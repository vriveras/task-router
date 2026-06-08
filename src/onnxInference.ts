/**
 * ONNX model inference wrapper for routing GBM models.
 * Uses onnxruntime-node for model inference with graceful fallback.
 *
 * Types below mirror the subset of the onnxruntime-node API we use.
 * The package is dynamically imported (optional dependency), so we define
 * local interfaces rather than importing from `onnxruntime-common`.
 */

import { readFile } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { formatError } from "./helpers.js";
import { getLogger } from "./logger.js";
import { RoutingErrorCode } from "./errors.js";

/** Minimal representation of an ONNX tensor returned by inference. */
interface OnnxTensor {
    /** Underlying typed-array data */
    data: Float32Array;
}

/** Subset of `ort.InferenceSession` used for model inference. */
interface OnnxInferenceSession {
    /** Names of the model's input tensors */
    inputNames: string[];
    /** Names of the model's output tensors */
    outputNames: string[];
    /** Run inference with a map of input-name → tensor, returning output-name → tensor */
    run(feeds: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>>;
}

/** Subset of the `onnxruntime-node` module API we consume. */
interface OnnxRuntimeModule {
    InferenceSession: {
        create(modelPath: string): Promise<OnnxInferenceSession>;
    };
    Tensor: new (type: "float32", data: Float32Array, dims: number[]) => OnnxTensor;
}

/** Cache ONNX inference sessions to avoid expensive re-creation on every call. */
const sessionCache = new Map<string, OnnxInferenceSession>();

/** Descriptor for a bundled ONNX model loaded from a JSON manifest. */
export interface ModelBundle {
    /** Path to the ONNX model file */
    modelPath: string;
    /** Ordered list of feature names expected by the model */
    featureOrder: string[];
    /** Classification threshold (probability >= threshold → class 1 / "large") */
    threshold: number;
    /** Optional metadata */
    metadata?: Record<string, unknown>;
    /** SHA-256 hash of the model file for integrity verification */
    sha256?: string;
}

/** Result of running a single inference pass. */
export interface InferenceResult {
    /** Probability of class 1 ("large") */
    probability: number;
    /** Whether inference succeeded */
    success: boolean;
    /** Error message if inference failed */
    error?: string;
    /** Standardized error code for programmatic handling */
    errorCode?: RoutingErrorCode;
}

/**
 * Load a model bundle descriptor from a JSON file.
 *
 * The JSON file must contain `modelPath` (relative to the bundle directory),
 * `featureOrder`, and optionally `threshold` and `metadata`.
 *
 * @param bundlePath - Path to the JSON bundle manifest file
 * @returns The resolved model bundle with an absolute model path and validated integrity
 * @throws If the bundle is invalid, the model path escapes the bundle directory, or integrity check fails
 */
export async function loadModelBundle(bundlePath: string): Promise<ModelBundle> {
    const raw = await readFile(bundlePath, "utf-8");
    const parsed = JSON.parse(raw) as Partial<ModelBundle>;

    if (!parsed.modelPath || !Array.isArray(parsed.featureOrder)) {
        const err = new Error(`Invalid model bundle at ${bundlePath}: missing modelPath or featureOrder`);
        (err as Error & { code: RoutingErrorCode }).code = RoutingErrorCode.INVALID_BUNDLE;
        getLogger().warning(`[OnnxInference] invalid model bundle: path=${bundlePath}`);
        throw err;
    }

    // Resolve modelPath relative to the bundle file's directory
    const bundleDir = dirname(bundlePath);
    const resolvedModelPath = join(bundleDir, parsed.modelPath);

    // Guard against path traversal attacks — modelPath must stay within bundleDir
    const canonicalBundleDir = resolve(bundleDir);
    const canonicalModelPath = resolve(resolvedModelPath);
    if (!canonicalModelPath.startsWith(canonicalBundleDir + sep)) {
        const err = new Error(`Model path escapes bundle directory: ${parsed.modelPath}`);
        (err as Error & { code: RoutingErrorCode }).code = RoutingErrorCode.PATH_TRAVERSAL;
        getLogger().warning(`[OnnxInference] path traversal blocked: modelPath=${parsed.modelPath} bundleDir=${canonicalBundleDir}`);
        throw err;
    }

    // Verify model file integrity if the bundle specifies a hash
    if (typeof parsed.sha256 === "string") {
        const { createHash } = await import("node:crypto");
        const modelData = await readFile(resolvedModelPath);
        const fileHash = createHash("sha256").update(modelData).digest("hex");
        if (fileHash !== parsed.sha256) {
            const err = new Error(
                `Model integrity check failed for ${resolvedModelPath}: expected ${parsed.sha256}, got ${fileHash}`,
            );
            (err as Error & { code: RoutingErrorCode }).code = RoutingErrorCode.INTEGRITY_CHECK_FAILED;
            getLogger().warning(`[OnnxInference] integrity check failed: path=${resolvedModelPath} expected=${parsed.sha256} actual=${fileHash}`);
            throw err;
        }
    }

    return {
        modelPath: resolvedModelPath,
        featureOrder: parsed.featureOrder,
        threshold: typeof parsed.threshold === "number" ? parsed.threshold : 0.5,
        metadata: parsed.metadata,
        sha256: typeof parsed.sha256 === "string" ? parsed.sha256 : undefined,
    };
}

/**
 * Run inference on a single sample using an ONNX model.
 *
 * Builds a feature vector in the order specified by `bundle.featureOrder`,
 * loads onnxruntime-node dynamically, and returns the probability of class 1
 * ("large"). If onnxruntime-node is not available or inference fails, returns
 * a failed result so callers can fall back to heuristics.
 *
 * @param bundle - The model bundle containing the ONNX model path and feature order
 * @param features - A record mapping feature names to their numeric values
 * @returns The inference result with the probability of the "large" class and a success flag
 */
export async function predictProbability(
    bundle: ModelBundle,
    features: Record<string, number>,
): Promise<InferenceResult> {
    // Build feature array in the model's expected order
    const featureArray = new Float32Array(bundle.featureOrder.length);
    for (let i = 0; i < bundle.featureOrder.length; i++) {
        featureArray[i] = features[bundle.featureOrder[i]] ?? 0;
    }

    // Dynamically import onnxruntime-node — it may not be installed.
    let ort: OnnxRuntimeModule;
    try {
        const moduleName = "onnxruntime-node";
        ort = (await import(/* webpackIgnore: true */ moduleName)) as OnnxRuntimeModule;
    } catch {
        getLogger().debug(`[OnnxInference] onnxruntime-node not available, skipping model inference`);
        return {
            probability: 0,
            success: false,
            error: "onnxruntime-node not available",
            errorCode: RoutingErrorCode.ONNX_RUNTIME_UNAVAILABLE,
        };
    }

    try {
        let session = sessionCache.get(bundle.modelPath);
        if (!session) {
            getLogger().debug(`[OnnxInference] creating inference session: model=${bundle.modelPath} features=${bundle.featureOrder.length}`);
            session = await ort.InferenceSession.create(bundle.modelPath);
            sessionCache.set(bundle.modelPath, session);
        }
        if (!session) {
            getLogger().warning(`[OnnxInference] failed to create inference session: model=${bundle.modelPath}`);
            return {
                probability: 0,
                success: false,
                error: "failed to create inference session",
                errorCode: RoutingErrorCode.SESSION_CREATION_FAILED,
            };
        }

        // Create input tensor — shape [1, numFeatures] for a single sample
        const inputTensor = new ort.Tensor("float32", featureArray, [1, bundle.featureOrder.length]);

        // Use the first input name from the model
        const inputName = session.inputNames[0];
        if (!inputName) {
            getLogger().warning(`[OnnxInference] model has no input names: model=${bundle.modelPath}`);
            return {
                probability: 0,
                success: false,
                error: "model has no input names",
                errorCode: RoutingErrorCode.MODEL_NO_INPUTS,
            };
        }

        const results = await session.run({ [inputName]: inputTensor });

        // GBM classifiers typically output probabilities via "probabilities" or
        // the last output. Try known names, then fall back to the last output.
        const probOutput =
            results["probabilities"] ??
            results["output_probability"] ??
            results[session.outputNames[session.outputNames.length - 1]];

        if (!probOutput?.data) {
            getLogger().warning(`[OnnxInference] no probability output from model: model=${bundle.modelPath} outputNames=${JSON.stringify(session.outputNames)}`);
            return {
                probability: 0,
                success: false,
                error: "no probability output from model",
                errorCode: RoutingErrorCode.NO_PROBABILITY_OUTPUT,
            };
        }

        // For binary classification the probabilities tensor is [1, 2] — take index 1
        const { data } = probOutput;
        const probability = data.length >= 2 ? Number(data[1]) : Number(data[0]);

        getLogger().debug(`[OnnxInference] inference succeeded: model=${bundle.modelPath} probability=${probability.toFixed(4)}`);
        return { probability, success: true };
    } catch (err: unknown) {
        const message = formatError(err);
        getLogger().warning(`[OnnxInference] inference failed: model=${bundle.modelPath} error=${message}`);
        return {
            probability: 0,
            success: false,
            error: `inference failed: ${message}`,
            errorCode: RoutingErrorCode.INFERENCE_FAILED,
        };
    }
}
