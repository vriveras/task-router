/**
 * Standardized error codes for the routing subsystem.
 *
 * Every error path in the router should set one of these codes so that
 * callers can programmatically react to specific failure modes without
 * parsing human-readable error strings.
 */
export enum RoutingErrorCode {
    // ── Model bundle loading ────────────────────────────────────────────
    /** Bundle JSON is missing required `modelPath` or `featureOrder` fields. */
    INVALID_BUNDLE = "INVALID_BUNDLE",
    /** The resolved model path escapes the bundle directory (path traversal). */
    PATH_TRAVERSAL = "PATH_TRAVERSAL",
    /** SHA-256 hash of the model file does not match the expected value. */
    INTEGRITY_CHECK_FAILED = "INTEGRITY_CHECK_FAILED",

    // ── ONNX runtime ────────────────────────────────────────────────────
    /** `onnxruntime-node` could not be dynamically imported. */
    ONNX_RUNTIME_UNAVAILABLE = "ONNX_RUNTIME_UNAVAILABLE",
    /** An ONNX `InferenceSession` could not be created for the model file. */
    SESSION_CREATION_FAILED = "SESSION_CREATION_FAILED",

    // ── Inference ────────────────────────────────────────────────────────
    /** The ONNX model reports no input tensor names. */
    MODEL_NO_INPUTS = "MODEL_NO_INPUTS",
    /** The model produced no recognisable probability output tensor. */
    NO_PROBABILITY_OUTPUT = "NO_PROBABILITY_OUTPUT",
    /** A generic inference-time failure (see accompanying `error` message). */
    INFERENCE_FAILED = "INFERENCE_FAILED",

    // ── Embedding classifier ─────────────────────────────────────────────
    /** The embedding ONNX model file was not found on disk. */
    EMBEDDING_MODEL_NOT_FOUND = "EMBEDDING_MODEL_NOT_FOUND",
    /** The embedding session failed to initialise or was previously null. */
    EMBEDDING_SESSION_NOT_LOADED = "EMBEDDING_SESSION_NOT_LOADED",
    /** The tokenizer JSON file could not be loaded. */
    TOKENIZER_NOT_FOUND = "TOKENIZER_NOT_FOUND",
    /** The GBM classifier ONNX model was not found on disk. */
    CLASSIFIER_NOT_FOUND = "CLASSIFIER_NOT_FOUND",
    /** The classifier session failed to initialise or was previously null. */
    CLASSIFIER_SESSION_NOT_LOADED = "CLASSIFIER_SESSION_NOT_LOADED",
    /** A catch-all for failures during embedding-based classification. */
    EMBEDDING_CLASSIFICATION_FAILED = "EMBEDDING_CLASSIFICATION_FAILED",
}

/**
 * Error codes specific to local model routing failures.
 *
 * Returned when the router decides *not* to route locally, so that
 * callers (telemetry, UI, retry logic) can react programmatically.
 */
export enum LocalRoutingErrorCode {
    /** The Ollama server could not be reached or the config failed to load. */
    OLLAMA_UNREACHABLE = "OLLAMA_UNREACHABLE",
    /** The prompt exceeds the local tier's context window. */
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW",
    /** The task's complexity exceeds the tier's maximum capability. */
    COMPLEXITY_EXCEEDED = "COMPLEXITY_EXCEEDED",
    /** All local GPU inference slots are occupied. */
    SLOT_SATURATED = "SLOT_SATURATED",
    /** No model mapping exists for the configured VRAM budget tier. */
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND",
}
