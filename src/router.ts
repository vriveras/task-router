/**
 * Main router that combines prompt-level (greenfield) classification with
 * trajectory-based escalation to decide whether a session turn should be
 * served by a "small" or "large" model.
 */

import type { EscalationState } from "./escalationTracker.js";
import {
    EscalationTracker,
    ESCALATION_CYCLE_DIVISOR,
    ESCALATION_GREP_DIVISOR,
    ESCALATION_MIN_TURNS,
    ESCALATION_TURN_DIVISOR,
} from "./escalationTracker.js";
import { extractGreenfieldFeatures as extractFullGreenfieldFeatures } from "./greenfieldFeatures.js";
import type { ModelBundle } from "./onnxInference.js";
import { loadModelBundle, predictProbability } from "./onnxInference.js";
import { getLogger } from "./logger.js";

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// ── Load routing config ─────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));

function loadRoutingConfig(): { bundles: Record<string, string>; embeddings: Record<string, string> } {
    const candidates = [
        join(__dirname, "..", "models", "routing-config.json"),
        join(__dirname, "..", "src", "data", "routing-config.json"),
    ];
    for (const candidate of candidates) {
        try {
            return JSON.parse(readFileSync(candidate, "utf-8"));
        } catch {
            // try next
        }
    }
    return {
        bundles: {
            greenfield: "greenfield_gbm.json",
            escalation: "escalation_gbm.json",
            brownfield: "brownfield_gbm.json",
            localRouterEmbed: "local_router_embed.json",
        },
        embeddings: {
            model: "embeddings/all-MiniLM-L6-v2.onnx",
            tokenizer: "embeddings/tokenizer.json",
            classifier: "local_router_embed.onnx",
        },
    };
}

const routingConfig = loadRoutingConfig();

// ── Types ───────────────────────────────────────────────────────────────────

/** The outcome of a single routing decision. */
export interface RouteDecision {
    /** "small" or "large" model class */
    modelClass: "small" | "large";
    /** Human-readable reason for the decision */
    reason: string;
    /** Probability of "large" class (from model or heuristic) */
    probabilityLarge: number;
    /** Threshold used for the decision */
    thresholdUsed: number;
    /** Which model artifact was used */
    source: "greenfield" | "brownfield" | "escalation" | "heuristic";
}

/** Configuration for the Router. */
export interface RouterConfig {
    /** Path to directory containing ONNX model bundles */
    modelsDir?: string;
    /** Default threshold when not specified in bundle (default: 0.5) */
    defaultThreshold?: number;
    /** Whether routing is enabled (default: true) */
    enabled?: boolean;
}

/** Default routing threshold when neither config nor bundle specifies one. */
const DEFAULT_THRESHOLD = 0.5;

/** Divisor for word count in the greenfield heuristic score. */
const GREENFIELD_WORD_COUNT_DIVISOR = 200;
/** Divisor for distinct symbol token count in the greenfield heuristic score. */
const GREENFIELD_SYMBOL_COUNT_DIVISOR = 5;
/** Score penalty applied when the prompt contains no examples. */
const GREENFIELD_NO_EXAMPLES_PENALTY = 0.3;

export class Router {
    private config: RouterConfig;
    private escalationTracker: EscalationTracker;
    private greenfieldBundle: ModelBundle | null = null;
    private escalationBundle: ModelBundle | null = null;
    private bundlesLoaded = false;

    /**
     * Create a new Router instance.
     *
     * @param config - Optional router configuration (models directory, threshold, enabled flag)
     * @throws {Error} If config contains invalid values
     */
    constructor(config?: RouterConfig) {
        if (config !== undefined && config !== null) {
            if (typeof config !== "object" || Array.isArray(config)) {
                throw new Error("Router config must be a plain object");
            }
            if (config.defaultThreshold !== undefined) {
                if (typeof config.defaultThreshold !== "number" || isNaN(config.defaultThreshold)) {
                    throw new Error("Router config.defaultThreshold must be a number");
                }
                if (config.defaultThreshold < 0 || config.defaultThreshold > 1) {
                    throw new Error(
                        `Router config.defaultThreshold must be between 0 and 1, got ${config.defaultThreshold}`,
                    );
                }
            }
            if (config.modelsDir !== undefined && typeof config.modelsDir !== "string") {
                throw new Error("Router config.modelsDir must be a string");
            }
            if (typeof config.modelsDir === "string" && config.modelsDir.trim() === "") {
                throw new Error("Router config.modelsDir must not be empty");
            }
            if (config.enabled !== undefined && typeof config.enabled !== "boolean") {
                throw new Error("Router config.enabled must be a boolean");
            }
        }
        this.config = config ?? {};
        this.escalationTracker = new EscalationTracker();
    }

    /**
     * Check if routing is enabled.
     *
     * @returns `true` if routing is enabled (the default), `false` if explicitly disabled
     */
    isEnabled(): boolean {
        return this.config.enabled !== false;
    }

    /**
     * Convenience method: route a prompt through the greenfield classifier.
     *
     * @param prompt - The user's prompt text to classify
     * @returns A routing decision indicating "small" or "large" model class
     */
    async route(prompt: string): Promise<RouteDecision> {
        return this.decideGreenfield(prompt);
    }

    /**
     * Make a greenfield routing decision based on prompt text features.
     *
     * If an ONNX greenfield model is available it is used; otherwise a
     * heuristic fallback scores the prompt by word count and symbol density.
     *
     * @param prompt - The user's prompt text to classify
     * @returns A routing decision indicating "small" or "large" model class
     * @throws {Error} If prompt is not a non-empty string
     */
    async decideGreenfield(prompt: string): Promise<RouteDecision> {
        if (typeof prompt !== "string") {
            throw new Error(`decideGreenfield expects a string prompt, got ${typeof prompt}`);
        }
        if (prompt.trim() === "") {
            throw new Error("decideGreenfield expects a non-empty prompt");
        }

        const features = extractFullGreenfieldFeatures(prompt);
        await this.ensureBundlesLoaded();

        const threshold = this.effectiveThreshold(this.greenfieldBundle);

        if (this.greenfieldBundle) {
            const result = await predictProbability(this.greenfieldBundle, features);
            if (result.success) {
                const modelClass = result.probability >= threshold ? "large" : "small";
                getLogger().debug(
                    `[Router] greenfield decision: modelClass=${modelClass} probability=${result.probability.toFixed(3)} threshold=${threshold.toFixed(3)}`,
                );
                return {
                    modelClass,
                    reason: `greenfield model p_large=${result.probability.toFixed(2)} threshold=${threshold.toFixed(2)}`,
                    probabilityLarge: result.probability,
                    thresholdUsed: threshold,
                    source: "greenfield",
                };
            }
            getLogger().debug(`[Router] greenfield model inference failed, falling back to heuristic`);
        }

        const decision = this.heuristicGreenfield(features, threshold);
        getLogger().debug(
            `[Router] greenfield heuristic decision: modelClass=${decision.modelClass} probability=${decision.probabilityLarge.toFixed(3)} threshold=${threshold.toFixed(3)}`,
        );
        return decision;
    }

    /**
     * Check whether escalation is needed based on the current trajectory.
     *
     * Escalation is sticky: once triggered for a session it stays "large"
     * for all subsequent turns.
     *
     * @returns A routing decision reflecting the current escalation status
     */
    async checkEscalation(): Promise<RouteDecision> {
        const state = this.escalationTracker.getState();

        // Already escalated — short-circuit
        if (state.escalated) {
            getLogger().debug(`[Router] escalation check: already escalated, returning large`);
            return {
                modelClass: "large",
                reason: "previously escalated",
                probabilityLarge: 1,
                thresholdUsed: 0,
                source: "escalation",
            };
        }

        const features = this.escalationTracker.getFeatures();
        await this.ensureBundlesLoaded();

        const threshold = this.effectiveThreshold(this.escalationBundle);

        if (this.escalationBundle) {
            const result = await predictProbability(this.escalationBundle, features);
            if (result.success) {
                const escalated = result.probability >= threshold;
                if (escalated) {
                    // Mark sticky escalation via shouldEscalate path
                    this.escalationTracker.shouldEscalate(0);
                }
                getLogger().debug(
                    `[Router] escalation model decision: escalated=${escalated} probability=${result.probability.toFixed(3)} threshold=${threshold.toFixed(3)} turns=${features["n_turns"]}`,
                );
                return {
                    modelClass: escalated ? "large" : "small",
                    reason: `escalation model p=${result.probability.toFixed(2)} threshold=${threshold.toFixed(2)}`,
                    probabilityLarge: result.probability,
                    thresholdUsed: threshold,
                    source: "escalation",
                };
            }
            getLogger().debug(`[Router] escalation model inference failed, falling back to heuristic`);
        }

        // Heuristic fallback — delegates to EscalationTracker
        const decision = this.heuristicEscalation(features, threshold);
        getLogger().debug(
            `[Router] escalation heuristic decision: modelClass=${decision.modelClass} source=${decision.source} turns=${features["n_turns"]}`,
        );
        return decision;
    }

    /**
     * Record a user turn for escalation tracking.
     *
     * Increments the internal turn counter used by the escalation heuristic.
     */
    recordTurn(): void {
        this.escalationTracker.recordTurn();
    }

    /**
     * Record tool calls for escalation tracking.
     *
     * @param toolNames - Array of tool names invoked in the assistant message
     * @throws {Error} If toolNames is not an array of strings
     */
    recordToolCalls(toolNames: string[]): void {
        if (!Array.isArray(toolNames)) {
            throw new Error(`recordToolCalls expects an array of tool names, got ${typeof toolNames}`);
        }
        for (let i = 0; i < toolNames.length; i++) {
            if (typeof toolNames[i] !== "string") {
                throw new Error(
                    `recordToolCalls expects all tool names to be strings, got ${typeof toolNames[i]} at index ${i}`,
                );
            }
        }
        this.escalationTracker.recordToolCalls(toolNames);
    }

    /**
     * Get current escalation state.
     *
     * @returns A read-only snapshot of the current escalation state
     */
    getEscalationState(): Readonly<EscalationState> {
        return this.escalationTracker.getState();
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /** Lazily load model bundles from the configured directory. */
    private async ensureBundlesLoaded(): Promise<void> {
        if (this.bundlesLoaded || !this.config.modelsDir) {
            return;
        }
        this.bundlesLoaded = true;

        try {
            this.greenfieldBundle = await loadModelBundle(`${this.config.modelsDir}/${routingConfig.bundles.greenfield}`);
            getLogger().debug(`[Router] loaded greenfield model bundle from=${this.config.modelsDir}`);
        } catch {
            getLogger().debug(`[Router] greenfield model not available, heuristic will be used`);
        }

        try {
            this.escalationBundle = await loadModelBundle(`${this.config.modelsDir}/${routingConfig.bundles.escalation}`);
            getLogger().debug(`[Router] loaded escalation model bundle from=${this.config.modelsDir}`);
        } catch {
            getLogger().debug(`[Router] escalation model not available, heuristic will be used`);
        }
    }

    /** Resolve the effective threshold from bundle → config → default. */
    private effectiveThreshold(bundle: ModelBundle | null): number {
        if (bundle?.threshold !== undefined) {
            return bundle.threshold;
        }
        return this.config.defaultThreshold ?? DEFAULT_THRESHOLD;
    }

    /**
     * Heuristic greenfield decision (mirrors Python `_heuristic_green`).
     *
     * Long or symbol-dense prompts → large; otherwise small.
     */
    private heuristicGreenfield(features: Record<string, number>, threshold: number): RouteDecision {
        const words = features["spec_word_count"] ?? 0;
        const symbols = features["spec_n_distinct_symbol_tokens"] ?? 0;
        const examples = features["spec_has_examples"] ?? 0;

        const score =
            words / GREENFIELD_WORD_COUNT_DIVISOR +
            symbols / GREENFIELD_SYMBOL_COUNT_DIVISOR +
            (examples ? 0 : GREENFIELD_NO_EXAMPLES_PENALTY);
        const probability = Math.min(1, score);

        return {
            modelClass: score >= threshold ? "large" : "small",
            reason: `heuristic green (words=${words}, syms=${symbols}, score=${score.toFixed(2)})`,
            probabilityLarge: probability,
            thresholdUsed: threshold,
            source: "heuristic",
        };
    }

    /**
     * Heuristic escalation decision (mirrors Python escalation tracker).
     *
     * ```
     * cycles = edit + run
     * score  = (cycles / 5) + (grep / 6) + (max(0, turns - 2) / 3)
     * ```
     */
    private heuristicEscalation(features: Record<string, number>, threshold: number): RouteDecision {
        const shouldEscalate = this.escalationTracker.shouldEscalate();
        const state = this.escalationTracker.getState();

        const nEdit = features["n_edit"] ?? 0;
        const nRun = features["n_run"] ?? 0;
        const nGrep = features["n_grep"] ?? 0;
        const nTurns = features["n_turns"] ?? 0;
        const cycles = nEdit + nRun;
        const score =
            cycles / ESCALATION_CYCLE_DIVISOR +
            nGrep / ESCALATION_GREP_DIVISOR +
            Math.max(0, nTurns - ESCALATION_MIN_TURNS) / ESCALATION_TURN_DIVISOR;
        const probability = Math.min(1, score);

        return {
            modelClass: shouldEscalate || state.escalated || score >= threshold ? "large" : "small",
            reason: shouldEscalate
                ? `heuristic escalation cycles=${cycles} grep=${nGrep} turns=${nTurns}`
                : `below escalation threshold (score=${score.toFixed(2)})`,
            probabilityLarge: probability,
            thresholdUsed: threshold,
            source: shouldEscalate ? "escalation" : "heuristic",
        };
    }
}
