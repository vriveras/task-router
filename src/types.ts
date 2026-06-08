/**
 * Shared types for task-router.
 */

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
    source: "greenfield" | "brownfield" | "escalation" | "heuristic" | "embedding";
}

export interface EmbeddingClassification {
    tier: "small" | "medium" | "large" | "cloud";
    probabilities: number[];
    features?: number[];
}

export interface RouterConfig {
    /** Path to directory containing ONNX model bundles */
    modelsDir?: string;
    /** Default threshold when not specified in bundle (default: 0.5) */
    defaultThreshold?: number;
    /** Whether routing is enabled (default: true) */
    enabled?: boolean;
}
