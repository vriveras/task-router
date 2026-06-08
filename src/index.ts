/**
 * @vriveras/task-router — Standalone ONNX task complexity router
 */

export { Router } from "./router.js";
export type { RouteDecision, RouterConfig } from "./router.js";
export type { EmbeddingClassification } from "./types.js";
export { classifyWithEmbeddings, initEmbeddingClassifier, clearEmbeddingCache } from "./embeddingClassifier.js";
export type { EmbeddingClassifyResult } from "./embeddingClassifier.js";
export { extractGreenfieldFeatures } from "./greenfieldFeatures.js";
export { extractEscalationFeatures } from "./escalationFeatures.js";
export type { TrajectoryMessage } from "./escalationFeatures.js";
export { EscalationTracker } from "./escalationTracker.js";
export type { EscalationState } from "./escalationTracker.js";
export { TfidfVectorizer, cosineSimilarity } from "./tfidf.js";
export type { TfidfVectorizerOptions } from "./tfidf.js";
export { classifyToolName } from "./toolClassification.js";
export type { ToolCategory } from "./toolClassification.js";
export { loadModelBundle, predictProbability } from "./onnxInference.js";
export type { ModelBundle, InferenceResult } from "./onnxInference.js";
export { RoutingErrorCode, LocalRoutingErrorCode } from "./errors.js";
export { setLogger, getLogger, silentLogger } from "./logger.js";
export type { Logger } from "./logger.js";
export { formatError } from "./helpers.js";

// Cloud consultation (delegation pattern)
export {
    verifyWithCloud,
    planWithCloud,
    pickConsultationStrategy,
    ConsultationStrategy,
} from "./cloudConsultation.js";
export type { ConsultationResult, CloudCompletionFn } from "./cloudConsultation.js";
