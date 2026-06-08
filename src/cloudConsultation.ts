/**
 * Cloud consultation module for delegation pattern.
 *
 * Implements verify-after and plan-before strategies where a local model
 * executes the task while a cloud/frontier model provides quality assurance.
 *
 * Design: fail-open — consultation failures accept the local result,
 * never blocking on cloud errors.
 */

import { getLogger } from "./logger.js";
import { formatError } from "./helpers.js";

/**
 * Result of a cloud consultation call.
 */
export interface ConsultationResult {
    /** Whether the consultation was successful */
    success: boolean;
    /** The cloud model's response */
    response: string;
    /** For verify-after: whether the cloud accepted the local result */
    accepted?: boolean;
    /** For verify-after: quality score 1-10 */
    score?: number;
    /** Latency of the consultation call in ms */
    latencyMs: number;
    /** Error message if consultation failed */
    error?: string;
}

/**
 * A function that sends a prompt to a cloud model and returns the response.
 * Provided by the caller so consultation uses the same model/auth they chose.
 *
 * @example
 * // OpenAI-compatible
 * const complete: CloudCompletionFn = async (prompt) => {
 *     const res = await fetch("https://api.openai.com/v1/chat/completions", {
 *         method: "POST",
 *         headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
 *         body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: prompt }] }),
 *     });
 *     const data = await res.json();
 *     return data.choices[0].message.content;
 * };
 */
export type CloudCompletionFn = (prompt: string) => Promise<string>;

/**
 * Consultation strategy — determines when/how the cloud is consulted.
 */
export enum ConsultationStrategy {
    /** No consultation — pure local execution */
    NONE = "none",
    /** Cloud verifies local result after execution */
    VERIFY_AFTER = "verify-after",
    /** Cloud creates a plan before local execution */
    PLAN_BEFORE = "plan-before",
}

/**
 * Pick the consultation strategy based on classifier confidence and complexity.
 *
 * @param decision - The routing decision with complexity and probability info
 * @param maxComplexity - Maximum complexity the local tier can handle (1-4)
 * @returns The strategy to use and optional model override
 */
export function pickConsultationStrategy(
    decision: { complexity: number; probLarge?: number },
    maxComplexity: number,
): { strategy: ConsultationStrategy } {
    // Pure local: low complexity relative to tier capacity
    if (decision.complexity <= 1 || decision.complexity <= maxComplexity - 1) {
        return { strategy: ConsultationStrategy.NONE };
    }

    // Near the tier's limit: verify with cloud after execution
    if (decision.complexity === maxComplexity) {
        return { strategy: ConsultationStrategy.VERIFY_AFTER };
    }

    // High large probability but still locally portable → cloud plans, local executes
    if (decision.probLarge !== undefined && decision.probLarge > 0.4) {
        return { strategy: ConsultationStrategy.PLAN_BEFORE };
    }

    // Default: verify after for anything at the boundary
    if (decision.complexity >= maxComplexity - 1) {
        return { strategy: ConsultationStrategy.VERIFY_AFTER };
    }

    return { strategy: ConsultationStrategy.NONE };
}

/**
 * Verify a local model's result with a cloud model.
 *
 * Sends the original task + local response to the cloud for quality scoring.
 * Returns whether the result should be accepted or re-run on cloud.
 *
 * @example
 * const result = await verifyWithCloud({
 *     task: "Fix the null check in auth.ts",
 *     localResponse: localModel.output,
 *     complete: openaiComplete,
 * });
 * if (!result.accepted) {
 *     // Re-run on cloud
 * }
 */
export async function verifyWithCloud(opts: {
    task: string;
    localResponse: string;
    complete: CloudCompletionFn;
}): Promise<ConsultationResult> {
    const { task, localResponse, complete } = opts;
    const logger = getLogger();
    const start = performance.now();

    try {
        const verifyPrompt = `You are a code review quality gate. Score this AI-generated response to a coding task.

## Task
${task.slice(0, 2000)}

## Response to evaluate
${localResponse.slice(0, 3000)}

## Instructions
Score the response from 1 to 10:
- 1-3: Wrong, incomplete, or harmful
- 4-6: Partially correct but has issues
- 7-8: Correct with minor issues
- 9-10: Excellent

Respond with JSON only: {"score": N, "accept": true/false, "reason": "brief explanation"}
Accept if score >= 7.`;

        let content: string | undefined;
        let lastError: unknown;
        const maxAttempts = 3;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                content = await complete(verifyPrompt);
                break;
            } catch (err) {
                lastError = err;
                if (attempt < maxAttempts - 1) {
                    const delayMs = 1000 * 2 ** attempt;
                    logger.warning(
                        `Consultation verify attempt ${attempt + 1} failed, retrying in ${delayMs}ms: ${formatError(err)}`,
                    );
                    await new Promise((resolve) => setTimeout(resolve, delayMs));
                }
            }
        }

        if (content === undefined) {
            throw lastError;
        }

        const latencyMs = performance.now() - start;

        // Parse JSON response
        const jsonMatch = content.match(/\{[\s\S]*"score"\s*:\s*\d+[\s\S]*\}/);
        if (jsonMatch) {
            try {
                const parsed = JSON.parse(jsonMatch[0]) as { score: number; accept: boolean; reason: string };
                logger.info(
                    `Consultation verify: score=${parsed.score}/10, accept=${parsed.accept} (${Math.round(latencyMs)}ms)`,
                );
                return {
                    success: true,
                    response: parsed.reason,
                    accepted: parsed.accept,
                    score: parsed.score,
                    latencyMs,
                };
            } catch {
                // Fall through to raw response
            }
        }

        logger.info(`Consultation verify: raw response (${Math.round(latencyMs)}ms)`);
        return { success: true, response: content, accepted: true, score: 7, latencyMs };
    } catch (error) {
        const latencyMs = performance.now() - start;
        const msg = formatError(error);
        logger.warning(`Consultation verify failed: ${msg}`);
        // Fail-open: accept local result on consultation failure
        return { success: false, response: "", accepted: true, latencyMs, error: msg };
    }
}

/**
 * Get an execution plan from a cloud model before local execution.
 *
 * The cloud model analyzes the task and returns numbered steps
 * that the local model can follow mechanically.
 *
 * @example
 * const plan = await planWithCloud({
 *     task: "Refactor auth module to use OAuth",
 *     complete: openaiComplete,
 * });
 * if (plan.success) {
 *     // Inject plan into local model's system prompt
 *     localPrompt += `\n\nFollow this plan:\n${plan.response}`;
 * }
 */
export async function planWithCloud(opts: {
    task: string;
    complete: CloudCompletionFn;
}): Promise<ConsultationResult> {
    const { task, complete } = opts;
    const logger = getLogger();
    const start = performance.now();

    try {
        const planPrompt = `You are a senior engineer creating an execution plan for a junior developer.

## Task
${task.slice(0, 3000)}

## Instructions
Create a numbered step-by-step execution plan (max 5 steps).
Each step should be specific and actionable — include file paths, function names, and exact changes.
The developer will follow these steps mechanically without needing to understand the big picture.

Keep it concise — under 200 words total. No explanations, just steps.`;

        const content = await complete(planPrompt);

        const latencyMs = performance.now() - start;

        logger.info(`Consultation plan: ${content.split("\n").length} steps (${Math.round(latencyMs)}ms)`);
        return { success: true, response: content, latencyMs };
    } catch (error) {
        const latencyMs = performance.now() - start;
        const msg = formatError(error);
        logger.warning(`Consultation plan failed: ${msg}`);
        // Fail-open: return empty plan, local model proceeds without guidance
        return { success: false, response: "", latencyMs, error: msg };
    }
}
