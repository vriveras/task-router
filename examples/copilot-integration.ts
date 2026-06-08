/**
 * Copilot CLI SDK integration example.
 *
 * Uses the Copilot API (free via GitHub Copilot subscription) as the cloud
 * model for consultation. Auth via `gh auth token`.
 *
 * Usage:
 *   npx tsx examples/copilot-integration.ts
 *   npx tsx examples/copilot-integration.ts "Refactor the auth module to use OAuth 2.0"
 */

import { execSync } from "node:child_process";
import { join } from "node:path";
import {
    Router,
    classifyWithEmbeddings,
    initEmbeddingClassifier,
    extractGreenfieldFeatures,
    pickConsultationStrategy,
    ConsultationStrategy,
    verifyWithCloud,
    planWithCloud,
    setLogger,
    type CloudCompletionFn,
} from "../src/index.js";

// ── Copilot API client ──────────────────────────────────────────────────────

const COPILOT_API_URL = "https://api.githubcopilot.com";
const COPILOT_MODEL = process.env.COPILOT_MODEL ?? "claude-sonnet-4.5";
const COPILOT_CONSULT_MODEL = process.env.COPILOT_CONSULT_MODEL ?? "claude-haiku-4.5";

function getGitHubToken(): string {
    const envToken = process.env.COPILOT_GITHUB_TOKEN ?? process.env.GH_TOKEN ?? process.env.GITHUB_TOKEN;
    if (envToken) return envToken;

    try {
        return execSync("gh auth token", { encoding: "utf-8", timeout: 5000 }).trim();
    } catch {
        throw new Error("No GitHub token found. Run `gh auth login` or set GH_TOKEN.");
    }
}

/**
 * Create a CloudCompletionFn backed by the Copilot API.
 */
function createCopilotCompletion(model: string): CloudCompletionFn {
    const token = getGitHubToken();

    return async (prompt: string): Promise<string> => {
        const response = await fetch(`${COPILOT_API_URL}/chat/completions`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
                "Copilot-Integration-Id": "copilot-developer",
            },
            body: JSON.stringify({
                model,
                messages: [{ role: "user", content: prompt }],
                max_tokens: 1024,
                temperature: 0.0,
            }),
        });

        if (!response.ok) {
            const body = await response.text();
            throw new Error(`Copilot API ${response.status}: ${body.slice(0, 200)}`);
        }

        const data = (await response.json()) as {
            choices: Array<{ message: { content: string } }>;
            usage?: { prompt_tokens: number; completion_tokens: number };
        };

        const usage = data.usage;
        if (usage) {
            console.log(
                `  [copilot] ${model} — ${usage.prompt_tokens} in / ${usage.completion_tokens} out tokens`,
            );
        }

        return data.choices[0]?.message?.content ?? "";
    };
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
    const modelsDir = join(import.meta.dirname, "../models");

    // Quiet logger for cleaner output
    setLogger({
        info: () => {},
        warning: (msg) => console.warn(`  ⚠ ${msg}`),
        error: (msg) => console.error(`  ✖ ${msg}`),
        debug: () => {},
    });

    console.log("🔧 Initializing task-router...");
    const router = new Router({ modelsDir });
    await initEmbeddingClassifier(modelsDir);

    // Get prompt from CLI arg or use default
    const prompt =
        process.argv[2] ??
        "Add comprehensive error handling to the payment processing module, " +
        "including retry logic for transient failures, dead letter queue for " +
        "permanent failures, and alerts via PagerDuty webhook.";

    console.log(`\n📝 Task: "${prompt.slice(0, 100)}${prompt.length > 100 ? "..." : ""}"\n`);

    // ── Step 1: Route ────────────────────────────────────────────────────

    console.log("━━━ Step 1: Route ━━━");
    const decision = await router.route(prompt);
    console.log(`  Model class: ${decision.modelClass}`);
    console.log(`  P(large):    ${decision.probabilityLarge.toFixed(3)}`);
    console.log(`  Source:      ${decision.source}`);
    console.log(`  Reason:      ${decision.reason}`);

    // ── Step 2: Embedding classification (4-class) ───────────────────────

    console.log("\n━━━ Step 2: Embedding classify ━━━");
    const textFeatures = extractGreenfieldFeatures(prompt);
    const embedding = await classifyWithEmbeddings(textFeatures, prompt, modelsDir);
    if (embedding.success) {
        const tiers = ["small", "medium", "large", "cloud"] as const;
        const tier = tiers[embedding.predictedClass] ?? "unknown";
        console.log(`  Tier:          ${tier} (class ${embedding.predictedClass})`);
        console.log(
            `  Probabilities: [${embedding.probabilities.map((p) => p.toFixed(3)).join(", ")}]`,
        );
        console.log(
            `                  small   medium  large   cloud`,
        );
    } else {
        console.log(`  ⚠ Embedding classifier not available: ${embedding.error}`);
    }

    // ── Step 3: Consultation strategy ────────────────────────────────────

    console.log("\n━━━ Step 3: Consultation strategy ━━━");
    const complexity = embedding.success ? embedding.predictedClass + 1 : (decision.modelClass === "large" ? 3 : 1);
    const maxComplexity = 3; // Simulating a "large" VRAM tier
    const strategy = pickConsultationStrategy(
        { complexity, probLarge: decision.probabilityLarge },
        maxComplexity,
    );
    console.log(`  Complexity:    ${complexity} (max: ${maxComplexity})`);
    console.log(`  Strategy:      ${strategy.strategy}`);

    // ── Step 4: Execute with consultation ────────────────────────────────

    if (strategy.strategy === ConsultationStrategy.NONE) {
        console.log("\n━━━ Step 4: Pure local execution (no consultation) ━━━");
        console.log("  → Task would run entirely on local model");
        console.log("  → No cloud tokens consumed");
        return;
    }

    const consultComplete = createCopilotCompletion(COPILOT_CONSULT_MODEL);

    if (strategy.strategy === ConsultationStrategy.PLAN_BEFORE) {
        console.log("\n━━━ Step 4a: Cloud plans (plan-before) ━━━");
        const plan = await planWithCloud({ task: prompt, complete: consultComplete });
        if (plan.success) {
            console.log(`  Plan (${Math.round(plan.latencyMs)}ms):`);
            for (const line of plan.response.split("\n").slice(0, 7)) {
                console.log(`    ${line}`);
            }
        } else {
            console.log(`  ⚠ Plan failed: ${plan.error}`);
        }

        // Simulate local execution
        console.log("\n━━━ Step 4b: Local executes plan ━━━");
        const execComplete = createCopilotCompletion(COPILOT_MODEL);
        const execPrompt = plan.success
            ? `Execute this plan step by step:\n\n${plan.response}\n\nOriginal task: ${prompt}`
            : prompt;
        const localResponse = await execComplete(execPrompt);
        console.log(`  Local response (${localResponse.length} chars):`);
        console.log(`    ${localResponse.slice(0, 200)}...`);

        // Verify after execution
        console.log("\n━━━ Step 4c: Cloud verifies ━━━");
        const verification = await verifyWithCloud({
            task: prompt,
            localResponse,
            complete: consultComplete,
        });
        console.log(`  Score:    ${verification.score}/10`);
        console.log(`  Accepted: ${verification.accepted}`);
        console.log(`  Reason:   ${verification.response}`);
        console.log(`  Latency:  ${Math.round(verification.latencyMs)}ms`);
    }

    if (strategy.strategy === ConsultationStrategy.VERIFY_AFTER) {
        console.log("\n━━━ Step 4a: Local executes ━━━");
        const execComplete = createCopilotCompletion(COPILOT_MODEL);
        const localResponse = await execComplete(prompt);
        console.log(`  Local response (${localResponse.length} chars):`);
        console.log(`    ${localResponse.slice(0, 200)}...`);

        console.log("\n━━━ Step 4b: Cloud verifies (verify-after) ━━━");
        const verification = await verifyWithCloud({
            task: prompt,
            localResponse,
            complete: consultComplete,
        });
        console.log(`  Score:    ${verification.score}/10`);
        console.log(`  Accepted: ${verification.accepted}`);
        console.log(`  Reason:   ${verification.response}`);
        console.log(`  Latency:  ${Math.round(verification.latencyMs)}ms`);

        if (!verification.accepted) {
            console.log("\n━━━ Step 4c: Cloud re-executes (rejected) ━━━");
            const cloudResponse = await execComplete(prompt);
            console.log(`  Cloud response (${cloudResponse.length} chars):`);
            console.log(`    ${cloudResponse.slice(0, 200)}...`);
        }
    }

    console.log("\n✅ Done");
}

main().catch(console.error);
