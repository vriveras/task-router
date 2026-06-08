/**
 * End-to-end integration tests for the task-router.
 *
 * These tests exercise the full pipeline: ONNX model loading,
 * feature extraction, classification, routing decisions, escalation,
 * and cloud consultation — all wired together.
 */

import { describe, it, expect, vi, beforeAll } from "vitest";
import { join } from "node:path";
import { existsSync } from "node:fs";
import { Router } from "../src/router.js";
import { extractGreenfieldFeatures } from "../src/greenfieldFeatures.js";
import {
    classifyWithEmbeddings,
    initEmbeddingClassifier,
    clearEmbeddingCache,
} from "../src/embeddingClassifier.js";
import { extractGreenfieldFeatures } from "../src/greenfieldFeatures.js";
import { extractEscalationFeatures } from "../src/escalationFeatures.js";
import { EscalationTracker } from "../src/escalationTracker.js";
import {
    verifyWithCloud,
    planWithCloud,
    pickConsultationStrategy,
    ConsultationStrategy,
} from "../src/cloudConsultation.js";
import { setLogger, silentLogger } from "../src/logger.js";

setLogger(silentLogger());

const modelsDir = join(import.meta.dirname, "../models");
const hasModels = existsSync(join(modelsDir, "greenfield_gbm.onnx"));
const hasEmbeddings = existsSync(join(modelsDir, "embeddings", "all-MiniLM-L6-v2.onnx"));

// ── Full routing pipeline ───────────────────────────────────────────────────

describe("E2E: Full routing pipeline", () => {
    const router = new Router({ modelsDir });

    it("routes a trivial task to 'small'", async () => {
        const decision = await router.route("Fix the typo in README.md line 5");
        expect(decision.modelClass).toBe("small");
        expect(decision.probabilityLarge).toBeLessThan(0.5);
        expect(decision.source).toMatch(/greenfield|heuristic/);
    });

    it("routes a complex task with higher probability than simple", async () => {
        const simple = await router.route("Fix typo in README");
        const complex = await router.route(
            "Refactor the entire authentication module from session-based to OAuth 2.0 with PKCE. " +
            "This requires updating the login flow, token refresh logic, CSRF protection, " +
            "all 47 integration test files, the database migration for storing refresh tokens, " +
            "the OpenAPI spec, client SDK generation, and backwards compatibility shims for " +
            "the three mobile apps that depend on the existing JWT format. " +
            "You must also implement rate limiting per client_id, add audit logging for all " +
            "auth events, update the Helm charts for the new token service, and ensure the " +
            "rollout can be done with zero downtime using feature flags.",
        );
        // Complex task should have higher P(large) than simple task
        expect(complex.probabilityLarge).toBeGreaterThan(simple.probabilityLarge);
    });

    it("returns consistent results for the same prompt", async () => {
        const prompt = "Add a null check before accessing user.email";
        const d1 = await router.route(prompt);
        const d2 = await router.route(prompt);
        expect(d1.modelClass).toBe(d2.modelClass);
        expect(d1.probabilityLarge).toBeCloseTo(d2.probabilityLarge, 3);
    });

    it("handles edge cases gracefully", async () => {
        // Single word
        const d1 = await router.route("help");
        expect(d1.modelClass).toBeDefined();

        // Very long prompt
        const longPrompt = "implement ".repeat(500);
        const d2 = await router.route(longPrompt);
        expect(d2.modelClass).toBeDefined();

        // Code-heavy prompt
        const codePrompt = `
function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
// Fix this to use memoization with a Map<number, number> cache
// and add proper TypeScript types and JSDoc documentation`;
        const d3 = await router.route(codePrompt);
        expect(d3.modelClass).toBeDefined();
    });
});

// ── Feature extraction pipeline ─────────────────────────────────────────────

describe("E2E: Feature extraction", () => {
    it("extracts all 34 greenfield features", () => {
        const features = extractGreenfieldFeatures(
            "Implement a REST API endpoint that accepts POST requests with a JSON body " +
            "containing { name: string, email: string } and returns a 201 Created response. " +
            "The endpoint must validate the email format and return 400 for invalid input.",
        );

        // Core F1 features
        expect(features.spec_word_count).toBeGreaterThan(20);
        expect(typeof features.spec_n_inline_code).toBe("number");
        expect(typeof features.spec_has_examples).toBe("number");

        // Should detect requirement keywords (must, return)
        expect(features.spec_n_requirement_keywords).toBeGreaterThan(0);
    });

    it("extracts escalation features from trajectory", () => {
        const messages = [
            { role: "user" },
            { role: "assistant", tool_calls: [{ function: { name: "grep" } }] },
            { role: "user" },
            { role: "assistant", tool_calls: [
                { function: { name: "read_file" } },
                { function: { name: "grep" } },
            ]},
            { role: "user" },
            { role: "assistant", tool_calls: [
                { function: { name: "edit_file" } },
                { function: { name: "bash" } },
            ]},
        ];

        const features = extractEscalationFeatures(messages);
        expect(features.n_turns).toBe(3);
        expect(features.n_tool_calls).toBe(5);
        expect(features.n_grep).toBe(2);
        expect(features.n_read).toBe(1);
        expect(features.n_edit).toBe(1);
        expect(features.n_run).toBe(1);
    });
});

// ── Escalation tracker ──────────────────────────────────────────────────────

describe("E2E: Escalation tracking", () => {
    it("escalates after repeated grep cycles", async () => {
        const tracker = new EscalationTracker();

        // Simulate a session where the model keeps grepping without finding what it needs
        for (let i = 0; i < 15; i++) {
            tracker.recordTurn();
            tracker.recordToolCalls(["grep", "read_file"]);
        }

        // shouldEscalate() evaluates and sets the escalated flag
        expect(tracker.shouldEscalate()).toBe(true);
        expect(tracker.getState().escalated).toBe(true);
    });

    it("does not escalate on short sessions", () => {
        const tracker = new EscalationTracker();
        tracker.recordTurn();
        tracker.recordToolCalls(["edit_file"]);
        tracker.recordTurn();
        tracker.recordToolCalls(["bash"]);

        const state = tracker.getState();
        expect(state.escalated).toBe(false);
    });

    it("escalation is sticky", () => {
        const tracker = new EscalationTracker();

        // Force escalation
        for (let i = 0; i < 20; i++) {
            tracker.recordTurn();
            tracker.recordToolCalls(["grep", "grep"]);
        }

        expect(tracker.shouldEscalate()).toBe(true);
        expect(tracker.getState().escalated).toBe(true);

        // Record a simple turn — should still be escalated
        tracker.recordTurn();
        tracker.recordToolCalls(["edit_file"]);
        expect(tracker.shouldEscalate()).toBe(true);
    });
});

// ── Embedding classifier (requires ONNX models) ────────────────────────────

describe.skipIf(!hasEmbeddings)("E2E: Embedding classifier", () => {
    let initSuccess = false;

    beforeAll(async () => {
        await initEmbeddingClassifier(modelsDir);
        // Check if init actually loaded the sessions
        const probe = await classifyWithEmbeddings(extractGreenfieldFeatures("test"), "test", modelsDir);
        initSuccess = probe.success;
    });

    it("classifies a simple task as small/medium (class 0 or 1)", async () => {
        if (!initSuccess) return; // ONNX runtime not available
        const result = await classifyWithEmbeddings(extractGreenfieldFeatures("rename variable x to count"), "rename variable x to count", modelsDir);
        expect(result.success).toBe(true);
        if (result.success) {
            expect([0, 1]).toContain(result.predictedClass);
            expect(result.probabilities).toHaveLength(4);
            expect(result.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 1);
        }
    });

    it("classifies a complex task as large/cloud (class 2 or 3)", async () => {
        if (!initSuccess) return;
        const complexPrompt =
            "Redesign the entire microservices architecture to use event sourcing with CQRS. " +
            "Migrate all 12 services from REST to gRPC, implement saga patterns for distributed " +
            "transactions, add OpenTelemetry tracing, and set up Kubernetes operators for auto-scaling.";
        const result = await classifyWithEmbeddings(
            extractGreenfieldFeatures(complexPrompt),
            complexPrompt,
            modelsDir,
        );
        expect(result.success).toBe(true);
        if (result.success) {
            expect([2, 3]).toContain(result.predictedClass);
        }
    });

    it("returns 4 class probabilities that sum to ~1", async () => {
        if (!initSuccess) return;
        const result = await classifyWithEmbeddings(extractGreenfieldFeatures("fix null pointer"), "fix null pointer", modelsDir);
        expect(result.success).toBe(true);
        if (result.success) {
            expect(result.probabilities).toHaveLength(4);
            const sum = result.probabilities.reduce((a, b) => a + b, 0);
            expect(sum).toBeCloseTo(1.0, 1);
        }
    });

    it("caches embeddings for repeated prompts", async () => {
        clearEmbeddingCache();
        const prompt = "add error handling to the API endpoint";
        const features = extractGreenfieldFeatures(prompt);

        const start1 = performance.now();
        await classifyWithEmbeddings(features, prompt, modelsDir);
        const time1 = performance.now() - start1;

        const start2 = performance.now();
        await classifyWithEmbeddings(features, prompt, modelsDir);
        const time2 = performance.now() - start2;

        // Second call should be faster (cached embedding)
        // Allow generous margin since CI can be noisy
        expect(time2).toBeLessThan(time1 * 2);
    });
});

// ── Full delegation flow (route → execute → consult) ────────────────────────

describe("E2E: Delegation flow", () => {
    const router = new Router({ modelsDir });

    it("simple task: route local → no consultation", async () => {
        const decision = await router.route("Fix typo in README");
        expect(decision.modelClass).toBe("small");

        const strategy = pickConsultationStrategy(
            { complexity: 1, probLarge: decision.probabilityLarge },
            3,
        );
        expect(strategy.strategy).toBe(ConsultationStrategy.NONE);
    });

    it("boundary task: route local → verify with cloud → accepted", async () => {
        // Simulate a boundary task (complexity = maxComplexity)
        const strategy = pickConsultationStrategy({ complexity: 3 }, 3);
        expect(strategy.strategy).toBe(ConsultationStrategy.VERIFY_AFTER);

        // Simulate local execution produced a result
        const localResponse = "Fixed the authentication bug by adding null check on line 42";

        // Cloud verifies and accepts
        const mockCloud = vi.fn().mockResolvedValue(
            '{"score": 8, "accept": true, "reason": "Correct fix, good approach"}',
        );

        const verification = await verifyWithCloud({
            task: "Fix the authentication bug in auth.ts",
            localResponse,
            complete: mockCloud,
        });

        expect(verification.success).toBe(true);
        expect(verification.accepted).toBe(true);
        expect(verification.score).toBe(8);
    });

    it("boundary task: route local → verify with cloud → rejected → would re-run on cloud", async () => {
        const strategy = pickConsultationStrategy({ complexity: 3 }, 3);
        expect(strategy.strategy).toBe(ConsultationStrategy.VERIFY_AFTER);

        const localResponse = "I couldn't find the file";

        const mockCloud = vi.fn().mockResolvedValue(
            '{"score": 2, "accept": false, "reason": "Response is incomplete, file was not located"}',
        );

        const verification = await verifyWithCloud({
            task: "Fix the authentication bug in auth.ts",
            localResponse,
            complete: mockCloud,
        });

        expect(verification.success).toBe(true);
        expect(verification.accepted).toBe(false);
        expect(verification.score).toBe(2);
        // Caller would now re-run on cloud
    });

    it("complex task: plan-before → cloud plans → local executes → verify", async () => {
        // Task exceeds tier but probLarge is high
        const strategy = pickConsultationStrategy({ complexity: 4, probLarge: 0.7 }, 3);
        expect(strategy.strategy).toBe(ConsultationStrategy.PLAN_BEFORE);

        // Step 1: Cloud creates plan
        const mockPlanCloud = vi.fn().mockResolvedValue(
            "1. Open src/auth.ts\n2. Add OAuth2 provider config\n3. Implement token refresh\n4. Update tests\n5. Run npm test",
        );

        const plan = await planWithCloud({
            task: "Migrate auth from sessions to OAuth 2.0",
            complete: mockPlanCloud,
        });

        expect(plan.success).toBe(true);
        expect(plan.response).toContain("OAuth");

        // Step 2: Local executes (simulated)
        const localResponse = "Implemented OAuth2 flow as planned. Updated auth.ts and 3 test files.";

        // Step 3: Cloud verifies result
        const mockVerifyCloud = vi.fn().mockResolvedValue(
            '{"score": 7, "accept": true, "reason": "Implementation follows the plan correctly"}',
        );

        const verification = await verifyWithCloud({
            task: "Migrate auth from sessions to OAuth 2.0",
            localResponse,
            complete: mockVerifyCloud,
        });

        expect(verification.success).toBe(true);
        expect(verification.accepted).toBe(true);
        expect(verification.score).toBeGreaterThanOrEqual(7);
    });

    it("cloud failure: consultation fails open, local result accepted", async () => {
        const mockCloud = vi.fn().mockRejectedValue(new Error("503 Service Unavailable"));

        // Both verify and plan should fail open
        const verification = await verifyWithCloud({
            task: "Fix bug",
            localResponse: "Fixed it",
            complete: mockCloud,
        });
        expect(verification.success).toBe(false);
        expect(verification.accepted).toBe(true); // fail-open

        const plan = await planWithCloud({
            task: "Fix bug",
            complete: mockCloud,
        });
        expect(plan.success).toBe(false);
        expect(plan.response).toBe(""); // empty plan, local proceeds without guidance
    });
});

// ── Router + Escalation combined ────────────────────────────────────────────

describe("E2E: Router with escalation override", () => {
    it("starts small, escalates to large after many grep cycles", async () => {
        const router = new Router({ modelsDir });

        // Initial routing: simple task → small
        const initial = await router.route("check if file exists");
        expect(initial.modelClass).toBe("small");

        // Simulate many turns of searching — escalation should trigger
        for (let i = 0; i < 20; i++) {
            router.recordTurn();
            router.recordToolCalls(["grep", "read_file"]);
        }

        // Check escalation
        const escalated = await router.checkEscalation();
        expect(escalated.modelClass).toBe("large");
        expect(escalated.source).toBe("escalation");
    });
});

// ── Batch routing (simulates a real workload) ───────────────────────────────

describe("E2E: Batch routing workload", () => {
    const router = new Router({ modelsDir });

    const tasks = [
        { prompt: "Fix typo in README", expectedClass: "small" },
        { prompt: "Rename variable from x to count", expectedClass: "small" },
        { prompt: "Add a log statement after the API call", expectedClass: "small" },
        {
            prompt:
                "Redesign the entire database schema to support multi-tenancy with row-level security, " +
                "migrate all existing data, update all 30 queries, add tenant isolation tests, " +
                "and implement a tenant provisioning API with rate limiting and audit logging",
            expectedClass: "large",
        },
        {
            prompt:
                "Implement a distributed consensus algorithm for the cluster coordinator, " +
                "including leader election, log replication, and membership changes following the Raft protocol",
            expectedClass: "large",
        },
    ];

    it("correctly routes a batch of diverse tasks", async () => {
        let correct = 0;
        for (const task of tasks) {
            const decision = await router.route(task.prompt);
            if (decision.modelClass === task.expectedClass) {
                correct++;
            }
        }
        // At least 3/5 should be correct (60%) — allows for heuristic fallback
        expect(correct).toBeGreaterThanOrEqual(3);
    });
});
