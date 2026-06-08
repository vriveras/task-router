import { describe, it, expect, vi } from "vitest";
import {
    verifyWithCloud,
    planWithCloud,
    pickConsultationStrategy,
    ConsultationStrategy,
} from "../src/cloudConsultation.js";
import { setLogger, silentLogger } from "../src/logger.js";

// Suppress console output during tests
setLogger(silentLogger());

describe("pickConsultationStrategy", () => {
    it("returns NONE for low complexity", () => {
        const result = pickConsultationStrategy({ complexity: 1 }, 3);
        expect(result.strategy).toBe(ConsultationStrategy.NONE);
    });

    it("returns NONE when well below max", () => {
        const result = pickConsultationStrategy({ complexity: 1 }, 4);
        expect(result.strategy).toBe(ConsultationStrategy.NONE);
    });

    it("returns VERIFY_AFTER at max complexity", () => {
        const result = pickConsultationStrategy({ complexity: 3 }, 3);
        expect(result.strategy).toBe(ConsultationStrategy.VERIFY_AFTER);
    });

    it("returns PLAN_BEFORE when complexity exceeds tier and probLarge > 0.4", () => {
        // complexity > maxComplexity: task exceeds tier, but probLarge triggers plan-before
        const result = pickConsultationStrategy({ complexity: 4, probLarge: 0.6 }, 3);
        expect(result.strategy).toBe(ConsultationStrategy.PLAN_BEFORE);
    });

    it("returns VERIFY_AFTER at boundary", () => {
        const result = pickConsultationStrategy({ complexity: 3, probLarge: 0.2 }, 3);
        expect(result.strategy).toBe(ConsultationStrategy.VERIFY_AFTER);
    });
});

describe("verifyWithCloud", () => {
    it("parses JSON response and returns score", async () => {
        const complete = vi.fn().mockResolvedValue('{"score": 8, "accept": true, "reason": "Good result"}');

        const result = await verifyWithCloud({
            task: "Fix the typo",
            localResponse: "Changed 'teh' to 'the'",
            complete,
        });

        expect(result.success).toBe(true);
        expect(result.accepted).toBe(true);
        expect(result.score).toBe(8);
        expect(result.response).toBe("Good result");
        expect(complete).toHaveBeenCalledOnce();
    });

    it("rejects low-score responses", async () => {
        const complete = vi.fn().mockResolvedValue('{"score": 3, "accept": false, "reason": "Incomplete fix"}');

        const result = await verifyWithCloud({
            task: "Refactor auth module",
            localResponse: "I renamed a variable",
            complete,
        });

        expect(result.success).toBe(true);
        expect(result.accepted).toBe(false);
        expect(result.score).toBe(3);
    });

    it("fails open on cloud error", async () => {
        const complete = vi.fn().mockRejectedValue(new Error("API timeout"));

        const result = await verifyWithCloud({
            task: "Fix bug",
            localResponse: "Fixed it",
            complete,
        });

        expect(result.success).toBe(false);
        expect(result.accepted).toBe(true); // fail-open
        expect(result.error).toContain("API timeout");
    });

    it("retries on transient failures", async () => {
        const complete = vi
            .fn()
            .mockRejectedValueOnce(new Error("429 rate limit"))
            .mockResolvedValue('{"score": 9, "accept": true, "reason": "Excellent"}');

        const result = await verifyWithCloud({
            task: "Add test",
            localResponse: "Added test coverage",
            complete,
        });

        expect(result.success).toBe(true);
        expect(result.score).toBe(9);
        expect(complete).toHaveBeenCalledTimes(2);
    });

    it("handles non-JSON response gracefully", async () => {
        const complete = vi.fn().mockResolvedValue("Looks good to me!");

        const result = await verifyWithCloud({
            task: "Fix bug",
            localResponse: "Fixed",
            complete,
        });

        expect(result.success).toBe(true);
        expect(result.accepted).toBe(true);
        expect(result.score).toBe(7);
    });
});

describe("planWithCloud", () => {
    it("returns cloud plan", async () => {
        const plan = "1. Open auth.ts\n2. Add null check on line 42\n3. Run tests";
        const complete = vi.fn().mockResolvedValue(plan);

        const result = await planWithCloud({
            task: "Fix null pointer in auth module",
            complete,
        });

        expect(result.success).toBe(true);
        expect(result.response).toBe(plan);
        expect(result.latencyMs).toBeGreaterThan(0);
    });

    it("fails open on cloud error", async () => {
        const complete = vi.fn().mockRejectedValue(new Error("Network error"));

        const result = await planWithCloud({
            task: "Refactor database layer",
            complete,
        });

        expect(result.success).toBe(false);
        expect(result.response).toBe("");
        expect(result.error).toContain("Network error");
    });
});
