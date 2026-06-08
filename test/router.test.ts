import { describe, it, expect } from "vitest";
import { extractGreenfieldFeatures } from "../src/greenfieldFeatures.js";
import { classifyToolName } from "../src/toolClassification.js";
import { TfidfVectorizer } from "../src/tfidf.js";
import { extractEscalationFeatures } from "../src/escalationFeatures.js";
import { EscalationTracker } from "../src/escalationTracker.js";
import { Router } from "../src/router.js";
import { silentLogger, setLogger } from "../src/logger.js";

// Suppress console output during tests
setLogger(silentLogger());

describe("toolClassification", () => {
    it("classifies grep tools", () => {
        expect(classifyToolName("grep")).toBe("grep");
        expect(classifyToolName("search_files")).toBe("grep");
    });
    it("classifies read tools", () => {
        expect(classifyToolName("read_file")).toBe("read");
        expect(classifyToolName("view")).toBe("read");
    });
    it("classifies edit tools", () => {
        expect(classifyToolName("edit_file")).toBe("edit");
        expect(classifyToolName("write")).toBe("edit");
    });
    it("classifies run tools", () => {
        expect(classifyToolName("bash")).toBe("run");
        expect(classifyToolName("shell_exec")).toBe("run");
    });
    it("returns null for unknown", () => {
        expect(classifyToolName("unknown_tool")).toBeNull();
    });
});

describe("greenfieldFeatures", () => {
    it("extracts features from a simple prompt", () => {
        const features = extractGreenfieldFeatures("Fix the typo in README.md");
        expect(features).toBeDefined();
        expect(typeof features.spec_word_count).toBe("number");
        expect(features.spec_word_count).toBeGreaterThan(0);
    });

    it("extracts more features from a complex prompt", () => {
        const simple = extractGreenfieldFeatures("rename variable x to count");
        const complex = extractGreenfieldFeatures(
            "Implement a full OAuth 2.0 authorization code flow with PKCE, " +
                "including token refresh, session management, and CSRF protection. " +
                "The implementation must handle edge cases like expired tokens, " +
                "concurrent requests, and support both web and mobile clients.",
        );
        expect(complex.spec_word_count).toBeGreaterThan(simple.spec_word_count);
    });

    it("extracts 34 features total", () => {
        const features = extractGreenfieldFeatures("Fix a bug in the authentication module");
        expect(Object.keys(features).length).toBe(34);
    });

    it("detects requirement keywords", () => {
        const features = extractGreenfieldFeatures("The function must return a list and should raise an error");
        expect(features.spec_n_requirement_keywords).toBeGreaterThan(0);
    });

    it("detects inline code references", () => {
        const features = extractGreenfieldFeatures("Update the `getUserById` function in `auth.ts`");
        expect(features.spec_n_inline_code).toBe(2);
        expect(features.spec_n_distinct_symbol_tokens).toBe(2);
    });
});

describe("escalationFeatures", () => {
    it("counts tool calls from trajectory", () => {
        const messages = [
            { role: "user" },
            { role: "assistant", tool_calls: [{ function: { name: "grep" } }] },
            { role: "user" },
            {
                role: "assistant",
                tool_calls: [{ function: { name: "read_file" } }, { function: { name: "edit_file" } }],
            },
        ];
        const features = extractEscalationFeatures(messages);
        expect(features.n_turns).toBe(2);
        expect(features.n_tool_calls).toBe(3);
        expect(features.n_grep).toBe(1);
        expect(features.n_read).toBe(1);
        expect(features.n_edit).toBe(1);
    });

    it("handles empty trajectory", () => {
        const features = extractEscalationFeatures([]);
        expect(features.n_turns).toBe(0);
        expect(features.n_tool_calls).toBe(0);
    });
});

describe("escalationTracker", () => {
    it("starts un-escalated", () => {
        const tracker = new EscalationTracker();
        expect(tracker.getState().escalated).toBe(false);
        expect(tracker.shouldEscalate()).toBe(false);
    });

    it("tracks turns and tool calls", () => {
        const tracker = new EscalationTracker();
        tracker.recordTurn();
        tracker.recordToolCalls(["grep", "read_file"]);
        const state = tracker.getState();
        expect(state.turnCount).toBe(1);
        expect(state.toolCounts.grep).toBe(1);
        expect(state.toolCounts.read).toBe(1);
        expect(state.toolCounts.total).toBe(2);
    });

    it("escalation is sticky", () => {
        const tracker = new EscalationTracker();
        // Simulate enough activity to trigger escalation
        for (let i = 0; i < 5; i++) {
            tracker.recordTurn();
            tracker.recordToolCalls(["edit_file", "bash"]);
        }
        expect(tracker.shouldEscalate()).toBe(true);
        expect(tracker.getState().escalated).toBe(true);
        // Should remain escalated even after reset of threshold
        expect(tracker.shouldEscalate(999)).toBe(true);
    });
});

describe("tfidf", () => {
    it("builds and queries a vectorizer", () => {
        const v = new TfidfVectorizer();
        v.fit(["hello world", "foo bar baz"]);
        const vec = v.transform("hello");
        expect(vec.length).toBeGreaterThan(0);
    });

    it("fitTransform returns matrix", () => {
        const v = new TfidfVectorizer();
        const matrix = v.fitTransform(["hello world", "foo bar"]);
        expect(matrix.length).toBe(2);
        expect(matrix[0].length).toBeGreaterThan(0);
    });

    it("supports bigram ngrams", () => {
        const v = new TfidfVectorizer({ ngramRange: [1, 2] });
        v.fit(["hello world foo"]);
        const vec = v.transform("hello world");
        expect(vec.length).toBeGreaterThan(0);
    });
});

describe("Router", () => {
    it("creates with default config", () => {
        const router = new Router();
        expect(router.isEnabled()).toBe(true);
    });

    it("can be disabled", () => {
        const router = new Router({ enabled: false });
        expect(router.isEnabled()).toBe(false);
    });

    it("validates config", () => {
        expect(() => new Router({ defaultThreshold: 2 })).toThrow();
        expect(() => new Router({ defaultThreshold: -1 })).toThrow();
        expect(() => new Router({ modelsDir: "" })).toThrow();
    });

    it("routes with heuristic fallback (no models)", async () => {
        const router = new Router();
        const decision = await router.route("Fix the typo in README.md");
        expect(decision.modelClass).toBe("small");
        expect(decision.source).toBe("heuristic");
        expect(decision.probabilityLarge).toBeGreaterThanOrEqual(0);
        expect(decision.probabilityLarge).toBeLessThanOrEqual(1);
    });

    it("routes complex prompts as large", async () => {
        const router = new Router();
        const decision = await router.route(
            "Refactor the entire authentication module to use OAuth 2.0 with PKCE flow, " +
                "update all 47 test files, migrate the database schema, and ensure backward " +
                "compatibility with the existing JWT-based sessions. " +
                "The implementation must handle token refresh, CSRF protection, " +
                "concurrent session management, and support both web and mobile clients. " +
                Array(200).fill("additional context words ").join(""),
        );
        expect(decision.modelClass).toBe("large");
    });

    it("tracks escalation state", () => {
        const router = new Router();
        const state = router.getEscalationState();
        expect(state.escalated).toBe(false);
        router.recordTurn();
        router.recordToolCalls(["grep"]);
        const updated = router.getEscalationState();
        expect(updated.turnCount).toBe(1);
        expect(updated.toolCounts.grep).toBe(1);
    });

    it("rejects non-string prompts", async () => {
        const router = new Router();
        await expect(router.route("")).rejects.toThrow();
    });
});
