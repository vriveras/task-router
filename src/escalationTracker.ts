/**
 * Per-session escalation state management.
 *
 * Tracks user turns and tool-call patterns across a session, then decides
 * whether the session should be escalated from a small model to a large model.
 * Escalation is **sticky**: once triggered it remains active for the rest of
 * the session.
 */

import { classifyToolName } from "./toolClassification.js";
import { getLogger } from "./logger.js";

/** Snapshot of escalation state for a single session. */
export interface EscalationState {
    /** Whether this session has been escalated to the large model */
    escalated: boolean;
    /** Number of user turns so far */
    turnCount: number;
    /** Cumulative tool call counts by category */
    toolCounts: {
        grep: number;
        read: number;
        edit: number;
        run: number;
        total: number;
    };
}

/** Default heuristic threshold for escalation. */
const DEFAULT_ESCALATION_THRESHOLD = 1.0;

/** Divisor for edit+run cycles in the escalation score formula. */
export const ESCALATION_CYCLE_DIVISOR = 5;
/** Divisor for grep tool calls in the escalation score formula. */
export const ESCALATION_GREP_DIVISOR = 6;
/** Minimum turns before escalation can trigger; also the offset subtracted from turn count. */
export const ESCALATION_MIN_TURNS = 2;
/** Divisor for turn count (above minimum) in the escalation score formula. */
export const ESCALATION_TURN_DIVISOR = 3;

export class EscalationTracker {
    private state: EscalationState;

    /** Create a new EscalationTracker with a fresh, un-escalated state. */
    constructor() {
        this.state = {
            escalated: false,
            turnCount: 0,
            toolCounts: { grep: 0, read: 0, edit: 0, run: 0, total: 0 },
        };
    }

    /**
     * Record a new user turn.
     *
     * Increments the internal turn counter used by the escalation heuristic.
     */
    recordTurn(): void {
        this.state.turnCount++;
        getLogger().debug(`[EscalationTracker] recorded turn: turnCount=${this.state.turnCount}`);
    }

    /**
     * Record tool calls from an assistant message.
     *
     * Each tool name is classified into one of the four canonical categories
     * (grep, read, edit, run). Unrecognised tools still count toward the total.
     *
     * @param toolNames - Array of tool names from the assistant's tool calls
     */
    recordToolCalls(toolNames: string[]): void {
        for (const name of toolNames) {
            this.state.toolCounts.total++;
            const category = classifyToolName(name);
            if (category) {
                this.state.toolCounts[category]++;
            }
        }
        getLogger().debug(
            `[EscalationTracker] recorded tool calls: count=${toolNames.length} totals={grep=${this.state.toolCounts.grep}, read=${this.state.toolCounts.read}, edit=${this.state.toolCounts.edit}, run=${this.state.toolCounts.run}}`,
        );
    }

    /**
     * Check whether escalation should trigger.
     *
     * The decision is **sticky** — once `true`, every subsequent call returns
     * `true` without re-evaluating.
     *
     * Heuristic (mirrors the Python escalation tracker):
     * ```
     * cycles = edit + run
     * score  = (cycles / 5) + (grep / 6) + (max(0, turns - 2) / 3)
     * escalate if score >= threshold (default 1.0)
     * ```
     *
     * @param threshold - Score threshold for triggering escalation (default: 1.0)
     * @returns `true` if the session should be escalated to the large model
     */
    shouldEscalate(threshold: number = DEFAULT_ESCALATION_THRESHOLD): boolean {
        if (this.state.escalated) {
            return true;
        }

        if (this.state.turnCount < ESCALATION_MIN_TURNS) {
            return false;
        }

        const { edit, run, grep } = this.state.toolCounts;
        const cycles = edit + run;
        const score =
            cycles / ESCALATION_CYCLE_DIVISOR +
            grep / ESCALATION_GREP_DIVISOR +
            Math.max(0, this.state.turnCount - ESCALATION_MIN_TURNS) / ESCALATION_TURN_DIVISOR;

        if (score >= threshold) {
            this.state.escalated = true;
            getLogger().info(
                `[EscalationTracker] escalation triggered: score=${score.toFixed(2)} threshold=${threshold} turns=${this.state.turnCount} cycles=${cycles} grep=${grep}`,
            );
            return true;
        }

        return false;
    }

    /**
     * Get current escalation state (read-only snapshot).
     *
     * @returns A frozen copy of the current escalation state
     */
    getState(): Readonly<EscalationState> {
        return {
            escalated: this.state.escalated,
            turnCount: this.state.turnCount,
            toolCounts: { ...this.state.toolCounts },
        };
    }

    /**
     * Get features dictionary suitable for model inference.
     *
     * Returns the same feature names used by the Python training pipeline.
     *
     * @returns A record mapping feature names (n_turns, n_tool_calls, etc.) to their numeric values
     */
    getFeatures(): Record<string, number> {
        return {
            n_turns: this.state.turnCount,
            n_tool_calls: this.state.toolCounts.total,
            n_grep: this.state.toolCounts.grep,
            n_read: this.state.toolCounts.read,
            n_edit: this.state.toolCounts.edit,
            n_run: this.state.toolCounts.run,
        };
    }

    /**
     * Reset the tracker to its initial un-escalated state.
     *
     * Intended for testing.
     */
    reset(): void {
        this.state = {
            escalated: false,
            turnCount: 0,
            toolCounts: { grep: 0, read: 0, edit: 0, run: 0, total: 0 },
        };
    }
}
