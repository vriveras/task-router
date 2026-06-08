/**
 * Trajectory / escalation features — counted from the message history.
 */

import { classifyToolName } from "./toolClassification.js";
import { getLogger } from "./logger.js";

export interface TrajectoryMessage {
    role: string;
    tool_calls?: Array<{ function: { name: string } }>;
}

/**
 * Extract the 6 escalation features from a conversation trajectory.
 *
 * @param messages - Array of conversation messages with roles and optional tool calls
 * @returns A record mapping feature names (n_turns, n_tool_calls, n_grep, n_read, n_edit, n_run) to counts
 */
export function extractEscalationFeatures(messages: TrajectoryMessage[]): Record<string, number> {
    let nTurns = 0;
    let nToolCalls = 0;
    let nGrep = 0;
    let nRead = 0;
    let nEdit = 0;
    let nRun = 0;

    for (const m of messages) {
        if (m.role === "user") {
            nTurns++;
        }
        if (m.role !== "assistant") {
            continue;
        }
        for (const tc of m.tool_calls ?? []) {
            nToolCalls++;
            const category = classifyToolName(tc.function?.name ?? "");
            if (category === "grep") {
                nGrep++;
            } else if (category === "read") {
                nRead++;
            } else if (category === "edit") {
                nEdit++;
            } else if (category === "run") {
                nRun++;
            }
        }
    }

    const result = {
        n_turns: nTurns,
        n_tool_calls: nToolCalls,
        n_grep: nGrep,
        n_read: nRead,
        n_edit: nEdit,
        n_run: nRun,
    };

    getLogger().debug(
        `[EscalationFeatures] extracted features: turns=${nTurns} toolCalls=${nToolCalls} grep=${nGrep} read=${nRead} edit=${nEdit} run=${nRun} messages=${messages.length}`,
    );

    return result;
}
