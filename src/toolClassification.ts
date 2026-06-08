/**
 * Shared tool-name classification.
 *
 * Categories mirror the Python `features_traj.py` logic:
 * - grep / search → `"grep"`
 * - read / view / cat → `"read"`
 * - edit / write / patch → `"edit"`
 * - run / shell / exec / bash → `"run"`
 */

/** Canonical tool categories used by escalation features and tracking. */
export type ToolCategory = "grep" | "read" | "edit" | "run";

/**
 * Classify a tool name into one of the four canonical categories,
 * or `null` if it doesn't match any known pattern.
 *
 * @param name - The tool name to classify (e.g., "grep", "bash", "edit_file")
 * @returns The canonical tool category, or `null` if the name is unrecognised
 */
export function classifyToolName(name: string): ToolCategory | null {
    const lower = name.toLowerCase();
    if (lower.includes("grep") || lower.includes("search")) {
        return "grep";
    }
    if (lower.includes("read") || lower.includes("view") || lower.includes("cat")) {
        return "read";
    }
    if (lower.includes("edit") || lower.includes("write") || lower.includes("patch")) {
        return "edit";
    }
    if (lower.includes("run") || lower.includes("shell") || lower.includes("exec") || lower.includes("bash")) {
        return "run";
    }
    return null;
}
