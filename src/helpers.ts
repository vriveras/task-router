/**
 * Format an unknown error value into a human-readable string.
 */
export function formatError(error: unknown): string {
    if (error instanceof Error) {
        return String(error);
    } else if (typeof error === "object" && error !== null) {
        try {
            return JSON.stringify(error) ?? "[object]";
        } catch {
            return "[object with circular reference]";
        }
    } else {
        return String(error);
    }
}
