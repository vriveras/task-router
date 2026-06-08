/**
 * Simple injectable logger for task-router.
 *
 * By default logs to console. Call `setLogger()` to inject your own
 * logger implementation (e.g., pino, winston, or a noop logger).
 */

export interface Logger {
    info(message: string): void;
    warning(message: string): void;
    error(message: string): void;
    debug(message: string): void;
}

const noopLogger: Logger = {
    info: () => {},
    warning: () => {},
    error: () => {},
    debug: () => {},
};

const consoleLogger: Logger = {
    info: (msg) => console.log(`[task-router] ${msg}`),
    warning: (msg) => console.warn(`[task-router] ${msg}`),
    error: (msg) => console.error(`[task-router] ${msg}`),
    debug: (msg) => console.debug(`[task-router] ${msg}`),
};

let activeLogger: Logger = consoleLogger;

export function setLogger(logger: Logger): void {
    activeLogger = logger;
}

export function getLogger(): Logger {
    return activeLogger;
}

export function silentLogger(): Logger {
    return noopLogger;
}
