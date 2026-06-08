import { Router } from "../src/index.js";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const router = new Router({
    modelsDir: join(__dirname, "../models"),
});

// Route a simple task
const decision = await router.route("Fix the typo in README.md line 42");
console.log(`Decision: ${decision.modelClass} (${decision.reason})`);
console.log(`P(large): ${decision.probabilityLarge.toFixed(3)}`);

// Route a complex task
const complex = await router.route(
    "Refactor the entire authentication module to use OAuth 2.0 with PKCE flow, " +
        "update all 47 test files, migrate the database schema, and ensure backward " +
        "compatibility with the existing JWT-based sessions",
);
console.log(`\nComplex: ${complex.modelClass} (${complex.reason})`);
console.log(`P(large): ${complex.probabilityLarge.toFixed(3)}`);
