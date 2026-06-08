# Task Router

Standalone ONNX-based task complexity router for AI agent orchestration. Routes tasks between local (small/medium) and cloud (large) models based on prompt analysis and conversation trajectory.

Extracted from `copilot-agent-runtime` for use in OpenClaw and other AI agent orchestrators.

## Features

- **Greenfield classification** — 34 prompt-only features (word count, code density, requirement keywords, TF-IDF similarity)
- **Embedding classification** — MiniLM-L6-v2 embeddings + GBM classifier (84% validation accuracy, 4-class)
- **Escalation tracking** — Trajectory-based features (tool call patterns, turns, grep/read/edit/run counts)
- **Zero external API calls** — All classification runs locally via ONNX Runtime
- **Framework agnostic** — Use as a library, HTTP server, or OpenAI-compatible proxy

## Quick Start

```bash
npm install
npm run build
npm run example:basic
```

## Usage

### As a library

```typescript
import { Router } from "@vriveras/task-router";

const router = new Router({ modelsDir: "./models" });
const decision = await router.route("Fix the typo in README.md");
// { modelClass: "small", reason: "...", probabilityLarge: 0.12, ... }
```

### As an HTTP server

```bash
npm run example:server
curl -X POST http://localhost:3456/route \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Fix typo"}'
```

### As an OpenAI-compatible proxy

```bash
LOCAL_MODEL=qwen3:30b CLOUD_MODEL=gpt-4o npm run example:openai
# Then point your client at http://localhost:4000/v1/chat/completions
```

## Architecture

```
Prompt → Greenfield Features (34) ─┐
                                    ├─→ ONNX GBM → P(large) → Route Decision
MiniLM Embeddings (384) ───────────┘

Trajectory → Escalation Features (6) → Escalation GBM → Override if escalating
```

## Models

| Model | Size | Purpose |
|-------|------|---------|
| `greenfield_gbm.onnx` | 6 KB | Prompt-only binary classifier |
| `escalation_gbm.onnx` | 28 KB | Trajectory escalation detector |
| `brownfield_gbm.onnx` | 28 KB | Multi-turn session classifier |
| `local_router_embed.onnx` | 1.5 MB | 4-class GBM with embeddings (84% acc) |
| `all-MiniLM-L6-v2.onnx` | 86 MB | Sentence embedding model |

## Custom Logger

By default, task-router logs to the console. Inject your own logger:

```typescript
import { setLogger } from "@vriveras/task-router";

setLogger({
    info: (msg) => myLogger.info(msg),
    warning: (msg) => myLogger.warn(msg),
    error: (msg) => myLogger.error(msg),
    debug: (msg) => myLogger.debug(msg),
});
```

Or silence logging entirely:

```typescript
import { setLogger, silentLogger } from "@vriveras/task-router";
setLogger(silentLogger());
```

## For OpenClaw Integration

```typescript
import { Router, classifyWithEmbeddings, initEmbeddingClassifier } from "@vriveras/task-router";

// Initialize once
const modelsDir = "./node_modules/@vriveras/task-router/models";
await initEmbeddingClassifier(modelsDir);

// Per-request routing
async function routeTask(prompt: string): Promise<"local" | "cloud"> {
    const result = await classifyWithEmbeddings({}, prompt, modelsDir);
    if (!result.success) return "cloud"; // fallback
    return result.predictedClass >= 2 ? "cloud" : "local";
}
```

## Testing

```bash
npm test
```

## License

MIT
