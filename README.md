# Task Router

Standalone ONNX-based task complexity router for AI agent orchestration. Routes tasks between local and cloud models based on prompt analysis, then optionally verifies results with a cloud model using the **delegation pattern**.

> **TL;DR**: Classify a prompt in <5ms with zero API calls, then optionally consult a cloud model to verify quality. The router runs entirely offline; you bring your own cloud provider.

## What It Does

```
User prompt → "Refactor auth module to use OAuth 2.0 with PKCE..."

1. CLASSIFY (offline, <5ms)
   Embedding classifier → "large" (P=[.13, .37, .45, .04])
                            small  medium  large  cloud

2. PICK STRATEGY (based on complexity vs tier capacity)
   Complexity 3 at max tier 3 → "verify-after"

3. EXECUTE locally (your local model runs the task)

4. VERIFY with cloud ($0.04, ~2s)
   Cloud scores result → 5/10, rejected
   → Re-execute on cloud (quality preserved)
```

This is the **delegation pattern** from [MSR Frontiers research](https://github.com/vriveras/router-testing/blob/main/docs/delegation-economics-results.md): the local model executes while the cloud model remains available for verification. Unlike one-shot routing, delegation preserves frontier access and avoids quality cliffs.

## Quick Start

```bash
git clone https://github.com/vriveras/task-router.git
cd task-router
npm install
npm run build
npm test                          # 57 tests
npm run example:basic             # Offline routing demo
npm run example:copilot           # Live Copilot API delegation demo
```

## How It Works

### Routing Pipeline (3 layers, checked in priority order)

```
Prompt arrives
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Layer 1: EMBEDDING CLASSIFIER (84% accuracy)        │
│   MiniLM-L6-v2 → 384-dim embedding                 │
│   + 34 text features = 418-dim feature vector       │
│   → GBM (ONNX) → 4 probabilities:                  │
│     [P(small), P(medium), P(large), P(cloud)]       │
│   Winner = argmax → routing tier                    │
└────────────────────────┬────────────────────────────┘
                         │ if unavailable
                         ▼
┌─────────────────────────────────────────────────────┐
│ Layer 2: GREENFIELD GBM (65% accuracy)              │
│   34 text-only features → ONNX GBM → P(large)      │
└────────────────────────┬────────────────────────────┘
                         │ if unavailable
                         ▼
┌─────────────────────────────────────────────────────┐
│ Layer 3: HEURISTIC FALLBACK                         │
│   word_count/200 + symbol_density/5 → P(large)      │
└────────────────────────┬────────────────────────────┘
                         ▼
              RouteDecision { modelClass, probabilityLarge }
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ CONSULTATION STRATEGY (based on complexity vs tier) │
│   Low complexity    → NONE (pure local)             │
│   At tier max       → VERIFY_AFTER                  │
│   Exceeds tier      → PLAN_BEFORE                   │
└─────────────────────────────────────────────────────┘
```

### In-Session Escalation

The `EscalationTracker` watches tool call patterns during execution. If the local model keeps grep-cycling without progress, escalation triggers and overrides to "large" mid-conversation. Escalation is **sticky** — once triggered, it stays for the rest of the session.

### Cloud Consultation (Delegation Pattern)

Three strategies, picked automatically based on classifier confidence:

| Strategy | When | What Happens |
|----------|------|-------------|
| **NONE** | Low complexity, high confidence | Pure local execution, zero cloud cost |
| **VERIFY_AFTER** | At tier boundary | Local executes → cloud scores result (1-10) → accept or re-run |
| **PLAN_BEFORE** | Exceeds tier, high P(large) | Cloud creates step-by-step plan → local follows plan → cloud verifies |

The consultation module is **fail-open**: if the cloud is unreachable, the local result is accepted. This ensures offline operation is never blocked.

## Usage

### 1. As a Library (Offline Routing Only)

```typescript
import { Router } from "@vriveras/task-router";

const router = new Router({ modelsDir: "./models" });

// Simple routing
const decision = await router.route("Fix the typo in README.md");
console.log(decision.modelClass);      // "small"
console.log(decision.probabilityLarge); // 0.138

// Complex task
const complex = await router.route("Refactor auth to OAuth 2.0 with PKCE...");
console.log(complex.modelClass);      // "large"
console.log(complex.probabilityLarge); // 0.72
```

### 2. With Embedding Classifier (4-class, 84% accuracy)

```typescript
import {
    initEmbeddingClassifier,
    classifyWithEmbeddings,
    extractGreenfieldFeatures,
} from "@vriveras/task-router";

// Initialize once at startup
await initEmbeddingClassifier("./models");

// Classify
const features = extractGreenfieldFeatures("Add retry logic to payment module");
const result = await classifyWithEmbeddings(features, prompt, "./models");

if (result.success) {
    const tiers = ["small", "medium", "large", "cloud"];
    console.log(tiers[result.predictedClass]); // "large"
    console.log(result.probabilities);         // [0.13, 0.37, 0.45, 0.04]
}
```

### 3. With Cloud Consultation (Delegation)

```typescript
import {
    Router,
    extractGreenfieldFeatures,
    classifyWithEmbeddings,
    initEmbeddingClassifier,
    pickConsultationStrategy,
    ConsultationStrategy,
    verifyWithCloud,
    planWithCloud,
    type CloudCompletionFn,
} from "@vriveras/task-router";

// You provide the cloud completion function — any provider works
const complete: CloudCompletionFn = async (prompt) => {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model: "gpt-4o-mini",
            messages: [{ role: "user", content: prompt }],
        }),
    });
    const data = await res.json();
    return data.choices[0].message.content;
};

// Route + classify
const router = new Router({ modelsDir: "./models" });
await initEmbeddingClassifier("./models");

const prompt = "Add comprehensive error handling to the payment module";
const features = extractGreenfieldFeatures(prompt);
const embedding = await classifyWithEmbeddings(features, prompt, "./models");
const complexity = embedding.success ? embedding.predictedClass + 1 : 2;

// Pick strategy
const { strategy } = pickConsultationStrategy({ complexity }, /* maxTierComplexity */ 3);

if (strategy === ConsultationStrategy.PLAN_BEFORE) {
    // Cloud plans, local executes
    const plan = await planWithCloud({ task: prompt, complete });
    console.log("Plan:", plan.response);
    // → Inject plan into local model's system prompt
}

if (strategy === ConsultationStrategy.VERIFY_AFTER) {
    // Local executes first, then cloud verifies
    const localResponse = await runLocalModel(prompt); // your local model

    const verification = await verifyWithCloud({
        task: prompt,
        localResponse,
        complete,
    });

    if (!verification.accepted) {
        // Re-run on cloud — quality preserved
        const cloudResponse = await complete(prompt);
    }
}
```

### 4. With Copilot API (Free via GitHub Copilot)

```bash
# Auth via GitHub CLI (no API key needed)
gh auth login

# Run the full delegation pipeline
npm run example:copilot

# Or with a custom task
npx tsx examples/copilot-integration.ts "Fix the auth bug in login.ts"
```

The Copilot integration uses `gh auth token` for authentication and hits `api.githubcopilot.com`. No API keys or billing setup required — included with GitHub Copilot subscription.

### 5. As an HTTP Server

```bash
npm run example:server

# Route a task
curl -X POST http://localhost:3456/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Fix the typo in README.md"}'

# 4-class embedding classification
curl -X POST http://localhost:3456/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refactor the auth module"}'
```

### 6. As an OpenAI-Compatible Proxy

Routes requests transparently between local Ollama and cloud API:

```bash
LOCAL_MODEL=qwen3:30b CLOUD_MODEL=gpt-4o npm run example:openai

# Point any OpenAI-compatible client at the proxy
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Fix typo"}]}'
```

### 7. Escalation Tracking (Mid-Session)

```typescript
import { Router } from "@vriveras/task-router";

const router = new Router({ modelsDir: "./models" });

// As the session progresses, record tool calls
router.recordTurn();
router.recordToolCalls(["grep", "read_file"]);

router.recordTurn();
router.recordToolCalls(["grep", "grep", "read_file"]); // lots of searching...

// Check if the session should escalate to a larger model
const escalation = await router.checkEscalation();
if (escalation.modelClass === "large") {
    console.log("Escalating to cloud model — local model is struggling");
}
```

## For OpenClaw Integration

```typescript
import {
    Router,
    initEmbeddingClassifier,
    classifyWithEmbeddings,
    extractGreenfieldFeatures,
    pickConsultationStrategy,
    verifyWithCloud,
    ConsultationStrategy,
} from "@vriveras/task-router";

// Initialize once
const modelsDir = "./node_modules/@vriveras/task-router/models";
const router = new Router({ modelsDir });
await initEmbeddingClassifier(modelsDir);

// Per-task routing decision
async function routeTask(prompt: string) {
    // 1. Classify
    const features = extractGreenfieldFeatures(prompt);
    const result = await classifyWithEmbeddings(features, prompt, modelsDir);
    const tier = result.success
        ? (["small", "medium", "large", "cloud"] as const)[result.predictedClass]
        : "cloud";

    // 2. Pick strategy
    const complexity = result.success ? result.predictedClass + 1 : 4;
    const { strategy } = pickConsultationStrategy({ complexity }, 3);

    return { tier, strategy, probabilities: result.probabilities };
}

// Example usage in OpenClaw task handler
const { tier, strategy } = await routeTask(userPrompt);

if (tier === "cloud" || tier === "large") {
    await executeOnCloud(userPrompt);
} else {
    const localResult = await executeOnLocal(userPrompt);

    if (strategy === ConsultationStrategy.VERIFY_AFTER) {
        const check = await verifyWithCloud({
            task: userPrompt,
            localResponse: localResult,
            complete: openclawCloudFn,
        });
        if (!check.accepted) {
            await executeOnCloud(userPrompt); // fallback
        }
    }
}
```

## ONNX Models

All models run offline via `onnxruntime-node`. No API calls during classification.

| Model | Size | Purpose | Accuracy |
|-------|------|---------|----------|
| `greenfield_gbm.onnx` | 6 KB | Prompt-only binary (small/large) | 65% |
| `escalation_gbm.onnx` | 28 KB | Trajectory escalation detector | — |
| `brownfield_gbm.onnx` | 28 KB | Multi-turn session classifier | — |
| `local_router_embed.onnx` | 1.5 MB | 4-class GBM with embeddings | **84%** |
| `all-MiniLM-L6-v2.onnx` | 86 MB | Sentence embedding model | — |

### 34 Greenfield Features

Extracted from the prompt text alone (no API calls):

- **Lexical**: word count, line count, code density, inline code blocks
- **Requirement**: keywords (must, should, return, throw, implement, etc.)
- **Structure**: function signatures, examples, doctest lines
- **Knowledge base**: TF-IDF similarity to bug-fix and spec-writing corpora

### Training Data

The classifier was trained on **460K synthetic samples** + **1,830 real anonymized CLI prompts**:
- v1: 95K clean, tier-correlated prompts
- v2: 183K with hard negatives and deceptive simples
- v3: 182K with AI-delegated patterns and investigation chains
- Validation: 486 separate samples (never trained on)

## Custom Logger

```typescript
import { setLogger, silentLogger } from "@vriveras/task-router";

// Inject your own
setLogger({
    info: (msg) => myLogger.info(msg),
    warning: (msg) => myLogger.warn(msg),
    error: (msg) => myLogger.error(msg),
    debug: (msg) => myLogger.debug(msg),
});

// Or silence entirely
setLogger(silentLogger());
```

## Empirical Results

From two full 70-task SWE benchmarks comparing local vs cloud:

| Model | Quality (vs cloud) | Economics | Root Cause |
|-------|-------------------|-----------|------------|
| **qwen3:30b** | 41% | Net-negative (-$39.80) | Poor quality → rework |
| **nemotron-3-super (86GB)** | **82%** | Near-break-even (-$1.57) | Orchestrator narration |
| **MSR Fara 9B (fine-tuned)** | ~95% | **77% savings** | Purpose-trained SLM |

Key finding: **pure routing is net-negative** — one-shot decisions without recovery create quality cliffs. The delegation pattern (this library) enables recovery via cloud verification, bringing economics toward the MSR target of 77% savings.

See the full analysis: [Delegation Economics Results](https://github.com/vriveras/router-testing/blob/main/docs/delegation-economics-results.md)

## Testing

```bash
npm test              # 57 tests (unit + E2E + consultation)
npm run example:basic # Offline routing demo
npm run example:copilot # Live Copilot API demo (requires gh auth login)
```

### Test Coverage

| File | Tests | What |
|------|-------|------|
| `router.test.ts` | 25 | Tool classification, features, TF-IDF, escalation, Router |
| `consultation.test.ts` | 12 | Strategy picker, verify/plan, retries, fail-open |
| `e2e.test.ts` | 20 | Full pipeline: ONNX → features → routing → escalation → delegation |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COPILOT_MODEL` | `claude-haiku-4.5` | Model for task execution |
| `COPILOT_CONSULT_MODEL` | `claude-haiku-4.5` | Model for verification/planning |
| `COPILOT_GITHUB_TOKEN` | `gh auth token` | GitHub token for Copilot API |
| `LOCAL_URL` | `http://127.0.0.1:11434/v1` | Ollama API URL (proxy mode) |
| `CLOUD_URL` | `https://api.openai.com/v1` | Cloud API URL (proxy mode) |
| `CLOUD_API_KEY` | — | Cloud API key (proxy mode) |
| `LOCAL_MODEL` | `qwen3:30b` | Local model name (proxy mode) |
| `CLOUD_MODEL` | `gpt-4o` | Cloud model name (proxy mode) |

## Project Structure

```
task-router/
├── src/
│   ├── index.ts              # All exports
│   ├── router.ts             # Main Router class (3-layer pipeline)
│   ├── embeddingClassifier.ts # MiniLM + GBM classifier (84% acc)
│   ├── cloudConsultation.ts   # verify-after, plan-before, strategy picker
│   ├── greenfieldFeatures.ts  # 34 prompt-only features
│   ├── escalationFeatures.ts  # Trajectory tool-call features
│   ├── escalationTracker.ts   # Stateful escalation (sticky)
│   ├── onnxInference.ts       # ONNX runtime wrapper
│   ├── tfidf.ts               # TF-IDF vectorizer
│   ├── toolClassification.ts  # Tool name categorization
│   ├── errors.ts              # Error codes
│   ├── logger.ts              # Injectable logger
│   └── helpers.ts             # Utilities
├── models/                    # ONNX models (offline, no API calls)
├── examples/
│   ├── basic-routing.ts       # Offline routing demo
│   ├── copilot-integration.ts # Live Copilot API delegation
│   ├── express-server.ts      # HTTP API server
│   └── openai-compatible.ts   # OpenAI proxy (local ↔ cloud)
├── test/
│   ├── router.test.ts         # Unit tests
│   ├── consultation.test.ts   # Consultation tests
│   └── e2e.test.ts            # Integration tests
└── docs/
    └── architecture.md        # Detailed architecture docs
```

## License

MIT
