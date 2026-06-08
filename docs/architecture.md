# Architecture

## Overview

Task Router is a standalone task complexity classifier that routes AI agent
requests between local (small/fast) and cloud (large/expensive) models.
All classification runs locally via ONNX Runtime — no external API calls.

## Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    User Prompt                                │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│  Greenfield Features (34)│     │  MiniLM Embeddings (384) │
│                          │     │  all-MiniLM-L6-v2.onnx   │
│  • spec_word_count       │     │                          │
│  • req_keywords          │     │  Simple hash tokenizer → │
│  • inline_code           │     │  ONNX → mean pool →     │
│  • distinct_symbols      │     │  L2 normalize            │
│  • doctest_lines         │     │                          │
│  • has_examples          │     │  384-dim output          │
│  • signature_args        │     └─────────┬────────────────┘
│  • 11 bug KB lex feats   │               │
│  • 3 bug KB TF-IDF feats │               │
│  • 10 spec KB lex feats  │               │
│  • 3 spec KB TF-IDF feats│               │
└──────────┬───────────────┘               │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────┐
│                   Feature Concatenation                    │
│                34 text + 384 embedding = 418 dims          │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌────────────────┐
    │ Greenfield│  │ Brownfield│  │ Embed Classifier│
    │ GBM      │  │ GBM       │  │ (4-class GBM)   │
    │ Binary   │  │ Binary    │  │ small/med/large/ │
    │ 34 feats │  │ 34 feats  │  │ cloud            │
    │          │  │           │  │ 418 feats        │
    └────┬─────┘  └─────┬─────┘  └───────┬──────────┘
         │              │                │
         ▼              ▼                ▼
    P(large)       P(large)        [P(small), P(med),
                                    P(large), P(cloud)]
```

## Escalation Pipeline

```
┌──────────────────────────────────────────┐
│           Conversation Trajectory         │
│  [user, assistant+tools, user, ...]       │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Escalation Features (6)     │
│  • n_turns                   │
│  • n_tool_calls              │
│  • n_grep                    │
│  • n_read                    │
│  • n_edit                    │
│  • n_run                     │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Escalation GBM          │
│  Binary classifier       │
│                          │
│  Heuristic fallback:     │
│  score = (edit+run)/5    │
│        + grep/6          │
│        + max(0,turns-2)/3│
│  escalate if score >= 1  │
└──────────┬───────────────┘
           │
           ▼
     Override → "large"
     (sticky per session)
```

## Tool Classification

Tool names are classified into 4 categories for escalation tracking:

| Category | Patterns |
|----------|----------|
| `grep` | grep, search |
| `read` | read, view, cat |
| `edit` | edit, write, patch |
| `run` | run, shell, exec, bash |

## Knowledge Bases

Two knowledge bases provide domain-specific TF-IDF features:

- **bugKnowledgeBase** — Bug-related patterns (assertions, exceptions, loops,
  recursion, I/O, concurrency, datetime, regex, math, collections, strings)
- **specKnowledgeBase** — Specification patterns (modal verbs must/should/may,
  NFRs like performance/security/reliability, scope markers, example markers)

## Model Files

All models are in the `models/` directory:

| File | Type | Input | Output |
|------|------|-------|--------|
| `greenfield_gbm.onnx` | GBM | 34 text features | P(large) |
| `escalation_gbm.onnx` | GBM | 6 trajectory features | P(escalate) |
| `brownfield_gbm.onnx` | GBM | 34 text features | P(large) |
| `local_router_embed.onnx` | GBM | 418 features (34 text + 384 embed) | 4-class probs |
| `embeddings/all-MiniLM-L6-v2.onnx` | Transformer | Token IDs (128 seq) | 384-dim embeddings |
| `embeddings/tokenizer.json` | Vocab | — | Token vocabulary |

Each `.onnx` model has a companion `.json` manifest specifying feature order,
threshold, and optional SHA-256 integrity hash.
