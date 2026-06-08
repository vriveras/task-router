import express from "express";
import { Router, classifyWithEmbeddings, initEmbeddingClassifier } from "../src/index.js";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(express.json());

const modelsDir = join(__dirname, "../models");
const router = new Router({ modelsDir });

// Initialize embedding classifier
await initEmbeddingClassifier(modelsDir);

// POST /route — classify a prompt
app.post("/route", async (req, res) => {
    const { prompt } = req.body;
    if (!prompt) {
        res.status(400).json({ error: "prompt required" });
        return;
    }

    const decision = await router.route(prompt);
    res.json(decision);
});

// POST /classify — 4-class embedding classification
app.post("/classify", async (req, res) => {
    const { prompt } = req.body;
    if (!prompt) {
        res.status(400).json({ error: "prompt required" });
        return;
    }

    const result = await classifyWithEmbeddings({}, prompt, modelsDir);
    res.json(result);
});

// GET /health
app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
});

const port = Number(process.env.PORT) || 3456;
app.listen(port, () => console.log(`Task router listening on :${port}`));
