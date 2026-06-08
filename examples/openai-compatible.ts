/**
 * OpenAI-compatible proxy that routes requests to local Ollama or cloud API
 * based on task complexity classification.
 */
import express from "express";
import { Router } from "../src/index.js";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(express.json());

const router = new Router({
    modelsDir: join(__dirname, "../models"),
});

const LOCAL_URL = process.env.LOCAL_URL || "http://127.0.0.1:11434/v1";
const CLOUD_URL = process.env.CLOUD_URL || "https://api.openai.com/v1";
const CLOUD_API_KEY = process.env.CLOUD_API_KEY || "";
const LOCAL_MODEL = process.env.LOCAL_MODEL || "qwen3:30b";
const CLOUD_MODEL = process.env.CLOUD_MODEL || "gpt-4o";

interface ChatMessage {
    role: string;
    content: string;
}

app.post("/v1/chat/completions", async (req, res) => {
    const messages: ChatMessage[] = req.body.messages || [];
    const lastUserMsg = messages.filter((m) => m.role === "user").pop();
    const prompt = lastUserMsg?.content || "";

    const decision = await router.route(prompt || "unknown");
    const useLocal = decision.modelClass === "small";

    const targetUrl = useLocal ? LOCAL_URL : CLOUD_URL;
    const targetModel = useLocal ? LOCAL_MODEL : CLOUD_MODEL;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (!useLocal && CLOUD_API_KEY) headers["Authorization"] = `Bearer ${CLOUD_API_KEY}`;

    console.log(`[${decision.modelClass}] P(large)=${decision.probabilityLarge.toFixed(2)} → ${targetModel}`);

    const upstream = await fetch(`${targetUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({ ...req.body, model: targetModel }),
    });

    const data = await upstream.json();
    res.status(upstream.status).json(data);
});

const port = Number(process.env.PORT) || 4000;
app.listen(port, () => {
    console.log(`OpenAI-compatible router on :${port}`);
    console.log(`  Local: ${LOCAL_URL} (${LOCAL_MODEL})`);
    console.log(`  Cloud: ${CLOUD_URL} (${CLOUD_MODEL})`);
});
