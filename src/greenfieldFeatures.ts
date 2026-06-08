/**
 * Greenfield (F1++) features — prompt-only.
 *
 * Computes features derivable from the user's prompt text alone, with no
 * repository access. Three cumulative sets:
 *   F1   — 7 spec-lex features
 *   F1+  — F1 + 11 bug-KB lex features + 3 bug-KB TF-IDF features
 *   F1++ — F1+ + 10 spec-KB features + 3 spec-KB TF-IDF features
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { TfidfVectorizer, cosineSimilarity } from "./tfidf.js";
import { getLogger } from "./logger.js";

// ── Load knowledge base data ────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const dataDir = join(__dirname, "..", "src", "data");

// Try src/data first (development), fall back to ../data (if running from dist/)
function loadJson(filename: string): Record<string, string[]> {
    try {
        return JSON.parse(readFileSync(join(dataDir, filename), "utf-8"));
    } catch {
        // When running from dist/, data is at ../../src/data relative to dist/
        const altDir = join(__dirname, "..", "data");
        return JSON.parse(readFileSync(join(altDir, filename), "utf-8"));
    }
}

const bugKnowledgeBase: Record<string, string[]> = loadJson("bugKnowledgeBase.json");
const specKnowledgeBase: Record<string, string[]> = loadJson("specKnowledgeBase.json");

// ── Constants ───────────────────────────────────────────────────────────────

const REQ_KEYWORDS = ["return", "must", "should", "raise", "throw", "compute", "implement", "given", "expect"] as const;

const ACTIVE_DOC_THRESHOLD = 0.05;

/** Maximum number of function signature arguments to count. */
const MAX_SIGNATURE_ARGS = 12;

// ── Cached TF-IDF indices ───────────────────────────────────────────────────

interface TfidfIndex {
    vectorizer: TfidfVectorizer;
    matrix: number[][];
}

function buildTfidfIndex(kb: Record<string, string[]>): TfidfIndex | null {
    const docs = Object.values(kb)
        .filter((keywords) => keywords.length > 0)
        .map((keywords) => keywords.join(" "));
    if (docs.length === 0) {
        return null;
    }
    const vectorizer = new TfidfVectorizer({ lowercase: true, ngramRange: [1, 2] });
    const matrix = vectorizer.fitTransform(docs);
    return { vectorizer, matrix };
}

let bugTfidfIndex: TfidfIndex | null | undefined;
function getBugTfidfIndex(): TfidfIndex | null {
    if (bugTfidfIndex === undefined) {
        bugTfidfIndex = buildTfidfIndex(bugKnowledgeBase);
    }
    return bugTfidfIndex;
}

let specTfidfIndex: TfidfIndex | null | undefined;
function getSpecTfidfIndex(): TfidfIndex | null {
    if (specTfidfIndex === undefined) {
        specTfidfIndex = buildTfidfIndex(specKnowledgeBase);
    }
    return specTfidfIndex;
}

// ── F1: Spec-lex features ───────────────────────────────────────────────────

function countSignatureArgs(text: string): number {
    const m = text.match(/def\s+\w+\s*\(([^)]*)\)/);
    if (!m) {
        return 0;
    }
    const args = m[1]
        .split(",")
        .map((a) => a.trim())
        .filter((a) => a.length > 0 && a !== "self" && a !== "cls");
    return Math.min(args.length, MAX_SIGNATURE_ARGS);
}

function specLexFeatures(text: string): Record<string, number> {
    const lower = text.toLowerCase();
    return {
        spec_word_count: text.split(/\s+/).filter((w) => w.length > 0).length,
        spec_n_requirement_keywords: REQ_KEYWORDS.reduce((sum, kw) => sum + countOccurrences(lower, kw), 0),
        spec_n_inline_code: (text.match(/`[^`]+`/g) ?? []).length,
        spec_n_distinct_symbol_tokens: new Set(
            (text.match(/`([A-Za-z_][A-Za-z0-9_.:]+)`/g) ?? []).map((m) => m.slice(1, -1)),
        ).size,
        spec_n_doctest_lines: (text.match(/^\s*>>>/gm) ?? []).length,
        spec_has_examples: /\b(example|for example|e\.g\.|sample (?:input|output))\b/i.test(text) ? 1 : 0,
        spec_signature_arg_count: countSignatureArgs(text),
    };
}

// ── KB lex features ─────────────────────────────────────────────────────────

function kbLexFeatures(text: string, kb: Record<string, string[]>): Record<string, number> {
    const lower = text.toLowerCase();
    const out: Record<string, number> = {};
    for (const [featName, keywords] of Object.entries(kb)) {
        out[featName] = keywords.reduce((sum, kw) => sum + countOccurrences(lower, kw.toLowerCase()), 0);
    }
    return out;
}

// ── KB TF-IDF features ─────────────────────────────────────────────────────

function kbTfidfFeatures(text: string, index: TfidfIndex | null, prefix: string): Record<string, number> {
    const out: Record<string, number> = {
        [`${prefix}_tfidf_max`]: 0,
        [`${prefix}_tfidf_mean`]: 0,
        [`${prefix}_active_docs`]: 0,
    };

    if (!text || !index) {
        return out;
    }

    const promptVec = index.vectorizer.transform(text);
    if (promptVec.every((v) => v === 0)) {
        return out;
    }

    const sims = index.matrix.map((docVec) => cosineSimilarity(docVec, promptVec));
    if (sims.length === 0) {
        return out;
    }

    out[`${prefix}_tfidf_max`] = Math.max(...sims);
    out[`${prefix}_tfidf_mean`] = sims.length > 0 ? sims.reduce((a, b) => a + b, 0) / sims.length : 0;
    out[`${prefix}_active_docs`] = sims.filter((s) => s >= ACTIVE_DOC_THRESHOLD).length;
    return out;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function countOccurrences(haystack: string, needle: string): number {
    let count = 0;
    let pos = 0;
    while ((pos = haystack.indexOf(needle, pos)) !== -1) {
        count++;
        pos += needle.length;
    }
    return count;
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Extract all 34 greenfield features from a user prompt.
 *
 * Returns a flat record with keys matching the ONNX model's expected
 * feature names.
 *
 * @param prompt - The raw user prompt text to extract features from
 * @returns A flat record mapping feature names to numeric values (spec-lex, KB-lex, and KB-TF-IDF features)
 */
export function extractGreenfieldFeatures(prompt: string): Record<string, number> {
    const text = prompt ?? "";
    const features = {
        ...specLexFeatures(text),
        ...kbLexFeatures(text, bugKnowledgeBase),
        ...kbTfidfFeatures(text, getBugTfidfIndex(), "spec_bug_kb"),
        ...kbLexFeatures(text, specKnowledgeBase),
        ...kbTfidfFeatures(text, getSpecTfidfIndex(), "spec_spec_kb"),
    };
    getLogger().debug(
        `[GreenfieldFeatures] extracted features: count=${Object.keys(features).length} wordCount=${features["spec_word_count"]} symbols=${features["spec_n_distinct_symbol_tokens"]}`,
    );
    return features;
}
