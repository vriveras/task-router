/**
 * Minimal TF-IDF vectorizer for knowledge-base similarity scoring.
 *
 * Mirrors the subset of sklearn.feature_extraction.text.TfidfVectorizer
 * used by the ONNX routing feature extractors.
 */

// ── Tokenization ────────────────────────────────────────────────────────────

const TOKEN_RE = /[a-z0-9]+/g;

function tokenize(text: string, lowercase: boolean): string[] {
    const normalized = lowercase ? text.toLowerCase() : text;
    const matches = normalized.match(TOKEN_RE);
    return matches ?? [];
}

function generateNgrams(tokens: string[], ngramRange: [number, number]): string[] {
    const [minN, maxN] = ngramRange;
    const ngrams: string[] = [];
    for (let n = minN; n <= maxN; n++) {
        for (let i = 0; i <= tokens.length - n; i++) {
            ngrams.push(tokens.slice(i, i + n).join(" "));
        }
    }
    return ngrams;
}

// ── TfidfVectorizer ─────────────────────────────────────────────────────────

export interface TfidfVectorizerOptions {
    lowercase?: boolean;
    ngramRange?: [number, number];
}

export class TfidfVectorizer {
    private readonly lowercase: boolean;
    private readonly ngramRange: [number, number];
    private vocabulary: Map<string, number> = new Map();
    private idf: number[] = [];
    private nDocs = 0;

    /**
     * Create a new TF-IDF vectorizer.
     *
     * @param options - Vectorizer configuration (lowercase normalization, n-gram range)
     */
    constructor(options: TfidfVectorizerOptions = {}) {
        this.lowercase = options.lowercase ?? true;
        this.ngramRange = options.ngramRange ?? [1, 1];
    }

    /**
     * Build the vocabulary and compute IDF weights from a corpus of documents.
     *
     * @param documents - Array of document strings to learn vocabulary from
     */
    fit(documents: string[]): void {
        this.train(documents);
    }

    /**
     * Build the vocabulary and compute IDF weights from a corpus of documents.
     *
     * @param documents - Array of document strings to learn vocabulary from
     */
    train(documents: string[]): void {
        this.nDocs = documents.length;
        const df = new Map<string, number>();

        // Build vocabulary and document frequencies
        for (const doc of documents) {
            const ngrams = this._analyze(doc);
            const seen = new Set(ngrams);
            for (const term of seen) {
                df.set(term, (df.get(term) ?? 0) + 1);
            }
        }

        // Assign term indices and compute IDF
        this.vocabulary = new Map();
        this.idf = [];
        let idx = 0;
        for (const [term, docFreq] of df) {
            this.vocabulary.set(term, idx);
            // IDF: log(N / (1 + df))
            this.idf.push(Math.log(this.nDocs / (1 + docFreq)));
            idx++;
        }
    }

    /**
     * Transform a single text into a TF-IDF weighted vector using the learned vocabulary.
     *
     * @param text - The input text to vectorize
     * @returns A TF-IDF vector with one element per vocabulary term
     */
    transform(text: string): number[] {
        const ngrams = this._analyze(text);
        const tf = new Map<string, number>();
        for (const term of ngrams) {
            if (this.vocabulary.has(term)) {
                tf.set(term, (tf.get(term) ?? 0) + 1);
            }
        }

        const totalTerms = ngrams.length || 1;
        const vec = new Array<number>(this.vocabulary.size).fill(0);
        for (const [term, count] of tf) {
            const idx = this.vocabulary.get(term)!;
            vec[idx] = (count / totalTerms) * this.idf[idx];
        }
        return vec;
    }

    /**
     * Train on the given documents and transform them in a single pass.
     *
     * @param documents - Array of document strings to learn from and transform
     * @returns A matrix of TF-IDF vectors, one per document
     */
    fitTransform(documents: string[]): number[][] {
        this.train(documents);
        return documents.map((doc) => this.transform(doc));
    }

    private _analyze(text: string): string[] {
        const tokens = tokenize(text, this.lowercase);
        return generateNgrams(tokens, this.ngramRange);
    }
}

// ── Cosine similarity ───────────────────────────────────────────────────────

/**
 * Compute the cosine similarity between two numeric vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns A value between 0 and 1 (0 = orthogonal, 1 = identical direction)
 */
export function cosineSimilarity(a: number[], b: number[]): number {
    const len = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < len; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}
