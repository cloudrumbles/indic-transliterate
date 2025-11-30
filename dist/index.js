"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.IndicTransliterator = void 0;
const ort = __importStar(require("onnxruntime-node"));
const path_1 = __importDefault(require("path"));
const fs_1 = __importDefault(require("fs"));
const https_1 = __importDefault(require("https"));
const readline_1 = __importDefault(require("readline"));
const fs_2 = require("fs");
const unzipper_1 = __importDefault(require("unzipper"));
// Dictionary download URL (zip file containing all language dicts)
const DICT_ZIP_URL = 'https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/word_prob_dicts.zip';
class IndicTransliterator {
    constructor(options = {}) {
        this.encoder = null;
        this.decoder = null;
        this.vocab = null;
        this.wordProbDicts = new Map();
        this.modelPath = options.modelPath || path_1.default.join(__dirname, '../models');
        this.beamWidth = options.beamWidth ?? 4;
        this.maxLen = options.maxLen ?? 20;
        this.rescore = options.rescore ?? false;
        this.dictPath = options.dictPath || path_1.default.join(this.modelPath, 'word_prob_dicts');
        this.rescoreAlpha = options.rescoreAlpha ?? 0.9;
    }
    /**
     * Check if models are loaded.
     */
    get isInitialized() {
        return this.encoder !== null && this.decoder !== null && this.vocab !== null;
    }
    /**
     * Get list of supported language codes.
     */
    getSupportedLanguages() {
        return ['as', 'bn', 'brx', 'gom', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'si', 'ta', 'te', 'ur'];
    }
    /**
     * Release ONNX sessions. Call this when done to free memory.
     */
    async dispose() {
        if (this.encoder) {
            await this.encoder.release();
            this.encoder = null;
        }
        if (this.decoder) {
            await this.decoder.release();
            this.decoder = null;
        }
        this.vocab = null;
        this.wordProbDicts.clear();
    }
    /**
     * Check if word probability dictionary is available for a language.
     */
    hasDictionary(langCode) {
        const dictFile = path_1.default.join(this.dictPath, `${langCode}_word_prob_dict.json`);
        return fs_1.default.existsSync(dictFile);
    }
    /**
     * Load word probability dictionary for a language.
     * Uses streaming to handle large dictionaries (1GB+).
     * Called automatically when rescore=true.
     */
    async loadDictionary(langCode) {
        if (this.wordProbDicts.has(langCode)) {
            return this.wordProbDicts.get(langCode);
        }
        const dictFile = path_1.default.join(this.dictPath, `${langCode}_word_prob_dict.json`);
        if (!fs_1.default.existsSync(dictFile)) {
            throw new Error(`Word probability dictionary not found for '${langCode}'. ` +
                `Run downloadDictionary('${langCode}') first, or disable rescoring.`);
        }
        // Stream-parse the JSON file line by line to avoid string length limits
        // The dictionary is a flat JSON object: {"word1": prob1, "word2": prob2, ...}
        const dict = new Map();
        const rl = readline_1.default.createInterface({
            input: (0, fs_2.createReadStream)(dictFile, { encoding: 'utf-8' }),
            crlfDelay: Infinity,
        });
        // Regex to extract key-value pairs from JSON lines
        // Matches: "word": 0.00123 or "word": 1.23e-05
        const kvRegex = /^\s*"([^"]+)":\s*([0-9.eE+-]+),?\s*$/;
        for await (const line of rl) {
            const match = line.match(kvRegex);
            if (match) {
                // Unescape Unicode sequences like \u0b92 to actual characters
                const word = match[1].replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
                const prob = parseFloat(match[2]);
                if (!isNaN(prob)) {
                    dict.set(word, prob);
                }
            }
        }
        this.wordProbDicts.set(langCode, dict);
        return dict;
    }
    /**
     * Rescore beam search results using word probability dictionary.
     * Uses interpolation: final = alpha * model_score + (1-alpha) * dict_prob
     * Words not in dictionary get score 0 (matching original AI4Bharat behavior)
     */
    rescoreResults(candidates, dict) {
        if (candidates.length === 0)
            return candidates;
        // Get model scores (already log probs, convert to probs for normalization)
        const modelProbs = candidates.map((c) => Math.exp(c.score));
        const totalModelProb = modelProbs.reduce((a, b) => a + b, 0);
        // Get dictionary probabilities (only for words in dict)
        const dictProbs = candidates.map((c) => dict.get(c.word) || 0);
        const totalDictProb = dictProbs.reduce((a, b) => a + b, 0);
        // Normalize and interpolate
        const alpha = this.rescoreAlpha;
        const rescored = candidates.map((c, i) => {
            const inDict = dict.has(c.word);
            // Words NOT in dictionary get score 0 (matching original AI4Bharat)
            if (!inDict) {
                return { word: c.word, score: 0 };
            }
            const normModelProb = totalModelProb > 0 ? modelProbs[i] / totalModelProb : 0;
            const normDictProb = totalDictProb > 0 ? dictProbs[i] / totalDictProb : 0;
            // Interpolated score (higher is better)
            const finalScore = alpha * normModelProb + (1 - alpha) * normDictProb;
            return { word: c.word, score: finalScore };
        });
        // Sort by final score descending
        rescored.sort((a, b) => b.score - a.score);
        return rescored;
    }
    /**
     * Download word probability dictionary for a language.
     * Downloads from AI4Bharat's IndicXlit releases and extracts the specific language dict.
     *
     * @param langCode Language code (e.g., 'ta', 'hi')
     * @param onProgress Optional progress callback (bytes downloaded, total bytes)
     */
    async downloadDictionary(langCode, onProgress) {
        const supportedLangs = this.getSupportedLanguages();
        if (!supportedLangs.includes(langCode)) {
            throw new Error(`Unsupported language: ${langCode}. Valid: ${supportedLangs.join(', ')}`);
        }
        // Create dict directory if needed
        if (!fs_1.default.existsSync(this.dictPath)) {
            fs_1.default.mkdirSync(this.dictPath, { recursive: true });
        }
        const destFile = path_1.default.join(this.dictPath, `${langCode}_word_prob_dict.json`);
        const targetFileName = `word_prob_dicts/${langCode}_word_prob_dict.json`;
        return new Promise((resolve, reject) => {
            const makeRequest = (requestUrl, redirectCount = 0) => {
                if (redirectCount > 10) {
                    reject(new Error('Too many redirects'));
                    return;
                }
                https_1.default
                    .get(requestUrl, (res) => {
                    // Handle redirects
                    if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307) {
                        const redirectUrl = res.headers.location;
                        if (redirectUrl) {
                            makeRequest(redirectUrl, redirectCount + 1);
                            return;
                        }
                    }
                    if (res.statusCode !== 200) {
                        reject(new Error(`Failed to download dictionary zip: HTTP ${res.statusCode}`));
                        return;
                    }
                    const totalBytes = parseInt(res.headers['content-length'] || '0', 10);
                    let downloadedBytes = 0;
                    res.on('data', (chunk) => {
                        downloadedBytes += chunk.length;
                        onProgress?.(downloadedBytes, totalBytes);
                    });
                    let found = false;
                    res
                        .pipe(unzipper_1.default.Parse())
                        .on('entry', (entry) => {
                        if (entry.path === targetFileName) {
                            found = true;
                            entry.pipe((0, fs_2.createWriteStream)(destFile))
                                .on('finish', () => resolve())
                                .on('error', reject);
                        }
                        else {
                            entry.autodrain();
                        }
                    })
                        .on('close', () => {
                        if (!found) {
                            reject(new Error(`Dictionary for '${langCode}' not found in zip`));
                        }
                    })
                        .on('error', reject);
                })
                    .on('error', reject);
            };
            makeRequest(DICT_ZIP_URL);
        });
    }
    /**
     * Initialize the ONNX models.
     * This is called automatically on first request, but can be called manually to warm up.
     */
    async initialize() {
        if (this.encoder)
            return;
        try {
            // Load Vocab
            const vocabPath = path_1.default.join(this.modelPath, 'vocab.json');
            if (!fs_1.default.existsSync(vocabPath)) {
                throw new Error(`Vocab file not found at ${vocabPath}`);
            }
            this.vocab = JSON.parse(fs_1.default.readFileSync(vocabPath, 'utf-8'));
            // Load Models
            const encoderPath = path_1.default.join(this.modelPath, 'indicxlit_encoder.onnx');
            const decoderPath = path_1.default.join(this.modelPath, 'indicxlit_decoder_v2.onnx');
            if (!fs_1.default.existsSync(encoderPath) || !fs_1.default.existsSync(decoderPath)) {
                throw new Error(`ONNX models not found at ${this.modelPath}`);
            }
            // Use CPU
            const options = { executionProviders: ['cpu'] };
            this.encoder = await ort.InferenceSession.create(encoderPath, options);
            this.decoder = await ort.InferenceSession.create(decoderPath, options);
        }
        catch (error) {
            throw new Error(`Failed to load transliteration models: ${error.message}`);
        }
    }
    tokenize(word, langCode) {
        if (!this.vocab)
            throw new Error('Vocab not loaded');
        if (langCode === 'en') {
            throw new Error(`Cannot transliterate to English. This library transliterates FROM English TO Indic scripts.`);
        }
        const langTag = `__${langCode}__`;
        // Check if language is supported
        if (!this.vocab.src.includes(langTag)) {
            const supported = this.vocab.src
                .filter(t => t.startsWith('__') && t.endsWith('__') && t !== '__en__')
                .map(t => t.slice(2, -2))
                .join(', ');
            throw new Error(`Language code '${langCode}' not supported. Valid codes: ${supported}`);
        }
        const chars = word.toLowerCase().split('').join(' ');
        const text = `${langTag} ${chars}`;
        const tokens = text.split(' ');
        const indices = tokens.map(token => {
            const idx = this.vocab.src.indexOf(token);
            return idx !== -1 ? idx : this.vocab.special_tokens.unk;
        });
        // Add EOS at the end
        indices.push(this.vocab.special_tokens.eos);
        return indices;
    }
    detokenize(indices) {
        if (!this.vocab)
            throw new Error('Vocab not loaded');
        const tokens = [];
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            // Skip start token (first EOS)
            if (i === 0)
                continue;
            // Stop at end token
            if (idx === this.vocab.special_tokens.eos)
                break;
            if (idx === this.vocab.special_tokens.bos ||
                idx === this.vocab.special_tokens.pad ||
                idx === this.vocab.special_tokens.unk) {
                continue;
            }
            const tokenStr = this.vocab.tgt[idx];
            if (tokenStr) {
                tokens.push(tokenStr);
            }
        }
        return tokens.join('').replace(/ /g, '');
    }
    /**
     * Transliterate an English word to an Indic language.
     *
     * @param word The English word to transliterate (e.g. "amma")
     * @param langCode The target language code (e.g. "ta", "hi", "ml")
     * @param count Number of candidates to return (default: 5)
     */
    async transliterate(word, langCode, count = 5) {
        if (typeof word !== 'string') {
            throw new Error(`Expected word to be a string, got ${typeof word}`);
        }
        if (typeof langCode !== 'string') {
            throw new Error(`Expected langCode to be a string, got ${typeof langCode}`);
        }
        if (!word.trim()) {
            throw new Error('Word cannot be empty');
        }
        if (!Number.isInteger(count) || count < 1) {
            throw new Error(`Expected count to be a positive integer, got ${count}`);
        }
        if (!this.encoder || !this.decoder || !this.vocab) {
            await this.initialize();
        }
        // Beam width should be at least equal to requested count for diversity
        const beamWidth = Math.max(this.beamWidth, count);
        const maxLen = this.maxLen;
        // 1. Tokenize
        const srcIndices = this.tokenize(word, langCode);
        // Prepare Encoder Input
        const srcTokensTensor = new ort.Tensor('int64', new BigInt64Array(srcIndices.map(BigInt)), [1, srcIndices.length]);
        // Run Encoder
        const encoderResults = await this.encoder.run({ src_tokens: srcTokensTensor });
        const encoderOut = encoderResults.encoder_out;
        // 2. Beam Search
        const startToken = this.vocab.special_tokens.eos;
        let beams = [{
                score: 0.0,
                tokens: [startToken],
                finished: false
            }];
        const finalCandidates = [];
        const minLen = 1;
        for (let step = 0; step < maxLen; step++) {
            const candidates = [];
            for (const beam of beams) {
                if (beam.finished)
                    continue;
                const prevTokensTensor = new ort.Tensor('int64', new BigInt64Array(beam.tokens.map(BigInt)), [1, beam.tokens.length]);
                const decoderResults = await this.decoder.run({
                    prev_tokens: prevTokensTensor,
                    encoder_out: encoderOut
                });
                const decoderOutData = decoderResults.decoder_out.data;
                const vocabSize = this.vocab.tgt.length;
                const offset = (beam.tokens.length - 1) * vocabSize;
                const logits = new Float32Array(vocabSize);
                for (let i = 0; i < vocabSize; i++) {
                    logits[i] = decoderOutData[offset + i];
                }
                // Apply min_len constraint
                if (beam.tokens.length - 1 < minLen) {
                    logits[this.vocab.special_tokens.eos] = -Infinity;
                }
                // Log Softmax
                const maxLogit = Math.max(...logits);
                let sumExp = 0;
                for (let i = 0; i < vocabSize; i++) {
                    sumExp += Math.exp(logits[i] - maxLogit);
                }
                const logSumExp = Math.log(sumExp);
                // Top K
                const nextCandidates = [];
                for (let i = 0; i < vocabSize; i++) {
                    const logProb = (logits[i] - maxLogit) - logSumExp;
                    nextCandidates.push({ idx: i, logProb });
                }
                nextCandidates.sort((a, b) => b.logProb - a.logProb);
                const topK = nextCandidates.slice(0, beamWidth * 2);
                for (const cand of topK) {
                    const newScore = beam.score + cand.logProb;
                    const newTokens = [...beam.tokens, cand.idx];
                    const isEos = cand.idx === this.vocab.special_tokens.eos;
                    if (isEos) {
                        finalCandidates.push({ score: newScore, tokens: newTokens, finished: true });
                    }
                    else {
                        candidates.push({ score: newScore, tokens: newTokens, finished: false });
                    }
                }
            }
            candidates.sort((a, b) => b.score - a.score);
            beams = candidates.slice(0, beamWidth);
            if (beams.length === 0)
                break;
            // Only stop early if we have enough finished candidates AND
            // the best unfinished beam's score is worse than all finished scores
            if (finalCandidates.length >= beamWidth) {
                const worstFinishedScore = Math.min(...finalCandidates.map((c) => c.score));
                const bestUnfinishedScore = beams[0]?.score ?? -Infinity;
                // Apply length penalty estimate to unfinished beam
                const estimatedLen = beams[0]?.tokens.length + 3; // assume 3 more tokens
                const lp = Math.pow((5 + estimatedLen) / 6, 1.0);
                const adjustedUnfinished = bestUnfinishedScore / lp;
                const adjustedFinished = worstFinishedScore / Math.pow((5 + finalCandidates[0].tokens.length) / 6, 1.0);
                if (adjustedUnfinished < adjustedFinished)
                    break;
            }
        }
        for (const beam of beams) {
            if (!beam.finished)
                finalCandidates.push(beam);
        }
        // Length penalty normalization (Google NMT)
        const lenPenalty = 1.0;
        const scoredResults = finalCandidates.map((cand) => {
            const len = Math.max(1, cand.tokens.length - 1);
            const lp = Math.pow((5 + len) / 6, lenPenalty);
            return { ...cand, normalizedScore: cand.score / lp };
        });
        scoredResults.sort((a, b) => b.normalizedScore - a.normalizedScore);
        // Get more candidates than requested for rescoring
        const rescoreCount = this.rescore ? Math.max(count * 2, 10) : count;
        const topResults = scoredResults.slice(0, rescoreCount);
        // Detokenize to get words
        const wordsWithScores = topResults.map((res) => ({
            word: this.detokenize(res.tokens),
            score: res.normalizedScore,
        }));
        // Remove duplicates (keep highest scored)
        const seen = new Set();
        const uniqueResults = wordsWithScores.filter((r) => {
            if (seen.has(r.word))
                return false;
            seen.add(r.word);
            return true;
        });
        // Apply rescoring if enabled
        if (this.rescore) {
            // Auto-download dictionary if not present
            if (!this.hasDictionary(langCode)) {
                await this.downloadDictionary(langCode);
            }
            const dict = await this.loadDictionary(langCode);
            const rescored = this.rescoreResults(uniqueResults, dict);
            return rescored.slice(0, count).map((r) => r.word);
        }
        return uniqueResults.slice(0, count).map((r) => r.word);
    }
}
exports.IndicTransliterator = IndicTransliterator;
