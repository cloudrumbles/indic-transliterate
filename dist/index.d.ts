export interface TransliterationOptions {
    /**
     * Beam width for beam search. Higher values give more candidates but are slower.
     * Default: 4
     */
    beamWidth?: number;
    /**
     * Maximum output length in characters.
     * Default: 20
     */
    maxLen?: number;
    /**
     * Path to the models directory containing .onnx files and vocab.json.
     * Default: bundled models
     */
    modelPath?: string;
    /**
     * Enable rescoring with word probability dictionaries.
     * Dictionary is auto-downloaded on first use if not present (~200MB per language).
     * Default: false
     */
    rescore?: boolean;
    /**
     * Path to directory containing word probability dictionaries.
     * Default: {modelPath}/word_prob_dicts
     */
    dictPath?: string;
    /**
     * Alpha value for rescoring interpolation.
     * final_score = alpha * model_score + (1-alpha) * dict_prob
     * Default: 0.9
     */
    rescoreAlpha?: number;
}
export declare class IndicTransliterator {
    private encoder;
    private decoder;
    private vocab;
    private modelPath;
    private beamWidth;
    private maxLen;
    private rescore;
    private dictPath;
    private rescoreAlpha;
    private wordProbDicts;
    constructor(options?: TransliterationOptions);
    /**
     * Check if models are loaded.
     */
    get isInitialized(): boolean;
    /**
     * Get list of supported language codes.
     */
    getSupportedLanguages(): string[];
    /**
     * Release ONNX sessions. Call this when done to free memory.
     */
    dispose(): Promise<void>;
    /**
     * Check if word probability dictionary is available for a language.
     */
    hasDictionary(langCode: string): boolean;
    /**
     * Load word probability dictionary for a language.
     * Uses streaming to handle large dictionaries (1GB+).
     * Called automatically when rescore=true.
     */
    private loadDictionary;
    /**
     * Rescore beam search results using word probability dictionary.
     * Uses interpolation: final = alpha * model_score + (1-alpha) * dict_prob
     * Words not in dictionary get score 0 (matching original AI4Bharat behavior)
     */
    private rescoreResults;
    /**
     * Download word probability dictionary for a language.
     * Downloads from AI4Bharat's IndicXlit releases and extracts the specific language dict.
     *
     * @param langCode Language code (e.g., 'ta', 'hi')
     * @param onProgress Optional progress callback (bytes downloaded, total bytes)
     */
    downloadDictionary(langCode: string, onProgress?: (downloaded: number, total: number) => void): Promise<void>;
    /**
     * Initialize the ONNX models.
     * This is called automatically on first request, but can be called manually to warm up.
     */
    initialize(): Promise<void>;
    private tokenize;
    private detokenize;
    /**
     * Transliterate an English word to an Indic language.
     *
     * @param word The English word to transliterate (e.g. "amma")
     * @param langCode The target language code (e.g. "ta", "hi", "ml")
     * @param count Number of candidates to return (default: 5)
     */
    transliterate(word: string, langCode: string, count?: number): Promise<string[]>;
}
