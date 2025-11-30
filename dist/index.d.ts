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
}
export declare class IndicTransliterator {
    private encoder;
    private decoder;
    private vocab;
    private modelPath;
    private beamWidth;
    private maxLen;
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
