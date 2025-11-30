import * as ort from 'onnxruntime-node'
import path from 'path'
import fs from 'fs'

// Types
interface Vocab {
  src: string[]
  tgt: string[]
  special_tokens: {
    unk: number
    pad: number
    bos: number
    eos: number
  }
}

interface BeamCandidate {
  score: number
  tokens: number[]
  finished: boolean
}

export interface TransliterationOptions {
  /**
   * Beam width for beam search. Higher values give more candidates but are slower.
   * Default: 4
   */
  beamWidth?: number
  
  /**
   * Maximum output length in characters.
   * Default: 20
   */
  maxLen?: number

  /**
   * Path to the models directory containing .onnx files and vocab.json.
   * Default: bundled models
   */
  modelPath?: string
}

export class IndicTransliterator {
  private encoder: ort.InferenceSession | null = null
  private decoder: ort.InferenceSession | null = null
  private vocab: Vocab | null = null
  private modelPath: string
  private beamWidth: number
  private maxLen: number

  constructor(options: TransliterationOptions = {}) {
    this.modelPath = options.modelPath || path.join(__dirname, '../models')
    this.beamWidth = options.beamWidth ?? 4
    this.maxLen = options.maxLen ?? 20
  }

  /**
   * Check if models are loaded.
   */
  get isInitialized(): boolean {
    return this.encoder !== null && this.decoder !== null && this.vocab !== null
  }

  /**
   * Get list of supported language codes.
   */
  getSupportedLanguages(): string[] {
    return ['as', 'bn', 'brx', 'gom', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'si', 'ta', 'te', 'ur']
  }

  /**
   * Release ONNX sessions. Call this when done to free memory.
   */
  async dispose(): Promise<void> {
    if (this.encoder) {
      await this.encoder.release()
      this.encoder = null
    }
    if (this.decoder) {
      await this.decoder.release()
      this.decoder = null
    }
    this.vocab = null
  }

  /**
   * Initialize the ONNX models.
   * This is called automatically on first request, but can be called manually to warm up.
   */
  async initialize(): Promise<void> {
    if (this.encoder) return

    try {
      // Load Vocab
      const vocabPath = path.join(this.modelPath, 'vocab.json')
      if (!fs.existsSync(vocabPath)) {
        throw new Error(`Vocab file not found at ${vocabPath}`)
      }
      this.vocab = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'))

      // Load Models
      const encoderPath = path.join(this.modelPath, 'indicxlit_encoder.onnx')
      const decoderPath = path.join(this.modelPath, 'indicxlit_decoder_v2.onnx')
      
      if (!fs.existsSync(encoderPath) || !fs.existsSync(decoderPath)) {
        throw new Error(`ONNX models not found at ${this.modelPath}`)
      }

      // Use CPU
      const options = { executionProviders: ['cpu'] }
      
      this.encoder = await ort.InferenceSession.create(encoderPath, options)
      this.decoder = await ort.InferenceSession.create(decoderPath, options)
    } catch (error) {
      throw new Error(`Failed to load transliteration models: ${(error as Error).message}`)
    }
  }

  private tokenize(word: string, langCode: string): number[] {
    if (!this.vocab) throw new Error('Vocab not loaded')
    
    if (langCode === 'en') {
      throw new Error(`Cannot transliterate to English. This library transliterates FROM English TO Indic scripts.`)
    }
    
    const langTag = `__${langCode}__`
    
    // Check if language is supported
    if (!this.vocab.src.includes(langTag)) {
      const supported = this.vocab.src
        .filter(t => t.startsWith('__') && t.endsWith('__') && t !== '__en__')
        .map(t => t.slice(2, -2))
        .join(', ')
      throw new Error(`Language code '${langCode}' not supported. Valid codes: ${supported}`)
    }

    const chars = word.toLowerCase().split('').join(' ')
    const text = `${langTag} ${chars}`
    const tokens = text.split(' ')
    
    const indices = tokens.map(token => {
      const idx = this.vocab!.src.indexOf(token)
      return idx !== -1 ? idx : this.vocab!.special_tokens.unk
    })
    
    // Add EOS at the end
    indices.push(this.vocab.special_tokens.eos)
    
    return indices
  }

  private detokenize(indices: number[]): string {
    if (!this.vocab) throw new Error('Vocab not loaded')
    
    const tokens: string[] = []
    
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i]
      
      // Skip start token (first EOS)
      if (i === 0) continue
      
      // Stop at end token
      if (idx === this.vocab.special_tokens.eos) break
      
      if (
        idx === this.vocab.special_tokens.bos || 
        idx === this.vocab.special_tokens.pad || 
        idx === this.vocab.special_tokens.unk
      ) {
        continue
      }
      
      const tokenStr = this.vocab.tgt[idx]
      if (tokenStr) {
        tokens.push(tokenStr)
      }
    }
    
    return tokens.join('').replace(/ /g, '')
  }

  /**
   * Transliterate an English word to an Indic language.
   * 
   * @param word The English word to transliterate (e.g. "amma")
   * @param langCode The target language code (e.g. "ta", "hi", "ml")
   * @param count Number of candidates to return (default: 5)
   */
  async transliterate(word: string, langCode: string, count = 5): Promise<string[]> {
    if (typeof word !== 'string') {
      throw new Error(`Expected word to be a string, got ${typeof word}`)
    }
    if (typeof langCode !== 'string') {
      throw new Error(`Expected langCode to be a string, got ${typeof langCode}`)
    }
    if (!word.trim()) {
      throw new Error('Word cannot be empty')
    }
    if (!Number.isInteger(count) || count < 1) {
      throw new Error(`Expected count to be a positive integer, got ${count}`)
    }
    
    if (!this.encoder || !this.decoder || !this.vocab) {
      await this.initialize()
    }
    
    // Beam width should be at least equal to requested count for diversity
    const beamWidth = Math.max(this.beamWidth, count)
    const maxLen = this.maxLen
    
    // 1. Tokenize
    const srcIndices = this.tokenize(word, langCode)
    
    // Prepare Encoder Input
    const srcTokensTensor = new ort.Tensor(
      'int64',
      new BigInt64Array(srcIndices.map(BigInt)),
      [1, srcIndices.length]
    )
    
    // Run Encoder
    const encoderResults = await this.encoder!.run({ src_tokens: srcTokensTensor })
    const encoderOut = encoderResults.encoder_out
    
    // 2. Beam Search
    const startToken = this.vocab!.special_tokens.eos
    
    let beams: BeamCandidate[] = [{
      score: 0.0,
      tokens: [startToken],
      finished: false
    }]
    
    const finalCandidates: BeamCandidate[] = []
    const minLen = 1
    
    for (let step = 0; step < maxLen; step++) {
      const candidates: BeamCandidate[] = []
      
      for (const beam of beams) {
        if (beam.finished) continue
        
        const prevTokensTensor = new ort.Tensor(
          'int64',
          new BigInt64Array(beam.tokens.map(BigInt)),
          [1, beam.tokens.length]
        )
        
        const decoderResults = await this.decoder!.run({
          prev_tokens: prevTokensTensor,
          encoder_out: encoderOut
        })
        
        const decoderOutData = decoderResults.decoder_out.data as Float32Array
        const vocabSize = this.vocab!.tgt.length
        const offset = (beam.tokens.length - 1) * vocabSize
        
        const logits = new Float32Array(vocabSize)
        for (let i = 0; i < vocabSize; i++) {
          logits[i] = decoderOutData[offset + i]
        }
        
        // Apply min_len constraint
        if (beam.tokens.length - 1 < minLen) {
          logits[this.vocab!.special_tokens.eos] = -Infinity
        }

        // Log Softmax
        const maxLogit = Math.max(...logits)
        let sumExp = 0
        for (let i = 0; i < vocabSize; i++) {
          sumExp += Math.exp(logits[i] - maxLogit)
        }
        const logSumExp = Math.log(sumExp)
        
        // Top K
        const nextCandidates: { idx: number, logProb: number }[] = []
        for (let i = 0; i < vocabSize; i++) {
          const logProb = (logits[i] - maxLogit) - logSumExp
          nextCandidates.push({ idx: i, logProb })
        }
        nextCandidates.sort((a, b) => b.logProb - a.logProb)
        
        const topK = nextCandidates.slice(0, beamWidth * 2)
        
        for (const cand of topK) {
          const newScore = beam.score + cand.logProb
          const newTokens = [...beam.tokens, cand.idx]
          const isEos = cand.idx === this.vocab!.special_tokens.eos
          
          if (isEos) {
            finalCandidates.push({ score: newScore, tokens: newTokens, finished: true })
          } else {
            candidates.push({ score: newScore, tokens: newTokens, finished: false })
          }
        }
      }
      
      candidates.sort((a, b) => b.score - a.score)
      beams = candidates.slice(0, beamWidth)
      
      if (beams.length === 0) break
      if (finalCandidates.length >= beamWidth * 2) break
    }
    
    for (const beam of beams) {
      if (!beam.finished) finalCandidates.push(beam)
    }
    
    // Length penalty normalization (Google NMT)
    const lenPenalty = 1.0
    const scoredResults = finalCandidates.map(cand => {
      const len = Math.max(1, cand.tokens.length - 1)
      const lp = Math.pow((5 + len) / 6, lenPenalty)
      return { ...cand, normalizedScore: cand.score / lp }
    })
    
    scoredResults.sort((a, b) => b.normalizedScore - a.normalizedScore)
    
    const topResults = scoredResults.slice(0, count)
    return topResults.map(res => this.detokenize(res.tokens))
  }
}

