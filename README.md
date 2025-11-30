# @cloudrumbles/indic-transliterate

Node.js transliteration for 21 Indian languages using AI4Bharat's IndicXlit models via ONNX Runtime.

This is an unofficial port. Models (~47MB) are bundled with the package.

## Install

```bash
npm install @cloudrumbles/indic-transliterate
```

## Usage

```typescript
import { IndicTransliterator } from '@cloudrumbles/indic-transliterate'

const transliterator = new IndicTransliterator()

// Tamil
const tamil = await transliterator.transliterate('amma', 'ta')
// => ['அம்மா', 'அம்ம', 'ஆம்மா', ...]

// Hindi
const hindi = await transliterator.transliterate('namaste', 'hi')
// => ['नमस्ते', 'नमस्ती', ...]

// Get more candidates
const results = await transliterator.transliterate('chennai', 'ta', 10)
```

### API

**`new IndicTransliterator(options?)`**

Creates a new transliterator instance. Models are loaded lazily on first use.

```typescript
const transliterator = new IndicTransliterator({
  beamWidth: 4,   // beam search width (default: 4)
  maxLen: 20,     // max output length (default: 20)
  modelPath: '...' // custom model path (default: bundled)
})
```

**`transliterate(word, langCode, count?): Promise<string[]>`**

Transliterates a romanized word to the target script. Returns `count` candidates (default 5) ranked by likelihood.

**`getSupportedLanguages(): string[]`**

Returns array of supported language codes.

**`initialize(): Promise<void>`**

Pre-loads the ONNX models. Optional - models load automatically on first `transliterate()` call.

**`dispose(): Promise<void>`**

Releases ONNX sessions to free memory. Call when done.

**`isInitialized: boolean`**

Whether models are currently loaded.

## Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `as` | Assamese | `hi` | Hindi | `ne` | Nepali |
| `bn` | Bengali | `kn` | Kannada | `or` | Odia |
| `brx` | Bodo | `ks` | Kashmiri | `pa` | Punjabi |
| `gom` | Konkani | `mai` | Maithili | `sa` | Sanskrit |
| `gu` | Gujarati | `ml` | Malayalam | `sd` | Sindhi |
| `mr` | Marathi | `mni` | Manipuri | `si` | Sinhala |
| `ta` | Tamil | `te` | Telugu | `ur` | Urdu |

## Credits

Models and vocabulary from [IndicXlit by AI4Bharat](https://github.com/AI4Bharat/IndicXlit) (MIT License).

```bibtex
@article{Madhani2022AksharantarTB,
  title={Aksharantar: Towards building open transliteration tools for the next billion users},
  author={Yash Madhani and Sushane Parthan and Priyanka A. Bedekar and Ruchi Khapra and Vivek Seshadri and Anoop Kunchukuttan and Pratyush Kumar and Mitesh M. Khapra},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.03018}
}
```

## License

MIT
