# SahitAssist

SahitAssist is a pipeline to transform long Punjabi (Anmol Lipi) literary works (e.g., 350-page books) into richly annotated English JSON translations.

Each sentence yields:
1. `tokens`: Ordered literal token breakdown (with optional lemma, POS, literal English, idiom notes)
2. `literal_sentence_en`: Literal (word-by-word) sentence rendering
3. `sentence_translation_en`: Natural, nuanced English translation
4. `context_explanation_en`: Deeper / cultural / theological context

## Key Capabilities

- Anmol Lipi → Unicode Gurmukhi normalization
- Robust segmentation (pages/paragraphs → sentences)
- Azure OpenAI GPT-based translation with enforced JSON schema
- Vector embeddings + semantic retrieval for:
  - Consistent terminology (Guru, Bani, Waheguru, Naam, etc.)
  - Reuse of near-duplicate lines (cost savings)
  - Context augmentation for better nuance
- Reuse threshold to automatically copy earlier validated translations
- Manifest-based checkpointing (resume-safe)
- Index persistence (`vectors.npy`, `meta.jsonl`)
- Optional retrieval-controlled prompting (`--use-context`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment Variables

Create a `.env` (see `.env.example`):

```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_DEPLOYMENT_NAME=gpt5-translation
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=embeddings-gpt5

TRANSLATION_MODEL_TEMPERATURE=0.2
EMBEDDING_MODEL_DIM=1536

RETRIEVAL_TOP_K=5
RETRIEVAL_REUSE_THRESHOLD=0.95
RETRIEVAL_MAX_CONTEXT_CHARS=1200
```

## CLI Overview

Install (editable) first, then:

```bash
sahitassist normalize raw/book1_anmol.txt work/book1_normalized.txt
sahitassist segment work/book1_normalized.txt work/book1_sentences.jsonl --book-id book1
sahitassist build-index work/book1_sentences.jsonl outputs/book1 --book-id book1
sahitassist translate --input work/book1_sentences.jsonl --output outputs/book1 --use-context
```

Or full pipeline (if you already have extracted Anmol Lipi text):

```bash
sahitassist pipeline --anmol-text raw/book1_anmol.txt --book-id book1 --output outputs/book1 --use-context
```

If you start from a PDF:

```bash
sahitassist pipeline --pdf input/book1.pdf --book-id book1 --output outputs/book1 --use-context
```

## Output

Translations accumulate in:
```
outputs/book1/manifest.jsonl
outputs/book1/book1.jsonl   # final exported copy
outputs/book1/index/        # embeddings index
```

Sample JSONL record:

```json
{
  "book_id": "book1",
  "page_number": 0,
  "paragraph_index": 3,
  "sentence_index": 2,
  "original": "ਗੁਰੂ ਦੀ ਕਿਰਪਾ ਨਾਲ ਸਭ ਹੋਇਆ।",
  "normalized": "ਗੁਰੂ ਦੀ ਕਿਰਪਾ ਨਾਲ ਸਭ ਹੋਇਆ।",
  "tokens": [
    {"surface": "ਗੁਰੂ", "lemma": "ਗੁਰੂ", "pos": null, "literal_en": "Guru", "note": null}
  ],
  "literal_sentence_en": "By Guru's grace all happened.",
  "sentence_translation_en": "All has occurred through the Guru's Grace.",
  "context_explanation_en": "Expresses complete attribution to divine guidance.",
  "metadata": {
    "uuid": "....",
    "translated_at": 1735853012.23,
    "checksum": "sha256:....",
    "reused_from": null,
    "similarity": null
  }
}
```

If reused from a prior sentence, `reused_from` and `similarity` are filled.

## Retrieval

When `--use-context`:
1. Embed all sentences (build-index step).
2. For each sentence:
   - Retrieve top-k similar earlier sentences.
   - If similarity ≥ reuse threshold → reuse prior translation directly.
   - Else inject a concise context block into the translation prompt.

Prompt Snippet:

```
Context for consistency (similar prior passages):
1. Original: ਗੁਰੂ ਦੀ ਕਿਰਪਾ ਨਾਲ ਸਭ ਹੋਇਆ।
   Translation: All occurred by the Guru's Grace.
   Notes: (n/a)

Target Sentence:
"""ਗੁਰੂ ਦੀ ਕਿਰਪਾ ਅਗਿਆਨਤਾ ਦੂਰ ਕਰਦੀ ਹੈ।"""
```

Then JSON schema instructions follow, forcing structured output.

## Reindexing

If you change embedding model:
- Remove or rename `outputs/book1/index/` and rebuild with `build-index`.

## Testing

```bash
pytest -q
```

Lint (if you install `dev` extras):
```bash
ruff check .
```

## Future Enhancements (Ideas)

- Glossary consensus builder (lemma → canonical translation)
- Human-in-the-loop review tool
- Outlier detection for inconsistent translations
- FastAPI service front-end
- FAISS or pgvector backend for large multi-book corpora

## License

MIT (see LICENSE)

## Disclaimer

The included Anmol Lipi mapping is partial and provided only as a placeholder. Expand and verify for production accuracy.
