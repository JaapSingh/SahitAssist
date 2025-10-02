from __future__ import annotations
import json
import time
from typing import List, Dict, Any, Callable, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .schema import SentenceRecord, TokenAnnotation
from .config import AzureConfig

JSON_SCHEMA_SNIPPET = """
{
  "type": "object",
  "properties": {
    "tokens": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "surface": {"type": "string"},
          "lemma": {"type": ["string","null"]},
          "pos": {"type": ["string","null"]},
          "literal_en": {"type": ["string","null"]},
          "note": {"type": ["string","null"]}
        },
        "required": ["surface"]
      }
    },
    "literal_sentence_en": {"type": ["string","null"]},
    "sentence_translation_en": {"type": ["string","null"]},
    "context_explanation_en": {"type": ["string","null"]}
  },
  "required": ["tokens","literal_sentence_en","sentence_translation_en","context_explanation_en"]
}
"""

BASE_RULES = """You are a careful Punjabi↔English literary translator and annotator.
Rules:
- Output ONLY valid JSON that matches the schema provided.
- Preserve nuance in Sikh texts: capitalize ‘Guru’, ‘Bani’, ‘Waheguru’.
- Provide literal word tokens in order; mark idioms with notes.
- Fill every field; use null if unknown.
"""

PROMPT_TEMPLATE = """{rules}

{context_block}

Schema:
{schema}

Target Sentence:
\"\"\"{text}\"\"\"
Return JSON ONLY, no markdown fences.
"""

class TranslationError(Exception):
    pass

def build_payload(azure: AzureConfig, prompt: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are a Punjabi to English scholarly translation engine."},
            {"role": "user", "content": prompt},
        ],
        "temperature": azure.temperature,
        "response_format": {"type": "json_object"},
    }

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((httpx.HTTPError, TranslationError)),
)
async def translate_batch(
    azure: AzureConfig,
    client: httpx.AsyncClient,
    batch: List[SentenceRecord],
    retrieval_context_fn: Optional[Callable[[SentenceRecord], str]] = None,
) -> List[SentenceRecord]:
    updated: List[SentenceRecord] = []
    for rec in batch:
        context_block = ""
        if retrieval_context_fn:
            context_block = retrieval_context_fn(rec)
        context_block = context_block.strip()
        if not context_block:
            context_block = "Context: (none)"
        prompt = PROMPT_TEMPLATE.format(
            rules=BASE_RULES,
            context_block=context_block,
            schema=JSON_SCHEMA_SNIPPET,
            text=rec.normalized
        )
        payload = build_payload(azure, prompt)
        url = f"{azure.endpoint}/openai/deployments/{azure.deployment}/chat/completions?api-version={azure.api_version}"
        try:
            resp = await client.post(
                url,
                headers={"api-key": azure.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise TranslationError(str(e)) from e

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except Exception as e:
            raise TranslationError(f"Invalid JSON from model: {e} :: {data}") from e

        tokens_raw = parsed.get("tokens", [])
        tokens = []
        for t in tokens_raw:
            tokens.append(
                TokenAnnotation(
                    surface=t.get("surface", ""),
                    lemma=t.get("lemma"),
                    pos=t.get("pos"),
                    literal_en=t.get("literal_en"),
                    note=t.get("note"),
                )
            )
        rec.tokens = tokens
        rec.literal_sentence_en = parsed.get("literal_sentence_en")
        rec.sentence_translation_en = parsed.get("sentence_translation_en")
        rec.context_explanation_en = parsed.get("context_explanation_en")
        rec.metadata["translated_at"] = time.time()
        rec.metadata["checksum"] = rec.compute_checksum()
        updated.append(rec)
    return updated