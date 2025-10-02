from __future__ import annotations
import os
import json
from pathlib import Path
import asyncio
from typing import List, Iterable, Optional, Callable
import httpx
from tqdm import tqdm
from filelock import FileLock

from .normalization import load_anmol_lipi_mapping, normalize_text
from .segmentation import segment_text
from .schema import SentenceRecord
from .translation import translate_batch
from .config import PipelineConfig, AzureConfig, EmbeddingConfig, RetrievalConfig
from .embeddings import embed_texts
from .vector_store import VectorStore

MANIFEST_NAME = "manifest.jsonl"
INDEX_DIR = "index"

def read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_lines(path: str, lines: Iterable[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def append_jsonl(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_existing_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {}
    data = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("metadata", {}).get("uuid") or rec.get("uuid")
            if key:
                data[key] = rec
    return data

def create_sentence_records(book_id: str, normalized_text: str, page_number: int = 0) -> List[SentenceRecord]:
    seg = segment_text(normalized_text)
    records: List[SentenceRecord] = []
    for para_idx, sent_idx, sentence in seg:
        rec = SentenceRecord.create_base(
            book_id=book_id,
            page_number=page_number,
            paragraph_index=para_idx,
            sentence_index=sent_idx,
            original=sentence,
            normalized=sentence,
        )
        records.append(rec)
    return records

async def build_index(
    embedding_cfg: EmbeddingConfig,
    records: List[SentenceRecord],
    output_dir: str,
):
    store = VectorStore(dim=embedding_cfg.dim)
    texts = [r.normalized for r in records]
    async with httpx.AsyncClient() as client:
        vectors = await embed_texts(embedding_cfg, client, texts)
    for rec, vec in zip(records, vectors):
        meta = {
            "uuid": rec.metadata["uuid"],
            "book_id": rec.book_id,
            "page_number": rec.page_number,
            "paragraph_index": rec.paragraph_index,
            "sentence_index": rec.sentence_index,
            "original": rec.original,
        }
        store.add(vec, meta)
    store.persist(os.path.join(output_dir, INDEX_DIR))
    return store

def load_or_build_index(
    embedding_cfg: EmbeddingConfig,
    records: List[SentenceRecord],
    output_dir: str,
    force_rebuild: bool = False,
):
    idx_dir = os.path.join(output_dir, INDEX_DIR)
    if (not force_rebuild) and os.path.exists(os.path.join(idx_dir, "vectors.npy")):
        try:
            return VectorStore.load(idx_dir)
        except Exception:
            pass
    # If cannot load, rebuild externally (caller must run build_index separately)
    return None

def build_retrieval_context_fn(
    store: VectorStore,
    embedding_vectors: dict,
    retrieval_cfg: RetrievalConfig,
    translate_cache: dict,
) -> Callable[[SentenceRecord], str]:
    def fn(rec: SentenceRecord) -> str:
        # If reuse candidate appears identical
        vec = embedding_vectors.get(rec.metadata["uuid"])
        if vec is None:
            return ""
        neighbors = store.search(vec, top_k=retrieval_cfg.top_k + 3, exclude_ids=[rec.metadata["uuid"]])
        ctx_lines = []
        reused_from = None
        similarity_used = None
        for sim, meta in neighbors:
            uid = meta.get("uuid")
            if uid in translate_cache:
                # Found a candidate translation
                tgt_rec = translate_cache[uid]
                if reused_from is None and sim >= retrieval_cfg.reuse_threshold:
                    # Mark reuse
                    rec.literal_sentence_en = tgt_rec.get("literal_sentence_en")
                    rec.sentence_translation_en = tgt_rec.get("sentence_translation_en")
                    rec.context_explanation_en = tgt_rec.get("context_explanation_en")
                    rec.metadata["reused_from"] = uid
                    rec.metadata["similarity"] = sim
                    return f"Context: (reused prior translation from {uid} similarity={sim:.4f})"
                # Build context line
                line = (
                    f"Original: {meta.get('original')}\n"
                    f"Translation: {tgt_rec.get('sentence_translation_en')}\n"
                    f"Notes: (n/a)"
                )
                ctx_lines.append(line)
            if len(ctx_lines) >= retrieval_cfg.top_k:
                break
        if not ctx_lines:
            return ""
        block = "Context for consistency (similar prior passages):\n"
        formatted = []
        char_budget = retrieval_cfg.max_context_chars
        used = 0
        for i, l in enumerate(ctx_lines, 1):
            if used + len(l) > char_budget:
                break
            formatted.append(f"{i}. {l}")
            used += len(l)
        return block + "\n".join(formatted)
    return fn

async def run_translation(
    azure: AzureConfig,
    embedding_cfg: EmbeddingConfig,
    cfg: PipelineConfig,
    records: List[SentenceRecord],
    output_dir: str,
):
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    lock = FileLock(manifest_path + ".lock")

    # Load existing translations (resume)
    existing = load_existing_manifest(manifest_path)
    # Map for retrieval: uuid -> translated content
    translate_cache = {
        k: v for k, v in existing.items() if v.get("sentence_translation_en")
    }

    # Load index if retrieval enabled
    store = None
    embedding_vectors = {}
    if cfg.retrieval.enable:
        idx_dir = os.path.join(output_dir, INDEX_DIR)
        if not os.path.exists(os.path.join(idx_dir, "vectors.npy")):
            raise RuntimeError("Retrieval enabled but index missing. Run build-index first.")
        store = VectorStore.load(idx_dir)
        # Build in-memory mapping uuid -> vector row
        # (Vectors already in store; we assume order alignment)
        for meta_idx, meta in enumerate(store.meta):
            uid = meta.get("uuid")
            if uid:
                embedding_vectors[uid] = store.vectors[meta_idx]

    retrieval_context_fn = None
    if cfg.retrieval.enable and store:
        retrieval_context_fn = build_retrieval_context_fn(
            store=store,
            embedding_vectors=embedding_vectors,
            retrieval_cfg=cfg.retrieval,
            translate_cache=translate_cache,
        )

    async with httpx.AsyncClient() as client:
        batch_size = cfg.batch_size
        for i in tqdm(range(0, len(records), batch_size), desc="Translating"):
            batch = records[i:i+batch_size]
            untranslated = []
            # Load existing translations into objects if present
            for r in batch:
                if r.metadata["uuid"] in existing:
                    existing_rec = existing[r.metadata["uuid"]]
                    if existing_rec.get("sentence_translation_en"):
                        # Already translated
                        continue
                untranslated.append(r)
            if not untranslated:
                continue
            if cfg.dry_run:
                continue
            updated = await translate_batch(
                azure, client, untranslated, retrieval_context_fn=retrieval_context_fn
            )
            with lock:
                for rec in updated:
                    append_jsonl(manifest_path, json.loads(rec.model_dump_json()))
                    existing[rec.metadata["uuid"]] = json.loads(rec.model_dump_json())
    return manifest_path

def export_final(manifest_path: str, final_output: str):
    with open(manifest_path, "r", encoding="utf-8") as f, open(final_output, "w", encoding="utf-8") as out:
        for line in f:
            out.write(line)

async def full_pipeline(
    *,
    book_id: str,
    anmol_input_path: str,
    mapping_path: str,
    azure: AzureConfig,
    embedding_cfg: EmbeddingConfig,
    cfg: PipelineConfig,
    output_dir: str,
    build_vector_index: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    raw_lines = []
    with open(anmol_input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    mapping = load_anmol_lipi_mapping(mapping_path)
    normalized_lines = list(normalize_text(raw_lines, mapping))
    normalized_text = "\n".join(normalized_lines)
    records = create_sentence_records(book_id, normalized_text)
    if build_vector_index:
        await build_index(embedding_cfg, records, output_dir)
    await run_translation(azure, embedding_cfg, cfg, records, output_dir)
    export_final(os.path.join(output_dir, MANIFEST_NAME), os.path.join(output_dir, f"{book_id}.jsonl"))