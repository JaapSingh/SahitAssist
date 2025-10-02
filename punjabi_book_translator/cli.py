import typer
import json
from pathlib import Path
from rich import print
import asyncio
import os
import httpx

from .config import (
    load_azure_config,
    validate_config,
    PipelineConfig,
    load_embedding_config,
    load_retrieval_config,
)
from .normalization import load_anmol_lipi_mapping, normalize_text
from .segmentation import segment_text
from .schema import SentenceRecord
from .pipeline import (
    full_pipeline,
    create_sentence_records,
    run_translation,
    export_final,
    load_existing_manifest,
)
from .embeddings import embed_texts
from .vector_store import VectorStore

app = typer.Typer(help="Punjabi Book Translator CLI")

@app.command()
def normalize(
    input: str = typer.Argument(..., help="Path to Anmol Lipi text file"),
    output: str = typer.Argument(..., help="Where to write normalized Unicode text"),
    mapping: str = typer.Option("data/anmol_lipi_mapping.csv")
):
    mapping_table = load_anmol_lipi_mapping(mapping)
    with open(input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    norm_lines = list(normalize_text(lines, mapping_table))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(norm_lines))
    print(f"[green]Wrote normalized text to {output}[/green]")

@app.command()
def segment(
    input: str = typer.Argument(...),
    output: str = typer.Argument(...),
    book_id: str = typer.Option("book"),
):
    text = Path(input).read_text(encoding="utf-8")
    seg = segment_text(text)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for para_idx, sent_idx, sent in seg:
            rec = SentenceRecord.create_base(book_id=book_id, page_number=0,
                                             paragraph_index=para_idx,
                                             sentence_index=sent_idx,
                                             original=sent,
                                             normalized=sent)
            f.write(rec.model_dump_json(ensure_ascii=False) + "\n")
    print(f"[green]Segmented sentences written to {output}[/green]")

@app.command()
def build_index(
    input: str = typer.Argument(..., help="Segmented sentences JSONL"),
    output: str = typer.Argument(..., help="Output directory (same used for translation)"),
    book_id: str = typer.Option("book"),
):
    embedding_cfg = load_embedding_config()
    records = []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            records.append(SentenceRecord(**data))
    async def _run():
        async with httpx.AsyncClient() as client:
            texts = [r.normalized for r in records]
            vectors = await embed_texts(embedding_cfg, client, texts)
        store = VectorStore(dim=embedding_cfg.dim)
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
        store.persist(Path(output, "index").as_posix())
    asyncio.run(_run())
    print(f"[green]Index built at {output}/index[/green]")

@app.command()
def translate(
    input: str = typer.Argument(..., help="JSONL of sentence records OR raw text file."),
    output: str = typer.Argument(...),
    batch_size: int = typer.Option(8),
    use_context: bool = typer.Option(False, "--use-context", help="Enable retrieval-based context"),
):
    azure = load_azure_config()
    validate_config(azure)
    embedding_cfg = load_embedding_config()
    retrieval_cfg = load_retrieval_config(enable=use_context)
    cfg = PipelineConfig(batch_size=batch_size, retrieval=retrieval_cfg)
    Path(output).mkdir(parents=True, exist_ok=True)

    records = []
    if input.endswith(".jsonl"):
        with open(input, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                records.append(SentenceRecord(**data))
    else:
        text = Path(input).read_text(encoding="utf-8")
        records = create_sentence_records("book", text)

    asyncio.run(run_translation(azure, embedding_cfg, cfg, records, output))
    export_final(os.path.join(output, "manifest.jsonl"), os.path.join(output, "final.jsonl"))
    print("[green]Translation complete[/green]")

@app.command()
def pipeline(
    pdf: str = typer.Option(None, help="Path to PDF (optional if providing normalized text)"),
    anmol_text: str = typer.Option(None, help="If already extracted Anmol Lipi text"),
    mapping: str = typer.Option("data/anmol_lipi_mapping.csv"),
    book_id: str = typer.Option("book1"),
    output: str = typer.Option("outputs/book1"),
    batch_size: int = typer.Option(8),
    use_context: bool = typer.Option(False, "--use-context"),
    skip_index: bool = typer.Option(False, "--skip-index"),
):
    if not anmol_text and not pdf:
        raise typer.BadParameter("Provide either --pdf or --anmol-text.")
    if pdf and not anmol_text:
        from pdfminer.high_level import extract_text
        raw = extract_text(pdf)
        anmol_text = os.path.join(output, f"{book_id}_raw.txt")
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(anmol_text).write_text(raw, encoding="utf-8")

    azure = load_azure_config()
    validate_config(azure)
    embedding_cfg = load_embedding_config()
    retrieval_cfg = load_retrieval_config(enable=use_context)
    cfg = PipelineConfig(batch_size=batch_size, retrieval=retrieval_cfg)
    asyncio.run(full_pipeline(
        book_id=book_id,
        anmol_input_path=anmol_text,
        mapping_path=mapping,
        azure=azure,
        embedding_cfg=embedding_cfg,
        cfg=cfg,
        output_dir=output,
        build_vector_index=not skip_index
    ))
    print(f"[green]Pipeline complete. See {output}[/green]")

@app.command()
def stats(output: str):
    idx = Path(output, "index")
    if not idx.exists():
        print("[red]Index not found[/red]")
        raise typer.Exit(1)
    from .vector_store import VectorStore
    store = VectorStore.load(idx.as_posix())
    print(f"[cyan]Index size: {store.size()} dim: {store.dim}[/cyan]")

@app.command()
def resume(manifest: str):
    if not os.path.exists(manifest):
        raise typer.BadParameter("Manifest does not exist.")
    data = load_existing_manifest(manifest)
    print(f"Loaded {len(data)} entries. Resume logic can be extended here.")

if __name__ == "__main__":
    app()