from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finroute.config import FinRouteConfig
from finroute.indexing import EmbeddingEncoder, HybridIndex
from finroute.preprocessing import build_narrative_chunks, extract_table_blocks, parse_pdf_pages


def document_routing_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(column, ""))
        for column in ["company", "doc_name", "year", "form"]
    ).strip()


def build_artifacts(
    pdf_directory: str | Path,
    metadata_path: str | Path,
    artifacts_directory: str | Path,
    config: FinRouteConfig,
) -> dict[str, int]:
    pdf_directory = Path(pdf_directory)
    artifacts = Path(artifacts_directory)
    artifacts.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    required = {"doc_name", "file_name", "company", "year", "form"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"documents.csv is missing columns: {sorted(missing)}")

    page_frames = []
    for record in metadata.to_dict(orient="records"):
        pdf_path = pdf_directory / str(record.pop("file_name"))
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        page_frames.append(parse_pdf_pages(pdf_path, record))
    pages = pd.concat(page_frames, ignore_index=True)
    children = build_narrative_chunks(pages, config.chunking)
    tables = extract_table_blocks(pages)
    documents = metadata.drop_duplicates("doc_name").copy()
    documents["context_text"] = documents.apply(document_routing_text, axis=1)

    pages.to_parquet(artifacts / "pages.parquet", index=False)
    encoder = EmbeddingEncoder(
        config.models.embedding,
        query_prefix=config.models.query_prefix,
        local_files_only=config.models.local_files_only,
    )
    chroma = artifacts / "chroma"
    for name, frame in [
        ("documents", documents),
        ("narrative", children),
        ("tables", tables),
    ]:
        if frame.empty:
            raise ValueError(f"{name} corpus is empty")
        embeddings = encoder.encode_documents(frame["context_text"].astype(str).tolist())
        index = HybridIndex(
            frame=frame,
            embeddings=embeddings,
            text_column="context_text",
            encoder=encoder,
            config=config,
            collection_name=f"finroute_{name}",
            chroma_directory=chroma,
        )
        index.save(artifacts / name)
        if config.runtime.use_chroma:
            index.build_chroma(recreate=True)

    manifest = {
        "documents": int(documents["doc_name"].nunique()),
        "pages": len(pages),
        "narrative_chunks": len(children),
        "table_blocks": len(tables),
    }
    (artifacts / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
