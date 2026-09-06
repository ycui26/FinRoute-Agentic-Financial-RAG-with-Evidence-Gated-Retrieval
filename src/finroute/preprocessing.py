from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from finroute.config import ChunkingConfig


NUMBER_RE = re.compile(r"(?<!\w)[(+-]?(?:\$|€|£)?\d[\d,]*(?:\.\d+)?%?\)?")
SECTION_RE = re.compile(
    r"(?i)^(item\s+\d+[a-z]?\.?|consolidated\s+statements?|notes?\s+to\s+)"
)
STATEMENT_PATTERNS = {
    "balance_sheet": re.compile(r"(?i)balance sheets?|financial position"),
    "income_statement": re.compile(r"(?i)statements? of (?:operations|income|earnings)"),
    "cash_flow_statement": re.compile(r"(?i)statements? of cash flows?"),
    "equity_statement": re.compile(r"(?i)statements? of (?:stockholders|shareholders).+equity"),
}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf_pages(pdf_path: str | Path, metadata: dict) -> pd.DataFrame:
    import pymupdf

    rows = []
    active_section = ""
    with pymupdf.open(pdf_path) as document:
        for page_idx0, page in enumerate(document):
            text = normalize_text(page.get_text("text"))
            for line in text.splitlines():
                if SECTION_RE.search(line.strip()):
                    active_section = line.strip()[:200]
                    break
            rows.append(
                {
                    **metadata,
                    "page_idx0": page_idx0,
                    "page_num": page_idx0 + 1,
                    "section": active_section,
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def split_with_offsets(text: str, size: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def build_narrative_chunks(pages: pd.DataFrame, config: ChunkingConfig) -> pd.DataFrame:
    rows = []
    for page in pages.to_dict(orient="records"):
        text = str(page["text"])
        parent_id = f"{page['doc_name']}_p{int(page['page_num']):04d}"
        parents = list(split_with_offsets(text, config.parent_size, config.parent_overlap))
        for parent_index, (parent_start, parent_end, parent_text) in enumerate(parents):
            for child_index, (child_start, child_end, child_text) in enumerate(
                split_with_offsets(parent_text, config.child_size, config.child_overlap)
            ):
                if len(child_text) < config.minimum_chars:
                    continue
                rows.append(
                    {
                        **page,
                        "parent_id": f"{parent_id}_w{parent_index:03d}",
                        "chunk_id": f"{parent_id}_w{parent_index:03d}_c{child_index:03d}",
                        "start_char": parent_start + child_start,
                        "end_char": min(parent_start + child_end, parent_end),
                        "parent_text": parent_text,
                        "context_text": child_text,
                        "source_type": "narrative_child",
                    }
                )
    return pd.DataFrame(rows)


def infer_statement_type(text: str) -> str:
    for name, pattern in STATEMENT_PATTERNS.items():
        if pattern.search(text):
            return name
    return "other_table"


def table_quality(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    numeric_lines = sum(bool(NUMBER_RE.search(line)) for line in lines)
    header_signal = int(any(pattern.search(text) for pattern in STATEMENT_PATTERNS.values()))
    density = numeric_lines / len(lines)
    return min(1.0, 0.75 * density + 0.25 * header_signal)


def extract_table_blocks(pages: pd.DataFrame, minimum_numbers: int = 4) -> pd.DataFrame:
    rows = []
    for page in pages.to_dict(orient="records"):
        lines = [line.strip() for line in str(page["text"]).splitlines() if line.strip()]
        groups: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            is_table_line = bool(NUMBER_RE.search(line)) or (current and len(line.split()) <= 12)
            if is_table_line:
                current.append(line)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        for block_index, group in enumerate(groups):
            block = "\n".join(group)
            if len(NUMBER_RE.findall(block)) < minimum_numbers:
                continue
            context = "\n".join(lines[max(0, lines.index(group[0]) - 2):])
            context = context[:4000]
            rows.append(
                {
                    **page,
                    "chunk_id": f"{page['doc_name']}_p{int(page['page_num']):04d}_t{block_index:03d}",
                    "parent_id": f"{page['doc_name']}_p{int(page['page_num']):04d}",
                    "statement_type": infer_statement_type(
                        f"{page.get('section', '')}\n{context}"
                    ),
                    "table_quality_score": table_quality(block),
                    "context_text": block,
                    "source_type": "table_block",
                }
            )
    return pd.DataFrame(rows)
