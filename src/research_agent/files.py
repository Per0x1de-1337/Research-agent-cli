from __future__ import annotations

import csv
import json
from pathlib import Path

from pypdf import PdfReader

from research_agent.models import LocalChunk, LocalDocument


SUPPORTED_SUFFIXES = {".txt", ".md", ".rst", ".json", ".csv", ".pdf"}


def chunk_text(text: str, chunk_size: int = 2800, overlap: int = 250) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader, start=1):
            rows.append(f"Row {index}: " + " | ".join(row))
    return "\n".join(rows)


def _read_json(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        data = json.load(handle)
    return json.dumps(data, indent=2, ensure_ascii=False)


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".json":
        return _read_json(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_local_documents(paths: list[Path], max_chars: int) -> list[LocalDocument]:
    documents: list[LocalDocument] = []
    for index, raw_path in enumerate(paths, start=1):
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local file does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type for {path}. Supported types: "
                + ", ".join(sorted(SUPPORTED_SUFFIXES))
            )

        content = read_text_file(path).strip()
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars].rstrip()
            truncated = True

        chunk_texts = chunk_text(content)
        chunks = [
            LocalChunk(
                source_id=f"DOC{index:03d}-C{chunk_index:02d}",
                locator=f"chunk {chunk_index}",
                text=chunk,
            )
            for chunk_index, chunk in enumerate(chunk_texts, start=1)
        ]

        documents.append(
            LocalDocument(
                doc_id=f"DOC{index:03d}",
                path=str(path),
                title=path.name,
                excerpt=content[:700],
                content=content,
                chunks=chunks,
                truncated=truncated,
            )
        )
    return documents
