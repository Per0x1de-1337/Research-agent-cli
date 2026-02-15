from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_agent.models import LocalDocument, SourceKind, SourceRecord


def _get(item: Any, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def extract_message_text_and_annotations(message) -> tuple[str, list[dict[str, Any]]]:
    blocks = getattr(message, "content_blocks", None)
    if not blocks:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            blocks = content
        elif isinstance(content, str):
            return content, []

    if not blocks:
        text = getattr(message, "text", None)
        if callable(text):
            text = text()
        return text or "", []

    text_fragments: list[str] = []
    annotations: list[dict[str, Any]] = []
    for block in blocks:
        block_type = _get(block, "type")
        if block_type in {"text", "output_text"}:
            text_fragments.append(_get(block, "text", "") or "")
            for annotation in _get(block, "annotations", []) or []:
                annotations.append(
                    {
                        "start_index": int(_get(annotation, "start_index", 0) or 0),
                        "end_index": int(_get(annotation, "end_index", 0) or 0),
                        "title": _get(annotation, "title"),
                        "url": _get(annotation, "url"),
                        "type": _get(annotation, "type"),
                        "extras": _get(annotation, "extras"),
                    }
                )

    text = "\n".join(fragment for fragment in text_fragments if fragment).strip()
    return text, annotations


@dataclass
class SourceRegistry:
    _records: list[SourceRecord] = field(default_factory=list)
    _by_key: dict[tuple[str, str], str] = field(default_factory=dict)
    _counter: int = 0

    @property
    def records(self) -> list[SourceRecord]:
        return list(self._records)

    def _next_id(self) -> str:
        self._counter += 1
        return f"SRC{self._counter:03d}"

    def register_local_document(self, document: LocalDocument) -> list[str]:
        source_ids: list[str] = []
        for chunk in document.chunks:
            key = ("local", f"{document.path}:{chunk.locator}")
            source_id = self._by_key.get(key)
            if source_id is None:
                source_id = chunk.source_id
                self._by_key[key] = source_id
                self._records.append(
                    SourceRecord(
                        source_id=source_id,
                        kind=SourceKind.local_file,
                        title=document.title,
                        path=document.path,
                        locator=chunk.locator,
                        snippet=chunk.text[:500],
                        note="User-supplied local document chunk",
                    )
                )
            source_ids.append(source_id)
        return source_ids

    def register_web_annotations(
        self,
        query: str,
        text: str,
        annotations: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        if not text.strip():
            return "", []

        source_ids: list[str] = []
        inserts: list[tuple[int, str]] = []

        for annotation in sorted(annotations, key=lambda item: item["end_index"]):
            title = annotation.get("title") or "Web source"
            url = annotation.get("url") or ""
            key = ("web", f"{title}|{url}")
            source_id = self._by_key.get(key)
            if source_id is None:
                source_id = self._next_id()
                self._by_key[key] = source_id
                snippet = text[annotation["start_index"] : annotation["end_index"]].strip()
                self._records.append(
                    SourceRecord(
                        source_id=source_id,
                        kind=SourceKind.web,
                        title=title,
                        url=url or None,
                        locator=annotation.get("type"),
                        snippet=snippet or None,
                        query=query,
                        note="OpenAI web-search citation",
                    )
                )
            source_ids.append(source_id)
            inserts.append((annotation["end_index"], f" [{source_id}]"))

        if not inserts:
            return text, []

        pieces: list[str] = []
        last_index = 0
        for end_index, marker in inserts:
            if end_index < last_index:
                continue
            pieces.append(text[last_index:end_index])
            pieces.append(marker)
            last_index = end_index
        pieces.append(text[last_index:])
        annotated_text = "".join(pieces).strip()
        deduped_ids = list(dict.fromkeys(source_ids))
        return annotated_text, deduped_ids

    def lookup(self, source_ids: list[str]) -> list[SourceRecord]:
        wanted = set(source_ids)
        return [record for record in self._records if record.source_id in wanted]
