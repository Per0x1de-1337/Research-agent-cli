from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


def to_serializable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


@dataclass(slots=True)
class JobWorkspace:
    job_id: str
    root: Path

    def path_for(self, name: str) -> Path:
        return self.root / name

    def write_text(self, name: str, content: str) -> Path:
        path = self.path_for(name)
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, name: str, content) -> Path:
        path = self.path_for(name)
        path.write_text(
            json.dumps(to_serializable(content), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def append_event(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        with self.path_for("events.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.jobs_root = root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def create_job(self) -> JobWorkspace:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        job_id = f"{timestamp}-{uuid4().hex[:8]}"
        root = self.jobs_root / job_id
        root.mkdir(parents=True, exist_ok=False)
        return JobWorkspace(job_id=job_id, root=root)

    def list_jobs(self) -> list[Path]:
        return sorted(
            (path for path in self.jobs_root.iterdir() if path.is_dir()),
            reverse=True,
        )

    def get(self, job_id: str) -> JobWorkspace:
        root = self.jobs_root / job_id
        if not root.exists():
            raise FileNotFoundError(f"Unknown job ID: {job_id}")
        return JobWorkspace(job_id=job_id, root=root)
