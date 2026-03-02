from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Job:
    raw: Dict[str, Any]
    path: Path

    @property
    def job_id(self) -> str:
        return str(self.raw.get("job_id", ""))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False, allow_unicode=True)


def load_job(path: Path) -> Job:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Job YAML must be a mapping.")
    return Job(raw=raw, path=path)


def ensure_job_defaults(job: Job) -> None:
    job.raw.setdefault("results", {})
    job.raw["results"].setdefault("run", {})
    job.raw["results"].setdefault("outputs", [])
    job.raw["results"].setdefault("notes", [])
    job.raw["results"].setdefault("errors", [])
