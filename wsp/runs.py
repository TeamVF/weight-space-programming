"""Write one experiment run under a gitignored runs/ directory.

Author: wanghaobo
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLog:
    """A single run directory: `runs/<experiment>/<run_id>/`."""

    def __init__(
        self,
        runs_root: str | Path,
        experiment: str,
        run_id: str | None = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.directory = Path(runs_root) / experiment / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_text(self, name: str, text: str) -> Path:
        path = self._path(name)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self._path(name)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self._path(name)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def append_jsonl(self, name: str, row: dict[str, Any]) -> Path:
        path = self._path(name)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _path(self, name: str) -> Path:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"refusing to write outside the run directory: {name}")
        return self.directory / name
