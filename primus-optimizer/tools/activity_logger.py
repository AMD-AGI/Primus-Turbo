"""Append-only JSONL activity logger for Worker progress tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ActivityLogger:
    def __init__(self, path: str | Path, worker_id: str):
        self.path = Path(path)
        self.worker_id = worker_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, phase: str, round_num: int, msg: str) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "worker": self.worker_id,
            "phase": phase,
            "round": round_num,
            "msg": msg,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def read_recent(path: str | Path, n: int = 10) -> list[dict]:
        path = Path(path)
        if not path.exists():
            return []
        with open(path) as f:
            lines = f.readlines()
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent if line.strip()]
