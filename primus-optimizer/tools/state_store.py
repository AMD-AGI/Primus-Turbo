"""Persistent JSON state store for optimizer sessions."""

from __future__ import annotations

import json
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkerStatus:
    PENDING = "pending"
    RUNNING = "running"
    BOTTLENECK = "bottleneck"
    STUCK = "stuck"
    COMPLETED = "completed"
    FAILED = "failed"


class StateStore:
    """Read/write optimizer-state.json with file-lock safety."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict[str, Any]:
        with open(self.path) as f:
            return json.load(f)

    def _write(self, state: dict[str, Any]) -> None:
        state["updated_at"] = self._now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.rename(self.path)

    def _update(self, fn) -> dict[str, Any]:
        """Read-modify-write with file lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self._read()
                fn(state)
                self._write(state)
                return state
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load(self) -> dict[str, Any]:
        return self._read()

    def init_session(
        self,
        session_id: str,
        hw: str,
        tasks: list[dict[str, str]],
        gpu_ids: list[int],
    ) -> None:
        state = {
            "session_id": session_id,
            "config": {
                "hw": hw,
                "tasks": tasks,
            },
            "gpu_pool": {
                "total": gpu_ids,
                "allocated": {},
            },
            "workers": {},
            "task_queue": [f"{t['operator']}:{t['backend']}" for t in tasks],
            "review_log": [],
            "started_at": self._now(),
            "updated_at": self._now(),
        }
        self._write(state)

    def add_worker(self, worker_id: str, gpu_id: int, worktree_branch: str) -> None:
        def _add(state):
            state["workers"][worker_id] = {
                "status": WorkerStatus.RUNNING,
                "gpu_id": gpu_id,
                "worktree_branch": worktree_branch,
                "current_round": 0,
                "rounds": [],
                "bottleneck": None,
                "started_at": self._now(),
            }
            state["gpu_pool"]["allocated"][worker_id] = gpu_id
        self._update(_add)

    def record_round(
        self,
        worker_id: str,
        round_num: int,
        strategy: str,
        baseline_tflops: float,
        result_tflops: float,
        status: str,
    ) -> None:
        improvement = ((result_tflops - baseline_tflops) / baseline_tflops) * 100 if baseline_tflops > 0 else 0.0

        def _record(state):
            w = state["workers"][worker_id]
            w["current_round"] = round_num
            w["rounds"].append({
                "round": round_num,
                "strategy": strategy,
                "baseline_tflops": baseline_tflops,
                "result_tflops": result_tflops,
                "improvement_pct": round(improvement, 1),
                "status": status,
            })
        self._update(_record)

    def update_worker_status(
        self, worker_id: str, status: str, bottleneck_level: str | None = None
    ) -> None:
        def _update_status(state):
            state["workers"][worker_id]["status"] = status
            if bottleneck_level is not None:
                state["workers"][worker_id]["bottleneck"] = bottleneck_level
        self._update(_update_status)

    def add_review_log(
        self, trigger: str, worker: str, round_num: int, decision: str,
        cross_pollination: list | None = None,
    ) -> None:
        def _add(state):
            state["review_log"].append({
                "trigger": trigger,
                "worker": worker,
                "round": round_num,
                "decision": decision,
                "cross_pollination": cross_pollination or [],
                "timestamp": self._now(),
            })
        self._update(_add)

    def pop_task_queue(self) -> str | None:
        result = [None]
        def _pop(state):
            if state["task_queue"]:
                result[0] = state["task_queue"].pop(0)
        self._update(_pop)
        return result[0]
