"""GPU resource pool for allocating exclusive devices to Workers."""

from __future__ import annotations


class GPUPool:
    def __init__(self, gpu_ids: list[int]):
        self._available: list[int] = list(gpu_ids)
        self._allocated: dict[str, int] = {}

    def acquire(self, task_id: str) -> int:
        if not self._available:
            raise RuntimeError("No GPUs available")
        gpu_id = self._available.pop(0)
        self._allocated[task_id] = gpu_id
        return gpu_id

    def release(self, task_id: str) -> None:
        gpu_id = self._allocated.pop(task_id)  # raises KeyError if unknown
        self._available.append(gpu_id)

    def has_available(self) -> bool:
        return len(self._available) > 0

    def available_count(self) -> int:
        return len(self._available)

    def allocated_map(self) -> dict[str, int]:
        return dict(self._allocated)

    def env_for(self, task_id: str) -> dict[str, str]:
        gpu_id = self._allocated[task_id]
        return {"HIP_VISIBLE_DEVICES": str(gpu_id)}
