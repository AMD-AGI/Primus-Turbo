###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP-safe ``num_combine_cu`` autotune for the unified fp8 combine kernel.

Mirrors bf16 ``grouped_gemm_combine_bf16_kernel``'s flydsl autotune sweep, but at Python level
because the fp8 combine compiles via ``_compile`` (Python closure) rather than a single @autotune
@flyc.jit entry. Disk cache is file-locked so concurrent EP ranks don't clobber each other.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from primus_turbo.flydsl.mega.tune_utils import _file_lock

# fwd L2 (K=I): GEMM is small -> favor more push CUs; bwd STEP3 (K=2I): smaller push share.
_CU_CANDIDATES_FWD = (24, 32, 40, 48, 56, 64)
_CU_CANDIDATES_BWD = (16, 24, 32, 40, 48)

_MEM: dict[tuple, int] = {}


def _cache_path() -> Path:
    base = Path(os.environ.get("FLYDSL_CACHE_DIR", Path.home() / ".flydsl"))
    return base / "fp8_combine_cu_cache.json"


def _load_disk() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_disk(disk: dict) -> None:
    p = _cache_path()
    with _file_lock(p.with_suffix(".lock")):
        merged = _load_disk()
        merged.update(disk)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(merged, indent=2))
        os.replace(tmp, p)


def _bench_launch(launch_fn: Callable[[], None], *, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        launch_fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = float(np.mean([s.elapsed_time(e) for s, e in zip(starts, ends)][1:]))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([ms], device="cuda")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        ms = float(t.item())
    return ms


def resolve_num_combine_cu(
    *,
    key: tuple,
    apply_weights: bool,
    explicit: int | None,
    make_launch: Callable[[int], Callable[[], None]],
) -> int:
    """Return ``num_combine_cu`` to use. ``explicit`` bypasses autotune (bench / tests)."""
    if explicit is not None:
        return int(explicit)
    if os.environ.get("PT_FP8_COMBINE_AUTOTUNE", "1") == "0":
        # shipped defaults when autotune disabled
        return 32 if apply_weights else 24

    if key in _MEM:
        return _MEM[key]

    disk_key = json.dumps(list(key))
    disk = _load_disk()
    if disk_key in disk:
        cu = int(disk[disk_key]["num_combine_cu"])
        _MEM[key] = cu
        return cu

    candidates: Sequence[int] = _CU_CANDIDATES_FWD if apply_weights else _CU_CANDIDATES_BWD
    best_cu, best_ms = candidates[0], float("inf")
    for cu in candidates:
        ms = _bench_launch(make_launch(cu), warmup=2, rep=5)
        if ms < best_ms:
            best_ms, best_cu = ms, cu

    _MEM[key] = best_cu
    _save_disk({disk_key: {"num_combine_cu": best_cu, "ms": best_ms}})
    return best_cu
