###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Process-wide state shared by every FlyDSL kernel autotuner.

The switch is ``PRIMUS_TURBO_AUTO_TUNE`` (default off), read through
``GlobalBackendManager.auto_tune_enabled()`` so ``set_auto_tune()`` reaches the kernel level as
well. When it is off a tuner times nothing and returns its base config (see ``Tuner``'s ``off``).
"""

import os
import threading
import time
import weakref
from typing import Any, Callable, List, Optional

import torch

from primus_turbo.common.constants import ENV_AUTO_TUNE
from primus_turbo.common.logger import logger

__all__ = [
    "enabled",
    "force",
    "blocked",
    "warn_blocked",
    "warn_unverified",
    "empty_cache",
    "record",
    "report",
    "clear",
    "register",
]

_forced: Optional[bool] = None
_switch: Optional[Callable[[], bool]] = None

_records: List[dict] = []
_records_lock = threading.Lock()
_tuners: "weakref.WeakSet" = weakref.WeakSet()
_clear_hooks: List[Callable[[], None]] = []


def _env_switch() -> bool:
    return os.environ.get(ENV_AUTO_TUNE, "0") == "1"


def enabled() -> bool:
    """Whether kernel-level autotune runs. Cheap enough for a per-launch check."""
    if _forced is not None:
        return _forced
    global _switch
    if _switch is None:
        try:
            # Lazy: pytorch.core imports the flydsl kernels, not the other way round.
            from primus_turbo.pytorch.core.backend import GlobalBackendManager

            _switch = GlobalBackendManager.auto_tune_enabled
        except ImportError:
            _switch = _env_switch
    return _switch()


def force(value: Optional[bool]) -> None:
    """Pin the kernel-level switch for tests and tools; ``None`` follows the backend switch again."""
    global _forced
    _forced = value


def _is_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except AttributeError:  # pragma: no cover - torch < 2.3
        return False


def blocked() -> Optional[str]:
    """Why timing is impossible right now, or None. A blocked tune returns base and caches nothing."""
    if torch.cuda.is_current_stream_capturing():
        return "CUDA-graph capture"
    if _is_compiling():
        return "torch.compile tracing"
    return None


def _short(key: Any, limit: int = 160) -> str:
    """A key for the log: flydsl's own keys carry the whole toolchain fingerprint."""
    s = repr(key)
    return s if len(s) <= limit else s[:limit] + "...)"


def warn_blocked(op: str, key: Any, reason: str) -> None:
    logger.warning(
        f"[flydsl-autotune] {op}: cache miss under {reason}, using the base config for key={_short(key)}; "
        "it is picked again on the next eager call, but what was captured keeps the base: run each "
        "shape once eagerly before capture",
        once=True,
    )


def warn_unverified(op: str, key: Any, config: Any) -> None:
    logger.warning(
        f"[flydsl-autotune] {op}: no candidate held on this call's data for key={_short(key)}, using "
        f"{config!r} uncached; it is checked again on the next call",
        once=True,
    )


def empty_cache() -> None:
    """Hand the caching allocator's free blocks back, e.g. once a search's probes are dead."""
    torch.cuda.empty_cache()


def record(op: str, key: Any, config: Any, *, candidates: int, scores: Any = None, secs: float = 0.0) -> None:
    rec = {
        "op": op,
        "key": key,
        "config": config,
        "candidates": candidates,
        "scores": scores,
        "secs": round(secs, 3),
        "time": time.time(),
    }
    with _records_lock:
        _records.append(rec)
    logger.debug(f"[flydsl-autotune] {op} key={_short(key)} -> {config} ({candidates} cands, {secs:.2f}s)")


def report() -> List[dict]:
    """Every tune this process ran, oldest first."""
    with _records_lock:
        return list(_records)


def register(tuner=None, clear_hook: Optional[Callable[[], None]] = None) -> None:
    """Track a tuner (and any op-level cache that holds its picks) for ``clear()``."""
    if tuner is not None:
        _tuners.add(tuner)
    if clear_hook is not None:
        _clear_hooks.append(clear_hook)


def clear() -> None:
    """Forget every pick and the report, so the next call per shape tunes (or takes base) again.

    It drops the artifacts and compiles the picks own, the replaced ones kept alive for graphs
    captured on them too, so a CUDA graph captured before may launch code or read workspaces that
    are gone: capture again after a clear instead of replaying. The
    launch-level caches stay, so a re-tune may call a kept launch raw once more (see ``Entry``).
    Refused while the current stream captures."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("[flydsl-autotune] runtime.clear() during CUDA-graph capture")
    for t in list(_tuners):
        if hasattr(t, "clear"):
            t.clear()
        else:
            t.cache.clear()
    for hook in _clear_hooks:
        hook()
    with _records_lock:
        _records.clear()
