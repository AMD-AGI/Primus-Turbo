###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HipKittens module loader (no-cache, no-JSON edition).

The HipKittens project ships a prebuilt Python extension per precision:

  * ``tk_bf16_layouts``  - BF16 dense + grouped (analysis/bf16_gemm/mi350x)
  * ``tk_fp8_layouts``   - FP8 tensorwise dense (analysis/fp8_gemm/mi350x)

Each lives under ``$PRIMUS_TURBO_HIPKITTEN_PATH/analysis/<dtype>_gemm/mi350x``.
The historical autotune-cache JSON files in those directories are *offline*
benchmark notes — useful at development time to derive the rules in
``config.py::select_default_config``, but never read at runtime.

This module is intentionally tiny: it does ``importlib.import_module`` on the
``tk_*_layouts.so`` and returns a frozen :class:`HipKittenModule`. The first
import goes through Python's own ``sys.modules`` mapping after that, so
subsequent ``load_*()`` calls cost ~tens of nanoseconds. The previous
implementation re-parsed a ~50 KB autotune-cache JSON on every dispatch which
dominated the host overhead on small GEMMs (4096^3 BF16 forward fell from
~120 us kernel time to ~770 us wall — i.e. ~6x slowdown vs hipBLASLt purely
from host fat). Removing the JSON read recovers that overhead.

NOTE: per the project policy, this module MUST NOT keep any per-shape cache
(no dict / weakref / data_ptr / _version / lru_cache / TTL). The
``importlib.import_module`` step itself goes through Python's own
``sys.modules`` mapping which we do not control; that is part of the
language runtime, not a cache we introduced.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_HIPKITTEN_MODULE_ENV = "PRIMUS_TURBO_HIPKITTEN_MODULE"
_HIPKITTEN_BF16_PATH_ENV = "PRIMUS_TURBO_HIPKITTEN_BF16_PATH"
_HIPKITTEN_FP8_PATH_ENV = "PRIMUS_TURBO_HIPKITTEN_FP8_PATH"
_HIPKITTEN_PATH_ENV = "PRIMUS_TURBO_HIPKITTEN_PATH"

_BF16_DEFAULT_MODULE = "tk_bf16_layouts"
_FP8_MODULE = "tk_fp8_layouts"


@dataclass(frozen=True)
class HipKittenModule:
    """A loaded HipKittens precision module + the alignment constants we need.

    ``block_size`` and ``k_block`` come straight from the binding (read once
    at import time via ``getattr``). They are used by callers to enforce
    alignment in ``can_handle``; we cache the values on the dataclass so
    ``can_handle`` does not re-issue ``getattr`` per dispatch.
    """

    module: Any
    dtype: Literal["bf16", "fp8"]
    block_size: int
    k_block: int


def _bf16_search_paths() -> list[Path]:
    paths: list[Path] = []
    if bf16_path := os.environ.get(_HIPKITTEN_BF16_PATH_ENV):
        paths.append(Path(bf16_path))
    if root := os.environ.get(_HIPKITTEN_PATH_ENV):
        root_path = Path(root)
        paths.extend([root_path, root_path / "analysis" / "bf16_gemm" / "mi350x"])
    return paths


def _fp8_search_paths() -> list[Path]:
    paths: list[Path] = []
    if fp8_path := os.environ.get(_HIPKITTEN_FP8_PATH_ENV):
        paths.append(Path(fp8_path))
    if root := os.environ.get(_HIPKITTEN_PATH_ENV):
        root_path = Path(root)
        paths.extend([root_path, root_path / "analysis" / "fp8_gemm" / "mi350x"])
    return paths


def _import_with_fallback(module_name: str, search_paths: list[Path]) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as first_error:
        for search_path in search_paths:
            if search_path.is_dir():
                sys.path.insert(0, str(search_path))
                try:
                    return importlib.import_module(module_name)
                except ImportError:
                    continue
        raise ImportError(
            f"HipKittens backend requires importable '{module_name}'. "
            f"Searched: {[str(p) for p in search_paths]}. "
            f"Set {_HIPKITTEN_PATH_ENV} or the per-precision path env."
        ) from first_error


def load_bf16() -> HipKittenModule:
    """Import ``tk_bf16_layouts`` and return its :class:`HipKittenModule` view.

    Re-runs ``importlib.import_module`` on every invocation, but Python
    caches the loaded extension via ``sys.modules`` so subsequent calls cost
    ~tens of nanoseconds. No JSON / pickle / dict lookup happens here.
    """
    module_name = os.environ.get(_HIPKITTEN_MODULE_ENV, _BF16_DEFAULT_MODULE)
    search_paths = _bf16_search_paths()
    module = _import_with_fallback(module_name, search_paths)
    return HipKittenModule(
        module=module,
        dtype="bf16",
        block_size=int(getattr(module, "BLOCK_SIZE", 256)),
        k_block=int(getattr(module, "K_STEP", 64)),
    )


def load_fp8() -> HipKittenModule:
    """Import ``tk_fp8_layouts`` and return its :class:`HipKittenModule` view.

    Same no-cache semantics as :func:`load_bf16`.
    """
    search_paths = _fp8_search_paths()
    module = _import_with_fallback(_FP8_MODULE, search_paths)
    return HipKittenModule(
        module=module,
        dtype="fp8",
        block_size=int(getattr(module, "BLOCK_SIZE", 256)),
        k_block=int(getattr(module, "K_BLOCK", 128)),
    )


def has_bf16() -> bool:
    try:
        load_bf16()
        return True
    except ImportError:
        return False


def has_fp8() -> bool:
    try:
        load_fp8()
        return True
    except ImportError:
        return False
