###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HipKittens module + autotune-cache loader.

The HipKittens project ships a prebuilt Python extension per precision:

  * ``tk_bf16_layouts``  - BF16 dense + grouped (analysis/bf16_gemm/mi350x)
  * ``tk_fp8_layouts``   - FP8 tensorwise dense (analysis/fp8_gemm/mi350x)

Each lives under ``$PRIMUS_TURBO_HIPKITTEN_PATH/analysis/<dtype>_gemm/mi350x``
together with the autotune cache that records the per-shape ``group_m`` /
``num_xcds`` / ``kernel`` choices made during offline autotune. This module
finds, imports, and parses both into a uniform :class:`HipKittenModule`
shape that the rest of the integration layer consumes.

NOTE: per the project policy, this module MUST NOT keep any cache (no
dict / weakref / data_ptr / _version / lru_cache / TTL of any kind). Each
``load_bf16()`` / ``load_fp8()`` call re-reads the on-disk autotune cache
file from scratch. ``importlib.import_module`` itself goes through Python's
own ``sys.modules`` mapping which we do not control; that is part of the
language runtime, not a cache we introduced.
"""
from __future__ import annotations

import importlib
import json
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
_BF16_CACHE_FILE = "bench_bf16_no_jit_final.json"
_FP8_CACHE_FILE = ".autotune_cache.json"


@dataclass(frozen=True)
class HipKittenModule:
    """A loaded HipKittens precision module plus its parsed autotune cache.

    The ``cache`` is normalized into a single shape regardless of the
    on-disk format: ``cache[(layout, M, N, K)]`` -> dict with keys
    ``{group_m, num_xcds, kernel}`` where ``num_xcds`` is ``None`` for FP8
    and ``kernel`` is ``None`` for BF16.
    """

    module: Any
    cache: dict[tuple[str, int, int, int], dict[str, Any]]
    dtype: Literal["bf16", "fp8"]
    block_size: int
    k_block: int
    default_group_m: int
    default_num_xcds: int | None


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


def _parse_bf16_cache(search_paths: list[Path]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    for search_path in search_paths:
        candidate = search_path / _BF16_CACHE_FILE
        if candidate.is_file():
            with candidate.open() as f:
                data = json.load(f)
            cache: dict[tuple[str, int, int, int], dict[str, Any]] = {}
            for row in data.get("rows", []):
                m = int(row["M"])
                n = int(row["N"])
                k = int(row["K"])
                for layout in ("rcr", "rrr", "crr"):
                    cache[(layout, m, n, k)] = {
                        "group_m": int(row.get(f"{layout}_gm", 4)),
                        "num_xcds": int(row.get(f"{layout}_xcd", 8)),
                        "kernel": None,
                    }
            return cache
    return {}


def _parse_fp8_cache(search_paths: list[Path]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    for search_path in search_paths:
        candidate = search_path / _FP8_CACHE_FILE
        if candidate.is_file():
            with candidate.open() as f:
                raw = json.load(f)
            cache: dict[tuple[str, int, int, int], dict[str, Any]] = {}
            for key, value in raw.items():
                # Keys are "<layout>_<M>_<N>_<K>" with layout in {rcr, rrr, crr}.
                parts = key.split("_")
                if len(parts) != 4 or parts[0] not in {"rcr", "rrr", "crr"}:
                    continue
                try:
                    layout = parts[0]
                    m, n, k = int(parts[1]), int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                cache[(layout, m, n, k)] = {
                    "group_m": int(value.get("group_m", 4)),
                    "num_xcds": None,
                    "kernel": str(value.get("kernel", "8")),
                }
            return cache
    return {}


def load_bf16() -> HipKittenModule:
    """Import ``tk_bf16_layouts`` and parse the BF16 autotune cache.

    Re-runs both the ``importlib.import_module`` call and the JSON parse on
    every invocation; no memoization. The ``importlib`` step is cheap once
    Python has the module in ``sys.modules``, but the ``json.load`` of the
    autotune cache file is a fresh disk read every time.
    """
    module_name = os.environ.get(_HIPKITTEN_MODULE_ENV, _BF16_DEFAULT_MODULE)
    search_paths = _bf16_search_paths()
    module = _import_with_fallback(module_name, search_paths)
    cache = _parse_bf16_cache(search_paths)
    return HipKittenModule(
        module=module,
        cache=cache,
        dtype="bf16",
        block_size=int(getattr(module, "BLOCK_SIZE", 256)),
        k_block=int(getattr(module, "K_STEP", 64)),
        default_group_m=4,
        default_num_xcds=8,
    )


def load_fp8() -> HipKittenModule:
    """Import ``tk_fp8_layouts`` and parse the FP8 autotune cache.

    Same no-cache semantics as :func:`load_bf16`.
    """
    search_paths = _fp8_search_paths()
    module = _import_with_fallback(_FP8_MODULE, search_paths)
    cache = _parse_fp8_cache(search_paths)
    return HipKittenModule(
        module=module,
        cache=cache,
        dtype="fp8",
        block_size=int(getattr(module, "BLOCK_SIZE", 256)),
        k_block=int(getattr(module, "K_BLOCK", 128)),
        default_group_m=int(getattr(module, "DEFAULT_GROUP_M", 4)),
        default_num_xcds=None,
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
