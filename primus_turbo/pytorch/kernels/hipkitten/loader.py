###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HipKittens module loader (process-singleton, no-JSON edition).

The HipKittens project ships a prebuilt Python extension per precision:

  * ``tk_bf16_layouts``  - BF16 dense + grouped (analysis/bf16_gemm/mi350x)
  * ``tk_fp8_layouts``   - FP8 tensorwise dense (analysis/fp8_gemm/mi350x)

Each lives under ``$PRIMUS_TURBO_HIPKITTEN_PATH/analysis/<dtype>_gemm/mi350x``.
The historical autotune-cache JSON files in those directories are *offline*
benchmark notes — useful at development time to derive the rules in
``config.py::select_default_config``, but never read at runtime.

This module's only state is the per-precision singleton stored in
``_BF16_SINGLETON`` / ``_FP8_SINGLETON``: a frozen :class:`HipKittenModule`
that bundles the imported ``.so`` with pre-resolved layout-keyed callables
(``gemm_rcr`` / ``gemm_rrr`` / ``gemm_crr`` and the optional FP8 ``_dscale``
variants and the optional grouped ``grouped_{rcr,rrr,crr}`` launchers).
The first :func:`load_bf16` / :func:`load_fp8` call constructs the
singleton; every subsequent call returns it as a single attribute load.
This avoids two sources of per-dispatch host overhead:

  1. Reconstructing the :class:`HipKittenModule` dataclass per call
     (~3-5 us in observed measurement on MI355X).
  2. Re-doing ``getattr(module, "gemm_<layout>")`` on every dispatch
     (~0.5-1 us each, of which the dispatch path issues 1-3).

Together these two changes shaved ~5 us off the BF16 dense forward
turbo-vs-direct overhead on 4096x4096x4096 (round 3 measurement: HK turbo
122 us → 117 us). The singletons are *constants*: they do not depend on
input shape, dtype, or data, so they are not "caches" in the project-policy
sense (which forbids per-shape / per-data caching).

NOTE: per the project policy, this module MUST NOT keep any per-shape cache
(no dict / weakref / data_ptr / _version / lru_cache / TTL). Process-level
singletons that depend only on environment variables (which `.so` to load)
are explicitly NOT in that prohibition — they are the moral equivalent of
``import torch``: a one-time process-wide constant.
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
    """A loaded HipKittens precision module + alignment constants + pre-resolved
    layout-keyed callables.

    ``block_size`` and ``k_block`` come straight from the binding (read once
    at import time via ``getattr``). They are used by callers to enforce
    alignment in ``can_handle``; we cache the values on the dataclass so
    ``can_handle`` does not re-issue ``getattr`` per dispatch.

    The ``gemm_*`` and ``grouped_*`` attributes hold direct refs to the
    binding's pybind11 callables, looked up once at import time. The
    dispatcher would otherwise pay ~0.5-1 us per call for ``getattr`` on
    the binding; for the BF16 4096x4096x4096 forward (kernel ≈ 100 us)
    that is a measurable fraction of host overhead, especially in the
    BF16 backward path which issues 3 dispatches per autograd step.

    None of these attributes depend on input shape / dtype / data: they
    are process-wide constants, computed once. They are NOT a per-shape
    cache.

    The grouped slots are aliased — see :func:`_resolve_grouped_attr` —
    so this dataclass exposes ``grouped_{rcr,rrr,crr}`` regardless of
    whether the underlying ``.so`` ships them as ``grouped_*`` (FP8,
    new HK builds) or with the legacy ``_balanced`` suffix (the BF16
    ``.so`` shipped today). New HK bindings MUST drop the
    ``_balanced`` suffix; balanced / unbalanced is a workload property,
    not an entrypoint property.
    """

    module: Any
    dtype: Literal["bf16", "fp8"]
    block_size: int
    k_block: int
    gemm_rcr: Any
    gemm_rrr: Any
    gemm_crr: Any
    # FP8-only device-pointer scale entries. None on BF16, and on FP8
    # builds where the binding does not yet expose ``gemm_*_dscale``.
    gemm_rcr_dscale: Any
    gemm_rrr_dscale: Any
    gemm_crr_dscale: Any
    # Grouped launchers (None when the binding does not expose them).
    grouped_rcr: Any
    grouped_rrr: Any
    grouped_crr: Any
    # FP8-only device-pointer scale grouped variants. None on BF16, and
    # on FP8 builds where the binding does not yet expose
    # ``grouped_*_dscale``. Mirrors the ``gemm_*_dscale`` family — saves
    # the per-dispatch ``.item()`` host sync the host-scalar grouped
    # path would otherwise pay (~10-20 us / dispatch on small grouped
    # FP8 shapes where the kernel itself is only ~1 ms).
    grouped_rcr_dscale: Any
    grouped_rrr_dscale: Any
    grouped_crr_dscale: Any
    # Variable-K grouped launchers (CRR-only — the var-K backward path
    # only uses CRR layout). ``grouped_variable_k_crr`` exists on both
    # BF16 and FP8 bindings; ``grouped_variable_k_crr_dscale`` is FP8-
    # only (None on BF16). Pre-resolving these saves the per-call
    # ``getattr(hk.module, "grouped_variable_k_crr", None)`` × 2 that
    # ``GroupedGEMM(FP8)?VariableKHipKittenBackend.execute`` previously
    # paid on every backward dB launch (~34 ns / call host-side; small
    # but the var-K execute body is otherwise pure-Python overhead and
    # this brings it to parity with the dense forward execute body
    # which has been pre-resolving since R3 of the BF16 series).
    grouped_variable_k_crr: Any
    grouped_variable_k_crr_dscale: Any

    def gemm(self, layout: str) -> Any:
        """Return the dense kernel callable for ``layout``."""
        if layout == "rcr":
            return self.gemm_rcr
        if layout == "rrr":
            return self.gemm_rrr
        return self.gemm_crr

    def gemm_dscale(self, layout: str) -> Any:
        """Return the device-scale dense kernel callable for ``layout`` or None."""
        if layout == "rcr":
            return self.gemm_rcr_dscale
        if layout == "rrr":
            return self.gemm_rrr_dscale
        return self.gemm_crr_dscale

    def grouped(self, layout: str) -> Any:
        """Return the grouped launcher for ``layout`` or None.

        ``None`` means the underlying ``.so`` does not ship a grouped
        entrypoint for the precision; callers must fall back to a
        per-group :func:`...dispatch.dense_run` loop.
        """
        if layout == "rcr":
            return self.grouped_rcr
        if layout == "rrr":
            return self.grouped_rrr
        return self.grouped_crr

    def grouped_dscale(self, layout: str) -> Any:
        """Return the device-scale grouped launcher for ``layout`` or None.

        FP8-only. None on BF16 (no scaling) and on FP8 builds where the
        binding does not yet expose the ``_dscale`` grouped variant.
        """
        if layout == "rcr":
            return self.grouped_rcr_dscale
        if layout == "rrr":
            return self.grouped_rrr_dscale
        return self.grouped_crr_dscale

    def has_grouped(self) -> bool:
        return self.grouped_rcr is not None


def _resolve_grouped_attr(module: Any, layout: str) -> Any:
    """Pull the grouped launcher off ``module`` for ``layout``.

    Tries the new (preferred) name ``grouped_<layout>`` first, then falls
    back to the legacy ``grouped_<layout>_balanced`` for the BF16 ``.so``
    shipped today. Returns ``None`` if neither attribute exists.
    """
    return getattr(module, f"grouped_{layout}", None) or getattr(
        module, f"grouped_{layout}_balanced", None
    )


def _build_module(module: Any, dtype: Literal["bf16", "fp8"], default_bs: int, default_kb: int) -> "HipKittenModule":
    return HipKittenModule(
        module=module,
        dtype=dtype,
        block_size=int(getattr(module, "BLOCK_SIZE", default_bs)),
        k_block=int(getattr(module, "K_STEP" if dtype == "bf16" else "K_BLOCK", default_kb)),
        gemm_rcr=getattr(module, "gemm_rcr"),
        gemm_rrr=getattr(module, "gemm_rrr"),
        gemm_crr=getattr(module, "gemm_crr"),
        gemm_rcr_dscale=getattr(module, "gemm_rcr_dscale", None),
        gemm_rrr_dscale=getattr(module, "gemm_rrr_dscale", None),
        gemm_crr_dscale=getattr(module, "gemm_crr_dscale", None),
        grouped_rcr=_resolve_grouped_attr(module, "rcr"),
        grouped_rrr=_resolve_grouped_attr(module, "rrr"),
        grouped_crr=_resolve_grouped_attr(module, "crr"),
        grouped_rcr_dscale=getattr(module, "grouped_rcr_dscale", None),
        grouped_rrr_dscale=getattr(module, "grouped_rrr_dscale", None),
        grouped_crr_dscale=getattr(module, "grouped_crr_dscale", None),
        grouped_variable_k_crr=getattr(module, "grouped_variable_k_crr", None),
        grouped_variable_k_crr_dscale=getattr(
            module, "grouped_variable_k_crr_dscale", None
        ),
    )


# Process-wide singletons. The first ``load_*`` call populates them; every
# subsequent call returns the same instance with a single global-attribute
# load. The dataclass is frozen, so handing the same instance to many
# threads is safe (no per-thread pre-resolution required).
_BF16_SINGLETON: "HipKittenModule | None" = None
_FP8_SINGLETON: "HipKittenModule | None" = None


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
    """Return the process-wide BF16 HipKittens module.

    First call constructs the :class:`HipKittenModule` (imports the ``.so``
    and pre-resolves the layout-keyed callables). Subsequent calls return
    the same singleton — a single global-attribute load.
    """
    global _BF16_SINGLETON
    if _BF16_SINGLETON is not None:
        return _BF16_SINGLETON
    module_name = os.environ.get(_HIPKITTEN_MODULE_ENV, _BF16_DEFAULT_MODULE)
    search_paths = _bf16_search_paths()
    module = _import_with_fallback(module_name, search_paths)
    _BF16_SINGLETON = _build_module(module, "bf16", default_bs=256, default_kb=64)
    return _BF16_SINGLETON


def load_fp8() -> HipKittenModule:
    """Return the process-wide FP8 HipKittens module (singleton)."""
    global _FP8_SINGLETON
    if _FP8_SINGLETON is not None:
        return _FP8_SINGLETON
    search_paths = _fp8_search_paths()
    module = _import_with_fallback(_FP8_MODULE, search_paths)
    _FP8_SINGLETON = _build_module(module, "fp8", default_bs=256, default_kb=128)
    return _FP8_SINGLETON


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
