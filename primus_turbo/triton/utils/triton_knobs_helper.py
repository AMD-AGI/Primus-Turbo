###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""AMDGPU Triton compiler knob helpers (shared across GEMM / grouped-GEMM kernels)."""

import contextlib
import functools
import os

import triton

# Triton ``knobs.amd`` attributes touched by the turbo GEMM kernels.
_AMD_KNOB_ATTRS = ("use_async_copy", "scalarize_packed_fops", "use_block_pingpong")

# Env-var fallback (older Triton without ``triton.knobs``).
_AMD_KNOB_ENVS = (
    "TRITON_HIP_USE_ASYNC_COPY",
    "AMDGCN_SCALARIZE_PACKED_FOPS",
    "TRITON_HIP_USE_BLOCK_PINGPONG",
)


def _amd_knobs_available() -> bool:
    return hasattr(triton, "knobs") and hasattr(triton.knobs, "amd")


def set_triton_knobs_gfx950() -> None:
    """Enable AMD compiler knobs for gfx950 (async_copy, block_pingpong, scalarize).

    Must be called from inside a :func:`scoped_amd_knobs`-decorated entry point
    so the change is reverted after the kernel compiles instead of leaking
    process-wide.  Unlike the previous implementation this applies the knobs on
    every call (the surrounding scope restores them afterwards).
    """
    if _amd_knobs_available():
        triton.knobs.amd.use_async_copy = True
        triton.knobs.amd.scalarize_packed_fops = True
        triton.knobs.amd.use_block_pingpong = True
    else:
        os.environ["TRITON_HIP_USE_ASYNC_COPY"] = "1"
        os.environ["AMDGCN_SCALARIZE_PACKED_FOPS"] = "1"
        os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"


@contextlib.contextmanager
def scoped_amd_triton_knobs():
    """Snapshot AMD Triton compiler knobs on entry, restore them on exit.

    Any knob flipping done inside the ``with`` block (via
    :func:`set_triton_knobs_gfx950` or a local ``_set_amd_knobs``) is undone on
    exit, so turbo's GEMM knobs never leak into other Triton kernels.
    """
    saved_attrs = {}
    if _amd_knobs_available():
        amd = triton.knobs.amd
        for name in _AMD_KNOB_ATTRS:
            if hasattr(amd, name):
                saved_attrs[name] = getattr(amd, name)
    saved_env = {name: os.environ.get(name) for name in _AMD_KNOB_ENVS}
    try:
        yield
    finally:
        if saved_attrs:
            amd = triton.knobs.amd
            for name, val in saved_attrs.items():
                setattr(amd, name, val)
        for name, val in saved_env.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val


def scoped_amd_knobs(func):
    """Decorator: run ``func`` under :func:`scoped_amd_triton_knobs`.

    Apply to every turbo GEMM / grouped-GEMM entry point that toggles AMD knobs,
    so the knobs are active only while that kernel compiles/launches.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with scoped_amd_triton_knobs():
            return func(*args, **kwargs)

    return wrapper
