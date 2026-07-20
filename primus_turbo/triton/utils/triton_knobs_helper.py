###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""AMDGPU Triton compiler knob helpers (shared across GEMM / grouped-GEMM kernels)."""

import contextlib
import functools
import os
from typing import Optional

import triton

from primus_turbo.pytorch.core.utils import is_gfx950

# Triton ``knobs.amd`` attributes touched by the turbo GEMM kernels.
_AMD_KNOB_ATTRS = ("use_async_copy", "scalarize_packed_fops", "use_block_pingpong")

# ping-pong / async-copy knobs disabled by the sparse-MLA attention kernels.
_AMD_PINGPONG_ATTRS = ("use_async_copy", "use_block_pingpong")
_AMD_PINGPONG_ENVS = ("TRITON_HIP_USE_ASYNC_COPY", "TRITON_HIP_USE_BLOCK_PINGPONG")

# Env-var fallback (older Triton without ``triton.knobs``).
_AMD_KNOB_ENVS = (
    "TRITON_HIP_USE_ASYNC_COPY",
    "AMDGCN_SCALARIZE_PACKED_FOPS",
    "TRITON_HIP_USE_BLOCK_PINGPONG",
)


def _amd_knobs_available() -> bool:
    return hasattr(triton, "knobs") and hasattr(triton.knobs, "amd")


@contextlib.contextmanager
def scoped_amd_triton_knobs(gfx942_enable: Optional[bool] = None):
    """Snapshot AMD Triton compiler knobs on entry, restore them on exit.

    On gfx950 the full gfx950 knob set (async_copy, scalarize_packed_fops,
    block_pingpong) is enabled for the duration of the scope.

    On gfx942, when ``gfx942_enable`` is not None, use_async_copy /
    scalarize_packed_fops are set to that value for the scope (NT/NN GEMMs
    benefit, but TN/wgrad regresses ~5-8% on MI300X so those callers pass
    ``False``); ``None`` leaves the gfx942 knobs at their defaults.
    block_pingpong is never touched on gfx942.

    All changes are restored on exit, so turbo's GEMM knobs never leak into
    other Triton kernels.
    """
    saved_attrs = {}
    if _amd_knobs_available():
        amd = triton.knobs.amd
        for name in _AMD_KNOB_ATTRS:
            if hasattr(amd, name):
                saved_attrs[name] = getattr(amd, name)
    saved_env = {name: os.environ.get(name) for name in _AMD_KNOB_ENVS}
    try:
        if is_gfx950():
            if _amd_knobs_available():
                # Only touch knobs that exist, so snapshot/restore stays
                # symmetric and we never AttributeError on a Triton build
                # missing one.
                amd = triton.knobs.amd
                for name in _AMD_KNOB_ATTRS:
                    if hasattr(amd, name):
                        setattr(amd, name, True)
            else:
                for name in _AMD_KNOB_ENVS:
                    os.environ[name] = "1"
        elif gfx942_enable is not None and _amd_knobs_available():
            # gfx942 touches only async_copy + scalarize (not block_pingpong).
            amd = triton.knobs.amd
            for name in ("use_async_copy", "scalarize_packed_fops"):
                if hasattr(amd, name):
                    setattr(amd, name, gfx942_enable)
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


@contextlib.contextmanager
def scoped_amd_triton_knobs_disabled():
    """Snapshot the AMD ping-pong / async-copy knobs, disable them for the scope, restore on exit.

    primus_turbo enables these globally for the GEMM kernels, but they pessimize the sparse-MLA
    attention kernels (~16-29% slower fwd, ~5-7% slower bwd on gfx950) and the wide dKV tiling
    overflows the 160 KB LDS limit with them on. They are read at compile time and are not part of
    Triton's compile cache key, so compiling a kernel inside this scope pins the non-ping-pong
    schedule for that kernel for the process while leaving every other (kernel compiled outside the
    scope) exactly as primus_turbo configured it.
    """
    saved_attrs = {}
    if _amd_knobs_available():
        amd = triton.knobs.amd
        for name in _AMD_PINGPONG_ATTRS:
            if hasattr(amd, name):
                saved_attrs[name] = getattr(amd, name)
    saved_env = {name: os.environ.get(name) for name in _AMD_PINGPONG_ENVS}
    try:
        if _amd_knobs_available():
            amd = triton.knobs.amd
            for name in _AMD_PINGPONG_ATTRS:
                if hasattr(amd, name):
                    setattr(amd, name, False)
        else:
            for name in _AMD_PINGPONG_ENVS:
                os.environ[name] = "0"
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


def scoped_amd_knobs(func=None, *, gfx942_enable: Optional[bool] = None):
    """Decorator: run ``func`` under :func:`scoped_amd_triton_knobs`.

    Apply to every turbo GEMM / grouped-GEMM entry point that toggles AMD knobs,
    so the knobs are active only while that kernel compiles/launches.

    Usable bare (``@scoped_amd_knobs``) or parametrised
    (``@scoped_amd_knobs(gfx942_enable=True)``) to pin a fixed gfx942 per-layout
    policy. For entry points whose layout is only known at call time (e.g. the
    unified dense blockwise kernel that dispatches NT/NN/TN internally), use the
    :func:`scoped_amd_triton_knobs` context manager directly instead.
    """
    if func is None:
        return functools.partial(scoped_amd_knobs, gfx942_enable=gfx942_enable)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with scoped_amd_triton_knobs(gfx942_enable=gfx942_enable):
            return func(*args, **kwargs)

    return wrapper
