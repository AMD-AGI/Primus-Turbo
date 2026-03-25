from __future__ import annotations

import functools
import os

import torch
import triton


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if is_compiling is not None and is_compiling():
            return True

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if is_compiling is not None and is_compiling():
            return True

    return False


@functools.lru_cache(maxsize=1)
def _get_current_target():
    try:
        return triton.runtime.driver.active.get_current_target()
    except (AttributeError, TypeError):
        return None


@functools.lru_cache(maxsize=1)
def _is_gfx950() -> bool:
    """Check if the active Triton target is gfx950 (CDNA4 / MI350X / MI355X)."""
    target = _get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


@functools.lru_cache(maxsize=8)
def _get_device_name_upper(device_index: int) -> str | None:
    try:
        return torch.cuda.get_device_name(device_index).upper()
    except (AssertionError, RuntimeError, TypeError):
        return None


@functools.lru_cache(maxsize=8)
def _is_mi355_quant_device_index(device_index: int) -> bool:
    if not torch.cuda.is_available() or not _is_gfx950():
        return False

    device_name = _get_device_name_upper(device_index)
    return device_name is not None and "MI355" in device_name


def _is_mi355_quant_device(device: torch.device | None = None) -> bool:
    """Return whether the runtime target is specifically MI355 on gfx950."""
    if _is_torch_compiling():
        return False

    if device is None:
        if not torch.cuda.is_available():
            return False
        device_index = torch.cuda.current_device()
    else:
        device_index = device.index if device.index is not None else torch.cuda.current_device()

    return _is_mi355_quant_device_index(device_index)


_KNOBS_SET = False


def _set_knobs_gfx950():
    """Enable AMD compiler knobs for gfx950 (async_copy, block_pingpong, scalarize)."""
    global _KNOBS_SET
    if _KNOBS_SET:
        return
    _KNOBS_SET = True
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = True
        triton.knobs.amd.scalarize_packed_fops = True
        triton.knobs.amd.use_block_pingpong = True
    else:
        os.environ.setdefault("TRITON_HIP_USE_ASYNC_COPY", "1")
        os.environ.setdefault("AMDGCN_SCALARIZE_PACKED_FOPS", "1")
        os.environ.setdefault("TRITON_HIP_USE_BLOCK_PINGPONG", "1")
