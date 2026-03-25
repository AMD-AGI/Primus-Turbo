from __future__ import annotations

import functools
import os

import torch
import triton


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


def _is_mi355_quant_device(device: torch.device | None = None) -> bool:
    """Return whether the runtime target is specifically MI355 on gfx950."""
    if not torch.cuda.is_available() or not _is_gfx950():
        return False

    if device is None:
        device_index = torch.cuda.current_device()
    else:
        device_index = device.index if device.index is not None else torch.cuda.current_device()

    device_name = _get_device_name_upper(device_index)
    return device_name is not None and "MI355" in device_name


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
