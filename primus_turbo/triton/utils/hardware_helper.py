from __future__ import annotations

import os

import torch
import triton


def _resolve_device_index(device: torch.device | int | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None

    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


def _get_device_arch_name(device_index: int) -> str:
    try:
        props = torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError, TypeError):
        return ""
    return getattr(props, "gcnArchName", "").split(":")[0]


def is_gfx950(device: torch.device | int | None = None) -> bool:
    """Check if the active device is gfx950 (CDNA4 / MI350X / MI355X)."""
    device_index = _resolve_device_index(device)
    return device_index is not None and _get_device_arch_name(device_index) == "gfx950"


def _get_device_name_upper(device_index: int) -> str | None:
    try:
        props = torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError, TypeError):
        return None
    return getattr(props, "name", "").upper() or None


def _is_mi355_quant_device_index(device_index: int) -> bool:
    if not is_gfx950(device_index):
        return False

    device_name = _get_device_name_upper(device_index)
    return device_name is not None and "MI355" in device_name


def is_mi355_quant_device(device: torch.device | None = None) -> bool:
    """Return whether the runtime target is specifically MI355 on gfx950."""
    device_index = _resolve_device_index(device)
    return device_index is not None and _is_mi355_quant_device_index(device_index)


_KNOBS_SET = False


def set_knobs_gfx950():
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
