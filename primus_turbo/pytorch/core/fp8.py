from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Tuple

import torch

from .utils import get_device_compute_capability


def is_fp8_dtype(dtype):
    TORCH_FP8_DTYPE = [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]
    return dtype in TORCH_FP8_DTYPE


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):  # MI300 and above
        return True, ""
    if get_device_compute_capability() < (9, 4):
        return False, "Device compute capability gfx942 or higher required for FP8 execution."
    return True, ""


def check_fp8_ocp_support() -> Tuple[bool, str]:
    if get_device_compute_capability() >= (9, 5):  # MI350 and above
        return True, ""
    return False, "Device compute capability gfx950 or higher required for FP8 OCP format."


class _FormatHelper(NamedTuple):
    fwd_dtype: torch.dtype
    bwd_dtype: torch.dtype


class Format(Enum):
    """
    Supported FP8 formats.

    Values
    ------
    E4M3 :
          All FP8 tensors are in e4m3 format
    E5M2 :
          All FP8 tensors are in e5m2 format
    HYBRID :
            FP8 tensors in the forward pass are in e4m3 format,
            FP8 tensors in the backward pass are in e5m2 format
    """

    E4M3 = _FormatHelper(
        fwd_dtype=torch.float8_e4m3fn if check_fp8_ocp_support()[0] else torch.float8_e4m3fnuz,
        bwd_dtype=torch.float8_e4m3fn if check_fp8_ocp_support()[0] else torch.float8_e4m3fnuz,
    )
    E5M2 = _FormatHelper(
        fwd_dtype=torch.float8_e5m2 if check_fp8_ocp_support()[0] else torch.float8_e5m2fnuz,
        bwd_dtype=torch.float8_e5m2 if check_fp8_ocp_support()[0] else torch.float8_e5m2fnuz,
    )
    HYBRID = _FormatHelper(fwd_dtype=E4M3.fwd_dtype, bwd_dtype=E5M2.bwd_dtype)
