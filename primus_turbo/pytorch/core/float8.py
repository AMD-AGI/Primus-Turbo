from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import get_device_compute_capability

__all__ = ["float8_e4m3", "float8_e5m2"]


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
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_fp8_ocp_support() -> Tuple[bool, str]:
    """Return if fp8 ocp support is available"""
    if not torch.version.hip:
        return True, ""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 OCP format.",
    )


###################################################

try:
    if check_fp8_ocp_support()[0]:
        float8_e4m3 = torch.float8_e4m3fn
        float8_e5m2 = torch.float8_e5m2
    else:
        float8_e4m3 = torch.float8_e4m3fnuz
        float8_e5m2 = torch.float8_e5m2fnuz
except AttributeError:
    raise RuntimeError("Your PyTorch build does not support FP8 types.")

###################################################


class Format(Enum):
    """
    Supported FP8 formats.
    """

    E4M3 = auto()
    E5M2 = auto()
    HYBRID = auto()


class ScalingGranularity(Enum):
    TENSORWISE = auto()
    ROWWISE = auto()
    BLOCKWISE = auto()


class ScalingStrategy(Enum):
    DYNAMIC = auto()
    # DELAYED_SCALING = auto() # TODO: undetermined


@dataclass
class Float8QuantConfig:
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    block_size: Optional[int] = None  # Default: not used for tensorwise/rowwise

    def __post_init__(self):
        if self.granularity == ScalingGranularity.BLOCKWISE and self.block_size is None:
            raise ValueError("block_size must be set when granularity is BLOCKWISE")


@dataclass
class MXQuantConfig(Float8QuantConfig):
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.BLOCKWISE
    block_size: int = 128  # Override: block_size required for blockwise

    def __post_init__(self):
        super().__post_init__()
