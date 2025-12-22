###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple

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


def is_fp4_dtype(dtype):
    TORCH_FP4_DTYPE = [
        torch.float4_e2m1fn_x2,
    ]
    return dtype in TORCH_FP4_DTYPE


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_mxfp4_support() -> Tuple[bool, str]:
    """Return if fp4 support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP4 execution.",
    )


def check_fp8_ocp_support() -> Tuple[bool, str]:
    """Return if fp8 ocp support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 OCP format.",
    )


def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if mxfp8 support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for MXFP8 execution.",
    )


###################################################

try:
    if check_fp8_ocp_support()[0]:
        float8_e4m3 = torch.float8_e4m3fn
        float8_e5m2 = torch.float8_e5m2
    else:
        float8_e4m3 = torch.float8_e4m3fnuz
        float8_e5m2 = torch.float8_e5m2fnuz
    if check_mxfp4_support()[0]:
        float4_e2m1fn_x2 = torch.float4_e2m1fn_x2
    else:
        float4_e2m1fn_x2 = None
except AttributeError:
    raise RuntimeError("Your PyTorch build does not support FP8 types.")

###################################################


class Format(Enum):
    """
    Supported FP8/FP4 formats.
    """

    E4M3 = auto()
    E5M2 = auto()
    E2M1_X2 = auto()
    HYBRID = auto()


class ScaleDtype(Enum):
    """
    Supported FP8/FP4 Scale data type.
    """

    FP32 = auto()
    E8M0 = auto()


class ScalingGranularity(Enum):
    """
    Supported FP8/FP4 scaling granularity.
    """

    TENSORWISE = auto()
    ROWWISE = auto()
    BLOCKWISE = auto()
    MX_BLOCKWISE = auto()


class ScalingStrategy(Enum):
    """
    Supported FP8/FP4 scaling strategy.
    """

    DYNAMIC = auto()
    # DELAYED_SCALING = auto() # TODO: undetermined


@dataclass
class MXScalingRecipe:
    """
    Supported MXFP8/MXFP4 scaling recipe.

    - use_2d_block: Whether to use 2D block in quantization. Available in MXFP8 and MXFP4.
    - use_sr: Whether to use stochastic rounding in quantization. Available in MXFP4.
    - philox_seed: The philox generator's seed for the stochastic rounding. Available in MXFP4.
    - philox_offset: The philox generator's offset for the stochastic rounding. Available in MXFP4.
    """

    use_2d_block: bool = False
    use_sr: bool = False
    philox_seed: Optional[int] = None
    philox_offset: Optional[int] = None


@dataclass
class Float8QuantConfig:
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    scale_dtype: ScaleDtype = ScaleDtype.FP32
    block_size: Optional[int] = None  # Default: not used for tensorwise/rowwise

    # Default scaling recipe. Assume b is weight.
    scaling_recipe: Dict[str, MXScalingRecipe] = field(
        default_factory=lambda: {
            "a_fwd": MXScalingRecipe(),
            "b_fwd": MXScalingRecipe(use_2d_block=True),
            "grad_bwd": MXScalingRecipe(),
            "a_bwd": MXScalingRecipe(),
            "b_bwd": MXScalingRecipe(use_2d_block=True),
        }
    )

    def __post_init__(self):
        if self.granularity == ScalingGranularity.BLOCKWISE:
            assert self.block_size is not None, "block_size must be set when granularity is BLOCKWISE"

        if self.granularity == ScalingGranularity.MX_BLOCKWISE:
            mx_support_block_size = [32]
            assert (
                self.block_size in mx_support_block_size
            ), f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"

            mx_support_scale_dtype = ScaleDtype.E8M0
            assert (
                self.scale_dtype == mx_support_scale_dtype
            ), f"scale_dtype should be {mx_support_scale_dtype} when granularity is MX_BLOCKWISE"


@dataclass
class Float4QuantConfig:
    format: Format = Format.E2M1_X2
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    scale_dtype: ScaleDtype = ScaleDtype.FP32
    block_size: int = 32

    # Default scaling recipe. Assume b is weight. Reference: https://arxiv.org/pdf/2509.25149
    scaling_recipe: Dict[str, MXScalingRecipe] = field(
        default_factory=lambda: {
            "a_fwd": MXScalingRecipe(),
            "b_fwd": MXScalingRecipe(use_2d_block=True),
            "grad_bwd": MXScalingRecipe(use_sr=True),
            "a_bwd": MXScalingRecipe(),
            "b_bwd": MXScalingRecipe(use_2d_block=True, use_sr=True),
        }
    )

    def __post_init__(self):
        assert (
            self.granularity == ScalingGranularity.MX_BLOCKWISE
        ), "Float4QuantConfig currently only supports MX_BLOCKWISE granularity"

        mx_support_block_size = [32]
        assert (
            self.block_size in mx_support_block_size
        ), f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"
        assert self.format == Format.E2M1_X2, "Format must be E2M1_X2 for Float4QuantConfig"

        mx_support_scale_dtype = ScaleDtype.E8M0
        assert (
            self.scale_dtype == mx_support_scale_dtype
        ), f"scale_dtype should be {mx_support_scale_dtype} when granularity is MX_BLOCKWISE"
