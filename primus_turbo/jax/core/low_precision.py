###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import jax.numpy as jnp

from primus_turbo.jax.core.utils import get_device_compute_capability

__all__ = [
    "float8_e4m3",
    "float8_e5m2",
    "Format",
    "ScalingGranularity",
    "ScalingStrategy",
    "Float8QuantConfig",
    "Float4QuantConfig",
]


def is_fp8_dtype(dtype):
    """Check if dtype is a FP8 dtype."""
    JAX_FP8_DTYPE = [
        jnp.float8_e4m3fn,
        jnp.float8_e4m3fnuz,
        jnp.float8_e5m2,
        jnp.float8_e5m2fnuz,
    ]
    return dtype in JAX_FP8_DTYPE


def is_fp4_dtype(dtype):
    """Check if dtype is a FP4 dtype."""
    # Note: JAX may not have FP4 support yet, this is for future compatibility
    JAX_FP4_DTYPE = []
    return dtype in JAX_FP4_DTYPE


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_fp4_support() -> Tuple[bool, str]:
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
        "Device compute capability gfx950 or higher required for FP8 execution.",
    )


###################################################

try:
    if check_fp8_ocp_support()[0]:
        float8_e4m3 = jnp.float8_e4m3fn
        float8_e5m2 = jnp.float8_e5m2
    else:
        float8_e4m3 = jnp.float8_e4m3fnuz
        float8_e5m2 = jnp.float8_e5m2fnuz
except AttributeError:
    raise RuntimeError("Your JAX build does not support FP8 types.")

###################################################


class Format(Enum):
    """
    Supported FP8/FP4 formats.
    """

    E4M3 = auto()
    E5M2 = auto()
    E2M1_X2 = auto()
    HYBRID = auto()


class ScalingGranularity(Enum):
    TENSORWISE = auto()
    ROWWISE = auto()
    BLOCKWISE = auto()
    MX_BLOCKWISE = auto()


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
        if self.granularity == ScalingGranularity.BLOCKWISE:
            assert self.block_size is not None, "block_size must be set when granularity is BLOCKWISE"

        if self.granularity == ScalingGranularity.MX_BLOCKWISE:
            mx_support_block_size = [32]
            assert (
                self.block_size in mx_support_block_size
            ), f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"


@dataclass
class Float4QuantConfig:
    format: Format = Format.E2M1_X2
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    block_size: Optional[int] = None

    def __post_init__(self):
        assert (
            self.granularity == ScalingGranularity.MX_BLOCKWISE
        ), "Float4QuantConfig currently only supports MX_BLOCKWISE granularity"

        if self.block_size is None:
            self.block_size = 32

        mx_support_block_size = [32]
        assert (
            self.block_size in mx_support_block_size
        ), f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"
        assert self.format == Format.E2M1_X2, "Format must be E2M1_X2 for Float4QuantConfig"
