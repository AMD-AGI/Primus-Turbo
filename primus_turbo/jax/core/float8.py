###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

__all__ = [
    "float8_e4m3",
    "float8_e5m2",
    "Format",
    "ScalingGranularity",
    "ScalingStrategy",
    "Float8QuantConfig",
]


@functools.lru_cache
def _get_device_compute_capability(device_id: int) -> Tuple[int, int]:
    """Get compute capability for a specific device."""
    devices = jax.devices("gpu")
    if device_id >= len(devices):
        raise ValueError(f"Device {device_id} not found")

    device = devices[device_id]
    # For AMD GPUs, parse the device kind (e.g., "gfx942" -> (9, 4, 2))
    device_kind = device.device_kind

    # Extract gfx version (e.g., "gfx942" -> "942")
    if device_kind.startswith("gfx"):
        gfx_version = device_kind[3:]
        if len(gfx_version) >= 2:
            major = int(gfx_version[0])
            minor = int(gfx_version[1])
            return (major, minor)

    # Default fallback
    return (0, 0)


def get_device_compute_capability(device_id: int = 0) -> Tuple[int, int]:
    """Get compute capability of specified GPU or current default GPU."""
    return _get_device_compute_capability(device_id)


def is_fp8_dtype(dtype):
    """Check if dtype is a FP8 dtype."""
    JAX_FP8_DTYPE = [
        jnp.float8_e4m3fn,
        jnp.float8_e4m3fnuz,
        jnp.float8_e5m2,
        jnp.float8_e5m2fnuz,
    ]
    return dtype in JAX_FP8_DTYPE


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 execution.",
    )


def check_fp8_ocp_support() -> Tuple[bool, str]:
    """Return if fp8 ocp support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 OCP format.",
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
    Supported FP8 formats.
    """

    E4M3 = auto()
    E5M2 = auto()
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
            assert self.format == Format.E4M3, "Format must be set E4M3 when granularity is MX_BLOCKWISE"
