###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import dtypes

from primus_turbo.jax._C import DType as TurboDType


@functools.lru_cache
def _get_device_compute_capability(device_id: int) -> Tuple[int, int]:
    """Get compute capability for a specific device."""
    devices = jax.devices("gpu")
    if device_id >= len(devices) or device_id < 0:
        raise ValueError(f"Device {device_id} not found")

    device = devices[device_id]
    device_kind = device.device_kind
    if device_kind.startswith("gfx"):
        gfx_version = device_kind[3:]
        if len(gfx_version) >= 2:
            try:
                major = int(gfx_version[0])
                minor = int(gfx_version[1])
                return (major, minor)
            except ValueError:
                pass

    return (0, 0)


def get_device_compute_capability(device_id: int = 0) -> Tuple[int, int]:
    """Get compute capability of specified GPU or current default GPU."""
    return _get_device_compute_capability(device_id)


def jnp_dtype_to_turbo_dtype(jnp_dtype):
    """Convert JAX NumPy dtype to Primus-Turbo DType."""
    jnp_dtype = dtypes.canonicalize_dtype(jnp_dtype)

    converter = {
        jnp.float32.dtype: TurboDType.kFloat32,
        jnp.float16.dtype: TurboDType.kFloat16,
        jnp.bfloat16.dtype: TurboDType.kBFloat16,
        jnp.int32.dtype: TurboDType.kInt32,
        jnp.int64.dtype: TurboDType.kInt64,
        jnp.float8_e4m3fn.dtype: TurboDType.kFloat8E4M3FN,
        jnp.float8_e4m3fnuz.dtype: TurboDType.kFloat8E4M3FNUZ,
        jnp.float8_e5m2.dtype: TurboDType.kFloat8E5M2,
        jnp.float8_e5m2fnuz.dtype: TurboDType.kFloat8E5M2FNUZ,
    }

    if jnp_dtype not in converter:
        raise ValueError(f"Unsupported {jnp_dtype=}")

    return converter[jnp_dtype]
