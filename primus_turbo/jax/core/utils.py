###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import logging
import subprocess
from typing import Tuple

import jax.numpy as jnp
from jax import dtypes

from primus_turbo.jax._C import DType as TurboDType

logger = logging.getLogger(__name__)


def _parse_gfx_string(gfx_name: str) -> Tuple[int, int]:
    """Parse a gfx architecture string (e.g. 'gfx942', 'gfx950') into (major, minor)."""
    if gfx_name.startswith("gfx"):
        gfx_version = gfx_name[3:]
        if len(gfx_version) >= 2:
            try:
                major = int(gfx_version[0])
                minor = int(gfx_version[1])
                return (major, minor)
            except ValueError:
                pass
    return (0, 0)


def _get_capability_from_hip(device_id: int) -> Tuple[int, int]:
    """Get compute capability via HIP runtime (C++ extension)."""
    try:
        from primus_turbo.jax._C import get_device_compute_capability as _hip_get_cc

        return _hip_get_cc(device_id)
    except (ImportError, AttributeError):
        return (0, 0)


def _get_capability_from_rocm_agent(device_id: int) -> Tuple[int, int]:
    """Get compute capability via rocm_agent_enumerator (fallback)."""
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator", "-t", "GPU"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.debug(
                "rocm_agent_enumerator failed with return code %d: %s",
                result.returncode,
                result.stderr.strip(),
            )
            return (0, 0)
        agents = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and l.strip() != "gfx000"]
        if 0 <= device_id < len(agents):
            return _parse_gfx_string(agents[device_id])
    except Exception:
        pass
    return (0, 0)


@functools.lru_cache
def _get_device_compute_capability(device_id: int) -> Tuple[int, int]:
    """Get compute capability for a specific device.

    Tries multiple detection methods:
      1. HIP runtime via C++ extension (most reliable)
      2. rocm_agent_enumerator CLI tool (fallback)
    """
    cap = _get_capability_from_hip(device_id)
    if cap != (0, 0):
        return cap

    cap = _get_capability_from_rocm_agent(device_id)
    if cap != (0, 0):
        return cap

    logger.warning("Could not determine GPU compute capability for device %d", device_id)
    return (0, 0)


def get_device_compute_capability(device_id: int = 0) -> Tuple[int, int]:
    """Get compute capability of specified GPU or current default GPU."""
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
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
