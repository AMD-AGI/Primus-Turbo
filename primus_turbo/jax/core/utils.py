###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from typing import Tuple

import jax


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
