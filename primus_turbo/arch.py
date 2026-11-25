###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU architectures and their names.

An architecture is named by its gfx ID (``gfx942``, ``gfx950``, ``gfx1250``). ROCm reports the
ID with target features attached (``gfx1250:sramecc+:xnack-``); ``normalize_arch`` strips them,
and new code should use it rather than splitting the name again.

The gfx ID is also the tag that architecture-specific code carries in its path
(``grouped_gemm_bf16_kernel_gfx1250.py``, ``attention/gfx1250_mla/``); product names are not
used in paths. What Primus-Turbo has for each architecture is in the kernel registry
(``primus_turbo.registry``).

``device_arch`` answers "which architecture is this device". It does not answer which target
FlyDSL compiles for (``flydsl.runtime.device.get_rocm_arch``, which ``FLYDSL_GPU_ARCH``
overrides) or which architectures the C++ kernels were built for (``setup.py``).

Importing this module imports nothing else; ``device_arch`` imports torch when it is called.
"""

from typing import Optional

KNOWN_ARCHS = ("gfx942", "gfx950", "gfx1250")

# The products of each architecture, for messages and for the "no product name in a path" check.
PRODUCTS = {
    "gfx942": ("MI300X", "MI325X"),
    "gfx950": ("MI355X",),
    "gfx1250": ("MI455X",),
}

# CUDA-style compute capability that ROCm's PyTorch reports for each architecture.
_COMPUTE_CAPABILITY = {(9, 4): "gfx942", (9, 5): "gfx950", (12, 5): "gfx1250"}


def normalize_arch(name: Optional[str]) -> str:
    """``"gfx1250:sramecc+:xnack-"`` -> ``"gfx1250"``; ``None`` or empty -> ``""``."""
    return str(name or "").split(":", 1)[0].strip().lower()


def device_arch(device=None) -> str:
    """Architecture of a torch device (default: the current one), ``""`` without a ROCm device.

    Read from the device's ``gcnArchName``; a PyTorch build that does not report it is
    identified by its compute capability instead.
    """
    try:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
    except Exception:
        return ""
    arch = normalize_arch(getattr(props, "gcnArchName", ""))
    if arch:
        return arch
    return _COMPUTE_CAPABILITY.get((getattr(props, "major", None), getattr(props, "minor", None)), "")
