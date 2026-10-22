###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trace-time target facts for FlyDSL kernels shared between gfx9 (gfx942 / gfx950) and gfx1250.

Every helper here is resolved while a kernel is traced, against the arch FlyDSL compiles for
(``FLYDSL_GPU_ARCH`` or the detected device), so one kernel source emits the right encoding on
either generation and the gfx9 encodings stay exactly what they were.
"""

from flydsl.runtime.device import get_rocm_arch


def is_gfx1250_target() -> bool:
    return get_rocm_arch().startswith("gfx1250")


def hw_wave_size() -> int:
    """Lanes of one hardware wave: 32 on gfx1250, 64 on gfx9."""
    return 32 if is_gfx1250_target() else 64


# Buffer load / store cache policy (the ``aux`` operand of the raw buffer intrinsics).
#
# gfx9 spells scope with SC1:SC0 (0 CU, 1 SE, 2 device, 3 system) plus NT = 2; gfx12 spells it
# ``TH | (SCOPE << 3)`` with TH NT = 1 and SCOPE 0 CU, 1 SE, 2 device, 3 system. Passing a gfx9
# value to gfx1250 silently changes its meaning (19 would read as TH 3 at device scope), so a
# cross-rank publish would lose system visibility without any error.
_GFX9_NT = 2
_GFX9_SC1 = 16
_GFX9_SC0 = 1
_GFX12_TH_NT = 1
_GFX12_SCOPE_DEV = 2 << 3
_GFX12_SCOPE_SYS = 3 << 3


def cpol_device_stream() -> int:
    """Non-temporal access coherent across the device: data another CU of this GPU produced."""
    if is_gfx1250_target():
        return _GFX12_SCOPE_DEV | _GFX12_TH_NT
    return _GFX9_SC1 | _GFX9_NT


def cpol_system_stream() -> int:
    """Non-temporal access coherent at system scope: a store a peer GPU reads, or a peer's store read here."""
    if is_gfx1250_target():
        return _GFX12_SCOPE_SYS | _GFX12_TH_NT
    return _GFX9_SC1 | _GFX9_SC0 | _GFX9_NT
