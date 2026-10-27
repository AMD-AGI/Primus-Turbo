###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Per-architecture choices of the FlyDSL DeepEP intranode kernels (V1 and V2).

``TRAITS`` holds the choices for the target FlyDSL compiles for (``FLYDSL_GPU_ARCH`` or the
device), read once at import time: a process builds for one architecture, and FlyDSL keys its
compile cache on the target.
"""

import functools
from typing import NamedTuple, Optional

from flydsl.runtime.device import get_rocm_arch


class ArchTraits(NamedTuple):
    """A NamedTuple: FlyDSL folds the globals a kernel reads into its cache key by value only for
    scalars and builtin containers, so an opaque object would let a changed trait reuse kernels cached
    with the old value."""

    name: str
    wave_size: int
    # buffer cache policy of a system-scope streaming load, which misses every GPU cache
    cpol_sys_load: int
    # buffer cache policy of a payload store into a peer's heap. The cheap fence publishes a tail
    # once every sending warp's stores are acknowledged: gfx942 / gfx950 acknowledge a store to the
    # uncached heap only from memory, gfx12 at the first cache its scope covers, so there the
    # stores need SCOPE_SYS (as `kSysScopePayloadStores` in the HIP kernels)
    cpol_sys_store: int
    # bytes of a row one copy_warp call loads before it stores them, by (op, num_channels)
    copy_piece_table: tuple
    # 16-byte row loads a combine reducer lane keeps in flight over all of a pass's sources; each
    # also holds a partial sum, so it bounds the reducer's VGPRs
    combine_loads_in_flight: int
    # spin-wait watchdog, in read_clock cycles (100 MHz), and whether it traps after printing or
    # prints and keeps spinning
    spin_timeout_cycles: int
    trap_on_timeout: bool

    def copy_piece_bytes(self, op: str, num_channels: int) -> int:
        for max_channels, piece in dict(self.copy_piece_table)[op]:
            if num_channels <= max_channels:
                return piece
        raise AssertionError("the last copy-piece entry must cover any channel count")


_INF = 1 << 30

# Measured on MI355X, 8 ranks, hidden 7168, a DeepEP forward + backward per step (num_sms -> best
# dispatch / combine): 20 -> 2 KiB / 8 KiB, 40 -> 1 KiB / 2 KiB, 64 and 80 -> 1 KiB / 1 KiB. gfx9's
# vmcnt counts stores too, so the loads of the next piece wait for this piece's (remote) stores;
# the more channels push at once, the smaller the piece that keeps a sender streaming.
_GFX9_COPY_PIECES = (
    ("dispatch", ((10, 2 * 1024), (_INF, 1024))),
    ("combine", ((10, 8 * 1024), (20, 2 * 1024), (_INF, 1024))),
)

_GFX9 = dict(
    wave_size=64,
    # SC0 (1) | NT (2) | SC1 (16)
    cpol_sys_load=0x13,
    cpol_sys_store=0,
    copy_piece_table=_GFX9_COPY_PIECES,
    combine_loads_in_flight=32,
    spin_timeout_cycles=3_000_000_000,
    trap_on_timeout=False,
)

# gfx12 buffer cache policy: TH in bits 2:0 (TH_NT = 1), SCOPE in bits 4:3 (SCOPE_SYS = 3 << 3)
_GFX12_SYS_STREAM = 0x1 | (0x3 << 3)

_TRAITS = {
    "gfx942": ArchTraits(name="gfx942", **_GFX9),
    "gfx950": ArchTraits(name="gfx950", **_GFX9),
    "gfx1250": ArchTraits(
        name="gfx1250",
        wave_size=32,
        cpol_sys_load=_GFX12_SYS_STREAM,
        cpol_sys_store=_GFX12_SYS_STREAM,
        # gfx12 counts loads and stores apart, so a piece only bounds the VGPRs of a copy: 8 KiB is 16
        # wave32 steps, 64 VGPRs. Fastest of the MI355X table and 2 / 4 / 8 KiB at 32-160 SMs on
        # MI455X (DSv3 shape, 2 and 4 ranks)
        copy_piece_table=(("dispatch", ((_INF, 8 * 1024),)), ("combine", ((_INF, 8 * 1024),))),
        combine_loads_in_flight=32,
        # 5 s, under the driver's 9 s queue preemption timeout: a queue still spinning when it is
        # evicted or torn down has left MES unrecoverable on MI455X, a trapped one has no waves left
        spin_timeout_cycles=500_000_000,
        trap_on_timeout=True,
    ),
}


@functools.lru_cache(maxsize=None)
def get_arch_traits(arch: Optional[str] = None) -> ArchTraits:
    """Traits of ``arch``, else of the target FlyDSL compiles for. Other targets get the gfx942 /
    gfx950 choices."""
    arch = (arch or get_rocm_arch()).split(":", 1)[0].lower()
    return _TRAITS.get(arch) or ArchTraits(name=arch, **_GFX9)


TRAITS = get_arch_traits()
