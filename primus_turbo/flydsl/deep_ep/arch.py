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

"""Per-architecture choices of the FlyDSL DeepEP intranode kernels.

The kernel builders read these at trace time, so each target gets its own IR. FlyDSL already
keys its compile cache on the target, so the traits need not appear in a launcher signature.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.runtime.device import get_rocm_arch

# gfx94x / gfx950 buffer cache policy: SC0 (1) | SC1 (16) is system scope, NT (2) streaming.
_GFX9_SYS_STREAM_LOAD = 0x13
# gfx12 buffer cache policy: TH in bits 2:0, SCOPE in bits 4:3. TH_NT (1) | SCOPE_SYS (3 << 3).
_GFX12_SYS_STREAM_LOAD = 0x1 | (0x3 << 3)


def _asm(text: str) -> None:
    llvm.inline_asm(res=None, operands_=[], asm_string=text, constraints="~{memory}", has_side_effects=True)


@dataclass(frozen=True)
class ArchTraits:
    name: str
    wave_size: int
    # buffer-load cache policy of a system-scope streaming read (misses every GPU cache)
    cpol_sys_load: int
    # Publish a ring tail as: every sending warp waits for its own stores, then a relaxed store
    # (the uncached heap makes an acknowledged store visible system-wide). Without it the
    # publish is a system-scope release and the poll an acquire.
    cheap_fence: bool
    # a load through a descriptor with num_records = 0 reads 0 (combine skips unrouted ranks so)
    oob_load_reads_zero: bool
    has_named_barrier: bool
    # bytes of a row one copy_warp call loads before it stores them, by (op, num_channels)
    copy_piece_table: tuple
    # 16-byte row loads a combine reducer lane keeps in flight over all of a pass's sources;
    # each also holds a partial sum, so it is what bounds the reducer's VGPRs
    combine_loads_in_flight: int
    # spin-wait watchdog, in read_clock cycles (100 MHz on gfx942 / gfx950 / gfx1250), and
    # whether it traps after printing or prints and keeps spinning
    spin_timeout_cycles: int
    trap_on_timeout: bool

    @property
    def is_gfx12(self) -> bool:
        return self.name.startswith("gfx12")

    def copy_piece_bytes(self, op: str, num_channels: int) -> int:
        for max_channels, piece in dict(self.copy_piece_table)[op]:
            if num_channels <= max_channels:
                return piece
        raise AssertionError("the last copy-piece entry must cover any channel count")


# Functions, not ArchTraits methods: the AST rewriter carries a local whose method is called in a
# dynamic loop as loop state, and traits are no IR value.
def wait_all(traits: ArchTraits) -> None:
    """Wait for every outstanding memory op of this warp."""
    if traits.is_gfx12:
        _asm("s_wait_dscnt 0\n\ts_wait_kmcnt 0\n\ts_wait_loadcnt 0\n\ts_wait_storecnt 0")
    else:
        fx.rocdl.s_waitcnt(0)


def wait_stores(traits: ArchTraits) -> None:
    """Wait for this warp's outstanding stores (gfx9 counts them with the loads on vmcnt)."""
    if traits.is_gfx12:
        _asm("s_wait_storecnt 0")
    else:
        fx.rocdl.s_waitcnt(0)


def spin_timeout_hit(traits: ArchTraits) -> None:
    """After a watchdog's report: abort the queue where the traits say so."""
    if traits.trap_on_timeout:
        _asm("s_trap 2")


_INF = 1 << 30

# Measured on MI355X, 8 ranks, hidden 7168, a DeepEP forward + backward per step (num_sms -> best
# dispatch / combine): 20 -> 2 KiB / 8 KiB, 40 -> 1 KiB / 2 KiB, 64 and 80 -> 1 KiB / 1 KiB. gfx9's
# vmcnt counts stores too, so the loads of the next piece wait for this piece's (remote) stores;
# the more channels push at once, the smaller the piece that keeps a sender streaming.
_GFX9_COPY_PIECES = (
    ("dispatch", ((10, 2 * 1024), (_INF, 1024))),
    ("combine", ((10, 8 * 1024), (20, 2 * 1024), (_INF, 1024))),
)
# gfx12 counts loads and stores apart, so the piece only bounds the VGPRs of a copy
# (a wave32 step is 512 B, 4 VGPRs a lane). Placeholder until scanned on MI455X.
_GFX12_COPY_PIECES = (
    ("dispatch", ((_INF, 2 * 1024),)),
    ("combine", ((_INF, 2 * 1024),)),
)

_GFX9 = dict(
    wave_size=64,
    cpol_sys_load=_GFX9_SYS_STREAM_LOAD,
    cheap_fence=True,
    oob_load_reads_zero=True,
    has_named_barrier=False,
    copy_piece_table=_GFX9_COPY_PIECES,
    combine_loads_in_flight=32,
    spin_timeout_cycles=3_000_000_000,
    trap_on_timeout=False,
)

_TRAITS = {
    "gfx942": ArchTraits(name="gfx942", **_GFX9),
    "gfx950": ArchTraits(name="gfx950", **_GFX9),
    "gfx1250": ArchTraits(
        name="gfx1250",
        wave_size=32,
        cpol_sys_load=_GFX12_SYS_STREAM_LOAD,
        cheap_fence=False,
        # probed: num_records = 0 and fully out-of-range loads read 0, such stores are dropped.
        # A 16-byte access straddling the end is not checked per dword (only aligned ones are used)
        oob_load_reads_zero=True,
        has_named_barrier=True,
        copy_piece_table=_GFX12_COPY_PIECES,
        # 32 spills 73 VGPRs at 2 ranks against the 128 cap at 1024 threads, 12 leaves 2
        combine_loads_in_flight=12,
        # 5 s, under the driver's 9 s queue preemption timeout: a queue still spinning when it
        # is evicted or torn down left MES unrecoverable, a trapped one has no waves left
        spin_timeout_cycles=500_000_000,
        trap_on_timeout=True,
    ),
}


@functools.lru_cache(maxsize=None)
def get_arch_traits(arch: Optional[str] = None) -> ArchTraits:
    """Traits of ``arch``, else of the target FlyDSL compiles for (``FLYDSL_GPU_ARCH`` or the device)."""
    arch = (arch or get_rocm_arch()).split(":", 1)[0].lower()
    if arch not in _TRAITS:
        raise RuntimeError(f"DeepEP FlyDSL intranode has no support for {arch}; supported: {sorted(_TRAITS)}")
    return _TRAITS[arch]
