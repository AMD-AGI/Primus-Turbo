# SPDX-License-Identifier: Apache-2.0
"""Producer/consumer LDS ring-buffer primitives for the CSA backward kernels.

A multi-stage (ping-pong) LDS ring buffer for the dominant gathered-backward
path of ``csa_bwd_full_kernel.py``. The ring lets a
*producer* step stream the next gathered ``[BLOCK_K, HEAD_DIM]`` tile
global->LDS (``buffer_load_lds``) into one stage while the *consumer* step still
issues the score-recompute / dV / dK MFMAs and the dgathered atomics out of the
*other*, already-resident stage. Overlapping the no-reuse HBM gathered loads
with the latency-bound MFMA chain hides the cold L2-miss latency that the prior
"issue ``buffer_load_lds`` -> immediate ``s_waitcnt(0)`` + ``s_barrier`` drain"
fully exposed on this 1-wave/wg kernel.

The ring "layout contract" lives here so the producer write address and the
consumer read index are derived from one shared definition and can never drift:
every stage occupies a contiguous ``ring_stride_elems`` f16 slice of the LDS
staging buffer, stage ``s`` starting at element ``s * ring_stride_elems``. The
per-column / per-lane offsets *inside* a stage are unchanged from the
single-buffer layout, so the MFMA A-frag and the ``g_vec`` consume reads are
bitwise-identical apart from the constant per-stage base.

All helpers return plain Python ints (compile-time ring geometry); they add no
MLIR ops and need no active Context.
"""
from __future__ import annotations

__all__ = [
    "ring_stage_elem_base",
    "ring_stage_byte_base",
    "vmcnt_waitcnt_bits",
]


def ring_stage_elem_base(stage, ring_stride_elems):
    """Element-index base of ring ``stage`` within the LDS staging buffer."""
    return int(stage) * int(ring_stride_elems)


def ring_stage_byte_base(stage, ring_stride_elems, elem_bytes=2):
    """Byte base of ring ``stage`` (``elem_bytes`` per element, f16 default)."""
    return int(stage) * int(ring_stride_elems) * int(elem_bytes)


def vmcnt_waitcnt_bits(n):
    """``s_waitcnt`` SIMM16 that waits for ``vmcnt <= n`` only (no lgkm/exp wait).

    CDNA4 (gfx950) ``s_waitcnt`` SIMM16 layout (vendor ISA, p.145):
      SIMM16[3:0]   = vmcnt[3:0]
      SIMM16[6:4]   = expcnt
      SIMM16[11:8]  = lgkmcnt
      SIMM16[15:14] = vmcnt[5:4]
    Setting expcnt=7 and lgkmcnt=15 disables waiting on those counters so the
    fence is a pure vmcnt(n) wait. Provided for completeness; the ring uses a
    full ``s_waitcnt(0)`` drain after the overlapped consume because the
    consume issues dgathered atomics (also vmcnt-tracked) that would otherwise
    perturb a partial vmcnt count.
    """
    n = int(n)
    return (((n >> 4) & 0x3) << 14) | (0xF << 8) | (0x7 << 4) | (n & 0xF)
