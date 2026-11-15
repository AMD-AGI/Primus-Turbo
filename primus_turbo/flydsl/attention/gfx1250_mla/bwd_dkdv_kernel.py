###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""dK / dV of the gfx1250 MLA attention backward over 192-row kv blocks, plus the dS^T scratch (k_dkdv).

The FlyDSL form of the HipKittens kernel k_dkdv96ds (csrc/kernels/attention/hipkittens/gfx1250_mla) with two of its
96-row kv blocks per item and persistent workgroups: block 128 = 4 waves (one per SIMD), at most one workgroup per CU
(grid = min(CUs, items)). An item is one 192-row kv block of one (batch, head) of the launch (the host launches the
heads in chunks to bound the scratch): kv rows [192 bid, 192 bid + 192), wave w owning the 48 rows from 192 bid +
48 w (NKV 3 sub-tiles of 16). Workgroup g runs the items of its own list back to back; the host makes the lists and
the table of their per-item values (bwd_dkdv_items: one row per item and wave, a row of zeros ends a list). The last
block's rows past Skv (Skv % 192 = 64 or 128) are idle: their K / V are not read (the TDM extent stops at Skv; a kv
row's arithmetic involves only its own row), every tile of that block is masked (those rows are past every query's
diagonal) and their dS^T / dK / dV stores fall outside the descriptors. An item is one sequence of query tiles (q
pairs of 32 queries; MHA, the kernel is specialized for G = Hq / Hkv = 1) over its block, the n_mask tiles that
straddle the diagonal with the causal mask on.
Q / dO tiles go through a DEPTH-stage LDS ring filled by the TDM (a stage = [32][D_V] dO + [32][D_QK] Q, each wave
moving 8 of the 32 rows) that all four waves read: every tile streamed from L2 feeds 192 kv rows (half the Q / dO
stream of 96-row blocks). K / V of the wave's rows stay in registers (the S / dP B operands; moved by the TDM into a
per-wave LDS staging area first); dV / dK accumulate in registers and leave through per-wave row-major images in
the wave's LDS image area (whole-line global stores).

The work is generated code (bwd_dkdv_sched: the HipKittens generator's software-pipelined schedule of a tile, with
this kernel's numerics, around it the item loop: one inline asm statement on a static plan of VGPRs
v[bwd_dkdv_sched.BASE:1024] and of the SGPRs). This file computes its operands, all VGPRs pinned below
bwd_dkdv_sched.BASE: the lane-dependent address parts, and a pool VGPR whose lanes hold the per-workgroup scalars
(the generated code owns every SGPR LLVM allocates, so nothing can reach it in one; nothing of this file's code is
live across it).

Every bf16 dS^T tile the kernel feeds its dK WMMA is also stored into the dS scratch (the HipKittens dslayout.h,
version 2; bwd_dkdv_items has its index math): one region per (batch, q head) of the launch, inside it the kv16 rows
in order, row kt holding the q pairs from qp_first(kt) = max(0, kt // 2 - (Skv - Sq) // 32) on as 1 KiB blocks
[16 kv][32 q] in the dK operand's chunk order, with the softmax scale inside dS. k_dqds reads it back for dQ = dS K.

Numerics: P = exp2(c1 S + NL), c1 = scale * log2(e), dS = P (scale dP + ND), with k_delta_bshd's NL = -log2(e) lse
and ND = -scale delta, every accumulator one WMMA per tile in tile order. A wave's per-row work does not depend on the
block height, the schedule or the order of the items: its kv rows see the query tiles in ascending order, the ones
before their diagonal fully masked (P = dS = 0, every accumulator still +0), so dK, dV and the dS^T bytes are those
of 96-row (and 64-row) blocks.

Ring protocol of an item (one barrier per tile; every prefetch is clamped to the last tile T_{N-1}, the ones past it
are NULL): T_0, the K / V TDM ops, T_1 .. T_{DEPTH-2}, tensorcnt TW -> BARRIER -> tile 0's hh0 readback; tile j in
stage j mod DEPTH: TDM of T_{j+DEPTH-1} into the stage of tile j - 1, hh1 readback, S / dP, softmax, tr16 dO^T / Q^T,
dV, tensorcnt TW, BARRIER, dK with the next stage's hh0 readback; exit: tensorcnt 0 (FINAL's images use the wave's
image area, not the ring).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): readfirstlane and disable_xdl_arb_stall have no stable API in FlyDSL 0.3.4.1.
from flydsl.expr import rocdl

from primus_turbo.flydsl.utils.prims import LOG2E

from . import SOURCE_DIGEST
from . import bwd_dkdv_sched as sched
from .bwd_dkdv_sched import DEPTH, HH_OFF, NKV, QOFF, STAGE_B, TDM_ROWS, XK, XV
from .common import inline_region, reserve_vgprs

D_QK = 192
D_V = 128
NW = 4  # waves per workgroup, one per SIMD (one workgroup per CU)
# LDS (bytes from the workgroup's base). The Q / dO ring: stage s holds query rows 0..15 (hh0) of its tile at
# s * STAGE_B and rows 16..31 (hh1) HH_OFF = 64 KiB higher, per half the [16][XV] dO rows, then the [16][XK] Q rows
# (QOFF). LDS has two read ports that serve one 64 KiB segment each per cycle, and split by rows, every wave's tr16
# operand pair and the stage's reads as a whole go half to each (split by tensor, the four waves' lock-step bursts
# used one port); one segment apart, the halves use both ports wherever the workgroup's LDS allocation starts.
# After the ring, per wave: its staging area of K / V rows (INIT's TDM: [48][XV] of V, then [48][XK] of K), then per
# wave an image area in the same layout for FINAL's dV / dK (waves 0..2 after the staging areas, wave 3 in the hole
# between the ring's hh0 and hh1 halves: measured 0.2% faster than the images in the staging area, which the next
# item's K / V ops write right after).
STG0 = HH_OFF + DEPTH * STAGE_B
EPI_W = 16 * NKV * (XV + XK)
EPI_DK = 16 * NKV * XV
IMG3 = DEPTH * STAGE_B  # wave 3's image area
LDS_B = STG0 + (2 * NW - 1) * EPI_W
RESERVE_V4 = 255  # common.reserve_vgprs: 16-byte values kept live
assert XK == D_QK * 2 + 16 and XV == D_V * 2 + 16  # LDS rows: the tensor's row + 16 B of padding
assert STAGE_B == 16 * (XV + XK) and QOFF == 16 * XV and TDM_ROWS * NW == 32
assert IMG3 + EPI_W <= HH_OFF and LDS_B <= 327680  # one CU's LDS


def _uniform(x):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.as_ir_value(x)))


@flyc.kernel(name="k_mla_bwd_dkdv", known_block_size=[32 * NW, 1, 1])
def k_dkdv(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    NL: fx.Tensor,
    ND: fx.Tensor,
    DV_: fx.Tensor,
    DK: fx.Tensor,
    DS: fx.Tensor,
    TBL: fx.Tensor,
    scale: fx.Float32,
    Hq: fx.Int32,
    nl_bytes: fx.Int32,
    nslot: fx.Int32,
):
    rocdl.disable_xdl_arb_stall()  # SCHED_MODE.DISABLE_XDL_ARB_STALL
    smem = fx.SharedAllocator().allocate(LDS_B)
    lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    reserve_vgprs(nslot < fx.Int32(0), lds0, RESERVE_V4)  # never taken
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & fx.Int32(31)
    wv = _uniform(tid) >> fx.Int32(5)
    half = lane >> fx.Int32(4)
    row = lane & fx.Int32(15)

    # ---- lane-dependent operand VGPRs (pinned to v0..v14): the K / V fragments' lane bases in the staging area
    # (INIT); lane bases of the readback rows and the tr16 pieces (half, stage and entry offsets are immediates); NL /
    # ND and dS^T voffsets; the mask's lane part; FINAL's image, readback and store lane parts (the store voffsets
    # without the item's first row)
    rs_k = Hq * fx.Int32(D_QK // 8)  # row strides of dK / dV in 16-B units (MHA: heads_kv = heads_q)
    rs_v = Hq * fx.Int32(D_V // 8)
    lane_r = half * fx.Int32(8) + (lane & fx.Int32(7))
    lane_c2 = ((lane >> fx.Int32(3)) & fx.Int32(1)) * fx.Int32(16)
    lr = lane >> fx.Int32(3)
    lc = lane & fx.Int32(7)
    wb = lds0 + fx.Int32(STG0) + wv * fx.Int32(EPI_W)
    # the wave's image area
    fb = (wv == fx.Int32(NW - 1)).select(lds0 + fx.Int32(IMG3), wb + fx.Int32(NW * EPI_W))
    vops = [
        wb + fx.Int32(EPI_DK) + row * fx.Int32(XK) + half * fx.Int32(16),
        wb + row * fx.Int32(XV) + half * fx.Int32(16),
        lds0 + fx.Int32(QOFF) + row * fx.Int32(XK) + half * fx.Int32(16),
        lds0 + row * fx.Int32(XV) + half * fx.Int32(16),
        lds0 + lane_r * fx.Int32(XV) + lane_c2,
        lds0 + fx.Int32(QOFF) + lane_r * fx.Int32(XK) + lane_c2,
        half * fx.Int32(32),
        lane * fx.Int32(16),
        row - half * fx.Int32(8),
        fb + row * fx.Int32(XV) + half * fx.Int32(16),
        fb + fx.Int32(EPI_DK) + row * fx.Int32(XK) + half * fx.Int32(16),
        fb + lr * fx.Int32(XV) + lc * fx.Int32(16),
        fb + fx.Int32(EPI_DK) + lr * fx.Int32(XK) + lc * fx.Int32(16),
        (lr * rs_v + lc) * fx.Int32(16),
        (lr * rs_k + lc) * fx.Int32(16),
    ]

    # ---- the pool VGPR (pinned to v15): the per-workgroup scalars, one per lane (sched.POOL); the generated code
    # owns every SGPR LLVM allocates, so nothing reaches it in one
    def addr(t):
        a = fx.Int64(fx.ptrtoint(fx.get_iter(t)))
        return [fx.Int32(a), fx.Int32(a >> fx.Int64(32))]

    def rsrc(t):
        """A buffer resource over nl_bytes of t: [base | num_records << 57, num_records >> 7] (gfx1250)."""
        lo, hi = addr(t)
        return [lo, hi | (nl_bytes << fx.Int32(25)), nl_bytes.shrui(fx.Int32(7)), fx.Int32(0)]

    # this wave's rows TDM_ROWS wv .. of a ring tile: half hh = rows // 16, rows % 16 inside it
    wr0 = wv * fx.Int32(TDM_ROWS)
    ring_w = (wr0 >> fx.Int32(4)) * fx.Int32(HH_OFF)
    ring_r = wr0 & fx.Int32(15)
    # the wave's rows of its workgroup's item list: workgroup g, slot s, wave w at ((g * nslot + s) * NW + w) rows
    tp = fx.Int64(fx.ptrtoint(fx.get_iter(TBL))) + fx.Int64(
        ((fx.Int32(fx.block_idx.x) * nslot) * fx.Int32(NW) + wv) * fx.Int32(4 * sched.ROW_DW)
    )
    scalars = {
        "rnl": rsrc(NL),
        "rnd": rsrc(ND),
        "st0": [Hq * fx.Int32(D_V)],
        "wv": [wv],
        "lw": [lds0 + ring_w + ring_r * fx.Int32(XV)],
        "lwq": [lds0 + fx.Int32(QOFF) + ring_w + ring_r * fx.Int32(XK)],
        "kvl": [wb],
        "tpl": [fx.Int32(tp)],
        "tph": [fx.Int32(tp >> fx.Int64(32))],
        "c1": [(scale * fx.Float32(LOG2E)).bitcast(fx.Int32)],
        "scl": [scale.bitcast(fx.Int32)],
        "do": addr(DO),
        "q": addr(Q),
        "k": addr(K),
        "v": addr(V),
        "ds": addr(DS),
        "dv": addr(DV_),
        "dk": addr(DK),
    }
    pool = fx.Int32(0)
    for name, vals in scalars.items():
        for i, x in enumerate(vals):
            pool = (lane == fx.Int32(sched.POOL[name] + i)).select(_uniform(x), pool)
    inline_region(sched, "MAIN", *vops, pool)


@flyc.jit
def launch_dkdv(
    Q,
    K,
    V,
    DO,
    NL,
    ND,
    DV_,
    DK,
    DS,
    TBL,
    scale: fx.Float32,
    Hq: fx.Int32,
    nl_bytes: fx.Int32,
    nslot: fx.Int32,
    nwg: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = the persistent workgroups (bwd_dkdv_items: at most one per CU), each running its list of TBL
    k_dkdv(Q, K, V, DO, NL, ND, DV_, DK, DS, TBL, scale, Hq, nl_bytes, nslot).launch(
        grid=(nwg, 1, 1), block=(32 * NW, 1, 1), stream=stream
    )


# amdgpu-prealloc-sgpr-spill-vgprs: the VGPR lanes LLVM spills SGPRs to are allocated with the kernel's own VGPRs
# (below bwd_dkdv_sched.BASE), not among the generated code's state.
launch_dkdv.compile_hints["llvm_options"] = {
    "structurizecfg-skip-uniform-regions": True,
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-enable-delay-alu": False,
    "greedy-regclass-priority-trumps-globalness": True,
    "amdgpu-prealloc-sgpr-spill-vgprs": True,
}
launch_dkdv.compile_hints["waves_per_eu"] = 1
