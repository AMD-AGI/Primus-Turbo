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

"""Generated code of k_dkdv's 48-row waves (bwd_dkdv_kernel, 192-row kv blocks, persistent workgroups): the
software-pipelined WMMA / LDS / softmax schedule of the HipKittens MLA backward generator with the FlyDSL backward's
numerics, as one inline asm statement (MAIN: the whole body of a workgroup) on a static register plan of VGPRs
v[BASE:1024] and of the SGPRs.
Standard library only (no FlyDSL): the kernel turns the region into one inline asm statement (region_asm).

  once     wave 0: the code itself as data, into L2 (region_warm); the per-workgroup scalars out of the pool
           VGPR's lanes, (c1, c1), (scale, scale), the ring's group-1 descriptors
  ITEM     the next item of the workgroup's list (bwd_dkdv_items): its table row into s0..s31, the tensor bases
           added; a row of zeros ends the list
  INIT     the ring's tile 0, the wave's K / V rows by TDM into its LDS staging area, the ring's tile 1, the
           accumulators cleared, the K / V B-operand fragments from the staging area, tensorcnt TW + barrier,
           tile 0's hh0 operands and NL / ND
  the tile loop (unrolled by the ring depth: tile k in stage k mod DEPTH; the item's first n_mask tiles in the
           masked bodies M<s>, then the unmasked U<s>, one conditional branch per unmasked tile): the TDM of tile
           k + DEPTH - 1 into the stage tile k - 1 used; S = Q K^T and dP = dO V^T from +0 (hh0's A operands loaded
           during the previous tile, hh1's streamed into the tuples hh0's WMMAs release); the softmax / dS in
           packed f32 straight into the bf16 dV / dK B operands; dV^T += dO^T P, dK^T += Q^T dS (A operands by
           ds_load_tr16_b128); the dS^T stores (masked tiles: unattended blocks dropped by an out-of-range
           voffset) and NL / ND of the next tile, behind a few dummy VALUs and one va_vdst wait in place of a
           full drain of the WMMA pipe (step_nodes); the ring barrier after the last LDS read of stage k
           (tensorcnt TW: tile k + 1 landed); the next tile's hh0 operands and scalars, the tile count
  exit     tensorcnt 0 (FINAL's images use the wave's image area, not the ring)
  FINAL    dV and dK to bf16 row-major images in the wave's image area, whole-line readback, global stores; then
           the next item
Numerics, per element (the FlyDSL backward's order): t = fma(S, c1, NL) (masked: -3e38), P = exp2(t), dS = P fma(dP,
scale, ND); bf16 P and dS feed dV / dK; NL = -log2(e) lse, ND = -scale delta (k_delta_bshd). Every accumulator takes
one WMMA per tile, in tile order, so dK, dV and the dS^T bytes are those of the LLVM-scheduled body.

Ring stage s (LDS bytes from the ring's base): query rows 0..15 (hh0) at s * STAGE_B, rows 16..31 (hh1) HH_OFF = 64 KiB
higher; per half the [16][XV] dO rows, then the [16][XK] Q rows (QOFF). LDS has two read ports that serve one 64 KiB
segment each per cycle: exactly one segment apart, the two halves of a stage use both ports wherever the workgroup's
LDS allocation starts. (It does not always start on a segment boundary: with the halves 32 KB apart around the
allocation's 64 KiB mark, every other workgroup of a CU read both halves through one port, and its tiles took 60 cycles
more.) hh0's lane bases are operands (lq, lv: readback rows, ltv, ltk: tr16 pieces); hh1's, HH_OFF higher and so beyond
the 16-bit DS immediate, are four VGPRs of the generated state, set once per workgroup (Layout.H1); the immediate holds
the stage offset. Each of the 4 waves moves 8 rows of a tile per TDM op (waves 0, 1: hh0). schedule_slot() orders a
tile around its in-order WMMA stream on the generator's cycle model; asm.emit_block adds s_set_vgpr_msb, the
expert-mode waits and the hazard v_nops; the operand VGPRs are pinned to v0..v15 (lane bases and the pool of
per-workgroup scalars)."""

import re

from . import asm as A
from .asm import add_order, vr

NKV = 3  # kv sub-tiles of 16 rows per wave
NDT_QK, NDT_V = 6, 4  # WMMA k-steps of S = Q K^T (D_QK 192) and dP = dO V^T (D_V 128)
NDO_V, NDO_QK = 8, 12  # 16-wide output tiles of dV^T (D_V) and dK^T (D_QK)
XK, XV = 400, 272  # LDS row strides (bytes) of Q / K and of dO / V rows (the TDM pads 16 B per row)
STAGE_B = 16 * (XV + XK)  # one half of a ring stage (16 query rows)
QOFF = 16 * XV  # the Q rows inside a half stage
DEPTH = 3  # ring stages
HH_OFF = 1 << 16  # the hh1 halves of the stages: one LDS segment above the hh0 halves
TW = 3 * (DEPTH - 2)  # tensorcnt that retires all but the newest DEPTH - 2 tiles' 3 TDM ops
# hh1's lane bases live in their own VGPRs (Layout.H1) when hh1's offsets do not fit the 16-bit DS immediate
HH_REG = HH_OFF + (DEPTH - 1) * STAGE_B + QOFF + 16 * XK >= 1 << 16
assert DEPTH * STAGE_B <= HH_OFF, "the hh0 halves end below the hh1 halves"
assert (DEPTH - 1) * STAGE_B + QOFF + 16 * XK < 1 << 16, "a stage offset must fit the DS immediate"
_H1 = {"lq": 0, "lv": 1, "ltv": 2, "ltk": 3}
DS_ST_B = (
    512  # dS^T stores (dslayout v2): dwords 4..7 of a dS operand (queries 16..31) go 512 B after dwords 0..3
)
NEG = "0xff61b1e6"  # fp32 -3e38: a masked logit, exp2 -> +0
FINAL_LAG = 8  # FINAL: a global store trails its readback load by this many loads
# k-steps of S whose hh0 Q operands a tile loads itself, at its start, not the previous tile at its end: their
# tuples (O[8], O[9]) are the last of dK's own Q^T operands to be released, and their loads would trail the tile's
# last WMMA
LATE_DT = (4, 5)
BASE = 16  # the generated code's first VGPR; the kernel's own code stays below it
SPILL_OFF = (
    0x40000000  # added to a dS^T store's voffset: past every descriptor's range (the store is dropped)
)
# the three TDM ops of a ring stage: 0 dO, 1 Q cols [0, 128), 2 Q cols [128, 192); word 0 of their group-1 descriptors
# (2-B elements, LDS row padding), tile columns, tile rows per wave, LDS / global column offsets
TDM_W0 = (0x07510000, 0x47510000, 0x87110000)
TDM_WIDTH = (128, 128, 64)
TDM_ROWS = 8
KV_ROWS = 16 * NKV  # the wave's K / V rows (INIT's TDM ops)
TDM_LDS = (0, 0, 256)  # from the wave's dO (op 0) / Q (ops 1, 2) row base
# the next tile's scalars redefine SGPRs that in-flight VMEM / TDM ops read (an XCNT wait): keep them this many
# cycles behind those ops so the wait finds them done
SCAL_XCNT_GAP = 160
# SGPRs of MAIN's loop (clobbered): TDM descriptors per ring op (group 1: 8 SGPRs, group 0: 4), the tile counter k
# and the per-tile scalars; s96..s99 are temporaries (INIT and FINAL use s92..s103 too)
SG_G1 = (48, 56, 64)
SG_G0 = (72, 76, 80)
SG_K, SG_QT, SG_T1, SG_T2, SG_NLS = "s84", "s85", "s86", "s87", "s88"
SG_SO = ("s89", "s90", "s91")
SG_THR = "s92"
SG_PT = ("s93", "s94", "s95")
# The generated code owns every SGPR that LLVM does not reserve (s32, s105; the kernel passes it VGPRs only):
#   s0..s31   the item's table row (ROW: word i lands in s<i>; bwd_dkdv_items builds the rows), the tensor bases
#             added in place
#   s33..s47  per workgroup: the NL / ND resources, the wave's LDS addresses, its table pointer, bytes per q pair
#   s48..s95  the tile loop (above). Before an item's loop s84..s87 hold its K / V row addresses; after it s88..s95
#             hold FINAL's copies of the dV / dK resources (SG_FV, SG_FK): the next item's first part rewrites none
#             of them, so it does not wait for FINAL's last stores
#   s96..s103 temporaries; group 1 of an item's K / V ops
#   s103, s104  temporaries of ring tile 1's descriptors (SG_TA, SG_TB); s104 then the soffset of tile 0's
#             NL / ND loads
ROW_DW = 32
ROW = {
    "n": 0, "nmask": 1, "qps": 2, "nl0": 3, "thr0": 4, "qpf0": 5, "qpf1": 6, "qpf2": 7, "rc0": 8, "rc1": 9,
    "rc2": 10, "ext": 11, "rds": 12, "rdv": 16, "rdk": 20, "ga0l": 24, "ga0h": 25, "ga1l": 26, "ga1h": 27,
    "gvo": 28, "gko": 29, "rsv": 30, "rsk": 31,
}  # fmt: skip
SG = {name: (f"s[{i}:{i + 3}]" if name in ("rds", "rdv", "rdk") else f"s{i}") for name, i in ROW.items()}
SG.update(
    rnl="s[40:43]",
    rnd="s[36:39]",
    lw="s33",
    lwq="s34",
    kvl="s35",
    tpl="s44",
    tph="s45",
    gs0="s46",
    gs1="s47",
    kgl="s84",
    kgh="s85",
    vgl="s86",
    vgh="s87",
    st0="s96",
    st1="s97",
    c1="s98",
    scl="s99",
)
# lanes of the pool VGPR (the kernel's last operand): the per-workgroup scalars
POOL = {
    "rnl": 0, "rnd": 4, "st0": 8, "wv": 9, "lw": 10, "lwq": 11, "kvl": 12, "tpl": 13, "tph": 14, "c1": 15, "scl": 16,
    "do": 17, "q": 19, "k": 21, "v": 23, "ds": 25, "dv": 27, "dk": 29,
}  # fmt: skip
WARM_PIECE = 2048  # code bytes per warm-up load: 32 lanes, one 64-byte line each
KERNEL = "k_mla_bwd_dkdv"  # the kernel's name, so its function's symbol: where its code starts (region_warm)
SG_FV, SG_FK = (
    "s[88:91]",
    "s[92:95]",
)  # FINAL's copies of the dV / dK resources (the loop's SGPRs are dead then)
SG_TA, SG_TB = "s103", "s104"
CLOBBER_S = tuple(r for r in range(0, 105) if r != 32)  # LLVM reserves s32 and s105


def offset(off):
    """The offset modifier of a memory instruction (none for 0)."""
    return f" offset:{off}" if off else ""


def wmma(d, a, b, c):
    """D = A B + C (v_wmma_f32_16x16x32_bf16) on 8-VGPR tuples; c None: from +0."""
    return f"v_wmma_f32_16x16x32_bf16 {vr(d, 8)}, {vr(a, 8)}, {vr(b, 8)}, {'0' if c is None else vr(c, 8)}"


class Layout:
    """VGPR map (true register numbers) of the generated state, all at or above `base`. An 8-VGPR tuple never crosses
    a 256-VGPR block (s_set_vgpr_msb selects one block per operand field). Over one tile:
      O[0..9]   S / dP A operands of one hh half: Q dt 0..3 at O[2 dt], Q dt 4, 5 at O[8], O[9], dO dt 0..3 at
                O[2 dt + 1]; hh0's are loaded during the previous tile, hh1's stream in as hh0's WMMAs release them;
                after hh1's S / dP: the Q^T operands 0..9 of dK; after dK: the next tile's hh0 operands
      D[0..7]   dO^T operands of dV; after dV: the Q^T operands 10, 11 (D[0], D[1])
      R0, R1    raw S / dP of hh0 / hh1: R[2 kh] = S, R[2 kh + 1] = dP of kv sub-tile kh
      P, DS     bf16 P / dS B operands of kv sub-tile kh (dwords 0..3: hh0, 4..7: hh1), written by the softmax in place
                of hh0's raw S / dP: P[kh] = R0[2 kh], DS[kh] = R0[2 kh + 1]
      PRE[e]    NL / ND quads, e = 4 hh + {0, 1}: NL, 4 hh + {2, 3}: ND
      acc       dV (kh, d) at acc[8 kh + d], dK (kh, d) at acc[8 NKV + 12 kh + d]
      kf, vf    K / V fragments [kh][dt]
      cc1, ccs  (c1, c1) and (scale, scale)
      T         mask scratch (2) and a masked tile's dS^T voffsets (NKV)
      TD        the destination of the softmax's dummy VALUs
      H1        hh1's lane bases lq, lv, ltv, ltk (HH_REG)"""

    def __init__(self, base):
        used = [[] for _ in range(4)]  # (first, count) taken in each 256-VGPR block

        def take(n, align):
            """First fit: the lowest block with room after everything already taken there."""
            for blk in range(4):
                r = max([max(base + base % 2, blk * 256)] + [a + m for a, m in used[blk]])
                r += (-r) % align
                if r + n <= (blk + 1) * 256:
                    used[blk].append((r, n))
                    return r
            raise ValueError(f"layout does not fit above base v{base}")

        # the big persistent arrays first, so that they pack the blocks
        self.acc = [take(8, 8) for _ in range(NKV * (NDO_V + NDO_QK))]
        self.kf = [[take(8, 8) for _ in range(NDT_QK)] for _ in range(NKV)]
        self.vf = [[take(8, 8) for _ in range(NDT_V)] for _ in range(NKV)]
        self.O = [take(8, 8) for _ in range(10)]
        self.R0 = [take(8, 8) for _ in range(2 * NKV)]
        self.R1 = [take(8, 8) for _ in range(2 * NKV)]
        self.D = [take(8, 8) for _ in range(NDO_V)]
        self.P = [self.R0[2 * kh] for kh in range(NKV)]
        self.DS = [self.R0[2 * kh + 1] for kh in range(NKV)]
        pre = [take(8, 8) for _ in range(4)]  # hh0 NL, hh0 ND, hh1 NL, hh1 ND
        self.PRE = [pre[e // 2] + 4 * (e % 2) for e in range(8)]
        self.cc1, self.ccs = take(2, 2), take(2, 2)
        self.T = [take(1, 1) for _ in range(2 + NKV)]
        self.TD = take(1, 1)
        self.H1 = [take(1, 1) for _ in range(4)] if HH_REG else None
        self.top = max(a + m for blk in used for a, m in blk)

    def sdp_a(self, isq, dt):
        """S / dP A operand of dt: Q (isq) or dO."""
        if isq:
            return self.O[2 * dt] if dt < 4 else self.O[8 + dt - 4]
        return self.O[2 * dt + 1]

    def raw(self, hh, kh, j):
        """Raw S (j 0) / dP (j 1) of (hh, kh)."""
        return (self.R0 if hh == 0 else self.R1)[2 * kh + j]

    def dv_a(self, d):
        """dO^T A operand d of dV."""
        return self.D[d]

    def dk_a(self, d):
        """Q^T A operand d of dK."""
        return self.O[d] if d < 10 else self.D[d - 10]

    def dv(self, kh, d):
        return self.acc[kh * NDO_V + d]

    def dk(self, kh, d):
        return self.acc[NKV * NDO_V + kh * NDO_QK + d]


# --------------------------------------------------------------------------------------------------------------
# dependence graph and slot scheduler (the generator's)
# --------------------------------------------------------------------------------------------------------------
# result latencies (cycles); DS_LD only for the critical path: the scheduler models the LDS queue instead. WMMA and
# TRANS as the ATT (the hardware's thread trace) shows them (a consumer issued 20 / 8 cycles behind stalled 2 to 4
# cycles at issue)
LAT = {
    "WMMA": 24,
    "VALU": 4,
    "TRANS": 12,
    "DS_LD": 90,
    "DS_ST": 4,
    "VM_LD": 800,
    "VM_ST": 4,
    "TDM": 4,
    "SALU": 2,
    "BAR_S": 1,
    "BAR_W": 1,
    "WAIT_CNT": 1,
    "OTHER": 1,
}
WMMA_CHAIN = 16  # WMMA -> WMMA on its accumulator
# a memory op that overwrites a register an in-flight WMMA reads waits for that WMMA (va_vdst, bounded by the WMMAs
# issued since): keep it WAR_XDL cycles and at least MIN_XDL_AFTER WMMAs behind the reader, so that the wait is free.
# The pipe holds 3 to 4 WMMAs and the softmax's VALUs count against the same bound (ATT: va_vdst(3) held the issue 9
# to 13 cycles, va_vdst(3) .. (7) under the softmax 5 to 21, (8) and more about 3); the Q^T pieces 0..9, which go out
# under hh1's softmax, keep QT_XDL_AFTER behind
WAR_XDL = 24
MIN_XDL_AFTER = 8
QT_XDL_AFTER = 10
# hh1's S / dP A operands of dt stream into the tuples hh0's dt releases, the late dt after hh1's first WMMAs: the
# first position in hh1's WMMA stream at which the operands of dt have landed (hh1_order)
HH1_AVAIL = (0, 2, 6, 12, 15, 18)
SOFT_TAIL = 8  # dummy VALUs behind the softmax, before its one va_vdst(SOFT_TAIL) wait (step_nodes)
DS_MAX = 56  # LDS loads in flight
# issue model (4 waves per WGP): a WMMA occupies the XDL pipe 8 cycles; inside its window a VALU / TRANS issues only
# at the offsets VALU_SLOTS, a memory op only at MEM_SLOTS, anything else at any offset but 0; one LDS op per LDS_RATE
# cycles, completing in order LDS_LAT cycles after issue; s_barrier_wait at least BAR_GAP WMMAs after the signal (the
# other waves' skew)
VALU_SLOTS, MEM_SLOTS = (2, 3, 6, 7), (1, 4, 5)
LDS_LAT, LDS_RATE = 70, 8
BAR_GAP = 12
MEM = ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM")


class Node:
    """One instruction of a tile: kind, VGPR / SGPR defs and uses, [(pred, latency)] edges, the WMMAs whose
    operands it overwrites (war_xdl) and how many WMMAs it keeps behind them (xdl_after; None: MIN_XDL_AFTER)."""

    __slots__ = ("text", "kind", "vdefs", "vuses", "sdefs", "suses", "preds", "war_xdl", "xdl_after")


def mk(text):
    """A node of one instruction; defs / uses from the asm model's parser (the model the emitter and the checks use)."""
    it = A.make_item("\t" + text)
    n = Node()
    n.text, n.kind = text, it.kind
    n.vdefs, n.vuses = set(it.vdefs), set(it.vuses)
    n.sdefs, n.suses = set(it.sdefs), set(it.suses)
    n.preds, n.war_xdl, n.xdl_after = [], set(), None
    return n


def build_dag(nodes):
    """Edges in list order: RAW (the producer's latency), WAR / WAW (order; WAR_XDL for a non-VALU writer behind a WMMA
    that reads the register); n.preds = [(pred, latency)]."""
    last_w, readers = {}, {}
    for i, n in enumerate(nodes):
        uses = [("v", x) for x in n.vuses] + [("s", x) for x in n.suses]
        defs = [("v", x) for x in n.vdefs] + [("s", x) for x in n.sdefs]
        preds = {}
        for r in uses:
            if r in last_w:
                p = last_w[r]
                lat = WMMA_CHAIN if (nodes[p].kind == "WMMA" and n.kind == "WMMA") else LAT[nodes[p].kind]
                preds[p] = max(preds.get(p, 0), lat)
        for r in defs:
            if r in last_w:
                preds[last_w[r]] = max(preds.get(last_w[r], 0), 1)
            for q in readers.get(r, ()):
                if q != i:
                    war_xdl = nodes[q].kind == "WMMA" and n.kind not in ("WMMA", "VALU", "TRANS")
                    preds[q] = max(preds.get(q, 0), WAR_XDL if war_xdl else 1)
                    if war_xdl:
                        n.war_xdl.add(q)
        n.preds = sorted(preds.items())
        for r in uses:
            readers.setdefault(r, set()).add(i)
        for r in defs:
            last_w[r] = i
            readers[r] = set()


def schedule_slot(nodes, wseq):
    """The nodes in issue order: a cycle-stepped list scheduler around the in-order WMMA stream wseq, one instruction
    per cycle. A WMMA issues once the previous one left the XDL pipe (8 cycles); the slots inside its window
    and the LDS queue follow the issue model above; a TRANS blocks VALU / TRANS the next cycle; s_barrier_signal (its
    s_wait_dscnt 0) waits until every LDS op issued so far completed; at most DS_MAX LDS loads are in flight (a consumer
    of a load retires it and every older one). Result latencies LAT, WMMA_CHAIN for a WMMA chain. Ties: barrier and
    wait nodes first, then the earliest deadline (the first WMMA that needs the node, transitively), the longest
    critical path, program order."""
    n = len(nodes)
    succ, dl, cp = A.dag_ranks(nodes, wseq)
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    out, wmma_at = [], {}
    t, last_w, wk, trans_at, ds_next, ds_last_done = 0, -100, 0, -100, 0, 0
    ds_issued, ds_retired, ds_seq = 0, -1, {}
    bs_wk = 10**6  # WMMAs issued before the barrier signal

    def xdl_ok(i):
        rs = nodes[i].war_xdl  # (all issued: they are predecessors of node i)
        if not rs:
            return True
        need = MIN_XDL_AFTER if nodes[i].xdl_after is None else nodes[i].xdl_after
        return wk - 1 - max(wmma_at[r] for r in rs) >= need or wk >= len(wseq)

    def avail(i):
        k = nodes[i].kind
        if earliest[i] > t:
            return False
        o = t - last_w
        inwin = 0 <= o < 8
        if k in ("VALU", "TRANS"):
            return (not inwin or o in VALU_SLOTS) and t != trans_at + 1
        if k in MEM:
            if not xdl_ok(i):
                return False
            if k in ("DS_LD", "DS_ST") and (t < ds_next or ds_issued - 1 - ds_retired >= DS_MAX):
                return False
            return not inwin or o in MEM_SLOTS
        if k == "BAR_S":
            return o != 0 and t >= ds_last_done
        if k == "BAR_W":
            return o != 0 and (wk >= bs_wk + BAR_GAP or wk >= len(wseq))
        return o != 0

    while len(out) < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        if nw is not None and nw in ready and earliest[nw] <= t and t >= last_w + 8:
            pick = nw
        else:
            cand = [i for i in ready if nodes[i].kind != "WMMA" and avail(i)]
            if cand:
                if 0 <= t - last_w < 8 and t - last_w in VALU_SLOTS:
                    cand = [i for i in cand if nodes[i].kind in ("VALU", "TRANS")] or cand
                pick = max(
                    cand, key=lambda i: (nodes[i].kind in ("BAR_S", "BAR_W", "WAIT_CNT"), -dl[i], cp[i], -i)
                )
        if pick is None:
            t += 1
            if t > 10**6:
                raise RuntimeError("slot scheduler deadlock")
            continue
        nd = nodes[pick]
        ready.discard(pick)
        k = nd.kind
        done_at = None
        if k in ("DS_LD", "DS_ST"):
            ds_next = max(t, ds_next) + LDS_RATE
            done_at = max(t + LDS_LAT, ds_last_done + LDS_RATE)
            ds_last_done = done_at
            ds_seq[pick] = ds_issued
            ds_issued += 1
        for p, _ in nd.preds:
            if nodes[p].kind == "DS_LD" and p in ds_seq and nd.vuses & nodes[p].vdefs:
                ds_retired = max(ds_retired, ds_seq[p])
        if k == "BAR_S":
            ds_retired = ds_issued - 1
            bs_wk = wk
        out.append(nd)
        if k == "WMMA":
            last_w = t
            wmma_at[pick] = wk
            wk += 1
        if k == "TRANS":
            trans_at = t
        for s_, lat in succ[pick]:
            if k == "DS_LD" and nodes[s_].vuses & nd.vdefs:
                lat = done_at - t
            elif k == "WMMA":
                lat = WMMA_CHAIN if nodes[s_].kind == "WMMA" and (nodes[s_].vuses & nd.vdefs) else max(lat, 1)
            else:
                lat = max(lat, LAT[k]) if (nodes[s_].vuses & nd.vdefs or nodes[s_].suses & nd.sdefs) else lat
            earliest[s_] = max(earliest[s_], t + lat)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
        t += 1
    return out


# --------------------------------------------------------------------------------------------------------------
# INIT, PRE
# --------------------------------------------------------------------------------------------------------------
def hbase(L, ops, name, hh):
    """(address register text, immediate) of the lane base ``name`` (lq, lv, ltv, ltk) for query half hh."""
    if hh and HH_REG:
        return vr(L.H1[_H1[name]]), 0
    return ops[name], hh * HH_OFF


def sreg(op, i):
    """SGPR i of a tuple operand s[a:b]."""
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", op)
    if not m or int(m.group(1)) + i > int(m.group(2)):
        raise ValueError(f"{op}: no element {i}")
    return f"s{int(m.group(1)) + i}"


def tdm_words(o, rows, width=None):
    """Words 0..4 of the group-1 descriptor of TDM op o (2-B elements, LDS row padding to TDM_XROW[o]): tensor extents
    (inner 0x7FFFFFFF: never clamps, outer `rows`), tile TDM_WIDTH[o] (or `width`) x `rows`. Words 5, 6 hold the row
    stride (elements), word 7 is 0."""
    w = TDM_WIDTH[o] if width is None else width
    return [hex(TDM_W0[o]), "0xffff0000", hex((rows << 16) | 0x7FFF), hex(w << 16), str(rows)]


def ring_g1(ops):
    """The loop-invariant group-1 descriptors of the ring's three TDM ops (SG_G1)."""
    out = []
    for o in range(3):
        words = tdm_words(o, TDM_ROWS) + [ops["st0" if o == 0 else "st1"], "0", "0"]
        out += [f"s_mov_b32 s{SG_G1[o] + i}, {w}" for i, w in enumerate(words)]
    return out


def ring_g0(ops, tile_off, stage, live=None, tmp="s96"):
    """Group 0 of the ring's three TDM ops of the tile `tile_off` tiles after the first (an SGPR, or None: the first)
    into ring stage `stage` (SG_G0); live: an SGPR that is 0 for the NULL descriptor (else count 1)."""
    out = []
    for o in range(3):
        g0 = SG_G0[o]
        ga = (ops["ga0l"], ops["ga0h"]) if o == 0 else (ops["ga1l"], ops["ga1h"])
        lw = ops["lw" if o == 0 else "lwq"]
        out.append(f"s_mov_b32 s{g0}, {live}" if live else f"s_mov_b32 s{g0}, 1")
        if tile_off is None:
            out += [f"s_mov_b32 s{g0 + 2}, {ga[0]}", f"s_mov_b32 s{g0 + 3}, {ga[1]}"]
        else:
            out += [
                f"s_mul_i32 {tmp}, {tile_off}, {ops['gs0' if o == 0 else 'gs1']}",
                f"s_add_co_u32 s{g0 + 2}, {ga[0]}, {tmp}",
                f"s_add_co_ci_u32 s{g0 + 3}, {ga[1]}, 0",
            ]
        if o == 2:
            out += [f"s_add_co_u32 s{g0 + 2}, s{g0 + 2}, 256", f"s_add_co_ci_u32 s{g0 + 3}, s{g0 + 3}, 0"]
        out += [
            f"s_bitset1_b32 s{g0 + 3}, 31",
            f"s_add_co_i32 s{g0 + 1}, {lw}, {stage * STAGE_B + TDM_LDS[o]}",
        ]
    return out


def region_wg_init(L, ops):
    """Once per workgroup: the per-workgroup scalars from the pool VGPR's lanes into their SGPRs (SG), (c1, c1) and
    (scale, scale), hh1's lane bases (HH_REG; written before anything reads them: one wait here, none in the loop),
    the ring's group-1 descriptors."""
    pool = ops["pool"]

    def lane(dst, name, i=0):
        return f"v_readlane_b32 {dst}, {pool}, {POOL[name] + i}"

    out = [lane(sreg(ops["rnl"], i), "rnl", i) for i in range(4)]
    out += [lane(sreg(ops["rnd"], i), "rnd", i) for i in range(4)]
    out += [lane(ops[x], x) for x in ("st0", "lw", "lwq", "kvl", "tpl", "tph", "c1", "scl")]
    # st1 = 1.5 st0 (row strides, elements: heads * 128 / heads * 192); bytes per q pair = 64 * stride
    out += [
        f"s_lshr_b32 {ops['st1']}, {ops['st0']}, 1",
        f"s_add_co_i32 {ops['st1']}, {ops['st1']}, {ops['st0']}",
    ]
    out += [f"s_lshl_b32 {ops['gs0']}, {ops['st0']}, 6", f"s_lshl_b32 {ops['gs1']}, {ops['st1']}, 6"]
    out += [f"v_mov_b32 {vr(L.cc1)}, {ops['c1']}", f"v_mov_b32 {vr(L.cc1 + 1)}, {ops['c1']}"]
    out += [f"v_mov_b32 {vr(L.ccs)}, {ops['scl']}", f"v_mov_b32 {vr(L.ccs + 1)}, {ops['scl']}"]
    if HH_REG:
        out += [f"v_add_nc_u32 {vr(L.H1[i])}, {HH_OFF}, {ops[name]}" for name, i in _H1.items()]
        out.append("s_wait_alu depctr_va_vdst(0)")
    out += ring_g1(ops)
    return out


_FOUR_B = (
    "v_exp_f32",
    "v_mov_b64",
    "v_mov_b32",
    "v_cmp_",
    "v_nop",
    "v_add_nc_u32",
    "v_subrev_nc_u32",
    "v_lshlrev_b32",
    "v_cndmask_b32",
)
_FOUR_B_S = ("s_wait", "s_set_vgpr_msb", "s_barrier", "s_cbranch", "s_branch")


def code_bytes(lines):
    """An upper bound of the encoded size of the instruction lines (a few hundred bytes over at most: 4 bytes, 8
    for VOP3 / DS / SMEM encodings or a 32-bit literal, 12 for buffer, global and TDM instructions)."""

    def literal(text):  # a label expression, or a number that is not an inline constant
        for tok in re.split(r"[,\s]+", text.partition(" ")[2]):
            if ".L" in tok or (re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", tok) and not -16 <= int(tok, 0) <= 64):
                return True
        return False

    n = 0
    for line in lines:
        text = line.partition(";")[0].strip()
        if not text or text.endswith(":"):
            continue
        mn = text.split()[0]
        if mn.startswith(("buffer_", "global_", "tensor_")):
            n += 12
        elif mn.startswith(("ds_", "s_load", "s_prefetch", "s_getpc", "v_readlane", "v_writelane")):
            n += 8
        elif mn.startswith(_FOUR_B_S) or (mn.startswith(("s_",) + _FOUR_B) and not literal(text)):
            n += 4
        else:
            n += 8
    return n


def region_warm(ops, lp, pieces):
    """The generated code into L2 through the data path, by wave 0 of every workgroup at its start: when another
    kernel ran since this one's last launch (every call in training) its code is neither in the instruction caches
    nor in L2, and each line's first fetch comes from HBM behind the kernel's own data stream (s_prefetch_inst and
    global_prefetch_b8 do nothing there). `pieces` loads of WARM_PIECE bytes of code each (32 lanes, 64 bytes
    apart), counted back from {lp}_END over the code's size bound: never past the code. Before it is the kernel's
    own code (the size bound and the rounding reach up to 2.3 KB into it), and a piece that would begin before the
    function's start begins there instead: every load reads the function's own bytes, however the compiler lays
    the kernel out. Their data goes to v[BASE + 4:], unused until the first item's accumulators are cleared
    (region_init waits for them there); each load's address offset is first written to its own destination."""
    pool = ops["pool"]
    if BASE + 4 + 4 * pieces > 256:
        raise ValueError("too many warm-up loads for the low VGPRs")
    regs = [BASE + 4 + 4 * k for k in range(pieces)]
    out = [
        f"v_readlane_b32 s100, {pool}, {POOL['wv']}",
        "s_cmp_lg_u32 s100, 0",
        f"s_cbranch_scc1 {lp}_WARM",
        "s_getpc_b64 s[100:101]",
        f"{lp}_PC:",
        f"s_mov_b32 s102, {KERNEL}-{lp}_PC",  # s[102:103]: the function's start (a signed 64-bit add)
        "s_ashr_i32 s103, s102, 31",
        "s_add_nc_u64 s[102:103], s[100:101], s[102:103]",
        f"v_lshlrev_b32 {vr(BASE)}, 2, {ops['vds']}",  # the lane's line: 64 * lane (vds = 16 * lane)
    ]
    # per piece: its first byte from the function's start, not before it, + the lane's line
    for k, r in enumerate(regs):
        out += [
            f"s_mov_b32 s96, {lp}_END-{KERNEL}-{WARM_PIECE * (k + 1)}",
            "s_max_i32 s96, s96, 0",
            f"v_add_nc_u32 {vr(r)}, s96, {vr(BASE)}",
        ]
    out.append("s_wait_alu depctr_va_vdst(0)")  # the offsets are written: one wait for all the loads
    out += [f"global_load_b128 {vr(r, 4)}, {vr(r)}, s[102:103]" for r in regs]
    return out


def _add_base(pool, lo, hi, name, ta, tb, zero_hi=False):
    """(hi, lo) += the 64-bit tensor base in the pool lanes of `name` (zero_hi: hi = the base's + the carry)."""
    return [
        f"v_readlane_b32 {ta}, {pool}, {POOL[name]}",
        f"v_readlane_b32 {tb}, {pool}, {POOL[name] + 1}",
        f"s_add_co_u32 {lo}, {lo}, {ta}",
        f"s_add_co_ci_u32 {hi}, {tb}, 0" if zero_hi else f"s_add_co_ci_u32 {hi}, {hi}, {tb}",
    ]


def _table_ptr(ops):
    return f"s[{ops['tpl'][1:]}:{ops['tph'][1:]}]"


def region_item(ops, lp):
    """The next item of the workgroup's list: its table row into s0..s31 (ROW), the table pointer advanced; a row
    with n = 0 ends the list ({lp}_EXIT); the row after it prefetched (two 64-B lines: the next item's load finds it
    in the cache). Then the item's addresses: the wave's K / V rows (kgl / kgh, vgl / vgh: the batch offset of the
    dK / dV resources + the row offsets gko / gvo + the tensor base), and the tensor bases added in place to the
    dS^T / dV / dK resources and the Q / dO tile addresses."""
    pool, tp = ops["pool"], _table_ptr(ops)
    out = [
        f"s_load_b512 s[0:15], {tp}, 0x0",
        f"s_load_b512 s[16:31], {tp}, {hex(4 * ROW_DW // 2)}",
        "s_wait_kmcnt 0x0",
        f"s_add_co_u32 {ops['tpl']}, {ops['tpl']}, {4 * ROW_DW * 4}",
        f"s_add_co_ci_u32 {ops['tph']}, {ops['tph']}, 0",
        f"s_cmp_eq_u32 {ops['n']}, 0",
        f"s_cbranch_scc1 {lp}_EXIT",
        f"s_prefetch_data {tp}, 0, null, 1",
        f"s_add_co_i32 {ops['kgl']}, {sreg(ops['rdk'], 0)}, {ops['gko']}",
        f"s_add_co_i32 {ops['vgl']}, {sreg(ops['rdv'], 0)}, {ops['gvo']}",
    ]
    out += _add_base(pool, ops["kgl"], ops["kgh"], "k", "s96", "s97", zero_hi=True)
    out += _add_base(pool, ops["vgl"], ops["vgh"], "v", "s96", "s97", zero_hi=True)
    for rs, name in (("rds", "ds"), ("rdv", "dv"), ("rdk", "dk")):
        out += _add_base(pool, sreg(ops[rs], 0), sreg(ops[rs], 1), name, "s96", "s97")
    out += _add_base(pool, ops["ga0l"], ops["ga0h"], "do", "s96", "s97")
    out += _add_base(pool, ops["ga1l"], ops["ga1h"], "q", "s96", "s97")
    return out


def region_init(L, ops):
    """INIT of an item: the ring's tiles 0 and 1 (tile 1 the NULL descriptor if there is one tile; their six small
    ops first: behind the K / V ops they stall at issue); the wave's K / V rows (KV_ROWS of them from the head's row
    kv0, the outer extent `ext`: rows past skv are not read) by three TDM ops into its staging area (V cols
    [0, 128), K cols [0, 128) and [128, 192): [KV_ROWS][XV] of V at kvl, [KV_ROWS][XK] of K after it); tile 0's
    NL / ND loads; the accumulators cleared while all of that is in flight; then the K / V B-operand fragments from
    the staging area (lane bases lk / lvs + kh * 16 rows + dt * 64 + u * 32), and the barrier after tile 0 landed
    in every wave. (Issuing the K / V ops a whole FINAL earlier, at the previous item's loop exit, measured slower:
    FINAL then shares the LDS with them, and the wait it removes here is idle time, which costs about half its
    share.)"""
    out = ring_g0(ops, None, 0)
    out += [
        f"tensor_load_to_lds s[{SG_G0[o]}:{SG_G0[o] + 3}], s[{SG_G1[o]}:{SG_G1[o] + 7}]" for o in range(3)
    ]
    # ring tile 1 (tile 0 again, as the NULL descriptor, when there is one tile)
    out += [f"s_cmp_gt_i32 {ops['n']}, 1", f"s_cselect_b32 {SG_TB}, 1, 0"]
    out += ring_g0(ops, SG_TB, 1, SG_TB, tmp=SG_TA)
    out += [
        f"tensor_load_to_lds s[{SG_G0[o]}:{SG_G0[o] + 3}], s[{SG_G1[o]}:{SG_G1[o] + 7}]" for o in range(3)
    ]
    g0, g1 = SG_G0[0], 96  # one K / V op at a time
    for o in range(3):
        ga = (ops["vgl"], ops["vgh"]) if o == 0 else (ops["kgl"], ops["kgh"])
        words = tdm_words(o, KV_ROWS) + [ops["gs0" if o == 0 else "gs1"], "0", "0"]
        out += [f"s_mov_b32 s{g1 + i}, {x}" for i, x in enumerate(words) if i not in (2, 5)]
        out += [f"s_lshr_b32 s{g1 + 5}, {words[5]}, 6"]  # the row stride (elements) = bytes per q pair / 64
        out += [f"s_lshl_b32 s{g1 + 2}, {ops['ext']}, 16", f"s_or_b32 s{g1 + 2}, s{g1 + 2}, 0x7fff"]
        out += [
            f"s_cmp_gt_i32 {ops['ext']}, 0",
            f"s_cselect_b32 s{g0}, 1, 0",
            f"s_add_co_i32 s{g0 + 1}, {ops['kvl']}, {0 if o == 0 else KV_ROWS * XV + (256 if o == 2 else 0)}",
            f"s_add_co_u32 s{g0 + 2}, {ga[0]}, {256 if o == 2 else 0}",
            f"s_add_co_ci_u32 s{g0 + 3}, {ga[1]}, 0",
            f"s_bitset1_b32 s{g0 + 3}, 31",
            f"tensor_load_to_lds s[{g0}:{g0 + 3}], s[{g1}:{g1 + 7}]",
        ]
    # no load in flight on a VGPR: the previous item's last NL / ND loads (never read), the first item's warm-up
    # loads (region_warm). Then tile 0's NL / ND (soffset in SG_TB: the loop's SG_NLS is one of FINAL's resource
    # copies until the loop starts): they land under the K / V rows' wait, not at the tile's softmax
    out.append("s_wait_loadcnt 0x0")
    out += [f"s_lshl_b32 {SG_TB}, {ops['qps']}, 7", f"s_add_co_i32 {SG_TB}, {ops['nl0']}, {SG_TB}"]
    out += nl_loads(L, dict(ops, nls=SG_TB))
    for x in L.acc:
        for q in range(0, 8, 2):
            out.append(f"v_mov_b64 {vr(x + q, 2)}, 0")
    # the K / V ops (the newest three) landed, and the ring's tiles before them
    out.append("s_wait_tensorcnt 0x0")
    for kh in range(NKV):
        for frag, lane, x, ndt in ((L.kf, ops["lk"], XK, NDT_QK), (L.vf, ops["lvs"], XV, NDT_V)):
            for dt in range(ndt):
                for u in range(2):
                    off = offset(kh * 16 * x + dt * 64 + u * 32)
                    out.append(f"ds_load_b128 {vr(frag[kh][dt] + 4 * u, 4)}, {lane}{off}")
    # tile 0 landed in every wave
    out += [f"s_wait_tensorcnt {hex(TW)}", "s_barrier_signal -1", "s_barrier_wait -1"]
    return out


def sdp_loads(L, ops, hh, stage):
    """[((Q / D, dt, u), text)] of the 20 ds_load_b128 of hh half's S / dP A operands from ring stage `stage` (the
    half's lane bases lq / lv, the stage offset in the immediate)."""
    out = []
    (lq, hq), (lv, hv) = hbase(L, ops, "lq", hh), hbase(L, ops, "lv", hh)
    so = stage * STAGE_B
    for dt in range(NDT_QK):
        for u in range(2):
            off = offset(hq + so + dt * 64 + u * 32)
            out.append((("Q", dt, u), f"ds_load_b128 {vr(L.sdp_a(1, dt) + 4 * u, 4)}, {lq}{off}"))
        for u in range(2 if dt < NDT_V else 0):
            off = offset(hv + so + dt * 64 + u * 32)
            out.append((("D", dt, u), f"ds_load_b128 {vr(L.sdp_a(0, dt) + 4 * u, 4)}, {lv}{off}"))
    return out


def nl_loads(L, ops):
    """The 8 buffer_load_b128 of a tile's NL / ND quads (constant offsets, soffset ops['nls'])."""
    out = []
    for e in range(8):
        r = ops["rnl"] if (e % 4) // 2 == 0 else ops["rnd"]
        off = offset((e // 4) * 64 + (e % 2) * 16)
        out.append(f"buffer_load_b128 {vr(L.PRE[e], 4)}, {ops['nlv']}, {r}, {ops['nls']} offen{off}")
    return out


# --------------------------------------------------------------------------------------------------------------
# one tile
# --------------------------------------------------------------------------------------------------------------
def softmax_unit(L, hh, kh, add, mask=None):
    """Node indices of the softmax / dS of (hh, kh), the FlyDSL backward's arithmetic: 4 element pairs into dwords
    4 hh + j of P[kh] / DS[kh]. mask: the VGPR holding dl - thr - 16 (hh - kh); element i is masked iff mask > i."""
    nodes = []
    s_, d_ = L.raw(hh, kh, 0), L.raw(hh, kh, 1)
    for j in range(4):
        nl = L.PRE[hh * 4 + j // 2] + 2 * (j % 2)
        nd = L.PRE[hh * 4 + 2 + j // 2] + 2 * (j % 2)
        s2, d2 = s_ + 2 * j, d_ + 2 * j
        nodes.append(add(f"v_pk_fma_f32 {vr(s2, 2)}, {vr(s2, 2)}, {vr(L.cc1, 2)}, {vr(nl, 2)}"))
        if mask is not None:
            for e in range(2):
                nodes.append(add(f"v_cmp_ge_i32 vcc_lo, {2 * j + e}, {vr(mask)}"))
                nodes.append(add(f"v_cndmask_b32 {vr(s2 + e)}, {NEG}, {vr(s2 + e)}, vcc_lo"))
        nodes.append(add(f"v_exp_f32 {vr(s2)}, {vr(s2)}"))
        nodes.append(add(f"v_exp_f32 {vr(s2 + 1)}, {vr(s2 + 1)}"))
        nodes.append(add(f"v_pk_fma_f32 {vr(d2, 2)}, {vr(d2, 2)}, {vr(L.ccs, 2)}, {vr(nd, 2)}"))
        nodes.append(add(f"v_pk_mul_f32 {vr(d2, 2)}, {vr(s2, 2)}, {vr(d2, 2)}"))
        w = hh * 4 + j
        nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(L.P[kh] + w)}, {vr(s2)}, {vr(s2 + 1)}"))
        nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(L.DS[kh] + w)}, {vr(d2)}, {vr(d2 + 1)}"))
    return nodes


def hh1_order(avail=None):
    """hh1's S / dP WMMAs, kh-major: kh 0's chains finish first, so its softmax (and so P[0], the first dV operand)
    hides under the later kh's WMMAs. The S and dP chains of one kh interleave; a chain's next WMMA stands >= 2
    positions after its previous one (16 cycles: WMMA_CHAIN); lower kh first, then the chain with more WMMAs left.
    avail[dt]: the first position at which hh1's operands of dt may be used (a chain waiting for its operands lets the
    next kh's chains go first instead of stalling the stream). Items (kind 'S' / 'P', kh, dt)."""
    chains = []
    for kh in range(NKV):
        chains.append([("S", kh, dt) for dt in range(NDT_QK)])
        chains.append([("P", kh, dt) for dt in range(NDT_V)])
    out, last, nxt = [], [-10] * len(chains), [0] * len(chains)
    while len(out) < sum(len(c) for c in chains):
        p = len(out)
        live = [c for c in range(len(chains)) if nxt[c] < len(chains[c])]
        if avail is not None:
            live = [c for c in live if avail[chains[c][nxt[c]][2]] <= p] or live
        cand = [c for c in live if last[c] <= p - 2] or live
        c = min(cand, key=lambda c: (chains[c][0][1], -(len(chains[c]) - nxt[c]), c))
        out.append(chains[c][nxt[c]])
        nxt[c] += 1
        last[c] = p
    return out


def step_nodes(L, ops, masked, stage, scal):
    """Nodes of one tile in ring stage `stage` (program order = semantics) and the WMMA issue order. masked: a diagonal
    tile (causal mask on, the dS^T stores of unattended blocks dropped). scal: the next tile's scalar instructions,
    after every use of this tile's values."""
    nodes = []

    def add(text):
        nodes.append(mk(text))
        return len(nodes) - 1

    so = stage * STAGE_B
    w_sdp0, w_sdp1, w_dv, w_dk = [], [], [], []
    stage_reads = []  # every LDS read of stage k: before the ring barrier
    # TDM of tile k + DEPTH - 1 into the stage tile k - 1 used (every wave's last reads of it preceded the previous
    # barrier); it issues before the tensorcnt wait below, which then retires tile k + 1
    tdm = [add(f"tensor_load_to_lds {ops['g0' + c]}, {ops['g1' + c]}") for c in "abc"]
    # hh0's Q operands of the last k-steps (LATE_DT): their tuples were dK's last Q^T operands of the previous tile
    for key, text in sdp_loads(L, ops, 0, stage):
        if key[0] == "Q" and key[1] in LATE_DT:
            stage_reads.append(add(text))

    def wmma_sdp(hh, kind, kh, dt):
        if kind == "S":  # S = Q K^T from +0
            s_ = L.raw(hh, kh, 0)
            return add(wmma(s_, L.sdp_a(1, dt), L.kf[kh][dt], None if dt == 0 else s_))
        d_ = L.raw(hh, kh, 1)  # dP = dO V^T from +0
        return add(wmma(d_, L.sdp_a(0, dt), L.vf[kh][dt], None if dt == 0 else d_))

    # S / dP of hh0, dt outer (S of every kh, then dP: consecutive WMMAs share the A operand); hh1's operands of dt
    # stream into the tuples hh0's dt just released
    hh1 = dict(sdp_loads(L, ops, 1, stage))
    for dt in range(NDT_QK):
        for kind in ("S", "P") if dt < NDT_V else ("S",):
            for kh in range(NKV):
                w_sdp0.append(wmma_sdp(0, kind, kh, dt))
        for u in range(2):
            stage_reads.append(add(hh1[("Q", dt, u)]))
            if dt < NDT_V:
                stage_reads.append(add(hh1[("D", dt, u)]))
    # dO^T operands (their own tuples, free since the previous tile's dV): piece p from half p's rows
    for d in range(NDO_V):
        for p in range(2):
            base, ho = hbase(L, ops, "ltv", p)
            off = offset(ho + so + d * 32)
            stage_reads.append(add(f"ds_load_tr16_b128 {vr(L.dv_a(d) + 4 * p, 4)}, {base}{off}"))
    for kind, kh, dt in hh1_order(HH1_AVAIL):
        w_sdp1.append(wmma_sdp(1, kind, kh, dt))
    # softmax / dS, hh0 then hh1, kh by kh. One dummy VALU stands between hh0's units and hh1's first unit, and one
    # va_vdst(14) wait behind that unit: more than 14 core VALUs are then behind every instruction of hh0's softmax,
    # which so counts as written (a class completes in order), and hh0's NL / ND loads of the next tile (they
    # overwrite the constants it read) need no wait of their own: they go out right behind the softmax
    soft, units = [], {}
    for hh in range(2):
        if hh:
            split_d = add(f"v_mov_b32 {vr(L.TD)}, 0")
        for kh in range(NKV):
            mask = None
            if masked:
                thr = "s96" if hh == 0 else "s97"
                add(f"s_add_co_i32 {thr}, {ops['thr']}, {16 * (hh - kh)}")
                add(f"v_subrev_nc_u32 {vr(L.T[hh])}, {thr}, {ops['dl']}")
                mask = L.T[hh]
            units[hh, kh] = softmax_unit(L, hh, kh, add, mask)
            soft += units[hh, kh]
            if (hh, kh) == (1, 0):
                split_w = add("s_wait_alu depctr_va_vdst(14)")

    def q_t_loads(d):
        for p in range(2):
            base, ho = hbase(L, ops, "ltk", p)
            off = offset(ho + so + d * 32)
            i = add(f"ds_load_tr16_b128 {vr(L.dk_a(d) + 4 * p, 4)}, {base}{off}")
            if d < 10:  # under hh1's softmax
                nodes[i].xdl_after = QT_XDL_AFTER
            stage_reads.append(i)

    # Q^T operands 0..9 into O (hh1's operands are dead after its S / dP)
    for d in range(10):
        q_t_loads(d)
    # dV^T += dO^T P, kh outer (dV of kh 0 needs P[0] only)
    for kh in range(NKV):
        for d in range(NDO_V):
            w_dv.append(add(wmma(L.dv(kh, d), L.dv_a(d), L.P[kh], L.dv(kh, d))))
    # Q^T operands 10, 11 into D[0], D[1] (dead after dV)
    for d in range(10, NDO_QK):
        q_t_loads(d)
    # the dS^T stores of tile k; in a masked tile a block without an attended entry goes to an out-of-range voffset
    # (dropped: dslayout does not store it). The softmax tail first: (the masked tile's voffsets,) SOFT_TAIL dummy
    # VALUs behind the last softmax instruction and one va_vdst(SOFT_TAIL) wait: every VALU before the dummies then
    # counts as written, so the dS^T stores (they read the softmax's results) and hh1's NL / ND loads (they overwrite
    # its constants) need no va_vdst(0), which drained the WMMA pipe once per tile
    ex = []
    if masked:
        for kh in range(NKV):
            ex.append(add(f"v_add_nc_u32 {vr(L.T[2 + kh])}, {ops['pt' + str(kh)]}, {ops['vds']}"))
    tail_d = [add(f"v_mov_b32 {vr(L.TD)}, 0") for _ in range(SOFT_TAIL)]
    tail_w = add(f"s_wait_alu depctr_va_vdst({SOFT_TAIL})")
    for kh in range(NKV):
        vd = vr(L.T[2 + kh]) if masked else ops["vds"]
        for h in range(2):
            st = f"buffer_store_b128 {vr(L.DS[kh] + 4 * h, 4)}, {vd}, {ops['rds']}, {ops['ds' + str(kh)]} offen"
            ex.append(add(st + offset(h * DS_ST_B) + " th:TH_STORE_NT"))
    # NL / ND of tile k+1 (PRE is dead after the softmax)
    nl = [add(t) for t in nl_loads(L, ops)]
    # the ring barrier after the last LDS read of stage k (tensorcnt TW: tile k+1 landed), dK (d outer), then the next
    # tile's hh0 operands into O as dK releases Q^T 0..9 (after the barrier wait)
    tw = add(f"s_wait_tensorcnt {hex(TW)}")
    bs = add("s_barrier_signal -1")
    bw = add("s_barrier_wait -1")
    for d in range(NDO_QK):
        for kh in range(NKV):
            w_dk.append(add(wmma(L.dk(kh, d), L.dk_a(d), L.DS[kh], L.dk(kh, d))))
    nxt_reads = [
        add(t)
        for key, t in sdp_loads(L, ops, 0, (stage + 1) % DEPTH)
        if not (key[0] == "Q" and key[1] in LATE_DT)
    ]
    scal_nodes = [add(t) for t in scal]
    build_dag(nodes)
    for j in scal_nodes:
        for i in range(j):
            if nodes[i].kind in ("VM_LD", "VM_ST", "TDM") and nodes[i].suses & nodes[j].sdefs:
                add_order(nodes, i, j, SCAL_XCNT_GAP)
    for i in tdm + stage_reads:
        add_order(nodes, i, tw)
    add_order(nodes, tw, bs)
    add_order(nodes, bs, bw)
    # the dS^T stores (RAW on the bf16 converts) and the NL / ND loads (WAR on PRE) after the whole softmax: a memory
    # op with a hazard against core VALU waits for every in-flight VALU while WMMAs run (va_vdst 0), so one such wait
    # covers them all
    for i in ex + nl:
        if nodes[i].kind in ("VM_ST", "VM_LD"):
            for j in soft:
                add_order(nodes, j, i)
    # the tail: softmax and voffsets, the dummies, the wait (the dummies written), then the stores and hh1's NL / ND
    for j in soft + [i for i in ex if nodes[i].kind == "VALU"]:
        add_order(nodes, j, tail_d[0])
    add_order(nodes, tail_d[-1], tail_w, 8)
    for i in ex + nl[4:]:
        if nodes[i].kind in ("VM_ST", "VM_LD"):
            add_order(nodes, tail_w, i)
    # hh0's softmax, the dummy, hh1's first unit, the va_vdst(14) wait, hh0's NL / ND
    for (hh, _), unit in units.items():
        if hh == 0:
            for j in unit:
                add_order(nodes, j, split_d)
    for j in units[1, 0]:
        add_order(nodes, split_d, j)
        add_order(nodes, j, split_w)
    for i in nl[:4]:
        add_order(nodes, split_w, i)
    for i in nxt_reads:
        add_order(nodes, bw, i)
    return nodes, w_sdp0 + w_sdp1 + w_dv + w_dk


def region_step(L, ops, masked, stage, scal):
    """The instruction texts of one tile."""
    return [n.text for n in schedule_slot(*step_nodes(L, ops, masked, stage, scal))]


# --------------------------------------------------------------------------------------------------------------
# the tile loop: tile 0's operands, then the tile loop (unrolled by the ring depth) with its scalar work
# --------------------------------------------------------------------------------------------------------------
def loop_ops(ops):
    """The region operands plus the names the tile bodies use for the loop's SGPRs."""
    o = dict(ops)
    for c, g0, g1 in zip("abc", SG_G0, SG_G1):
        o["g0" + c], o["g1" + c] = f"s[{g0}:{g0 + 3}]", f"s[{g1}:{g1 + 7}]"
    for kh, (so, pt) in enumerate(zip(SG_SO, SG_PT)):
        o["ds" + str(kh)], o["pt" + str(kh)] = so, pt
    o["nls"], o["thr"] = SG_NLS, SG_THR
    return o


def loop_init(L, ops):
    """Tile 0's hh0 operands (stage 0; but the Q operands of LATE_DT, which every tile loads itself; its NL / ND are
    loaded in INIT), k = 0. Scalar
    operands: ga0l / ga0h, ga1l / ga1h (the global byte address of the wave's rows of tile 0: op 0 dO, op 1 Q; op 2 =
    op 1 + 256), gs0 / gs1 (bytes per tile), lw / lwq (the LDS byte address of the wave's dO / Q rows of stage 0),
    st0 / st1 (row
    strides, elements), n / nmask (tiles, masked tiles), qps (the first query pair), nl0 (the NL / ND soffset of query
    pair 0), thr0 (cshift - kv0), qpf<kh> (the first stored pair of the wave's kv16 rows), rc<kh> (dS^T row
    constants)."""
    out = [t for key, t in sdp_loads(L, ops, 0, 0) if not (key[0] == "Q" and key[1] in LATE_DT)]
    out.append(f"s_mov_b32 {SG_K}, 0")
    return out


def loop_scalars(ops, stage, kofs):
    """The scalars of tile j = k + kofs (k in SG_K), which runs in ring stage `stage`: its query pair qt, the NL / ND
    soffset of tile j + 1, the TDM descriptors of tile j + DEPTH - 1 (both clamped to tile N - 1; a TDM tile past the
    last is the NULL descriptor), the dS^T soffsets, the mask threshold and the stored-block flags of tile j."""
    qps, n, nm1 = ops["qps"], ops["n"], "s99"
    out = [f"s_add_co_i32 {SG_QT}, {qps}, {SG_K}"]
    if kofs:
        out.append(f"s_add_co_i32 {SG_QT}, {SG_QT}, {kofs}")
    out += [
        f"s_add_co_i32 {nm1}, {n}, -1",
        f"s_add_co_i32 {SG_T1}, {SG_K}, {1 + kofs}",
        f"s_min_i32 {SG_T1}, {SG_T1}, {nm1}",
        f"s_add_co_i32 {SG_T2}, {SG_K}, {DEPTH - 1 + kofs}",
        f"s_min_i32 {SG_T2}, {SG_T2}, {nm1}",
        f"s_add_co_i32 s98, {SG_T1}, {qps}",
        "s_lshl_b32 s98, s98, 7",
        f"s_add_co_i32 {SG_NLS}, {ops['nl0']}, s98",
        f"s_add_co_i32 s98, {SG_K}, {DEPTH - 1 + kofs}",
        f"s_cmp_ge_i32 s98, {n}",
    ]
    # group 0: count (0 past the last tile: the NULL descriptor), LDS address of the tile's stage, global address =
    # tile 0's + tile * bytes per tile
    so_tdm = ((stage + DEPTH - 1) % DEPTH) * STAGE_B
    out += [f"s_cselect_b32 s{SG_G0[o]}, 0, 1" for o in range(3)]  # before anything that writes scc
    for o in range(3):
        g0 = SG_G0[o]
        ga = (ops["ga0l"], ops["ga0h"]) if o == 0 else (ops["ga1l"], ops["ga1h"])
        gs = ops["gs0" if o == 0 else "gs1"]
        out += [
            f"s_mul_i32 s96, {SG_T2}, {gs}",
            "s_mov_b32 s97, 0",
            f"s_add_co_u32 s{g0 + 2}, {ga[0]}, s96",
            f"s_add_co_ci_u32 s{g0 + 3}, {ga[1]}, s97",
        ]
        if o == 2:
            out += [f"s_add_co_u32 s{g0 + 2}, s{g0 + 2}, 256", f"s_add_co_ci_u32 s{g0 + 3}, s{g0 + 3}, 0"]
        lw = ops["lw" if o == 0 else "lwq"]
        out += [f"s_bitset1_b32 s{g0 + 3}, 31", f"s_add_co_i32 s{g0 + 1}, {lw}, {so_tdm + TDM_LDS[o]}"]
    # dS^T soffset of kh: 1024 qt + rc[kh]
    out += [f"s_lshl_b32 s98, {SG_QT}, 10"]
    for kh in range(NKV):
        out.append(f"s_add_co_i32 {SG_SO[kh]}, s98, {ops['rc' + str(kh)]}")
    # mask threshold: thr0 + 32 qt; kh's dS^T block is stored iff qt >= its first stored pair (else voffset + 2^30)
    out += [f"s_lshl_b32 s98, {SG_QT}, 5", f"s_add_co_i32 {SG_THR}, s98, {ops['thr0']}"]
    for kh in range(NKV):
        out += [
            f"s_cmp_ge_i32 {SG_QT}, {ops['qpf' + str(kh)]}",
            f"s_cselect_b32 {SG_PT[kh]}, 0, {hex(SPILL_OFF)}",
        ]
    return out


def loop_entry(ops, lp):
    """After tile 0's scalars: an item's masked tiles come first (k < n_mask), in M<s>; with none, U0 follows."""
    return [f"s_cmp_gt_i32 {ops['nmask']}, 0", f"s_cbranch_scc1 {lp}_M0"]


def region_loop(L, ops, lp):
    """The loop's blocks [(name, label, texts, successors in the region)] in layout order
    U0 U1 U2 M0 M1 M2 (a tile in ring stage s runs U<s> or, with the causal mask on, M<s>). A body computes the next
    tile's scalars and, last of them, counts k and compares it with N: the compare's SCC is long there at the block
    end (no instruction after it writes SCC), so the one conditional branch of an unmasked tile does not wait for it.
    U0 and U1 leave for {lp}_X ("X") once k = N and fall through into the next stage's body; U2 branches back to U0
    while k < N. The masked tiles are the item's first n_mask: M<s> (out of line) leaves for X once k = N, goes on in
    M<s+1> while k < n_mask and enters the unmasked bodies at U<s+1> after the last masked tile."""
    lo = loop_ops(ops)
    n, n_mask = ops["n"], ops["nmask"]
    ublocks, mblocks = [], []
    for s in range(DEPTH):
        s1 = (s + 1) % DEPTH
        nxt = loop_scalars(lo, s1, 1) + [f"s_add_co_i32 {SG_K}, {SG_K}, 1"]
        back = s == DEPTH - 1
        tu = region_step(L, lo, False, s, nxt + [f"s_cmp_{'lt' if back else 'ge'}_i32 {SG_K}, {n}"])
        tm = region_step(L, lo, True, s, nxt + [f"s_cmp_ge_i32 {SG_K}, {n}"])
        if back:
            ublocks.append((f"U{s}", tu + [f"s_cbranch_scc1 {lp}_U0", f"s_branch {lp}_X"], ["U0", "X"]))
        else:
            ublocks.append((f"U{s}", tu + [f"s_cbranch_scc1 {lp}_X"], ["X", f"U{s1}"]))
        tail = [
            f"s_cbranch_scc1 {lp}_X",
            f"s_cmp_lt_i32 {SG_K}, {n_mask}",
            f"s_cbranch_scc1 {lp}_M{s1}",
            f"s_branch {lp}_U{s1}",
        ]
        mblocks.append((f"M{s}", tm + tail, ["X", f"M{s1}", f"U{s1}"]))
    return [(name, f"{lp}_{name}", texts, succ) for name, texts, succ in ublocks + mblocks]


# --------------------------------------------------------------------------------------------------------------
# FINAL
# --------------------------------------------------------------------------------------------------------------
def region_final(L, ops):
    """FINAL, in program order: dV and dK to bf16 images in the wave's image area (lane bases ev / ek + kh * 16 rows +
    d * 32), whole-line readback (lane bases rv / rk + rb * 4 rows + li * 128) and global stores (voffset gv / gk
    (the lane's part) + gvo / gko (the wave's first row of the item) + rb * rsv / rsk: the row in the range-checked
    voffset, so a row past skv is dropped; offset li * 128). The
    conversions run group by group, (V, kh) then (K, kh), each image store one conversion after its own; a group's
    readback rows (rb = 4 kh .. 4 kh + 3) become loadable after its last image store (a wave's LDS ops complete in
    order), one ds_load_b128 per conversion, each global store FINAL_LAG pairs after its load. Per image, row block
    rb's voffset is its own VGPR of D (dead after the loop), so no VGPR an in-flight store reads is redefined."""
    convs = []  # (group, VALU texts, image store text)
    pairs = {}  # group -> [(load text, texts before the store, store text)]
    tmp = L.O + L.R0 + L.R1  # readback buffers: dead after the loop
    k = 0
    # the stores read copies of the resources in SGPRs the next item's code rewrites late: the workgroup goes on to
    # the next item while the last stores are still on their way
    head = [
        f"s_mov_b32 {sreg(d, i)}, {sreg(ops[r], i)}"
        for r, d in (("rdv", SG_FV), ("rdk", SG_FK))
        for i in range(4)
    ]
    ops = dict(ops, rdv=SG_FV, rdk=SG_FK)
    for isk, x, n_d, base in ((0, XV, NDO_V, ops["ev"]), (1, XK, NDO_QK, ops["ek"])):
        nli, rbase, gv, go, rs, rsrc = (
            (2, ops["rv"], ops["gv"], ops["gvo"], ops["rsv"], ops["rdv"]),
            (3, ops["rk"], ops["gk"], ops["gko"], ops["rsk"], ops["rdk"]),
        )[isk]
        vo = [L.D[4 * isk + q // 8] + q % 8 for q in range(4 * NKV)]  # rb's voffset
        for kh in range(NKV):
            for d in range(n_d):
                a = L.dk(kh, d) if isk else L.dv(kh, d)
                c = [f"v_cvt_pk_bf16_f32 {vr(a + q)}, {vr(a + 2 * q)}, {vr(a + 2 * q + 1)}" for q in range(4)]
                convs.append(
                    ((isk, kh), c, f"ds_store_b128 {base}, {vr(a, 4)}{offset(kh * 16 * x + d * 32)}")
                )
            lst = []
            for rb in range(4 * kh, 4 * kh + 4):
                pre = [
                    f"v_add_nc_u32 {vr(vo[rb])}, {rs}, {vr(vo[rb - 1])}"
                    if rb
                    else f"v_add_nc_u32 {vr(vo[0])}, {go}, {gv}"
                ]
                va = vr(vo[rb])
                for li in range(nli):
                    t = tmp[k % len(tmp)] + 4 * ((k // len(tmp)) % 2)
                    k += 1
                    ld = f"ds_load_b128 {vr(t, 4)}, {rbase}{offset(rb * 4 * x + li * 128)}"
                    st = f"buffer_store_b128 {vr(t, 4)}, {va}, {rsrc}, null offen{offset(li * 128)}"
                    lst.append((ld, pre if li == 0 else [], st))
            pairs[(isk, kh)] = lst
    text, ready, flight = head, [], []

    def advance(final=False):
        if ready:
            ld, pre, st = ready.pop(0)
            text.append(ld)
            flight.append((pre, st))
        if flight and (len(flight) > FINAL_LAG or final and not ready):
            pre, st = flight.pop(0)
            text.extend(pre + [st])

    for i, (g, c, _) in enumerate(convs):
        text.extend(c)
        if i:
            pg, _, pst = convs[i - 1]
            text.append(pst)
            if pg != g:  # group pg's last image store is out: its rows can be read back
                ready.extend(pairs[pg])
        advance()
    text.append(convs[-1][2])
    ready.extend(pairs[convs[-1][0]])
    while ready or flight:
        advance(final=True)
    return text


# --------------------------------------------------------------------------------------------------------------
# the kernel body as one inline asm statement
# --------------------------------------------------------------------------------------------------------------
# the VGPR operands in inline asm order, pinned to v0.. in this order: the lane bases (the kernel's lane-dependent
# values) and the pool (POOL: the per-workgroup scalars, one per lane)
_VOPS = (
    "lk",
    "lvs",
    "lq",
    "lv",
    "ltv",
    "ltk",
    "nlv",
    "vds",
    "dl",
    "ev",
    "ek",
    "rv",
    "rk",
    "gv",
    "gk",
    "pool",
)
OPERANDS = {"MAIN": tuple((x, "v") for x in _VOPS)}
VPIN = {name: i for i, name in enumerate(_VOPS)}  # operand -> its pinned VGPR


def _ops():
    """{name: register text} the regions use: the VGPR operands at their pins, the SGPR plan (SG)."""
    if len(VPIN) > BASE:
        raise ValueError("operand registers collide with the generated code's")
    out = {name: f"v{i}" for name, i in VPIN.items()}
    out.update(SG)
    return out


def _unknown():
    """What may be in flight where control comes from code the model does not follow."""
    return A.Flow([A.State(unknown=True)], [A.XState(True)], [[]])


# region tail: nothing of the region may still read an SGPR (XCNT) or a VGPR (vm_vsrc) the kernel's code after it
# could rewrite
POST = ["s_wait_xcnt 0x0", "s_wait_alu depctr_vm_vsrc(0)"]
LP = ".LHKD"  # MAIN's labels: LP + "${:uid}_" + block (one MAIN per kernel)
_LINES = {}


def _body(L, ops, lp, entry_x, pieces):
    """(lines, SGPRs the instructions in flight at an item's entry may still read): the body for the item entry's
    in-flight SGPR reads `entry_x` and `pieces` warm-up loads."""
    lo = loop_ops(ops)
    # the warm-up block (skipped by waves 1..: its label is a block start), then the setup from unknown code again
    warm = region_warm(ops, lp, pieces)
    k = warm.index(f"{lp}_PC:")
    main, _ = A.emit(warm[:k], _unknown())
    main.append(warm[k])
    lines_w, wfl0 = A.emit(warm[k + 1 :], _unknown())
    main += lines_w + [f"{lp}_WARM:"]
    lines_i, wfl = A.emit(region_wg_init(L, ops), _unknown())
    main += lines_i
    wfl.xs += wfl0.xs  # the warm-up loads' SGPR reads
    main.append(f"{lp}_ITEM:")
    item = region_item(ops, lp) + region_init(L, ops) + loop_init(L, lo) + loop_scalars(lo, 0, 0)
    item += loop_entry(ops, lp)
    # an item's entry: anything may be in flight on the VGPRs; the SGPRs an instruction in flight may still read
    # are entry_x
    lines_item, fl = A.emit(item, A.Flow([A.State(unknown=True)], [A.XState(False, entry_x)], [[]]))
    main += lines_item
    blocks = region_loop(L, ops, lp)
    items = {name: [A.make_item("\t" + x) for x in texts] for name, _, texts, _ in blocks}
    preds = {name: [] for name, *_ in blocks}
    for name, _, _, succ in blocks:
        for x in succ:
            if x != "X":
                preds[x].append(name)
    for name in ("U0", "M0"):  # the entry: loop_entry
        preds[name].append("RB")
    exits, sig, lines = {"RB": fl}, {}, {}
    for _ in range(96):
        changed = False
        for name in preds:
            flows = [exits[p] for p in preds[name] if p in exits]
            if not flows:
                continue
            fin = A.Flow.join(flows)
            if sig.get(name) == fin.sig():
                continue
            sig[name] = fin.sig()
            lines[name], exits[name] = A.emit_block(items[name], fin)
            changed = True
        if not changed:
            break
    else:
        raise RuntimeError("the loop's entry flows did not converge")
    for name, label, _, _ in blocks:
        main += [f"{label}:"] + lines[name]
    main.append(f"{lp}_X:")
    xfl = A.Flow.join([exits[n] for n, _, _, succ in blocks if "X" in succ])
    # exit: every TDM op of the item landed (FINAL's images use the wave's image area, not the ring), and nothing of
    # the loop still reads an SGPR; then FINAL and the next item
    ex, xfl = A.emit(["s_wait_tensorcnt 0x0"], xfl)
    main += ex + ["\ts_wait_xcnt 0x0"]
    for x in xfl.xs:
        x.unknown, x.regs = False, set()
    final, ffl = A.emit(region_final(L, ops) + [f"s_branch {lp}_ITEM"], xfl)
    main += final
    main.append(f"{lp}_EXIT:")
    main.append("\ts_wait_loadcnt 0x0")  # (a workgroup without an item: the warm-up loads)
    if any(x.unknown for x in wfl.xs + ffl.xs):
        raise RuntimeError("unknown SGPR reads in flight at an item's entry")
    main += ["\t" + x for x in POST] + [f"{lp}_END:"]
    return main, set().union(*(x.regs for x in wfl.xs + ffl.xs))


def all_regions(base=BASE):
    """{"MAIN": instruction lines} with the operand stand-ins: the persistent workgroup's whole body.
    wave 0's warm-up loads of the code itself (region_warm), the per-workgroup setup (from unknown code);
    {LP}_ITEM: the next item's row and addresses, INIT, the tile loop's blocks (to a fixed point of their entry
    flows), the exit waits, FINAL, back to {LP}_ITEM. Every pass enters with anything in flight on the VGPRs, but
    only the SGPR reads of the instructions issued since the exit waits: FINAL's stores (the item's first part
    rewrites none of their SGPRs, so it does not wait for FINAL's last stores);
    {LP}_EXIT: the end of the workgroup's list."""
    if base in _LINES:
        return _LINES[base]
    L = Layout(base)
    ops = _ops()
    lp = LP + "${:uid}"
    entry_x, pieces = set(), 0
    for _ in range(6):
        main, x = _body(L, ops, lp, entry_x, pieces)
        n = -(-code_bytes(main) // WARM_PIECE)
        if x == entry_x and n == pieces:
            break
        entry_x, pieces = x, n
    else:
        raise RuntimeError("the item entry's in-flight SGPR reads / the warm-up loads did not converge")
    out = {"MAIN": main}
    _LINES[base] = out
    return out


_VTOK = re.compile(r"(?<![\w$.\[])v(\d+)\b(?!\s*/\*)")


def region_asm(name, base=BASE):
    """(asm template, constraints) of region ``name`` as one inline asm statement: its operands (OPERANDS[name]) as
    $0.. inputs, pinned (VPIN), clobbering v[base:255] (the generated code's VGPRs that LLVM could allocate),
    CLOBBER_S, vcc, scc and memory. The generated code also uses v256..v1023, which LLVM can neither name nor
    allocate here (the kernel reserves them)."""
    sig = OPERANDS[name]
    lines = all_regions(base)[name]

    def sub_v(m):
        r = int(m.group(1))
        if r >= base:
            return f"v{r}"
        if r >= len(sig):
            raise RuntimeError(f"{name}: VGPR v{r} below the generated code's base is not an operand")
        return f"${r}"

    body = []
    for line in lines:
        code, sep, comment = line.strip().partition(";")
        body.append(_VTOK.sub(sub_v, code).rstrip() + (f" ;{comment}" if sep else ""))
    for m in re.finditer(r"(?<![\w$.])s\[?(\d+)", "\n".join(body)):
        if int(m.group(1)) not in CLOBBER_S:
            raise RuntimeError(f"{name}: SGPR s{m.group(1)} is not one of the generated code's")
    cons = [f"{{v{VPIN[n]}}}" for n, _ in sig]
    clob = [f"~{{v{r}}}" for r in range(base, 256)] + [f"~{{s{r}}}" for r in CLOBBER_S]
    clob += ["~{vcc}", "~{scc}", "~{memory}"]
    return "\n".join(body), ",".join(cons + clob)
