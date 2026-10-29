###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""r8gen -- the generated regions of k_dkdv96ds's 96-row blocks (MHA, NKV 3): r8a's loop structure, re-scheduled.

The shell (hip/src/k_dkdv96ds.hip, -DHKGEN=1) runs one STEP / STEPM marker per tile (the 3-stage Q / dO TDM ring,
tile t in stage t mod 3, prefetch distance 2). build.py reads the registers LLVM gave the marker operands; this module
returns the real instructions of every region:

  INIT   K / V fragments (60 buffer_load_b128), accumulators = 0, (c1, c1), (scale, scale)
  RB     hh0 operands of tile 0 (stage 0: 20 ds_load_b128), NL / ND of tile 0 (8 buffer_load_b128)
  STEP   one tile k (STEPM: a diagonal tile -- the causal mask on, the dS^T stores of unstored blocks dropped):
           TDM of tile k+2 (3 tensor_load_to_lds, operands from the shell)
           S / dP of hh0 (30 WMMA, dt outer, operands read back in the previous step), the hh1 operands of stage k
             streamed into the hh0 tuples as each dt dies, S / dP of hh1 (30 WMMA)
           softmax / dS (k_dkdv96ds's arithmetic, packed f32): hh0 under hh1's WMMAs, hh1 kh by kh in front of and
             under dV; bf16 P / dS straight into the dK / dV operand tuples
           dV^T += dO^T P (24 WMMA, kh outer: the 8 dO^T tr16 operands), dK^T += Q^T dS (36 WMMA, d outer: 12 Q^T tr16
             operands, the first 8 loaded early, the last 4 into dead hh1 raw tuples)
           the dS^T stores of tile k (dslayout v2: each dS operand's two 16-B halves), NL / ND of tile k+1
           s_wait_tensorcnt 3 + s_barrier_signal after the last LDS read of stage k, s_barrier_wait, then the hh0
             operands of stage k+1 (into the dO^T tuples of this step, dead after dV) under dK
  FINAL  accumulators -> bf16 dV / dK images in LDS (epi_images' layout), whole-line readback, global stores

Numerics are k_dkdv96ds's per element and per accumulator: S / dP chains over dt ascending from +0, the same fma /
exp2 / mul / RNE bf16 conversions, every dV / dK accumulator one WMMA per tile in tile order, so dk / dv / dS are
meant to be bitwise equal to r9a's (dslayout v2).

Register map (true VGPR numbers, all at or above the shell's `base`): see Layout. Program order is the semantics; the
list scheduler (bwdgen.schedule) reorders along the dependence graph (asmkit's parser: VGPR / SGPR / scc / vcc RAW,
WAR, WAW) plus order edges for LDS ordering (barrier, epilogue images). asmkit.emit adds s_set_vgpr_msb, the expert
waits (va_vdst: --rule warv) and hazard v_nops."""

import bwdgen as B
from bwdgen import add_order, build_dag, mk, schedule, vr

NKV, NDT_QK, NDT_V, NDO_V, NDO_QK = 3, 6, 4, 8, 12
XK, XV = 400, 272
QOFF = 32 * XV  # 8704: Q tile offset inside a ring stage
QDO_B = 32 * (XV + XK)  # 21504: one ring stage
DEPTH = 3  # ring stages
DS_ST_B = 512
NEG = "0xff61b1e6"  # K1_NEG = -3e38
EPI_RB = 4 * NKV  # whole-line readback rows of 4 kv rows
XS_DK = 12  # dK WMMAs between the barrier signal and its wait
ABL = set()  # timing-only ablations (garbage numerics): "nosoft" (no softmax VALU), "nolds" (no S/dP-operand or tr16 LDS
#              loads: the WMMAs read stale registers), "nonl" (no NL/ND loads, no dS^T stores), "notdm" (no TDM)

# machine model for the list scheduler (micro/wmma_mix: a WMMA issues every 8 cycles; ~3 core VALU + 1 TRANS behind
# each are nearly free; LDS loads ~200 cycles under load)
MODEL = {
    "COST": {"VALU": 2, "TRANS": 3, "DS_LD": 2, "DS_ST": 2, "VM_LD": 2, "VM_ST": 2, "TDM": 2, "SALU": 1, "BAR_S": 1,
             "BAR_W": 1, "WAIT_CNT": 0, "OTHER": 1},  # fmt: skip
    "LAT": {"WMMA": 20, "VALU": 4, "TRANS": 8, "DS_LD": 90, "DS_ST": 4, "VM_LD": 800, "VM_ST": 4, "TDM": 4,
            "SALU": 2, "BAR_S": 1, "BAR_W": 1, "WAIT_CNT": 1, "OTHER": 1},  # fmt: skip
    "XDL_BUSY": 8,
    "LDS_OCC": 8,
    "LDS_LAT": 60,
    "WMMA_CHAIN": 16,
    "WAR_XDL": 24,
}


def set_nkv(n):
    """Generate for blocks of 16 n kv rows per wave (3: the 96-row blocks, 2: the 64-row tail)."""
    global NKV, EPI_RB
    NKV, EPI_RB = n, 4 * n


def apply_model():
    B.LDS_OCC, B.LDS_LAT = MODEL.get("LDS_OCC"), MODEL.get("LDS_LAT")
    B.COST.clear()
    B.COST.update(MODEL["COST"])
    B.LAT.clear()
    B.LAT.update(MODEL["LAT"])
    B.XDL_BUSY, B.WMMA_CHAIN, B.WAR_XDL = MODEL["XDL_BUSY"], MODEL["WMMA_CHAIN"], MODEL["WAR_XDL"]


class Layout:
    """VGPR map (true numbers) of the generated state. Everything at or above `base`; 8-VGPR tuples never cross a
    256 block. Roles of the shared tuples over one step:
      O[0..9]  hh0 operands (Q dt 0..5 at O[2 dt] / O[8 + dt - 4], dO dt 0..3 at O[2 dt + 1]); the hh1 operand of the
               same slot streams in once hh0's dt died; then dO^T operands BDO[d] = O[d] (d < 8, after hh1's dt
               died) and Q^T operands BQ[6], BQ[7] = O[8], O[9]; after dV, the next step's hh0 operands
      R0[0..5] hh0 raw S / dP (R0[2 kh] = S, R0[2 kh + 1] = dP); after hh0's softmax BQ[0..5]
      R1[0..5] hh1 raw S / dP; after hh1's softmax BQ[8..11] = R1[0..3]
      P[kh], DS[kh]  bf16 P / dS operands of kv sub-tile kh (dwords 0..3: hh0, 4..7: hh1)
      PRE[e]   NL / ND quads (e = hh * 4 + {0, 1}: NL, + {2, 3}: ND), the next tile's after the last read
      ACC, KF, VF  accumulators (dV (kh, d) at kh * 8 + d, dK at 24 + kh * 12 + d), K / V fragments
      T[0..1]  mask scratch (STEPM)"""

    def __init__(self, base):
        self.base = base
        nxt = [base + (base % 2)]
        blocks = {0: [], 1: [], 2: [], 3: []}

        def take(n, align=2):
            for blk in range(4):
                lo = max(nxt[0] if blk == 0 else 0, blk * 256)
                # first fit inside this block, after everything taken there
                used = blocks[blk]
                r = max([lo] + [a + m for a, m in used])
                r += (-r) % align
                if r + n <= (blk + 1) * 256:
                    used.append((r, n))
                    return r
            raise ValueError(f"layout does not fit above base v{base}")

        # big persistent arrays first, block by block (so the tuples land where planned)
        self.acc = [take(8, 8) for _ in range(NKV * (NDO_V + NDO_QK))]  # NKV 3: 480, blocks 1, 2 (+ rest of 0 / 3)
        self.kf = [[take(8, 8) for _ in range(NDT_QK)] for _ in range(NKV)]
        self.vf = [[take(8, 8) for _ in range(NDT_V)] for _ in range(NKV)]
        self.O = [take(8, 8) for _ in range(10)]
        self.R0 = [take(8, 8) for _ in range(2 * NKV)]
        self.R1 = [take(8, 8) for _ in range(2 * NKV)]
        self.D = [take(8, 8) for _ in range(NDO_V)]
        # bf16 P / dS operands in place of hh0's raw S / dP (hh0's dwords 0..3, hh1's 4..7)
        self.P = [self.R0[2 * kh] for kh in range(NKV)]
        self.DS = [self.R0[2 * kh + 1] for kh in range(NKV)]
        pre8 = [take(8, 8) for _ in range(4)]  # (hh0 NL, hh0 ND, hh1 NL, hh1 ND): 8-VGPR tuples (ND: a C operand)
        self.PRE = [pre8[e // 2] + 4 * (e % 2) for e in range(8)]
        self.cc1, self.ccs = take(2), take(2)
        self.T = [take(1, 1) for _ in range(2 + NKV)]  # mask scratch (2), STEPM's dS^T voffsets (NKV)
        self.PRB = [take(1, 1) for _ in range(4)]  # probe: per-segment cycle sums (HKGEN_PROBE builds)
        self.top = max(a + m for blk in blocks.values() for a, m in blk)

    # roles
    def op_hh0(self, isq, dt):
        """hh0 operand tuple: Q (isq 1) of dt, or dO (isq 0) of dt < 4."""
        return self.O[2 * dt] if (isq and dt < 4) else (self.O[8 + dt - 4] if isq else self.O[2 * dt + 1])

    op_hh1 = op_hh0  # the same slots, once hh0's dt died

    def raw(self, hh, kh, j):
        return (self.R0 if hh == 0 else self.R1)[2 * kh + j]

    def bdo(self, d):
        return self.D[d]

    def bq(self, d):
        return self.O[d] if d < 10 else self.D[d - 10]

    def dv(self, kh, d):
        return self.acc[kh * NDO_V + d]

    def dk(self, kh, d):
        return self.acc[NKV * NDO_V + kh * NDO_QK + d]

    def all_regs(self):
        regs = set()
        for t in self.acc + [x for r in self.kf for x in r] + [x for r in self.vf for x in r] + self.O + self.R0 + \
                self.R1 + self.D:  # fmt: skip
            regs |= set(range(t, t + 8))
        for q in self.PRE:
            regs |= set(range(q, q + 4))
        regs |= {self.cc1, self.cc1 + 1, self.ccs, self.ccs + 1} | set(self.T)
        return regs


# --------------------------------------------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------------------------------------------
def region_init(L, ops):
    """INIT: (c1, c1), (scale, scale) (ND_FOLD: (1 / scale, 1 / scale), FINAL's dV factor); accumulators = 0; K / V
    fragments (buffer loads: lane part + kv sub-tile part + dt * 64 + u * 32)."""
    out = [f"v_mov_b32 {vr(L.cc1)}, {ops['c1']}", f"v_mov_b32 {vr(L.cc1 + 1)}, {ops['c1']}"]
    if ND_FOLD:
        out += [f"v_rcp_f32 {vr(L.ccs)}, {ops['sc']}", f"v_mov_b32 {vr(L.ccs + 1)}, {vr(L.ccs)}"]
    else:
        out += [f"v_mov_b32 {vr(L.ccs)}, {ops['sc']}", f"v_mov_b32 {vr(L.ccs + 1)}, {ops['sc']}"]
    if "lk" in ops:  # shell -DHKGEN_KV_TDM: K / V rows by TDM into the wave's LDS staging area ([16 NKV][XV] of V,
        #              then [16 NKV][XK] of K; ops as a ring stage's, descriptors in s96..s103 / s92..s95 one op
        #              at a time: INIT may clobber only s90..s103), the accumulators cleared while they are in
        #              flight, the fragments read once every TDM op landed
        rows = 16 * NKV
        kg, vg = ops["kg"], ops["vg"]
        for o in range(3):
            g1, g0 = 96, 92
            ga = (vg if o == 0 else kg)
            lo, hi = _sv(ga, 0), _sv(ga, 1)
            words = [hex(TDM_W0[o]), "0xffff0000", hex((rows << 16) | 0x7FFF), hex((64 if o == 2 else 128) << 16),
                     str(rows)]
            out += [f"s_mov_b32 s{g1 + i}, {w}" for i, w in enumerate(words)]
            out += [f"s_mul_i32 s{g1 + 5}, {ops['hkv']}, {128 if o == 0 else 192}", f"s_mov_b32 s{g1 + 6}, 0",
                    f"s_mov_b32 s{g1 + 7}, 0", f"s_mov_b32 s{g0}, 1",
                    f"s_add_co_i32 s{g0 + 1}, {ops['kvl']}, {0 if o == 0 else rows * XV + (256 if o == 2 else 0)}",
                    f"s_add_co_u32 s{g0 + 2}, {lo}, {256 if o == 2 else 0}", f"s_add_co_ci_u32 s{g0 + 3}, {hi}, 0",
                    f"s_bitset1_b32 s{g0 + 3}, 31", f"tensor_load_to_lds s[{g0}:{g0 + 3}], s[{g1}:{g1 + 7}]"]  # fmt: skip
        for t in L.acc:
            for q in range(0, 8, 2):
                out.append(f"v_mov_b64 {vr(t + q, 2)}, 0")
        out += [f"v_mov_b32 {vr(r)}, 0" for r in L.PRB]
        out.append("s_wait_tensorcnt 0x0")
        for kh in range(NKV):
            for dt in range(NDT_QK):
                for u in range(2):
                    off = kh * 16 * XK + dt * 64 + u * 32
                    out.append(f"ds_load_b128 {vr(L.kf[kh][dt] + 4 * u, 4)}, {ops['lk']}" + (f" offset:{off}" if off else ""))
            for dt in range(NDT_V):
                for u in range(2):
                    off = kh * 16 * XV + dt * 64 + u * 32
                    out.append(f"ds_load_b128 {vr(L.vf[kh][dt] + 4 * u, 4)}, {ops['lv']}" + (f" offset:{off}" if off else ""))
        return out
    for kh in range(NKV):
        for dt in range(NDT_QK):
            for u in range(2):
                off = dt * 64 + u * 32
                out.append(f"buffer_load_b128 {vr(L.kf[kh][dt] + 4 * u, 4)}, {ops['vk']}, {ops['rk']}, "
                           f"{ops['sk' + str(kh)]} offen" + (f" offset:{off}" if off else ""))  # fmt: skip
        for dt in range(NDT_V):
            for u in range(2):
                off = dt * 64 + u * 32
                out.append(f"buffer_load_b128 {vr(L.vf[kh][dt] + 4 * u, 4)}, {ops['vv']}, {ops['rv']}, "
                           f"{ops['sv' + str(kh)]} offen" + (f" offset:{off}" if off else ""))  # fmt: skip
    for t in L.acc:
        for q in range(0, 8, 2):
            out.append(f"v_mov_b64 {vr(t + q, 2)}, 0")
    out += [f"v_mov_b32 {vr(r)}, 0" for r in L.PRB]
    return out


def rb_texts(L, qbase, vbase, hh, dst_of, stage):
    """The 20 ds_load_b128 of one hh half's S / dP operands of ring stage `stage` (k1_rd_const offsets from the lane
    bases; the stage offset in the immediate)."""
    out = []
    so = stage * QDO_B
    for dt in range(NDT_QK):
        for u in range(2):
            off = so + QOFF + hh * 16 * XK + dt * 64 + u * 32
            out.append((("Q", dt, u), f"ds_load_b128 {vr(dst_of(1, dt) + 4 * u, 4)}, {qbase} offset:{off}"))
        for u in range(2 if dt < NDT_V else 0):
            off = so + hh * 16 * XV + dt * 64 + u * 32
            out.append((("D", dt, u), f"ds_load_b128 {vr(dst_of(0, dt) + 4 * u, 4)}, {vbase}"
                        + (f" offset:{off}" if off else "")))  # fmt: skip
    return out


def nl_texts(L, ops):
    out = []
    for e in range(8):
        r = ops["rnl"] if (e % 4) // 2 == 0 else ops["rnd"]
        off = (e // 4) * 64 + (e % 2) * 16
        out.append(f"buffer_load_b128 {vr(L.PRE[e], 4)}, {ops['nlv']}, {r}, {ops['nls']} offen"
                   + (f" offset:{off}" if off else ""))  # fmt: skip
    return out


def region_rb(L, ops):
    """RB (prologue, after the shell's barrier): hh0 operands of stage 0, NL / ND of tile 0."""
    return [t for _, t in rb_texts(L, ops["lq"], ops["lv"], 0, L.op_hh0, 0)] + nl_texts(L, ops)


def softmax_unit(L, hh, kh, masked, add, mask_tmp=None):
    if "nosoft" in ABL:
        return [add(f"v_mov_b32 {vr(L.P[kh] + hh * 4)}, {vr(L.raw(hh, kh, 0))}")]
    """Node indices of one (hh, kh) unit's softmax / dS: 4 element pairs, straight into P[kh] / DS[kh] dwords
    hh * 4 + j. masked: tt = kv > q + cshift ? K1_NEG : tt (mask_tmp: VGPR holding dl - thr - 16 (hh - kh), so the
    element si is masked iff mask_tmp > si)."""
    nodes = []
    s_, d_ = L.raw(hh, kh, 0), L.raw(hh, kh, 1)
    for j in range(4):
        nl = L.PRE[hh * 4 + j // 2] + 2 * (j % 2)
        nd = L.PRE[hh * 4 + 2 + j // 2] + 2 * (j % 2)
        s2, d2 = s_ + 2 * j, d_ + 2 * j
        nodes.append(add(f"v_pk_fma_f32 {vr(s2, 2)}, {vr(s2, 2)}, {vr(L.cc1, 2)}, {vr(nl, 2)}"))
        if masked:
            for e in range(2):
                si = 2 * j + e
                nodes.append(add(f"v_cmp_ge_i32 vcc_lo, {si}, {vr(mask_tmp)}"))
                nodes.append(add(f"v_cndmask_b32 {vr(s2 + e)}, {NEG}, {vr(s2 + e)}, vcc_lo"))
        nodes.append(add(f"v_exp_f32 {vr(s2)}, {vr(s2)}"))
        nodes.append(add(f"v_exp_f32 {vr(s2 + 1)}, {vr(s2 + 1)}"))
        if not ND_FOLD:
            nodes.append(add(f"v_pk_fma_f32 {vr(d2, 2)}, {vr(d2, 2)}, {vr(L.ccs, 2)}, {vr(nd, 2)}"))
        nodes.append(add(f"v_pk_mul_f32 {vr(d2, 2)}, {vr(s2, 2)}, {vr(d2, 2)}"))
        w = hh * 4 + j
        nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(L.P[kh] + w)}, {vr(s2)}, {vr(s2 + 1)}"))
        nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(L.DS[kh] + w)}, {vr(d2)}, {vr(d2 + 1)}"))
    return nodes


ND_FOLD = False  # k_delta -DHK_ND_FOLD: ND' = -delta is the dP chains' C init, NL' = NL + log2(scale) makes the
#                 softmax's P' = scale * P, so dS = P' dP' needs no dP fma; dV = (dO^T P') / scale in FINAL
SCAL_IN_BODY = False  # each body computes the next tile's scalars (SALU nodes in its WMMA windows); the headers keep
#                      only the M / U branch, and U falls through into N (the M bodies out of line)
TDM_DUMMY = False  # a tile past the last (k + 2 >= N) moves 16 B (tile 8 x 1) instead of re-loading tile N-1: no
#                   wasted L2 / HBM traffic, and the epilogue's tensorcnt-0 wait finds the ring settled
FINAL_PIPE = False  # FINAL in program order, pipelined: group (V / K image, kh) converted + stored to its image while
#                    the previous group's rows are read back and stored to global memory (a readback needs only its
#                    own group's image stores; LDS ops of a wave are in order). Else list-scheduled, every image
#                    store before every readback
SCAL_XCNT_GAP = 160  # cycles between a VMEM / TDM op and the next tile's redefinition of an SGPR it reads
HH1_SPLIT = 0  # hh1's S / dP order: dt < HH1_SPLIT dt outer (every kh), the rest kh-major (6: all dt outer)
BDO_AFTER_HH1 = False  # the dO^T loads issue only after every hh1 operand load (they do not crowd the LDS queue)


def hh1_order():
    """hh1's S / dP WMMAs, kh-major: kh 0's chains finish a third into hh1's run, so its softmax (and so P[0], the
    first dV operand) hides under kh 1 / 2's WMMAs. The S and dP chains of one kh interleave; a chain's next WMMA
    stands >= 2 positions after its previous one (16 cycles: WMMA_CHAIN); lower kh first, then the chain with more
    WMMAs left. Items (kind 'S' / 'P', kh, dt)."""
    out = []
    for dt in range(min(HH1_SPLIT, NDT_QK)):
        for kind in ("S", "P"):
            if kind == "P" and dt >= NDT_V:
                continue
            out += [(kind, kh, dt) for kh in range(NKV)]
    chains = []
    for kh in range(NKV):
        chains.append([("S", kh, dt) for dt in range(HH1_SPLIT, NDT_QK)])
        chains.append([("P", kh, dt) for dt in range(HH1_SPLIT, NDT_V)])
    chains = [c for c in chains if c]
    last, nxt = [-10] * len(chains), [0] * len(chains)
    n0 = len(out)
    while len(out) - n0 < sum(len(c) for c in chains):
        p = len(out)
        live = [c for c in range(len(chains)) if nxt[c] < len(chains[c])]
        cand = [c for c in live if last[c] <= p - 2] or live
        c = min(cand, key=lambda c: (chains[c][0][1], -(len(chains[c]) - nxt[c]), c))
        out.append(chains[c][nxt[c]])
        nxt[c] += 1
        last[c] = p
    return out


def step_nodes(L, ops, masked, stage, scal=None):
    """Nodes of one tile in ring stage `stage` (program order = semantics) and the WMMA stream order.

    Register roles over the step (Layout): hh0's S / dP operands in O (read back in the previous step), hh0's raw S /
    dP in R0; hh1's operands stream into O as hh0's dt die, hh1's raw in R1. Softmax / dS write bf16 P / dS in place
    of hh0's raw (P[kh] = R0[2 kh], dS[kh] = R0[2 kh + 1]: hh0's dwords 0..3, hh1's 4..7). dO^T operands in their own
    tuples D (loaded any time in the step); Q^T 0..9 into O once hh1's S / dP ended, Q^T 10, 11 into D[0], D[1] once
    dV read them; the next step's hh0 operands into O as dK releases Q^T 0..9 (after the ring barrier)."""
    nodes = []

    def add(text):
        nodes.append(mk(text))
        return len(nodes) - 1

    w_sdp0, w_sdp1, w_dv, w_dk = [], [], [], []
    stage_reads, nxt_reads, hh1_ld, bdo_loads = [], [], [], []
    so = stage * QDO_B
    nolds = "nolds" in ABL
    # TDM of tile k+2 (its stage's previous tile was last read before the previous step's barrier); it must issue
    # before the tensorcnt wait below, which then retires tile k+1 (outstanding: k+1, k+2)
    tdm = [add(f"tensor_load_to_lds {ops['g0' + c]}, {ops['g1' + c]}") for c in "abc"] if "notdm" not in ABL else []

    def wmma_sdp(hh, kind, kh, dt):
        if kind == "S":
            s_ = L.raw(hh, kh, 0)
            c = "0" if dt == 0 else vr(s_, 8)
            return add(f"v_wmma_f32_16x16x32_bf16 {vr(s_, 8)}, {vr(L.op_hh0(1, dt), 8)}, {vr(L.kf[kh][dt], 8)}, {c}")
        d_ = L.raw(hh, kh, 1)
        c = (vr(L.PRE[hh * 4 + 2], 8) if ND_FOLD else "0") if dt == 0 else vr(d_, 8)
        return add(f"v_wmma_f32_16x16x32_bf16 {vr(d_, 8)}, {vr(L.op_hh0(0, dt), 8)}, {vr(L.vf[kh][dt], 8)}, {c}")

    # ---- S / dP of hh0, dt outer (S of every kh, then dP: the A operand shared by consecutive WMMAs); hh1's operands
    # of each dt stream into the slots hh0's dt just released (program order after their readers)
    hh1_loads = dict(rb_texts(L, ops["lq"], ops["lv"], 1, L.op_hh1, stage))
    for dt in range(NDT_QK):
        for kind in ("S", "P"):
            if kind == "P" and dt >= NDT_V:
                continue
            for kh in range(NKV):
                w_sdp0.append(wmma_sdp(0, kind, kh, dt))
        for u in range(2 if not nolds else 0):
            stage_reads.append(add(hh1_loads[("Q", dt, u)]))
            hh1_ld.append(stage_reads[-1])
            if dt < NDT_V:
                stage_reads.append(add(hh1_loads[("D", dt, u)]))
                hh1_ld.append(stage_reads[-1])
    # dO^T operands (own tuples: free since the previous step's dV)
    for d in range(NDO_V if not nolds else 0):
        for p in range(2):
            off = so + d * 32 + p * 16 * XV
            stage_reads.append(add(f"ds_load_tr16_b128 {vr(L.bdo(d) + 4 * p, 4)}, {ops['ltv']}"
                                   + (f" offset:{off}" if off else "")))  # fmt: skip
            bdo_loads.append(stage_reads[-1])
    # ---- S / dP of hh1, kh-major
    for kind, kh, dt in hh1_order():
        w_sdp1.append(wmma_sdp(1, kind, kh, dt))
    # ---- softmax / dS: hh0 then hh1, kh by kh; bf16 P / dS in place of hh0's raw
    soft = {}
    for hh in range(2):
        for kh in range(NKV):
            mt = None
            if masked:
                sreg = "s96" if hh == 0 else "s97"
                add(f"s_add_co_i32 {sreg}, {ops['thr']}, {16 * (hh - kh)}")
                add(f"v_subrev_nc_u32 {vr(L.T[hh])}, {sreg}, {ops['dl']}")
                mt = L.T[hh]
            soft[(hh, kh)] = softmax_unit(L, hh, kh, masked, add, mt)
    # Q^T operands 0..9 into O (hh1's operands dead after its S / dP)
    bq_loads = {}

    def qt_load(d):
        bq_loads[d] = []
        for p in range(2 if not nolds else 0):
            off = so + QOFF + d * 32 + p * 16 * XK
            bq_loads[d].append(add(f"ds_load_tr16_b128 {vr(L.bq(d) + 4 * p, 4)}, {ops['ltk']} offset:{off}"))
            stage_reads.append(bq_loads[d][-1])

    for d in range(10):
        qt_load(d)
    # ---- dV^T += dO^T P (kh outer: dV of kh 0 needs P[0] only)
    for kh in range(NKV):
        for d in range(NDO_V):
            a = L.dv(kh, d)
            w_dv.append(add(f"v_wmma_f32_16x16x32_bf16 {vr(a, 8)}, {vr(L.bdo(d), 8)}, {vr(L.P[kh], 8)}, {vr(a, 8)}"))
    # Q^T operands 10, 11 into D[0], D[1] (dead after dV)
    for d in range(10, NDO_QK):
        qt_load(d)
    # ---- dS^T stores of tile k (dslayout v2) and NL / ND of tile k+1 (PRE dead after the softmax)
    ex = []
    for kh in range(NKV):
        vd = ops["vds"]
        if masked:  # a block with no attended entry in this pair is not stored (dslayout): out-of-range voffset
            ex.append(add(f"v_add_nc_u32 {vr(L.T[2 + kh])}, {ops['pt' + str(kh)]}, {ops['vds']}"))
            vd = vr(L.T[2 + kh])
        ex.append(add(f"buffer_store_b128 {vr(L.DS[kh], 4)}, {vd}, {ops['rds']}, {ops['ds' + str(kh)]} offen"))
        ex.append(add(f"buffer_store_b128 {vr(L.DS[kh] + 4, 4)}, {vd}, {ops['rds']}, {ops['ds' + str(kh)]} offen "
                      f"offset:{DS_ST_B}"))  # fmt: skip
    nl = [add(t) for t in nl_texts(L, ops)] if "nonl" not in ABL else []
    if "nonl" in ABL:
        ex = [i for i in ex if nodes[i].kind != "VM_ST"]
    # ---- the ring barrier after the last LDS read of stage k (tensorcnt 3: tile k+1 landed), dK (d outer), the
    # next step's hh0 operands (stage k+1) into O as dK releases Q^T 0..9 (program order: after the barrier wait)
    tw = add("s_wait_tensorcnt 0x3")
    bs = add("s_barrier_signal -1")
    bw = add("s_barrier_wait -1")
    for d in range(NDO_QK):
        for kh in range(NKV):
            a = L.dk(kh, d)
            w_dk.append(add(f"v_wmma_f32_16x16x32_bf16 {vr(a, 8)}, {vr(L.bq(d), 8)}, {vr(L.DS[kh], 8)}, {vr(a, 8)}"))
    for _, t in rb_texts(L, ops["lq"], ops["lv"], 0, L.op_hh0, (stage + 1) % DEPTH) if not nolds else []:
        nxt_reads.append(add(t))
    scal_nodes = [add(t) for t in scal or []]  # the next tile's scalars (after every use of this tile's values)
    build_dag(nodes)
    for j in scal_nodes:  # an SGPR an in-flight VMEM / TDM op still reads: redefine it well after (XCNT)
        for i in range(len(nodes)):
            if nodes[i].kind in ("VM_LD", "VM_ST", "TDM") and i < j and nodes[i].suses & nodes[j].sdefs:
                add_order(nodes, i, j, SCAL_XCNT_GAP)
    for i in tdm:
        add_order(nodes, i, tw)
    add_order(nodes, tw, bs)
    add_order(nodes, bs, bw)
    for i in stage_reads:
        add_order(nodes, i, tw)  # the tensorcnt wait (and so the signal) behind the stage's last LDS read
    # memory ops with a hazard against core VALU (LLVM's va_vdst rule drains every in-flight VALU for them while
    # WMMAs run): the dS^T stores (RAW on the bf16 converts) and the NL / ND loads (WAR on PRE) behind the whole
    # softmax, so one drain covers them
    sm = [i for k in soft for i in soft[k]]
    for i in ex + nl:
        if nodes[i].kind in ("VM_ST", "VM_LD"):
            for j in sm:
                add_order(nodes, j, i)
    for i in nxt_reads:
        add_order(nodes, bw, i)
    if BDO_AFTER_HH1:
        for i in bdo_loads:
            for j in hh1_ld:
                add_order(nodes, j, i)
    return nodes, w_sdp0 + w_sdp1 + w_dv + w_dk


PROBE_PTS = None  # HKGEN_PROBE builds: the three probe points as WMMA counts (None: S/dP end, dV end, barrier wait)
PROBE_SEGS = ("S/dP hh0+hh1 (+softmax hh0)", "softmax hh1 + dV", "dK to the barrier wait", "dK rest")


def region_step(L, ops, masked, stage, mode="list", probe=False, scal=None):
    """(texts, modelled cycles). probe: s_get_shader_cycles at the region start, after the last S / dP WMMA, after the
    last dV WMMA, after the barrier wait and at the end; segment k's delta added to L.PRB[k] (VALU, per lane)."""
    apply_model()
    nodes, wseq = step_nodes(L, ops, masked, stage, scal)
    if mode == "win":
        out, cyc = B.schedule_win(nodes, wseq)
    elif mode == "slot":
        out, cyc = B.schedule_slot(nodes, wseq)
    else:
        out, cyc = schedule(nodes, wseq, mode)
    texts = [n.text for n in out]
    if probe:
        wpos = [k for k, n in enumerate(out) if n.kind == "WMMA"]
        bw = next(k for k, n in enumerate(out) if n.kind == "BAR_W")
        n_sdp, n_dv = 20 * NKV, 8 * NKV
        pts = sorted([wpos[n_sdp - 1] + 1, wpos[n_sdp + n_dv - 1] + 1, bw + 1])
        if PROBE_PTS:  # after WMMA #k (1-based) of the step, k in PROBE_PTS (3 points: 4 segments)
            pts = sorted(wpos[min(k, len(wpos)) - 1] + 1 for k in PROBE_PTS)
        res, k0, prev = ["s_get_shader_cycles_u64 s[102:103]"], 0, "s102"
        for seg, k in enumerate(pts + [len(texts)]):
            res += texts[k0:k]
            cur = "s100" if prev == "s102" else "s102"
            res += [f"s_get_shader_cycles_u64 s[{cur[1:]}:{int(cur[1:]) + 1}]",
                    f"v_add_nc_u32 {vr(L.PRB[seg])}, {cur}, {vr(L.PRB[seg])}",
                    f"v_subrev_nc_u32 {vr(L.PRB[seg])}, {prev}, {vr(L.PRB[seg])}"]  # fmt: skip
            prev, k0 = cur, k
        texts = res
    return texts, cyc


def region_final(L, ops):
    """FINAL: accumulators -> bf16 images (k3_epi_wr: lane base ev / ek + kh * 16 * x + d * 32), whole-line readback
    (k3_epi_rd: lane base rv / rk + rb * 4 * x + li * 128) and global stores (lane voffset gv / gk, soffset sov / sok
    + rb * rsv / rsk, offset li * 128). The image stores precede every readback (order edges: LDS); FINAL_PIPE: see
    there."""
    if FINAL_PIPE:
        return region_final_pipe(L, ops)
    nodes = []

    def add(text):
        nodes.append(mk(text))
        return len(nodes) - 1

    stores, loads = [], []
    for isk, x, n_d, base in ((0, XV, NDO_V, ops["ev"]), (1, XK, NDO_QK, ops["ek"])):
        for kh in range(NKV):
            for d in range(n_d):
                a = L.dv(kh, d) if isk == 0 else L.dk(kh, d)
                if ND_FOLD and isk == 0:  # dV' = dO^T P' = scale * dV
                    for q in range(4):
                        add(f"v_pk_mul_f32 {vr(a + 2 * q, 2)}, {vr(a + 2 * q, 2)}, {vr(L.ccs, 2)}")
                for q in range(4):
                    add(f"v_cvt_pk_bf16_f32 {vr(a + q)}, {vr(a + 2 * q)}, {vr(a + 2 * q + 1)}")
                off = kh * 16 * x + d * 32
                stores.append(add(f"ds_store_b128 {base}, {vr(a, 4)}" + (f" offset:{off}" if off else "")))
    tmp = [L.O[i] for i in range(10)] + list(L.R0) + list(L.R1)
    k = 0
    for isk, x, nli, rbase, gv, so, rs, rsrc in ((0, XV, 2, ops["rv"], ops["gv"], ops["sov"], ops["rsv"], ops["rdv"]),
                                                 (1, XK, 3, ops["rk"], ops["gk"], ops["sok"], ops["rsk"], ops["rdk"])):
        sreg = "s92" if isk == 0 else "s93"
        add(f"s_mov_b32 {sreg}, {so}")
        for rb in range(EPI_RB):
            if rb:
                add(f"s_add_co_i32 {sreg}, {sreg}, {rs}")
            for li in range(nli):
                t = tmp[k % len(tmp)] + 4 * ((k // len(tmp)) % 2)
                k += 1
                off = rb * 4 * x + li * 128
                loads.append(add(f"ds_load_b128 {vr(t, 4)}, {rbase}" + (f" offset:{off}" if off else "")))
                add(f"buffer_store_b128 {vr(t, 4)}, {gv}, {rsrc}, {sreg} offen" + (f" offset:{li * 128}" if li else ""))
    build_dag(nodes)
    for s in stores:
        for l in loads:
            add_order(nodes, s, l)
    apply_model()
    out, _ = schedule(nodes, [i for i, n in enumerate(nodes) if n.kind == "WMMA"], "list")
    return [n.text for n in out]


FINAL_LAG = 8  # FINAL_PIPE: readback loads issued this many (load, store) pairs ahead of their global stores


def region_final_pipe(L, ops):
    """FINAL_PIPE: conversions in the order (V, kh = 0 .. NKV-1), (K, kh = 0 .. NKV-1) (ND_FOLD: dV * 1/scale), each
    image store one conversion later (its cvt results are in by then); a group's readback rows rb = 4 kh .. 4 kh + 3
    become loadable after its last image store (a wave's LDS ops are in order), one ds_load_b128 per conversion,
    its buffer_store_b128 FINAL_LAG pairs later; the soffset of row block rb in one of 4 rotating SGPRs per image
    (a store's SGPR is never redefined while it is in flight). asmkit derives the waits."""
    convs = []  # (group, VALU texts, image store text)
    pairs = {}  # group -> [(load text, store pre texts, store text)]
    tmp = [L.O[i] for i in range(10)] + list(L.R0) + list(L.R1)
    k = 0
    for isk, x, n_d, base in ((0, XV, NDO_V, ops["ev"]), (1, XK, NDO_QK, ops["ek"])):
        for kh in range(NKV):
            for d in range(n_d):
                a_ = L.dv(kh, d) if isk == 0 else L.dk(kh, d)
                c = []
                if ND_FOLD and isk == 0:  # dV' = dO^T P' = scale * dV
                    c += [f"v_pk_mul_f32 {vr(a_ + 2 * q, 2)}, {vr(a_ + 2 * q, 2)}, {vr(L.ccs, 2)}" for q in range(4)]
                c += [f"v_cvt_pk_bf16_f32 {vr(a_ + q)}, {vr(a_ + 2 * q)}, {vr(a_ + 2 * q + 1)}" for q in range(4)]
                off = kh * 16 * x + d * 32
                convs.append(((isk, kh), c, f"ds_store_b128 {base}, {vr(a_, 4)}" + (f" offset:{off}" if off else "")))
            nli, rbase, gv, so, rs, rsrc = ((2, ops["rv"], ops["gv"], ops["sov"], ops["rsv"], ops["rdv"]),
                                            (3, ops["rk"], ops["gk"], ops["sok"], ops["rsk"], ops["rdk"]))[isk]
            sregs = ("s92", "s94", "s96", "s98") if isk == 0 else ("s93", "s95", "s97", "s99")
            lst = []
            for rb in range(4 * kh, 4 * kh + 4):
                sr = sregs[rb % 4]
                pre = [f"s_mov_b32 {sr}, {so}" if rb == 0 else f"s_add_co_i32 {sr}, {sregs[(rb - 1) % 4]}, {rs}"]
                for li in range(nli):
                    t = tmp[k % len(tmp)] + 4 * ((k // len(tmp)) % 2)
                    k += 1
                    off = rb * 4 * x + li * 128
                    lst.append((f"ds_load_b128 {vr(t, 4)}, {rbase}" + (f" offset:{off}" if off else ""),
                                pre if li == 0 else [],
                                f"buffer_store_b128 {vr(t, 4)}, {gv}, {rsrc}, {sr} offen"
                                + (f" offset:{li * 128}" if li else "")))  # fmt: skip
            pairs[(isk, kh)] = lst
    text, ready, flight = [], [], []

    def step(final=False):
        if ready:
            ld, pre, st = ready.pop(0)
            text.append(ld)
            flight.append((pre, st))
        if flight and (len(flight) > FINAL_LAG or final and not ready):
            pre, st = flight.pop(0)
            text.extend(pre + [st])

    for i, (g, c, st) in enumerate(convs):
        text.extend(c)
        if i:
            pg, _, pst = convs[i - 1]
            text.append(pst)
            if pg != g:  # group pg's last image store is out: its rows can be read back
                ready.extend(pairs[pg])
        step()
    text.append(convs[-1][2])
    ready.extend(pairs[convs[-1][0]])
    while ready or flight:
        step(final=True)
    return text


# --------------------------------------------------------------------------------------------------------------
# the LOOP marker: the readback of tile 0, then the tile loop (unrolled by the ring depth) with its scalar work
# --------------------------------------------------------------------------------------------------------------
# generated SGPRs inside LOOP (the marker clobbers s48..s103)
SG_G1 = ("s[48:55]", "s[56:63]", "s[64:71]")
SG_G0 = (72, 76, 80)
SG_K, SG_QT, SG_T1, SG_T2, SG_NLS = "s84", "s85", "s86", "s87", "s88"
SG_SO = ("s89", "s90", "s91")
SG_THR = "s92"
SG_PT = ("s93", "s94", "s95")
TDM_W0 = (0x7510000, 0x47510000, 0x87110000)
TDM_W3 = (128 << 16, 128 << 16, 64 << 16)


def _sv(op, i):
    """Element i of an SGPR tuple operand text s[a:b]."""
    import re

    m = re.fullmatch(r"s\[(\d+):(\d+)\]", op)
    if not m or int(m.group(1)) + i > int(m.group(2)):
        raise ValueError(f"{op}: no element {i}")
    return f"s{int(m.group(1)) + i}"


def loop_ops(ops):
    """The step bodies' operand names inside LOOP."""
    o = dict(ops)
    o.update({"g0a": f"s[{SG_G0[0]}:{SG_G0[0] + 3}]", "g0b": f"s[{SG_G0[1]}:{SG_G0[1] + 3}]",
              "g0c": f"s[{SG_G0[2]}:{SG_G0[2] + 3}]", "g1a": SG_G1[0], "g1b": SG_G1[1], "g1c": SG_G1[2],
              "nls": SG_NLS, "ds0": SG_SO[0], "ds1": SG_SO[1], "ds2": SG_SO[2], "thr": SG_THR, "pt0": SG_PT[0],
              "pt1": SG_PT[1], "pt2": SG_PT[2]})  # fmt: skip
    return o


def loop_init(L, ops):
    """Tile 0's hh0 operands and NL/ND (stage 0), k = 0, the TDM descriptors' constant words."""
    tdm = ops["tdm"]
    # NL/ND of tile 0 = query pair qp_start: soffset sc[2] (pair 0) + 128 qp_start
    out = [f"s_lshl_b32 s98, {_sv(ops['sc'], 0)}, 7", f"s_add_co_i32 s98, {_sv(ops['sc'], 2)}, s98"]
    out += [t for _, t in rb_texts(L, ops["lq"], ops["lv"], 0, L.op_hh0, 0)] + nl_texts(L, dict(ops, nls="s98"))
    out.append(f"s_mov_b32 {SG_K}, 0")
    for o in range(3):
        g1 = int(SG_G1[o][2:].split(":")[0])
        words = [hex(TDM_W0[o]), "0xffff0000", "0x107fff", hex(TDM_W3[o]), "16", _sv(tdm, 12 if o == 0 else 13), "0", "0"]
        out += [f"s_mov_b32 s{g1 + i}, {w}" for i, w in enumerate(words)]
        out.append(f"s_mov_b32 s{SG_G0[o]}, 1")
    # the K / V fragments (INIT's 60 loads, older than tile 0's 8 NL / ND loads) complete before the loop: its blocks
    # then see only NL / ND loads in flight (no per-step loadcnt waits for the fragments)
    out.append("s_wait_loadcnt 0x8")
    return out


def loop_scalars(ops, stage, kofs=0):
    """The per-tile scalars of tile k + kofs (k in SG_K) for a body in ring stage `stage`."""
    tdm, sc, rc = ops["tdm"], ops["sc"], ops["rc"]
    qps, nm1 = _sv(sc, 0), "s99"
    out = [f"s_add_co_i32 {SG_QT}, {qps}, {SG_K}"] + ([f"s_add_co_i32 {SG_QT}, {SG_QT}, {kofs}"] if kofs else [])
    out += [f"s_add_co_i32 {nm1}, {_sv(tdm, 14)}, -1",
           f"s_add_co_i32 {SG_T1}, {SG_K}, {1 + kofs}", f"s_min_i32 {SG_T1}, {SG_T1}, {nm1}",
           f"s_add_co_i32 {SG_T2}, {SG_K}, {2 + kofs}", f"s_min_i32 {SG_T2}, {SG_T2}, {nm1}",
           f"s_add_co_i32 s98, {SG_T1}, {qps}", f"s_lshl_b32 s98, s98, 7", f"s_add_co_i32 {SG_NLS}, {_sv(sc, 2)}, s98"]  # fmt: skip
    so_tdm = ((stage + 2) % DEPTH) * QDO_B
    if TDM_DUMMY:  # tile k + kofs + 2 past the last: the three ops move 16 B each (words 3, 4: tile0 8, tile1 1)
        out += [f"s_add_co_i32 s98, {SG_K}, {2 + kofs}", f"s_cmp_ge_i32 s98, {_sv(tdm, 14)}",
                f"s_mov_b32 s98, {hex(8 << 16)}"]  # one literal per SALU op (s_mov keeps scc)
        for o in range(3):
            g1 = int(SG_G1[o][2:].split(":")[0])
            out += [f"s_cselect_b32 s{g1 + 3}, s98, {hex(TDM_W3[o])}", f"s_cselect_b32 s{g1 + 4}, 1, 16"]
    for o in range(3):
        g0 = SG_G0[o]
        out += [f"s_mul_i32 s96, {SG_T2}, {_sv(tdm, 4 * o + 2)}", "s_mov_b32 s97, 0",
                f"s_add_nc_u64 s[{g0 + 2}:{g0 + 3}], s[{int(_sv(tdm, 4 * o)[1:])}:{int(_sv(tdm, 4 * o)[1:]) + 1}], s[96:97]",
                f"s_bitset1_b32 s{g0 + 3}, 31",
                f"s_add_co_i32 s{g0 + 1}, {_sv(tdm, 4 * o + 3)}, {so_tdm}"]  # fmt: skip
    out += [f"s_lshl_b32 s98, {SG_QT}, 10"]
    for kh in range(NKV):
        out.append(f"s_add_co_i32 {SG_SO[kh]}, s98, {_sv(rc, kh)}")
    out += [f"s_lshl_b32 s98, {SG_QT}, 5", f"s_add_co_i32 {SG_THR}, s98, {_sv(sc, 3)}"]
    for kh in range(NKV):
        out += [f"s_cmp_ge_i32 {SG_QT}, {_sv(sc, 4 + kh)}", f"s_cselect_b32 {SG_PT[kh]}, 0, 0x40000000"]
    return out


def region_loop(L, ops, mode="list", probe=False, lp=".LHKG"):
    """(init texts, blocks, model cycles): blocks = [(name, label, texts, internal successors)] in layout order; the
    header is H0 (entered from the init code and by the back edge from N2); exits (k >= N) go to label {lp}_X."""
    lo = loop_ops(ops)
    tdm = ops["tdm"]
    blocks, cyc = [], {}
    if SCAL_IN_BODY:
        # H0 U0 N0 H1 U1 N1 H2 U2 N2 M0 M1 M2: a full tile falls through header -> body -> counter; the bodies compute
        # the next tile's scalars, so a header is only the masked / full branch
        mblocks = []
        for s in range(DEPTH):
            blocks.append((f"H{s}", f"{lp}_H{s}", [f"s_cmp_lt_i32 {SG_K}, {_sv(tdm, 15)}", f"s_cbranch_scc1 {lp}_M{s}"],
                           [f"U{s}", f"M{s}"]))  # fmt: skip
            nxt = loop_scalars(ops, (s + 1) % DEPTH, 1)
            tu, cyc[f"U{s}"] = region_step(L, lo, False, s, mode, probe, nxt)
            blocks.append((f"U{s}", f"{lp}_U{s}", tu, [f"N{s}"]))
            if s < DEPTH - 1:
                blocks.append((f"N{s}", f"{lp}_N{s}", [f"s_add_co_i32 {SG_K}, {SG_K}, 1",
                               f"s_cmp_ge_i32 {SG_K}, {_sv(tdm, 14)}", f"s_cbranch_scc1 {lp}_X"], ["X", f"H{s + 1}"]))  # fmt: skip
            else:
                blocks.append((f"N{s}", f"{lp}_N{s}", [f"s_add_co_i32 {SG_K}, {SG_K}, 1",
                               f"s_cmp_lt_i32 {SG_K}, {_sv(tdm, 14)}", f"s_cbranch_scc1 {lp}_H0", f"s_branch {lp}_X"],
                               ["H0", "X"]))  # fmt: skip
            tm, cyc[f"M{s}"] = region_step(L, lo, True, s, mode, probe, nxt)
            mblocks.append((f"M{s}", f"{lp}_M{s}", tm + [f"s_branch {lp}_N{s}"], [f"N{s}"]))
        return loop_init(L, ops) + loop_scalars(ops, 0, 0), blocks + mblocks, cyc
    for s in range(DEPTH):
        hdr = []
        if s == 0:
            name_h = "H0"
        else:
            name_h = f"N{s - 1}"
            hdr = [f"s_add_co_i32 {SG_K}, {SG_K}, 1", f"s_cmp_ge_i32 {SG_K}, {_sv(tdm, 14)}",
                   f"s_cbranch_scc1 {lp}_X"]  # fmt: skip
        sc_txt = loop_scalars(ops, s) + [f"s_cmp_lt_i32 {SG_K}, {_sv(tdm, 15)}", f"s_cbranch_scc1 {lp}_M{s}"]
        if hdr:  # the exit test ends a block of its own
            blocks.append((name_h, f"{lp}_N{s - 1}", hdr, ["X", f"H{s}"]))
        blocks.append((f"H{s}", f"{lp}_H{s}", sc_txt, [f"U{s}", f"M{s}"]))
        tu, cyc[f"U{s}"] = region_step(L, lo, False, s, mode, probe)
        blocks.append((f"U{s}", f"{lp}_U{s}", tu + [f"s_branch {lp}_N{s}"], [f"N{s}"]))
        tm, cyc[f"M{s}"] = region_step(L, lo, True, s, mode, probe)
        blocks.append((f"M{s}", f"{lp}_M{s}", tm, [f"N{s}"]))
    blocks.append(("N2", f"{lp}_N2", [f"s_add_co_i32 {SG_K}, {SG_K}, 1", f"s_cmp_lt_i32 {SG_K}, {_sv(tdm, 14)}",
                                      f"s_cbranch_scc1 {lp}_H0"], ["H0", "X"]))  # fmt: skip
    return loop_init(L, ops), blocks, cyc
