###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwdgen -- the generated regions of k_dkdv96ds (HK MLA backward: dK, dV and the dS^T scratch).

The shell (hip/src/k_dkdv96ds.hip, -DHKGEN=1) keeps the workgroup set-up, the first two TDM tiles and the barriers
before and after the tile loop, and marks three regions per block height (nkv = 3: the 96-row blocks, nkv = 2: the
64-row tail; 16 nkv kv rows per wave). build.py reads the registers LLVM gave the marker operands; this module returns
the instructions of every region:

  INIT   (c1, c1), (1 / scale, 1 / scale); the wave's K / V rows by TDM into its LDS staging area, the accumulators
         cleared while they are in flight, then the K / V B-operand fragments read from there
  LOOP   tile 0's hh0 S / dP operands and NL / ND, then every query tile k = 0 .. N-1 (the first n_mask diagonal:
         causal mask on, the dS^T stores of unattended blocks dropped) in a loop unrolled by the depth of the Q / dO
         TDM ring (3 stages, tile k in stage k mod 3, prefetch distance 2). One tile:
           TDM of tile k+2
           S = Q K^T and dP = dO V^T of hh0 (dt outer, operands loaded during the previous tile); hh1's operands
             streamed into the tuples hh0's WMMAs release; S / dP of hh1 (kh-major)
           softmax / dS in packed f32, bf16 P / dS written straight into the dV / dK B operands
           dV^T += dO^T P, dK^T += Q^T dS (transposed A operands by ds_load_tr16_b128)
           the dS^T stores of tile k, NL / ND of tile k+1, the scalars of tile k+1
           s_wait_tensorcnt 3 + s_barrier_signal after the last LDS read of stage k, s_barrier_wait, then tile k+1's
             hh0 operands from its stage
  FINAL  dV * (1 / scale) and dK to bf16 row-major images in LDS (the shell's epi_images layout), whole-line
         readback and global stores, software-pipelined

Indices: a query tile is 32 queries in two halves hh of 16; a wave's kv rows are NKV sub-tiles kh of 16; dt is a WMMA
k-step (32 columns of D), d a 16-column output tile of dV^T / dK^T.

Numerics: NL = log2(scale) - log2(e) lse and ND = -delta are k_delta's per-query constants. S chains over dt ascending from +0, dP chains from ND; exp2(c1 S + NL) with c1 = scale log2(e) is
scale * P, so dS = P dP takes one multiply (scale included) and FINAL scales dV by 1 / scale. Packed f32 fma / mul,
exp2, RNE bf16 conversions; every dV / dK accumulator takes one WMMA per tile, in tile order.

The node lists of a tile are in program order, which is the semantics; schedule_slot reorders them along the
dependence graph (asmkit's parser: RAW / WAR / WAW on VGPRs, SGPRs, scc, vcc) plus order edges for the ring barrier
and the XCNT spacing. The output is real instructions only: asmkit.emit adds s_set_vgpr_msb, the expert-mode waits
and the hazard v_nops."""

import re

from asmkit.emit import make_item

NKV = 3  # kv sub-tiles of 16 rows per wave (set_nkv)
NDT_QK, NDT_V = 6, 4  # WMMA k-steps of S = Q K^T (D_QK 192) and dP = dO V^T (D_V 128)
NDO_V, NDO_QK = 8, 12  # 16-wide output tiles of dV^T (D_V) and dK^T (D_QK)
XK, XV = 400, 272  # LDS row strides (bytes) of Q / K and of dO / V rows (the TDM pads 16 B per row)
QOFF = 32 * XV  # the Q tile inside a ring stage, after the [32][D_V] dO tile
QDO_B = 32 * (XV + XK)  # one ring stage
DEPTH = 3  # ring stages
# dS^T stores (dslayout v2): dwords 4..7 of a dS operand (queries 16..31) go 512 B after dwords 0..3
DS_ST_B = 512
NEG = "0xff61b1e6"  # fp32 -3e38 (K1_NEG): a masked logit, exp2 -> +0
# the three TDM ops of a ring stage: 0 dO (V in INIT), 1 Q (K) cols [0, 128), 2 Q (K) cols [128, 192); word 0 of
# their group-1 descriptors (k1_tdm_w0: 2-B elements, LDS row padding), tile columns, tile rows per wave
TDM_W0 = (0x07510000, 0x47510000, 0x87110000)
TDM_WIDTH = (128, 128, 64)
TDM_ROWS = 16
# the next tile's scalars redefine SGPRs that in-flight VMEM / TDM ops read (an XCNT wait): keep them this many
# cycles behind those ops so the wait finds them done
SCAL_XCNT_GAP = 160
FINAL_LAG = 8  # FINAL: a global store trails its readback load by this many loads

# SGPRs of the LOOP region (its marker clobbers s48..s103): TDM descriptors per ring op (group 1: 8 SGPRs, group 0:
# 4), the tile counter k and the per-tile scalars; s96..s99 are temporaries
SG_G1 = (48, 56, 64)
SG_G0 = (72, 76, 80)
SG_K, SG_QT, SG_T1, SG_T2, SG_NLS = "s84", "s85", "s86", "s87", "s88"
SG_SO = ("s89", "s90", "s91")
SG_THR = "s92"
SG_PT = ("s93", "s94", "s95")


def set_nkv(n):
    """Generate for blocks of 16 n kv rows per wave."""
    global NKV
    NKV = n


def vr(r, n=1):
    """Operand text of the n VGPRs at true register r: the encoded low 8 bits, the true numbers in a comment."""
    if r // 256 != (r + n - 1) // 256:
        raise ValueError(f"tuple v[{r}:{r + n - 1}] crosses a 256-register block")
    lo = r % 256
    if n == 1:
        return f"v{lo}" if r < 256 else f"v{lo} /*v{r}*/"
    return f"v[{lo}:{lo + n - 1}]" if r < 256 else f"v[{lo}:{lo + n - 1}] /*v[{r}:{r + n - 1}]*/"


def offset(off):
    """The offset modifier of a memory instruction (none for 0)."""
    return f" offset:{off}" if off else ""


def wmma(d, a, b, c):
    """D = A B + C (v_wmma_f32_16x16x32_bf16) on 8-VGPR tuples; c None: from +0."""
    return f"v_wmma_f32_16x16x32_bf16 {vr(d, 8)}, {vr(a, 8)}, {vr(b, 8)}, {'0' if c is None else vr(c, 8)}"


def sreg(op, i):
    """SGPR i of a tuple operand s[a:b]."""
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", op)
    if not m or int(m.group(1)) + i > int(m.group(2)):
        raise ValueError(f"{op}: no element {i}")
    return f"s{int(m.group(1)) + i}"


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
      PRE[e]    NL / ND quads, e = 4 hh + {0, 1}: NL, 4 hh + {2, 3}: ND (one 8-VGPR tuple: a dP chain's C init)
      acc       dV (kh, d) at acc[8 kh + d], dK (kh, d) at acc[8 NKV + 12 kh + d]
      kf, vf    K / V fragments [kh][dt]
      cc1, ccs  (c1, c1) and (1 / scale, 1 / scale)
      T         mask scratch (2) and a masked tile's dS^T voffsets (NKV)
      spare     4 VGPRs that INIT zeroes and nothing else uses (kept: the pinned code object has them)"""

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
        self.spare = [take(1, 1) for _ in range(4)]
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
# dependence graph and slot scheduler
# --------------------------------------------------------------------------------------------------------------
# result latencies (cycles); DS_LD only for the critical path: the scheduler models the LDS queue instead
LAT = {
    "WMMA": 20,
    "VALU": 4,
    "TRANS": 8,
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
# issued since): keep it WAR_XDL cycles and at least MIN_XDL_AFTER WMMAs behind the reader, so that the wait is free
WAR_XDL, MIN_XDL_AFTER = 24, 3
DS_MAX = 56  # LDS loads in flight
# issue model (4 waves per WGP): a WMMA occupies the XDL pipe 8 cycles; inside its window a VALU / TRANS issues only
# at the offsets VALU_SLOTS, a memory op only at MEM_SLOTS, anything else at any offset but 0; one LDS op per LDS_RATE
# cycles, completing in order LDS_LAT cycles after issue; s_barrier_wait at least BAR_GAP WMMAs after the signal (the
# other wave's skew)
VALU_SLOTS, MEM_SLOTS = (2, 3, 6, 7), (1, 4, 5)
LDS_LAT, LDS_RATE = 70, 8
BAR_GAP = 12
MEM = ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM")


class Node:
    """One instruction of a tile: asmkit kind, VGPR / SGPR defs and uses, [(pred, latency)] edges, and the WMMAs whose
    operands it overwrites (war_xdl)."""

    __slots__ = ("text", "kind", "vdefs", "vuses", "sdefs", "suses", "preds", "war_xdl")


def mk(text):
    """A node of one instruction; defs / uses from asmkit's parser (the model the emitter and the checks use)."""
    it = make_item("\t" + text)
    n = Node()
    n.text, n.kind = text, it.kind
    n.vdefs, n.vuses = set(it.vdefs), set(it.vuses)
    n.sdefs, n.suses = set(it.sdefs), set(it.suses)
    n.preds, n.war_xdl = [], set()
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


def add_order(nodes, a, b, lat=1):
    """An order edge a -> b (a before b in list order) of at least lat cycles."""
    if not a < b:
        raise ValueError("order edges must point forward in program order")
    d = dict(nodes[b].preds)
    d[a] = max(d.get(a, 0), lat)
    nodes[b].preds = sorted(d.items())


def schedule_slot(nodes, wseq):
    """(ordered nodes, modelled cycles): a cycle-stepped list scheduler around the in-order WMMA stream wseq, one
    instruction per cycle. A WMMA issues once the previous one left the XDL pipe (8 cycles); the slots inside its window
    and the LDS queue follow the issue model above; a TRANS blocks VALU / TRANS the next cycle; s_barrier_signal (its
    s_wait_dscnt 0) waits until every LDS op issued so far completed; at most DS_MAX LDS loads are in flight (a consumer
    of a load retires it and every older one). Result latencies LAT, WMMA_CHAIN for a WMMA chain. Ties: barrier and
    wait nodes first, then the earliest deadline (the first WMMA that needs the node, transitively), the longest
    critical path, program order."""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p, l in nd.preds:
            succ[p].append((i, l))
    wpos = {i: k for k, i in enumerate(wseq)}
    dl = [10**6] * n
    for i in range(n - 1, -1, -1):
        for s_, _ in succ[i]:
            dl[i] = min(dl[i], wpos[s_] if s_ in wpos else dl[s_])
    cp = [0] * n
    for i in range(n - 1, -1, -1):
        cp[i] = (8 if nodes[i].kind == "WMMA" else 1) + max((l + cp[s] for s, l in succ[i]), default=0)
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    out, wmma_at = [], {}
    t, last_w, wk, trans_at, ds_next, ds_last_done = 0, -100, 0, -100, 0, 0
    ds_issued, ds_retired, ds_seq = 0, -1, {}
    bs_wk = 10**6  # WMMAs issued before the barrier signal

    def xdl_ok(i):
        rs = nodes[i].war_xdl
        if not rs:
            return True
        if not all(r in wmma_at for r in rs):
            return False
        return wk - 1 - max(wmma_at[r] for r in rs) >= MIN_XDL_AFTER or wk >= len(wseq)

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
        for s_, l in succ[pick]:
            if k == "DS_LD" and nodes[s_].vuses & nd.vdefs:
                lat = done_at - t
            elif k == "WMMA":
                lat = WMMA_CHAIN if nodes[s_].kind == "WMMA" and (nodes[s_].vuses & nd.vdefs) else max(l, 1)
            else:
                lat = max(l, LAT[k]) if (nodes[s_].vuses & nd.vdefs or nodes[s_].suses & nd.sdefs) else l
            earliest[s_] = max(earliest[s_], t + lat)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
        t += 1
    return out, max(t, last_w + 8)


# --------------------------------------------------------------------------------------------------------------
# INIT
# --------------------------------------------------------------------------------------------------------------
def tdm_words(o, rows):
    """Words 0..4 of the group-1 descriptor of TDM op o (the shell's g1 layout): data size and LDS row padding,
    tensor extents (inner 0x7FFFFFFF: never clamps, outer `rows`), tile TDM_WIDTH[o] x `rows`. Words 5, 6 hold the
    row stride (elements), word 7 is 0."""
    return [hex(TDM_W0[o]), "0xffff0000", hex((rows << 16) | 0x7FFF), hex(TDM_WIDTH[o] << 16), str(rows)]


def region_init(L, ops):
    """INIT: (c1, c1), (1 / scale, 1 / scale) (FINAL's dV factor); the wave's K / V rows by TDM into its staging area
    ([16 NKV][XV] of V, then [16 NKV][XK] of K; the ops as a ring stage's: V cols [0, 128), K cols [0, 128) and
    [128, 192)); the accumulators cleared while they are in flight; then the K / V fragments (lane bases lk / lv +
    kh * 16 rows + dt * 64 + u * 32). The descriptors use s92..s103, one op at a time: INIT may clobber only
    s90..s103."""
    out = [f"v_mov_b32 {vr(L.cc1)}, {ops['c1']}", f"v_mov_b32 {vr(L.cc1 + 1)}, {ops['c1']}"]
    out += [f"v_rcp_f32 {vr(L.ccs)}, {ops['sc']}", f"v_mov_b32 {vr(L.ccs + 1)}, {vr(L.ccs)}"]
    rows = 16 * NKV
    g1, g0 = 96, 92
    for o in range(3):
        ga = ops["vg"] if o == 0 else ops["kg"]
        out += [f"s_mov_b32 s{g1 + i}, {w}" for i, w in enumerate(tdm_words(o, rows))]
        out += [
            f"s_mul_i32 s{g1 + 5}, {ops['hkv']}, {128 if o == 0 else 192}",
            f"s_mov_b32 s{g1 + 6}, 0",
            f"s_mov_b32 s{g1 + 7}, 0",
            f"s_mov_b32 s{g0}, 1",
            f"s_add_co_i32 s{g0 + 1}, {ops['kvl']}, {0 if o == 0 else rows * XV + (256 if o == 2 else 0)}",
            f"s_add_co_u32 s{g0 + 2}, {sreg(ga, 0)}, {256 if o == 2 else 0}",
            f"s_add_co_ci_u32 s{g0 + 3}, {sreg(ga, 1)}, 0",
            f"s_bitset1_b32 s{g0 + 3}, 31",
            f"tensor_load_to_lds s[{g0}:{g0 + 3}], s[{g1}:{g1 + 7}]",
        ]
    for t in L.acc:
        for q in range(0, 8, 2):
            out.append(f"v_mov_b64 {vr(t + q, 2)}, 0")
    out += [f"v_mov_b32 {vr(r)}, 0" for r in L.spare]
    out.append("s_wait_tensorcnt 0x0")
    for kh in range(NKV):
        for frag, lane, x, ndt in ((L.kf, ops["lk"], XK, NDT_QK), (L.vf, ops["lv"], XV, NDT_V)):
            for dt in range(ndt):
                for u in range(2):
                    off = offset(kh * 16 * x + dt * 64 + u * 32)
                    out.append(f"ds_load_b128 {vr(frag[kh][dt] + 4 * u, 4)}, {lane}{off}")
    return out


# --------------------------------------------------------------------------------------------------------------
# one tile
# --------------------------------------------------------------------------------------------------------------
def sdp_loads(L, qbase, vbase, hh, stage):
    """[((Q / D, dt, u), text)] of the 20 ds_load_b128 of hh half's S / dP A operands from ring stage `stage`
    (k1_rd_const offsets from the lane bases qbase / vbase, the stage offset in the immediate)."""
    out = []
    so = stage * QDO_B
    for dt in range(NDT_QK):
        for u in range(2):
            off = offset(so + QOFF + hh * 16 * XK + dt * 64 + u * 32)
            out.append((("Q", dt, u), f"ds_load_b128 {vr(L.sdp_a(1, dt) + 4 * u, 4)}, {qbase}{off}"))
        for u in range(2 if dt < NDT_V else 0):
            off = offset(so + hh * 16 * XV + dt * 64 + u * 32)
            out.append((("D", dt, u), f"ds_load_b128 {vr(L.sdp_a(0, dt) + 4 * u, 4)}, {vbase}{off}"))
    return out


def nl_loads(L, ops):
    """The 8 buffer_load_b128 of a tile's NL / ND quads (k1_nl_sconst offsets, soffset ops['nls'])."""
    out = []
    for e in range(8):
        r = ops["rnl"] if (e % 4) // 2 == 0 else ops["rnd"]
        off = offset((e // 4) * 64 + (e % 2) * 16)
        out.append(f"buffer_load_b128 {vr(L.PRE[e], 4)}, {ops['nlv']}, {r}, {ops['nls']} offen{off}")
    return out


def softmax_unit(L, hh, kh, add, mask=None):
    """Node indices of the softmax / dS of (hh, kh): 4 element pairs into dwords 4 hh + j of P[kh] / DS[kh]. mask: the
    VGPR holding dl - thr - 16 (hh - kh); element i is masked iff mask > i."""
    nodes = []
    s_, d_ = L.raw(hh, kh, 0), L.raw(hh, kh, 1)
    for j in range(4):
        nl = L.PRE[hh * 4 + j // 2] + 2 * (j % 2)
        s2, d2 = s_ + 2 * j, d_ + 2 * j
        nodes.append(add(f"v_pk_fma_f32 {vr(s2, 2)}, {vr(s2, 2)}, {vr(L.cc1, 2)}, {vr(nl, 2)}"))
        if mask is not None:
            for e in range(2):
                nodes.append(add(f"v_cmp_ge_i32 vcc_lo, {2 * j + e}, {vr(mask)}"))
                nodes.append(add(f"v_cndmask_b32 {vr(s2 + e)}, {NEG}, {vr(s2 + e)}, vcc_lo"))
        nodes.append(add(f"v_exp_f32 {vr(s2)}, {vr(s2)}"))
        nodes.append(add(f"v_exp_f32 {vr(s2 + 1)}, {vr(s2 + 1)}"))
        nodes.append(add(f"v_pk_mul_f32 {vr(d2, 2)}, {vr(s2, 2)}, {vr(d2, 2)}"))
        w = hh * 4 + j
        nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(L.P[kh] + w)}, {vr(s2)}, {vr(s2 + 1)}"))
        nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(L.DS[kh] + w)}, {vr(d2)}, {vr(d2 + 1)}"))
    return nodes


def hh1_order():
    """hh1's S / dP WMMAs, kh-major: kh 0's chains finish first, so its softmax (and so P[0], the first dV operand)
    hides under the later kh's WMMAs. The S and dP chains of one kh interleave; a chain's next WMMA stands >= 2
    positions after its previous one (16 cycles: WMMA_CHAIN); lower kh first, then the chain with more WMMAs left.
    Items (kind 'S' / 'P', kh, dt)."""
    chains = []
    for kh in range(NKV):
        chains.append([("S", kh, dt) for dt in range(NDT_QK)])
        chains.append([("P", kh, dt) for dt in range(NDT_V)])
    out, last, nxt = [], [-10] * len(chains), [0] * len(chains)
    while len(out) < sum(len(c) for c in chains):
        p = len(out)
        live = [c for c in range(len(chains)) if nxt[c] < len(chains[c])]
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

    so = stage * QDO_B
    w_sdp0, w_sdp1, w_dv, w_dk = [], [], [], []
    stage_reads = []  # every LDS read of stage k: before the ring barrier
    # TDM of tile k+2 into the stage tile k-1 used (both waves' last reads of it preceded the previous barrier); it
    # issues before the tensorcnt wait below, which then retires tile k+1 (in flight: k+1, k+2)
    tdm = [add(f"tensor_load_to_lds {ops['g0' + c]}, {ops['g1' + c]}") for c in "abc"]

    def wmma_sdp(hh, kind, kh, dt):
        if kind == "S":  # S = Q K^T from +0
            s_ = L.raw(hh, kh, 0)
            return add(wmma(s_, L.sdp_a(1, dt), L.kf[kh][dt], None if dt == 0 else s_))
        d_ = L.raw(hh, kh, 1)  # dP = dO V^T from ND = -delta
        return add(wmma(d_, L.sdp_a(0, dt), L.vf[kh][dt], L.PRE[hh * 4 + 2] if dt == 0 else d_))

    # S / dP of hh0, dt outer (S of every kh, then dP: consecutive WMMAs share the A operand); hh1's operands of dt
    # stream into the tuples hh0's dt just released
    hh1 = dict(sdp_loads(L, ops["lq"], ops["lv"], 1, stage))
    for dt in range(NDT_QK):
        for kind in ("S", "P") if dt < NDT_V else ("S",):
            for kh in range(NKV):
                w_sdp0.append(wmma_sdp(0, kind, kh, dt))
        for u in range(2):
            stage_reads.append(add(hh1[("Q", dt, u)]))
            if dt < NDT_V:
                stage_reads.append(add(hh1[("D", dt, u)]))
    # dO^T operands (their own tuples, free since the previous tile's dV)
    for d in range(NDO_V):
        for p in range(2):
            off = offset(so + d * 32 + p * 16 * XV)
            stage_reads.append(add(f"ds_load_tr16_b128 {vr(L.dv_a(d) + 4 * p, 4)}, {ops['ltv']}{off}"))
    for kind, kh, dt in hh1_order():
        w_sdp1.append(wmma_sdp(1, kind, kh, dt))
    # softmax / dS, hh0 then hh1, kh by kh
    soft = []
    for hh in range(2):
        for kh in range(NKV):
            mask = None
            if masked:
                thr = "s96" if hh == 0 else "s97"
                add(f"s_add_co_i32 {thr}, {ops['thr']}, {16 * (hh - kh)}")
                add(f"v_subrev_nc_u32 {vr(L.T[hh])}, {thr}, {ops['dl']}")
                mask = L.T[hh]
            soft += softmax_unit(L, hh, kh, add, mask)

    def q_t_loads(d):
        for p in range(2):
            off = offset(so + QOFF + d * 32 + p * 16 * XK)
            stage_reads.append(add(f"ds_load_tr16_b128 {vr(L.dk_a(d) + 4 * p, 4)}, {ops['ltk']}{off}"))

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
    # (dropped: dslayout does not store it)
    ex = []
    for kh in range(NKV):
        vd = ops["vds"]
        if masked:
            ex.append(add(f"v_add_nc_u32 {vr(L.T[2 + kh])}, {ops['pt' + str(kh)]}, {ops['vds']}"))
            vd = vr(L.T[2 + kh])
        for h in range(2):
            st = f"buffer_store_b128 {vr(L.DS[kh] + 4 * h, 4)}, {vd}, {ops['rds']}, {ops['ds' + str(kh)]} offen"
            ex.append(add(st + offset(h * DS_ST_B)))
    # NL / ND of tile k+1 (PRE is dead after the softmax)
    nl = [add(t) for t in nl_loads(L, ops)]
    # the ring barrier after the last LDS read of stage k (tensorcnt 3: tile k+1 landed), dK (d outer), then the next
    # tile's hh0 operands into O as dK releases Q^T 0..9 (after the barrier wait)
    tw = add("s_wait_tensorcnt 0x3")
    bs = add("s_barrier_signal -1")
    bw = add("s_barrier_wait -1")
    for d in range(NDO_QK):
        for kh in range(NKV):
            w_dk.append(add(wmma(L.dk(kh, d), L.dk_a(d), L.DS[kh], L.dk(kh, d))))
    nxt_reads = [add(t) for _, t in sdp_loads(L, ops["lq"], ops["lv"], 0, (stage + 1) % DEPTH)]
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
    for i in nxt_reads:
        add_order(nodes, bw, i)
    return nodes, w_sdp0 + w_sdp1 + w_dv + w_dk


def region_step(L, ops, masked, stage, scal):
    """(instruction texts, modelled cycles) of one tile."""
    nodes, wseq = step_nodes(L, ops, masked, stage, scal)
    order, cycles = schedule_slot(nodes, wseq)
    return [n.text for n in order], cycles


# --------------------------------------------------------------------------------------------------------------
# LOOP: tile 0's operands, then the tile loop (unrolled by the ring depth) with its scalar work
# --------------------------------------------------------------------------------------------------------------
def loop_ops(ops):
    """The marker operands plus the names the tile bodies use for the loop's SGPRs."""
    o = dict(ops)
    for c, g0, g1 in zip("abc", SG_G0, SG_G1):
        o["g0" + c], o["g1" + c] = f"s[{g0}:{g0 + 3}]", f"s[{g1}:{g1 + 7}]"
    for kh, (so, pt) in enumerate(zip(SG_SO, SG_PT)):
        o["ds" + str(kh)], o["pt" + str(kh)] = so, pt
    o["nls"], o["thr"] = SG_NLS, SG_THR
    return o


def loop_init(L, ops):
    """Tile 0's hh0 operands (stage 0) and NL / ND, k = 0, the loop-invariant words of the TDM descriptors."""
    tdm, sc = ops["tdm"], ops["sc"]
    # NL / ND soffset of tile 0: query pair 0's (sc[2]) + 128 qp_start (sc[0])
    out = [f"s_lshl_b32 s98, {sreg(sc, 0)}, 7", f"s_add_co_i32 s98, {sreg(sc, 2)}, s98"]
    out += [t for _, t in sdp_loads(L, ops["lq"], ops["lv"], 0, 0)] + nl_loads(L, dict(ops, nls="s98"))
    out.append(f"s_mov_b32 {SG_K}, 0")
    for o in range(3):
        words = tdm_words(o, TDM_ROWS) + [sreg(tdm, 12 if o == 0 else 13), "0", "0"]
        out += [f"s_mov_b32 s{SG_G1[o] + i}, {w}" for i, w in enumerate(words)]
        out.append(f"s_mov_b32 s{SG_G0[o]}, 1")
    # every VMEM load older than tile 0's 8 NL / ND loads completes before the loop: its blocks then wait only for
    # NL / ND loads
    out.append("s_wait_loadcnt 0x8")
    return out


def loop_scalars(ops, stage, kofs):
    """The scalars of tile j = k + kofs (k in SG_K), which runs in ring stage `stage`: its query pair qt, the NL / ND
    soffset of tile j+1, the TDM descriptors of tile j+2 (both clamped to tile N - 1), the dS^T soffsets, the mask
    threshold and the stored-block flags of tile j."""
    tdm, sc, rc = ops["tdm"], ops["sc"], ops["rc"]
    qps, n, nm1 = sreg(sc, 0), sreg(tdm, 14), "s99"
    out = [f"s_add_co_i32 {SG_QT}, {qps}, {SG_K}"]
    if kofs:
        out.append(f"s_add_co_i32 {SG_QT}, {SG_QT}, {kofs}")
    out += [
        f"s_add_co_i32 {nm1}, {n}, -1",
        f"s_add_co_i32 {SG_T1}, {SG_K}, {1 + kofs}",
        f"s_min_i32 {SG_T1}, {SG_T1}, {nm1}",
        f"s_add_co_i32 {SG_T2}, {SG_K}, {2 + kofs}",
        f"s_min_i32 {SG_T2}, {SG_T2}, {nm1}",
        f"s_add_co_i32 s98, {SG_T1}, {qps}",
        "s_lshl_b32 s98, s98, 7",
        f"s_add_co_i32 {SG_NLS}, {sreg(sc, 2)}, s98",
    ]
    # a TDM tile past the last moves 16 B per op (tile 8 x 1) instead of loading tile N - 1 again: no wasted traffic,
    # and the epilogue's tensorcnt-0 wait finds the ring settled (one literal per SALU op; s_mov keeps scc)
    out += [
        f"s_add_co_i32 s98, {SG_K}, {2 + kofs}",
        f"s_cmp_ge_i32 s98, {n}",
        f"s_mov_b32 s98, {hex(8 << 16)}",
    ]
    for o in range(3):
        width = hex(TDM_WIDTH[o] << 16)
        out += [
            f"s_cselect_b32 s{SG_G1[o] + 3}, s98, {width}",
            f"s_cselect_b32 s{SG_G1[o] + 4}, 1, {TDM_ROWS}",
        ]
    # group 0: global address = tile 0's + tile * bytes per tile, LDS address of the tile's stage
    so_tdm = ((stage + 2) % DEPTH) * QDO_B
    for o in range(3):
        g0, ga = SG_G0[o], int(sreg(tdm, 4 * o)[1:])
        out += [
            f"s_mul_i32 s96, {SG_T2}, {sreg(tdm, 4 * o + 2)}",
            "s_mov_b32 s97, 0",
            f"s_add_nc_u64 s[{g0 + 2}:{g0 + 3}], s[{ga}:{ga + 1}], s[96:97]",
            f"s_bitset1_b32 s{g0 + 3}, 31",
            f"s_add_co_i32 s{g0 + 1}, {sreg(tdm, 4 * o + 3)}, {so_tdm}",
        ]
    # dS^T soffset of kh: 1024 qt + rc[kh]
    out += [f"s_lshl_b32 s98, {SG_QT}, 10"]
    for kh in range(NKV):
        out.append(f"s_add_co_i32 {SG_SO[kh]}, s98, {sreg(rc, kh)}")
    # mask threshold: thr0 + 32 qt; kh's dS^T block is stored iff qt >= its first stored pair (else voffset 2^30)
    out += [f"s_lshl_b32 s98, {SG_QT}, 5", f"s_add_co_i32 {SG_THR}, s98, {sreg(sc, 3)}"]
    for kh in range(NKV):
        out += [f"s_cmp_ge_i32 {SG_QT}, {sreg(sc, 4 + kh)}", f"s_cselect_b32 {SG_PT[kh]}, 0, 0x40000000"]
    return out


def region_loop(L, ops, lp):
    """(prologue texts, blocks, modelled cycles per body); blocks = [(name, label, texts, successors in the region)]
    in layout order H0 U0 N0 H1 U1 N1 H2 U2 N2 M0 M1 M2. Header H<s> branches to the masked body M<s> while k < n_mask
    and falls through into the full body U<s>; N<s> counts k and leaves for {lp}_X ("X") once k = N; M<s> is out of
    line and returns to N<s>. A body computes the next tile's scalars, so the prologue computes tile 0's."""
    lo = loop_ops(ops)
    n, n_mask = sreg(ops["tdm"], 14), sreg(ops["tdm"], 15)
    blocks, mblocks, cyc = [], [], {}
    for s in range(DEPTH):
        nxt = loop_scalars(ops, (s + 1) % DEPTH, 1)
        tu, cyc[f"U{s}"] = region_step(L, lo, False, s, nxt)
        tm, cyc[f"M{s}"] = region_step(L, lo, True, s, nxt)
        head = [f"s_cmp_lt_i32 {SG_K}, {n_mask}", f"s_cbranch_scc1 {lp}_M{s}"]
        count = [f"s_add_co_i32 {SG_K}, {SG_K}, 1"]
        if s < DEPTH - 1:
            count += [f"s_cmp_ge_i32 {SG_K}, {n}", f"s_cbranch_scc1 {lp}_X"]
            count_succ = ["X", f"H{s + 1}"]
        else:
            count += [f"s_cmp_lt_i32 {SG_K}, {n}", f"s_cbranch_scc1 {lp}_H0", f"s_branch {lp}_X"]
            count_succ = ["H0", "X"]
        blocks += [
            (f"H{s}", head, [f"U{s}", f"M{s}"]),
            (f"U{s}", tu, [f"N{s}"]),
            (f"N{s}", count, count_succ),
        ]
        mblocks.append((f"M{s}", tm + [f"s_branch {lp}_N{s}"], [f"N{s}"]))
    blocks = [(name, f"{lp}_{name}", texts, succ) for name, texts, succ in blocks + mblocks]
    return loop_init(L, ops) + loop_scalars(ops, 0, 0), blocks, cyc


# --------------------------------------------------------------------------------------------------------------
# FINAL
# --------------------------------------------------------------------------------------------------------------
def region_final(L, ops):
    """FINAL, in program order: dV * (1 / scale) and dK to bf16 images in LDS (k3_epi_wr: lane bases ev / ek +
    kh * 16 rows + d * 32), whole-line readback (k3_epi_rd: lane bases rv / rk + rb * 4 rows + li * 128) and global
    stores (voffset gv / gk, soffset sov / sok + rb * rsv / rsk, offset li * 128). The conversions run group by group,
    (V, kh) then (K, kh), each image store one conversion after its own; a group's readback rows (rb = 4 kh ..
    4 kh + 3) become loadable after its last image store (a wave's LDS ops complete in order), one ds_load_b128 per
    conversion, each global store FINAL_LAG pairs after its load. Row block rb's soffset is in one of 4 rotating
    SGPRs per image, so no SGPR an in-flight store reads is redefined."""
    convs = []  # (group, VALU texts, image store text)
    pairs = {}  # group -> [(load text, texts before the store, store text)]
    tmp = L.O + L.R0 + L.R1  # readback buffers: dead after the loop
    k = 0
    for isk, x, n_d, base in ((0, XV, NDO_V, ops["ev"]), (1, XK, NDO_QK, ops["ek"])):
        nli, rbase, gv, so, rs, rsrc = (
            (2, ops["rv"], ops["gv"], ops["sov"], ops["rsv"], ops["rdv"]),
            (3, ops["rk"], ops["gk"], ops["sok"], ops["rsk"], ops["rdk"]),
        )[isk]
        sregs = ("s93", "s95", "s97", "s99") if isk else ("s92", "s94", "s96", "s98")
        for kh in range(NKV):
            for d in range(n_d):
                a = L.dk(kh, d) if isk else L.dv(kh, d)
                c = []
                if not isk:  # the softmax yields scale * P
                    c = [f"v_pk_mul_f32 {vr(a + q, 2)}, {vr(a + q, 2)}, {vr(L.ccs, 2)}" for q in (0, 2, 4, 6)]
                c += [
                    f"v_cvt_pk_bf16_f32 {vr(a + q)}, {vr(a + 2 * q)}, {vr(a + 2 * q + 1)}" for q in range(4)
                ]
                convs.append(
                    ((isk, kh), c, f"ds_store_b128 {base}, {vr(a, 4)}{offset(kh * 16 * x + d * 32)}")
                )
            lst = []
            for rb in range(4 * kh, 4 * kh + 4):
                sr, prev = sregs[rb % 4], sregs[(rb - 1) % 4]
                pre = [f"s_mov_b32 {sr}, {so}" if rb == 0 else f"s_add_co_i32 {sr}, {prev}, {rs}"]
                for li in range(nli):
                    t = tmp[k % len(tmp)] + 4 * ((k // len(tmp)) % 2)
                    k += 1
                    ld = f"ds_load_b128 {vr(t, 4)}, {rbase}{offset(rb * 4 * x + li * 128)}"
                    st = f"buffer_store_b128 {vr(t, 4)}, {gv}, {rsrc}, {sr} offen{offset(li * 128)}"
                    lst.append((ld, pre if li == 0 else [], st))
            pairs[(isk, kh)] = lst
    text, ready, flight = [], [], []

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
