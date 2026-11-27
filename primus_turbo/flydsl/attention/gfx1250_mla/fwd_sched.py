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

"""The generated regions of the gfx1250 MLA forward (fwd_kernel): one wave's tile math as straight-line schedules
on a static plan of the VGPRs v[BASE:1024] (Layout), each region one inline asm statement (region_asm). Ported from
the HipKittens MLA forward generator.

  INIT      a block's state: m, d, O, (c, c), the Q fragments of the wave's four q tiles (resident for the block)
            and the state that makes the first prefix a no-op (for a block whose tile 0 is a diagonal tile)
  INIT_C0   INIT and the block's tile 0 when that needs no mask, without a prefix (no previous tile); it issues
            the tile's ring slot (a TDM op)
  TILE_CL   the block's other unmasked tiles as one loop: per tile its TDM op, its math and its barrier
  TILE_Dk   diagonal tile k = 0..2: q tiles below k skipped, the causal mask on q tile k
  TILE_D3F  diagonal tile 3 and FINAL (whose prefix skips q tiles 0..2), the block's last barrier inside
  FINAL_A   the last tile's prefix, then out = bf16(O / d) and lse of the four q tiles into the staging slots
            (a block that ends before a diagonal tile 3)

The math is FlyDSL's forward op for op, so out / lse are bitwise FlyDSL's: per q tile, S = Q K^T over 6 k-steps from
+0, the mask (-inf), lm = the row max (any tree: max is exact), the rescale vote fma(lm, c, -m) > log2(e) over the
wave, m' = max(m, c lm) if it fires, corr = exp2(m - m'), p = exp2(fma(s, c, -m')), FlyDSL's 3-ary sum tree,
d = fma(corr, d, sum), O = O corr + bf16(p) V; c = scale log2(e).

The TILE regions are software-pipelined across tiles: the prefix of region t finishes tile t - 1 (the softmax of q
tiles 2, 3, then the PV WMMAs of all four), the body reads tile t's K / V, runs QK of the four q tiles and the softmax
of q tiles 0, 1; every q tile keeps its sequence of operations, so no result bit changes. O *= corr runs under a
wave-uniform branch per q-tile pair (a RESC node), only when m moved: corr is exactly 1 otherwise.

The node lists' program order is the semantics: schedule() reorders along the dependence graph on a cycle model of
one wave; asm.emit_block adds s_set_vgpr_msb, the expert-mode waits and the hazard v_nops."""

import functools
import re

from . import asm as A
from .asm import add_order, vr

LOG2E = 0x3FB8AA3B  # fp32(log2 e): the rescale threshold
NEG_INF = 0xFF800000
BIG_NEG = 0xF149F2CA  # fp32(-1e30): the initial m
LN2 = 0x3F317218  # fp32(1 / log2 e)
SEL_LO, SEL_HI = 0x76543210, 0xFEDCBA98  # v_permlanex16_b32 selects: lane l reads lane l of the other half

XK, XV = 400, 272  # LDS row strides of K and V (bytes)
QOFF1 = 64 * XK  # q tile 1 (3) rows are 64 rows after q tile 0 (2)
QT, NKV, NKS, NDT = 4, 4, 6, 8  # q tiles per wave, kv16 tiles per tile, QK k-steps, PV d tiles

# Scalar temporaries, among the SGPRs every region declares clobbered (CLOBBER_S; plus vcc and scc)
S_SEL = "s90"  # SEL_LO for v_permlanex16_b32
S_NEED = ("s91", "s92", "s93", "s94")  # per q tile: the lanes whose row max grew past the threshold
S_ANY = ("s95", "s96", "s97", "s98")  # per q tile: -1 if any lane did, else 0
S_MASK = ("s99", "s100", "s101")  # causal-mask compares, in rotation
S_C = ("s102", "s103")  # (c, c) in the TILE regions, copied from the region's c2 operand
CLOBBER_S = range(89, 104)

# Scheduler model (cycles; ATT: measured on MI455X with the hardware's thread trace)
DS_LAT = 290  # LDS load-to-use latency
DS_RATE = 6  # one LDS load issued per DS_RATE cycles
DS_MAX = 63  # LDS loads in flight at most (the dscnt field is 6 bits)
# the in-order WMMA stream, by node tag (PV3f: TILE_D3F's FINAL prefix, after the tile's QK3)
WORDER = ("PV0p", "PV1p", "PV2p", "PV3p", "QK0", "QK1", "QK2", "QK3", "PV3f")
# latency of a dependence edge by producer kind (it ranks nodes by critical path), and result latency by kind
# in the issue model (WMMA 28 and VALU 6: the values that scheduled the TILE regions best on MI455X)
LAT = {
    "WMMA": 28,
    "VALU": 6,
    "TRANS": 8,
    "PERM": 8,
    "DS": DS_LAT,
    "SALU": 2,
    "RESC": 2,
    "TDM": 2,
    "DSST": 1,
    "BARW": 1,
}
SLOT_LAT = {
    "WMMA": 28,
    "VALU": 6,
    "TRANS": 8,
    "PERM": 6,
    "SALU": 2,
    "RESC": 2,
    "TDM": 2,
    "DSST": 1,
    "BARW": 1,
}
BARW_LAT = (
    200  # TILE_D3F: its barrier wait no earlier than this many cycles behind its arrive (the other waves')
)
RESC_COST = 6  # modelled cycles of a RESC node (its s_cbranch over the block) when the branch skips it
BR_LAT = 9  # a RESC node's s_cmp -> s_cbranch (ATT: 9 cycles; a taken branch costs ~25 more)
SGPR_LAT = 20  # a VALU-written SGPR to a SALU / RESC reader (ATT: v_cmp_lt sX -> s_cmp_lg sX ~20 cycles)
TDM_XLAT = 48  # a TDM op to a write of an SGPR it reads (s_wait_xcnt): keep the rewrite well behind the issue
# The cycles after a WMMA's issue (0..7) in which a VALU / TRANS may issue (bit k: offset k), and in which the
# scheduler prefers a TRANS over a VALU (ATT, MI455X: a VALU at offset 1 delays the next WMMA)
VALU_OFFS = 0b11001100
TRANS_OFFS = 0b10001000
SQT_B = 4480  # the staging slot bytes per wave and q tile (fwd_kernel.SQT)
STG_B = 43008  # the bytes of a ring stage (fwd_kernel.STG_B)


class Layout:
    """VGPR map of the generated code (true register numbers): block 0 from `base` up, blocks 1..3 whole.

    The softmax's registers share block 2 (S, the sum-tree leaves PR, the (m', d) and (corr, m) pairs: the
    packed ops read their low halves only), so the softmax VALU runs in one s_set_vgpr_msb mode; its scratch
    is the parity's sum-tree leaves 0..17 (dead from the sum tree that consumed them to the q tile's exps that
    write them), the (peer sum, total) pairs of a sum tree two leaf pairs it consumed (sum_nodes). O fills
    block 1; block 3 holds K and the V fragments of d tiles 0..3; block 0 (after the shell's VGPRs) the mask
    bounds, the Q fragments of q tiles 0..2 and the V fragments of d tiles 4..7; block 2 after the pairs the Q
    fragments of q tile 3.
    Live across regions: m, d, (c, c), the Q bases and O; through a block's tiles the Q fragments of all four
    q tiles (INIT reads them); from one TILE region to the next (and into FINAL_A / TILE_D3F) also V, P(0),
    P(1), S(2), S(3). FINAL's scratch: qbuf[0] (12 VGPRs per q tile: Q(0) is dead after the block's last QK) and
    its bf16 outputs (32 VGPRs per q tile): K in FINAL_A, d3f_outs in TILE_D3F."""

    def __init__(self, base):
        nxt = [base + (base % 2)]

        def take(n, align=2):
            r = nxt[0] + ((-nxt[0]) % align)
            nxt[0] = r + n
            if nxt[0] > 256:
                raise ValueError(f"block 0 overflows (base {base})")
            return r

        b2 = [704]

        def take2(n, align=1):
            r = b2[0] + ((-b2[0]) % align)
            b2[0] = r + n
            if b2[0] > 768:
                raise ValueError("block 2 overflows")
            return r

        self.base = base
        self.cc = take(2)  # (c, c)
        self.qa, self.qb = take(1, 1), take(1, 1)  # Q fragment bases of q tiles 0, 1 and 2, 3
        self.dl = [None] + [take(1, 1) for _ in range(3)]  # mask bound of q tile i: dl + 64 i
        self.qbuf = [take(48, 8) for _ in range(3)]  # Q fragments of q tiles 0..2, see Q()
        self.vhi = take(64, 8)  # V[dt][kt] of d tiles 4..7
        self.end0 = nxt[0]
        self.s = 512  # block 2: S[i][j] = 512 + 32 i + 8 j
        self.pr = 640  #          PR: leaf x of q tile i at 640 + 2 x + (i % 2)
        self.sm = [[self.pr + 2 * k + p for k in range(18)] for p in range(2)]  # softmax scratch of parity p
        self.mn = [take2(2, 2) for _ in range(QT)]  # (m', d)
        self.corr = [take2(2, 2) for _ in range(QT)]  # (corr, m): packed ops broadcast the low half
        self.d = [self.mn[i] + 1 for i in range(QT)]
        self.m = [self.corr[i] + 1 for i in range(QT)]
        self.qbuf.append(take2(48, 8))  # Q fragments of q tile 3
        self.o = 256  # block 1: O[i][dt] = 256 + 64 i + 8 dt
        self.k = 768  # block 3: K[j][ks] = 768 + 48 j + 8 ks, then V of d tiles 0..3 from 960
        self.vlo = 960

    def O(self, i, dt):
        return self.o + 64 * i + 8 * dt

    def V(self, dt, kt):
        return (self.vlo + 16 * dt if dt < 4 else self.vhi + 16 * (dt - 4)) + 8 * kt

    def S(self, i, j):
        return self.s + 32 * i + 8 * j

    def K(self, j, ks):
        return self.k + 48 * j + 8 * ks

    def Q(self, i, ks):
        """Q fragments of q tile i, k-step ks (resident in qbuf[i] through the block)."""
        return self.qbuf[i] + 8 * ks

    def PR(self, i, leaf):
        """p of q tile i, leaf 8 j + e; a pair's two q tiles interleave, the sum tree adds both at once."""
        return self.pr + 2 * leaf + (i % 2)

    def P(self, i, kt):
        """bf16 P^T (B operand) of q tile i, kv half kt: in place of S(i, 2 kt) once the exps have read it."""
        return self.S(i, 2 * kt)


# ------------------------------------------------------------------------------------------------------------
# instruction nodes and the dependence graph
# ------------------------------------------------------------------------------------------------------------
class Node:
    __slots__ = ("text", "kind", "vdefs", "vuses", "sdefs", "suses", "preds", "idx", "tag")

    def __init__(self, text, kind, tag=""):
        self.text, self.kind, self.tag = text, kind, tag
        self.preds, self.idx = [], -1
        self.vdefs, self.vuses, self.sdefs, self.suses = set(), set(), set(), set()


def _vregs(op):
    """True VGPRs of one operand text (asm's parser)."""
    return A.regs_of(A.first_tok(op))


def _sregs(op):
    """Scalar registers of one operand text (asm's parser)."""
    return A.sregs(A.first_tok(op))


def mk(text, kind, tag=""):
    """A node with the registers its text defines and uses. Covers the forms the regions schedule: ds_load
    (vdst, vaddr), ds_store (vaddr, data), s_mov / s_cmp / s_cselect, tensor_load_to_lds (reads its two
    descriptor groups), other SALU (first operand the destination, scc written unless the op is known not to),
    v_cmp_*_e64 with an SGPR destination, v_div_scale (vdst, sdst vcc or null), v_div_fmas (reads vcc), and
    VALU / WMMA ops whose first operand is their only destination and that read no implicit operand."""
    n = Node(text, kind, tag)
    if kind in ("BAR", "BARW"):  # s_barrier_signal / s_barrier_wait: ordered by explicit edges
        return n
    # "RESC2 a b oa ob corra corrb scc": O(a), O(b), one branch
    if kind == "RESC" and text.startswith("RESC2"):
        _, _, _, oa, ob, ca, cb, sany = text.split()
        n.vdefs = set(range(int(oa), int(oa) + 64)) | set(range(int(ob), int(ob) + 64))
        n.vuses = n.vdefs | {int(ca), int(cb)}
        n.suses = {sany}
        return n
    if kind == "RESC":  # "RESC i o corr sany": the block O(i) *= corr(i) under S_ANY[i]
        _, _, o, corr, sany = text.split()
        n.vdefs = set(range(int(o), int(o) + 64))
        n.vuses = n.vdefs | {int(corr)}
        n.suses = {sany}
        return n
    mn, _, rest = text.strip().partition(" ")
    ops = A.split_operands(rest)
    if mn.startswith("ds_load"):
        n.vdefs, n.vuses = set(_vregs(ops[0])), set(_vregs(ops[1]))
    elif mn.startswith("ds_store"):  # address, data: uses only
        n.vuses = set(_vregs(ops[0])) | set(_vregs(ops[1]))
    elif mn.startswith("s_cmp"):
        n.sdefs, n.suses = {"scc"}, set(_sregs(ops[0])) | set(_sregs(ops[1]))
    elif mn.startswith("s_cselect"):
        n.sdefs, n.suses = set(_sregs(ops[0])), {"scc"} | {r for o in ops[1:] for r in _sregs(o)}
    elif mn.startswith("s_mov"):
        n.sdefs, n.suses = set(_sregs(ops[0])), set(_sregs(ops[1]))
    elif mn.startswith("tensor_load_to_lds"):
        n.suses = set(_sregs(ops[0])) | set(_sregs(ops[1]))
    # other SALU: the first operand is the destination; scc unless it is known not to
    elif mn.startswith("s_"):
        n.sdefs = set(_sregs(ops[0]))
        if not mn.startswith(("s_mul_i32", "s_add_nc_u64", "s_bitset1_b32")):
            n.sdefs.add("scc")
        n.suses = {r for o in ops[1:] for r in _sregs(o)}
    elif mn.startswith("v_div_scale"):  # vdst, sdst (vcc_lo or null), sources
        n.vdefs = set(_vregs(ops[0]))
        n.sdefs = {"vcc"} if "vcc" in ops[1] else set()
        n.vuses = {r for o in ops[2:] for r in _vregs(o)}
        n.suses = {r for o in ops[2:] for r in _sregs(o)}
    elif mn.startswith("v_"):
        if mn.startswith("v_cmp"):
            n.sdefs = set(_sregs(ops[0]))
        else:
            n.vdefs = set(_vregs(ops[0]))
        n.vuses = {r for o in ops[1:] for r in _vregs(o)}
        n.suses = {r for o in ops[1:] for r in _sregs(o)}
        if mn.startswith("v_div_fmas"):  # reads vcc implicitly
            n.suses.add("vcc")
    else:
        raise ValueError(f"unmodelled instruction {text}")
    return n


def _key(r):
    return ("v", r) if isinstance(r, int) else ("s", r)


def build_dag(nodes):
    """RAW (producer latency; SGPR_LAT for a VALU-written SGPR read by SALU / RESC, BR_LAT for the scc of a
    RESC node's s_cmp), WAR / WAW (latency 1; TDM_XLAT for the rewrite of an SGPR a TDM op reads) edges in list
    order: node.preds = [(index, latency)]."""
    last_w, readers = {}, {}
    for i, n in enumerate(nodes):
        n.idx = i
        preds = {}
        for r in list(n.vuses) + list(n.suses):
            k = _key(r)
            if k in last_w:
                p = last_w[k]
                lat = 8 if nodes[p].kind == n.kind == "WMMA" else LAT[nodes[p].kind]  # 8: accumulator chain
                pk = nodes[p].kind
                if (
                    k[0] == "s"
                    and r not in ("scc", "vcc")
                    and pk in ("VALU", "TRANS", "PERM")
                    and (n.kind in ("SALU", "RESC"))
                ):
                    lat = SGPR_LAT
                if r == "scc" and pk == "SALU" and n.kind == "RESC":
                    lat = BR_LAT
                preds[p] = max(preds.get(p, 0), lat)
        for r in list(n.vdefs) + list(n.sdefs):
            k = _key(r)
            if k in last_w and last_w[k] != i:
                preds[last_w[k]] = max(preds.get(last_w[k], 0), 1)
            for q in readers.get(k, ()):
                if q != i:
                    lat = TDM_XLAT if (k[0] == "s" and nodes[q].kind == "TDM") else 1
                    preds[q] = max(preds.get(q, 0), lat)
        n.preds = sorted(preds.items())
        for r in list(n.vuses) + list(n.suses):
            readers.setdefault(_key(r), []).append(i)
        for r in list(n.vdefs) + list(n.sdefs):
            last_w[_key(r)] = i
            readers[_key(r)] = []
    return nodes


def node_list():
    """(nodes, add): add(text, kind, tag) appends a node to nodes and returns it."""
    nodes = []

    def add(text, kind, tag=""):
        nd = mk(text, kind, tag)
        nodes.append(nd)
        return nd

    return nodes, add


# ------------------------------------------------------------------------------------------------------------
# scheduler
# ------------------------------------------------------------------------------------------------------------
def schedule(nodes):
    """The nodes in issue order: a cycle-stepped list scheduler on an issue model of one wave.

    WMMAs issue in WORDER (program order within a tag), one per 8 cycles at most (XDL occupancy). Within 8
    cycles of a WMMA, a VALU / TRANS issues only at offsets 2, 3, 6, 7 and v_permlanex16 only at 8; other
    nodes issue at any offset but 0. A TRANS blocks VALU / TRANS / permlane the next cycle; a permlane takes 2
    cycles and goes before a ready WMMA (else a dense WMMA stream starves it); one SALU or TDM op issues per 2
    cycles; a RESC block (its s_cmp + s_cbranch) blocks the wave for RESC_COST cycles.
    LDS loads issue one per DS_RATE cycles and complete in order DS_LAT cycles after issue, at most DS_MAX in
    flight (not yet complete; a consumer of a load retires it and every older one, the arrive
    (s_barrier_signal behind s_wait_dscnt 0) retires all). Other results are ready SLOT_LAT cycles after
    issue (an SGPR_LAT / BR_LAT edge: that many cycles). Among the ready nodes:
    a VALU at offsets 2, 6, a TRANS at 3, 7, an LDS load elsewhere first; then the arrive, the earliest
    deadline (the first WMMA of the stream that needs the node), the longest critical path (LAT), program
    order. An s_wait_dscnt goes in where a load would exceed DS_MAX."""
    n = len(nodes)
    wmmas = [i for i in range(n) if nodes[i].kind == "WMMA"]
    wseq = [i for tag in WORDER for i in wmmas if nodes[i].tag == tag]
    if sorted(wseq) != wmmas:
        raise ValueError("WORDER does not cover the WMMAs")
    succ, deadline, cp = A.dag_ranks(nodes, wseq)
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = {i for i in range(n) if npred[i] == 0}
    ds_done = {}  # LDS load -> completion cycle
    issued_ds, retired_ds, last_done = [], 0, 0
    out = []
    t, last_w, wk, trans_at, perm_busy, salu_free, ds_next, done = 0, -100, 0, -100, -1, 0, 0, 0

    def avail(i):
        k = nodes[i].kind
        if earliest[i] > t:
            return False
        o = t - last_w
        if k in ("VALU", "TRANS"):
            return (not 0 <= o <= 8 or (o < 8 and VALU_OFFS >> o & 1)) and t != trans_at + 1 and t > perm_busy
        if k == "PERM":
            return o >= 8 and t != trans_at + 1 and t > perm_busy
        if k == "DS":  # loads in flight: neither retired by a consumer nor complete in the model
            inflight = sum(1 for p in issued_ds[retired_ds:] if ds_done[p] > t)
            return o != 0 and t >= ds_next and inflight < DS_MAX
        if k in ("SALU", "RESC", "TDM"):
            return o != 0 and t >= salu_free and (k != "RESC" or t > perm_busy)
        return o != 0

    while done < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        perms = [i for i in ready if nodes[i].kind == "PERM" and avail(i)]
        if perms:
            pick = max(perms, key=lambda i: (-deadline[i], cp[i], -i))
        elif nw is not None and nw in ready and earliest[nw] <= t and t >= last_w + 8 and t > perm_busy:
            pick = nw
        else:
            cand = [i for i in ready if nodes[i].kind != "WMMA" and avail(i)]
            if cand:
                o = t - last_w
                if 0 <= o < 8 and VALU_OFFS >> o & 1:
                    vt = [i for i in cand if nodes[i].kind in ("VALU", "TRANS")]
                    if vt:
                        want = "TRANS" if TRANS_OFFS >> o & 1 else "VALU"
                        cand = [i for i in vt if nodes[i].kind == want] or vt
                elif any(nodes[i].kind == "DS" for i in cand):
                    cand = [i for i in cand if nodes[i].kind in ("DS", "BAR")]
                pick = max(cand, key=lambda i: (nodes[i].kind == "BAR", -deadline[i], cp[i], -i))
        if pick is None:
            t += 1
            if t > 10**6:
                raise RuntimeError("scheduler deadlock")
            continue
        nd = nodes[pick]
        ready.discard(pick)
        k = nd.kind
        if k == "DS":
            last_done = max(t + DS_LAT, last_done)
            ds_done[pick] = last_done
            issued_ds.append(pick)
            ds_next = t + DS_RATE
        elif k == "BAR":
            retired_ds = len(issued_ds)
        for p, _ in nd.preds:
            if nodes[p].kind == "DS" and k != "DS" and p in ds_done and nodes[p].vdefs & nd.vuses:
                retired_ds = max(retired_ds, issued_ds.index(p) + 1)
        out.append(nd)
        done += 1
        if k == "WMMA":
            last_w = t
            wk += 1
        elif k == "TRANS":
            trans_at = t
        elif k == "PERM":
            perm_busy = t + 1
        elif k in ("SALU", "TDM"):
            salu_free = t + 2
        elif k == "RESC":  # s_cmp + s_cbranch: the wave issues nothing else meanwhile
            salu_free = t + RESC_COST
            perm_busy = t + RESC_COST - 1
            ds_next = max(ds_next, t + RESC_COST)
        for s, lat in succ[pick]:
            if k == "DS":
                e = ds_done[pick] if nd.vdefs & nodes[s].vuses else t + 1
                if lat > 1 and nodes[s].kind == "BAR":
                    e = max(e, ds_done[pick])
            elif lat <= 1:
                e = t + 1  # WAR / WAW / order edge
            elif k == "BAR":
                e = t + lat  # TILE_D3F: arrive -> wait
            elif (
                (lat == SGPR_LAT and k in ("VALU", "TRANS", "PERM"))
                or (lat == BR_LAT and nodes[s].kind == "RESC")
                or (lat == TDM_XLAT and k == "TDM")
            ):
                e = t + lat  # a VALU-written SGPR read by SALU; scc to the RESC branch
            else:
                e = t + SLOT_LAT[k]
            earliest[s] = max(earliest[s], e)
            npred[s] -= 1
            if npred[s] == 0:
                ready.add(s)
        t += 1
    # the scheduler bounds the loads in flight only through their consumers: add the explicit waits
    res, nds, ret, seqn = [], 0, 0, {}
    for nd in out:
        if nd.kind == "DS":
            if nds - ret >= DS_MAX:
                res.append(Node(f"s_wait_dscnt 0x{DS_MAX - 1:x}", "WAIT", "dsmax"))
                ret = nds - (DS_MAX - 1)
            seqn[nd.idx] = nds
            nds += 1
        elif nd.kind == "BAR":
            ret = nds
        else:
            for p, _ in nd.preds:
                if p in seqn and nodes[p].vdefs & nd.vuses:
                    ret = max(ret, seqn[p] + 1)
        res.append(nd)
    return res


# ------------------------------------------------------------------------------------------------------------
# regions
# ------------------------------------------------------------------------------------------------------------
def q_load(L, i, ks, h):
    """Half h of the Q fragment of q tile i, k-step ks."""
    off = (i & 1) * QOFF1 + ks * 64 + 32 * h
    return f"ds_load_b128 {vr(L.Q(i, ks) + 4 * h, 4)}, {vr(L.qa if i < 2 else L.qb)} offset:{off}"


def k_load(L, kb, j, ks, h):
    """Half h of the K fragment of kv16 tile j, k-step ks (kb: the stage's K base VGPR)."""
    off = j * 16 * XK + ks * 64 + 32 * h
    return f"ds_load_b128 {vr(L.K(j, ks) + 4 * h, 4)}, {vr(kb)} offset:{off}"


def v_load(L, vb, dt, kt, h):
    """Half h of the V^T fragment of d tile dt, kv half kt (transposing load; vb: the stage's V base VGPR)."""
    off = (32 * kt + 16 * h) * XV + dt * 32
    return f"ds_load_tr16_b128 {vr(L.V(dt, kt) + 4 * h, 4)}, {vr(vb)} offset:{off}"


def region_init(L, ops):
    """INIT (program order). ops: qa / qb (VGPRs: Q fragment bases of q tiles 0, 1 and 2, 3), c2 (SGPR). The resident Q
    fragments of all four q tiles are read first (the Q staging area does not change until the block's last tile);
    the zeroing of O and of the first prefix's inputs (S(2), S(3) = -inf; P(0), P(1) = 0; V = 0: the first prefix is a
    no-op) runs under their latency."""
    out = []
    out.append(f"v_mov_b32_e32 {vr(L.qa)}, {vr(ops['qa'])}")
    out.append(f"v_mov_b32_e32 {vr(L.qb)}, {vr(ops['qb'])}")
    for i in range(QT):
        out.append(f"v_mov_b32_e32 {vr(L.m[i])}, 0x{BIG_NEG:x}")
        out.append(f"v_mov_b32_e32 {vr(L.d[i])}, 0")
    out.append(f"v_mov_b32_e32 {vr(L.cc)}, {ops['c2']}")
    out.append(f"v_mov_b32_e32 {vr(L.cc + 1)}, {ops['c2']}")
    out += [q_load(L, i, ks, h) for i in range(QT) for ks in range(NKS) for h in range(2)]
    zero = [(r, 0) for r in range(L.o, L.o + 256)]
    zero += [(r, NEG_INF) for i in (2, 3) for r in range(L.S(i, 0), L.S(i, 0) + 32)]
    zero += [(L.P(i, kt) + e, 0) for i in (0, 1) for kt in range(2) for e in range(8)]
    zero += [(L.V(dt, kt) + e, 0) for dt in range(NDT) for kt in range(2) for e in range(8)]
    for k in range(0, len(zero), 2):  # in register pairs
        (r, v), (r2, v2) = zero[k], zero[k + 1]
        assert r % 2 == 0 and r2 == r + 1 and v == v2
        out.append(f"v_mov_b64_e32 {vr(r, 2)}, 0x{(v << 32) | v:x}" if v else f"v_mov_b64_e32 {vr(r, 2)}, 0")
    out.append("s_wait_dscnt 0x0")
    return out


def mask_nodes(L, i, dl, add):
    """The causal mask of q tile i: s = -inf where 16 j + e > dl (kv > q)."""
    for j in range(NKV):
        for e in range(8):
            sm = S_MASK[(8 * j + e) % len(S_MASK)]
            s = vr(L.S(i, j) + e)
            add(f"v_cmp_gt_i32_e64 {sm}, {16 * j + e}, {vr(dl)}", "VALU", f"mask{i}")
            add(f"v_cndmask_b32_e64 {s}, {s}, 0x{NEG_INF:x}, {sm}", "VALU", f"mask{i}")


def softmax_nodes(L, i, add, c_sgpr, resc=True):
    """The softmax of q tile i up to P and O *= corr: scores in S(i), p into the leaves PR(i). c from the SGPR
    pair S_C (c_sgpr: the packed fma then has two VGPR sources, not three) or from the VGPR pair L.cc. resc
    False: no branch of q tile i's own (resc_pair_nodes covers it)."""
    S = [[L.S(i, j) + e for e in range(8)] for j in range(NKV)]
    sc = L.sm[i % 2]
    # row max over the lane's 32 values (exact; any tree): 32 -> 11 -> 4 -> 2 -> 1 in scratch sc[0..10]
    vals = [S[j][e] for j in range(NKV) for e in range(8)]
    while len(vals) > 1:
        nxt = []
        for a in range(0, len(vals), 3):
            grp = vals[a : a + 3]
            if len(grp) == 1:
                nxt.append(grp[0])
                continue
            r = sc[len(nxt)]
            if len(grp) == 3:
                add(f"v_max3_num_f32 {vr(r)}, {vr(grp[0])}, {vr(grp[1])}, {vr(grp[2])}", "VALU", f"max{i}")
            else:
                add(f"v_max_num_f32_e32 {vr(r)}, {vr(grp[0])}, {vr(grp[1])}", "VALU", f"max{i}")
            nxt.append(r)
        vals = nxt
    lmx, pl, lm, need, rmax, mfull, dm = vals[0], sc[11], sc[12], sc[13], sc[14], sc[15], sc[16]
    add(f"v_permlanex16_b32 {vr(pl)}, {vr(lmx)}, {S_SEL}, 0x{SEL_HI:x}", "PERM", f"max{i}")
    add(f"v_max_num_f32_e32 {vr(lm)}, {vr(lmx)}, {vr(pl)}", "VALU", f"max{i}")
    c = S_C[0] if c_sgpr else vr(L.cc)
    add(f"v_fma_f32 {vr(need)}, {vr(lm)}, {c}, -{vr(L.m[i])}", "VALU", f"dec{i}")
    add(f"v_mul_f32_e32 {vr(rmax)}, {c}, {vr(lm)}", "VALU", f"dec{i}")
    add(f"v_max_num_f32_e32 {vr(mfull)}, {vr(L.m[i])}, {vr(rmax)}", "VALU", f"dec{i}")
    add(f"v_cmp_lt_f32_e64 {S_NEED[i]}, 0x{LOG2E:x}, {vr(need)}", "VALU", f"dec{i}")
    add(f"s_cmp_lg_u32 {S_NEED[i]}, 0", "SALU", f"dec{i}")
    add(f"s_cselect_b32 {S_ANY[i]}, -1, 0", "SALU", f"dec{i}")
    mn, corr = L.mn[i], L.corr[i]
    add(f"v_cndmask_b32_e64 {vr(mn)}, {vr(L.m[i])}, {vr(mfull)}, {S_ANY[i]}", "VALU", f"dec{i}")
    add(f"v_sub_f32_e32 {vr(dm)}, {vr(L.m[i])}, {vr(mn)}", "VALU", f"dec{i}")
    add(f"v_exp_f32_e32 {vr(corr)}, {vr(dm)}", "TRANS", f"dec{i}")
    add(f"v_mov_b32_e32 {vr(L.m[i])}, {vr(mn)}", "VALU", f"dec{i}")
    # p = exp2(fma(s, c, -m')): packed fma in place, then exp into the leaves. An SGPR-pair source of a packed
    # f32 op takes op_sel_hi 0 (the assembler rejects 1): both halves read c from s102
    cc = f"s[{S_C[0][1:]}:{S_C[1][1:]}]" if c_sgpr else vr(L.cc, 2)
    for j in range(NKV):
        for e in range(0, 8, 2):
            add(
                f"v_pk_fma_f32 {vr(S[j][e], 2)}, {vr(S[j][e], 2)}, {cc}, {vr(mn, 2)} "
                f"op_sel_hi:[1,{0 if c_sgpr else 1},0] neg_lo:[0,0,1] neg_hi:[0,0,1]",
                "VALU",
                f"p{i}",
            )
            for x in range(2):
                add(f"v_exp_f32_e32 {vr(L.PR(i, 8 * j + e + x))}, {vr(S[j][e + x])}", "TRANS", f"p{i}")
    for kt in range(2):  # P^T (bf16) for PV
        for r in range(8):
            j, e = 2 * kt + r // 4, 2 * (r % 4)
            lo, hi = L.PR(i, 8 * j + e), L.PR(i, 8 * j + e + 1)
            add(f"v_cvt_pk_bf16_f32 {vr(L.P(i, kt) + r)}, {vr(lo)}, {vr(hi)}", "VALU", f"cvt{i}")
    if not resc:
        return
    # O *= corr under a wave-uniform branch: the s_cmp is a node of its own (the scheduler may issue it ahead of
    # the branch), the branch and the block one node (RESC, reading its scc; _emit_segments expands it: resc_block)
    add(f"s_cmp_lg_u32 {S_ANY[i]}, 0", "SALU", f"resc{i}")
    add(f"RESC {i} {L.O(i, 0)} {corr} scc", "RESC", f"resc{i}")


def resc_lines(L, i):
    """O(i) *= corr(i): 32 packed multiplies."""
    out = []
    for dt in range(NDT):
        for r in range(0, 8, 2):
            o = vr(L.O(i, dt) + r, 2)
            out.append(f"v_pk_mul_f32 {o}, {o}, {vr(L.corr[i], 2)} op_sel_hi:[1,0]")
    return out


def resc_pair_nodes(L, a, b, add):
    """One branch for q tiles a, b: O(a) *= corr(a) and O(b) *= corr(b) when either needs it (the other's corr is
    exactly 1 then: x * 1.0 is x, the same bits). S_ANY[a] |= S_ANY[b] sets scc (S_ANY[a] is dead after its
    v_cndmask)."""
    add(f"s_or_b32 {S_ANY[a]}, {S_ANY[a]}, {S_ANY[b]}", "SALU", f"resc{a}{b}")
    add(f"RESC2 {a} {b} {L.O(a, 0)} {L.O(b, 0)} {L.corr[a]} {L.corr[b]} scc", "RESC", f"resc{a}{b}")


def resc_block(L, line, label):
    """A RESC pseudo-line's expansion: (the line ending the block before it: s_cbranch over the conditional
    block when the scc of its s_cmp / s_or found S_ANY 0, the conditional block, the label line starting the
    block after it)."""
    if line.startswith("RESC2"):
        _, a, b = line.split()[:3]
        return [f"s_cbranch_scc0 {label}"], resc_lines(L, int(a)) + resc_lines(L, int(b)), f"{label}:"
    _, i, _, _, cond = line.split()
    if cond != "scc":
        raise ValueError(f"RESC node without its own s_cmp: {line}")
    return [f"s_cbranch_scc0 {label}"], resc_lines(L, int(i)), f"{label}:"


def _bank_pair(r):
    """The two VGPR banks (register % 4) a 64-bit pair starting at r reads."""
    return {r % 4, (r + 1) % 4}


def sum_nodes(L, i0, add, upd):
    """FlyDSL's sum tree over the 32 leaves of q tiles (i0, i0 + 1), packed (low half i0), in place in the
    leaves (the cvts have read them); then d of the q tiles in upd (one outside it has stale leaves here), the
    (peer sum, total) of each in two leaf pairs the tree consumed (the last two but its result pair).
    A group's partial sum goes to the leaf register whose VGPR banks differ from the other members' (a + b
    computed as b + a into b, then + c: the same additions, IEEE-commutative, so the same bits): an
    accumulate-form v_pk_add whose other source shares a bank with the accumulated pair costs a cycle."""
    leaves = [L.pr + 2 * x for x in range(32)]
    while len(leaves) > 1:  # (a + b) + c per group of 3; a trailing pair a + b, a trailing single passed on
        groups = [leaves[a : a + 3] for a in range(0, len(leaves), 3)]
        nxt = []
        for g in groups:
            acc = g[0]
            if len(g) >= 2:
                # the accumulator must be one of the first two (the first add is a + b); prefer the one whose banks
                # differ from every other member, then the second (keeps the partial sums' banks alternating)
                def clash(c, g=g):
                    return sum(1 for y in g if y != c and _bank_pair(y) & _bank_pair(c))

                acc = min((g[1], g[0]), key=clash)
            for y in (y for y in g if y != acc):
                add(f"v_pk_add_f32 {vr(acc, 2)}, {vr(acc, 2)}, {vr(y, 2)}", "VALU", f"sum{i0}")
            nxt.append(acc)
        leaves = nxt
    spare = [y for y in reversed([L.pr + 2 * x for x in range(32)]) if y != leaves[0]][:2]
    tots = [spare[0], spare[1], spare[0] + 1, spare[1] + 1]
    for x in range(2):
        qi = i0 + x
        if qi not in upd:
            continue
        pl, tot = tots[2 * x], tots[2 * x + 1]
        add(f"v_permlanex16_b32 {vr(pl)}, {vr(leaves[0] + x)}, {S_SEL}, 0x{SEL_HI:x}", "PERM", f"sum{qi}")
        add(f"v_add_f32_e32 {vr(tot)}, {vr(leaves[0] + x)}, {vr(pl)}", "VALU", f"sum{qi}")
        add(f"v_fma_f32 {vr(L.d[qi])}, {vr(L.corr[qi])}, {vr(L.d[qi])}, {vr(tot)}", "VALU", f"sum{qi}")


def prefix_nodes(L, add, c_sgpr, skip=0, pv_tag="p"):
    """The tail of the previous tile: the softmax of q tiles 2, 3 (masked scores in S(2), S(3)) with one RESC
    branch for both, their sum tree, then the PV WMMAs of all four q tiles (V of the previous tile), tags
    PV{i}p. skip: the previous tile skipped q tiles 0 .. skip - 1 (P 0, S(2) -inf): none of their PVs, no
    softmax of q tile 2 (q tile 3 then has its own RESC branch; the sum tree's q tile 2 half adds stale
    leaves into nothing)."""
    soft = [i for i in (2, 3) if i >= skip]
    pair = len(soft) == 2
    for i in soft:
        softmax_nodes(L, i, add, c_sgpr, resc=not pair)
    if pair:
        resc_pair_nodes(L, 2, 3, add)
    sum_nodes(L, 2, add, tuple(soft))
    for i in range(skip, QT):
        for kt in range(2):
            for dt in range(NDT):
                o = vr(L.O(i, dt), 8)
                add(
                    f"v_wmma_f32_16x16x32_bf16 {o}, {vr(L.V(dt, kt), 8)}, {vr(L.P(i, kt), 8)}, {o}",
                    "WMMA",
                    f"PV{i}{pv_tag}",
                )


def tile_nodes(L, ops, k=None, tail=None):
    """The nodes of a TILE region in program order, their dependence graph built. ops: kb / vb (VGPRs: the K / V
    fragment bases of the stage), c2 (SGPR: c), for a diagonal tile dl (VGPR: q + Skv - Sq - kv0 - 8 (lane / 16)
    of q tile 0); "loop": the body of TILE_CL (loop_nodes), "init": INIT_C0 (init_c0_nodes). k: None for an
    unmasked tile, else the diagonal tile k (q tiles below k skipped, the mask on q tile k only). tail(add, bar):
    appends nodes after the tile's (bar: its arrive) and returns a function that adds their order edges to the
    graph (TILE_D3F)."""
    first = 0 if k is None else k
    nodes, add = node_list()
    add(f"s_mov_b32 {S_SEL}, 0x{SEL_LO:x}", "SALU", "sel")
    for r in S_C:
        add(f"s_mov_b32 {r}, {ops['c2']}", "SALU", "c2")
    if "loop" in ops:  # the body of TILE_CL (tile_loop_lines)
        ops = dict(ops, kb=L.qa, vb=L.qb)
        loop_nodes(L, ops, add)
    init = "init" in ops  # INIT_C0: INIT and the block's tile 0 (no previous tile: no prefix)
    if init:
        init_c0_nodes(L, ops, add)
    # the K reads first (k-step major: FlyDSL's issue order); the prefix reads no K. The Q fragments are resident
    loads = [
        add(k_load(L, ops["kb"], j, ks, h), "DS", f"K{j}.{ks}.{h}")
        for ks in range(NKS)
        for j in range(NKV)
        for h in range(2)
    ]
    if not init:
        prefix_nodes(L, add, c_sgpr=True, skip=max(first - 1, 0))  # the q tiles TILE_D(k - 1) skipped
    # the next prefix must be a no-op for the q tiles this body skips. The next diagonal tile's prefix skips them;
    # only FINAL_A, the last region of a block ending after TILE_D1 / D2, reads the P of q tile first - 1 (the P
    # of the q tiles before it are 0 since an earlier diagonal tile); TILE_D3F's FINAL none
    if first in (1, 2):
        for kt in range(2):
            for e in range(8):
                add(f"v_mov_b32_e32 {vr(L.P(first - 1, kt) + e)}, 0", "VALU", f"skip{first - 1}")
    if k:
        add(f"v_add_nc_u32_e32 {vr(L.dl[k])}, {64 * k}, {vr(ops['dl'])}", "VALU", "dl")
    # the V^T fragments, after the prefix's PV WMMAs have read the previous tile's
    loads += [
        add(v_load(L, ops["vb"], dt, kt, h), "DS", f"V{dt}.{kt}.{h}")
        for dt in range(NDT)
        for kt in range(2)
        for h in range(2)
    ]
    for i in range(first, QT):
        for ks in range(NKS):  # QK(i): S[i][j] = sum_ks K[j][ks] Q[i][ks], chains from +0
            for j in range(NKV):
                s = vr(L.S(i, j), 8)
                c = "0" if ks == 0 else s
                qk = f"v_wmma_f32_16x16x32_bf16 {s}, {vr(L.K(j, ks), 8)}, {vr(L.Q(i, ks), 8)}, {c}"
                add(qk, "WMMA", f"QK{i}")
        if i == k:
            mask_nodes(L, i, ops["dl"] if i == 0 else L.dl[i], add)
        if i < 2:  # the rest of q tiles 2, 3 runs in the next region's prefix
            # q tile 0's RESC waits for q tile 1's (one branch for both); a body that skips q tile 0 (TILE_D1)
            # keeps q tile 1's own
            softmax_nodes(L, i, add, c_sgpr=True, resc=i == 1 and first == 1)
        if i == 1 and first == 0 and not init:  # (tile 0: O is 0, O *= corr leaves it 0)
            resc_pair_nodes(L, 0, 1, add)
        if i == 1:
            sum_nodes(L, 0, add, range(first, 2))
    if "loop" in ops:
        loop_base_nodes(L, ops, add)
    # the arrive, after every LDS read of the region and one LDS latency after the last (its s_wait_dscnt 0
    # then does not stall); no LDS read follows it, as the next block's Q may then overwrite the staging area
    bar = add("s_barrier_signal -1", "BAR", "arrive")
    tail_edges = tail(add, bar) if tail else None  # TILE_D3F: FINAL's nodes behind the tile's
    build_dag(nodes)
    for nd in loads:
        add_order(nodes, nd.idx, bar.idx, DS_LAT)
    for nd in nodes:  # the ring slot's TDM ops before the arrive (its s_wait_tensorcnt counts them)
        if nd.kind == "TDM":
            add_order(nodes, nd.idx, bar.idx)
    # an LDS load that overwrites a source of a WMMA in flight waits for va_vdst(0) (expert mode, more than
    # one VALU class in flight): the V reads go after the last prefix PV WMMA, so the group waits once instead of
    # once per buffer slice
    last = {nd.tag: nd.idx for nd in nodes if nd.kind == "WMMA"}
    pvs = [last[x] for x in (f"PV{i}p" for i in range(QT)) if x in last]
    if pvs:
        lpv = max(pvs)
        for nd in loads:
            if nd.tag.startswith("V"):
                add_order(nodes, lpv, nd.idx)
    # the first read goes before the first WMMA: the va_vdst(0) wait at the region entry then drains no WMMA (the
    # tile loop's head has no such wait: its first PV WMMA goes out at once)
    if "loop" not in ops:
        add_order(nodes, loads[0].idx, min(nd.idx for nd in nodes if nd.kind == "WMMA"))
    if init:
        # INIT_C0's 128 reads per wave (the resident Q fragments too) complete later than the model's latency: an
        # arrive among the WMMAs would hold the ones behind it at its s_wait_dscnt 0, so it goes behind the last one
        add_order(nodes, max(nd.idx for nd in nodes if nd.kind == "WMMA"), bar.idx)
    if tail_edges:
        tail_edges(nodes)
    return nodes


def init_c0_nodes(L, ops, add):
    """INIT's work in INIT_C0: m, d, (c, c), the resident Q fragments (read from the operand bases qa / qb), O = 0 (the
    next tile's PVs accumulate into it; nothing in this region reads it), and this wave's TDM op of ring slot 3 (quad
    q, octet g1). The first prefix's inputs (S(2), S(3), P(0), P(1), V) need no zeroing: no prefix runs here."""
    for i in range(QT):
        add(f"v_mov_b32_e32 {vr(L.m[i])}, 0x{BIG_NEG:x}", "VALU", "init")
        add(f"v_mov_b32_e32 {vr(L.d[i])}, 0", "VALU", "init")
    add(f"v_mov_b32_e32 {vr(L.cc)}, {ops['c2']}", "VALU", "init")
    add(f"v_mov_b32_e32 {vr(L.cc + 1)}, {ops['c2']}", "VALU", "init")
    for i in range(QT):
        for ks in range(NKS):
            for h in range(2):
                off = (i & 1) * QOFF1 + ks * 64 + 32 * h
                add(
                    f"ds_load_b128 {vr(L.Q(i, ks) + 4 * h, 4)}, {vr(ops['qa'] if i < 2 else ops['qb'])} offset:{off}",
                    "DS",
                    f"Q{i}.{ks}.{h}",
                )
    for r in range(L.o, L.o + 256, 2):
        add(f"v_mov_b64_e32 {vr(r, 2)}, 0", "VALU", "ozero")
    add(f"tensor_load_to_lds {ops['q']}, {ops['g1']}", "TDM", "tdm")


def d3f_outs(L):
    """TILE_D3F's FINAL output buffers: registers dead in diagonal tile 3 (not the K registers its QK3 reads): q
    tiles 0, 1 in S(0), S(1) (their P was consumed by TILE_D1 / D2's prefix), q tiles 2, 3 in the Q fragments of q
    tiles 1, 2 (no QK of theirs after TILE_D2)."""
    return [L.S(0, 0), L.S(1, 0), L.qbuf[1], L.qbuf[2]]


def region_d3f(L, ops):
    """The lines of TILE_D3F: diagonal tile 3 and FINAL (final_nodes, skip 3) in one region, FINAL's
    work under the tile's latencies. The tile's arrive is barrier B_n; FINAL stages q tiles 0, 1 in the ring stage
    of tile n - 1 (the one this tile reads) only behind its wait (every wave's reads of that stage done), q tiles
    2, 3 at TAIL at any time. The kernel issues the next block's Q (two TDM ops per wave) before the region: the
    arrive's tensorcnt wait leaves those in flight. ops: kb, vb, dl, c2 (tile_nodes), wa, wb, la, lb (final_nodes)."""
    fops = {k: ops[k] for k in ("wa", "wb", "la", "lb")}

    def tail(add, bar):
        barw = add("s_barrier_wait -1", "BARW", "bwait")
        staged = final_nodes(L, fops, add, skip=3, outs=d3f_outs(L), pv_tag="f")

        def edges(nodes):
            add_order(nodes, bar.idx, barw.idx, BARW_LAT)
            for i in (0, 1):
                for nd in staged[i]:
                    add_order(nodes, barw.idx, nd.idx)

        return edges

    out = []
    for nd in schedule(tile_nodes(L, ops, 3, tail=tail)):
        if nd.kind == "BAR":
            out += ["s_wait_dscnt 0x0", f"s_wait_tensorcnt 0x{2 * TDM_OPS + 2:x}"]
        out.append(nd.text)
    return out


def region_tile(L, ops, k=None):
    """The lines of an unmasked tile (k None: INIT_C0, the TILE_CL body) or TILE_Dk."""
    out = []
    for nd in schedule(tile_nodes(L, ops, k)):
        if nd.kind == "BAR":  # this wave's LDS reads done, its TDM op of the next tile's ring slot landed
            out += ["s_wait_dscnt 0x0", f"s_wait_tensorcnt 0x{2 * TDM_OPS:x}"]
        out.append(nd.text)
    return out


def region_final(L, ops):
    """The lines of FINAL_A (final_nodes)."""
    nodes, add = node_list()
    final_nodes(L, ops, add)
    return [nd.text for nd in schedule(build_dag(nodes))]


def final_nodes(L, ops, add, skip=0, outs=None, pv_tag="p"):
    """FINAL's nodes: the last tile's prefix (c from the VGPR pair: the S_C SGPRs do not live across regions), then
    per q tile i: inv = d > 0 ? 1 / d : 0 (LLVM's fp32 1.0 / d, denormals on: no mode switches),
    out = bf16(O(i) inv) into the dead K registers (K + 32 i), lse = (m + log2 d) ln 2 (LLVM's log2: v_log_f32 of
    d scaled by 2^32 below 2^-126), and their ds_stores into the staging slot: out rows at (i < 2 ? wa : wb) +
    (i & 1) 4 SQT + 32 dt, lse at (i < 2 ? la : lb) + (i & 1) 4 SQT. Per q tile its own scratch (12 VGPRs of the
    dead qbuf[0]) and SGPRs, so the four overlap. ops: wa, wb, la, lb (VGPRs: the lane's staging addresses).
    skip 3: after diagonal tile 3 (TILE_D3F), the prefix skips q tiles 0..2 (prefix_nodes). outs: the 32-VGPR
    output buffers of the four q tiles (default the dead K registers K + 32 i). Returns {q tile: its staging store
    nodes}."""
    add(f"s_mov_b32 {S_SEL}, 0x{SEL_LO:x}", "SALU", "sel")
    prefix_nodes(L, add, c_sgpr=False, skip=skip, pv_tag=pv_tag)
    staged = {}
    for i in range(QT):
        d, m = L.d[i], L.m[i]
        # 12 scratch VGPRs of the dead qbuf[0] per q tile (inv at an even register: a pair operand)
        a, b, rc, e1, q, e2, sh, sub, x, lg, inv = (L.qbuf[0] + 12 * i + k for k in range(11))
        sv = lg
        s1, s2 = S_NEED[i], S_ANY[i]
        tag = f"fin{i}"
        for t in (
            f"v_div_scale_f32 {vr(a)}, null, {vr(d)}, {vr(d)}, 1.0",
            f"v_div_scale_f32 {vr(b)}, vcc_lo, 1.0, {vr(d)}, 1.0",
            f"v_rcp_f32_e32 {vr(rc)}, {vr(a)}",
            f"v_fma_f32 {vr(e1)}, -{vr(a)}, {vr(rc)}, 1.0",
            f"v_fmac_f32_e32 {vr(rc)}, {vr(e1)}, {vr(rc)}",
            f"v_mul_f32_e32 {vr(q)}, {vr(b)}, {vr(rc)}",
            f"v_fma_f32 {vr(e2)}, -{vr(a)}, {vr(q)}, {vr(b)}",
            f"v_fmac_f32_e32 {vr(q)}, {vr(e2)}, {vr(rc)}",
            f"v_fma_f32 {vr(a)}, -{vr(a)}, {vr(q)}, {vr(b)}",
            f"v_div_fmas_f32 {vr(a)}, {vr(a)}, {vr(rc)}, {vr(q)}",
            f"v_div_fixup_f32 {vr(a)}, {vr(a)}, {vr(d)}, 1.0",
            f"v_cmp_lt_f32_e64 {s1}, 0, {vr(d)}",
            f"v_cndmask_b32_e64 {vr(inv)}, 0, {vr(a)}, {s1}",
        ):
            add(t, "TRANS" if t.startswith("v_rcp") else "VALU", tag)
        out = L.k + 32 * i if outs is None else outs[i]
        base = ops["wa"] if i < 2 else ops["wb"]
        for dt in range(NDT):
            for r in range(0, 8, 2):
                o = vr(L.O(i, dt) + r, 2)
                add(f"v_pk_mul_f32 {o}, {o}, {vr(inv, 2)} op_sel_hi:[1,0]", "VALU", tag)
                add(
                    f"v_cvt_pk_bf16_f32 {vr(out + 4 * dt + r // 2)}, {vr(L.O(i, dt) + r)}, {vr(L.O(i, dt) + r + 1)}",
                    "VALU",
                    tag,
                )
            staged.setdefault(i, []).append(
                add(
                    f"ds_store_b128 {vr(base)}, {vr(out + 4 * dt, 4)} offset:{(i & 1) * 4 * SQT_B + 32 * dt}",
                    "DSST",
                    tag,
                )
            )
        for t in (
            f"v_cmp_gt_f32_e64 {s2}, 0x800000, {vr(d)}",
            f"v_cndmask_b32_e64 {vr(sh)}, 0, 32, {s2}",
            f"v_cndmask_b32_e64 {vr(sub)}, 0, 0x42000000, {s2}",
            f"v_ldexp_f32 {vr(x)}, {vr(d)}, {vr(sh)}",
            f"v_log_f32_e32 {vr(x)}, {vr(x)}",
            f"v_sub_f32_e32 {vr(lg)}, {vr(x)}, {vr(sub)}",
            f"v_add_f32_e32 {vr(sv)}, {vr(m)}, {vr(lg)}",
            f"v_mul_f32_e32 {vr(sv)}, 0x{LN2:x}, {vr(sv)}",
        ):
            add(t, "TRANS" if t.startswith("v_log") else "VALU", tag)
        lbase = ops["la"] if i < 2 else ops["lb"]
        staged[i].append(add(f"ds_store_b32 {vr(lbase)}, {vr(sv)} offset:{(i & 1) * 4 * SQT_B}", "DSST", tag))
    return staged


# ------------------------------------------------------------------------------------------------------------
# TILE_CL: a run of unmasked tiles (a block's tiles [1, ncc)) as one loop in one inline asm statement
# ------------------------------------------------------------------------------------------------------------
# the loop's scalar state (clobbered): the tile counter t, the stage LDS base of tile t, scratch, and the group-0 quad
# of this wave's TDM op of ring slot t + 3 (count, LDS address, global address)
S_T, S_ST, S_TMP = "s83", "s82", "s81"
S_Q = 84
LOOP_CLOBBER_S = range(81, 88)


def loop_nodes(L, ops, add):
    """Per tile of the loop (tile t; L.qa / L.qb, free after INIT, hold its K / V^T fragment bases): this wave's TDM op
    of ring slot t + 3, the quad's advance to slot t + 4 (into the stage of tile t, its tile min(t + 4, n - 1)), the
    stage base's to tile t + 1 and the fragment bases of tile t + 1 (behind the reads of tile t), so the next tile
    starts with its first LDS read."""
    add(f"tensor_load_to_lds s[{S_Q}:{S_Q + 3}], {ops['g1']}", "TDM", "tdm")
    add(f"s_add_co_i32 s{S_Q + 1}, {S_ST}, {ops['ldsoff']}", "SALU", "tdmadv")
    add(f"s_cmp_lt_i32 {S_T}, {ops['n4']}", "SALU", "tdmadv")
    add(f"s_cselect_b64 s[100:101], {ops['step']}, 0", "SALU", "tdmadv")
    add(f"s_add_nc_u64 s[{S_Q + 2}:{S_Q + 3}], s[{S_Q + 2}:{S_Q + 3}], s[100:101]", "SALU", "tdmadv")
    add(f"s_add_co_i32 {S_ST}, {S_ST}, 0x{STG_B:x}", "SALU", "stadv")
    add(f"s_cmp_eq_u32 {S_ST}, {ops['lim']}", "SALU", "stadv")
    add(f"s_cselect_b32 {S_TMP}, 0x{(-4 * STG_B) & 0xFFFFFFFF:x}, 0", "SALU", "stadv")
    add(f"s_add_co_i32 {S_ST}, {S_ST}, {S_TMP}", "SALU", "stadv")


def loop_base_nodes(L, ops, add):
    """The K / V^T fragment bases of the next tile (after the stage advance; build_dag orders them behind this tile's
    reads of L.qa / L.qb)."""
    add(f"v_add_nc_u32_e32 {vr(L.qa)}, {S_ST}, {ops['klane']}", "VALU", "kb")
    add(f"v_add_nc_u32_e32 {vr(L.qb)}, {S_ST}, {ops['vlane']}", "VALU", "vb")


def _emit_segments(texts, L, flow, label, out_of_line):
    """asm.emit of a straight-line region split at its RESC pseudo-lines (see emit_region). out_of_line: the
    conditional O *= corr blocks go after the region (the common path falls through its s_cbranch_scc1); else in
    line behind an s_cbranch_scc0 over them. Returns (lines, out-of-line lines, the flow at the end)."""
    segs, cur = [], []
    for x in texts:
        if x.startswith("RESC"):
            segs.append((cur, x))
            cur = []
        else:
            cur.append(x)
    out, ool, lead = [], [], []
    for k, (tx, rline) in enumerate(segs):
        head, body, lab = resc_block(L, rline, f"{label}_s{k}")
        if out_of_line:
            head = [f"s_cbranch_scc1 {label}_r{k}"]
        lines, ex_a = A.emit(tx + head, flow)
        lines_b, ex_b = A.emit(body, ex_a)
        if out_of_line:
            out += lead + lines
            lead = [f"{label}_b{k}:"]
            ool += [f"{label}_r{k}:"] + lines_b + [f"s_branch {label}_b{k}"]
        else:
            out += lead + lines + lines_b
            lead = [lab]
        flow = A.Flow.join([ex_a, ex_b])
    lines, flow = A.emit(cur, flow)
    return out + lead + lines, ool, flow


def tile_loop_lines(L, ops):
    """TILE_CL: ncc tiles (operand) from the ring stage stage0 on as one loop (ops: TILE_CL operands). The preamble
    drains whatever the kernel left in flight and its own VALU, so the loop head is reached from a known-empty state
    and from the back edge: the waits and v_nops of the body hold for both (iterated to a fixed point)."""
    body = region_tile(L, dict(ops, loop=True))
    lab = ".LTC${:uid}"
    # the preamble drains everything the kernel may have left in flight, then sets up the loop state
    drain = ["s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)", "s_wait_dscnt 0x0", "s_wait_xcnt 0x0"]
    setup = [
        f"s_mov_b32 {S_T}, 0",
        f"s_mov_b32 {S_ST}, {ops['stage0']}",
        f"s_mov_b32 s{S_Q}, {ops['cnt']}",
        f"s_mov_b32 s{S_Q + 1}, {ops['ldsq0']}",
        f"s_mov_b64 s[{S_Q + 2}:{S_Q + 3}], {ops['addr0']}",
        f"v_add_nc_u32_e32 {vr(L.qa)}, {ops['stage0']}, {ops['klane']}",
        f"v_add_nc_u32_e32 {vr(L.qb)}, {ops['stage0']}, {ops['vlane']}",
    ]
    pre, entry = A.emit(setup, A.Flow([A.State()], [A.XState(False)], [[]]))
    # the setup's VALU (the first tile's fragment bases) complete here, once per block: the loop head then waits
    # only for what the back edge leaves in flight (a per-class va_vdst bound), not for every WMMA of the tile before
    pre = drain + pre + ["s_wait_alu depctr_va_vdst(0)"]
    for st in entry.sts:
        A.apply_va(st, 0)
    # the loop control's SCC latency behind the barrier wait (which keeps SCC)
    tail = [f"s_add_co_i32 {S_T}, {S_T}, 1", f"s_cmp_lt_i32 {S_T}, {ops['ncc']}", "s_barrier_wait -1"]

    def merged(flows):
        """One conservative wait state and XCNT state for everything reaching the loop head (a finite lattice: the
        iteration converges), the VALU histories kept apart."""
        j = A.Flow.join(flows)
        xs = A.XState(any(x.unknown for x in j.xs), set().union(*(x.regs for x in j.xs)))
        return A.Flow([A.join_states(j.sts)], [xs], j.valus)

    head = merged([entry])
    for _ in range(16):
        # the O *= corr blocks out of line: 2 taken branches when a q-tile pair rescales, none else (in line, 1 taken
        # branch whenever it does not; random inputs rescale ~half the pairs of a tile, long real sequences rarely)
        lines, ool, ex = _emit_segments(body, L, head, lab, out_of_line=True)
        tl, ex = A.emit(tail, ex)
        nxt = merged([entry, ex])
        if nxt.sig() == head.sig():
            break
        head = nxt
    else:
        raise RuntimeError("TILE_CL: the loop-head state did not converge")
    return (
        pre
        + [f"{lab}_top:"]
        + lines
        + tl
        + [f"s_cbranch_scc1 {lab}_top", f"s_branch {lab}_end"]
        + ool
        + [f"{lab}_end:", "s_wait_xcnt 0x0"]
    )


# ------------------------------------------------------------------------------------------------------------
# the regions as inline asm statements of the FlyDSL kernel (fwd_kernel)
# ------------------------------------------------------------------------------------------------------------
BASE = 40  # the generated code's first VGPR; the kernel's own code stays below it
# TDM ops per wave and ring slot (waves 0..2 one each, wave 3 a NULL op): a tile waits until its slot t + 1 landed,
# with slots t + 2 and t + 3 in flight
TDM_OPS = 1
# the operands of each region in inline asm order: (name, kind), kind v (VGPR), s (SGPR), s4 / s8 (SGPR quad / octet)
_TILE_D_OPS = (("kb", "v"), ("vb", "v"), ("dl", "v"), ("c2", "s"))
_FINAL_OPS = (("wa", "v"), ("wb", "v"), ("la", "v"), ("lb", "v"))
OPERANDS = {
    "INIT": (("qa", "v"), ("qb", "v"), ("c2", "s")),
    # INIT and the block's tile 0 (tile_nodes init): Q fragment bases, K / V^T bases of tile 0, c, ring slot 3's op
    "INIT_C0": (("qa", "v"), ("qb", "v"), ("kb", "v"), ("vb", "v"), ("c2", "s"), ("q", "s4"), ("g1", "s8")),
    # the block's other unmasked tiles as one loop (tile_loop_lines)
    "TILE_CL": (
        ("klane", "v"),
        ("vlane", "v"),
        ("c2", "s"),
        ("stage0", "s"),
        ("cnt", "s"),
        ("ldsq0", "s"),
        ("ldsoff", "s"),
        ("lim", "s"),
        ("ncc", "s"),
        ("n4", "s"),
        ("addr0", "s2"),
        ("step", "s2"),
        ("g1", "s8"),
    ),
    **{f"TILE_D{k}": _TILE_D_OPS for k in range(3)},
    "TILE_D3F": _TILE_D_OPS + _FINAL_OPS,
    "FINAL_A": _FINAL_OPS,
}


def _stand_ins(sig):
    """{operand: stand-in register text} for the analysis: VGPRs v0.. (below BASE), SGPRs from s40 (below the
    clobbered ones), quads / octets aligned."""
    out, v, s = {}, 0, 40
    for name, kind in sig:
        if kind == "v":
            out[name] = f"v{v}"
            v += 1
        else:
            n = {"s": 1, "s2": 2, "s4": 4, "s8": 8}[kind]
            s += (-s) % min(n, 4)
            out[name] = f"s{s}" if n == 1 else f"s[{s}:{s + n - 1}]"
            s += n
    if v > BASE or s > LOOP_CLOBBER_S[0]:  # TILE_CL owns LOOP_CLOBBER_S, every region CLOBBER_S
        raise ValueError("stand-in registers collide with the generated code's")
    return out


def _vfirst(t):
    """First VGPR of an operand (v5, v[8:39])."""
    m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", t.strip())
    if not m:
        raise ValueError(f"not a VGPR operand: {t}")
    return int(m.group(1) or m.group(2))


def emit_region(texts, L):
    """asm.emit on one region, entered with anything in flight (VGPR and memory state unknown, any SGPR possibly read
    by an in-flight VMEM / TDM op). A RESC pseudo-line splits it: the block before ends in s_cmp + s_cbranch over the
    conditional O *= corr block, the block after is entered from both. Labels .LHKR${:uid}_<n>: unique per inline asm
    instance."""
    flow = A.Flow([A.State(unknown=True)], [A.XState(True)], [[]])
    return _emit_segments(texts, L, flow, ".LHKR${:uid}", out_of_line=False)[0]


def region_lines(name, ops, base=BASE):
    """The instruction lines of region ``name`` with the operand registers ops ({name: register text})."""
    L = Layout(base)
    if name == "TILE_CL":
        return tile_loop_lines(L, ops)
    if name == "TILE_D3F":
        o = {x: _vfirst(ops[x]) for x in ("kb", "vb", "dl", "wa", "wb", "la", "lb")}
        o["c2"] = ops["c2"]
        # after it the kernel reads the staging back and rewrites the stores' address VGPRs
        post = ["s_wait_dscnt 0x0", "s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)", "s_wait_xcnt 0x0"]
        return emit_region(region_d3f(L, o), L) + post
    if name == "INIT_C0":
        o = {x: _vfirst(ops[x]) for x in ("qa", "qb", "kb", "vb")}
        o.update(c2=ops["c2"], q=ops["q"], g1=ops["g1"], init=True)
        return emit_region(region_tile(L, o), L) + ["s_wait_xcnt 0x0"]
    if name == "INIT":
        return emit_region(
            region_init(L, {"qa": _vfirst(ops["qa"]), "qb": _vfirst(ops["qb"]), "c2": ops["c2"]}), L
        )
    if name.startswith("TILE"):
        o = {"kb": _vfirst(ops["kb"]), "vb": _vfirst(ops["vb"]), "c2": ops["c2"]}
        k = None
        if name.startswith("TILE_D"):
            k, o["dl"] = int(name[-1]), _vfirst(ops["dl"])
        # every TDM op of the region has read its descriptors when it ends: the kernel may rewrite them
        return emit_region(region_tile(L, o, k), L) + ["s_wait_xcnt 0x0"]
    texts = region_final(L, {k: _vfirst(ops[k]) for k in ("wa", "wb", "la", "lb")})
    # the preceding region's WMMAs may be in flight; after it the kernel reads the staging back (or rewrites the
    # stores' address VGPRs / resource SGPRs)
    pre = ["s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"] + ["v_nop"] * 4
    post = ["s_wait_dscnt 0x0", "s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)", "s_wait_xcnt 0x0"]
    return pre + emit_region(texts, L) + post


_VTOK = re.compile(r"(?<![\w$.\[])v(\d+)\b(?!\s*/\*)")
_STOK = re.compile(r"(?<![\w$.])s\[(\d+):(\d+)\]|(?<![\w$.\[])s(\d+)\b")


def region_label(name):
    """The label at the start of region ``name``'s code (and ``<label>_end`` after it): each region is one inline asm
    statement emitted once per kernel (a second copy would not assemble)."""
    return f"fwd_rgn_{name}"


def region_bytes(name):
    """The region's code size as the kernel's warm-up counts it: 8 bytes per instruction line. The regions' lines
    average 5.6 to 6.4 bytes (4, 8 or 12 each), so their real size is below it."""
    return 8 * sum(1 for x in region_asm(name)[0].split("\n") if x.strip() and not x.strip().endswith(":"))


@functools.lru_cache(maxsize=None)
def region_asm(name, base=BASE):
    """(asm template, constraints) of region ``name`` as one inline asm statement: its operands (OPERANDS[name]) as
    $0.. inputs, clobbering v[base:255] (the generated code's VGPRs that LLVM could allocate), CLOBBER_S (TILE_CL
    also LOOP_CLOBBER_S), vcc, scc and memory. The generated code also uses v256..v1023, which LLVM can neither name
    nor allocate here (the kernel reserves them, fwd_kernel)."""
    sig = OPERANDS[name]
    ops = _stand_ins(sig)
    idx = {txt: i for i, txt in enumerate(ops[n] for n, _ in sig)}
    lines = region_lines(name, ops, base)

    def sub_v(m):
        t = f"v{m.group(1)}"
        if int(m.group(1)) < base:
            if t not in idx:
                raise RuntimeError(f"{name}: VGPR {t} below the generated code's base is not an operand")
            return f"${idx[t]}"
        return t

    def sub_s(m):
        t = m.group(0)
        lo = int(m.group(1) if m.group(1) is not None else m.group(3))
        if lo in CLOBBER_S or (name == "TILE_CL" and lo in LOOP_CLOBBER_S):
            return t
        if t not in idx:
            raise RuntimeError(f"{name}: SGPR {t} is neither an operand nor clobbered")
        return f"${idx[t]}"

    body = [f"{region_label(name)}:"]
    for line in lines:
        code, sep, comment = line.strip().partition(";")
        code = _STOK.sub(sub_s, _VTOK.sub(sub_v, code))
        body.append(code.rstrip() + (f" ;{comment}" if sep else ""))
    body.append(f"{region_label(name)}_end:")
    cons = [{"v": "v"}.get(kind, "s") for _, kind in sig]
    clob = [f"~{{v{r}}}" for r in range(base, 256)] + [f"~{{s{r}}}" for r in CLOBBER_S]
    if name == "TILE_CL":
        clob += [f"~{{s{r}}}" for r in LOOP_CLOBBER_S]
    clob += ["~{vcc}", "~{scc}", "~{memory}"]
    return "\n".join(body), ",".join(cons + clob)
