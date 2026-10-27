###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fwdgen -- the generated regions of the HK MLA forward (W-S1a: no cross-tile pipelining).

The shell (hip/k_mla_fwd.hip) carries inline-asm markers; build.py compiles it, reads the registers LLVM gave each
marker operand, and asks this module for the code of every marker:

  INIT        m = fp32(-1e30), d = 0, O = 0; the (c, c) pair; copies of the Q fragment bases
  TILE_C/M    one 64-kv tile: Q / K / V fragment reads, the ring barrier's arrive (s_wait_dscnt 0, s_wait_tensorcnt 6,
              s_barrier_signal -1) once this wave's K / V reads of the stage are issued (the dscnt wait retires them),
              then per q tile i: QK(i) -> [mask] -> softmax(i) -> O(i) *= corr(i) -> PV(i), list-scheduled
  FINAL_i     q tile i: inv = d > 0 ? 1 / d : 0 (LLVM's IEEE division sequence), lse = (m + log2 d) ln 2 (LLVM's
              log2 expansion), out = bf16(O * inv) into the marker's output registers

The math is FlyDSL's forward op for op (op order read from its gfx1250 ISA, see ref/ref_softmax.py): the QK WMMA
chains over k-steps 0..5 from +0; [mask: s = -inf where kv > q]; lm = max over the lane's 32 values (any tree: max is
exact), lm = max(lm, peer(lm)); need = fma(lm, c, -m) > fp32(log2 e); ballot over the wave; m' = any ? max(m, fp32(c
lm)) : m; corr = exp2(m - m'); p = exp2(fma(s, c, -m')); P = bf16_rne(p); FlyDSL's 3-ary sum tree over the lane's 32 p
(kv16 tile major, then the 8 rows) as packed pairs of q tiles (2 j, 2 j + 1); d = fma(corr, d, ls + peer(ls));
O *= corr (always: corr is exactly 1 when m did not move); O += P V as WMMA chains over the two 32-kv halves.

Program order (the order of the node lists) is the semantics; the list scheduler only reorders along the dependence
graph (RAW / WAR / WAW on every VGPR, SGPR, scc, vcc) plus the barrier's order edges. The output is real instructions
only (no s_set_vgpr_msb, register-driven waits or hazard v_nops): asmkit.emit derives those."""

import os
import re

# test hook (tools/mutants.sh): a named deliberate error, so the CPU checks can be shown to catch it. Never set in a
# build that goes anywhere.
MUTATE = os.environ.get("HKGEN_MUTATE", "")

LOG2E = 0x3FB8AA3B  # fp32(log2 e), FlyDSL's threshold constant
NEG_INF = 0xFF800000
BIG_NEG = 0xF149F2CA  # fp32(-1e30)
LN2 = 0x3F317218  # fp32(1 / log2 e)
SEL_LO = 0x76543210
SEL_HI = 0xFEDCBA98

XK, XV = 400, 272  # LDS row strides of K and V (bytes)
QOFF1 = 64 * XK  # q tile 1 (3) rows are 64 rows after q tile 0 (2)
QT, NKV, NKS, NDT = 4, 4, 6, 8

# SGPRs the shell declares clobbered at every marker (HKGEN_CLOBBERS), and their uses here
S_SEL = "s90"
S_NEED = ("s91", "s92", "s93", "s94")
S_ANY = ("s95", "s96", "s97", "s98")
S_MASK = ("s99", "s100", "s101", "s102", "s103")
CLOBBERED = {f"s{i}" for i in range(90, 104)} | {"vcc", "scc"}


def vr(r, n=1):
    """Operand text of the n VGPRs starting at true register r: the encoded low 8 bits and the true numbers in a
    comment (the compiler's own spelling under s_set_vgpr_msb)."""
    if r // 256 != (r + n - 1) // 256:
        raise ValueError(f"tuple v[{r}:{r + n - 1}] crosses a 256-register block")
    lo = r % 256
    if n == 1:
        return f"v{lo}" if r < 256 else f"v{lo} /*v{r}*/"
    return f"v[{lo}:{lo + n - 1}]" if r < 256 else f"v[{lo}:{lo + n - 1}] /*v[{r}:{r + n - 1}]*/"


class Layout:
    """VGPR map of the generated code (true register numbers), everything at or above `base`."""

    def __init__(self, base):
        nxt = [base + (base % 2)]

        def take(n, align=2):
            r = nxt[0] + ((-nxt[0]) % align)
            nxt[0] = r + n
            if nxt[0] > 256:
                raise ValueError(f"block 0 overflows (base {base})")
            return r

        # block 0: persistent scalars, per-tile scalars, Q fragment buffers, softmax scratch
        self.m = [take(1, 1) for _ in range(4)]
        self.d = [take(1, 1) for _ in range(4)]
        self.cc = take(2)  # (c, c)
        self.qa, self.qb = take(1, 1), take(1, 1)
        self.dl = [None] + [take(1, 1) for _ in range(3)]  # masked tiles: dl + 64 i, i = 1..3
        self.corr = [take(2) for _ in range(4)]  # (corr, junk) pairs: packed ops broadcast the low half
        self.mn = [take(2) for _ in range(4)]  # (m', junk)
        self.qbuf = [take(48, 8) for _ in range(2)]  # Q fragments of q tile i in qbuf[i % 2]
        # per q tile parity: max tree (<= 11 live) + lmx/pl/lm/need/rmax/mfull/dm
        self.sm = [[take(1, 1) for _ in range(18)] for _ in range(2)]
        self.tree = [take(2) for _ in range(11)]  # sum tree (in place by level)
        self.tot = [take(1, 1) for _ in range(4)]  # pl0, tot0, pl1, tot1
        self.fin = [take(1, 1) for _ in range(14)]  # FINAL scratch (+ the inv pair below)
        self.inv = take(2)
        self.end0 = nxt[0]
        self.base = base
        self.o = 256  # block 1: O[i][dt] = 256 + 64 i + 8 dt
        self.v = 512  # block 2: V[dt][kt] = 512 + 16 dt + 8 kt
        self.s = 640  #          S[i][j] = 640 + 32 i + 8 j
        self.k = 768  # block 3: K[j][ks] = 768 + 48 j + 8 ks
        self.pr = 960  #          PR: leaf x of q tile i at 960 + 2 x + (i % 2)

    def O(self, i, dt):
        return self.o + 64 * i + 8 * dt

    def V(self, dt, kt):
        return self.v + 16 * dt + 8 * kt

    def S(self, i, j):
        return self.s + 32 * i + 8 * j

    def K(self, j, ks):
        return self.k + 48 * j + 8 * ks

    def Q(self, i, ks):
        return self.qbuf[i % 2] + 8 * ks

    def PR(self, i, leaf):
        return self.pr + 2 * leaf + (i % 2)

    def P(self, i, kt):
        """bf16 P^T B operand of q tile i, kv half kt: in place in S(i, 2 kt) (dead once its exps have read it)."""
        return self.S(i, 2 * kt)

    def persistent(self):
        """VGPRs whose values live across markers (INIT -> TILEs -> FINALs)."""
        return set(self.m) | set(self.d) | {self.cc, self.cc + 1, self.qa, self.qb} | set(range(256, 512))

    def all_regs(self):
        return set(range(self.base, self.end0)) | set(range(256, 1024))


# --------------------------------------------------------------------------------------------------------------
# instruction nodes and the dependence graph
# --------------------------------------------------------------------------------------------------------------
class Node:
    __slots__ = ("text", "kind", "vdefs", "vuses", "sdefs", "suses", "preds", "idx", "tag")

    def __init__(self, text, kind, tag=""):
        self.text, self.kind, self.tag = text, kind, tag
        self.preds, self.idx = [], -1
        self.vdefs, self.vuses, self.sdefs, self.suses = set(), set(), set(), set()


STOK = re.compile(r"(?<![\w$.])(s\[(\d+):(\d+)\]|s(\d+)|vcc_lo|vcc|scc)(?![\w\[])")


def _vregs(op):
    """True VGPRs of one operand text (the comment's numbers win over the encoded ones)."""
    m = re.search(r"/\*v\[(\d+):(\d+)\]\*/", op)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.search(r"/\*v(\d+)\*/", op)
    if m:
        return [int(m.group(1))]
    m = re.match(r"\s*-?v\[(\d+):(\d+)\]", op)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.match(r"\s*-?v(\d+)\b", op)
    if m:
        return [int(m.group(1))]
    return []


def _sregs(op):
    out = []
    for m in STOK.finditer(re.sub(r"/\*.*?\*/", "", op)):
        if m.group(2) is not None:
            out += [f"s{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
        elif m.group(4) is not None:
            out.append(f"s{m.group(4)}")
        else:
            out.append("vcc" if m.group(1).startswith("vcc") else "scc")
    return out


def split_ops(s):
    out, cur, depth = [], "", 0
    i = 0
    while i < len(s):
        if s.startswith("/*", i):
            j = s.index("*/", i)
            cur += s[i : j + 2]
            i = j + 2
            continue
        c = s[i]
        depth += (c in "[(") - (c in "])")
        if c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def mk(text, kind, tag=""):
    """A node with its register defs / uses read from its text (own parser; build.py cross-checks every emitted line
    against asmkit's parser)."""
    n = Node(text, kind, tag)
    mn, _, rest = text.strip().partition(" ")
    ops = split_ops(rest)
    vd, vu, sd, su = set(), set(), set(), set()
    if kind in ("BAR", "WAIT"):
        pass
    elif mn.startswith("ds_load"):
        vd |= set(_vregs(ops[0]))
        vu |= set(_vregs(ops[1]))
    elif mn.startswith("s_"):
        if mn.startswith("s_cmp"):
            su |= set(_sregs(ops[0])) | set(_sregs(ops[1]))
            sd.add("scc")
        elif mn.startswith("s_cselect"):
            sd |= set(_sregs(ops[0]))
            su.add("scc")
        elif mn.startswith("s_mov"):
            sd |= set(_sregs(ops[0]))
        else:
            raise ValueError(f"unmodelled SALU {text}")
    elif mn.startswith("v_"):
        regs = [_vregs(o) for o in ops]
        if mn.startswith("v_cmp"):
            sd |= set(_sregs(ops[0]))
            vu |= {r for rr in regs[1:] for r in rr}
            su |= {r for o in ops[1:] for r in _sregs(o)}
        else:
            vd |= set(regs[0])
            vu |= {r for rr in regs[1:] for r in rr}
            first_src = 2 if mn.startswith("v_div_scale") else 1
            if mn.startswith("v_div_scale"):
                sd |= {r for r in _sregs(ops[1])}
            for o in ops[first_src:]:
                su |= set(_sregs(o))
            if mn.startswith("v_fmac"):
                vu |= set(regs[0])
            if mn.startswith("v_div_fmas"):
                su.add("vcc")
    else:
        raise ValueError(f"unmodelled instruction {text}")
    n.vdefs, n.vuses, n.sdefs, n.suses = vd, vu, sd, su
    return n


# issue cost (cycles) and result latency per kind (a model; WMMA: 8-cycle XDL occupancy handled by the scheduler)
COST = {"WMMA": 0, "VALU": 2, "TRANS": 2, "PERM": 4, "DS": 1, "SALU": 1, "BAR": 1, "WAIT": 0}
LAT = {"WMMA": 20, "VALU": 5, "TRANS": 8, "PERM": 8, "DS": 120, "SALU": 2, "BAR": 1, "WAIT": 1}


def _key(r):
    return ("v", r) if isinstance(r, int) else ("s", r)


def build_dag(nodes):
    """RAW (producer latency), WAR / WAW (order) edges in list order."""
    last_w, readers = {}, {}
    for i, n in enumerate(nodes):
        n.idx = i
        preds = {}
        for r in list(n.vuses) + list(n.suses):
            k = _key(r)
            if k in last_w:
                p = last_w[k]
                lat = LAT[nodes[p].kind]
                if nodes[p].kind == "WMMA" and n.kind == "WMMA":
                    lat = 8  # accumulator chain
                preds[p] = max(preds.get(p, 0), lat)
        for r in list(n.vdefs) + list(n.sdefs):
            k = _key(r)
            if k in last_w and last_w[k] != i:
                preds[last_w[k]] = max(preds.get(last_w[k], 0), 1)
            for q in readers.get(k, ()):
                if q != i:
                    preds[q] = max(preds.get(q, 0), 1)
        n.preds = sorted(preds.items())
        for r in list(n.vuses) + list(n.suses):
            readers.setdefault(_key(r), []).append(i)
        for r in list(n.vdefs) + list(n.sdefs):
            last_w[_key(r)] = i
            readers[_key(r)] = []
    return nodes


def add_order(nodes, a, b, lat=1):
    """Extra edge a -> b (indices, a < b)."""
    if not a < b:
        raise ValueError("order edges must point forward in program order")
    d = dict(nodes[b].preds)
    d[a] = max(d.get(a, 0), lat)
    nodes[b].preds = sorted(d.items())


DS_MAX = 56  # outstanding LDS loads a schedule allows (the dscnt field is 6 bits)


def schedule(nodes, mode="list", trace=None):
    """(ordered node list, modelled cycles). mode 'program': list order; 'list': cycle-driven list scheduling around
    an in-order WMMA stream QK(0..3), PV(0..3) (WMMAs occupy the XDL pipe 8 cycles), the other instructions picked
    earliest-deadline-first (deadline = the first WMMA that needs them), then by critical path ('cp': critical path
    only); other kinds take COST issue cycles; a TRANS right after a TRANS costs double. Either way LDS loads are kept at <= DS_MAX outstanding (in-order completion: a
    consumer of load L retires every load issued before L; the arrive retires all), with an explicit s_wait_dscnt
    where needed."""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p, l in nd.preds:
            succ[p].append((i, l))
    seq = {}  # DS node -> issue sequence number
    state = {"issued": 0, "retired": -1}

    def consume(nd):
        for p, _ in nd.preds:
            if (
                nodes[p].kind == "DS"
                and nd.kind != "DS"
                and p in seq
                and any(r in nodes[p].vdefs for r in nd.vuses)
            ):
                state["retired"] = max(state["retired"], seq[p])
        if nd.kind == "BAR":
            state["retired"] = state["issued"] - 1

    def ds_room():
        return state["issued"] - 1 - state["retired"] < DS_MAX

    def force_wait(out):
        keep = DS_MAX - 1
        out.append(Node(f"s_wait_dscnt 0x{keep:x}", "WAIT", "dsmax"))
        state["retired"] = max(state["retired"], state["issued"] - 1 - keep)

    if mode == "program":
        out = []
        for nd in nodes:
            if nd.kind == "DS":
                if not ds_room():
                    force_wait(out)
                seq[nd.idx] = state["issued"]
                state["issued"] += 1
            consume(nd)
            out.append(nd)
        return out, None
    cp = [0] * n
    for i in range(n - 1, -1, -1):
        c = 8 if nodes[i].kind == "WMMA" else COST[nodes[i].kind]
        cp[i] = c + max((l + cp[s] for s, l in succ[i]), default=0)
    # the WMMA stream, in order: QK(0..3) then PV(0..3) (program order within each group)
    wseq = [i for i in range(n) if nodes[i].kind == "WMMA" and nodes[i].tag.startswith("QK")]
    wseq += [i for i in range(n) if nodes[i].kind == "WMMA" and nodes[i].tag.startswith("PV")]
    wpos = {i: k for k, i in enumerate(wseq)}
    # deadline of a non-WMMA node: position of the first WMMA that needs it (transitively); none: after the last
    dl = [10**6] * n
    for i in range(n - 1, -1, -1):
        for s_, _ in succ[i]:
            dl[i] = min(dl[i], wpos[s_] if s_ in wpos else dl[s_])
    if mode == "cp":
        prio = lambda i: (nodes[i].kind == "BAR", cp[i])  # noqa: E731
    else:
        prio = lambda i: (nodes[i].kind == "BAR", -dl[i], cp[i])  # noqa: E731
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    t, xdl_free, out, done, last_trans, wk = 0, 0, [], 0, False, 0
    while done < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        if nw is not None and nw in ready and earliest[nw] <= t and xdl_free <= t:
            pick = nw
        else:
            o = [
                i
                for i in ready
                if earliest[i] <= t and nodes[i].kind != "WMMA" and (nodes[i].kind != "DS" or ds_room())
            ]
            if o:
                pick = max(
                    o, key=lambda i: prio(i) + ((-6 if (last_trans and nodes[i].kind == "TRANS") else 0), -i)
                )
        if pick is None:
            nxt = [earliest[i] for i in ready if earliest[i] > t]
            if nw is not None and nw in ready and xdl_free > t:
                nxt.append(xdl_free)
            if not nxt:
                if not any(nodes[i].kind == "DS" for i in ready):
                    raise RuntimeError("scheduler deadlock")
                force_wait(out)
                t += 1
                continue
            t = max(t + 1, min(nxt))
            continue
        nd = nodes[pick]
        ready.discard(pick)
        ti = t
        if trace is not None:
            trace.append((ti, nd))
        if nd.kind == "DS":
            seq[pick] = state["issued"]
            state["issued"] += 1
        consume(nd)
        out.append(nd)
        done += 1
        if nd.kind == "WMMA":
            xdl_free = t + 8
            wk += 1
        else:
            t += COST[nd.kind] + (2 if (last_trans and nd.kind == "TRANS") else 0)
        last_trans = nd.kind == "TRANS"
        for s_, l in succ[pick]:
            earliest[s_] = max(earliest[s_], ti + l)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
    return out, max(t, xdl_free)


# --------------------------------------------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------------------------------------------
def region_init(L, ops):
    """INIT (program order). ops = {'qa': VGPR, 'qb': VGPR, 'c2': SGPR name}."""
    out = []
    for i in range(4):
        out.append(f"v_mov_b32_e32 {vr(L.m[i])}, 0x{BIG_NEG:x}")
        out.append(f"v_mov_b32_e32 {vr(L.d[i])}, 0")
    out.append(f"v_mov_b32_e32 {vr(L.cc)}, {ops['c2']}")
    out.append(f"v_mov_b32_e32 {vr(L.cc + 1)}, {ops['c2']}")
    out.append(f"v_mov_b32_e32 {vr(L.qa)}, {vr(ops['qa'])}")
    out.append(f"v_mov_b32_e32 {vr(L.qb)}, {vr(ops['qb'])}")
    for r in range(L.o, L.o + 256):
        out.append(f"v_mov_b32_e32 {vr(r)}, 0")
    return out


def tile_nodes(L, ops, masked, first=0, mask_only=None):
    """The nodes of one tile in program order (which defines the semantics). ops: {'kb', 'vb'[, 'dl']}: the VGPRs of
    the marker operands (K / V fragment bases of the stage, dl = q + Skv - Sq - kv0 - 8 (lane / 16) of q tile 0).
    first: q tiles below it are skipped (diagonal tile ncc + first of a TILE_D body); mask_only: None = the mask on
    every q tile (TILE_M), else on that q tile only."""
    active = [i for i in range(QT) if i >= first]
    masked_q = [i for i in active if masked and (mask_only is None or i == mask_only)]
    nodes = []

    def add(text, kind, tag=""):
        nd = mk(text, kind, tag)
        nodes.append(nd)
        return nd

    add(f"s_mov_b32 {S_SEL}, 0x{SEL_LO:x}", "SALU", "sel")
    dl = [ops.get("dl")] + L.dl[1:]
    for i in masked_q:
        if i > 0:
            add(f"v_add_nc_u32_e32 {vr(dl[i])}, {64 * i}, {vr(ops['dl'])}", "VALU", "dl")
    kv_loads = []
    for ks in range(NKS):  # K fragments, k-step major (FlyDSL's issue order)
        for j in range(NKV):
            for h in range(2):
                off = (
                    j * 16 * XK
                    + ks * 64
                    + 32 * h
                    + (16 if (MUTATE == "koff" and (j, ks, h) == (1, 2, 0)) else 0)
                )
                kv_loads.append(
                    add(
                        f"ds_load_b128 {vr(L.K(j, ks) + 4 * h, 4)}, {vr(ops['kb'])} offset:{off}",
                        "DS",
                        f"K{j}.{ks}.{h}",
                    )
                )
    for dt in range(NDT):  # V^T fragments (transposing loads, FlyDSL's lane pattern)
        for kt in range(2):
            for h in range(2):
                off = (32 * kt + 16 * h) * XV + dt * 32
                kv_loads.append(
                    add(
                        f"ds_load_tr16_b128 {vr(L.V(dt, kt) + 4 * h, 4)}, {vr(ops['vb'])} offset:{off}",
                        "DS",
                        f"V{dt}.{kt}.{h}",
                    )
                )

    def q_loads(i):
        qbase = L.qa if i < 2 else L.qb
        return [
            add(
                f"ds_load_b128 {vr(L.Q(i, ks) + 4 * h, 4)}, {vr(qbase)} offset:{(i & 1) * QOFF1 + ks * 64 + 32 * h}",
                "DS",
                f"Q{i}.{ks}.{h}",
            )
            for ks in range(NKS)
            for h in range(2)
        ]

    q01 = [x for i in (0, 1) if i in active for x in q_loads(i)]
    bar = add("s_barrier_signal -1", "BAR", "arrive")
    q23 = []
    for i in active:
        if i >= 2:
            q23 += q_loads(i)
        for ks in range(NKS):  # QK(i): S[i][j] = sum_ks K[j][ks] Q[i][ks] (chains from +0)
            for j in range(NKV):
                c = "0" if ks == 0 else vr(L.S(i, j), 8)
                kk = {1: 2, 2: 1}.get(ks, ks) if (MUTATE == "chain" and i == 1 and j == 3) else ks
                add(
                    f"v_wmma_f32_16x16x32_bf16 {vr(L.S(i, j), 8)}, {vr(L.K(j, kk), 8)}, {vr(L.Q(i, kk), 8)}, {c}",
                    "WMMA",
                    f"QK{i}",
                )
        softmax_nodes(L, i, dl[i] if i in masked_q else None, add, active)
        for kt in range(2):  # PV(i): O[i][dt] += V[dt][kt] P[i][kt], kt = 0 then 1
            for dt in range(NDT):
                o = vr(L.O(i, dt), 8)
                add(
                    f"v_wmma_f32_16x16x32_bf16 {o}, {vr(L.V(dt, kt), 8)}, {vr(L.P(i, kt), 8)}, {o}",
                    "WMMA",
                    f"PV{i}",
                )
    build_dag(nodes)
    # the arrive: after every K / V read of the stage (about one LDS latency after the last, so its s_wait_dscnt 0
    # does not stall) and after the Q reads of q tiles 0, 1; the Q reads of q tiles 2, 3 after it (so no fresh LDS
    # read is in flight at the arrive)
    for nd in kv_loads:
        add_order(nodes, nd.idx, bar.idx, LAT["DS"])
    for nd in q01:
        add_order(nodes, nd.idx, bar.idx, 1)
    for nd in q23:
        add_order(nodes, bar.idx, nd.idx, 1)
    return nodes


def softmax_nodes(L, i, dl, add, active=(0, 1, 2, 3)):
    S = [[L.S(i, j) + e for e in range(8)] for j in range(NKV)]
    sc = L.sm[i % 2]
    if dl is not None:  # s = -inf where 16 j + e > dl (kv > q)
        k = 0
        for j in range(NKV):
            for e in range(8):
                sm = S_MASK[k % len(S_MASK)]
                k += 1
                if MUTATE == "mask" and i == 0 and j == 0 and e == 5:
                    continue
                add(f"v_cmp_gt_i32_e64 {sm}, {16 * j + e}, {vr(dl)}", "VALU", f"mask{i}")
                add(
                    f"v_cndmask_b32_e64 {vr(S[j][e])}, {vr(S[j][e])}, 0x{NEG_INF:x}, {sm}", "VALU", f"mask{i}"
                )
    # row max over the lane's 32 values (exact; any tree): 32 -> 11 -> 4 -> 2 -> 1 in scratch sc[0..10]
    vals, slot = [S[j][e] for j in range(NKV) for e in range(8)], 0
    while len(vals) > 1:
        nxt = []
        k = 0
        for a in range(0, len(vals), 3):
            grp = vals[a : a + 3]
            if len(grp) == 1:
                nxt.append(grp[0])
                continue
            r = sc[k]
            k += 1
            if len(grp) == 3:
                add(f"v_max3_num_f32 {vr(r)}, {vr(grp[0])}, {vr(grp[1])}, {vr(grp[2])}", "VALU", f"max{i}")
            else:
                add(f"v_max_num_f32_e32 {vr(r)}, {vr(grp[0])}, {vr(grp[1])}", "VALU", f"max{i}")
            nxt.append(r)
        vals = nxt
        slot += 1
    lmx, pl, lm, need, rmax, mfull, dm = vals[0], sc[11], sc[12], sc[13], sc[14], sc[15], sc[16]
    add(f"v_permlanex16_b32 {vr(pl)}, {vr(lmx)}, {S_SEL}, 0x{SEL_HI:x}", "PERM", f"max{i}")
    add(f"v_max_num_f32_e32 {vr(lm)}, {vr(lmx)}, {vr(pl)}", "VALU", f"max{i}")
    add(f"v_fma_f32 {vr(need)}, {vr(lm)}, {vr(L.cc)}, -{vr(L.m[i])}", "VALU", f"dec{i}")
    add(f"v_mul_f32_e32 {vr(rmax)}, {vr(L.cc)}, {vr(lm)}", "VALU", f"dec{i}")
    add(f"v_max_num_f32_e32 {vr(mfull)}, {vr(L.m[i])}, {vr(rmax)}", "VALU", f"dec{i}")
    add(f"v_cmp_lt_f32_e64 {S_NEED[i]}, 0x{LOG2E:x}, {vr(need)}", "VALU", f"dec{i}")
    add(f"s_cmp_lg_u32 {S_NEED[i]}, 0", "SALU", f"dec{i}")
    add(f"s_cselect_b32 {S_ANY[i]}, -1, 0", "SALU", f"dec{i}")
    mn, corr = L.mn[i], L.corr[i]
    add(f"v_cndmask_b32_e64 {vr(mn)}, {vr(L.m[i])}, {vr(mfull)}, {S_ANY[i]}", "VALU", f"dec{i}")
    add(f"v_sub_f32_e32 {vr(dm)}, {vr(L.m[i])}, {vr(mn)}", "VALU", f"dec{i}")
    add(f"v_exp_f32_e32 {vr(corr)}, {vr(dm)}", "TRANS", f"dec{i}")
    add(f"v_mov_b32_e32 {vr(L.m[i])}, {vr(mn)}", "VALU", f"dec{i}")
    # p = exp2(fma(s, c, -m')): packed fma in place, then exp into the pair buffer (leaf 8 j + e)
    for j in range(NKV):
        for e in range(0, 8, 2):
            add(
                f"v_pk_fma_f32 {vr(S[j][e], 2)}, {vr(S[j][e], 2)}, {vr(L.cc, 2)}, "
                f"{vr(L.corr[i], 2) if (MUTATE == 'exparg' and i == 2) else vr(mn, 2)} "
                f"op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]",
                "VALU",
                f"p{i}",
            )
            for x in range(2):
                add(f"v_exp_f32_e32 {vr(L.PR(i, 8 * j + e + x))}, {vr(S[j][e + x])}", "TRANS", f"p{i}")
    for kt in range(2):  # P^T (bf16) for PV, in place in S(i, 2 kt)
        for r in range(8):
            j, e = 2 * kt + r // 4, 2 * (r % 4)
            add(
                f"v_cvt_pk_bf16_f32 {vr(L.P(i, kt) + r)}, {vr(L.PR(i, 8 * j + e))}, {vr(L.PR(i, 8 * j + e + 1))}",
                "VALU",
                f"cvt{i}",
            )
    for dt in range(NDT):  # O *= corr (always)
        for r in range(0, 8, 2):
            o = vr(L.O(i, dt) + r, 2)
            add(f"v_pk_mul_f32 {o}, {o}, {vr(corr, 2)} op_sel_hi:[1,0]", "VALU", f"resc{i}")
    if i % 2 == 1:  # the pair's tree (an inactive lo tile: its half is computed on stale values and not used)
        sum_nodes(L, i - 1, add, [x for x in (i - 1, i) if x in active])


def sum_nodes(L, i0, add, upd=None):
    """FlyDSL's sum tree over the 32 leaves of q tiles (i0, i0 + 1), packed (lo = i0), then d for both."""
    leaves = [L.pr + 2 * x for x in range(32)]
    level = 0
    while len(leaves) > 1:
        nxt, a, k = [], 0, 0
        while a < len(leaves):
            rem = len(leaves) - a
            if rem >= 2:
                # level 1 writes the tree slots; later levels write in place into their first operand's slot
                r = L.tree[k] if level == 0 else leaves[a]
                k += 1
                b1, b2 = (a + 2, a + 1) if (MUTATE == "tree" and level == 0 and a == 0) else (a + 1, a + 2)
                add(f"v_pk_add_f32 {vr(r, 2)}, {vr(leaves[a], 2)}, {vr(leaves[b1], 2)}", "VALU", f"sum{i0}")
                if rem >= 3:
                    add(f"v_pk_add_f32 {vr(r, 2)}, {vr(r, 2)}, {vr(leaves[b2], 2)}", "VALU", f"sum{i0}")
                    a += 3
                else:
                    a += 2
                nxt.append(r)
            else:
                nxt.append(leaves[a])
                a += 1
        leaves = nxt
        level += 1
    ls = leaves[0]
    for x in range(2):
        qi = i0 + x
        if upd is not None and qi not in upd:
            continue
        pl, tot = L.tot[2 * x], L.tot[2 * x + 1]
        add(f"v_permlanex16_b32 {vr(pl)}, {vr(ls + x)}, {S_SEL}, 0x{SEL_HI:x}", "PERM", f"sum{qi}")
        add(f"v_add_f32_e32 {vr(tot)}, {vr(ls + x)}, {vr(pl)}", "VALU", f"sum{qi}")
        cq = L.corr[i0 + 1 - x] if MUTATE == "corr" else L.corr[qi]
        add(f"v_fma_f32 {vr(L.d[qi])}, {vr(cq)}, {vr(L.d[qi])}, {vr(tot)}", "VALU", f"sum{qi}")


def region_tile(L, ops, masked, mode="list", first=0, mask_only=None):
    """(lines, modelled cycles) of a TILE region."""
    nodes = tile_nodes(L, ops, masked, first, mask_only)
    order, cyc = schedule(nodes, mode)
    out = []
    for nd in order:
        if nd.kind == "BAR":  # the arrive: this wave's LDS reads retired, wave 0's tile t + 1 landed
            out += ["s_wait_dscnt 0x0", "s_wait_tensorcnt 0x6", nd.text]
        else:
            out.append(nd.text)
    return out, cyc


def region_final(L, ops, i):
    """FINAL_i (program order). ops = {'o': first of the 32 output VGPRs, 'lse': output VGPR}."""
    d, m = L.d[i], L.m[i]
    a, b, rc, e1, q, e2, sh, sub, x, lg, s = L.fin[:11]
    inv = L.inv
    out = [  # LLVM's fp32 1.0 / d (denormals on: no mode switches), then d > 0 ? . : 0
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
        f"v_cmp_lt_f32_e64 {S_NEED[0]}, 0, {vr(d)}",
        f"v_cndmask_b32_e64 {vr(inv)}, 0, {vr(a)}, {S_NEED[0]}",
    ]
    for dt in range(NDT):
        for r in range(0, 8, 2):
            o = vr(L.O(i, dt) + r, 2)
            out.append(f"v_pk_mul_f32 {o}, {o}, {vr(inv, 2)} op_sel_hi:[1,0]")
            out.append(
                f"v_cvt_pk_bf16_f32 {vr(ops['o'] + 4 * dt + r // 2)}, {vr(L.O(i, dt) + r)}, "
                f"{vr(L.O(i, dt) + r + 1)}"
            )
    out += [  # LLVM's log2: v_log_f32 of d scaled by 2^32 below 2^-126
        f"v_cmp_gt_f32_e64 {S_NEED[1]}, 0x800000, {vr(d)}",
        f"v_cndmask_b32_e64 {vr(sh)}, 0, 32, {S_NEED[1]}",
        f"v_cndmask_b32_e64 {vr(sub)}, 0, 0x42000000, {S_NEED[1]}",
        f"v_ldexp_f32 {vr(x)}, {vr(d)}, {vr(sh)}",
        f"v_log_f32_e32 {vr(x)}, {vr(x)}",
        f"v_sub_f32_e32 {vr(lg)}, {vr(x)}, {vr(sub)}",
        f"v_add_f32_e32 {vr(s)}, {vr(m)}, {vr(lg)}",
        f"v_mul_f32_e32 {vr(ops['lse'])}, 0x{LN2 + (1 if MUTATE == 'lse' else 0):x}, {vr(s)}",
    ]
    return out
