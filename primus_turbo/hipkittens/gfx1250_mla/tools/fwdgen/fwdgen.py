###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fwdgen -- the generated regions of the HK MLA forward kernel ``k_mla_fwd``.

The shell (``hip/src/k_mla_fwd.hip``) carries inline-asm
markers; build.py compiles it, reads the registers LLVM gave each marker operand and asks this module for
every region:

  INIT     once per q block: m = fp32(-1e30), d = 0, O = 0, the (c, c) pair, copies of the Q fragment bases,
           the resident Q fragments of q tiles 1, 2 and the state that makes the first tile's prefix a no-op
  TILE_C   a 64-kv tile below the diagonal (no mask)
  TILE_Dk  diagonal tile k = 0..3 (Sq == Skv): q tiles below k skipped, the causal mask on q tile k only
  FINAL_i  q tile i: inv = d > 0 ? 1 / d : 0 (LLVM's IEEE division sequence), lse = (m + log2 d) ln 2 (LLVM's
           log2 expansion), out = bf16(O inv) into the marker's output registers; FINAL_0 first runs the
           prefix of the last tile

A wave owns four 16-row q tiles; c = scale log2 e. The math is FlyDSL's forward, op for op: the QK WMMA chains
over k-steps 0..5 from +0; [mask: s = -inf where kv > q]; lm = max over the lane's 32 values (any tree: max is
exact), lm = max(lm, peer(lm)); need = fma(lm, c, -m) > fp32(log2 e); ballot over the wave;
m' = any ? max(m, fp32(c lm)) : m; corr = exp2(m - m'); p = exp2(fma(s, c, -m')); P = bf16_rne(p); FlyDSL's
3-ary sum tree over the lane's 32 p (kv16 tile major, then the 8 rows), packed over the q tile pairs (0, 1)
and (2, 3); d = fma(corr, d, ls + peer(ls)); O *= corr (always: corr is exactly 1 when m did not move);
O += P V as WMMA chains over the two 32-kv halves.

The TILE regions are software-pipelined across tiles. The region of tile t opens with a prefix that finishes
tile t - 1: the softmax of q tiles 2, 3 (their masked scores are live in S(2), S(3)), their sum tree and
O *= corr, then the PV WMMAs of all four q tiles (P(0), P(1) come from region t - 1). Its body reads the
tile's K / V / Q fragments and runs QK(0..3), the full softmax of q tiles 0, 1 (P in place of their scores)
with their sum tree, and only the mask of q tiles 2, 3. The K / Q(0) reads issued at the region start thus
land while the prefix's 64 WMMAs and its softmax run, and the softmax VALU spreads over the whole region.
Every q tile keeps its sequence of operations: the pipelining changes no result bit. A body that skips q
tile i leaves the next prefix a no-op for it, as INIT does for the first prefix: P(i) = 0 for i < 2,
S(i) = -inf for i >= 2 (corr = 1, p = +0, d = fma(1, d, +0), O += V * 0).

Program order (the order of the node lists) is the semantics: the scheduler reorders only along the dependence
graph (RAW / WAR / WAW on every VGPR, SGPR, scc, vcc) and the added order edges. The output is real
instructions only; asmkit.emit derives s_set_vgpr_msb, the expert-mode waits and the hazard v_nops."""

import re

LOG2E = 0x3FB8AA3B  # fp32(log2 e): the rescale threshold
NEG_INF = 0xFF800000
BIG_NEG = 0xF149F2CA  # fp32(-1e30): the initial m
LN2 = 0x3F317218  # fp32(1 / log2 e)
SEL_LO, SEL_HI = 0x76543210, 0xFEDCBA98  # v_permlanex16_b32 selects: lane l reads lane l of the other half

XK, XV = 400, 272  # LDS row strides of K and V (bytes)
QOFF1 = 64 * XK  # q tile 1 (3) rows are 64 rows after q tile 0 (2)
QT, NKV, NKS, NDT = 4, 4, 6, 8  # q tiles per wave, kv16 tiles per tile, QK k-steps, PV d tiles

# Scalar temporaries: the SGPRs every marker declares clobbered (HKGEN_CLOBBERS), plus vcc and scc
S_SEL = "s90"  # SEL_LO for v_permlanex16_b32
S_NEED = ("s91", "s92", "s93", "s94")  # per q tile: the lanes whose row max grew past the threshold
S_ANY = ("s95", "s96", "s97", "s98")  # per q tile: -1 if any lane did, else 0
S_MASK = ("s99", "s100", "s101")  # causal-mask compares, in rotation
S_C = ("s102", "s103")  # (c, c) in the TILE regions, copied from the marker's c2 operand
CLOBBERED = {f"s{i}" for i in range(90, 104)} | {"vcc", "scc"}

# Scheduler model (cycles)
DS_LAT = 290  # LDS load-to-use latency
DS_RATE = 6  # one LDS load issued per DS_RATE cycles
DS_MAX = 56  # LDS loads in flight at most (the dscnt field is 6 bits)
WORDER = ("PV0p", "PV1p", "PV2p", "PV3p", "QK0", "QK1", "QK2", "QK3")  # the in-order WMMA stream, by node tag
# latency of a dependence edge by producer kind (it ranks nodes by critical path), and result latency by kind
# in the issue model
LAT = {"WMMA": 20, "VALU": 5, "TRANS": 8, "PERM": 8, "DS": DS_LAT, "SALU": 2}
SLOT_LAT = {"WMMA": 20, "VALU": 3, "TRANS": 8, "PERM": 6, "SALU": 2}


def vr(r, n=1):
    """Operand text of the n VGPRs from true register r: the encoded low 8 bits, the true numbers in a comment
    (the compiler's spelling under s_set_vgpr_msb)."""
    if r // 256 != (r + n - 1) // 256:
        raise ValueError(f"tuple v[{r}:{r + n - 1}] crosses a 256-register block")
    lo = r % 256
    if n == 1:
        return f"v{lo}" if r < 256 else f"v{lo} /*v{r}*/"
    return f"v[{lo}:{lo + n - 1}]" if r < 256 else f"v[{lo}:{lo + n - 1}] /*v[{r}:{r + n - 1}]*/"


class Layout:
    """VGPR map of the generated code (true register numbers): block 0 from `base` up, blocks 1..3 whole.

    Live across regions: m, d, (c, c), the Q bases and O; from one TILE region to the next (and into FINAL_0)
    also V, P(0), P(1), S(2), S(3) and the resident Q fragments of q tiles 1, 2."""

    def __init__(self, base):
        nxt = [base + (base % 2)]

        def take(n, align=2):
            r = nxt[0] + ((-nxt[0]) % align)
            nxt[0] = r + n
            if nxt[0] > 256:
                raise ValueError(f"block 0 overflows (base {base})")
            return r

        self.base = base
        self.m = [take(1, 1) for _ in range(QT)]
        self.d = [take(1, 1) for _ in range(QT)]
        self.cc = take(2)  # (c, c)
        self.qa, self.qb = take(1, 1), take(1, 1)  # Q fragment bases of q tiles 0, 1 and 2, 3
        self.dl = [None] + [take(1, 1) for _ in range(3)]  # mask bound of q tile i: dl + 64 i
        self.corr = [take(2) for _ in range(QT)]  # (corr, unused): packed ops broadcast the low half
        self.mn = [take(2) for _ in range(QT)]  # (m', unused)
        self.qbuf = [take(48, 8) for _ in range(3)]  # Q fragments, see Q()
        self.sm = [[take(1, 1) for _ in range(18)] for _ in range(2)]  # softmax scratch per q tile parity
        self.tot = [take(1, 1) for _ in range(4)]  # (peer sum, total) of the two q tiles of a sum tree
        self.end0 = nxt[0]
        # FINAL scratch and the (inv, unused) pair: qbuf[0] is dead once the last tile's QK has read it
        self.fin = [self.qbuf[0] + k for k in range(11)]
        self.inv = self.qbuf[0] + 14
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
        """Q fragments of q tile i: 1 and 2 resident in qbuf[1], qbuf[2]; 0 and 3 share qbuf[0]."""
        return self.qbuf[(0, 1, 2, 0)[i]] + 8 * ks

    def PR(self, i, leaf):
        """p of q tile i, leaf 8 j + e; a pair's two q tiles interleave, the sum tree adds both at once."""
        return self.pr + 2 * leaf + (i % 2)

    def P(self, i, kt):
        """bf16 P^T (B operand) of q tile i, kv half kt: in place of S(i, 2 kt) once the exps have read it."""
        return self.S(i, 2 * kt)

    def all_regs(self):
        return set(range(self.base, self.end0)) | set(range(256, 1024))


# ------------------------------------------------------------------------------------------------------------
# instruction nodes and the dependence graph
# ------------------------------------------------------------------------------------------------------------
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


def _split_ops(s):
    out, cur, depth, i = [], "", 0, 0
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
    """A node with the registers its text defines and uses. Covers the forms the regions schedule: ds_load
    (vdst, vaddr), s_mov / s_cmp / s_cselect, v_cmp_*_e64 with an SGPR destination, and VALU / WMMA ops whose
    first operand is their only destination and that read no implicit operand."""
    n = Node(text, kind, tag)
    if kind == "BAR":
        return n
    mn, _, rest = text.strip().partition(" ")
    ops = _split_ops(rest)
    if mn.startswith("ds_load"):
        n.vdefs, n.vuses = set(_vregs(ops[0])), set(_vregs(ops[1]))
    elif mn.startswith("s_cmp"):
        n.sdefs, n.suses = {"scc"}, set(_sregs(ops[0])) | set(_sregs(ops[1]))
    elif mn.startswith("s_cselect"):
        n.sdefs, n.suses = set(_sregs(ops[0])), {"scc"}
    elif mn.startswith("s_mov"):
        n.sdefs, n.suses = set(_sregs(ops[0])), set(_sregs(ops[1]))
    elif mn.startswith("v_"):
        if mn.startswith("v_cmp"):
            n.sdefs = set(_sregs(ops[0]))
        else:
            n.vdefs = set(_vregs(ops[0]))
        n.vuses = {r for o in ops[1:] for r in _vregs(o)}
        n.suses = {r for o in ops[1:] for r in _sregs(o)}
    else:
        raise ValueError(f"unmodelled instruction {text}")
    return n


def _key(r):
    return ("v", r) if isinstance(r, int) else ("s", r)


def build_dag(nodes):
    """RAW (producer latency), WAR / WAW (latency 1) edges in list order: node.preds = [(index, latency)]."""
    last_w, readers = {}, {}
    for i, n in enumerate(nodes):
        n.idx = i
        preds = {}
        for r in list(n.vuses) + list(n.suses):
            k = _key(r)
            if k in last_w:
                p = last_w[k]
                lat = 8 if nodes[p].kind == n.kind == "WMMA" else LAT[nodes[p].kind]  # 8: accumulator chain
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
    """(ordered nodes, modelled cycles): a cycle-stepped list scheduler on an issue model of one wave.

    WMMAs issue in WORDER (program order within a tag), one per 8 cycles at most (XDL occupancy). Within 8
    cycles of a WMMA, a VALU / TRANS issues only at offsets 2, 3, 6, 7 and v_permlanex16 only at 8; other
    nodes issue at any offset but 0. A TRANS blocks VALU / TRANS / permlane the next cycle; a permlane takes 2
    cycles and goes before a ready WMMA (else a dense WMMA stream starves it); one SALU issues per 2 cycles.
    LDS loads issue one per DS_RATE cycles and complete in order DS_LAT cycles after issue, at most DS_MAX in
    flight: a consumer of a load retires it and every older one, the arrive (s_barrier_signal behind
    s_wait_dscnt 0) retires all. Other results are ready SLOT_LAT cycles after issue. Among the ready nodes:
    a VALU at offsets 2, 6, a TRANS at 3, 7, an LDS load elsewhere first; then the arrive, the earliest
    deadline (the first WMMA of the stream that needs the node), the longest critical path (LAT), program
    order. An s_wait_dscnt goes in where a load would exceed DS_MAX."""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p, lat in nd.preds:
            succ[p].append((i, lat))
    wmmas = [i for i in range(n) if nodes[i].kind == "WMMA"]
    wseq = [i for tag in WORDER for i in wmmas if nodes[i].tag == tag]
    if sorted(wseq) != wmmas:
        raise ValueError("WORDER does not cover the WMMAs")
    wpos = {i: k for k, i in enumerate(wseq)}
    deadline = [10**6] * n
    for i in range(n - 1, -1, -1):
        for s, _ in succ[i]:
            deadline[i] = min(deadline[i], wpos[s] if s in wpos else deadline[s])
    cp = [0] * n
    for i in range(n - 1, -1, -1):
        cp[i] = (8 if nodes[i].kind == "WMMA" else 1) + max((lat + cp[s] for s, lat in succ[i]), default=0)
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
            return (not 0 <= o <= 8 or o in (2, 3, 6, 7)) and t != trans_at + 1 and t > perm_busy
        if k == "PERM":
            return o >= 8 and t != trans_at + 1 and t > perm_busy
        if k == "DS":
            return o != 0 and t >= ds_next and len(issued_ds) - retired_ds < DS_MAX
        if k == "SALU":
            return o != 0 and t >= salu_free
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
                if o in (2, 3, 6, 7):
                    vt = [i for i in cand if nodes[i].kind in ("VALU", "TRANS")]
                    if vt:
                        want = "VALU" if o in (2, 6) else "TRANS"
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
        elif k == "SALU":
            salu_free = t + 2
        for s, lat in succ[pick]:
            if k == "DS":
                e = ds_done[pick] if nd.vdefs & nodes[s].vuses else t + 1
                if lat > 1 and nodes[s].kind == "BAR":
                    e = max(e, ds_done[pick])
            elif lat <= 1:
                e = t + 1  # WAR / WAW / order edge
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
    return res, max(t, last_w + 8)


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
    """INIT (program order). ops: qa / qb (VGPRs: Q fragment bases of q tiles 0, 1 and 2, 3), c2 (SGPR)."""
    out = []
    for i in range(QT):
        out.append(f"v_mov_b32_e32 {vr(L.m[i])}, 0x{BIG_NEG:x}")
        out.append(f"v_mov_b32_e32 {vr(L.d[i])}, 0")
    out.append(f"v_mov_b32_e32 {vr(L.cc)}, {ops['c2']}")
    out.append(f"v_mov_b32_e32 {vr(L.cc + 1)}, {ops['c2']}")
    out.append(f"v_mov_b32_e32 {vr(L.qa)}, {vr(ops['qa'])}")
    out.append(f"v_mov_b32_e32 {vr(L.qb)}, {vr(ops['qb'])}")
    out += [f"v_mov_b32_e32 {vr(r)}, 0" for r in range(L.o, L.o + 256)]
    # the resident Q fragments of q tiles 1, 2: the Q staging area does not change until the block's last tile
    out += [q_load(L, i, ks, h) for i in (1, 2) for ks in range(NKS) for h in range(2)]
    # the first prefix is a no-op: S(2), S(3) = -inf; P(0), P(1) = 0; V = 0
    out += [f"v_mov_b32_e32 {vr(r)}, 0x{NEG_INF:x}" for i in (2, 3) for r in range(L.S(i, 0), L.S(i, 0) + 32)]
    out += [f"v_mov_b32_e32 {vr(L.P(i, kt) + e)}, 0" for i in (0, 1) for kt in range(2) for e in range(8)]
    out += [f"v_mov_b32_e32 {vr(r)}, 0" for r in range(L.v, L.v + 128)]
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


def softmax_nodes(L, i, add, c_sgpr):
    """The softmax of q tile i up to P and O *= corr: scores in S(i), p into the leaves PR(i). c from the SGPR
    pair S_C (c_sgpr: the packed fma then has two VGPR sources, not three) or from the VGPR pair L.cc."""
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
    for dt in range(NDT):  # O *= corr
        for r in range(0, 8, 2):
            o = vr(L.O(i, dt) + r, 2)
            add(f"v_pk_mul_f32 {o}, {o}, {vr(corr, 2)} op_sel_hi:[1,0]", "VALU", f"resc{i}")


def sum_nodes(L, i0, add, upd):
    """FlyDSL's sum tree over the 32 leaves of q tiles (i0, i0 + 1), packed (low half i0), in place in the
    leaves (the cvts have read them); then d of the q tiles in upd (one outside it has stale leaves here)."""
    leaves = [L.pr + 2 * x for x in range(32)]
    while len(leaves) > 1:  # (a + b) + c per group of 3; a trailing pair a + b, a trailing single passed on
        groups = [leaves[a : a + 3] for a in range(0, len(leaves), 3)]
        for g in groups:
            for y in g[1:]:
                add(f"v_pk_add_f32 {vr(g[0], 2)}, {vr(g[0], 2)}, {vr(y, 2)}", "VALU", f"sum{i0}")
        leaves = [g[0] for g in groups]
    for x in range(2):
        qi = i0 + x
        if qi not in upd:
            continue
        pl, tot = L.tot[2 * x], L.tot[2 * x + 1]
        add(f"v_permlanex16_b32 {vr(pl)}, {vr(leaves[0] + x)}, {S_SEL}, 0x{SEL_HI:x}", "PERM", f"sum{qi}")
        add(f"v_add_f32_e32 {vr(tot)}, {vr(leaves[0] + x)}, {vr(pl)}", "VALU", f"sum{qi}")
        add(f"v_fma_f32 {vr(L.d[qi])}, {vr(L.corr[qi])}, {vr(L.d[qi])}, {vr(tot)}", "VALU", f"sum{qi}")


def prefix_nodes(L, add, c_sgpr):
    """The tail of the previous tile: the softmax of q tiles 2, 3 (masked scores in S(2), S(3)), their sum
    tree, then the PV WMMAs of all four q tiles (V of the previous tile), tags PV{i}p."""
    softmax_nodes(L, 2, add, c_sgpr)
    softmax_nodes(L, 3, add, c_sgpr)
    sum_nodes(L, 2, add, (2, 3))
    for i in range(QT):
        for kt in range(2):
            for dt in range(NDT):
                o = vr(L.O(i, dt), 8)
                add(
                    f"v_wmma_f32_16x16x32_bf16 {o}, {vr(L.V(dt, kt), 8)}, {vr(L.P(i, kt), 8)}, {o}",
                    "WMMA",
                    f"PV{i}p",
                )


def tile_nodes(L, ops, k=None):
    """The nodes of a TILE region in program order. ops: kb / vb (VGPRs: the K / V fragment bases of the
    stage), c2 (SGPR: c), and for a diagonal tile dl (VGPR: q + Skv - Sq - kv0 - 8 (lane / 16) of q tile 0).
    k: None for TILE_C, else the diagonal tile k (q tiles below k skipped, the mask on q tile k only)."""
    first = 0 if k is None else k
    nodes, add = node_list()
    add(f"s_mov_b32 {S_SEL}, 0x{SEL_LO:x}", "SALU", "sel")
    for r in S_C:
        add(f"s_mov_b32 {r}, {ops['c2']}", "SALU", "c2")
    # the K (k-step major: FlyDSL's issue order) and Q(0) reads first; the prefix reads neither K nor qbuf[0]
    loads = [
        add(k_load(L, ops["kb"], j, ks, h), "DS", f"K{j}.{ks}.{h}")
        for ks in range(NKS)
        for j in range(NKV)
        for h in range(2)
    ]
    if first == 0:
        loads += [add(q_load(L, 0, ks, h), "DS", f"Q0.{ks}.{h}") for ks in range(NKS) for h in range(2)]
    prefix_nodes(L, add, c_sgpr=True)
    for i in range(first):  # the next prefix must be a no-op for the q tiles this body skips
        if i < 2:
            for kt in range(2):
                for e in range(8):
                    add(f"v_mov_b32_e32 {vr(L.P(i, kt) + e)}, 0", "VALU", f"skip{i}")
        else:
            for r in range(L.S(i, 0), L.S(i, 0) + 32):
                add(f"v_mov_b32_e32 {vr(r)}, 0x{NEG_INF:x}", "VALU", f"skip{i}")
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
        if i == 3:  # into qbuf[0], after QK(0) has read Q(0)
            loads += [add(q_load(L, 3, ks, h), "DS", f"Q3.{ks}.{h}") for ks in range(NKS) for h in range(2)]
        for ks in range(NKS):  # QK(i): S[i][j] = sum_ks K[j][ks] Q[i][ks], chains from +0
            for j in range(NKV):
                s = vr(L.S(i, j), 8)
                c = "0" if ks == 0 else s
                qk = f"v_wmma_f32_16x16x32_bf16 {s}, {vr(L.K(j, ks), 8)}, {vr(L.Q(i, ks), 8)}, {c}"
                add(qk, "WMMA", f"QK{i}")
        if i == k:
            mask_nodes(L, i, ops["dl"] if i == 0 else L.dl[i], add)
        if i < 2:  # the rest of q tiles 2, 3 runs in the next region's prefix
            softmax_nodes(L, i, add, c_sgpr=True)
        if i == 1:
            sum_nodes(L, 0, add, range(first, 2))
    # the arrive, after every LDS read of the region and one LDS latency after the last (its s_wait_dscnt 0
    # then does not stall); no LDS read follows it, as the next block's Q may then overwrite the staging area
    bar = add("s_barrier_signal -1", "BAR", "arrive")
    build_dag(nodes)
    for nd in loads:
        add_order(nodes, nd.idx, bar.idx, DS_LAT)
    # an LDS load that overwrites a source of a WMMA in flight waits for va_vdst(0) (expert mode, more than
    # one VALU class in flight): the V reads go after the last prefix PV WMMA, the Q(3) reads after the last
    # QK(0) WMMA, so each group waits once instead of once per buffer slice
    last = {nd.tag: nd.idx for nd in nodes if nd.kind == "WMMA"}
    lpv = max(last[f"PV{i}p"] for i in range(QT))
    for nd in loads:
        if nd.tag.startswith("V"):
            add_order(nodes, lpv, nd.idx)
        if nd.tag.startswith("Q3") and "QK0" in last:
            add_order(nodes, last["QK0"], nd.idx)
    # the first read goes before the first WMMA: the va_vdst(0) wait at the region entry then drains no WMMA
    add_order(nodes, loads[0].idx, min(nd.idx for nd in nodes if nd.kind == "WMMA"))
    return nodes


def region_tile(L, ops, k=None):
    """(lines, modelled cycles) of TILE_C (k None) or TILE_Dk."""
    order, cycles = schedule(tile_nodes(L, ops, k))
    out = []
    for nd in order:
        if nd.kind == "BAR":  # this wave's LDS reads done, wave 0's TDM ops of the next stage landed
            out += ["s_wait_dscnt 0x0", "s_wait_tensorcnt 0x6"]
        out.append(nd.text)
    return out, cycles


def region_final(L, ops, i):
    """FINAL_i (program order). FINAL_0 starts with the scheduled prefix of the last tile, c from the VGPR
    pair (the SGPRs are clobbered across markers). ops: o (the first of the 32 output VGPRs), lse (VGPR)."""
    out = []
    if i == 0:
        nodes, add = node_list()
        add(f"s_mov_b32 {S_SEL}, 0x{SEL_LO:x}", "SALU", "sel")
        prefix_nodes(L, add, c_sgpr=False)
        out = [nd.text for nd in schedule(build_dag(nodes))[0]]
    d, m = L.d[i], L.m[i]
    a, b, rc, e1, q, e2, sh, sub, x, lg, s = L.fin
    inv = L.inv
    out += [  # LLVM's fp32 1.0 / d (denormals on: no mode switches), then d > 0 ? . : 0
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
        f"v_mul_f32_e32 {vr(ops['lse'])}, 0x{LN2:x}, {vr(s)}",
    ]
    return out
