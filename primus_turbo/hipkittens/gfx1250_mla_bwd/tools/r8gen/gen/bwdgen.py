###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwdgen -- generated regions of the HK MLA dK/dV ABC loop (timing skeleton, step S0).

The shell (hip/src/k_dkdv96ds.hip, -DHKGEN=1) runs every tile of a 96-row K3 workgroup through one marker per step;
build.py reads the registers LLVM gave the marker operands and asks this module for the code:

  INIT      (c1, c1) / (scale, scale) pairs; K / V fragments, accumulators, raw S / dP, P / dS, NL / ND set to
            constants (skeleton: no global loads)
  STEP<p>   one ABC step (p = step parity: which P / dS set C reads and B writes):
              B(k-1)  softmax / dS of the raw S / dP in registers (k_dkdv96ds's arithmetic, packed f32 ops), pack
                      to bf16 (P -> PB.pk, dS -> PB.ds), dS^T DPP regroup -> DSX / DSY
              C(k-2)  dV^T += dO^T P (tr16 of stage k-2, 8 operands x NKV), dK^T += Q^T dS (12 x NKV)
              ring barrier: s_wait_tensorcnt 3, s_barrier_signal (after C's last LDS read), s_barrier_wait
              export  6 x buffer_store_b128 of DSX / DSY
              NL / ND 8 x buffer_load_b128 of tile k (next step's B)
              TDM     3 x tensor_load_to_lds of tile k+2 into stage k+2
              A(k)    S = Q K^T, dP = dO V^T of stage k: units (dt, hh), 2 operands of 2 b128 each, NKV WMMAs per
                      operand; S / dP chains from +0 over dt ascending
            A's operand units rotate over the P / dS set C just consumed (dead after C).

Program order is the semantics; the list scheduler reorders along the dependence graph (RAW / WAR / WAW on VGPRs,
SGPRs, scc, vcc; asmkit's own parser) plus order edges for the barrier. Output: real instructions only; asmkit.emit
derives s_set_vgpr_msb, expert-mode waits and hazard v_nops."""

import sys

NKV, NDT_QK, NDT_V, NDO_V, NDO_QK = 3, 6, 4, 8, 12
XK, XV = 400, 272
QOFF = 32 * XV  # Q tile offset inside a ring stage
DS_ST_B = 512


def vr(r, n=1):
    """Operand text of the n VGPRs at true register r (encoded low 8 bits + the true numbers in a comment)."""
    if r // 256 != (r + n - 1) // 256:
        raise ValueError(f"tuple v[{r}:{r + n - 1}] crosses a 256-register block")
    lo = r % 256
    if n == 1:
        return f"v{lo}" if r < 256 else f"v{lo} /*v{r}*/"
    return f"v[{lo}:{lo + n - 1}]" if r < 256 else f"v[{lo}:{lo + n - 1}] /*v[{r}:{r + n - 1}]*/"


class Layout:
    """VGPR map (true numbers) of the generated state, everything at or above `base`."""

    def __init__(self, base, cslots=5):
        if base > 12:
            raise ValueError(f"shell VGPRs reach v{base - 1}: the layout needs base <= 12")
        self.base = base
        self.cc1, self.ccs = 12, 14  # (c1, c1), (scale, scale)
        self.pre = [16 + 4 * e for e in range(8)]  # NL / ND quads
        self.dsx = [48 + 8 * kh for kh in range(NKV)]
        self.dsy = [52 + 8 * kh for kh in range(NKV)]
        self.ct = [72, 80, 88, 1008, 1016]  # C operand slots
        # timing skeleton only: more C slots borrowed from the K fragments (their data is destroyed: garbage numerics)
        self.ct += [768 + 8 * (17 - i) for i in range(max(0, cslots - 5))]
        self.rawb = 96  # raw[hh][kh][j] = 96 + 8 (6 hh + 2 kh + j)
        # P / dS sets: pk[kh], ds[kh] tuples
        self.P = [
            {"pk": [192, 200, 208], "ds": [216, 224, 232]},
            {"pk": [736, 744, 752], "ds": [760, 240, 248]},
        ]
        self.kfb, self.vfb = 768, 912

    def raw(self, hh, kh, j):
        return self.rawb + 8 * (6 * hh + 2 * kh + j)

    def acc(self, i):
        return 256 + 8 * i if i < 32 else 512 + 8 * (i - 32)

    def dv(self, kh, d):
        return self.acc(kh * NDO_V + d)

    def dk(self, kh, d):
        return self.acc(NKV * NDO_V + kh * NDO_QK + d)

    def kf(self, kh, dt):
        return self.kfb + 8 * (kh * NDT_QK + dt)

    def vf(self, kh, dt):
        return self.vfb + 8 * (kh * NDT_V + dt)

    def all_tuples(self):
        """Every generated register (for INIT and the layout check)."""
        regs = set(range(self.cc1, self.cc1 + 4))
        for q in self.pre:
            regs |= set(range(q, q + 4))
        for t in self.dsx + self.dsy:
            regs |= set(range(t, t + 4))
        for t in self.ct:
            regs |= set(range(t, t + 8))
        regs |= set(range(self.rawb, self.rawb + 96))
        for s in self.P:
            for t in s["pk"] + s["ds"]:
                regs |= set(range(t, t + 8))
        for i in range(60):
            regs |= set(range(self.acc(i), self.acc(i) + 8))
        regs |= set(range(self.kfb, self.kfb + 144)) | set(range(self.vfb, self.vfb + 96))
        return regs


# --------------------------------------------------------------------------------------------------------------
# nodes (asmkit's parser gives defs / uses: the same model the emitter and the checks use)
# --------------------------------------------------------------------------------------------------------------
_M = None


def _asmkit():
    global _M
    if _M is None:
        from asmkit import emit as M

        _M = M
    return _M


class Node:
    __slots__ = ("text", "kind", "tag", "vdefs", "vuses", "sdefs", "suses", "preds", "idx", "war_xdl")


def mk(text, tag=""):
    M = _asmkit()
    it = M.make_item("\t" + text)
    n = Node()
    n.text, n.kind, n.tag = text, it.kind, tag
    n.vdefs, n.vuses = set(it.vdefs), set(it.vuses)
    n.sdefs, n.suses = set(it.sdefs), set(it.suses)
    n.preds, n.idx, n.war_xdl = [], -1, set()
    return n


# --------------------------------------------------------------------------------------------------------------
# dependence graph and list scheduler
# --------------------------------------------------------------------------------------------------------------
# issue cycles (non-WMMA) and result latency; WMMA: XDL pipe busy XDL_BUSY cycles, chain latency WMMA_CHAIN
COST = {"VALU": 2, "TRANS": 4, "DS_LD": 2, "DS_ST": 2, "VM_LD": 2, "VM_ST": 2, "TDM": 2, "SALU": 1, "BAR_S": 1,
        "BAR_W": 1, "WAIT_CNT": 0, "OTHER": 1}  # fmt: skip
LAT = {"WMMA": 20, "VALU": 4, "TRANS": 8, "DS_LD": 120, "DS_ST": 4, "VM_LD": 800, "VM_ST": 4, "TDM": 4, "SALU": 2,
       "BAR_S": 1, "BAR_W": 1, "WAIT_CNT": 1, "OTHER": 1}  # fmt: skip
XDL_BUSY, WMMA_CHAIN, WAR_XDL = 10, 16, 30
DS_MAX = 56
MIN_XDL_AFTER = 3  # WMMAs issued after a reader before a memory op may overwrite its operand (its va_vdst then >= 3)
# LDS as a queue (micro/wmma_mix: with 4 waves per WGP each wave gets one 512-B LDS load per 8 cycles): a DS op
# occupies it LDS_OCC cycles; its data is back LDS_LAT cycles after its turn ends (None: the static LAT["DS_LD"])
LDS_OCC, LDS_LAT = None, None
EX_AT = 12  # the export stores follow A's WMMA #EX_AT


def build_dag(nodes):
    """RAW (producer latency), WAR / WAW (order) edges in list order."""
    last_w, readers = {}, {}
    for i, n in enumerate(nodes):
        n.idx = i
        preds = {}
        for r in [("v", x) for x in n.vuses] + [("s", x) for x in n.suses]:
            if r in last_w:
                p = last_w[r]
                lat = WMMA_CHAIN if (nodes[p].kind == "WMMA" and n.kind == "WMMA") else LAT[nodes[p].kind]
                preds[p] = max(preds.get(p, 0), lat)
        for r in [("v", x) for x in n.vdefs] + [("s", x) for x in n.sdefs]:
            if r in last_w:
                preds[last_w[r]] = max(preds.get(last_w[r], 0), 1)
            for q in readers.get(r, ()):
                if q != i:
                    # a memory op overwriting a register an in-flight WMMA reads waits for it (va_vdst): keep
                    # WAR_XDL cycles (about 3 WMMAs) between them so the wait finds it retired
                    lat = WAR_XDL if (nodes[q].kind == "WMMA" and n.kind not in ("WMMA", "VALU", "TRANS")) else 1
                    preds[q] = max(preds.get(q, 0), lat)
                    if nodes[q].kind == "WMMA" and n.kind not in ("WMMA", "VALU", "TRANS"):
                        n.war_xdl.add(q)
        n.preds = sorted(preds.items())
        for r in [("v", x) for x in n.vuses] + [("s", x) for x in n.suses]:
            readers.setdefault(r, set()).add(i)
        for r in [("v", x) for x in n.vdefs] + [("s", x) for x in n.sdefs]:
            last_w[r] = i
            readers[r] = set()


def add_order(nodes, a, b, lat=1):
    if not a < b:
        raise ValueError("order edges must point forward in program order")
    d = dict(nodes[b].preds)
    d[a] = max(d.get(a, 0), lat)
    nodes[b].preds = sorted(d.items())


def schedule(nodes, wseq, mode="list"):
    """(ordered nodes, modelled cycles). 'program': list order. 'list': the WMMAs in the order wseq (XDL busy
    XDL_BUSY cycles each); every other node earliest-deadline-first (deadline = position of the first WMMA that
    needs it, transitively), then by critical path; barrier nodes first. LDS loads kept <= DS_MAX outstanding."""
    n = len(nodes)
    if mode == "program":
        return list(nodes), None
    succ = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p, l in nd.preds:
            succ[p].append((i, l))
    cp = [0] * n
    for i in range(n - 1, -1, -1):
        c = XDL_BUSY if nodes[i].kind == "WMMA" else COST[nodes[i].kind]
        cp[i] = c + max((l + cp[s] for s, l in succ[i]), default=0)
    wpos = {i: k for k, i in enumerate(wseq)}
    dl = [10**6] * n
    for i in range(n - 1, -1, -1):
        for s_, _ in succ[i]:
            dl[i] = min(dl[i], wpos[s_] if s_ in wpos else dl[s_])
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    ds_issued, ds_retired = 0, -1
    ds_seq = {}
    wmma_at = {}  # node -> stream index of an issued WMMA
    lds_free = 0

    def xdl_ok(i):
        rs = nodes[i].war_xdl
        return not rs or all(r in wmma_at for r in rs) and wk - 1 - max(wmma_at[r] for r in rs) >= MIN_XDL_AFTER or \
            (nw_left() == 0)

    def nw_left():
        return len(wseq) - wk
    t, xdl_free, out, wk, last_trans = 0, 0, [], 0, False
    while len(out) < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        if nw is not None and nw in ready and earliest[nw] <= t and xdl_free <= t:
            pick = nw
        else:
            o = [i for i in ready if earliest[i] <= t and nodes[i].kind != "WMMA"
                 and (nodes[i].kind != "DS_LD" or ds_issued - 1 - ds_retired < DS_MAX) and xdl_ok(i)]  # fmt: skip
            if o:
                pick = max(o, key=lambda i: (nodes[i].kind in ("BAR_S", "BAR_W", "WAIT_CNT"), -dl[i],
                                             cp[i] - (6 if last_trans and nodes[i].kind == "TRANS" else 0), -i))  # fmt: skip
        if pick is None:
            nxt = [earliest[i] for i in ready if earliest[i] > t]
            if nw is not None and nw in ready and xdl_free > t:
                nxt.append(xdl_free)
            if not nxt:
                raise RuntimeError(f"scheduler deadlock at t={t}, {len(out)}/{n} placed")
            t = max(t + 1, min(nxt))
            continue
        nd = nodes[pick]
        ready.discard(pick)
        ti = t
        if nd.kind == "DS_LD":
            ds_seq[pick] = ds_issued
            ds_issued += 1
        for p, _ in nd.preds:  # a consumer of an LDS load retires every LDS load issued before it
            if nodes[p].kind == "DS_LD" and p in ds_seq and nd.vuses & nodes[p].vdefs:
                ds_retired = max(ds_retired, ds_seq[p])
        if nd.kind == "BAR_S":
            ds_retired = ds_issued - 1
        out.append(nd)
        if nd.kind == "WMMA":
            xdl_free = t + XDL_BUSY
            wmma_at[pick] = wk
            wk += 1
        else:
            t += COST[nd.kind] + (2 if (last_trans and nd.kind == "TRANS") else 0)
        last_trans = nd.kind == "TRANS"
        done_at = None
        if LDS_OCC is not None and nd.kind in ("DS_LD", "DS_ST"):
            start = max(ti, lds_free)
            lds_free = start + LDS_OCC
            done_at = lds_free + LDS_LAT
        for s_, l in succ[pick]:
            lat = l
            if done_at is not None and nd.kind == "DS_LD" and nodes[s_].vuses & nd.vdefs:
                lat = max(1, done_at - ti)  # RAW on the loaded data: when its LDS turn and the latency are over
            earliest[s_] = max(earliest[s_], ti + lat)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
    return out, max(t, xdl_free)


# --------------------------------------------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------------------------------------------
BF16_ONE2 = 0x3F803F80  # two bf16 1.0


def region_init(L, ops):
    """INIT: constants into every generated register; (c1, c1), (scale, scale) from the marker's SGPRs."""
    out = [f"v_mov_b32 {vr(L.cc1)}, {ops['c1']}", f"v_mov_b32 {vr(L.cc1 + 1)}, {ops['c1']}",
           f"v_mov_b32 {vr(L.ccs)}, {ops['sc']}", f"v_mov_b32 {vr(L.ccs + 1)}, {ops['sc']}"]  # fmt: skip
    out += [f"s_mov_b32 {r}, 0" for r in PROBE_ACC]
    regs = sorted(L.all_tuples() - set(range(L.cc1, L.cc1 + 4)))
    kv = set(range(L.kfb, L.kfb + 144)) | set(range(L.vfb, L.vfb + 96))
    for r in regs:
        out.append(f"v_mov_b32 {vr(r)}, {hex(BF16_ONE2) if r in kv else '0'}")
    return out


def step_nodes(L, ops, p, b_before_barrier=True):
    """Nodes of one ABC step (program order) and the WMMA stream order. ops: marker operands (register texts)."""
    PA, PB = L.P[p], L.P[1 - p]
    nodes, tags = [], {}

    def add(text, tag=""):
        n = mk(text, tag)
        nodes.append(n)
        return len(nodes) - 1

    b_nodes, c_tr, w_c, w_a = [], [], [], []
    # ---- B(k-1): softmax / dS (in place in raw), pack into PB, regroup into DSX / DSY
    for hh in range(2):
        for kh in range(NKV):
            s_, d_ = L.raw(hh, kh, 0), L.raw(hh, kh, 1)
            for j in range(4):
                nl = L.pre[hh * 4 + j // 2] + 2 * (j % 2)
                nd = L.pre[hh * 4 + 2 + j // 2] + 2 * (j % 2)
                b_nodes.append(add(f"v_pk_fma_f32 {vr(s_ + 2 * j, 2)}, {vr(s_ + 2 * j, 2)}, {vr(L.cc1, 2)}, {vr(nl, 2)}"))
                b_nodes.append(add(f"v_exp_f32 {vr(s_ + 2 * j)}, {vr(s_ + 2 * j)}"))
                b_nodes.append(add(f"v_exp_f32 {vr(s_ + 2 * j + 1)}, {vr(s_ + 2 * j + 1)}"))
                b_nodes.append(add(f"v_pk_fma_f32 {vr(d_ + 2 * j, 2)}, {vr(d_ + 2 * j, 2)}, {vr(L.ccs, 2)}, {vr(nd, 2)}"))
                b_nodes.append(add(f"v_pk_mul_f32 {vr(d_ + 2 * j, 2)}, {vr(s_ + 2 * j, 2)}, {vr(d_ + 2 * j, 2)}"))
                w = hh * 4 + j
                b_nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(PB['pk'][kh] + w)}, {vr(s_ + 2 * j)}, {vr(s_ + 2 * j + 1)}"))
                b_nodes.append(add(f"v_cvt_pk_bf16_f32 {vr(PB['ds'][kh] + w)}, {vr(d_ + 2 * j)}, {vr(d_ + 2 * j + 1)}"))
    for kh in range(NKV):
        w = PB["ds"][kh]
        for k in range(4):
            b_nodes.append(add(f"v_mov_b32 {vr(L.dsx[kh] + k)}, {vr(w + k)}"))
            b_nodes.append(add(f"v_mov_b32_dpp {vr(L.dsx[kh] + k)}, {vr(w + 4 + k)} row_ror:8 row_mask:0xf bank_mask:0xc"))
            b_nodes.append(add(f"v_mov_b32 {vr(L.dsy[kh] + k)}, {vr(w + 4 + k)}"))
            b_nodes.append(add(f"v_mov_b32_dpp {vr(L.dsy[kh] + k)}, {vr(w + k)} row_ror:8 row_mask:0xf bank_mask:0x3"))
    # ---- C(k-2): dV then dK, d outer / kh inner, operand j in slot j % 5
    j = 0
    for isq, nd_, acc, base, src in ((0, NDO_V, L.dv, ops["tv"], PA["pk"]), (1, NDO_QK, L.dk, ops["tk"], PA["ds"])):
        for d in range(nd_):
            t = L.ct[j % len(L.ct)]
            for pc in range(2):
                off = (QOFF if isq else 0) + d * 32 + pc * 16 * (XK if isq else XV)
                c_tr.append(add(f"ds_load_tr16_b128 {vr(t + 4 * pc, 4)}, {base}" + (f" offset:{off}" if off else "")))
            for kh in range(NKV):
                a = acc(kh, d)
                w_c.append(add(f"v_wmma_f32_16x16x32_bf16 {vr(a, 8)}, {vr(t, 8)}, {vr(src[kh], 8)}, {vr(a, 8)}"))
            j += 1
    # ---- ring barrier
    tw = add("s_wait_tensorcnt 0x3")
    bs = add("s_barrier_signal -1")
    bw = add("s_barrier_wait -1")
    # ---- export, NL / ND, TDM
    ld = []
    for e in range(8):
        r = ops["rnl"] if (e % 4) // 2 == 0 else ops["rnd"]
        off = (e // 4) * 64 + (e % 2) * 16
        ld.append(add(f"buffer_load_b128 {vr(L.pre[e], 4)}, {ops['nlv']}, {r}, {ops['nls']} offen"
                      + (f" offset:{off}" if off else "")))  # fmt: skip
    td = [add(f"tensor_load_to_lds {ops['g0' + c]}, {ops['g1' + c]}") for c in "abc"]
    # ---- A(k): units (dt, hh) in 3 slots over PA (dead after C)
    slots = [(PA["pk"][0], PA["pk"][1]), (PA["pk"][2], PA["ds"][0]), (PA["ds"][1], PA["ds"][2]),
             (L.ct[0], L.ct[1]), (L.ct[2], L.ct[3])]  # C's operand slots are free in A
    a_ld = []
    for dt in range(NDT_QK):
        for hh in range(2):
            u = dt * 2 + hh
            qt, ot = slots[u % len(slots)]
            for pc in range(2):
                off = QOFF + hh * 16 * XK + dt * 64 + pc * 32
                a_ld.append(add(f"ds_load_b128 {vr(qt + 4 * pc, 4)}, {ops['qa']} offset:{off}"))
            if dt < NDT_V:
                for pc in range(2):
                    off = hh * 16 * XV + dt * 64 + pc * 32
                    a_ld.append(add(f"ds_load_b128 {vr(ot + 4 * pc, 4)}, {ops['va']}" + (f" offset:{off}" if off else "")))
            for kh in range(NKV):
                s_ = L.raw(hh, kh, 0)
                c = "0" if dt == 0 else vr(s_, 8)
                w_a.append(add(f"v_wmma_f32_16x16x32_bf16 {vr(s_, 8)}, {vr(qt, 8)}, {vr(L.kf(kh, dt), 8)}, {c}"))
                if dt < NDT_V:
                    d_ = L.raw(hh, kh, 1)
                    c = "0" if dt == 0 else vr(d_, 8)
                    w_a.append(add(f"v_wmma_f32_16x16x32_bf16 {vr(d_, 8)}, {vr(ot, 8)}, {vr(L.vf(kh, dt), 8)}, {c}"))
    # ---- export (program order after A: it only reads DSX / DSY)
    ex = []
    for kh in range(NKV):
        ex.append(add(f"buffer_store_b128 {vr(L.dsx[kh], 4)}, {ops['vds']}, {ops['rds']}, {ops['ds' + str(kh)]} offen"))
        ex.append(add(f"buffer_store_b128 {vr(L.dsy[kh], 4)}, {ops['vds']}, {ops['rds']}, {ops['ds' + str(kh)]} offen "
                      f"offset:{DS_ST_B}"))  # fmt: skip
    build_dag(nodes)
    # order edges: tensorcnt -> signal -> wait; C's LDS reads before the signal; A's LDS reads, the TDM, the export
    # and the NL / ND loads after the wait; (option) B before the signal
    add_order(nodes, tw, bs)
    add_order(nodes, bs, bw)
    for i in c_tr:
        add_order(nodes, i, bs)
    if b_before_barrier:
        for i in b_nodes:
            add_order(nodes, i, bs)
    for i in a_ld + td + ex + ld:
        add_order(nodes, bw, i)
    for i in ex:  # the export stores under A's WMMAs (B's DPP results long retired)
        add_order(nodes, w_a[EX_AT], i)
    return nodes, w_c + w_a


PROBE_ACC = ("s96", "s97", "s98", "s99")  # phase sums (INIT zeroes them, PROBE hands them to the shell)


def region_step(L, ops, p, mode="list", b_before_barrier=True, probe=False, info=None):
    """(texts, modelled cycles). probe: s_get_shader_cycles at the step start, before the barrier signal, after the
    barrier wait, after C's last WMMA (in schedule order) and at the end; the four segment deltas summed into
    PROBE_ACC (info["probe_segments"]: what each segment ends at)."""
    nodes, wseq = step_nodes(L, ops, p, b_before_barrier)
    out, cyc = schedule(nodes, wseq, mode)
    texts = [n.text for n in out]
    if probe:
        nwc = NKV * (NDO_V + NDO_QK)
        last_c = [k for k, n in enumerate(out) if n.kind == "WMMA"][nwc - 1]
        bs = next(k for k, n in enumerate(out) if n.kind == "BAR_S")
        bw = next(k for k, n in enumerate(out) if n.kind == "BAR_W")
        pts = sorted([(bs, "signal"), (bw + 1, "wait"), (last_c + 1, "C end")])
        regs = ["s[90:91]", "s[92:93]"]
        res, k0 = ["s_get_shader_cycles_u64 s[90:91]"], 0
        for seg, (k, lab) in enumerate(pts + [(len(texts), "end")]):
            res += texts[k0:k]
            cur, prev = regs[(seg + 1) % 2], regs[seg % 2]
            res += [f"s_get_shader_cycles_u64 {cur}", f"s_sub_co_i32 s94, s{cur[2:4]}, s{prev[2:4]}",
                    f"s_add_co_i32 {PROBE_ACC[seg]}, {PROBE_ACC[seg]}, s94"]  # fmt: skip
            k0 = k
        texts = res
        if info is not None:
            info["probe_segments"] = [lab for _, lab in pts] + ["end"]
    return texts, cyc


if __name__ == "__main__":
    sys.exit(0)


# --------------------------------------------------------------------------------------------------------------
# slot scheduler (the forward ladder's S1b issue model, measured again here: micro/wmma_mix, micro/lds_lat)
# --------------------------------------------------------------------------------------------------------------
SLOT = {"dslat": 70, "dsrate": 8, "valu_offsets": (2, 3, 6, 7), "mem_offsets": (1, 4, 5), "bar_gap": 12}


def schedule_slot(nodes, wseq):
    """(ordered nodes, modelled cycles): a cycle-stepped list scheduler around the in-order WMMA stream wseq. A WMMA
    occupies the XDL pipe 8 cycles; inside its window only SLOT['valu_offsets'] take a VALU / TRANS and
    SLOT['mem_offsets'] (and any offset >= 8 when the next WMMA is not ready) an LDS / VMEM / TDM op or a SALU; a
    TRANS blocks VALU / TRANS the next cycle; one LDS op per SLOT['dsrate'] cycles (4 waves share the WGP's LDS),
    completing in order SLOT['dslat'] cycles after its turn; one SALU per cycle. Result latencies VALU 4, TRANS 8,
    WMMA 16 (chain) / 20. A memory op that overwrites a WMMA operand waits until MIN_XDL_AFTER more WMMAs issued.
    Ties: deadline (first WMMA needing the node), barrier first, then critical path, then program order."""
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
        c = 8 if nodes[i].kind == "WMMA" else 1
        cp[i] = c + max((l + cp[s] for s, l in succ[i]), default=0)
    LATK = {"WMMA": 20, "VALU": 4, "TRANS": 8, "SALU": 2, "VM_LD": 800, "VM_ST": 4, "TDM": 4, "DS_ST": 4,
            "BAR_S": 1, "BAR_W": 1, "WAIT_CNT": 1, "OTHER": 1}  # fmt: skip
    MEM = ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM")
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    out, wmma_at = [], {}
    t, last_w, wk, trans_at, ds_next, ds_last_done = 0, -100, 0, -100, 0, 0
    ds_issued, ds_retired, ds_seq = 0, -1, {}
    bs_wk = [10**6]

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
            return (not inwin or o in SLOT["valu_offsets"]) and t != trans_at + 1
        if k in MEM:
            if not xdl_ok(i):
                return False
            if k in ("DS_LD", "DS_ST") and (t < ds_next or ds_issued - 1 - ds_retired >= DS_MAX):
                return False
            return not inwin or o in SLOT["mem_offsets"]
        if k == "SALU":
            return o != 0
        if k == "BAR_S":  # its s_wait_dscnt 0: every LDS op issued so far complete
            return o != 0 and t >= ds_last_done
        if k == "BAR_W":  # bar_gap WMMAs after the signal (the other wave's skew)
            return o != 0 and (wk >= bs_wk[0] + SLOT.get("bar_gap", 0) or wk >= len(wseq))
        return o != 0  # waits

    while len(out) < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        if nw is not None and nw in ready and earliest[nw] <= t and t >= last_w + 8:
            pick = nw
        else:
            cand = [i for i in ready if nodes[i].kind != "WMMA" and avail(i)]
            if cand:
                o = t - last_w
                if 0 <= o < 8 and o in SLOT["valu_offsets"]:
                    vt = [i for i in cand if nodes[i].kind in ("VALU", "TRANS")]
                    if vt:
                        cand = vt
                pick = max(cand, key=lambda i: (nodes[i].kind in ("BAR_S", "BAR_W", "WAIT_CNT"), -dl[i], cp[i], -i))
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
            start = max(t, ds_next - SLOT["dsrate"])
            ds_next = max(t, ds_next) + SLOT["dsrate"]
            done_at = max(t + SLOT["dslat"], ds_last_done + SLOT["dsrate"])
            ds_last_done = done_at
            ds_seq[pick] = ds_issued
            ds_issued += 1
        for p, _ in nd.preds:
            if nodes[p].kind == "DS_LD" and p in ds_seq and nd.vuses & nodes[p].vdefs:
                ds_retired = max(ds_retired, ds_seq[p])
        if k == "BAR_S":
            ds_retired = ds_issued - 1
            bs_wk[0] = wk
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
                lat = max(l, LATK.get(k, 1)) if (nodes[s_].vuses & nd.vdefs or nodes[s_].suses & nd.sdefs) else l
            earliest[s_] = max(earliest[s_], t + lat)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
        t += 1
    return out, max(t, last_w + 8)


# --------------------------------------------------------------------------------------------------------------
# window scheduler: the WMMA stream cut into windows (one WMMA + what issues behind it), each window's length from
# the measured issue costs of its contents (micro/wmma_mix, expert mode, 4 waves per WGP, 2026-10-29)
# --------------------------------------------------------------------------------------------------------------
# extra cycles over the WMMA's 8 for a window holding (P, E, C) = (packed / FMA VALU, TRANS, light VALU: converts,
# moves, integer and compare ops); LDS loads add nothing alone, 0.22 next to two P, 0.69 next to two P and an E
WIN_COST = {
    (0, 0, 0): 0.0, (0, 1, 0): 0.0, (0, 2, 0): 0.11, (0, 3, 0): 2.16,
    (0, 0, 1): 0.0, (0, 0, 2): 0.08, (0, 0, 3): 0.19, (0, 0, 4): 0.22, (0, 0, 5): 2.2,
    (0, 1, 1): 0.08, (0, 1, 2): 0.42, (0, 2, 1): 2.36, (0, 2, 2): 3.66,
    (1, 0, 0): 1.0, (1, 1, 0): 1.28, (1, 2, 0): 1.41, (1, 3, 0): 4.1, (1, 0, 1): 1.03, (1, 0, 2): 1.14,
    (1, 1, 1): 1.39, (1, 1, 2): 2.14,
    (2, 0, 0): 1.05, (2, 1, 0): 1.41, (2, 2, 0): 4.1, (2, 0, 1): 1.39, (2, 0, 2): 2.14, (2, 1, 1): 4.08,
    (3, 0, 0): 1.64, (3, 1, 0): 2.16, (4, 0, 0): 2.16,
}  # fmt: skip
WIN = {"budget": 1.06, "budget_hi": 1.42, "dslat": 70, "dsrate": 8, "lat_valu": 6, "lat_trans": 10, "lat_wmma": 20,
       "chain": 16, "bar_gap": 12, "drain": 16, "other_free": 2, "l_max": 1, "vm_max": 2, "slack": 6}  # fmt: skip
TRACE = None  # schedule_win debug: a list to receive (issue time, node, kind, class, text)
P_MNEM = ("v_pk_fma_f32", "v_pk_mul_f32", "v_pk_add_f32", "v_fma_f32", "v_mul_f32", "v_add_f32", "v_sub_f32")


def vclass(nd):
    if nd.kind == "TRANS":
        return "E"
    if nd.kind == "VALU":
        return "P" if nd.text.split()[0] in P_MNEM else "C"
    if nd.kind in ("DS_LD", "DS_ST"):
        return "L"
    return "O"


def win_extra(cnt):
    """Extra cycles of one window from its class counts (unknown mixes: a pessimistic sum)."""
    p, e, c, l, o = cnt["P"], cnt["E"], cnt["C"], cnt["L"], cnt["O"]
    key = (min(p, 4), e, c)
    if key in WIN_COST:
        x = WIN_COST[key]
    else:
        x = 1.0 * p + 1.1 * max(0, e - 1) + 0.6 * max(0, c - 2) + 2.0
    if l and p >= 2:
        x += 0.69 if e else 0.22
    x += max(0, o - WIN["other_free"]) * 1.0
    return x


def schedule_win(nodes, wseq):
    """(ordered nodes, modelled cycles): the WMMAs in the order wseq; every other node into the window (behind a
    WMMA) that keeps that window cheap (WIN_COST), earliest deadline first (deadline: the first WMMA that needs the
    node, transitively), forced in when its deadline WMMA would otherwise wait. LDS: one op accepted per
    WIN['dsrate'] cycles, data back WIN['dslat'] cycles later, in order. A memory op overwriting a WMMA operand waits
    until MIN_XDL_AFTER more WMMAs issued; the ring barrier's signal until every LDS op before it completed, its wait
    until WIN['bar_gap'] WMMAs after the signal."""
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
        c = 8 if nodes[i].kind == "WMMA" else 1
        cp[i] = c + max((l + cp[s] for s, l in succ[i]), default=0)
    cls = [vclass(nd) for nd in nodes]
    npred = [len(nd.preds) for nd in nodes]
    avail = [0.0] * n
    ready = set(i for i in range(n) if npred[i] == 0 and nodes[i].kind != "WMMA")
    st = {"ds_next": 0.0, "ds_done": 0.0, "ds_issued": 0, "bs_wk": None, "wk": 0}
    wmma_at, out, issue_t = {}, [], {}

    def lat_of(p, s_):
        kp = nodes[p].kind
        dep = nodes[s_].vuses & nodes[p].vdefs or nodes[s_].suses & nodes[p].sdefs
        if kp == "WMMA":
            return WIN["chain"] if nodes[s_].kind == "WMMA" and dep else (WIN["lat_wmma"] if dep else 1)
        if kp == "DS_LD":
            return None  # completion time (queue)
        if kp == "VALU":
            return WIN["lat_valu"] if dep else 1
        if kp == "TRANS":
            return WIN["lat_trans"] if dep else 1
        if kp == "VM_LD":
            return 0 if dep else 1  # NL / ND: loaded a step earlier (the emitter's loadcnt wait)
        return 1

    def place(i, t):
        nd = nodes[i]
        issue_t[i] = t
        if TRACE is not None:
            TRACE.append((t, i, nd.kind, cls[i], nd.text))
        done = t
        if nd.kind in ("DS_LD", "DS_ST"):
            acc = max(t, st["ds_next"])
            st["ds_next"] = acc + WIN["dsrate"]
            done = max(acc + WIN["dslat"], st["ds_done"])
            st["ds_done"] = done
            st["ds_issued"] += 1
        if nd.kind == "WMMA":
            wmma_at[i] = st["wk"]
            st["wk"] += 1
        if nd.kind == "BAR_S":
            st["bs_wk"] = st["wk"]
        out.append(nd)
        for s_, _ in succ[i]:
            l = lat_of(i, s_)
            a = done if l is None else t + l
            avail[s_] = max(avail[s_], a)
            npred[s_] -= 1
            if npred[s_] == 0 and nodes[s_].kind != "WMMA":
                ready.add(s_)

    def can(i, t_end):
        nd = nodes[i]
        if avail[i] > t_end:
            return False
        if nd.war_xdl and not all(r in wmma_at and st["wk"] - 1 - wmma_at[r] >= MIN_XDL_AFTER for r in nd.war_xdl) \
                and st["wk"] < len(wseq):  # fmt: skip
            return False
        if nd.kind == "BAR_S" and st["ds_done"] > t_end:
            return False
        if nd.kind == "BAR_W" and (st["bs_wk"] is None or (st["wk"] < st["bs_wk"] + WIN["bar_gap"]
                                                             and st["wk"] < len(wseq))):  # fmt: skip
            return False
        return True

    t = 0.0
    for k in range(len(wseq) + 1):
        nxt = wseq[k] if k < len(wseq) else None
        # fill the window in front of W_k (k = 0: the prologue window; k = len: the tail)
        cnt = {"P": 0, "E": 0, "C": 0, "L": 0, "O": 0}
        t0 = t
        while True:
            extra = win_extra(cnt)
            base_end = t0 + (8 + extra if k else 0)
            w_ok = nxt is not None and npred[nxt] == 0
            end = max(base_end, avail[nxt]) if w_ok else base_end
            room = end - (t0 + 8) if k else end - t0
            cand = [i for i in ready if dl[i] <= k or can(i, end - 1)]
            cand.sort(key=lambda i: (nodes[i].kind not in ("BAR_S", "BAR_W", "WAIT_CNT"), dl[i], -cp[i], i))
            # pressure: VALU / TRANS work due within the next WIN['slack'] windows vs. what cheap windows can hold
            due = sum(1 for i in ready if cls[i] in "PEC" and dl[i] <= k + WIN["slack"])
            budget = WIN["budget_hi"] if due > 2 * WIN["slack"] else WIN["budget"]
            pick = None
            for i in cand:
                if dl[i] <= k and not can(i, float("inf")):
                    continue
                c = dict(cnt)
                c[cls[i]] += 1
                urgent = dl[i] <= k
                if not urgent and cls[i] == "L" and c["L"] > WIN["l_max"]:
                    continue
                if not urgent and nodes[i].kind in ("VM_LD", "VM_ST") and \
                        sum(1 for x in out[-12:] if x.kind in ("VM_LD", "VM_ST") and issue_t[x.idx] >= t0) >= WIN["vm_max"]:
                    continue
                ex = win_extra(c)
                if urgent or (cls[i] == "O" and nodes[i].kind not in ("VM_LD", "VM_ST")) or ex <= max(budget, room):
                    pick, cnt_new = i, c
                    break
            if pick is None:
                break
            ready.discard(pick)
            place(pick, max(avail[pick], t0 + 1 + sum(cnt.values())))
            cnt = cnt_new
        if nxt is None:
            t = t0 + 8 + win_extra(cnt)
            break
        guard = 0
        while npred[nxt] > 0:  # its operands not all placed (their own predecessors were late): force them
            cand = sorted(ready, key=lambda i: (dl[i], -cp[i], i))
            guard += 1
            if not cand or guard > n:
                raise RuntimeError("window scheduler deadlock")
            i = cand[0]
            ready.discard(i)
            cnt[cls[i]] += 1
            place(i, max(avail[i], t0 + 1 + sum(cnt.values())))
        t_w = max(t0 + (8 + win_extra(cnt) if k else 0), avail[nxt])
        place(nxt, t_w)
        t = t_w
    while ready:  # leftovers (none expected)
        i = min(ready, key=lambda i: (dl[i], i))
        ready.discard(i)
        place(i, t)
    if len(out) != n:
        raise RuntimeError(f"window scheduler placed {len(out)} of {n} nodes")
    return out, int(round(t))
