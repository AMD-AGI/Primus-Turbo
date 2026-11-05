###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.emit -- the derived lines of a generated block, and the re-check of emitted code.

A generator hands over the real instructions of a straight-line block in their final order (``make_item``), without
``s_set_vgpr_msb``, ``s_wait_alu``, register-driven ``s_wait_*cnt`` and ``v_nop``. ``emit_block`` adds them for
gfx1250 in expert scheduling mode, as LLVM would:
  * ``s_set_vgpr_msb`` wherever the VGPR operands need another mode. Which operand of a mnemonic uses which mode
    field is learned from the compiler's own output (``learn_msb_map``). The mode is 0 at the block start, before
    a branch and at the block end;
  * the waits of the expert-mode model (asmkit.expert): ``s_wait_alu depctr_va_vdst(N)`` with the caller's rule
    (``expert.va_need``) and ``depctr_vm_vsrc(N)``; ``s_wait_dscnt`` / ``loadcnt`` / ``storecnt``; and
    ``s_wait_dscnt 0`` before an ``s_barrier_signal`` while LDS instructions are in flight;
  * ``s_wait_xcnt 0`` before a write of a scalar register that an in-flight VMEM / FLAT / TDM instruction still
    reads. ``s_set_vgpr_msb`` does not wait for XCNT;
  * ``v_nop`` for the WMMA and TRANS wait states (``nops_needed``).
Every wait and v_nop holds for everything that may reach the block (a ``Flow``). ``s_wait_*cnt`` items (fences such
as ``s_wait_tensorcnt``) stay where they are and count.

``walk_block`` walks emitted lines with their own waits and reports every hazard they leave open. It uses the same
model as the emitter, so it catches emitter mistakes, not model mistakes."""

import re

from . import expert as E
from .isa import MEM, VALU, cnt_fields, first_tok, parse_line, regs_of, sgpr_defs_uses, strip_comment

FIELDS = ("src0", "src1", "src2", "dst")  # s_set_vgpr_msb mode fields, 2 bits each from bit 0
VALU_CLS = ("XDL", "CSMACC", "DPMACC", "TRANS", "VNOP")  # the instructions the v_nop rules count
VALU_HISTORY = 12  # VALU-class instructions kept per path (the v_nop rules look back at most 9)
VA_VDST_MAX = 14  # depctr_va_vdst: 15 means "no wait"
VM_VSRC_MAX = 6  # depctr_vm_vsrc is a 3-bit field whose 7 means "no wait": a need >= 7 is emitted as 6
CNT_MAX = 63  # largest emitted s_wait_*cnt value; the model applies the needed one (conservative)
MAX_STATES = 8  # more distinct wait (or XCNT) states than this at one program point are merged into one
MAX_HISTORIES = 64  # more distinct VALU histories than this at one program point are an error
_NOP = parse_line(-1, "\tv_nop")


# --------------------------------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------------------------------
def kind_of(x):
    """The kind of an Ins: WMMA, VALU, TRANS, DS_LD, DS_ST, VM_LD, VM_ST, TDM, WAIT_CNT, BAR_S, BAR_W, BR, SALU or
    OTHER."""
    c, mn = x.cls, x.mn
    if c == "XDL":
        return "WMMA"
    if c in ("CSMACC", "DPMACC"):
        return "VALU"
    if c == "TRANS":
        return "TRANS"
    if c == "DS":
        return "DS_LD" if mn.startswith(("ds_load", "ds_read")) else "DS_ST"
    if c in ("VMEM", "FLAT"):
        return "VM_ST" if ("_store" in mn or "_atomic" in mn) else "VM_LD"
    if c == "TDM":
        return "TDM"
    if c == "WAIT_CNT":
        return "WAIT_CNT"
    if c == "BARRIER":
        return "BAR_S" if "signal" in mn else "BAR_W"
    if mn.startswith(("s_cbranch", "s_branch")):
        return "BR"
    if c == "SALU":
        return "SALU"
    return "OTHER"


class Item:
    """A real instruction of a block: its text, parsed Ins (x), kind (kind_of), VGPR and scalar defs / uses; pos:
    its index, set by the caller (for display)."""

    __slots__ = ("text", "x", "kind", "vdefs", "vuses", "sdefs", "suses", "pos")

    def __repr__(self):
        return f"I{self.pos}:{self.x.mn}"


def make_item(text):
    """The Item of one instruction line."""
    x = parse_line(-1, text)
    n = Item()
    n.text, n.x, n.kind = text, x, kind_of(x)
    n.vdefs, n.vuses = list(x.defs), list(x.uses)
    n.sdefs, n.suses = sgpr_defs_uses(x)
    n.pos = -1
    return n


# --------------------------------------------------------------------------------------------------------------
# s_set_vgpr_msb operand-field map (learned from the compiler's own .s)
# --------------------------------------------------------------------------------------------------------------
MSB_RE = re.compile(r"msbs:\s*dst=(\d)\s+src0=(\d)\s+src1=(\d)\s+src2=(\d)")


def msb_mode(line):
    """{field: block} that an s_set_vgpr_msb line sets (from its ``msbs:`` comment, else its immediate)."""
    m = MSB_RE.search(line)
    if m:
        d, a, b, c = (int(g) for g in m.groups())
        return {"dst": d, "src0": a, "src1": b, "src2": c}
    imm = int(strip_comment(line).split()[1], 0) & 0xFF
    return {f: (imm >> (2 * k)) & 3 for k, f in enumerate(FIELDS)}


def op_blocks(x):
    """[(operand position, block)] of the VGPR operands of x (block = true register // 256)."""
    out = []
    for p, o in enumerate(x.ops):
        if o == "::":
            continue
        rr = regs_of(first_tok(o))
        if rr:
            out.append((p, rr[0] // 256))
    return out


def learn_msb_map(ins):
    """{(mn, pos): {fields}}: for every VGPR operand of the text, the fields whose mode in effect there (0 after a
    label) equals the operand's block, intersected over all its occurrences."""
    cand = {}
    mode = {f: 0 for f in FIELDS}
    for x in ins:
        if x.cls == "LABEL":
            mode = {f: 0 for f in FIELDS}
            continue
        if x.cls == "MSB":
            mode = msb_mode(x.text)
            continue
        if not x.mn or x.cls in ("DIR",):
            continue
        for p, b in op_blocks(x):
            key = (x.mn, p)
            ok = {f for f in FIELDS if mode[f] == b}
            cand[key] = cand.get(key, set(FIELDS)) & ok
    return cand


def merge_msb_maps(*maps):
    """Field map from several texts (a generator's vocabulary may exceed one kernel's)."""
    out = {}
    for m in maps:
        for k, v in m.items():
            out[k] = out.get(k, set(FIELDS)) & v
    return out


def msb_required(x, fmap):
    """{field: block} x needs; raises if an operand's field is unknown or ambiguous. An ambiguous VALU / WMMA
    operand among the first four takes the VOP3 order (dst, src0, src1, src2) when that field is a candidate."""
    if "::" in x.mn:
        raise RuntimeError(f"VOPD is not supported: {x.text.strip()}")
    req = {}
    for p, b in op_blocks(x):
        fs = fmap.get((x.mn, p))
        if not fs:
            raise RuntimeError(f"no MSB field for {x.mn} operand {p}: {x.text.strip()}")
        if len(fs) > 1 and x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS"):
            dflt = ("dst", "src0", "src1", "src2")[p] if p < 4 else None
            if dflt in fs:
                fs = {dflt}
        if len(fs) > 1:
            raise RuntimeError(f"ambiguous MSB field {sorted(fs)} for {x.mn} operand {p}: {x.text.strip()}")
        f = next(iter(fs))
        if f in req and req[f] != b:
            raise RuntimeError(f"conflicting MSB needs in {x.text.strip()}")
        req[f] = b
    return req


def msb_line(new, old):
    """The s_set_vgpr_msb from mode old to mode new: bits 0-7 hold the new mode, bits 8-15 the old one (as the
    compiler writes it)."""
    imm = sum(new[f] << (2 * k) for k, f in enumerate(FIELDS)) | (
        sum(old[f] << (2 * k) for k, f in enumerate(FIELDS)) << 8
    )
    return (
        f"\ts_set_vgpr_msb 0x{imm:x}                   ;  msbs: dst={new['dst']} src0={new['src0']} "
        f"src1={new['src1']} src2={new['src2']}"
    )


# --------------------------------------------------------------------------------------------------------------
# hazards needing v_nop (gfx1250 co-execution rules, GCNHazardRecognizer)
# --------------------------------------------------------------------------------------------------------------
def wmma_regs(x):
    """(D, A | B): the VGPRs a WMMA writes and the ones it multiplies."""
    o = [regs_of(first_tok(t)) for t in x.ops]
    return set(o[0]), set(o[1]) | set(o[2])


def nops_needed(prev, x):
    """The v_nops x needs; prev: the VALU-class instructions before it, newest first. Wait states, counted in
    VALU-class instructions (v_nop included): 5 from a WMMA's D to a WMMA reading it as A / B; 4 from a WMMA to a
    VALU that reads its D or writes its D / A / B; 1 from a TRANS to an immediately following non-TRANS VALU that
    reads its result or writes its source."""
    need = 0
    if x.cls == "XDL":
        _, ab = wmma_regs(x)
        for k, y in enumerate(prev[:9]):
            if y.cls == "XDL":
                d, _ = wmma_regs(y)
                if d & ab:
                    need = max(need, 5 - k)
                    break
    elif x.cls in ("CSMACC", "DPMACC", "TRANS"):
        uses, defs = set(x.uses), set(x.defs)
        for k, y in enumerate(prev[:8]):
            if y.cls == "XDL":
                d, ab = wmma_regs(y)
                if (d & uses) or (defs & (d | ab)):
                    need = max(need, 4 - k)
                    break
        if x.cls != "TRANS" and prev:
            y = prev[0]
            if y.cls == "TRANS" and ((set(y.defs) & uses) or (defs & set(y.uses))):
                need = max(need, 1)
    return max(need, 0)


def _push(valus, x):
    """Add the VALU-class instruction x to every history."""
    for v in valus:
        v.insert(0, x)
        del v[VALU_HISTORY:]


# --------------------------------------------------------------------------------------------------------------
# waits
# --------------------------------------------------------------------------------------------------------------
class XState:
    """The scalar registers that in-flight VMEM / FLAT / TDM instructions still read (XCNT); unknown: any may be."""

    def __init__(self, unknown):
        self.unknown = unknown
        self.regs = set()

    def copy(self):
        s = XState(self.unknown)
        s.regs = set(self.regs)
        return s

    def sig(self):
        return (self.unknown, tuple(sorted(self.regs)))


def wait_needs(sts, xs, x, sx, release, va_rule):
    """{field: n}: the waits x needs, the smallest over the states sts, and xcnt if any XCNT state of xs asks.
    sx: the scalar registers x writes; release: x is an s_barrier_signal."""
    need = {}

    def put(f, n):
        if n is not None:
            need[f] = n if f not in need else min(need[f], n)

    for st in sts:
        if x.cls in MEM:
            put("va_vdst", E.va_need(st, x, va_rule))
        if x.cls in VALU or x.cls in MEM:
            put("vm_vsrc", E.vm_need(st, x))
        for c, n in E.load_need(st, x):
            put(c, n)
        if release and (st.unknown_ds or any(not r.done and r.ctr == "dscnt" for r in st.loads + st.msrc)):
            put("dscnt", 0)
    if sx and any(s.unknown or set(sx) & s.regs for s in xs):
        put("xcnt", 0)
    return need


def wait_lines(need):
    """The wait lines of need: va_vdst / vm_vsrc and the counters, capped at their largest emitted values."""
    out = []
    caps = (("va_vdst", VA_VDST_MAX), ("vm_vsrc", VM_VSRC_MAX))
    alu = [f"depctr_{k}({min(need[k], m)})" for k, m in caps if k in need]
    if alu:
        out.append("\ts_wait_alu " + " ".join(alu))
    for c in ("loadcnt", "storecnt", "dscnt", "xcnt"):
        if c in need:
            out.append(f"\ts_wait_{c} 0x{min(need[c], CNT_MAX):x}")
    return out


def apply_needs(sts, xs, need):
    """Advance sts / xs over the waits of need as wait_lines emits them (counters: see CNT_MAX)."""
    for st in sts:
        if "va_vdst" in need:
            E.apply_va(st, min(need["va_vdst"], VA_VDST_MAX))
        if "vm_vsrc" in need:
            E.apply_vm(st, min(need["vm_vsrc"], VM_VSRC_MAX))
        cnt = {k: v for k, v in need.items() if k in ("loadcnt", "storecnt", "dscnt")}
        if cnt:
            E.apply_cnt(st, cnt)
    if "xcnt" in need:
        for s in xs:
            s.unknown, s.regs = False, set()


class Flow:
    """What can reach a program point: wait states, XCNT states and VALU histories (newest first), kept as three
    independent lists (a wait is the min over the states, an XCNT wait is needed if any XCNT state asks, the v_nops
    are the max over the histories: a sound over-approximation of the per-path combinations)."""

    __slots__ = ("sts", "xs", "valus")

    def __init__(self, sts, xs, valus):
        self.sts, self.xs, self.valus = sts, xs, valus

    def copy(self):
        return Flow([s.copy() for s in self.sts], [x.copy() for x in self.xs], [list(v) for v in self.valus])

    def sig(self):
        return (
            sorted(repr(s.rel_sig()) for s in self.sts),
            sorted(x.sig() for x in self.xs),
            sorted(tuple(y.text for y in v) for v in self.valus),
        )

    @staticmethod
    def join(flows):
        """The flow at a merge point: the distinct states, XCNT states and histories of flows. More than MAX_STATES
        states (XCNT states) are merged into one; more than MAX_HISTORIES histories raise."""
        sts, xs, vs = [], [], []
        seen = (set(), set(), set())
        for f in flows:
            for s in f.sts:
                k = repr(s.rel_sig())
                if k not in seen[0]:
                    seen[0].add(k)
                    sts.append(s.copy())
            for x in f.xs:
                k = x.sig()
                if k not in seen[1]:
                    seen[1].add(k)
                    xs.append(x.copy())
            for v in f.valus:
                k = tuple(y.text for y in v)
                if k not in seen[2]:
                    seen[2].add(k)
                    vs.append(list(v))
        if len(sts) > MAX_STATES:
            sts = [E.join(sts)]
        if len(xs) > MAX_STATES:
            m = XState(any(x.unknown for x in xs))
            for x in xs:
                m.regs |= x.regs
            xs = [m]
        if len(vs) > MAX_HISTORIES:
            raise RuntimeError(f"{len(vs)} distinct VALU histories reach one block")
        return Flow(sts, xs, vs)


# --------------------------------------------------------------------------------------------------------------
# emission and re-check of a straight-line block
# --------------------------------------------------------------------------------------------------------------
def emit_block(items, flow, fmap, va_rule):
    """(lines, the flow at the end) of a block: the items in their order with the derived lines added, for
    everything ``flow`` (not modified) says may reach the block. fmap: the MSB field map; va_rule: the va_vdst rule
    (expert.va_need)."""
    fl = flow.copy()
    sts, xs, valus = fl.sts, fl.xs, fl.valus
    out = []
    mode = {f: 0 for f in FIELDS}
    for n in items:
        x = n.x
        pre = []
        req = {f: 0 for f in FIELDS} if n.kind == "BR" else msb_required(x, fmap)
        if any(mode[f] != v for f, v in req.items()):
            new = dict(mode, **req)
            pre.append(msb_line(new, mode))
            mode = new
        sx = [r for r in n.sdefs if r != "scc"]
        need = wait_needs(sts, xs, x, sx, n.kind == "BAR_S", va_rule)
        if need:
            pre += wait_lines(need)
            apply_needs(sts, xs, need)
        if x.cls in VALU_CLS:
            nn = max(nops_needed(v, x) for v in valus)
            pre += ["\tv_nop"] * nn
            for _ in range(nn):
                _push(valus, _NOP)
        out += pre
        out.append(n.text)
        for st in sts:
            if x.cls == "WAIT_CNT":
                E.apply_cnt(st, cnt_fields(x))
            else:
                E.issue(st, x)
        if x.cls in ("VMEM", "FLAT", "TDM"):
            for s in xs:
                s.regs |= set(n.suses)
        if x.cls in VALU_CLS:
            _push(valus, x)
    if any(mode[f] for f in FIELDS):  # the next block starts in mode 0
        out.append(msb_line({f: 0 for f in FIELDS}, mode))
    for st in sts:
        E.cap(st)
    return out, fl


def walk_block(ins, flow, on_need):
    """Walk the parsed lines ins of a straight-line block with their own waits from flow (advanced in place) and
    report every hazard they leave open as on_need(i, field, n): a missing va_vdst (per-class bound), vm_vsrc or
    counter wait (n: the largest value that would do), xcnt or v_nop (n = 0)."""
    for i, x in enumerate(ins):
        if x.cls in VALU_CLS and x.cls != "VNOP" and any(nops_needed(v, x) for v in flow.valus):
            on_need(i, "v_nop", 0)
        sd, su = sgpr_defs_uses(x)
        guard = {r for r in sd if r != "scc"}
        if guard and any(s.unknown or guard & s.regs for s in flow.xs):
            on_need(i, "xcnt", 0)
        for st in flow.sts:
            E.step(st, ins, i, on_need)
        if x.cls == "WAIT_CNT" and cnt_fields(x).get("xcnt") == 0:
            for s in flow.xs:
                s.unknown, s.regs = False, set()
        elif x.cls in ("VMEM", "FLAT", "TDM"):
            for s in flow.xs:
                s.regs |= set(su)
        if x.cls in VALU_CLS:
            _push(flow.valus, x)
