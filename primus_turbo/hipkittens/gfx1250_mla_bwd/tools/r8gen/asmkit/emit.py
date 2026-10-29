###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.emit -- re-derive the "derived" lines of a loop (region) body for a given instruction order.

A generator or a scheduler hands over the real instructions of a loop body in their final order (``Item``s, no
``s_set_vgpr_msb``, ``s_wait_alu``, register-driven ``s_wait_*cnt``, ``v_nop``, ``s_clause``, ``s_delay_alu``);
this module adds them back with LLVM's rules for gfx1250:
  * ``s_set_vgpr_msb``: the operand-field map is learned from the compiler's own output (``learn_msb_map``); the
    mode is 0 at every block start and before every branch (LLVM: a block that is not a fall-through starts at 0);
  * expert scheduling mode waits: ``va_vdst`` with LLVM's rule, ``vm_vsrc`` (<= 7, 3-bit field), in-order
    ``dscnt`` / ``loadcnt`` / ``storecnt``, ``s_wait_dscnt 0`` before ``s_barrier_signal`` while LDS ops are in
    flight (release), ``s_wait_xcnt 0`` before a scalar def that an in-flight VMEM / TDM op still reads (an
    ``s_set_vgpr_msb`` waits XCNT == 0 implicitly). Non-expert mode (``expert=False``): no ``s_wait_alu``; XCNT
    also before a VGPR def an in-flight VMEM op still reads (what LLVM emits there);
  * hazard ``v_nop``: WMMA D -> WMMA A/B 5, WMMA -> VALU 4, TRANS -> VALU 1 wait states (GCNHazardRecognizer);
  * WMMA ``matrix_[ab]_reuse``: dropped where the operand is not the previous WMMA's (same registers, not written).
Waits are the minimum over every path into each instruction (entry from the preheader, back edges, the blocks of a
multi-block region), iterated to a fixed point of the back-edge states.

Fence waits are not derived: ``s_wait_tensorcnt`` / ``kmcnt`` / ``samplecnt`` / ``bvhcnt`` lines and any
non-``dscnt`` wait right before an ``s_barrier_signal`` stay as fixed items in the order given."""

import re

from . import expert as E
from .isa import (
    MEM,
    VALU,
    cnt_fields,
    first_tok,
    fmt_wait,
    parse_line,
    regs_of,
    sgpr_defs_uses,
    strip_comment,
)

DERIVED = ("MSB", "WAIT_ALU", "VNOP", "CLAUSE", "DELAY")
# abcgen patches: va_vdst rule (True: LLVM's, N = 0 once more than one VALU class pends; False: the per-class bound,
# only after the WAR-V stress test and the independent checker are clean) and the vm_vsrc cap (the 3-bit field's 7
# means "no wait": a need >= 7 must be rounded down to 6, not up to "none")
VA_RULE_LLVM = True
VM_VSRC_CAP = 6
DERIVED_CNT = ("dscnt", "loadcnt", "storecnt", "xcnt")
FIELDS = ("src0", "src1", "src2", "dst")
VALU_CLS = ("XDL", "CSMACC", "DPMACC", "TRANS", "VNOP")


# --------------------------------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------------------------------
def kind_of(x):
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
    """A real instruction of a body: its source line, parsed Ins, kind, scalar defs / uses."""

    __slots__ = ("li", "text", "x", "kind", "vdefs", "vuses", "sdefs", "suses", "fence", "pos", "reuse_src")

    def __repr__(self):
        return f"I{self.pos}:{self.x.mn}"


def make_item(text, li=-1, x=None):
    x = x if x is not None else parse_line(li, text)
    n = Item()
    n.li, n.text, n.x, n.kind = li, text, x, kind_of(x)
    n.vdefs, n.vuses = list(x.defs), list(x.uses)
    n.sdefs, n.suses = sgpr_defs_uses(x)
    n.fence = n.kind in ("BAR_S", "BAR_W", "TDM") or n.kind == "WAIT_CNT"
    n.pos, n.reuse_src = -1, -1
    return n


def is_derived(ins, i, last):
    """Line i is re-derived (dropped from the items): MSB / s_wait_alu / v_nop / clause / delay, and the
    register-driven counter waits (dscnt / loadcnt / storecnt / xcnt) except fence waits right before an
    s_barrier_signal (non-dscnt)."""
    x = ins[i]
    if x.cls in DERIVED:
        return True
    if x.cls != "WAIT_CNT":
        return False
    f = cnt_fields(x)
    if not f or any(k not in DERIVED_CNT for k in f):
        return False
    if all(k == "dscnt" for k in f):
        return True
    j = i + 1  # a fence before a barrier signal: keep
    while j <= last and (not ins[j].mn or ins[j].cls in DERIVED or ins[j].cls == "WAIT_CNT"):
        j += 1
    return not (j <= last and ins[j].cls == "BARRIER" and "signal" in ins[j].mn)


def build_items(lines, ins, first, last):
    """Items of the body lines [first, last] (last = the branch), derived lines dropped, in source order."""
    out = []
    for i in range(first, last + 1):
        x = ins[i]
        if not x.mn or x.cls in ("LABEL", "DIR") or is_derived(ins, i, last):
            continue
        n = make_item(lines[i], i, x)
        n.pos = len(out)
        out.append(n)
    prevw = None
    for n in out:
        if n.kind == "WMMA":
            if "matrix_a_reuse" in n.text or "matrix_b_reuse" in n.text:
                n.reuse_src = prevw.pos if prevw is not None else -1
            prevw = n
    return out


# --------------------------------------------------------------------------------------------------------------
# s_set_vgpr_msb operand-field map (learned from the compiler's own .s)
# --------------------------------------------------------------------------------------------------------------
MSB_RE = re.compile(r"msbs:\s*dst=(\d)\s+src0=(\d)\s+src1=(\d)\s+src2=(\d)")


def msb_mode(line):
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
    """{(mn, pos): {fields}} from every instruction of the text under its in-effect mode."""
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
    """{field: block} x needs; raises if an operand's field is unknown or ambiguous with a nonzero block."""
    req = {}
    vopd = "::" in x.mn
    half = x.ops.index("::") if vopd else None
    for p, b in op_blocks(x):
        if vopd:  # AMDGPULowerVGPREncoding: X and Y halves share the fields, operands in order dst, src0, src1, src2
            q = p if p < half else p - half - 1
            f = ("dst", "src0", "src1", "src2")[q]
            if f in req and req[f] != b:
                raise RuntimeError(f"conflicting MSB needs in VOPD {x.text.strip()}")
            req[f] = b
            continue
        fs = fmap.get((x.mn, p))
        if not fs:
            raise RuntimeError(f"no MSB field for {x.mn} operand {p}: {x.text.strip()}")
        if len(fs) > 1 and x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS") and "::" not in x.mn:
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
    o = [regs_of(first_tok(t)) for t in x.ops]
    return set(o[0]), set(o[1]) | set(o[2])  # D, A|B


def nops_needed(prev, x):
    """prev: VALU-class instructions before x, newest first (Ins). Number of v_nop to put before x."""
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


def check_nops(lines_body, preheader_valu, trips=2):
    """Body line indices (straight-line body, repeated `trips` times) where a hazard has too few v_nops."""
    from .isa import parse

    ins = parse(lines_body)
    prev = list(preheader_valu)
    bad = []
    for _ in range(trips):
        for i, x in enumerate(ins):
            if x.cls in VALU_CLS and x.cls != "VNOP":
                if nops_needed(prev, x):
                    bad.append(i)
            if x.cls in VALU_CLS:
                prev.insert(0, x)
                del prev[12:]
    return sorted(set(bad))


def preheader_valu_list(ins, first):
    """The last 12 VALU-class instructions before line `first` (newest first)."""
    out = []
    p = first - 1
    while p >= 0 and len(out) < 12:
        if ins[p].cls in VALU_CLS:
            out.append(ins[p])
        p -= 1
    return out


# --------------------------------------------------------------------------------------------------------------
# waits
# --------------------------------------------------------------------------------------------------------------
class XState:
    """Sources of in-flight VMEM / FLAT / TDM ops (XCNT): scalar names, and VGPR numbers in non-expert mode."""

    def __init__(self, unknown):
        self.unknown = unknown
        self.regs = set()

    def copy(self):
        s = XState(self.unknown)
        s.regs = set(self.regs)
        return s

    def sig(self):
        return (self.unknown, tuple(sorted(map(str, self.regs))))


def wait_needs(sts, xs, x, sx, release, expert=True):
    """{field: n} the waits x needs on every path state in sts (min over paths). sx: the defs XCNT guards."""
    need = {}

    def put(f, n):
        if n is not None:
            need[f] = n if f not in need else min(need[f], n)

    for st in sts:
        if expert and x.cls in MEM:
            put("va_vdst", E.va_need(st, x, llvm=VA_RULE_LLVM))
        if expert and (x.cls in VALU or x.cls in MEM or x.cls == "SALU"):
            put("vm_vsrc", E.vm_need(st, x))
        for c, n in E.load_need(st, x):
            put(c, n)
        if release:
            if st.unknown_ds or any(not r.done and r.ctr == "dscnt" for r in st.loads + st.msrc):
                put("dscnt", 0)
    for s in xs:
        if s.unknown and sx:
            put("xcnt", 0)
        elif set(sx) & s.regs:
            put("xcnt", 0)
    return need


def apply_needs(sts, xs, need):
    need = dict(need)
    if "vm_vsrc" in need:
        need["vm_vsrc"] = min(need["vm_vsrc"], VM_VSRC_CAP)
    if "va_vdst" in need:
        need["va_vdst"] = min(need["va_vdst"], 14)
    for st in sts:
        if "va_vdst" in need:
            E.apply_va(st, need["va_vdst"])
        if "vm_vsrc" in need:
            E.apply_vm(st, need["vm_vsrc"])
        cnt = {k: v for k, v in need.items() if k in ("dscnt", "loadcnt", "storecnt", "tensorcnt")}
        if cnt:
            E.apply_cnt(st, cnt)
    if "xcnt" in need:
        for s in xs:
            s.unknown, s.regs = False, set()


def wait_lines(need):
    out = []
    alu = {k: need[k] for k in ("va_vdst", "vm_vsrc") if k in need}
    if "vm_vsrc" in alu:
        alu["vm_vsrc"] = min(alu["vm_vsrc"], VM_VSRC_CAP)  # 3-bit field, 7 = no wait; a smaller value is stricter
    if alu:
        out.append(fmt_wait(alu))
    for c in ("loadcnt", "storecnt", "dscnt", "xcnt"):
        if c in need:
            out.append(f"\ts_wait_{c} 0x{min(need[c], 63):x}")
    return out


def fix_reuse(lines_body):
    """Drop matrix_a_reuse / matrix_b_reuse flags whose operand is not the previous WMMA's (same registers, not
    written in between): the compiler's own invariant; a dropped flag reads VGPRs (always correct)."""
    out, prevw, written, dropped = [], None, set(), 0
    for t in lines_body:
        x = parse_line(0, t)
        if x.cls == "XDL":
            o = [tuple(regs_of(first_tok(q))) for q in x.ops]
            for flag, idx in (("matrix_a_reuse", 1), ("matrix_b_reuse", 2)):
                if flag in t and not (
                    prevw is not None and prevw[idx] == o[idx] and not (set(o[idx]) & written)
                ):
                    t = t.replace(" " + flag, "")
                    dropped += 1
            prevw, written = o, set()
        elif x.mn:
            written |= set(x.defs)
        out.append(t)
    return out, dropped


def _xdefs(n, expert):
    """The defs an XCNT wait guards: scalar defs (not scc); in non-expert mode the VGPR defs too."""
    sx = [r for r in n.sdefs if r != "scc"]
    if not expert:
        sx += list(n.vdefs)
    return sx


def _xsrcs(n, expert):
    return set(n.suses) | (set(n.vuses) if not expert else set())


# --------------------------------------------------------------------------------------------------------------
# single-block loop (the original AO emitter; ao_id e4b638c8 was built with it)
# --------------------------------------------------------------------------------------------------------------
def emit_body(seq, fmap, entry_state, preheader_valu, iters=6):
    """Lines of a single-block loop body (label excluded, branch included) for the items `seq` in their final
    order: waits, v_nops and MSBs re-derived (expert mode). entry_state: expert.State at the header entered from
    outside. The back-edge XCNT state starts empty (the original AO behaviour; emit_region carries it)."""
    back = None
    back_valu = []
    for _ in range(iters):
        sts = [entry_state.copy()] + ([back.copy()] if back is not None else [])
        xs = [XState(True)] + ([XState(False)] if back is not None else [])
        out = []
        mode = {f: 0 for f in FIELDS}
        prev_valu = list(preheader_valu)  # newest first
        prev_back = list(back_valu)
        for n in seq:
            x = n.x
            pre = []
            req = {f: 0 for f in FIELDS} if n.kind == "BR" else msb_required(x, fmap)
            if any(mode[f] != v for f, v in req.items()):
                new = dict(mode, **req)
                pre.append(msb_line(new, mode))
                mode = new
                for s in xs:
                    s.unknown, s.regs = False, set()
            sx = [r for r in n.sdefs if r.startswith("s")]
            need = wait_needs(sts, xs, x, sx, release=(n.kind == "BAR_S"))
            if need:
                pre += wait_lines(need)
                apply_needs(sts, xs, need)
            if x.cls in VALU_CLS:
                k = max(nops_needed(prev_valu, x), nops_needed(prev_back, x) if prev_back else 0)
                pre += ["\tv_nop"] * k
                nop = parse_line(0, "\tv_nop")
                for _ in range(k):
                    prev_valu.insert(0, nop)
                    prev_back.insert(0, nop)
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
                prev_valu.insert(0, x)
                del prev_valu[12:]
                prev_back.insert(0, x)
                del prev_back[12:]
        new_back = sts[-1]
        new_bv = [y for y in prev_valu]
        if (
            back is not None
            and new_back.sig() == back.sig()
            and [y.text for y in new_bv] == [y.text for y in back_valu]
        ):
            break
        back, back_valu = new_back, new_bv
    out, _ = fix_reuse(out)
    return out


# --------------------------------------------------------------------------------------------------------------
# multi-block region (a natural loop with forward branches inside, or a single block)
# --------------------------------------------------------------------------------------------------------------
class Path:
    """One way into a region from outside: wait state, XCNT state, the VALU history (newest first)."""

    __slots__ = ("st", "xs", "valu")

    def __init__(self, st, xs, valu):
        self.st, self.xs, self.valu = st, xs, valu


class Flow:
    """What can reach a program point: wait states, XCNT states and VALU histories, kept as three independent lists
    (a wait is the min over the states, an XCNT wait is needed if any XCNT state asks, the v_nops are the max over
    the histories: a sound over-approximation of the per-path combinations)."""

    __slots__ = ("sts", "xs", "valus")

    def __init__(self, sts, xs, valus):
        self.sts, self.xs, self.valus = sts, xs, valus

    @staticmethod
    def of(paths):
        return Flow([p.st.copy() for p in paths], [p.xs.copy() for p in paths], [list(p.valu) for p in paths])

    def copy(self):
        return Flow([s.copy() for s in self.sts], [x.copy() for x in self.xs], [list(v) for v in self.valus])

    def sig(self):
        return (
            sorted(repr(s.rel_sig()) for s in self.sts),
            sorted(x.sig() for x in self.xs),
            sorted(tuple(y.text for y in v) for v in self.valus),
        )

    @staticmethod
    def join(flows, max_states=8, max_hist=64):
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
        if len(sts) > max_states:
            sts = [E.join(sts)]
        if len(xs) > max_states:
            m = XState(any(x.unknown for x in xs))
            for x in xs:
                m.regs |= x.regs
            xs = [m]
        if len(vs) > max_hist:
            raise RuntimeError(f"{len(vs)} distinct VALU histories reach one block")
        return Flow(sts, xs, vs)


# s_set_vgpr_msb does NOT wait for XCNT == 0 on gfx1250 (micro xcnt_mix 2026-10-29: buffer_load + s_set_vgpr_msb
# costs 0.4 cycles more than the load alone, + s_wait_xcnt 0 costs 46): an MSB write releases no in-flight VMEM / TDM
# scalar source; only s_wait_xcnt does
MSB_CLEARS_XCNT = False


def _emit_block(items, flow, fmap, expert, last_block_mode_reset=True):
    """Emit one block for everything that reaches it; returns (lines, the flow at its end)."""
    fl = flow.copy()
    sts, xs, valus = fl.sts, fl.xs, fl.valus
    out = []
    mode = {f: 0 for f in FIELDS}
    for k, n in enumerate(items):
        x = n.x
        pre = []
        req = {f: 0 for f in FIELDS} if n.kind == "BR" else msb_required(x, fmap)
        if any(mode[f] != v for f, v in req.items()):
            new = dict(mode, **req)
            pre.append(msb_line(new, mode))
            mode = new
            if MSB_CLEARS_XCNT:
                for s in xs:
                    s.unknown, s.regs = False, set()
        need = wait_needs(sts, xs, x, _xdefs(n, expert), release=(n.kind == "BAR_S"), expert=expert)
        if need:
            pre += wait_lines(need)
            apply_needs(sts, xs, need)
        if x.cls in VALU_CLS:
            nn = max(nops_needed(v, x) for v in valus)
            pre += ["\tv_nop"] * nn
            nop = parse_line(0, "\tv_nop")
            for _ in range(nn):
                for v in valus:
                    v.insert(0, nop)
        out += pre
        out.append(n.text)
        for st in sts:
            if x.cls == "WAIT_CNT":
                E.apply_cnt(st, cnt_fields(x))
            else:
                E.issue(st, x)
            if not expert:
                E.forget_alu(st)
        if x.cls in ("VMEM", "FLAT", "TDM"):
            for s in xs:
                s.regs |= _xsrcs(n, expert)
        if x.cls in VALU_CLS:
            for v in valus:
                v.insert(0, x)
                del v[12:]
    if last_block_mode_reset and any(mode[f] for f in FIELDS):  # fall-through end: mode 0 for the next block
        out.append(msb_line({f: 0 for f in FIELDS}, mode))
        if MSB_CLEARS_XCNT:
            for s in xs:
                s.unknown, s.regs = False, set()
    for st in sts:
        E.cap(st)
    return out, fl


def emit_region(
    region,
    fmap,
    entry_state,
    preheader_valu,
    expert=True,
    iters=12,
    max_states=8,
    header=0,
    entry_paths=None,
    info=None,
):
    """Lines of every block of a loop region for the items in their final order.

    region: list of blocks in layout order; each a dict with
      items  : [Item] in final order (a branch, if any, last);
      succ   : successor block indices inside the region; exits omitted.
    header: the block entered from outside (edges to it from the region are the back edges); every other edge must
    keep the region acyclic. entry_paths: [Path] into the header from outside (default: one path from
    entry_state / preheader_valu). info (dict, optional) receives 'exit_paths' {block: [Path]} (states at the end
    of every block, for exit checks) and 'iterations'.
    Returns [lines per block] (labels not included)."""
    nb = len(region)
    preds = [[] for _ in range(nb)]
    for b, blk in enumerate(region):
        for t in blk["succ"]:
            preds[t].append(b)
    # topological order of the region without its back edges, from the header
    indeg = [sum(1 for q in preds[b] if b != header) for b in range(nb)]
    order, ready = [], [header]
    while ready:
        b = ready.pop(0)
        order.append(b)
        for t in region[b]["succ"]:
            if t == header:
                continue
            indeg[t] -= 1
            if indeg[t] == 0:
                ready.append(t)
    if len(order) != nb:
        raise ValueError(f"region has a cycle not through the header or unreachable blocks: order {order}")
    if entry_paths is None:
        entry_paths = [Path(entry_state.copy(), XState(True), list(preheader_valu))]
    entry = Flow.of(entry_paths)
    if not expert:
        for st in entry.sts:
            E.forget_alu(st)
    back = None
    for it in range(iters):
        outs, texts = [None] * nb, [None] * nb
        for b in order:
            if b == header:
                inc = Flow.join([entry] + ([back] if back is not None else []), max_states)
            else:
                inc = Flow.join([outs[q] for q in preds[b]], max_states)
            texts[b], outs[b] = _emit_block(region[b]["items"], inc, fmap, expert)
        new_back = Flow.join([outs[q] for q in preds[header]], max_states)
        if back is not None and new_back.sig() == back.sig():
            break
        back = new_back
    else:
        raise RuntimeError("back-edge states did not converge")
    if info is not None:
        info["exit_paths"] = {b: outs[b] for b in range(nb)}
        info["iterations"] = it + 1
    return [fix_reuse(t)[0] for t in texts]
