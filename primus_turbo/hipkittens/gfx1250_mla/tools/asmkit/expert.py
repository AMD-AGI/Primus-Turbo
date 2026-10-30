###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.expert -- what is in flight at each instruction in gfx1250 expert scheduling mode, and the basic blocks.

In expert scheduling mode the code itself waits for these hazards (the rules of LLVM's SIInsertWaitcnts):
  * a memory instruction that reads (RAW) or writes (WAW) a register an in-flight VALU writes, or writes one it
    reads (WAR): ``s_wait_alu depctr_va_vdst(N)``, which waits until at most N VALU instructions are in flight.
    WMMA (XDL), TRANS and core VALU (CSMACC) complete in order within their class and out of order across
    classes, so N <= (the counted instructions of X's class issued after X) proves X complete: the per-class
    bound. Other VALU (DPMACC) are not counted: a hazard on one needs N = 0. ``va_need`` has the rules;
  * any instruction that writes a register an in-flight memory instruction still reads: ``s_wait_alu
    depctr_vm_vsrc(N)``; memory instructions read their sources in order within LDS / VMEM / FLAT;
  * any instruction that touches a register an in-flight load writes: ``s_wait_dscnt`` / ``loadcnt`` /
    ``storecnt``, which retire in order.
VALU -> VALU dependences are interlocked, apart from the WMMA / TRANS wait states (``emit.nops_needed``).
A ``State`` entered from unknown code (``unknown=True``) assumes an unknown VALU write, memory source read and LDS
load in flight on every VGPR: the first hazard against one of them waits for 0 (va_vdst, vm_vsrc, dscnt)."""

from collections import defaultdict

from .isa import COUNTED, MEM, VALU, cnt_fields, function_range, strip_comment, wait_fields

VA_RULES = ("class", "llvm", "llvm_war_xdl")  # see va_need


class Rec:
    """An in-flight instruction. VALU: cls XDL / TRANS / CSMACC / UNK, seq = its number in its class. Memory op:
    cls LDS / VMEM / FLAT, seq = its number in that class, ctr / cseq = its wait counter and its number there."""

    __slots__ = ("cls", "reads", "writes", "counted", "done", "seq", "ctr", "cseq")

    def __init__(self, cls, reads, writes, counted, seq, ctr=None, cseq=None):
        self.cls, self.reads, self.writes, self.counted = cls, set(reads), set(writes), counted
        self.done, self.seq, self.ctr, self.cseq = False, seq, ctr, cseq

    def copy(self):
        return Rec(self.cls, self.reads, self.writes, self.counted, self.seq, self.ctr, self.cseq)


class State:
    """What is in flight at one program point on one path."""

    def __init__(self, unknown=False):
        self.valu, self.msrc, self.loads = [], [], []  # in-flight VALU, memory source reads, load writes
        self.vcnt, self.mcnt, self.ccnt = defaultdict(int), defaultdict(int), defaultdict(int)
        self.unknown_v = unknown  # entry from unknown code: unknown VALU in flight on every VGPR,
        self.unknown_m = unknown  # unknown memory ops reading every VGPR,
        self.unknown_ds = unknown  # unknown DS loads writing every VGPR

    def copy(self):
        s = State()
        pending = ([r.copy() for r in v if not r.done] for v in (self.valu, self.msrc, self.loads))
        s.valu, s.msrc, s.loads = pending
        s.vcnt, s.mcnt, s.ccnt = (defaultdict(int, c) for c in (self.vcnt, self.mcnt, self.ccnt))
        s.unknown_v, s.unknown_m, s.unknown_ds = self.unknown_v, self.unknown_m, self.unknown_ds
        return s

    def rel_sig(self):
        """Signature with sequence numbers relative to the current counters (comparable across trips)."""

        def recs(v, cnt):
            return sorted(
                (
                    r.cls,
                    tuple(sorted(r.reads)),
                    tuple(sorted(r.writes)),
                    r.counted,
                    cnt[r.cls] - r.seq,
                    r.ctr,
                    (self.ccnt[r.ctr] - r.cseq) if r.ctr else None,
                )
                for r in v
                if not r.done
            )

        return (
            recs(self.valu, self.vcnt),
            recs(self.msrc, self.mcnt),
            recs(self.loads, self.mcnt),
            self.unknown_v,
            self.unknown_m,
            self.unknown_ds,
        )


def join(states):
    """A conservative merge of several states (any path): the union of the pending records, each at its most recent
    position over the states (smallest distance to the current counters), unknown flags or-ed. Every wait the
    merged state asks for is at least as strict as each input state's."""
    base = 1 << 20  # the merged state's counters
    out = State()
    out.unknown_v = any(s.unknown_v for s in states)
    out.unknown_m = any(s.unknown_m for s in states)
    out.unknown_ds = any(s.unknown_ds for s in states)
    for name, cnt_name in (("valu", "vcnt"), ("msrc", "mcnt"), ("loads", "mcnt")):
        best = {}
        for s in states:
            cnt = getattr(s, cnt_name)
            for r in getattr(s, name):
                if r.done:
                    continue
                key = (r.cls, frozenset(r.reads), frozenset(r.writes), r.counted, r.ctr)
                d = cnt[r.cls] - r.seq
                cd = (s.ccnt[r.ctr] - r.cseq) if r.ctr else None
                if key in best:
                    d0, cd0 = best[key]
                    best[key] = (min(d0, d), None if cd is None else min(cd0, cd))
                else:
                    best[key] = (d, cd)
        recs = []
        for (cls, reads, writes, counted, ctr), (d, cd) in best.items():
            getattr(out, cnt_name)[cls] = base
            if ctr:
                out.ccnt[ctr] = base
            recs.append(Rec(cls, reads, writes, counted, base - d, ctr, None if cd is None else base - cd))
        setattr(out, name, recs)
    return out


def cap(st):
    """Clamp every pending record's distance to the current counters at 64. No wait field exceeds 63 (va_vdst 14,
    vm_vsrc 6, counters 63), so this changes no wait; it keeps the state space finite for the fixed points."""
    h = 64
    for name, cnt in (("valu", st.vcnt), ("msrc", st.mcnt), ("loads", st.mcnt)):
        for r in getattr(st, name):
            if cnt[r.cls] - r.seq > h:
                r.seq = cnt[r.cls] - h
            if r.ctr and st.ccnt[r.ctr] - r.cseq > h:
                r.cseq = st.ccnt[r.ctr] - h
    return st


def counter_of(x):
    """The counter a memory instruction is waited for with."""
    if x.cls == "DS":
        return "dscnt"
    if x.cls in ("VMEM", "FLAT"):
        return "storecnt" if ("_store" in x.mn or "_atomic" in x.mn) else "loadcnt"
    return "tensorcnt" if x.cls == "TDM" else None


def issue(st, x):
    """Add instruction x to st (VALU and memory instructions; everything else leaves st as it is)."""
    if x.cls in VALU:
        if x.defs and not x.rmw:  # a newer full write of r supersedes older in-flight writes (as in LLVM)
            d = set(x.defs)
            for r in st.valu:
                if not r.done and r.writes & d:
                    r.writes = r.writes - d
        cls = x.cls if x.cls in COUNTED else "UNK"
        counted = cls == "XDL" or (cls != "UNK" and bool(x.defs))  # a VALU without a VGPR dst may not count
        st.vcnt[cls] += counted
        st.valu.append(Rec(cls, x.uses, x.defs, counted, st.vcnt[cls]))
    elif x.cls in MEM:
        mc = {"DS": "LDS", "FLAT": "FLAT", "VMEM": "VMEM", "TDM": "VMEM"}[x.cls]
        st.mcnt[mc] += 1
        k = counter_of(x)
        st.ccnt[k] += 1
        if x.uses:
            st.msrc.append(Rec(mc, x.uses, [], True, st.mcnt[mc], k, st.ccnt[k]))
        if x.defs:
            st.loads.append(Rec(mc, [], x.defs, True, st.mcnt[mc], k, st.ccnt[k]))


def va_need(st, x, rule):
    """The largest va_vdst N that resolves every VALU hazard of memory instruction x, None if it has none.
      "class": the per-class bound (what the hardware guarantees);
      "llvm": LLVM's rule: the per-class bound while one class is pending, N = 0 once more than one class (or the
        UNK class, DPMACC) is pending;
      "llvm_war_xdl": LLVM's rule, except that a hazard which is only a write-after-read against an in-flight WMMA
        keeps the per-class bound.
    Each rule is at least as strict as the per-class bound."""
    if rule not in VA_RULES:
        raise ValueError(f"unknown va_vdst rule {rule!r}")
    if st.unknown_v and (x.uses or x.defs):
        return 0
    rs, ws = set(x.uses), set(x.defs)
    pending = {r.cls for r in st.valu if not r.done}
    multi = rule != "class" and (len(pending) > 1 or "UNK" in pending)
    need = None
    for r in st.valu:
        if r.done:
            continue
        raw_waw = bool(r.writes & rs or r.writes & ws)
        if not (raw_waw or r.reads & ws):
            continue
        counted = r.cls in COUNTED and r.counted
        n = st.vcnt[r.cls] - r.seq if counted else 0
        if multi and not (rule == "llvm_war_xdl" and not raw_waw and r.cls == "XDL" and counted):
            n = 0
        need = n if need is None else min(need, n)
    return need


def vm_need(st, x):
    """The largest vm_vsrc N after which no in-flight memory op still reads a register x writes."""
    if not x.defs:
        return None
    if st.unknown_m:
        return 0
    ws, need = set(x.defs), None
    for r in st.msrc:
        if not r.done and r.reads & ws:
            n = st.mcnt[r.cls] - r.seq
            need = n if need is None else min(need, n)
    return need


def load_need(st, x):
    """[(counter, N)]: the waits that retire every in-flight load writing a register x touches."""
    rs = set(x.uses) | set(x.defs)
    if st.unknown_ds and rs:
        return [("dscnt", 0)]
    return [(r.ctr, st.ccnt[r.ctr] - r.cseq) for r in st.loads if not r.done and r.writes & rs]


def apply_va(st, n):
    """st after ``s_wait_alu depctr_va_vdst(n)``."""
    for r in st.valu:
        if not r.done and (n == 0 or (r.cls in COUNTED and r.counted and st.vcnt[r.cls] - r.seq >= n)):
            r.done = True
    if n == 0:
        st.unknown_v = False


def apply_vm(st, n):
    """st after ``s_wait_alu depctr_vm_vsrc(n)``."""
    for r in st.msrc:
        if not r.done and (n == 0 or st.mcnt[r.cls] - r.seq >= n):
            r.done = True
    if n == 0:
        st.unknown_m = False


def apply_cnt(st, fields):
    """st after ``s_wait_<counter> N`` for every counter: N of fields."""
    for k, v in fields.items():
        for r in st.loads + st.msrc:
            if not r.done and r.ctr == k and st.ccnt[k] - r.cseq >= v:
                r.done = True
    if fields.get("dscnt") == 0:
        st.unknown_ds = False


def step(st, ins, i, on_need):
    """Advance st over instruction i with the waits in place; on_need(i, field, n) reports every hazard i still has
    (va_vdst with the per-class bound)."""
    x = ins[i]
    if x.cls == "WAIT_ALU":
        f = wait_fields(x)
        if "va_vdst" in f:
            apply_va(st, f["va_vdst"])
        if "vm_vsrc" in f:
            apply_vm(st, f["vm_vsrc"])
    elif x.cls == "WAIT_CNT":
        apply_cnt(st, cnt_fields(x))
    elif x.cls in VALU or x.cls in MEM:
        needs = [("va_vdst", va_need(st, x, "class"))] if x.cls in MEM else []
        needs += [("vm_vsrc", vm_need(st, x))] + load_need(st, x)
        for field, n in needs:
            if n is not None:
                on_need(i, field, n)
        issue(st, x)


# --------------------------------------------------------------------------------------------------------------
# Basic blocks
# --------------------------------------------------------------------------------------------------------------
def blocks(ins):
    """[[start, end, successors]] of the kernel function: a block starts at a label or ``; %bb.N:``."""
    lo, hi = function_range(ins)
    starts = set()
    for i in range(lo, hi + 1):
        x = ins[i]
        if x.cls == "LABEL" or x.text.strip().startswith("; %bb."):
            starts.add(i)
        if x.mn.startswith(("s_branch", "s_cbranch", "s_endpgm", "s_setpc")):
            starts.add(i + 1)
    starts = sorted(s for s in starts if lo <= s <= hi)
    bl = [[s, (starts[k + 1] - 1) if k + 1 < len(starts) else hi, []] for k, s in enumerate(starts)]
    block_of = {ins[i].label: b for b, (s, e, _) in enumerate(bl) for i in range(s, e + 1) if ins[i].label}
    for b, (s, e, succ) in enumerate(bl):
        j = e
        while j > s and not ins[j].mn:
            j -= 1
        mn = ins[j].mn
        if mn.startswith("s_branch"):
            succ.append(block_of[strip_comment(ins[j].ops[0]).strip()])
        elif mn.startswith("s_cbranch"):
            succ.append(block_of[strip_comment(ins[j].ops[0]).strip()])
            if b + 1 < len(bl):
                succ.append(b + 1)
        elif not (mn == "s_endpgm" or mn.startswith("s_setpc")) and b + 1 < len(bl):
            succ.append(b + 1)
    return bl
