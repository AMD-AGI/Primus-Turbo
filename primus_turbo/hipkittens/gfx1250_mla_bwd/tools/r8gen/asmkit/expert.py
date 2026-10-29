###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.expert -- what is in flight at each instruction (expert scheduling mode, gfx1250), and the CFG.

Model, from the toolchain's LLVM (SIInsertWaitcnts.cpp, AMDGPUHWEvents.cpp; vendored from
``gfx1250_mla_bwd/tools/regen_asm_patch.py``):
  * VA_VDST counts in-flight VALU instructions; ``s_wait_alu depctr_va_vdst(N)`` waits until at most N are in
    flight. XDL (WMMA), TRANS, core (CSMACC) and DP VALU complete out of order with respect to each other but in
    order within their class, so N <= (counted class-c instructions issued after X) proves X (class c) complete.
    LLVM uses that bound only while one class is pending and N = 0 otherwise (``llvm=True``).
  * VALU -> VALU is interlocked in hardware; the hazards are: a memory op reading (RAW) or writing (WAW) a register
    an in-flight VALU writes, or writing one it reads (WAR) -> va_vdst; any instruction writing a register an
    in-flight memory op still reads -> vm_vsrc (in order per FLAT / LDS / VMEM); anything touching a register an
    in-flight load writes -> dscnt / loadcnt / storecnt.
  * Entering a region from outside: unknown VALU and memory ops in flight on every register.
"""

from collections import defaultdict

from .isa import COUNTED, MEM, VALU, cnt_fields, function_range, mask, strip_comment, wait_fields


class Rec:
    """An in-flight instruction. VALU: cls XDL / TRANS / CSMACC / UNK, seq = its number in its class.
    Memory op: cls LDS / VMEM / FLAT, seq = its number in that class, ctr / cseq = its wait counter."""

    __slots__ = ("cls", "reads", "writes", "counted", "done", "seq", "ctr", "cseq")

    def __init__(self, cls, reads, writes, counted, seq, ctr=None, cseq=None):
        self.cls, self.reads, self.writes, self.counted = cls, set(reads), set(writes), counted
        self.done, self.seq, self.ctr, self.cseq = False, seq, ctr, cseq

    def copy(self):
        return Rec(self.cls, self.reads, self.writes, self.counted, self.seq, self.ctr, self.cseq)


class State:
    def __init__(self, unknown=False):
        self.valu, self.msrc, self.loads = [], [], []  # in-flight VALU, memory source reads, load writes
        self.vcnt, self.mcnt, self.ccnt = defaultdict(int), defaultdict(int), defaultdict(int)
        self.unknown_v = unknown  # entry from outside: unknown VALU in flight on every VGPR,
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

    def sig(self):
        def recs(v):
            return sorted(
                (r.cls, tuple(sorted(r.reads)), tuple(sorted(r.writes)), r.counted, r.seq, r.ctr, r.cseq)
                for r in v
                if not r.done
            )

        return (
            recs(self.valu),
            recs(self.msrc),
            recs(self.loads),
            dict(self.vcnt),
            dict(self.mcnt),
            dict(self.ccnt),
            self.unknown_v,
            self.unknown_m,
            self.unknown_ds,
        )


def join(states, base=1 << 20):
    """A conservative merge of several states (any path): the union of the pending records, each at its most recent
    position over the states (smallest distance to the current counters), unknown flags or-ed. Every wait the
    merged state asks for is at least as strict as each input state's."""
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


def cap(st, h=64):
    """Clamp every pending record's distance to the current counters at h. Behaviour-preserving (every wait field
    is < h: va_vdst <= 14, vm_vsrc <= 7, counters <= 63), and it makes the state space finite (fixed points)."""
    for name, cnt in (("valu", st.vcnt), ("msrc", st.mcnt), ("loads", st.mcnt)):
        for r in getattr(st, name):
            if cnt[r.cls] - r.seq > h:
                r.seq = cnt[r.cls] - h
            if r.ctr and st.ccnt[r.ctr] - r.cseq > h:
                r.cseq = st.ccnt[r.ctr] - h
    return st


def forget_alu(st):
    """Non-expert mode: the hardware interlocks VALU results and memory source reads; only the counters matter."""
    st.valu, st.msrc = [], []
    st.unknown_v = st.unknown_m = False


def counter_of(x):
    if x.cls == "DS":
        return "dscnt"
    if x.cls in ("VMEM", "FLAT"):
        return "storecnt" if ("_store" in x.mn or "_atomic" in x.mn) else "loadcnt"
    return "tensorcnt" if x.cls == "TDM" else None


def issue(st, x):
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


def va_need(st, x, llvm=False):
    """The largest va_vdst N that resolves every VALU hazard of memory instruction x, None if it has none.
    Per-class bound; llvm=True: LLVM's rule (N = 0 once more than one class, or an unknown one, pends);
    llvm="warv" (abcgen): LLVM's rule except for WAR-V -- x overwrites registers only in-flight WMMAs (XDL) read --
    which gets the per-class bound (the user-approved relaxation, tools/warv_regen.py's rule)."""
    if llvm == "warv":
        return _va_need_warv(st, x)
    if st.unknown_v and (x.uses or x.defs):
        return 0
    rs, ws = set(x.uses), set(x.defs)
    need = None
    for r in st.valu:
        if not r.done and (r.writes & rs or r.writes & ws or r.reads & ws):
            n = st.vcnt[r.cls] - r.seq if (r.cls in COUNTED and r.counted) else 0
            need = n if need is None else min(need, n)
    if need is not None and llvm:
        pending = {r.cls for r in st.valu if not r.done}
        if len(pending) > 1 or "UNK" in pending:
            return 0
    return need


def _va_need_warv(st, x):
    if st.unknown_v and (x.uses or x.defs):
        return 0
    rs, ws = set(x.uses), set(x.defs)
    pending = {r.cls for r in st.valu if not r.done}
    multi = len(pending) > 1 or "UNK" in pending
    need = None
    for r in st.valu:
        if r.done:
            continue
        raw_waw = bool(r.writes & rs or r.writes & ws)
        war = bool(r.reads & ws)
        if not (raw_waw or war):
            continue
        counted = r.cls in COUNTED and r.counted
        n = st.vcnt[r.cls] - r.seq if counted else 0
        warv = (not raw_waw) and war and r.cls == "XDL" and counted
        if multi and not warv:
            n = 0  # LLVM's rule for everything but WAR-V
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
    for r in st.valu:
        if not r.done and (n == 0 or (r.cls in COUNTED and r.counted and st.vcnt[r.cls] - r.seq >= n)):
            r.done = True
    if n == 0:
        st.unknown_v = False


def apply_vm(st, n):
    for r in st.msrc:
        if not r.done and (n == 0 or st.mcnt[r.cls] - r.seq >= n):
            r.done = True
    if n == 0:
        st.unknown_m = False


def apply_cnt(st, fields):
    for k, v in fields.items():
        for r in st.loads + st.msrc:
            if not r.done and r.ctr == k and st.ccnt[k] - r.cseq >= v:
                r.done = True
    if fields.get("dscnt") == 0:
        st.unknown_ds = False


def step(st, ins, i, on_need=None):
    """Advance st over instruction i; on_need(i, field, n) reports every hazard i still has."""
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
        if on_need is not None:
            needs = [("va_vdst", va_need(st, x))] if x.cls in MEM else []
            needs += [("vm_vsrc", vm_need(st, x))] + load_need(st, x)
            for field, n in needs:
                if n is not None:
                    on_need(i, field, n)
        issue(st, x)


def walk(st, ins, rng, on_need=None):
    for i in rng:
        step(st, ins, i, on_need)
    return st


def block_start(ins, i):
    """First line of the basic block holding line i (after a label or a ``; %bb.N:`` comment)."""
    p = i
    while p > 0 and not (ins[p].cls == "LABEL" or ins[p].text.strip().startswith("; %bb.")):
        p -= 1
    return p


def entry_state(ins, first):
    """State at a header entered from outside: the preceding block walked from an unknown state."""
    p = block_start(ins, first - 1)
    return walk(State(unknown=True), ins, range(p + 1, first))


def backedge_state(ins, first, last, passes=4):
    """State at a single-block loop header entered from the back edge: the body iterated (close to) a fixed point."""
    st = walk(entry_state(ins, first), ins, range(first, last + 1))
    for _ in range(passes):
        st = walk(st.copy(), ins, range(first, last + 1))
    return st


def unresolved(ins, first, last):
    """{path: {(instruction ordinal, field)}}: hazards the waits in place leave open in a single-block loop, on the
    entry and back-edge paths."""
    ordinal = {}
    for i in range(first, last + 1):
        if ins[i].mn and ins[i].cls != "WAIT_ALU":
            ordinal[i] = len(ordinal)
    out = {}
    for path, st in (("entry", entry_state(ins, first)), ("back edge", backedge_state(ins, first, last))):
        found = set()
        walk(st, ins, range(first, last + 1), lambda i, f, n, found=found: found.add((ordinal.get(i), f)))
        out[path] = found
    return out


def preloop_vmem_regs(ins, first):
    """VGPRs written by any VMEM load before the loop (the entry path is not tracked further back)."""
    return {r for x in ins[:first] if x.cls in ("VMEM", "FLAT") for r in x.defs}


def clause_violations(ins):
    """Hard clauses holding anything but memory instructions / s_set_vgpr_msb (an illegal instruction)."""
    bad = []
    for k, x in enumerate(ins):
        if x.cls != "CLAUSE":
            continue
        n, q = int(strip_comment(x.text).split()[1], 0) + 1, k + 1
        while n > 0 and q < len(ins):
            y = ins[q]
            if y.cls not in ("DIR", "LABEL"):
                if y.cls not in (*MEM, "MSB") and not y.mn.startswith(("s_load", "s_buffer_load")):
                    bad.append(q + 1)
                n -= 1
            q += 1
    return bad


# --------------------------------------------------------------------------------------------------------------
# Control flow and VGPR liveness
# --------------------------------------------------------------------------------------------------------------
def blocks(ins, name=None):
    """[[start, end, successors]] of a kernel function: a block starts at a label or ``; %bb.N:``."""
    lo, hi = function_range(ins, name)
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


def liveness(ins, bl):
    """Live-out VGPR mask of every block (a full write kills, a partial write is also a read)."""
    use, dfn = [], []
    for s, e, _ in bl:
        u = d = 0
        for i in range(s, e + 1):
            x = ins[i]
            u |= mask(x.uses) & ~d
            if not x.rmw:
                d |= mask(x.defs)
        use.append(u)
        dfn.append(d)
    lin, lout = [0] * len(bl), [0] * len(bl)
    changed = True
    while changed:
        changed = False
        for b in range(len(bl) - 1, -1, -1):
            o = 0
            for s in bl[b][2]:
                o |= lin[s]
            n = use[b] | (o & ~dfn[b])
            if n != lin[b] or o != lout[b]:
                lin[b], lout[b], changed = n, o, True
    return lout


def live_in(ins, bl, lout):
    """Live-in VGPR mask of every block."""
    out = []
    for b, (s, e, _) in enumerate(bl):
        live = lout[b]
        for i in range(e, s - 1, -1):
            x = ins[i]
            if not x.rmw:
                live &= ~mask(x.defs)
            live |= mask(x.uses)
        out.append(live)
    return out


def live_after(ins, block, live_out):
    """{i: mask of the VGPRs live after instruction i} over one block."""
    s, e, _ = block
    live, out = live_out, {}
    for i in range(e, s - 1, -1):
        out[i] = live
        x = ins[i]
        if not x.rmw:
            live &= ~mask(x.defs)
        live |= mask(x.uses)
    return out


def scalar_liveness(ins, bl, defs_uses):
    """(live-in, live-out) scalar register sets of every block; defs_uses(x) -> (defs, uses) (isa.sgpr_defs_uses).
    Conservative for partial writes: a def of an SCC-writing / sub-register op kills it."""
    use, dfn = [], []
    for s, e, _ in bl:
        u, d = set(), set()
        for i in range(s, e + 1):
            sd, su = defs_uses(ins[i])
            u |= set(su) - d
            d |= set(sd)
        use.append(u)
        dfn.append(d)
    lin, lout = [set() for _ in bl], [set() for _ in bl]
    changed = True
    while changed:
        changed = False
        for b in range(len(bl) - 1, -1, -1):
            o = set()
            for s in bl[b][2]:
                o |= lin[s]
            n = use[b] | (o - dfn[b])
            if n != lin[b] or o != lout[b]:
                lin[b], lout[b], changed = n, o, True
    return lin, lout
