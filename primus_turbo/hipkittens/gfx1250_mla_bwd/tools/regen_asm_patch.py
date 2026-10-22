###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regenerate ``hip/asm/k_dkdv96ds.s.patch``, the assembly patch hsaco.py applies to k_dkdv96ds.

Why: in k_dkdv96ds's full loop the 24 Q^T ``ds_load_tr16_b128`` of an iteration are interleaved with
the dV WMMAs, and the register allocator puts their destinations into the b_do registers those WMMAs
have just read. In expert scheduling mode (``-amdgpu-expert-scheduling-mode``) a load that overwrites
a register an in-flight WMMA still reads must first drain the VALU queue (``s_wait_alu
depctr_va_vdst(0)``), and LLVM has no knob against it (its allocator does not know about in-flight
VALU reads). The patch moves those destination tuples, and the WMMA operands that read them, to
registers that are free over the value's whole live range, inside the same 256-VGPR block (so every
``s_set_vgpr_msb`` stays valid), then recomputes the loop's ``va_vdst`` waits with LLVM's own rules.
Only VGPR numbers and ``s_wait_alu`` lines change; dK / dV / dS stay bitwise identical.

Steps:
  1. compile k_dkdv96ds exactly as hsaco.py does (manifest flags, ``-save-temps=obj``) and read the
     compiler's own ``.s``: its sha256 is the manifest's ``asm_input_sha256``;
  2. find the G == 1 full loop and its Q^T tr16 loads (``full_loop``, ``qt_loads``), group them into
     WMMA operand tuples and re-target every tuple whose loads need a near-drain (``retarget``);
  3. recompute every ``va_vdst`` of the loop with LLVM's model (``recompute_va_vdst``);
  4. run the static checks (``static_checks``) and write the unified diff;
  5. assemble the patched ``.s`` through hsaco.py's own build path and print the sha256 values that go
     into manifest.json (``asm_input_sha256``, ``hsaco_sha256``).

    python primus_turbo/hipkittens/gfx1250_mla_bwd/tools/regen_asm_patch.py [--check] [--cxx PATH]
        [--out PATH] [--keep DIR] [--asm FILE]

``--check`` writes nothing and fails unless the regenerated patch equals the committed one byte for
byte and both sha256 values equal the manifest's. ``--asm`` regenerates from a saved ``.s`` (no
compiler, no code object). Standard library only, like hsaco.py (loaded by path). The analysis
depends on the compiler's ``.s`` format: the ``/*vN*/`` comments with the full register numbers under
``s_set_vgpr_msb``, and the ``s_clause`` / ``s_wait_*`` / ``s_set_vgpr_msb`` syntax.

Model of expert scheduling mode, from the toolchain's LLVM (SIInsertWaitcnts.cpp, AMDGPUHWEvents.cpp):
  * VA_VDST counts in-flight VALU instructions; ``s_wait_alu depctr_va_vdst(N)`` waits until at most N
    are in flight. XDL (WMMA), TRANS, core (CSMACC) and DP VALU complete out of order with respect to
    each other but in order within their class, so N <= (counted class-c instructions issued after X)
    proves X (class c) complete. LLVM uses that bound only while one class is pending and N = 0
    otherwise. The patch's waits follow LLVM's rule; the per-class bound only picks what to re-target.
  * VALU -> VALU is interlocked in hardware; the hazards are: a memory op reading (RAW) or writing (WAW)
    a register an in-flight VALU writes, or writing one it reads (WAR) -> va_vdst; any instruction
    writing a register an in-flight memory op still reads -> vm_vsrc (in order per FLAT / LDS / VMEM);
    anything touching a register an in-flight load writes -> dscnt / loadcnt / storecnt.
  * Entering the loop from outside: unknown VALU and memory ops in flight on every register.
"""

import argparse
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
KERNEL = "k_dkdv96ds"

# The G == 1 full loop of k_dkdv96ds.hip's k3 body, as compiled: two iterations per trip of 120 WMMAs.
# An iteration starts with its NL/ND buffer_loads, then the dO^T tr16 burst (NDO_V = 8 B operands of 2
# tr16) and the Q^T burst (NDO_QK = 12 of 2). qt_loads checks this structure instead of assuming it.
LOOP_WMMAS = 240
DOT_TR16, QT_TR16 = 16, 24
# A tuple is re-targeted when its loads need va_vdst < MIN_WAIT (a near-drain), and only to registers
# that need no wait below MIN_WAIT either.
MIN_WAIT = 6
VA_VDST_MAX = 14  # LLVM's largest explicit va_vdst (15 = no wait)


class RegenError(RuntimeError):
    """The kernel's loop has not the expected shape, or the edited assembly fails a static check."""


# --------------------------------------------------------------------------------------------------
# Parsing the compiler's .s
# --------------------------------------------------------------------------------------------------
TRANS_OPS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_", "v_tanh_")
VALU = ("XDL", "TRANS", "CSMACC", "DPMACC")
MEM = ("DS", "VMEM", "FLAT", "TDM")
COUNTED = ("XDL", "TRANS", "CSMACC")  # VALU classes whose in-class order bounds va_vdst
# Lines between a memory instruction and the s_wait_alu that protects it.
WINDOW = ("MSB", "DELAY", "WAIT_CNT", "WAIT_ALU", "DIR", "CLAUSE")
REG_RE = re.compile(r"^v\[(\d+):(\d+)\]$|^v(\d+)$")
TOKEN_RE = re.compile(r"v\[(\d+):(\d+)\](\s*/\*\s*v\[(\d+):(\d+)\]\s*\*/)?|v(\d+)(\s*/\*\s*v(\d+)\s*\*/)?")
CNT_RE = re.compile(
    r"s_wait_(dscnt|loadcnt|storecnt|loadcnt_dscnt|storecnt_dscnt|tensorcnt|xcnt|kmcnt|samplecnt|bvhcnt)"
    r"\s+(0x[0-9a-fA-F]+|\d+)"
)


def regs_of(tok):
    """Full VGPR numbers of an operand token (``v[26:33] /*v[794:801]*/``: the comment has the true ones)."""
    m = re.search(r"/\*\s*(v\[?[\d:]+\]?)\s*\*/", tok)
    t = (m.group(1) if m else tok).strip().strip("|-")
    t = re.sub(r"^(neg|abs)\((.*)\)$", r"\2", t)
    m = REG_RE.match(t)
    if not m:
        return []
    if m.group(3) is not None:
        return [int(m.group(3))]
    return list(range(int(m.group(1)), int(m.group(2)) + 1))


def strip_comment(line):
    """The code of a line: ``;`` starts a comment, ``/* */`` comments are kept (they carry registers)."""
    out, i = "", 0
    while i < len(line):
        if line.startswith("/*", i):
            j = line.index("*/", i)
            out += line[i : j + 2]
            i = j + 2
            continue
        if line[i] == ";":
            break
        out += line[i]
        i += 1
    return out.rstrip()


def split_operands(s):
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


def first_tok(op):
    """The register part of an operand (``v[0:7] /*v[256:263]*/ matrix_a_reuse`` -> without the modifier)."""
    m = re.match(r"^(\S+(\s*/\*[^*]*\*/)?)", op)
    return m.group(1) if m else op


def classify(mn):
    if mn.startswith(("v_wmma", "v_swmmac")):
        return "XDL"
    if mn.startswith("v_"):
        if mn == "v_nop":
            return "VNOP"
        if mn.startswith(TRANS_OPS):
            return "TRANS"
        if "_f64" in mn or re.match(r"v_cmpx?_\w+_[iuf](64|16)", mn):
            return "DPMACC"
        return "CSMACC"
    if mn.startswith("ds_"):
        return "DS"
    if mn.startswith(("buffer_", "tbuffer_")):
        return "VMEM"
    if mn.startswith(("global_", "flat_", "scratch_")):
        return "FLAT"
    if mn.startswith("tensor_"):
        return "TDM"
    if mn.startswith("s_wait_alu"):
        return "WAIT_ALU"
    if mn.startswith("s_wait_"):
        return "WAIT_CNT"
    if mn == "s_set_vgpr_msb":
        return "MSB"
    if mn == "s_clause":
        return "CLAUSE"
    if mn.startswith("s_barrier"):
        return "BARRIER"
    if mn.startswith("s_delay_alu"):
        return "DELAY"
    if mn.startswith("s_"):
        return "SALU"
    return "OTHER"


class Ins:
    """One line: mnemonic, operands, VGPRs read (uses) and written (defs); rmw: partial write (a use too)."""

    __slots__ = ("li", "text", "mn", "ops", "defs", "uses", "cls", "rmw", "label")


def parse_line(li, line):
    x = Ins()
    x.li, x.text, x.label, x.mn, x.ops, x.defs, x.uses, x.rmw = li, line, None, "", [], [], [], False
    s = strip_comment(line).strip()
    if not s:
        x.cls = "DIR"
        return x
    if s.endswith(":") and " " not in s:
        x.cls, x.label = "LABEL", s[:-1]
        return x
    if s.startswith("."):
        x.cls = "DIR"
        return x
    if "::" in s:  # VOPD dual issue: both halves
        a, b = s.split("::", 1)
        i1, i2 = parse_line(li, "\t" + a), parse_line(li, "\t" + b)
        x.mn, x.ops = i1.mn + "::" + i2.mn, i1.ops + ["::"] + i2.ops
        x.defs, x.uses, x.cls = i1.defs + i2.defs, i1.uses + i2.uses, "CSMACC"
        return x
    parts = s.split(None, 1)
    x.mn = parts[0]
    x.cls = classify(x.mn)
    x.ops = split_operands(parts[1]) if len(parts) > 1 else []
    regs = [regs_of(first_tok(o)) for o in x.ops]
    d, u = [], []
    if x.cls in VALU:
        if x.mn.startswith(("v_cmp", "v_readlane", "v_readfirstlane")):
            u = [r for rr in regs for r in rr]
        elif x.mn.startswith("v_writelane"):
            d, x.rmw = list(regs[0]), True
            u = [r for rr in regs for r in rr]
        else:
            d = list(regs[0]) if regs else []
            u = [r for rr in regs[1:] for r in rr]
            if "_dpp" in x.mn or "row_" in line or "quad_perm" in line:  # DPP: partial unless all lanes write
                m1 = re.search(r"row_mask:(0x[0-9a-fA-F]+|\d+)", line)
                m2 = re.search(r"bank_mask:(0x[0-9a-fA-F]+|\d+)", line)
                full = (not m1 or int(m1.group(1), 0) == 0xF) and (not m2 or int(m2.group(1), 0) == 0xF)
                x.rmw = x.rmw or not full or "bound_ctrl" not in line
            if re.search(r"v\d+\.(h|l)\b", line) or "_t16" in x.mn:  # 16-bit halves
                x.rmw = True
            if x.rmw:
                u += d
    elif x.cls in ("DS", "VMEM", "FLAT") and (
        x.mn.startswith(("ds_load", "ds_read")) or (x.cls != "DS" and "_load" in x.mn and "lds" not in x.mn)
    ):
        d = list(regs[0])
        u = [r for rr in regs[1:] for r in rr]
    else:
        u = [r for rr in regs for r in rr]
    x.defs, x.uses = d, u
    return x


def parse(lines):
    return [parse_line(i, line) for i, line in enumerate(lines)]


def wait_fields(x):
    return {k: int(v) for k, v in re.findall(r"depctr_(\w+)\((\d+)\)", x.text)}


def fmt_wait(fields):
    if "va_vdst" in fields:
        fields = dict(fields, va_vdst=min(fields["va_vdst"], VA_VDST_MAX))
    order = ["va_vdst", "vm_vsrc", "va_sdst", "va_ssrc", "hold_cnt", "va_vcc", "sa_sdst"]
    keys = sorted(fields, key=lambda k: order.index(k) if k in order else 99)
    return "\ts_wait_alu " + " ".join(f"depctr_{k}({fields[k]})" for k in keys)


def cnt_fields(x):
    m = CNT_RE.search(strip_comment(x.text))
    if not m:
        return {}
    k, v = m.group(1), int(m.group(2), 0)
    if k in ("loadcnt_dscnt", "storecnt_dscnt"):
        return {k.split("_")[0]: v >> 8, "dscnt": v & 0xFF}
    return {k: v}


def mask(regs):
    m = 0
    for r in regs:
        m |= 1 << r
    return m


# --------------------------------------------------------------------------------------------------
# Control flow and VGPR liveness
# --------------------------------------------------------------------------------------------------
def blocks(ins):
    """[[start, end, successors]] of the kernel function: a block starts at a label or ``; %bb.N:``."""
    lo = next(i for i, x in enumerate(ins) if x.label == KERNEL)
    hi = next(i for i in range(lo, len(ins)) if ins[i].text.strip().startswith(".Lfunc_end")) - 1
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
            succ.append(block_of[ins[j].ops[0]])
        elif mn.startswith("s_cbranch"):
            succ.append(block_of[ins[j].ops[0]])
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


def full_loop(ins):
    """(label, first, last) of the G == 1 full loop: the first single-block loop (a label, then an
    s_cbranch back to it before any other label) with LOOP_WMMAS WMMAs."""
    for i, x in enumerate(ins):
        if not x.label:
            continue
        for j in range(i + 1, len(ins)):
            y = ins[j]
            if y.label:
                break
            if y.mn.startswith("s_cbranch") and strip_comment(y.text).strip().endswith(" " + x.label):
                if sum(ins[k].cls == "XDL" for k in range(i, j)) == LOOP_WMMAS:
                    return x.label, i, j
                break
    raise RegenError(f"no single-block loop with {LOOP_WMMAS} WMMAs (k_dkdv96ds's G == 1 full loop)")


# --------------------------------------------------------------------------------------------------
# Expert-mode hazard model: what is in flight at each instruction of the loop
# --------------------------------------------------------------------------------------------------
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
    Per-class bound; llvm=True: LLVM's rule (N = 0 once more than one class, or an unknown one, pends)."""
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


def step(st, ins, i, on_need=None, va_override=None):
    """Advance st over instruction i; on_need(i, field, n) reports every hazard i still has."""
    x = ins[i]
    if x.cls == "WAIT_ALU":
        f = wait_fields(x)
        va = va_override[i] if va_override is not None and i in va_override else f.get("va_vdst")
        if va is not None:
            apply_va(st, va)
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


def walk(st, ins, rng, on_need=None, va_override=None):
    for i in rng:
        step(st, ins, i, on_need, va_override)
    return st


def entry_state(ins, first):
    """State at the loop header entered from outside: the preheader block walked from an unknown state."""
    p = first - 1
    while p > 0 and not (ins[p].cls == "LABEL" or ins[p].text.strip().startswith("; %bb.")):
        p -= 1
    return walk(State(unknown=True), ins, range(p + 1, first))


def backedge_state(ins, first, last, va_override=None, passes=4):
    """State at the loop header entered from the back edge: the body iterated (close to) a fixed point."""
    st = walk(entry_state(ins, first), ins, range(first, last + 1), va_override=va_override)
    for _ in range(passes):
        st = walk(st.copy(), ins, range(first, last + 1), va_override=va_override)
    return st


def unresolved(ins, first, last):
    """{path: {(instruction ordinal, field)}}: hazards the waits in place leave open, on both paths."""
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


# --------------------------------------------------------------------------------------------------
# Re-targeting the Q^T tr16 loads
# --------------------------------------------------------------------------------------------------
def qt_loads(ins, first, last):
    """The Q^T tr16 loads of the loop: per iteration (each starts at its NL/ND buffer_loads) the
    ds_load_tr16_b128 after the first DOT_TR16 (dO^T). Raises if the loop no longer has that shape."""
    iters, cur = [], None
    for i in range(first, last + 1):
        x = ins[i]
        if x.cls == "VMEM" and x.defs and (cur is None or cur):
            cur = []
            iters.append(cur)
        elif x.mn == "ds_load_tr16_b128":
            if cur is None:
                raise RegenError(f"tr16 load before the NL/ND loads at line {i + 1}")
            cur.append(i)
    if len(iters) != LOOP_WMMAS // 120 or any(len(it) != DOT_TR16 + QT_TR16 for it in iters):
        raise RegenError(
            f"expected {LOOP_WMMAS // 120} iterations of {DOT_TR16} dO^T + {QT_TR16} Q^T tr16 loads, got "
            f"{[len(it) for it in iters]}: the kernel changed shape, update qt_loads"
        )
    return [i for it in iters for i in it[DOT_TR16:]]


def operand_units(ins, first, last, loads):
    """Group the loads whose destinations end up in one operand tuple of a later instruction (two tr16
    b128 halves form one 8-VGPR WMMA operand); in program order of their first load."""
    parent = {i: i for i in loads}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}  # register -> the selected load whose value it holds
    for j in range(first, last + 1):
        x = ins[j]
        if not x.mn:
            continue
        for o in x.ops:
            roots = {find(owner[r]) for r in regs_of(first_tok(o)) if r in owner}
            if len(roots) > 1:
                root = roots.pop()
                for k in roots:
                    parent[k] = root
        for r in x.defs:
            if j in parent:
                owner[r] = j
            else:
                owner.pop(r, None)
    groups = defaultdict(list)
    for i in sorted(loads):
        groups[find(i)].append(i)
    return sorted(groups.values())


def value_range(ins, last, unit):
    """(registers, last use, users) of the value the unit's loads write (until each register is rewritten)."""
    regs = {r for i in unit for r in ins[i].defs}
    alive, last_use, users = set(), None, []
    for j in range(unit[0], last + 1):
        x = ins[j]
        if j in unit:
            alive |= set(x.defs)
            continue
        if not x.mn:
            continue
        if set(x.uses) & alive:
            last_use = j
            users.append(j)
        if not x.rmw:
            alive -= set(x.defs)
        if not alive and j > unit[-1]:
            break
    return regs, last_use, users


def clause_of(ins, i, lo):
    """Index of the s_clause whose hard clause (the N+1 instructions after it) holds i, or None."""
    k, n = i - 1, 0
    while k >= lo and k > i - 64:
        x = ins[k]
        if x.cls == "CLAUSE":
            return k if n < int(strip_comment(x.text).split()[1], 0) + 1 else None
        if x.cls not in ("DIR", "LABEL"):
            n += 1
        k -= 1
    return None


def window_start(ins, i, lo):
    """First line of the run of wait / msb / clause lines in front of memory instruction i (its clause's
    s_clause if it is in one): the waits that protect i."""
    c = clause_of(ins, i, lo)
    k = i if c is None else c
    while k - 1 >= lo and ins[k - 1].cls in WINDOW:
        k -= 1
    return k


def unit_need(ins, first, last, unit):
    """(N, mem): the smallest va_vdst the unit's loads need with their current registers (99: none), per
    class, on both paths into the body, in the state before each load's own waits; mem: an in-flight
    memory op (tracked in the loop) still reads or writes them."""
    worst, mem = 99, False
    for st in (entry_state(ins, first), backedge_state(ins, first, last)):
        pos, snap = first, None
        for j in unit:
            k0 = window_start(ins, j, first)
            if k0 < pos:  # shares the previous load's window (one clause): from that window's start
                st = snap.copy()
            else:
                walk(st, ins, range(pos, k0))
            snap = st.copy()
            n = va_need(st, ins[j])
            worst = min(worst, 99 if n is None else n)
            ws = set(ins[j].defs)
            mem |= any(not r.done and r.reads & ws for r in st.msrc)
            mem |= any(not r.done and r.writes & ws for r in st.loads)
            walk(st, ins, range(k0, j + 1))
            pos = j + 1
    return worst, mem


def render(lo, hi):
    """A register tuple as the compiler prints it: number in its 256-VGPR block, true number in a comment."""
    blk = lo // 256
    a, b = lo - 256 * blk, hi - 256 * blk
    if lo == hi:
        return f"v{a}" + (f" /*v{lo}*/" if blk else "")
    return f"v[{a}:{b}]" + (f" /*v[{lo}:{hi}]*/" if blk else "")


def remap(text, rmap):
    """text with every register token whose registers are all in rmap renamed (a partial overlap raises)."""
    k = text.rfind("*/")
    semi = text.find(";", k + 2 if k >= 0 else 0)
    code, comment = (text, "") if semi < 0 else (text[:semi], text[semi:])
    out, pos = "", 0
    for m in TOKEN_RE.finditer(code):
        if m.group(1) is not None:
            lo, hi = (int(m.group(4)), int(m.group(5))) if m.group(3) else (int(m.group(1)), int(m.group(2)))
        else:
            lo = hi = int(m.group(8)) if m.group(7) else int(m.group(6))
        hit = [r in rmap for r in range(lo, hi + 1)]
        if any(hit):
            if not all(hit):
                raise RegenError(f"partial overlap with a re-targeted tuple: {text.strip()}")
            out += code[pos : m.start()] + render(rmap[lo], rmap[hi])
            pos = m.end()
    return out + code[pos:] + comment


def retarget(lines, first, last, loads, log):
    """Re-target the units of ``loads`` in place in ``lines``; returns the units (line indices) moved.

    A unit (``operand_units``) is moved when its loads need va_vdst < MIN_WAIT (``unit_need``) and its
    value is dead after its last use (no use across the back edge). The new tuple:
      * lies in the old tuple's 256-VGPR block (every s_set_vgpr_msb stays valid), even-aligned, disjoint
        from the old one;
      * is dead over the whole live range (function-level liveness) and referenced by no other
        instruction inside it; no VMEM load before the loop writes it (entry path);
      * at the unit's first load, on both paths into the body: no in-flight memory op reads or writes it,
        and in-flight VALU on it needs no va_vdst below MIN_WAIT.
    Candidates rank by that wait (none best), then by the distance to their last reference before the
    load, then by register number. Units are taken greedily in program order, liveness recomputed after
    each move."""
    ins = parse(lines)
    pre_vmem = mask(preloop_vmem_regs(ins, first))
    bl = blocks(ins)
    b = next(k for k, (s, e, _) in enumerate(bl) if s <= first <= e)
    if bl[b][:2] != [first, last]:
        raise RegenError(f"the full loop (lines {first + 1}-{last + 1}) is not one basic block")
    moved = []
    for unit in operand_units(ins, first, last, loads):
        where = f"lines {'/'.join(str(i + 1) for i in unit)}"
        la = live_after(ins, bl[b], liveness(ins, bl)[b])
        cost = unit_need(ins, first, last, unit)[0]
        if cost >= MIN_WAIT:
            continue
        regs, lu, users = value_range(ins, last, unit)
        lo, size = min(regs), max(regs) - min(regs) + 1
        if lu is None or mask(regs) & la[lu] or sorted(regs) != list(range(lo, lo + size)):
            log(f"  {where} v[{lo}:{lo + size - 1}]: unused, live around the loop or not one tuple; kept")
            continue
        busy = pre_vmem | la.get(unit[0] - 1, 0)
        for j in range(unit[0], lu + 1):
            busy |= la[j]
            if j not in unit and j not in users:
                busy |= mask(ins[j].uses) | mask(ins[j].defs)
        paths = (entry_state(ins, first), backedge_state(ins, first, last))
        states = [walk(st, ins, range(first, unit[0])) for st in paths]
        cands = []
        blk = lo // 256 * 256
        for base in range(blk, blk + 256 - size + 1, 2):
            new = set(range(base, base + size))
            if new & regs or mask(new) & busy:
                continue
            worst, ok = 99, True
            for st in states:
                for r in st.valu:
                    if not r.done and (r.reads | r.writes) & new:
                        worst = min(worst, st.vcnt[r.cls] - r.seq if (r.cls in COUNTED and r.counted) else 0)
                ok &= not any(not r.done and r.reads & new for r in st.msrc)
                ok &= not any(not r.done and r.writes & new for r in st.loads)
            if not ok or worst < MIN_WAIT:
                continue
            m_new = mask(new)
            refs = (j for j in range(unit[0] - 1, first - 1, -1) if mask(ins[j].uses + ins[j].defs) & m_new)
            dist = unit[0] - 1 - next(refs, first - 1)
            cands.append((-worst, -dist, base))
        if not cands:
            log(f"  {where} v[{lo}:{lo + size - 1}]: needs va_vdst({cost}), no free tuple in its block; kept")
            continue
        base = min(cands)[2]
        rmap = {lo + k: base + k for k in range(size)}
        for j in list(unit) + users:
            text = remap(lines[j], rmap)
            if text == lines[j]:
                raise RegenError(f"line {j + 1} does not reference the moved tuple: {text.strip()}")
            lines[j] = text
        ins = parse(lines)
        moved.append(unit)
        log(
            f"  {where} v[{lo}:{lo + size - 1}] -> v[{base}:{base + size - 1}] "
            f"(needed va_vdst({cost}); {len(users)} users, {len(cands)} candidates)"
        )
    return moved


# --------------------------------------------------------------------------------------------------
# va_vdst recomputation (LLVM's rule)
# --------------------------------------------------------------------------------------------------
def wait_slot(ins, i, lo):
    """(existing s_wait_alu to tighten or None, line to insert a new one before) for memory instruction i:
    the s_wait_alu among the wait / msb lines right before i; never inside a hard clause (an s_wait_alu
    there is an illegal instruction): before its s_clause instead."""
    c = clause_of(ins, i, lo)
    if c is not None:
        i = c
    k = i - 1
    while k >= lo and ins[k].cls in WINDOW:
        if ins[k].cls == "WAIT_ALU":
            return k, i
        k -= 1
    return None, i


def recompute_va_vdst(ins, first, last, iters=6):
    """New va_vdst of the loop: {line: N or None (drop the field)} for its s_wait_alu lines and
    {("ins", line): N} for new lines inserted before ``line``. Every memory instruction gets the
    smallest wait LLVM's rule asks for on either path into the body (the back-edge state from the
    previous round's waits; a few rounds reach the fixed point)."""
    body = range(first, last + 1)
    plan = {i: None for i in body if ins[i].cls == "WAIT_ALU"}
    for _ in range(iters):
        new = {i: None for i in plan}
        override = {k: v for k, v in plan.items() if not isinstance(k, tuple)}
        states = [entry_state(ins, first), backedge_state(ins, first, last, override)]
        for i in body:
            x = ins[i]
            if x.cls in MEM:
                ns = [n for n in (va_need(st, x, llvm=True) for st in states) if n is not None]
                if ns:
                    n = min(ns)
                    at, before = wait_slot(ins, i, first)
                    key = at if at is not None else ("ins", before)
                    new[key] = n if new.get(key) is None else min(new[key], n)
                    for st in states:
                        apply_va(st, n)
            if x.cls == "WAIT_ALU":
                f = wait_fields(x)
                for st in states:
                    if "vm_vsrc" in f:
                        apply_vm(st, f["vm_vsrc"])
                    if new.get(i) is not None:
                        apply_va(st, new[i])
            elif x.cls == "WAIT_CNT":
                for st in states:
                    apply_cnt(st, cnt_fields(x))
            else:
                for st in states:
                    issue(st, x)
        if new == plan:
            break
        plan = new
    return plan


def apply_plan(lines, ins, plan):
    """(before, after) per line: the s_wait_alu inserted before it (or None), its new text (None: dropped)."""
    before, after = [None] * len(lines), list(lines)
    for k, v in plan.items():
        if isinstance(k, tuple):
            before[k[1]] = fmt_wait({"va_vdst": v})
            continue
        f = wait_fields(ins[k])
        g = {key: val for key, val in f.items() if key != "va_vdst"}
        if v is not None:
            g["va_vdst"] = v
        if g != f:
            after[k] = fmt_wait(g) if g else None
    return before, after


# --------------------------------------------------------------------------------------------------
# Static checks of the edited assembly
# --------------------------------------------------------------------------------------------------
def dataflow(ins, first, last):
    """Reaching definitions (instruction ordinal, position in its defs) of every register each body
    instruction reads, over two trips (the second sees the back edge), and the definitions at the exit."""
    body = [i for i in range(first, last + 1) if ins[i].mn and ins[i].cls != "WAIT_ALU"]
    cur, reads = {}, []
    for _ in range(2):
        for k, i in enumerate(body):
            reads.append([cur.get(r, ("in", r)) for r in ins[i].uses])
            cur.update({r: (k, p) for p, r in enumerate(ins[i].defs)})
    return reads, cur


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


def static_checks(orig, edited, moved, log):
    """Failures (empty = pass) of the edited text against the compiler's (moved: the re-targeted units as
    line indices of the edited text):
      1. outside the loop identical; inside, apart from s_wait_alu lines, the same instructions with only
         VGPR numbers changed, every operand keeping its tuple size and its 256-VGPR block (MSB state);
      2. registers free: every instruction reads the same definitions (two trips, so across the back
         edge too), the same values are live at the exit, no new register was a pre-loop VMEM target;
      3. the moved loads need no near-drain any more and no tracked memory op is in flight on them;
      4. no hazard left open that the original does not leave open too (entry and back-edge paths);
      5. no s_wait_alu (nor anything but memory ops / s_set_vgpr_msb) inside a hard clause;
      6. every va_vdst at least as strict as LLVM's own model asks for."""
    fail = []
    io, ie = parse(orig), parse(edited)
    _, fo, lo = full_loop(io)
    _, fe, le = full_loop(ie)
    if orig[:fo] != edited[:fe] or orig[lo + 1 :] != edited[le + 1 :]:
        fail.append("text outside the loop differs")
    keep_o = [i for i in range(fo, lo + 1) if io[i].cls != "WAIT_ALU"]
    keep_e = [i for i in range(fe, le + 1) if ie[i].cls != "WAIT_ALU"]
    new_regs = set()
    if len(keep_o) != len(keep_e):
        fail.append(f"the loop has {len(keep_e)} non-wait lines, the original {len(keep_o)}")
    for a, b in zip(keep_o, keep_e):
        if orig[a] == edited[b]:
            continue
        if TOKEN_RE.sub("R", orig[a]) != TOKEN_RE.sub("R", edited[b]):
            fail.append(f"line {b + 1} changes more than register numbers: {edited[b].strip()}")
            continue
        for ta, tb in zip(io[a].ops, ie[b].ops):
            ra, rb = regs_of(first_tok(ta)), regs_of(first_tok(tb))
            if len(ra) != len(rb) or (ra and ra[0] // 256 != rb[0] // 256):
                fail.append(f"line {b + 1}: {ta} -> {tb} changes the tuple size or the 256-VGPR block")
            if ra != rb:
                new_regs |= set(rb)
    reads_o, out_o = dataflow(io, fo, lo)
    reads_e, out_e = dataflow(ie, fe, le)
    bl = blocks(io)
    live_out = liveness(io, bl)[next(k for k, (s, e, _) in enumerate(bl) if s == fo)]
    exits = [r for r in range(1024) if live_out >> r & 1]
    if reads_o != reads_e or any(out_o.get(r, r) != out_e.get(r, r) for r in exits):
        fail.append("dataflow differs: a re-targeted tuple overwrites a live value")
    pre_vmem = preloop_vmem_regs(io, fo)
    if new_regs & pre_vmem:
        fail.append(f"re-targeted to VMEM load targets before the loop {sorted(new_regs & pre_vmem)}")
    for unit in moved:
        n, mem = unit_need(ie, fe, le, unit)
        if n < MIN_WAIT or mem:
            fail.append(f"moved loads at lines {[i + 1 for i in unit]} need va_vdst({n}) or a memory wait")
    ho, he = unresolved(io, fo, lo), unresolved(ie, fe, le)
    for path in he:
        if he[path] - ho[path]:
            fail.append(f"{len(he[path] - ho[path])} new open hazards on the {path} path")
    if clause_violations(ie):
        fail.append(f"non-memory instructions inside hard clauses at lines {clause_violations(ie)}")
    plan = recompute_va_vdst(ie, fe, le)
    loose = [
        k
        for k, v in plan.items()
        if v is not None and (isinstance(k, tuple) or wait_fields(ie[k]).get("va_vdst", 99) > v)
    ]
    if loose:
        fail.append(f"{len(loose)} va_vdst waits looser than LLVM's model, e.g. {loose[:4]}")

    def drains(ins, lines):
        return sum(ins[i].cls == "WAIT_ALU" and wait_fields(ins[i]).get("va_vdst") == 0 for i in lines)

    log(f"  va_vdst(0) per loop trip: {drains(io, range(fo, lo))} -> {drains(ie, range(fe, le))}")
    return fail


# --------------------------------------------------------------------------------------------------
# The patch
# --------------------------------------------------------------------------------------------------
def unified_diff(ops, old_label, new_label, context=3):
    """``diff -u`` text of an edit script [(tag, line)]: tag ' ' (kept), '-' (old only), '+' (new only)."""
    old_no, new_no, o, n = [], [], 0, 0
    for tag, _ in ops:
        old_no.append(o)
        new_no.append(n)
        o += tag != "+"
        n += tag != "-"
    hunks = []
    for k in (k for k, (tag, _) in enumerate(ops) if tag != " "):
        if hunks and k - hunks[-1][1] <= 2 * context:
            hunks[-1][1] = k + 1
        else:
            hunks.append([k, k + 1])

    def rng(start, count):
        return f"{start + 1}" if count == 1 else f"{start + (count > 0)},{count}"

    out = [f"--- {old_label}\n", f"+++ {new_label}\n"]
    for s, e in hunks:
        s, e = max(0, s - context), min(len(ops), e + context)
        body = ops[s:e]
        n_old, n_new = sum(t != "+" for t, _ in body), sum(t != "-" for t, _ in body)
        out.append(f"@@ -{rng(old_no[s], n_old)} +{rng(new_no[s], n_new)} @@\n")
        k = 0
        while k < len(body):
            if body[k][0] == " ":
                out.append(f" {body[k][1]}\n")
                k += 1
                continue
            j = k
            while j < len(body) and body[j][0] != " ":
                j += 1
            change = body[k:j]  # deletions first, then insertions, as diff prints a change
            out += [f"-{t}\n" for tag, t in change if tag == "-"]
            out += [f"+{t}\n" for tag, t in change if tag == "+"]
            k = j
    return "".join(out)


def regenerate(text, log=print):
    """(edited text, patch) for the compiler's .s text; RegenError if the loop has no expected shape or
    a static check fails."""
    if not text.endswith("\n"):
        raise RegenError("the .s does not end with a newline")
    orig = text[:-1].split("\n")
    ins = parse(orig)
    label, first, last = full_loop(ins)
    loads = qt_loads(ins, first, last)
    log(f"  full loop {label}: lines {first + 1}-{last + 1}, {len(loads)} Q^T tr16 loads")
    cur = list(orig)
    moved = retarget(cur, first, last, loads, log)
    cur_ins = parse(cur)
    before, after = apply_plan(cur, cur_ins, recompute_va_vdst(cur_ins, first, last))
    ops, edited, new_index = [], [], {}
    for i, line in enumerate(orig):
        if before[i] is not None:
            ops.append(("+", before[i]))
            edited.append(before[i])
        if after[i] is None:
            ops.append(("-", line))
            continue
        ops += [("-", line), ("+", after[i])] if after[i] != line else [(" ", line)]
        new_index[i] = len(edited)
        edited.append(after[i])
    fail = static_checks(orig, edited, [[new_index[i] for i in unit] for unit in moved], log)
    if fail:
        raise RegenError("static checks failed:\n    " + "\n    ".join(fail))
    log(f"  static checks passed; {len(moved)} tuples re-targeted")
    sha = hashlib.sha256(text.encode()).hexdigest()
    patch = unified_diff(ops, f"{KERNEL}.s\tsha256 {sha}", f"{KERNEL}.s\tregen_asm_patch.py")
    return "\n".join(edited) + "\n", patch


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def load_hsaco():
    """The package's hsaco.py by path (importing the package would import torch)."""
    spec = importlib.util.spec_from_file_location("_hk_mla_bwd_hsaco", PACKAGE_DIR / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="write nothing; fail unless patch and shas match")
    ap.add_argument("--cxx", default=None, help="compiler (default: $PRIMUS_TURBO_HK_CXX, PATH, $ROCM_PATH)")
    ap.add_argument("--out", default=None, help="patch path (default: the manifest's asm_patch under hip/)")
    ap.add_argument("--keep", default=None, help="save the compiler's .s, the patched .s and the hsaco here")
    ap.add_argument("--asm", default=None, help="regenerate from this .s instead of compiling (no hsaco)")
    a = ap.parse_args(argv)
    hs = load_hsaco()
    m = hs.load_manifest()
    k = m["kernels"][KERNEL]
    committed = hs.HIP_DIR / k.get("asm_patch", f"asm/{KERNEL}.s.patch")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        obj, cmd = tmp / f"{KERNEL}.hsaco", None
        if a.asm:
            s_path = Path(a.asm)
        else:
            cxx = hs.find_compiler(a.cxx)
            if cxx is None:
                print(f"amdclang++ not found (set {hs.CXX_ENV} or --cxx)", file=sys.stderr)
                return 2
            version = hs.compiler_version(cxx)
            if version != m["toolchain"]:
                print(f"WARNING: {version}, the manifest's is {m['toolchain']}", file=sys.stderr)
            cmd = hs.compile_command(cxx, KERNEL, obj)
            r = subprocess.run(cmd, cwd=hs.HIP_DIR, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{' '.join(cmd)} failed ({r.returncode}):\n{r.stderr[-4000:]}", file=sys.stderr)
                return 2
            s_files = list(tmp.glob(f"{KERNEL}-hip-*.s"))
            if len(s_files) != 1:
                print(f"expected one saved .s, found {s_files}", file=sys.stderr)
                return 2
            s_path = s_files[0]
        data = s_path.read_bytes()
        text, in_sha = data.decode(), hashlib.sha256(data).hexdigest()
        print(f"[regen_asm_patch] {KERNEL}: compiler .s sha256 {in_sha}")
        try:
            edited, patch = regenerate(text)
        except RegenError as e:
            print(f"[regen_asm_patch] {e}", file=sys.stderr)
            return 1
        if edited == text:
            print("[regen_asm_patch] nothing to patch: drop asm_patch / asm_input_sha256 from manifest.json")
            return 0
        if hs._apply_unified_diff(text, patch) != edited:  # the build's own applier must give the same text
            print("[regen_asm_patch] internal error: the patch does not give the edited .s", file=sys.stderr)
            return 1
        tmp_patch = tmp / "asm.patch"
        tmp_patch.write_text(patch)
        out_sha = None
        if cmd is not None:  # hsaco.py's build path: sha check, patch, the driver's own -cc1as + lld
            meta = dict(k, asm_input_sha256=in_sha, asm_patch=str(tmp_patch))
            try:
                hs._patch_and_reassemble(KERNEL, meta, cmd, tmp)
            except RuntimeError as e:
                print(f"[regen_asm_patch] {e}", file=sys.stderr)
                return 1
            out_sha = hashlib.sha256(obj.read_bytes()).hexdigest()
            print(f"[regen_asm_patch] {KERNEL}.hsaco sha256 {out_sha}")
        if a.keep:
            keep = Path(a.keep)
            keep.mkdir(parents=True, exist_ok=True)
            (keep / f"{KERNEL}.compiler.s").write_text(text)
            (keep / f"{KERNEL}.patched.s").write_text(edited)
            shutil.copyfile(tmp_patch, keep / committed.name)
            if out_sha:
                shutil.copyfile(obj, keep / obj.name)
    same_patch = committed.is_file() and committed.read_text() == patch
    current = in_sha == k.get("asm_input_sha256") and out_sha in (None, k["hsaco_sha256"])
    if a.check:
        print(f"[regen_asm_patch] patch {'matches' if same_patch else 'DIFFERS FROM'} {committed}")
        print(f"[regen_asm_patch] manifest shas {'match' if current else 'DO NOT MATCH'}")
        return 0 if same_patch and current else 1
    out = Path(a.out) if a.out else committed
    out.write_text(patch)
    print(f"[regen_asm_patch] wrote {out} ({patch.count(chr(10))} lines)")
    if current:
        print("[regen_asm_patch] manifest.json is up to date")
    else:
        print(f"[regen_asm_patch] set in manifest.json kernels.{KERNEL}:")
        if "asm_patch" not in k:
            print(f'  "asm_patch": "{committed.relative_to(hs.HIP_DIR)}",')
        print(f'  "asm_input_sha256": "{in_sha}",')
        if out_sha:
            print(f'  "hsaco_sha256": "{out_sha}",')
    return 0


if __name__ == "__main__":
    sys.exit(main())
