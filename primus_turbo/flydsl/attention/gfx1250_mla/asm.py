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

"""The gfx1250 assembly model behind the generated code of the MLA kernels (fwd_sched, bwd_dkdv_sched; ported from
the HipKittens generators). Standard library only.

A generator hands over the real instructions of a straight-line block in their final order; ``emit_block`` adds the
lines that order implies in expert scheduling mode, as LLVM would:
  * ``s_set_vgpr_msb`` wherever the VGPR operands need another mode (VGPR block = register // 256; ``msb_field``
    gives the mode field of an operand); the mode is 0 at the block start, before a branch and at the block end;
  * ``s_wait_alu depctr_va_vdst(N)`` before a memory instruction that reads or writes a register an in-flight VALU
    writes, or writes one it reads. WMMA, TRANS and core VALU complete in order within their class, so N may be the
    number of same-class instructions issued since; LLVM's rule (``va_need``) takes N = 0 once more than one class
    is pending, except for a write that only an in-flight WMMA still reads;
  * ``s_wait_alu depctr_vm_vsrc(N)`` before a write of a register an in-flight memory instruction still reads:
    LDS, FLAT and buffer instructions read their VGPR sources in order within their class, so N is the number of
    instructions of that class issued since (``vm_need``; a TDM op reads no VGPR and counts in none of them).
    LLVM takes N = 0 while more than one class is pending; the class's own N holds there too (an instruction that
    has not read its sources has its N younger classmates behind it, so more than N are outstanding);
    ``s_wait_dscnt`` / ``loadcnt`` / ``storecnt`` before an access to a register an in-flight load writes, and
    ``s_wait_dscnt 0`` before ``s_barrier_signal`` while LDS operations are in flight;
  * ``s_wait_xcnt 0`` before a write of a scalar register an in-flight VMEM / TDM instruction still reads;
  * ``v_nop`` for the wait states, in VALU-class instructions: WMMA D -> WMMA A / B: 5; WMMA -> a VALU that reads
    its D or writes its D / A / B: 4; TRANS -> an immediately following VALU that reads its result or writes its
    source: 1.
Every wait and v_nop holds for everything that may reach the block (a ``Flow``). An ``s_wait_alu`` among the
instructions (a wait the generator places itself, e.g. va_vdst(N) behind N core VALUs so that the ones before them
count as written) is applied to that state like a derived one.

What the model covers, and what it leaves out (the reference is LLVM's AMDGPU backend for gfx1250: SIInsertWaitcnts
for the waits, GCNHazardRecognizer for the wait states):
  * VGPRs only. Expert scheduling mode hands the compiler the hazards between VALU and memory instructions on
    VGPRs. The scalar side stays with the hardware in both directions (a VALU that reads an SGPR, vcc or scc a
    SALU just wrote; a SALU or branch that reads one a VALU just wrote, as s_cmp behind v_readlane): LLVM places
    no va_sdst / sa_sdst / va_vcc wait for gfx1250, and neither does the model. Two rules of LLVM's that the model
    does not implement hold by the generators' construction, and the codegen test checks them on the regions: a
    buffer, global or TDM instruction that reads an SGPR a VALU wrote wants 5 instructions between the two (such
    SGPRs reach memory instructions through SALU copies, or long after the write), and a TDM op is issued with
    at most 10 others in flight (s_wait_tensorcnt);
  * the classes the generators emit: every ``v_`` instruction of theirs is a WMMA (XDL), a transcendental (TRANS)
    or a core VALU (CSMACC), the packed f32 operations and the bf16 conversions included; none is DPMACC (f64),
    which ``classify`` only tells apart by name;
  * the v_nop counts are those of ``v_wmma_f32_16x16x32_bf16`` on gfx1250 (8 cycles in the XDL pipe; gfx1251 runs
    it in 16 and needs 9 / 8), for a WMMA whose accumulator is +0 or its own destination (``wmma_regs`` checks both);
  * a block entered from code the model does not follow (``State(unknown=True)``, an empty VALU history) waits for
    VALU writes, memory source reads and LDS loads on the VGPRs it touches, but it knows neither a buffer / global
    load still in flight into one of them (the generators place ``s_wait_loadcnt`` where the kernel leaves one) nor
    the VALU-class instructions issued just before it: the v_nop rules hold across such an entry only when a
    va_vdst(0) wait stands before the block's first VALU-class instruction, or no WMMA / TRANS can be in flight
    there (test_attention_flydsl_gfx1250_mla_codegen pins which regions rely on which).

The generators' common pieces are here too: ``vr`` (operand text of VGPRs), ``add_order`` and ``dag_ranks`` (the
dependence graph their list schedulers rank nodes by)."""

import re
from collections import defaultdict

# ----------------------------------------------------------------------------------------------------------------
# instruction text: one Ins per line, VGPRs with their true numbers (the /*vN*/ comments under s_set_vgpr_msb)
# ----------------------------------------------------------------------------------------------------------------
TRANS_OPS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_", "v_tanh_")
VALU = ("XDL", "TRANS", "CSMACC", "DPMACC")
MEM = ("DS", "VMEM", "FLAT", "TDM")
COUNTED = ("XDL", "TRANS", "CSMACC")  # VALU classes whose in-class order bounds va_vdst
REG_RE = re.compile(r"^v\[(\d+):(\d+)\]$|^v(\d+)$")
CNT_RE = re.compile(
    r"s_wait_(dscnt|loadcnt|storecnt|loadcnt_dscnt|storecnt_dscnt|tensorcnt|xcnt|kmcnt|samplecnt|bvhcnt)"
    r"\s+(0x[0-9a-fA-F]+|\d+)"
)


def regs_of(tok):
    """VGPR numbers of an operand token (``v[26:33] /*v[794:801]*/``: the comment has the true ones)."""
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
    """Operands of an instruction's operand text, split at top-level commas."""
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
    """The register part of an operand (without a trailing modifier)."""
    m = re.match(r"^(\S+(\s*/\*[^*]*\*/)?)", op)
    return m.group(1) if m else op


def classify(mn):
    """XDL (WMMA), TRANS, CSMACC (core VALU), DPMACC (f64, 64- / 16-bit compares), VNOP, DS, VMEM, FLAT, TDM,
    WAIT_ALU, WAIT_CNT, MSB, CLAUSE, BARRIER, DELAY, SALU or OTHER."""
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
    for prefix, c in (
        ("ds_", "DS"),
        (("buffer_", "tbuffer_"), "VMEM"),
        (("global_", "flat_", "scratch_"), "FLAT"),
        ("tensor_", "TDM"),
        ("s_wait_alu", "WAIT_ALU"),
        ("s_wait_", "WAIT_CNT"),
    ):
        if mn.startswith(prefix):
            return c
    if mn == "s_set_vgpr_msb":
        return "MSB"
    if mn == "s_clause":
        return "CLAUSE"
    if mn.startswith("s_barrier"):
        return "BARRIER"
    if mn.startswith("s_delay_alu"):
        return "DELAY"
    return "SALU" if mn.startswith("s_") else "OTHER"


class Ins:
    """One line: mnemonic, operands, VGPRs read (uses) and written (defs), class, label."""

    __slots__ = ("li", "text", "mn", "ops", "defs", "uses", "cls", "label")


def parse_line(li, line):
    """The Ins of line li. Labels get cls LABEL, directives and blank lines DIR; a VOPD pair is one Ins with both
    halves' operands. Defs / uses are exact for the generated instructions (full-register writes); of the compiler's
    lines only their union is used."""
    x = Ins()
    x.li, x.text, x.label, x.mn, x.ops, x.defs, x.uses = li, line, None, "", [], [], []
    s = strip_comment(line).strip()
    if s.endswith(":") and " " not in s:
        x.cls, x.label = "LABEL", s[:-1]
        return x
    if not s or s.startswith("."):
        x.cls = "DIR"
        return x
    if "::" in s:  # VOPD dual issue
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
    if x.cls in VALU and not x.mn.startswith(("v_cmp", "v_readlane", "v_readfirstlane")):
        d = list(regs[0]) if regs else []
        u = [r for rr in regs[1:] for r in rr]
    elif x.cls in ("DS", "VMEM", "FLAT") and (
        x.mn.startswith(("ds_load", "ds_read")) or (x.cls != "DS" and "_load" in x.mn and "lds" not in x.mn)
    ):
        d = list(regs[0])
        u = [r for rr in regs[1:] for r in rr]
    else:
        u = [r for rr in regs for r in rr]
    x.defs, x.uses = d, u
    return x


def cnt_fields(x):
    """{counter: N} of an ``s_wait_<counter> N`` line (the combined forms give two counters)."""
    m = CNT_RE.search(strip_comment(x.text))
    if not m:
        return {}
    k, v = m.group(1), int(m.group(2), 0)
    if k in ("loadcnt_dscnt", "storecnt_dscnt"):
        return {k.split("_")[0]: v >> 8, "dscnt": v & 0xFF}
    return {k: v}


# lookarounds, not \b: a \b after "]" does not match at the end of a token
SREG_RE = re.compile(
    r"(?<![\w$.])(s\[(\d+):(\d+)\]|s(\d+)|ttmp\[(\d+):(\d+)\]|ttmp(\d+)|vcc_lo|vcc_hi|vcc|exec_lo|exec_hi|exec|m0)"
    r"(?![\w\[])"
)
# SALU that do not write SCC (LLVM MIR: no implicit-def $scc); every other s_ instruction is taken to write it
NO_SCC = tuple(
    "s_mov s_movk s_bitset s_add_nc s_sub_nc s_mul_ s_wait s_barrier s_nop s_set_vgpr_msb s_getreg s_setreg s_clause"
    " s_delay s_setprio s_cbranch s_branch".split()
)
READ_SCC = ("s_cselect", "s_cbranch_scc", "s_addc", "s_subb", "s_cmov")
NO_SDEF = tuple(
    "s_cmp s_bitcmp s_wait s_barrier s_nop s_cbranch s_branch s_set_vgpr_msb s_clause s_delay s_setprio".split()
)
_ALIAS = {"vcc_lo": "vcc", "exec_lo": "exec"}  # wave32


def sregs(tok):
    """Scalar registers of an operand token: s<N> per register, ttmp<N>, vcc, vcc_hi, exec, exec_hi, m0."""
    out = []
    for m in SREG_RE.finditer(tok):
        if m.group(2) is not None:
            out += [f"s{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
        elif m.group(4) is not None:
            out.append(f"s{m.group(4)}")
        elif m.group(5) is not None:
            out += [f"ttmp{i}" for i in range(int(m.group(5)), int(m.group(6)) + 1)]
        elif m.group(7) is not None:
            out.append(f"ttmp{m.group(7)}")
        else:
            out.append(_ALIAS.get(m.group(1), m.group(1)))
    return out


def sgpr_defs_uses(x):
    """Scalar defs and uses of an instruction (s<N>, scc, vcc, exec, m0; conservative). A VALU's scalar operands are
    uses; the destination of a compare / v_readlane is its def too (and stays among the uses), v_div_scale's
    carry-out (its operand 1: vcc or null) is a def only, and v_div_fmas reads vcc implicitly."""
    if not x.mn or x.cls in ("LABEL", "DIR"):
        return [], []
    ops, mn = x.ops, x.mn.split("::")[0]
    d, u = [], []
    if x.cls in ("SALU", "BARRIER") or mn.startswith("s_"):
        regs = [sregs(first_tok(o)) for o in ops]
        if regs and not mn.startswith(NO_SDEF):
            d = list(regs[0])
            u = [r for rr in regs[1:] for r in rr]
            if mn.startswith(("s_bitset", "s_addk", "s_mulk", "s_cmovk", "s_cmov")):
                u += d
        else:
            u = [r for rr in regs for r in rr]
        if not mn.startswith(NO_SCC):
            d.append("scc")
        if mn.startswith(READ_SCC) or mn.startswith(("s_addk", "s_cmovk")):
            u.append("scc")
        if mn.startswith("s_cbranch_vcc"):
            u.append("vcc")
        if mn.startswith("s_cbranch_exec"):
            u.append("exec")
    else:
        carry_out = mn.startswith("v_div_scale")
        for p, o in enumerate(ops):
            if carry_out and p == 1:
                d += sregs(first_tok(o))
            elif o != "::":
                u += sregs(first_tok(o))
        if mn.startswith(("v_cmp", "v_readlane", "v_readfirstlane")):
            d = u[:1] if u else ["vcc"]
        if "_dpp" in mn or mn.startswith(("v_cndmask", "v_addc", "v_subb", "v_div_fmas")):
            u.append("vcc")
        if x.cls in ("CSMACC", "DPMACC", "TRANS", "XDL", "DS", "VMEM", "FLAT", "TDM"):
            u.append("exec")
    return d, u


# ----------------------------------------------------------------------------------------------------------------
# expert scheduling mode: what is in flight at a program point (one path)
# ----------------------------------------------------------------------------------------------------------------
class Rec:
    """An in-flight instruction. VALU: cls XDL / TRANS / CSMACC / UNK, seq = its number in its class. Memory op:
    cls LDS / VMEM / FLAT (the class its VGPR source reads are ordered in), seq = its number there, ctr / cseq =
    its wait counter and its number there."""

    __slots__ = ("cls", "reads", "writes", "counted", "done", "seq", "ctr", "cseq")

    def __init__(self, cls, reads, writes, counted, seq, ctr=None, cseq=None):
        self.cls, self.reads, self.writes, self.counted = cls, set(reads), set(writes), counted
        self.done, self.seq, self.ctr, self.cseq = False, seq, ctr, cseq

    def copy(self):
        return Rec(self.cls, self.reads, self.writes, self.counted, self.seq, self.ctr, self.cseq)


class State:
    """In-flight VALU, memory source reads and load writes. unknown: entered from code the model does not follow, where
    a VALU write, a memory instruction's source read or an LDS load may be in flight on every VGPR (not a buffer or
    global load, and no VALU history for the v_nop rules: the module docstring)."""

    def __init__(self, unknown=False):
        self.valu, self.msrc, self.loads = [], [], []
        self.vcnt, self.mcnt, self.ccnt = defaultdict(int), defaultdict(int), defaultdict(int)
        self.unknown_v = self.unknown_m = self.unknown_ds = unknown

    def copy(self):
        s = State()
        s.valu, s.msrc, s.loads = (
            [r.copy() for r in v if not r.done] for v in (self.valu, self.msrc, self.loads)
        )
        s.vcnt, s.mcnt, s.ccnt = (defaultdict(int, c) for c in (self.vcnt, self.mcnt, self.ccnt))
        s.unknown_v, s.unknown_m, s.unknown_ds = self.unknown_v, self.unknown_m, self.unknown_ds
        return s

    def rel_sig(self):
        """Signature with sequence numbers relative to the current counters (comparable across loop trips)."""

        def recs(v, cnt):
            out = []
            for r in v:
                if not r.done:
                    key = (
                        r.cls,
                        tuple(sorted(r.reads)),
                        tuple(sorted(r.writes)),
                        r.counted,
                        cnt[r.cls] - r.seq,
                    )
                    out.append(key + (r.ctr, (self.ccnt[r.ctr] - r.cseq) if r.ctr else None))
            return sorted(out)

        flags = (self.unknown_v, self.unknown_m, self.unknown_ds)
        return (recs(self.valu, self.vcnt), recs(self.msrc, self.mcnt), recs(self.loads, self.mcnt), *flags)


def join_states(states):
    """A conservative merge (any path): every pending record at its most recent position over the states."""
    base = 1 << 20  # the merged counters
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
                d, cd = cnt[r.cls] - r.seq, (s.ccnt[r.ctr] - r.cseq) if r.ctr else None
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
    """Clamp every pending record's distance at 64 (no wait field exceeds 63) and keep one of the records that are
    then equal (a record no wait of the code retires, e.g. a VALU without a VGPR destination in a loop without a
    va_vdst(0), would pile up once per trip): a finite state space."""
    for name, cnt in (("valu", st.vcnt), ("msrc", st.mcnt), ("loads", st.mcnt)):
        seen, keep = set(), []
        for r in getattr(st, name):
            if r.done:
                continue
            r.seq = max(r.seq, cnt[r.cls] - 64)
            if r.ctr:
                r.cseq = max(r.cseq, st.ccnt[r.ctr] - 64)
            key = (r.cls, frozenset(r.reads), frozenset(r.writes), r.counted, r.seq, r.ctr, r.cseq)
            if key not in seen:
                seen.add(key)
                keep.append(r)
        setattr(st, name, keep)
    return st


def issue(st, x):
    """Add instruction x to st (VALU and memory instructions)."""
    if x.cls in VALU:
        if x.defs:  # a newer write supersedes older in-flight writes (as in LLVM)
            d = set(x.defs)
            for r in st.valu:
                if not r.done and r.writes & d:
                    r.writes = r.writes - d
        cls = x.cls if x.cls in COUNTED else "UNK"
        counted = cls == "XDL" or (cls != "UNK" and bool(x.defs))  # a VALU without a VGPR dst may not count
        st.vcnt[cls] += counted
        st.valu.append(Rec(cls, x.uses, x.defs, counted, st.vcnt[cls]))
    elif x.cls in MEM:
        # the class whose order bounds vm_vsrc; a TDM op reads no VGPR and is in a class of its own: counted among
        # the buffer instructions it would loosen the waits for their sources (LLVM does not count it either)
        mc = {"DS": "LDS", "FLAT": "FLAT", "VMEM": "VMEM", "TDM": "TDM"}[x.cls]
        st.mcnt[mc] += 1
        if x.cls == "DS":
            k = "dscnt"
        elif x.cls == "TDM":
            k = "tensorcnt"
        else:
            k = "storecnt" if ("_store" in x.mn or "_atomic" in x.mn) else "loadcnt"
        st.ccnt[k] += 1
        if x.uses:
            st.msrc.append(Rec(mc, x.uses, [], True, st.mcnt[mc], k, st.ccnt[k]))
        if x.defs:
            st.loads.append(Rec(mc, [], x.defs, True, st.mcnt[mc], k, st.ccnt[k]))


def va_need(st, x):
    """The va_vdst N memory instruction x needs (None: no hazard), LLVM's rule: the per-class bound while one class
    is pending, else 0, except for a write-after-read against an in-flight WMMA (the per-class bound)."""
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
        if not (raw_waw or r.reads & ws):
            continue
        counted = r.cls in COUNTED and r.counted
        n = st.vcnt[r.cls] - r.seq if counted else 0
        if multi and (raw_waw or r.cls != "XDL" or not counted):
            n = 0
        need = n if need is None else min(need, n)
    return need


def vm_need(st, x):
    """The vm_vsrc N after which no in-flight memory op still reads a register x writes."""
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


def alu_fields(x):
    """{va_vdst / vm_vsrc: N} of an ``s_wait_alu`` line."""
    return {k: int(v) for k, v in re.findall(r"depctr_(va_vdst|vm_vsrc)\((\d+)\)", strip_comment(x.text))}


def apply_cnt(st, fields):
    for k, v in fields.items():
        for r in st.loads + st.msrc:
            if not r.done and r.ctr == k and st.ccnt[k] - r.cseq >= v:
                r.done = True
    if fields.get("dscnt") == 0:
        st.unknown_ds = False


# ----------------------------------------------------------------------------------------------------------------
# emission of a straight-line block
# ----------------------------------------------------------------------------------------------------------------
FIELDS = ("src0", "src1", "src2", "dst")  # s_set_vgpr_msb mode fields, 2 bits each from bit 0
VALU_CLS = ("XDL", "CSMACC", "DPMACC", "TRANS", "VNOP")  # what the v_nop rules count
VALU_HISTORY = 12  # VALU-class instructions kept per path (the v_nop rules look back at most 9)
VA_VDST_MAX, VM_VSRC_MAX, CNT_MAX = 14, 6, 63  # largest emitted wait values
MAX_STATES, MAX_HISTORIES = 8, 64  # more distinct states at one point are merged; more histories are an error
_NOP = parse_line(-1, "\tv_nop")


def kind_of(x):
    """WMMA, VALU, TRANS, DS_LD, DS_ST, VM_LD, VM_ST, TDM, WAIT_CNT, BAR_S, BAR_W, BR, SALU or OTHER."""
    c, mn = x.cls, x.mn
    if c == "XDL":
        return "WMMA"
    if c in ("CSMACC", "DPMACC"):
        return "VALU"
    if c in ("TRANS", "TDM", "WAIT_CNT"):
        return c
    if c == "DS":
        return "DS_LD" if mn.startswith(("ds_load", "ds_read")) else "DS_ST"
    if c in ("VMEM", "FLAT"):
        return "VM_ST" if ("_store" in mn or "_atomic" in mn) else "VM_LD"
    if c == "BARRIER":
        return "BAR_S" if "signal" in mn else "BAR_W"
    if mn.startswith(("s_cbranch", "s_branch")):
        return "BR"
    return "SALU" if c == "SALU" else "OTHER"


class Item:
    """A real instruction of a block: text, parsed Ins (x), kind, VGPR and scalar defs / uses."""

    __slots__ = ("text", "x", "kind", "vdefs", "vuses", "sdefs", "suses")


def make_item(text):
    x = parse_line(-1, text)
    n = Item()
    n.text, n.x, n.kind = text, x, kind_of(x)
    n.vdefs, n.vuses = list(x.defs), list(x.uses)
    n.sdefs, n.suses = sgpr_defs_uses(x)
    return n


def msb_field(mn, p):
    """The s_set_vgpr_msb field of VGPR operand p (LLVM's VGPR lowering tables): VALU / WMMA dst, src0, src1, src2
    by position (v_div_scale's operand 1 is its scalar carry-out); DS / buffer loads and buffer stores dst, src0;
    DS stores src0, src1."""
    if mn.startswith("ds_store"):
        return ("src0", "src1")[p]
    if mn.startswith(("ds_load", "buffer_")):
        return ("dst", "src0")[p]
    if mn.startswith("v_div_scale"):
        p -= p > 0
    return ("dst", "src0", "src1", "src2")[p]


def msb_line(new, old):
    """s_set_vgpr_msb from mode old to mode new: bits 0-7 the new mode, bits 8-15 the old one (as LLVM writes it)."""
    imm = sum(new[f] << (2 * k) for k, f in enumerate(FIELDS)) | (
        sum(old[f] << (2 * k) for k, f in enumerate(FIELDS)) << 8
    )
    return (
        f"\ts_set_vgpr_msb 0x{imm:x}                   ;  msbs: dst={new['dst']} src0={new['src0']} "
        f"src1={new['src1']} src2={new['src2']}"
    )


WMMA = "v_wmma_f32_16x16x32_bf16"  # the XDL instruction the wait states of nops_needed are for


def wmma_regs(x):
    """(D, A | B) of a WMMA: every register it reads or writes, as its accumulator C is +0 or D itself (LLVM's rule
    for a VALU that writes a register of an in-flight WMMA looks at C too)."""
    o = [regs_of(first_tok(t)) for t in x.ops]
    if x.mn != WMMA or len(o) != 4 or (o[3] and o[3] != o[0]):
        raise ValueError(f"the WMMA hazard rules cover {WMMA} with C = 0 or C = D only: {x.text.strip()}")
    return set(o[0]), set(o[1]) | set(o[2])


def nops_needed(prev, x):
    """The v_nops x needs after the VALU-class instructions prev (newest first): 5 / 4 behind a WMMA (wmma_regs;
    gfx1250's counts for an 8-cycle WMMA), 1 behind a transcendental."""
    need = 0
    if x.cls == "XDL":
        _, ab = wmma_regs(x)
        for k, y in enumerate(prev[:9]):
            if y.cls == "XDL":
                if wmma_regs(y)[0] & ab:
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
    for v in valus:
        v.insert(0, x)
        del v[VALU_HISTORY:]


class XState:
    """The scalar registers in-flight VMEM / FLAT / TDM instructions still read (XCNT); unknown: any may be."""

    def __init__(self, unknown, regs=()):
        self.unknown, self.regs = unknown, set(regs)

    def copy(self):
        return XState(self.unknown, self.regs)

    def sig(self):
        return (self.unknown, tuple(sorted(self.regs)))


class Flow:
    """What can reach a program point: wait states, XCNT states and VALU histories (newest first), kept as three
    independent lists: a sound over-approximation of the per-path combinations."""

    __slots__ = ("sts", "xs", "valus")

    def __init__(self, sts, xs, valus):
        self.sts, self.xs, self.valus = sts, xs, valus

    def copy(self):
        return Flow([s.copy() for s in self.sts], [x.copy() for x in self.xs], [list(v) for v in self.valus])

    def sig(self):
        histories = sorted(tuple(y.text for y in v) for v in self.valus)
        return (sorted(repr(s.rel_sig()) for s in self.sts), sorted(x.sig() for x in self.xs), histories)

    @staticmethod
    def join(flows):
        """The flow at a merge point: the distinct states, XCNT states and histories."""
        sts, xs, vs = [], [], []
        seen = (set(), set(), set())
        for f in flows:
            for s in f.sts:
                k = repr(s.rel_sig())
                if k not in seen[0]:
                    seen[0].add(k)
                    sts.append(s.copy())
            for x in f.xs:
                if x.sig() not in seen[1]:
                    seen[1].add(x.sig())
                    xs.append(x.copy())
            for v in f.valus:
                k = tuple(y.text for y in v)
                if k not in seen[2]:
                    seen[2].add(k)
                    vs.append(list(v))
        if len(sts) > MAX_STATES:
            sts = [join_states(sts)]
        if len(xs) > MAX_STATES:
            xs = [XState(any(x.unknown for x in xs), set().union(*(x.regs for x in xs)))]
        if len(vs) > MAX_HISTORIES:
            raise RuntimeError(f"{len(vs)} distinct VALU histories reach one block")
        return Flow(sts, xs, vs)


def wait_needs(sts, xs, x, sx, release):
    """{field: n}: the waits x needs (the smallest over the states), xcnt if any XCNT state asks. sx: the scalar
    registers x writes; release: x is an s_barrier_signal."""
    need = {}

    def put(f, n):
        if n is not None:
            need[f] = n if f not in need else min(need[f], n)

    for st in sts:
        if x.cls in MEM:
            put("va_vdst", va_need(st, x))
        if x.cls in VALU or x.cls in MEM:
            put("vm_vsrc", vm_need(st, x))
        for c, n in load_need(st, x):
            put(c, n)
        if release and (st.unknown_ds or any(not r.done and r.ctr == "dscnt" for r in st.loads + st.msrc)):
            put("dscnt", 0)
    if sx and any(s.unknown or set(sx) & s.regs for s in xs):
        put("xcnt", 0)
    return need


def emit_block(items, flow):
    """(lines, the flow at the end) of a block: the items in their order with the derived lines added, for
    everything flow (not modified) says may reach the block."""
    fl = flow.copy()
    sts, xs, valus = fl.sts, fl.xs, fl.valus
    out = []
    mode = {f: 0 for f in FIELDS}
    for n in items:
        x = n.x
        req = {}
        if n.kind != "BR":
            for p, o in enumerate(x.ops):
                rr = regs_of(first_tok(o)) if o != "::" else []
                if rr:
                    f = msb_field(x.mn, p)
                    if req.get(f, rr[0] // 256) != rr[0] // 256:
                        raise RuntimeError(f"conflicting MSB needs in {x.text.strip()}")
                    req[f] = rr[0] // 256
        else:
            req = {f: 0 for f in FIELDS}
        if any(mode[f] != v for f, v in req.items()):
            new = dict(mode, **req)
            out.append(msb_line(new, mode))
            mode = new
        need = wait_needs(sts, xs, x, [r for r in n.sdefs if r != "scc"], n.kind == "BAR_S")
        if need:
            caps = {"va_vdst": VA_VDST_MAX, "vm_vsrc": VM_VSRC_MAX}
            alu = [f"depctr_{k}({min(need[k], m)})" for k, m in caps.items() if k in need]
            if alu:
                out.append("\ts_wait_alu " + " ".join(alu))
            cnts = ("loadcnt", "storecnt", "dscnt", "xcnt")
            out += [f"\ts_wait_{c} 0x{min(need[c], CNT_MAX):x}" for c in cnts if c in need]
            for st in sts:
                if "va_vdst" in need:
                    apply_va(st, min(need["va_vdst"], VA_VDST_MAX))
                if "vm_vsrc" in need:
                    apply_vm(st, min(need["vm_vsrc"], VM_VSRC_MAX))
                cnt = {k: v for k, v in need.items() if k in ("loadcnt", "storecnt", "dscnt")}
                if cnt:
                    apply_cnt(st, cnt)
            if "xcnt" in need:
                for s in xs:
                    s.unknown, s.regs = False, set()
        if x.cls in VALU_CLS:
            nn = max(nops_needed(v, x) for v in valus)
            out += ["\tv_nop"] * nn
            for _ in range(nn):
                _push(valus, _NOP)
        out.append(n.text)
        for st in sts:
            if x.cls == "WAIT_CNT":
                apply_cnt(st, cnt_fields(x))
            elif x.cls == "WAIT_ALU":  # a wait the generator placed itself
                for k, v in alu_fields(x).items():
                    (apply_va if k == "va_vdst" else apply_vm)(st, v)
            else:
                issue(st, x)
        if x.cls in ("VMEM", "FLAT", "TDM"):
            for s in xs:
                s.regs |= set(n.suses)
        if x.cls in VALU_CLS:
            _push(valus, x)
    if any(mode[f] for f in FIELDS):  # the next block starts in mode 0
        out.append(msb_line({f: 0 for f in FIELDS}, mode))
    for st in sts:
        cap(st)
    return out, fl


def emit(texts, flow):
    """emit_block on instruction texts (without the leading tab)."""
    return emit_block([make_item("\t" + t) for t in texts], flow)


# ----------------------------------------------------------------------------------------------------------------
# what the generators share
# ----------------------------------------------------------------------------------------------------------------
def vr(r, n=1):
    """Operand text of the n VGPRs from true register r: the encoded low 8 bits, the true numbers in a comment
    (the compiler's spelling under s_set_vgpr_msb)."""
    if r // 256 != (r + n - 1) // 256:
        raise ValueError(f"tuple v[{r}:{r + n - 1}] crosses a 256-register block")
    lo = r % 256
    if n == 1:
        return f"v{lo}" if r < 256 else f"v{lo} /*v{r}*/"
    return f"v[{lo}:{lo + n - 1}]" if r < 256 else f"v[{lo}:{lo + n - 1}] /*v[{r}:{r + n - 1}]*/"


def add_order(nodes, a, b, lat=1):
    """An order edge a -> b (indices, a before b in program order) of at least lat cycles; node.preds =
    [(pred, latency)]."""
    if not a < b:
        raise ValueError("order edges must point forward in program order")
    d = dict(nodes[b].preds)
    d[a] = max(d.get(a, 0), lat)
    nodes[b].preds = sorted(d.items())


def dag_ranks(nodes, wseq):
    """(successors, deadline, critical path) of a dependence graph (node.preds, node.kind) for a list scheduler around
    the in-order WMMA stream wseq (node indices): succ[i] = [(successor, latency)]; deadline[i] = the position in
    wseq of the first WMMA that needs node i, transitively (10**6: none does); cp[i] = the longest path from node i
    on, a WMMA counting 8 cycles and any other node 1."""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p, lat in nd.preds:
            succ[p].append((i, lat))
    wpos = {i: k for k, i in enumerate(wseq)}
    deadline, cp = [10**6] * n, [0] * n
    for i in range(n - 1, -1, -1):
        for s, _ in succ[i]:
            deadline[i] = min(deadline[i], wpos[s] if s in wpos else deadline[s])
        cp[i] = (8 if nodes[i].kind == "WMMA" else 1) + max((lat + cp[s] for s, lat in succ[i]), default=0)
    return succ, deadline, cp
