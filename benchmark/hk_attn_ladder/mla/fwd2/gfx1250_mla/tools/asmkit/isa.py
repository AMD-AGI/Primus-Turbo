###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.isa -- the gfx1250 assembly text model.

One ``Ins`` per line of a compiler ``.s`` (``-save-temps``) or an ``llvm-objdump -d`` listing: mnemonic, operands,
the VGPRs it reads and writes with their true numbers (0..1023, from the ``/*vN*/`` comments the compiler prints
under ``s_set_vgpr_msb``), and the instruction class the hazard and wait models use; ``sgpr_defs_uses`` gives the
scalar defs / uses. Every dependence the other modules see comes from these defs / uses, so ``coverage_errors``
checks that each register token of a line is in them."""

import re
from pathlib import Path

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
    """The operands of an instruction's operand text, split at top-level commas."""
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
    """The class of a mnemonic: XDL (WMMA), TRANS, CSMACC (core VALU), DPMACC (f64 ops, 64- / 16-bit compares),
    VNOP, DS, VMEM, FLAT, TDM, WAIT_ALU, WAIT_CNT, MSB, CLAUSE, BARRIER, DELAY, SALU or OTHER."""
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
    """The Ins of line number li. Labels get cls LABEL, directives and blank / comment lines DIR; a VOPD pair is
    one Ins with both halves' operands (separated by ``::``)."""
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
    """{field: N} of an ``s_wait_alu depctr_<field>(N) ...`` line."""
    return {k: int(v) for k, v in re.findall(r"depctr_(\w+)\((\d+)\)", x.text)}


def cnt_fields(x):
    """{counter: N} of an ``s_wait_<counter> N`` line (the combined forms give two counters)."""
    m = CNT_RE.search(strip_comment(x.text))
    if not m:
        return {}
    k, v = m.group(1), int(m.group(2), 0)
    if k in ("loadcnt_dscnt", "storecnt_dscnt"):
        return {k.split("_")[0]: v >> 8, "dscnt": v & 0xFF}
    return {k: v}


# --------------------------------------------------------------------------------------------------------------
# SGPR / scc / vcc / exec / m0 operands
# --------------------------------------------------------------------------------------------------------------
# Lookarounds, not \b: a \b after "]" does not match at the end of a token, which would drop every s[a:b] range.
SREG_RE = re.compile(
    r"(?<![\w$.])(s\[(\d+):(\d+)\]|s(\d+)|ttmp\[(\d+):(\d+)\]|ttmp(\d+)|vcc_lo|vcc_hi|vcc|exec_lo|exec_hi|exec|m0)"
    r"(?![\w\[])"
)
# SALU that do not write SCC (LLVM MIR: no implicit-def $scc; the compiler itself places them between s_cmp and the
# SCC reader). Everything else starting with s_ is treated as an SCC writer (conservative).
NO_SCC = (
    "s_mov",
    "s_movk",
    "s_bitset",
    "s_add_nc",
    "s_sub_nc",
    "s_mul_",
    "s_wait",
    "s_barrier",
    "s_nop",
    "s_set_vgpr_msb",
    "s_getreg",
    "s_setreg",
    "s_clause",
    "s_delay",
    "s_setprio",
    "s_cbranch",
    "s_branch",
)
READ_SCC = ("s_cselect", "s_cbranch_scc", "s_addc", "s_subb", "s_cmov")
NO_SDEF = (
    "s_cmp",
    "s_bitcmp",
    "s_wait",
    "s_barrier",
    "s_nop",
    "s_cbranch",
    "s_branch",
    "s_set_vgpr_msb",
    "s_clause",
    "s_delay",
    "s_setprio",
)
# wave32: vcc == vcc_lo, exec == exec_lo
_ALIAS = {"vcc_lo": "vcc", "exec_lo": "exec"}


def sregs(tok):
    """Scalar register names of an operand token: s<N> per register, ttmp<N>, vcc, vcc_hi, exec, exec_hi, m0."""
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
    """Scalar defs and uses of an instruction (s<N>, scc, vcc, exec, m0; conservative)."""
    if not x.mn or x.cls in ("LABEL", "DIR"):
        return [], []
    ops = x.ops
    mn = x.mn.split("::")[0]
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
        # VALU / memory: every scalar operand is a use. That includes v_div_scale_f32's vcc carry-out, which the
        # instruction writes. Harmless here: the generated code reads that vcc only in the v_div_fmas_f32 of the
        # same division, a VALU -> VALU dependence that needs no wait (LLVM emits the same sequence in expert mode
        # without one). Listing it as a def would change the generated code.
        for o in ops:
            if o == "::":
                continue
            u += sregs(first_tok(o))
        if mn.startswith(("v_cmp", "v_readlane", "v_readfirstlane")):
            d = u[:1] if u else ["vcc"]
        if "_dpp" in mn or mn.startswith(("v_cndmask", "v_addc", "v_subb")):
            u.append("vcc")
        if x.cls in ("CSMACC", "DPMACC", "TRANS", "XDL", "DS", "VMEM", "FLAT", "TDM"):
            u.append("exec")
    return d, u


# --------------------------------------------------------------------------------------------------------------
# Loading (.s or objdump) and structure
# --------------------------------------------------------------------------------------------------------------
OBJ_ADDR = re.compile(r"//\s*([0-9A-Fa-f]{6,16}):")
OBJ_TGT = re.compile(r"<[\w.$]+\+0x([0-9a-fA-F]+)>")


def load_any(path):
    """(lines, ins, fmt): a compiler .s (fmt 'cc') or an llvm-objdump -d listing (fmt 'objdump': the
    ``// addr: encoding`` tails are removed, branch targets become labels ``T<offset>``)."""
    raw = Path(path).read_text().rstrip("\n").split("\n")
    if any(l.strip().endswith("file format elf64-amdgpu") for l in raw[:3]):
        base = None
        addr_of, clean = [], []
        for l in raw:
            m = OBJ_ADDR.search(l)
            a = int(m.group(1), 16) if m else None
            addr_of.append(a)
            t = OBJ_TGT.search(l)
            code = re.sub(r"//\s*[0-9A-Fa-f]{6,16}:.*$", "", l).rstrip()
            if "/*" in code and "*/" not in code[code.rfind("/*") :]:
                code = code[: code.rfind("/*")].rstrip()
            if t and re.match(r"\s*s_c?branch", code):
                code = re.sub(
                    r"(s_c?branch\w*)\s+\S+", lambda mm, t=t: mm.group(1) + " T%x" % int(t.group(1), 16), code
                )
            clean.append(code)
        for l in raw:
            mm = re.match(r"^([0-9a-f]{16}) <(\w+)>:", l)
            if mm:
                base = int(mm.group(1), 16)
                break
        tg = set()
        for c in clean:
            mm = re.search(r"s_c?branch\w*\s+T([0-9a-f]+)", c)
            if mm:
                tg.add(int(mm.group(1), 16) + base)
        out = []
        for c, a in zip(clean, addr_of):
            if a is not None and a in tg:
                out.append("T%x:" % (a - base))
            out.append(c)
        return out, parse(out), "objdump"
    return raw, parse(raw), "cc"


def function_range(ins):
    """(first, last) line index of the kernel function: from its label (the first one not starting with ``.`` or
    ``T``, load_any's branch targets) to the line before ``.Lfunc_end*``; a listing without one: all of it."""
    lo = next((i for i, x in enumerate(ins) if x.label and not x.label.startswith((".", "T"))), 0)
    hi = next((i for i in range(lo, len(ins)) if ins[i].text.strip().startswith(".Lfunc_end")), len(ins)) - 1
    return lo, hi


# --------------------------------------------------------------------------------------------------------------
# Parse coverage: every register token of the code part is in the parsed operands
# --------------------------------------------------------------------------------------------------------------
_VTOK = re.compile(r"(?<![\w$.])v\[\d+:\d+\]|(?<![\w$.])v\d+(?![\w\[])")


def coverage_errors(x):
    """Register tokens of x's code that its VGPR defs / uses or scalar defs / uses miss ([] = complete)."""
    if not x.mn or x.cls in ("LABEL", "DIR"):
        return []
    code = strip_comment(x.text)
    err = []
    vparsed = set(x.defs) | set(x.uses)
    for o in x.ops:
        if o == "::":
            continue
        tok = first_tok(o)
        for m in _VTOK.finditer(re.sub(r"/\*.*?\*/", "", tok)):
            if not regs_of(tok):
                err.append(f"VGPR token {m.group(0)} not parsed")
            elif not set(regs_of(tok)) <= vparsed and x.cls not in ("VNOP",):
                err.append(f"VGPR token {tok} not in defs/uses")
    sd, su = sgpr_defs_uses(x)
    sparsed = set(sd) | set(su)
    for o in x.ops:
        if o == "::":
            continue
        for r in sregs(first_tok(o)):
            if r not in sparsed:
                err.append(f"scalar {r} not in defs/uses")
    # a scalar-looking token that SREG_RE missed (e.g. a new syntax)
    for m in re.finditer(
        r"(?<![\w$.])s\[[^\]]*\]|(?<![\w$.])s\d+\b",
        re.sub(r"/\*.*?\*/", "", code.split(None, 1)[1] if " " in code.strip() else ""),
    ):
        if not sregs(m.group(0)):
            err.append(f"scalar token {m.group(0)} not parsed")
    return err
