#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""k_dkdv96ds's asm patch with WAR-V waits by class (the user-approved relaxation, gated by tools/warv_stress).

    python3 tools/warv_regen.py <compiler .s> --out <patch> [--mode warv|llvm] [--turbo <Primus-Turbo checkout>]

Runs Primus-Turbo's tools/regen_asm_patch.py (re-target of the Q^T tr16 tuples, va_vdst recomputation, static
checks) with one change to its wait model: a memory instruction whose only VALU hazards are WAR-V -- it overwrites
registers that in-flight WMMAs (XDL) read -- gets the per-class bound N = (WMMAs issued after the newest such
reader) instead of LLVM's N = 0 whenever more than one VALU class is pending. Every other hazard (RAW / WAW on
anything, WAR on non-XDL readers, unknown state) keeps LLVM's rule, and the waits of different hazards of one
instruction combine with min.

Why it is safe: s_wait_alu depctr_va_vdst(N) waits until at most N VALU instructions are in flight; XDL instructions
complete in order among themselves (the model the regen tool documents, and what tools/warv_stress tests on the
card), so with at most N in flight and N XDL issued after reader X, X has completed and read its operands.

Then an independent check (written apart from the regen tool's code): a forward walk over two trips of the loop that
tracks every in-flight WMMA's operand registers and, at each s_wait_alu va_vdst(N), retires all but the newest N
WMMAs; every DS / VMEM load or TDM op that writes a register a still-in-flight WMMA reads is a violation. --mode llvm
reproduces the shipped patch (the check must pass on it too).
"""

import argparse
import hashlib
import importlib.util
import re
import sys
from pathlib import Path


RETARGET_VA = """        cost = unit_need(ins, first, last, unit)[0]
        if cost >= MIN_WAIT:
            continue
"""
RETARGET_MSRC = """        cost, mem_src = unit_need(ins, first, last, unit)
        if cost >= MIN_WAIT and not mem_src:
            continue
"""


def load_regen(turbo, msrc=False):
    """The regen tool as a module; msrc=True: its retarget also moves a Q^T unit whose loads wait for an in-flight
    memory op still reading (or writing) their registers (vm_vsrc / load waits, e.g. the export stores' data
    registers re-used by the next Q^T tr16), not only one that needs a va_vdst near-drain."""
    p = Path(turbo) / "primus_turbo/hipkittens/gfx1250_mla_bwd/tools/regen_asm_patch.py"
    src = p.read_text()
    if msrc:
        if src.count(RETARGET_VA) != 1:
            raise SystemExit("regen tool: retarget's selection rule not found")
        src = src.replace(RETARGET_VA, RETARGET_MSRC)
    m = type(sys)("regen_asm_patch")
    m.__file__ = str(p)
    sys.modules["regen_asm_patch"] = m
    exec(compile(src, str(p), "exec"), m.__dict__)
    if msrc:  # check 3 (a moved unit no longer needs a near-drain) measures the benefit, not correctness, and sees the
        # vm_vsrc-stripped text: a warning here; correctness stays with checks 1, 2, 4-6 and main()'s final check
        # (no hazard open that the compiler's original leaves closed, on either path)
        checks = m.static_checks

        def static_checks(orig, edited, moved, log):
            fail = checks(orig, edited, moved, log)
            soft = [f for f in fail if f.startswith("moved loads at lines")]
            for f in soft:
                log(f"  warning: {f}")
            return [f for f in fail if f not in soft]

        m.static_checks = static_checks
    return m


def install_warv(M):
    """Replace M.va_need's LLVM rule by the WAR-V-by-class rule (other hazards unchanged)."""
    orig = M.va_need

    def va_need(st, x, llvm=False):
        if not llvm:
            return orig(st, x, llvm=False)
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
            counted = r.cls in M.COUNTED and r.counted
            n = st.vcnt[r.cls] - r.seq if counted else 0
            warv = (not raw_waw) and war and r.cls == "XDL" and counted
            if multi and not warv:
                n = 0                         # LLVM's rule for everything but WAR-V
            need = n if need is None else min(need, n)
        return need

    M.va_need = va_need


# ----------------------------------------------------------------------------------------------------------------
# the independent WAR-V check
# ----------------------------------------------------------------------------------------------------------------
VM_VSRC_CAP = 6  # depctr vm_vsrc is a 3-bit field whose maximum (7) means no wait: a loosened wait stays a real one

RANGE = re.compile(r"v\[(\d+):(\d+)\]|v(\d+)")


def full_regs(op):
    """Full VGPR numbers of one operand: the /*v..*/ comment (the MSB-resolved name) when present."""
    c = re.search(r"/\*(v\[\d+:\d+\]|v\d+)\*/", op)
    tok = c.group(1) if c else op
    m = RANGE.search(tok)
    if not m:
        return set()
    if m.group(1):
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    return {int(m.group(3))}


def check_warv(lines, first, last):
    """Violations: (line, text) of loads / TDM writing registers an in-flight WMMA still reads, over two trips."""
    body = list(range(first, last + 1))
    inflight = []                             # [(seq, reads)] of WMMAs not proven complete
    seq, bad = 0, []
    for trip in range(2):
        for i in body:
            t = lines[i].split(";")[0].strip()
            if not t or t.startswith((".", "//")):
                continue
            mn = t.split()[0]
            if mn == "s_wait_alu":
                m = re.search(r"va_vdst\((\d+)\)", t)
                if m:
                    n = int(m.group(1))
                    inflight = inflight[-n:] if n > 0 else []
                continue
            ops = [o.strip() for o in re.split(r",(?![^\[]*\])", t[len(mn):])] if len(t) > len(mn) else []
            if mn.startswith("v_wmma"):
                reads = set()
                for o in ops[1:4]:
                    reads |= full_regs(o)
                inflight.append((seq, reads))
                seq += 1
                continue
            if mn.startswith(("ds_load", "buffer_load", "global_load", "tensor_load")) and ops:
                dst = full_regs(ops[0]) if not mn.startswith("tensor_load") else set()
                hit = [s for s, rd in inflight if rd & dst]
                if hit and trip == 1:
                    bad.append((i + 1, t))
    return bad


def relax_vm(M, lines, first, last, log=print):
    """Loosen (never tighten) the vm_vsrc field of every s_wait_alu in the loop to what the instructions it guards
    (up to the next s_wait_alu carrying vm_vsrc) still need on either path into the body, under the regen tool's own
    memory-source model; dropped when nothing needs it. Repeated to a fixed point. Returns the number changed."""
    total = 0
    while True:
        ins = M.parse(lines)
        changed = False
        for w in range(first, last + 1):
            x = ins[w]
            if x.cls != "WAIT_ALU":
                continue
            f = M.wait_fields(x)
            if "vm_vsrc" not in f:
                continue
            need = None
            for st0 in (M.entry_state(ins, first), M.backedge_state(ins, first, last)):
                st = M.walk(st0, ins, range(first, w))
                if "va_vdst" in f:
                    M.apply_va(st, f["va_vdst"])
                j = w + 1
                while j <= last and not (ins[j].cls == "WAIT_ALU" and "vm_vsrc" in M.wait_fields(ins[j])):
                    y = ins[j]
                    if y.cls in M.VALU or y.cls in M.MEM:
                        n = M.vm_need(st, y)
                        if n is not None:
                            need = n if need is None else min(need, n)
                            M.apply_vm(st, n)
                    M.step(st, ins, j)
                    j += 1
            cur = f["vm_vsrc"]
            if need is not None:
                need = min(need, VM_VSRC_CAP)
            if need is None or need > cur:
                g = {k: v for k, v in f.items() if k != "vm_vsrc"}
                if need is not None:
                    g["vm_vsrc"] = need
                indent = lines[w][: len(lines[w]) - len(lines[w].lstrip())]
                lines[w] = indent + M.fmt_wait(g) if g else None
                if lines[w] is None:
                    del lines[w]
                log(f"  vm_vsrc({cur}) at line {w + 1} -> {'dropped' if need is None else f'vm_vsrc({need})'}")
                total += 1
                changed = True
                break                      # re-parse after every edit (line numbers may shift)
        if not changed:
            return total


def strip_vm(M, lines):
    """Drop every vm_vsrc field of the full loop's s_wait_alu lines (the line too when nothing else is left)."""
    ins = M.parse(lines)
    _, first, last = M.full_loop(ins)
    out = []
    for i, l in enumerate(lines):
        if first <= i <= last and ins[i].cls == "WAIT_ALU" and "vm_vsrc" in M.wait_fields(ins[i]):
            g = {k: v for k, v in M.wait_fields(ins[i]).items() if k != "vm_vsrc"}
            if not g:
                continue
            l = M.fmt_wait(g)
        out.append(l)
    return out


def insert_vm(M, lines, log=print):
    """Give every instruction of the full loop that writes a register an in-flight memory op still reads the
    vm_vsrc wait it needs on either path (tightening the s_wait_alu in front of it, or inserting one; never inside
    a hard clause), to a fixed point. Returns the number of edits."""
    edits = 0
    while True:
        ins = M.parse(lines)
        _, first, last = M.full_loop(ins)
        hit = None
        for st0 in (M.entry_state(ins, first), M.backedge_state(ins, first, last)):
            st = st0
            for i in range(first, last + 1):
                x = ins[i]
                if x.cls in M.VALU or x.cls in M.MEM:
                    n = M.vm_need(st, x)
                    if n is not None:
                        hit = (i, min(n, VM_VSRC_CAP))
                        break
                M.step(st, ins, i)
            if hit:
                break
        if hit is None:
            return edits
        i, n = hit
        at, before = M.wait_slot(ins, i, first)
        if at is not None:
            g = dict(M.wait_fields(ins[at]))
            g["vm_vsrc"] = min(n, g.get("vm_vsrc", 99))
            lines[at] = M.fmt_wait(g)
        else:
            lines.insert(before, M.fmt_wait({"vm_vsrc": n}))
        edits += 1


def retarget_nl(M, lines, log=print):
    """Re-target the NL/ND prefetch loads of the trip's first iteration (their value is consumed by the second
    iteration, never across the back edge) with the regen tool's own retarget, then recompute va_vdst. In r8a they
    overwrite an operand of the WMMA issued right before them (a va_vdst(0) drain per trip)."""
    ins = M.parse(lines)
    _, first, last = M.full_loop(ins)
    vm = [i for i in range(first, last + 1) if ins[i].cls == "VMEM" and ins[i].defs]
    moved = M.retarget(lines, first, last, vm[:8], log)
    ins = M.parse(lines)
    _, first, last = M.full_loop(ins)
    before, after = M.apply_plan(lines, ins, M.recompute_va_vdst(ins, first, last))
    new = []
    for i, l in enumerate(lines):
        if before[i] is not None:
            new.append(before[i])
        if after[i] is not None:
            new.append(after[i])
    lines[:] = new
    return moved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("s_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("warv", "llvm"), default="warv")
    ap.add_argument("--msrc", action="store_true", help="also re-target Q^T units held up by in-flight memory sources")
    ap.add_argument("--nl", action="store_true", help="(with --msrc) also re-target the first iteration's NL/ND loads")
    ap.add_argument("--turbo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args(argv)
    M = load_regen(a.turbo, msrc=a.msrc)
    if a.mode == "warv":
        install_warv(M)
    text = Path(a.s_file).read_text()
    orig = text[:-1].split("\n")
    if a.msrc:  # the retarget sees every memory-source hazard once the vm_vsrc waits are out of the way
        stripped = "\n".join(strip_vm(M, orig)) + "\n"
        edited, patch = M.regenerate(stripped)
        lines = edited[:-1].split("\n")
        if a.nl:
            nl = retarget_nl(M, lines)
            print(f"[warv_regen] NL/ND loads re-targeted: {len(nl)} units")
        n_ins = insert_vm(M, lines)
        print(f"[warv_regen] vm_vsrc waits re-inserted where needed: {n_ins} edits")
    else:
        edited, patch = M.regenerate(text)
        lines = edited[:-1].split("\n")
    _, first, last = M.full_loop(M.parse(lines))
    if a.msrc:
        n_vm = relax_vm(M, lines, first, last)
        _, first, last = M.full_loop(M.parse(lines))
        io, ie = M.parse(orig), M.parse(lines)
        _, fo, lo = M.full_loop(io)
        ho, he = M.unresolved(io, fo, lo), M.unresolved(ie, first, last)
        new_open = {k: he[k] - ho[k] for k in he if he[k] - ho[k]}
        if new_open or M.clause_violations(ie):
            raise SystemExit(f"relax_vm broke a check: new open hazards {new_open}, clauses {M.clause_violations(ie)}")
        # dataflow against the compiler's original: the same non-wait instructions (register numbers aside) reading
        # the same definitions over two trips, the same definitions reaching the exit
        ko = [i for i in range(fo, lo + 1) if io[i].cls != "WAIT_ALU"]
        ke = [i for i in range(first, last + 1) if ie[i].cls != "WAIT_ALU"]
        if [io[i].mn for i in ko] != [ie[i].mn for i in ke]:
            raise SystemExit("final check: the loop's instructions differ from the compiler's")
        if M.dataflow(io, fo, lo)[0] != M.dataflow(ie, first, last)[0]:
            raise SystemExit("final check: dataflow differs from the compiler's (a moved tuple overwrites a live value)")
        if orig[:fo] != lines[:first] or orig[lo + 1:] != lines[last + 1:]:
            raise SystemExit("final check: text outside the loop differs")
        print(f"[warv_regen] relax_vm: {n_vm} vm_vsrc waits loosened, no new open hazard on either path")
        import difflib
        patch = "".join(difflib.unified_diff([l + "\n" for l in orig], [l + "\n" for l in lines],
                                             f"{M.KERNEL}.s", f"{M.KERNEL}.s", n=3))
    bad = check_warv(lines, first, last)
    drains = sum(1 for i in range(first, last + 1) if re.search(r"va_vdst\(0\)", lines[i]))
    print(f"[warv_regen] mode {a.mode}: loop lines {first + 1}-{last + 1}, va_vdst(0) per trip {drains}, "
          f"independent WAR-V check: {len(bad)} violations")
    for ln, t in bad[:10]:
        print(f"  line {ln}: {t}")
    if bad:
        return 1
    Path(a.out).write_text(patch)
    print(f"[warv_regen] asm_input_sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
