###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""build.py -- build k_dkdv96ds from its marker shell and the generated regions (bwdgen).

    python3 build.py --out DIR [--cxx CXX]

 1. compile the shell (hip/src/k_dkdv96ds.hip, exactly hsaco.py's command: the manifest's flags, -DHKGEN=1):
    DIR/shell.s; and the LLVM-compiled kernel (-DHKGEN=0), whose operands in high VGPR blocks teach, with the
    shell's, the s_set_vgpr_msb operand-field map;
 2. read the markers (INIT, LOOP, FINAL per block height nkv) and the registers LLVM gave their operands; base = 1 +
    the highest VGPR the shell references on a path from INIT to FINAL (shellbase) or as a marker operand;
 3. generate the regions (bwdgen); asmkit.emit derives s_set_vgpr_msb, the expert-mode waits and the hazard v_nops
    (the loop's blocks to a fixed point of their entry states); every region ends with s_wait_xcnt 0 + vm_vsrc 0; a
    re-walk of every block from its entry state must find no open hazard;
 4. splice the regions in place of the marker bodies, VGPR allocation 1024: DIR/k_dkdv96ds.s;
 5. assemble with the driver's own cc1as / lld (asmkit.build): DIR/k_dkdv96ds.hsaco; asmkit.verify_obj;
 6. contract: the generated code writes VGPRs only at or above base and SGPRs only in the markers' clobber lists;
    no spills.
Writes DIR/build.json; exit status 1 on any failed check. CPU only."""

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
HIP = TOOLS.parent / "hip"
sys.path[:0] = [str(HERE), str(TOOLS)]
import bwdgen as G  # noqa: E402
import shellbase  # noqa: E402
from asmkit import build as B  # noqa: E402
from asmkit import emit as M  # noqa: E402
from asmkit import expert as E  # noqa: E402
from asmkit import verify_obj as VO  # noqa: E402
from asmkit.isa import coverage_errors, parse, parse_line, sgpr_defs_uses  # noqa: E402

KERNEL = "k_dkdv96ds"
MARK = re.compile(r";\s*HKGEN\s+(\w+)\s*(.*)$")
REGIONS = ["FINAL", "INIT", "LOOP"]
VOP3_FIELDS = ("dst", "src0", "src1", "src2")
# SGPRs a region may write: the shell declares them clobbered (HKGEN_CLOBBERS, HKGEN_LOOP_CLOBBERS)
CLOBBERED = {f"s{i}" for i in range(90, 104)} | {"vcc", "scc"}
LOOP_CLOBBERED = CLOBBERED | {f"s{i}" for i in range(48, 90)}
# region tail: the shell's code after a region may rewrite the SGPRs (XCNT) and VGPRs (vm_vsrc) an in-flight memory
# op of the region still reads
POST = ["\ts_wait_xcnt 0x0", "\ts_wait_alu depctr_vm_vsrc(0)"]


def log(*a):
    print(*a, flush=True)


def load_hsaco():
    spec = importlib.util.spec_from_file_location("_gfx1250_mla_hsaco", TOOLS.parent / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_variant(hsaco, cxx, out_dir, reference):
    """Compile the shell (exactly hsaco.py's command) or, reference=True, the LLVM-compiled kernel (-DHKGEN=0 in
    place of the -DHKGEN* flags) into out_dir; returns (compile argv, the compiler's .s)."""
    tmp = out_dir / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    extra = list(hsaco.load_manifest()["kernels"][KERNEL]["extra_flags"])
    if reference:
        extra = [f for f in extra if not f.startswith("-DHKGEN")] + ["-DHKGEN=0"]
    cmd = hsaco.compile_command(cxx, KERNEL, tmp / f"{KERNEL}.cc.hsaco", extra)
    r = subprocess.run(cmd, cwd=HIP, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    return cmd, next(tmp.glob("*-gfx1250.s"))


def parse_markers(lines):
    """{nkv: {region: (operands, line index)}} of the HKGEN markers; each must be an inline asm statement of its own,
    and every block height needs exactly the INIT, LOOP and FINAL markers."""
    sets = {}
    for i, l in enumerate(lines):
        m = MARK.search(l)
        if not m:
            continue
        if ";;#ASMSTART" not in lines[i - 1] or ";;#ASMEND" not in lines[i + 1]:
            raise ValueError(f"marker at line {i + 1} is not a lone inline asm statement")
        name, ops = m.group(1), dict(kv.split("=", 1) for kv in m.group(2).split())
        if "nkv" not in ops:
            raise ValueError(f"marker {name} at line {i + 1} has no nkv")
        by = sets.setdefault(int(ops.pop("nkv")), {})
        if name in by:
            raise ValueError(f"marker {name} at line {i + 1}: a second one for its nkv")
        by[name] = (ops, i)
    for nkv, by in sets.items():
        if sorted(by) != REGIONS:
            raise ValueError(f"nkv={nkv}: markers {sorted(by)}, want {REGIONS}")
    return sets


def items_of(texts, fmap):
    """asmkit items of instruction texts. A VALU / WMMA mnemonic missing from the learned field map gets the VOP3
    field order (dst, src0, src1, src2)."""
    for t in texts:
        x = parse_line(0, "\t" + t)
        if x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS"):
            for p in range(min(len(x.ops), 4)):
                fmap.setdefault((x.mn, p), {VOP3_FIELDS[p]})
    items = [M.make_item("\t" + t) for t in texts]
    for k, it in enumerate(items):
        it.pos = k
    return items


def emit_block(items, flow, fmap):
    """(lines, exit flow) of one block entered by `flow`: s_set_vgpr_msb, expert-mode waits and v_nops derived."""
    # va_vdst: LLVM's rule, but the per-class bound for a memory op that overwrites registers only in-flight WMMAs read
    return M.emit_block(items, flow, fmap, va_rule="llvm_war_xdl")


def unknown_flow():
    return M.Flow([E.State(unknown=True)], [M.XState(True)], [[]])


def with_unknown(fl, regs):
    """fl after unknown shell code: VALU writes in flight on regs, unknown XCNT sources."""
    fl = fl.copy()
    for st in fl.sts:
        st.valu.append(E.Rec("UNK", [], sorted(regs), False, 0))
    for x in fl.xs:
        x.unknown = True
    return fl


def tail(fl):
    """fl after the region tail POST."""
    fl = fl.copy()
    for x in fl.xs:
        x.unknown, x.regs = False, set()
    for st in fl.sts:
        E.apply_vm(st, 0)
    return fl


def open_hazards(lines, flow):
    """(hazards left open, parse-coverage errors) of lines walked from flow with their own waits."""
    ins = parse(lines)
    found = []

    def note(k, field, _):
        found.append((k, field, ins[k].text.strip()))

    M.walk_block(ins, flow.copy(), note)
    return found, [(k, e) for k, x in enumerate(ins) for e in coverage_errors(x)]


def generate(nkv, by, ins, bl, shell_hi, fmap, rec):
    """Emit and check the regions of block height nkv (by = {region: (operands, marker line)}); returns {marker line:
    region lines}. Failed checks go to rec["fail"]."""
    fail = rec["fail"]
    G.set_nkv(nkv)
    hi, seen = shellbase.shell_span(ins, bl, by["INIT"][1], by["FINAL"][1])
    base, why = hi + 1, f"shell code v{hi}"
    for name, (ops, _) in by.items():
        for k, v in ops.items():
            m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", v)
            if m and int(m.group(3) or m.group(1)) + 1 > base:
                base, why = int(m.group(3) or m.group(1)) + 1, f"{name} operand {k}={v}"
    log(f"nkv={nkv}: base v{base} set by {why}")
    if shell_hi[0] >= base:
        fail.append(
            f"nkv={nkv}: the shell references v{shell_hi[0]} (line {shell_hi[1]}) at or above base v{base}"
        )
    L = G.Layout(base)
    rec["layout"][nkv] = {"base": base, "top": L.top}
    log(
        f"nkv={nkv} shell: highest VGPR between INIT and FINAL v{hi}, base v{base}, generated state up to "
        f"v{L.top - 1}"
    )
    # every load and store the regions wait for is their own: the shell's code between INIT and FINAL may use the
    # TDM only
    for i in sorted(seen):
        if ins[i].cls in ("DS", "VMEM", "FLAT"):
            fail.append(
                f"nkv={nkv}: shell memory op between INIT and FINAL at line {i + 1}: {ins[i].text.strip()}"
            )
    shell_regs = set(range(base))
    entry, lines = {}, {}  # per block: entry flow, emitted lines

    entry["INIT"] = unknown_flow()
    t_init = G.region_init(L, by["INIT"][0])
    lines["INIT"], fl = emit_block(items_of(t_init, fmap), entry["INIT"], fmap)
    exit_init = tail(fl)

    # LOOP: its prologue RB, then the loop's blocks to a fixed point of their entry flows
    lp = f".LHKG{nkv}"
    t_rb, blocks, model = G.region_loop(L, by["LOOP"][0], lp)
    entry["RB"] = with_unknown(exit_init, shell_regs)
    lines["RB"], exit_rb = emit_block(items_of(t_rb, fmap), entry["RB"], fmap)
    names = [b[0] for b in blocks]
    items = {name: items_of(texts, fmap) for name, _, texts, _ in blocks}
    preds = {n: [] for n in names}
    for name, _, _, succ in blocks:
        for t in succ:
            if t != "X":
                preds[t].append(name)
    preds["H0"].append("RB")
    exits, sig = {"RB": exit_rb}, {}
    for it in range(96):
        changed = False
        for n in names:
            ins_fl = [exits[p] for p in preds[n] if p in exits]
            if not ins_fl:
                continue
            fl_in = M.Flow.join(ins_fl)
            sg = fl_in.sig()
            if sig.get(n) == sg:
                continue
            sig[n], entry[n] = sg, fl_in
            lines[n], exits[n] = emit_block(items[n], fl_in, fmap)
            changed = True
        if not changed:
            break
    else:
        fail.append(f"nkv={nkv}: LOOP entry states did not converge")
    rec["checks"]["loop_fixed_point_iterations"][nkv] = it + 1
    loop_lines = list(lines["RB"])
    for n, label, _, _ in blocks:
        loop_lines += [f"{label}:"] + lines[n]
    loop_lines.append(f"{lp}_X:")

    # FINAL: after INIT (no tile) or a loop exit, through shell code
    x_exits = [tail(exits[n]) for n, _, _, succ in blocks if "X" in succ]
    entry["FINAL"] = with_unknown(M.Flow.join([exit_init] + x_exits), shell_regs)
    t_fin = G.region_final(L, by["FINAL"][0])
    lines["FINAL"], _ = emit_block(items_of(t_fin, fmap), entry["FINAL"], fmap)

    region_lines = {"INIT": lines["INIT"], "LOOP": loop_lines, "FINAL": lines["FINAL"]}
    n_ins = {"INIT": len(t_init), "LOOP": len(t_rb) + sum(len(b[2]) for b in blocks), "FINAL": len(t_fin)}
    regions = {}
    for name, (_, i) in by.items():
        out = region_lines[name] + POST
        allowed = LOOP_CLOBBERED if name == "LOOP" else CLOBBERED
        for l in out:
            x = parse_line(0, l)
            if not x.mn:
                continue
            low = [r for r in x.defs if r < base]
            if low:
                fail.append(f"nkv={nkv} {name}: writes shell VGPRs {low}: {l.strip()}")
            bad = [r for r in sgpr_defs_uses(x)[0] if r not in allowed]
            if bad and not x.mn.startswith("s_wait") and x.cls != "MSB":
                fail.append(f"nkv={nkv} {name}: writes SGPRs {bad}: {l.strip()}")
        regions[i] = out
        nw = sum(1 for l in out if "v_wmma" in l)
        nva0 = sum(1 for l in out if "va_vdst(0)" in l)
        log(f"nkv={nkv} {name}: {n_ins[name]} instructions -> {len(out)} lines, {nw} WMMA, {nva0} va_vdst(0)")
    log(f"nkv={nkv} LOOP model cycles per body: {model}")
    rec["model_cycles"][nkv] = model
    for name in ["INIT", "RB"] + names + ["FINAL"]:
        found, cov = open_hazards(lines[name], entry[name])
        if found or cov:
            fail.append(
                f"nkv={nkv} {name}: {len(found)} open hazards, {len(cov)} coverage errors, e.g. "
                f"{(found + cov)[:3]}"
            )
    return regions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cxx", default="amdclang++", help="compiler (the manifest's toolchain)")
    a = ap.parse_args()
    hsaco = load_hsaco()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "recipe": {},
        "checks": {"loop_fixed_point_iterations": {}},
        "fail": [],
        "layout": {},
        "model_cycles": {},
    }
    fail = rec["fail"]
    cmd, s_path = compile_variant(hsaco, a.cxx, out / "shell", reference=False)
    rec["recipe"]["compile"] = cmd
    shell = s_path.read_text().rstrip("\n").split("\n")
    (out / "shell.s").write_text("\n".join(shell) + "\n")
    _, ref_s = compile_variant(hsaco, a.cxx, out / "ref", reference=True)
    ins = parse(shell)
    fmap = M.merge_msb_maps(M.learn_msb_map(ins), M.learn_msb_map(parse(ref_s.read_text().split("\n"))))
    # DS operand fields (vdst: dst, address: src0, data: src1), whatever the compiles left ambiguous
    for mn in ("ds_load_b128", "ds_load_tr16_b128"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    fmap[("ds_store_b128", 0)], fmap[("ds_store_b128", 1)] = {"src0"}, {"src1"}

    sets = parse_markers(shell)
    bl = E.blocks(ins)
    for x in ins:
        if x.cls in ("XDL", "TRANS"):
            fail.append(f"shell line {x.li + 1} has a WMMA / TRANS: {x.text.strip()}")
    # the highest VGPR the shell's code references anywhere: a value the compiler keeps in a VGPR across the regions
    # (defined before INIT, used after FINAL) is not referenced between them, so all of it must stay below every base
    shell_hi = (-1, None)
    for x in ins:
        vs = x.defs + x.uses
        if x.mn and vs and max(vs) > shell_hi[0]:
            shell_hi = (max(vs), (x.li + 1, x.text.strip()[:80]))
    rec["checks"]["shell_highest_vgpr"] = shell_hi[0]
    regions = {}
    for nkv in sorted(sets, reverse=True):
        regions.update(generate(nkv, sets[nkv], ins, bl, shell_hi, fmap, rec))

    new = list(shell)
    for i in sorted(
        regions, reverse=True
    ):  # each region right behind its marker, inside the inline asm statement
        new[i + 1 : i + 1] = regions[i]
    new = B.set_vgpr_count(new, 1024)
    spl = out / f"{KERNEL}.s"
    spl.write_text("\n".join(new) + "\n")
    cc1as, lld = B.driver_steps(cmd, str(HIP))
    meta = B.assemble(spl, out / f"{KERNEL}.hsaco", cc1as, lld, cwd=str(HIP))
    rec["hsaco"] = meta
    log(f"hsaco: {meta}")
    bad = VO.compare(str(spl), str(out / f"{KERNEL}.objdump.s"), log=log)
    rec["checks"]["verify_obj"] = "ok" if not bad else [str(b) for b in bad]
    if bad:
        fail.append(f"verify_obj: {bad[:3]}")
    if meta.get("vgpr_spill_count", 0) or meta.get("private_segment_fixed_size", 0):
        fail.append(f"spills / scratch: {meta}")
    for f in fail:
        log("FAIL:", f)
    rec["verdict"] = "PASS" if not fail else "FAIL"
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    log(f"VERDICT {rec['verdict']}  {out / (KERNEL + '.hsaco')}  sha256 {meta.get('sha256')}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
