###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""build.py -- build k_dkdv96ds from the HKGEN marker shell (hip/src/k_dkdv96ds.hip -DHKGEN=1) and r8gen's regions.

    python3 gen/build.py --out DIR [--sched list|program] [--rule llvm|warv] [-D NAME=V ...] [--cxx amdclang++]

 1. compile the shell (the ladder's k_dkdv96ds flags + -DHKGEN=1, -save-temps): DIR/shell.s; and the LLVM-compiled
    variant (-DHKGEN=0: r9a) as DIR/ref.s, whose high-VGPR buffer / DS / VALU operands teach the MSB field map;
 2. markers -> operand registers; base = 1 + the highest VGPR the shell's code touches on any path from INIT to FINAL
    (gen/shellbase.py) or any VGPR marker operand;
 3. r8gen's regions (INIT, RB, STEP, STEPM, FINAL); asmkit.emit derives s_set_vgpr_msb / expert waits (va_vdst:
    --rule llvm = LLVM's, warv = per-class bound for WAR-V only) / hazard v_nops from an unknown entry state; every
    region ends with s_wait_xcnt 0 + vm_vsrc 0 (the shell may rewrite the operands an in-flight memory op reads); a
    re-walk must find no open hazard;
 4. splice in place of the marker bodies, VGPR allocation 1024: DIR/k_dkdv96ds.s;
 5. assemble with the driver's own cc1as / lld (asmkit.build): DIR/k_dkdv96ds.hsaco; asmkit.verify_obj;
 6. contract: generated VGPR defs only at or above base; SGPR defs only in s90..s103 / vcc / scc; no spills.
Writes DIR/build.json; exit 1 on any failed check. CPU only."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
# the HIP tree: the package's own hip/ (the ladder's copy), else Primus-Turbo's gfx1250_mla_bwd/hip (tools/r8gen/..)
HIP = PKG / "hip" if (PKG / "hip").is_dir() else PKG.parent.parent / "hip"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKG))
import r8gen as G  # noqa: E402
import shellbase  # noqa: E402

FLAGS = ["-x", "hip", "--offload-arch=gfx1250", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++20",
         "-DNDEBUG", "-DKITTENS_CDNA5", "-Iinclude", "-I.", "-save-temps=obj"]  # fmt: skip
EXTRA = ["-mllvm", "-structurizecfg-skip-uniform-regions", "-mllvm", "-amdgpu-expert-scheduling-mode", "-mllvm",
         "-amdgpu-enable-delay-alu=false", "-mllvm", "-greedy-regclass-priority-trumps-globalness"]  # fmt: skip
KERNEL = "k_dkdv96ds"
MARK = re.compile(r";\s*HKGEN\s+(\w+)\s*(.*)$")
CLOBBERED = {f"s{i}" for i in range(90, 104)} | {"vcc", "scc"}
LOOP_CLOBBERED = CLOBBERED | {f"s{i}" for i in range(48, 90)}
# region tails: the shell may rewrite the SGPR operands an in-flight memory op still reads (XCNT); after INIT / RB /
# FINAL also the VGPR operands (the shell's VALU) -- between steps it writes no VGPR a region reads (checked)
POST_STEP = ["\ts_wait_xcnt 0x0"]
POST = ["\ts_wait_xcnt 0x0", "\ts_wait_alu depctr_vm_vsrc(0)"]
STEPS = []  # (the per-tile markers of the first generation; the LOOP marker replaces them)


def log(*a):
    print(*a, flush=True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compile_variant(cxx, out_dir, defs):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    cmd = [cxx, *FLAGS, *EXTRA, f"-cuid={KERNEL}", *[f"-D{d}" for d in defs], "-c", "src/k_dkdv96ds.hip", "-o",
           str(out_dir / f"{KERNEL}.cc.hsaco")]  # fmt: skip
    r = subprocess.run(cmd, cwd=HIP, capture_output=True, text=True)
    for d in (HIP, out_dir):
        for f in d.glob(f"{KERNEL}-hip-*"):
            shutil.move(str(f), str(tmp / f.name))
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    return cmd, next(tmp.glob("*-gfx1250.s"))


def parse_markers(lines):
    out = []
    for i, l in enumerate(lines):
        m = MARK.search(l)
        if not m:
            continue
        if ";;#ASMSTART" not in lines[i - 1] or ";;#ASMEND" not in lines[i + 1]:
            raise ValueError(f"marker at line {i + 1} is not a lone inline asm statement")
        ops = dict(kv.split("=", 1) for kv in m.group(2).split())
        out.append((m.group(1), ops, i, i - 1, i + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sched", default="slot", choices=("win", "slot", "list", "program"))
    ap.add_argument("--rule", default="warv", choices=("llvm", "warv"))
    ap.add_argument("--cxx", default="amdclang++")
    ap.add_argument("-D", dest="defs", action="append", default=[])
    ap.add_argument("--abl", default="", help="timing-only ablations, comma separated (r8gen.ABL)")
    ap.add_argument("--probe-pts", default="", help="HKGEN_PROBE: three WMMA counts (comma separated) as probe points")
    ap.add_argument("--hh1-split", type=int, default=0, help="r8gen.HH1_SPLIT (0 kh-major .. 6 dt outer)")
    ap.add_argument("--bdo-after-hh1", action="store_true", help="r8gen.BDO_AFTER_HH1")
    ap.add_argument("--win", default="", help="bwdgen.WIN overrides for --sched win: KEY=VAL,KEY=VAL")
    ap.add_argument("--nd-fold", action="store_true", help="r8gen.ND_FOLD (needs k_delta -DHK_ND_FOLD=1)")
    ap.add_argument("--scal-in-body", action="store_true", help="r8gen.SCAL_IN_BODY (the next tile's scalars in the bodies)")
    ap.add_argument("--tdm-dummy", action="store_true", help="r8gen.TDM_DUMMY (16-B ops past the last tile)")
    ap.add_argument("--asmkit-root", default=None, help="ignored: the package carries its own asmkit")
    a = ap.parse_args()
    from asmkit import build as B
    from asmkit import emit as M
    from asmkit import expert as E
    from asmkit import splice as SP
    from asmkit import verify_obj as VO
    from asmkit.isa import cnt_fields, coverage_errors, parse, parse_line, sgpr_defs_uses

    M.VA_RULE_LLVM = {"llvm": True, "warv": "warv"}[a.rule]
    G.ABL = set(x for x in a.abl.split(",") if x)
    G.PROBE_PTS = [int(x) for x in a.probe_pts.split(",") if x] or None
    G.HH1_SPLIT, G.BDO_AFTER_HH1, G.ND_FOLD = a.hh1_split, a.bdo_after_hh1, a.nd_fold
    G.SCAL_IN_BODY, G.TDM_DUMMY = a.scal_in_body, a.tdm_dummy
    import bwdgen as BG

    for kv in [x for x in a.win.split(",") if x]:
        k, v = kv.split("=")
        if k not in BG.WIN:
            raise SystemExit(f"--win: unknown key {k}")
        BG.WIN[k] = type(BG.WIN[k])(float(v)) if isinstance(BG.WIN[k], (int, float)) else v
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = {"recipe": {"sched": a.sched, "rule": a.rule, "defs": a.defs}, "checks": {}, "fail": []}
    fail = rec["fail"]
    cmd, s_path = compile_variant(a.cxx, out / "shell", ["HKGEN=1"] + a.defs)
    shell = s_path.read_text().rstrip("\n").split("\n")
    (out / "shell.s").write_text("\n".join(shell) + "\n")
    rec["recipe"]["compile"] = cmd
    _, ref_s = compile_variant(a.cxx, out / "ref", ["HKGEN=0"])
    fmap = M.merge_msb_maps(M.learn_msb_map(parse(shell)), M.learn_msb_map(parse(ref_s.read_text().split("\n"))))
    fields = ("dst", "src0", "src1", "src2")
    for mn in ("ds_load_b128", "ds_load_tr16_b128"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    fmap[("ds_store_b128", 0)], fmap[("ds_store_b128", 1)] = {"src0"}, {"src1"}

    markers = parse_markers(shell)
    sets = {}
    for m in markers:
        nkv = int(m[1].pop("nkv", "3"))
        if m[0] in sets.setdefault(nkv, {}):
            raise RuntimeError(f"marker {m[0]} nkv={nkv} twice")
        sets[nkv][m[0]] = m
    for nkv, by in sets.items():
        if sorted(n for n in by if n != "PROBE") != ["FINAL", "INIT", "LOOP"]:
            raise RuntimeError(f"nkv={nkv}: markers {sorted(by)}, want FINAL INIT LOOP (+ PROBE)")
    ins = parse(shell)
    bl = E.blocks(ins)
    for i, l in enumerate(shell):
        x = parse_line(i, l)
        if x.cls in ("XDL", "TRANS"):
            fail.append(f"shell line {i + 1} has a WMMA / TRANS: {l.strip()}")
    rec["layout"], rec["model_cycles"], rec["checks"]["loop_fixed_point_iterations"] = {}, {}, {}

    def walk_check(lines):
        ins2 = parse(["\t" + l.strip() if not l.startswith("\t") else l for l in lines])
        fl = M.Flow([E.State(unknown=True)], [M.XState(True)], [[]])
        found = []
        SP.walk_block(ins2, 0, len(ins2) - 1, fl, True, lambda k, f, n: found.append((k, f, ins2[k].text.strip())))
        cov = [(k, e) for k, x in enumerate(ins2) for e in coverage_errors(x)]
        return found + [("coverage", c) for c in cov]

    def items_of(texts):
        for t in texts:
            x = parse_line(0, "\t" + t)
            for p, o in enumerate(x.ops):
                key = (x.mn, p)
                if key not in fmap and x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS") and p < 4:
                    fmap[key] = {fields[p]}
        items = [M.make_item("\t" + t) for t in texts]
        for k, it in enumerate(items):
            it.pos = k
        return items

    def with_unknown(fl, regs):
        """fl + unknown VALU writes in flight on regs (the shell's code in front of the region) + unknown XCNT."""
        fl = fl.copy()
        for st in fl.sts:
            st.valu.append(E.Rec("UNK", [], sorted(regs), False, 0))
        for x in fl.xs:
            x.unknown = True
        return fl

    def tail(fl):
        """The region's tail waits (xcnt 0, vm_vsrc 0) applied to its exit flow."""
        fl = fl.copy()
        for x in fl.xs:
            x.unknown, x.regs = False, set()
        for st in fl.sts:
            E.apply_vm(st, 0)
        return fl

    def emit(items, fl):
        return M._emit_block(items, fl, fmap, True)

    def explain(items, fl, label):
        """Debug (HKGEN_EXPLAIN): every instruction of a block that needs va_vdst 0, with the in-flight records
        behind it."""
        fl = fl.copy()
        for n in items:
            x = n.x
            for st in fl.sts:
                if x.cls in ("DS", "VMEM", "FLAT", "TDM"):
                    nd = E.va_need(st, x, llvm=M.VA_RULE_LLVM)
                    if nd == 0:
                        rs, ws = set(x.uses), set(x.defs)
                        why = [(r.cls, r.counted, sorted(r.reads & ws)[:2], sorted(r.writes & (rs | ws))[:2])
                               for r in st.valu if not r.done and (r.writes & (rs | ws) or r.reads & ws)]
                        log(f"  [{label}] va0 before {x.text.strip()[:70]}  unknown_v={st.unknown_v} recs={why[:4]}")
                        E.apply_va(st, 0)
                if x.cls == "WAIT_CNT":
                    E.apply_cnt(st, cnt_fields(x))
                else:
                    E.issue(st, x)

    # every VGPR the shell's own code references anywhere in the function (not just between INIT and FINAL): a value
    # the compiler keeps in a VGPR across the generated regions (defined before INIT, used after FINAL) is not
    # "touched" in between, so the whole function must stay below every base
    shell_hi, shell_hi_at = -1, None
    marker_lines = set()
    for m in markers:
        marker_lines |= set(range(m[3], m[4] + 1))
    for i, x in enumerate(ins):
        if i in marker_lines or not x.mn or x.cls in ("LABEL", "DIR"):
            continue
        vs = [r for r in list(x.defs) + list(x.uses) if isinstance(r, int)]
        if vs and max(vs) > shell_hi:
            shell_hi, shell_hi_at = max(vs), (i + 1, shell[i].strip()[:80])
    rec["checks"]["shell_highest_vgpr"] = shell_hi
    regions = {}  # marker line -> lines
    for nkv in sorted(sets, reverse=True):
        by = sets[nkv]
        G.set_nkv(nkv)
        mlines = {m[2] for m in markers}
        hi, seen = shellbase.shell_span(shell, ins, bl, by["INIT"][2], by["FINAL"][2], mlines)
        base, why = hi + 1, f"shell code v{hi}"
        for mk, (_, ops, _, _, _) in by.items():
            for k, v in ops.items():
                m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", v)
                if m and int(m.group(3) or m.group(1)) + 1 > base:
                    base, why = int(m.group(3) or m.group(1)) + 1, f"{mk} operand {k}={v}"
        log(f"nkv={nkv}: base v{base} set by {why}")
        if hi >= 15:
            for i in sorted(seen):
                vs = [r for r in list(ins[i].defs) + list(ins[i].uses) if r >= 15]
                if vs:
                    log(f"   line {i + 1}: {shell[i].strip()[:100]}")
        if shell_hi >= base:
            fail.append(f"nkv={nkv}: the shell references v{shell_hi} (line {shell_hi_at}) at or above base v{base}")
        L = G.Layout(base)
        rec["layout"][nkv] = {"base": base, "top": L.top}
        log(f"nkv={nkv} shell: highest VGPR between INIT and FINAL v{hi}, base v{base}, generated state up to "
            f"v{L.top - 1}")  # fmt: skip
        # shell instructions between INIT and FINAL: no memory op but the TDM (the loads / stores the regions count
        # on are all generated); the LOOP region runs no shell code inside
        for i in seen:
            x = ins[i]
            if x.cls in ("DS", "VMEM", "FLAT"):
                fail.append(f"nkv={nkv}: shell memory op between INIT and FINAL at line {i + 1}: {shell[i].strip()}")
        shell_regs = set(range(base))
        unknown = M.Flow([E.State(unknown=True)], [M.XState(True)], [[]])
        lines_of, entry_fl = {}, {}
        # INIT
        t_init = G.region_init(L, by["INIT"][1])
        entry_fl["INIT"] = unknown
        lines_of["INIT"], fl = emit(items_of(t_init), unknown)
        exit_init = tail(fl)
        # LOOP: its init block, then the loop blocks to a fixed point of the entry flows
        probe = "PROBE" in by
        lp = f".LHKG{nkv}"
        t_rb, blocks, model = G.region_loop(L, by["LOOP"][1], a.sched, probe, lp)
        entry_fl["RB"] = with_unknown(exit_init, shell_regs)
        lines_of["RB"], exit_rb = emit(items_of(t_rb), entry_fl["RB"])
        names = [b[0] for b in blocks]
        its_b = {b[0]: items_of(b[2]) for b in blocks}
        preds = {n: [] for n in names}
        for n, _, _, succ in blocks:
            for t in succ:
                if t != "X":
                    preds[t].append(n)
        preds["H0"].append("RB")
        exits = {"RB": exit_rb}
        lines_b, sig_b = {}, {}
        for it in range(96):
            changed = False
            for n in names:
                ins_fl = [exits[p] for p in preds[n] if p in exits]
                if not ins_fl:
                    continue
                fl_in = M.Flow.join(ins_fl)
                sg = fl_in.sig()
                if sig_b.get(n) == sg:
                    continue
                sig_b[n] = sg
                entry_fl[n] = fl_in
                lines_b[n], exits[n] = emit(its_b[n], fl_in)
                changed = True
            if not changed:
                break
        else:
            (log if G.ABL else fail.append)(f"nkv={nkv}: LOOP entry states did not converge")
        rec["checks"]["loop_fixed_point_iterations"][nkv] = it + 1
        if "HKGEN_EXPLAIN=1" in a.defs:
            for n in ("U1",):
                explain(its_b[n], entry_fl[n], f"nkv{nkv} {n}")
        loop_lines = list(lines_of["RB"])
        for n, label, _, _ in blocks:
            loop_lines += [f"{label}:"] + lines_b[n]
        loop_lines.append(f"{lp}_X:")
        lines_of["LOOP"] = loop_lines
        x_exits = [tail(exits[n]) for n, _, _, succ in blocks if "X" in succ]
        fin_in = with_unknown(M.Flow.join([exit_init] + x_exits), shell_regs)
        t_fin = G.region_final(L, by["FINAL"][1])
        entry_fl["FINAL"] = fin_in
        lines_of["FINAL"], _ = emit(items_of(t_fin), fin_in)
        texts_of = {"INIT": t_init, "LOOP": t_rb + [t for b in blocks for t in b[2]], "FINAL": t_fin}
        if probe:
            t_pr = [f"v_readfirstlane_b32 {by['PROBE'][1][k]}, {G.vr(L.PRB[j])}" for j, k in enumerate("abcd")]
            lines_of["PROBE"], _ = emit(items_of(t_pr), unknown)
            texts_of["PROBE"] = t_pr

        for name, (_, ops, i, _, _) in by.items():
            lines = lines_of[name] + POST
            allowed = LOOP_CLOBBERED if name == "LOOP" else CLOBBERED
            for l in lines:
                x = parse_line(0, l)
                if not x.mn or x.cls in ("LABEL", "DIR"):
                    continue
                low = [r for r in x.defs if r < base]
                if low:
                    fail.append(f"nkv={nkv} {name}: writes shell VGPRs {low}: {l.strip()}")
                sd, _ = sgpr_defs_uses(x)
                bad = [r for r in sd if r not in allowed]
                if bad and not x.mn.startswith("s_wait") and x.cls != "MSB" and name != "PROBE":
                    fail.append(f"nkv={nkv} {name}: writes SGPRs {bad}: {l.strip()}")
            regions[i] = lines
            nw = sum(1 for l in lines if "v_wmma" in l)
            nva0 = sum(1 for l in lines if "va_vdst(0)" in l)
            log(f"nkv={nkv} {name}: {len(texts_of[name])} instructions -> {len(lines)} lines, {nw} WMMA, "
                f"{nva0} va_vdst(0)")  # fmt: skip
        log(f"nkv={nkv} LOOP model cycles per body: {model}")
        rec["model_cycles"][nkv] = model
        # re-walk every block's emitted lines from its own entry flow: nothing may stay open
        for name, ls0 in [("INIT", lines_of["INIT"]), ("RB", lines_of["RB"])] + [(n, lines_b[n]) for n in names] + \
                [("FINAL", lines_of["FINAL"])]:  # fmt: skip
            ls = ["\t" + l.strip() if not l.startswith("\t") else l for l in ls0]
            ins2 = parse(ls)
            found = []
            SP.walk_block(ins2, 0, len(ins2) - 1, entry_fl[name].copy(), True,
                          lambda k, f, n: found.append((k, f, ins2[k].text.strip())))  # fmt: skip
            cov = [(k, e) for k, x in enumerate(ins2) for e in coverage_errors(x)]
            if found or cov:
                fail.append(f"nkv={nkv} {name}: {len(found)} open hazards, {len(cov)} coverage errors, e.g. "
                            f"{(found + cov)[:3]}")  # fmt: skip

    new = list(shell)
    for name, ops, i, s0, s1 in sorted(markers, key=lambda m: -m[2]):
        new[s0 + 1 : s1] = [new[i]] + regions[i]
    new = SP.set_vgpr_count(new, 1024)
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
