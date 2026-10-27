###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""build.py -- build the HK MLA forward code object (W-S1a) from the marker shell and the generated regions.

    python gen/build.py --out DIR [--sched list|program] [--asmkit-root DIR] [--cxx amdclang++] [--ref]

Steps (all on the CPU; nothing here touches a GPU):
 1. compile hip/k_mla_fwd.hip (the shell, HKGEN_REF=0) with the manifest-style flags + expert scheduling mode and
    -save-temps: DIR/shell.s is the compiler's own assembly;
 2. read every HKGEN marker (inline asm) and the registers LLVM gave its operands; base = 1 + the highest VGPR the
    shell's code references (the generated code lives above it);
 3. generate each region (fwdgen), let asmkit.emit add s_set_vgpr_msb / expert waits / hazard v_nops (entry state:
    unknown), and check it again with asmkit's walk (no hazard left open);
 4. splice the regions in place of the marker bodies, raise the VGPR allocation to 1024: DIR/k_mla_fwd.s;
 5. assemble with the clang driver's own cc1as / lld commands (asmkit.build): DIR/k_mla_fwd.hsaco (+ .notes,
    .objdump.s), and compare the code object with the intended .s (asmkit.verify_obj);
 6. contract checks on the code object's listing (regions found by instruction ordinal): mnemonic whitelist (no
    global / buffer / flat / scratch / SMEM store / TDM in generated code), VGPR defs only in the generated range or
    the marker outputs, SGPR defs only in the clobbered set, no exec / m0 writes, LDS read bases and offsets, one
    s_barrier_signal per TILE, MSB mode 0 at every region end, no WMMA / TRANS in the shell's code;
 7. --ref: also build the HIP reference variant (HKGEN_REF=1) as DIR/ref/k_mla_fwd.hsaco.
Writes DIR/build.json (recipe, tool versions, sha256s, register map, check results). Exit 1 on any failed check."""

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
HIP = PKG / "hip"
sys.path.insert(0, str(HERE))
import fwdgen as G  # noqa: E402

FLAGS = [
    "-x",
    "hip",
    "--offload-arch=gfx1250",
    "--cuda-device-only",
    "--no-gpu-bundle-output",
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-I.",
    "-save-temps=obj",
]
EXTRA = ["-mllvm", "-amdgpu-expert-scheduling-mode"]
KERNEL = "k_mla_fwd"
SRC = "k_mla_fwd.hip"

MARK = re.compile(r";\s*HKGEN\s+(\w+)\s*(.*)$")
ALLOWED = (
    "v_wmma_f32_16x16x32_bf16",
    "ds_load_b128",
    "ds_load_tr16_b128",
    "v_max3_num_f32",
    "v_max_num_f32",
    "v_permlanex16_b32",
    "v_fma_f32",
    "v_mul_f32",
    "v_cmp_lt_f32",
    "v_cmp_gt_f32",
    "v_cmp_gt_i32",
    "v_cndmask_b32",
    "v_sub_f32",
    "v_exp_f32",
    "v_mov_b32",
    "v_pk_fma_f32",
    "v_cvt_pk_bf16_f32",
    "v_pk_mul_f32",
    "v_pk_add_f32",
    "v_add_f32",
    "v_add_nc_u32",
    "v_div_scale_f32",
    "v_rcp_f32",
    "v_fmac_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_ldexp_f32",
    "v_log_f32",
    "s_mov_b32",
    "s_cmp_lg_u32",
    "s_cselect_b32",
    "s_barrier_signal",
    "s_wait_dscnt",
    "s_wait_tensorcnt",
    "s_wait_alu",
    "s_wait_xcnt",
    "s_set_vgpr_msb",
    "v_nop",
    "s_nop",
)


def log(*a):
    print(*a, flush=True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compile_variant(cxx, out_dir, ref, diag=0):
    """Compile the shell (ref=0) or the HIP reference (ref=1) into out_dir; returns (compile argv, .s path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    cmd = (
        [cxx]
        + FLAGS
        + EXTRA
        + [
            f"-cuid={KERNEL}",
            f"-DHKGEN_REF={ref}",
            f"-DHKGEN_DIAG={diag}",
            "-c",
            str(HIP / SRC),
            "-o",
            str(out_dir / f"{KERNEL}.cc.hsaco"),
        ]
    )
    # -I. resolves against the cwd: compile from hip/ so hkf.h is found; the temporaries land in the cwd
    r = subprocess.run(cmd, cwd=HIP, capture_output=True, text=True)
    for d in (HIP, out_dir):
        for f in d.glob(f"{KERNEL}-hip-*"):
            shutil.move(str(f), str(tmp / f.name))
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    s = next(tmp.glob("*-gfx1250.s"))
    return cmd, s


def parse_markers(lines):
    """[(name, {key: operand text}, line index of the marker comment, asmstart index, asmend index)]."""
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


def vfirst(t):
    m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", t.strip())
    if not m:
        raise ValueError(f"not a VGPR operand: {t}")
    return int(m.group(1) or m.group(2))


def vcount(t):
    m = re.fullmatch(r"v\[(\d+):(\d+)\]", t.strip())
    return int(m.group(2)) - int(m.group(1)) + 1 if m else 1


def shell_vgprs(lines, markers):
    """Highest VGPR any shell instruction references (marker bodies excluded), and the WMMA / TRANS lines."""
    from asmkit.isa import parse_line

    skip = {k for _, _, i, _, _ in markers for k in (i,)}
    hi, bad = -1, []
    for i, l in enumerate(lines):
        if i in skip:
            continue
        x = parse_line(i, l)
        if not x.mn or x.cls in ("LABEL", "DIR"):
            continue
        for r in x.defs + x.uses:
            hi = max(hi, r)
        if x.cls in ("XDL", "TRANS"):
            bad.append((i + 1, l.strip()))
    return hi, bad


def exec_narrowed_before(lines, idx):
    """True if an exec-narrowing instruction precedes line idx in its block without a later restore (heuristic for
    'the marker runs with all lanes'): scan back to the previous label."""
    j = idx - 1
    while j >= 0 and not re.match(r"^[.\w$]+:", lines[j]):
        t = lines[j].split(";")[0].strip()
        if re.match(r"s_or_b32\s+exec_lo,\s*exec_lo", t) or re.match(r"s_mov_b32\s+exec_lo,\s*-1", t):
            return False
        if re.match(
            r"(v_cmpx_|s_and_saveexec|s_and_b32\s+exec_lo|s_andn2_b32\s+exec_lo|s_mov_b32\s+exec_lo)", t
        ):
            return True
        j -= 1
    return False


def clobber_check(lines, markers):
    """The shell keeps nothing in the clobbered scalars across a marker: s90..s103 are never referenced by its code,
    and after every marker vcc / scc are written before they are read (scan to the end of the block)."""
    from asmkit.isa import parse_line, sgpr_defs_uses

    bad = []
    for i, line in enumerate(lines):
        x = parse_line(i, line)
        if not x.mn or x.cls in ("LABEL", "DIR") or MARK.search(line):
            continue
        sd, su = sgpr_defs_uses(x)
        hit = {r for r in sd + su if r.startswith("s") and r[1:].isdigit() and 90 <= int(r[1:]) <= 103}
        if hit:
            bad.append(f"line {i + 1} uses clobbered {sorted(hit)}: {line.strip()}")
    for name, _, mi, _, _ in markers:
        live = {"vcc", "scc"}
        j = mi + 2
        while j < len(lines) and live and not re.match(r"^[.\w$]+:", lines[j]) and not MARK.search(lines[j]):
            x = parse_line(j, lines[j])
            if x.mn and x.cls not in ("LABEL", "DIR"):
                sd, su = sgpr_defs_uses(x)
                if x.mn.startswith(
                    "v_cmp"
                ):  # its first scalar operand is the destination (asmkit lists it in both)
                    su = [r for r in su if r not in sd]
                if live & set(su):
                    bad.append(
                        f"{name}: {sorted(live & set(su))} read at line {j + 1} before a write: {lines[j].strip()}"
                    )
                    break
                live -= set(sd)
            j += 1
    return bad


def emit_region(texts, fmap, M, E, check_walk):
    """asmkit.emit on one straight-line region from an unknown entry state; returns the lines and the open hazards
    found by re-walking the result (must be empty)."""
    items = [M.make_item("\t" + t) for t in texts]
    for k, it in enumerate(items):
        it.pos = k
    paths = [M.Path(E.State(unknown=True), M.XState(True), [])]
    (lines,) = M.emit_region(
        [{"items": items, "succ": []}], fmap, None, None, expert=True, header=0, entry_paths=paths
    )
    return lines, check_walk(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sched", default="list", choices=("list", "program"))
    ap.add_argument("--asmkit-root", default=None, help="directory that contains the asmkit package")
    ap.add_argument("--cxx", default="amdclang++")
    ap.add_argument(
        "--diag", action="store_true", help="specialized diagonal-tile bodies TILE_D0..3 (Sq == Skv)"
    )
    ap.add_argument(
        "--no-ref",
        action="store_true",
        help="skip the HIP reference variant (its .s also feeds the MSB field map: DS operands)",
    )
    a = ap.parse_args()
    if a.asmkit_root:
        sys.path.insert(0, a.asmkit_root)
    else:
        sys.path.insert(0, str(PKG.parent))
    from asmkit import build as B
    from asmkit import emit as M
    from asmkit import expert as E
    from asmkit import splice as SP
    from asmkit import verify_obj as VO
    from asmkit.isa import coverage_errors, parse, parse_line

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = {"recipe": {}, "checks": {}, "fail": []}
    fail = rec["fail"]
    ver = subprocess.run([a.cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    rec["recipe"]["toolchain"] = ver

    # 1. the shell
    diag = 1 if a.diag else 0
    rec["recipe"]["variant"] = {"sched": a.sched, "diag": diag}
    cmd, s_path = compile_variant(a.cxx, out / "shell", 0, diag)
    rec["recipe"]["compile"] = cmd
    shell = s_path.read_text().rstrip("\n").split("\n")
    (out / "shell.s").write_text("\n".join(shell) + "\n")
    rec["recipe"]["shell_s_sha256"] = sha256(out / "shell.s")
    fmap = M.learn_msb_map(parse(shell))
    ref_s = None
    if not a.no_ref:
        cmd_r, ref_s = compile_variant(a.cxx, out / "ref", 1, diag)
        shutil.copy(out / "ref" / f"{KERNEL}.cc.hsaco", out / "ref" / f"{KERNEL}.hsaco")
        rec["recipe"]["compile_ref"] = cmd_r
        rec["ref_hsaco_sha256"] = sha256(out / "ref" / f"{KERNEL}.hsaco")
        fmap = M.merge_msb_maps(fmap, M.learn_msb_map(parse(ref_s.read_text().split("\n"))))
    # MSB fields by operand (LLVM AMDGPUBaseInfo getVGPRLoweringOperandTables: VOP* (src0, src1, src2, vdst), DS
    # (addr, data0, data1, vdst)); pinned where the compiler's outputs leave them ambiguous, and the default for VALU
    # operands they never showed (text operand k -> dst, src0, src1, src2)
    fields = ("dst", "src0", "src1", "src2")
    for mn in ("ds_load_b128", "ds_load_tr16_b128"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    for p, f in ((0, "dst"), (2, "src0"), (3, "src1"), (4, "src2")):
        fmap[("v_div_scale_f32", p)] = {f}

    # 2. markers and the register map
    markers = parse_markers(shell)
    names = [m[0] for m in markers]
    want = (
        ["INIT", "TILE_C"]
        + (["TILE_D0", "TILE_D1", "TILE_D2", "TILE_D3"] if diag else ["TILE_M"])
        + ["FINAL_0", "FINAL_1", "FINAL_2", "FINAL_3"]
    )
    if sorted(names) != sorted(want):
        raise RuntimeError(f"markers {names}, want {want}")
    hi, xdl_lines = shell_vgprs(shell, markers)
    base = hi + 1
    for name, ops, i, _, _ in markers:
        for k, v in ops.items():
            if v.startswith("v"):
                base = max(base, vfirst(v) + vcount(v))
    L = G.Layout(base)
    rec["layout"] = {
        "base": base,
        "block0_end": L.end0,
        "m": L.m,
        "d": L.d,
        "cc": L.cc,
        "qa": L.qa,
        "qb": L.qb,
        "O": [256, 511],
        "V": [512, 639],
        "S": [640, 767],
        "K": [768, 959],
        "PR": [960, 1023],
        "qbuf": L.qbuf,
    }
    rec["markers"] = {m[0]: m[1] for m in markers}
    if xdl_lines:
        fail.append(f"shell code has WMMA / TRANS instructions: {xdl_lines[:3]}")
    for name, ops, i, _, _ in markers:
        if exec_narrowed_before(shell, i - 1):
            fail.append(f"marker {name} may run with a narrowed exec")
    cb = clobber_check(shell, markers)
    rec["checks"]["clobbers"] = cb or "ok"
    fail += cb
    log(f"shell: highest VGPR v{hi}, generated base v{base}, block 0 up to v{L.end0 - 1}")

    def walk_check(lines):
        ins = parse(["\t" + l.strip() if not l.startswith("\t") else l for l in lines])
        fl = M.Flow([E.State(unknown=True)], [M.XState(True)], [[]])
        found = []
        SP.walk_block(
            ins, 0, len(ins) - 1, fl, True, lambda k, f, n: found.append((k, f, ins[k].text.strip()))
        )
        cov = [(k, e) for k, x in enumerate(ins) for e in coverage_errors(x)]
        return found + [("coverage", c) for c in cov]

    # 3. regions
    regions, models = {}, {}
    for name, ops, i, _, _ in markers:
        if name == "INIT":
            texts = G.region_init(L, {"qa": vfirst(ops["qa"]), "qb": vfirst(ops["qb"]), "c2": ops["c2"]})
            pre = []
        elif name.startswith("TILE"):
            o = {"kb": vfirst(ops["kb"]), "vb": vfirst(ops["vb"])}
            if "dl" in ops:
                o["dl"] = vfirst(ops["dl"])
            if name.startswith("TILE_D"):
                k = int(name[-1])
                texts, cyc = G.region_tile(L, o, True, a.sched, first=k, mask_only=k)
            else:
                texts, cyc = G.region_tile(L, o, name == "TILE_M", a.sched)
            models[name] = cyc
            pre = []
        else:
            if vcount(ops["o"]) != 32 or vcount(ops["lse"]) != 1:
                raise RuntimeError(f"{name}: unexpected output operands {ops}")
            texts = G.region_final(L, {"o": vfirst(ops["o"]), "lse": vfirst(ops["lse"])}, int(name[-1]))
            # in-flight WMMAs of the last tile write O: retire every VALU first, then the WMMA -> VALU wait states
            pre = ["\ts_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"] + ["\tv_nop"] * 4
        for t in texts:
            x = parse_line(0, "\t" + t)
            for p, o in enumerate(x.ops):
                key = (x.mn, p)
                if key not in fmap and x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS") and p < 4:
                    fmap[key] = {fields[p]}
        lines, open_ = emit_region(texts, fmap, M, E, walk_check)
        if open_:
            fail.append(f"{name}: {len(open_)} hazards / coverage errors after emission, e.g. {open_[:3]}")
        regions[name] = pre + lines
        log(
            f"{name}: {len(texts)} instructions -> {len(pre) + len(lines)} lines"
            + (f", modelled {models[name]} cycles" if name in models else "")
        )
    rec["model_cycles"] = models

    # 4. splice (last marker first: indices stay valid)
    new = list(shell)
    for name, ops, i, s0, s1 in sorted(markers, key=lambda m: -m[2]):
        body = [new[i]] + regions[name]
        new[s0 + 1 : s1] = body
    new = SP.set_vgpr_count(new, 1024)
    spl = out / f"{KERNEL}.s"
    spl.write_text("\n".join(new) + "\n")

    # 5. assemble + verify_obj
    cc1as, lld = B.driver_steps(cmd, str(HIP))
    rec["recipe"]["cc1as"] = cc1as
    rec["recipe"]["lld"] = lld
    meta = B.assemble(spl, out / f"{KERNEL}.hsaco", cc1as, lld, cwd=str(HIP))
    rec["hsaco"] = meta
    log(f"hsaco: {meta}")
    bad = VO.compare(str(spl), str(out / f"{KERNEL}.objdump.s"), log=log)
    rec["checks"]["verify_obj"] = "ok" if not bad else [str(b) for b in bad]
    if bad:
        fail.append(f"verify_obj: {bad[:3]}")
    if (
        meta.get("vgpr_spill_count", 0)
        or meta.get("sgpr_spill_count", 0)
        or meta.get("private_segment_fixed_size", 0)
    ):
        fail.append(f"spills / scratch: {meta}")

    # 6. contract checks on the listing (regions by instruction ordinal)
    rec["checks"]["contract"] = contract_checks(spl, out / f"{KERNEL}.objdump.s", markers, L, fail)

    for f in fail:
        log("FAIL:", f)
    rec["verdict"] = "PASS" if not fail else "FAIL"
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    log(f"VERDICT {rec['verdict']}  {out / (KERNEL + '.hsaco')}  sha256 {meta['sha256']}")
    return 0 if not fail else 1


def contract_checks(spl, objdump, markers, L, fail):
    """Check the generated regions in the code object's listing."""
    from asmkit.isa import load_any, parse_line, sgpr_defs_uses

    s_lines = spl.read_text().split("\n")
    # ordinals of the instructions of each region in the intended .s (same filter as verify_obj)
    mnre = re.compile(r"^[a-z][a-z0-9_]*(::[a-z][a-z0-9_]*)?$")
    ins_s = [parse_line(i, l) for i, l in enumerate(s_lines)]
    start = next(i for i, x in enumerate(ins_s) if x.label == KERNEL)
    ordn, n, region_of, cur = {}, 0, {}, None
    for i in range(start, len(ins_s)):
        x = ins_s[i]
        m = MARK.search(s_lines[i])
        if m:
            cur = m.group(1)
        if ";;#ASMEND" in s_lines[i]:
            cur = None
        if x.mn and x.cls not in ("LABEL", "DIR", "CLAUSE") and mnre.match(x.mn) and x.mn != "s_code_end":
            ordn[i] = n
            if cur:
                region_of[n] = cur
            n += 1
        if s_lines[i].strip().startswith(".Lfunc_end"):
            break
    o_lines, o_ins, _ = load_any(objdump)
    seq = [
        x
        for x in o_ins
        if x.mn and x.cls not in ("LABEL", "DIR", "CLAUSE") and mnre.match(x.mn) and x.mn != "s_code_end"
    ]
    ops_of = {m[0]: m[1] for m in markers}
    out_regs = {}
    for name, ops, *_ in markers:
        if name.startswith("FINAL"):
            out_regs[name] = set(range(vfirst(ops["o"]), vfirst(ops["o"]) + 32)) | {vfirst(ops["lse"])}
    gen_regs = L.all_regs()
    stats = {}
    msb_mode = {}
    for k, x in enumerate(seq):
        name = region_of.get(k)
        if name is None:
            continue
        st = stats.setdefault(name, {"insts": 0, "signal": 0, "ds": 0, "wmma": 0})
        st["insts"] += 1
        mn = x.mn.replace("_e32", "").replace("_e64", "")
        if not mn.startswith(ALLOWED) and mn not in ALLOWED:
            fail.append(f"{name}: instruction outside the whitelist: {x.text.strip()}")
        if mn == "s_set_vgpr_msb":
            msb_mode[name] = int(x.ops[0], 0) & 0xFF
        allowed = gen_regs | out_regs.get(name, set())
        if set(x.defs) - allowed:
            fail.append(f"{name}: writes VGPRs outside the generated range: {x.text.strip()}")
        sd, su = sgpr_defs_uses(x)
        if set(sd) - G.CLOBBERED - {"scc"} and mn not in ("s_barrier_signal",):
            fail.append(f"{name}: writes scalars outside the clobbered set: {sd} {x.text.strip()}")
        if x.mn.startswith("ds_"):
            st["ds"] += 1
            ok_base = {L.qa, L.qb}
            if name.startswith("TILE"):
                ok_base |= {vfirst(ops_of[name]["kb"]), vfirst(ops_of[name]["vb"])}
            if not set(x.uses) <= ok_base:
                fail.append(f"{name}: LDS read from an unexpected base: {x.text.strip()}")
            mo = re.search(r"offset:(\d+)", x.text)
            off = int(mo.group(1)) if mo else 0
            b = x.uses[0] if x.uses else None
            lim = 25952 if b in (L.qa, L.qb) else (13280 if x.mn == "ds_load_tr16_b128" else 19552)
            if off > lim or off % 16:
                fail.append(f"{name}: LDS offset {off} out of the layout: {x.text.strip()}")
        if x.mn == "s_barrier_signal":
            st["signal"] += 1
        if x.cls == "XDL":
            st["wmma"] += 1
    for name in ops_of:
        st = stats.get(name, {})
        want = 1 if name.startswith("TILE") else 0
        if st.get("signal", 0) != want:
            fail.append(f"{name}: {st.get('signal', 0)} s_barrier_signal, want {want}")
        if msb_mode.get(name, 0) != 0:
            fail.append(f"{name}: MSB mode {msb_mode[name]:#x} at the region end")
    return stats


if __name__ == "__main__":
    sys.exit(main())
