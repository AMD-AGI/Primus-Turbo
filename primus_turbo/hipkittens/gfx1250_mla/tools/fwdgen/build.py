###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build the k_mla_fwd code object from its marker shell and the regions fwdgen generates.

    python3 build.py --out DIR [--cxx CXX]

CPU only. Steps:
 1. compile the shell exactly as hsaco.py does (DIR/shell.s is the compiler's .s) and the HIP reference
    (-DHKGEN_REF=1); the VGPR MSB operand fields are learned from both;
 2. read the HKGEN markers (inline asm) and the registers LLVM gave their operands; the generated code lives
    at and above 1 + the highest VGPR the shell's code or a marker operand references;
 3. generate every region (fwdgen) and let asmkit.emit derive its s_set_vgpr_msb, expert-mode waits and hazard
    v_nops (va_vdst: LLVM's rule, but the per-class bound for a write that only an in-flight WMMA still
    reads, as the backward's); a RESC node (O *= corr under a branch) splits a region into blocks, each
    emitted from the joined flows of its predecessors; a re-walk of the result must find no open hazard and
    complete parse coverage;
 4. splice the regions in place of the marker bodies, raise the VGPR allocation to 1024: DIR/k_mla_fwd.s;
 5. assemble with the clang driver's own cc1as / lld commands: DIR/k_mla_fwd.hsaco (+ .notes, .objdump.s); the
    code object must hold the intended instructions (asmkit.verify_obj) and use no spills or scratch;
 6. check the contract with the shell on the code object's listing (contract_checks), and that the shell's
    code has no WMMA / TRANS, keeps nothing in the clobbered scalars across a marker and reaches no marker
    with exec narrowed;
 7. check the shell's epilogue stores (writeback_check).
Writes DIR/build.json (recipe, sha256s, register map, check results); exit status 1 on any failed check."""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIP = HERE.parents[1] / "hip"
sys.path[:0] = [str(HERE), str(HERE.parent)]
import fwdgen as G  # noqa: E402
from asmkit import build as B  # noqa: E402
from asmkit import emit as M  # noqa: E402
from asmkit import expert as E  # noqa: E402
from asmkit import verify_obj as VO  # noqa: E402
from asmkit.isa import coverage_errors, load_any, parse, parse_line, regs_of, sgpr_defs_uses  # noqa: E402

KERNEL = "k_mla_fwd"
MARKERS = ("INIT", "TILE_C", *(f"TILE_D{k}" for k in range(4)), *(f"FINAL_{i}" for i in range(4)))
MARK = re.compile(r";\s*HKGEN\s+(\w+)\s*(.*)$")
LABEL = re.compile(r"^[.\w$]+:")
EXEC_RESTORE = re.compile(r"s_or_b32\s+exec_lo,\s*exec_lo|s_mov_b32\s+exec_lo,\s*-1")
EXEC_NARROW = re.compile(r"v_cmpx_|s_and_saveexec|s_and(n2)?_b32\s+exec_lo|s_mov_b32\s+exec_lo")
EXEC_M0 = {"exec", "exec_lo", "exec_hi", "m0"}
# the mnemonics generated code may contain (prefixes; no global / buffer / flat / scratch / SMEM / TDM access)
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
    "s_cbranch_scc0",
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


def _hsaco():
    spec = importlib.util.spec_from_file_location("_gfx1250_mla_hsaco", HIP.parent / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_variant(cxx, out_dir, ref):
    """Compile the shell (ref False: exactly hsaco.py's command) or the HIP reference (ref True: -DHKGEN_REF=1
    instead of the marker options) into a fresh out_dir; returns (argv, the compiler's .s)."""
    hsaco = _hsaco()
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    extra = list(hsaco.load_manifest()["kernels"][KERNEL]["extra_flags"])
    if ref:
        extra = [f for f in extra if not f.startswith("-DHKGEN_REF=")] + ["-DHKGEN_REF=1"]
    cmd = hsaco.compile_command(cxx, KERNEL, out_dir / f"{KERNEL}.cc.hsaco", extra)
    r = subprocess.run(cmd, cwd=HIP, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    (s,) = out_dir.glob("*-gfx1250.s")
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
    """First VGPR of a marker operand (v5, v[8:39])."""
    m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", t.strip())
    if not m:
        raise ValueError(f"not a VGPR operand: {t}")
    return int(m.group(1) or m.group(2))


def vcount(t):
    m = re.fullmatch(r"v\[(\d+):(\d+)\]", t.strip())
    return int(m.group(2)) - int(m.group(1)) + 1 if m else 1


def scan_shell(ins):
    """The shell's code: (highest VGPR referenced, WMMA / TRANS lines, scalar registers referenced)."""
    hi, bad, scalars = -1, [], set()
    for x in ins:
        if not x.mn or x.cls in ("LABEL", "DIR"):
            continue
        hi = max([hi, *x.defs, *x.uses])
        if x.cls in ("XDL", "TRANS"):
            bad.append((x.li + 1, x.text.strip()))
        sd, su = sgpr_defs_uses(x)
        scalars |= set(sd) | set(su)
    return hi, bad, scalars


def exec_narrowed_before(lines, idx):
    """True if an exec-narrowing instruction precedes line idx in its block with no restore after it (the
    markers must run with all lanes): scan back to the previous label."""
    j = idx - 1
    while j >= 0 and not LABEL.match(lines[j]):
        t = lines[j].split(";")[0].strip()
        if EXEC_RESTORE.match(t):
            return False
        if EXEC_NARROW.match(t):
            return True
        j -= 1
    return False


def clobber_check(lines, ins, markers):
    """The shell keeps nothing in the clobbered scalars across a marker: its code never references s90..s103,
    and after every marker it writes vcc / scc before reading them (scan to the end of the block)."""
    bad = []
    s_clobbered = G.CLOBBERED - {"vcc", "scc"}
    for x in ins:
        if not x.mn or x.cls in ("LABEL", "DIR"):
            continue
        sd, su = sgpr_defs_uses(x)
        hit = s_clobbered & set(sd + su)
        if hit:
            bad.append(f"line {x.li + 1} uses clobbered {sorted(hit)}: {x.text.strip()}")
    for name, _, mi, _, _ in markers:
        live = {"vcc", "scc"}
        j = mi + 2
        while j < len(lines) and live and not LABEL.match(lines[j]) and not MARK.search(lines[j]):
            x = ins[j]
            if x.mn and x.cls not in ("LABEL", "DIR"):
                sd, su = sgpr_defs_uses(x)
                if x.mn.startswith("v_cmp"):  # asmkit lists the scalar destination as a use too
                    su = [r for r in su if r not in sd]
                read = sorted(live & set(su))
                if read:
                    bad.append(f"{name}: {read} read at line {j + 1} before a write: {x.text.strip()}")
                    break
                live -= set(sd)
            j += 1
    return bad


def msb_field_map(*texts):
    """{(mnemonic, operand index): {MSB field}} learned from the compiler's .s texts (the HIP reference has
    the operands in the high VGPR blocks), pinned where they leave it ambiguous. Fields by operand follow
    LLVM's getVGPRLoweringOperandTables: VOP (src0, src1, src2, vdst), DS (addr, data0, data1, vdst)."""
    fmap = M.merge_msb_maps(*(M.learn_msb_map(parse(t)) for t in texts))
    for mn in ("ds_load_b128", "ds_load_tr16_b128"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    for p, f in ((0, "dst"), (2, "src0"), (3, "src1"), (4, "src2")):  # operand 1: the scalar carry-out
        fmap[("v_div_scale_f32", p)] = {f}
    return fmap


def msb_defaults(fmap, texts):
    """VALU operand positions the compiler's output never showed: text operand k -> dst, src0, src1, src2."""
    for t in texts:
        x = parse_line(0, "\t" + t)
        if x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS"):
            for p in range(min(len(x.ops), 4)):
                fmap.setdefault((x.mn, p), {("dst", "src0", "src1", "src2")[p]})


def xcnt_entry(scalars):
    """XCNT state at a region entry: known, holding every scalar register the shell's code references. The
    regions issue no VMEM / FLAT / TDM op, so the memory ops in flight at a region entry are the shell's and
    their scalar sources are among these registers; the emitter waits for XCNT before a region write to any of
    them. The SGPRs the regions write, s90..s103, are not among them (clobber_check)."""
    xs = M.XState(False)
    xs.regs = set(scalars)
    return xs


def walk_block_check(lines, flow):
    """Open hazards and parse-coverage errors of one emitted block walked from its entry flow."""
    ins = parse([l if l.startswith("\t") else "\t" + l.strip() for l in lines])
    fl = flow.copy()
    found = []
    M.walk_block(ins, fl, lambda k, f, n: found.append((k, f, ins[k].text.strip())))
    return found + [("coverage", (k, e)) for k, x in enumerate(ins) for e in coverage_errors(x)]


RESC_LABELS = [0]  # unique labels of the RESC blocks across the regions


def emit_region(texts, fmap, scalars, L):
    """asmkit.emit on one region: (lines, what the re-walk finds in them, which must be nothing). Entry
    state: anything may be in flight (unknown), XCNT as xcnt_entry, no VALU history. A RESC pseudo-line splits
    the region: the block before it ends in s_cmp + s_cbranch over the conditional O *= corr block, and the
    block after it is entered from both (joined flows)."""
    segs, cur = [], []
    for x in texts:
        if x.startswith("HKRESC"):
            segs.append((cur, x))
            cur = []
        else:
            cur.append(x)
    flow = M.Flow([E.State(unknown=True)], [xcnt_entry(scalars)], [[]])

    def emit(tx, fl):
        items = [M.make_item("\t" + x) for x in tx]
        for k, it in enumerate(items):
            it.pos = k
        lines, ex = M.emit_block(items, fl, fmap, va_rule="llvm_war_xdl")
        return lines, ex, walk_block_check(lines, fl)

    out, found, lead = [], [], []
    for tx, rline in segs:
        label = f".LHKR{RESC_LABELS[0]}"
        RESC_LABELS[0] += 1
        head, body, lab = G.resc_block(L, rline, label)
        lines, ex_a, f1 = emit(tx + head, flow)
        lines_b, ex_b, f2 = emit(body, ex_a)
        out += lead + lines + lines_b
        lead = [lab]
        found += f1 + f2
        flow = M.Flow.join([ex_a, ex_b])
    lines, _, f3 = emit(cur, flow)
    return out + lead + lines, found + f3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cxx", default="amdclang++", help="compiler (the manifest's toolchain)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = {"recipe": {}, "checks": {}, "fail": []}
    fail = rec["fail"]
    version = subprocess.run([a.cxx, "--version"], capture_output=True, text=True).stdout
    rec["recipe"]["toolchain"] = version.splitlines()[0]

    # 1. the shell and the HIP reference
    cmd, s_path = compile_variant(a.cxx, out / "shell", ref=False)
    rec["recipe"]["compile"] = cmd
    shell = s_path.read_text().rstrip("\n").split("\n")
    (out / "shell.s").write_text("\n".join(shell) + "\n")
    rec["recipe"]["shell_s_sha256"] = sha256(out / "shell.s")
    cmd_ref, ref_path = compile_variant(a.cxx, out / "ref", ref=True)
    rec["recipe"]["compile_ref"] = cmd_ref
    fmap = msb_field_map(shell, ref_path.read_text().rstrip("\n").split("\n"))

    # 2. markers and the register map
    ins = parse(shell)
    markers = parse_markers(shell)
    names = [m[0] for m in markers]
    if sorted(names) != sorted(MARKERS):
        raise RuntimeError(f"markers {names}, want {list(MARKERS)}")
    hi, wmma_trans, scalars = scan_shell(ins)
    ends = [vfirst(v) + vcount(v) for _, ops, *_ in markers for v in ops.values() if v.startswith("v")]
    base = max([hi + 1, *ends])
    L = G.Layout(base)
    rec["layout"] = {
        "base": base,
        "block0_end": L.end0,
        "m": L.m,
        "d": L.d,
        "cc": L.cc,
        "qa": L.qa,
        "qb": L.qb,
        "qbuf": L.qbuf,
        "O": [L.o, L.o + 255],
        "V": [[L.V(0, 0), L.V(3, 1) + 7], [L.V(4, 0), L.V(7, 1) + 7]],
        "S": [L.s, L.s + 127],
        "K": [L.k, L.k + 191],
        "PR": [L.pr, L.pr + 63],
    }
    rec["markers"] = {m[0]: m[1] for m in markers}
    if wmma_trans:
        fail.append(f"shell code has WMMA / TRANS instructions: {wmma_trans[:3]}")
    for name, _, i, _, _ in markers:
        if exec_narrowed_before(shell, i - 1):
            fail.append(f"marker {name} may run with a narrowed exec")
    cb = clobber_check(shell, ins, markers)
    rec["checks"]["clobbers"] = cb or "ok"
    fail += cb
    log(f"shell: highest VGPR v{hi}, generated base v{base}, block 0 up to v{L.end0 - 1}")

    # 3. regions
    regions, cycles = {}, {}
    for name, ops, *_ in markers:
        pre, post = [], []
        if name == "INIT":
            texts = G.region_init(L, {"qa": vfirst(ops["qa"]), "qb": vfirst(ops["qb"]), "c2": ops["c2"]})
        elif name.startswith("TILE"):
            c2 = ops.get("c2", "")
            if not re.fullmatch(r"s\d+", c2) or c2 in G.CLOBBERED:
                raise RuntimeError(f"{name}: c2 operand {c2!r} is not a plain, unclobbered SGPR")
            o = {"kb": vfirst(ops["kb"]), "vb": vfirst(ops["vb"]), "c2": c2}
            k = None
            if name.startswith("TILE_D"):
                k, o["dl"] = int(name[-1]), vfirst(ops["dl"])
            texts, cycles[name] = G.region_tile(L, o, k)
        else:
            if vcount(ops["o"]) != 32 or vcount(ops["lse"]) != 1:
                raise RuntimeError(f"{name}: unexpected output operands {ops}")
            texts = G.region_final(L, {"o": vfirst(ops["o"]), "lse": vfirst(ops["lse"])}, int(name[-1]))
            # WMMAs of the preceding region may still be in flight: retire every VALU, then the WMMA -> VALU
            # wait states
            pre = ["\ts_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"] + ["\tv_nop"] * 4
            # the shell's LDS stores read the outputs, and LLVM puts no va_vdst wait after an inline-asm def
            post = ["\ts_wait_alu depctr_va_vdst(0)"]
        msb_defaults(fmap, [x for x in texts if not x.startswith("HKRESC")] + G.resc_lines(L, 0))
        lines, open_ = emit_region(texts, fmap, scalars, L)
        if open_:
            fail.append(f"{name}: {len(open_)} hazards / coverage errors after emission, e.g. {open_[:3]}")
        regions[name] = pre + lines + post
        log(
            f"{name}: {len(texts)} instructions -> {len(pre) + len(lines)} lines"
            + (f", modelled {cycles[name]} cycles" if name in cycles else "")
        )
    rec["model_cycles"] = cycles

    # 4. splice (last marker first: the indices stay valid)
    new = list(shell)
    for name, _, i, s0, s1 in sorted(markers, key=lambda m: -m[2]):
        new[s0 + 1 : s1] = [new[i]] + regions[name]
    new = B.set_vgpr_count(new, 1024)
    spl = out / f"{KERNEL}.s"
    spl.write_text("\n".join(new) + "\n")

    # 5. assemble, verify_obj, spills
    cc1as, lld = B.driver_steps(cmd, str(HIP))
    rec["recipe"]["cc1as"], rec["recipe"]["lld"] = cc1as, lld
    meta = B.assemble(spl, out / f"{KERNEL}.hsaco", cc1as, lld, cwd=str(HIP))
    rec["hsaco"] = meta
    log(f"hsaco: {meta}")
    objdump = out / f"{KERNEL}.objdump.s"
    bad = VO.compare(str(spl), str(objdump), log=log)
    rec["checks"]["verify_obj"] = [str(b) for b in bad] or "ok"
    if bad:
        fail.append(f"verify_obj: {bad[:3]}")
    if any(meta.get(k, 0) for k in ("vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size")):
        fail.append(f"spills / scratch: {meta}")

    # 6., 7. the listing
    rec["checks"]["contract"] = contract_checks(spl, objdump, markers, L, fail)
    rec["checks"]["writeback"] = writeback_check(objdump, fail)

    for f in fail:
        log("FAIL:", f)
    rec["verdict"] = "PASS" if not fail else "FAIL"
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    log(f"VERDICT {rec['verdict']}  {out / (KERNEL + '.hsaco')}  sha256 {meta['sha256']}")
    return 0 if not fail else 1


def writeback_check(objdump, fail):
    """The epilogue's buffer stores, on the code object's listing. A store phase runs from a VMEM store to the
    next branch or s_endpgm (both drain XCNT: LLVM's SIInstrInfo::isXcntDrain). The block's stores must form
    exactly one phase, and in it:
      * no other VMEM / FLAT op, no EXEC / M0 write, no s_set_vgpr_msb (the shell's code references VGPRs
        below 256 only and never switches the MSB mode, so one here would mean a generated region inside the
        phase, which must come after the last FINAL region);
      * no write of a register an earlier store of the phase reads (data, address VGPR, resource / offset
        SGPRs) unless an s_wait_xcnt N has retired that store (XCNT retires in order: at most N younger
        stores are pending); `early_waits` counts the writes other than data reloads (ds_load) that needed
        such a wait;
      * between the last s_barrier_wait before the phase and its first store, every ds_store (the staging of
        the FINAL outputs) precedes every ds_load (their read-back).
    Returns per-phase statistics; failures go to `fail`."""
    _, ins, _ = load_any(objdump)
    seq = [x for x in ins if x.mn and x.cls not in ("LABEL", "DIR")]

    def is_store(x):
        return x.mn.startswith(("buffer_store", "global_store"))

    def is_end(x):
        return x.mn == "s_endpgm" or x.mn.startswith(("s_branch", "s_cbranch", "s_setpc", "s_swappc"))

    res = {"phases": [], "stores": 0, "early_waits": 0}
    stores_at = [k for k, x in enumerate(seq) if is_store(x)]
    if not stores_at:
        fail.append("writeback: no store phase")
        return res
    k = stores_at[0]
    while k is not None:
        stores = []  # (data VGPRs, other operand registers) of the phase's stores, oldest first
        retired = 0
        ph = {"first": k, "stores": 0, "xcnt_waits": []}
        while not is_end(seq[k]):
            x = seq[k]
            t, mn = x.text.strip(), x.mn
            if mn.startswith("s_set_vgpr_msb"):
                fail.append(f"writeback: {t} inside a store phase")
            sd, su = sgpr_defs_uses(x)
            dst = t.split(",")[0].split()[-1] if t else ""
            if set(sd) & EXEC_M0 or "saveexec" in mn or re.fullmatch(r"exec(_lo|_hi)?|m0", dst):
                fail.append(f"writeback: EXEC / M0 write inside a store phase: {t}")
            if mn == "s_wait_xcnt":
                n = int(x.ops[0], 0)
                retired = max(retired, len(stores) - n)
                ph["xcnt_waits"].append(n)
            elif is_store(x):
                ops = [o.strip() for o in x.ops]
                data = {f"v{r}" for r in regs_of(ops[0])}
                other = {f"v{r}" for r in regs_of(ops[1])} if len(ops) > 1 else set()
                stores.append((data, other | {r for r in su if not r.startswith(("exec", "m0"))}))
                ph["stores"] += 1
            else:
                if mn.startswith(("buffer_", "global_", "flat_", "scratch_")):
                    fail.append(f"writeback: other VMEM op inside a store phase: {t}")
                wr = {f"v{r}" for r in x.defs} | set(sd)
                for j in range(retired, len(stores)):
                    hit = wr & (stores[j][0] | stores[j][1])
                    if hit:
                        fail.append(
                            f"writeback: rewrites {sorted(hit)} of store #{j} of the phase (translation "
                            f"pending, {len(stores) - 1 - j} younger) without an s_wait_xcnt retiring it: {t}"
                        )
                if not mn.startswith("ds_load") and any(wr & (d | o) for d, o in stores[:retired]):
                    res["early_waits"] += 1
            k += 1
        ph["end"] = seq[k].text.strip()[:40] or seq[k].mn
        res["stores"] += ph["stores"]
        res["phases"].append(ph)
        k = next((j for j in stores_at if j > k), None)
    if len(res["phases"]) != 1:
        fail.append(
            f"writeback: {len(res['phases'])} store phases (the block's stores must form one straight-line "
            f"phase ending at the back edge / s_endpgm: {[p['end'] for p in res['phases']]})"
        )
    for ph in res["phases"]:
        lastbar = max((j for j in range(ph["first"]) if seq[j].mn == "s_barrier_wait"), default=0)
        dss = [j for j in range(lastbar, ph["first"]) if seq[j].mn.startswith("ds_store")]
        dsl = [j for j in range(lastbar, ph["first"]) if seq[j].mn.startswith("ds_load")]
        ph["staging"] = {"ds_store": len(dss), "ds_load": len(dsl)}
        if dss and dsl and max(dss) > min(dsl):
            fail.append("writeback: an epilogue ds_load precedes a ds_store (staging order)")
    return res


def contract_checks(spl, objdump, markers, L, fail):
    """The generated regions on the code object's listing (found by instruction ordinal in the intended .s):
    only ALLOWED mnemonics; VGPR writes only in the generated range (L.all_regs) or the FINAL outputs; scalar
    writes only in G.CLOBBERED; LDS reads only at the marker bases plus 16-byte aligned offsets inside the
    layout; one s_barrier_signal per TILE region and no LDS read after it (the next block's Q TDM, issued
    right after the block's last barrier, overwrites the whole Q staging area); MSB mode 0 at every region
    end. Returns per-region statistics; failures go to `fail`."""
    s_lines = spl.read_text().split("\n")
    mnre = re.compile(r"^[a-z][a-z0-9_]*(::[a-z][a-z0-9_]*)?$")

    def counted(x):  # an instruction as verify_obj counts them
        return x.mn and x.cls not in ("LABEL", "DIR", "CLAUSE") and mnre.match(x.mn) and x.mn != "s_code_end"

    ins_s = parse(s_lines)
    start = next(i for i, x in enumerate(ins_s) if x.label == KERNEL)
    n, region_of, cur = 0, {}, None
    for i in range(start, len(ins_s)):
        m = MARK.search(s_lines[i])
        if m:
            cur = m.group(1)
        if ";;#ASMEND" in s_lines[i]:
            cur = None
        if counted(ins_s[i]):
            if cur:
                region_of[n] = cur
            n += 1
        if s_lines[i].strip().startswith(".Lfunc_end"):
            break
    _, o_ins, _ = load_any(objdump)
    seq = [x for x in o_ins if counted(x)]
    ops_of = {m[0]: m[1] for m in markers}
    out_regs = {
        name: set(range(vfirst(ops["o"]), vfirst(ops["o"]) + 32)) | {vfirst(ops["lse"])}
        for name, ops in ops_of.items()
        if name.startswith("FINAL")
    }
    gen_regs = L.all_regs()
    # the largest LDS offsets of the layout: Q (from qa / qb), K (kb), V^T (vb)
    q_lim = G.QOFF1 + (G.NKS - 1) * 64 + 32
    k_lim = (G.NKV - 1) * 16 * G.XK + (G.NKS - 1) * 64 + 32
    v_lim = 48 * G.XV + (G.NDT - 1) * 32
    stats, msb_mode = {}, {}
    for k, x in enumerate(seq):
        name = region_of.get(k)
        if name is None:
            continue
        st = stats.setdefault(name, {"insts": 0, "signal": 0, "ds": 0, "wmma": 0, "ds_after_signal": 0})
        st["insts"] += 1
        mn = x.mn.replace("_e32", "").replace("_e64", "")
        if not mn.startswith(ALLOWED):
            fail.append(f"{name}: instruction outside the whitelist: {x.text.strip()}")
        if mn == "s_set_vgpr_msb":
            msb_mode[name] = int(x.ops[0], 0) & 0xFF
        if set(x.defs) - gen_regs - out_regs.get(name, set()):
            fail.append(f"{name}: writes VGPRs outside the generated range: {x.text.strip()}")
        sd, _ = sgpr_defs_uses(x)
        if set(sd) - G.CLOBBERED:
            fail.append(f"{name}: writes scalars outside the clobbered set: {sd} {x.text.strip()}")
        if x.mn.startswith("ds_"):
            st["ds"] += 1
            if st["signal"]:
                st["ds_after_signal"] += 1
            ok_base = {L.qa, L.qb}
            if name.startswith("TILE"):
                ok_base |= {vfirst(ops_of[name]["kb"]), vfirst(ops_of[name]["vb"])}
            if not set(x.uses) <= ok_base:
                fail.append(f"{name}: LDS read from an unexpected base: {x.text.strip()}")
            mo = re.search(r"offset:(\d+)", x.text)
            off = int(mo.group(1)) if mo else 0
            b = x.uses[0] if x.uses else None
            lim = q_lim if b in (L.qa, L.qb) else (v_lim if x.mn == "ds_load_tr16_b128" else k_lim)
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
        if st.get("ds_after_signal", 0):
            fail.append(f"{name}: {st['ds_after_signal']} LDS reads after the s_barrier_signal")
    return stats


if __name__ == "__main__":
    sys.exit(main())
