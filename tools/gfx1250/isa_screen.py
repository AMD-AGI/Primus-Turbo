#!/usr/bin/env python3
"""Offline ISA pre-screen for the gfx1250 fused attention backward.

Compiles candidate configurations WITHOUT a GPU and reports the register and
instruction facts that decide whether a candidate is worth a benchmark slot:

    vgpr   spill_st/spill_ld   s_set_vgpr_msb   wmma   total_instr

Why this exists: of the 43 variants compiled during the 2026-09-13 session,
every one with vgpr below the 1024 cap spilled exactly zero, and every one at
the cap spilled -- up to 1814 scratch stores. That is a compile-time fact, but
each of those verdicts cost real GPU minutes to learn. This screens them first.

Runs in a CPU-only container (no /dev/kfd, no /dev/dri):

    docker run --rm -v /home/lihuzhan/code:/home/lihuzhan/code fa-tune:deps \
        python /path/to/isa_screen.py --grid default

See output/0913__opt_plan__claude/phase2/isa/ISA-FINDINGS.md.
"""
import argparse, json, re, sys, collections, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

TARGET = GPUTarget("hip", "gfx1250", 32)

# Argument name -> triton signature type. Pointer args are the tensors; the
# kernel casts strides to int64 internally when USE_INT64_STRIDES, so they are
# passed as i32 exactly as the real launch does.
PTR_BF16 = {"Q", "K", "V", "DO", "DQ", "DK", "DV", "Sink", "DSink"}
PTR_FP32 = {"M", "Delta", "Alibi_slopes", "Descale_q", "Descale_k", "Descale_v", "Descale_do"}
PTR_I32 = {"cu_seqlens_q", "cu_seqlens_k", "Dropout_mask"}
FP32 = {"sm_scale", "dropout_p"}
I64 = {"philox_offset_base_in"}


# Calibration against the cached ground-truth compilations (see ISA-FINDINGS.md):
# the runtime specializes on 16-byte divisibility for pointers and non-unit
# strides, and folds the unit innermost strides to constants. Without both, the
# offline compile produces 4x the code and 4x the transpose loads, which would
# make this screen actively misleading rather than merely approximate.
UNIT_STRIDE = re.compile(r"^stride_.*d_in$")


def signature_for(fn, constants):
    sig, params = {}, fn.params
    for p in params:
        n = p.name
        if n in constants:
            sig[n] = "constexpr"
        elif n in PTR_BF16:
            sig[n] = "*bf16"
        elif n in PTR_FP32:
            sig[n] = "*fp32"
        elif n in PTR_I32:
            sig[n] = "*i32"
        elif n in FP32:
            sig[n] = "fp32"
        elif n in I64:
            sig[n] = "i64"
        else:
            sig[n] = "i32"
    return sig


def analyse(asm):
    ins = collections.Counter(re.findall(r"^\s+([a-z_0-9]+)", asm, re.M))
    m = re.search(r"vgpr_count:\s*(\d+)", asm)
    return {
        "vgpr": int(m.group(1)) if m else None,
        "spill_st": sum(v for k, v in ins.items() if k.startswith("scratch_store")),
        "spill_ld": sum(v for k, v in ins.items() if k.startswith("scratch_load")),
        "s_set_vgpr_msb": ins.get("s_set_vgpr_msb", 0),
        "wmma": sum(v for k, v in ins.items() if "wmma" in k),
        "ds_load_tr16_b128": ins.get("ds_load_tr16_b128", 0),
        "total_instr": sum(ins.values()),
    }


def compile_one(fn, cfg, num_warps, num_stages, waves_per_eu):
    constants = {
        "BLOCK_M1": cfg["BLOCK_M1"], "BLOCK_N1": cfg["BLOCK_N1"],
        "BLOCK_M2": cfg["BLOCK_M2"], "BLOCK_N2": cfg["BLOCK_N2"],
        "BLK_SLICE_FACTOR": cfg["BLK_SLICE_FACTOR"],
        "HEAD_DIM": 128, "ACTUAL_HEAD_DIM": 128, "PE_HEAD_DIM": 0,
        "ENABLE_DROPOUT": False, "IS_VARLEN": False, "USE_ALIBI": False,
        "USE_EXP2": True, "IS_FP8": False, "FP8_MAX": None,
        "DEBUG_TRITON": False, "DEBUG_TRITON_DETAIL": False,
        "USE_INT64_STRIDES": True, "ENABLE_SINK": False, "SLIDING_WINDOW": 0,
    }
    for p in fn.params:  # fold unit innermost strides, as the runtime does
        if UNIT_STRIDE.match(p.name):
            constants[p.name] = 1
    sig = signature_for(fn, constants)
    attrs = {}
    for i, p in enumerate(fn.params):
        n = p.name
        if sig[n] == "constexpr":
            continue
        aligned = ("max_seqlen_q", "max_seqlen_k", "HQ")
        if sig[n].startswith("*") or n.startswith("stride_") or n in aligned:
            attrs[(i,)] = [["tt.divisibility", 16]]
    src = ASTSource(fn=fn, signature=sig, constexprs=constants, attrs=attrs)
    opts = {"num_warps": num_warps, "num_stages": num_stages,
            "matrix_instr_nonkdim": 16, "kpack": 1}
    if waves_per_eu is not None:
        opts["waves_per_eu"] = waves_per_eu
    compiled = triton.compile(src, target=TARGET, options=opts)
    return compiled.asm["amdgcn"]


# The pairing constraint BLOCK_N1 == BLOCK_M2 and BLOCK_M1 == BLOCK_N2 is
# structural: breaking it computes half the dq rows (see PROGRESS.md rule 4).
def tile(m1, n1, bsf):
    return {"BLOCK_M1": m1, "BLOCK_N1": n1, "BLOCK_M2": n1, "BLOCK_N2": m1,
            "BLK_SLICE_FACTOR": bsf}


GRIDS = {
    # Reproduces the champion plus the untested neighbours it should be
    # screened against. Champion == (32,256,bsf=1,warps=4,wpe=1).
    "default": [
        (tile(m1, n1, bsf), w, s, wpe)
        for m1, n1 in ((32, 256), (32, 128), (64, 128), (64, 256))
        for bsf in (1, 2)
        for w in (4, 8)
        for s in (1, 2)
        for wpe in (1, None)
    ],
    "champion": [(tile(32, 256, 1), 4, 1, 1)],
}


def validate(fn, census_path):
    """Does the screen agree with a real compilation on the decisions it is used for?

    Absolute instruction counts are NOT expected to match -- the offline
    reconstruction of the runtime's specialization is close but not exact. What
    must match is what the screen is used to decide: whether a config pins VGPRs
    at the 1024 cap, and whether it spills.
    """
    import ast
    truth = {}
    for line in open(census_path):
        r = json.loads(line)
        m = re.search(r"BLOCK_M1_(\d+)_BLOCK_N1_(\d+)_BLOCK_M2_(\d+)_BLOCK_N2_(\d+)_BLK_SLICE_FACTOR_(\d+)", r["name"])
        if not m:
            continue
        m1, n1, m2, n2, bsf = (int(x) for x in m.groups())
        if (n1, m1) != (m2, n2):
            continue
        truth[(m1, n1, bsf, r["num_warps"], r["num_stages"], r["waves_per_eu"])] = r

    agree_cap = agree_spill = total = 0
    print(f"{'config':<34} {'vgpr pred/true':>16} {'spills pred/true':>18}  verdict")
    for key, t in sorted(truth.items()):
        m1, n1, bsf, w, s, wpe = key
        try:
            st = analyse(compile_one(fn, tile(m1, n1, bsf), w, s, wpe))
        except Exception as e:
            print(f"  {key} -> compile failed: {type(e).__name__}")
            continue
        total += 1
        cap_p, cap_t = st["vgpr"] == 1024, t["vgpr"] == 1024
        sp_p, sp_t = st["spill_st"] > 0, t["spill_st"] > 0
        agree_cap += cap_p == cap_t
        agree_spill += sp_p == sp_t
        name = f"M1={m1} N1={n1} bsf={bsf} w={w} s={s} wpe={wpe}"
        mark = "ok" if (cap_p == cap_t and sp_p == sp_t) else "MISMATCH"
        print(f"{name:<34} {st['vgpr']:>7}/{t['vgpr']:<7} "
              f"{st['spill_st']+st['spill_ld']:>8}/{t['spill_st']+t['spill_ld']:<8}  {mark}")
    print(f"\nagreement on 'pins VGPRs at cap': {agree_cap}/{total}")
    print(f"agreement on 'spills at all':      {agree_spill}/{total}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="default", choices=sorted(GRIDS))
    ap.add_argument("--out", default="-")
    ap.add_argument("--validate", metavar="CENSUS.jsonl",
                    help="compare predictions against cached ground-truth compilations")
    args = ap.parse_args()

    from primus_turbo.triton.attention.fused_mha_bwd_kernel import bwd_kernel_causal as fn

    if args.validate:
        return validate(fn, args.validate)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    ok = fail = 0
    for cfg, w, s, wpe in GRIDS[args.grid]:
        row = {"BLOCK_M1": cfg["BLOCK_M1"], "BLOCK_N1": cfg["BLOCK_N1"],
               "BLK_SLICE_FACTOR": cfg["BLK_SLICE_FACTOR"],
               "num_warps": w, "num_stages": s, "waves_per_eu": wpe}
        try:
            row.update(analyse(compile_one(fn, cfg, w, s, wpe)))
            ok += 1
        except Exception as e:  # a config that will not compile is a real verdict
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            fail += 1
        print(json.dumps(row), file=out, flush=True)
    print(f"# compiled={ok} failed={fail}", file=sys.stderr)


if __name__ == "__main__":
    main()
