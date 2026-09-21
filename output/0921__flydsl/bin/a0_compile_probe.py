#!/usr/bin/env python3
"""A0 gate: can FlyDSL compile to gfx1250 ISA with NO GPU present?

Runs inside a container started WITHOUT /dev/kfd and /dev/dri, so it physically
cannot touch the card. Tries three argument-binding strategies in order and
reports which (if any) reaches a dumped 21_final_isa.s.

Every run asserts the resolved target is gfx1250. Without FLYDSL_GPU_ARCH and
with no /dev/kfd, get_rocm_arch() silently answers 'gfx942' -- an audit against
that target answers a different question.
"""
import glob
import json
import os
import sys
import traceback

DUMP = os.environ.get("FLYDSL_DUMP_DIR", "/tmp/a0dump")
os.makedirs(DUMP, exist_ok=True)

sys.path.insert(0, "/home/lihuzhan/.local/flydsl032")

report = {"stages": []}


def stage(name, fn):
    rec = {"name": name}
    try:
        rec["result"] = fn()
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - the whole point is to record any failure
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-1200:]
    report["stages"].append(rec)
    return rec


def s_import():
    import flydsl
    from flydsl.runtime.device import get_rocm_arch

    arch = get_rocm_arch()
    # The load-bearing assertion. Never let a gfx942 answer pass as a gfx1250 one.
    assert arch == "gfx1250", f"resolved target is {arch!r}, not gfx1250"
    return {
        "flydsl_version": flydsl.__version__,
        "flydsl_file": flydsl.__file__,
        "arch": arch,
        "COMPILE_ONLY": os.environ.get("COMPILE_ONLY"),
        "ARCH": os.environ.get("ARCH"),
        "FLYDSL_GPU_ARCH": os.environ.get("FLYDSL_GPU_ARCH"),
    }


def s_torch():
    import torch

    return {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}


def s_env_manager():
    from flydsl.utils import env as E

    names = {}
    for cls in ("CompileEnvManager", "DebugEnvManager"):
        c = getattr(E, cls)
        names[cls] = {k: getattr(v, "env_var", None)
                      for k, v in vars(c).items() if hasattr(v, "env_var")}
    return names


def s_compiler_import():
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    return {"compiler": flyc.__name__, "expr": fx.__name__}


def s_build_tiny(binding):
    """Compile a trivial elementwise kernel with the given argument binding."""
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k_copy(A: fx.Tensor, B: fx.Tensor, n: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x) + fx.Int32(fx.block_idx.x) * fx.Int32(64)
        if tid < n:
            B[tid] = A[tid]

    @flyc.jit
    def launch(A, B, n: fx.Int32, nblk: fx.Int32):
        # No stream parameter: None is not a JitArgument, and a compile-only
        # driver has no torch.cuda.Stream to hand it. launch() takes stream=None.
        k_copy(A, B, n).launch(grid=(nblk, 1, 1), block=(64, 1, 1))

    n = 256
    if binding == "cpu":
        a = torch.zeros(n, dtype=torch.float32)
        b = torch.zeros(n, dtype=torch.float32)
    elif binding == "meta":
        a = torch.zeros(n, dtype=torch.float32, device="meta")
        b = torch.zeros(n, dtype=torch.float32, device="meta")
    else:
        raise RuntimeError(f"unknown binding {binding}")

    launch(a, b, fx.Int32(n), fx.Int32(4))
    return {"binding": binding, "launched": True}


def s_dump_files():
    files = sorted(glob.glob(os.path.join(DUMP, "**", "*"), recursive=True))
    isa = [f for f in files if f.endswith("_final_isa.s")]
    out = {"n_files": len(files), "isa_files": isa[:5], "sample": files[:15]}
    if isa:
        txt = open(isa[0]).read()
        out["isa_bytes"] = len(txt)
        out["has_gfx1250"] = "gfx1250" in txt
        for key in ("amdhsa_next_free_vgpr", "amdhsa_private_segment_fixed_size",
                    "amdhsa_group_segment_fixed_size", "amdhsa_accum_offset"):
            for line in txt.splitlines():
                if key in line:
                    out[key] = line.strip()
                    break
    return out


stage("import_flydsl", s_import)
stage("import_torch", s_torch)
stage("env_manager", s_env_manager)
stage("import_compiler", s_compiler_import)
stage("build_cpu_tensor", lambda: s_build_tiny("cpu"))
stage("build_meta_tensor", lambda: s_build_tiny("meta"))
stage("dump_files", s_dump_files)

print(json.dumps(report, indent=1)[:6000])
