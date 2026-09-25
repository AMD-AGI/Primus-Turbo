#!/usr/bin/env python3
"""Compile the three gfx1250 FlyDSL backward kernels to ISA with NO GPU present.

WHY THIS FILE EXISTS. Every existing bring-up script builds device tensors, launches,
and checks -- so none of them can run where there is no card. But the only instrument
that works on this machine is static: rocprofv3 faults the GPU on this op, and of 51
gfx1250 counters the WMMA FLOP ones read exactly 0 on a kernel proven to issue 4.096M
WMMAs. Meanwhile round 1 showed a pure WMMA count predicts this op's runtime to within
4%. So the screen has to be compile-time, and compile-time has to work without a card.

HOW. flydsl ships a first-class mode for this. Note the two variables have NO `FLYDSL_`
prefix -- `flydsl/utils/env.py` sets their `env_var` explicitly, and the metaclass only
prefixes the ones left as None:

    COMPILE_ONLY=1      -> flydsl.utils.env.CompileEnvManager.compile_only
    ARCH=gfx1250        -> flydsl.utils.env.CompileEnvManager.arch
    FLYDSL_GPU_ARCH=... -> flydsl.runtime.device.get_rocm_arch

BOTH arch variables must be set. `ARCH` steers the compile backend; `FLYDSL_GPU_ARCH`
steers buffer-descriptor selection through get_rocm_arch(). Setting only one yields an
ISA with a mixed target -- and it compiles silently.

`ARCH` is a dangerously common name for an environment variable. Anything in an outer
script that sets it for its own purposes silently retargets the compiler. That is why
every record this file writes carries an `arch` witness and why resolution is asserted,
exactly as every timing number carries an sclk witness.

The kernels' own `launch_*` wrappers take `stream: fx.Stream`, and None is not a
JitArgument ("NoneType is neither a JitArgument nor has a registered constructor").
There is no torch.cuda.Stream to hand them without a card, so this file declares its own
stream-free @flyc.jit wrappers around the same @flyc.kernel builders. The kernel bodies
are what gets screened; the launch wrapper is not.

Argument binding: torch CPU tensors work (measured). flydsl only reads data_ptr/shape/
stride to build the cache key, and COMPILE_ONLY returns before the execution engine is
ever built -- JitExecutor.ir()/source_ir()/dump() never call _ensure_engine().

usage (inside a container started WITHOUT /dev/kfd and /dev/dri):
    compile_only_driver.py --impl <dir-with-kernels.py> --dump-dir <dir> [--json out.json]
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

REQUIRED_ARCH = "gfx1250"


AITER_SRC = "/home/lihuzhan/code/aiter-src"

# The two vendor symbols kernels.py needs, and the only two.
AITER_SHIMS = (
    "aiter.ops.flydsl.kernels.kernels_common",
    "aiter.ops.flydsl.kernels.tensor_shim",
)


def preload_aiter_shims():
    """Load the two vendor helper modules by path, WITHOUT running aiter/__init__.py.

    kernels.py needs exactly `create_llvm_ptr` (8 lines of flydsl pointer casting) and
    `_to_raw` (3 lines of ir.Value coercion). Reaching them the normal way runs
    `import aiter`, which on this card-less container walks into hardware detection
    (`get_gfx_runtime` calls rocminfo directly and ignores the GPU_ARCHS override) and
    then into an unrelated Triton/gluon path that wants jax, which this image lacks.

    So seed empty parent packages and exec the two real vendor files into them. This is
    the vendor's own source, not a reimplementation -- the point is to skip the package
    __init__, not to fork the helpers. Both depend only on flydsl, torch and numpy.

    Returns the resolved file of each shim so the record can prove which copy was used.
    """
    import types

    used = {}
    for pkg in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [os.path.join(AITER_SRC, *pkg.split("."))]
            sys.modules[pkg] = m
    for dotted in AITER_SHIMS:
        if dotted in sys.modules and getattr(sys.modules[dotted], "__file__", None):
            used[dotted] = sys.modules[dotted].__file__
            continue
        path = os.path.join(AITER_SRC, *dotted.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        used[dotted] = path
    return used


def materialise(impl_dir, overrides, workdir):
    """Copy an implementation and rewrite module-level constants in its kernels.py.

    WHY A COPY AND NOT setattr AFTER IMPORT. The tunable constants are module-level and
    most of the interesting ones have DERIVED constants computed beside them at import
    time -- DELTA_THREADS feeds LANES_PER_ROW, ROWS_PER_PASS and PASSES_PER_WG; D feeds
    NDT and NDO. Rebinding the parent after import leaves every derived value at its old
    setting, and the kernel still compiles. It would produce a candidate that is not the
    candidate you asked for and reports no error at all.

    Rewriting the source and re-importing is also how the job's own arms are built:
    armA/armB/armAB under rounds/001/_scratch are whole directory copies, not patched
    imports. Same identity rule as load_kernels -- the directory IS the candidate.

    Only `NAME = <number>` at column zero is rewritten, so a same-named local inside a
    function is untouched. A name that is not found is an error rather than a silent
    no-op: a typo'd knob that quietly screens the unmodified kernel is exactly the shape
    of result that looks real.
    """
    import re as _re
    import shutil

    dst = os.path.join(workdir, os.path.basename(impl_dir.rstrip("/")))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(impl_dir, dst)
    path = os.path.join(dst, "kernels.py")
    src = open(path).read()
    applied = {}
    for name, value in overrides.items():
        pat = _re.compile(rf"^({_re.escape(name)}\s*=\s*)([-\w.]+)", _re.M)
        hit = pat.search(src)
        if not hit:
            raise SystemExit(
                f"knob {name!r} is not a module-level constant in {path}. "
                f"Screening would have silently compiled the unmodified kernel."
            )
        applied[name] = {"from": hit.group(2), "to": value}
        src = pat.sub(rf"\g<1>{value}", src, count=1)
    open(path, "w").write(src)
    return dst, applied


def load_kernels(impl_dir):
    """Import kernels.py from impl_dir under a name unique to that directory.

    The directory is the identity. A plain `import kernels` binds sys.modules["kernels"],
    so a second implementation loaded in the same process silently reuses the first one's
    module -- and with it the first one's compiled kernels. The two arms then differ by
    almost nothing and produce identical ISA, which looks exactly like a real result.
    This mirrors op/baseline/impl.py:_sibling, deliberately.
    """
    path = os.path.join(impl_dir, "kernels.py")
    if not os.path.isfile(path):
        raise SystemExit(f"no kernels.py in {impl_dir}")
    name = f"kernels__{abs(hash(os.path.abspath(impl_dir)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def assert_arch():
    """The load-bearing assertion.

    With no /dev/kfd and FLYDSL_GPU_ARCH unset, get_rocm_arch() does not raise and does
    not return an empty string -- _arch_from_hardware() falls back to 'gfx942'. An audit
    against that target answers a different question and reports success while doing it.
    Also pin the flydsl copy: the image carries 0.2.4 and 0.3.2 is a bind mount, so
    "which flydsl" is a real question every record has to answer.
    """
    import flydsl
    from flydsl.runtime.device import get_rocm_arch, get_warp_size

    arch = get_rocm_arch()
    if arch != REQUIRED_ARCH:
        raise SystemExit(
            f"resolved target is {arch!r}, not {REQUIRED_ARCH!r}. "
            f"Set FLYDSL_GPU_ARCH and check nothing else in the environment defines ARCH."
        )
    return {
        "arch": arch,
        "warp_size": get_warp_size(arch),
        "flydsl_version": flydsl.__version__,
        "flydsl_file": flydsl.__file__,
        "compile_only": os.environ.get("COMPILE_ONLY"),
        "env_ARCH": os.environ.get("ARCH"),
        "env_FLYDSL_GPU_ARCH": os.environ.get("FLYDSL_GPU_ARCH"),
    }


def build_all(k, shape):
    """Compile all three kernels. Returns which ones reached codegen."""
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    b, sq, skv, hq, hkv = shape["b"], shape["sq"], shape["skv"], shape["hq"], shape["hkv"]
    d = k.D
    g = hq // hkv
    dev, bf16, f32 = "cpu", torch.bfloat16, torch.float32

    q = torch.zeros((b, sq, hq, d), dtype=bf16, device=dev)
    o = torch.zeros_like(q)
    do = torch.zeros_like(q)
    kk = torch.zeros((b, skv, hkv, d), dtype=bf16, device=dev)
    vv = torch.zeros_like(kk)
    lse = torch.zeros((b, hq, sq), dtype=f32, device=dev)
    dele = torch.zeros((b, hq, sq), dtype=f32, device=dev)
    dq32 = torch.zeros((b, sq, hq, d), dtype=f32, device=dev)
    dk32 = torch.zeros((b, skv, hkv, d), dtype=f32, device=dev)
    dv32 = torch.zeros_like(dk32)

    n_rows = b * sq * hq
    scale = float(d) ** -0.5

    # Stream-free wrappers around the same @flyc.kernel builders. See the module
    # docstring: the kernels' own launch_* take fx.Stream, which cannot be supplied here.
    @flyc.jit
    def go_delta(DO, O, DEL, S: fx.Int32, H: fx.Int32, nr: fx.Int32, nblk: fx.Int32):
        k.k_delta_bshd(DO, O, DEL, S, H, nr).launch(
            grid=(nblk, 1, 1), block=(k.DELTA_THREADS, 1, 1))

    @flyc.jit
    def go_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, sc: fx.Float32,
                Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32):
        k.k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, sc, Sq, Skv, Hq, Hkv, G,
                 nqt, cshift, causal).launch(
            grid=(nblk, nhkv, nb), block=(getattr(k, "DKDV_THREADS", 32), 1, 1))

    # r2: the g07 address-clamp arm takes one extra kernel argument (batch count), so
    # the screen needs both signatures. Declared, never guessed -- a driver that silently
    # screened the wrong arity would report a pass for a kernel it never compiled.
    @flyc.jit
    def go_dkdv_b(Q, K, V, DO, LSE, DEL, DV_, DK, sc: fx.Float32,
                  Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                  nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                  nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32):
        k.k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, sc, Sq, Skv, Hq, Hkv, G,
                 nqt, cshift, causal, nb).launch(
            grid=(nblk, nhkv, nb), block=(getattr(k, "DKDV_THREADS", 32), 1, 1))

    @flyc.jit
    def go_dq(Q, K, V, DO, LSE, DEL, DQ, sc: fx.Float32,
              Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
              nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
              nblk: fx.Int32, nhq: fx.Int32, nb: fx.Int32):
        k.k_dq(Q, K, V, DO, LSE, DEL, DQ, sc, Sq, Skv, Hq, Hkv, G,
               nkvt, cshift, causal).launch(
            grid=(nblk, nhq, nb), block=(32, 1, 1))

    I = fx.Int32
    built = {}
    for name, fn in (
        ("delta", lambda: go_delta(do, o, dele, I(sq), I(hq), I(n_rows),
                                   I(n_rows // k.ROWS_DELTA))),
        ("dkdv", lambda: (go_dkdv_b if getattr(k, "_G07_CLAMP", False) else go_dkdv)(
            q, kk, vv, do, lse, dele, dv32, dk32, fx.Float32(scale),
            I(sq), I(skv), I(hq), I(hkv), I(g), I(sq // 16),
            I(skv - sq), I(1), I(skv // getattr(k, "BLOCK_KV", 16)), I(hkv), I(b))),
        ("dq", lambda: go_dq(q, kk, vv, do, lse, dele, dq32, fx.Float32(scale),
                             I(sq), I(skv), I(hq), I(hkv), I(g), I(skv // k.KV_STEP),
                             I(skv - sq), I(1), I(sq // 16), I(hq), I(b))),
    ):
        try:
            fn()
            built[name] = {"ok": True}
        except Exception as e:  # noqa: BLE001 - record any failure, never abort the sweep
            built[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return built


META = ("amdhsa_next_free_vgpr", "amdhsa_next_free_sgpr", "amdhsa_accum_offset",
        "amdhsa_private_segment_fixed_size", "amdhsa_group_segment_fixed_size")

# gfx1250 instruction mix. Kyle's gfx950 OPS table CANNOT be reused: v_mfma_*,
# accvgpr, ds_read_b64_tr_b16 and buffer_atomic_pk_add_bf16 are all identically 0
# here, so a gate written against them passes every candidate forever.
OPS = ("v_wmma_f32_16x16x32_bf16", "ds_load_tr16_b128", "ds_read_b128", "ds_write_b128",
       "s_barrier", "buffer_load_dwordx4", "buffer_store_dwordx4", "v_exp_f32",
       "s_set_vgpr_msb", "scratch_", "global_load_lds", "s_wait_dscnt",
       "s_wait_asynccnt", "sched_barrier")


def scan_isa(dump_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, "**", "*_final_isa.s"),
                                 recursive=True)):
        kern = os.path.basename(os.path.dirname(path))
        txt = open(path).read()
        rec = {"path": path, "bytes": len(txt)}
        m = re.search(r'\.amdgcn_target\s+"([^"]+)"', txt)
        rec["target"] = m.group(1) if m else None
        # An ISA whose target is not gfx1250 answers a different question. Flag, never drop.
        rec["target_ok"] = bool(m and REQUIRED_ARCH in m.group(1))
        rec["wave32"] = ".amdhsa_wavefront_size32 1" in txt
        for key in META:
            mm = re.search(rf"\.{key}\s+(\d+)", txt)
            if mm:
                rec[key.replace("amdhsa_", "")] = int(mm.group(1))
        rec["total_instr"] = len(re.findall(r"^\s+(?:s_|v_|ds_|buffer_|global_|flat_|scratch_)",
                                            txt, re.M))
        rec["ops"] = {op: txt.count(op) for op in OPS}
        # The only hard kill. A spilling build does not merely run slow on gfx1250 --
        # it hangs after the first launch, which costs an AC cycle to recover.
        rec["spill"] = rec.get("private_segment_fixed_size", 0) + rec["ops"]["scratch_"]
        rec["verdict"] = "KILL-spill" if rec["spill"] > 0 else "pass"
        out[kern] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, help="directory containing kernels.py")
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--sq", type=int, default=8192)
    ap.add_argument("--skv", type=int, default=8192)
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a module-level constant in kernels.py (repeatable)")
    ap.add_argument("--workdir", default="/tmp/flyscreen",
                    help="where --set materialises the patched copy")
    ap.add_argument("--tag", default=None, help="label carried into the JSON record")
    a = ap.parse_args()

    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = a.dump_dir
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    os.makedirs(a.dump_dir, exist_ok=True)

    sys.path.insert(0, "/home/lihuzhan/.local/flydsl032")
    sys.path.insert(0, "/home/lihuzhan/code/aiter-src")

    witness = assert_arch()
    witness["aiter_shims"] = preload_aiter_shims()
    shape = {"b": a.b, "sq": a.sq, "skv": a.skv, "hq": a.hq, "hkv": a.hkv}
    impl = a.impl
    applied = {}
    if a.set:
        overrides = dict(kv.split("=", 1) for kv in a.set)
        os.makedirs(a.workdir, exist_ok=True)
        impl, applied = materialise(a.impl, overrides, a.workdir)
    k = load_kernels(impl)
    built = build_all(k, shape)
    isa = scan_isa(a.dump_dir)

    report = {"witness": witness, "impl": os.path.abspath(impl),
              "source_impl": os.path.abspath(a.impl), "tag": a.tag,
              "overrides": applied, "shape": shape, "built": built, "isa": isa}
    text = json.dumps(report, indent=1)
    if a.json:
        open(a.json, "w").write(text)
    print(text)
    # Exit non-zero only when nothing compiled -- a per-kernel failure is a ledger row.
    return 0 if any(v.get("ok") for v in built.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
