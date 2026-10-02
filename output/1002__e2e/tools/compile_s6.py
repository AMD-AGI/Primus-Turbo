"""Compile-only (no GPU dispatch) of an s6-family bwd arm at the prod shape (b4 s8192 hq32 hkv8 d128 causal).

usage (inside fa-repro, via tools/compile_s6.sh): compile_s6.py <arm_dir> [delta] [dkdv] [dkdv_sp] [dqg]
The arm's own _env.py decides the flydsl version (0.3.2 for 0930__bwd/armsrc/s6, 0.3.4.1 for bwd_s6_0341).
Adapted from output/0930__bwd/tools/compile_arm.py: same launch arguments as impl.py at prod
(k_dkdv nsp=1; k_dkdv_sp with nsp=2 like compile_bwd_0341.py; k_dqg nsp_q=1, DQ_NW=1), plus
an explicit arch / wave-size / flydsl-origin assertion so a silent gfx942 or wrong-flydsl build
cannot pass as a result.
"""
import importlib.util
import os
import pathlib
import sys
import types

assert os.environ.get("COMPILE_ONLY") == "1", "compile-only tool: COMPILE_ONLY=1 required"
assert os.environ.get("HIP_VISIBLE_DEVICES") == "-1", "compile-only tool: HIP_VISIBLE_DEVICES=-1 required"
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import _env  # noqa: E402,F401  (pins flydsl + aiter-src on sys.path)

AITER = "/home/lihuzhan/code/aiter-src"
# Stub the aiter package chain so aiter/__init__ (which may touch the device) never runs.
for pkg in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(AITER, *pkg.split("."))]
    sys.modules[pkg] = m
import torch  # noqa: E402
import flydsl  # noqa: E402
import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime import device as _dev  # noqa: E402

arch = _dev.get_rocm_arch()
ws = _dev.get_warp_size(arch)
print(f"flydsl {flydsl.__version__} {flydsl.__file__} arch={arch} warp={ws} torch={torch.__version__}", flush=True)
assert arch == "gfx1250", f"resolved arch {arch!r}, not gfx1250 (set FLYDSL_GPU_ARCH)"
assert ws == 32, f"warp size {ws} for {arch}"
want = os.environ.get("EXPECT_FLYDSL")
if want:
    assert flydsl.__version__ == want, f"flydsl {flydsl.__version__} at {flydsl.__file__}, expected {want}"

sp = importlib.util.spec_from_file_location("bwdk", root / "kernels.py")
K = importlib.util.module_from_spec(sp)
sys.modules["bwdk"] = K
sp.loader.exec_module(K)
print("aiter modules loaded:", sorted(n for n in sys.modules if n.startswith("aiter.ops.flydsl.kernels.")), flush=True)
which = sys.argv[2:] or ["delta", "dkdv", "dkdv_sp", "dqg"]
M = "meta"
b, sq, hq, hkv, d = 4, 8192, 32, 8, 128
skv = sq
g = hq // hkv
bf, f32 = torch.bfloat16, torch.float32
q = torch.empty((b, sq, hq, d), dtype=bf, device=M)
k = torch.empty((b, skv, hkv, d), dtype=bf, device=M)
v = torch.empty_like(k)
do = torch.empty_like(q)
o = torch.empty_like(q)
lse = torch.empty((b, hq, sq), dtype=f32, device=M)
delta = torch.empty_like(lse)
scale = d ** -0.5
n_rows = b * sq * hq
nsp = 2
jobs = {
    "delta": (K.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // K.ROWS_DELTA, None)),
    "dkdv": (K.launch_dkdv, (q, k, v, do, lse, delta, torch.empty_like(k), torch.empty_like(k),
             scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b, None)),
    "dkdv_sp": (K.launch_dkdv_sp, (q, k, v, do, lse, delta,
                torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
                torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
                scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b,
                nsp, hkv * nsp, None)),
    "dqg": (K.launch_dqg, (q, k, v, do, o, lse, delta, torch.empty_like(q), scale,
            sq, skv, hq, hkv, g, skv // K.KV_STEP, skv - sq, 1, sq // K.DQ_BQW, hq // K.DQ_NW, b, None)),
}
base = os.environ["FLYDSL_DUMP_DIR"]
for name in which:
    os.environ["FLYDSL_DUMP_DIR"] = os.path.join(base, name)
    fn, args = jobs[name]
    r = flyc.compile(fn, *args)
    print(f"[{name}] compile returned {type(r).__name__}", flush=True)
print("COMPILE_DONE", flush=True)
