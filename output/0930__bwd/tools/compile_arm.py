"""Compile-only (no GPU dispatch) of an r29-family bwd arm at the prod shape: k_dkdv (nsp=1) and k_dqg.
usage (inside fa-repro, via tools/compile.sh): compile_arm.py <arm_dir> [dkdv] [dqg] [delta]
The arm's own _env.py decides the flydsl version (0.3.2 for the r29 champion)."""
import importlib.util, os, sys, types, pathlib
assert os.environ.get("COMPILE_ONLY") == "1"
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import _env  # noqa: F401  (pins flydsl + aiter-src on sys.path)
AITER = "/home/lihuzhan/code/aiter-src"
for pkg in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
    m = types.ModuleType(pkg); m.__path__ = [os.path.join(AITER, *pkg.split("."))]
    sys.modules[pkg] = m
import torch
import flydsl, flydsl.compiler as flyc
print("flydsl", flydsl.__version__, flydsl.__file__, flush=True)
sp = importlib.util.spec_from_file_location("bwdk", root / "kernels.py")
K = importlib.util.module_from_spec(sp); sys.modules["bwdk"] = K; sp.loader.exec_module(K)
which = sys.argv[2:] or ["dkdv", "dqg"]
M = "meta"
b, sq, hq, hkv, d = 4, 8192, 32, 8, 128
skv = sq; g = hq // hkv
bf, f32 = torch.bfloat16, torch.float32
q = torch.empty((b, sq, hq, d), dtype=bf, device=M)
k = torch.empty((b, skv, hkv, d), dtype=bf, device=M)
v = torch.empty_like(k); do = torch.empty_like(q); o = torch.empty_like(q)
lse = torch.empty((b, hq, sq), dtype=f32, device=M); delta = torch.empty_like(lse)
scale = d ** -0.5
n_rows = b * sq * hq
jobs = {
    "delta": (K.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // K.ROWS_DELTA, None)),
    "dkdv": (K.launch_dkdv, (q, k, v, do, lse, delta, torch.empty_like(k), torch.empty_like(k),
             scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b, None)),
    "dqg": (K.launch_dqg, (q, k, v, do, o, lse, delta, torch.empty_like(q), scale,
            sq, skv, hq, hkv, g, skv // K.KV_STEP, skv - sq, 1, sq // K.DQ_BQW, hq // K.DQ_NW, b, None)),
}
base = os.environ["FLYDSL_DUMP_DIR"]
for name in which:
    os.environ["FLYDSL_DUMP_DIR"] = os.path.join(base, name)
    fn, args = jobs[name]
    r = flyc.compile(fn, *args)
    print(f"[{name}] compile returned {type(r).__name__}", flush=True)
