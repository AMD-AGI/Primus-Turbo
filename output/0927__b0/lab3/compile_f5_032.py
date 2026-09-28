"""[fused5 copy of api-audit/.bwd_work/compile_bwd.py: + dkdv_f5, dkdv_sp_f5, cvt_dq]
Compile-only (no GPU dispatch) of the bwd kernel group at the prod shape.

usage: compile_bwd.py <impl_dir>
prod: b4 s8192 hq32 hkv8 d128 causal bf16 -> delta, dkdv (nsp=1), dq (nsp_q=1).
The split-K variants (dkdv_sp, dq_sp, redsp, redsp_q) are compiled too with nsp=2,
since their kernels take every shape-dependent value as a runtime Int32.
"""
import importlib.util, os, sys, types, pathlib
assert os.environ.get("COMPILE_ONLY") == "1"
AITER = "/home/lihuzhan/code/aiter-src"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl032"); sys.path.insert(0, AITER)
# Stub the aiter package chain so aiter/__init__ (which may touch the device) never runs.
for pkg in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
    m = types.ModuleType(pkg); m.__path__ = [os.path.join(AITER, *pkg.split("."))]
    sys.modules[pkg] = m
import torch
import flydsl, flydsl.compiler as flyc
print("flydsl", flydsl.__version__, flydsl.__file__)
root = pathlib.Path(sys.argv[1]).resolve()
sp = importlib.util.spec_from_file_location("bwdk", root / "kernels.py")
K = importlib.util.module_from_spec(sp); sys.modules["bwdk"] = K; sp.loader.exec_module(K)
print("aiter modules loaded:", sorted(n for n in sys.modules if n.startswith("aiter.ops.flydsl.kernels.")))
which = sys.argv[2:] or ["delta", "dkdv", "dq", "dkdv_sp", "dq_sp", "redsp", "redsp_q"]
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
nsp = 2
jobs = {
    "delta": (K.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // K.ROWS_DELTA, None)),
    "dkdv": (K.launch_dkdv, (q, k, v, do, lse, delta, torch.empty_like(k), torch.empty_like(k),
             scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b, None)),
    "dq": (K.launch_dq, (q, k, v, do, lse, delta, torch.empty_like(q), scale,
           sq, skv, hq, hkv, g, skv // K.KV_STEP, skv - sq, 1, sq // K.BLOCK_Q, hq, b, None)),
    "dkdv_sp": (K.launch_dkdv_sp, (q, k, v, do, lse, delta,
                torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
                torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
                scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b,
                nsp, hkv * nsp, None)),
    "dq_sp": (K.launch_dq_sp, (q, k, v, do, lse, delta,
              torch.empty((nsp, b, sq, hq, d), dtype=f32, device=M), scale,
              sq, skv, hq, hkv, g, skv // K.KV_STEP, skv - sq, 1, sq // K.BLOCK_Q, hq, b,
              nsp, hq * nsp, None)),
    "dkdv_f5": (getattr(K,"launch_dkdv_f5",None), (q, k, v, do, lse, delta, torch.empty_like(k), torch.empty_like(k),
             torch.empty((b, sq, hq, d), dtype=f32, device=M),
             scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b, None)),
    "dkdv_sp_f5": (getattr(K,"launch_dkdv_sp_f5",None), (q, k, v, do, lse, delta,
                torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
                torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
                torch.empty((b, sq, hq, d), dtype=f32, device=M),
                scale, sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // K.BLOCK_KV, hkv, b,
                nsp, hkv * nsp, None)),
    "cvt_dq": (K.launch_redsp_q, (torch.empty((b, sq, hq, d), dtype=f32, device=M),
                torch.empty_like(q), (b * sq * hq * d) // K.RED_VEC, 1,
                ((b * sq * hq * d) // K.RED_VEC + K.RED_THREADS - 1) // K.RED_THREADS, None)),
    "redsp": (K.launch_redsp, (torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
              torch.empty((nsp, b, skv, hkv, d), dtype=f32, device=M),
              torch.empty_like(k), torch.empty_like(k),
              (b * skv * hkv * d) // K.RED_VEC, nsp,
              ((b * skv * hkv * d) // K.RED_VEC + K.RED_THREADS - 1) // K.RED_THREADS, None)),
    "redsp_q": (K.launch_redsp_q, (torch.empty((nsp, b, sq, hq, d), dtype=f32, device=M),
                torch.empty_like(q), (b * sq * hq * d) // K.RED_VEC, nsp,
                ((b * sq * hq * d) // K.RED_VEC + K.RED_THREADS - 1) // K.RED_THREADS, None)),
}
base = os.environ["FLYDSL_DUMP_DIR"]
for name in which:
    os.environ["FLYDSL_DUMP_DIR"] = os.path.join(base, name)
    fn, args = jobs[name]
    r = flyc.compile(fn, *args)
    print(f"[{name}] compile returned {type(r).__name__}", flush=True)
