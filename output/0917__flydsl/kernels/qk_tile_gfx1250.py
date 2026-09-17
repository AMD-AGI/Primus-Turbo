"""Second increment toward dkdv: S^T = K @ Q^T for one 16x16 tile, contracting d.

The layout probe did one WMMA. This adds the thing the probe could not exercise: the
accumulation chain across d-tiles. At d=128 that is 4 chained `v_wmma_f32_16x16x32` calls
feeding the same v8 accumulator, which is the inner loop of both dkdv and dq.

Deliberately expressed the way dkdv needs it -- S^T[kv, q] = K @ Q^T with K as the
A-operand and Q as the B-operand -- rather than the more obvious S = Q @ K^T, so that what
is validated here is what gets reused.
"""
import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, "/tmp/flydsl032")
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")

import torch, flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir

print(f"flydsl {flydsl.__version__}   arch {torch.cuda.get_device_properties(0).gcnArchName}")
WMMA_M = WMMA_N = 16
WMMA_K = 32


def _bv(t, nbytes, dt, vec=1):
    ntile = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nbytes))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((ntile, vec), (vec, 1))))


def _atom(dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)


def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
    return f.load()


def _st1(val, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))


def build(D):
    NDT = D // WMMA_K                      # contraction tiles along d
    ROW_T = D // 8                         # v8 tiles per row of a [16, D] operand

    def frag(buf, row, half, dt_idx):
        """One v16 operand fragment for d-tile dt_idx, per the confirmed layout."""
        base = row * fx.Int32(ROW_T) + half + fx.Int32(dt_idx * (WMMA_K // 8))
        lo = _ldv(buf, base, fx.BFloat16, 8)
        hi = _ldv(buf, base + fx.Int32(2), fx.BFloat16, 8)
        return lo.shuffle(hi, list(range(16)))

    @flyc.kernel(known_block_size=[32, 1, 1])
    def k_qk(Q: fx.Tensor, K: fx.Tensor, S: fx.Tensor):
        lane = fx.Int32(fx.thread_idx.x)
        g_q = _bv(Q, WMMA_N * D * 2, fx.BFloat16, 8)
        g_k = _bv(K, WMMA_M * D * 2, fx.BFloat16, 8)
        g_s = _bv(S, WMMA_M * WMMA_N * 4, fx.Float32)

        row = lane % fx.Int32(16)
        half = lane // fx.Int32(16)
        v8f32 = fx.Vector.make_type(8, fx.Float32)
        acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        for dt in range_constexpr(NDT):
            a = frag(g_k, row, half, dt)     # K is the A-operand: M = kv
            b = frag(g_q, row, half, dt)     # Q is the B-operand: N = q
            acc = rocdl.wmma_f32_16x16x32_bf16(
                v8f32, _ir(a), _ir(b), acc, reuseA=False, reuseB=False
            ).result
        ov = fx.Vector(acc)
        for si in range_constexpr(8):
            kv = half * fx.Int32(8) + fx.Int32(si)     # accumulator M axis = kv
            _st1(ov[si], g_s, kv * fx.Int32(16) + row, fx.Float32)  # S^T[kv, q]

    @flyc.jit
    def launch(Q: fx.Tensor, K: fx.Tensor, S: fx.Tensor, stream: fx.Stream):
        k_qk(Q, K, S).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    return launch


res = []
for D in (32, 64, 128, 256):
    torch.manual_seed(0)
    q = torch.randn(WMMA_N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(WMMA_M, D, device="cuda", dtype=torch.bfloat16)
    s = torch.full((WMMA_M, WMMA_N), float("nan"), device="cuda", dtype=torch.float32)
    build(D)(q, k, s, torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = k.float() @ q.float().T                   # S^T[kv, q]
    fin = int(torch.isfinite(s).sum())
    db = float(10 * torch.log10(ref.pow(2).mean() /
                                (ref - s).pow(2).mean().clamp_min(1e-30)))
    ok = fin == s.numel() and db >= 40
    print(f"  D={D:4d}  ndt={D//WMMA_K}  isfinite {fin}/{s.numel()}  SQNR {db:7.2f} dB  "
          f"max|err| {float((ref-s).abs().max()):.3e}  {'OK' if ok else 'FAIL'}")
    res.append({"D": D, "ndt": D // WMMA_K, "isfinite": f"{fin}/{s.numel()}",
                "sqnr_db": round(db, 2), "pass": ok})

json.dump(res, open("/tmp/qk_tile.json", "w"), indent=2)
allok = all(r["pass"] for r in res)
print("RESULT:", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 2)
