"""Isolate the one thing dkdv's correctness rests on: the gfx1250 WMMA fragment layout.

`v_wmma_f32_16x16x32_bf16` on wave32: C[16,16] f32 = A[16,32] @ B[32,16] + C.

Layout read off aiter's gfx1250 forward (fmha_b16_buffer_managers.py:1124-1143 builds the
fragments, fmha_fwd_prefill_a16w16_m32x8.py:273-276 documents the accumulator), and A and B
turn out to be symmetric:

  operand fragment, lane l in 0..31, element e in 0..15
      the 16-axis (A's M, B's N) = l % 16
      the 32-axis (contraction K) = (l // 16) * 8 + (e % 8) + (e // 8) * 16
  accumulator, lane l, element si in 0..7
      C[(l // 16) * 8 + si, l % 16]

Both are read off someone else's working kernel, so this file exists to confirm them against
torch on one tile before 700 lines of dkdv are written on top. A wrong operand layout gives a
plausible-looking wrong matrix, not an error.
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
M = N = 16
K = 32


def _buffer_view(tensor, nbytes, fx_dt, vec=1):
    ntile = (1 << 31) // (vec * (fx_dt.width // 8))
    buf = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=fx.Int64(nbytes))
    return fx.Tensor(fx.make_view(fx.get_iter(buf), fx.make_layout((ntile, vec), (vec, 1))))


def _atom(fx_dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * fx_dt.width), fx_dt)


def _ldv(buf, tile, fx_dt, vec):
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx_dt)
    fx.copy_atom_call(_atom(fx_dt, vec), fx.slice(buf, (tile, None)), frag)
    return frag.load()


def _st1(val, buf, idx, fx_dt):
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=fx_dt), frag)
    fx.copy_atom_call(_atom(fx_dt, 1), frag, fx.slice(buf, (idx, None)))


@flyc.kernel(known_block_size=[32, 1, 1])
def k_wmma(A: fx.Tensor, Bt: fx.Tensor, C: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    g_a = _buffer_view(A, M * K * 2, fx.BFloat16, 8)
    g_bt = _buffer_view(Bt, N * K * 2, fx.BFloat16, 8)
    g_c = _buffer_view(C, M * N * 4, fx.Float32)

    row = lane % fx.Int32(16)      # A's M index / Bt's N index
    half = lane // fx.Int32(16)    # which 8-wide column group of the 32-axis
    t_lo = row * fx.Int32(K // 8) + half          # cols  half*8 .. +7
    t_hi = t_lo + fx.Int32(2)                     # cols  half*8+16 .. +7

    a_lo = _ldv(g_a, t_lo, fx.BFloat16, 8)
    a_hi = _ldv(g_a, t_hi, fx.BFloat16, 8)
    b_lo = _ldv(g_bt, t_lo, fx.BFloat16, 8)
    b_hi = _ldv(g_bt, t_hi, fx.BFloat16, 8)
    a_frag = a_lo.shuffle(a_hi, list(range(16)))
    b_frag = b_lo.shuffle(b_hi, list(range(16)))

    v8f32 = fx.Vector.make_type(8, fx.Float32)
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    out = rocdl.wmma_f32_16x16x32_bf16(
        v8f32, _ir(a_frag), _ir(b_frag), _ir(acc), reuseA=False, reuseB=False
    ).result
    ov = fx.Vector(out)
    for si in range_constexpr(8):
        r = half * fx.Int32(8) + fx.Int32(si)
        _st1(ov[si], g_c, r * fx.Int32(16) + row, fx.Float32)


@flyc.jit
def launch(A: fx.Tensor, Bt: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
    k_wmma(A, Bt, C).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


torch.manual_seed(0)
A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
Bt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)          # Bt[n,k] == B[k,n]
C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
launch(A, Bt, C, torch.cuda.current_stream())
torch.cuda.synchronize()

ref = A.float() @ Bt.float().T
fin = int(torch.isfinite(C).sum())
err = (ref - C).pow(2).mean().clamp_min(1e-30)
db = float(10 * torch.log10(ref.pow(2).mean() / err))
print(f"isfinite {fin}/{C.numel()}   SQNR {db:.2f} dB   max|err| {float((ref-C).abs().max()):.4e}")
ok = fin == C.numel() and db >= 40
if not ok:
    print("\nref[:4,:4]\n", ref[:4, :4])
    print("got[:4,:4]\n", C[:4, :4])
json.dump({"isfinite": f"{fin}/{C.numel()}", "sqnr_db": round(db, 2), "pass": ok},
          open("/tmp/wmma_probe.json", "w"), indent=2)
print("RESULT:", "PASS -- the documented layout is correct" if ok else "FAIL")
sys.exit(0 if ok else 2)
