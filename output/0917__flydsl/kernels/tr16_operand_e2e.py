"""Close the loop: LDS + two ds_load_tr16_b128 really do build a WMMA A-operand.

Decoded in tr16.py: `ds_load_tr16_b128` returns, for lane l element e,
`src[row = (l//16)*8 + e, col = l%16]` -- i.e. each lane gets ONE COLUMN of the
source tile, eight rows of it.

That is exactly what dkdv needs, for a reason worth spelling out:

  the A-operand wants   lane l, e -> A[l%16, (l//16)*8 + (e%8) + (e//8)*16]
  storing P as [q][kv]  a tr16 read at row offset 0  gives P[q=(l//16)*8+e, kv=l%16]
                                                      == A[kv=l%16, q=(l//16)*8+e]  for e=0..7
                        a second tr16 at row offset +16 gives the operand's e=8..15

and the STORE side is cheap: the accumulator hands lane l eight values that are
consecutive in kv for a fixed q, so writing P[q][kv] row-major is one b128 store, not
eight strided ones. Total per 16x32 operand: one b128 store, one barrier, two tr16 loads.

This validates the read half end to end: a [32 q, 16 kv] tile in LDS, two tr16 loads, one
WMMA, against torch.
"""
import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, "/tmp/flydsl032"); sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch, flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr
from flydsl._mlir.dialects import llvm as llvm_dialect
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir
print(f"flydsl {flydsl.__version__}   arch {torch.cuda.get_device_properties(0).gcnArchName}")

QN, KV = 32, 16                 # P stored as [32 q, 16 kv]
ROW_BYTES = KV * 2

def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))
def _atom(dt, vec): return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)
def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f); return f.load()
def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))

@flyc.kernel(known_block_size=[32, 1, 1])
def k_e2e(P: fx.Tensor, Bt: fx.Tensor, C: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    g_p = _bv(P, QN * KV * 2, fx.BFloat16, 8)
    g_bt = _bv(Bt, 16 * 32 * 2, fx.BFloat16, 8)
    g_c = _bv(C, 16 * 16 * 4, fx.Float32)
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)

    smem = fx.SharedAllocator().allocate(QN * ROW_BYTES)
    lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8 = fx.Vector.make_type(8, fx.BFloat16)
    # stage P into LDS: 512 elements, 32 lanes x 2 b128 stores
    for j in range_constexpr(2):
        t = lane + fx.Int32(j * 32)
        llvm_dialect.store(fx.as_ir_value(_ldv(g_p, t, fx.BFloat16, 8)),
                           create_llvm_ptr(lds0 + t * fx.Int32(16), address_space=3))
    fx.barrier()

    # two transposing reads, 16 source rows apart -> the operand's e=0..7 and e=8..15
    lane_kv = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_d = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    base = lds0 + lane_kv * fx.Int32(ROW_BYTES) + lane_d * fx.Int32(2)
    lo = fx.Vector(rocdl.ds_load_tr16_b128(v8, create_llvm_ptr(base, address_space=3)))
    hi = fx.Vector(rocdl.ds_load_tr16_b128(
        v8, create_llvm_ptr(base + fx.Int32(16 * ROW_BYTES), address_space=3)))
    a_frag = lo.shuffle(hi, list(range(16)))

    t = row * fx.Int32(4) + half
    b_frag = _ldv(g_bt, t, fx.BFloat16, 8).shuffle(
        _ldv(g_bt, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))
    out = rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(8, fx.Float32), _ir(a_frag), _ir(b_frag),
        _ir(fx.Vector.filled(8, 0.0, fx.Float32)), reuseA=False, reuseB=False).result
    ov = fx.Vector(out)
    for si in range_constexpr(8):
        _st1(ov[si], g_c, (half * fx.Int32(8) + fx.Int32(si)) * fx.Int32(16) + row, fx.Float32)

@flyc.jit
def launch(P, Bt, C, stream: fx.Stream):
    k_e2e(P, Bt, C).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

torch.manual_seed(0)
P = torch.randn(QN, KV, device="cuda", dtype=torch.bfloat16)     # [32 q, 16 kv]
Bt = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)    # B[32 q, 16 n] transposed
C = torch.full((16, 16), float("nan"), device="cuda", dtype=torch.float32)
launch(P, Bt, C, torch.cuda.current_stream()); torch.cuda.synchronize()

ref = P.float().T @ Bt.float().T          # A = P^T [16 kv, 32 q];  C = A @ B [16 kv, 16 n]
fin = int(torch.isfinite(C).sum())
db = float(10*torch.log10(ref.pow(2).mean()/(ref-C).pow(2).mean().clamp_min(1e-30)))
print(f"isfinite {fin}/{C.numel()}   SQNR {db:.2f} dB   max|err| {float((ref-C).abs().max()):.3e}")
ok = fin == C.numel() and db >= 40
print("RESULT:", "PASS -- LDS + 2x ds_load_tr16_b128 builds the A-operand" if ok else "FAIL")
json.dump({"isfinite": f"{fin}/{C.numel()}", "sqnr_db": round(db,2), "pass": ok},
          open("/tmp/tr16_e2e.json","w"), indent=2)
sys.exit(0 if ok else 2)
