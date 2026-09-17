"""Does gfx1250's WMMA accumulator feed back as an operand for free, as MFMA's does?

This decides a real cost in dkdv. aiter's gfx942 template contracts dV = P^T . dO and
dK = dS^T . Q over the QUERY index and says P^T / dS^T "come free from the accumulators" --
on MFMA 16x16x16 it packs 4 accumulator fp32 into a v4 bf16 operand IN LANE, no cross-lane
movement (`_pack4_trunc`).

On gfx1250 the two layouts look incompatible on paper:

    accumulator      lane l holds C[(l//16)*8 + si, l%16] for si in 0..7
                     -> ONE column, EIGHT rows
    A-operand        lane l holds A[l%16, (l//16)*8 + (e%8) + (e//8)*16] for e in 0..15
                     -> ONE row, SIXTEEN columns

If that is right, the template's central trick does not port, and dkdv needs an LDS round
trip or a cross-lane transpose for P^T / dS^T -- on its most expensive kernel.

Test: C1 = A @ B, then feed C1's accumulator straight back in-lane as the next A-operand and
compare against the true C1 @ B2. Matching means free; not matching means it is a transpose.
"""
import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, "/tmp/flydsl032"); sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch, flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir
print(f"flydsl {flydsl.__version__}   arch {torch.cuda.get_device_properties(0).gcnArchName}")

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
def k_probe(A: fx.Tensor, B1: fx.Tensor, B2: fx.Tensor, C1o: fx.Tensor, C2o: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    g_a  = _bv(A,  16*32*2, fx.BFloat16, 8)
    g_b1 = _bv(B1, 16*32*2, fx.BFloat16, 8)
    g_b2 = _bv(B2, 16*32*2, fx.BFloat16, 8)
    g_c1 = _bv(C1o, 16*16*4, fx.Float32)
    g_c2 = _bv(C2o, 16*16*4, fx.Float32)
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)
    def frag(buf):
        t = row * fx.Int32(4) + half
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(_ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    c1 = rocdl.wmma_f32_16x16x32_bf16(v8f32, _ir(frag(g_a)), _ir(frag(g_b1)),
                                      _ir(fx.Vector.filled(8, 0.0, fx.Float32)),
                                      reuseA=False, reuseB=False).result
    c1v = fx.Vector(c1)
    for si in range_constexpr(8):
        _st1(c1v[si], g_c1, (half * fx.Int32(8) + fx.Int32(si)) * fx.Int32(16) + row, fx.Float32)
    # THE QUESTION: pack the 8 accumulator fp32 in-lane into a v16 bf16 A-operand.
    # 8 values, 16 slots -- duplicate into both halves, which is the most charitable
    # in-lane reading of "free". If the layouts matched this would reproduce C1 @ B2.
    packed = fx.Vector.from_elements([c1v[i % 8].to(fx.BFloat16) for i in range(16)],
                                     dtype=fx.BFloat16)
    c2 = rocdl.wmma_f32_16x16x32_bf16(v8f32, _ir(packed), _ir(frag(g_b2)),
                                      _ir(fx.Vector.filled(8, 0.0, fx.Float32)),
                                      reuseA=False, reuseB=False).result
    c2v = fx.Vector(c2)
    for si in range_constexpr(8):
        _st1(c2v[si], g_c2, (half * fx.Int32(8) + fx.Int32(si)) * fx.Int32(16) + row, fx.Float32)

@flyc.jit
def launch(A: fx.Tensor, B1: fx.Tensor, B2: fx.Tensor, C1: fx.Tensor, C2: fx.Tensor, stream: fx.Stream):
    k_probe(A, B1, B2, C1, C2).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

torch.manual_seed(0)
A  = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
B1 = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
B2 = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
C1 = torch.full((16,16), float("nan"), device="cuda", dtype=torch.float32)
C2 = torch.full((16,16), float("nan"), device="cuda", dtype=torch.float32)
launch(A, B1, B2, C1, C2, torch.cuda.current_stream()); torch.cuda.synchronize()

ref1 = A.float() @ B1.float().T
def db(r, g): return float(10*torch.log10(r.pow(2).mean()/(r-g).pow(2).mean().clamp_min(1e-30)))
print(f"  stage 1 (sanity)            SQNR {db(ref1, C1):7.2f} dB")
# If the accumulator were operand-shaped, C2 would be C1[16x32-ish] @ B2 -- it cannot even
# be well-typed (C1 is 16x16, the operand slot is 16x32), which is the point. Compare
# against the only interpretation that would make "free" true.
ref2 = C1.float() @ B2.float().T[:16, :]        # best-case charitable reading
print(f"  stage 2 vs 'free' reading   SQNR {db(ref2, C2):7.2f} dB")
free = db(ref2, C2) >= 40
print("\nVERDICT:", "accumulator IS operand-shaped (free)" if free
      else "accumulator is NOT operand-shaped -- P^T/dS^T need a transpose on gfx1250")
json.dump({"stage1_db": round(db(ref1,C1),2), "stage2_db": round(db(ref2,C2),2),
           "accumulator_is_operand_shaped": free}, open("/tmp/acc2op.json","w"), indent=2)
