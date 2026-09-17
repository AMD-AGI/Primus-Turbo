"""dq on gfx1250: dQ = dS . K, the half where the operand comes free.

dkdv had to stage P^T / dS^T through LDS because its contraction is over the query axis.
dq contracts over kv instead, and dq_operand_probe.py showed the consequence: two
consecutive kv-tile dS^T accumulators CONCATENATE IN LANE into one v16 A-operand, with no
LDS round trip and no barrier for that operand.

    S^T[kv,q]  = K @ Q^T * scale        dP^T[kv,q] = V @ dO^T
    dS^T       = P^T*(dP^T - delta)*scale
    dQ[q,d]    = sum_kv dS[q,kv] K[kv,d]        A = dS  (free from two accumulators)
                                                B = K   (tr16 over K staged [kv][d])

Scope: one wave, 16 queries, KV=32 keys (exactly one WMMA contraction), D=128, non-causal.
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

NQ, KV, D = 16, 32, 128
NDT, NKT, NDO = D // 32, KV // 16, D // 16
LOG2E = 1.4426950408889634
K_ROW_B = D * 2

def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))
def _atom(dt, vec): return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)
def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f); return f.load()
def _ld1(buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.copy_atom_call(_atom(dt, 1), fx.slice(buf, (idx, None)), f); return f.load()[0]
def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))
def _exp2(x): return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))

@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
         LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor, scale: fx.Float32):
    lane = fx.Int32(fx.thread_idx.x)
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)
    g_q = _bv(Q, NQ * D * 2, fx.BFloat16, 8); g_k = _bv(K, KV * D * 2, fx.BFloat16, 8)
    g_v = _bv(V, KV * D * 2, fx.BFloat16, 8); g_do = _bv(DO, NQ * D * 2, fx.BFloat16, 8)
    g_lse = _bv(LSE, NQ * 4, fx.Float32); g_del = _bv(DEL, NQ * 4, fx.Float32)
    g_dq = _bv(DQ, NQ * D * 4, fx.Float32)

    smem = fx.SharedAllocator().allocate(KV * K_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16); v8f = fx.Vector.make_type(8, fx.Float32)

    for j in range_constexpr(KV * D // 8 // 32):       # stage K row-major [kv][d]
        t = lane + fx.Int32(j * 32)
        llvm_dialect.store(fx.as_ir_value(_ldv(g_k, t, fx.BFloat16, 8)),
                           create_llvm_ptr(lds_k + t * fx.Int32(16), address_space=3))

    def frag(buf, base_row, dt):
        t = (base_row + row) * fx.Int32(D // 8) + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    lse_q = _ld1(g_lse, row, fx.Float32)
    del_q = _ld1(g_del, row, fx.Float32)
    ds_halves = []
    for kt in range_constexpr(NKT):
        s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        for dt in range_constexpr(NDT):
            s_acc = rocdl.wmma_f32_16x16x32_bf16(
                v8f, _ir(frag(g_k, fx.Int32(kt * 16), dt)),
                _ir(frag(g_q, fx.Int32(0), dt)), s_acc, reuseA=False, reuseB=False).result
            p_acc = rocdl.wmma_f32_16x16x32_bf16(
                v8f, _ir(frag(g_v, fx.Int32(kt * 16), dt)),
                _ir(frag(g_do, fx.Int32(0), dt)), p_acc, reuseA=False, reuseB=False).result
        sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
        pf = [_exp2((sv[si] * scale - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
        ds_halves.append([(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16)
                          for si in range(8)])
    # FREE: the two kv-tile accumulators concatenate in-lane into the dS A-operand.
    a_ds = fx.Vector.from_elements(ds_halves[0] + ds_halves[1], dtype=fx.BFloat16)
    fx.barrier()   # for the K staging above, not for a_ds

    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    for dtile in range_constexpr(NDO):
        base = (lds_k + lane_r * fx.Int32(K_ROW_B)
                + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2))
        b_k = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))
            ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * K_ROW_B), address_space=3))),
                list(range(16)))
        out = fx.Vector(rocdl.wmma_f32_16x16x32_bf16(
            v8f, _ir(a_ds), _ir(b_k), _ir(fx.Vector.filled(8, 0.0, fx.Float32)),
            reuseA=False, reuseB=False).result)
        for si in range_constexpr(8):
            q_i = half * fx.Int32(8) + fx.Int32(si)
            _st1(out[si], g_dq, q_i * fx.Int32(D) + fx.Int32(dtile * 16) + row, fx.Float32)

@flyc.jit
def launch(Q, K, V, DO, LSE, DEL, DQ, scale: fx.Float32, stream: fx.Stream):
    k_dq(Q, K, V, DO, LSE, DEL, DQ, scale).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

torch.manual_seed(0)
q = torch.randn(NQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(KV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(KV, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(NQ, D, device="cuda", dtype=torch.bfloat16)
scale = D ** -0.5
s = (q.float() @ k.float().T) * scale
lse = torch.logsumexp(s, -1)
p = torch.exp(s - lse[:, None])
o = p.to(torch.bfloat16).float() @ v.float()
delta = (do.float() * o).sum(-1)
ds = p * (do.float() @ v.float().T - delta[:, None]) * scale
ref_dq = ds.to(torch.bfloat16).float() @ k.float()

dq = torch.full((NQ, D), float("nan"), device="cuda", dtype=torch.float32)
launch(q, k, v, do, lse.contiguous(), delta.contiguous(), dq, scale, torch.cuda.current_stream())
torch.cuda.synchronize()
fin = int(torch.isfinite(dq).sum())
db = float(10*torch.log10(ref_dq.pow(2).mean()/(ref_dq-dq).pow(2).mean().clamp_min(1e-30)))
print(f"  dQ  isfinite {fin}/{dq.numel()}   SQNR {db:7.2f} dB   "
      f"max|err| {float((ref_dq-dq).abs().max()):.3e}")
ok = fin == dq.numel() and db >= 40
print("RESULT:", "PASS -- dq works, operand free from the accumulators" if ok else "FAIL")
json.dump({"dq_db": round(db,2), "pass": ok}, open("/tmp/dq_kernel.json","w"), indent=2)
sys.exit(0 if ok else 2)
