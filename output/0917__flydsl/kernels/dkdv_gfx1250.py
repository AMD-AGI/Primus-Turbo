"""dkdv on gfx1250: dV and dK in one kernel.

dK is structurally dV with substitutions, which is why they belong in one pass:

    S^T[kv,q]  = K @ Q^T * scale          dP^T[kv,q] = V @ dO^T      <- same GEMM shape
    P^T        = exp(S^T - lse[q])        dS^T = P^T*(dP^T - delta[q])*scale
    dV[kv,d]   = sum_q P^T  dO[q,d]       dK[kv,d] = sum_q dS^T Q[q,d]  <- same GEMM shape

Four operands, all through the same `ds_load_tr16_b128`-over-row-major-LDS path:
P and dS staged [q][kv], dO and Q staged [q][d]. The P and dS stores are one b128 each
per q-tile, because the accumulators hand a lane eight values consecutive in kv at fixed q.

Scope: one wave, one 16-row kv tile, NQ queries, D head dim, non-causal, no GQA, no tail.
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

KV, NQ, D = 16, 32, 128
NDT, NQT, NDO = D // 32, NQ // 16, D // 16
LOG2E = 1.4426950408889634
S_ROW_B = KV * 2          # LDS row stride for P / dS, both [q][kv]
X_ROW_B = D * 2           # LDS row stride for dO / Q,  both [q][d]

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
def k_dkdv(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
           LSE: fx.Tensor, DEL: fx.Tensor, DV: fx.Tensor, DK: fx.Tensor,
           scale: fx.Float32):
    lane = fx.Int32(fx.thread_idx.x)
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)
    g_q  = _bv(Q,  NQ * D * 2, fx.BFloat16, 8); g_k  = _bv(K,  KV * D * 2, fx.BFloat16, 8)
    g_v  = _bv(V,  KV * D * 2, fx.BFloat16, 8); g_do = _bv(DO, NQ * D * 2, fx.BFloat16, 8)
    g_lse = _bv(LSE, NQ * 4, fx.Float32); g_del = _bv(DEL, NQ * 4, fx.Float32)
    g_dv = _bv(DV, KV * D * 4, fx.Float32); g_dk = _bv(DK, KV * D * 4, fx.Float32)

    smem = fx.SharedAllocator().allocate(2 * NQ * S_ROW_B + 2 * NQ * X_ROW_B)
    lds_p  = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_ds = lds_p + fx.Int32(NQ * S_ROW_B)
    lds_do = lds_ds + fx.Int32(NQ * S_ROW_B)
    lds_q  = lds_do + fx.Int32(NQ * X_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16); v8f = fx.Vector.make_type(8, fx.Float32)

    for j in range_constexpr(NQ * D // 8 // 32):          # stage dO and Q row-major
        t = lane + fx.Int32(j * 32)
        llvm_dialect.store(fx.as_ir_value(_ldv(g_do, t, fx.BFloat16, 8)),
                           create_llvm_ptr(lds_do + t * fx.Int32(16), address_space=3))
        llvm_dialect.store(fx.as_ir_value(_ldv(g_q, t, fx.BFloat16, 8)),
                           create_llvm_ptr(lds_q + t * fx.Int32(16), address_space=3))

    def frag(buf, base_row, dt):
        t = (base_row + row) * fx.Int32(D // 8) + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    for qt in range_constexpr(NQT):
        s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        for dt in range_constexpr(NDT):
            qf = _ir(frag(g_q, fx.Int32(qt * 16), dt))
            df = _ir(frag(g_do, fx.Int32(qt * 16), dt))
            s_acc = rocdl.wmma_f32_16x16x32_bf16(
                v8f, _ir(frag(g_k, fx.Int32(0), dt)), qf, s_acc,
                reuseA=False, reuseB=False).result          # S^T  = K @ Q^T
            p_acc = rocdl.wmma_f32_16x16x32_bf16(
                v8f, _ir(frag(g_v, fx.Int32(0), dt)), df, p_acc,
                reuseA=False, reuseB=False).result          # dP^T = V @ dO^T
        sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
        q_glob = fx.Int32(qt * 16) + row
        lse_q = _ld1(g_lse, q_glob, fx.Float32)
        del_q = _ld1(g_del, q_glob, fx.Float32)
        # List comprehensions, not a statement-level `for`: the AST rewriter turns the
        # latter into an scf.for with carried variables and rejects a growing list.
        p_f32 = [_exp2((sv[si] * scale - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
        p_list = [x.to(fx.BFloat16) for x in p_f32]
        ds_list = [(p_f32[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16) for si in range(8)]
        off = q_glob * fx.Int32(S_ROW_B) + half * fx.Int32(16)
        llvm_dialect.store(fx.as_ir_value(fx.Vector.from_elements(p_list, dtype=fx.BFloat16)),
                           create_llvm_ptr(lds_p + off, address_space=3))
        llvm_dialect.store(fx.as_ir_value(fx.Vector.from_elements(ds_list, dtype=fx.BFloat16)),
                           create_llvm_ptr(lds_ds + off, address_space=3))
    fx.barrier()

    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    def tr_pair(base, row_bytes):
        return fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))
            ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * row_bytes), address_space=3))),
                list(range(16)))

    a_p  = tr_pair(lds_p  + lane_r * fx.Int32(S_ROW_B) + lane_c * fx.Int32(2), S_ROW_B)
    a_ds = tr_pair(lds_ds + lane_r * fx.Int32(S_ROW_B) + lane_c * fx.Int32(2), S_ROW_B)

    for dtile in range_constexpr(NDO):
        c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
        b_do = tr_pair(lds_do + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)
        b_q  = tr_pair(lds_q  + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)
        z = lambda: _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        o_dv = fx.Vector(rocdl.wmma_f32_16x16x32_bf16(
            v8f, _ir(a_p), _ir(b_do), z(), reuseA=False, reuseB=False).result)
        o_dk = fx.Vector(rocdl.wmma_f32_16x16x32_bf16(
            v8f, _ir(a_ds), _ir(b_q), z(), reuseA=False, reuseB=False).result)
        for si in range_constexpr(8):
            idx = (half * fx.Int32(8) + fx.Int32(si)) * fx.Int32(D) + fx.Int32(dtile * 16) + row
            _st1(o_dv[si], g_dv, idx, fx.Float32)
            _st1(o_dk[si], g_dk, idx, fx.Float32)

@flyc.jit
def launch(Q, K, V, DO, LSE, DEL, DV, DK, scale: fx.Float32, stream: fx.Stream):
    k_dkdv(Q, K, V, DO, LSE, DEL, DV, DK, scale).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

torch.manual_seed(0)
mk = lambda n: torch.randn(n, D, device="cuda", dtype=torch.bfloat16)
q, k, v, do = mk(NQ), mk(KV), mk(KV), mk(NQ)
scale = D ** -0.5
s = (q.float() @ k.float().T) * scale
lse = torch.logsumexp(s, -1)
p = torch.exp(s - lse[:, None])
o = p.to(torch.bfloat16).float() @ v.float()
delta = (do.float() * o).sum(-1)
dp = do.float() @ v.float().T
ds = p * (dp - delta[:, None]) * scale
ref_dv = p.T.to(torch.bfloat16).float() @ do.float()
ref_dk = ds.T.to(torch.bfloat16).float() @ q.float()

dv = torch.full((KV, D), float("nan"), device="cuda", dtype=torch.float32)
dk = dv.clone()
launch(q, k, v, do, lse.contiguous(), delta.contiguous(), dv, dk, scale,
       torch.cuda.current_stream())
torch.cuda.synchronize()

def rep(name, ref, got):
    fin = int(torch.isfinite(got).sum())
    db = float(10*torch.log10(ref.pow(2).mean()/(ref-got).pow(2).mean().clamp_min(1e-30)))
    print(f"  {name}  isfinite {fin}/{got.numel()}   SQNR {db:7.2f} dB   "
          f"max|err| {float((ref-got).abs().max()):.3e}")
    return fin == got.numel() and db >= 40, round(db, 2)
ok_v, db_v = rep("dV", ref_dv, dv)
ok_k, db_k = rep("dK", ref_dk, dk)
ok = ok_v and ok_k
print("RESULT:", "PASS -- dkdv (dV and dK) works on gfx1250" if ok else "FAIL")
json.dump({"dv_db": db_v, "dk_db": db_k, "pass": ok}, open("/tmp/dkdv.json","w"), indent=2)
sys.exit(0 if ok else 2)
