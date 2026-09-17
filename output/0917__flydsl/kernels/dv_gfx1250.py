"""First real slice of dkdv: dV = P^T . dO for one kv tile, on gfx1250.

Chains everything validated so far into one kernel that is checkable end to end:

    S^T[kv,q] = K @ Q^T * scale          (qk_tile_gfx1250: the d-tile accumulation chain)
    P^T[kv,q] = exp(S^T - lse[q])        (fp32 in the accumulator)
    dV[kv,d]  = sum_q P^T[kv,q] dO[q,d]  (contracts over q -- needs BOTH operands transposed)

Both operands of the second GEMM come from `ds_load_tr16_b128` over row-major LDS, which is
the same trick twice:

    A = P^T, M=kv, K=q  <- P staged as [q][kv];  tr16 gives lane l  P[q=(l//16)*8+e, kv=l%16]
    B = dO,   N=d,  K=q <- dO staged as [q][d];  tr16 gives lane l dO[q=(l//16)*8+e, d=l%16]

The P store is one b128 per q-tile: the S^T accumulator hands a lane eight values consecutive
in kv for a fixed q, which is exactly one row-major [q][kv] run.

Scope: one wave, one 16-row kv tile, NQ queries, D head dim, non-causal. Enough to be wrong
in every way that matters and be caught.
"""
import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# flydsl 0.3.2 is required (0.2.4 cannot build aiter's gfx1250 kernels) and is installed
# side by side rather than over the image's 0.2.4, which other things in this container
# still need. Durable path first: /tmp does not survive a container restart, and the
# unattended op-evolve job runs for up to 48h with a supervisor that restarts things.
for _p in ("/home/lihuzhan/.local/flydsl032", "/tmp/flydsl032"):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch, flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr
from flydsl._mlir.dialects import llvm as llvm_dialect
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir
print(f"flydsl {flydsl.__version__}   arch {torch.cuda.get_device_properties(0).gcnArchName}")

KV, NQ, D = 16, 32, 128
NDT = D // 32          # d-tiles contracted by the QK gemm
NQT = NQ // 16         # q-tiles (each a 16-wide WMMA N)
NDO = D // 16          # dV output tiles along d (each a 16-wide WMMA N)
LOG2E = 1.4426950408889634
P_ROW_B = KV * 2       # LDS row stride for P  [q][kv]
DO_ROW_B = D * 2       # LDS row stride for dO [q][d]


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
def k_dv(Q: fx.Tensor, K: fx.Tensor, DO: fx.Tensor, LSE: fx.Tensor,
         DV: fx.Tensor, scale: fx.Float32):
    lane = fx.Int32(fx.thread_idx.x)
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)
    g_q = _bv(Q, NQ * D * 2, fx.BFloat16, 8)
    g_k = _bv(K, KV * D * 2, fx.BFloat16, 8)
    g_do = _bv(DO, NQ * D * 2, fx.BFloat16, 8)
    g_lse = _bv(LSE, NQ * 4, fx.Float32)
    g_dv = _bv(DV, KV * D * 4, fx.Float32)

    smem = fx.SharedAllocator().allocate(NQ * P_ROW_B + NQ * DO_ROW_B)
    lds_p = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_do = lds_p + fx.Int32(NQ * P_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    # stage dO row-major [q][d] into LDS
    for j in range_constexpr(NQ * D // 8 // 32):
        t = lane + fx.Int32(j * 32)
        llvm_dialect.store(fx.as_ir_value(_ldv(g_do, t, fx.BFloat16, 8)),
                           create_llvm_ptr(lds_do + t * fx.Int32(16), address_space=3))

    def frag(buf, base_row, dt):
        t = (base_row + row) * fx.Int32(D // 8) + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    # S^T = K @ Q^T per q-tile, then P^T = exp(S^T*scale - lse), stored as P[q][kv]
    for qt in range_constexpr(NQT):
        acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
        for dt in range_constexpr(NDT):
            acc = rocdl.wmma_f32_16x16x32_bf16(
                v8f, _ir(frag(g_k, fx.Int32(0), dt)),
                _ir(frag(g_q, fx.Int32(qt * 16), dt)), acc,
                reuseA=False, reuseB=False).result
        av = fx.Vector(acc)
        q_glob = fx.Int32(qt * 16) + row                       # this lane's query
        lse_q = _ld1(g_lse, q_glob, fx.Float32)
        # accumulator si -> kv = half*8 + si, all for the SAME q -> one contiguous [q][kv] run
        pv = fx.Vector.from_elements(
            [_exp2((av[si] * scale - lse_q) * fx.Float32(LOG2E)).to(fx.BFloat16)
             for si in range(8)], dtype=fx.BFloat16)
        llvm_dialect.store(fx.as_ir_value(pv),
                           create_llvm_ptr(lds_p + q_glob * fx.Int32(P_ROW_B)
                                           + half * fx.Int32(8 * 2), address_space=3))
    fx.barrier()

    # both operands via tr16 over row-major LDS
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    p_base = lds_p + lane_r * fx.Int32(P_ROW_B) + lane_c * fx.Int32(2)
    a_frag = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(p_base, address_space=3))
                       ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
        v8b, create_llvm_ptr(p_base + fx.Int32(16 * P_ROW_B), address_space=3))), list(range(16)))

    for dtile in range_constexpr(NDO):
        do_base = (lds_do + lane_r * fx.Int32(DO_ROW_B)
                   + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2))
        b_frag = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(do_base, address_space=3))
                           ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
            v8b, create_llvm_ptr(do_base + fx.Int32(16 * DO_ROW_B), address_space=3))),
            list(range(16)))
        out = rocdl.wmma_f32_16x16x32_bf16(
            v8f, _ir(a_frag), _ir(b_frag), _ir(fx.Vector.filled(8, 0.0, fx.Float32)),
            reuseA=False, reuseB=False).result
        ov = fx.Vector(out)
        for si in range_constexpr(8):
            kv = half * fx.Int32(8) + fx.Int32(si)
            _st1(ov[si], g_dv, kv * fx.Int32(D) + fx.Int32(dtile * 16) + row, fx.Float32)


@flyc.jit
def launch(Q, K, DO, LSE, DV, scale: fx.Float32, stream: fx.Stream):
    k_dv(Q, K, DO, LSE, DV, scale).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


torch.manual_seed(0)
q = torch.randn(NQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(KV, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(NQ, D, device="cuda", dtype=torch.bfloat16)
scale = D ** -0.5
s = (q.float() @ k.float().T) * scale                 # [NQ, KV]
lse = torch.logsumexp(s, dim=-1)                      # [NQ]
dv = torch.full((KV, D), float("nan"), device="cuda", dtype=torch.float32)
launch(q, k, do, lse.contiguous(), dv, scale, torch.cuda.current_stream())
torch.cuda.synchronize()

p = torch.exp(s - lse[:, None])                        # [NQ, KV]
ref = p.T.to(torch.bfloat16).float() @ do.float()      # dV[kv, d]; P truncated as the kernel does
fin = int(torch.isfinite(dv).sum())
db = float(10 * torch.log10(ref.pow(2).mean() / (ref - dv).pow(2).mean().clamp_min(1e-30)))
print(f"isfinite {fin}/{dv.numel()}   SQNR {db:.2f} dB   max|err| {float((ref-dv).abs().max()):.3e}")
ok = fin == dv.numel() and db >= 40
print("RESULT:", "PASS -- dV = P^T . dO works on gfx1250" if ok else "FAIL")
json.dump({"isfinite": f"{fin}/{dv.numel()}", "sqnr_db": round(db, 2), "pass": ok},
          open("/tmp/dv_kernel.json", "w"), indent=2)
sys.exit(0 if ok else 2)
