"""dkdv with a runtime query loop and causal masking -- a real kernel, not a single tile.

One workgroup owns a 16-row kv block and STREAMS every query tile, accumulating dV and dK in
registers. That is the shape the gfx942 template uses and the shape that can be timed.

Runtime loop idiom taken from the gfx1250 forward (fmha_fwd_prefill:1199): a @flyc.jit
generator over fx.Index with the accumulators as carried state.

Causal (bottom-right, equal seqlens): a query at index q attends kv <= q, so the S^T tile is
masked where kv > q.
"""
import os, sys, json, time
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

D = 128
NDT, NDO = D // 32, D // 16
LOG2E = 1.4426950408889634
S_ROW_B = 16 * 2
X_ROW_B = D * 2
NEG = -3.0e38

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
def k_dkdv_loop(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
                LSE: fx.Tensor, DEL: fx.Tensor, DV: fx.Tensor, DK: fx.Tensor,
                scale: fx.Float32, nqt: fx.Int32, causal: fx.Int32):
    lane = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)                 # kv tile index
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(16)
    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8); g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8); g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32); g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dv = _bv(DV, 1 << 30, fx.Float32); g_dk = _bv(DK, 1 << 30, fx.Float32)

    smem = fx.SharedAllocator().allocate(2 * 16 * S_ROW_B + 2 * 16 * X_ROW_B)
    lds_p = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_ds = lds_p + fx.Int32(16 * S_ROW_B)
    lds_do = lds_ds + fx.Int32(16 * S_ROW_B)
    lds_q = lds_do + fx.Int32(16 * X_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16); v8f = fx.Vector.make_type(8, fx.Float32)

    # K and V fragments for this kv tile are loop-invariant: hoist them.
    def gfrag(buf, base_row, dt):
        t = (base_row + row) * fx.Int32(D // 8) + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))
    kf = [gfrag(g_k, kv0, dt) for dt in range(NDT)]
    vf = [gfrag(g_v, kv0, dt) for dt in range(NDT)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    @flyc.jit
    def qloop(state, n):
        final = state
        for qt, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            qi = fx.Int32(qt)
            q0 = qi * fx.Int32(16)
            acc = list(carried)
            # stage Q and dO for this query tile: 16 rows x D, 2 b128 per lane at D=128
            for j in range_constexpr(16 * D // 8 // 32):
                t = lane + fx.Int32(j * 32)
                src = q0 * fx.Int32(D // 8) + t
                llvm_dialect.store(fx.as_ir_value(_ldv(g_do, src, fx.BFloat16, 8)),
                                   create_llvm_ptr(lds_do + t * fx.Int32(16), address_space=3))
                llvm_dialect.store(fx.as_ir_value(_ldv(g_q, src, fx.BFloat16, 8)),
                                   create_llvm_ptr(lds_q + t * fx.Int32(16), address_space=3))
            s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
            p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
            for dt in range_constexpr(NDT):
                qfr = gfrag(g_q, q0, dt); dfr = gfrag(g_do, q0, dt)
                s_acc = rocdl.wmma_f32_16x16x32_bf16(v8f, _ir(kf[dt]), _ir(qfr), s_acc,
                                                     reuseA=False, reuseB=False).result
                p_acc = rocdl.wmma_f32_16x16x32_bf16(v8f, _ir(vf[dt]), _ir(dfr), p_acc,
                                                     reuseA=False, reuseB=False).result
            sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
            q_glob = q0 + row
            lse_q = _ld1(g_lse, q_glob, fx.Float32)
            del_q = _ld1(g_del, q_glob, fx.Float32)
            # causal: this lane's q is q_glob; accumulator si -> kv = kv0 + half*8 + si
            # Causal, bottom-right with equal seqlens: query q attends kv <= q, so mask
            # where kv > q. `cond.select(a, b)` is the scalar ternary (the gfx942 template
            # uses the same call for its bounds-steered store).
            masked = [
                ((kv0 + half * fx.Int32(8) + fx.Int32(si) > q_glob)
                 & (causal != fx.Int32(0))).select(fx.Float32(NEG), sv[si] * scale)
                for si in range(8)
            ]
            pf = [_exp2((masked[si] - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
            p_l = [x.to(fx.BFloat16) for x in pf]
            ds_l = [(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16) for si in range(8)]
            off = row * fx.Int32(S_ROW_B) + half * fx.Int32(16)
            llvm_dialect.store(fx.as_ir_value(fx.Vector.from_elements(p_l, dtype=fx.BFloat16)),
                               create_llvm_ptr(lds_p + off, address_space=3))
            llvm_dialect.store(fx.as_ir_value(fx.Vector.from_elements(ds_l, dtype=fx.BFloat16)),
                               create_llvm_ptr(lds_ds + off, address_space=3))
            fx.barrier()
            # only 16 queries staged, so the operand's upper half is the same tile again:
            # a 16-wide contraction padded to the WMMA's 32. Correct, not yet efficient.
            def tr(base, rb):
                lo = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3)))
                return lo.shuffle(fx.Vector.filled(8, 0.0, fx.BFloat16), list(range(16)))
            a_p = tr(lds_p + lane_r * fx.Int32(S_ROW_B) + lane_c * fx.Int32(2), S_ROW_B)
            a_ds = tr(lds_ds + lane_r * fx.Int32(S_ROW_B) + lane_c * fx.Int32(2), S_ROW_B)
            new = []
            for dtile in range_constexpr(NDO):
                c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
                b_do = tr(lds_do + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)
                b_q = tr(lds_q + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)
                new.append(rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_p), _ir(b_do), acc[dtile], reuseA=False, reuseB=False).result)
                new.append(rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_q), acc[NDO + dtile], reuseA=False, reuseB=False).result)
            fx.barrier()
            ordered = [new[2 * i] for i in range(NDO)] + [new[2 * i + 1] for i in range(NDO)]
            final = yield ordered
        return final

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(2 * NDO)]
    out = qloop(init, nqt)
    for dtile in range_constexpr(NDO):
        ov = fx.Vector(out[dtile]); ok_ = fx.Vector(out[NDO + dtile])
        for si in range_constexpr(8):
            kv = kv0 + half * fx.Int32(8) + fx.Int32(si)
            idx = kv * fx.Int32(D) + fx.Int32(dtile * 16) + row
            _st1(ov[si], g_dv, idx, fx.Float32)
            _st1(ok_[si], g_dk, idx, fx.Float32)

@flyc.jit
def launch(Q, K, V, DO, LSE, DEL, DV, DK, scale: fx.Float32, nqt: fx.Int32,
           causal: fx.Int32, nblk: fx.Int32, stream: fx.Stream):
    k_dkdv_loop(Q, K, V, DO, LSE, DEL, DV, DK, scale, nqt, causal).launch(
        grid=(nblk, 1, 1), block=(32, 1, 1), stream=stream)


import statistics as st, subprocess
def sclk():
    return subprocess.run(["bash","-lc","cat /sys/class/drm/card*/device/pp_dpm_sclk|grep '\\*'"],
                          capture_output=True, text=True, timeout=10).stdout.strip()

print("NOTE: one wave per workgroup, no causal tile skipping, 16-wide contraction padded")
print("      to the WMMA's 32. This is a correctness vehicle; the number is an order of")
print("      magnitude, taken to decide whether the production work is worth doing.\n")
for S in (1024, 2048, 4096):
    torch.manual_seed(0)
    q = torch.randn(S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, D, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(S, D, device="cuda", dtype=torch.bfloat16)
    lse = torch.randn(S, device="cuda", dtype=torch.float32)
    delta = torch.randn(S, device="cuda", dtype=torch.float32)
    dv = torch.empty(S, D, device="cuda", dtype=torch.float32); dk = torch.empty_like(dv)
    scale = D ** -0.5
    stream = torch.cuda.current_stream()
    args = (q, k, v, do, lse, delta, dv, dk, scale, S // 16, 1, S // 16)
    for _ in range(3):
        launch(*args, stream)
    torch.cuda.synchronize()
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    ts = []
    for _ in range(10):
        flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); a.record(); launch(*args, stream); b.record()
        torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    med = st.median(ts)
    executed = 4 * 2 * S * S * D
    print(f"  S={S:5d}  median {med:8.3f} ms   executed {executed/med*1e-9:7.1f} TFLOP/s"
          f"   useful {executed/2/med*1e-9:7.1f} TFLOP/s   grid={S//16} wg x 32 thr")
print(f"\nsclk {sclk()}")
print("reference on this card: aiter's prebuilt ASM backward does the production shape")
print("(b=4 s=8192 hq=32 hkv=8 d=128 causal) in 10.160 ms == about 540 TFLOP/s useful.")
