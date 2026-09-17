"""gfx1250 FlyDSL flash-attention backward -- the three kernels, batched and GQA-aware.

Copied from Primus-Turbo/output/0917__flydsl/kernels/ (see PROVENANCE.md) and extended
with exactly what those bring-up files did not have: a batch/head grid, GQA, and -- for
dq -- a runtime kv loop and the causal mask. Nothing else is changed; the data path, the
fragment layouts and the LDS staging are the validated originals.

    delta[b,h,s] = sum_d dO*O                              k_delta_bshd  (odo, verbatim)
    dV[kv,d] = sum_q P^T dO ; dK[kv,d] = sum_q dS^T Q      k_dkdv        (+ b/h/GQA)
    dQ[q,d]  = sum_kv dS K                                 k_dq          (+ kv loop, causal)

Layouts: q/o/do [B, Sq, Hq, D] bf16; k/v [B, Skv, Hkv, D] bf16; lse/delta [B, Hq, Sq] fp32
natural log; dq [B, Sq, Hq, D] fp32; dk/dv [B, Skv, Hkv, D] fp32.

GQA is reduced INSIDE k_dkdv: one workgroup owns a kv tile of one kv head and streams every
query tile of all `G = Hq/Hkv` q heads that share it, accumulating into the same registers.
So every output element is written exactly once, with no atomics and no host reduction --
the determinism gate in op.config.determinism holds by construction.

Divisibility is asserted, not handled: Sq % 16 == 0, Skv % 32 == 0. The bring-up files
assert the same way and every shape in op.shape divides.
"""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

# _env sets sys.path (flydsl 0.3.2, aiter) and TORCH_BLAS_PREFER_HIPBLASLT. It must run
# before flydsl/aiter are imported, and it is loaded by path so two implementation
# directories in one process never share a module object.
_n = f"_env__{abs(hash(str(_pl.Path(__file__).resolve().parent)))}"
_sp = _ilu.spec_from_file_location(_n, _pl.Path(__file__).resolve().parent / "_env.py")
_env = _ilu.module_from_spec(_sp)
_sys.modules[_n] = _env
_sp.loader.exec_module(_env)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import range_constexpr, rocdl

from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir

D = 128
DV8 = D // 8                 # vec8 tiles per row of one head
NDT = D // 32                # WMMA k-steps to contract 128
NDO = D // 16                # 16-wide output tiles across d
WAVE = 32                    # gfx1250 dispatches wave32; gfx942's odo says 64 here
LOG2E = 1.4426950408889634
NEG = -3.0e38
S_ROW_B = 16 * 2             # LDS row stride, bytes, for a [q][kv] 16-wide tile
X_ROW_B = D * 2              # LDS row stride, bytes, for a [q][d] tile

DELTA_THREADS = 256
LANES_PER_ROW = D // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


# ---- shared helpers, verbatim from the bring-up files ------------------------------
def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))


def _atom(dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)


def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
    return f.load()


def _ld1(buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.copy_atom_call(_atom(dt, 1), fx.slice(buf, (idx, None)), f)
    return f.load()[0]


def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


# ==================================================================== odo ==========
@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor,
                 S: fx.Int32, H: fx.Int32, n_rows: fx.Int32):
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32. Verbatim from odo."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    g_do = _bv(DO, n_rows * (D * 2), fx.BFloat16, 8)
    g_o = _bv(O, n_rows * (D * 2), fx.BFloat16, 8)
    g_delta = _bv(DEL, n_rows * 4, fx.Float32)

    tile = bid * (ROWS_DELTA * DV8) + tid
    do_vecs = [_ldv(g_do, tile + u * (ROWS_PER_PASS * DV8), fx.BFloat16, 8).ir_value()
               for u in range_constexpr(PASSES_PER_WG)]
    o_vecs = [_ldv(g_o, tile + u * (ROWS_PER_PASS * DV8), fx.BFloat16, 8).ir_value()
              for u in range_constexpr(PASSES_PER_WG)]

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(4):
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)
        idx = bid * fx.Int32(ROWS_DELTA) + u * fx.Int32(ROWS_PER_PASS) + row_in_group
        ok = (lane_in_row == fx.Int32(0)) & (idx < n_rows)
        b = idx // (S * H)
        rem = idx - b * (S * H)
        s_ = rem // H
        h = rem - s_ * H
        dst = (b * H + h) * S + s_
        _st1(acc, g_delta, ok.select(dst, n_rows), fx.Float32)


@flyc.jit
def launch_delta(DO, O, DEL, S: fx.Int32, H: fx.Int32, n_rows: fx.Int32,
                 nblk: fx.Int32, stream: fx.Stream):
    k_delta_bshd(DO, O, DEL, S, H, n_rows).launch(
        grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream)


# =================================================================== dkdv =========
@flyc.kernel(known_block_size=[32, 1, 1])
def k_dkdv(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
           LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
           scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
           G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32):
    """One wave owns a 16-row kv tile of one kv head and streams every (q head, q tile).

    grid = (Skv/16, Hkv, B). The accumulators persist across the G q heads that share this
    kv head, so the GQA reduction happens in registers -- atomic-free, written once.
    """
    lane = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)              # kv tile
    hkv = fx.Int32(fx.block_idx.y)              # kv head
    bat = fx.Int32(fx.block_idx.z)              # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(16)

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dv = _bv(DV_, 1 << 30, fx.Float32)
    g_dk = _bv(DK, 1 << 30, fx.Float32)

    rs_q = Hq * fx.Int32(DV8)                   # vec8 tiles between consecutive q rows
    rs_kv = Hkv * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    # 32 query rows staged, not 16: the output GEMM's contraction is over queries, and
    # the WMMA contracts 32. Staging only 16 was padding half of every dK/dV matrix
    # instruction with zeros. LDS goes 9216 B -> 18432 B, still far inside 320 KB.
    smem = fx.SharedAllocator().allocate(2 * 32 * S_ROW_B + 2 * 32 * X_ROW_B)
    lds_p = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_ds = lds_p + fx.Int32(32 * S_ROW_B)
    lds_do = lds_ds + fx.Int32(32 * S_ROW_B)
    lds_q = lds_do + fx.Int32(32 * X_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    # K and V fragments for this kv tile are invariant over every query and every q head.
    kf = [gfrag(g_k, base_kv, rs_kv, kv0, dt) for dt in range(NDT)]
    vf = [gfrag(g_v, base_kv, rs_kv, kv0, dt) for dt in range(NDT)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    nqt2 = nqt // fx.Int32(2)                   # query tiles come in PAIRS now
    # CAUSAL TILE SKIP (r1.i1.g01), in PAIR units: this workgroup's kv tile
    # [kv0, kv0+16) is untouched by every query pair below (kv0 - cshift) // 32.
    _c = kv0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    @flyc.jit
    def qloop(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            gh = ii // nqp_eff                   # which q head inside the GQA group
            qt = qp_start + (ii - gh * nqp_eff)  # which PAIR of query tiles (skip-shifted)
            qh = hkv * G + gh
            base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
            base_l = (bat * Hq + qh) * Sq
            q0 = qt * fx.Int32(32)
            acc = list(carried)

            # stage Q and dO for 32 query rows: 32 rows x D, 16 b128 per lane at D=128
            for j in range_constexpr(32 * D // 8 // 32):
                t = lane + fx.Int32(j * 32)
                r_ = t // fx.Int32(DV8)
                c_ = t - r_ * fx.Int32(DV8)
                src = base_q + (q0 + r_) * rs_q + c_
                llvm_dialect.store(fx.as_ir_value(_ldv(g_do, src, fx.BFloat16, 8)),
                                   create_llvm_ptr(lds_do + t * fx.Int32(16),
                                                   address_space=3))
                llvm_dialect.store(fx.as_ir_value(_ldv(g_q, src, fx.BFloat16, 8)),
                                   create_llvm_ptr(lds_q + t * fx.Int32(16),
                                                   address_space=3))
            # S and P for each 16-query half. This half of the work does NOT shrink: it
            # already contracts a full 32 over d. Only the dK/dV GEMM below was padded.
            for hh in range_constexpr(2):
                qh0 = q0 + fx.Int32(hh * 16)
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT):
                    qfr = gfrag(g_q, base_q, rs_q, qh0, dt)
                    dfr = gfrag(g_do, base_q, rs_q, qh0, dt)
                    s_acc = rocdl.wmma_f32_16x16x32_bf16(v8f, _ir(kf[dt]), _ir(qfr), s_acc,
                                                         reuseA=False, reuseB=False).result
                    p_acc = rocdl.wmma_f32_16x16x32_bf16(v8f, _ir(vf[dt]), _ir(dfr), p_acc,
                                                         reuseA=False, reuseB=False).result
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                q_glob = qh0 + row
                lse_q = _ld1(g_lse, base_l + q_glob, fx.Float32)
                del_q = _ld1(g_del, base_l + q_glob, fx.Float32)
                # causal, BOTTOM-RIGHT: query q attends kv <= q + (Skv - Sq). cshift
                # carries that offset so unequal seqlens are right; 0 when equal.
                masked = [
                    ((kv0 + half * fx.Int32(8) + fx.Int32(si) > q_glob + cshift)
                     & (causal != fx.Int32(0))).select(fx.Float32(NEG), sv[si] * scale)
                    for si in range(8)
                ]
                pf = [_exp2((masked[si] - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
                p_l = [x.to(fx.BFloat16) for x in pf]
                ds_l = [(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16)
                        for si in range(8)]
                off = (fx.Int32(hh * 16) + row) * fx.Int32(S_ROW_B) + half * fx.Int32(16)
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(p_l, dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_p + off, address_space=3))
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(ds_l, dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ds + off, address_space=3))
            fx.barrier()

            # 32 queries staged, so the contraction is FULL: rows [lane_r] and
            # [lane_r+16] concatenate in lane into a v16 operand and no WMMA lane is
            # multiplied by a padding zero. This is exactly the idiom k_dq already uses
            # for its K operand, applied to the four dK/dV operands.
            def tr(base, rowb):
                return fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                    fx.Vector(rocdl.ds_load_tr16_b128(
                        v8b, create_llvm_ptr(base + fx.Int32(16 * rowb),
                                             address_space=3))), list(range(16)))

            a_p = tr(lds_p + lane_r * fx.Int32(S_ROW_B) + lane_c * fx.Int32(2), S_ROW_B)
            a_ds = tr(lds_ds + lane_r * fx.Int32(S_ROW_B) + lane_c * fx.Int32(2), S_ROW_B)
            new = []
            for dtile in range_constexpr(NDO):
                c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
                b_do = tr(lds_do + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)
                b_q = tr(lds_q + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)
                new.append(rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_p), _ir(b_do), acc[dtile],
                    reuseA=False, reuseB=False).result)
                new.append(rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_q), acc[NDO + dtile],
                    reuseA=False, reuseB=False).result)
            fx.barrier()
            ordered = [new[2 * i] for i in range(NDO)] + [new[2 * i + 1] for i in range(NDO)]
            final = yield ordered
        return final

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(2 * NDO)]
    out = qloop(init, G * nqp_eff)
    base_o = bat * Skv * Hkv * fx.Int32(D) + hkv * fx.Int32(D)
    for dtile in range_constexpr(NDO):
        ov = fx.Vector(out[dtile])
        ok_ = fx.Vector(out[NDO + dtile])
        for si in range_constexpr(8):
            kv = kv0 + half * fx.Int32(8) + fx.Int32(si)
            idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row
            _st1(ov[si], g_dv, idx, fx.Float32)
            _st1(ok_[si], g_dk, idx, fx.Float32)


@flyc.jit
def launch_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale: fx.Float32,
                Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G,
           nqt, cshift, causal).launch(
        grid=(nblk, nhkv, nb), block=(32, 1, 1), stream=stream)


# ===================================================================== dq =========
KV_STEP = 32                 # two 16-kv tiles: exactly one WMMA contraction of dS
NKT = KV_STEP // 16


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
         LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
         scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
         G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32):
    """One wave owns 16 queries of one q head and streams every 32-key block.

    grid = (Sq/16, Hq, B). The A operand of the dQ GEMM comes free: the two kv-tile dS^T
    accumulators concatenate in lane into one v16, so the only LDS traffic is staging K.
    """
    lane = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)              # query tile
    qh = fx.Int32(fx.block_idx.y)               # q head
    bat = fx.Int32(fx.block_idx.z)              # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0 = bid * fx.Int32(16)
    hkv = qh // G

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq = _bv(DQ, 1 << 30, fx.Float32)

    rs_q = Hq * fx.Int32(DV8)
    rs_kv = Hkv * fx.Int32(DV8)
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(KV_STEP * X_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    # Q, dO, lse and delta are invariant over the whole kv loop: hoist them.
    qf = [gfrag(g_q, base_q, rs_q, q0, dt) for dt in range(NDT)]
    dof = [gfrag(g_do, base_q, rs_q, q0, dt) for dt in range(NDT)]
    q_glob = q0 + row
    lse_q = _ld1(g_lse, base_l + q_glob, fx.Float32)
    del_q = _ld1(g_del, base_l + q_glob, fx.Float32)
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    @flyc.jit
    def kvloop(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            kv0 = fx.Int32(it) * fx.Int32(KV_STEP)
            acc = list(carried)

            for j in range_constexpr(KV_STEP * D // 8 // 32):   # stage K row-major [kv][d]
                t = lane + fx.Int32(j * 32)
                r_ = t // fx.Int32(DV8)
                c_ = t - r_ * fx.Int32(DV8)
                llvm_dialect.store(
                    fx.as_ir_value(_ldv(g_k, base_kv + (kv0 + r_) * rs_kv + c_,
                                        fx.BFloat16, 8)),
                    create_llvm_ptr(lds_k + t * fx.Int32(16), address_space=3))

            ds_halves = []
            for kt in range_constexpr(NKT):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT):
                    kfr = gfrag(g_k, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                    vfr = gfrag(g_v, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                    s_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[dt]), s_acc,
                        reuseA=False, reuseB=False).result
                    p_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[dt]), p_acc,
                        reuseA=False, reuseB=False).result
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                masked = [
                    ((kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                      > q_glob + cshift) & (causal != fx.Int32(0))
                     ).select(fx.Float32(NEG), sv[si] * scale)
                    for si in range(8)
                ]
                pf = [_exp2((masked[si] - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
                ds_halves.append([(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16)
                                  for si in range(8)])
            # FREE: the two kv-tile accumulators concatenate in-lane into the dS A-operand.
            a_ds = fx.Vector.from_elements(ds_halves[0] + ds_halves[1], dtype=fx.BFloat16)
            fx.barrier()   # for the K staging above, not for a_ds

            new = []
            for dtile in range_constexpr(NDO):
                base = (lds_k + lane_r * fx.Int32(X_ROW_B)
                        + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2))
                b_k = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(base, address_space=3))
                ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(base + fx.Int32(16 * X_ROW_B), address_space=3))),
                    list(range(16)))
                new.append(rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_k), acc[dtile],
                    reuseA=False, reuseB=False).result)
            fx.barrier()   # LDS is reused by the next kv block
            final = yield new
        return final

    # CAUSAL TILE SKIP, the mirror of k_dkdv's. Queries [q0, q0+16) reach at most key
    # q0 + 15 + cshift, so every kv block past that is fully masked and contributes
    # exactly zero to dQ. Bit-identical, not an approximation.
    _lim = (q0 + fx.Int32(16) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NDO)]
    out = kvloop(init, nkvt_eff)
    base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for dtile in range_constexpr(NDO):
        ov = fx.Vector(out[dtile])
        for si in range_constexpr(8):
            q_i = q0 + half * fx.Int32(8) + fx.Int32(si)
            _st1(ov[si], g_dq,
                 base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row, fx.Float32)


@flyc.jit
def launch_dq(Q, K, V, DO, LSE, DEL, DQ, scale: fx.Float32,
              Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
              nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
              nblk: fx.Int32, nhq: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dq(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
         nkvt, cshift, causal).launch(
        grid=(nblk, nhq, nb), block=(32, 1, 1), stream=stream)
