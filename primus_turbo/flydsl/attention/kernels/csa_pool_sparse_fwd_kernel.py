# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""csa_pool_sparse_fwd: V4 CSA sparse-branch forward, head-as-M MFMA + AV-MFMA.

The CSA sparse branch gathers, for each query row m, the top-K pool rows given by
``topk_idxs[b, m, :]`` — INDEPENDENT of the query head. So for a fixed (b, m),
every head attends to the SAME gathered rows: the query-head axis is a genuine
MFMA M-dimension. Both GEMMs run on the matrix cores:

  QK: ``score[key, head] = gathered[key, d] @ q[head, d]^T``     (contract d)
  AV: ``acc[d, head]     = gathered[key, d]^T @ p[key, head]``   (contract key)

Grid: (Sq, cdiv(HEAD_Q, 16), B). One wave per (b, h_block, m). BLOCK_H = 16
(one MFMA M-tile of heads), BLOCK_K = 32 (one MFMA K-tile of gathered keys).

MFMA ``v_mfma_f32_16x16x32`` operand layouts (lane L, group g = L // 16):
  A/B pack (vec8): lane L holds row/col ``L % 16`` and K-subgroup ``g*8 + 0..7``.
  C acc  (vec4):   lane L holds ``C[g*4 + r, L % 16]`` for r in 0..3.

QK C-layout: lane L holds ``score[key = g*4 + r, head = L % 16]`` — the 32 keys of
a BLOCK_K tile are split into two 16-key sub-tiles (s = 0, 1), each key spread
across the 4 lane groups. Online softmax reduces across groups (shuffle_xor 16,
32) to per-head (head = L % 16) running max / sum, exactly as before.

AV: ``head`` is the MFMA N-dimension, so it stays ``L % 16`` in the C output too —
no cross-lane head remap, the per-head l/alpha computed by softmax rescale the
accumulators locally. ``gathered`` is staged transposed into LDS as
``lds_gT[d * BLOCK_K + kloc]`` (contiguous vec8 over kloc for the A operand); ``p``
is transposed C-layout -> B-operand layout via ``lds_p[head * BLOCK_K + kloc]``.
For each of the ``HEAD_DIM / 16`` d-tiles one MFMA contracts all 32 keys, with the
running ``acc[dtile] * alpha`` fed as the MFMA C input.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from primus_turbo.flydsl.attention.kernels.kernels_common import dtype_to_elem_type, mfma_f32_16x16x32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm, math as math_dialect
from flydsl.expr import const_expr  # noqa: E402


KERNEL_NAME = "csa_pool_sparse_fwd_kernel"
_LOG2E = math.log2(math.e)
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_csa_pool_sparse_fwd_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    """Build the CSA sparse-branch forward (head-as-M QK + AV MFMA) launcher.

    Returns ``(out_sparse, lse_sparse)``; empty rows carry ``lse = NEG_INF``.
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    BLOCK_H = 16
    BLOCK_K = 32
    assert HEAD_DIM % 32 == 0
    K_CHUNKS = HEAD_DIM // 32       # QK contraction chunks over d
    D_TILES = HEAD_DIM // 16        # AV output d-tiles (MFMA N=16 heads, M=16 dims)
    N_SUB = BLOCK_K // 16           # QK key sub-tiles (2)
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # ---- LDS: gathered^T [d][kloc] then p [head][kloc] ----
    LDS_GT_ELEMS = HEAD_DIM * BLOCK_K   # 512 * 32
    LDS_P_ELEMS = BLOCK_H * BLOCK_K     # 16 * 32
    LDS_TOTAL_ELEMS = LDS_GT_ELEMS + LDS_P_ELEMS

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="csa_pool_sparse_fwd_smem",
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_pool_sparse_fwd_kernel(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        pool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), POOL)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TOPK)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL_ELEMS,)).get()
        c_lds_p_base = arith.index(LDS_GT_ELEMS)

        def _gep(base_p, elem_idx, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(_llvm_ptr_ty(), base_p, [idx_i64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v(base_p, elem_idx, n):
            return _llvm.LoadOp(T.vec(n, f16_ty), _gep(base_p, elem_idx, f16_ty)).result

        def load_i32_scalar(base_p, elem_idx):
            return _llvm.LoadOp(T.i32, _gep(base_p, elem_idx, T.i32)).result

        def store_f16(val, base_p, elem_idx):
            _llvm.StoreOp(val, _gep(base_p, elem_idx, f16_ty))

        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        pid_hblk = arith.index_cast(T.index, gpu.block_idx.y)
        bid = arith.index_cast(T.index, gpu.block_idx.z)
        lane = arith.index_cast(T.index, gpu.thread_idx.x)

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)
        pool_size_v = arith.index_cast(T.index, pool_size)
        pool_size_i32 = pool_size
        if const_expr(hasattr(pool_size_i32, "ir_value")):
            pool_size_i32 = pool_size_i32.ir_value()
        K_topk_i32 = K_topk
        if const_expr(hasattr(K_topk_i32, "ir_value")):
            K_topk_i32 = K_topk_i32.ir_value()

        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)   # group g in 0..3
        lane_mod_16_i32 = arith.index_cast(T.i32, lane_mod_16)
        lane_div_16_i32 = arith.index_cast(T.i32, lane_div_16)
        h_base = pid_hblk * arith.index(BLOCK_H)

        NEG_INF_F = -1.0e30
        c_neg_inf = arith.constant(NEG_INF_F, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_one_f = arith.constant(1.0, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        c_log2e_f = arith.constant(_LOG2E, type=f32_ty)
        zero_mfma_pack = arith.constant_vector(0.0, T.vec(8, f16_ty))
        c_zero_mfma_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)

        # ---- Q B-operand packs: lane L -> head (h_base + L%16), dims chunk. ----
        my_head = h_base + lane_mod_16
        head_ib = arith.cmpi(arith.CmpIPredicate.slt, my_head, arith.index(NUM_HEADS))
        head_safe = arith.select(head_ib, my_head, arith.index(0))
        q_row_base = ((bid * arith.index(NUM_HEADS) + head_safe) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
        q_valid = arith.AndIOp(head_ib, q_active).result
        q_packs = []
        for ck in range_constexpr(K_CHUNKS):
            q_off = q_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
            qp = load_f16_v(q_ptr, q_off, 8)
            q_packs.append(arith.select(q_valid, qp, zero_mfma_pack))

        def group_reduce_max(v):
            cur = v
            for off in (16, 32):
                peer = arith.ArithValue(cur).shuffle_xor(arith.constant(off, type=T.i32), width_i32)
                cur = arith.MaxNumFOp(cur, peer, fastmath=fm_fast).result
            return cur

        def group_reduce_add(v):
            cur = v
            for off in (16, 32):
                peer = arith.ArithValue(cur).shuffle_xor(arith.constant(off, type=T.i32), width_i32)
                cur = arith.AddFOp(cur, peer, fastmath=fm_fast).result
            return cur

        # Validity + pool row base for a global key index kk (i32).
        def key_meta(kk_i32):
            ko = arith.cmpi(arith.CmpIPredicate.sge, kk_i32, K_topk_i32)
            kk_idx = arith.index_cast(T.index, arith.select(ko, arith.constant(0, type=T.i32), kk_i32))
            toff = (bid * seq_len_v + pid_m_safe) * K_topk_v + kk_idx
            idr = load_i32_scalar(topk_ptr, toff)
            ineg = arith.cmpi(arith.CmpIPredicate.slt, idr, arith.constant(0, type=T.i32))
            igp = arith.cmpi(arith.CmpIPredicate.sge, idr, pool_size_i32)
            inv = arith.OrIOp(arith.OrIOp(ko, ineg).result, igp).result
            idx_safe = arith.index_cast(T.index, arith.select(inv, arith.constant(0, type=T.i32), idr))
            prow = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
            return inv, prow

        # State: per-head (this lane's head = L%16), replicated over the 4 groups.
        m_i = c_neg_inf
        l_i = c_zero_f
        acc = [c_zero_mfma_acc for _ in range_constexpr(D_TILES)]

        for k_start, carry, results in scf.for_(
            arith.index(0), K_topk_v, arith.index(BLOCK_K), iter_args=[m_i, l_i] + acc,
        ):
            m_i = carry[0]
            l_i = carry[1]
            acc = [carry[2 + d] for d in range_constexpr(D_TILES)]
            k_start_i32 = arith.index_cast(T.i32, k_start)

            gpu.barrier()  # protect LDS from the previous iteration's AV reads

            # ---- QK: two 16-key sub-tiles -> owned scores per (s, r) ----
            # The gathered A-operand packs loaded here are ALSO scattered into the
            # LDS gathered^T buffer, so the AV MFMA reuses them without a second
            # HBM read. lds_gT[d * BLOCK_K + kloc], kloc = s*16 + L%16.
            owned_scores = []  # [s][r] scaled+masked score, key = s*16 + g*4 + r
            for s in range_constexpr(N_SUB):
                # A-operand: gathered key rows. lane L -> key (s*16 + L%16), dims chunk.
                key_pos_i32 = arith.AddIOp(
                    arith.AddIOp(k_start_i32, arith.constant(s * 16, type=T.i32)).result,
                    lane_mod_16_i32).result
                _inv_a, pool_row_base = key_meta(key_pos_i32)
                kloc_base = arith.index(s * 16) + lane_mod_16
                c_acc = c_zero_mfma_acc
                for ck in range_constexpr(K_CHUNKS):
                    poff = pool_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    a_pack = load_f16_v(pool_ptr, poff, 8)
                    c_acc = mfma_f32_16x16x32(a_pack, q_packs[ck], c_acc, dtype_str)
                    # Scatter to gathered^T: d = ck*32 + g*8 + e, kloc = s*16 + L%16.
                    d_base = arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    for e in range_constexpr(8):
                        gv = vector.extract(a_pack, static_position=[e], dynamic_position=[])
                        lds_idx = (d_base + arith.index(e)) * arith.index(BLOCK_K) + kloc_base
                        vector.store(vector.from_elements(T.vec(1, f16_ty), [gv]), lds, [lds_idx])
                c_regs = [vector.extract(c_acc, static_position=[r], dynamic_position=[]) for r in range_constexpr(4)]
                s_row = []
                for r in range_constexpr(4):
                    krow_i32 = arith.AddIOp(
                        arith.AddIOp(k_start_i32, arith.constant(s * 16, type=T.i32)).result,
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    inv_r, _pr = key_meta(krow_i32)
                    sc = arith.MulFOp(c_regs[r], c_sm_scale_f, fastmath=fm_fast).result
                    sc = arith.select(inv_r, c_neg_inf, sc)
                    s_row.append(sc)
                owned_scores.append(s_row)

            # ---- Online softmax over the 32 keys (8 owned per lane) ----
            m_tile_local = owned_scores[0][0]
            for s in range_constexpr(N_SUB):
                for r in range_constexpr(4):
                    if const_expr(s == 0 and r == 0):
                        continue
                    m_tile_local = arith.MaxNumFOp(m_tile_local, owned_scores[s][r], fastmath=fm_fast).result
            m_tile = group_reduce_max(m_tile_local)
            m_new = arith.MaxNumFOp(m_i, m_tile, fastmath=fm_fast).result
            alpha = arith.ArithValue(arith.MulFOp(arith.SubFOp(m_i, m_new, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)

            p_owned = []
            l_tile_local = c_zero_f
            for s in range_constexpr(N_SUB):
                p_row = []
                for r in range_constexpr(4):
                    pe = arith.ArithValue(arith.MulFOp(arith.SubFOp(owned_scores[s][r], m_new, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)
                    p_row.append(pe)
                    l_tile_local = arith.AddFOp(l_tile_local, pe, fastmath=fm_fast).result
                p_owned.append(p_row)
            l_tile = group_reduce_add(l_tile_local)
            l_i = arith.AddFOp(arith.MulFOp(l_i, alpha, fastmath=fm_fast).result, l_tile, fastmath=fm_fast).result
            m_i = m_new

            # ---- Transpose p into B-operand layout: lds_p[head * BLOCK_K + kloc] ----
            # lane L (group g), head = L%16, owned key kloc = s*16 + g*4 + r.
            for s in range_constexpr(N_SUB):
                for r in range_constexpr(4):
                    kloc = arith.AddIOp(
                        arith.constant(s * 16, type=T.i32),
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    kloc_idx = arith.index_cast(T.index, kloc)
                    p_f16 = arith.trunc_f(f16_ty, p_owned[s][r])
                    lds_pidx = c_lds_p_base + lane_mod_16 * arith.index(BLOCK_K) + kloc_idx
                    vector.store(vector.from_elements(T.vec(1, f16_ty), [p_f16]), lds, [lds_pidx])

            gpu.barrier()  # gathered^T and p fully staged before AV reads

            # ---- AV MFMA: acc[d, head] += gathered^T[d, kloc] @ p[kloc, head] ----
            # A pack (gathered): lane L -> d = dtile*16 + L%16, kloc = g*8 + e.
            # B pack (p):        lane L -> kloc = g*8 + e, head = L%16.
            g_koff = lane_div_16 * arith.index(8)
            b_pidx = c_lds_p_base + lane_mod_16 * arith.index(BLOCK_K) + g_koff
            b_pack = vector.load_op(T.vec(8, f16_ty), lds, [b_pidx])
            alpha_vec = vector.broadcast(T.vec(4, f32_ty), alpha)
            new_acc = []
            for dt in range_constexpr(D_TILES):
                a_d = arith.index(dt * 16) + lane_mod_16
                a_idx = a_d * arith.index(BLOCK_K) + g_koff
                a_pack = vector.load_op(T.vec(8, f16_ty), lds, [a_idx])
                c_in = arith.MulFOp(acc[dt], alpha_vec, fastmath=fm_fast).result
                new_acc.append(mfma_f32_16x16x32(a_pack, b_pack, c_in, dtype_str))
            acc = new_acc

            yield [m_i, l_i] + acc

        m_i = results[0]
        l_i = results[1]
        acc = [results[2 + d] for d in range_constexpr(D_TILES)]

        # ---- Normalize + store. acc[dt][r] = out[d = dt*16 + g*4 + r, head = L%16]. ----
        head_active = arith.AndIOp(head_ib, q_active).result
        empty = arith.cmpf(arith.CmpFPredicate.OEQ, l_i, c_zero_f)
        safe_l = arith.select(empty, c_one_f, l_i)
        inv_l = arith.DivFOp(c_one_f, safe_l, fastmath=fm_fast).result

        _if = scf.IfOp(head_active, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            o_row_base = ((bid * arith.index(NUM_HEADS) + my_head) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            for dt in range_constexpr(D_TILES):
                for r in range_constexpr(4):
                    ov = vector.extract(acc[dt], static_position=[r], dynamic_position=[])
                    ov = arith.MulFOp(ov, inv_l, fastmath=fm_fast).result
                    ov = arith.select(empty, c_zero_f, ov)
                    of16 = arith.trunc_f(f16_ty, ov)
                    d = arith.index(dt * 16) + lane_div_16 * arith.index(4) + arith.index(r)
                    store_f16(of16, o_ptr, o_row_base + d)
            # LSE once per head-row (group 0 owns).
            is_owner = arith.cmpi(arith.CmpIPredicate.eq, lane_div_16, arith.index(0))
            _if2 = scf.IfOp(is_owner, [], has_else=False)
            with ir.InsertionPoint(_if2.then_block):
                ln_l = math_dialect.log(safe_l, fastmath=fm_fast)
                lse_val = arith.AddFOp(m_i, ln_l, fastmath=fm_fast).result
                lse_val = arith.select(empty, c_neg_inf, lse_val)
                lse_off = (bid * arith.index(NUM_HEADS) + my_head) * seq_len_v + pid_m_safe
                lse_off_i32 = arith.index_cast(T.i32, lse_off)
                buffer_ops.buffer_store(lse_val, lse_rsrc, lse_off_i32)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_csa_pool_sparse_fwd(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        n_hblk = (NUM_HEADS + BLOCK_H - 1) // BLOCK_H
        launcher = csa_pool_sparse_fwd_kernel(Q, POOL, TOPK, O, LSE, seq_len, K_topk, pool_size)
        if const_expr(waves_per_eu is not None):
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign")]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")]))
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(grid=(sl_idx, arith.index(n_hblk), bs_idx), block=(BLOCK_SIZE, 1, 1), stream=stream)

    compile_hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_pool_sparse_fwd(*args, **kwargs)

    return _launch
