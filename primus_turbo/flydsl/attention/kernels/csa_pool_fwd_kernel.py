# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""csa_pool_fwd: V4 CSA forward with in-kernel pool gather (FlyDSL per-row).

The sparse branch gathers each top-K key directly from the compressed ``POOL``
``[B, P, D]`` using ``TOPK`` ``[B, Sq, K_topk]`` indices, instead of consuming a
materialised ``GATHERED`` ``[B, Sq, K_topk, D]`` tensor + additive
``SPARSE_MASK``. This avoids materialising the ``[B, Sq, K_topk, D]`` gathered
buffer, saving the gather pass + its memory traffic (design §4.7).

Per the Triton reference (`_csa_attention_pool_fwd_kernel`), a slot is masked
(``NEG_INF``) when its index is out of the K-loop, ``< 0`` (the ``-1`` pad), or
``>= pool_size``; otherwise the row is gathered from ``POOL[bid, idx]`` and used
with the plain ``qk * sm_scale`` score (no additive bias). ``HEAD_GROUP`` is
forced to 1 here (no banking).

Layout: BHLD (Q/K_local/V_local/O all [B, H, Sq, D]).
  - POOL: [B, P, D] (no H dim -- shared across heads).
  - TOPK: [B, Sq, K_topk] int32 (-1 / out-of-range slots masked).
  - sink: [H] fp32 or None.
  - LSE: [B, H, Sq] fp32 (raw-domain m_final + ln(l_final)).
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from primus_turbo.flydsl.attention.kernels.kernels_common import dtype_to_elem_type, mfma_f32_16x16x32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref, scf, fly as _fly, llvm as _llvm, math as math_dialect
from flydsl.expr import const_expr  # noqa: E402


KERNEL_NAME = "csa_pool_fwd_kernel"

_LOG2E = math.log2(math.e)

_LLVM_GEP_DYNAMIC = -2147483648


def _waitcnt_lgkm_0():
    """s_waitcnt lgkmcnt(0): drain LDS/SMEM ops. vmcnt=63, expcnt=7."""
    val = 0xF | (0x7 << 4) | (0 << 8) | (0x3 << 14)
    rocdl.s_waitcnt(val)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_csa_pool_fwd_module(
    num_heads,
    head_dim,
    swa_window,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    block_n=32,
    block_k=32,
    has_sink=False,
    has_sparse=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    mqa_kv=False,
    head_group=1,
):
    """Build the V4 CSA-from-pool forward per-row launcher (in-kernel gather).

    ``head_group`` is forced to 1 (no banking) and the sparse branch gathers
    from ``POOL`` via ``TOPK`` rather than reading a pre-gathered tensor.
    ``has_sparse`` is always effectively
    True here (the ``K_topk == 0`` case short-circuits to the dense kernel in
    the op layer).
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE  # one wave per program
    BLOCK_N = int(block_n)
    BLOCK_K = int(block_k)
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    assert HEAD_DIM % WARP_SIZE == 0, f"head_dim must be divisible by {WARP_SIZE}"
    D_PER_LANE = HEAD_DIM // WARP_SIZE  # 8 for D=512
    HEAD_GROUP = 1  # from-pool kernel does not bank heads
    NUM_HEAD_GROUPS = NUM_HEADS // HEAD_GROUP
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # No LDS gather cache in the from-pool kernel (indices are per-slot).
    ENABLE_LDS_CACHE = False
    lds_gather_offset = 0

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"csa_pool_fwd_smem_N{BLOCK_N}_K{BLOCK_K}_HG{HEAD_GROUP}",
    )

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_pool_fwd_kernel(
        Q: fx.Tensor,
        K_LOCAL: fx.Tensor,
        V_LOCAL: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        Sink: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K_LOCAL)
        vl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V_LOCAL)
        pool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), POOL)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TOPK)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        if const_expr(has_sink):
            sink_rsrc = buffer_ops.create_buffer_resource(Sink, max_size=True)

        f16_ty = elem_type
        f32_ty = T.f32

        # ---- Helpers ----
        def _gep_load_scalar(base_ptr, elem_idx, vec_type, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=elem_t,
                              noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def load_f16_v(base_ptr, elem_idx, n):
            vt = T.vec(n, f16_ty)
            return _gep_load_scalar(base_ptr, elem_idx, vt, f16_ty)

        def load_f32_scalar(base_ptr, elem_idx):
            return _gep_load_scalar(base_ptr, elem_idx, f32_ty, f32_ty)

        def load_i32_scalar(base_ptr, elem_idx):
            return _gep_load_scalar(base_ptr, elem_idx, T.i32, T.i32)

        # ---- Thread / program ----
        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        pid_bh = arith.index_cast(T.index, gpu.block_idx.y)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        lane = tid

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)
        pool_size_i32 = pool_size
        if const_expr(hasattr(pool_size_i32, "ir_value")):
            pool_size_i32 = pool_size_i32.ir_value()
        pool_size_v = arith.index_cast(T.index, pool_size)

        bid = pid_bh // arith.index(NUM_HEAD_GROUPS)
        qhid_group = pid_bh % arith.index(NUM_HEAD_GROUPS)
        # qhid_base is the head index of the first head in this program's head group.
        qhid_base = qhid_group * arith.index(HEAD_GROUP)

        # ---- LDS view for sparse-branch gathered cache ----
        if const_expr(ENABLE_LDS_CACHE):
            base_ptr = allocator.get_base()
            lds_gather = SmemPtr(
                base_ptr,
                lds_gather_offset,
                elem_type,
                shape=(LDS_GATHER_TILE_ELEMS,),
            ).get()

        # ---- Q row in-bounds guard ----
        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        # ---- MFMA tiling constants (CDNA4 v_mfma_f32_16x16x32) ----
        # QK^T is routed through the matrix cores: each MFMA computes a
        # 16(key) x 16 x 32(head-dim chunk) tile with the single query row
        # broadcast across the 16 M-rows. The 16 per-key scores land in the
        # column-major C-layout at lanes 0..15 (agpr[0]); we read them back
        # warp-uniform via readlane so the downstream softmax / AV path is
        # unchanged. cbsz/abid/blgp = 0 (broadcast via per-lane Q load).
        #
        # The MFMA keys-as-M tiling needs the key-tile dimension to be a
        # multiple of 16 (one 16x16x32 MFMA per key group) and the head-dim a
        # multiple of 32 (the contraction chunk). When a tile dim is not a
        # multiple of 16 (e.g. the default local BLOCK_N=8) we fall back to the
        # original VALU warp-reduce QK path for that branch instead of
        # asserting -- this keeps every representative shape correct while the
        # sparse branch (BLOCK_K=16) still engages the matrix cores.
        USE_MFMA_LOCAL = (HEAD_DIM % 32 == 0) and (BLOCK_N % 16 == 0)
        USE_MFMA_SPARSE = (HEAD_DIM % 32 == 0) and (BLOCK_K % 16 == 0)
        USE_MFMA_ANY = USE_MFMA_LOCAL or USE_MFMA_SPARSE
        K_CHUNKS = HEAD_DIM // 32
        N_GROUPS_LOCAL = BLOCK_N // 16 if USE_MFMA_LOCAL else 0
        N_GROUPS_SPARSE = BLOCK_K // 16 if USE_MFMA_SPARSE else 0
        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)
        lane_mod_16_i32 = arith.index_cast(T.i32, lane_mod_16)
        mfma_pack_ty = T.vec(8, f16_ty)
        zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_ty)
        c_zero_mfma_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))

        # ---- Preload Q operands ----
        # MFMA path: B-operand packs (register-resident, broadcast over M).
        # B-operand layout: lane L holds B[m = L%16, k = (L//16)*8 + 0..7]. The
        # query is broadcast across the 16 M-rows, so each pack depends only on
        # (L//16) and the head-dim chunk, not on L%16.
        # VALU path: per-lane f32 Q vector of D_PER_LANE elems (for warp-reduce).
        zero_f32_vec = arith.constant_vector(0.0, T.vec(D_PER_LANE, f32_ty))
        q_b_packs_per_head = []
        q_f32_vecs = []
        for h_off in range_constexpr(HEAD_GROUP):
            qhid_h = qhid_base + arith.index(h_off)
            q_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            if const_expr(USE_MFMA_ANY):
                packs = []
                for ck in range_constexpr(K_CHUNKS):
                    q_off = q_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    qp = load_f16_v(q_ptr, q_off, 8)
                    qp = arith.select(q_active, qp, zero_mfma_pack)
                    packs.append(qp)
                q_b_packs_per_head.append(packs)
            if const_expr(not (USE_MFMA_LOCAL and USE_MFMA_SPARSE)):
                q_lane_off = q_row_base + lane * arith.index(D_PER_LANE)
                q_vec = load_f16_v(q_ptr, q_lane_off, D_PER_LANE)
                q_f32_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), q_vec)
                q_f32_vec = arith.select(q_active, q_f32_vec, zero_f32_vec)
                q_f32_vecs.append(q_f32_vec)

        # ---- Constants ----
        NEG_INF_F = -1.0e30
        c_neg_inf = arith.constant(NEG_INF_F, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_one_f = arith.constant(1.0, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        c_log2e_f = arith.constant(_LOG2E, type=f32_ty)

        lane_i32 = arith.index_cast(T.i32, lane)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)

        def warp_reduce_sum_f32(v):
            cur = v
            for off in [32, 16, 8, 4, 2, 1]:
                xor_amt = arith.constant(off, type=T.i32)
                peer = arith.ArithValue(cur).shuffle_xor(xor_amt, width_i32)
                cur = arith.AddFOp(cur, peer, fastmath=fm_fast).result
            return cur

        def vec_dot_f32(a_vec, b_vec):
            s = c_zero_f
            for i in range_constexpr(D_PER_LANE):
                av = vector.extract(a_vec, static_position=[i], dynamic_position=[])
                bv = vector.extract(b_vec, static_position=[i], dynamic_position=[])
                p = arith.MulFOp(av, bv, fastmath=fm_fast).result
                s = arith.AddFOp(s, p, fastmath=fm_fast).result
            return s

        # ---- Local SWA loop bounds ----
        _pid_p1 = pid_m + arith.index(1)
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _pid_p1, seq_len_v)
        n_loop_end = arith.select(_le_seq, _pid_p1, seq_len_v)
        SWA = arith.index(int(swa_window))
        _ge_w = arith.cmpi(arith.CmpIPredicate.sge, _pid_p1, SWA)
        _n_lo_raw = arith.select(_ge_w, _pid_p1 - SWA, arith.index(0))
        BN_idx = arith.index(BLOCK_N)
        n_loop_start = (_n_lo_raw // BN_idx) * BN_idx

        # Init state: HEAD_GROUP copies of (m_i, l_i, acc[D_PER_LANE]).
        # Layout: [m_0, l_0, acc_0_0..acc_0_{D-1}, m_1, l_1, acc_1_0..]
        STATE_PER_HEAD = 2 + D_PER_LANE
        init_args = []
        for _h in range_constexpr(HEAD_GROUP):
            init_args.append(c_neg_inf)  # m
            init_args.append(c_zero_f)   # l
            for _ in range_constexpr(D_PER_LANE):
                init_args.append(c_zero_f)

        # ==== LOCAL SWA loop ====
        for n_start, inner_args, loop_results_local in scf.for_(
            n_loop_start,
            n_loop_end,
            BN_idx,
            iter_args=init_args,
        ):
            # Unpack HEAD_GROUP states from inner_args
            m_is = [inner_args[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
            l_is = [inner_args[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
            accs = [[inner_args[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

            n_start_i32 = arith.index_cast(T.i32, n_start)
            pid_m_i32 = arith.index_cast(T.i32, pid_m)
            seq_len_i32 = seq_len
            if const_expr(hasattr(seq_len_i32, "ir_value")):
                seq_len_i32 = seq_len_i32.ir_value()
            w_i32 = arith.constant(int(swa_window), type=T.i32)

            # Per-head QK values: [HEAD_GROUP][BLOCK_N].
            qk_vals_per_head = [[] for _ in range_constexpr(HEAD_GROUP)]
            if const_expr(USE_MFMA_LOCAL):
                # CDNA4 MFMA keys-as-M: each MFMA group handles 16 contiguous
                # local keys; the head-dim D is contracted in K_CHUNKS chunks of
                # 32. A-operand = K (lane L -> key L%16, dims (L//16)*8..),
                # B-operand = broadcast Q.
                for h_off in range_constexpr(HEAD_GROUP):
                    qhid_h = qhid_base + arith.index(h_off)
                    q_packs_h = q_b_packs_per_head[h_off]
                    raw_scores = [None] * BLOCK_N
                    for g in range_constexpr(N_GROUPS_LOCAL):
                        # Per-lane key column for group: n_start + g*16 + lane%16
                        key_col_i32 = arith.AddIOp(
                            arith.AddIOp(n_start_i32, arith.constant(g * 16, type=T.i32)).result,
                            lane_mod_16_i32,
                        ).result
                        key_is_oob = arith.cmpi(arith.CmpIPredicate.sge, key_col_i32, seq_len_i32)
                        key_idx = arith.index_cast(T.index, key_col_i32)
                        key_safe = arith.select(key_is_oob, arith.index(0), key_idx)
                        # MQA: local K shared across heads. MHA (HG>1): per-head K row.
                        if const_expr(mqa_kv):
                            kl_row_base = (bid * seq_len_v + key_safe) * arith.index(HEAD_DIM)
                        else:
                            kl_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + key_safe) * arith.index(HEAD_DIM)
                        c_acc = c_zero_mfma_acc
                        for ck in range_constexpr(K_CHUNKS):
                            koff = kl_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                            a_pack = load_f16_v(kl_ptr, koff, 8)
                            c_acc = mfma_f32_16x16x32(a_pack, q_packs_h[ck], c_acc, dtype_str)
                        # Column-major MFMA output: lane L holds agpr[k] =
                        # C[(L//16)*4 + k, L%16]. With Q broadcast across all 16
                        # columns the score for key row i is identical in every
                        # column, so read it from column j=0: register k = i%4 at
                        # lane (i//4)*16.
                        c_regs = [
                            vector.extract(c_acc, static_position=[r], dynamic_position=[])
                            for r in range_constexpr(4)
                        ]
                        for i in range_constexpr(16):
                            key_i = g * 16 + i
                            raw_scores[key_i] = rocdl.readlane(
                                f32_ty, c_regs[i % 4], arith.constant((i // 4) * 16, type=T.i32))

                    # Per-key causal / SWA / boundary mask (warp-uniform scalars).
                    for n_off in range_constexpr(BLOCK_N):
                        kv_col_i32 = arith.AddIOp(
                            n_start_i32,
                            arith.constant(n_off, type=T.i32),
                        ).result
                        _kv_plus_w_lo = arith.AddIOp(kv_col_i32, w_i32).result
                        is_swa_lo = arith.cmpi(
                            arith.CmpIPredicate.sle, _kv_plus_w_lo, pid_m_i32)
                        is_causal_lo = arith.cmpi(
                            arith.CmpIPredicate.sgt, kv_col_i32, pid_m_i32)
                        is_oob = arith.cmpi(
                            arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                        bad_lo = arith.OrIOp(
                            arith.OrIOp(is_causal_lo, is_swa_lo).result,
                            is_oob).result
                        qk_scaled = arith.MulFOp(raw_scores[n_off], c_sm_scale_f, fastmath=fm_fast).result
                        qk_masked = arith.select(bad_lo, c_neg_inf, qk_scaled)
                        qk_vals_per_head[h_off].append(qk_masked)
            else:
                # VALU fallback (BLOCK_N % 16 != 0, e.g. default BLOCK_N=8):
                # per-key warp-reduce dot product over the head-dim.
                for n_off in range_constexpr(BLOCK_N):
                    kv_col_i32 = arith.AddIOp(
                        n_start_i32,
                        arith.constant(n_off, type=T.i32),
                    ).result
                    _kv_plus_w_lo = arith.AddIOp(kv_col_i32, w_i32).result
                    is_swa_lo = arith.cmpi(
                        arith.CmpIPredicate.sle, _kv_plus_w_lo, pid_m_i32)
                    is_causal_lo = arith.cmpi(
                        arith.CmpIPredicate.sgt, kv_col_i32, pid_m_i32)
                    is_oob = arith.cmpi(
                        arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                    bad_lo = arith.OrIOp(
                        arith.OrIOp(is_causal_lo, is_swa_lo).result,
                        is_oob).result

                    kv_col_idx = arith.index_cast(T.index, kv_col_i32)
                    kv_col_safe = arith.select(is_oob, arith.index(0), kv_col_idx)
                    if const_expr(mqa_kv):
                        kl_row_base = (bid * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                        kl_lane_off = kl_row_base + lane * arith.index(D_PER_LANE)
                        kl_vec = load_f16_v(kl_ptr, kl_lane_off, D_PER_LANE)
                        kl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), kl_vec)
                        for h_off in range_constexpr(HEAD_GROUP):
                            lane_dot = vec_dot_f32(kl_f32, q_f32_vecs[h_off])
                            qk_full = warp_reduce_sum_f32(lane_dot)
                            qk_scaled = arith.MulFOp(qk_full, c_sm_scale_f, fastmath=fm_fast).result
                            qk_masked = arith.select(bad_lo, c_neg_inf, qk_scaled)
                            qk_vals_per_head[h_off].append(qk_masked)
                    else:
                        for h_off in range_constexpr(HEAD_GROUP):
                            qhid_h = qhid_base + arith.index(h_off)
                            kl_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                            kl_lane_off = kl_row_base + lane * arith.index(D_PER_LANE)
                            kl_vec = load_f16_v(kl_ptr, kl_lane_off, D_PER_LANE)
                            kl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), kl_vec)
                            lane_dot = vec_dot_f32(kl_f32, q_f32_vecs[h_off])
                            qk_full = warp_reduce_sum_f32(lane_dot)
                            qk_scaled = arith.MulFOp(qk_full, c_sm_scale_f, fastmath=fm_fast).result
                            qk_masked = arith.select(bad_lo, c_neg_inf, qk_scaled)
                            qk_vals_per_head[h_off].append(qk_masked)

            # Per-head softmax update
            new_m_is = []
            new_l_is = []
            new_accs = []
            p_vals_per_head = []
            for h_off in range_constexpr(HEAD_GROUP):
                qk_vals_h = qk_vals_per_head[h_off]
                m_tile = qk_vals_h[0]
                for n_off in range_constexpr(BLOCK_N - 1):
                    m_tile = arith.MaxNumFOp(m_tile, qk_vals_h[n_off + 1], fastmath=fm_fast).result
                m_new = arith.MaxNumFOp(m_is[h_off], m_tile, fastmath=fm_fast).result

                diff_m = arith.SubFOp(m_is[h_off], m_new, fastmath=fm_fast).result
                diff_m_log2 = arith.MulFOp(diff_m, c_log2e_f, fastmath=fm_fast).result
                alpha = arith.ArithValue(diff_m_log2).exp2(fastmath=fm_fast)

                p_vals = []
                tile_sum = c_zero_f
                for n_off in range_constexpr(BLOCK_N):
                    d = arith.SubFOp(qk_vals_h[n_off], m_new, fastmath=fm_fast).result
                    dl = arith.MulFOp(d, c_log2e_f, fastmath=fm_fast).result
                    p = arith.ArithValue(dl).exp2(fastmath=fm_fast)
                    p_vals.append(p)
                    tile_sum = arith.AddFOp(tile_sum, p, fastmath=fm_fast).result
                p_vals_per_head.append(p_vals)

                l_alpha = arith.MulFOp(l_is[h_off], alpha, fastmath=fm_fast).result
                l_new = arith.AddFOp(l_alpha, tile_sum, fastmath=fm_fast).result

                acc_h = accs[h_off]
                new_acc_h = []
                for d_off in range_constexpr(D_PER_LANE):
                    new_acc_h.append(arith.MulFOp(acc_h[d_off], alpha, fastmath=fm_fast).result)
                new_m_is.append(m_new)
                new_l_is.append(l_new)
                new_accs.append(new_acc_h)

            # AV phase: load V once (MQA), reuse across HEAD_GROUP heads.
            for n_off in range_constexpr(BLOCK_N):
                kv_col_i32 = arith.AddIOp(
                    n_start_i32,
                    arith.constant(n_off, type=T.i32),
                ).result
                is_oob = arith.cmpi(
                    arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                kv_col_idx = arith.index_cast(T.index, kv_col_i32)
                kv_col_safe = arith.select(is_oob, arith.index(0), kv_col_idx)
                if const_expr(mqa_kv):
                    # Shared local V across heads (MQA): load once.
                    vl_row_base = (bid * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                    vl_lane_off = vl_row_base + lane * arith.index(D_PER_LANE)
                    vl_vec = load_f16_v(vl_ptr, vl_lane_off, D_PER_LANE)
                    vl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), vl_vec)
                    for h_off in range_constexpr(HEAD_GROUP):
                        p_vals_h = p_vals_per_head[h_off]
                        new_acc_h = new_accs[h_off]
                        for d_off in range_constexpr(D_PER_LANE):
                            vv = vector.extract(vl_f32, static_position=[d_off], dynamic_position=[])
                            contrib = arith.MulFOp(p_vals_h[n_off], vv, fastmath=fm_fast).result
                            new_acc_h[d_off] = arith.AddFOp(new_acc_h[d_off], contrib, fastmath=fm_fast).result
                else:
                    # MHA (head_group > 1): each head reads its own local V row.
                    for h_off in range_constexpr(HEAD_GROUP):
                        qhid_h = qhid_base + arith.index(h_off)
                        vl_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                        vl_lane_off = vl_row_base + lane * arith.index(D_PER_LANE)
                        vl_vec = load_f16_v(vl_ptr, vl_lane_off, D_PER_LANE)
                        vl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), vl_vec)
                        p_vals_h = p_vals_per_head[h_off]
                        new_acc_h = new_accs[h_off]
                        for d_off in range_constexpr(D_PER_LANE):
                            vv = vector.extract(vl_f32, static_position=[d_off], dynamic_position=[])
                            contrib = arith.MulFOp(p_vals_h[n_off], vv, fastmath=fm_fast).result
                            new_acc_h[d_off] = arith.AddFOp(new_acc_h[d_off], contrib, fastmath=fm_fast).result

            # Pack yield args
            yield_args = []
            for h_off in range_constexpr(HEAD_GROUP):
                yield_args.append(new_m_is[h_off])
                yield_args.append(new_l_is[h_off])
                for d in range_constexpr(D_PER_LANE):
                    yield_args.append(new_accs[h_off][d])
            yield yield_args

        # Unpack HEAD_GROUP states from local SWA results
        m_is = [loop_results_local[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
        l_is = [loop_results_local[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
        accs = [[loop_results_local[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

        # ==== SPARSE branch ====
        if const_expr(has_sparse):
            init_sparse = []
            for h_off in range_constexpr(HEAD_GROUP):
                init_sparse.append(m_is[h_off])
                init_sparse.append(l_is[h_off])
                for d in range_constexpr(D_PER_LANE):
                    init_sparse.append(accs[h_off][d])
            for k_start, inner_args, loop_results_sparse in scf.for_(
                arith.index(0),
                K_topk_v,
                arith.index(BLOCK_K),
                iter_args=init_sparse,
            ):
                m_is = [inner_args[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
                l_is = [inner_args[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
                accs = [[inner_args[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

                k_start_i32 = arith.index_cast(T.i32, k_start)
                K_topk_i32 = K_topk
                if const_expr(hasattr(K_topk_i32, "ir_value")):
                    K_topk_i32 = K_topk_i32.ir_value()

                # QK^T via CDNA4 MFMA over the gathered keys (keys-as-M). Each
                # lane gathers its own slot (lane%16) of the group from POOL via
                # TOPK; the head-dim D is contracted in K_CHUNKS chunks of 32.
                # A slot is masked (NEG_INF) when it is out of the K-loop, < 0
                # (the -1 pad), or >= pool_size.
                qk_vals_sparse_per_head = [[] for _ in range_constexpr(HEAD_GROUP)]
                if const_expr(USE_MFMA_SPARSE):
                  for h_off in range_constexpr(HEAD_GROUP):
                    q_packs_h = q_b_packs_per_head[h_off]
                    raw_scores = [None] * BLOCK_K
                    invalid_flags = [None] * BLOCK_K
                    for g in range_constexpr(N_GROUPS_SPARSE):
                        # Per-lane gathered slot: k_pos = k_start + g*16 + lane%16
                        k_pos_i32 = arith.AddIOp(
                            arith.AddIOp(k_start_i32, arith.constant(g * 16, type=T.i32)).result,
                            lane_mod_16_i32,
                        ).result
                        is_koob = arith.cmpi(arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                        k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                        k_pos_safe = arith.select(is_koob, arith.index(0), k_pos_idx)
                        topk_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                        idx_i32 = load_i32_scalar(topk_ptr, topk_off)
                        is_neg = arith.cmpi(arith.CmpIPredicate.slt, idx_i32, arith.constant(0, type=T.i32))
                        is_ge_p = arith.cmpi(arith.CmpIPredicate.sge, idx_i32, pool_size_i32)
                        invalid = arith.OrIOp(arith.OrIOp(is_koob, is_neg).result, is_ge_p).result
                        idx_safe_i32 = arith.select(invalid, arith.constant(0, type=T.i32), idx_i32)
                        idx_safe = arith.index_cast(T.index, idx_safe_i32)
                        pool_row_base = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
                        invalid_i32 = arith.select(
                            invalid, arith.constant(1, type=T.i32), arith.constant(0, type=T.i32))
                        c_acc = c_zero_mfma_acc
                        for ck in range_constexpr(K_CHUNKS):
                            poff = pool_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                            a_pack = load_f16_v(pool_ptr, poff, 8)
                            c_acc = mfma_f32_16x16x32(a_pack, q_packs_h[ck], c_acc, dtype_str)
                        # Column-major MFMA output: lane L holds agpr[k] =
                        # C[(L//16)*4 + k, L%16]. Score for gathered slot row i
                        # (Q broadcast across columns) -> register k = i%4 at
                        # lane (i//4)*16. The per-slot invalid flag instead lives
                        # in the lane whose lane%16 == i (it was computed from
                        # lane_mod_16), i.e. lane i of group 0.
                        c_regs = [
                            vector.extract(c_acc, static_position=[r], dynamic_position=[])
                            for r in range_constexpr(4)
                        ]
                        for i in range_constexpr(16):
                            slot_i = g * 16 + i
                            raw_scores[slot_i] = rocdl.readlane(
                                f32_ty, c_regs[i % 4], arith.constant((i // 4) * 16, type=T.i32))
                            inv_i = rocdl.readlane(T.i32, invalid_i32, arith.constant(i, type=T.i32))
                            invalid_flags[slot_i] = arith.cmpi(
                                arith.CmpIPredicate.ne, inv_i, arith.constant(0, type=T.i32))
                    for k_off in range_constexpr(BLOCK_K):
                        qk_scaled = arith.MulFOp(raw_scores[k_off], c_sm_scale_f, fastmath=fm_fast).result
                        qk_masked = arith.select(invalid_flags[k_off], c_neg_inf, qk_scaled)
                        qk_vals_sparse_per_head[h_off].append(qk_masked)
                else:
                  # VALU fallback (BLOCK_K % 16 != 0): per-slot warp-reduce dot.
                  for k_off in range_constexpr(BLOCK_K):
                    k_pos_i32 = arith.AddIOp(
                        k_start_i32,
                        arith.constant(k_off, type=T.i32),
                    ).result
                    is_oob = arith.cmpi(
                        arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                    k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                    k_pos_safe = arith.select(is_oob, arith.index(0), k_pos_idx)

                    topk_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                    idx_i32 = load_i32_scalar(topk_ptr, topk_off)
                    is_neg = arith.cmpi(arith.CmpIPredicate.slt, idx_i32, arith.constant(0, type=T.i32))
                    is_ge_p = arith.cmpi(arith.CmpIPredicate.sge, idx_i32, pool_size_i32)
                    invalid = arith.OrIOp(arith.OrIOp(is_oob, is_neg).result, is_ge_p).result
                    idx_safe_i32 = arith.select(invalid, arith.constant(0, type=T.i32), idx_i32)
                    idx_safe = arith.index_cast(T.index, idx_safe_i32)
                    pool_row_base = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
                    pool_lane_off = pool_row_base + lane * arith.index(D_PER_LANE)
                    g_vec = load_f16_v(pool_ptr, pool_lane_off, D_PER_LANE)
                    g_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), g_vec)
                    for h_off in range_constexpr(HEAD_GROUP):
                        lane_dot = vec_dot_f32(g_f32, q_f32_vecs[h_off])
                        qk_full = warp_reduce_sum_f32(lane_dot)
                        qk_scaled = arith.MulFOp(qk_full, c_sm_scale_f, fastmath=fm_fast).result
                        qk_masked = arith.select(invalid, c_neg_inf, qk_scaled)
                        qk_vals_sparse_per_head[h_off].append(qk_masked)

                new_m_is = []
                new_l_is = []
                new_accs = []
                p_vals_per_head = []
                for h_off in range_constexpr(HEAD_GROUP):
                    qk_vals_h = qk_vals_sparse_per_head[h_off]
                    m_tile = qk_vals_h[0]
                    for k_off in range_constexpr(BLOCK_K - 1):
                        m_tile = arith.MaxNumFOp(m_tile, qk_vals_h[k_off + 1], fastmath=fm_fast).result
                    m_new = arith.MaxNumFOp(m_is[h_off], m_tile, fastmath=fm_fast).result

                    diff_m = arith.SubFOp(m_is[h_off], m_new, fastmath=fm_fast).result
                    diff_m_log2 = arith.MulFOp(diff_m, c_log2e_f, fastmath=fm_fast).result
                    alpha = arith.ArithValue(diff_m_log2).exp2(fastmath=fm_fast)

                    p_vals = []
                    tile_sum = c_zero_f
                    for k_off in range_constexpr(BLOCK_K):
                        d = arith.SubFOp(qk_vals_h[k_off], m_new, fastmath=fm_fast).result
                        dl = arith.MulFOp(d, c_log2e_f, fastmath=fm_fast).result
                        p = arith.ArithValue(dl).exp2(fastmath=fm_fast)
                        p_vals.append(p)
                        tile_sum = arith.AddFOp(tile_sum, p, fastmath=fm_fast).result
                    p_vals_per_head.append(p_vals)

                    l_alpha = arith.MulFOp(l_is[h_off], alpha, fastmath=fm_fast).result
                    l_new = arith.AddFOp(l_alpha, tile_sum, fastmath=fm_fast).result

                    acc_h = accs[h_off]
                    new_acc_h = []
                    for d_off in range_constexpr(D_PER_LANE):
                        new_acc_h.append(arith.MulFOp(acc_h[d_off], alpha, fastmath=fm_fast).result)
                    new_m_is.append(m_new)
                    new_l_is.append(l_new)
                    new_accs.append(new_acc_h)

                # ---- AV phase: re-read gathered K-block once per k_off, reuse for all heads ----
                if const_expr(ENABLE_LDS_CACHE):
                    _waitcnt_lgkm_0()
                for k_off in range_constexpr(BLOCK_K):
                    # Re-gather POOL[bid, idx] for the AV phase. Invalid slots
                    # read row 0 but contribute 0 (their p == exp(NEG_INF) == 0).
                    k_pos_i32 = arith.AddIOp(
                        k_start_i32,
                        arith.constant(k_off, type=T.i32),
                    ).result
                    is_oob = arith.cmpi(
                        arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                    k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                    k_pos_safe = arith.select(is_oob, arith.index(0), k_pos_idx)
                    topk_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                    idx_i32 = load_i32_scalar(topk_ptr, topk_off)
                    is_neg = arith.cmpi(arith.CmpIPredicate.slt, idx_i32, arith.constant(0, type=T.i32))
                    is_ge_p = arith.cmpi(arith.CmpIPredicate.sge, idx_i32, pool_size_i32)
                    invalid = arith.OrIOp(arith.OrIOp(is_oob, is_neg).result, is_ge_p).result
                    idx_safe_i32 = arith.select(invalid, arith.constant(0, type=T.i32), idx_i32)
                    idx_safe = arith.index_cast(T.index, idx_safe_i32)
                    pool_row_base = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
                    pool_lane_off = pool_row_base + lane * arith.index(D_PER_LANE)
                    g_vec = load_f16_v(pool_ptr, pool_lane_off, D_PER_LANE)
                    g_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), g_vec)
                    for h_off in range_constexpr(HEAD_GROUP):
                        p_vals_h = p_vals_per_head[h_off]
                        new_acc_h = new_accs[h_off]
                        for d_off in range_constexpr(D_PER_LANE):
                            vv = vector.extract(g_f32, static_position=[d_off], dynamic_position=[])
                            contrib = arith.MulFOp(p_vals_h[k_off], vv, fastmath=fm_fast).result
                            new_acc_h[d_off] = arith.AddFOp(new_acc_h[d_off], contrib, fastmath=fm_fast).result

                # Pack yield args
                yield_args = []
                for h_off in range_constexpr(HEAD_GROUP):
                    yield_args.append(new_m_is[h_off])
                    yield_args.append(new_l_is[h_off])
                    for d in range_constexpr(D_PER_LANE):
                        yield_args.append(new_accs[h_off][d])
                yield yield_args

            m_is = [loop_results_sparse[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
            l_is = [loop_results_sparse[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
            accs = [[loop_results_sparse[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

        # ==== Sink epilogue (per-head) ====
        if const_expr(has_sink):
            for h_off in range_constexpr(HEAD_GROUP):
                qhid_h = qhid_base + arith.index(h_off)
                qhid_i32 = arith.index_cast(T.i32, qhid_h)
                sink_h_val = buffer_ops.buffer_load(
                    sink_rsrc, qhid_i32, vec_width=1, dtype=f32_ty,
                )
                m_i_h = m_is[h_off]
                l_i_h = l_is[h_off]
                acc_h = accs[h_off]
                m_new = arith.MaxNumFOp(m_i_h, sink_h_val, fastmath=fm_fast).result
                d_alpha = arith.SubFOp(m_i_h, m_new, fastmath=fm_fast).result
                d_alpha_log2 = arith.MulFOp(d_alpha, c_log2e_f, fastmath=fm_fast).result
                alpha_sink = arith.ArithValue(d_alpha_log2).exp2(fastmath=fm_fast)
                d_beta = arith.SubFOp(sink_h_val, m_new, fastmath=fm_fast).result
                d_beta_log2 = arith.MulFOp(d_beta, c_log2e_f, fastmath=fm_fast).result
                beta_sink = arith.ArithValue(d_beta_log2).exp2(fastmath=fm_fast)
                l_alpha = arith.MulFOp(l_i_h, alpha_sink, fastmath=fm_fast).result
                l_is[h_off] = arith.AddFOp(l_alpha, beta_sink, fastmath=fm_fast).result
                new_acc_h = []
                for d_off in range_constexpr(D_PER_LANE):
                    new_acc_h.append(arith.MulFOp(acc_h[d_off], alpha_sink, fastmath=fm_fast).result)
                accs[h_off] = new_acc_h
                m_is[h_off] = m_new

        # ==== Final divide and store (per-head) ====
        _o_guard = scf.IfOp(q_active, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for h_off in range_constexpr(HEAD_GROUP):
                qhid_h = qhid_base + arith.index(h_off)
                l_i_h = l_is[h_off]
                m_i_h = m_is[h_off]
                acc_h = accs[h_off]
                inv_l = arith.DivFOp(c_one_f, l_i_h, fastmath=fm_fast).result
                o_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
                o_lane_off = o_row_base + lane * arith.index(D_PER_LANE)
                for d_off in range_constexpr(D_PER_LANE):
                    o_f32 = arith.MulFOp(acc_h[d_off], inv_l, fastmath=fm_fast).result
                    o_f16 = arith.trunc_f(elem_type, o_f32)
                    elem_off = o_lane_off + arith.index(d_off)
                    idx_i64 = arith.index_cast(T.i64, elem_off)
                    gep = _llvm.GEPOp(_llvm_ptr_ty(), o_ptr, [idx_i64],
                                      rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                      elem_type=elem_type,
                                      noWrapFlags=0)
                    _llvm.StoreOp(o_f16, gep.result)

                is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.index(0))
                _lse_if = scf.IfOp(is_lane0, [], has_else=False)
                with ir.InsertionPoint(_lse_if.then_block):
                    ln_l = math_dialect.log(l_i_h, fastmath=fm_fast)
                    lse_val = arith.AddFOp(m_i_h, ln_l, fastmath=fm_fast).result
                    lse_off = (bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + pid_m_safe
                    lse_off_i32 = arith.index_cast(T.i32, lse_off)
                    buffer_ops.buffer_store(lse_val, lse_rsrc, lse_off_i32)
                    scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_csa_pool_fwd(
        Q: fx.Tensor,
        K_LOCAL: fx.Tensor,
        V_LOCAL: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        Sink: fx.Tensor,
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
        grid_x = sl_idx
        grid_y = bs_idx * arith.index(NUM_HEAD_GROUPS)

        launcher = csa_pool_fwd_kernel(
            Q, K_LOCAL, V_LOCAL, POOL, TOPK, Sink, O, LSE,
            seq_len, K_topk, pool_size,
        )

        # Occupancy / resource: pin a higher waves-per-eu target on the
        # dominant pool-fwd kernel. The AMDGPU register allocator treats
        # rocdl.waves_per_eu as a minimum-occupancy budget and caps VGPRs at
        # ~512/N per SIMD, so raising N from 2->3 asks the JIT to compact the
        # long MFMA->exp2->rescale online-softmax live ranges and keep a 3rd
        # wave/SIMD resident to hide the dependency-chain latency that the
        # baseline 2-wave occupancy cannot. Env FLYDSL_WAVES_PER_EU can still
        # raise it further.
        OCC_WAVES_PER_EU_TARGET = 3
        if const_expr(waves_per_eu is not None):
            _wpe = max(int(waves_per_eu), OCC_WAVES_PER_EU_TARGET)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe)

        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"),
                ir.StringAttr.get("true"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"),
                ir.StringAttr.get("true"),
            ]))
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_pool_fwd(*args, **kwargs)

    def _compile(Q, K_LOCAL, V_LOCAL, POOL, TOPK, Sink, O, LSE, batch_size, seq_len, K_topk, pool_size, stream=None):
        with CompilationContext.compile_hints(compile_hints):
            return flyc.compile(
                launch_csa_pool_fwd, Q, K_LOCAL, V_LOCAL, POOL, TOPK, Sink, O, LSE,
                batch_size, seq_len, K_topk, pool_size, fx.Stream(stream))

    _launch.compile = _compile

    return _launch


build_csa_pool_fwd_module_primary = build_csa_pool_fwd_module
