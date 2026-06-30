# SPDX-License-Identifier: Apache-2.0
"""csa_bwd_full: V4 CSA backward kernel, full output set.

Emits dq + dk_local + dv_local + dgathered + dsink in one launch.
Mirrors `_csa_attention_bwd_kernel` (Triton, _triton/csa_attention_bwd.py)
1:1: grid=(Sq, B*HQ); each program owns one query row. dq is direct-stored;
dk_local, dv_local, dgathered, dsink are accumulated via atomic_fadd.
"""
from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from primus_turbo.flydsl.attention.kernels.kernels_common import (
    dgathered_atomic_elem_base,
    dgathered_split_elem_base,
    dtype_to_elem_type,
    mfma_mv_reduce_16,
)
from primus_turbo.flydsl.attention.kernels.warp_pipeline_common import (
    ring_stage_byte_base,
    ring_stage_elem_base,
)
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref, scf, fly as _fly, llvm as _llvm, math as math_dialect
from flydsl.expr import const_expr  # noqa: E402


KERNEL_NAME = "csa_bwd_full_kernel"
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_csa_bwd_full_module(
    num_heads, head_dim, swa_window,
    dtype_str="bf16", sm_scale=None, waves_per_eu=2,
    block_n=32, block_k=32, qrow_block=2, head_block=2,
    has_sink=True, has_sparse=True,
    unsafe_fp_math=True, fast_fp_math=True, daz=True,
    mqa_kv=True,
):
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    BLOCK_N = int(block_n)
    BLOCK_K = int(block_k)
    # Query-row blocking. One program owns QROW_BLOCK contiguous query rows
    # (grid X -> ceil(Sq/QROW_BLOCK)). The shared LOCAL k_local /
    # v_local A-frag (and the per-lane k_local dq vectors) for each key tile are
    # loaded ONCE and consumed across all QROW_BLOCK rows in the block before
    # eviction, converting cold L2 re-misses on the shared local K/V cache lines
    # into register/L2 hits. The online-softmax recompute, the dq direct-store,
    # the dgathered / dsink atomic-accumulation algorithm, and the gathered-tile
    # layout are all unchanged -- only the iteration order (key-tile loop hoisted
    # above a per-row consume loop) and the per-program row count move.
    QROW_BLOCK = int(qrow_block)
    assert QROW_BLOCK >= 1, "qrow_block must be >= 1"
    NUM_HEADS = int(num_heads)
    # Head-grouping. The GATHERED tensor is [B, Sq, K_topk, D] with NO head axis
    # (identical across all HQ query heads), and for MQA (mqa_kv) the LOCAL
    # k_local/v_local are head-shared too. A per-head grid would map one program
    # to a single (b, head, query-row-block), so each of the HQ heads would
    # re-read the SAME gathered (and shared local) K/V tiles from HBM. Instead
    # one program owns a (b, query-row-block) over a *block* of HEAD_BLOCK query
    # heads: the head-shared gathered g_afrag / g_vec (and, for MQA, the local
    # k/v A-frags + the LDS-staged K tile) are fetched from HBM ONCE per program
    # and reused across all HEAD_BLOCK heads' score-recompute / dq / dgathered
    # contractions (only the per-head q / dO B-frags are re-loaded per head).
    # As a structural consequence the head-shared dgathered contributions for one
    # (query-row, k_pos) are summed in-register across the HEAD_BLOCK heads before
    # a single atomic_fadd, collapsing HEAD_BLOCK-fold cross-head atomic traffic.
    # Head-grouping only buys reuse when the local K/V are head-shared, so it is
    # gated on mqa_kv; for MHA (and indivisible head counts) HEAD_BLOCK falls back
    # to 1 (one program per head). The grid Y shrinks from B*HQ to B*(HQ/HB).
    _hb_req = int(head_block)
    assert _hb_req >= 1, "head_block must be >= 1"
    if mqa_kv and NUM_HEADS % _hb_req == 0:
        HEAD_BLOCK = _hb_req
    else:
        HEAD_BLOCK = 1
    NUM_HEAD_GROUPS = NUM_HEADS // HEAD_BLOCK
    HEAD_DIM = int(head_dim)
    assert HEAD_DIM % WARP_SIZE == 0, f"head_dim must be divisible by {WARP_SIZE}"
    D_PER_LANE = HEAD_DIM // WARP_SIZE
    # Route the per-key QK^T and dP=dO.V cross-lane reductions through
    # the CDNA4 matrix core (v_mfma_f32_16x16x32_bf16) instead of the per-key
    # lane-partial vec_dot_f32 + 6-step warp_reduce (bpermute) butterfly. The
    # BLOCK_N/BLOCK_K key tile supplies the MFMA M dimension (16-key sub-tiles),
    # head_dim the K contraction (K_STEPS x 32), and the single owned query/dO
    # the broadcast N operand (only column 0 is real). dq=ds.K stays a per-lane
    # f32 accumulation (it has no bpermute and feeds the contiguous-D dq store).
    MFMA_K_STEP = 32
    assert HEAD_DIM % MFMA_K_STEP == 0, "head_dim must be a multiple of 32 for MFMA 16x16x32"
    K_STEPS = HEAD_DIM // MFMA_K_STEP
    assert BLOCK_N % 16 == 0, "block_n must be a multiple of 16 (MFMA M-tile)"
    assert BLOCK_K % 16 == 0, "block_k must be a multiple of 16 (MFMA M-tile)"
    N_MTILES = BLOCK_N // 16
    K_MTILES = BLOCK_K // 16
    assert gpu_arch.startswith("gfx950"), (
        "csa_bwd_full MFMA path requires gfx950 (CDNA4 bf16 MFMA)."
    )
    # ``mqa_kv`` selects how the LOCAL k_local / v_local inputs are indexed:
    #   True  -> shared [B, Sq, D] (K_H == 1, broadcast across query heads);
    #   False -> per-head [B, HQ, Sq, D] (K_H == HQ, MHA — the Primus-Turbo CSA
    #            local branch). The dk_local / dv_local *outputs* are per-head
    #            [B, HQ, Sq, D] in both cases (the MQA caller reduces over heads).
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    allocator = SmemAllocator(
        None, arch=gpu_arch,
        global_sym_name=f"csa_bwd_full_smem_N{BLOCK_N}_K{BLOCK_K}_S{int(has_sink)}_HS{int(has_sparse)}",
    )
    # The streaming LOCAL per-lane K tile (kl) and the gathered tile are staged
    # in LDS rather than held as kl_f32_cache (BLOCK_N x D_PER_LANE f32 SSA
    # values per lane) live in VGPRs across the whole MFMA + per-row consume
    # span. The bf16->f32 extf runs after the ds_read at point of use, freeing
    # the dominant persistent VGPR block to lift realized waves/SIMD.
    #
    # The producer of those LDS tiles uses a single CDNA4 hardware
    # direct-to-LDS load (buffer_load_lds dwordx4: D_PER_LANE==8 f16 == 16 B
    # per lane) per column, retiring the contiguous per-lane row straight from
    # device memory into the LDS staging buffer with no intermediate v_mov +
    # ds_write VGPR roundtrip. buffer_load_lds writes lane L's payload to
    # base + L*size, so the LDS layout is warp-contiguous per column (tile
    # offset occupies WARP_SIZE*D_PER_LANE f16, lane L at +L*D_PER_LANE) and the
    # consume ds_read index matches.
    #
    # The same LDS scratchpad stages BOTH the streaming LOCAL K tile (BLOCK_N
    # columns) and -- in the GATHERED branch -- the per-query gathered tile
    # (BLOCK_K columns), reused across the two disjoint loop phases. One stage is
    # sized for the wider of the two so either fits.
    #
    # The staging buffer is a LDS_RING_STAGES-deep ring. The dedicated gathered
    # producer keeps TWO no-reuse gathered [BLOCK_K, HEAD_DIM] tiles outstanding
    # global->LDS (the next two pipeline slots) while the consumer drives the
    # score-recompute / dgathered MFMA + atomic chain out of the current
    # resident slot, so the cold L2-miss gathered loads are decoupled a full
    # extra slot ahead of the latency-bound consume instead of being exposed by
    # an "issue buffer_load_lds -> immediate s_waitcnt(0) + barrier" drain. A
    # three-way slot rotation with a two-slot prologue prefetch + drain epilogue
    # drives it; the online-softmax recompute and the dq/dgathered accumulation
    # are algebraically unchanged (deeper buffering only). The LOCAL SWA loop
    # uses stage 0 only. LDS budget: LDS_KL_ELEMS * 3 * 2 B must stay under the
    # gfx950 per-CU ceiling with occupancy held.
    LDS_RING_STAGES = 3
    # Diagonal LDS row-skew to kill the staged-tile bank-conflict serialization.
    # A flat layout gives every staged key row a WARP_SIZE*D_PER_LANE
    # (== HEAD_DIM) element stride; at D=512 f16 that is 1024 B ==
    # (1024/4)%64 == 0 on the gfx950 64-bank LDS, so all 16 rows of a tile alias
    # the SAME banks on every ds_read / MFMA A-frag pack (only 16 of 64 banks
    # ever touched -> ~16-way conflict). buffer_load_lds fixes lane L's payload
    # at base + L*size, so an intra-row XOR swizzle is not expressible on the
    # store side; the algebraically-identical, store-expressible equivalent is
    # to PAD each row by LDS_ROW_PAD f16 so the row stride is no longer a
    # multiple of the 64*4B bank period. PAD=8 keeps the row base 16 B aligned
    # (1040 B, legal for the dwordx4 buffer_load_lds) and shifts each row by
    # (1040/4)%64 == 4 banks == exactly one 8xf16 access width, spreading the
    # 16 rows across all 64 banks. Pure LDS address remap applied IDENTICALLY to
    # the buffer_load_lds store base and every ds_read / MFMA-pack consume base;
    # the per-lane payload, online-softmax recompute and dq/dgathered results are
    # bitwise-identical -- only the row stride constant moves.
    LDS_ROW_PAD = 8
    LDS_KL_ROW_STRIDE = WARP_SIZE * D_PER_LANE + LDS_ROW_PAD
    LDS_KL_ELEMS = max(BLOCK_N, BLOCK_K) * LDS_KL_ROW_STRIDE
    LDS_RING_ELEMS = LDS_KL_ELEMS * LDS_RING_STAGES
    lds_kl_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kl_offset + LDS_RING_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_bwd_full_kernel(
        Q: fx.Tensor, K_LOCAL: fx.Tensor, V_LOCAL: fx.Tensor,
        GATHERED: fx.Tensor, SPARSE_MASK: fx.Tensor,
        DOUT: fx.Tensor, LSE: fx.Tensor, DELTAS: fx.Tensor, SINK: fx.Tensor,
        DQ: fx.Tensor, DK_LOCAL: fx.Tensor, DV_LOCAL: fx.Tensor,
        DGATHERED: fx.Tensor, DSINK: fx.Tensor,
        seq_len: fx.Int32, K_topk: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K_LOCAL)
        vl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V_LOCAL)
        g_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), GATHERED)
        sm_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), SPARSE_MASK)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DQ)
        # DGATHERED is the WARP-DISJOINT packed-bf16 split-K scratch
        # DGATHERED_SPLIT[B, Sq, K_topk, NUM_HEAD_GROUPS, D] (bf16). Each
        # head-group program writes its disjoint group_id stripe with a plain
        # packed 2xbf16 store (no atomic_fadd RMW); the launcher's finalize pass
        # sums the NUM_HEAD_GROUPS stripes back into the fp32 dgathered.
        dg_split_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DGATHERED)
        # Buffer resource for K_LOCAL so the streaming K tile can be fetched via
        # the hardware direct-to-LDS path (buffer_load_lds).
        kl_rsrc = buffer_ops.create_buffer_resource(K_LOCAL, max_size=True)
        # Buffer resource for GATHERED so the per-query gathered tile can be
        # fetched via the hardware direct-to-LDS path (buffer_load_lds),
        # mirroring the LOCAL-K staging.
        g_rsrc = buffer_ops.create_buffer_resource(GATHERED, max_size=True)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        deltas_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)
        dkl_rsrc = buffer_ops.create_buffer_resource(DK_LOCAL, max_size=True)
        dvl_rsrc = buffer_ops.create_buffer_resource(DV_LOCAL, max_size=True)
        dg_rsrc = buffer_ops.create_buffer_resource(DGATHERED, max_size=True)
        if const_expr(has_sink):
            sink_rsrc = buffer_ops.create_buffer_resource(SINK, max_size=True)
            dsink_rsrc = buffer_ops.create_buffer_resource(DSINK, max_size=True)

        # LDS staging buffer for the streaming LOCAL K tile.
        _smem_base = allocator.get_base()
        lds_kl = SmemPtr(_smem_base, lds_kl_offset, f16_ty,
                         shape=(LDS_RING_ELEMS,)).get()
        # Absolute LDS byte base of the staging view + the constant
        # operands shared by every buffer_load_lds (per-lane payload = D_PER_LANE
        # f16 == 16 B, soffset/offset 0, aux 1).
        lds_kl_base_idx = _memref.extract_aligned_pointer_as_index(lds_kl)
        _dma_size_kl = arith.constant(D_PER_LANE * 2, type=T.i32)
        _dma_soff0 = arith.constant(0, type=T.i32)
        _dma_off0 = arith.constant(0, type=T.i32)
        _dma_aux1 = arith.constant(1, type=T.i32)

        def _gep_load(base_ptr, elem_idx, vec_type, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_t, noWrapFlags=0,
            )
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store_f32(val, base_ptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=T.f32, noWrapFlags=0,
            )
            _llvm.StoreOp(val, gep.result)

        def load_f16_v(base_ptr, elem_idx, n):
            vt = T.vec(n, f16_ty)
            return _gep_load(base_ptr, elem_idx, vt, f16_ty)

        # Packed 2xbf16 store helpers for the warp-disjoint dgathered split
        # scratch. Two adjacent head-dim columns (d_off, d_off+1) owned by
        # the SAME lane are packed into one 32-bit word and stored race-free (the
        # group_id stripe is written by exactly one program, so no atomics).
        _v2_elem_type = T.vec(2, f16_ty)
        _v1i32_type = T.vec(1, T.i32)

        def _pack2_dg(f0, f1):
            if const_expr(dtype_str == "bf16"):
                _c16 = arith.constant(16, type=T.i32)
                _cmask = arith.constant(0xFFFF0000, type=T.i32)
                a = arith.ArithValue(f0).bitcast(T.i32)
                b = arith.ArithValue(f1).bitcast(T.i32)
                # low 16 bits = trunc(f0), high 16 bits = trunc(f1).
                p = arith.OrIOp(arith.AndIOp(b, _cmask).result,
                                arith.ShRUIOp(a, _c16).result).result
                return vector.bitcast(
                    _v2_elem_type,
                    vector.from_elements(_v1i32_type, [p]),
                )
            e0 = arith.trunc_f(f16_ty, f0)
            e1 = arith.trunc_f(f16_ty, f1)
            return vector.from_elements(_v2_elem_type, [e0, e1])

        def _gep_store_pack2(val, base_ptr, elem_idx_i64):
            # ``elem_idx_i64`` is a 64-bit element index: the split scratch can
            # exceed 2^31 elements for wide MHA shapes, so the index must stay
            # i64 all the way to the GEP (no index round-trip that might be
            # 32-bit on this target).
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [elem_idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=f16_ty, noWrapFlags=0,
            )
            _llvm.StoreOp(val, gep.result)

        pid_m_block = arith.index_cast(T.index, gpu.block_idx.x)
        pid_bh = arith.index_cast(T.index, gpu.block_idx.y)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        lane = tid
        # MFMA lane decomposition (16x16x32): lane%16 selects the
        # MFMA row (a key) / N-column (a query); lane//16 selects the 8-wide
        # K-subgroup of head_dim handled by this lane.
        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)

        # Head-grouping: grid Y enumerates (b, head_group). One program owns
        # HEAD_BLOCK contiguous query heads head_base + h for h in [0, HB).
        HB = HEAD_BLOCK
        bid = pid_bh // arith.index(NUM_HEAD_GROUPS)
        group_id = pid_bh % arith.index(NUM_HEAD_GROUPS)
        # This program's disjoint dgathered split stripe index.
        group_i32 = arith.index_cast(T.i32, group_id)
        num_groups_i32 = arith.constant(NUM_HEAD_GROUPS, type=T.i32)
        head_base = group_id * arith.index(HEAD_BLOCK)
        qhid_list = [head_base + arith.index(h) for h in range_constexpr(HB)]
        qhid_i32_list = [arith.index_cast(T.i32, qh) for qh in qhid_list]

        # Query-row blocking: this program owns QROW_BLOCK contiguous query rows
        # base_row + r for r in [0, QROW_BLOCK).
        QR = QROW_BLOCK
        base_row = pid_m_block * arith.index(QR)

        NEG_INF_F = -1.0e30
        c_neg_inf = arith.constant(NEG_INF_F, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_sm_scale = arith.constant(float(sm_scale), type=f32_ty)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        c_four_i32 = arith.constant(4, type=T.i32)
        c_zero_i32 = arith.constant(0, type=T.i32)

        zero_f32_vec = arith.constant_vector(0.0, T.vec(D_PER_LANE, f32_ty))

        def _load_afrag_packs(base_ptr, row_base):
            packs = []
            for ks in range_constexpr(K_STEPS):
                d_off = arith.index(ks * MFMA_K_STEP) + lane_div_16 * arith.index(8)
                packs.append(load_f16_v(base_ptr, row_base + d_off, 8))
            return packs

        def _load_afrag_packs_lds(lds_buf, row_elems):
            # Same MFMA A-frag pack layout as ``_load_afrag_packs`` but read
            # from the LDS-staged tile (row-major [k, HEAD_DIM]); row_elems
            # is the element-base of the staged key row within the buffer.
            packs = []
            for ks in range_constexpr(K_STEPS):
                d_off = arith.index(ks * MFMA_K_STEP) + lane_div_16 * arith.index(8)
                packs.append(vector.load_op(T.vec(8, f16_ty), lds_buf, [row_elems + d_off]))
            return packs

        # ---- per-row program state (index r in [0, QR)); row-only, shared
        # across all HEAD_BLOCK heads (pid_m / masking depend on the query row,
        # not the head). ----
        pid_m_list = []
        q_active_list = []
        pid_m_safe_list = []
        pid_m_i32_list = []
        pid_m_safe_i32_list = []
        for r in range_constexpr(QR):
            pid_m_r = base_row + arith.index(r)
            q_active_r = arith.cmpi(arith.CmpIPredicate.slt, pid_m_r, seq_len_v)
            pid_m_safe_r = arith.select(q_active_r, pid_m_r, arith.index(0))
            pid_m_list.append(pid_m_r)
            q_active_list.append(q_active_r)
            pid_m_safe_list.append(pid_m_safe_r)
            pid_m_i32_list.append(arith.index_cast(T.i32, pid_m_r))
            pid_m_safe_i32_list.append(arith.index_cast(T.i32, pid_m_safe_r))

        # ---- per-(head, row) program state. q / dO and lse / delta are indexed
        # by query head; loaded once and reused across every key tile. ----
        q_f32_hr = [[None] * QR for _ in range_constexpr(HB)]
        do_f32_hr = [[None] * QR for _ in range_constexpr(HB)]
        q_b_packs_hr = [[None] * QR for _ in range_constexpr(HB)]
        do_b_packs_hr = [[None] * QR for _ in range_constexpr(HB)]
        lse_val_hr = [[None] * QR for _ in range_constexpr(HB)]
        delta_val_hr = [[None] * QR for _ in range_constexpr(HB)]
        for h in range_constexpr(HB):
            qh = qhid_list[h]
            for r in range_constexpr(QR):
                q_active_r = q_active_list[r]
                pid_m_safe_r = pid_m_safe_list[r]
                q_row_base = (
                    (bid * arith.index(NUM_HEADS) + qh) * seq_len_v + pid_m_safe_r
                ) * arith.index(HEAD_DIM)
                q_lane_off = q_row_base + lane * arith.index(D_PER_LANE)
                q_vec_raw = load_f16_v(q_ptr, q_lane_off, D_PER_LANE)
                q_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), q_vec_raw)
                q_f32 = arith.select(q_active_r, q_f32, zero_f32_vec)
                q_f32_hr[h][r] = q_f32

                do_lane_off = q_lane_off
                do_vec_raw = load_f16_v(do_ptr, do_lane_off, D_PER_LANE)
                do_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), do_vec_raw)
                do_f32 = arith.select(q_active_r, do_f32, zero_f32_vec)
                do_f32_hr[h][r] = do_f32

                # Broadcast MFMA B operands (this head/row's query / dO)
                # in MFMA 16x16x32 B-frag layout; reused across all key tiles.
                qb = []
                dob = []
                for ks in range_constexpr(K_STEPS):
                    _d_off = arith.index(ks * MFMA_K_STEP) + lane_div_16 * arith.index(8)
                    qb.append(load_f16_v(q_ptr, q_row_base + _d_off, 8))
                    dob.append(load_f16_v(do_ptr, q_row_base + _d_off, 8))
                q_b_packs_hr[h][r] = qb
                do_b_packs_hr[h][r] = dob

                lse_delta_off = (
                    bid * arith.index(NUM_HEADS) + qh
                ) * seq_len_v + pid_m_safe_r
                lse_delta_off_i32 = arith.index_cast(T.i32, lse_delta_off)
                lse_val_hr[h][r] = buffer_ops.buffer_load(
                    lse_rsrc, lse_delta_off_i32, vec_width=1, dtype=f32_ty,
                )
                delta_val_hr[h][r] = buffer_ops.buffer_load(
                    deltas_rsrc, lse_delta_off_i32, vec_width=1, dtype=f32_ty,
                )

        bid_i32 = arith.index_cast(T.i32, bid)
        lane_i32 = arith.index_cast(T.i32, lane)
        head_dim_i32 = arith.constant(HEAD_DIM, type=T.i32)
        d_per_lane_i32 = arith.constant(D_PER_LANE, type=T.i32)

        if const_expr(has_sink):
            for h in range_constexpr(HB):
                qh_i32 = qhid_i32_list[h]
                sink_h = buffer_ops.buffer_load(
                    sink_rsrc, qh_i32, vec_width=1, dtype=f32_ty,
                )
                sink_h = rocdl.readfirstlane(f32_ty, sink_h)
                for r in range_constexpr(QR):
                    sub_sh = arith.SubFOp(sink_h, lse_val_hr[h][r], fastmath=fm_fast).result
                    p_sink = math_dialect.exp(sub_sh, fastmath=fm_fast)
                    neg_p_sink = arith.SubFOp(c_zero_f, p_sink, fastmath=fm_fast).result
                    dsink_contrib = arith.MulFOp(
                        neg_p_sink, delta_val_hr[h][r], fastmath=fm_fast,
                    ).result
                    is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.index(0))
                    do_sink = arith.AndIOp(is_lane0, q_active_list[r]).result
                    _if_sink = scf.IfOp(do_sink, [], has_else=False)
                    with ir.InsertionPoint(_if_sink.then_block):
                        _dsink_byte_off = arith.MulIOp(qh_i32, c_four_i32).result
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            dsink_contrib, dsink_rsrc, _dsink_byte_off,
                            c_zero_i32, c_zero_i32,
                        )
                        scf.YieldOp([])

        SWA = arith.index(int(swa_window))
        BN_idx = arith.index(BLOCK_N)
        # Union SWA window across the QR contiguous rows: the lower bound comes
        # from the FIRST row (smallest pid_m), the upper bound from the LAST row
        # (largest pid_m, clamped to seq_len). Per-row causal/SWA masking inside
        # the loop restricts each row to its own sub-window.
        _p1_first = pid_m_list[0] + arith.index(1)
        _ge_w = arith.cmpi(arith.CmpIPredicate.sge, _p1_first, SWA)
        _n_lo_raw = arith.select(_ge_w, _p1_first - SWA, arith.index(0))
        n_loop_start = (_n_lo_raw // BN_idx) * BN_idx
        _p1_last = pid_m_list[QR - 1] + arith.index(1)
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _p1_last, seq_len_v)
        n_loop_end_row = arith.select(_le_seq, _p1_last, seq_len_v)
        n_loop_end_blk = (
            (n_loop_end_row + BN_idx - arith.index(1)) // BN_idx
        ) * BN_idx

        # dq accumulators: HB heads x QR rows x D_PER_LANE lanes, flattened into
        # iter_args as [(h*QR + r)*D_PER_LANE + d].
        N_ACC = HB * QR * D_PER_LANE
        _PAD = 1 if N_ACC == 1 else 0
        init_local = []
        for _ in range_constexpr(N_ACC):
            init_local.append(c_zero_f)
        for _ in range_constexpr(_PAD):
            init_local.append(c_zero_f)

        seq_len_i32 = arith.index_cast(T.i32, seq_len_v)
        w_i32 = arith.constant(int(swa_window), type=T.i32)
        K_topk_i32 = arith.index_cast(T.i32, K_topk_v)

        # ==== LOCAL SWA loop ====
        for n_start, inner_args, loop_results_local in scf.for_(
            n_loop_start, n_loop_end_blk, BN_idx, iter_args=init_local,
        ):
            dq_accs = [
                [[inner_args[(h * QR + r) * D_PER_LANE + d]
                  for d in range_constexpr(D_PER_LANE)]
                 for r in range_constexpr(QR)]
                for h in range_constexpr(HB)
            ]
            n_start_i32 = arith.index_cast(T.i32, n_start)

            # The streaming K_local per-lane vectors (and the V/K MFMA A-frags
            # below) depend only on the key column -- NOT on the
            # query row, and (for MQA) NOT on the query head. Load them ONCE for
            # this BLOCK_N tile and consume across all QR rows x HB heads -> the
            # shared local K/V cache lines are read from DRAM/L2 once per block
            # instead of once per (head, query row).
            kv_col_i32_cache = []
            for n_off in range_constexpr(BLOCK_N):
                kv_col_i32 = arith.AddIOp(
                    n_start_i32, arith.constant(n_off, type=T.i32),
                ).result
                kv_col_i32_cache.append(kv_col_i32)
                is_oob = arith.cmpi(arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                kv_col_idx = arith.index_cast(T.index, kv_col_i32)
                kv_col_safe = arith.select(is_oob, arith.index(0), kv_col_idx)
                if const_expr(mqa_kv):
                    # Shared K/V across heads: K_LOCAL is [B, Sq, D].
                    kl_row_base = (bid * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                else:
                    # Per-head K/V (MHA, HB==1): K_LOCAL is [B, HQ, Sq, D].
                    kl_row_base = (
                        (bid * arith.index(NUM_HEADS) + qhid_list[0]) * seq_len_v
                        + kv_col_safe
                    ) * arith.index(HEAD_DIM)
                kl_lane_off = kl_row_base + lane * arith.index(D_PER_LANE)
                # Hardware direct global->LDS load. This lane's
                # D_PER_LANE contiguous f16 (one dwordx4) stream straight from
                # K_LOCAL device memory into the LDS staging buffer, with no
                # VGPR landing + ds_write. buffer_load_lds writes lane L's 16 B
                # payload at lds_base + L*size, so column n_off occupies a
                # warp-contiguous WARP_SIZE*D_PER_LANE f16 slot.
                kl_byte_off = arith.index_cast(T.i32, kl_lane_off * arith.index(2))
                # Padded (skewed) row stride LDS_KL_ROW_STRIDE.
                lds_byte = lds_kl_base_idx + arith.index(
                    n_off * LDS_KL_ROW_STRIDE * 2
                )
                lds_i64 = arith.index_cast(T.i64, lds_byte)
                lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_lane0).result
                rocdl.raw_ptr_buffer_load_lds(
                    kl_rsrc, lds_ptr, _dma_size_kl, kl_byte_off,
                    _dma_soff0, _dma_off0, _dma_aux1,
                )
            # The direct-to-LDS load completion is tracked by vmcnt
            # (the global fetch AND its LDS write retire together); the AMDGPU
            # backend does not auto-insert a dependency between buffer_load_lds
            # and the consume ds_read, so wait on ALL counters and barrier
            # before any lane reads its staged tile back.
            rocdl.s_waitcnt(0)
            gpu.barrier()

            # Load each k_local / v_local MFMA A-frag tile ONCE, then reduce it
            # against every (head, query row) broadcast B-frag
            # (q / dO). For MQA the A-frag is head-independent so it is hoisted
            # above the head loop; only the cheap per-(head,row) MFMA + readlane
            # is replicated. For MHA (HB==1) the A-frag is per-head and loaded
            # inside the (single-iteration) head loop.
            qk_full_cache = [[[None] * BLOCK_N for _ in range_constexpr(QR)]
                             for _ in range_constexpr(HB)]
            dp_cache = [[[None] * BLOCK_N for _ in range_constexpr(QR)]
                        for _ in range_constexpr(HB)]
            for mt in range_constexpr(N_MTILES):
                key_rel = arith.index(mt * 16) + lane_mod_16
                key_abs = n_start + key_rel
                key_abs_i32 = arith.index_cast(T.i32, key_abs)
                is_oob_k = arith.cmpi(arith.CmpIPredicate.sge, key_abs_i32, seq_len_i32)
                key_safe = arith.select(is_oob_k, arith.index(0), key_abs)
                if const_expr(mqa_kv):
                    a_row_base = (bid * seq_len_v + key_safe) * arith.index(HEAD_DIM)
                    k_afrag = _load_afrag_packs(kl_ptr, a_row_base)
                    v_afrag = _load_afrag_packs(vl_ptr, a_row_base)
                for h in range_constexpr(HB):
                    if const_expr(not mqa_kv):
                        a_row_base = (
                            (bid * arith.index(NUM_HEADS) + qhid_list[h]) * seq_len_v
                            + key_safe
                        ) * arith.index(HEAD_DIM)
                        k_afrag = _load_afrag_packs(kl_ptr, a_row_base)
                        v_afrag = _load_afrag_packs(vl_ptr, a_row_base)
                    for r in range_constexpr(QR):
                        qk_scalars = mfma_mv_reduce_16(k_afrag, q_b_packs_hr[h][r], 16, dtype_str)
                        dp_scalars = mfma_mv_reduce_16(v_afrag, do_b_packs_hr[h][r], 16, dtype_str)
                        for j in range_constexpr(16):
                            n_off = mt * 16 + j
                            qk_full_cache[h][r][n_off] = qk_scalars[j]
                            dp_cache[h][r][n_off] = dp_scalars[j]

            # Per-(head,row) consume: softmax recompute, dq accumulation. Masking
            # is per row; lse/delta per (head,row); the LDS-staged kl tile is
            # shared across heads (MQA) and rows.
            for h in range_constexpr(HB):
                for r in range_constexpr(QR):
                    pid_m_i32_r = pid_m_i32_list[r]
                    lse_val_r = lse_val_hr[h][r]
                    delta_val_r = delta_val_hr[h][r]
                    for n_off in range_constexpr(BLOCK_N):
                        kv_col_i32 = kv_col_i32_cache[n_off]
                        _kv_plus_w = arith.AddIOp(kv_col_i32, w_i32).result
                        is_swa = arith.cmpi(arith.CmpIPredicate.sle, _kv_plus_w, pid_m_i32_r)
                        is_causal = arith.cmpi(arith.CmpIPredicate.sgt, kv_col_i32, pid_m_i32_r)
                        is_oob = arith.cmpi(arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                        bad = arith.OrIOp(
                            arith.OrIOp(is_causal, is_swa).result, is_oob,
                        ).result

                        qk_full = qk_full_cache[h][r][n_off]
                        qk_scaled = arith.MulFOp(qk_full, c_sm_scale, fastmath=fm_fast).result
                        qk_masked = arith.select(bad, c_neg_inf, qk_scaled)
                        diff_qk = arith.SubFOp(qk_masked, lse_val_r, fastmath=fm_fast).result
                        p = math_dialect.exp(diff_qk, fastmath=fm_fast)

                        dp = dp_cache[h][r][n_off]
                        diff = arith.SubFOp(dp, delta_val_r, fastmath=fm_fast).result
                        ds = arith.MulFOp(p, diff, fastmath=fm_fast).result
                        ds_scaled = arith.MulFOp(ds, c_sm_scale, fastmath=fm_fast).result
                        # ds_read this lane's staged K tile back from
                        # LDS (warp-contiguous: column n_off slot at
                        # [n_off*WARP_SIZE*D_PER_LANE], lane at +lane*D_PER_LANE)
                        # and extf to f32 at point of use. Shared across heads
                        # (MQA local K is head-invariant).
                        lds_kl_idx = (
                            arith.index(n_off * LDS_KL_ROW_STRIDE)
                            + lane * arith.index(D_PER_LANE)
                        )
                        kl_vec_ld = vector.load_op(
                            T.vec(D_PER_LANE, f16_ty), lds_kl, [lds_kl_idx],
                        )
                        kl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), kl_vec_ld)

                        for d_off in range_constexpr(D_PER_LANE):
                            klv = vector.extract(kl_f32, static_position=[d_off], dynamic_position=[])
                            contrib = arith.MulFOp(ds_scaled, klv, fastmath=fm_fast).result
                            dq_accs[h][r][d_off] = arith.AddFOp(
                                dq_accs[h][r][d_off], contrib, fastmath=fm_fast,
                            ).result

            _yield = []
            for h in range_constexpr(HB):
                for r in range_constexpr(QR):
                    for d in range_constexpr(D_PER_LANE):
                        _yield.append(dq_accs[h][r][d])
            for _ in range_constexpr(_PAD):
                _yield.append(c_zero_f)
            yield _yield

        dq_accs = [
            [[loop_results_local[(h * QR + r) * D_PER_LANE + d]
              for d in range_constexpr(D_PER_LANE)]
             for r in range_constexpr(QR)]
            for h in range_constexpr(HB)
        ]

        # ==== GATHERED branch ====
        # The GATHERED tensor is indexed by query row ([B, Sq, K_topk, D]) with
        # NO head axis -> the gathered g_afrag (score recompute) and g_vec (dq /
        # dgathered values) are HEAD-INVARIANT. The query rows are looped
        # outermost; for each (row, k-tile) the head-shared g_afrag / g_vec are
        # fetched from HBM ONCE and reused across all HB heads (only the per-head
        # q / dO B-frags differ). The sparse-mask (sm_val) and causal mask (bad)
        # are also head-independent and computed once per row. The head-shared
        # dgathered contributions for one (row, k_pos) are summed in-register
        # across the HB heads before a SINGLE atomic_fadd (was HB atomics).
        # Adjacent query rows still read disjoint gathered memory, so the dq
        # accumulators flow through as iter_args [(h*QR + r)*D_PER_LANE + d].
        if const_expr(has_sparse):
            init_sparse = []
            for h in range_constexpr(HB):
                for r in range_constexpr(QR):
                    for d in range_constexpr(D_PER_LANE):
                        init_sparse.append(dq_accs[h][r][d])
            for _ in range_constexpr(_PAD):
                init_sparse.append(c_zero_f)
            for k_start, inner_args_g, loop_results_g in scf.for_(
                arith.index(0), K_topk_v, arith.index(BLOCK_K),
                iter_args=init_sparse,
            ):
                dq_accs_g = [
                    [[inner_args_g[(h * QR + r) * D_PER_LANE + d]
                      for d in range_constexpr(D_PER_LANE)]
                     for r in range_constexpr(QR)]
                    for h in range_constexpr(HB)
                ]
                k_start_i32 = arith.index_cast(T.i32, k_start)

                # Sparse/causal tile-skip. Precompute the
                # head-independent per-(row, k_off) sparse-mask bias + causal/oob
                # mask ONCE, and OR-reduce a wave-uniform "tile has >= 1 live
                # key" flag. A gathered key contributes a strictly-zero gradient
                # (p == 0 exactly, so ds == 0 and dg_val == 0) when it is out of
                # range (k_pos >= K_topk), its query row is inactive, OR its
                # sparse-mask bias is the -inf sentinel. A BLOCK_K tile with no
                # live key across the owned query rows can therefore be skipped
                # entirely -- its gathered K/V/dO global loads, the score-recompute
                # MFMA, and the dgathered atomic_fadd of zeros -- with
                # bitwise-identical results (skipping an all-zero contribution is
                # algebraically exact, and the carried dq accumulators pass
                # through unchanged). Every predicate operand is lane-invariant,
                # so the branch is uniform across the wave.
                _i1_ty = ir.IntegerType.get_signless(1)
                c_false_i1 = arith.constant(0, type=_i1_ty)
                c_mask_thresh = arith.constant(NEG_INF_F, type=f32_ty)
                tile_alive = c_false_i1
                sm_caches = []
                for r in range_constexpr(QR):
                    pid_m_safe_r = pid_m_safe_list[r]
                    pid_m_i32_r = pid_m_i32_list[r]
                    q_active_r = q_active_list[r]

                    # Head-independent per-(row,k) state: sparse-mask, causal
                    # mask, k positions. Computed once and reused across heads.
                    sm_val_cache = []
                    bad_cache_g = []
                    k_pos_i32_cache = []

                    for k_off in range_constexpr(BLOCK_K):
                        k_pos_i32 = arith.AddIOp(
                            k_start_i32, arith.constant(k_off, type=T.i32),
                        ).result
                        k_pos_i32_cache.append(k_pos_i32)
                        is_oob = arith.cmpi(arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                        k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                        k_pos_safe = arith.select(is_oob, arith.index(0), k_pos_idx)

                        sm_off = (bid * seq_len_v + pid_m_safe_r) * K_topk_v + k_pos_safe
                        sm_raw_v1 = _gep_load(sm_ptr, sm_off, T.vec(1, elem_type), elem_type)
                        sm_raw = vector.extract(sm_raw_v1, static_position=[0], dynamic_position=[])
                        sm_val = arith.extf(f32_ty, sm_raw)
                        sm_val = arith.select(is_oob, c_zero_f, sm_val)
                        sm_val_cache.append(sm_val)

                        bad = arith.OrIOp(
                            is_oob,
                            arith.cmpi(arith.CmpIPredicate.sge, pid_m_i32_r, seq_len_i32),
                        ).result
                        bad_cache_g.append(bad)

                        # Live-key predicate (exact): in-range AND row-active AND
                        # the sparse bias is strictly greater than the -inf
                        # sentinel (a finite bias keeps p > 0, so the key must
                        # stay live; only sentinel-masked keys are skippable).
                        in_range_pred = arith.cmpi(
                            arith.CmpIPredicate.slt, k_pos_i32, K_topk_i32)
                        not_masked = arith.cmpf(
                            arith.CmpFPredicate.OGT, sm_val, c_mask_thresh)
                        alive_rk = arith.AndIOp(
                            arith.AndIOp(in_range_pred, q_active_r).result,
                            not_masked,
                        ).result
                        tile_alive = arith.OrIOp(tile_alive, alive_rk).result

                    sm_caches.append((sm_val_cache, bad_cache_g, k_pos_i32_cache))

                # Skip the whole tile when no live key is present; carried dq
                # accumulators flow through unchanged via the else branch.
                _if_tile = scf.IfOp(
                    tile_alive, [f32_ty] * (N_ACC + _PAD), has_else=True,
                )
                with ir.InsertionPoint(_if_tile.then_block):
                    # Producer/consumer LDS ring over the QR query rows.
                    # ``_issue_gathered_row`` is the PRODUCER: it streams query
                    # row ``r``'s gathered [BLOCK_K, HEAD_DIM] tile global->LDS
                    # (buffer_load_lds) into ring ``stage``. ``_consume_row`` is
                    # the CONSUMER: it MFMA-recomputes scores / dP, the softmax
                    # probs, and accumulates dq + the dgathered atomics out of an
                    # already-resident ``stage``. The pipeline issues the NEXT
                    # row's producer load BEFORE consuming the current row, so the
                    # no-reuse cold gathered HBM load overlaps the latency-bound
                    # MFMA chain instead of being fully exposed by the old
                    # "issue -> immediate s_waitcnt(0)" drain. OOB columns
                    # (k_pos >= K_topk) stage gathered row 0 exactly as before, so
                    # results are bitwise-identical.
                    def _issue_gathered_row(r, stage):
                        pid_m_safe_r = pid_m_safe_list[r]
                        _k_pos_i32_cache = sm_caches[r][2]
                        _stage_byte = ring_stage_byte_base(stage, LDS_KL_ELEMS)
                        for k_off in range_constexpr(BLOCK_K):
                            k_pos_i32_s = _k_pos_i32_cache[k_off]
                            is_oob_s = arith.cmpi(
                                arith.CmpIPredicate.sge, k_pos_i32_s, K_topk_i32)
                            k_pos_idx_s = arith.index_cast(T.index, k_pos_i32_s)
                            k_pos_safe_s = arith.select(is_oob_s, arith.index(0), k_pos_idx_s)
                            g_row_base_s = (
                                (bid * seq_len_v + pid_m_safe_r) * K_topk_v + k_pos_safe_s
                            ) * arith.index(HEAD_DIM)
                            g_lane_off_s = g_row_base_s + lane * arith.index(D_PER_LANE)
                            g_byte_off = arith.index_cast(T.i32, g_lane_off_s * arith.index(2))
                            # Padded (skewed) row stride LDS_KL_ROW_STRIDE.
                            lds_byte = lds_kl_base_idx + arith.index(
                                _stage_byte + k_off * LDS_KL_ROW_STRIDE * 2
                            )
                            lds_i64 = arith.index_cast(T.i64, lds_byte)
                            lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                            lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_lane0).result
                            rocdl.raw_ptr_buffer_load_lds(
                                g_rsrc, lds_ptr, _dma_size_kl, g_byte_off,
                                _dma_soff0, _dma_off0, _dma_aux1,
                            )

                    def _consume_gathered_row(r, stage):
                        pid_m_safe_i32_r = pid_m_safe_i32_list[r]
                        q_active_r = q_active_list[r]
                        sm_val_cache, bad_cache_g, k_pos_i32_cache = sm_caches[r]
                        _stage_elem = ring_stage_elem_base(stage, LDS_KL_ELEMS)

                        # Head-shared g_afrag: read ONCE per k-tile from the LDS
                        # staged tile (row k_pos_rel at [stage + k_pos_rel*HEAD_DIM]),
                        # MFMA-reduced against every head's q / dO B-frag.
                        qk_full_cache_g = [[None] * BLOCK_K for _ in range_constexpr(HB)]
                        dp_cache_g = [[None] * BLOCK_K for _ in range_constexpr(HB)]
                        for mt in range_constexpr(K_MTILES):
                            k_pos_rel = arith.index(mt * 16) + lane_mod_16
                            # Padded (skewed) row stride LDS_KL_ROW_STRIDE.
                            ga_row_elems = (
                                arith.index(_stage_elem)
                                + k_pos_rel * arith.index(LDS_KL_ROW_STRIDE)
                            )
                            g_afrag = _load_afrag_packs_lds(lds_kl, ga_row_elems)
                            for h in range_constexpr(HB):
                                qk_scalars = mfma_mv_reduce_16(g_afrag, q_b_packs_hr[h][r], 16, dtype_str)
                                dp_scalars = mfma_mv_reduce_16(g_afrag, do_b_packs_hr[h][r], 16, dtype_str)
                                for j in range_constexpr(16):
                                    k_off = mt * 16 + j
                                    qk_full_cache_g[h][k_off] = qk_scalars[j]
                                    dp_cache_g[h][k_off] = dp_scalars[j]

                        # Per-head softmax-prob recompute.
                        p_cache_g = [[None] * BLOCK_K for _ in range_constexpr(HB)]
                        for h in range_constexpr(HB):
                            lse_val_r = lse_val_hr[h][r]
                            for k_off in range_constexpr(BLOCK_K):
                                sm_val = sm_val_cache[k_off]
                                bad = bad_cache_g[k_off]
                                qk_full = qk_full_cache_g[h][k_off]
                                qk_scaled = arith.MulFOp(qk_full, c_sm_scale, fastmath=fm_fast).result
                                qk_biased = arith.AddFOp(qk_scaled, sm_val, fastmath=fm_fast).result
                                qk_masked = arith.select(bad, c_neg_inf, qk_biased)
                                diff_qk = arith.SubFOp(qk_masked, lse_val_r, fastmath=fm_fast).result
                                p = math_dialect.exp(diff_qk, fastmath=fm_fast)
                                p_cache_g[h][k_off] = p

                        # Consume: load the head-shared g_vec ONCE per k_off, loop
                        # the HB heads for dq accumulation and the dgathered
                        # in-register cross-head pre-sum, then ONE atomic_fadd of
                        # the summed gathered grad (offset is head-invariant).
                        for k_off in range_constexpr(BLOCK_K):
                            k_pos_i32 = k_pos_i32_cache[k_off]
                            # Read this lane's gathered row back from the LDS-staged
                            # ring tile (stage base + column k_off at
                            # k_off*WARP_SIZE*D_PER_LANE == k_off*HEAD_DIM, lane at
                            # +lane*D_PER_LANE).
                            lds_g_idx = (
                                arith.index(_stage_elem + k_off * LDS_KL_ROW_STRIDE)
                                + lane * arith.index(D_PER_LANE)
                            )
                            g_vec = vector.load_op(
                                T.vec(D_PER_LANE, f16_ty), lds_kl, [lds_g_idx],
                            )
                            g_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), g_vec)

                            dg_accum = [c_zero_f for _ in range_constexpr(D_PER_LANE)]
                            for h in range_constexpr(HB):
                                p = p_cache_g[h][k_off]
                                dp = dp_cache_g[h][k_off]
                                delta_val_r = delta_val_hr[h][r]
                                diff = arith.SubFOp(dp, delta_val_r, fastmath=fm_fast).result
                                ds = arith.MulFOp(p, diff, fastmath=fm_fast).result
                                ds_scaled = arith.MulFOp(ds, c_sm_scale, fastmath=fm_fast).result

                                q_f32_h = q_f32_hr[h][r]
                                do_f32_h = do_f32_hr[h][r]
                                for d_off in range_constexpr(D_PER_LANE):
                                    gv = vector.extract(g_f32, static_position=[d_off], dynamic_position=[])
                                    contrib = arith.MulFOp(ds_scaled, gv, fastmath=fm_fast).result
                                    dq_accs_g[h][r][d_off] = arith.AddFOp(
                                        dq_accs_g[h][r][d_off], contrib, fastmath=fm_fast,
                                    ).result

                                    qv = vector.extract(q_f32_h, static_position=[d_off], dynamic_position=[])
                                    dov = vector.extract(do_f32_h, static_position=[d_off], dynamic_position=[])
                                    t1 = arith.MulFOp(ds_scaled, qv, fastmath=fm_fast).result
                                    t2 = arith.MulFOp(p, dov, fastmath=fm_fast).result
                                    dg_val = arith.AddFOp(t1, t2, fastmath=fm_fast).result
                                    dg_accum[d_off] = arith.AddFOp(
                                        dg_accum[d_off], dg_val, fastmath=fm_fast,
                                    ).result

                            in_range_k = arith.cmpi(arith.CmpIPredicate.slt, k_pos_i32, K_topk_i32)
                            do_atom_k = arith.AndIOp(in_range_k, q_active_r).result
                            _if_dg = scf.IfOp(do_atom_k, [], has_else=False)
                            with ir.InsertionPoint(_if_dg.then_block):
                                # Packed 2xbf16 store into this program's
                                # DISJOINT group_id stripe of the split scratch
                                # (no atomic_fadd RMW, half the store bytes). The
                                # finalize pass reduces the stripes to fp32.
                                _row_d = dgathered_split_elem_base(
                                    bid_i32, seq_len_i32, pid_m_safe_i32_r, K_topk_i32,
                                    k_pos_i32, group_i32, num_groups_i32,
                                    lane_i32, d_per_lane_i32, head_dim_i32,
                                )
                                for j in range_constexpr(D_PER_LANE // 2):
                                    d0 = 2 * j
                                    packed = _pack2_dg(dg_accum[d0], dg_accum[d0 + 1])
                                    elem_idx = arith.AddIOp(
                                        _row_d, arith.constant(d0, type=T.i64),
                                    ).result
                                    _gep_store_pack2(packed, dg_split_ptr, elem_idx)
                                scf.YieldOp([])

                    # ---- software-pipelined producer/consumer ring drive ----
                    # 3-stage ring. Prologue frees the staging buffer
                    # from the LOCAL loop's last reads, then PREFETCHES the first
                    # TWO pipeline slots (rows 0 and 1, when present) into ring
                    # stages 0 and 1 so the producer is a full extra slot ahead of
                    # the consumer before the steady-state loop. Both
                    # buffer_load_lds DMAs are made resident with a single drain so
                    # consume(0) and consume(1) never stall on HBM.
                    gpu.barrier()
                    _issue_gathered_row(0, 0)
                    _prime_row1 = const_expr(QR > 1)
                    if _prime_row1:
                        _issue_gathered_row(1, 1)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    for r in range_constexpr(QR):
                        cur_stage = r % LDS_RING_STAGES
                        _has_next = const_expr((r + 1) < QR)
                        _has_next2 = const_expr((r + 2) < QR)
                        if _has_next2:
                            # PRODUCER: prefetch the slot TWO rows ahead into its
                            # ring stage (three-way slot rotation s<-n1, n1<-n2,
                            # n2<-(r+2)%3). This buffer_load_lds DMA is left in
                            # flight and overlaps the CONSUMER MFMA/atomics below;
                            # row r+1 is already resident (primed in the prologue or
                            # drained two iterations back), so the producer now runs
                            # a full extra slot ahead of the consume.
                            _issue_gathered_row(r + 2, (r + 2) % LDS_RING_STAGES)
                        # CONSUMER on the already-resident current stage.
                        _consume_gathered_row(r, cur_stage)
                        if _has_next:
                            # Two-slot drain: wait the in-flight prefetch DMA + this
                            # row's dgathered atomics (all vmcnt-tracked) and
                            # barrier. The slot needed next (row r+1) is resident,
                            # the slot r+2 just prefetched is now resident/visible,
                            # AND every lane has finished reading the current stage,
                            # so the 3-deep ring slot is safe to reuse three rows on.
                            rocdl.s_waitcnt(0)
                            gpu.barrier()

                    _yield_then = []
                    for h in range_constexpr(HB):
                        for r in range_constexpr(QR):
                            for d in range_constexpr(D_PER_LANE):
                                _yield_then.append(dq_accs_g[h][r][d])
                    for _ in range_constexpr(_PAD):
                        _yield_then.append(c_zero_f)
                    scf.YieldOp(_yield_then)

                with ir.InsertionPoint(_if_tile.else_block):
                    _yield_else = [inner_args_g[i] for i in range_constexpr(N_ACC + _PAD)]
                    scf.YieldOp(_yield_else)

                _yield_g = [_if_tile.results[i] for i in range_constexpr(N_ACC + _PAD)]
                yield _yield_g

            dq_accs = [
                [[loop_results_g[(h * QR + r) * D_PER_LANE + d]
                  for d in range_constexpr(D_PER_LANE)]
                 for r in range_constexpr(QR)]
                for h in range_constexpr(HB)
            ]

        # ==== Store dq direct (per head, per row) ====
        for h in range_constexpr(HB):
            qh = qhid_list[h]
            for r in range_constexpr(QR):
                _o_guard = scf.IfOp(q_active_list[r], [], has_else=False)
                with ir.InsertionPoint(_o_guard.then_block):
                    dq_row_base = (
                        (bid * arith.index(NUM_HEADS) + qh) * seq_len_v
                        + pid_m_safe_list[r]
                    ) * arith.index(HEAD_DIM)
                    dq_lane_off = dq_row_base + lane * arith.index(D_PER_LANE)
                    for d_off in range_constexpr(D_PER_LANE):
                        elem_off = dq_lane_off + arith.index(d_off)
                        _gep_store_f32(dq_accs[h][r][d_off], dq_ptr, elem_off)
                    scf.YieldOp([])

    @flyc.jit
    def launch_csa_bwd_full(
        Q: fx.Tensor, K_LOCAL: fx.Tensor, V_LOCAL: fx.Tensor,
        GATHERED: fx.Tensor, SPARSE_MASK: fx.Tensor,
        DOUT: fx.Tensor, LSE: fx.Tensor, DELTAS: fx.Tensor, SINK: fx.Tensor,
        DQ: fx.Tensor, DK_LOCAL: fx.Tensor, DV_LOCAL: fx.Tensor,
        DGATHERED: fx.Tensor, DSINK: fx.Tensor,
        batch_size: fx.Int32, seq_len: fx.Int32, K_topk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        # Query-row blocking: each program owns QROW_BLOCK query rows,
        # so grid X = ceil(Sq / QROW_BLOCK).
        grid_x = (sl_idx + arith.index(QROW_BLOCK - 1)) // arith.index(QROW_BLOCK)
        # Head-grouping: each program owns HEAD_BLOCK query heads, so
        # grid Y = B * (HQ / HEAD_BLOCK) head-groups.
        grid_y = bs_idx * arith.index(NUM_HEAD_GROUPS)

        launcher = csa_bwd_full_kernel(
            Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK,
            DOUT, LSE, DELTAS, SINK,
            DQ, DK_LOCAL, DV_LOCAL, DGATHERED, DSINK,
            seq_len, K_topk,
        )

        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)

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

    compile_hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_bwd_full(*args, **kwargs)

    def _compile(
        Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK,
        DOUT, LSE, DELTAS, SINK,
        DQ, DK_LOCAL, DV_LOCAL, DGATHERED, DSINK,
        batch_size, seq_len, K_topk, stream=None,
    ):
        with CompilationContext.compile_hints(compile_hints):
            return flyc.compile(
                launch_csa_bwd_full,
                Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK,
                DOUT, LSE, DELTAS, SINK,
                DQ, DK_LOCAL, DV_LOCAL, DGATHERED, DSINK,
                batch_size, seq_len, K_topk, fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch


build_csa_bwd_full_module_primary = build_csa_bwd_full_module
