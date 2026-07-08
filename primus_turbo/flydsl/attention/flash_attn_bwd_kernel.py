# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn backward kernel builders for FlyDSL (gfx950 / MI355X).

Forked from flash_attn_fwd_kernel.py: reuses the verified forward machine
(KV-head-major XCD remap, d64-general K XOR-swizzle, K@Q^T GEMM1 so S lands in
PV-aligned registers, ds_read_tr16_b64 hardware transpose read, causal per-lane
select, wave64 peer reduce) and only swaps the epilogue for the backward math.

Three build modes (Q-outer, one work-group owns one q-tile -> single write, no
float atomics -> deterministic):

  mode="delta": delta[b,hq,s] = sum_j P_ij * dP_ij   (fp32 P, fp32 accumulate).
      P recomputed from the saved LSE (softmax prob); dP = dO @ V^T reuses the
      GEMM1 template with V as the "K" A-operand and dO as the "Q" B-operand.

  mode="dq": dQ = sm_scale * sum_j dS_ij * k_j  with dS = P (.) (dP - delta).
      P/dP recomputed exactly as in the delta kernel (bit-identical), so the
      near-diagonal cancellation sum_j dS_ij = 0 stays exact against the
      consistent delta buffer. dQ = dS @ K reuses the GEMM2 template with K read
      transposed (ds_read_tr) as the "V" A-operand and dS as the "P" B-operand;
      for head_dim=64 the K-swizzle equals the forward V-swizzle so that read
      path is reused verbatim. Result [D,q] is stored transposed like O.

  mode="fused_dq_delta": folds the delta and dq passes into one kv-loop so S/P/dP
      are recomputed once (not twice), removing the whole delta kernel's exp2
      pass. Uses the identity dQ = sm*(A - delta*B) with, accumulated in one pass,
      delta_i = sum_j P_ij*dP_ij, A_i = sum_j (P_ij*dP_ij) k_j, B_i = sum_j P_ij k_j.
      A and B feed their P/(P*dP) B-operands (and the transpose-read K) as single fp16
      (10-bit mantissa = tf32) so the epilogue subtraction survives the near-diagonal
      catastrophic cancellation at half the double-bf16 pack+MFMA cost (single-bf16's
      8-bit mantissa fails the s8192 dq gate). Writes both dQ (transposed) and the
      recomputed fp32 delta (for the downstream dkdv kernel).

Target: gfx950 only (32x32x16 bf16 MFMA, ds_read_tr16_b64, permlane32_swap +
cvt_pk_bf16_f32 store). bf16, causal, GQA/MQA (num_kv_heads <= num_heads).
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LOG2E = host_math.log2(host_math.e)


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def dtype_to_elem_type(dtype_str):
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "f16":
        return fx.Float16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'bf16' or 'f16')")


def build_flash_attn_bwd_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    block_m=128,
    num_kv_heads=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    mode="dq",
    enable_dma=True,
):
    """Build one backward launcher. ``mode`` in {"delta", "dq", "fused_dq_delta"}."""
    assert mode in ("delta", "dq", "fused_dq_delta"), mode
    assert causal, "backward kernel is causal-only for the GPT-OSS campaign"
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "backward kernel targets gfx950"
    assert dtype_str == "bf16", "backward kernel targets bf16"

    # DMA-to-LDS (buffer_load_dwordx4 ... lds) bypasses the VGPR staging of the K/V
    # tile loads (gfx950+ only); relieves register pressure / removes the ds_write
    # spill on this 168-VGPR delta/dq kernel.
    ENABLE_DMA = enable_dma and not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64
    BLOCK_M = block_m
    flat_work_group_size = 256 if BLOCK_M <= 128 else 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES

    K_STEP_QK = 16
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0 and head_dim >= 64
    assert head_dim % 16 == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    # K and V both go to LDS in the same K-swizzle layout (stride = head_dim);
    # K is additionally read transposed (ds_read_tr) for the dQ GEMM.
    K_STRIDE = HEAD_DIM
    LDS_TILE = BLOCK_N * K_STRIDE
    LDS_V_BASE = LDS_TILE
    # Fused dual-tile: keep a 2nd K copy in fp16 (LDS_K16_BASE) alongside the bf16
    # K/V tiles. GEMM1 (S=K@Q^T) reads the bf16 K tile; GEMM2's A/B fp16 MFMA reads
    # this fp16 tile via a transpose-read directly, dropping the per-read bf16->fp16
    # conversion from the VALU-issue-bound loop (host pre-casts the small K tensor).
    USE_K16 = mode == "fused_dq_delta"
    LDS_K16_BASE = 2 * LDS_TILE
    LDS_TOTAL = (3 if USE_K16 else 2) * LDS_TILE

    VEC_WIDTH = 16
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"flash_attn_bwd_smem_{mode}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    IS_DQ = mode == "dq"
    IS_DELTA = mode == "delta"
    IS_FUSED = mode == "fused_dq_delta"

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_bwd_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        K16: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)

        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type
        MFMA_LANE_K = 8

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)

        _f16 = fx.Float16

        def mfma_f16(a, b, c):
            # Single-fp16 MFMA (K=16, same throughput/accumulator layout as the bf16
            # 32x32x16). Used by the fused dQ+delta A/B GEMM2 where fp16's 10-bit
            # mantissa (= tf32) is enough for the near-diagonal dS cancellation.
            return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)

        seq_len_v = fx.Index(seq_len)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        # ds_read_tr16_b64 lane decomposition (4x4 transpose within 16-lane blocks).
        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_off
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # Same 16-bit transpose read, typed as real fp16, for the fused dual-tile
        # fp16 K copy (bit-level reinterpret: the DMA wrote fp16 bit patterns there).
        v4realf16_type = Vec.make_type(4, _f16)

        def ds_read_tr_realf16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_off
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4realf16_type, ptr).result

        wave_q_offset = wave_id * ROWS_PER_WAVE

        # KV-head-major block_id decode (XCD/L2 locality; bijection -> det-neutral).
        if const_expr(GQA_GROUP_SIZE == 1):
            q_head_idx = block_id % NUM_HEADS_Q
            batch_q_tile_id = block_id // NUM_HEADS_Q
            kv_head_idx = q_head_idx
        else:
            kv_head_idx = block_id % NUM_HEADS_KV
            _bid_rest = block_id // NUM_HEADS_KV
            _q_in_group = _bid_rest % GQA_GROUP_SIZE
            batch_q_tile_id = _bid_rest // GQA_GROUP_SIZE
            q_head_idx = kv_head_idx * GQA_GROUP_SIZE + _q_in_group
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        _qt_disp = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        # Causal load-balance two-pointer interleave (mirrors the forward kernel):
        # a q-tile's kv-loop length grows with q_tile_idx (tile 0 -> 1 kv-block,
        # tile N-1 -> N), so natural dispatch runs only the heaviest tiles at the
        # tail (low occupancy). Reorder dispatch to (0, N-1, 1, N-2, ...) so
        # concurrent work-groups mix light+heavy loads. Bijection over q-tiles ->
        # each output tile still computed by exactly one WG (corr/det-neutral);
        # kv_head stays the fastest block_id axis so the XCD/L2 remap is untouched.
        _qt_half = _qt_disp // fx.Index(2)
        _qt_is_odd = ArithValue(_qt_disp % fx.Index(2) == fx.Index(1))
        q_tile_idx = fx.Index(_qt_is_odd.select(num_q_tiles - fx.Index(1) - _qt_half, _qt_half))
        q_start = q_tile_idx * BLOCK_M

        # Fold the per-batch element offset into the raw KV pointers (0-based rows).
        _kv_ptr_batch_off = batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_KV)
        k_ptr = buffer_ops.get_element_ptr(k_ptr, _kv_ptr_batch_off, elem_type=elem_type)
        v_ptr = buffer_ops.get_element_ptr(v_ptr, _kv_ptr_batch_off, elem_type=elem_type)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx_q(token_idx, col):
            return token_idx * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _kv_row_clamp(row_idx):
            last = seq_len_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_v).select(row_idx, last))

        def _load_global_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def bf16_trunc_pack_v8(f32_vals):
            # Hardware f32->bf16 pack (RNE, 1 VALU op/pair) instead of the manual
            # &/>>/| truncation (3 VALU ops/pair); cuts the VALU-issue-bound path.
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        def _to_f16_v8(f32_vals):
            # Single-fp16 pack of one 8-slot B-operand group (no bot half). fp16 keeps
            # a 10-bit mantissa (= tf32), enough to preserve the near-diagonal dS
            # cancellation that single-bf16 (8-bit) breaks, at half the double-bf16
            # pack+MFMA cost.
            return Vec.from_elements(
                [fx.Float32(_raw(f32_vals[i])).to(_f16) for i in range_constexpr(8)], _f16
            ).ir_value()

        def _k_to_f16(kv8):
            # Convert the transpose-read bf16 K sub-tile to fp16 for the fp16 MFMA.
            # Must round-trip through f32 (a direct bf16->f16 arith.truncf is not
            # lowered on this toolchain).
            return Vec(kv8).to(fx.Float32).to(_f16).ir_value()

        # ---- K XOR swizzle (d64-general): col ^ ((row & (K_STRIDE//16-1)) << 4). ----
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)
            return col_idx ^ mask

        def _coop_load(src_ptr, base, tile_start):
            """Cooperative row-major XOR-swizzled load of a BLOCK_N x head_dim tile."""
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                lds_row = load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    if load_row_in_batch < fx.Index(BLOCK_N):
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        vec = _load_global_vec(src_ptr, g_idx, VEC_WIDTH)
                        Vec(vec).store(lds, [base + lds_row * K_STRIDE + swz_col])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    vec = _load_global_vec(src_ptr, g_idx, VEC_WIDTH)
                    Vec(vec).store(lds, [base + lds_row * K_STRIDE + swz_col])

        # ---- Per-batch buffer descriptors (batch base folded into SRD base). ----
        _q_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _q_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        _lse_per_batch = seq_len_v * fx.Index(NUM_HEADS_Q)
        _lse_nrec_bytes = _raw(_lse_per_batch * fx.Index(4))
        _lse_batch_byte_off = _raw(batch_idx * _lse_per_batch * fx.Index(4))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )
        if const_expr(IS_DQ):
            delta_in_rsrc = buffer_ops.create_buffer_resource(
                DELTA, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
            )
        if const_expr(IS_DQ or IS_FUSED):
            dq_rsrc = buffer_ops.create_buffer_resource(
                DQ, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
            )
        if const_expr(IS_DELTA or IS_FUSED):
            delta_out_rsrc = buffer_ops.create_buffer_resource(
                DELTA, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
            )

        # ---- DMA-to-LDS for the K/V tiles (buffer_load_dwordx4 ... lds). ----
        # K_STRIDE == head_dim, so the swizzled LDS layout matches the forward's K
        # DMA path verbatim (LDS[row][c] = Global[row][c ^ ((row&3)<<4)]); serves
        # both the normal read (_a_idx) and the transpose read (_read_k_tr for dQ).
        if const_expr(ENABLE_DMA):
            _kv_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
            _kv_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
            k_rsrc = buffer_ops.create_buffer_resource(
                K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            v_rsrc = buffer_ops.create_buffer_resource(
                V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            if const_expr(USE_K16):
                # fp16 K copy (same [B,S,Hkv,D] layout, 2 bytes/elem -> identical
                # byte math and swizzle as the bf16 K DMA).
                k16_rsrc = buffer_ops.create_buffer_resource(
                    K16, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
                )
            lds_base_idx = buffer_ops.extract_base_index(lds, address_space=3)
            DMA_BYTES = 16
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            KV_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_KV = KV_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_KV_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_tile(src_rsrc, lds_byte_base, tile_start):
                """DMA a BLOCK_N x head_dim K/V tile into the swizzled LDS layout."""
                for d in range_constexpr(NUM_DMA_KV):
                    lds_addr = (
                        lds_byte_base
                        + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_addr))
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
                    row_in_tile = tid // LANES_PER_KV_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                    swiz_col_f16 = (tid % LANES_PER_KV_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                        + kv_head_idx * fx.Index(HEAD_DIM * 2)
                        + col_byte
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        src_rsrc, lds_ptr, _dma_size, fx.Int32(global_byte), _dma_soff, _dma_off, _dma_aux
                    )

        # ---- Preload Q and dO B-operand packs (register-resident). ----
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_b_packs = []
        do_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            q_b_packs.append(
                buffer_ops.buffer_load(
                    q_rsrc, global_idx_q(q_row, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
            )
            do_b_packs.append(
                buffer_ops.buffer_load(
                    do_rsrc, global_idx_q(q_row, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
            )

        # ---- Load LSE (and delta for dq) for this lane's q_row. ----
        _lse_elem = q_head_idx * seq_len_v + q_row
        lse_val = fx.Float32(buffer_ops.buffer_load(lse_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32))
        if const_expr(IS_DQ):
            delta_val = fx.Float32(
                buffer_ops.buffer_load(delta_in_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32)
            )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        # LSE arrives pre-scaled by -log2e from the host, so it is the exp2 addend
        # directly: P = exp2(s*sm*log2e + lse). Saves the (-log2e)*lse mul.
        neg_lse_log2e = lse_val
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound (causal). ----
        _q_end = q_start + BLOCK_M
        kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))

        k_swz_mask = (lane_mod_32 & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)

        def _a_idx_lo(a_base, ks):
            col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return a_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

        def _a_idx_hi(a_base, ks):
            col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return a_base + fx.Index(K_SUB_N * K_STRIDE) + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

        def _gemm_kq(a_base, b_packs):
            """GEMM1 template: acc[M=rows, N=q] = A[rows,D] @ B[q,D]^T over D."""
            acc_lo = c_zero_v16f32
            acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                a_lo = Vec.load(mfma_pack_type, lds, [_a_idx_lo(a_base, ks)])
                a_hi = Vec.load(mfma_pack_type, lds, [_a_idx_hi(a_base, ks)])
                acc_lo = mfma_acc(a_lo, b_packs[ks], acc_lo)
                acc_hi = mfma_acc(a_hi, b_packs[ks], acc_hi)
            return acc_lo, acc_hi

        _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
        TOTAL_PV = len(_steps)

        def _read_k_tr(step_idx):
            """Transpose-read K from LDS -> A-operand [M=D, ctr=kv] (like fwd V)."""
            dc, pks = _steps[step_idx]
            d_col = fx.Index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
            k_row = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
            d_col_eff = _k_swizzle(k_row, d_col)
            lds_lo = fx.Index(0) + k_row * K_STRIDE + d_col_eff
            lds_hi = lds_lo + fx.Index(K_SUB_N * K_STRIDE)
            vl_a = ds_read_tr_v4f16(lds_lo)
            vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * K_STRIDE))
            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            vh_a = ds_read_tr_v4f16(lds_hi)
            vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * K_STRIDE))
            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            return vl, vh

        def _read_k16_tr(step_idx):
            """Transpose-read the fp16 K copy (LDS_K16_BASE) -> A-operand [M=D, ctr=kv]
            as real fp16, fed to mfma_f16 directly (no bf16->fp16 conversion)."""
            dc, pks = _steps[step_idx]
            d_col = fx.Index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
            k_row = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
            d_col_eff = _k_swizzle(k_row, d_col)
            lds_lo = fx.Index(LDS_K16_BASE) + k_row * K_STRIDE + d_col_eff
            lds_hi = lds_lo + fx.Index(K_SUB_N * K_STRIDE)
            vl_a = ds_read_tr_realf16(lds_lo)
            vl_b = ds_read_tr_realf16(lds_lo + fx.Index(8 * K_STRIDE))
            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            vh_a = ds_read_tr_realf16(lds_hi)
            vh_b = ds_read_tr_realf16(lds_hi + fx.Index(8 * K_STRIDE))
            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            return vl, vh

        # ---- Loop-carried init ----
        if const_expr(IS_DQ):
            init_args = [c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]
        elif const_expr(IS_FUSED):
            # [A_accs(D_CHUNKS), B_accs(D_CHUNKS), delta] carried in one kv pass.
            init_args = [c_zero_v16f32 for _ in range_constexpr(2 * D_CHUNKS)] + [c_zero_f]
        else:
            init_args = [c_zero_f]

        def _kv_body(kv_start, inner_iter_args, apply_mask):
            if const_expr(IS_DQ):
                dq_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]
            elif const_expr(IS_FUSED):
                a_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]
                b_accs = [inner_iter_args[D_CHUNKS + i] for i in range_constexpr(D_CHUNKS)]
                delta_acc = inner_iter_args[2 * D_CHUNKS]
            else:
                # A single loop-carried value can arrive unwrapped (not a list).
                delta_acc = (
                    inner_iter_args[0] if isinstance(inner_iter_args, (list, tuple)) else inner_iter_args
                )

            # WAR guard: the single K/V LDS region is overwritten each iteration
            # (no double buffer), so wait for the previous iteration's LDS reads.
            gpu.barrier()
            if const_expr(ENABLE_DMA):
                coop_dma_tile(k_rsrc, lds_base_idx, kv_start)
                coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), kv_start)
                if const_expr(USE_K16):
                    coop_dma_tile(k16_rsrc, lds_base_idx + fx.Index(LDS_K16_BASE * 2), kv_start)
                rocdl.s_waitcnt(0)
            else:
                _coop_load(k_ptr, fx.Index(0), kv_start)
                _coop_load(v_ptr, fx.Index(LDS_V_BASE), kv_start)
            gpu.barrier()

            # GEMM1: S[kv,q] = K @ Q^T
            s_lo_acc, s_hi_acc = _gemm_kq(fx.Index(0), q_b_packs)
            # dP[kv,q] = V @ dO^T (same template, V as "K", dO as "Q")
            dp_lo_acc, dp_hi_acc = _gemm_kq(fx.Index(LDS_V_BASE), do_b_packs)

            s_lo = [Vec(s_lo_acc)[r] for r in range_constexpr(16)]
            s_hi = [Vec(s_hi_acc)[r] for r in range_constexpr(16)]

            # Causal mask: only diagonal tiles (kv_start >= q_start = min q_row of
            # the block) can have kv_col > q_row; below-diagonal tiles are provably
            # unmasked, so the caller skips the compare+select there (apply_mask=False).
            kv_start_i32 = fx.Int32(kv_start)
            lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)

            def _p_exp2(r):
                # P[r] = exp2(sm*log2e*S[r] + lse) with the causal mask on diagonal
                # tiles only. Returns (p_lo_r, p_hi_r) for the two 32-kv sub-blocks.
                if const_expr(apply_mask):
                    off = (r // 4) * 8 + (r % 4)
                    kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(off)
                    s_lo_r = ArithValue(kv_col > q_row_i32).select(c_neg_inf, s_lo[r])
                    s_hi_r = ArithValue(kv_col + fx.Int32(K_SUB_N) > q_row_i32).select(c_neg_inf, s_hi[r])
                else:
                    s_lo_r = s_lo[r]
                    s_hi_r = s_hi[r]
                diff_lo = fmath.fma(s_lo_r, c_sm_scale_log2e, neg_lse_log2e, fastmath=fm_fast)
                diff_hi = fmath.fma(s_hi_r, c_sm_scale_log2e, neg_lse_log2e, fastmath=fm_fast)
                return (
                    ArithValue(diff_lo).exp2(fastmath=fm_fast),
                    ArithValue(diff_hi).exp2(fastmath=fm_fast),
                )

            if const_expr(not IS_FUSED):
                p_lo = []
                p_hi = []
                for r in range_constexpr(16):
                    plo_r, phi_r = _p_exp2(r)
                    p_lo.append(plo_r)
                    p_hi.append(phi_r)
                dp_lo = [Vec(dp_lo_acc)[r] for r in range_constexpr(16)]
                dp_hi = [Vec(dp_hi_acc)[r] for r in range_constexpr(16)]

            # Build the loop-carried yield args conditionally, then yield ONCE at the
            # tail (a single scf.yield per loop body, mirroring the forward).
            if const_expr(IS_DELTA):
                local = c_zero_f
                for r in range_constexpr(16):
                    local = _fadd(local, _fmul(p_lo[r], dp_lo[r]))
                    local = _fadd(local, _fmul(p_hi[r], dp_hi[r]))
                delta_acc = _fadd(delta_acc, local)
                return [delta_acc]
            elif const_expr(IS_FUSED):
                # One pass accumulates delta=sum_j P*dP, A=sum_j (P*dP)*K, B=sum_j P*K
                # (dQ = sm*(A - delta*B)). To keep VGPR pressure low (dual A/B
                # accumulators already cost 2x dq's), process one 8-slot group at a
                # time: exp2 -> C=P*dP -> fp16 pack -> GEMM2 immediately, so only the
                # current group's packs stay live (not all 16 at once).
                local = c_zero_f
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * 8
                    plo_g = []
                    phi_g = []
                    clo_g = []
                    chi_g = []
                    for i in range_constexpr(8):
                        r = base + i
                        plo, phi = _p_exp2(r)
                        clo = _fmul(plo, Vec(dp_lo_acc)[r])
                        chi = _fmul(phi, Vec(dp_hi_acc)[r])
                        local = _fadd(local, clo)
                        local = _fadd(local, chi)
                        plo_g.append(plo)
                        phi_g.append(phi)
                        clo_g.append(clo)
                        chi_g.append(chi)
                    plo_p = _to_f16_v8(plo_g)
                    phi_p = _to_f16_v8(phi_g)
                    clo_p = _to_f16_v8(clo_g)
                    chi_p = _to_f16_v8(chi_g)
                    # GEMM2: A[D,q] += K^T @ C ; B[D,q] += K^T @ P (K transpose-read
                    # as "V"; C/P single-fp16 packs as "P"). K read fp16 (dual-tile,
                    # no conversion). One MFMA per sub-block (no double-bf16 bot).
                    for dc in range_constexpr(D_CHUNKS):
                        if const_expr(ENABLE_DMA):
                            # Dual-tile: read the fp16 K copy directly (no conversion).
                            k_lo, k_hi = _read_k16_tr(dc * PV_K_STEPS + pks)
                        else:
                            k_lo, k_hi = _read_k_tr(dc * PV_K_STEPS + pks)
                            k_lo = _k_to_f16(k_lo)
                            k_hi = _k_to_f16(k_hi)
                        a_accs[dc] = mfma_f16(k_lo, clo_p, a_accs[dc])
                        a_accs[dc] = mfma_f16(k_hi, chi_p, a_accs[dc])
                        b_accs[dc] = mfma_f16(k_lo, plo_p, b_accs[dc])
                        b_accs[dc] = mfma_f16(k_hi, phi_p, b_accs[dc])
                delta_acc = _fadd(delta_acc, local)
                return (
                    [a_accs[i] for i in range_constexpr(D_CHUNKS)]
                    + [b_accs[i] for i in range_constexpr(D_CHUNKS)]
                    + [delta_acc]
                )
            else:
                ds_lo = []
                ds_hi = []
                for r in range_constexpr(16):
                    ds_lo.append(_fmul(p_lo[r], _fsub(dp_lo[r], delta_val)))
                    ds_hi.append(_fmul(p_hi[r], _fsub(dp_hi[r], delta_val)))
                ds_packs_lo = []
                ds_packs_hi = []
                for pks in range_constexpr(PV_K_STEPS):
                    b = pks * 8
                    ds_packs_lo.append(bf16_trunc_pack_v8(ds_lo[b : b + 8]))
                    ds_packs_hi.append(bf16_trunc_pack_v8(ds_hi[b : b + 8]))

                # GEMM2: dQ[D,q] += K^T @ dS  (K transpose-read as "V", dS as "P")
                k_lo_cur, k_hi_cur = _read_k_tr(0)
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < TOTAL_PV):
                        k_lo_nxt, k_hi_nxt = _read_k_tr(si + 1)
                    dq_accs[dc] = mfma_acc(k_lo_cur, ds_packs_lo[pks], dq_accs[dc])
                    dq_accs[dc] = mfma_acc(k_hi_cur, ds_packs_hi[pks], dq_accs[dc])
                    if const_expr(si + 1 < TOTAL_PV):
                        k_lo_cur = k_lo_nxt
                        k_hi_cur = k_hi_nxt
                return [dq_accs[i] for i in range_constexpr(D_CHUNKS)]

        # Split the causal kv-loop: [0, q_start) is fully below the diagonal (no
        # mask), [q_start, kv_upper) straddles it (mask). This drops the per-tile
        # compare+select from every below-diagonal tile (the large majority).
        loop_results = init_args
        for kv_start, inner_iter_args in range(0, q_start, BLOCK_N, init=init_args):
            loop_results = yield _kv_body(kv_start, inner_iter_args, False)
        # A single loop-carried value (delta mode) is yielded back unwrapped; the
        # next loop's init= needs a list.
        _tail_init = loop_results if isinstance(loop_results, (list, tuple)) else [loop_results]
        for kv_start, inner_iter_args in range(q_start, kv_upper, BLOCK_N, init=_tail_init):
            loop_results = yield _kv_body(kv_start, inner_iter_args, True)

        # ---- Epilogue ----
        if const_expr(IS_DELTA):
            delta_final = loop_results[0] if isinstance(loop_results, (list, tuple)) else loop_results
            delta_full = _fadd(delta_final, reduction_peer(delta_final))
            if lane_div_32 == fx.Index(0):
                buffer_ops.buffer_store(
                    fx.Float32(delta_full),
                    delta_out_rsrc,
                    _lse_elem * fx.Index(4),
                    mask=ArithValue(q_row < seq_len_v),
                    offset_is_bytes=True,
                )
        else:
            if const_expr(IS_FUSED):
                # delta = sum_j P*dP (peer-reduce over the kv split held by lane+-32);
                # both lane halves then hold the full delta for their q=lane_mod_32.
                a_finals = [loop_results[dc] for dc in range_constexpr(D_CHUNKS)]
                b_finals = [loop_results[D_CHUNKS + dc] for dc in range_constexpr(D_CHUNKS)]
                delta_final = loop_results[2 * D_CHUNKS]
                delta_full = _fadd(delta_final, reduction_peer(delta_final))
                if lane_div_32 == fx.Index(0):
                    buffer_ops.buffer_store(
                        fx.Float32(delta_full),
                        delta_out_rsrc,
                        _lse_elem * fx.Index(4),
                        mask=ArithValue(q_row < seq_len_v),
                        offset_is_bytes=True,
                    )
                # dQ = sm*(A - delta*B). The accumulators are [M=D, N=q] with N=q ==
                # lane_mod_32, so delta_full (per-q, per-lane) is constant across the 16
                # D-registers -> a single scalar multiply per register.
                dq_finals = []
                for dc in range_constexpr(D_CHUNKS):
                    a_v = Vec(a_finals[dc])
                    b_v = Vec(b_finals[dc])
                    vals = [
                        fx.Float32(_fsub(a_v[r], _fmul(delta_full, b_v[r])))
                        for r in range_constexpr(16)
                    ]
                    dq_finals.append(Vec.from_elements(vals, fx.Float32).ir_value())
            else:
                dq_finals = [loop_results[dc] for dc in range_constexpr(D_CHUNKS)]
            sm_vec = Vec.from_elements([fx.Float32(sm_scale)], fx.Float32).broadcast_to(16)
            v_o = [Vec(dq_finals[dc]) * sm_vec for dc in range_constexpr(D_CHUNKS)]

            pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            is_hi_half = ArithValue(lane_div_32 != fx.Index(0))

            def _o_pack_2dw(dc, store_group):
                r_base = store_group * 4
                lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
                hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
                return lo, hi

            def _swap_halves(dw):
                swapped = rocdl.permlane32_swap(pair_i32_ty, _raw(dw), _raw(dw), False, False)
                lo_res = llvm.extractvalue(T.i32, swapped, [0])
                hi_res = llvm.extractvalue(T.i32, swapped, [1])
                return is_hi_half.select(lo_res, hi_res)

            for dc in range_constexpr(D_CHUNKS):
                for g in range_constexpr(2):
                    d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                    d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                    y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                    y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                    w0 = is_hi_half.select(y0_b, _raw(d0_a))
                    w1 = is_hi_half.select(y1_b, _raw(d1_a))
                    w2 = is_hi_half.select(_raw(d0_b), y0_a)
                    w3 = is_hi_half.select(_raw(d1_b), y1_a)
                    o_pack = Vec.from_elements(
                        [fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)], fx.Int32
                    )
                    d_col = fx.Index(dc * D_CHUNK) + (fx.Index(2 * g) + lane_div_32) * fx.Index(8)
                    o_global = global_idx_q(q_row, d_col)
                    buffer_ops.buffer_store(o_pack, dq_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

    @flyc.jit
    def launch_flash_attn_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        K16: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS_Q

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_bwd_kernel(
            Q,
            K,
            V,
            DO,
            LSE,
            DELTA,
            DQ,
            K16,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{int(flat_work_group_size)},{int(flat_work_group_size)}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        # enable-post-misched=True: the split backward is VALU/exp2-issue-bound with
        # the MFMA pipeline mostly idle, so the post-RA machine scheduler interleaves
        # the gradient-GEMM MFMAs into the exp2/reduce VALU shadow. Reorder of
        # independent ops only -> bit-identical output (corr/det unchanged).
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_flash_attn_bwd(*args, **kwargs)

    def _compile(*args):
        with CompilationContext.compile_hints(_hints):
            return flyc.compile(launch_flash_attn_bwd, *args)

    _launch.compile = _compile
    return _launch


def build_flash_attn_bwd_dkdv_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    block_kv=128,
    num_kv_heads=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    q_split=2,
    enable_dma=True,
    fast_exp2=False,
):
    """Build the dK/dV KV-outer backward launcher (clean mirror of the forward).

    One work-group owns BLOCK_KV key/value rows of one kv-head and loops over the
    GQA group's q-heads and (causal) q-blocks, accumulating dK/dV in registers ->
    single write, no float atomics -> deterministic. Roles vs the forward are
    swapped q<->kv:

    Deterministic causal split-K over the q-loop (``q_split``): block_id carries a
    split_idx (kv_head stays the fastest-varying axis so the forward XCD/L2
    mapping is preserved). Each split owns a cyclic subset of the causal q-blocks
    (q_start = kv_start + split_idx*BLOCK_Q, step = q_split*BLOCK_Q) and writes its
    own slot of a [B, q_split, S, Hkv, D] workspace exactly once (no float
    atomics); the host reduces slot-wise with a fixed-order fp32 sum. This lifts
    the grid from B*Hkv*(S/BLOCK_KV) to that times q_split, which raises the grid
    -wave count and hides latency at the cost of redundant work; callers tune
    q_split per shape. q_split=1 degenerates to the single-owner path.
      * K,V owned as MFMA B-operands (register-resident, like the forward's Q).
      * Q,dO streamed to LDS (like the forward's K,V), read normally for the
        S/dP GEMMs and transpose-read (ds_read_tr) for the dV/dK GEMMs.
      * GEMM1a S[q,kv]=Q@K^T, GEMM1b dP[q,kv]=dO@V^T.
      * GEMM2a dV^T[D,kv] += dO_tr @ P, GEMM2b dK^T[D,kv] += Q_tr @ dS, where the
        P/dS accumulators feed directly as B-operands (K@Q^T PV-alignment), so no
        explicit accumulator transpose is needed. For head_dim=64 the K-swizzle
        equals the V-swizzle, so one &3-swizzled LDS tile serves both the normal
        and the transpose read. Output [D,kv] is stored transposed to [kv,D].
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "bwd dkdv kernel targets gfx950"
    assert dtype_str == "bf16", "bwd dkdv kernel targets bf16"
    assert causal, "bwd dkdv kernel is causal-only for the GPT-OSS campaign"

    # buffer_load_dwordx4 ... lds (16B DMA-to-LDS) needs gfx950+ (gfx94x has only
    # the 4B dword variant). DMA bypasses the VGPR staging of the Q/dO tile loads,
    # relieving register pressure on this VGPR-locked (236 VGPR, occ ~2) kernel.
    ENABLE_DMA = enable_dma and not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    BLOCK_Q = 64
    Q_SUB = 32
    WARP_SIZE = 64
    BLOCK_KV = block_kv
    Q_SPLIT = q_split
    assert q_split >= 1
    flat_work_group_size = 256
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_KV = BLOCK_KV // NUM_WAVES

    K_STEP_QK = 16
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16
    PV_K_STEPS = Q_SUB // PV_K_STEP

    assert BLOCK_KV % NUM_WAVES == 0
    assert head_dim % 32 == 0 and head_dim >= 64

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    Q_STRIDE = HEAD_DIM
    LDS_TILE = BLOCK_Q * Q_STRIDE
    LDS_DO_BASE = LDS_TILE
    LDS_TOTAL = 2 * LDS_TILE

    VEC_WIDTH = 16
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    assert ROWS_PER_BATCH_LOAD >= BLOCK_Q and ROWS_PER_BATCH_LOAD % BLOCK_Q == 0
    NUM_BATCHES_Q = 1
    Q_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_Q

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="flash_attn_bwd_smem_dkdv")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_bwd_dkdv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        q_ptr = _extract_aligned_pointer(Q)
        do_ptr = _extract_aligned_pointer(DO)

        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type
        MFMA_LANE_K = 8

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)

        seq_len_v = fx.Index(seq_len)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32
        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_off
            ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Decompose block_id: kv_head fastest (XCD/L2), then split_idx. ----
        num_kv_tiles = (seq_len_v + BLOCK_KV - 1) // BLOCK_KV
        kv_head_idx = block_id % NUM_HEADS_KV
        _rest = block_id // NUM_HEADS_KV
        if const_expr(Q_SPLIT > 1):
            split_idx = _rest % fx.Index(Q_SPLIT)
            _rest = _rest // fx.Index(Q_SPLIT)
        else:
            split_idx = fx.Index(0)
        kv_tile_idx = _rest % num_kv_tiles
        batch_idx = _rest // num_kv_tiles
        kv_start = kv_tile_idx * BLOCK_KV
        kv_row = kv_start + wave_id * ROWS_PER_WAVE_KV + lane_mod_32
        kv_row_i32 = fx.Int32(kv_row)

        # Fold per-batch element offset into raw Q/dO pointers (0-based rows).
        _q_ptr_batch_off = batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_Q)
        q_ptr = buffer_ops.get_element_ptr(q_ptr, _q_ptr_batch_off, elem_type=elem_type)
        do_ptr = buffer_ops.get_element_ptr(do_ptr, _q_ptr_batch_off, elem_type=elem_type)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx_q(token_idx, col, q_head):
            return token_idx * STRIDE_TOKEN_Q + q_head * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _q_row_clamp(row_idx):
            last = seq_len_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_v).select(row_idx, last))

        def _load_global_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def bf16_trunc_pack_v8(f32_vals):
            # Hardware f32->bf16 pack (RNE, 1 VALU op/pair) instead of the manual
            # &/>>/| truncation (3 VALU ops/pair); cuts the VALU-issue-bound path.
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        def _swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(Q_STRIDE // 16 - 1)) << fx.Index(4)
            return col_idx ^ mask

        def _coop_load(src_ptr, base, tile_start, q_head):
            """Cooperative row-major XOR-swizzled load of a BLOCK_Q x head_dim tile."""
            for batch in range_constexpr(NUM_BATCHES_Q):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _q_row_clamp(tile_start + load_row_in_batch + row_offset)
                lds_row = load_row_in_batch + row_offset
                if const_expr(Q_NEEDS_GUARD):
                    if load_row_in_batch < fx.Index(BLOCK_Q):
                        g_idx = global_idx_q(row_idx, load_col_base, q_head)
                        swz_col = _swizzle(lds_row, load_col_base)
                        vec = _load_global_vec(src_ptr, g_idx, VEC_WIDTH)
                        Vec(vec).store(lds, [base + lds_row * Q_STRIDE + swz_col])
                else:
                    g_idx = global_idx_q(row_idx, load_col_base, q_head)
                    swz_col = _swizzle(lds_row, load_col_base)
                    vec = _load_global_vec(src_ptr, g_idx, VEC_WIDTH)
                    Vec(vec).store(lds, [base + lds_row * Q_STRIDE + swz_col])

        # ---- Per-batch descriptors (batch base folded into SRD base). ----
        _q_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _q_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _kv_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        _kv_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
        )
        # DK/DV point at this split's slot of the [B, q_split, S, Hkv, D] workspace
        # (slot index = batch*q_split + split_idx); one WG writes it exactly once.
        _ws_slot = batch_idx * fx.Index(Q_SPLIT) + split_idx
        _dkv_ws_byte_off = _raw(_ws_slot * seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        dk_rsrc = buffer_ops.create_buffer_resource(
            DK, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_dkv_ws_byte_off
        )
        dv_rsrc = buffer_ops.create_buffer_resource(
            DV, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_dkv_ws_byte_off
        )
        _lse_per_batch = seq_len_v * fx.Index(NUM_HEADS_Q)
        _lse_nrec_bytes = _raw(_lse_per_batch * fx.Index(4))
        _lse_batch_byte_off = _raw(batch_idx * _lse_per_batch * fx.Index(4))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )

        # ---- DMA-to-LDS for the Q/dO tiles (buffer_load_dwordx4 ... lds). ----
        # Q_STRIDE == head_dim, so the swizzled LDS layout matches the forward's K
        # DMA path verbatim (LDS[row][c] = Global[row][c ^ ((row&3)<<4)]); both the
        # normal read (_a_idx) and the transpose read (_read_tr) expect that layout.
        if const_expr(ENABLE_DMA):
            q_rsrc = buffer_ops.create_buffer_resource(
                Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
            )
            do_rsrc = buffer_ops.create_buffer_resource(
                DO, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
            )
            lds_base_idx = buffer_ops.extract_base_index(lds, address_space=3)
            DMA_BYTES = 16
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            Q_TILE_BYTES = BLOCK_Q * Q_STRIDE * 2
            NUM_DMA_Q = Q_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_Q_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_tile(src_rsrc, lds_byte_base, tile_start, q_head):
                """DMA a BLOCK_Q x head_dim Q/dO tile into the swizzled LDS layout."""
                for d in range_constexpr(NUM_DMA_Q):
                    lds_addr = (
                        lds_byte_base
                        + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_addr))
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
                    row_in_tile = tid // LANES_PER_Q_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                    swiz_col_f16 = (tid % LANES_PER_Q_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(Q_STRIDE // 16 - 1)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_Q * 2) + q_head * fx.Index(HEAD_DIM * 2) + col_byte
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        src_rsrc, lds_ptr, _dma_size, fx.Int32(global_byte), _dma_soff, _dma_off, _dma_aux
                    )

        # ---- Owned K,V B-operand packs (register-resident; kv fixed per WG). ----
        k_b_packs = []
        v_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            kv_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            k_b_packs.append(
                buffer_ops.buffer_load(
                    k_rsrc, global_idx_kv(kv_row, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
            )
            v_b_packs.append(
                buffer_ops.buffer_load(
                    v_rsrc, global_idx_kv(kv_row, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
            )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

        # Crude Schraudolph 2^x (fast_exp2): reinterpret the integer
        # x*2^23 + (127*2^23 + magic) as f32 -> piecewise-linear 2^x, replacing the
        # quarter-rate v_exp with 3 full-rate ops (max+fma+fptosi+bitcast). The
        # maximumf clamp bounds the int convert AND is the all-mask guard (masked
        # -inf -> 2^-88=0, no exp2(-inf)=NaN; pitfalls/04). dK/dV tolerate the
        # ~1-3% P error (no near-diagonal cancellation), unlike the dq/LSE path.
        _c_exp2_floor = fx.Float32(-87.0)
        _c_exp2_scale = fx.Float32(float(1 << 23))
        _c_exp2_bias = fx.Float32(float(127 * (1 << 23) - 486411))
        _compute_type = fx.Float32.ir_type

        def _exp2_of(diff):
            if const_expr(fast_exp2):
                xc = ArithValue(diff).maximumf(_c_exp2_floor)
                scaled = fmath.fma(xc, _c_exp2_scale, _c_exp2_bias, fastmath=fm_fast)
                i = arith.fptosi(fx.Int32.ir_type, _raw(scaled))
                return ArithValue(i).bitcast(_compute_type)
            return ArithValue(diff).exp2(fastmath=fm_fast)

        # A-operand read (Q/dO from LDS, M=q=lane_mod_32) for the S/dP GEMMs.
        a_swz_mask = (lane_mod_32 & fx.Index(Q_STRIDE // 16 - 1)) << fx.Index(4)

        def _a_idx_lo(a_base, ks):
            col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return a_base + lane_mod_32 * Q_STRIDE + (col ^ a_swz_mask)

        def _a_idx_hi(a_base, ks):
            col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return a_base + fx.Index(Q_SUB * Q_STRIDE) + lane_mod_32 * Q_STRIDE + (col ^ a_swz_mask)

        def _gemm_qk(a_base, b_packs):
            """S[q,kv]=A(streamed,M=q)@B(owned,N=kv)^T over D."""
            acc_lo = c_zero_v16f32
            acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                a_lo = Vec.load(mfma_pack_type, lds, [_a_idx_lo(a_base, ks)])
                a_hi = Vec.load(mfma_pack_type, lds, [_a_idx_hi(a_base, ks)])
                acc_lo = mfma_acc(a_lo, b_packs[ks], acc_lo)
                acc_hi = mfma_acc(a_hi, b_packs[ks], acc_hi)
            return acc_lo, acc_hi

        _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
        TOTAL_PV = len(_steps)

        def _read_tr(a_base, step_idx):
            """Transpose-read Q/dO from LDS -> A-operand [M=D, ctr=q] (like fwd V)."""
            dc, pks = _steps[step_idx]
            d_col = fx.Index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
            q_row_tr = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
            d_col_eff = _swizzle(q_row_tr, d_col)
            lds_lo = a_base + q_row_tr * Q_STRIDE + d_col_eff
            lds_hi = lds_lo + fx.Index(Q_SUB * Q_STRIDE)
            vl_a = ds_read_tr_v4f16(lds_lo)
            vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * Q_STRIDE))
            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            vh_a = ds_read_tr_v4f16(lds_hi)
            vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * Q_STRIDE))
            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            return vl, vh

        # ---- q-slot -> q offset within the 64-block (M output layout). ----
        _q_off = [(r // 4) * 8 + (r % 4) for r in range(16)]

        dv_accs = [c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]
        dk_accs = [c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

        # This split owns q-blocks [kv_start + split_idx*BLOCK_Q : seq : q_split*BLOCK_Q].
        _q_loop_start = kv_start + split_idx * fx.Index(BLOCK_Q)
        # Only the diagonal q-block (q_start < kv_end) needs the causal mask; the
        # split step (q_split*BLOCK_Q) > BLOCK_KV so that is at most the first
        # iteration. Run it masked, then the rest mask-free (drops compare+select
        # from every below-diagonal q-block, x GQA_GROUP_SIZE q-heads).
        _kv_end = kv_start + fx.Index(BLOCK_KV)
        _step = Q_SPLIT * BLOCK_Q
        _masked_upper = ArithValue(_kv_end < seq_len_v).select(_kv_end, seq_len_v)
        _unmask_start = ArithValue(_q_loop_start < _kv_end).select(
            _q_loop_start + fx.Index(_step), _q_loop_start
        )
        for q_head_local in range_constexpr(GQA_GROUP_SIZE):
            q_head = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(q_head_local)
            _lse_head_base = q_head * seq_len_v

            # q_head/_lse_head_base are rebound once per q_head_local iteration; _q_body
            # is only ever called within that same iteration (never stored past it), so
            # the B023 late-binding warnings below are false positives.
            def _q_body(q_start, inner, apply_mask):
                dv_cur = [inner[i] for i in range_constexpr(D_CHUNKS)]
                dk_cur = [inner[D_CHUNKS + i] for i in range_constexpr(D_CHUNKS)]

                gpu.barrier()
                if const_expr(ENABLE_DMA):
                    coop_dma_tile(q_rsrc, lds_base_idx, q_start, q_head)  # noqa: B023
                    do_lds_base = lds_base_idx + fx.Index(LDS_DO_BASE * 2)
                    coop_dma_tile(do_rsrc, do_lds_base, q_start, q_head)  # noqa: B023
                    rocdl.s_waitcnt(0)
                else:
                    _coop_load(q_ptr, fx.Index(0), q_start, q_head)  # noqa: B023
                    _coop_load(do_ptr, fx.Index(LDS_DO_BASE), q_start, q_head)  # noqa: B023
                gpu.barrier()

                # GEMM1a S[q,kv]=Q@K^T ; GEMM1b dP[q,kv]=dO@V^T.
                s_lo_acc, s_hi_acc = _gemm_qk(fx.Index(0), k_b_packs)
                dp_lo_acc, dp_hi_acc = _gemm_qk(fx.Index(LDS_DO_BASE), v_b_packs)

                s_lo = [Vec(s_lo_acc)[r] for r in range_constexpr(16)]
                s_hi = [Vec(s_hi_acc)[r] for r in range_constexpr(16)]
                dp_lo = [Vec(dp_lo_acc)[r] for r in range_constexpr(16)]
                dp_hi = [Vec(dp_hi_acc)[r] for r in range_constexpr(16)]

                q_start_i32 = fx.Int32(q_start)
                lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
                # Compute+pack P/dS one 8-slot group at a time, and issue that
                # group's lse/delta vec4 loads inside the same step (16 q-slots per
                # lane = 4 contiguous runs of 4 -> one 16B vec4 load each). Only one
                # group's f32 P/dS/lse/delta stay live at a time (halved peak), which
                # cuts spill on this register-locked kernel. Per-element independent
                # math -> bit-identical to the all-upfront version.
                p_packs_lo = [None] * PV_K_STEPS
                p_packs_hi = [None] * PV_K_STEPS
                ds_packs_lo = [None] * PV_K_STEPS
                ds_packs_hi = [None] * PV_K_STEPS
                for pks in range_constexpr(PV_K_STEPS):
                    lse_lo_g = [None] * 8
                    lse_hi_g = [None] * 8
                    dl_lo_g = [None] * 8
                    dl_hi_g = [None] * 8
                    for gg in range_constexpr(2):
                        g = 2 * pks + gg
                        base_lo = _lse_head_base + fx.Index(  # noqa: B023
                            ArithValue(q_start_i32 + lane_off_i32 + fx.Int32(8 * g))
                        )
                        base_hi = base_lo + fx.Index(Q_SUB)
                        lse_lo_vec = buffer_ops.buffer_load(lse_rsrc, base_lo, vec_width=4, dtype=fx.Float32)
                        lse_hi_vec = buffer_ops.buffer_load(lse_rsrc, base_hi, vec_width=4, dtype=fx.Float32)
                        dl_lo_vec = buffer_ops.buffer_load(delta_rsrc, base_lo, vec_width=4, dtype=fx.Float32)
                        dl_hi_vec = buffer_ops.buffer_load(delta_rsrc, base_hi, vec_width=4, dtype=fx.Float32)
                        for i in range_constexpr(4):
                            lse_lo_g[4 * gg + i] = fx.Float32(Vec(lse_lo_vec)[i])
                            lse_hi_g[4 * gg + i] = fx.Float32(Vec(lse_hi_vec)[i])
                            dl_lo_g[4 * gg + i] = fx.Float32(Vec(dl_lo_vec)[i])
                            dl_hi_g[4 * gg + i] = fx.Float32(Vec(dl_hi_vec)[i])
                    plo_g = []
                    phi_g = []
                    dslo_g = []
                    dshi_g = []
                    for i in range_constexpr(8):
                        r = pks * 8 + i
                        q_slot_lo_i32 = q_start_i32 + lane_off_i32 + fx.Int32(_q_off[r])
                        q_slot_hi_i32 = q_slot_lo_i32 + fx.Int32(Q_SUB)
                        lse_lo = lse_lo_g[i]
                        lse_hi = lse_hi_g[i]
                        dl_lo = dl_lo_g[i]
                        dl_hi = dl_hi_g[i]
                        # causal mask (only the diagonal q-block per split needs it).
                        if const_expr(apply_mask):
                            s_lo_r = ArithValue(kv_row_i32 > q_slot_lo_i32).select(c_neg_inf, s_lo[r])
                            s_hi_r = ArithValue(kv_row_i32 > q_slot_hi_i32).select(c_neg_inf, s_hi[r])
                        else:
                            s_lo_r = s_lo[r]
                            s_hi_r = s_hi[r]
                        # lse_lo/lse_hi arrive pre-scaled by -log2e (host), so they are
                        # the exp2 addend directly -> no (-log2e)*lse mul per q-slot.
                        plo = _exp2_of(fmath.fma(s_lo_r, c_sm_scale_log2e, lse_lo, fastmath=fm_fast))
                        phi = _exp2_of(fmath.fma(s_hi_r, c_sm_scale_log2e, lse_hi, fastmath=fm_fast))
                        plo_g.append(plo)
                        phi_g.append(phi)
                        dslo_g.append(_fmul(plo, _fsub(dp_lo[r], dl_lo)))
                        dshi_g.append(_fmul(phi, _fsub(dp_hi[r], dl_hi)))
                    p_packs_lo[pks] = bf16_trunc_pack_v8(plo_g)
                    p_packs_hi[pks] = bf16_trunc_pack_v8(phi_g)
                    ds_packs_lo[pks] = bf16_trunc_pack_v8(dslo_g)
                    ds_packs_hi[pks] = bf16_trunc_pack_v8(dshi_g)

                # GEMM2a dV^T[D,kv] += dO_tr @ P ; GEMM2b dK^T[D,kv] += Q_tr @ dS.
                do_lo_cur, do_hi_cur = _read_tr(fx.Index(LDS_DO_BASE), 0)
                q_lo_cur, q_hi_cur = _read_tr(fx.Index(0), 0)
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < TOTAL_PV):
                        do_lo_nxt, do_hi_nxt = _read_tr(fx.Index(LDS_DO_BASE), si + 1)
                        q_lo_nxt, q_hi_nxt = _read_tr(fx.Index(0), si + 1)
                    dv_cur[dc] = mfma_acc(do_lo_cur, p_packs_lo[pks], dv_cur[dc])
                    dv_cur[dc] = mfma_acc(do_hi_cur, p_packs_hi[pks], dv_cur[dc])
                    dk_cur[dc] = mfma_acc(q_lo_cur, ds_packs_lo[pks], dk_cur[dc])
                    dk_cur[dc] = mfma_acc(q_hi_cur, ds_packs_hi[pks], dk_cur[dc])
                    if const_expr(si + 1 < TOTAL_PV):
                        do_lo_cur, do_hi_cur = do_lo_nxt, do_hi_nxt
                        q_lo_cur, q_hi_cur = q_lo_nxt, q_hi_nxt

                return dv_cur + dk_cur

            _carry = dv_accs + dk_accs
            loop_results = _carry
            for q_start, inner in range(_q_loop_start, _masked_upper, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
            for q_start, inner in range(_unmask_start, seq_len_v, _step, init=loop_results):
                loop_results = yield _q_body(q_start, inner, False)
            dv_accs = [loop_results[i] for i in range_constexpr(D_CHUNKS)]
            dk_accs = [loop_results[D_CHUNKS + i] for i in range_constexpr(D_CHUNKS)]

        # ---- Store dV[kv,D] and dK[kv,D] (permlane transpose, mirror O-store). ----
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        is_hi_half = ArithValue(lane_div_32 != fx.Index(0))
        sm_vec = Vec.from_elements([fx.Float32(sm_scale)], fx.Float32).broadcast_to(16)
        dk_scaled = [Vec(dk_accs[dc]) * sm_vec for dc in range_constexpr(D_CHUNKS)]

        def _swap_halves(dw):
            swapped = rocdl.permlane32_swap(pair_i32_ty, _raw(dw), _raw(dw), False, False)
            lo_res = llvm.extractvalue(T.i32, swapped, [0])
            hi_res = llvm.extractvalue(T.i32, swapped, [1])
            return is_hi_half.select(lo_res, hi_res)

        def _store(vals, rsrc):
            def _pack_2dw(dc, store_group):
                r_base = store_group * 4
                lo = rocdl.cvt_pk_bf16_f32(Vec(vals[dc])[r_base], Vec(vals[dc])[r_base + 1])
                hi = rocdl.cvt_pk_bf16_f32(Vec(vals[dc])[r_base + 2], Vec(vals[dc])[r_base + 3])
                return lo, hi

            for dc in range_constexpr(D_CHUNKS):
                for g in range_constexpr(2):
                    d0_a, d1_a = _pack_2dw(dc, 2 * g)
                    d0_b, d1_b = _pack_2dw(dc, 2 * g + 1)
                    y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                    y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                    w0 = is_hi_half.select(y0_b, _raw(d0_a))
                    w1 = is_hi_half.select(y1_b, _raw(d1_a))
                    w2 = is_hi_half.select(_raw(d0_b), y0_a)
                    w3 = is_hi_half.select(_raw(d1_b), y1_a)
                    o_pack = Vec.from_elements(
                        [fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)], fx.Int32
                    )
                    d_col = fx.Index(dc * D_CHUNK) + (fx.Index(2 * g) + lane_div_32) * fx.Index(8)
                    g_idx = global_idx_kv(kv_row, d_col)
                    buffer_ops.buffer_store(o_pack, rsrc, g_idx * fx.Index(2), offset_is_bytes=True)

        _store([Vec(dv_accs[dc]) for dc in range_constexpr(D_CHUNKS)], dv_rsrc)
        _store(dk_scaled, dk_rsrc)

    @flyc.jit
    def launch_flash_attn_bwd_dkdv(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_kv_tiles = (sl_idx + BLOCK_KV - 1) // BLOCK_KV
        grid_x = bs_idx * num_kv_tiles * NUM_HEADS_KV * Q_SPLIT

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_bwd_dkdv_kernel(
            Q,
            K,
            V,
            DO,
            LSE,
            DELTA,
            DK,
            DV,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{int(flat_work_group_size)},{int(flat_work_group_size)}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        # enable-post-misched=True: the split backward is VALU/exp2-issue-bound with
        # the MFMA pipeline mostly idle, so the post-RA machine scheduler interleaves
        # the gradient-GEMM MFMAs into the exp2/reduce VALU shadow. Reorder of
        # independent ops only -> bit-identical output (corr/det unchanged).
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_flash_attn_bwd_dkdv(*args, **kwargs)

    def _compile(*args):
        with CompilationContext.compile_hints(_hints):
            return flyc.compile(launch_flash_attn_bwd_dkdv, *args)

    _launch.compile = _compile
    return _launch
