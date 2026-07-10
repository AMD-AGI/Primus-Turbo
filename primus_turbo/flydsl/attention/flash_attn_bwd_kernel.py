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

  mode="fused_dq_delta": folds delta and dq into one kv-loop so S/P/dP are recomputed
      once. dQ = sm/R*(A - (delta/R)*B) with, in one pass, A_i = sum_j C_ij k_j,
      B_i = sum_j P_ij k_j, and a scalar reduce. Two variants:
      * identity_center=True (production): C = P*(dP - delta_id) is CENTERED by a
        precomputed identity delta_id = rowsum(O.dO) read from DELTA, so C is small
        and the A/B GEMM2 uses plain bf16 operands; the scalar becomes the residual
        rho = sum_j P*(dP-delta_id) and the epilogue's (rho/R)*B correction recovers
        the exact consistent dq. No delta is written (delta_id already serves dkdv);
        this eliminates the separate S/dP delta pass entirely.
      * identity_center=False (legacy): C = P*dP is UNCENTERED, so A/B feed single
        fp16 (tf32, 10-bit mantissa) operands to survive the near-diagonal
        catastrophic cancellation (single-bf16's 8-bit fails); writes the recomputed
        fp32 delta for the downstream dkdv kernel.

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
    fast_exp2=False,
    identity_center=False,
):
    """Build one backward launcher. ``mode`` in {"delta", "dq", "fused_dq_delta"}.

    ``identity_center`` (fused mode only): instead of computing delta in-kernel and
    forming the uncentered A = sum_j (P*dP)*k (which needs fp16/tf32 operands to
    survive the near-diagonal cancellation), read a precomputed identity delta
    delta_id = rowsum_d(O*dO) from DELTA and center in-loop: C = P*(dP - delta_id).
    Because dP-delta_id is already small, the A/B GEMM2 uses plain bf16 operands.
    The residual rho = sum_j P*(dP-delta_id) then corrects the epilogue exactly:
    dQ = sm/R * (A - (rho/R)*B), recovering the consistent dq WITHOUT the separate
    delta pass (delta kernel eliminated; DELTA is filled cheaply on the host / by an
    O.dO pass and reused by dkdv). No delta is written back."""
    assert mode in ("delta", "dq", "fused_dq_delta"), mode
    assert not identity_center or mode == "fused_dq_delta", "identity_center is fused-only"
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
    # identity_center fuses with plain bf16 operands (centered A survives bf16), so
    # it needs no separate fp16 K copy; only the legacy uncentered fused path does.
    USE_K16 = mode == "fused_dq_delta" and not identity_center
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

        # ---- LDS bank-conflict swizzles (d64, gfx950 64 banks). A K/V row = 64 elems
        # = 32 dwords, so same-parity rows share a 32-bank half; the 16 same-parity
        # rows must be spread across the 8 aligned 16 B slots of that half to hit the
        # b128 2-way floor. Two masks because the two tiles are read differently:
        #   * K is BOTH normal-read (S=K@Q^T) AND transpose-read (dQ=K^T@dS). The
        #     ds_read_tr16_b64 network only tolerates a 16-elem-granular mask, which on
        #     d64 (4 blocks) cannot spread same-parity rows past 2 offsets -> stuck at
        #     the legacy (row&3)<<4 (4-8 way). Keep legacy for K.
        #   * V is normal-read ONLY (dP=V@dO^T), so it can use the finer 8-elem-granular
        #     ((row//2)%8)*8 mask -> 8 distinct offsets -> 2-way. This halves the GEMM's
        #     normal-read bank conflicts (measured ~60% of LDS-active cycles).
        # Both round-trip (XOR self-inverse; period divides K_SUB_N=32 so lo/hi reads
        # share the lane_mod_32 mask). D128 keeps the legacy mask for both.
        def _k_bank_mask(row_idx):
            return (row_idx & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)

        def _v_bank_mask(row_idx):
            if const_expr(K_STRIDE == 64):
                return ((row_idx >> fx.Index(1)) & fx.Index(7)) << fx.Index(3)
            return (row_idx & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)

        def _k_swizzle(row_idx, col_idx):
            return col_idx ^ _k_bank_mask(row_idx)

        def _v_swizzle(row_idx, col_idx):
            return col_idx ^ _v_bank_mask(row_idx)

        def _coop_load(src_ptr, base, tile_start, swizzle=_k_swizzle):
            """Cooperative row-major XOR-swizzled load of a BLOCK_N x head_dim tile."""
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                lds_row = load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    if load_row_in_batch < fx.Index(BLOCK_N):
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        swz_col = swizzle(lds_row, load_col_base)
                        vec = _load_global_vec(src_ptr, g_idx, VEC_WIDTH)
                        Vec(vec).store(lds, [base + lds_row * K_STRIDE + swz_col])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    swz_col = swizzle(lds_row, load_col_base)
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
        if const_expr(IS_DQ or (IS_FUSED and identity_center)):
            delta_in_rsrc = buffer_ops.create_buffer_resource(
                DELTA, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
            )
        if const_expr(IS_DQ or IS_FUSED):
            dq_rsrc = buffer_ops.create_buffer_resource(
                DQ, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
            )
        if const_expr((IS_DELTA or IS_FUSED) and not (IS_FUSED and identity_center)):
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

            def coop_dma_tile(src_rsrc, lds_byte_base, tile_start, bank_mask=_k_bank_mask):
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
                    xor_mask = bank_mask(row_in_tile)
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
        if const_expr(IS_DQ or (IS_FUSED and identity_center)):
            # DELTA holds -delta_id (negated, matching the dkdv fold convention), so
            # dP - delta_id == dP + delta_val in the loop below.
            delta_val = fx.Float32(
                buffer_ops.buffer_load(delta_in_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32)
            )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        # LSE arrives host-prescaled as lse_s23 = (-log2e*lse)*2^23 + bias, i.e. the
        # Schraudolph exp2 addend already scaled by 2^23, so _exp2_of folds its two
        # fmas into one: scaled = s*(sm*log2e*2^23) + lse_s23 -> fptosi.
        lse_s23_val = lse_val
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)

        # Crude Schraudolph 2^x (fast_exp2): P~ = bitcast(fptosi((s*sm*log2e + lse)*
        # 2^23 + bias)). With lse host-prescaled to lse_s23 (see lse_s23_val above),
        # _exp2_of collapses to a SINGLE fma: scaled = s*(sm*log2e*2^23) + lse_s23,
        # trading the quarter-rate v_exp for 2 full-rate ops (fma+fptosi+bitcast).
        # The epilogue renormalizes P = P~/rowsum(P~), restoring sum_j P=1 so the
        # near-diagonal dS cancellation stays exact (dq decoupled from the exp approx).
        _c_scaled_scale = fx.Float32(sm_scale * _LOG2E * float(1 << 23))
        _c_scaled_floor = fx.Float32(-87.0 * float(1 << 23) + float(127 * (1 << 23) - 486411))
        _exp2_compute_type = fx.Float32.ir_type

        def _exp2_of(s_r, lse_t, apply_mask):
            if const_expr(fast_exp2):
                # maximumf guards the all-mask -inf (masked s_r=-inf -> scaled -inf
                # -> maximumf(floor) -> 2^-87=0, no exp2(-inf)=NaN; pitfalls/04), so
                # it is load-bearing only on masked (diagonal) tiles. The mask-free
                # bulk has bounded args (>> -87) so the clamp is dropped there.
                scaled = fmath.fma(s_r, _c_scaled_scale, lse_t, fastmath=fm_fast)
                if const_expr(apply_mask):
                    scaled = ArithValue(scaled).maximumf(_c_scaled_floor)
                i = arith.fptosi(fx.Int32.ir_type, _raw(scaled))
                return ArithValue(i).bitcast(_exp2_compute_type)
            # Exact path (fast_exp2=False, unused) expects lse_t = plain -log2e*lse.
            diff = fmath.fma(s_r, c_sm_scale_log2e, lse_t, fastmath=fm_fast)
            return ArithValue(diff).exp2(fastmath=fm_fast)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound (causal). ----
        _q_end = q_start + BLOCK_M
        kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))

        k_swz_mask = _k_bank_mask(lane_mod_32)
        v_swz_mask = _v_bank_mask(lane_mod_32)

        def _a_idx_lo(a_base, ks, swz_mask):
            col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return a_base + lane_mod_32 * K_STRIDE + (col ^ swz_mask)

        def _a_idx_hi(a_base, ks, swz_mask):
            col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return a_base + fx.Index(K_SUB_N * K_STRIDE) + lane_mod_32 * K_STRIDE + (col ^ swz_mask)

        def _gemm_kq(a_base, b_packs, swz_mask=k_swz_mask, init=None):
            """GEMM1 template: acc[M=rows, N=q] = A[rows,D] @ B[q,D]^T over D. `init`
            pre-loads BOTH MFMA accumulators (used to fold the per-q delta-center add
            into the dP GEMM for free, mirroring dkdv's _neg_delta_acc)."""
            acc_lo = c_zero_v16f32 if init is None else init
            acc_hi = c_zero_v16f32 if init is None else init
            for ks in range_constexpr(K_STEPS_QK):
                a_lo = Vec.load(mfma_pack_type, lds, [_a_idx_lo(a_base, ks, swz_mask)])
                a_hi = Vec.load(mfma_pack_type, lds, [_a_idx_hi(a_base, ks, swz_mask)])
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
            # dQ accumulators; +rowsum(P~) when fast_exp2 so the epilogue can
            # renormalize dQ = sm/R * dQ~ (P~ is the unnormalized Schraudolph prob).
            init_args = [c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]
            if const_expr(fast_exp2):
                init_args = init_args + [c_zero_f]
        elif const_expr(IS_FUSED):
            # [A_accs(D_CHUNKS), B_accs(D_CHUNKS), delta(, rowsum)] in one kv pass.
            # fast_exp2 adds a rowsum(P~) accumulator so the epilogue can renormalize
            # P = P~/rowsum(P~) (restores sum_j P=1 for the near-diagonal dS cancel).
            init_args = [c_zero_v16f32 for _ in range_constexpr(2 * D_CHUNKS)] + [c_zero_f]
            if const_expr(fast_exp2):
                init_args = init_args + [c_zero_f]
        else:
            # delta mode: delta_acc; +rowsum(P~) when fast_exp2 (renorm to true delta).
            init_args = [c_zero_f]
            if const_expr(fast_exp2):
                init_args = init_args + [c_zero_f]

        def _kv_body(kv_start, inner_iter_args, apply_mask):
            if const_expr(IS_DQ):
                dq_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]
                if const_expr(fast_exp2):
                    r_acc = inner_iter_args[D_CHUNKS]
            elif const_expr(IS_FUSED):
                a_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]
                b_accs = [inner_iter_args[D_CHUNKS + i] for i in range_constexpr(D_CHUNKS)]
                delta_acc = inner_iter_args[2 * D_CHUNKS]
                if const_expr(fast_exp2):
                    r_acc = inner_iter_args[2 * D_CHUNKS + 1]
            elif const_expr(fast_exp2):
                # delta mode + fast_exp2: [delta_acc, rowsum(P~)] (always a 2-list).
                delta_acc = inner_iter_args[0]
                r_acc = inner_iter_args[1]
            else:
                # A single loop-carried value can arrive unwrapped (not a list).
                delta_acc = (
                    inner_iter_args[0] if isinstance(inner_iter_args, (list, tuple)) else inner_iter_args
                )

            # WAR guard: the single K/V LDS region is overwritten each iteration (no
            # double buffer), so wait for the previous iteration's LDS reads. s_barrier
            # alone only syncs wave *execution*, not outstanding lgkmcnt (ds_read) ops;
            # drain them first so the next DMA can't overwrite LDS mid-read. (The legacy
            # bank-conflict-serialized reads hid this WAR hazard; the finer V swizzle
            # issues reads fast enough to expose it as run-to-run nondeterminism.)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(ENABLE_DMA):
                coop_dma_tile(k_rsrc, lds_base_idx, kv_start)
                coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), kv_start, _v_bank_mask)
                if const_expr(USE_K16):
                    coop_dma_tile(k16_rsrc, lds_base_idx + fx.Index(LDS_K16_BASE * 2), kv_start)
                rocdl.s_waitcnt(0)
            else:
                _coop_load(k_ptr, fx.Index(0), kv_start)
                _coop_load(v_ptr, fx.Index(LDS_V_BASE), kv_start, _v_swizzle)
            gpu.barrier()

            # GEMM1: S[kv,q] = K @ Q^T
            s_lo_acc, s_hi_acc = _gemm_kq(fx.Index(0), q_b_packs)
            # dP[kv,q] = V @ dO^T (same template, V as "K", dO as "Q"). V is normal-read
            # only, so it uses the finer 8-granular v_swz_mask (2-way vs K's stuck 8-way).
            # identity_center: pre-load the accumulator with the per-q delta_val (uniform
            # over this lane's 16 kv elements) so dp_acc = dO@V^T + delta_val comes out of
            # the MFMA directly -> the per-element dS-centering add below is removed. Same
            # deterministic re-association class as P3/H1 (fp add-order shift only).
            if const_expr(IS_FUSED and identity_center):
                _dp_init = Vec.from_elements([delta_val], fx.Float32).broadcast_to(16).ir_value()
                dp_lo_acc, dp_hi_acc = _gemm_kq(fx.Index(LDS_V_BASE), do_b_packs, v_swz_mask, init=_dp_init)
            else:
                dp_lo_acc, dp_hi_acc = _gemm_kq(fx.Index(LDS_V_BASE), do_b_packs, v_swz_mask)

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
                return (_exp2_of(s_lo_r, lse_s23_val, apply_mask), _exp2_of(s_hi_r, lse_s23_val, apply_mask))

            if const_expr(not IS_FUSED):
                p_lo = []
                p_hi = []
                for r in range_constexpr(16):
                    plo_r, phi_r = _p_exp2(r)
                    p_lo.append(plo_r)
                    p_hi.append(phi_r)
                dp_lo = [Vec(dp_lo_acc)[r] for r in range_constexpr(16)]
                dp_hi = [Vec(dp_hi_acc)[r] for r in range_constexpr(16)]
                # rowsum(P~) for the fast_exp2 renorm (delta & dq share the recompute,
                # so R is bit-identical between the two kernels -> exact cancellation).
                if const_expr(fast_exp2):
                    r_local = c_zero_f
                    for r in range_constexpr(16):
                        r_local = _fadd(r_local, _fadd(p_lo[r], p_hi[r]))

            # Build the loop-carried yield args conditionally, then yield ONCE at the
            # tail (a single scf.yield per loop body, mirroring the forward).
            if const_expr(IS_DELTA):
                local = c_zero_f
                for r in range_constexpr(16):
                    local = _fadd(local, _fmul(p_lo[r], dp_lo[r]))
                    local = _fadd(local, _fmul(p_hi[r], dp_hi[r]))
                delta_acc = _fadd(delta_acc, local)
                if const_expr(fast_exp2):
                    r_acc = _fadd(r_acc, r_local)
                    return [delta_acc, r_acc]
                return [delta_acc]
            elif const_expr(IS_FUSED and identity_center and not apply_mask):
                # Vectorized bulk (below-diagonal) fused path. Same math as the scalar
                # branch below, but each 8-slot group is carried as vector<8xf32> so the
                # elementwise softmax/dS ops (exp2 fma, dP centering add, C=P*dP mul,
                # rowsum/rho reductions) lower to packed v_pk_* instead of scalar
                # v_add/v_mul/v_fma -> cuts VALU issues on this VALU-issue-bound kernel.
                # The exp2 approx and C=P*dP are strictly elementwise, so plo/phi/clo/chi
                # are bit-identical to the scalar path (A/B GEMM operands unchanged); only
                # the scalar rho/R reductions are re-associated (still deterministic ->
                # det gate holds; cos/l2 unaffected within margin). apply_mask handled by
                # the scalar branch (diagonal tiles only; the maximumf floor clamp is a
                # no-op off-diagonal). identity_center only (plain bf16 operands).
                v8i32_ty = Vec.make_type(8, fx.Int32)
                lse_v8 = Vec.from_elements([lse_s23_val], fx.Float32).broadcast_to(8).ir_value()
                scale_v8 = Vec.filled(8, sm_scale * _LOG2E * float(1 << 23), fx.Float32).ir_value()

                def _slice8(acc, base):
                    v = Vec(acc)
                    return v.shuffle(v, [base + j for j in range_constexpr(8)]).ir_value()

                def _exp2_v8(s_v8):
                    scaled = fmath.fma(_raw(s_v8), scale_v8, lse_v8, fastmath=fm_fast)
                    i = arith.fptosi(v8i32_ty, _raw(scaled))
                    return Vec(i).bitcast(fx.Float32).ir_value()

                def _hred8(v8):
                    v = Vec(v8)
                    s4 = Vec(_fadd(v.shuffle(v, [0, 1, 2, 3]).ir_value(), v.shuffle(v, [4, 5, 6, 7]).ir_value()))
                    s2 = Vec(_fadd(s4.shuffle(s4, [0, 1]).ir_value(), s4.shuffle(s4, [2, 3]).ir_value()))
                    return _fadd(s2[0], s2[1])

                # Accumulate the per-group C/P sums as vector<8xf32> across the PV_K
                # steps and reduce ONCE at the tail, instead of an _hred8 per step:
                # sum_pks hred8(g_pks) == hred8(sum_pks g_pks) (re-associated the same
                # way P3's rho/R reduction already is -> deterministic, det gate holds;
                # cos/l2 unaffected within margin). Trades PV_K_STEPS-1 extra v8 adds
                # for PV_K_STEPS-1 fewer narrowing-shuffle reductions on this partly
                # VALU-issue-bound kernel.
                c_sum_v8 = None
                p_sum_v8 = None
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * 8
                    plo_v = _exp2_v8(_slice8(s_lo_acc, base))
                    phi_v = _exp2_v8(_slice8(s_hi_acc, base))
                    # dp_lo/hi_acc already hold (dO@V^T + delta_val) via the GEMM acc init.
                    clo_v = _fmul(plo_v, _slice8(dp_lo_acc, base))
                    chi_v = _fmul(phi_v, _slice8(dp_hi_acc, base))
                    c_g = _fadd(clo_v, chi_v)
                    c_sum_v8 = c_g if c_sum_v8 is None else _fadd(c_sum_v8, c_g)
                    if const_expr(fast_exp2):
                        p_g = _fadd(plo_v, phi_v)
                        p_sum_v8 = p_g if p_sum_v8 is None else _fadd(p_sum_v8, p_g)
                    plo_p = bf16_trunc_pack_v8([Vec(plo_v)[i] for i in range_constexpr(8)])
                    phi_p = bf16_trunc_pack_v8([Vec(phi_v)[i] for i in range_constexpr(8)])
                    clo_p = bf16_trunc_pack_v8([Vec(clo_v)[i] for i in range_constexpr(8)])
                    chi_p = bf16_trunc_pack_v8([Vec(chi_v)[i] for i in range_constexpr(8)])
                    for dc in range_constexpr(D_CHUNKS):
                        k_lo, k_hi = _read_k_tr(dc * PV_K_STEPS + pks)
                        a_accs[dc] = mfma_acc(k_lo, clo_p, a_accs[dc])
                        a_accs[dc] = mfma_acc(k_hi, chi_p, a_accs[dc])
                        b_accs[dc] = mfma_acc(k_lo, plo_p, b_accs[dc])
                        b_accs[dc] = mfma_acc(k_hi, phi_p, b_accs[dc])
                delta_acc = _fadd(delta_acc, _hred8(c_sum_v8))
                if const_expr(fast_exp2):
                    r_local = _hred8(p_sum_v8)
                _fused_yield = (
                    [a_accs[i] for i in range_constexpr(D_CHUNKS)]
                    + [b_accs[i] for i in range_constexpr(D_CHUNKS)]
                    + [delta_acc]
                )
                if const_expr(fast_exp2):
                    r_acc = _fadd(r_acc, r_local)
                    _fused_yield = _fused_yield + [r_acc]
                return _fused_yield
            elif const_expr(IS_FUSED):
                # One pass accumulates delta=sum_j P*dP, A=sum_j (P*dP)*K, B=sum_j P*K
                # (dQ = sm*(A - delta*B)). To keep VGPR pressure low (dual A/B
                # accumulators already cost 2x dq's), process one 8-slot group at a
                # time: exp2 -> C=P*dP -> fp16 pack -> GEMM2 immediately, so only the
                # current group's packs stay live (not all 16 at once).
                local = c_zero_f
                if const_expr(fast_exp2):
                    r_local = c_zero_f
                # identity_center: C = P*(dP - delta_id) (dP + delta_val, delta_val =
                # -delta_id) so A accumulates the centered dS and `local` becomes the
                # residual rho = sum_j P*(dP-delta_id); operands are plain bf16.
                # legacy: C = P*dP (uncentered), `local` is delta = sum_j P*dP; fp16.
                _pack8 = bf16_trunc_pack_v8 if identity_center else _to_f16_v8
                _mfma_ab = mfma_acc if identity_center else mfma_f16
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * 8
                    plo_g = []
                    phi_g = []
                    clo_g = []
                    chi_g = []
                    for i in range_constexpr(8):
                        r = base + i
                        plo, phi = _p_exp2(r)
                        # identity_center: delta_val is folded into the dP GEMM acc init
                        # (see above), so dp_lo/hi_acc already hold dO@V^T + delta_val.
                        dp_lo_r = Vec(dp_lo_acc)[r]
                        dp_hi_r = Vec(dp_hi_acc)[r]
                        clo = _fmul(plo, dp_lo_r)
                        chi = _fmul(phi, dp_hi_r)
                        local = _fadd(local, clo)
                        local = _fadd(local, chi)
                        if const_expr(fast_exp2):
                            r_local = _fadd(r_local, plo)
                            r_local = _fadd(r_local, phi)
                        plo_g.append(plo)
                        phi_g.append(phi)
                        clo_g.append(clo)
                        chi_g.append(chi)
                    plo_p = _pack8(plo_g)
                    phi_p = _pack8(phi_g)
                    clo_p = _pack8(clo_g)
                    chi_p = _pack8(chi_g)
                    # GEMM2: A[D,q] += K^T @ C ; B[D,q] += K^T @ P (K transpose-read
                    # as "V"; C/P packs as "P"). identity_center reads the bf16 K tile
                    # directly; the legacy fp16 path reads the dual-tile fp16 K copy.
                    for dc in range_constexpr(D_CHUNKS):
                        if const_expr(identity_center):
                            k_lo, k_hi = _read_k_tr(dc * PV_K_STEPS + pks)
                        elif const_expr(ENABLE_DMA):
                            # Dual-tile: read the fp16 K copy directly (no conversion).
                            k_lo, k_hi = _read_k16_tr(dc * PV_K_STEPS + pks)
                        else:
                            k_lo, k_hi = _read_k_tr(dc * PV_K_STEPS + pks)
                            k_lo = _k_to_f16(k_lo)
                            k_hi = _k_to_f16(k_hi)
                        a_accs[dc] = _mfma_ab(k_lo, clo_p, a_accs[dc])
                        a_accs[dc] = _mfma_ab(k_hi, chi_p, a_accs[dc])
                        b_accs[dc] = _mfma_ab(k_lo, plo_p, b_accs[dc])
                        b_accs[dc] = _mfma_ab(k_hi, phi_p, b_accs[dc])
                delta_acc = _fadd(delta_acc, local)
                _fused_yield = (
                    [a_accs[i] for i in range_constexpr(D_CHUNKS)]
                    + [b_accs[i] for i in range_constexpr(D_CHUNKS)]
                    + [delta_acc]
                )
                if const_expr(fast_exp2):
                    r_acc = _fadd(r_acc, r_local)
                    _fused_yield = _fused_yield + [r_acc]
                return _fused_yield
            else:
                # dS = P~ .* (dP - delta_true). The delta buffer stores -delta_true
                # (negated for dkdv's accumulator fold), so dP - delta_true = dP +
                # delta_val. The subtraction is fp32 (near-diagonal cancellation done
                # before the bf16 pack), so the dQ GEMM below is pure bf16 (CK-style:
                # fp32 dS -> bf16 operand -> fp32 accumulate, no fp16 operand).
                ds_lo = []
                ds_hi = []
                for r in range_constexpr(16):
                    ds_lo.append(_fmul(p_lo[r], _fadd(dp_lo[r], delta_val)))
                    ds_hi.append(_fmul(p_hi[r], _fadd(dp_hi[r], delta_val)))
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
                if const_expr(fast_exp2):
                    r_acc = _fadd(r_acc, r_local)
                    return [dq_accs[i] for i in range_constexpr(D_CHUNKS)] + [r_acc]
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
            if const_expr(fast_exp2):
                delta_final = loop_results[0]
                r_final = loop_results[1]
            else:
                delta_final = (
                    loop_results[0] if isinstance(loop_results, (list, tuple)) else loop_results
                )
            delta_full = _fadd(delta_final, reduction_peer(delta_final))
            # With fast_exp2 the recomputed P~ is the unnormalized Schraudolph prob
            # (rowsum R != 1); renormalize to the true delta = (sum_j P~*dP)/R so the
            # dq kernel's near-diagonal cancellation (which recomputes the same R) is
            # exact. Store -delta so dkdv folds it straight into its dP accumulator.
            if const_expr(fast_exp2):
                r_full = _fadd(r_final, reduction_peer(r_final))
                inv_r = rocdl.rcp(T.f32, _raw(r_full))
                delta_full = _fmul(delta_full, inv_r)
            if lane_div_32 == fx.Index(0):
                buffer_ops.buffer_store(
                    fx.Float32(_fsub(c_zero_f, delta_full)),
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
                # With fast_exp2, P~ is unnormalized (rowsum R = sum_j P~ != 1).
                # Renormalize P = P~/R so sum_j P = 1 (exact near-diagonal cancel):
                # true delta = (sum_j P~*dP)/R and dQ = sm/R*(A~ - (delta~/R)*B~).
                # inv_r is per-q (== lane_mod_32), constant over the 16 D-registers.
                if const_expr(fast_exp2):
                    r_final = loop_results[2 * D_CHUNKS + 1]
                    r_full = _fadd(r_final, reduction_peer(r_final))
                    inv_r = rocdl.rcp(T.f32, _raw(r_full))
                    delta_full = _fmul(delta_full, inv_r)  # true delta (for dkdv)
                    dq_scale = fx.Float32(_fmul(fx.Float32(sm_scale), fx.Float32(inv_r)))
                else:
                    dq_scale = fx.Float32(sm_scale)
                if const_expr(not identity_center):
                    if lane_div_32 == fx.Index(0):
                        # Store -delta: dkdv folds it into its dP MFMA accumulator init
                        # (dP - delta = dO@V^T + (-delta)), removing its per-element dS
                        # subtract. dQ below still uses the positive delta_full.
                        buffer_ops.buffer_store(
                            fx.Float32(_fsub(c_zero_f, delta_full)),
                            delta_out_rsrc,
                            _lse_elem * fx.Index(4),
                            mask=ArithValue(q_row < seq_len_v),
                            offset_is_bytes=True,
                        )
                # identity_center: delta_full is now rho/R (peer-reduced sum_j
                # P~*(dP-delta_id), renormalized), so A - delta_full*B == A - (rho/R)*B
                # exactly recovers the consistent dq; DELTA already holds delta_id (for
                # dkdv) and is not overwritten.
                # dQ = dq_scale*(A - delta*B). The accumulators are [M=D, N=q] with
                # N=q == lane_mod_32, so delta_full (per-q, per-lane) is constant across
                # the 16 D-registers -> a single scalar multiply per register.
                dq_finals = []
                for dc in range_constexpr(D_CHUNKS):
                    a_v = Vec(a_finals[dc])
                    b_v = Vec(b_finals[dc])
                    vals = [
                        fx.Float32(_fsub(a_v[r], _fmul(delta_full, b_v[r])))
                        for r in range_constexpr(16)
                    ]
                    dq_finals.append(Vec.from_elements(vals, fx.Float32).ir_value())
                sm_vec = Vec.from_elements([dq_scale], fx.Float32).broadcast_to(16)
            else:
                # dq mode: dQ~ = sum_j dS~_j @ K accumulated in the loop. With
                # fast_exp2 the dS used the unnormalized P~, so renormalize with
                # inv_r = 1/rowsum(P~) (per-q, factors out of the GEMM -> a single
                # epilogue scalar): dQ = sm * inv_r * dQ~.
                dq_finals = [loop_results[dc] for dc in range_constexpr(D_CHUNKS)]
                if const_expr(fast_exp2):
                    r_final = loop_results[D_CHUNKS]
                    r_full = _fadd(r_final, reduction_peer(r_final))
                    inv_r = rocdl.rcp(T.f32, _raw(r_full))
                    dq_scale = fx.Float32(_fmul(fx.Float32(sm_scale), fx.Float32(inv_r)))
                else:
                    dq_scale = fx.Float32(sm_scale)
                sm_vec = Vec.from_elements([dq_scale], fx.Float32).broadcast_to(16)
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


def build_flash_attn_bwd_odo_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    num_kv_heads=None,
    causal=True,
    sm_scale=None,
    waves_per_eu=4,
    block=256,
):
    """Identity-delta ("odo") kernel: DELTA[b,hq,s] = -sum_d O[b,s,hq,d]*dO[b,s,hq,d].

    Memory-bound O.dO row-reduce that replaces the torch (out*dout).sum(-1): one
    thread owns one (b,s,hq) row, reads its D-vector of O and dO (coalesced bf16
    buffer loads), multiplies and sums in fp32, negates (the dkdv/dq fold convention
    stores -delta_id) and scatter-stores the scalar to the transposed [B,Hq,S] delta.
    delta_id is only a centering value (the fused dq kernel corrects rho/R*B exactly),
    so the bf16*bf16 product with fp32 accumulate is enough precision.

    waves_per_eu=4 (not 8): the row's O/dO loads are hoisted ahead of the reduction
    (below) so all NVEC*2 dwordx4 loads are in flight before the first is consumed,
    which needs ~80 VGPR. At the old wpe=8 (64-VGPR budget) that spilled (116 B
    scratch) and the loads still drained one pair at a time (latency-exposed, ~35%
    HBM BW); wpe=4 (128-VGPR budget) fits with 0 spill -> odo ~60us -> ~31us."""
    assert dtype_str == "bf16", "odo kernel targets bf16"
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "odo kernel targets gfx950"
    elem_dtype = dtype_to_elem_type(dtype_str)
    HEAD_DIM = head_dim
    NUM_HEADS_Q = num_heads
    VEC = 8
    assert HEAD_DIM % VEC == 0
    NVEC = HEAD_DIM // VEC
    BLOCK = block

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_odo_kernel(
        O: fx.Tensor,
        DO: fx.Tensor,
        DELTA: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
    ):
        elem_dtype_l = elem_dtype
        fm = fx.arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm)

        c_zero_f = fx.Float32(0.0)

        bid = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        row = bid * fx.Index(BLOCK) + tid
        sl = fx.Index(seq_len)
        total = fx.Index(batch_size) * sl * fx.Index(NUM_HEADS_Q)
        in_range = ArithValue(row < total)
        # OOB rows fold to row 0 for the loads; the store is masked off. (The buffer
        # descriptor also OOB-guards, but clamping keeps the offset well-formed.)
        row_c = fx.Index(in_range.select(row, fx.Index(0)))

        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=True)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=True)

        base = row_c * fx.Index(HEAD_DIM)
        # Hoist the whole row's O/dO loads ahead of the reduction so all NVEC*2
        # dwordx4 loads are issued (and in flight) before the first is consumed.
        # The original per-iter load->use pattern drained one pair at a time
        # (one s_waitcnt per load), exposing the load latency (~35% of HBM BW on
        # this memory-bound row-reduce). Same product/accumulate order -> the
        # fp32 sum is bit-identical (det-safe) and it stays an exact O*dO.
        ovs = []
        dvs = []
        for c in range_constexpr(NVEC):
            off = base + fx.Index(c * VEC)
            ovs.append(buffer_ops.buffer_load(o_rsrc, off, vec_width=VEC, dtype=elem_dtype_l))
            dvs.append(buffer_ops.buffer_load(do_rsrc, off, vec_width=VEC, dtype=elem_dtype_l))
        acc = fx.Float32(0.0)
        for c in range_constexpr(NVEC):
            prod = Vec(ovs[c]).to(fx.Float32) * Vec(dvs[c]).to(fx.Float32)
            for i in range_constexpr(VEC):
                acc = _fadd(acc, Vec(prod)[i])

        # row = ((b*S + s)*Hq + hq)  ->  delta[b,hq,s] at (b*Hq + hq)*S + s.
        hq = row_c % fx.Index(NUM_HEADS_Q)
        tmp = row_c // fx.Index(NUM_HEADS_Q)
        s = tmp % sl
        b = tmp // sl
        delta_off = (b * fx.Index(NUM_HEADS_Q) + hq) * sl + s
        neg_acc = arith.subf(_raw(c_zero_f), _raw(acc), fastmath=fm)
        buffer_ops.buffer_store(
            fx.Float32(neg_acc),
            delta_rsrc,
            delta_off * fx.Index(4),
            mask=in_range,
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch_flash_attn_bwd_odo(
        O: fx.Tensor,
        DO: fx.Tensor,
        DELTA: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream,
    ):
        total = fx.Index(batch_size) * fx.Index(seq_len) * fx.Index(NUM_HEADS_Q)
        grid_x = (total + fx.Index(BLOCK - 1)) // fx.Index(BLOCK)
        flash_attn_bwd_odo_kernel(
            O,
            DO,
            DELTA,
            batch_size,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{int(BLOCK)},{int(BLOCK)}",
            },
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    def _launch(*args, **kwargs):
        return launch_flash_attn_bwd_odo(*args, **kwargs)

    def _compile(*args):
        return flyc.compile(launch_flash_attn_bwd_odo, *args)

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

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32). Splits each old 32x32
    # accumulator into 4 independent 16x16 chains -> 4x the MFMA-latency ILP at
    # the SAME accumulator VGPR total (dep-wait is the dkdv bottleneck; MFMA is
    # latency-bound, not throughput-bound). Lane layout: lane%16 = M/N index,
    # lane//16 = K-subgroup (4 groups x 8 = K32) and, on the C output, the
    # M-block ((lane//16)*4 + t, t in 0..3 -> 4 f32/lane).
    M_TILE = 16
    N_TILE = 16
    D_TILE = 16
    K_STEP_QK = 32                          # K=32 per GEMM1 MFMA (contract over D)
    K_STEPS_QK = head_dim // K_STEP_QK      # d64 -> 2
    NT = ROWS_PER_WAVE_KV // N_TILE         # kv 16-tiles per wave: 32/16 = 2
    MT = BLOCK_Q // M_TILE                  # q 16-tiles: 64/16 = 4
    DT = head_dim // D_TILE                 # D 16-tiles: 64/16 = 4
    PV_K_STEP = 32                          # K=32 per GEMM2 MFMA (contract over q)
    PV_K_STEPS = BLOCK_Q // PV_K_STEP       # 64/32 = 2

    assert BLOCK_KV % NUM_WAVES == 0
    assert ROWS_PER_WAVE_KV % N_TILE == 0
    assert BLOCK_Q % M_TILE == 0
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
    if ENABLE_DMA:
        # DMA path tiles the Q/dO copy by NUM_DMA_Q batches (BLOCK_Q independent of
        # ROWS_PER_BATCH_LOAD); the VGPR-staged _coop_load fallback is unused.
        NUM_BATCHES_Q = 1
        Q_NEEDS_GUARD = False
    else:
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
        v4f32_type = Vec.make_type(4, fx.Float32)
        mfma_pack_type = v8f16_type
        MFMA_LANE_K = 8  # 8 bf16/lane; 4 lane-groups (lane//16) -> K=32

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v4f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_16x16x32_bf16, a, b, c)

        seq_len_v = fx.Index(seq_len)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16      # M/N index within a 16-tile
        kg = lane // 16         # 0..3: K-subgroup (inputs) / M-block (C output)

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
        # This wave owns ROWS_PER_WAVE_KV kv rows, split into NT 16-wide N-tiles.
        # In the 16x16 layout the owned kv row for a lane is nt*16 + lane16.
        kv_row_wave = kv_start + wave_id * ROWS_PER_WAVE_KV

        def kv_row_of(nt):
            return kv_row_wave + fx.Index(nt * N_TILE) + lane16

        def kv_row_i32_of(nt):
            return fx.Int32(kv_row_of(nt))

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

        # ---- Owned K,V B-operand packs: B[k=D][n=kv], n=lane16, k=kg*8+s. Per wave
        # NT kv 16-tiles x K_STEPS_QK D-steps; k_b_packs[nt][ks] is a v8 bf16. ----
        k_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(NT)]
        v_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(NT)]
        for nt in range_constexpr(NT):
            _kvr = kv_row_of(nt)
            for ks in range_constexpr(K_STEPS_QK):
                kv_col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
                k_b_packs[nt][ks] = buffer_ops.buffer_load(
                    k_rsrc, global_idx_kv(_kvr, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
                v_b_packs[nt][ks] = buffer_ops.buffer_load(
                    v_rsrc, global_idx_kv(_kvr, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        # Crude Schraudolph 2^x (fast_exp2): P = bitcast(fptosi((s*sm*log2e + lse)*
        # 2^23 + bias)). The (lse*2^23 + bias) addend is pre-scaled on the host (see
        # attention_flydsl_impl), so _p_of collapses to a SINGLE fma
        # scaled = s*(sm*log2e*2^23) + lse_s23 -> fptosi: the diff fma and the
        # Schraudolph *2^23+bias fma fold into one. lse_t is a plain loaded addend
        # (not an in-kernel prescale), keeping it a clean fma(var,const,loaded)->fptosi.
        _c_scaled_scale = fx.Float32(sm_scale * _LOG2E * float(1 << 23))
        _c_scaled_floor = fx.Float32(-87.0 * float(1 << 23) + float(127 * (1 << 23) - 486411))
        _compute_type = fx.Float32.ir_type

        def _p_of(s_r, lse_t, apply_mask):
            if const_expr(fast_exp2):
                # The floor clamp is load-bearing only on masked (diagonal) slots
                # (masked s_r=-inf -> scaled -inf -> maximumf(floor) -> 2^-87=0;
                # pitfalls/04). In the mask-free bulk causal-valid softmax args are
                # bounded (>> -87), so the clamp is a no-op there and is dropped.
                scaled = fmath.fma(s_r, _c_scaled_scale, lse_t, fastmath=fm_fast)
                if const_expr(apply_mask):
                    scaled = ArithValue(scaled).maximumf(_c_scaled_floor)
                i = arith.fptosi(fx.Int32.ir_type, _raw(scaled))
                return ArithValue(i).bitcast(_compute_type)
            # Exact path (fast_exp2=False, unused in the campaign) expects lse_t to be
            # the plain -log2e*lse, not the s23-prescaled addend.
            diff = fmath.fma(s_r, c_sm_scale_log2e, lse_t, fastmath=fm_fast)
            return ArithValue(diff).exp2(fastmath=fm_fast)

        # A-operand read (Q/dO from LDS): A[m=q=lane16][k=D=kg*8+s]. mt selects the
        # 16-q tile (row = mt*16 + lane16), ks the D 32-step (D = ks*32 + kg*8).
        a_swz_mask = (lane16 & fx.Index(Q_STRIDE // 16 - 1)) << fx.Index(4)

        def _a_idx(a_base, mt, ks):
            row = fx.Index(mt * M_TILE) + lane16
            col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
            return a_base + row * Q_STRIDE + (col ^ a_swz_mask)

        def _gemm_qk(a_base, b_packs, inits=None):
            """S[mt][nt] (v4f32) = A(Q/dO)[mt] @ B(owned K/V)[nt]^T over D. A is
            loaded once per (mt,ks) and reused across nt. inits[mt] optionally
            pre-loads the accumulator (folds -delta into the dP GEMM for free)."""
            a = [
                [Vec.load(mfma_pack_type, lds, [_a_idx(a_base, mt, ks)]) for ks in range_constexpr(K_STEPS_QK)]
                for mt in range_constexpr(MT)
            ]
            out = [[None] * NT for _ in range_constexpr(MT)]
            for mt in range_constexpr(MT):
                for nt in range_constexpr(NT):
                    acc = c_zero_v4f32 if inits is None else inits[mt]
                    for ks in range_constexpr(K_STEPS_QK):
                        acc = mfma_acc(a[mt][ks], b_packs[nt][ks], acc)
                    out[mt][nt] = acc
            return out

        def _read_tr(a_base, dt, pks):
            """Transpose-read Q/dO -> GEMM2 A-operand [m=D=dt*16+lane16][k=q=kg*8+s].
            Two ds_read_tr16 (4 q each): read0->s0..3 (q=pks*32+kg*4+j), read1->s4..7
            (q=pks*32+16+kg*4+j). See .claude/memory/ds_read_tr16_b64_gfx950.md."""
            col = fx.Index(dt * D_TILE) + (lane % fx.Index(4)) * fx.Index(4)
            row0 = fx.Index(pks * PV_K_STEP) + kg * fx.Index(4) + (lane16 // fx.Index(4))
            row1 = row0 + fx.Index(N_TILE)
            v0 = ds_read_tr_v4f16(a_base + row0 * Q_STRIDE + _swizzle(row0, col))
            v1 = ds_read_tr_v4f16(a_base + row1 * Q_STRIDE + _swizzle(row1, col))
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # dv/dk accumulators flat over (dt,nt): index dt*NT+nt, each v4f32,
        # C[m=D=dt*16+kg*4+t][n=kv=nt*16+lane16].
        dv_accs = [c_zero_v4f32 for _ in range_constexpr(DT * NT)]
        dk_accs = [c_zero_v4f32 for _ in range_constexpr(DT * NT)]

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
                dv_cur = [[inner[dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)]
                dk_cur = [
                    [inner[DT * NT + dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)
                ]

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

                q_start_i32 = fx.Int32(q_start)
                # This lane's q for tile mt slot t = q_start + kg*4 + mt*16 + t; the 4
                # t-values are contiguous q -> one vec4 lse/delta load per mt.
                kg_off_i32 = fx.Int32(kg) * fx.Int32(4)

                def _qblk_base(mt):  # noqa: B023
                    return _lse_head_base + fx.Index(  # noqa: B023
                        ArithValue(q_start_i32 + kg_off_i32 + fx.Int32(mt * M_TILE))
                    )

                # -delta[q] init for the dP GEMM accumulator (delta stored negated by
                # the fused kernel; same init broadcast across nt). One vec4/mt.
                delta_inits = [
                    buffer_ops.buffer_load(delta_rsrc, _qblk_base(mt), vec_width=4, dtype=fx.Float32)
                    for mt in range_constexpr(MT)
                ]

                # GEMM1a S[mt][nt]=Q@K^T ; GEMM1b dP[mt][nt]=dO@V^T (acc init=-delta).
                s_tiles = _gemm_qk(fx.Index(0), k_b_packs)
                dp_tiles = _gemm_qk(fx.Index(LDS_DO_BASE), v_b_packs, delta_inits)

                lse_vecs = [
                    buffer_ops.buffer_load(lse_rsrc, _qblk_base(mt), vec_width=4, dtype=fx.Float32)
                    for mt in range_constexpr(MT)
                ]

                # P[mt][nt]/dS[mt][nt] (each 4 f32 at q=mt*16+kg*4+t, kv=nt*16+lane16).
                P = [[None] * NT for _ in range_constexpr(MT)]
                dS = [[None] * NT for _ in range_constexpr(MT)]
                for mt in range_constexpr(MT):
                    lse_v = lse_vecs[mt]
                    for nt in range_constexpr(NT):
                        s_v = s_tiles[mt][nt]
                        dp_v = dp_tiles[mt][nt]
                        p_vals = []
                        ds_vals = []
                        for t in range_constexpr(4):
                            s_r = fx.Float32(Vec(s_v)[t])
                            if const_expr(apply_mask):
                                q_slot = q_start_i32 + kg_off_i32 + fx.Int32(mt * M_TILE + t)
                                s_r = ArithValue(kv_row_i32_of(nt) > q_slot).select(c_neg_inf, s_r)
                            p = _p_of(s_r, fx.Float32(Vec(lse_v)[t]), apply_mask)
                            p_vals.append(p)
                            ds_vals.append(_fmul(p, Vec(dp_v)[t]))
                        P[mt][nt] = p_vals
                        dS[mt][nt] = ds_vals

                # B-operand packs for GEMM2: pack pks combines mt=2*pks (k=0..3) and
                # 2*pks+1 (k=4..7) -> 8 q values/lane matching _read_tr's q ordering.
                p_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
                ds_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
                for pks in range_constexpr(PV_K_STEPS):
                    ma, mb = 2 * pks, 2 * pks + 1
                    for nt in range_constexpr(NT):
                        p_pack[pks][nt] = bf16_trunc_pack_v8(P[ma][nt] + P[mb][nt])
                        ds_pack[pks][nt] = bf16_trunc_pack_v8(dS[ma][nt] + dS[mb][nt])

                # GEMM2a dV^T[dt][nt] += dO_tr[dt] @ P ; GEMM2b dK^T[dt][nt] += Q_tr @ dS.
                # Depth-1 prefetch across dt: issue dt+1's dO transpose-reads before
                # dt's dV MFMAs and dt+1's Q reads between the dV and dK MFMAs, so the
                # ds_read_tr16 LDS latency hides in the MFMA shadow. pks is the outer
                # MFMA loop (nt inner) so the two nt accumulators of each 16x16 chain
                # interleave -> more independent MFMA ILP. Pure reorder -> det-neutral.
                do_tr = [_read_tr(fx.Index(LDS_DO_BASE), 0, pks) for pks in range_constexpr(PV_K_STEPS)]
                q_tr = [_read_tr(fx.Index(0), 0, pks) for pks in range_constexpr(PV_K_STEPS)]
                for dt in range_constexpr(DT):
                    if const_expr(dt + 1 < DT):
                        do_tr_n = [
                            _read_tr(fx.Index(LDS_DO_BASE), dt + 1, pks) for pks in range_constexpr(PV_K_STEPS)
                        ]
                    for pks in range_constexpr(PV_K_STEPS):
                        for nt in range_constexpr(NT):
                            dv_cur[dt][nt] = mfma_acc(do_tr[pks], p_pack[pks][nt], dv_cur[dt][nt])
                    if const_expr(dt + 1 < DT):
                        q_tr_n = [_read_tr(fx.Index(0), dt + 1, pks) for pks in range_constexpr(PV_K_STEPS)]
                    for pks in range_constexpr(PV_K_STEPS):
                        for nt in range_constexpr(NT):
                            dk_cur[dt][nt] = mfma_acc(q_tr[pks], ds_pack[pks][nt], dk_cur[dt][nt])
                    if const_expr(dt + 1 < DT):
                        do_tr, q_tr = do_tr_n, q_tr_n

                out = [dv_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]
                out += [dk_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]
                return out

            _carry = dv_accs + dk_accs
            loop_results = _carry
            for q_start, inner in range(_q_loop_start, _masked_upper, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
            for q_start, inner in range(_unmask_start, seq_len_v, _step, init=loop_results):
                loop_results = yield _q_body(q_start, inner, False)
            dv_accs = [loop_results[i] for i in range_constexpr(DT * NT)]
            dk_accs = [loop_results[DT * NT + i] for i in range_constexpr(DT * NT)]

        # ---- Store dV[kv,D], dK[kv,D]. The 16x16 C-layout gives each lane 4
        # CONTIGUOUS D values (D = dt*16 + kg*4 + t) at kv = nt*16 + lane16, so the
        # store is direct (no permlane32 transpose needed, unlike the 32x32 path). ----
        sm_vec4 = Vec.from_elements([fx.Float32(sm_scale)], fx.Float32).broadcast_to(4)

        def _store(accs, rsrc, scale):
            for dt in range_constexpr(DT):
                for nt in range_constexpr(NT):
                    v = Vec(accs[dt * NT + nt])
                    if const_expr(scale):
                        v = v * sm_vec4
                    lo = rocdl.cvt_pk_bf16_f32(v[0], v[1])
                    hi = rocdl.cvt_pk_bf16_f32(v[2], v[3])
                    o_pack = Vec.from_elements([fx.Int32(_raw(lo)), fx.Int32(_raw(hi))], fx.Int32)
                    d_col = fx.Index(dt * D_TILE) + kg * fx.Index(4)
                    g_idx = global_idx_kv(kv_row_of(nt), d_col)
                    buffer_ops.buffer_store(o_pack, rsrc, g_idx * fx.Index(2), offset_is_bytes=True)

        _store(dv_accs, dv_rsrc, False)
        _store(dk_accs, dk_rsrc, True)

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


def build_flash_attn_bwd_dq_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    block_kv=64,
    num_kv_heads=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    enable_dma=True,
    fast_exp2=True,
):
    """Build the dQ Q-outer backward launcher (16x16x32 mirror of dkdv).

    One work-group owns BLOCK_M q rows of one q-head and loops the causal kv
    blocks, accumulating dQ in registers -> single write, deterministic. Fused
    identity-center path: DELTA holds -delta_id = -rowsum_d(O.dO); the kernel
    centers dP by it in-loop (plain bf16 operands) and corrects the residual rho/R
    in the epilogue -> exact consistent dQ in one pass (the K16 arg is unused).

    Roles vs dkdv are swapped q<->kv:
      * Q,dO owned as MFMA B-operands (register-resident, per wave's q rows).
      * K,V streamed to LDS, read normally for the S/dP GEMMs and K transpose-read
        (ds_read_tr) for the A/B GEMMs.
      * GEMM1a S[kv,q]=K@Q^T, GEMM1b dP[kv,q]=V@dO^T (acc init folds -delta_id).
      * GEMM2a A[D,q] += K_tr @ C (C=P*(dP-delta_id)), GEMM2b B[D,q] += K_tr @ P.
      * rho=sum_kv C, R=sum_kv P~ reduced across the K-subgroups (lane^16/^32) in
        the epilogue: dQ = sm/R * (A - (rho/R)*B), stored [q,D] (direct 16x16
        C-layout: 4 contiguous D/lane -> no permlane32 transpose).
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "bwd dq kernel targets gfx950"
    assert dtype_str == "bf16", "bwd dq kernel targets bf16"
    assert causal, "bwd dq kernel is causal-only for the GPT-OSS campaign"
    assert fast_exp2, "bwd dq 16x16 kernel is the fast_exp2 identity-center path"

    ENABLE_DMA = enable_dma and not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    BLOCK_M = 128                            # q rows per work-group (owned)
    WARP_SIZE = 64
    BLOCK_KV = block_kv                      # kv rows per loop iteration (LDS tile)
    flat_work_group_size = 256
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_Q = BLOCK_M // NUM_WAVES   # 32

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32); q<->kv mirror of dkdv. ----
    M_TILE = 16
    N_TILE = 16
    D_TILE = 16
    K_STEP_QK = 32                           # K=32 per GEMM1 MFMA (contract over D)
    K_STEPS_QK = head_dim // K_STEP_QK       # d64 -> 2
    QT = ROWS_PER_WAVE_Q // N_TILE           # owned q 16-tiles per wave: 2
    KVT = BLOCK_KV // M_TILE                 # looped kv 16-tiles in the LDS block: 4
    DT = head_dim // D_TILE                  # D 16-tiles: 4
    PV_K_STEP = 32                           # K=32 per GEMM2 MFMA (contract over kv)
    PV_K_STEPS = BLOCK_KV // PV_K_STEP       # 64/32 = 2

    assert BLOCK_M % NUM_WAVES == 0
    assert ROWS_PER_WAVE_Q % N_TILE == 0
    assert BLOCK_KV % M_TILE == 0
    assert head_dim % 32 == 0 and head_dim >= 64

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    K_STRIDE = HEAD_DIM
    LDS_TILE = BLOCK_KV * K_STRIDE
    LDS_V_BASE = LDS_TILE
    LDS_TOTAL = 2 * LDS_TILE

    VEC_WIDTH = 16
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="flash_attn_bwd_smem_dq16")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_bwd_dq_kernel(
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
        v4f32_type = Vec.make_type(4, fx.Float32)
        mfma_pack_type = v8f16_type
        MFMA_LANE_K = 8  # 8 bf16/lane; 4 lane-groups (lane//16) -> K=32

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v4f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_16x16x32_bf16, a, b, c)

        seq_len_v = fx.Index(seq_len)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16      # M/N index within a 16-tile
        kg = lane // 16         # 0..3: K-subgroup (inputs) / M-block (C output)

        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_off
            ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- block_id decode: kv_head fastest (XCD/L2), q-in-group, then q-tile with
        # the causal load-balance interleave (mirror old dq / the forward). ----
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
        _qt_half = _qt_disp // fx.Index(2)
        _qt_is_odd = ArithValue(_qt_disp % fx.Index(2) == fx.Index(1))
        q_tile_idx = fx.Index(_qt_is_odd.select(num_q_tiles - fx.Index(1) - _qt_half, _qt_half))
        q_start = q_tile_idx * BLOCK_M

        # Fold per-batch element offset into raw K/V pointers (0-based rows).
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

        def bf16_trunc_pack_v8(f32_vals):
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        def _swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)
            return col_idx ^ mask

        # ---- Per-batch descriptors (batch base folded into SRD base). ----
        _q_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _q_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _kv_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        _kv_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        _lse_per_batch = seq_len_v * fx.Index(NUM_HEADS_Q)
        _lse_nrec_bytes = _raw(_lse_per_batch * fx.Index(4))
        _lse_batch_byte_off = _raw(batch_idx * _lse_per_batch * fx.Index(4))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )
        delta_in_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )

        # ---- DMA-to-LDS for the K/V tiles (buffer_load_dwordx4 ... lds). ----
        if const_expr(ENABLE_DMA):
            k_rsrc = buffer_ops.create_buffer_resource(
                K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            v_rsrc = buffer_ops.create_buffer_resource(
                V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            lds_base_idx = buffer_ops.extract_base_index(lds, address_space=3)
            DMA_BYTES = 16
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            KV_TILE_BYTES = BLOCK_KV * K_STRIDE * 2
            NUM_DMA_KV = KV_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_KV_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_tile(src_rsrc, lds_byte_base, tile_start):
                """DMA a BLOCK_KV x head_dim K/V tile into the swizzled LDS layout."""
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

        # ---- Owned Q,dO B-operand packs: B[k=D][n=q], n=lane16, k=kg*8+s. Per wave
        # QT q 16-tiles x K_STEPS_QK D-steps; q_b_packs[qt][ks] is a v8 bf16. ----
        q_row_wave = q_start + wave_id * ROWS_PER_WAVE_Q

        def q_row_of(qt):
            return q_row_wave + fx.Index(qt * N_TILE) + lane16

        q_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        do_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        for qt in range_constexpr(QT):
            _qr = q_row_of(qt)
            for ks in range_constexpr(K_STEPS_QK):
                q_col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
                q_b_packs[qt][ks] = buffer_ops.buffer_load(
                    q_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
                do_b_packs[qt][ks] = buffer_ops.buffer_load(
                    do_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )

        # ---- Owned LSE/-delta_id per q (one scalar per qt, q = qt*16 + lane16). ----
        lse_owned = []
        delta_owned = []
        for qt in range_constexpr(QT):
            _lse_elem = q_head_idx * seq_len_v + q_row_of(qt)
            lse_owned.append(
                fx.Float32(buffer_ops.buffer_load(lse_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32))
            )
            delta_owned.append(
                fx.Float32(buffer_ops.buffer_load(delta_in_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32))
            )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)

        _c_scaled_scale = fx.Float32(sm_scale * _LOG2E * float(1 << 23))
        _c_scaled_floor = fx.Float32(-87.0 * float(1 << 23) + float(127 * (1 << 23) - 486411))
        _compute_type = fx.Float32.ir_type
        v4i32_ty = Vec.make_type(4, fx.Int32)
        scale_v4 = Vec.filled(4, sm_scale * _LOG2E * float(1 << 23), fx.Float32)

        def _hred4(v4):
            v = Vec(v4)
            return _fadd(_fadd(v[0], v[1]), _fadd(v[2], v[3]))

        def _p_of(s_r, lse_t, apply_mask):
            if const_expr(fast_exp2):
                scaled = fmath.fma(s_r, _c_scaled_scale, lse_t, fastmath=fm_fast)
                if const_expr(apply_mask):
                    scaled = ArithValue(scaled).maximumf(_c_scaled_floor)
                i = arith.fptosi(fx.Int32.ir_type, _raw(scaled))
                return ArithValue(i).bitcast(_compute_type)
            diff = fmath.fma(s_r, c_sm_scale_log2e, lse_t, fastmath=fm_fast)
            return ArithValue(diff).exp2(fastmath=fm_fast)

        # A-operand read (K/V from LDS): A[m=kv=lane16][k=D=kg*8+s]. kvt selects the
        # 16-kv tile (row = kvt*16 + lane16), ks the D 32-step (D = ks*32 + kg*8).
        a_swz_mask = (lane16 & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)

        def _a_idx(a_base, kvt, ks):
            row = fx.Index(kvt * M_TILE) + lane16
            col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
            return a_base + row * K_STRIDE + (col ^ a_swz_mask)

        def _gemm1(a_base, b_packs, inits_q=None):
            """S[kvt][qt] (v4f32) = A(K/V)[kvt] @ B(owned Q/dO)[qt]^T over D. A is
            loaded once per (kvt,ks) and reused across qt. inits_q[qt] optionally
            pre-loads the accumulator (folds -delta_id into the dP GEMM for free)."""
            a = [
                [Vec.load(mfma_pack_type, lds, [_a_idx(a_base, kvt, ks)]) for ks in range_constexpr(K_STEPS_QK)]
                for kvt in range_constexpr(KVT)
            ]
            out = [[None] * QT for _ in range_constexpr(KVT)]
            for kvt in range_constexpr(KVT):
                for qt in range_constexpr(QT):
                    acc = c_zero_v4f32 if inits_q is None else inits_q[qt]
                    for ks in range_constexpr(K_STEPS_QK):
                        acc = mfma_acc(a[kvt][ks], b_packs[qt][ks], acc)
                    out[kvt][qt] = acc
            return out

        def _read_tr(a_base, dt, pks):
            """Transpose-read K -> GEMM2 A-operand [m=D=dt*16+lane16][k=kv=kg*8+s]."""
            col = fx.Index(dt * D_TILE) + (lane % fx.Index(4)) * fx.Index(4)
            row0 = fx.Index(pks * PV_K_STEP) + kg * fx.Index(4) + (lane16 // fx.Index(4))
            row1 = row0 + fx.Index(N_TILE)
            v0 = ds_read_tr_v4f16(a_base + row0 * K_STRIDE + _swizzle(row0, col))
            v1 = ds_read_tr_v4f16(a_base + row1 * K_STRIDE + _swizzle(row1, col))
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # Per-q delta init (broadcast over the 4 kv output rows) and q-slot i32.
        delta_inits = [
            Vec.from_elements([delta_owned[qt]], fx.Float32).broadcast_to(4).ir_value()
            for qt in range_constexpr(QT)
        ]
        q_slot_i32 = [fx.Int32(q_row_of(qt)) for qt in range_constexpr(QT)]

        # ---- Loop-carried: A(D_CHUNKS*QT), B(D_CHUNKS*QT), rho(QT), R(QT). ----
        A_accs = [c_zero_v4f32 for _ in range_constexpr(DT * QT)]
        B_accs = [c_zero_v4f32 for _ in range_constexpr(DT * QT)]
        rho_accs = [c_zero_f for _ in range_constexpr(QT)]
        r_accs = [c_zero_f for _ in range_constexpr(QT)]

        _q_end = q_start + BLOCK_M
        kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))

        def _kv_body(kv_start, inner, apply_mask):
            A_cur = [[inner[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]
            B_cur = [
                [inner[DT * QT + dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)
            ]
            rho_cur = [inner[2 * DT * QT + qt] for qt in range_constexpr(QT)]
            r_cur = [inner[2 * DT * QT + QT + qt] for qt in range_constexpr(QT)]

            gpu.barrier()
            if const_expr(ENABLE_DMA):
                coop_dma_tile(k_rsrc, lds_base_idx, kv_start)  # noqa: B023
                coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), kv_start)  # noqa: B023
                rocdl.s_waitcnt(0)
            gpu.barrier()

            # GEMM1a S[kv,q]=K@Q^T ; GEMM1b dP[kv,q]=V@dO^T (acc init=-delta_id).
            s_tiles = _gemm1(fx.Index(0), q_b_packs)
            dp_tiles = _gemm1(fx.Index(LDS_V_BASE), do_b_packs, delta_inits)

            kv_start_i32 = fx.Int32(kv_start)
            # P[kvt][qt]/C[kvt][qt]: each 4 f32 at kv=kvt*16+kg*4+t, q=qt*16+lane16.
            P = [[None] * QT for _ in range_constexpr(KVT)]
            C = [[None] * QT for _ in range_constexpr(KVT)]
            if const_expr(not apply_mask):
                # Vectorized bulk (below-diagonal): exp2/C/reduce as packed v4 ops
                # (v_pk_*), mirroring the 32x32 kernel's v8 path. exp2 and C=P*dP are
                # strictly elementwise so P/C are bit-identical to the scalar branch;
                # rho/R re-associated in a fixed order -> deterministic (det gate holds).
                for qt in range_constexpr(QT):
                    lse_v4 = Vec.from_elements([lse_owned[qt]], fx.Float32).broadcast_to(4)
                    rho4 = None
                    r4 = None
                    for kvt in range_constexpr(KVT):
                        scaled = fmath.fma(
                            _raw(s_tiles[kvt][qt]), _raw(scale_v4), _raw(lse_v4), fastmath=fm_fast
                        )
                        p4 = Vec(arith.fptosi(v4i32_ty, _raw(scaled))).bitcast(fx.Float32)
                        c4 = p4 * Vec(dp_tiles[kvt][qt])
                        P[kvt][qt] = [p4[t] for t in range_constexpr(4)]
                        C[kvt][qt] = [c4[t] for t in range_constexpr(4)]
                        rho4 = c4 if rho4 is None else (rho4 + c4)
                        r4 = p4 if r4 is None else (r4 + p4)
                    rho_cur[qt] = _fadd(rho_cur[qt], _hred4(rho4.ir_value()))
                    r_cur[qt] = _fadd(r_cur[qt], _hred4(r4.ir_value()))
            else:
                for qt in range_constexpr(QT):
                    lse_q = lse_owned[qt]
                    rho_local = c_zero_f
                    r_local = c_zero_f
                    for kvt in range_constexpr(KVT):
                        dp_v = dp_tiles[kvt][qt]
                        s_v = s_tiles[kvt][qt]
                        p_vals = []
                        c_vals = []
                        for t in range_constexpr(4):
                            kv_slot = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4 + t)
                            s_r = ArithValue(kv_slot > q_slot_i32[qt]).select(c_neg_inf, fx.Float32(Vec(s_v)[t]))
                            p = _p_of(s_r, lse_q, True)
                            c = _fmul(p, Vec(dp_v)[t])
                            p_vals.append(p)
                            c_vals.append(c)
                            rho_local = _fadd(rho_local, c)
                            r_local = _fadd(r_local, p)
                        P[kvt][qt] = p_vals
                        C[kvt][qt] = c_vals
                    rho_cur[qt] = _fadd(rho_cur[qt], rho_local)
                    r_cur[qt] = _fadd(r_cur[qt], r_local)

            # B-operand packs for GEMM2 (contract over kv): combine kvt=2*pks (k=0..3)
            # and 2*pks+1 (k=4..7) -> 8 kv values/lane matching _read_tr's kv ordering.
            c_pack = [[None] * QT for _ in range_constexpr(PV_K_STEPS)]
            p_pack = [[None] * QT for _ in range_constexpr(PV_K_STEPS)]
            for pks in range_constexpr(PV_K_STEPS):
                ka, kb = 2 * pks, 2 * pks + 1
                for qt in range_constexpr(QT):
                    c_pack[pks][qt] = bf16_trunc_pack_v8(C[ka][qt] + C[kb][qt])
                    p_pack[pks][qt] = bf16_trunc_pack_v8(P[ka][qt] + P[kb][qt])

            # GEMM2a A^T[D,q] += K_tr @ C ; GEMM2b B^T[D,q] += K_tr @ P. Both use the
            # same K transpose-read A-operand; depth-1 prefetch across dt.
            k_tr = [_read_tr(fx.Index(0), 0, pks) for pks in range_constexpr(PV_K_STEPS)]
            for dt in range_constexpr(DT):
                if const_expr(dt + 1 < DT):
                    k_tr_n = [_read_tr(fx.Index(0), dt + 1, pks) for pks in range_constexpr(PV_K_STEPS)]
                for pks in range_constexpr(PV_K_STEPS):
                    for qt in range_constexpr(QT):
                        A_cur[dt][qt] = mfma_acc(k_tr[pks], c_pack[pks][qt], A_cur[dt][qt])
                for pks in range_constexpr(PV_K_STEPS):
                    for qt in range_constexpr(QT):
                        B_cur[dt][qt] = mfma_acc(k_tr[pks], p_pack[pks][qt], B_cur[dt][qt])
                if const_expr(dt + 1 < DT):
                    k_tr = k_tr_n

            out = [A_cur[dt][qt] for dt in range_constexpr(DT) for qt in range_constexpr(QT)]
            out += [B_cur[dt][qt] for dt in range_constexpr(DT) for qt in range_constexpr(QT)]
            out += [rho_cur[qt] for qt in range_constexpr(QT)]
            out += [r_cur[qt] for qt in range_constexpr(QT)]
            return out

        # Split the causal kv-loop: [0, q_start) below the diagonal (no mask),
        # [q_start, kv_upper) straddles it (mask).
        _carry = A_accs + B_accs + rho_accs + r_accs
        loop_results = _carry
        for kv_start, inner in range(0, q_start, BLOCK_KV, init=_carry):
            loop_results = yield _kv_body(kv_start, inner, False)
        for kv_start, inner in range(q_start, kv_upper, BLOCK_KV, init=loop_results):
            loop_results = yield _kv_body(kv_start, inner, True)

        A_finals = [[loop_results[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]
        B_finals = [
            [loop_results[DT * QT + dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)
        ]
        rho_finals = [loop_results[2 * DT * QT + qt] for qt in range_constexpr(QT)]
        r_finals = [loop_results[2 * DT * QT + QT + qt] for qt in range_constexpr(QT)]

        # ---- Epilogue: reduce rho/R across the 4 K-subgroups (lane^16, lane^32) so
        # every lane holds the full per-q sum, then dQ = sm/R*(A - (rho/R)*B). The
        # 16x16 C-layout gives 4 CONTIGUOUS D/lane at q=qt*16+lane16 -> direct store. ----
        def _kg_allreduce(v):
            v = _fadd(v, fx.Float32(v).shuffle_xor(fx.Int32(16), width_i32))
            v = _fadd(v, fx.Float32(v).shuffle_xor(fx.Int32(32), width_i32))
            return v

        for qt in range_constexpr(QT):
            rho_f = _kg_allreduce(rho_finals[qt])
            r_f = _kg_allreduce(r_finals[qt])
            inv_r = rocdl.rcp(T.f32, _raw(r_f))
            dq_scale = fx.Float32(_fmul(fx.Float32(sm_scale), fx.Float32(inv_r)))
            rho_over_r = fx.Float32(_fmul(rho_f, inv_r))
            _q_row = q_row_of(qt)
            _store_mask = ArithValue(_q_row < seq_len_v)
            for dt in range_constexpr(DT):
                a_v = Vec(A_finals[dt][qt])
                b_v = Vec(B_finals[dt][qt])
                vals = [
                    fx.Float32(_fmul(dq_scale, _fsub(a_v[t], _fmul(rho_over_r, b_v[t]))))
                    for t in range_constexpr(4)
                ]
                lo = rocdl.cvt_pk_bf16_f32(_raw(vals[0]), _raw(vals[1]))
                hi = rocdl.cvt_pk_bf16_f32(_raw(vals[2]), _raw(vals[3]))
                o_pack = Vec.from_elements([fx.Int32(_raw(lo)), fx.Int32(_raw(hi))], fx.Int32)
                d_col = fx.Index(dt * D_TILE) + kg * fx.Index(4)
                g_idx = global_idx_q(_q_row, d_col)
                buffer_ops.buffer_store(
                    o_pack, dq_rsrc, g_idx * fx.Index(2), mask=_store_mask, offset_is_bytes=True
                )

    @flyc.jit
    def launch_flash_attn_bwd_dq(
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
        flash_attn_bwd_dq_kernel(
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
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_flash_attn_bwd_dq(*args, **kwargs)

    def _compile(*args):
        with CompilationContext.compile_hints(_hints):
            return flyc.compile(launch_flash_attn_bwd_dq, *args)

    _launch.compile = _compile
    return _launch
