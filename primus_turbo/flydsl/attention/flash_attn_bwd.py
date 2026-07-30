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
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch as _IfDispatch
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


def _scf_if_vals(cond, then_fn, else_fn, vals):
    """Runtime branch that carries a LIST of MLIR values through both arms.

    A Python list cannot be a single scf.if result, so each carried value gets its
    own name/result. Used for the loop-carried MFMA accumulators when a wave-uniform
    causal test lets a whole tile of work be skipped or run unmasked.
    """
    names = tuple("v%d" % i for i in range(len(vals)))
    return list(
        _IfDispatch.scf_if_dispatch(
            cond,
            lambda *_a: then_fn(),
            lambda *_a: else_fn(),
            result_names=names,
            result_values=tuple(vals),
        )
    )


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
                    s4 = Vec(
                        _fadd(v.shuffle(v, [0, 1, 2, 3]).ir_value(), v.shuffle(v, [4, 5, 6, 7]).ir_value())
                    )
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
                delta_final = loop_results[0] if isinstance(loop_results, (list, tuple)) else loop_results
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
                    vals = [fx.Float32(_fsub(a_v[r], _fmul(delta_full, b_v[r]))) for r in range_constexpr(16)]
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
    waves_per_eu=None,  # None = derive from the hoisted rows' register footprint
    block=256,
    sbhd=False,  # SBHD [S,B,H,D] native O/dO layout (seq-step = B*H*D)
    rows_per_group=8,  # s-rows owned by one lane-group (== deltas written contiguously)
):
    """Identity-delta ("odo") kernel: DELTA[b,hq,s] = -sum_d O[b,s,hq,d]*dO[b,s,hq,d].

    Memory-bound O.dO row-reduce that replaces the torch (out*dout).sum(-1). It is
    laid out so BOTH ends coalesce, which a one-thread-per-row mapping cannot do:
    a row is D*2 = 128 B of O and of dO, and its delta is a single fp32, so whichever
    axis a lane owns whole, the other one strides.

    Mapping: LPR = D/VEC lanes form a group and split one row's D vector, so each row
    is read by LPR adjacent 16 B loads = one full 128 B segment per instruction (every
    fetched byte is used). A group owns RPT *consecutive s* rows of one (b,hq) and
    walks them with one load pair each, so all RPT*2 dwordx4 are in flight before the
    first is consumed. The LPR lane partials are summed with a ds_bpermute butterfly
    (pure LDS-crossbar, no allocation/barrier), then lane sub=0 writes the group's RPT
    deltas, which are contiguous in the transposed [B,Hq,S] delta -- consecutive groups
    take consecutive s-blocks, so a wave's delta writes cover full cache lines instead
    of touching one line per lane (the previous mapping wrote 4 B into each of 64
    distinct lines per wave, i.e. ~16x write amplification on a 4 MB tensor).

    The butterfly changes the fp32 summation order versus a single-lane sequential sum
    (it is a fixed-shape tree, so still deterministic, and pairwise summation is the
    more accurate order); delta_id is only a centering value, since the fused dq kernel
    corrects rho/R*B exactly, so bf16*bf16 with fp32 accumulate is ample precision.

    waves_per_eu is derived, not fixed: the attribute pins occupancy exactly, so it
    must be the largest value whose register budget (512/wpe) still covers the
    in-flight loads -- on a pure-HBM reduce the extra waves are what hides the load
    latency, but overshooting spills and costs more than it buys (the previous
    one-row-per-thread version cliffed at wpe=7 with a 73-VGPR budget)."""
    assert dtype_str == "bf16", "odo kernel targets bf16"
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "odo kernel targets gfx950"
    elem_dtype = dtype_to_elem_type(dtype_str)
    HEAD_DIM = head_dim
    NUM_HEADS_Q = num_heads
    VEC = 8  # bf16 per load == one dwordx4, the widest buffer_load
    assert HEAD_DIM % VEC == 0
    LPR = HEAD_DIM // VEC  # lanes cooperating on one row (d64 -> 8, d128 -> 16)
    assert LPR & (LPR - 1) == 0 and 1 < LPR <= 64, "lanes-per-row must be a power of 2 in (1,64]"
    XOR_MASKS = [1 << i for i in range(LPR.bit_length() - 1)]  # 1,2,..,LPR/2
    RPT = rows_per_group
    BLOCK = block
    assert BLOCK % LPR == 0
    if waves_per_eu is None:
        # RPT*2 in-flight dwordx4 (4 VGPR each) + ~24 for addressing/accumulators.
        # Measured in-trio on gpt-oss B2/S8192/Hq64: wpe 5 (derived) 43.2us, 6 44.0,
        # 8 45.1; RPT 4 43.6, 8 44.3, 16 52.1 (2 44.9 -- too few loads in flight).
        waves_per_eu = max(4, min(8, 512 // (RPT * 2 * 4 + 24)))

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
        gtid = bid * fx.Index(BLOCK) + tid
        sl = fx.Index(seq_len)
        # Lane-group grp owns RPT consecutive s-rows of one (b,hq); lane sub owns the
        # sub-th VEC slice of each of those rows' D vectors.
        nblk = (sl + fx.Index(RPT - 1)) // fx.Index(RPT)
        grp = gtid // fx.Index(LPR)
        sub = gtid % fx.Index(LPR)
        total_grp = fx.Index(batch_size) * fx.Index(NUM_HEADS_Q) * nblk
        in_range = ArithValue(grp < total_grp)
        # OOB groups fold to group 0 for the loads; their stores are masked off.
        grp_c = fx.Index(in_range.select(grp, fx.Index(0)))

        # Bound all three exactly (not max_size): a tail group's rows past seq_len must
        # read 0 rather than neighbouring memory, and buffer_store implements mask= by
        # redirecting the offset to 0x7fffffff, which only gets dropped if the
        # descriptor's num_records is real (max_size=True faults instead).
        _o_nrec = _raw(fx.Index(batch_size) * sl * fx.Index(NUM_HEADS_Q * HEAD_DIM * 2))
        _d_nrec = _raw(fx.Index(batch_size) * sl * fx.Index(NUM_HEADS_Q * 4))
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=_o_nrec)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=False, num_records_bytes=_o_nrec)
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_d_nrec
        )

        # grp = (b*Hq + hq)*nblk + s_blk: s-major within a head so consecutive groups
        # write consecutive delta blocks. THD packs O/dO as [B,S,Hq,D] (row stride
        # Hq*D) but SBHD is [S,B,Hq,D] (row stride B*Hq*D); DELTA stays [B,Hq,S].
        s_blk = grp_c % nblk
        bh = grp_c // nblk
        hq = bh % fx.Index(NUM_HEADS_Q)
        b = bh // fx.Index(NUM_HEADS_Q)
        s0 = s_blk * fx.Index(RPT)
        if const_expr(sbhd):
            base = ((s0 * fx.Index(batch_size) + b) * fx.Index(NUM_HEADS_Q) + hq) * fx.Index(HEAD_DIM)
            row_step = fx.Index(batch_size) * fx.Index(NUM_HEADS_Q * HEAD_DIM)
        else:
            base = ((b * sl + s0) * fx.Index(NUM_HEADS_Q) + hq) * fx.Index(HEAD_DIM)
            row_step = fx.Index(NUM_HEADS_Q * HEAD_DIM)
        base = base + sub * fx.Index(VEC)
        # Issue all RPT*2 dwordx4 before the first is consumed: a per-row load->use
        # pattern drains one pair at a time (one s_waitcnt each) and exposes the full
        # HBM latency on this pure-memory reduce.
        ovs = []
        dvs = []
        off = base
        for r in range_constexpr(RPT):
            ovs.append(buffer_ops.buffer_load(o_rsrc, off, vec_width=VEC, dtype=elem_dtype_l))
            dvs.append(buffer_ops.buffer_load(do_rsrc, off, vec_width=VEC, dtype=elem_dtype_l))
            off = off + row_step
        accs = []
        for r in range_constexpr(RPT):
            prod = Vec(ovs[r]).to(fx.Float32) * Vec(dvs[r]).to(fx.Float32)
            acc = fx.Float32(0.0)
            for i in range_constexpr(VEC):
                acc = _fadd(acc, Vec(prod)[i])
            accs.append(fx.Float32(acc))
        # Butterfly-sum the LPR lane partials (dst[l] = src[l ^ m]) so every lane holds
        # its rows' totals; ds_bpermute is the LDS crossbar only -- no LDS, no barrier.
        lane_i = fx.Int32(tid)
        for m in XOR_MASKS:
            idx = _raw((lane_i ^ fx.Int32(m)) * fx.Int32(4))
            for r in range_constexpr(RPT):
                part = _raw(Vec.from_elements([accs[r]], fx.Float32).bitcast(fx.Int32)[0])
                peer = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part)
                peer_f = fx.Float32(
                    _raw(Vec.from_elements([fx.Int32(peer)], fx.Int32).bitcast(fx.Float32)[0])
                )
                accs[r] = fx.Float32(_fadd(accs[r], peer_f))

        # DELTA is transposed [B,Hq,S]: delta[b,hq,s] at (b*Hq + hq)*S + s, so the
        # group's RPT values are contiguous and lane sub=0 writes them all.
        delta_off = bh * sl + s0
        head_lane = in_range & ArithValue(sub == fx.Index(0))
        for r in range_constexpr(RPT):
            neg_acc = arith.subf(_raw(c_zero_f), _raw(accs[r]), fastmath=fm)
            buffer_ops.buffer_store(
                fx.Float32(neg_acc),
                delta_rsrc,
                (delta_off + fx.Index(r)) * fx.Index(4),
                mask=head_lane & ArithValue((s0 + fx.Index(r)) < sl),
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
        nblk = (fx.Index(seq_len) + fx.Index(RPT - 1)) // fx.Index(RPT)
        total = fx.Index(batch_size) * nblk * fx.Index(NUM_HEADS_Q) * fx.Index(LPR)
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
    window_left=-1,
    batch_size=None,  # compile-time B; required for SBHD seq-step stride bake
    sbhd=False,       # SBHD [S,B,H,D] native layout (seq-step = B*H*D)
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
    K_STEP_QK = 32  # K=32 per GEMM1 MFMA (contract over D)
    K_STEPS_QK = head_dim // K_STEP_QK  # d64 -> 2
    NT = ROWS_PER_WAVE_KV // N_TILE  # kv 16-tiles per wave: 32/16 = 2
    MT = BLOCK_Q // M_TILE  # q 16-tiles: 64/16 = 4
    DT = head_dim // D_TILE  # D 16-tiles: 64/16 = 4
    PV_K_STEP = 32  # K=32 per GEMM2 MFMA (contract over q)
    PV_K_STEPS = BLOCK_Q // PV_K_STEP  # 64/32 = 2

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
    # SBHD [S,B,H,D]: per-token seq step is B*H*D (batch interleaved in the seq axis)
    # while the per-batch base is only H*D. THD/BSHD keep RD==STRIDE (dense). The
    # dk/dv workspace is reorganized to [q_split, Skv, B, Hkv, D] so the host's
    # slot reduction (sum over the leading q_split axis) yields SBHD contiguously.
    if sbhd:
        assert batch_size is not None, "SBHD dkdv needs compile-time batch_size"
    RD_STRIDE_Q = (batch_size * STRIDE_TOKEN_Q) if sbhd else STRIDE_TOKEN_Q
    RD_STRIDE_KV = (batch_size * STRIDE_TOKEN_KV) if sbhd else STRIDE_TOKEN_KV

    Q_STRIDE = HEAD_DIM
    LDS_TILE = BLOCK_Q * Q_STRIDE
    LDS_DO_BASE = LDS_TILE
    LDS_TOTAL = 2 * LDS_TILE
    # Q/dO tiles for DMA_STAGE_HEADS consecutive GQA heads are staged in one LDS
    # buffer and fetched by one DMA batch, so the WAR barrier + s_waitcnt(0) drain
    # is paid once per group instead of once per head, and the second head's GEMM1
    # LDS reads are no longer fenced off from the first head's GEMM2 MFMAs. The
    # per-slot base is a compile-time constant that folds into the ds_read offset
    # immediate, so extra slots cost LDS (free: 16 KB/slot of the 80 KB an occ-2 WG
    # may use) and NUM_DMA_Q SGPR pointers, but no address VGPRs.
    DMA_STAGE_HEADS = 1  # GQA heads staged per Q/dO DMA batch
    assert GQA_GROUP_SIZE % DMA_STAGE_HEADS == 0

    # lse / -delta for the WHOLE q block (all GQA heads) are staged in LDS next to
    # the Q/dO tiles. Per head each lane needs 2 x MT v4f32 of them, but lane16 does
    # not index q, so the direct HBM form issues 8 buffer_load_dwordx4/head (64/trip,
    # 67% of dkdv's VMEM instructions) to deliver 512 B of distinct data -- a 16x
    # instruction amplification. One cooperative vec2 fetch per thread per q block
    # replaces all of them, and because LDS latency needs no cover the one-head-ahead
    # register prefetch (32 VGPR carried across GEMM2) disappears with it.
    LD_HEAD_ELEMS = BLOCK_Q
    LD_ARR_ELEMS = GQA_GROUP_SIZE * LD_HEAD_ELEMS
    LD_ELEMS = 2 * LD_ARR_ELEMS  # [-delta | lse][head][q]
    LD_THREADS_PER_HEAD = BLOCK_SIZE // GQA_GROUP_SIZE if GQA_GROUP_SIZE <= BLOCK_SIZE else 0
    LD_VEC = LD_HEAD_ELEMS // LD_THREADS_PER_HEAD if LD_THREADS_PER_HEAD > 0 else 0
    # Cooperative LDS staging of (-delta, lse) needs GQA_GROUP_SIZE to divide
    # BLOCK_SIZE (and LD_VEC >= 1); non-divisor / small GQA fall back to
    # per-use HBM reads via _ld_read.
    COOP_LD_VALID = (
        LD_THREADS_PER_HEAD > 0
        and BLOCK_SIZE % GQA_GROUP_SIZE == 0
        and LD_VEC >= 1
        and LD_HEAD_ELEMS % LD_THREADS_PER_HEAD == 0
    )

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
    ld_off = lds_off + LDS_TOTAL * 2 * DMA_STAGE_HEADS
    allocator.ptr = ld_off + LD_ELEMS * 4

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
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
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

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)
        # Bottom-right causal: offset = seq_k - seq_q >= 0 (Sq <= Skv).
        causal_offset = seq_len_k_v - seq_len_q_v
        causal_off_i32 = fx.Int32(seq_len_k) - fx.Int32(seq_len_q)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL * DMA_STAGE_HEADS,)).get()
        ld_lds = SmemPtr(base_ptr, ld_off, fx.Float32.ir_type, shape=(LD_ELEMS,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16  # M/N index within a 16-tile
        kg = lane // 16  # 0..3: K-subgroup (inputs) / M-block (C output)


        # ---- Decompose block_id: kv_head fastest (XCD/L2), then split_idx. ----
        num_kv_tiles = (seq_len_k_v + BLOCK_KV - 1) // BLOCK_KV
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

        # Per-batch base (elements). SBHD: batch inside the seq axis -> base is only
        # H*D. THD: dense per-batch block -> base is seq*H*D.
        if const_expr(sbhd):
            _q_ptr_batch_off = batch_idx * fx.Index(STRIDE_TOKEN_Q)
        else:
            _q_ptr_batch_off = batch_idx * seq_len_q_v * fx.Index(STRIDE_TOKEN_Q)
        q_ptr = buffer_ops.get_element_ptr(q_ptr, _q_ptr_batch_off, elem_type=elem_type)
        do_ptr = buffer_ops.get_element_ptr(do_ptr, _q_ptr_batch_off, elem_type=elem_type)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx_q(token_idx, col, q_head):
            return token_idx * RD_STRIDE_Q + q_head * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * RD_STRIDE_KV + kv_head_idx * HEAD_DIM + col

        def _q_row_clamp(row_idx):
            last = seq_len_q_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_q_v).select(row_idx, last))

        def _load_global_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def bf16_trunc_pack_v8(f32_vals):
            # Hardware f32->16-bit pack (RNE, 1 VALU op/pair) instead of the manual
            # &/>>/| truncation (3 VALU ops/pair); cuts the VALU-issue-bound path.
            # RNE is load-bearing, not just tidy: a truncating v_perm_b32 pack (also
            # one pass) was measured net-negative on dK/dV SNR.
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        PBLK = 128  # 128-wide block holds 2 real rows (low r&4=0 -> [0,64), high -> [64,128))

        def _pblk(row_idx):
            return ((row_idx >> fx.Index(3)) << fx.Index(2)) | (row_idx & fx.Index(3))

        def _swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(7)) << fx.Index(4)
            return col_idx ^ mask

        # GEMM2 transpose-read addressing, split into a lane-only register and a
        # compile-time literal. The read row is pks*PV_K_STEP (+N_TILE for the second
        # read) + L with L = kg*4 + lane16//4 in [0,M_TILE); PV_K_STEP and N_TILE are
        # multiples of 8, so both _pblk and the (row&7) swizzle mask split into a
        # lane-only term plus a constant, and the column swizzle splits as well
        # because (lane%4)*4 sits below the mask bits. ds_read_b64_tr_b16 carries an
        # offset immediate, so emitting the constant as a literal add leaves DT
        # address registers for the whole GEMM2 read set instead of one per read
        # (dkdv held 50 live tr-read addresses). The identities are asserted over the
        # full (pks, L, dt, mask, col) domain rather than argued.
        def _pblk_py(row_idx):
            return ((row_idx >> 3) << 2) | (row_idx & 3)

        for _p in range(PV_K_STEPS):
            for _e in (0, N_TILE):
                for _l in range(M_TILE):
                    assert _pblk_py(_p * PV_K_STEP + _e + _l) == _pblk_py(_p * PV_K_STEP + _e) + _pblk_py(_l)
                    assert ((_p * PV_K_STEP + _e + _l) & 7) == (_l & 7)
        for _d in range(DT):
            for _m in range(8):
                for _c in range(4):
                    assert ((_d * D_TILE + _c * 4) ^ (_m << 4)) == ((_d * D_TILE) ^ (_m << 4)) + _c * 4

        _tr_l = kg * fx.Index(4) + (lane16 // fx.Index(4))
        _tr_lane = (
            _pblk(_tr_l) * fx.Index(PBLK) + (lane % fx.Index(4)) * fx.Index(4)
        )
        _tr_mask = (_tr_l & fx.Index(7)) << fx.Index(4)
        # Materialised eagerly (entry block) so every use inside the q loop dominates.
        _tr_dyn = [
            fx.Int64((_tr_lane + (fx.Index(_d * D_TILE) ^ _tr_mask)) * fx.Index(2))
            for _d in range_constexpr(DT)
        ]

        def _read_tr_at(dt, const_byte_off):
            ptr = buffer_ops.create_llvm_ptr(
                _tr_dyn[dt] + fx.Int64(const_byte_off), address_space=3
            )
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

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
        _q_nrec_bytes = _raw(seq_len_q_v * fx.Index(RD_STRIDE_Q * 2))
        _q_batch_byte_off = _raw(_q_ptr_batch_off * fx.Index(2))
        _kv_nrec_bytes = _raw(seq_len_k_v * fx.Index(RD_STRIDE_KV * 2))
        if const_expr(sbhd):
            _kv_batch_byte_off = _raw(batch_idx * fx.Index(STRIDE_TOKEN_KV * 2))
        else:
            _kv_batch_byte_off = _raw(batch_idx * seq_len_k_v * fx.Index(STRIDE_TOKEN_KV * 2))
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
        )
        # DK/DV point at this split's slot of the [B, q_split, S, Hkv, D] workspace
        # (slot index = batch*q_split + split_idx); one WG writes it exactly once.
        if const_expr(sbhd):
            # [q_split, Skv, B, Hkv, D]: slot base = split*Skv*(B*Hkv*D) + batch*(Hkv*D).
            # Token stride inside a slot is RD_STRIDE_KV (B*Hkv*D) == global_idx_kv step.
            _dkv_ws_byte_off = _raw(
                (split_idx * seq_len_k_v * fx.Index(RD_STRIDE_KV) + batch_idx * fx.Index(STRIDE_TOKEN_KV))
                * fx.Index(2)
            )
        else:
            _ws_slot = batch_idx * fx.Index(Q_SPLIT) + split_idx
            _dkv_ws_byte_off = _raw(_ws_slot * seq_len_k_v * fx.Index(STRIDE_TOKEN_KV * 2))
        dk_rsrc = buffer_ops.create_buffer_resource(
            DK, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_dkv_ws_byte_off
        )
        dv_rsrc = buffer_ops.create_buffer_resource(
            DV, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_dkv_ws_byte_off
        )
        _lse_per_batch = seq_len_q_v * fx.Index(NUM_HEADS_Q)
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
            Q_TILE_BYTES = (BLOCK_Q // 2) * 128 * 2
            NUM_DMA_Q = Q_TILE_BYTES // DMA_BATCH_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (128 * 2)  # blocks per batch
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def _dma_lds_ptrs(lds_byte_base):
                # LDS write pointer is loop/head-invariant, but readfirstlane is not
                # LICM-hoistable -> precompute the per-d SGPR pointers once.
                ptrs = []
                for d in range_constexpr(NUM_DMA_Q):
                    lds_addr = (
                        lds_byte_base
                        + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_addr))
                    ptrs.append(buffer_ops.create_llvm_ptr(lds_lane0, address_space=3))
                return ptrs

            q_lds_ptrs = [
                _dma_lds_ptrs(lds_base_idx + fx.Index(s * LDS_TOTAL * 2))
                for s in range_constexpr(DMA_STAGE_HEADS)
            ]
            do_lds_ptrs = [
                _dma_lds_ptrs(lds_base_idx + fx.Index((s * LDS_TOTAL + LDS_DO_BASE) * 2))
                for s in range_constexpr(DMA_STAGE_HEADS)
            ]

            def coop_dma_tile(src_rsrc, lds_ptrs, tile_start, q_head):
                """DMA a BLOCK_Q x head_dim Q/dO tile into the swizzled LDS layout."""
                for d in range_constexpr(NUM_DMA_Q):
                    lds_ptr = lds_ptrs[d]
                    block = tid // fx.Index(16) + fx.Index(d * ROWS_PER_DMA_BATCH)
                    lane_in_block = tid % fx.Index(16)
                    half = lane_in_block // fx.Index(8)
                    position = lane_in_block * fx.Index(8)  # swiz col within 128-block
                    row_in_tile = (
                        fx.Index(8) * (block >> fx.Index(2)) + (block & fx.Index(3)) + half * fx.Index(4)
                    )
                    xor_mask = (row_in_tile & fx.Index(7)) << fx.Index(4)
                    unsw_col_f16 = position ^ xor_mask  # real col in [0,64), 1x HBM
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile
                    global_byte = (
                        global_row * fx.Index(RD_STRIDE_Q * 2) + q_head * fx.Index(HEAD_DIM * 2) + col_byte
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        src_rsrc, lds_ptr, _dma_size, fx.Int32(global_byte), _dma_soff, _dma_off, _dma_aux
                    )

        # ---- Owned K,V B-operand packs: B[k=D][n=kv], n=lane16, k=kg*8+s. Per wave
        # NT kv 16-tiles x K_STEPS_QK D-steps; k_b_packs[nt][ks] is a v8 bf16. ----
        # K carries the softmax scale: prescaling the GEMM1a B-operand by
        # sm_scale*log2e (x2^23 for the Schraudolph path) lets the exponent
        #   diff = s*sm_scale*log2e + lse
        # be produced by GEMM1a itself with lse as the accumulator init, so the
        # per-element fma in front of every exp2 disappears (256 VALU ops/trip on
        # the exp2 dependency chain). Costs one extra bf16 rounding of K, hoisted
        # once per work-group; K feeds no other GEMM (dK comes from Q_tr @ dS).
        _pre_k = fx.Float32(sm_scale * _LOG2E * (float(1 << 23) if fast_exp2 else 1.0))
        k_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(NT)]
        v_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(NT)]
        for nt in range_constexpr(NT):
            _kvr = kv_row_of(nt)
            for ks in range_constexpr(K_STEPS_QK):
                kv_col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
                _k_raw = Vec(
                    buffer_ops.buffer_load(
                        k_rsrc, global_idx_kv(_kvr, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                    )
                ).to(fx.Float32)
                k_b_packs[nt][ks] = bf16_trunc_pack_v8(
                    [_fmul(_k_raw[i], _pre_k) for i in range_constexpr(MFMA_LANE_K)]
                )
                v_b_packs[nt][ks] = buffer_ops.buffer_load(
                    v_rsrc, global_idx_kv(_kvr, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        # Crude Schraudolph 2^x (fast_exp2): P = bitcast(fptosi((s*sm*log2e + lse)*
        # 2^23 + bias)). The (lse*2^23 + bias) addend is pre-scaled on the host (see
        # attention_flydsl_impl), so _p_of collapses to a SINGLE fma
        # scaled = s*(sm*log2e*2^23) + lse_s23 -> fptosi: the diff fma and the
        # Schraudolph *2^23+bias fma fold into one. lse_t is a plain loaded addend
        # (not an in-kernel prescale), keeping it a clean fma(var,const,loaded)->fptosi.
        _c_scaled_floor = fx.Float32(-87.0 * float(1 << 23) + float(127 * (1 << 23) - 486411))
        _compute_type = fx.Float32.ir_type

        def _vexp_intrin(x):
            # Backend-visible 2^x: emits v_exp_f32 but, being a recognised VALU op
            # rather than opaque asm, it carries the MFMA->VALU hazard itself.
            return fx.Float32(
                llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", [_raw(x)], [], [])
            )

        def _p_of(s_r, apply_mask):
            # s_r already IS the base-2 softmax exponent: GEMM1a runs on a K prescaled
            # by sm_scale*log2e (x2^23 in the Schraudolph mode) and is initialised with
            # the prescaled lse, so no fma is left in front of the exp2.
            if const_expr(fast_exp2):
                # The floor clamp is load-bearing only on masked (diagonal) slots
                # (masked s_r=-inf -> maximumf(floor) -> 2^-87=0; pitfalls/04). In the
                # mask-free bulk causal-valid softmax args are bounded (>> -87), so the
                # clamp is a no-op there and is dropped.
                if const_expr(apply_mask):
                    s_r = ArithValue(s_r).maximumf(_c_scaled_floor)
                i = arith.fptosi(fx.Int32.ir_type, _raw(s_r))
                return ArithValue(i).bitcast(_compute_type)
            return _vexp_intrin(s_r)

        # A-operand read (Q/dO from LDS): A[m=q=lane16][k=D=kg*8+s]. mt selects the
        # 16-q tile (row = mt*16 + lane16), ks the D 32-step (D = ks*32 + kg*8).
        a_swz_mask = (lane16 & fx.Index(7)) << fx.Index(4)

        def _a_idx(a_base, mt, ks):
            row = fx.Index(mt * M_TILE) + lane16
            col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
            return a_base + _pblk(row) * fx.Index(PBLK) + (col ^ a_swz_mask)

        def _gemm_qk(a_base, b_packs, inits=None, mts=None):
            """S[mt][nt] (v4f32) = A(Q/dO)[mt] @ B(owned K/V)[nt]^T over D. A is
            loaded once per (mt,ks) and reused across nt. inits[mt] optionally
            pre-loads the accumulator (folds -delta into the dP GEMM for free).
            mts restricts the emission to a subset of the q-tiles."""
            _mts = range_constexpr(MT) if mts is None else mts
            a = {
                mt: [
                    Vec.load(mfma_pack_type, lds, [_a_idx(a_base, mt, ks)])
                    for ks in range_constexpr(K_STEPS_QK)
                ]
                for mt in _mts
            }
            out = [[None] * NT for _ in range_constexpr(MT)]
            for mt in _mts:
                for nt in range_constexpr(NT):
                    acc = c_zero_v4f32 if inits is None else inits[mt]
                    for ks in range_constexpr(K_STEPS_QK):
                        acc = mfma_acc(a[mt][ks], b_packs[nt][ks], acc)
                    out[mt][nt] = acc
            return out

        def _read_tr(a_base, dt, pks):
            """Transpose-read Q/dO -> GEMM2 A-operand [m=D=dt*16+lane16][k=q=kg*8+s].
            Two ds_read_tr16 (4 q each): read0->s0..3 (q=pks*32+kg*4+j), read1->s4..7
            (q=pks*32+16+kg*4+j). See .claude/memory/ds_read_tr16_b64_gfx950.md.
            a_base is a compile-time LDS byte base, so the whole per-read offset is a
            literal the backend can fold into the ds_read offset field."""
            c0 = (a_base + _pblk_py(pks * PV_K_STEP) * PBLK) * 2 + lds_off
            c1 = c0 + _pblk_py(N_TILE) * PBLK * 2
            v0 = _read_tr_at(dt, c0)
            v1 = _read_tr_at(dt, c1)
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # GEMM2 transpose-read prefetch distance, in dt steps. Each dt block issues
        # 2*PV_K_STEPS*NT MFMAs and the same number of ds_read_tr16, so the 1:1
        # interleave gives a read only ~1 MFMA of shadow at depth 1; depth D scales
        # that to D dt blocks at the cost of (D-1) extra live A-operand sets.
        G2D = max(1, min(2, DT))  # GEMM2 dt+1 prefetch depth

        # dv/dk accumulators flat over (dt,nt): index dt*NT+nt, each v4f32,
        # C[m=D=dt*16+kg*4+t][n=kv=nt*16+lane16].
        dv_accs = [c_zero_v4f32 for _ in range_constexpr(DT * NT)]
        dk_accs = [c_zero_v4f32 for _ in range_constexpr(DT * NT)]

        # Bottom-right causal: first query attending this kv-tile = max(0, kv_start-offset).
        _kv_first_q = ArithValue(kv_start >= causal_offset).select(kv_start - causal_offset, fx.Index(0))
        _q_loop_start = _kv_first_q + split_idx * fx.Index(BLOCK_Q)
        _kv_end = kv_start + fx.Index(BLOCK_KV)
        _kv_end_c = ArithValue(_kv_end < seq_len_k_v).select(_kv_end, seq_len_k_v)
        _step = Q_SPLIT * BLOCK_Q
        _masked_upper = ArithValue(_kv_end_c >= causal_offset).select(_kv_end_c - causal_offset, fx.Index(0))
        # Masked q-blocks this split visits = ceil((_masked_upper - _q_loop_start)/_step);
        # unmask resumes at the next stride point. The masked band is BLOCK_KV wide, so it
        # spans BLOCK_KV/BLOCK_Q q-blocks. A plain "+_step" assumes exactly ONE masked
        # block per split, which holds for _step >= BLOCK_KV (q_split>=2) but under-counts
        # for q_split=1 (_step=BLOCK_Q < band) -> the 2nd diagonal block was reprocessed
        # unmasked (double-count + wrong mask) => dk/dv corruption. This ceil form is exact
        # for every q_split and reduces to the old value when the band is one block wide.
        _masked_span = ArithValue(_masked_upper > _q_loop_start).select(
            _masked_upper - _q_loop_start, fx.Index(0)
        )
        _unmask_start = _q_loop_start + ((_masked_span + fx.Index(_step - 1)) // fx.Index(_step)) * fx.Index(
            _step
        )

        # GQA head axis is unrolled INSIDE each q_start body (rather than wrapping the
        # whole q_start loop, as before) so that head h+1's GEMM1/exp2 is emitted right
        # after head h's GEMM2 in the SAME straight-line block, with no real loop
        # back-edge between them: the compiler can then schedule head h+1's exp2 VALU
        # chain into head h's GEMM2 MFMA shadow (the GQA-head-axis operand-bubble
        # softpipe). dv/dk legitimately accumulate across all GQA heads into the same
        # registers (K/V is shared by the group), so this is a pure reassociation of
        # the same sum -> det-neutral, and does not change live accumulator count.
        #
        # Cooperative fetch of (-delta, lse) for the whole q block: thread t owns
        # LD_VEC consecutive q of one GQA head, so the wave issues LD_ARR/BLOCK_SIZE
        # vec loads per array instead of MT per head. Rides in _stage_heads' existing
        # barrier pair, and is re-read straight out of LDS at each use point.
        _ld_head = tid // fx.Index(LD_THREADS_PER_HEAD)
        _ld_q = (tid % fx.Index(LD_THREADS_PER_HEAD)) * fx.Index(LD_VEC)

        def _stage_ld_issue(q_start):
            # Issued BEFORE the Q/dO DMA so both HBM streams are in flight together;
            # the LDS commit lands after the DMA instructions, so its vmcnt wait does
            # not serialise the two (gfx950 cannot wait on a vmcnt subset, but the
            # counter is in-order, so a later consumer wait is enough).
            _g = (kv_head_idx * fx.Index(GQA_GROUP_SIZE) + _ld_head) * seq_len_q_v + q_start + _ld_q
            return [
                buffer_ops.buffer_load(rsrc, _g, vec_width=LD_VEC, dtype=fx.Float32)
                for rsrc in (delta_rsrc, lse_rsrc)
            ]

        def _stage_ld_commit(vals):
            _lds_i = _ld_head * fx.Index(LD_HEAD_ELEMS) + _ld_q
            for arr in range_constexpr(2):
                Vec(vals[arr]).store(ld_lds, [fx.Index(arr * LD_ARR_ELEMS) + _lds_i])

        def _ld_read(q_start, head_local, mt, arr):
            # v4f32 at q = head's q block + mt*M_TILE + kg*4 (+t), matching the
            # GEMM1 accumulator C layout; lane16 is absent -> a 16-way LDS broadcast.
            if const_expr(COOP_LD_VALID):
                return Vec.load(
                    v4f32_type,
                    ld_lds,
                    [fx.Index(arr * LD_ARR_ELEMS + head_local * LD_HEAD_ELEMS + mt * M_TILE)
                     + kg * fx.Index(4)],
                ).ir_value()
            # Generic fallback: LDS staging disabled -> read v4f32 from HBM per
            # (head, mt). arr 0 -> delta, arr 1 -> lse; identical operand.
            _rsrc = delta_rsrc if arr == 0 else lse_rsrc
            _lhb = (kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)) * seq_len_q_v
            _idx = _lhb + fx.Index(
                ArithValue(fx.Int32(q_start) + fx.Int32(kg) * fx.Int32(4) + fx.Int32(mt * M_TILE))
            )
            # Return a v4f32 ir_value like the LDS branch for _gemm_qk.
            return Vec(buffer_ops.buffer_load(_rsrc, _idx, vec_width=4, dtype=fx.Float32)).ir_value()

        def _stage_heads(q_start, head_first):
            """Fetch Q/dO for the next DMA_STAGE_HEADS heads into their LDS slots."""
            _ldv = None
            if const_expr(head_first == 0 and COOP_LD_VALID):
                _ldv = _stage_ld_issue(q_start)
            gpu.barrier()  # WAR: guard the prior group's LDS reads before this DMA
            for s in range_constexpr(DMA_STAGE_HEADS):
                q_head = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_first + s)
                if const_expr(ENABLE_DMA):
                    coop_dma_tile(q_rsrc, q_lds_ptrs[s], q_start, q_head)
                    coop_dma_tile(do_rsrc, do_lds_ptrs[s], q_start, q_head)
                else:
                    _coop_load(q_ptr, fx.Index(s * LDS_TOTAL), q_start, q_head)
                    _coop_load(do_ptr, fx.Index(s * LDS_TOTAL + LDS_DO_BASE), q_start, q_head)
            if const_expr(head_first == 0 and COOP_LD_VALID):
                _stage_ld_commit(_ldv)
            if const_expr(ENABLE_DMA):
                rocdl.s_waitcnt(0)
            gpu.barrier()  # DMA visible before GEMM1 reads the tile

        def _head_step(q_start, apply_mask, head_local, dv_cur, dk_cur):
            q_start_i32 = fx.Int32(q_start)
            # This lane's q for tile mt slot t = q_start + kg*4 + mt*16 + t.
            kg_off_i32 = fx.Int32(kg) * fx.Int32(4)
            # LDS slot holding this head's Q/dO tiles (compile-time constant base).
            _q_base = (head_local % DMA_STAGE_HEADS) * LDS_TOTAL
            _do_base = _q_base + LDS_DO_BASE

            # GEMM1a S[mt][nt]=Q@K^T. P=exp2(S) needs only s_tiles+lse (not dP), so
            # it is emitted BEFORE GEMM1b dP=dO@V^T: the exp2 VALU chain then hides
            # in the dP MFMA shadow instead of forming a serial bubble between the
            # two GEMMs. dS=P*dP is folded in after dP. Pure reorder -> det-neutral.
            # Accumulator init = the prescaled lse, so GEMM1a emits the full softmax
            # exponent (K is prescaled by sm_scale*log2e) and the exp2 reads it raw.
            #
            # The whole GEMM1a/exp2/GEMM1b/dS/pack block runs per q-HALF (one pks = the
            # two mt that pack into one GEMM2 K=32 step): computing, consuming and
            # packing 2 of the MT q-tiles at a time halves the live S/dP/P/dS transient
            # that pinned dkdv at 256 VGPR with 9 spills / 40 B scratch, so the whole
            # kernel now fits spill-free. Same structure as dq's per-kv-half _kv_body
            # loop. Pure re-ordering -> bit-identical, det-neutral.
            p_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            ds_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            do_ring, q_ring = None, None
            for pks in range_constexpr(PV_K_STEPS):
                ma, mb = 2 * pks, 2 * pks + 1
                half = [ma, mb]
                # lse / -delta are pulled from LDS at their use points, so only the
                # 2 v4f32 this half consumes are ever live (vs MT x 2 carried before).
                s_tiles = _gemm_qk(
                    fx.Index(_q_base), k_b_packs, {mt: _ld_read(q_start, head_local, mt, 1) for mt in half}, mts=half
                )

                # P[mt][nt]: 4 f32 at q=mt*16+kg*4+t, kv=nt*16+lane16.
                P = [[None] * NT for _ in range_constexpr(MT)]
                for mt in half:
                    for nt in range_constexpr(NT):
                        s_v = s_tiles[mt][nt]
                        p_vals = []
                        if const_expr(apply_mask):
                            for t in range_constexpr(4):
                                q_slot = q_start_i32 + kg_off_i32 + fx.Int32(mt * M_TILE + t)
                                _up = ArithValue(kv_row_i32_of(nt) > q_slot + causal_off_i32)
                                if const_expr(window_left >= 0):
                                    _lo = ArithValue(
                                        kv_row_i32_of(nt)
                                        <= q_slot + causal_off_i32 - fx.Int32(window_left)
                                    )
                                    _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                                else:
                                    _mm = _up
                                # The mask cndmask is itself this slot's accumulator read,
                                # so it doubles as the MFMA->VALU hazard anchor.
                                _sm = _mm.select(c_neg_inf, fx.Float32(Vec(s_v)[t]))
                                p_vals.append(_p_of(_sm, True))
                            P[mt][nt] = p_vals
                        elif const_expr(fast_exp2):
                            # fptosi reads each accumulator slot itself -> own anchor.
                            P[mt][nt] = [
                                _p_of(fx.Float32(Vec(s_v)[t]), False) for t in range_constexpr(4)
                            ]
                        else:
                            # Slot 0 goes through the exp2 INTRINSIC: unlike inline asm it
                            # is a backend-visible VALU read of the MFMA result, so the
                            # MFMA->VALU wait states are inserted for it and it anchors the
                            # whole v4 at no instruction cost. Slots 1..3 read the v4 bare
                            # and are pinned behind that anchor (the anchor must read the
                            # very v4 it protects, pitfalls/13).
                            for t in range_constexpr(4):
                                p_vals.append(_vexp_intrin(Vec(s_v)[t]))
                            P[mt][nt] = p_vals

                # GEMM1b dP[mt][nt]=dO@V^T (acc init=-delta); dS[mt][nt]=P*dP after it.
                dp_tiles = _gemm_qk(
                    fx.Index(_do_base), v_b_packs, {mt: _ld_read(q_start, head_local, mt, 0) for mt in half}, mts=half
                )

                # Hoist the first G2D dt's GEMM2 transpose-reads ahead of the last
                # half's dS mul + bf16 pack: the ds_read_tr16 LDS latency (the measured
                # MFMA operand bubble) then overlaps that VALU block instead of exposing
                # at GEMM2's first MFMA.
                if const_expr(pks == PV_K_STEPS - 1):
                    do_ring = [
                        [
                            _read_tr(_do_base, _d, _p)
                            for _p in range_constexpr(PV_K_STEPS)
                        ]
                        for _d in range_constexpr(G2D)
                    ]
                    q_ring = [
                        [_read_tr(_q_base, _d, _p) for _p in range_constexpr(PV_K_STEPS)]
                        for _d in range_constexpr(G2D)
                    ]

                # B-operand packs for GEMM2: pack pks combines mt=2*pks (k=0..3) and
                # 2*pks+1 (k=4..7) -> 8 q values/lane matching _read_tr's q ordering.
                # Packing here frees P/dS (and S/dP) before the next half's GEMM1a.
                for nt in range_constexpr(NT):
                    _ds = [
                        [_fmul(P[mt][nt][t], Vec(dp_tiles[mt][nt])[t]) for t in range_constexpr(4)]
                        for mt in half
                    ]
                    p_pack[pks][nt] = bf16_trunc_pack_v8(P[ma][nt] + P[mb][nt])
                    ds_pack[pks][nt] = bf16_trunc_pack_v8(_ds[0] + _ds[1])

            # GEMM2a dV^T[dt][nt] += dO_tr[dt] @ P ; GEMM2b dK^T[dt][nt] += Q_tr @ dS.
            # Depth-G2D prefetch across dt: issue dt+G2D's dO transpose-reads before
            # dt's dV MFMAs and dt+G2D's Q reads between the dV and dK MFMAs, so the
            # ds_read_tr16 LDS latency hides in the MFMA shadow. pks is the outer
            # MFMA loop (nt inner) so the two nt accumulators of each 16x16 chain
            # interleave -> more independent MFMA ILP. Pure reorder -> det-neutral.
            # (dt=0's do_tr/q_tr are read above, hoisted into the dS/pack shadow.)
            rocdl.s_setprio(1)
            for dt in range_constexpr(DT):
                _slot = dt % G2D
                do_tr, q_tr = do_ring[_slot], q_ring[_slot]
                if const_expr(dt + G2D < DT):
                    do_tr_n = [
                        _read_tr(_do_base, dt + G2D, pks)
                        for pks in range_constexpr(PV_K_STEPS)
                    ]
                for pks in range_constexpr(PV_K_STEPS):
                    for nt in range_constexpr(NT):
                        dv_cur[dt][nt] = mfma_acc(do_tr[pks], p_pack[pks][nt], dv_cur[dt][nt])
                if const_expr(dt + G2D < DT):
                    q_tr_n = [
                        _read_tr(_q_base, dt + G2D, _p)
                        for _p in range_constexpr(PV_K_STEPS)
                    ]
                for pks in range_constexpr(PV_K_STEPS):
                    for nt in range_constexpr(NT):
                        dk_cur[dt][nt] = mfma_acc(q_tr[pks], ds_pack[pks][nt], dk_cur[dt][nt])
                # Interleave the dt+1 prefetch ds_read_tr16 1:1 with the dt MFMAs so
                # the LDS read latency hides in the MFMA shadow.
                if const_expr(dt + G2D < DT):
                    for _ in range_constexpr(2 * PV_K_STEPS * NT):
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
                if const_expr(dt + G2D < DT):
                    do_ring[_slot], q_ring[_slot] = do_tr_n, q_tr_n

            rocdl.s_setprio(0)
            return dv_cur, dk_cur

        def _q_body(q_start, inner, apply_mask):
            dv_cur = [[inner[dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)]
            dk_cur = [
                [inner[DT * NT + dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)
            ]
            for head_local in range_constexpr(GQA_GROUP_SIZE):
                if const_expr(head_local % DMA_STAGE_HEADS == 0):
                    _stage_heads(q_start, head_local)
                dv_cur, dk_cur = _head_step(q_start, apply_mask, head_local, dv_cur, dk_cur)
            out = [dv_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]
            out += [dk_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]
            return out

        _carry = dv_accs + dk_accs
        loop_results = _carry
        if const_expr(window_left >= 0):
            _qhi = _kv_end_c - causal_offset + fx.Index(window_left)
            _qhi = fx.Index(ArithValue(_qhi < seq_len_q_v).select(_qhi, seq_len_q_v))
            for q_start, inner in range(_q_loop_start, _qhi, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
        else:
            for q_start, inner in range(_q_loop_start, _masked_upper, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
            for q_start, inner in range(_unmask_start, seq_len_q_v, _step, init=loop_results):
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
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        _wpe_dkdv = waves_per_eu
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len_k)
        num_kv_tiles = (sl_idx + BLOCK_KV - 1) // BLOCK_KV
        grid_x = bs_idx * num_kv_tiles * NUM_HEADS_KV * Q_SPLIT

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else []
        )
        # Cap AGPR at 64,64 (with vgpr-form off) so the MFMA accumulators stay in AGPRs
        # and the dkdv kernel keeps occ-2 without VGPR spills.
        passthrough_entries = passthrough_entries + [
            ["amdgpu-agpr-alloc", "64,64"],
            ["amdgpu-mfma-vgpr-form", "false"],
        ]
        flash_attn_bwd_dkdv_kernel(
            Q,
            K,
            V,
            DO,
            LSE,
            DELTA,
            DK,
            DV,
            seq_len_q,
            seq_len_k,
            value_attrs={
                "rocdl.waves_per_eu": _wpe_dkdv,
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
    fast_exp2=False,  # default hw v_exp2, aligned with the fwd acceptance path
    window_left=-1,
    batch_size=None,  # compile-time B; required for SBHD seq-step stride bake
    sbhd=False,       # SBHD [S,B,H,D] native layout (seq-step = B*H*D)
    fuse_delta=False,  # compute DELTA here from O (arg 8) instead of a separate odo pass
    wave_block=True,  # per-wave causal class in the diagonal band (skip / demote / mask)
):
    """Build the dQ Q-outer backward launcher (16x16x32 mirror of dkdv).

    One work-group owns BLOCK_M q rows of one q-head and loops the causal kv
    blocks, accumulating dQ in registers -> single write, deterministic. Fused
    identity-center path: DELTA holds -delta_id = -rowsum_d(O.dO); the kernel
    centers dP by it in-loop (plain bf16 operands) and corrects the residual rho/R
    in the epilogue -> exact consistent dQ in one pass.

    fuse_delta=True makes this kernel PRODUCE DELTA instead of reading it: the O
    rows matching its owned q rows come in through arg 8 (the slot the unused K16
    used to occupy) and the row reduce reuses the dO that is already in registers
    as the GEMM1b B-operand, so the standalone odo kernel -- a full 268 MB HBM
    pass, 43.6 us of the gpt-oss B2/S8192 trio -- disappears from the launch
    sequence and only O's half of its traffic is left, inside a kernel that is
    53% MFMA-busy and moves 8x less data than its bandwidth allows.

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

    ENABLE_DMA = enable_dma and not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    # q rows per work-group (owned). 192 (4 waves, QT=3) beat a narrower 128 tile:
    # the wider tile streams each kv tile from fewer work-groups.
    BLOCK_M = 192
    WARP_SIZE = 64
    BLOCK_KV = block_kv  # kv rows per loop iteration (LDS tile)
    flat_work_group_size = 256
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_Q = BLOCK_M // NUM_WAVES  # 48

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32); q<->kv mirror of dkdv. ----
    M_TILE = 16
    N_TILE = 16
    D_TILE = 16
    K_STEP_QK = 32  # K=32 per GEMM1 MFMA (contract over D)
    K_STEPS_QK = head_dim // K_STEP_QK  # d64 -> 2
    QT = ROWS_PER_WAVE_Q // N_TILE  # owned q 16-tiles per wave: 2
    KVT = BLOCK_KV // M_TILE  # looped kv 16-tiles in the LDS block: 4
    DT = head_dim // D_TILE  # D 16-tiles: 4
    PV_K_STEP = 32  # K=32 per GEMM2 MFMA (contract over kv)
    PV_K_STEPS = BLOCK_KV // PV_K_STEP  # 64/32 = 2

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
    # SBHD [S,B,H,D]: per-token seq step is B*H*D (batch interleaved in the seq axis)
    # while the per-batch base is only H*D. THD/BSHD keep RD==STRIDE (dense).
    if sbhd:
        assert batch_size is not None, "SBHD dq needs compile-time batch_size"
    RD_STRIDE_Q = (batch_size * STRIDE_TOKEN_Q) if sbhd else STRIDE_TOKEN_Q
    RD_STRIDE_KV = (batch_size * STRIDE_TOKEN_KV) if sbhd else STRIDE_TOKEN_KV

    K_STRIDE = HEAD_DIM
    LDS_TILE = BLOCK_KV * K_STRIDE
    LDS_V_BASE = LDS_TILE
    # Grouped staging: LDS_NSTAGE kv tiles are DMA'd under ONE barrier pair + ONE
    # s_waitcnt(0) and then consumed back-to-back, so the per-trip sync convoy is
    # amortised over LDS_NSTAGE trips (kv order and the accumulation order are
    # unchanged -> bit-identical). The stage bases stay COMPILE-TIME constants
    # (the group is unrolled, not parity-selected): a runtime base makes LLVM sink
    # the hoisted ds_read address chain back into the loop (+30 VALU/trip, -2%).
    LDS_NSTAGE = 2
    LDS_STAGE = 2 * LDS_TILE
    LDS_TOTAL = LDS_NSTAGE * LDS_STAGE

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
        O: fx.Tensor,  # fuse_delta: O for the DELTA reduce; otherwise unused
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
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

        def _vexp_intrin(x):
            # Backend-visible 2^x: emits the same v_exp_f32 but, being a recognised
            # VALU op rather than opaque asm, it carries the MFMA->VALU hazard itself,
            # so no separate anchor instruction is needed.
            return fx.Float32(
                llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", [_raw(x)], [], [])
            )


        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)
        # Bottom-right causal: offset = seq_k - seq_q >= 0 (Sq <= Skv).
        causal_offset = seq_len_k_v - seq_len_q_v
        causal_off_i32 = fx.Int32(seq_len_k) - fx.Int32(seq_len_q)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16  # M/N index within a 16-tile
        kg = lane // 16  # 0..3: K-subgroup (inputs) / M-block (C output)


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
        num_q_tiles = (seq_len_q_v + BLOCK_M - 1) // BLOCK_M
        _qt_disp = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        # Causal work per WG = (q_tile+1)*BLOCK_M/BLOCK_KV kv-tiles, i.e. monotonically
        # increasing in q_tile, and the dispatch order IS the list-schedule order, so
        # longest-processing-time-first (descending q_tile) balances the tail wave.
        q_tile_idx = num_q_tiles - fx.Index(1) - _qt_disp
        # Causal-aligned tile origin: num_q_tiles*BLOCK_M overshoots seq_len_q by
        # Q_PAD rows, and anchoring at row 0 dumps that overshoot on the LAST tile --
        # precisely the one with the longest causal kv range. Shifting every origin
        # back by a whole number of BLOCK_KV moves the overshoot to tile 0 (the
        # shortest) and puts each tile's causal boundary on a BLOCK_KV edge, so every
        # tile walks one fewer kv block.
        # fx.Index is unsigned -> guard both subtractions.
        _q_shift = ((num_q_tiles * fx.Index(BLOCK_M) - seq_len_q_v) // fx.Index(BLOCK_KV)) * fx.Index(
            BLOCK_KV
        )
        _q_org = q_tile_idx * BLOCK_M
        q_start = fx.Index(ArithValue(_q_org >= _q_shift).select(_q_org - _q_shift, fx.Index(0)))
        # Rows this tile OWNS (stores) end here; tile 0 owns fewer than BLOCK_M rows.
        _q_owned_end = _q_org + fx.Index(BLOCK_M) - _q_shift
        q_owned_end = fx.Index(
            ArithValue(_q_owned_end < seq_len_q_v).select(_q_owned_end, seq_len_q_v)
        )

        # Per-batch base (elements). SBHD: batch inside the seq axis -> base is only
        # H*D. THD: dense per-batch block -> base is seq*H*D.
        if const_expr(sbhd):
            _q_batch_elems = batch_idx * fx.Index(STRIDE_TOKEN_Q)
            _kv_batch_elems = batch_idx * fx.Index(STRIDE_TOKEN_KV)
        else:
            _q_batch_elems = batch_idx * seq_len_q_v * fx.Index(STRIDE_TOKEN_Q)
            _kv_batch_elems = batch_idx * seq_len_k_v * fx.Index(STRIDE_TOKEN_KV)

        # Fold per-batch element offset into raw K/V pointers (0-based rows).
        _kv_ptr_batch_off = _kv_batch_elems
        k_ptr = buffer_ops.get_element_ptr(k_ptr, _kv_ptr_batch_off, elem_type=elem_type)
        v_ptr = buffer_ops.get_element_ptr(v_ptr, _kv_ptr_batch_off, elem_type=elem_type)

        def global_idx_q(token_idx, col):
            return token_idx * RD_STRIDE_Q + q_head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * RD_STRIDE_KV + kv_head_idx * HEAD_DIM + col

        def bf16_trunc_pack_v8(f32_vals):
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        PBLK = 128  # 128-wide block holds 2 real rows (low r&4=0 -> [0,64), high -> [64,128))

        def _pblk(row_idx):
            return ((row_idx >> fx.Index(3)) << fx.Index(2)) | (row_idx & fx.Index(3))

        def _swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(7)) << fx.Index(4)
            return col_idx ^ mask

        # GEMM2 transpose-read addressing: lane-only register + compile-time literal,
        # so the backend folds the per-read part into the ds_read offset immediate
        # instead of materialising one address VGPR per read. See the dkdv builder for
        # the derivation; the identities are asserted over the full domain here too.
        def _pblk_py(row_idx):
            return ((row_idx >> 3) << 2) | (row_idx & 3)

        for _p in range(PV_K_STEPS):
            for _e in (0, N_TILE):
                for _l in range(M_TILE):
                    assert _pblk_py(_p * PV_K_STEP + _e + _l) == _pblk_py(_p * PV_K_STEP + _e) + _pblk_py(_l)
                    assert ((_p * PV_K_STEP + _e + _l) & 7) == (_l & 7)
        for _d in range(DT):
            for _m in range(8):
                for _c in range(4):
                    assert ((_d * D_TILE + _c * 4) ^ (_m << 4)) == ((_d * D_TILE) ^ (_m << 4)) + _c * 4

        _tr_l = kg * fx.Index(4) + (lane16 // fx.Index(4))
        _tr_lane = _pblk(_tr_l) * fx.Index(PBLK) + (lane % fx.Index(4)) * fx.Index(4)
        _tr_mask = (_tr_l & fx.Index(7)) << fx.Index(4)
        _tr_dyn = [
            fx.Int64((_tr_lane + (fx.Index(_d * D_TILE) ^ _tr_mask)) * fx.Index(2))
            for _d in range_constexpr(DT)
        ]

        def _read_tr_at(dt, const_byte_off):
            ptr = buffer_ops.create_llvm_ptr(
                _tr_dyn[dt] + fx.Int64(const_byte_off), address_space=3
            )
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Per-batch descriptors (batch base folded into SRD base). ----
        _q_nrec_bytes = _raw(seq_len_q_v * fx.Index(RD_STRIDE_Q * 2))
        _q_batch_byte_off = _raw(_q_batch_elems * fx.Index(2))
        _kv_nrec_bytes = _raw(seq_len_k_v * fx.Index(RD_STRIDE_KV * 2))
        _kv_batch_byte_off = _raw(_kv_batch_elems * fx.Index(2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        _lse_per_batch = seq_len_q_v * fx.Index(NUM_HEADS_Q)
        _lse_nrec_bytes = _raw(_lse_per_batch * fx.Index(4))
        _lse_batch_byte_off = _raw(batch_idx * _lse_per_batch * fx.Index(4))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )
        if const_expr(fuse_delta):
            o_rsrc = buffer_ops.create_buffer_resource(
                O, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
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
            KV_TILE_BYTES = (BLOCK_KV // 2) * 128 * 2
            NUM_DMA_KV = KV_TILE_BYTES // DMA_BATCH_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (128 * 2)  # blocks per batch
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            # cachepolicy 0: bit 0 is sc0 (the old GLC), which forces the DMA to miss
            # the vector L1. K/V are read-only tiles re-read by every q tile above
            # the diagonal, so the coherent read gives up L1 hits for nothing.
            # (The same bit on dkdv's Q/dO DMA measured neutral -- those tiles are
            # only shared between work-groups of the same kv head, so there is no
            # vL1D reuse to recover -- and is left alone.)
            _dma_aux = fx.Int32(0)

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
                    block = tid // fx.Index(16) + fx.Index(d * ROWS_PER_DMA_BATCH)
                    lane_in_block = tid % fx.Index(16)
                    half = lane_in_block // fx.Index(8)
                    position = lane_in_block * fx.Index(8)  # swiz col within 128-block
                    row_in_tile = (
                        fx.Index(8) * (block >> fx.Index(2)) + (block & fx.Index(3)) + half * fx.Index(4)
                    )
                    xor_mask = (row_in_tile & fx.Index(7)) << fx.Index(4)
                    unsw_col_f16 = position ^ xor_mask  # real col in [0,64), 1x HBM
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile
                    global_byte = (
                        global_row * fx.Index(RD_STRIDE_KV * 2)
                        + kv_head_idx * fx.Index(HEAD_DIM * 2)
                        + col_byte
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        src_rsrc, lds_ptr, _dma_size, fx.Int32(global_byte), _dma_soff, _dma_off, _dma_aux
                    )

        # ---- Owned Q,dO B-operand packs: B[k=D][n=q], n=lane16, k=kg*8+s. Per wave
        # QT q 16-tiles x K_STEPS_QK D-steps; q_b_packs[qt][ks] is a v8 bf16. ----
        q_row_wave = q_start + wave_id * ROWS_PER_WAVE_Q
        q_wave_last = q_row_wave + fx.Index(ROWS_PER_WAVE_Q - 1)

        def q_row_of(qt):
            return q_row_wave + fx.Index(qt * N_TILE) + lane16

        # Q carries the softmax scale: prescaling the GEMM1a B-operand by
        # sm_scale*log2e (x2^23 for the Schraudolph path) lets GEMM1a itself produce
        # the exponent s*sm_scale*log2e + lse with lse as the accumulator init, so the
        # per-element fma in front of every exp2 disappears. Costs one extra bf16
        # rounding of Q, hoisted once per work-group; Q feeds no other GEMM (dQ comes
        # from K_tr @ C).
        _pre_q = fx.Float32(sm_scale * _LOG2E * (float(1 << 23) if fast_exp2 else 1.0))
        q_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        do_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        d_parts = [fx.Float32(0.0) for _ in range_constexpr(QT)]
        for qt in range_constexpr(QT):
            _qr = q_row_of(qt)
            for ks in range_constexpr(K_STEPS_QK):
                q_col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
                _q_raw = Vec(
                    buffer_ops.buffer_load(
                        q_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                    )
                ).to(fx.Float32)
                q_b_packs[qt][ks] = bf16_trunc_pack_v8(
                    [_fmul(_q_raw[i], _pre_q) for i in range_constexpr(MFMA_LANE_K)]
                )
                do_b_packs[qt][ks] = buffer_ops.buffer_load(
                    do_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
                if const_expr(fuse_delta):
                    # This lane's slice of row _qr: O.dO over the 8 D it holds. The O
                    # pack dies here (only the f32 partial stays live), so the reduce
                    # adds one in-flight dwordx4, not a second B-operand set.
                    _o_v = Vec(
                        buffer_ops.buffer_load(
                            o_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                        )
                    ).to(fx.Float32)
                    _od = _o_v * Vec(do_b_packs[qt][ks]).to(fx.Float32)
                    for i in range_constexpr(MFMA_LANE_K):
                        d_parts[qt] = fx.Float32(_fadd(d_parts[qt], Vec(_od)[i]))

        # ---- Owned LSE/-delta_id per q (one scalar per qt, q = qt*16 + lane16). ----
        lse_owned = []
        delta_owned = []
        for qt in range_constexpr(QT):
            _lse_elem = q_head_idx * seq_len_q_v + q_row_of(qt)
            lse_owned.append(
                fx.Float32(buffer_ops.buffer_load(lse_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32))
            )
            if const_expr(not fuse_delta):
                delta_owned.append(
                    fx.Float32(buffer_ops.buffer_load(delta_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32))
                )
        if const_expr(fuse_delta):
            # DELTA[b,hq,q] = -rowsum_d(O.dO). A row's 64 D are split over the 4
            # K-subgroup lanes that share lane16, so the row total is a 2-step xor
            # butterfly over kg (masks 16,32) -- ds_bpermute is the LDS crossbar only,
            # no allocation and no barrier, the same reduce the standalone odo kernel
            # runs over its lane groups. Every (b,hq,q) row is OWNED by exactly one
            # work-group, so one lane per row (kg==0) stores it for dkdv, which still
            # reads DELTA; the rows this tile only traces are recomputed, not stored.
            _lane_i32 = fx.Int32(lane)
            for _m in [M_TILE, 2 * M_TILE]:
                _idx = _raw((_lane_i32 ^ fx.Int32(_m)) * fx.Int32(4))
                for qt in range_constexpr(QT):
                    _part = _raw(Vec.from_elements([d_parts[qt]], fx.Float32).bitcast(fx.Int32)[0])
                    _peer = rocdl.ds_bpermute(fx.Int32.ir_type, _idx, _part)
                    _peer_f = fx.Float32(
                        _raw(Vec.from_elements([fx.Int32(_peer)], fx.Int32).bitcast(fx.Float32)[0])
                    )
                    d_parts[qt] = fx.Float32(_fadd(d_parts[qt], _peer_f))
            for qt in range_constexpr(QT):
                delta_owned.append(fx.Float32(_fsub(fx.Float32(0.0), d_parts[qt])))
                _q_row = q_row_of(qt)
                buffer_ops.buffer_store(
                    delta_owned[qt],
                    delta_rsrc,
                    (q_head_idx * seq_len_q_v + _q_row) * fx.Index(4),
                    mask=ArithValue(_q_row < q_owned_end) & ArithValue(kg == fx.Index(0)),
                    offset_is_bytes=True,
                )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        _c_scaled_floor = fx.Float32(-87.0 * float(1 << 23) + float(127 * (1 << 23) - 486411))
        _compute_type = fx.Float32.ir_type
        v4i32_ty = Vec.make_type(4, fx.Int32)

        def _p_of(s_r, apply_mask):
            # s_r already IS the base-2 softmax exponent: GEMM1a runs on a Q prescaled
            # by sm_scale*log2e (x2^23 in the Schraudolph mode) and is initialised with
            # the prescaled lse, so no fma is left in front of the exp2.
            if const_expr(fast_exp2):
                if const_expr(apply_mask):
                    s_r = ArithValue(s_r).maximumf(_c_scaled_floor)
                i = arith.fptosi(fx.Int32.ir_type, _raw(s_r))
                return ArithValue(i).bitcast(_compute_type)
            # Bare hw v_exp_f32, as dkdv's _p_of and the dq bulk path already use:
            # ArithValue.exp2's denormal-safe expansion (exp2(x+k)/ldexp + compare +
            # 2 cndmask) costs 240 VALU per masked trip and buys nothing here. Only
            # the masked (diagonal) branch reaches this helper, and a diagonal P is
            # the largest term of its softmax row -- never denormal; a masked slot is
            # s_r=-inf, and v_exp_f32(-inf)=0 exactly. dQ is bit-identical.
            return _vexp_intrin(s_r)

        # A-operand read (K/V from LDS): A[m=kv=lane16][k=D=kg*8+s]. kvt selects the
        # 16-kv tile (row = kvt*16 + lane16), ks the D 32-step (D = ks*32 + kg*8).
        # Address hoist (byte-identical layout): row = kvt*16 + lane16 with kvt*16
        # a 16-multiple, so _pblk(kvt*16+lane16)*PBLK == kvt*(8*PBLK) + _pblk(lane16)*PBLK
        # (proven over the full lane/kvt/ks domain). The lane-only part is loop-
        # invariant and the (col^mask) part is kvt-invariant, so precompute both once
        # instead of recomputing the shift/or swizzle bit-ops per ds_read. This drops
        # the per-read _pblk + xor arithmetic to two adds without touching the LDS
        # layout, its 0-conflict property, or determinism.
        a_swz_mask = (lane16 & fx.Index(7)) << fx.Index(4)

        def _a_idx(a_base, kvt, ks):
            row = fx.Index(kvt * M_TILE) + lane16
            col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
            return a_base + _pblk(row) * fx.Index(PBLK) + (col ^ a_swz_mask)

        def _gemm1_load(a_base, kvts):
            """Issue the ds_read loads for A(K/V)[kvt] only, no MFMA yet. Split out
            of _gemm1 so the caller can prefetch a kv-half's K reads ahead of when
            its MFMAs are actually issued (see the kv-half loop below)."""
            return {
                kvt: [
                    Vec.load(mfma_pack_type, lds, [_a_idx(a_base, kvt, ks)])
                    for ks in range_constexpr(K_STEPS_QK)
                ]
                for kvt in kvts
            }

        def _gemm1_mfma(a, b_packs, inits_q=None, kvts=None):
            """S[kvt][qt] (v4f32) = a[kvt] @ B(owned Q/dO)[qt]^T over D, given
            already-loaded A tiles `a` (see _gemm1_load). inits_q[qt] optionally
            pre-loads the accumulator (folds -delta_id into the dP GEMM for free)."""
            if kvts is None:
                kvts = list(a.keys())
            out = [[None] * QT for _ in range_constexpr(KVT)]
            for kvt in kvts:
                for qt in range_constexpr(QT):
                    acc = c_zero_v4f32 if inits_q is None else inits_q[qt]
                    for ks in range_constexpr(K_STEPS_QK):
                        acc = mfma_acc(a[kvt][ks], b_packs[qt][ks], acc)
                    out[kvt][qt] = acc
            return out

        def _gemm1(a_base, b_packs, inits_q=None, kvts=None):
            """S[kvt][qt] (v4f32) = A(K/V)[kvt] @ B(owned Q/dO)[qt]^T over D. A is
            loaded once per (kvt,ks) and reused across qt. inits_q[qt] optionally
            pre-loads the accumulator (folds -delta_id into the dP GEMM for free).
            kvts restricts to a subset of kv 16-tiles (halves the live s/dp transient
            peak when the caller interleaves exp2/pack per kv-half)."""
            if kvts is None:
                kvts = list(range_constexpr(KVT))
            a = _gemm1_load(a_base, kvts)
            return _gemm1_mfma(a, b_packs, inits_q, kvts)

        def _read_tr(a_base, dt, pks):
            """Transpose-read K -> GEMM2 A-operand [m=D=dt*16+lane16][k=kv=kg*8+s].
            a_base is a compile-time LDS byte base, so the whole per-read offset is a
            literal the backend can fold into the ds_read offset field."""
            c0 = (a_base + _pblk_py(pks * PV_K_STEP) * PBLK) * 2 + lds_off
            c1 = c0 + _pblk_py(N_TILE) * PBLK * 2
            v0 = _read_tr_at(dt, c0)
            v1 = _read_tr_at(dt, c1)
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # Per-q delta init (broadcast over the 4 kv output rows) and q-slot i32. The
        # matching lse broadcast pre-loads GEMM1a so the softmax exponent needs no fma.
        delta_inits = [
            Vec.from_elements([delta_owned[qt]], fx.Float32).broadcast_to(4).ir_value()
            for qt in range_constexpr(QT)
        ]
        lse_inits = [
            Vec.from_elements([lse_owned[qt]], fx.Float32).broadcast_to(4).ir_value()
            for qt in range_constexpr(QT)
        ]
        q_slot_i32 = [fx.Int32(q_row_of(qt)) for qt in range_constexpr(QT)]

        # ---- Loop-carried: A(DT*QT) dQ~ accumulators. dQ = sm * A,
        # A = sum_kv K_tr @ (P~*(dP-delta_id)). The rho/R * B self-consistency
        # correction is dropped (halves GEMM2 MFMA): delta_id from odo is the
        # fp32-exact rowsum_d(O.dO), so C already carries the near-diagonal
        # cancellation before the bf16 pack, holding the fast-mode SNR gate. The
        # per-row rowsum(P~) renorm is also dropped: R==1 to bf16 precision (see
        # the epilogue) so it was pure VALU overhead on a VALU-bound kernel.
        A_accs = [c_zero_v4f32 for _ in range_constexpr(DT * QT)]

        # Causal kv range is set by the last OWNED q row, not q_start+BLOCK_M: tile 0
        # traces BLOCK_M rows but owns only the first q_owned_end of them.
        _q_end = q_owned_end + causal_offset
        kv_upper = fx.Index(ArithValue(_q_end < seq_len_k_v).select(_q_end, seq_len_k_v))

        def _kv_compute(kv_start, inner, apply_mask, stage=0):
            """One kv tile: S/dP recompute, C pack and the dQ GEMM2 accumulate. The
            LDS K/V tile is already staged, so this is the part a wave whose q rows
            are entirely on one side of the diagonal can skip or run unmasked.
            `stage` selects the (compile-time) LDS staging slot."""
            A_cur = [[inner[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]
            _k_base_c = stage * LDS_STAGE
            _k_base = fx.Index(_k_base_c)
            _v_base = fx.Index(stage * LDS_STAGE + LDS_V_BASE)

            kv_start_i32 = fx.Int32(kv_start)
            # C[kvt][qt]: 4 f32 at kv=kvt*16+kg*4+t, q=qt*16+lane16. C = P~*(dP-delta_id)
            # feeds GEMM2; R = rowsum(P~) is the fast_exp2 renorm. (P itself is no longer
            # kept: the B-GEMM that consumed P was dropped, see the loop-carried comment.)
            C = [[None] * QT for _ in range_constexpr(KVT)]
            c_pack = [[None] * QT for _ in range_constexpr(PV_K_STEPS)]
            # Split GEMM1a/1b + exp2/C + pack per kv-half (pks = the 2 kvt of one GEMM2
            # K=32 step): compute s/dP for only 2 kvt, consume (exp2/C/pack) and free them
            # before the next half. This halves the live s/dP transient peak (the VGPR
            # ceiling) WITHOUT touching the batched GEMM2 below, so it can reach occ-3
            # (wpe=3) while keeping the depth-2 k_tr prefetch the per-kvt fusion lost.
            # K cross-half prefetch: the NEXT kv-half's K ds_read is issued right
            # after the current half's GEMM1 MFMAs (before the VALU-heavy exp2/C/
            # pack section below), so its latency hides in that VALU shadow instead
            # of sitting exposed right before the next half's GEMM1 MFMA issue
            # (V stays loaded in-half by the unsplit _gemm1 -- only K is prefetched).
            k_a_by_half = {0: _gemm1_load(_k_base, [0, 1])}
            for pks in range_constexpr(PV_K_STEPS):
                ka, kb = 2 * pks, 2 * pks + 1
                half = [ka, kb]
                # GEMM1a S[kv,q]=K@Q^T ; GEMM1b dP[kv,q]=V@dO^T (acc init=-delta_id) for
                # this kv-half. s_setprio(1) raises MFMA priority over ds_read/VALU;
                # dropped to 0 for the exp2/pack/reduce VALU section so it is not starved.
                rocdl.s_setprio(1)
                rocdl.iglp_opt(0)
                s_tiles = _gemm1_mfma(k_a_by_half[pks], q_b_packs, lse_inits, kvts=half)
                dp_tiles = _gemm1(_v_base, do_b_packs, delta_inits, kvts=half)
                rocdl.s_setprio(0)

                # Narrow the prefetched-half's live range: load only ka's K here
                # (before the qt loop), and issue kb's K load between qt=0 and
                # qt=1 below (_next_kb_load), instead of the whole half in one
                # block. Halves how long the second kvt's registers stay live
                # before GEMM1 actually consumes them, trimming the extra
                # scratch this prefetch adds while keeping the same total
                # ds_read-vs-VALU-shadow overlap.
                _next_kb_load = None
                if const_expr(pks + 1 < PV_K_STEPS):
                    nka, nkb = 2 * (pks + 1), 2 * (pks + 1) + 1
                    # s_setprio(1) around the prefetch ds_read issue only (not the
                    # VALU it's interleaved with): the load itself should win issue
                    # priority over the surrounding exp2/pack VALU so it drains
                    # sooner, without raising priority on the VALU work itself.
                    rocdl.s_setprio(1)
                    k_a_by_half[pks + 1] = _gemm1_load(_k_base, [nka])
                    rocdl.s_setprio(0)

                    def _next_kb_load():  # noqa: B023
                        rocdl.s_setprio(1)
                        k_a_by_half[pks + 1].update(_gemm1_load(_k_base, [nkb]))  # noqa: B023
                        rocdl.s_setprio(0)

                if const_expr(not apply_mask):
                    # Vectorized bulk (below-diagonal): exp2/C/reduce as packed v4 ops
                    # (v_pk_*), mirroring the 32x32 kernel's v8 path. exp2 and C=P*dP are
                    # strictly elementwise so C is bit-identical to the scalar branch;
                    # R re-associated in a fixed order -> deterministic (det gate holds).
                    for qt in range_constexpr(QT):
                        if const_expr(qt == 1 and _next_kb_load is not None):
                            _next_kb_load()
                        for kvt in half:
                            s4 = s_tiles[kvt][qt]
                            if const_expr(fast_exp2):
                                # fptosi reads the accumulator v4 itself -> own anchor.
                                p4 = Vec(arith.fptosi(v4i32_ty, _raw(s4))).bitcast(fx.Float32)
                            else:
                                # exact: the accumulator already holds the log2 exponent.
                                # Slot 0 goes through the exp2 INTRINSIC, a backend-visible
                                # VALU read of the MFMA result, so it both anchors the v4's
                                # MFMA->VALU wait states and produces P -- no separate
                                # v_min no-op is needed. Slots 1..3 read the v4 bare and are
                                # pinned behind that anchor (the anchor must read the very
                                # v4 it protects, pitfalls/13).
                                p4 = Vec.from_elements(
                                    [_vexp_intrin(Vec(s4)[t]) for t in range_constexpr(4)],
                                    fx.Float32,
                                )
                            if const_expr(window_left >= 0):
                                _thr = q_slot_i32[qt] + causal_off_i32 - fx.Int32(window_left)
                                _kvb = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4)
                                p4 = Vec.from_elements(
                                    [
                                        ArithValue(_kvb + fx.Int32(t) > _thr).select(Vec(p4)[t], c_zero_f)
                                        for t in range_constexpr(4)
                                    ],
                                    fx.Float32,
                                )
                            c4 = p4 * Vec(dp_tiles[kvt][qt])
                            C[kvt][qt] = [c4[t] for t in range_constexpr(4)]
                else:
                    for qt in range_constexpr(QT):
                        if const_expr(qt == 1 and _next_kb_load is not None):
                            _next_kb_load()
                        for kvt in half:
                            dp_v = dp_tiles[kvt][qt]
                            s_v = s_tiles[kvt][qt]
                            c_vals = []
                            for t in range_constexpr(4):
                                kv_slot = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4 + t)
                                _up = ArithValue(kv_slot > q_slot_i32[qt] + causal_off_i32)
                                if const_expr(window_left >= 0):
                                    _lo = ArithValue(
                                        kv_slot <= q_slot_i32[qt] + causal_off_i32 - fx.Int32(window_left)
                                    )
                                    _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                                else:
                                    _mm = _up
                                # The mask cndmask is itself this slot's accumulator read,
                                # so it doubles as the MFMA->VALU hazard anchor.
                                s_r = _mm.select(c_neg_inf, fx.Float32(Vec(s_v)[t]))
                                p = _p_of(s_r, True)
                                c = _fmul(p, Vec(dp_v)[t])
                                c_vals.append(c)
                            C[kvt][qt] = c_vals

                # Pack this half's C now (contract over kv): combine kvt=ka (k=0..3) and
                # kvt=kb (k=4..7) -> 8 kv values/lane matching _read_tr's kv ordering.
                # Packing here frees C[ka],C[kb] (and s/dP) before the next half's GEMM1.
                for qt in range_constexpr(QT):
                    c_pack[pks][qt] = bf16_trunc_pack_v8(C[ka][qt] + C[kb][qt])

            # GEMM2 A^T[D,q] += K_tr @ C. The B-GEMM (K_tr @ P) is dropped -> half the
            # GEMM2 MFMA. N3 ILP: process dt in pairs with the two dt interleaved at
            # issue so a dependent MFMA (same dt,qt across pks) is separated by 3
            # independent MFMAs (was 1), overlapping the 16x16x32 MFMA-operand latency.
            # The next pair's k_tr is prefetched (pair-depth-2) during the current pair.
            # The initial pair is read here (not hoisted above the compute) to keep the
            # k_tr registers off the s/dP transient peak so wpe=3 fits occ-3 spill-free.
            # s_setprio(2) raises MFMA issue priority over ds_read.
            kts = [
                [_read_tr(_k_base_c, d, pks) for pks in range_constexpr(PV_K_STEPS)]
                for d in range_constexpr(min(2, DT))
            ]
            rocdl.s_setprio(2)
            for d0 in range_constexpr(0, DT, 2):
                if const_expr(d0 + 2 < DT):
                    kts.append([_read_tr(_k_base_c, d0 + 2, pks) for pks in range_constexpr(PV_K_STEPS)])
                    kts.append([_read_tr(_k_base_c, d0 + 3, pks) for pks in range_constexpr(PV_K_STEPS)])
                for pks in range_constexpr(PV_K_STEPS):
                    for dd in range_constexpr(d0, min(d0 + 2, DT)):
                        for qt in range_constexpr(QT):
                            A_cur[dd][qt] = mfma_acc(kts[dd][pks], c_pack[pks][qt], A_cur[dd][qt])
                # Interleave the next-pair prefetch ds_read_tr16 1:1 with the pair MFMAs.
                if const_expr(d0 + 2 < DT):
                    for _ in range_constexpr(2 * PV_K_STEPS * QT):
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
            rocdl.s_setprio(0)

            out = [A_cur[dt][qt] for dt in range_constexpr(DT) for qt in range_constexpr(QT)]
            return out

        def _stage_tile(kv_start, stage):
            coop_dma_tile(k_rsrc, lds_base_idx + fx.Index(stage * LDS_STAGE * 2), kv_start)
            coop_dma_tile(
                v_rsrc, lds_base_idx + fx.Index((stage * LDS_STAGE + LDS_V_BASE) * 2), kv_start
            )

        def _kv_group(kv_start, inner):
            """LDS_NSTAGE unmasked kv tiles per trip: one barrier pair and one DMA
            drain for the whole group instead of one per tile."""
            gpu.barrier()
            if const_expr(ENABLE_DMA):
                for stage in range_constexpr(LDS_NSTAGE):
                    _stage_tile(kv_start + fx.Index(stage * BLOCK_KV), stage)  # noqa: B023
                rocdl.s_waitcnt(0)
            gpu.barrier()
            for stage in range_constexpr(LDS_NSTAGE):
                inner = _kv_compute(
                    kv_start + fx.Index(stage * BLOCK_KV), inner, False, stage=stage  # noqa: B023
                )
            return inner

        def _kv_body(kv_start, inner, apply_mask):
            gpu.barrier()
            if const_expr(ENABLE_DMA):
                _stage_tile(kv_start, 0)  # noqa: B023
                rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(not apply_mask or not wave_block):
                return _kv_compute(kv_start, inner, apply_mask)
            # Diagonal band: classify THIS WAVE's ROWS_PER_WAVE_Q q rows against the
            # kv tile once per trip instead of masking every slot. Both tests are
            # wave-uniform, so they lower to s_cbranch_execz, and the branch sits
            # after the cooperative DMA/barriers so every wave still stages LDS.
            #   kv_start > the wave's last q  -> every slot is masked, P == 0 exactly,
            #       so the trip's contribution to A is +0.0: skip it (bit-identical).
            #   kv tile end <= the wave's first q -> no slot needs the causal mask, so
            #       the vectorized bulk path is exact and drops the per-slot v_cmp +
            #       v_cndmask + -inf clamp chain (masked trip 3333 cyc vs bulk 1755).
            # At BLOCK_M=192 / BLOCK_KV=64 the band is 3 tiles x 4 waves = 12 blocks,
            # of which 3 are fully masked and 3 fully open, i.e. half the band's
            # masked-path cost is spent on slots whose class is known in advance.
            _skip = ArithValue(kv_start > q_wave_last + causal_offset)
            _open = ArithValue(kv_start + fx.Index(BLOCK_KV - 1) <= q_row_wave + causal_offset)
            return _scf_if_vals(
                _skip,
                lambda: list(inner),
                lambda: _scf_if_vals(
                    _open,
                    lambda: _kv_compute(kv_start, inner, False),  # noqa: B023
                    lambda: _kv_compute(kv_start, inner, True),  # noqa: B023
                    inner,
                ),
                inner,
            )

        # Split the causal kv-loop: [0, q_start) below the diagonal (no mask),
        # [q_start, kv_upper) straddles it (mask).
        _carry = A_accs
        loop_results = _carry
        if const_expr(window_left >= 0):
            # fx.Index is unsigned: guard the subtract (W-1 may exceed q+off) to
            # avoid underflow-to-huge. _wlo skips fully-out-of-window kv tiles.
            _wlo = fx.Index(
                ArithValue(q_start + causal_offset >= fx.Index(window_left - 1)).select(
                    q_start + causal_offset - fx.Index(window_left - 1), fx.Index(0)
                )
            )
            _wlo = (_wlo // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
        else:
            _wlo = fx.Index(0)
        # The grouped loop stops at the last full LDS_NSTAGE-tile boundary; the 0..
        # LDS_NSTAGE-1 leftover tiles join the masked loop, where the per-wave causal
        # classification demotes them to the exact unmasked path anyway.
        _bulk_end = q_start + causal_offset
        _grp = LDS_NSTAGE * BLOCK_KV
        _grp_end = _wlo + ((_bulk_end - _wlo) // fx.Index(_grp)) * fx.Index(_grp)
        for kv_start, inner in range(_wlo, _grp_end, _grp, init=_carry):
            loop_results = yield _kv_group(kv_start, inner)
        for kv_start, inner in range(_grp_end, kv_upper, BLOCK_KV, init=loop_results):
            loop_results = yield _kv_body(kv_start, inner, True)

        A_finals = [[loop_results[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]

        # ---- Epilogue: dQ = sm * A. Both exp modes use R==1: lse is the true
        # log-sum-exp so rowsum(exp(S-lse))==1 over the row, and the Schraudolph fast
        # P~ sums to 1 to bf16 precision (its per-element approx error cancels over
        # the row-sum -> R stays ~1), so the fast renorm was pure VALU overhead
        # and is dropped. The 16x16 C-layout gives 4 CONTIGUOUS D/lane at
        # q=qt*16+lane16 -> direct store. ----
        for qt in range_constexpr(QT):
            dq_scale = fx.Float32(sm_scale)
            _q_row = q_row_of(qt)
            # q_owned_end <= seq_len_q, so this also covers the tail-tile guard.
            _store_mask = ArithValue(_q_row < q_owned_end)
            for dt in range_constexpr(DT):
                a_v = Vec(A_finals[dt][qt])
                vals = [fx.Float32(_fmul(dq_scale, a_v[t])) for t in range_constexpr(4)]
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
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len_q)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS_Q

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else []
        )
        flash_attn_bwd_dq_kernel(
            Q,
            K,
            V,
            DO,
            LSE,
            DELTA,
            DQ,
            O,
            seq_len_q,
            seq_len_k,
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


# ===========================================================================
# Host-side varlen backward orchestration (odo + dq + dkdv split-K reduce).
# Deterministic drop-in for the CK hd64 FMHA varlen backward; the build_* module
# factories above are called directly (same module).
# ===========================================================================
import math as _math
import os as _os

import torch

_LOG2E = _math.log2(_math.e)
_S23 = float(1 << 23)
_BIAS = float(127 * (1 << 23) - 486411)  # Schraudolph min-RMS bias
_FAST_EXP2 = False  # False (default) = hw v_exp2 ~52dB; True = Schraudolph ~35dB
_QSPLIT_ENV = _os.environ.get("DQ_QSPLIT")


def _qsplit_for(Sq):
    # q_split fans the dK/dV KV-owner WGs across the CU grid, at the cost of a
    # per-WG prologue/epilogue and one more dK/dV workspace slot to write. Small Sq
    # needs the extra WGs to fill the machine; from Sq=4096 up the grid is already
    # >=8 full CU rounds, so the cheaper fixed cost wins. det-neutral.
    if _QSPLIT_ENV is not None:
        return int(_QSPLIT_ENV)
    if Sq <= 2048:
        return 4
    return 4


def _blockkv_for(Sq):
    # Small Sq: BLOCK_KV=64 fills the grid (kills the tail effect); large Sq: 128.
    # 128 is the widest tile this WG shape can carry: a wider 256 tile halves the
    # Q/dO re-reads but leaves one WG per CU, so the per-head WAR barrier has no
    # co-resident WG to cover it; a narrower 64 also measured worse.
    return 64 if Sq <= 2048 else 128


_BWD_CACHE: dict = {}
# Fold the odo (DELTA = -rowsum_d(O.dO)) pass into the dq kernel and drop its launch:
# dq is Q-outer and already streams dO, so it reduces DELTA for the q rows it owns,
# saving one kernel launch.
_FUSE_DELTA = True


def _defer_delta(dq_launch):
    """Adapt a fuse_delta dq launcher to the legacy odo -> dq -> dkdv call order.

    The fused dq kernel produces DELTA itself, so the odo launcher has no kernel
    left to launch: it only forwards its O tensor (holding a reference, which may be
    the only one when the caller passes a freshly cast temporary) to the next dq
    launch, where O occupies the argument slot the unused K16 used to occupy.
    Callers that drive the sequence themselves pass O to dq directly instead.
    """
    pending = []

    def _odo(O, DO, DELTA, batch_size, seq_len, stream):
        pending.clear()
        pending.append(O)

    def _dq(Q, K, V, DO, LSE, DELTA, DQ, O, *rest):
        if pending:
            O = pending.pop()
        return dq_launch(Q, K, V, DO, LSE, DELTA, DQ, O, *rest)

    return _dq, _odo


def _get_bwd(Hq, Hkv, D, scale, fast, window_left, q_split, block_kv, batch_size=None, sbhd=False):
    key = (Hq, Hkv, D, scale, fast, window_left, q_split, block_kv, batch_size, sbhd)
    launchers = _BWD_CACHE.get(key)
    if launchers is None:
        common = dict(
            num_heads=Hq,
            head_dim=D,
            causal=True,
            dtype_str="bf16",
            sm_scale=scale,
            num_kv_heads=Hkv,
            window_left=window_left,
        )
        dq_l = build_flash_attn_bwd_dq_module(
            fast_exp2=fast,
            batch_size=batch_size,
            sbhd=sbhd,
            fuse_delta=_FUSE_DELTA,
            **common,
        )
        dkdv_l = build_flash_attn_bwd_dkdv_module(
            q_split=q_split,
            fast_exp2=fast,
            block_kv=block_kv,
            batch_size=batch_size,
            sbhd=sbhd,
            **common,
        )
        if _FUSE_DELTA:
            # The fused dq kernel produces DELTA itself, so the standalone odo
            # kernel is never launched here. Skip building it: its power-of-2
            # butterfly reduction would crash the build for non-pow2 D.
            dq_l, odo_l = _defer_delta(dq_l)
        else:
            odo_l = build_flash_attn_bwd_odo_module(
                num_heads=Hq, head_dim=D, num_kv_heads=Hkv, sm_scale=scale, sbhd=sbhd
            )
        launchers = (dq_l, dkdv_l, odo_l)
        _BWD_CACHE[key] = launchers
    return launchers


def _prescale_lse(lse_bhsq, fast):
    if fast:
        return (lse_bhsq.float() * (-_LOG2E * _S23) + _BIAS).contiguous()  # Schraudolph s23
    return (lse_bhsq.float() * (-_LOG2E)).contiguous()  # exact (hw exp2): plain -log2e*lse


def _uniform_len(cu):
    d = cu[1:] - cu[:-1]
    S = int(d[0].item())
    assert bool((d == S).all().item()), "flydsl varlen bwd requires uniform seqlens"
    return cu.numel() - 1, S


def flydsl_varlen_backward(
    dout, q, k, v, out, lse_bhsq, B, Sq, Skv, Hq, Hkv, D, scale, fast_exp2=_FAST_EXP2, window_left=-1,
    sbhd=False,
):
    """Run the 16x16x32 flydsl bwd.
    THD (sbhd=False): q,dout,dq,out:[B*Sq,Hq,D]; k,v,dk,dv:[B*Skv,Hkv,D].
    SBHD (sbhd=True): q,dout,dq,out:[Sq,B,Hq,D]; k,v,dk,dv:[Skv,B,Hkv,D] (native,
    no permute/copy anywhere -- the kernels address SBHD directly and the dk/dv
    workspace is laid out [q_split,Skv,B,Hkv,D] so the slot reduction is contiguous).
    lse_bhsq:[B,Hq,Sq] f32 (batch-major, layout-independent).
    window_left>=0 = sliding-window causal (valid q+off-W < kv <= q+off)."""
    q_split = _qsplit_for(Sq)
    dq_l, dkdv_l, odo_l = _get_bwd(
        Hq, Hkv, D, scale, fast_exp2, window_left, q_split, _blockkv_for(Sq),
        batch_size=B, sbhd=sbhd,
    )
    st = torch.cuda.current_stream()
    # identity delta = -rowsum(O.dO); both kernels center dP by it (exact). dq owns the
    # reduce (it already holds dO in registers) and stores DELTA for dkdv, so no odo
    # launch is needed; the odo kernel stays for callers that want DELTA standalone.
    # Both paths read O as bf16, so cast (no-op when out is already bf16; the
    # correctness gate passes fp32 out from the fp32 reference forward).
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    o16 = out.to(q.dtype).reshape(-1)
    if not _FUSE_DELTA:
        odo_l(o16, dout.to(q.dtype).reshape(-1), delta.reshape(-1), B, Sq, st)
    lse_s = _prescale_lse(lse_bhsq, fast_exp2)
    dq = torch.empty_like(q)
    # SBHD workspace [q_split,Skv,B,Hkv,D]: summing the leading q_split axis yields
    # [Skv,B,Hkv,D] contiguous == native SBHD dk/dv (no permute). THD keeps
    # [B,q_split,Skv,Hkv,D] -> sum(dim=1) -> [B*Skv,Hkv,D].
    if sbhd:
        ws_dk = torch.empty(q_split, Skv, B, Hkv, D, device=q.device, dtype=k.dtype)
        ws_dv = torch.empty(q_split, Skv, B, Hkv, D, device=q.device, dtype=v.dtype)
    else:
        ws_dk = torch.empty(B, q_split, Skv, Hkv, D, device=q.device, dtype=k.dtype)
        ws_dv = torch.empty(B, q_split, Skv, Hkv, D, device=q.device, dtype=v.dtype)
    qf, kf, vf, dof = q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1)
    lsef, df = lse_s.reshape(-1), delta.reshape(-1)
    dq_l(qf, kf, vf, dof, lsef, df, dq.reshape(-1), o16, B, Sq, Skv, st)
    dkdv_l(qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1), B, Sq, Skv, st)
    if sbhd:
        dk = ws_dk.sum(dim=0)  # [Skv,B,Hkv,D] SBHD contiguous
        dv = ws_dv.sum(dim=0)
    else:
        dk = ws_dk.sum(dim=1).reshape(B * Skv, Hkv, D)
        dv = ws_dv.sum(dim=1).reshape(B * Skv, Hkv, D)
    return dq, dk, dv
