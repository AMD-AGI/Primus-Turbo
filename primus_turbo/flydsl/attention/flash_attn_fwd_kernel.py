# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn_func kernel builder for FlyDSL.

- True MFMA32 remap: `mfma_f32_32x32x16bf16` / `mfma_f32_32x32x16f16` for both GEMM stages.
- Tile shape: BLOCK_M=128 or 256 (auto-selected), BLOCK_N=64.
- BLOCK_M=128: 4 waves (256 threads), BLOCK_M=256: 8 waves (512 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V use separate LDS regions with DMA-to-LDS prefetch and XOR swizzle.
- For H>=32, both M=128 and M=256 variants are built and dispatched at runtime.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) or (512,) depending on BLOCK_M.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % 128 == 0.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LOG2E = host_math.log2(host_math.e)  # 1.4426950408889634
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer load ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    """Extract the aligned LLVM pointer from a FlyDSL tensor/memref."""
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _read_exec_i64():
    """Read the current wave exec mask (ballot of an all-true predicate)."""
    return rocdl.ballot(T.i64, fx.Boolean(True).ir_value())


def dtype_to_elem_type(dtype_str):
    """Map a dtype string to its FlyDSL numeric type (vendored from kernels_common)."""
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")


def build_flash_attn_fwd_module(
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
    path_tag="auto",
    enable_dma=None,
    fast_exp2=False,
):
    """Build a single-variant dense FlyDSL forward flash-attention launcher.

    Vendored from FlyDSL ``kernels/flash_attn_generic.py`` for a fixed ``block_m``
    (GPT-OSS uses 128) plus campaign additions: a head_dim-general K XOR-swizzle
    mask (fixes the d64 correctness bug), an LSE output ``[B, num_heads, S]`` f32
    for the backward, and an explicit ``enable_dma`` override so the d64 path can
    opt into the DMA-to-LDS software-pipelined K load (default None keeps the
    legacy env/N128 behavior).

    ``path_tag`` selects the outer KV granularity: "N32" -> BLOCK_N_OUT=64 (one
    sub-tile/step), "N128" -> BLOCK_N_OUT=128 (two BLOCK_N=64 sub-tiles/step, so
    the intra-block prefetch pipelines the second sub-tile behind the first);
    "auto" picks N128 only for the d128 causal case.

    For GQA/MQA pass ``num_kv_heads < num_heads`` (num_heads % num_kv_heads == 0).
    Q/O have ``num_heads`` heads; K/V have ``num_kv_heads`` heads, with every
    ``num_heads // num_kv_heads`` consecutive Q heads sharing one KV head.
    """
    gpu_arch = get_hip_arch()
    if block_m is None:
        block_m = 128

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    BLOCK_M = block_m
    flat_work_group_size = 256 if BLOCK_M <= 128 else 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    if path_tag.upper() in ("N32", "N128"):
        PATH_TAG = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        PATH_TAG = "N128"
    else:
        PATH_TAG = "N32"
    BLOCK_N_OUT = 128 if PATH_TAG == "N128" else BLOCK_N
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    ENABLE_PREFETCH_3BUF = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    # buffer_load_dwordx4_lds (16B DMA-to-LDS) requires gfx950+; gfx94x only has dword (4B).
    _has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    if enable_dma is None:
        # Legacy behavior: DMA on for the N128 (d128) path or via env override.
        ENABLE_DMA = _has_lds_load_b128 and (
            PATH_TAG == "N128" or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1")
        )
    else:
        # Explicit override (impl passes enable_dma=True for the d64 GPT-OSS path):
        # async global->LDS DMA double-buffers K across the KV loop (software
        # pipelining), overlapping the K load with the current tile's MFMAs.
        ENABLE_DMA = _has_lds_load_b128 and bool(enable_dma)
    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    REDUCE_MODE = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    # gfx950+ has ds_read_tr16_b64 (HW transpose LDS read); gfx942 needs V^T stored in LDS.
    USE_HW_TR = gpu_arch.startswith("gfx950")

    # MFMA32 K-dimension: 16 on gfx950+ (CDNA4) for both GEMMs.
    USE_K16 = gpu_arch.startswith("gfx950")

    # 128-bit permlane-fused O-store needs gfx950 (permlane32_swap + cvt_pk_bf16_f32,
    # both CDNA4-only); gfx942 falls back to a per-lane dwordx2 store via .to(elem_dtype).
    USE_PERMLANE_OSTORE = gpu_arch.startswith("gfx950")
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16 if USE_K16 else 8
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 steps per sub-tile (K=16) or 4 (K=8)

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert flat_work_group_size in (
        128,
        256,
        512,
    ), f"flat_work_group_size must be 128, 256, or 512, got {flat_work_group_size}"
    assert dtype_str in ("f16", "bf16"), "flash_attn_func only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT % BLOCK_N == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    # Bank-conflict-free LDS strides.
    # K uses XOR swizzle (col ^ ((row & 7) << 4)) at 16-element granularity
    # instead of padding. This enables ds_read_b128 (stride is 256B-aligned).
    K_STRIDE = HEAD_DIM
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        VT_STRIDE = BLOCK_N + 2
        V_STRIDE = VT_STRIDE

    # Vectorized cooperative load constants.
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
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

    # K/V circular buffers; defaults to 1/1, optional 3/3 with CK-like LDS sequence.
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    if USE_HW_TR:
        LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    else:
        LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_generic_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type
        MFMA_LANE_K = 8 if USE_K16 else 4

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        # Crude Schraudolph 2^x (fast_exp2): P = bitcast(fptosi((s*sm*log2e - sm*
        # log2e*m)*2^23 + bias)) -> piecewise-linear 2^x. The (-sm*log2e*m)*2^23 +
        # bias addend is prescaled ONCE per kv-tile (neg_max_s23, shared by the tile's
        # 32 exp2 evals for this q-row), so _exp2_of folds its two fmas into one:
        # scaled = s*(sm*log2e*2^23) + neg_max_s23 -> fptosi. Safe in the forward
        # because O = sum P*V / l self-normalizes (the /l cancels the approx scale);
        # the LSE is approximate but the backward's fused dQ renormalizes P internally,
        # so the near-diagonal dS cancellation is independent of the fwd exp precision.
        _c_exp2_scale = fx.Float32(float(1 << 23))
        _c_exp2_bias = fx.Float32(float(127 * (1 << 23) - 486411))
        _c_scaled_scale = fx.Float32(sm_scale * _LOG2E * float(1 << 23))
        _c_scaled_floor = fx.Float32(-87.0 * float(1 << 23) + float(127 * (1 << 23) - 486411))

        def _exp2_of(s_r, neg_max_s23, apply_clamp=True):
            if const_expr(fast_exp2):
                # maximumf guards the masked -inf (masked s_r=-inf -> scaled -inf ->
                # maximumf(floor) -> 2^-87=0, no exp2(-inf)=NaN; pitfalls/04), so it is
                # load-bearing only on masked (diagonal) tiles; the below-diagonal bulk
                # has bounded args (>> -87) so the clamp is dropped there.
                scaled = fmath.fma(s_r, _c_scaled_scale, neg_max_s23, fastmath=fm_fast)
                if const_expr(apply_clamp):
                    scaled = ArithValue(scaled).maximumf(_c_scaled_floor)
                i = arith.fptosi(fx.Int32.ir_type, _raw(scaled))
                return ArithValue(i).bitcast(compute_type)
            # Exact path (fast_exp2=False) reconstructs the plain diff = sm*log2e*
            # (s - m) from the un-prescaled neg_max passed by the caller.
            diff = fmath.fma(s_r, c_sm_scale_log2e, neg_max_s23, fastmath=fm_fast)
            return ArithValue(diff).exp2(fastmath=fm_fast)

        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
                a = Vec(a).bitcast(fx.Int16)
                b = Vec(b).bitcast(fx.Int16)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            if const_expr(USE_K16):
                return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        seq_len_v = fx.Index(seq_len)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ---- ds_read_b64_tr_b16 lane decomposition ----
        # Hardware does 4×4 transpose within blocks of 16 lanes.
        # tr_k_group selects which of 4 K-rows within the block,
        # tr_col_sub selects which 4-column sub-group within 16 columns.
        tr_k_group = (lane % 16) // 4  # 0..3: K-row offset within 4-row group
        tr_col_sub = lane % 4  # 0..3: 4-column sub-group
        tr_col_half = (lane % 32) // 16  # 0 or 1: first/second 16-column half

        # ---- ds_read_b64_tr_b16 helper ----

        def ds_read_tr_v4f16(lds_elem_idx):
            """Read v4f16 from LDS with hardware transpose.

            Within each block of 16 lanes, the hardware performs a 4×4
            transpose across 4 groups of 4 lanes.  After the transpose,
            result[lane, elem_e] = Input[source_lane, lane%4] where
            source_lane = e*4 + (lane%16)//4.  This naturally produces
            the MFMA A-operand layout when per-lane addresses point to
            the correct K-row and D-column sub-group.
            """
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Wave offsets ----
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id (KV-head-major for L2/XCD locality) ----
        # Each block computes one Q head (per batch, per Q-tile). block_id digits,
        # fastest-varying first: kv_head, q-head-in-group, q_tile, batch. MI355X
        # dispatches work-groups round-robin across its 8 XCDs by (block_id % 8),
        # so placing kv_head as the fastest digit pins every tile sharing a KV head
        # to one XCD -> that head's K/V stays resident in its L2 slice. This is a
        # bijection over the whole grid, so correctness and determinism are
        # unchanged. The non-GQA case keeps the identity (q-head-major) decode.
        # Use a compile-time `const_expr` branch (GQA_GROUP_SIZE is a build-time
        # constant); only one branch is emitted, so no dynamic dispatch is traced.
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
        # Causal load-balance: a q-tile's kv-loop length grows with q_tile_idx
        # (tile 0 -> 1 kv-block, tile 63 -> 64), so dispatching them in natural
        # order leaves only the heaviest tiles running at the tail (occ min ~1
        # WG/CU vs LDS-cap ~4). Two-pointer interleave the dispatch order
        # (0, N-1, 1, N-2, ...) so concurrent work-groups mix light+heavy loads
        # and the tail stays populated. Bijection over q-tiles -> each output
        # tile still computed by exactly one WG (corr/det-neutral); kv_head stays
        # the fastest block_id axis so the XCD/L2 remap is untouched.
        if const_expr(CAUSAL):
            _qt_half = _qt_disp // fx.Index(2)
            _qt_is_odd = ArithValue(_qt_disp % fx.Index(2) == fx.Index(1))
            q_tile_idx = fx.Index(
                _qt_is_odd.select(num_q_tiles - fx.Index(1) - _qt_half, _qt_half)
            )
        else:
            q_tile_idx = _qt_disp
        q_start = q_tile_idx * BLOCK_M

        # Non-DMA KV loads use raw k_ptr/v_ptr; fold the per-batch element
        # offset so 0-based global_idx_kv reads this batch (DMA path uses k/v_rsrc).
        _kv_ptr_batch_off = batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_KV)
        k_ptr = buffer_ops.get_element_ptr(k_ptr, _kv_ptr_batch_off, elem_type=elem_type)
        v_ptr = buffer_ops.get_element_ptr(v_ptr, _kv_ptr_batch_off, elem_type=elem_type)

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat indices ----
        # Q/O are laid out with NUM_HEADS_Q heads; K/V with NUM_HEADS_KV.
        # batch_idx*seq_len is folded into each tensor's descriptor base (see q/k/v/o_rsrc),
        # so token indices are 0-based within the batch.
        def global_idx_q(token_idx, col):
            return token_idx * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _kv_row_clamp(row_idx):
            # Non-DMA KV loads use raw pointers (no hardware bounds), so clamp the
            # global KV row to the last valid token; partial-tile lanes then read a
            # duplicated in-bounds row whose contribution the score-side causal /
            # padding mask discards. (The DMA path is bounded by num_records.)
            last = seq_len_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_v).select(row_idx, last))

        def _load_global_half_vec(ptr, base_idx, vec_elems: int):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            _pointer_store(val, gep)

        def load_global_f16x4(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, 4)

        def load_global_mfma_pack(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, MFMA_LANE_K)

        def load_global_f16xN(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, VEC_WIDTH)

        def bf16_trunc_pack_v4(f32_vals):
            """Pack 4 f32 -> v4bf16 via hardware cvt_pk_bf16_f32 (RNE, 1 VALU op/pair)."""
            packed = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[0]), _raw(f32_vals[1])),
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[2]), _raw(f32_vals[3])),
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in packed], fx.Int32)
                .bitcast(elem_dtype)
                .ir_value()
            )

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 -> v8bf16 via hardware cvt_pk_bf16_f32 (RNE, 1 VALU op/pair)."""
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        # ---- K LDS bank-conflict swizzle (gfx950, 64 banks, bf16 b128 reads) ----
        # A d64 K row is 64 elems = 128 B = 32 dwords, so on 64 banks same-parity
        # rows share a 32-bank half; the 16 same-parity rows must be spread across
        # all 8 aligned 16 B (8-elem) slots of that half to hit the b128 2-way floor.
        # The legacy (row & 3) << 4 mask gives only TWO distinct values per parity
        # -> 8 rows collapse onto one 16 B slot = 8-way conflict (measured ~60% of
        # LDS-active cycles). ((row>>1) & 7) << 3 (= ((row//2)%8)*8 elems) yields 8
        # distinct aligned offsets -> 2-way. Applied identically on the read, the DMA
        # global fetch and the (unused-under-DMA) VGPR write paths (XOR self-inverse,
        # so store@(c^mask) + read^mask round-trips). D128 keeps the legacy mask.
        def _k_bank_mask(row_idx):
            if const_expr(K_STRIDE == 64):
                return ((row_idx // fx.Index(2)) % fx.Index(8)) * fx.Index(8)
            return (row_idx & fx.Index(K_STRIDE // 16 - 1)) << fx.Index(4)

        def _k_swizzle(row_idx, col_idx):
            return col_idx ^ _k_bank_mask(row_idx)

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(k_ptr, g_idx)
                    Vec(vec).store(lds_kv, [lds_idx])

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = Vec(vec)[_e]
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                v1 = Vec.from_elements([elem], elem_dtype)
                v1.store(lds_kv, [vt_idx])

        _v_store_to_lds = _v_store_row_major if USE_HW_TR else _v_store_transposed

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global loads for V, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                g_idx = global_idx_kv(row_idx, load_col_base)
                vecs.append(load_global_f16xN(v_ptr, g_idx))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write previously-loaded V vectors to LDS."""
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vecs[batch])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vecs[batch])

        # Per-batch descriptors: fold batch_idx*seq_len into each tensor's 48-bit base and
        # bound num_records to ONE batch. Global indices are 0-based within the batch (see
        # global_idx_q/kv + DMA global_row), so the 32-bit voffset and the int32 C-ABI
        # shape field never see the whole-tensor numel (which can reach 2^31). OOB rows
        # within the batch still read 0 / drop on store (arbitrary-seqlen safe).
        _kv_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        _q_nrec_bytes = _raw(seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _q_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_Q * 2))
        _kv_batch_byte_off = _raw(batch_idx * seq_len_v * fx.Index(STRIDE_TOKEN_KV * 2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        # LSE[B, NUM_HEADS_Q, S] f32; batch base folded in like Q/O so the epilogue
        # writes 0-based (q_head, q_row) within this batch's [NUM_HEADS_Q, S] slab.
        _lse_per_batch = seq_len_v * fx.Index(NUM_HEADS_Q)
        _lse_nrec_bytes = _raw(_lse_per_batch * fx.Index(4))
        _lse_batch_byte_off = _raw(batch_idx * _lse_per_batch * fx.Index(4))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if const_expr(ENABLE_DMA):
            k_rsrc = buffer_ops.create_buffer_resource(
                K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            DMA_BYTES = 16  # buffer_load_dwordx4 = 16 bytes per lane
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_k(tile_start, buf_id=0):
                """Load K tile via DMA with XOR-swizzled global fetch."""
                if const_expr(isinstance(buf_id, int)):
                    k_lds_byte_base = lds_kv_base_idx + fx.Index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * fx.Index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (
                        k_lds_byte_base
                        + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_K_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    # Same bank-conflict swizzle as the read path (see _k_bank_mask):
                    # store logical col c at physical c^mask so read^mask round-trips.
                    xor_mask = _k_bank_mask(row_in_tile)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                        + kv_head_idx * fx.Index(HEAD_DIM * 2)
                        + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- V XOR swizzle: col ^ ((row & 3) << 4) at 16-element granularity ----
        def _v_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(0x3)) << fx.Index(4)
            return col_idx ^ mask

        # ---- DMA loading for V (buffer_load_dwordx4 ... lds) ----
        if const_expr(ENABLE_DMA):
            v_rsrc = buffer_ops.create_buffer_resource(
                V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                """Load V tile via DMA with XOR-swizzled global fetch."""
                v_lds_byte_base = lds_kv_base_idx + fx.Index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2)
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (
                        v_lds_byte_base
                        + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_V_ROW + fx.Index(d * ROWS_PER_DMA_BATCH_V)
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(0x3)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                        + kv_head_idx * fx.Index(HEAD_DIM * 2)
                        + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row, q_col)
            q_b_packs.append(buffer_ops.buffer_load(q_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)
        lane_xor_32_i32 = lane_i32 ^ shuf_32_i32
        lane_xor_32_byte = lane_xor_32_i32 * c4_i32

        def reduction_peer(v_f32):
            if const_expr(REDUCE_MODE == "ds_bpermute"):
                v_i32 = fx.Int32(ArithValue(v_f32).bitcast(fx.Int32.ir_type))
                peer_i32 = rocdl.ds_bpermute(fx.Int32.ir_type, lane_xor_32_byte, v_i32)
                return fx.Float32(ArithValue(peer_i32).bitcast(compute_type))
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound ----
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))
        else:
            kv_upper = seq_len_v

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(fx.Index(0))
            coop_dma_k(fx.Index(0), buf_id=0)
        # gfx950 frozen-basis path: the exp2 addend neg_max_arg is loop-invariant
        # after the tile-0 seed, so carry it (compute once in the seed branch) to
        # drop the per-tile mul/sub/fma recompute off the pre-exp2 critical path.
        if const_expr(USE_HW_TR):
            init_args.append(c_zero_f)

        def _kv_outer_body(kv_block_start, inner_iter_args, apply_mask):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _cur_buf_id = inner_iter_args[2 + D_CHUNKS] if _use_dma_dbuf else None
            if const_expr(USE_HW_TR):
                neg_max_arg_carried = inner_iter_args[2 + D_CHUNKS + (1 if _use_dma_dbuf else 0)]
            else:
                neg_max_arg_carried = None
            preload_k_count = NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES

            if const_expr(ENABLE_PREFETCH_3BUF):
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * BLOCK_N
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        coop_load_k(pre_k_start, pre_k_slot)
                if const_expr(ENABLE_DMA):
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N

                if const_expr(ENABLE_PREFETCH_3BUF):
                    k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                elif const_expr(_use_dma_dbuf):
                    if const_expr(kv_sub % 2 == 0):
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = fx.Index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = fx.Index(1) - _k_buf_id
                    if const_expr(kv_sub + 1 < N_SUBTILES):
                        coop_dma_k(
                            kv_block_start + (kv_sub + 1) * BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + fx.Index(BLOCK_N_OUT)
                        _has_next = _next_kv < kv_upper
                        if _has_next:
                            coop_dma_k(_next_kv, _next_k_buf_id)
                    rocdl.sched_barrier(0)
                    k_base = k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = k_buf_base(k_slot)

                if const_expr(not USE_HW_TR or (not ENABLE_DMA and not ENABLE_PREFETCH_3BUF)):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = K_SUB_N * K_STRIDE
                # Bank-conflict swizzle (see _k_bank_mask): must match the write/DMA
                # side exactly (XOR self-inverse). d64 -> ((row//2)%8)*8 (2-way floor).
                k_swz_mask = _k_bank_mask(lane_mod_32)

                # k_base/k_swz_mask/k_hi_offset are rebound once per kv_sub iteration and
                # both helpers are only ever called within that same iteration (never
                # stored past it), so the B023 late-binding warning is a false positive.
                def _k_idx_lo(ks):
                    col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)  # noqa: B023

                def _k_idx_hi(ks):
                    col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + k_hi_offset + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)  # noqa: B023

                _QK_PREFETCH_DEPTH = 2
                k_packs_lo = [None] * K_STEPS_QK
                k_packs_hi = [None] * K_STEPS_QK
                for p in range_constexpr(_QK_PREFETCH_DEPTH):
                    k_packs_lo[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                    k_packs_hi[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                if const_expr(ENABLE_DMA and not ENABLE_PREFETCH_3BUF):
                    coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                for ks in range_constexpr(K_STEPS_QK):
                    s_acc_lo = mfma_acc(k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                    s_acc_hi = mfma_acc(k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    if const_expr(ks + _QK_PREFETCH_DEPTH < K_STEPS_QK):
                        k_packs_lo[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)]
                        )
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)]
                        )

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(Vec(s_acc_lo)[r])
                    s_raw_hi.append(Vec(s_acc_hi)[r])

                if const_expr(CAUSAL):
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_div_32_i32 = fx.Int32(lane_div_32)
                    # Compile-time (caller-split): below-diagonal blocks pass
                    # apply_mask=False and skip the compare+select entirely.
                    tile_needs_mask = apply_mask
                    s_raw_lo_0 = s_raw_lo[0]
                    s_raw_lo_1 = s_raw_lo[1]
                    s_raw_lo_2 = s_raw_lo[2]
                    s_raw_lo_3 = s_raw_lo[3]
                    s_raw_lo_4 = s_raw_lo[4]
                    s_raw_lo_5 = s_raw_lo[5]
                    s_raw_lo_6 = s_raw_lo[6]
                    s_raw_lo_7 = s_raw_lo[7]
                    s_raw_lo_8 = s_raw_lo[8]
                    s_raw_lo_9 = s_raw_lo[9]
                    s_raw_lo_10 = s_raw_lo[10]
                    s_raw_lo_11 = s_raw_lo[11]
                    s_raw_lo_12 = s_raw_lo[12]
                    s_raw_lo_13 = s_raw_lo[13]
                    s_raw_lo_14 = s_raw_lo[14]
                    s_raw_lo_15 = s_raw_lo[15]
                    s_raw_hi_0 = s_raw_hi[0]
                    s_raw_hi_1 = s_raw_hi[1]
                    s_raw_hi_2 = s_raw_hi[2]
                    s_raw_hi_3 = s_raw_hi[3]
                    s_raw_hi_4 = s_raw_hi[4]
                    s_raw_hi_5 = s_raw_hi[5]
                    s_raw_hi_6 = s_raw_hi[6]
                    s_raw_hi_7 = s_raw_hi[7]
                    s_raw_hi_8 = s_raw_hi[8]
                    s_raw_hi_9 = s_raw_hi[9]
                    s_raw_hi_10 = s_raw_hi[10]
                    s_raw_hi_11 = s_raw_hi[11]
                    s_raw_hi_12 = s_raw_hi[12]
                    s_raw_hi_13 = s_raw_hi[13]
                    s_raw_hi_14 = s_raw_hi[14]
                    s_raw_hi_15 = s_raw_hi[15]

                    if tile_needs_mask:
                        lane_off_i32 = lane_div_32_i32 * fx.Int32(4)
                        kv_col_lo_0 = kv_start_i32 + lane_off_i32 + fx.Int32(0)
                        s_raw_lo_0 = ArithValue(kv_col_lo_0 > q_row_i32).select(c_neg_inf, s_raw_lo_0)
                        s_raw_hi_0 = ArithValue(kv_col_lo_0 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_0
                        )
                        kv_col_lo_1 = kv_start_i32 + lane_off_i32 + fx.Int32(1)
                        s_raw_lo_1 = ArithValue(kv_col_lo_1 > q_row_i32).select(c_neg_inf, s_raw_lo_1)
                        s_raw_hi_1 = ArithValue(kv_col_lo_1 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_1
                        )
                        kv_col_lo_2 = kv_start_i32 + lane_off_i32 + fx.Int32(2)
                        s_raw_lo_2 = ArithValue(kv_col_lo_2 > q_row_i32).select(c_neg_inf, s_raw_lo_2)
                        s_raw_hi_2 = ArithValue(kv_col_lo_2 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_2
                        )
                        kv_col_lo_3 = kv_start_i32 + lane_off_i32 + fx.Int32(3)
                        s_raw_lo_3 = ArithValue(kv_col_lo_3 > q_row_i32).select(c_neg_inf, s_raw_lo_3)
                        s_raw_hi_3 = ArithValue(kv_col_lo_3 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_3
                        )
                        kv_col_lo_4 = kv_start_i32 + lane_off_i32 + fx.Int32(8)
                        s_raw_lo_4 = ArithValue(kv_col_lo_4 > q_row_i32).select(c_neg_inf, s_raw_lo_4)
                        s_raw_hi_4 = ArithValue(kv_col_lo_4 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_4
                        )
                        kv_col_lo_5 = kv_start_i32 + lane_off_i32 + fx.Int32(9)
                        s_raw_lo_5 = ArithValue(kv_col_lo_5 > q_row_i32).select(c_neg_inf, s_raw_lo_5)
                        s_raw_hi_5 = ArithValue(kv_col_lo_5 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_5
                        )
                        kv_col_lo_6 = kv_start_i32 + lane_off_i32 + fx.Int32(10)
                        s_raw_lo_6 = ArithValue(kv_col_lo_6 > q_row_i32).select(c_neg_inf, s_raw_lo_6)
                        s_raw_hi_6 = ArithValue(kv_col_lo_6 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_6
                        )
                        kv_col_lo_7 = kv_start_i32 + lane_off_i32 + fx.Int32(11)
                        s_raw_lo_7 = ArithValue(kv_col_lo_7 > q_row_i32).select(c_neg_inf, s_raw_lo_7)
                        s_raw_hi_7 = ArithValue(kv_col_lo_7 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_7
                        )
                        kv_col_lo_8 = kv_start_i32 + lane_off_i32 + fx.Int32(16)
                        s_raw_lo_8 = ArithValue(kv_col_lo_8 > q_row_i32).select(c_neg_inf, s_raw_lo_8)
                        s_raw_hi_8 = ArithValue(kv_col_lo_8 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_8
                        )
                        kv_col_lo_9 = kv_start_i32 + lane_off_i32 + fx.Int32(17)
                        s_raw_lo_9 = ArithValue(kv_col_lo_9 > q_row_i32).select(c_neg_inf, s_raw_lo_9)
                        s_raw_hi_9 = ArithValue(kv_col_lo_9 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_9
                        )
                        kv_col_lo_10 = kv_start_i32 + lane_off_i32 + fx.Int32(18)
                        s_raw_lo_10 = ArithValue(kv_col_lo_10 > q_row_i32).select(c_neg_inf, s_raw_lo_10)
                        s_raw_hi_10 = ArithValue(kv_col_lo_10 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_10
                        )
                        kv_col_lo_11 = kv_start_i32 + lane_off_i32 + fx.Int32(19)
                        s_raw_lo_11 = ArithValue(kv_col_lo_11 > q_row_i32).select(c_neg_inf, s_raw_lo_11)
                        s_raw_hi_11 = ArithValue(kv_col_lo_11 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_11
                        )
                        kv_col_lo_12 = kv_start_i32 + lane_off_i32 + fx.Int32(24)
                        s_raw_lo_12 = ArithValue(kv_col_lo_12 > q_row_i32).select(c_neg_inf, s_raw_lo_12)
                        s_raw_hi_12 = ArithValue(kv_col_lo_12 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_12
                        )
                        kv_col_lo_13 = kv_start_i32 + lane_off_i32 + fx.Int32(25)
                        s_raw_lo_13 = ArithValue(kv_col_lo_13 > q_row_i32).select(c_neg_inf, s_raw_lo_13)
                        s_raw_hi_13 = ArithValue(kv_col_lo_13 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_13
                        )
                        kv_col_lo_14 = kv_start_i32 + lane_off_i32 + fx.Int32(26)
                        s_raw_lo_14 = ArithValue(kv_col_lo_14 > q_row_i32).select(c_neg_inf, s_raw_lo_14)
                        s_raw_hi_14 = ArithValue(kv_col_lo_14 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_14
                        )
                        kv_col_lo_15 = kv_start_i32 + lane_off_i32 + fx.Int32(27)
                        s_raw_lo_15 = ArithValue(kv_col_lo_15 > q_row_i32).select(c_neg_inf, s_raw_lo_15)
                        s_raw_hi_15 = ArithValue(kv_col_lo_15 + fx.Int32(K_SUB_N) > q_row_i32).select(
                            c_neg_inf, s_raw_hi_15
                        )

                    s_raw_lo = [
                        s_raw_lo_0,
                        s_raw_lo_1,
                        s_raw_lo_2,
                        s_raw_lo_3,
                        s_raw_lo_4,
                        s_raw_lo_5,
                        s_raw_lo_6,
                        s_raw_lo_7,
                        s_raw_lo_8,
                        s_raw_lo_9,
                        s_raw_lo_10,
                        s_raw_lo_11,
                        s_raw_lo_12,
                        s_raw_lo_13,
                        s_raw_lo_14,
                        s_raw_lo_15,
                    ]
                    s_raw_hi = [
                        s_raw_hi_0,
                        s_raw_hi_1,
                        s_raw_hi_2,
                        s_raw_hi_3,
                        s_raw_hi_4,
                        s_raw_hi_5,
                        s_raw_hi_6,
                        s_raw_hi_7,
                        s_raw_hi_8,
                        s_raw_hi_9,
                        s_raw_hi_10,
                        s_raw_hi_11,
                        s_raw_hi_12,
                        s_raw_hi_13,
                        s_raw_hi_14,
                        s_raw_hi_15,
                    ]
                else:
                    # Non-causal KV padding mask: keys with absolute column >= seq_len
                    # -> -inf, so OOB KV (0 or duplicated row) doesn't leak into softmax.
                    # Col layout (mirrors causal): lo = kv_start + lane_div_32*4 +
                    # ((r//4)*8 + r%4); hi = +K_SUB_N.
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
                    seq_len_i32 = fx.Int32(seq_len_v)
                    for r in range_constexpr(16):
                        _off = (r // 4) * 8 + (r % 4)
                        kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(_off)
                        s_raw_lo[r] = ArithValue(kv_col >= seq_len_i32).select(c_neg_inf, s_raw_lo[r])
                        s_raw_hi[r] = ArithValue(kv_col + fx.Int32(K_SUB_N) >= seq_len_i32).select(
                            c_neg_inf, s_raw_hi[r]
                        )

                # ==== Softmax basis: seed-once then freeze (VALU-issue lever) ====
                # PMC shows this kernel is regular-VALU-issue-bound (exp2 ~10% of
                # VALU); the per-tile row-max reduction (31 fmax + peer) is the
                # largest cuttable regular-VALU block. The ported lazy-rescale kept
                # the basis across tiles but still recomputed the max EVERY tile to
                # decide whether to rescale -- and for realistic score spreads a
                # rescale needs a >2^8 jump over the basis (~5.5 sigma), which never
                # happens after the first tile, so that max was dead VALU. Here the
                # basis is computed ONLY on the first KV subtile (m_running is the
                # -inf init only there; uniform ballot == exec branch) and frozen
                # afterwards. O/l are never rescaled (basis never changes), so the
                # output is bit-identical to the never-rescale path; the /l
                # normalization cancels the basis and P = exp2(s - basis0) stays far
                # inside f32 (score spread << 127 log2 units). Determinism holds: the
                # branch is data-independent (loop-position only). gfx942 (non-HW-TR)
                # keeps the original per-tile interleaved rescale.
                def _prescale_addend(m_b):
                    # Prescale the exp2 addend from the (frozen) basis. fast_exp2 folds
                    # *2^23+bias in so _exp2_of is a single fma; exact keeps plain -max.
                    _sm = _fmul(c_sm_scale_log2e, m_b)
                    _nsm = _fsub(c_zero_f, _sm)
                    if const_expr(fast_exp2):
                        return fmath.fma(_nsm, _c_exp2_scale, _c_exp2_bias, fastmath=fm_fast)
                    return _nsm

                if const_expr(USE_HW_TR):
                    _need_seed = ArithValue(fx.Float32(m_running) == c_neg_inf)
                    _bf = rocdl.ballot(T.i64, _raw(_need_seed))
                    _seed = arith.cmpi(arith.CmpIPredicate.eq, _raw(_bf), _read_exec_i64())
                    _f32ty = _raw(c_zero_f).type
                    _if = scf.IfOp(_seed, [_f32ty, _f32ty], has_else=True, loc=ir.Location.unknown())
                    with ir.InsertionPoint(_if.regions[0].blocks[0]):
                        _lmax = s_raw_lo[0]
                        for r in range_constexpr(15):
                            _lmax = _fmax(_lmax, s_raw_lo[r + 1])
                        for r in range_constexpr(16):
                            _lmax = _fmax(_lmax, s_raw_hi[r])
                        _rmax = _fmax(_lmax, reduction_peer(_lmax))
                        scf.YieldOp([_raw(_rmax), _raw(_prescale_addend(_rmax))])
                    if len(_if.regions[1].blocks) == 0:
                        _if.regions[1].blocks.append(*[])
                    with ir.InsertionPoint(_if.regions[1].blocks[0]):
                        scf.YieldOp([_raw(m_running), _raw(_prescale_addend(m_running))])
                    _res = list(_if.results)
                    m_basis = fx.Float32(_res[0])
                    neg_max_arg = fx.Float32(_res[1])
                    corr_vec = None
                else:
                    local_max = s_raw_lo[0]
                    for r in range_constexpr(15):
                        local_max = _fmax(local_max, s_raw_lo[r + 1])
                    for r in range_constexpr(16):
                        local_max = _fmax(local_max, s_raw_hi[r])
                    peer_max = reduction_peer(local_max)
                    row_max = _fmax(local_max, peer_max)
                    m_new_raw = _fmax(m_running, row_max)
                    diff_m_raw = _fsub(m_running, m_new_raw)
                    diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
                    corr = ArithValue(diff_m_scaled).exp2(fastmath=fm_fast)
                    corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
                    o_accs[0] = _fmul(Vec(o_accs[0]), corr_vec)
                    l_running = _fmul(corr, l_running)
                    m_basis = m_new_raw
                    neg_max_arg = _prescale_addend(m_basis)

                # ==== exp2 P + grouped pack (VGPR-liveness: cap live f32 P) ====
                # Interleave the Schraudolph exp2 with the bf16/fp16 pack one MFMA
                # pack group at a time, so only PV_GRP f32 P values are live at once
                # instead of all 32 (which previously coexisted with the 16-VGPR
                # packed operands). local_sum accumulates in-flight.
                PV_GRP = 16 // PV_K_STEPS

                def _pack_grp(grp_f32):
                    if const_expr(dtype_str == "bf16" and not USE_K16):
                        return bf16_trunc_pack_v4(grp_f32)
                    if const_expr(dtype_str == "bf16" and USE_K16):
                        return bf16_trunc_pack_v8(grp_f32)
                    p_f16 = [fx.Float32(x).to(elem_dtype) for x in grp_f32]
                    return Vec.from_elements(p_f16, elem_dtype).ir_value()

                local_sum = c_zero_f
                p_packs_lo = []
                p_packs_hi = []
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * PV_GRP
                    grp_lo = []
                    for j in range_constexpr(PV_GRP):
                        p_lo = _exp2_of(s_raw_lo[base + j], neg_max_arg, apply_clamp=apply_mask)
                        grp_lo.append(p_lo)
                        local_sum = _fadd(local_sum, p_lo)
                    p_packs_lo.append(_pack_grp(grp_lo))
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * PV_GRP
                    grp_hi = []
                    for j in range_constexpr(PV_GRP):
                        p_hi = _exp2_of(s_raw_hi[base + j], neg_max_arg, apply_clamp=apply_mask)
                        grp_hi.append(p_hi)
                        local_sum = _fadd(local_sum, p_hi)
                    p_packs_hi.append(_pack_grp(grp_hi))

                peer_sum = reduction_peer(local_sum)
                tile_sum = _fadd(local_sum, peer_sum)
                l_new = _fadd(l_running, tile_sum)

                if const_expr(ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                if const_expr(ENABLE_PREFETCH_3BUF):
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(ENABLE_DMA):
                    v_base = v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                # _steps/v_base are rebound once per kv_sub iteration; _read_v_pack is
                # only ever called within that same iteration (never stored past it), so
                # the B023 late-binding warning below is a false positive.
                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]  # noqa: B023
                    if const_expr(USE_HW_TR):
                        d_col = fx.Index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
                        k_row = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
                        _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * V_STRIDE + _d_col_eff  # noqa: B023
                        lds_hi = lds_lo + fx.Index(K_SUB_N * V_STRIDE)
                        if const_expr(USE_K16):
                            vl_a = ds_read_tr_v4f16(lds_lo)
                            vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * V_STRIDE))
                            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                            vh_a = ds_read_tr_v4f16(lds_hi)
                            vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * V_STRIDE))
                            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        else:
                            vl = ds_read_tr_v4f16(lds_lo)
                            vh = ds_read_tr_v4f16(lds_hi)
                    else:
                        d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
                        k_base = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4
                        v_lo_idx = v_base + d_pos * VT_STRIDE + k_base  # noqa: B023
                        v_hi_idx = v_lo_idx + fx.Index(K_SUB_N)
                        vl = Vec.load(v4f16_type, lds_kv, [v_lo_idx])
                        vh = Vec.load(v4f16_type, lds_kv, [v_hi_idx])
                    return vl, vh

                # Pre-read V for the first step.
                v_lo_cur, v_hi_cur = _read_v_pack(0)

                # ==== GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_nxt, v_hi_nxt = _read_v_pack(si + 1)
                    o_accs[dc] = mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
                    if const_expr(not USE_HW_TR and dc == 0 and pks < D_CHUNKS - 1):
                        o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_cur = v_lo_nxt
                        v_hi_cur = v_hi_nxt

                m_running = m_basis
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(N_SUBTILES % 2 == 1):
                    _yield_args.append(fx.Index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            if const_expr(USE_HW_TR):
                _yield_args.append(neg_max_arg)
            return _yield_args

        # Split the causal kv-loop at the diagonal: [0, q_start) is fully below the
        # diagonal (no mask, no exp2 clamp), [q_start, kv_upper) straddles it (mask
        # + clamp). q_start = q_tile_idx * BLOCK_M is a multiple of BLOCK_N_OUT, so
        # the split is exact and reproduces the per-tile tile_needs_mask decision.
        # This drops the mask compare+select AND the exp2 v_max clamp from every
        # below-diagonal outer block (the large majority), mirroring the bwd fused
        # kernel. buf_id/m/l/o loop-carry threads through both loops unchanged, and
        # the last unmasked iteration's DMA prefetch feeds the first masked tile.
        loop_results = init_args
        if const_expr(CAUSAL):
            for kv_block_start, inner_iter_args in range(0, q_start, BLOCK_N_OUT, init=init_args):
                loop_results = yield _kv_outer_body(kv_block_start, inner_iter_args, False)
            _tail_init = loop_results if isinstance(loop_results, (list, tuple)) else [loop_results]
            for kv_block_start, inner_iter_args in range(q_start, kv_upper, BLOCK_N_OUT, init=_tail_init):
                loop_results = yield _kv_outer_body(kv_block_start, inner_iter_args, True)
        else:
            for kv_block_start, inner_iter_args in range(0, kv_upper, BLOCK_N_OUT, init=init_args):
                loop_results = yield _kv_outer_body(kv_block_start, inner_iter_args, True)

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        # gfx950: pack 4 f32 -> 2 bf16 dwords (cvt_pk_bf16_f32), permlane32_swap fuses
        # each lane's 4 cols with its half-wave partner's -> 8 cols/store. O is
        # num_records-bounded (o_rsrc) -> partial-q-tile OOB rows drop.
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        # ---- Emit LSE[b, hq, q_row] = sm_scale * m_final + ln(l_final) ----
        # m_final (loop_results[0], raw unscaled QK max) and l_final are peer-reduced,
        # so lane_div_32 == 0 and == 1 hold bit-identical values. A single peer
        # (lane_div_32 == 0) writes each q_row -> deterministic, no atomics (det gate).
        # Fully-masked rows (l_final <= 0) write -inf so the backward's exp(S - lse)
        # stays well defined (pitfalls/04 attention neutral value).
        m_final = loop_results[0]
        _lse_ln = fmath.log(l_final, fastmath=fm_fast)
        _lse_val = _fadd(_fmul(fx.Float32(sm_scale), m_final), _lse_ln)
        _lse_out = ArithValue(l_final > c_zero_f).select(_lse_val, _raw(c_neg_inf))
        if lane_div_32 == fx.Index(0):
            _lse_elem = q_head_idx * seq_len_v + q_row
            buffer_ops.buffer_store(
                fx.Float32(_lse_out),
                lse_rsrc,
                _lse_elem * fx.Index(4),
                mask=ArithValue(q_row < seq_len_v),
                offset_is_bytes=True,
            )

        inv_l = rocdl.rcp(T.f32, l_final)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        v_o = [Vec(o_finals[dc]) * inv_l_vec for dc in range_constexpr(D_CHUNKS)]

        if const_expr(USE_PERMLANE_OSTORE):
            # gfx950: 128-bit permlane-fused store (cvt_pk_bf16_f32 + permlane32_swap).
            pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            is_hi_half = ArithValue(lane_div_32 != fx.Index(0))

            def _o_pack_2dw(dc, store_group):
                # 4 f32 outputs -> 2 packed-16bit dwords (lo = cols 0,1; hi = cols 2,3).
                r_base = store_group * 4
                if const_expr(dtype_str == "bf16"):
                    lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
                    hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
                    return lo, hi
                o_f16 = [fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype) for i in range_constexpr(4)]
                pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                return _raw(pack[0]), _raw(pack[1])

            def _swap_halves(dw):
                # permlane32_swap(a,b) -> (a.lo|b.lo, a.hi|b.hi); with a=b=dw the
                # partner dword dw[lane^32] is result[1] on low lanes, [0] on high.
                swapped = rocdl.permlane32_swap(pair_i32_ty, _raw(dw), _raw(dw), False, False)
                lo_res = llvm.extractvalue(T.i32, swapped, [0])
                hi_res = llvm.extractvalue(T.i32, swapped, [1])
                return is_hi_half.select(lo_res, hi_res)

            for dc in range_constexpr(D_CHUNKS):
                for g in range_constexpr(2):
                    d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                    d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                    # low lanes: own group-2g cols 0-3 ++ partner's cols 4-7;
                    # high lanes: partner's group-(2g+1) cols 0-3 ++ own cols 4-7.
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
                    buffer_ops.buffer_store(o_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)
        else:
            # gfx942 fallback (no permlane32_swap / cvt_pk_bf16_f32): each lane stores
            # its 16 cols as 4 dwordx2 groups via .to(elem_dtype); col map d_col =
            # dc*D_CHUNK + lane_div_32*4 + 8*grp + r. num_records bound drops OOB rows.
            for dc in range_constexpr(D_CHUNKS):
                for grp in range_constexpr(4):
                    r0 = grp * 4
                    o_f16 = [fx.Float32(Vec(v_o[dc])[r0 + i]).to(elem_dtype) for i in range_constexpr(4)]
                    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                    o2 = Vec.from_elements([_raw(pack[0]), _raw(pack[1])], fx.Int32)
                    d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                    o_global = global_idx_q(q_row, d_col)
                    buffer_ops.buffer_store(o2, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

    @flyc.jit
    def launch_flash_attn_generic(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
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
        flash_attn_generic_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                    if const_expr(flat_work_group_size is not None)
                    else None
                ),
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Tuned against ROCm/llvm-project `felix/tune_fmha` (c8cf6da43); other LLVM
    # revisions still compile/run correctly but may leave throughput on the table.
    # enable-post-misched=True: this d64 path is latency/dependency-bound, so the
    # post-RA scheduler co-issues independent QK/PV MFMAs into the exp2 VALU
    # shadow -- a reorder of independent ops only, bit-identical output.
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": True,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_generic(*args, **kwargs)

    def _compile(*args):
        # Eager one-time compile (mirrors flydsl/gemm's _get_compiled_dense): the
        # returned object is called with the SAME positional args, incl. the raw
        # torch stream in the trailing fx.Stream slot.
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(launch_flash_attn_generic, *args)

    _launch.compile = _compile

    return _launch
