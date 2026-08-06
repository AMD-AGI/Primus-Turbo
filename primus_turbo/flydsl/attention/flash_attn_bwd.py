# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn backward kernel builders for FlyDSL (gfx950 / MI355X).

Three deterministic kernels -- odo (identity delta), dkdv (KV-outer), dq
(Q-outer). Each work-group owns one output tile and writes it once, so there
are no float atomics. Built on the verified forward machine.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LOG2E = host_math.log2(host_math.e)

# dkdv MFMA-accumulator AGPR forcing (amdgpu-agpr-alloc): only pays off once the body
# is VGPR-lean, so it is disabled on the current body. Layout-agnostic.
_DKDV_AGPR = 0


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


def _cached_launch(cache, jit_fn, hints, args, kwargs):
    """Reuse the compiled artifact across calls that share a scalar signature."""
    if kwargs:
        if hints is None:
            return jit_fn(*args, **kwargs)
        with CompilationContext.compile_hints(hints):
            return jit_fn(*args, **kwargs)
    key = tuple(type(a) for a in args[:-1] if not isinstance(a, torch.Tensor))
    fn = cache.get(key)
    if fn is None:
        if len(cache) >= 64:
            cache.clear()
        if hints is None:
            fn = flyc.compile(jit_fn, *args)
        else:
            with CompilationContext.compile_hints(hints):
                fn = flyc.compile(jit_fn, *args)
        cache[key] = fn
    return fn(*args)


def dtype_to_elem_type(dtype_str):
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "f16":
        return fx.Float16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'bf16' or 'f16')")


def build_flash_attn_bwd_odo_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    num_kv_heads=None,
    causal=True,
    sm_scale=None,
    waves_per_eu=4,
    block=256,
    sbhd=False,  # SBHD [S,B,H,D] native O/dO layout (seq-step = B*H*D)
):
    """Identity-delta ("odo") kernel: DELTA[b,hq,s] = -sum_d O[b,s,hq,d]*dO[b,s,hq,d].

    One thread owns one (b,s,hq) row and stores the negated scalar (the dkdv/dq fold
    convention) to the transposed [B,Hq,S] delta. waves_per_eu=4: the hoisted
    whole-row O/dO loads do not fit the wpe=8 register budget.
    """
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

        # Decompose the flat row = ((b*S + s)*Hq + hq) once, up front: THD packs
        # O/dO as [B,S,Hq,D] (base = row*D) but SBHD is [S,B,Hq,D] so the element
        # base must be ((s*B + b)*Hq + hq)*D. DELTA stays batch-major [B,Hq,S].
        hq = row_c % fx.Index(NUM_HEADS_Q)
        tmp = row_c // fx.Index(NUM_HEADS_Q)
        s = tmp % sl
        b = tmp // sl
        if const_expr(sbhd):
            base = ((s * fx.Index(batch_size) + b) * fx.Index(NUM_HEADS_Q) + hq) * fx.Index(HEAD_DIM)
        else:
            base = row_c * fx.Index(HEAD_DIM)
        # Hoist the whole row's O/dO loads ahead of the reduction so all NVEC*2 dwordx4
        # loads are in flight before the first is consumed. Accumulate order is unchanged,
        # so the fp32 sum stays bit-identical (det-safe).
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

        # DELTA is transposed [B,Hq,S]: delta[b,hq,s] at (b*Hq + hq)*S + s.
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

    _compiled: dict = {}

    def _launch(*args, **kwargs):
        return _cached_launch(_compiled, launch_flash_attn_bwd_odo, None, args, kwargs)

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
    window_left=-1,
    fold_lse=None,  # None = fold on the hw-exp path only (see below)
    batch_size=None,  # compile-time B; required for SBHD seq-step stride bake
    sbhd=False,  # SBHD [S,B,H,D] native layout (seq-step = B*H*D)
    agpr=_DKDV_AGPR,  # force N MFMA accumulators into AGPRs (0 disables); layout-agnostic
    # exp_intrin: FOLD bulk exp via exp2 intrinsic anchor (vs v_min+dead-op asm) -- a
    # win once the body is VGPR-lean (spill-free). Layout-agnostic.
    exp_intrin=True,
    # g2d: GEMM2 transpose-read prefetch depth (ring across dt). Depth-1 wins -- a deeper
    # ring's extra live transpose-reads outweigh the latency it hides on this body.
    g2d=1,
    varlen=False,  # ragged / block-causal: per-segment [tok_base,tok_end) from cu_seqlens
):
    """Build the dK/dV KV-outer backward launcher (clean mirror of the forward).

    One work-group owns BLOCK_KV rows of one kv-head and loops the GQA group's
    q-heads and causal q-blocks, accumulating dK/dV in registers. q_split splits
    the q-loop deterministically: cyclic subsets, reduced afterwards.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "bwd dkdv kernel targets gfx950"
    assert dtype_str == "bf16", "bwd dkdv kernel targets bf16"
    assert causal, "bwd dkdv kernel is causal-only for the GPT-OSS campaign"

    # Prescale the owned K by sm*log2e and fold -log2e*lse into GEMM1a's MFMA C-init, so its
    # accumulator already IS the base-2 softmax exponent. Not combinable with Schraudolph:
    # its lse*2^23+bias addend loses the low mantissa bits through the f32 MFMA accumulator.
    if fold_lse is None:
        fold_lse = True

    # buffer_load_dwordx4 ... lds (16B DMA-to-LDS) needs gfx950+ (gfx94x has only
    # the 4B dword variant). DMA bypasses the VGPR staging of the Q/dO tile loads,
    # relieving register pressure on this VGPR-locked (236 VGPR, occ ~2) kernel.
    ENABLE_DMA = enable_dma and not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    BLOCK_Q = 64
    WARP_SIZE = 64
    NUM_XCD = 8  # gfx950 XCDs; the dispatcher hands block_id to xcd = block_id % NUM_XCD
    BLOCK_KV = block_kv
    Q_SPLIT = q_split
    assert q_split >= 1
    flat_work_group_size = 256
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_KV = BLOCK_KV // NUM_WAVES

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32): four independent 16x16 accumulator
    # chains at the same accumulator VGPR total (dkdv is MFMA dep-wait bound). Lane layout:
    # lane%16 = M/N index, lane//16 = K-subgroup (4 x 8 = K32) and, on the C output, the
    # M-block ((lane//16)*4 + t, t in 0..3 -> 4 f32/lane). ----
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

    # sched_barrier(TRANS) pins MFMA/ds_read/VALU in place and frees only the softmax's
    # quarter-rate v_exp to migrate, so the exps are what fills GEMM1b's MFMA latency shadow
    # (schedule-only: opcode multiset and output unchanged).
    SCHED_TRANS = 0x400  # LLVM SchedGroupMask: TRANS (v_exp)

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

    # LDS staging region for (-delta, lse) of the whole GQA group's q-block. One
    # cooperative vec fetch per array (LD_ARR/BLOCK_SIZE loads) replaces MT per-head
    # buffer_loads carried in registers, and each use point re-reads straight from
    # LDS -- removing the +MT*2 v4f32 register carry that pinned dkdv at spill.
    # Layout-agnostic (delta/lse are [B,Hq,S] batch-major in both THD and SBHD).
    LD_HEAD_ELEMS = BLOCK_Q
    LD_ARR_ELEMS = GQA_GROUP_SIZE * LD_HEAD_ELEMS
    LD_ELEMS = 2 * LD_ARR_ELEMS
    LD_THREADS_PER_HEAD = BLOCK_SIZE // GQA_GROUP_SIZE
    LD_VEC = LD_HEAD_ELEMS // LD_THREADS_PER_HEAD
    assert BLOCK_SIZE % GQA_GROUP_SIZE == 0 and LD_HEAD_ELEMS % LD_THREADS_PER_HEAD == 0
    ld_off = allocator._align(allocator.ptr, 16)
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
        CuSeqQ: fx.Tensor,  # varlen: cu_seqlens_q [num_seg+1] i32; else unused placeholder slot
        CuSeqKv: fx.Tensor,  # varlen: cu_seqlens_kv [num_seg+1] i32; else unused placeholder slot
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        total_kv: fx.Int32,  # varlen: sum of kv seglens (packed dk/dv workspace split stride); else unused
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

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_16x16x32_bf16, a, b, c)

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)
        causal_off_i32 = fx.Int32(seq_len_k) - fx.Int32(seq_len_q)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16  # M/N index within a 16-tile
        kg = lane // 16  # 0..3: K-subgroup (inputs) / M-block (C output)

        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_off
            ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # block_id decode. The dispatcher round-robins work-groups over the XCDs and each XCD
        # owns a private L2 slice, so XCD-major decode gives each XCD a whole (batch, kv-head)
        # chunk: its L2 streams one kv-head's K/V and the GQA work-groups reading byte-identical
        # rows stay co-resident. Bijective when B*NUM_HEADS_KV % NUM_XCD == 0, which
        # NUM_HEADS_KV % NUM_XCD == 0 guarantees; other head counts keep the plain decode.
        num_kv_tiles = (seq_len_k_v + BLOCK_KV - 1) // BLOCK_KV
        if const_expr(NUM_HEADS_KV % NUM_XCD == 0):
            _xcd = block_id % fx.Index(NUM_XCD)
            _slot = block_id // fx.Index(NUM_XCD)
            if const_expr(Q_SPLIT > 1):
                split_idx = _slot % fx.Index(Q_SPLIT)
                _slot = _slot // fx.Index(Q_SPLIT)
            else:
                split_idx = fx.Index(0)
            kv_tile_idx = _slot % num_kv_tiles
            _u = _slot // num_kv_tiles
            _bkv = _u * fx.Index(NUM_XCD) + _xcd
            kv_head_idx = _bkv % NUM_HEADS_KV
            batch_idx = _bkv // NUM_HEADS_KV
        else:
            kv_head_idx = block_id % NUM_HEADS_KV
            _rest = block_id // NUM_HEADS_KV
            if const_expr(Q_SPLIT > 1):
                split_idx = _rest % fx.Index(Q_SPLIT)
                _rest = _rest // fx.Index(Q_SPLIT)
            else:
                split_idx = fx.Index(0)
            kv_tile_idx = _rest % num_kv_tiles
            batch_idx = _rest // num_kv_tiles
        # SHADOW seq_len_q_v/k_v to the per-segment length so downstream base/SRD/loop-bounds follow the segment (byte-identical when uniform; grid tiles were fixed from max above).
        if const_expr(varlen):
            _seg = batch_idx
            _cuq_rsrc = buffer_ops.create_buffer_resource(CuSeqQ, max_size=True)
            _cukv_rsrc = buffer_ops.create_buffer_resource(CuSeqKv, max_size=True)
            _qb_i = fx.Int32(buffer_ops.buffer_load(_cuq_rsrc, _seg, vec_width=1, dtype=fx.Int32))
            _qe_i = fx.Int32(
                buffer_ops.buffer_load(_cuq_rsrc, _seg + fx.Index(1), vec_width=1, dtype=fx.Int32)
            )
            _kb_i = fx.Int32(buffer_ops.buffer_load(_cukv_rsrc, _seg, vec_width=1, dtype=fx.Int32))
            _ke_i = fx.Int32(
                buffer_ops.buffer_load(_cukv_rsrc, _seg + fx.Index(1), vec_width=1, dtype=fx.Int32)
            )
            q_tok_base = fx.Index(_qb_i)
            kv_tok_base = fx.Index(_kb_i)
            seq_len_q_v = fx.Index(_qe_i) - q_tok_base
            seq_len_k_v = fx.Index(_ke_i) - kv_tok_base
            causal_off_i32 = (_ke_i - _kb_i) - (_qe_i - _qb_i)
        else:
            q_tok_base = batch_idx * seq_len_q_v
            kv_tok_base = batch_idx * seq_len_k_v
        causal_offset = seq_len_k_v - seq_len_q_v
        kv_start = kv_tile_idx * BLOCK_KV
        # This wave owns ROWS_PER_WAVE_KV kv rows, split into NT 16-wide N-tiles.
        # In the 16x16 layout the owned kv row for a lane is nt*16 + lane16.
        kv_row_wave = kv_start + wave_id * ROWS_PER_WAVE_KV

        def global_idx_kv(token_idx, col):
            return token_idx * RD_STRIDE_KV + kv_head_idx * HEAD_DIM + col

        def kv_row_of(nt):
            return kv_row_wave + fx.Index(nt * N_TILE) + lane16

        def kv_row_i32_of(nt):
            return fx.Int32(kv_row_of(nt))

        # Per-batch base (elements). SBHD: batch inside the seq axis -> base is only
        # H*D. THD: dense per-batch block -> base is seq*H*D.
        if const_expr(sbhd):
            _q_ptr_batch_off = batch_idx * fx.Index(STRIDE_TOKEN_Q)
        else:
            _q_ptr_batch_off = q_tok_base * fx.Index(STRIDE_TOKEN_Q)
        q_ptr = buffer_ops.get_element_ptr(q_ptr, _q_ptr_batch_off, elem_type=elem_type)
        do_ptr = buffer_ops.get_element_ptr(do_ptr, _q_ptr_batch_off, elem_type=elem_type)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx_q(token_idx, col, q_head):
            return token_idx * RD_STRIDE_Q + q_head * HEAD_DIM + col

        def _q_row_clamp(row_idx):
            last = seq_len_q_v - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seq_len_q_v).select(row_idx, last))

        def _load_global_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

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
            _kv_batch_byte_off = _raw(kv_tok_base * fx.Index(STRIDE_TOKEN_KV * 2))
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
        elif const_expr(varlen):
            # Packed [q_split,total_kv,Hkv,D]: slot base = (split*total_kv + kv_tok_base); host sum(dim=0) -> packed dk/dv.
            _dkv_ws_byte_off = _raw(
                (split_idx * fx.Index(total_kv) + kv_tok_base) * fx.Index(STRIDE_TOKEN_KV * 2)
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
        if const_expr(varlen):
            _lse_batch_byte_off = _raw(q_tok_base * fx.Index(NUM_HEADS_Q) * fx.Index(4))
        else:
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

            def _dma_bases(tile_start):
                """Head-independent part of the Q/dO DMA byte offset, one per batch.

                Only the q_head term differs between GQA heads sharing a q-block, so hoisting
                the row/swizzle/column derivation collapses each head's DMA to a single add
                and takes the kernel's scratch spill to zero.
                """
                bases = []
                for d in range_constexpr(NUM_DMA_Q):
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
                    bases.append(global_row * fx.Index(RD_STRIDE_Q * 2) + col_byte)
                return bases

            q_lds_ptrs = _dma_lds_ptrs(lds_base_idx)
            do_lds_ptrs = _dma_lds_ptrs(lds_base_idx + fx.Index(LDS_DO_BASE * 2))

            def coop_dma_tile(src_rsrc, lds_ptrs, bases, q_head):
                """DMA a BLOCK_Q x head_dim Q/dO tile into the swizzled LDS layout.

                Address math is recomputed per tile on purpose: keeping the offsets live
                across the k_tr peak pushes VGPRs past the occ-2 boundary.
                """
                _qoff = q_head * fx.Index(HEAD_DIM * 2)
                for d in range_constexpr(NUM_DMA_Q):
                    rocdl.raw_ptr_buffer_load_lds(
                        src_rsrc,
                        lds_ptrs[d],
                        _dma_size,
                        fx.Int32(bases[d] + _qoff),
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
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

        # ---- FOLD: prescale the owned K by sm*log2e once per kv-block (amortized over
        # the GQA group's heads). K feeds GEMM1a only -- dK is a separate accumulator --
        # so scaling k_b_packs is safe. Together with -log2e*lse folded into GEMM1a's
        # C-init, GEMM1a's raw output already IS the base-2 softmax exponent. ----
        if const_expr(fold_lse):
            _kscale_v8 = Vec.filled(MFMA_LANE_K, sm_scale * _LOG2E, fx.Float32)
            for nt in range_constexpr(NT):
                for ks in range_constexpr(K_STEPS_QK):
                    k_b_packs[nt][ks] = (
                        (Vec(k_b_packs[nt][ks]).to(fx.Float32) * _kscale_v8).to(elem_dtype).ir_value()
                    )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        def _vexp(x):
            # Bare v_exp_f32 (hardware 2^x), NON-side-effecting -> the compiler overlaps
            # it into the GEMM MFMA bubbles naturally. No ldexp (softmax diff <= 0, so
            # 2^diff is in (0,1] and needs no range reduction).
            return fx.Float32(
                llvm.inline_asm(
                    ir.F32Type.get(), [_raw(x)], "v_exp_f32 $0, $1", "=v,v", has_side_effects=False
                )
            )

        def _vexp_after(x, dep):
            # Same v_exp plus a DEAD input operand ($2 is unreferenced by the asm text,
            # so nothing is emitted for it) whose only job is to order this read after
            # `dep` -- the FOLD hazard anchor in _head_step_lds needs one compiler-visible
            # read of the MFMA accumulator to buy the wait states for later reads.
            return fx.Float32(
                llvm.inline_asm(
                    ir.F32Type.get(),
                    [_raw(x), _raw(dep)],
                    "v_exp_f32 $0, $1",
                    "=v,v,v",
                    has_side_effects=False,
                )
            )

        def _vexp_intrin(x):
            # Backend-visible 2^x: emits the same v_exp_f32 but, being a recognised VALU
            # op rather than opaque inline asm, it IS a compiler-visible read of the MFMA
            # accumulator -- so it carries the MFMA->VALU hazard itself and anchors the v4
            # at no extra instruction (replaces the v_min anchor + _vexp_after dead-operand
            # trick with zero added VALU on the exp-issue-bound critical path).
            return fx.Float32(
                llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", [_raw(x)], [], [])
            )

        def _p_of(s_r, lse_t, apply_mask):
            if const_expr(fold_lse):
                assert apply_mask, "FOLD bulk uses the hazard-anchored path in _head_step_lds"
                # FOLD: masked (diagonal) tiles keep a ZERO C-init, so lse is added by this fma,
                # which doubles as the compiler-visible plain-VALU accumulator read that buys the
                # MFMA hazard wait states. Do NOT fold lse into the masked C-init and drop it.
                s_r = fmath.fma(s_r, fx.Float32(1.0), lse_t, fastmath=fm_fast)
                return _vexp(s_r)
            # Exact path (fold_lse=False) expects lse_t = plain -log2e*lse, so
            # diff = log2e*(s*sm - lse) is the true base-2 softmax exponent.
            diff = fmath.fma(s_r, c_sm_scale_log2e, lse_t, fastmath=fm_fast)
            return fx.Float32(
                llvm.inline_asm(
                    ir.F32Type.get(), [_raw(diff)], "v_exp_f32 $0, $1", "=v,v", has_side_effects=False
                )
            )

        # A-operand read (Q/dO from LDS): A[m=q=lane16][k=D=kg*8+s]. mt selects the
        # 16-q tile (row = mt*16 + lane16), ks the D 32-step (D = ks*32 + kg*8).
        a_swz_mask = (lane16 & fx.Index(7)) << fx.Index(4)

        def _a_idx(a_base, mt, ks):
            row = fx.Index(mt * M_TILE) + lane16
            col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
            return a_base + _pblk(row) * fx.Index(PBLK) + (col ^ a_swz_mask)

        def _keepalive_v4(v4list):
            """Pin the -lse C-init registers live past GEMM1a.

            Without a later use the RA may reuse them as a later nt's MFMA output D while
            an earlier nt still reads them as C -- a WAR the hardware cannot guard. Empty
            side-effecting asm: no instruction emitted, liveness constraint only.
            """
            for v4 in v4list:
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [_raw(fx.Float32(Vec(v4)[t])) for t in range_constexpr(4)],
                    "",
                    "=v,v,v,v,v",
                    has_side_effects=True,
                )

        def _gemm_qk(a_base, b_packs, inits=None, mts=None):
            """S[mt][nt] (v4f32) = A(Q/dO)[mt] @ B(owned K/V)[nt]^T over D. A is
            loaded once per (mt,ks) and reused across nt. inits[mt] optionally
            pre-loads the accumulator (folds -delta into the dP GEMM for free).
            mts restricts work to a subset of the MT q-tiles (per-half GEMM1); the
            output is keyed by mt so [2,3] halves index correctly."""
            _mts = list(range_constexpr(MT)) if mts is None else list(mts)
            a = {
                mt: [
                    Vec.load(mfma_pack_type, lds, [_a_idx(a_base, mt, ks)])
                    for ks in range_constexpr(K_STEPS_QK)
                ]
                for mt in _mts
            }
            out = {mt: [None] * NT for mt in _mts}
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
            (q=pks*32+16+kg*4+j)."""
            col = fx.Index(dt * D_TILE) + (lane % fx.Index(4)) * fx.Index(4)
            row0 = fx.Index(pks * PV_K_STEP) + kg * fx.Index(4) + (lane16 // fx.Index(4))
            row1 = row0 + fx.Index(N_TILE)
            v0 = ds_read_tr_v4f16(a_base + _pblk(row0) * fx.Index(PBLK) + _swizzle(row0, col))
            v1 = ds_read_tr_v4f16(a_base + _pblk(row1) * fx.Index(PBLK) + _swizzle(row1, col))
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

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
        # Masked q-blocks this split visits = ceil((_masked_upper - _q_loop_start)/_step): the
        # masked band is BLOCK_KV wide, so for q_split=1 (_step=BLOCK_Q < band) it spans more
        # than one q-block and a plain "+_step" would reprocess a diagonal block unmasked.
        # for every q_split and reduces to the old value when the band is one block wide.
        _masked_span = ArithValue(_masked_upper > _q_loop_start).select(
            _masked_upper - _q_loop_start, fx.Index(0)
        )
        _unmask_start = _q_loop_start + ((_masked_span + fx.Index(_step - 1)) // fx.Index(_step)) * fx.Index(
            _step
        )

        # The GQA head axis is unrolled INSIDE each q_start body so head h+1's GEMM1/exp2 is
        # emitted in the same straight-line block as head h's GEMM2 and schedules into its
        # MFMA shadow; accumulating dv/dk across heads is a pure reassociation (det-neutral).
        ld_lds = SmemPtr(base_ptr, ld_off, fx.Float32.ir_type, shape=(LD_ELEMS,)).get()
        # Thread t owns LD_VEC consecutive q of one GQA head.
        _ld_head = tid // fx.Index(LD_THREADS_PER_HEAD)
        _ld_q = (tid % fx.Index(LD_THREADS_PER_HEAD)) * fx.Index(LD_VEC)

        def _stage_ld_issue(q_start):
            # Issued BEFORE the Q/dO DMA so both HBM streams are in flight together;
            # the LDS commit lands after the DMA, so its vmcnt wait does not serialise
            # them (gfx950 has no vmcnt subset wait, but the counter is in-order).
            if const_expr(varlen):
                # Packed [total_q,Hq]: consecutive q for a fixed head are stride-NUM_HEADS_Q apart, so gather scalars (uniform head-major loads a single vec below).
                _qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + _ld_head
                _q0 = q_start + _ld_q
                return [
                    Vec.from_elements(
                        [
                            fx.Float32(
                                buffer_ops.buffer_load(
                                    rsrc,
                                    (_q0 + fx.Index(j)) * fx.Index(NUM_HEADS_Q) + _qh,
                                    vec_width=1,
                                    dtype=fx.Float32,
                                )
                            )
                            for j in range_constexpr(LD_VEC)
                        ],
                        fx.Float32,
                    ).ir_value()
                    for rsrc in (delta_rsrc, lse_rsrc)
                ]
            _g = (kv_head_idx * fx.Index(GQA_GROUP_SIZE) + _ld_head) * seq_len_q_v + q_start + _ld_q
            return [
                buffer_ops.buffer_load(rsrc, _g, vec_width=LD_VEC, dtype=fx.Float32)
                for rsrc in (delta_rsrc, lse_rsrc)
            ]

        def _stage_ld_commit(vals):
            _lds_i = _ld_head * fx.Index(LD_HEAD_ELEMS) + _ld_q
            for arr in range_constexpr(2):
                Vec(vals[arr]).store(ld_lds, [fx.Index(arr * LD_ARR_ELEMS) + _lds_i])

        def _ld_read(head_local, mt, arr):
            # v4f32 at q = head's q-block + mt*M_TILE + kg*4 (+t), matching the GEMM1
            # accumulator C layout; lane16 is absent -> a 16-way LDS broadcast.
            # arr=0 -> -delta (GEMM1b init), arr=1 -> prescaled lse (GEMM1a init/masked add).
            return Vec.load(
                v4f32_type,
                ld_lds,
                [fx.Index(arr * LD_ARR_ELEMS + head_local * LD_HEAD_ELEMS + mt * M_TILE) + kg * fx.Index(4)],
            ).ir_value()

        def _head_step_lds(q_start, apply_mask, head_local, dv_cur, dk_cur, bases=None):
            sb_bulk = not apply_mask  # exps only exist on these paths
            q_head = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            q_start_i32 = fx.Int32(q_start)
            kg_off_i32 = fx.Int32(kg) * fx.Int32(4)
            # Stage the whole GQA group's (-delta, lse) into LDS on head 0, riding
            # head 0's existing barrier pair; heads 1..7 re-read straight from LDS.
            _ldv = None
            if const_expr(head_local == 0):
                _ldv = _stage_ld_issue(q_start)
            gpu.barrier()  # WAR: guard prior head's LDS reads before this head's DMA
            if const_expr(ENABLE_DMA):
                coop_dma_tile(q_rsrc, q_lds_ptrs, bases, q_head)
                coop_dma_tile(do_rsrc, do_lds_ptrs, bases, q_head)
                if const_expr(head_local == 0):
                    _stage_ld_commit(_ldv)
                rocdl.s_waitcnt(0)
            else:
                _coop_load(q_ptr, fx.Index(0), q_start, q_head)
                _coop_load(do_ptr, fx.Index(LDS_DO_BASE), q_start, q_head)
                if const_expr(head_local == 0):
                    _stage_ld_commit(_ldv)
            gpu.barrier()  # DMA + ld_lds commit visible before GEMM1 reads

            # GEMM1a/exp2/GEMM1b/dS/pack per q-HALF (one pks = two mt packing into one
            # GEMM2 K=32 step): processing 2 of the MT q-tiles at a time halves the live
            # S/dP/P/dS transient that pinned dkdv at spill, so the kernel fits spill-free.
            # lse/-delta are pulled from LDS at their use points (only the 2 v4f32 this
            # half consumes are ever live). Pure re-ordering -> bit-identical, det-neutral.
            p_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            ds_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            do_ring, q_ring = None, None
            for pks in range_constexpr(PV_K_STEPS):
                ma, mb = 2 * pks, 2 * pks + 1
                half = [ma, mb]
                if const_expr(fold_lse and not apply_mask):
                    # FOLD unmasked: prescaled -log2e*lse is GEMM1a's C-init, so the
                    # accumulator already IS the base-2 softmax exponent.
                    s_tiles = _gemm_qk(
                        fx.Index(0),
                        k_b_packs,
                        inits={mt: _ld_read(head_local, mt, 1) for mt in half},
                        mts=half,
                    )
                else:
                    s_tiles = _gemm_qk(fx.Index(0), k_b_packs, mts=half)
                if const_expr(sb_bulk and not exp_intrin):
                    rocdl.sched_barrier(SCHED_TRANS)

                P = [[None] * NT for _ in range_constexpr(MT)]
                if const_expr(fold_lse and not apply_mask):
                    for mt in half:
                        for nt in range_constexpr(NT):
                            s_v = Vec(s_tiles[mt][nt])
                            if const_expr(exp_intrin):
                                P[mt][nt] = [_vexp_intrin(fx.Float32(s_v[t])) for t in range_constexpr(4)]
                            else:
                                _smin_anchor = fx.Float32(
                                    arith.minimumf(_raw(fx.Float32(s_v[0])), _raw(c_zero_f))
                                )
                                P[mt][nt] = [_vexp(_smin_anchor)] + [
                                    _vexp_after(fx.Float32(s_v[t]), _smin_anchor)
                                    for t in range_constexpr(1, 4)
                                ]
                else:
                    for mt in half:
                        lse_v = _ld_read(head_local, mt, 1)
                        for nt in range_constexpr(NT):
                            s_v = s_tiles[mt][nt]
                            p_vals = []
                            for t in range_constexpr(4):
                                s_r = fx.Float32(Vec(s_v)[t])
                                if const_expr(apply_mask):
                                    q_slot = q_start_i32 + kg_off_i32 + fx.Int32(mt * M_TILE + t)
                                    _up = ArithValue(kv_row_i32_of(nt) > q_slot + causal_off_i32)
                                    if const_expr(window_left >= 0):  # W+1 keys: mask kv <= q+off-(W+1)
                                        _lo = ArithValue(
                                            kv_row_i32_of(nt)
                                            <= q_slot + causal_off_i32 - fx.Int32(window_left + 1)
                                        )
                                        _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                                    else:
                                        _mm = _up
                                    s_r = _mm.select(c_neg_inf, s_r)
                                p_vals.append(_p_of(s_r, fx.Float32(Vec(lse_v)[t]), apply_mask))
                            P[mt][nt] = p_vals

                dp_tiles = _gemm_qk(
                    fx.Index(LDS_DO_BASE),
                    v_b_packs,
                    inits={mt: _ld_read(head_local, mt, 0) for mt in half},
                    mts=half,
                )

                # Hoist the first g2d dt's GEMM2 transpose-reads into the last half's
                # dS/pack shadow: the ds_read_tr16 LDS latency overlaps that VALU block
                # instead of exposing at GEMM2's first MFMA.
                if const_expr(pks == PV_K_STEPS - 1):
                    do_ring = [
                        [_read_tr(fx.Index(LDS_DO_BASE), _d, _p) for _p in range_constexpr(PV_K_STEPS)]
                        for _d in range_constexpr(g2d)
                    ]
                    q_ring = [
                        [_read_tr(fx.Index(0), _d, _p) for _p in range_constexpr(PV_K_STEPS)]
                        for _d in range_constexpr(g2d)
                    ]

                for nt in range_constexpr(NT):
                    _ds = [
                        [_fmul(P[mt][nt][t], Vec(dp_tiles[mt][nt])[t]) for t in range_constexpr(4)]
                        for mt in half
                    ]
                    p_pack[pks][nt] = bf16_trunc_pack_v8(P[ma][nt] + P[mb][nt])
                    ds_pack[pks][nt] = bf16_trunc_pack_v8(_ds[0] + _ds[1])

            # GEMM2a dV^T += dO_tr @ P ; GEMM2b dK^T += Q_tr @ dS. Depth-g2d dt prefetch
            # ring: issue dt+g2d's transpose-reads before dt's MFMAs so the ds_read_tr16
            # LDS latency hides in the MFMA shadow. g2d=1 -> depth-1 baseline.
            rocdl.s_setprio(1)
            for dt in range_constexpr(DT):
                _slot = dt % g2d
                do_tr, q_tr = do_ring[_slot], q_ring[_slot]
                if const_expr(dt + g2d < DT):
                    do_tr_n = [
                        _read_tr(fx.Index(LDS_DO_BASE), dt + g2d, pks) for pks in range_constexpr(PV_K_STEPS)
                    ]
                for pks in range_constexpr(PV_K_STEPS):
                    for nt in range_constexpr(NT):
                        dv_cur[dt][nt] = mfma_acc(do_tr[pks], p_pack[pks][nt], dv_cur[dt][nt])
                if const_expr(dt + g2d < DT):
                    q_tr_n = [_read_tr(fx.Index(0), dt + g2d, pks) for pks in range_constexpr(PV_K_STEPS)]
                for pks in range_constexpr(PV_K_STEPS):
                    for nt in range_constexpr(NT):
                        dk_cur[dt][nt] = mfma_acc(q_tr[pks], ds_pack[pks][nt], dk_cur[dt][nt])
                if const_expr(dt + g2d < DT):
                    for _ in range_constexpr(2 * PV_K_STEPS * NT):
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
                    do_ring[_slot], q_ring[_slot] = do_tr_n, q_tr_n
            rocdl.s_setprio(0)
            return dv_cur, dk_cur

        def _q_body(q_start, inner, apply_mask):
            dv_cur = [[inner[dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)]
            dk_cur = [
                [inner[DT * NT + dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)
            ]
            # Head-invariant DMA offsets: computed once per q-block, reused by all heads.
            _bases = _dma_bases(q_start) if const_expr(ENABLE_DMA) else None
            for head_local in range_constexpr(GQA_GROUP_SIZE):
                dv_cur, dk_cur = _head_step_lds(q_start, apply_mask, head_local, dv_cur, dk_cur, bases=_bases)
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
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        total_kv: fx.Int32,
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
        if const_expr(agpr != 0):
            passthrough_entries = passthrough_entries + [
                ["amdgpu-agpr-alloc", f"{int(agpr)},{int(agpr)}"],
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
            CuSeqQ,
            CuSeqKv,
            seq_len_q,
            seq_len_k,
            total_kv,
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
        # Backward is VALU/exp2-issue-bound with the MFMA pipe mostly idle; post-RA
        # misched hides the gradient-GEMM MFMAs in the exp2/reduce VALU shadow.
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    _compiled: dict = {}

    def _launch(*args, **kwargs):
        return _cached_launch(_compiled, launch_flash_attn_bwd_dkdv, _hints, args, kwargs)

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
    window_left=-1,
    fold_lse=None,  # None = fold on the hw-exp path only (see below)
    batch_size=None,  # compile-time B; required for SBHD seq-step stride bake
    sbhd=False,  # SBHD [S,B,H,D] native layout (seq-step = B*H*D)
    fuse_delta=False,  # compute DELTA here from O (K16 slot) instead of a separate odo pass
    varlen=False,  # ragged / block-causal: per-segment [tok_base,tok_end) from cu_seqlens
):
    """Build the dQ Q-outer backward launcher (16x16x32 mirror of dkdv).

    One work-group owns BLOCK_M q rows and loops the causal kv blocks. Q/dO are
    register-resident B-operands, K/V stream through LDS, and C = P*(dP-delta_id)
    is centered by odo's identity delta so GEMM2 runs on plain bf16.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "bwd dq kernel targets gfx950"
    assert dtype_str == "bf16", "bwd dq kernel targets bf16"
    assert causal, "bwd dq kernel is causal-only for the GPT-OSS campaign"

    # Prescale the owned Q by sm*log2e and fold -log2e*lse into GEMM1a's MFMA C-init,
    # so the accumulator already IS the base-2 softmax exponent and the per-slot diff
    # FMA disappears.
    if fold_lse is None:
        fold_lse = True

    ENABLE_DMA = enable_dma and not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    BLOCK_M = 192  # q rows per work-group (owned)
    WARP_SIZE = 64
    NUM_XCD = 8  # gfx950 XCDs; the dispatcher hands block_id to xcd = block_id % NUM_XCD
    BLOCK_KV = block_kv  # kv rows per loop iteration (LDS tile)
    flat_work_group_size = 256
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_Q = BLOCK_M // NUM_WAVES  # 32

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32); q<->kv mirror of dkdv. ----
    M_TILE = 16
    N_TILE = 16
    D_TILE = 16
    K_STEP_QK = 32  # K=32 per GEMM1 MFMA (contract over D)
    K_STEPS_QK = head_dim // K_STEP_QK  # d64 -> 2
    QT = ROWS_PER_WAVE_Q // N_TILE  # owned q 16-tiles per wave: 2
    KVT = BLOCK_KV // M_TILE
    DT = head_dim // D_TILE
    PV_K_STEP = 32  # GEMM2 MFMA contracts over kv (vs K_STEP_QK over D)
    PV_K_STEPS = BLOCK_KV // PV_K_STEP

    # sched_barrier(TRANS) pins MFMA/ds_read/VALU in place and frees only the
    # quarter-rate v_exp to migrate, so the exps are what fills the MFMA latency
    # shadow (schedule-only, opcode multiset unchanged).
    SCHED_TRANS = 0x400  # LLVM SchedGroupMask: TRANS (v_exp)

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
    LDS_TOTAL = 2 * LDS_TILE

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
        O: fx.Tensor,  # fuse_delta: O for the DELTA reduce; otherwise unused placeholder slot
        CuSeqQ: fx.Tensor,  # varlen: cu_seqlens_q [num_seg+1] i32; else unused placeholder slot
        CuSeqKv: fx.Tensor,  # varlen: cu_seqlens_kv [num_seg+1] i32; else unused placeholder slot
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

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_16x16x32_bf16, a, b, c)

        def _vexp(x):
            # Hardware 2^x as the raw v_exp_f32 intrinsic: one instruction (math.exp2 adds ldexp
            # range reduction the softmax argument <= 0 never needs), compiler-visible so it owns
            # the MFMA->VALU wait states for the accumulator it reads, and side-effect free so it
            # still sinks into the GEMM2 bubbles. dkdv keeps the anchor form -- there the exps
            # follow GEMM1a directly and this pads with s_nop instead.
            return fx.Float32(
                llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", [_raw(x)], [], [])
            )

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16  # M/N index within a 16-tile
        kg = lane // 16  # 0..3: K-subgroup (inputs) / M-block (C output)

        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_off
            ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # block_id decode. The dispatcher round-robins work-groups over the XCDs and each XCD
        # owns a private L2 slice, so XCD-major decode gives each XCD a whole (batch, kv-head)
        # chunk: its L2 streams one kv-head's K/V and the GQA work-groups reading byte-identical
        # rows stay co-resident. Bijective when B*NUM_HEADS_KV % NUM_XCD == 0, which
        # NUM_HEADS_KV % NUM_XCD == 0 guarantees; other head counts keep the plain decode.
        num_q_tiles = (seq_len_q_v + BLOCK_M - 1) // BLOCK_M
        if const_expr(NUM_HEADS_KV % NUM_XCD == 0):
            _xcd = block_id % fx.Index(NUM_XCD)
            _slot = block_id // fx.Index(NUM_XCD)
            _q_in_group = _slot % GQA_GROUP_SIZE
            _u = _slot // GQA_GROUP_SIZE
            _qt_disp = _u % num_q_tiles
            _bkv = (_u // num_q_tiles) * fx.Index(NUM_XCD) + _xcd
            kv_head_idx = _bkv % NUM_HEADS_KV
            batch_idx = _bkv // NUM_HEADS_KV
            q_head_idx = kv_head_idx * GQA_GROUP_SIZE + _q_in_group
        elif const_expr(GQA_GROUP_SIZE == 1):
            q_head_idx = block_id % NUM_HEADS_Q
            batch_q_tile_id = block_id // NUM_HEADS_Q
            kv_head_idx = q_head_idx
            _qt_disp = batch_q_tile_id % num_q_tiles
            batch_idx = batch_q_tile_id // num_q_tiles
        else:
            kv_head_idx = block_id % NUM_HEADS_KV
            _bid_rest = block_id // NUM_HEADS_KV
            _q_in_group = _bid_rest % GQA_GROUP_SIZE
            batch_q_tile_id = _bid_rest // GQA_GROUP_SIZE
            q_head_idx = kv_head_idx * GQA_GROUP_SIZE + _q_in_group
            _qt_disp = batch_q_tile_id % num_q_tiles
            batch_idx = batch_q_tile_id // num_q_tiles
        # SHADOW seq_len_q_v/k_v to the per-segment length so downstream base/SRD/clamp follow the segment (byte-identical when uniform; grid tiles were fixed from max above).
        if const_expr(varlen):
            _seg = batch_idx
            _cuq_rsrc = buffer_ops.create_buffer_resource(CuSeqQ, max_size=True)
            _cukv_rsrc = buffer_ops.create_buffer_resource(CuSeqKv, max_size=True)
            _qb_i = fx.Int32(buffer_ops.buffer_load(_cuq_rsrc, _seg, vec_width=1, dtype=fx.Int32))
            _qe_i = fx.Int32(
                buffer_ops.buffer_load(_cuq_rsrc, _seg + fx.Index(1), vec_width=1, dtype=fx.Int32)
            )
            _kb_i = fx.Int32(buffer_ops.buffer_load(_cukv_rsrc, _seg, vec_width=1, dtype=fx.Int32))
            _ke_i = fx.Int32(
                buffer_ops.buffer_load(_cukv_rsrc, _seg + fx.Index(1), vec_width=1, dtype=fx.Int32)
            )
            q_tok_base = fx.Index(_qb_i)
            kv_tok_base = fx.Index(_kb_i)
            seq_len_q_v = fx.Index(_qe_i) - q_tok_base
            seq_len_k_v = fx.Index(_ke_i) - kv_tok_base
            causal_off_i32 = (_ke_i - _kb_i) - (_qe_i - _qb_i)
        else:
            q_tok_base = batch_idx * seq_len_q_v
            kv_tok_base = batch_idx * seq_len_k_v
            causal_off_i32 = fx.Int32(seq_len_k) - fx.Int32(seq_len_q)
        causal_offset = seq_len_k_v - seq_len_q_v
        # Descending q_tile = longest-processing-time-first: causal work grows with
        # q_tile and block_ids are handed out in order, so dispatch order IS the
        # list-schedule order.
        if const_expr(varlen):
            # Per-segment tile count; no causal-aligned shift (uniform-only opt). Out-of-segment tiles clamp to 0 and store nothing via the store-end mask.
            _nqt_seg = (seq_len_q_v + fx.Index(BLOCK_M - 1)) // fx.Index(BLOCK_M)
            _qt_in_seg = ArithValue(_qt_disp < _nqt_seg)
            _qt_c = fx.Index(_qt_in_seg.select(_qt_disp, _nqt_seg - fx.Index(1)))
            q_tile_idx = _nqt_seg - fx.Index(1) - _qt_c
            q_start = q_tile_idx * BLOCK_M
            _q_owned_end = q_start + fx.Index(BLOCK_M)
            _q_store_end = fx.Index(
                _qt_in_seg.select(
                    ArithValue(_q_owned_end < seq_len_q_v).select(_q_owned_end, seq_len_q_v),
                    fx.Index(0),
                )
            )
        else:
            q_tile_idx = num_q_tiles - fx.Index(1) - _qt_disp
            # Causal-aligned origin: shift every tile down by the largest BLOCK_KV pad multiple so the overshoot lands on tile 0 (stays kv-block aligned; det unchanged).
            _q_pad = num_q_tiles * fx.Index(BLOCK_M) - seq_len_q_v
            _q_shift = (_q_pad // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
            _q_raw = q_tile_idx * BLOCK_M
            q_start = fx.Index(
                ArithValue(_q_raw >= _q_shift).select(_q_raw - _q_shift, fx.Index(0))
            )  # fx.Index is unsigned: the discarded branch would underflow, so select it away
            _q_owned_end = _q_raw + fx.Index(BLOCK_M) - _q_shift  # exclusive, always > q_start
            _q_store_end = fx.Index(ArithValue(_q_owned_end < seq_len_q_v).select(_q_owned_end, seq_len_q_v))

        # Per-batch base (elements). SBHD: batch inside the seq axis -> base is only
        # H*D. THD: dense per-batch block -> base is seq*H*D.
        if const_expr(sbhd):
            _q_batch_elems = batch_idx * fx.Index(STRIDE_TOKEN_Q)
            _kv_batch_elems = batch_idx * fx.Index(STRIDE_TOKEN_KV)
        else:
            _q_batch_elems = q_tok_base * fx.Index(STRIDE_TOKEN_Q)
            _kv_batch_elems = kv_tok_base * fx.Index(STRIDE_TOKEN_KV)

        # Fold per-batch element offset into raw K/V pointers (0-based rows).
        _kv_ptr_batch_off = _kv_batch_elems
        k_ptr = buffer_ops.get_element_ptr(k_ptr, _kv_ptr_batch_off, elem_type=elem_type)
        v_ptr = buffer_ops.get_element_ptr(v_ptr, _kv_ptr_batch_off, elem_type=elem_type)

        def global_idx_q(token_idx, col):
            return token_idx * RD_STRIDE_Q + q_head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * RD_STRIDE_KV + kv_head_idx * HEAD_DIM + col

        def _ld_delta_elem(q_row):
            # VARLEN: packed [total_q,Hq] token-major. Uniform/SBHD: [B,Hq,Sq] head-major.
            if const_expr(varlen):
                return q_row * fx.Index(NUM_HEADS_Q) + q_head_idx
            return q_head_idx * seq_len_q_v + q_row

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
        if const_expr(varlen):
            _lse_batch_byte_off = _raw(q_tok_base * fx.Index(NUM_HEADS_Q) * fx.Index(4))
        else:
            _lse_batch_byte_off = _raw(batch_idx * _lse_per_batch * fx.Index(4))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_lse_nrec_bytes, base_byte_offset=_lse_batch_byte_off
        )
        delta_in_rsrc = buffer_ops.create_buffer_resource(
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
            _dma_aux = fx.Int32(1)

            def coop_dma_tile(src_rsrc, lds_byte_base, tile_start):
                """DMA a tile into the swizzled LDS layout.

                Address math is recomputed per tile on purpose: keeping the offsets live
                across the k_tr peak pushes VGPRs past the occ-2 boundary.
                """
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

        def q_row_of(qt):
            return q_row_wave + fx.Index(qt * N_TILE) + lane16

        q_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        do_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        d_parts = [fx.Float32(0.0) for _ in range_constexpr(QT)]
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

        # ---- FOLD: prescale the owned Q by sm*log2e once per work-group (amortized
        # over the whole causal kv-loop). Q feeds GEMM1a only -- dQ is accumulated from
        # K_tr and never from Q -- so scaling q_b_packs is safe. ----
        if const_expr(fold_lse):
            _qscale_v8 = Vec.filled(MFMA_LANE_K, sm_scale * _LOG2E, fx.Float32)
            for qt in range_constexpr(QT):
                for ks in range_constexpr(K_STEPS_QK):
                    q_b_packs[qt][ks] = (
                        (Vec(q_b_packs[qt][ks]).to(fx.Float32) * _qscale_v8).to(elem_dtype).ir_value()
                    )

        # ---- Owned LSE/-delta_id per q (one scalar per qt, q = qt*16 + lane16). ----
        lse_owned = []
        delta_owned = []
        for qt in range_constexpr(QT):
            _lse_elem = _ld_delta_elem(q_row_of(qt))
            lse_owned.append(
                fx.Float32(buffer_ops.buffer_load(lse_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32))
            )
            if const_expr(not fuse_delta):
                delta_owned.append(
                    fx.Float32(
                        buffer_ops.buffer_load(delta_in_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32)
                    )
                )
        if const_expr(fuse_delta):
            # DELTA[b,hq,q] = -rowsum_d(O.dO). A row's 64 D split over the 4 K-subgroup
            # lanes sharing lane16, so the row total is a 2-step xor butterfly over kg
            # (masks 16,32); ds_bpermute is the LDS crossbar only (no alloc, no barrier).
            # Each (b,hq,q) row is owned by one work-group, so one lane (kg==0) stores it
            # for dkdv; rows this tile only traces are recomputed, not stored.
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
                    delta_in_rsrc,
                    _ld_delta_elem(_q_row) * fx.Index(4),
                    mask=ArithValue(_q_row < _q_store_end) & ArithValue(kg == fx.Index(0)),
                    offset_is_bytes=True,
                )

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        _scale_log2e_v4 = Vec.filled(4, sm_scale * _LOG2E, fx.Float32)  # exact (hw exp2) v4 scale

        def _p_of(s_r, lse_t, apply_mask):
            if const_expr(fold_lse):
                # FOLD: s_r already = sm*log2e*S (prescaled Q). Masked (diagonal) tiles
                # keep a ZERO C-init so lse is added here; the bulk gets it from the
                # C-init and only needs the clamp below.
                if const_expr(apply_mask):
                    s_r = fmath.fma(s_r, fx.Float32(1.0), lse_t, fastmath=fm_fast)
                else:
                    s_r = fx.Float32(arith.minimumf(_raw(s_r), _raw(c_zero_f)))
                return _vexp(s_r)
            diff = fmath.fma(s_r, c_sm_scale_log2e, lse_t, fastmath=fm_fast)
            return ArithValue(diff).exp2(fastmath=fm_fast)

        # A-operand read (K/V from LDS): A[m=kv=lane16][k=D=kg*8+s]. Address hoist: kvt*16 is a
        # 16-multiple, so _pblk(kvt*16+lane16)*PBLK == kvt*(8*PBLK) + _pblk(lane16)*PBLK -- the
        # lane-only part is loop-invariant and the (col^mask) part kvt-invariant, so both
        # precompute once. Byte-identical layout, 0-conflict property and determinism kept.
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
            """Transpose-read K -> GEMM2 A-operand [m=D=dt*16+lane16][k=kv=kg*8+s]."""
            col = fx.Index(dt * D_TILE) + (lane % fx.Index(4)) * fx.Index(4)
            row0 = fx.Index(pks * PV_K_STEP) + kg * fx.Index(4) + (lane16 // fx.Index(4))
            row1 = row0 + fx.Index(N_TILE)
            v0 = ds_read_tr_v4f16(a_base + _pblk(row0) * fx.Index(PBLK) + _swizzle(row0, col))
            v1 = ds_read_tr_v4f16(a_base + _pblk(row1) * fx.Index(PBLK) + _swizzle(row1, col))
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # Per-q delta init (broadcast over the 4 kv output rows) and q-slot i32. The
        # GEMM1a C-layout is C[m=kv][n=q], so a lane's 4 accumulator slots share one q
        # and -log2e*lse is a broadcast exactly like -delta_id (FOLD path).
        delta_inits = [
            Vec.from_elements([delta_owned[qt]], fx.Float32).broadcast_to(4).ir_value()
            for qt in range_constexpr(QT)
        ]
        if const_expr(fold_lse):
            lse_inits = [
                Vec.from_elements([lse_owned[qt]], fx.Float32).broadcast_to(4).ir_value()
                for qt in range_constexpr(QT)
            ]
            # Reuse slot 0 of the broadcast for the masked path instead of keeping the
            # scalar alive too (same register, no extra live value).
            lse_owned = [fx.Float32(Vec(lse_inits[qt])[0]) for qt in range_constexpr(QT)]
        q_slot_i32 = [fx.Int32(q_row_of(qt)) for qt in range_constexpr(QT)]

        # Loop-carried A(DT*QT) accumulators: dQ = sm * A, A = sum_kv K_tr @ (P~*(dP-delta_id)).
        # The rho/R correction is dropped (halves GEMM2 MFMA): delta_id from odo is the
        # fp32-exact rowsum_d(O.dO), so C already carries the near-diagonal cancellation before
        # the bf16 pack. The rowsum(P~) renorm is dropped too -- R == 1 to bf16 precision.
        A_accs = [c_zero_v4f32 for _ in range_constexpr(DT * QT)]

        # Causal upper bound of the rows this work-group OWNS (not of the BLOCK_M rows
        # it walks): tile 0's shared rows are recomputed with this truncated range and
        # discarded at the store, which is what saves the pad tile's kv blocks.
        _q_end = _q_owned_end + causal_offset
        kv_upper = fx.Index(ArithValue(_q_end < seq_len_k_v).select(_q_end, seq_len_k_v))

        def _kv_body(kv_start, inner, apply_mask):
            # The LDS tile for kv_start is already resident (prologue for the first
            # iteration, the previous body for the rest): this body consumes it and
            # then issues the DMA for kv_start+BLOCK_KV under its own GEMM2 (see the
            # k_tr / DMA hand-over below).
            A_cur = [[inner[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]
            sb_bulk = not apply_mask  # exps only exist on these paths

            kv_start_i32 = fx.Int32(kv_start)
            # C[kvt][qt]: 4 f32 at kv=kvt*16+kg*4+t, q=qt*16+lane16. C = P~*(dP-delta_id)
            # feeds GEMM2.
            C = [[None] * QT for _ in range_constexpr(KVT)]
            c_pack = [[None] * QT for _ in range_constexpr(PV_K_STEPS)]
            # Split GEMM1a/1b + exp2/C + pack per kv-half (pks = the 2 kvt of one GEMM2 K=32 step):
            # only 2 kvt of s/dP are live at a time, halving the transient VGPR peak without touching
            # the batched GEMM2 below. The next half's K ds_read is issued right after this half's
            # GEMM1 MFMAs so its latency hides in the VALU-heavy exp2/C/pack shadow (V stays in-half).
            k_a_by_half = {0: _gemm1_load(fx.Index(0), [0, 1])}
            for pks in range_constexpr(PV_K_STEPS):
                ka, kb = 2 * pks, 2 * pks + 1
                half = [ka, kb]
                # GEMM1a S[kv,q]=K@Q^T ; GEMM1b dP[kv,q]=V@dO^T (acc init=-delta_id) for
                # this kv-half. s_setprio(1) raises MFMA priority over ds_read/VALU;
                # dropped to 0 for the exp2/pack/reduce VALU section so it is not starved.
                rocdl.s_setprio(1)
                rocdl.iglp_opt(0)
                if const_expr(fold_lse and not apply_mask):
                    _s_inits = lse_inits
                else:
                    _s_inits = None
                s_tiles = _gemm1_mfma(k_a_by_half[pks], q_b_packs, inits_q=_s_inits, kvts=half)
                if const_expr(sb_bulk):
                    rocdl.sched_barrier(SCHED_TRANS)
                dp_tiles = _gemm1(fx.Index(LDS_V_BASE), do_b_packs, delta_inits, kvts=half)
                rocdl.s_setprio(0)

                # Narrow the prefetched half's live range: load only ka's K here and issue kb's between
                # qt=0 and qt=1 (_next_kb_load), so the second kvt's registers stay live for half as long
                # before GEMM1 consumes them, at the same ds_read-vs-VALU overlap.
                _next_kb_load = None
                if const_expr(pks + 1 < PV_K_STEPS):
                    nka, nkb = 2 * (pks + 1), 2 * (pks + 1) + 1
                    # s_setprio(1) around the prefetch ds_read issue only (not the
                    # VALU it's interleaved with): the load itself should win issue
                    # priority over the surrounding exp2/pack VALU so it drains
                    # sooner, without raising priority on the VALU work itself.
                    rocdl.s_setprio(1)
                    k_a_by_half[pks + 1] = _gemm1_load(fx.Index(0), [nka])
                    rocdl.s_setprio(0)

                    def _next_kb_load():  # noqa: B023
                        rocdl.s_setprio(1)
                        k_a_by_half[pks + 1].update(_gemm1_load(fx.Index(0), [nkb]))  # noqa: B023
                        rocdl.s_setprio(0)

                if const_expr(not apply_mask):
                    # Vectorized bulk (below-diagonal): exp2/C/reduce as packed v4 ops
                    # (v_pk_*), mirroring the 32x32 kernel's v8 path. exp2 and C=P*dP are
                    # strictly elementwise so C is bit-identical to the scalar branch;
                    # R re-associated in a fixed order -> deterministic (det gate holds).
                    for qt in range_constexpr(QT):
                        if const_expr(qt == 1 and _next_kb_load is not None):
                            _next_kb_load()
                        if const_expr(not fold_lse):
                            lse_v4 = Vec.from_elements([lse_owned[qt]], fx.Float32).broadcast_to(4)
                        for kvt in half:
                            if const_expr(fold_lse):
                                # either (see _vexp).
                                _s_v = Vec(s_tiles[kvt][qt])
                                p4 = Vec.from_elements(
                                    [_vexp(fx.Float32(_s_v[t])) for t in range_constexpr(4)],
                                    fx.Float32,
                                )
                            else:
                                # exact: 2^diff on the log2 exponent (lse arrives as
                                # plain -log2e*lse), elementwise over the v4.
                                diff4 = fmath.fma(
                                    _raw(s_tiles[kvt][qt]),
                                    _raw(_scale_log2e_v4),
                                    _raw(lse_v4),
                                    fastmath=fm_fast,
                                )
                                p4 = Vec.from_elements(
                                    [_vexp(Vec(diff4)[t]) for t in range_constexpr(4)], fx.Float32
                                )
                            if const_expr(window_left >= 0):
                                _thr = q_slot_i32[qt] + causal_off_i32 - fx.Int32(window_left + 1)
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
                        lse_q = lse_owned[qt]
                        for kvt in half:
                            dp_v = dp_tiles[kvt][qt]
                            s_v = s_tiles[kvt][qt]
                            c_vals = []
                            for t in range_constexpr(4):
                                kv_slot = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4 + t)
                                _up = ArithValue(kv_slot > q_slot_i32[qt] + causal_off_i32)
                                if const_expr(window_left >= 0):
                                    _lo = ArithValue(
                                        kv_slot <= q_slot_i32[qt] + causal_off_i32 - fx.Int32(window_left + 1)
                                    )
                                    _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                                else:
                                    _mm = _up
                                s_r = _mm.select(c_neg_inf, fx.Float32(Vec(s_v)[t]))
                                p = _p_of(s_r, lse_q, True)
                                c = _fmul(p, Vec(dp_v)[t])
                                c_vals.append(c)
                            C[kvt][qt] = c_vals

                # Pack this half's C now (contract over kv): combine kvt=ka (k=0..3) and
                # kvt=kb (k=4..7) -> 8 kv values/lane matching _read_tr's kv ordering.
                # Packing here frees C[ka],C[kb] (and s/dP) before the next half's GEMM1.
                if const_expr(sb_bulk):
                    rocdl.sched_barrier(SCHED_TRANS)
                for qt in range_constexpr(QT):
                    c_pack[pks][qt] = bf16_trunc_pack_v8(C[ka][qt] + C[kb][qt])

            # GEMM2 A^T[D,q] += K_tr @ C. Process dt in interleaved pairs so a dependent MFMA is
            # separated by 3 independent ones, covering the 16x16x32 operand latency. The next pair's
            # k_tr is prefetched during the current one; the initial pair is read here, not hoisted,
            # to keep k_tr off the s/dP transient peak. s_setprio(2) puts MFMA issue over ds_read.
            kts = [
                [_read_tr(fx.Index(0), d, pks) for pks in range_constexpr(PV_K_STEPS)]
                for d in range_constexpr(min(2, DT))
            ]
            rocdl.s_setprio(2)
            for d0 in range_constexpr(0, DT, 2):
                if const_expr(d0 + 2 < DT):
                    kts.append([_read_tr(fx.Index(0), d0 + 2, pks) for pks in range_constexpr(PV_K_STEPS)])
                    kts.append([_read_tr(fx.Index(0), d0 + 3, pks) for pks in range_constexpr(PV_K_STEPS)])
                for pks in range_constexpr(PV_K_STEPS):
                    for dd in range_constexpr(d0, min(d0 + 2, DT)):
                        for qt in range_constexpr(QT):
                            A_cur[dd][qt] = mfma_acc(kts[dd][pks], c_pack[pks][qt], A_cur[dd][qt])
                # Interleave the next-pair prefetch ds_read_tr16 1:1 with the pair MFMAs.
                if const_expr(d0 + 2 < DT):
                    for _ in range_constexpr(2 * PV_K_STEPS * QT):
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
                # LDS hand-over: this pair issues the tile's last k_tr read, so the next tile is DMA'd
                # into the SAME buffer while the remaining register-only GEMM2 pair covers the transfer.
                # Unlike an LDS double buffer (measured net-negative) the in-flight writes never contend
                # with LDS reads and no second buffer is needed. Issuing it here also keeps the DMA
                # address registers off the all-DT k_tr peak.
                if const_expr(d0 == max(0, DT - 4)):
                    gpu.barrier()
                    if const_expr(ENABLE_DMA):
                        # One tile past the causal range on the tail iteration: the SRD
                        # bounds it, so it lands as zeros with no memory traffic.
                        _kv_next = kv_start + fx.Index(BLOCK_KV)  # noqa: B023
                        coop_dma_tile(k_rsrc, lds_base_idx, _kv_next)  # noqa: B023
                        coop_dma_tile(  # noqa: B023
                            v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), _kv_next
                        )
            rocdl.s_setprio(0)
            if const_expr(ENABLE_DMA):
                rocdl.s_waitcnt(0)
            gpu.barrier()

            out = [A_cur[dt][qt] for dt in range_constexpr(DT) for qt in range_constexpr(QT)]
            return out

        # Split the causal kv-loop: [_wlo, _split) below the diagonal (no mask),
        # [_split, kv_upper) straddles it (mask).
        _carry = A_accs
        loop_results = _carry
        if const_expr(window_left >= 0):
            # fx.Index is unsigned: guard the subtract (W-1 may exceed q+off) to
            # avoid underflow-to-huge. _wlo skips fully-out-of-window kv tiles.
            _wlo = fx.Index(
                ArithValue(q_start + causal_offset >= fx.Index(window_left)).select(
                    q_start + causal_offset - fx.Index(window_left), fx.Index(0)
                )
            )
            _wlo = (_wlo // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
        else:
            _wlo = fx.Index(0)
        _split = ((q_start + causal_offset) // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
        # Prologue for the software-pipelined body: it expects its own tile already in
        # LDS and leaves the next one there. _wlo is the first kv tile of whichever of
        # the two loops below runs first (they are contiguous).
        if const_expr(ENABLE_DMA):
            coop_dma_tile(k_rsrc, lds_base_idx, _wlo)
            coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), _wlo)
            rocdl.s_waitcnt(0)
        gpu.barrier()
        for kv_start, inner in range(_wlo, _split, BLOCK_KV, init=_carry):
            loop_results = yield _kv_body(kv_start, inner, False)
        for kv_start, inner in range(_split, kv_upper, BLOCK_KV, init=loop_results):
            loop_results = yield _kv_body(kv_start, inner, True)

        A_finals = [[loop_results[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]

        # Epilogue: dQ = sm * A. Both exp modes use R == 1 -- lse is the true log-sum-exp so
        # rowsum(exp(S-lse)) == 1, and the Schraudolph fast P~ sums to 1 to bf16 precision -- so
        # the renorm is dropped. The 16x16 C-layout gives 4 contiguous D per lane, direct store.
        for qt in range_constexpr(QT):
            dq_scale = fx.Float32(sm_scale)
            _q_row = q_row_of(qt)
            # Owned rows only: the shifted origin makes tile 0 walk BLOCK_M rows while
            # owning fewer, and _q_store_end also absorbs the old seq_len_q clamp.
            _store_mask = ArithValue(_q_row < _q_store_end)
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
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
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
            O,
            CuSeqQ,
            CuSeqKv,
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

    _compiled: dict = {}

    def _launch(*args, **kwargs):
        return _cached_launch(_compiled, launch_flash_attn_bwd_dq, _hints, args, kwargs)

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


def _qsplit_for(Sq):
    # q_split fans the dK/dV KV-owner WGs across the CU grid; the optimum rises with
    # Sq before split-reduction overhead dominates.
    if Sq <= 8192:
        return 4
    return 3


def _blockkv_for(Skv):
    # dkdv is KV-outer, so Skv (not Sq) sets its grid: a short Skv needs BLOCK_KV=64
    # to fill the CU array, a long one wants 128 to amortise the per-tile cost. Keying
    # this on Sq costs 19% on rectangular shapes such as Sq=2048, Skv=16384.
    return 64 if Skv <= 2048 else 128


def _dq_block_kv(Sq):
    """dq's kv tile, swept with dkdv running before it (L2 contention moves the
    optimum). Only multiples of 32 are valid -- other sizes silently produce the
    wrong dQ, so keep to the checked set {32, 64, 96, 192}.
    """
    return 96 if Sq >= 16384 else 64


_BWD_CACHE: dict = {}
# Fold the odo (DELTA = -rowsum_d(O.dO)) pass into the dq kernel and drop its launch:
# dq is Q-outer and already streams dO, so it reduces DELTA for the q rows it owns,
# saving one kernel launch and the whole O HBM re-read.
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


def _get_bwd(
    Hq, Hkv, D, scale, window_left, q_split, block_kv, dq_block_kv=64, batch_size=None, sbhd=False, varlen=False
):
    # batch_size is baked only into the SBHD seq-step stride (RD_STRIDE_*); THD takes it
    # as a runtime kernel argument. Keeping it in the key there rebuilds the module -- and
    # so recompiles -- for every num_seq an e2e run happens to hit.
    if not sbhd:
        batch_size = None
    key = (Hq, Hkv, D, scale, window_left, q_split, block_kv, dq_block_kv, batch_size, sbhd, varlen)
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
            block_kv=dq_block_kv,
            waves_per_eu=1,
            batch_size=batch_size,
            sbhd=sbhd,
            fuse_delta=_FUSE_DELTA,
            varlen=varlen,
            **common,
        )
        dkdv_l = build_flash_attn_bwd_dkdv_module(
            q_split=q_split, block_kv=block_kv, batch_size=batch_size, sbhd=sbhd, varlen=varlen, **common
        )
        if _FUSE_DELTA:
            # The fused dq kernel produces DELTA itself; the standalone odo kernel is
            # never launched here. _defer_delta forwards O into dq's freed slot and
            # keeps the legacy odo -> dq -> dkdv call order for callers.
            dq_l, odo_l = _defer_delta(dq_l)
        else:
            odo_l = build_flash_attn_bwd_odo_module(
                num_heads=Hq, head_dim=D, num_kv_heads=Hkv, sm_scale=scale, sbhd=sbhd
            )
        launchers = (dq_l, dkdv_l, odo_l)
        _BWD_CACHE[key] = launchers
    return launchers


def _prescale_lse(lse_bhsq):
    """Fold -log2e into lse host-side so the kernel's exp2 argument is a bare fma."""
    return (lse_bhsq.float() * (-_LOG2E)).contiguous()


def flydsl_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    lse_bhsq,
    B,
    Sq,
    Skv,
    Hq,
    Hkv,
    D,
    scale,
    window_left=-1,
    sbhd=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
):
    """Run the 16x16x32 flydsl bwd.
    THD (sbhd=False): q,dout,dq,out:[B*Sq,Hq,D]; k,v,dk,dv:[B*Skv,Hkv,D].
    SBHD (sbhd=True): q,dout,dq,out:[Sq,B,Hq,D]; k,v,dk,dv:[Skv,B,Hkv,D] (native,
    no permute/copy anywhere -- the kernels address SBHD directly and the dk/dv
    workspace is laid out [q_split,Skv,B,Hkv,D] so the slot reduction is contiguous).
    lse_bhsq:[B,Hq,Sq] f32 (batch-major, layout-independent).
    window_left>=0 = sliding-window causal (valid q+off-W < kv <= q+off).

    Ragged / block-causal (cu_seqlens_q given, THD only): q/k/v/dq/dk/dv and lse_bhsq
    all packed; each segment [cu[i],cu[i+1]) is an independent document (per-segment
    bottom-right causal + cross-segment masking). Grid tiles by max_seqlen_q/kv."""
    varlen = cu_seqlens_q is not None
    st = torch.cuda.current_stream()
    lse_s = _prescale_lse(lse_bhsq)
    qf, kf, vf, dof = q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1)
    o16 = out.to(q.dtype).reshape(-1)

    if varlen:
        assert not sbhd, "ragged / block-causal backward is THD only"
        assert _FUSE_DELTA, "ragged bwd fuses DELTA into dq (no odo launch)"
        num_seg = cu_seqlens_q.numel() - 1
        total_q, total_kv = q.shape[0], k.shape[0]
        max_sq = int(max_seqlen_q) if max_seqlen_q is not None else Sq
        max_skv = int(max_seqlen_kv) if max_seqlen_kv is not None else Skv
        q_split = _qsplit_for(max_sq)
        dq_l, dkdv_l, _ = _get_bwd(
            Hq, Hkv, D, scale, window_left, q_split,
            _blockkv_for(max_skv), _dq_block_kv(max_sq),
            batch_size=num_seg, sbhd=False, varlen=True,
        )
        delta = torch.empty(total_q, Hq, device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        ws_dk = torch.zeros(q_split, total_kv, Hkv, D, device=q.device, dtype=k.dtype)
        ws_dv = torch.zeros(q_split, total_kv, Hkv, D, device=q.device, dtype=v.dtype)
        lsef, df = lse_s.reshape(-1), delta.reshape(-1)
        dq_l(
            qf, kf, vf, dof, lsef, df, dq.reshape(-1), o16,
            cu_seqlens_q, cu_seqlens_kv, num_seg, max_sq, max_skv, st,
        )
        dkdv_l(
            qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1),
            cu_seqlens_q, cu_seqlens_kv, num_seg, max_sq, max_skv, total_kv, st,
        )
        dk = ws_dk.sum(dim=0)
        dv = ws_dv.sum(dim=0)
        return dq, dk, dv

    q_split = _qsplit_for(Sq)
    dq_l, dkdv_l, odo_l = _get_bwd(
        Hq,
        Hkv,
        D,
        scale,
        window_left,
        q_split,
        _blockkv_for(Skv),
        _dq_block_kv(Sq),
        batch_size=B,
        sbhd=sbhd,
    )
    # identity delta = -rowsum(O.dO); both kernels center dP by it (exact). dq owns the
    # reduce (it already holds dO in registers) and stores DELTA for dkdv when
    # _FUSE_DELTA is on, so no odo launch is needed; O is cast to bf16 (no-op when out
    # is already bf16) and passed into dq's freed slot via _defer_delta.
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    if not _FUSE_DELTA:
        odo_l(o16, dout.to(q.dtype).reshape(-1), delta.reshape(-1), B, Sq, st)
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
    lsef, df = lse_s.reshape(-1), delta.reshape(-1)
    cu_ph = torch.zeros(1, device=q.device, dtype=torch.int32)  # placeholder: cu args read only under const_expr(varlen)
    dq_l(qf, kf, vf, dof, lsef, df, dq.reshape(-1), o16, cu_ph, cu_ph, B, Sq, Skv, st)
    dkdv_l(
        qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1),
        cu_ph, cu_ph, B, Sq, Skv, 0, st,
    )
    if sbhd:
        dk = ws_dk.sum(dim=0)  # [Skv,B,Hkv,D] SBHD contiguous
        dv = ws_dv.sum(dim=0)
    else:
        dk = ws_dk.sum(dim=1).reshape(B * Skv, Hkv, D)
        dv = ws_dv.sum(dim=1).reshape(B * Skv, Hkv, D)
    return dq, dk, dv
