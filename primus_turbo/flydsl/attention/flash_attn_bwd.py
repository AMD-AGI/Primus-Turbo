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

# Fold dQ into the KV-outer dkdv kernel (fuse_dq): the split pair runs 7 GEMMs because
# dq (Q-outer) and dkdv (KV-outer) each recompute S=Q@K^T, dP=dO@V^T and the softmax. A
# fifth GEMM inside dkdv -- dQ^T[D][q] = K^T @ dS^T over the block's kv rows -- makes it
# 5 GEMMs in one pass. A kv band owns only part of dQ, so the fifth GEMM lands in a bf16
# split-K workspace that build_flash_attn_bwd_dqred_module folds in a fixed
# band-ascending fp32 order (deterministic, no atomics), exactly like dk/dv's q_split.
_FUSE_DQ = True


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
    key = tuple(a for a in args[:-1] if not isinstance(a, torch.Tensor))
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
    spw=8,  # q rows per work-group tile (the rest of the tile is q-heads)
):
    """Identity-delta ("odo") kernel: DELTA[b,hq,s] = -sum_d O[b,s,hq,d]*dO[b,s,hq,d].

    LPR lanes cooperate on one (b,s,hq) row -- one 16 B chunk of O and of dO each -- and
    fold their partials with an xor butterfly over the low lane bits (ds_bpermute is the
    LDS crossbar only: no allocation, no barrier), then one lane stores the negated scalar
    (the dkdv/dq fold convention) to the transposed [B,Hq,S] delta.

    A row is D*2 = 128 B, so one lane per row makes every load instruction touch 64
    separate lines and read 1.95x the bytes it needs (measured L1->L2 0.788 GB against
    0.405 GB of O/dO); with LPR lanes per row a load instruction covers 64/LPR whole rows
    instead, which took the kernel from 108 to 72 us (3.8 -> 5.7 TB/s) at B=3 S=8192 Hq64
    D64. The reduction is a tree rather than a linear chain, so DELTA is not bit-identical
    to the one-lane-per-row form (still fully deterministic). block=512 measured 78 us;
    waves_per_eu no longer matters (two loads per thread).

    Which (b, hq, s) a lane owns comes from an SPW x HPW tile of (q, q-head) rather than
    from a flat row index -- see SPW below -- which took it to 65 us / 6.26 TB/s and also
    removes the per-thread dynamic division by seq_len that the flat form needed.
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
    LPR = NVEC  # lanes per row: the whole row in one 16 B load each
    ROWS_PER_WG = BLOCK // LPR
    assert BLOCK % LPR == 0 and LPR in (2, 4, 8, 16, 32, 64), f"bad lanes/row {LPR}"
    # A work-group owns SPW consecutive q of HPW consecutive q-heads. DELTA is transposed
    # [B,Hq,S], so a work-group that owns one q and ROWS_PER_WG heads (the natural flat-row
    # tiling) writes ROWS_PER_WG scalars S floats apart -- one line touched per 4 B, which
    # even after cross-work-group combining cost 51 MB of DRAM writes for 6.3 MB of DELTA.
    # Trading heads for q makes each work-group write SPW*4 contiguous bytes per head while
    # the O/dO side keeps SPW runs of HPW*D*2 contiguous bytes. SPW=8 measured best at
    # B=3 S=8192 Hq64 D64 (73.8 / 70.5 / 67.2 / 64.9 / 69.5 / 75.3 us for SPW=1..32): past
    # 8 the shrinking O/dO run costs more than the extra DELTA coalescing pays for.
    HPW = ROWS_PER_WG // min(spw, ROWS_PER_WG)
    while HPW > 1 and NUM_HEADS_Q % HPW:
        HPW //= 2
    SPW = ROWS_PER_WG // HPW

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
        chunk = tid % fx.Index(LPR)
        sl = fx.Index(seq_len)
        total = fx.Index(batch_size) * sl * fx.Index(NUM_HEADS_Q)
        # (b, h-tile, s-tile) from the work-group id; only the s extent is dynamic, so
        # only that one division is a real (work-group uniform, hence SALU) divide.
        # The head tile is the fastest-varying axis so that NUM_HEADS_Q/HPW consecutive
        # work-groups sweep one contiguous SPW*Hq*D run of O/dO; making the s tile fastest
        # instead spreads the read stream over SPW*Hq*D-strided addresses (80 vs 68 us).
        n_stile = (sl + fx.Index(SPW - 1)) // fx.Index(SPW)
        ht = bid % fx.Index(NUM_HEADS_Q // HPW)
        _r = bid // fx.Index(NUM_HEADS_Q // HPW)
        st = _r % n_stile
        b = _r // n_stile
        row_local = tid // fx.Index(LPR)
        hq = ht * fx.Index(HPW) + row_local % fx.Index(HPW)
        s = st * fx.Index(SPW) + row_local // fx.Index(HPW)
        in_range = ArithValue(s < sl)
        # O/dO ride an unbounded (max_size) descriptor, so a tail work-group's OOB rows are
        # clamped to q row 0 here rather than relying on num_records; their store is masked.
        s = fx.Index(in_range.select(s, fx.Index(0)))

        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=True)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=True)
        # DELTA must carry its real bound: a masked-off buffer_store is lowered to an
        # offset of 0x7fffffff and dropped by num_records, so an unbounded descriptor
        # would let the LPR-1 non-storing lanes of every row write 2 GB past the tensor.
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(total * fx.Index(4))
        )

        # THD packs O/dO as [B,S,Hq,D] but SBHD is [S,B,Hq,D], so the seq step is B*Hq*D
        # there. DELTA stays batch-major [B,Hq,S] in both cases.
        if const_expr(sbhd):
            base = ((s * fx.Index(batch_size) + b) * fx.Index(NUM_HEADS_Q) + hq) * fx.Index(HEAD_DIM)
        else:
            base = ((b * sl + s) * fx.Index(NUM_HEADS_Q) + hq) * fx.Index(HEAD_DIM)
        # This lane's 16 B slice of the row; both loads are in flight before either is used.
        off = base + chunk * fx.Index(VEC)
        ov = buffer_ops.buffer_load(o_rsrc, off, vec_width=VEC, dtype=elem_dtype_l)
        dv = buffer_ops.buffer_load(do_rsrc, off, vec_width=VEC, dtype=elem_dtype_l)
        prod = Vec(ov).to(fx.Float32) * Vec(dv).to(fx.Float32)
        acc = fx.Float32(0.0)
        for i in range_constexpr(VEC):
            acc = fx.Float32(_fadd(acc, Vec(prod)[i]))

        # Fold the LPR lanes of a row: they differ only in the low log2(LPR) lane bits.
        lane_i32 = fx.Int32(tid % fx.Index(64))
        for m in [1 << i for i in range_constexpr(LPR.bit_length() - 1)]:
            idx = _raw((lane_i32 ^ fx.Int32(m)) * fx.Int32(4))
            part = _raw(Vec.from_elements([acc], fx.Float32).bitcast(fx.Int32)[0])
            peer = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part)
            peer_f = fx.Float32(_raw(Vec.from_elements([fx.Int32(peer)], fx.Int32).bitcast(fx.Float32)[0]))
            acc = fx.Float32(_fadd(acc, peer_f))

        # DELTA is transposed [B,Hq,S]: delta[b,hq,s] at (b*Hq + hq)*S + s.
        delta_off = (b * fx.Index(NUM_HEADS_Q) + hq) * sl + s
        neg_acc = arith.subf(_raw(c_zero_f), _raw(acc), fastmath=fm)
        buffer_ops.buffer_store(
            fx.Float32(neg_acc),
            delta_rsrc,
            delta_off * fx.Index(4),
            mask=in_range & ArithValue(chunk == fx.Index(0)),
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
        grid_x = (
            fx.Index(batch_size)
            * fx.Index(NUM_HEADS_Q // HPW)
            * ((fx.Index(seq_len) + fx.Index(SPW - 1)) // fx.Index(SPW))
        )
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


def build_flash_attn_bwd_lset_module(B, Sq, Hq, scale, block=256):
    """OUT[b,h,s] = scale * IN[b,s,h], fp32 (the lse transpose-prescale).

    A work-group moves a TS x TH tile through LDS so both the load (TH consecutive heads of
    one q) and the store (TS consecutive q of one head) are 128 B contiguous. Row stride is
    padded by 4 floats so the strided LDS read hits four different banks.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "lse transpose kernel targets gfx950"
    TS = TH = _LSET_TILE
    VEC = 4
    BLOCK = block
    ROW = TH + VEC  # LDS row stride in floats
    TPR = TH // VEC  # threads covering one tile row on the load side
    assert TS * TH == BLOCK * VEC and Sq % TS == 0 and Hq % TH == 0
    NST, NHT = Sq // TS, Hq // TH
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="flash_attn_bwd_lset_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + TS * ROW * 4

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_lset_kernel(LSE: fx.Tensor, OUT: fx.Tensor):
        lds = SmemPtr(allocator.get_base(), lds_off, fx.Float32.ir_type, shape=(TS * ROW,)).get()
        bid = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        ht = bid % fx.Index(NHT)
        _r = bid // fx.Index(NHT)
        st = _r % fx.Index(NST)
        b = _r // fx.Index(NST)
        in_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
        s0 = st * fx.Index(TS)
        h0 = ht * fx.Index(TH)

        # load: VEC consecutive heads of one q
        _s = tid // fx.Index(TPR)
        _h = (tid % fx.Index(TPR)) * fx.Index(VEC)
        Vec(
            buffer_ops.buffer_load(
                in_rsrc,
                ((b * fx.Index(Sq) + s0 + _s) * fx.Index(Hq)) + h0 + _h,
                vec_width=VEC,
                dtype=fx.Float32,
            )
        ).store(lds, [_s * fx.Index(ROW) + _h])
        gpu.barrier()

        # store: VEC consecutive q of one head
        _h = tid // fx.Index(TPR)
        _s = (tid % fx.Index(TPR)) * fx.Index(VEC)
        _sv = Vec.make_type(1, fx.Float32)
        _out = Vec.from_elements(
            [
                fx.Float32(Vec.load(_sv, lds, [(_s + fx.Index(j)) * fx.Index(ROW) + _h])[0])
                for j in range_constexpr(VEC)
            ],
            fx.Float32,
        ) * Vec.filled(VEC, scale, fx.Float32)
        buffer_ops.buffer_store(
            _out.ir_value(),
            out_rsrc,
            (((b * fx.Index(Hq) + h0 + _h) * fx.Index(Sq)) + s0 + _s) * fx.Index(4),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch_flash_attn_bwd_lset(LSE: fx.Tensor, OUT: fx.Tensor, stream: fx.Stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        flash_attn_bwd_lset_kernel(
            LSE,
            OUT,
            value_attrs={"rocdl.flat_work_group_size": f"{int(BLOCK)},{int(BLOCK)}"},
        ).launch(grid=(fx.Index(B * NST * NHT), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    _compiled: dict = {}

    def _launch(*args, **kwargs):
        return _cached_launch(_compiled, launch_flash_attn_bwd_lset, None, args, kwargs)

    def _compile(*args):
        return flyc.compile(launch_flash_attn_bwd_lset, *args)

    _launch.compile = _compile
    return _launch


def build_flash_attn_bwd_dqred_module(
    num_heads,
    head_dim,
    batch_size,
    seq_len_q,
    block_kv,
    sm_scale,
    dtype_str="bf16",
    block=None,  # None: widest work-group that still tiles rows_per_wg*Hq*D (see below)
    rows_per_wg=2,
    lpt=True,
):
    """Fold the fused kernel's dQ split-K partials: DQ[b,q] = sm * Sum_b' WSQ[b',b,q].

    Only the bands a q row causally sees (b' <= q/BLOCK_KV) are read, in ascending
    order and with an fp32 accumulator, so the result is bitwise reproducible without
    atomics. One pass replaces torch's sum -> mul_ -> cast chain, which materialises an
    fp32 [B, BLOCK_KV*Hq*D] temporary per q group and touches it three more times.

    A work-group owns ``rows_per_wg`` q rows (one q group, hence one band count) and
    every thread carries ``RPW*Hq*D/(block*VEC)`` independent 16 B chunks, so that many
    loads per band are in flight -- the band loop is dynamic and cannot be unrolled.
    Sweeping it measured 1 -> 752.6, 2 -> 756.7, 4 -> 734.2 TF on the scored shape.
    The work-group width is a wash without ``lpt`` (612.9 to 642 us over 8 shapes) but
    not with it: 512x2 -> 563.8 us / 6.07 TB/s against 256x1 581.5, 256x2 583.3,
    512x8 570.3, 128x2 596.7, 64x1 607.6. Every shape returns bit-identical dQ, so this
    is purely a rate knob -- re-sweep with _r5_dqred.py if ``lpt`` or the layout changes.
    Isolated is not in-situ, though: 1024x8 wins the sweep (557.2 us) and loses 0.4% of
    the whole backward, so confirm any pick with the end-to-end bench as well.

    Both sides are non-temporal: a partial is read exactly once by exactly this kernel
    and dQ is not read again in the backward, so keeping either in L2/MALL only evicts
    the dO the fused kernel re-reads once per band (+1.4% for the pair). The same hint
    is a LOSS on the odo kernel's O/dO reads (-2.6%) and on the fused kernel's partial
    store, so it is not a general lever -- see the sign flip in memory.md.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "dq reduce kernel targets gfx950"
    elem_dtype = dtype_to_elem_type(dtype_str)
    HD = num_heads * head_dim
    SQ = seq_len_q
    VEC = 8
    RPW = rows_per_wg
    if block is None:
        cands = [b for b in (512, 256, 128, 64, 32) if (RPW * HD) % (b * VEC) == 0]
        assert cands, f"cannot tile {RPW}*{HD} elements into {VEC}-element lanes"
        block = cands[0]
    BLOCK = block
    LPT = lpt
    UC = RPW * HD // (BLOCK * VEC)
    assert RPW * HD == BLOCK * VEC * UC, "rows_per_wg*Hq*D must tile the work-group"
    assert block_kv % RPW == 0 and (batch_size * SQ) % RPW == 0
    BAND_BYTES = batch_size * SQ * HD * 2

    NWG = batch_size * SQ // RPW

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_dqred_kernel(WSQ: fx.Tensor, DQ: fx.Tensor):
        bid = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        if const_expr(LPT):
            # Longest-processing-time-first. A work-group reads g+1 bands with
            # g = q/BLOCK_KV, so the work per work-group ramps 1 -> Skv/BLOCK_KV along the
            # grid and the heaviest ones are dispatched LAST, leaving them to drain alone
            # once the light ones are done. Walking the grid backwards starts every batch
            # with its heaviest rows and ends on its lightest, at identical addresses
            # (reversed) so the DRAM stream stays sequential and the output is bit-identical.
            bid = fx.Index(NWG - 1) - bid
        row0 = bid * fx.Index(RPW)  # b*SQ + q of this work-group's first row
        g = (row0 % fx.Index(SQ)) // fx.Index(block_kv)  # topmost band this group sees
        base = row0 * fx.Index(HD) + tid * fx.Index(VEC)
        offs = [base + fx.Index(c * BLOCK * VEC) for c in range_constexpr(UC)]
        c_zero_vec = Vec.filled(VEC, 0.0, fx.Float32).ir_value()

        acc = [c_zero_vec for _ in range_constexpr(UC)]
        for band, inner in range(fx.Index(0), g + fx.Index(1), fx.Index(1), init=acc):
            # One descriptor per band: the whole workspace overflows a 32-bit
            # num_records, a single band slab does not, and the band base is 64-bit.
            band_rsrc = buffer_ops.create_buffer_resource(
                WSQ,
                max_size=False,
                num_records_bytes=_raw(fx.Index(BAND_BYTES)),
                base_byte_offset=_raw(band * fx.Index(BAND_BYTES)),
            )
            parts = [
                buffer_ops.buffer_load(band_rsrc, o, vec_width=VEC, dtype=elem_dtype, cache_modifier=2)
                for o in offs
            ]
            acc = yield [
                (Vec(inner[c]) + Vec(parts[c]).to(fx.Float32)).ir_value()
                for c in range_constexpr(UC)
            ]

        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=True)
        sm_vec = Vec.filled(VEC, sm_scale, fx.Float32)
        for c in range_constexpr(UC):
            buffer_ops.buffer_store(
                (Vec(acc[c]) * sm_vec).to(elem_dtype).ir_value(),
                dq_rsrc,
                offs[c] * fx.Index(2),
                cache_modifier=2,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch_flash_attn_bwd_dqred(WSQ: fx.Tensor, DQ: fx.Tensor, stream: fx.Stream):
        flash_attn_bwd_dqred_kernel(
            WSQ,
            DQ,
            value_attrs={"rocdl.flat_work_group_size": f"{int(BLOCK)},{int(BLOCK)}"},
        ).launch(grid=(fx.Index(batch_size * SQ // RPW), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    _compiled: dict = {}

    def _launch(*args, **kwargs):
        return _cached_launch(_compiled, launch_flash_attn_bwd_dqred, None, args, kwargs)

    def _compile(*args):
        return flyc.compile(launch_flash_attn_bwd_dqred, *args)

    _launch.compile = _compile
    return _launch


def build_flash_attn_bwd_slotred_module(
    n_slots,
    n_groups,
    n_elems,
    dtype_str="bf16",
    block=256,
    uc=2,
):
    """Fold two split-K workspaces in one pass: OUT[g,i] = Sum_{s<NS} WS[g,s,i].

    This is the dK/dV q_split reduction. torch's ``sum(dim=1)`` reduces a strided axis and
    runs it at 4.5 TB/s over two launches; folding both tensors in one flat pass keeps NS
    loads per thread in flight and reaches the dQ reduce kernel's ~6 TB/s. Ascending slot
    order with an fp32 accumulator, so the result is bitwise reproducible.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "slot reduce kernel targets gfx950"
    elem_dtype = dtype_to_elem_type(dtype_str)
    VEC = 8
    BLOCK = block
    UC = uc
    TILE = BLOCK * UC * VEC  # elements one work-group folds, per tensor
    assert n_elems % TILE == 0, "n_elems must tile the work-group"
    WPG = n_elems // TILE
    NS = n_slots

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_slotred_kernel(
        WSK: fx.Tensor, DK: fx.Tensor, WSV: fx.Tensor, DV: fx.Tensor
    ):
        bid = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        grp = bid // fx.Index(WPG)
        tile = bid % fx.Index(WPG)
        o_base = grp * fx.Index(n_elems) + tile * fx.Index(TILE) + tid * fx.Index(VEC)
        w_base = grp * fx.Index(NS * n_elems) + tile * fx.Index(TILE) + tid * fx.Index(VEC)
        # Both tensors are read exactly once and their outputs are not read again in the
        # backward, so nothing here belongs in L2 -- the same non-temporal pair the dQ
        # reduce uses.
        for _ws, _out in ((WSK, DK), (WSV, DV)):
            ws_rsrc = buffer_ops.create_buffer_resource(_ws, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(_out, max_size=True)
            parts = [
                [
                    buffer_ops.buffer_load(
                        ws_rsrc,
                        w_base + fx.Index(s * n_elems + c * BLOCK * VEC),
                        vec_width=VEC,
                        dtype=elem_dtype,
                        cache_modifier=2,
                    )
                    for s in range_constexpr(NS)
                ]
                for c in range_constexpr(UC)
            ]
            for c in range_constexpr(UC):
                acc = Vec(parts[c][0]).to(fx.Float32)
                for s in range_constexpr(1, NS):
                    acc = acc + Vec(parts[c][s]).to(fx.Float32)
                buffer_ops.buffer_store(
                    acc.to(elem_dtype).ir_value(),
                    out_rsrc,
                    (o_base + fx.Index(c * BLOCK * VEC)) * fx.Index(2),
                    cache_modifier=2,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch_flash_attn_bwd_slotred(
        WSK: fx.Tensor, DK: fx.Tensor, WSV: fx.Tensor, DV: fx.Tensor, stream: fx.Stream
    ):
        flash_attn_bwd_slotred_kernel(
            WSK,
            DK,
            WSV,
            DV,
            value_attrs={"rocdl.flat_work_group_size": f"{int(BLOCK)},{int(BLOCK)}"},
        ).launch(grid=(fx.Index(n_groups * WPG), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    _compiled: dict = {}

    def _launch(*args, **kwargs):
        return _cached_launch(_compiled, launch_flash_attn_bwd_slotred, None, args, kwargs)

    def _compile(*args):
        return flyc.compile(launch_flash_attn_bwd_slotred, *args)

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
    # q_dbuf: stage the GQA group's Q/dO tiles into two alternating LDS slots so head h's
    # step issues head h+1's DMA up front and drains it only at its own tail. One head-step
    # of MFMA then covers the fetch, and a single barrier per head-step serves both edges
    # (it makes the prefetch visible AND fences the slot that the next step overwrites),
    # halving the rendezvous count. Costs LDS_TOTAL extra bytes -- only affordable when a
    # work-group owns the CU (waves_per_eu=1). On the fused body it measured 753 vs 780 at
    # equal GEMM3 ring depth: the DMA's LDS writes then interleave with the GEMMs' LDS reads
    # for a whole head-step, and that costs more than the drain it hides.
    q_dbuf=False,
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
    # sched_strategy: LLVM amdgpu-sched-strategy override (None = compiler default). At D128
    # the GEMM2 ds_read_tr16 transpose-reads are latency-bound and scattered across compute
    # clusters (LdsUtil/MfmaUtil both <60%); "max-memory-clause" clusters those LDS reads to
    # hide their latency. D64 (MfmaUtil-bound) keeps None -> byte-identical.
    sched_strategy=None,
    # dma_grp: how many GQA heads stage their Q/dO tiles in one shot, see _q_body.
    dma_grp=1,
    # pf_ring: double the Q/dO slot ring (2*dma_grp deep) and stage one head-group ahead,
    # so the whole rendezvous collapses to ONE barrier parked inside a GEMM2 run instead
    # of a barrier pair at the head boundary. See _head_step_lds/_q_body.
    pf_ring=False,
    # g1_ks_outer: emit GEMM1's D-contraction outermost so its accumulator chains
    # interleave instead of running one dependent MFMA after another. See _gemm_qk.
    g1_ks_outer=None,  # None = on for D128
    varlen=False,  # ragged / block-causal: per-segment [tok_base,tok_end) from cu_seqlens
    # fuse_dq: also emit dQ (the fifth GEMM) into a split-K workspace, replacing the
    # separate Q-outer dq kernel. See _FUSE_DQ and _gemm3 below.
    fuse_dq=False,
    # v_lds: stage the owned V rows in LDS as GEMM1b's B operand instead of keeping them
    # in registers. K has to be staged either way (GEMM3 transpose-reads it), V does not.
    # Registers win by 2.1%: they cost 16 VGPR but remove NT*K_STEPS_QK LDS reads per
    # q-half, and this body stalls on LDS instruction issue, not on LDS capacity -- the
    # 32 KB freed stays unspent, since both ways of spending it (a second Q/dO slot via
    # q_dbuf or via dma_grp=2) measured slower.
    v_lds=False,
    k_reg=True,  # feed GEMM1a's B from the K register packs, not the LDS tile (see K_REG)
    # q_pref: stage the Q/dO tiles through VGPRs and issue head h+1's fetch at the top of
    # head-step h, so a whole head-step covers it. See Q_PREF.
    q_pref=False,
    # flat_wg: work-group size. 512 (8 waves, one work-group per CU) doubles the kv rows
    # a work-group owns at CONSTANT per-wave state -- ROWS_PER_WAVE_KV, NT and every
    # NT-scaled accumulator stay put -- which is the only way to widen the band without
    # crossing the register cliff. The fused path buys halved dQ split-K traffic with it.
    flat_wg=256,
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
    flat_work_group_size = flat_wg
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_KV = BLOCK_KV // NUM_WAVES

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32): four independent 16x16 accumulator
    # chains at the same accumulator VGPR total (dkdv is MFMA dep-wait bound). Lane layout:
    # lane%16 = M/N index, lane//16 = K-subgroup (4 x 8 = K32) and, on the C output, the
    # M-block ((lane//16)*4 + t, t in 0..3 -> 4 f32/lane).
    # 32x32x16 buys nothing here, and GEMM2 cannot even take it. On the LDS side a transpose
    # read delivers 64 lanes x 8 B and a ds_read_b128 64 x 16 B whatever the MFMA shape, and a
    # 2x2 patch of 16-wide MFMAs spans the same 32x32 output block from the same fragments as
    # one 32-wide MFMA -- GEMM3 reads 64 tr per carrier per head-step either way. So all it
    # does is halve the MFMA COUNT (1536 -> 768), and issue is nowhere near a limit (0.14 LDS
    # + 0.06 MFMA per CU cycle). GEMM2's B operand rules it out outright: P/dS sit in the
    # GEMM1 C-layout, where lanes 0-15 hold q 0-3 and lanes 16-31 hold q 4-7, while 32x32x16
    # wants all 32 lanes of a half on the SAME 8 k -- no k-permutation fixes that (the map has
    # to be lane-uniform within the half), only a cross-lane shuffle or a second LDS trip. ----
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

    # ---- Fifth GEMM (dQ). dQ^T[m=D][n=q] = K^T @ dS^T contracts over the block's kv
    # rows, so both operands need the kv axis as the MFMA k axis: the prescaled K tile
    # ([kv][D]) and a dS staging tile ([kv][BLOCK_Q]) both live in LDS in the Q/dO tile
    # layout and are read transposed (ds_read_tr16). The two reads share one row->k
    # mapping, so the kv permutation the transpose imposes cancels out. Each wave owns a
    # G3_DT x G3_QT patch of the DT x MT output tiles and contracts the WHOLE band, so no
    # cross-wave reduction is needed (only the RAW barrier on the dS tile). ----
    FUSE_DQ = bool(fuse_dq)
    # K has to be in LDS for GEMM3; V only ever feeds GEMM1b, so staging it too is purely
    # a register trade. It wins by a wide margin: leaving the V packs in registers costs
    # 16 VGPR for the whole kernel and measured 225 spill dwords (vs 36) and -6%.
    G3_KSTEPS = BLOCK_KV // PV_K_STEP  # kv 32-steps per band
    # A wave's output patch is the squarest G3_TILES-tile rectangle, because the transpose
    # reads per MFMA are (G3_DT + G3_QT) / (G3_DT * G3_QT): 2x2 reads 4 fragments for 4
    # MFMAs, 1x2 reads 3 for 2. One head has DT*MT = 16 output tiles, so the square patch
    # exists for exactly 4 waves -- at 8 waves spreading it over all of them would force the
    # 1x2 shape and 50% more LDS traffic for the same MFMAs. GEMM3 therefore runs on the
    # FIRST G3_WAVES waves only, which is MFMA-neutral (waves are round-robin over the 4
    # SIMDs, so one carrier per SIMD does the 32 MFMAs its two waves would have done 16 each)
    # and cuts GEMM3's share of the LDS data path by a third. Covering a head PAIR in one run
    # would amortise the head-INVARIANT K^T fragments as well -- reads per MFMA
    # (G3_DT + G3_QT)/(G3_DT*G3_QT) = 1.0 becomes (G3_DT + 2*G3_QT)/(2*G3_DT*G3_QT) = 0.75 --
    # and stays fence-free on a 3-slot dS ring -- but every form of it loses. Sharing the K^T
    # fragments needs both heads' accumulators live at once (32 dwords instead of 16) and that
    # clears 256 VGPR at EVERY ring depth: occ 1 and 664..788 TF against 812 (depth 6/1/2/3),
    # so the quarter of GEMM3's reads it saves is simply unreachable. Giving each of the
    # NUM_WAVES waves a head of its OWN instead keeps both the register footprint and the read
    # count exactly as they are and perfectly balances the carriers' extra 32 MFMAs, yet still
    # measured 8/9 losses (-0.5%): GEMM3's reads then arrive in half as many, twice as dense
    # windows. Neither direction is worth revisiting, because LDS reads are not what this body
    # waits on -- deleting a quarter of them measures 0.0% (see memory.md read probes), and
    # deleting a third of GEMM3's own is 6/11. Spreading the SAME head over all NUM_WAVES as
    # a D-wide 2-tile patch (G3_DT=2, G3_QT=1, which keeps the dQ store at one 64 B request
    # per row) is the third losing form: 1280 MFMA / 243 VGPR / 0 spill but 0/11 wins at
    # -3.9%, and running the upper half's patch mid head-step to de-phase the two waves of a
    # SIMD recovers only 0.4% of that (-3.5%, 2/11). The 4-carrier 2x2 patch is what pays:
    # the carriers' 32-MFMA run at the head-step top is also what offsets a SIMD's two waves
    # so their quarter-rate exp runs do not collide, and moving that run to mid head-step
    # (after q-half 0's GEMM2) is 9/11 one session and 5/15 the next = noise.
    # GEMM3 kstep transpose-read prefetch depth. Sweep at the head-step-top emission point:
    # 2 -> 739.3, 3 -> 741.9, 4 -> 743.7, 6 -> 746.3, 8 (all ksteps) -> 738.8 TF. Depth 3 was
    # a 392 TF spill cliff while GEMM3 still sat after GEMM2, so the position is what buys
    # the depth: with the ring's fragments no longer overlapping GEMM2's live values, the
    # ds_read_tr16 latency hides under the MFMAs instead of exposing at each kstep. Re-swept
    # once GEMM1's B operands moved to registers: 2 -> 780.3, 6 -> 783.2.
    G3D = min(6, G3_KSTEPS)
    G3_WAVES = min(NUM_WAVES, max(1, DT * MT // min(4, DT * MT)))  # waves carrying GEMM3
    G3_TILES = max(1, DT * MT // G3_WAVES)  # output tiles per carrier wave
    G3_QT = min(MT, 2 if G3_TILES >= 2 else 1)  # q 16-tiles per wave
    G3_DT = G3_TILES // G3_QT  # D 16-tiles per wave
    G3_QGRP = MT // G3_QT  # q-tile groups; wave -> (D group, q group)
    if FUSE_DQ:
        assert head_dim == 64 and window_left < 0 and not sbhd and not varlen
        assert fold_lse, "fused dQ reads the prescaled K tile; see _reduce_dq_partials"
        assert DT * MT == G3_WAVES * G3_DT * G3_QT, "GEMM3 tiles must partition over carriers"
        assert BLOCK_KV % PV_K_STEP == 0
        assert batch_size is not None, "fused dQ needs compile-time B for the workspace stride"
        assert dma_grp == 1, (
            "fused dQ rides the per-head Q/dO staging barriers as the dS WAR fence"
        )

    # sched_barrier(TRANS) pins MFMA/ds_read/VALU in place and frees only the softmax's
    # quarter-rate v_exp to migrate, so the exps are what fills GEMM1b's MFMA latency shadow
    # (schedule-only: opcode multiset and output unchanged).
    SCHED_TRANS = 0x400  # LLVM SchedGroupMask: TRANS (v_exp)
    # G2_HALF: run GEMM2 once per q-half instead of once per head-step (fused body only;
    # the split bodies keep the single call so their ISA stays byte-identical). See the
    # emission point in _head_step_lds for what it buys and what it does not.
    G2_HALF = FUSE_DQ
    # s_waitcnt SIMM16 selecting lgkmcnt(0) alone: vmcnt/expcnt stay at their maxima, so the
    # wait retires the LDS traffic without also retiring in-flight global stores.
    WAIT_LGKM = 0xC07F

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
    # DMA_GRP heads share one Q/dO staging round-trip (see _q_body). One LDS slot per
    # head of the group; occ is register-bound at 1 for D128, so the extra LDS is free.
    DMA_GRP = max(1, int(dma_grp))
    assert GQA_GROUP_SIZE % DMA_GRP == 0, "dma_grp must divide the GQA group"
    # PF_RING doubles the ring so a group's tiles are staged one group-step before they
    # are read. The slots a refill overwrites were then last read a whole group earlier,
    # so ONE barrier both publishes the pending group and fences those slots -- the WAR
    # barrier of the pair disappears. That single barrier is parked on the last GEMM2
    # step of the group's last head rather than at the head boundary, which is what
    # actually pays: it leaves every head boundary fence-free, so head h+1's GEMM1 and
    # exp2 chain schedule into head h's GEMM2 shadow.
    PF_RING = bool(pf_ring) and ENABLE_DMA and not q_dbuf
    # Q_PREF: fetch the Q/dO tile into VGPRs and ds_write it, instead of buffer_load ... lds.
    # The DMA route cannot be given a shadow at all: a pending buffer_load ... lds forces
    # vmcnt(0) before every later ds_read of the same LDS allocation, so wherever the issue
    # point is moved the drain reappears at the next LDS read (see G3_SHADOW, q_dbuf,
    # dma_grp=2 -- all measured losses). Through VGPRs the fetch is an ordinary VMEM load
    # with no LDS dependence, so head h+1's tile is issued at the top of head-step h and
    # only waited on at its ds_write one head-step later, and every other wait in the body
    # becomes a partial vmcnt. The price is one 16 B slice per tensor held across the body.
    # It runs at LDS_SLOTS == 1 (hence the DMA_GRP == 1 assert). A 2-slot ring DOES retire the
    # WAR barrier of the staging pair -- the slot head h overwrites was last read by head h-2,
    # whose reads all precede head h-1's publish barrier, and the same one remaining barrier
    # still fences the dS ring -- and the ISA confirms it: 34 s_barrier -> 18, LDS 118,784 ->
    # 135,168 B, occ unchanged. It is still a loss, and NOT because of the spill it brings:
    # 83 dwords / 240 B of scratch and 744 TF before G2_HALF, 26 dwords / 76 B and 9/9 losses
    # at -3.3% after it, and taking that spill to 2 dwords / 12 B -- by feeding GEMM1a's B
    # from LDS, which frees the K packs' 32 whole-loop dwords (k_reg=False) -- still measures
    # 11/11 losses at -2.9% and -2.8% in two separate sessions. So the 24 spilled dwords were
    # worth 0.2%, this body pays almost nothing for a small spill, and what the second slot
    # actually costs is elsewhere: dropping 14 s_barrier ADDS 13 lgkmcnt(0) full LDS drains
    # (185 -> 198), because the rendezvous is what the drains were already attached to. The
    # NOBAR probe's 0.339 ms therefore prices the DRAINS, not the barriers, and no amount of
    # register relief buys it. Issuing the fetch even earlier is also spent: QPF_AT=0, ahead
    # of the deferred GEMM3's partial stores so the commit's vmcnt(0) no longer retires them,
    # is 5/9.
    Q_PREF = bool(q_pref) and ENABLE_DMA and not q_dbuf and not PF_RING and DMA_GRP == 1
    # Issue point for that fetch. Earlier gives it more of the step as shadow but keeps its
    # 16 B per tensor live over GEMM2's accumulator peak. Interleaved A/B: before GEMM1 and
    # before GEMM2 tie (5/9 wins, 28 B of scratch against 48), inside GEMM2 loses 3.5% -- by
    # then too little of the step is left to cover the fetch. The cheaper-scratch tie wins.
    QPF_AT = 2
    # PF_QB: the LAST head-step of a q-block has no next head to fetch for, so it issues
    # head 0's fetch of the NEXT q-block instead -- Q/dO and the group's (-delta, lse) --
    # and the values ride the q-loop's iter_args. The fused body runs ONE work-group per CU
    # (8 waves, LDS 118,784 B), so nothing else is resident to cover a q-block prologue:
    # without this, every 8th head-step starts by waiting out the full HBM latency of its
    # first fetch, with only a barrier in between. The last head-step is also where the
    # register pressure is lowest (it is the one step that currently prefetches nothing),
    # so the extra live values land in the cheapest place in the body.
    # Fused body only, so the split bodies stay byte-identical. Interleaved A/B against
    # the same tree: +0.29% (8/11) and, once GEMM1 also emits ks-outer, +0.55% (10/11)
    # and 15/15 for the pair -- the two compound, since ks-outer only helps while the
    # operands are already there and the prologue fetch is what keeps them coming.
    PF_QB = Q_PREF and FUSE_DQ
    # MASK_SKIP: let a wave sit out a diagonal q-block whose kv rows it cannot see. Its
    # P and dS are zero there, so this only removes work -- the output is bitwise equal.
    # Full-causal only (a left window would need the lower edge too), fused only.
    MASK_SKIP = FUSE_DQ and window_left < 0
    # GEMM1 emits ks-outer: the four kv tiles of one k-step first, so consecutive MFMAs
    # write different accumulators and the next one issues without waiting on the last
    # one's result. At D128 (one wave per SIMD) there is no sibling wave to cover that
    # latency at all; at D64 it still pays, together with PF_QB (see below).
    G1_KS_OUTER = (HEAD_DIM == 128 or FUSE_DQ) if g1_ks_outer is None else bool(g1_ks_outer)
    LDS_SLOTS = 2 if q_dbuf else ((2 * DMA_GRP) if PF_RING else DMA_GRP)
    assert GQA_GROUP_SIZE % LDS_SLOTS == 0
    # Share one SGPR LDS base pointer across every DMA destination (D128 only; D64 keeps
    # the per-destination pointer table so its ISA stays byte-identical).
    DMA_SHARED_PTR = HEAD_DIM == 128

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

    # The Q/dO slot ring, and under FUSE_DQ the K, V and dS tiles, share one element-indexed
    # view so every reader (_a_idx / _read_tr / _kv_lds_idx / _g3s_idx) addresses them the
    # same way.
    LDS_VIEW_ELEMS = LDS_TOTAL * LDS_SLOTS
    V_LDS = FUSE_DQ and bool(v_lds)
    # K_REG: GEMM3 transpose-reads the staged K tile, but GEMM1a's B operand can come
    # from the register packs that filled it instead of being re-read from LDS once per
    # q-half. Costs NT*K_STEPS_QK v8 for the whole loop, saves 2*NT*K_STEPS_QK LDS reads
    # per head-step (the LDS copy stays -- this is a read-side choice, not a staging one).
    # It is worth 2.0% (11/11 losses without it, 384 -> 512 ds_read_b128 and 249 -> 243
    # VGPR), which is the ONE asymmetry in this body's LDS accounting: DELETING a quarter
    # of the LDS reads measures 0.0%, but ADDING these 8 per head-step costs 2.0%, because
    # they land INSIDE GEMM1's MFMA run as fresh SrcB dependencies and break its density.
    # Hoisting reads earlier does not reproduce the gain -- issuing the next q-half's A
    # fragments a half-step early is 7/15, and the next head's (-delta, lse) a head-step
    # early (legal: the staging is written once per q-block for the whole GQA group, and
    # -9 lgkmcnt(0) drains confirm they leave the prerequisite set) is 7..9/15 against.
    # So it is MFMA-run density that this body pays for, not LDS latency or read count.
    K_REG = FUSE_DQ and bool(k_reg)
    G3K_BASE = LDS_VIEW_ELEMS  # prescaled K [BLOCK_KV][HEAD_DIM]
    G3V_BASE = G3K_BASE + (BLOCK_KV * HEAD_DIM if FUSE_DQ else 0)  # V [BLOCK_KV][HEAD_DIM]
    G3S_BASE = G3V_BASE + (BLOCK_KV * HEAD_DIM if V_LDS else 0)  # dS [slot][BLOCK_KV][BLOCK_Q]
    # GEMM3 lags one head-step behind the head that produced dS and reads the OTHER slot,
    # so its RAW edge is covered by the head boundary's own staging barrier pair (drain +
    # publish) and its WAR edge by the pair one step later. Both dS fences per head-step
    # then disappear; the price is one extra dS slot.
    G3S_SLOT_ELEMS = BLOCK_KV * BLOCK_Q
    # Spending this slot's LDS on a second Q/dO ring slot instead (q_dbuf, so the DMA drain
    # lands a whole head-step after its issue) and letting GEMM3 fence its own dS costs 6%:
    # a barrier plus an lgkmcnt fence per head-step is dearer than the drain it removes.
    G3_DEFER = FUSE_DQ
    G3S_SLOTS = 2 if G3_DEFER else 1
    # G3_SHADOW: emit the deferred GEMM3 INSIDE the rendezvous, between the Q/dO DMA issue
    # and its drain. The rendezvous is otherwise a bare [barrier, 2x buffer_load ... lds,
    # full drain, barrier] with nothing in the fetch's shadow, once per head-step, and
    # GEMM3 is the only work that reads neither the slot being filled nor GEMM1's output.
    # The drain has to land BETWEEN GEMM3's MFMA run and its dQ partial stores: with the
    # stores ahead of a trailing full drain (gfx950 shares one vmcnt between loads and
    # stores and has no subset wait) the drain waits on them instead, which costs 5.6%.
    # It does not pay even then: a pending buffer_load ... lds forces vmcnt(0) before every
    # subsequent ds_read of the same LDS allocation, and LLVM plants that wait at GEMM3's
    # first transpose-read, so the shadow never materialises (30 full drains against 16,
    # 740.1 TF against 779.0). DMA-to-LDS latency can only be covered by work that touches
    # no LDS at all, which is also why q_dbuf (753) and dma_grp=2 (767) lose.
    G3_SHADOW = False
    if FUSE_DQ:
        LDS_VIEW_ELEMS = G3S_BASE + G3S_SLOTS * G3S_SLOT_ELEMS

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="flash_attn_bwd_smem_dkdv")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_VIEW_ELEMS * 2
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
        WSQ: fx.Tensor,  # fuse_dq: dQ partials [kv_band, B, Sq, Hq, D] bf16; else placeholder
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
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_VIEW_ELEMS,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16  # M/N index within a 16-tile
        kg = lane // 16  # 0..3: K-subgroup (inputs) / M-block (C output)

        def ds_read_tr_v4f16(lds_elem_idx, const_elem_off=0):
            # const_elem_off is a compile-time element offset that the backend folds into the
            # ds_read offset field, letting a family of reads share one address register.
            byte_offset = lds_elem_idx * 2 + lds_off
            ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
            if const_expr(const_elem_off != 0):
                ptr = buffer_ops.get_element_ptr(ptr, fx.Int64(const_elem_off), elem_type=elem_type)
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
            # The q_split axis is deliberately the FASTEST after the XCD term. All Q_SPLIT
            # work-groups of a band walk interleaved q-blocks of the same band, so a
            # resident window covers every q-block of a whole band group with Q_SPLIT
            # readers each; both alternatives lose. Making the band the slowest axis
            # (longest-first list scheduling, since a band's work is num_kv_tiles -
            # kv_tile_idx q-blocks) is 0/11 at -1.3%, and making it the fastest -- one
            # split's 32 bands co-resident, which spreads the same window over Q_SPLIT
            # times the q range with one reader each -- is 0/11 at -1.9%. Whatever the
            # dispatch order is worth here, it is spent on L2 sharing, not on makespan.
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

        # A vector fptrunc selects the same v_cvt_pk_bf16_f32 pairs as the inline-asm
        # intrinsic, but as a scored op: the backend sees the VGPR def and places the
        # pack-to-MFMA wait states itself, so the GEMM2 consumers need no hand fence.
        # The asm form hides the def from GCNHazardRecognizer, which is why the D64 path
        # (kept bit-identical) still pays for one. Same rounding -> identical bits.
        # The fused body needs the scored form for a second reason: the pack feeds GEMM2's
        # SrcB, so with the def hidden the ONLY thing satisfying that hazard is the
        # incidental instruction distance the default schedule happens to leave. Any
        # reschedule of the head-step then loses bitwise determinism.
        SCORED_PACK = HEAD_DIM == 128 or FUSE_DQ
        # A ds_read offset immediate is 16-bit unsigned, so once the top slot of the ring
        # reaches 65536 bytes the backend can no longer carry a tile base in the offset
        # field: it materialises a separate live address per A-fragment family, which on
        # this register-full body costs 125 spill dwords, 376 B of scratch and 74% of the
        # runtime (measured). Pinning one address per tile instead removes that entirely.
        # Below the limit the compile-time form is cheaper (pinning it costs ~0.7%), and
        # pinning only the overflowing slots is worse than pinning all of them (+1.3%):
        # a mixed addressing mode gives the allocator two live-range shapes to juggle.
        A_PIN = HEAD_DIM == 128 and (
            lds_off + ((LDS_SLOTS - 1) * LDS_TOTAL + LDS_DO_BASE + LDS_TILE) * 2 > 65536
        )

        def bf16_trunc_pack_v8(f32_vals):
            if const_expr(SCORED_PACK):
                f32_vec = Vec.from_elements([_raw(v) for v in f32_vals], fx.Float32)
                trunc = llvm.FPTruncOp(Vec.make_type(8, elem_dtype), _raw(f32_vec))
                trunc.operation.attributes["fastmathFlags"] = ir.Attribute.parse("#llvm.fastmath<fast>")
                return trunc.result
            pairs = [
                rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1]))
                for j in range_constexpr(4)
            ]
            return (
                Vec.from_elements([fx.Int32(_raw(p)) for p in pairs], fx.Int32).bitcast(elem_dtype).ir_value()
            )

        def bf16_trunc_scored_v4(f32_vec4):
            """Scored f32x4 -> bf16x4 pack, returned as the 2 dwords a dwordx2 store wants.

            GEMM3 reads its MFMA accumulator in the same instruction group that produced
            it, so the pack MUST be a scored op: the inline-asm form hides the read from
            GCNHazardRecognizer, which then emits no wait states and lets src0 (t=0,2)
            latch the pre-MFMA value while src1 (t=1,3) latches the new one.
            """
            trunc = llvm.FPTruncOp(Vec.make_type(4, elem_dtype), _raw(Vec(f32_vec4)))
            trunc.operation.attributes["fastmathFlags"] = ir.Attribute.parse("#llvm.fastmath<fast>")
            return Vec(trunc.result).bitcast(fx.Int32)

        # D64 packs 2 real rows into one 128-wide LDS block (low r&4=0 -> [0,64),
        # high -> [64,128)); D128 is already 128-wide, so one row == one block.
        PACK_2ROW = HEAD_DIM == 64  # host bool; gate tracer branches with const_expr()
        PBLK = 128 if PACK_2ROW else HEAD_DIM

        def _pblk(row_idx):
            if const_expr(PACK_2ROW):
                return ((row_idx >> fx.Index(3)) << fx.Index(2)) | (row_idx & fx.Index(3))
            return row_idx

        def _swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(7)) << fx.Index(4)
            return col_idx ^ mask

        def _kv_lds_idx(base, nt, ks):
            """Owned-K/V LDS slot of B[k=D=ks*32+kg*8][n=kv=nt*16+lane16], one v8 per lane.

            Same [row][col] Q/dO tile layout as the Q/dO DMA, so writer and reader share
            this one address and the fragment round-trips bit-exactly.
            """
            _r = wave_id * ROWS_PER_WAVE_KV + fx.Index(nt * N_TILE) + lane16
            _c = fx.Index(ks * K_STEP_QK) + kg * fx.Index(MFMA_LANE_K)
            return fx.Index(base) + _pblk(_r) * fx.Index(PBLK) + _swizzle(_r, _c)

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
        if const_expr(FUSE_DQ):
            # dQ partials are [kv_band, B, Sq, Hq, D]: one (band, batch) slice per SRD, so the
            # descriptor stays inside the 32-bit num_records while the whole workspace (bands x
            # B x this slice) is reached through the 64-bit base. The slice bound also clamps
            # the tail q-block when Sq % BLOCK_Q != 0.
            _wsq_slice = seq_len_q_v * fx.Index(NUM_HEADS_Q * HEAD_DIM * 2)
            wsq_rsrc = buffer_ops.create_buffer_resource(
                WSQ,
                max_size=False,
                num_records_bytes=_raw(_wsq_slice),
                base_byte_offset=_raw((kv_tile_idx * fx.Index(batch_size or 1) + batch_idx) * _wsq_slice),
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
            # D64: (BLOCK_Q/2) blocks, 2 rows each. D128: BLOCK_Q blocks, 1 row each.
            # BLOCK_Q*HEAD_DIM*2 covers both (D64: 64*64*2 == 32*128*2).
            Q_TILE_BYTES = BLOCK_Q * HEAD_DIM * 2
            NUM_DMA_Q = Q_TILE_BYTES // DMA_BATCH_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (128 * 2)  # 128-wide blocks per batch
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

            # Every Q/dO DMA destination is this wave's LDS write base plus a compile-time
            # byte offset (batch, Q vs dO, slot), and the destination reaches the hardware
            # through m0. Materialising one uniform pointer per destination pins an SGPR
            # pair each; folding the offsets into the SALU add that feeds m0 keeps a single
            # pair live, which is what lets the slot count grow past two.
            if const_expr(DMA_SHARED_PTR):
                _dma_lds_base = buffer_ops.create_llvm_ptr(
                    rocdl.readfirstlane(
                        fx.Int64.ir_type,
                        fx.Int64(lds_base_idx + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)),
                    ),
                    address_space=3,
                )

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
                    position = lane_in_block * fx.Index(8)  # swiz col within 128-block
                    if const_expr(PACK_2ROW):
                        # D64: block holds 2 rows; 8 lanes/half, real col in [0,64).
                        half = lane_in_block // fx.Index(8)
                        row_in_tile = (
                            fx.Index(8) * (block >> fx.Index(2)) + (block & fx.Index(3)) + half * fx.Index(4)
                        )
                    else:
                        # D128: block == row; 16 lanes span the full 128-wide row.
                        row_in_tile = block
                    xor_mask = (row_in_tile & fx.Index(7)) << fx.Index(4)
                    unsw_col_f16 = position ^ xor_mask  # real col (1x HBM)
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile
                    bases.append(global_row * fx.Index(RD_STRIDE_Q * 2) + col_byte)
                return bases

            if const_expr(not DMA_SHARED_PTR):
                q_lds_ptrs = [
                    _dma_lds_ptrs(lds_base_idx + fx.Index(sl * LDS_TOTAL * 2))
                    for sl in range_constexpr(LDS_SLOTS)
                ]
                do_lds_ptrs = [
                    _dma_lds_ptrs(lds_base_idx + fx.Index((sl * LDS_TOTAL + LDS_DO_BASE) * 2))
                    for sl in range_constexpr(LDS_SLOTS)
                ]

            def coop_dma_tile(src_rsrc, lds_dst, bases, q_head):
                """DMA a BLOCK_Q x head_dim Q/dO tile into the swizzled LDS layout.

                lds_dst is either the per-batch pointer list or, on the shared-pointer
                path, the tile's compile-time byte offset off _dma_lds_base.

                Address math is recomputed per tile on purpose: keeping the offsets live
                across the k_tr peak pushes VGPRs past the occ-2 boundary.
                """
                _qoff = q_head * fx.Index(HEAD_DIM * 2)
                for d in range_constexpr(NUM_DMA_Q):
                    if const_expr(DMA_SHARED_PTR):
                        _dst = buffer_ops.get_element_ptr(
                            _dma_lds_base, lds_dst + d * DMA_BATCH_BYTES
                        )
                    else:
                        _dst = lds_dst[d]
                    rocdl.raw_ptr_buffer_load_lds(
                        src_rsrc,
                        _dst,
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

        if const_expr(FUSE_DQ):
            # GEMM3 contracts over kv, so its A operand is K^T: stage the owned K into LDS
            # as [kv][D] in the Q/dO tile layout and transpose-read it back. Every lane
            # already holds MFMA_LANE_K contiguous D of its own kv row, so this is a pure
            # register->LDS repack, once per kv-block.
            # V is staged the same way for GEMM1b. Neither B operand then has to stay in
            # registers for the whole kernel: those 2*NT*K_STEPS_QK v8 packs are live from
            # the prologue to the last head-step, and dropping them is what takes the fused
            # body off the spill cliff (234 spill dwords -> see _gemm_qk).
            # K goes in ALREADY PRESCALED, so GEMM1a reads it directly; that leaves the dQ
            # partial scaled by sm*log2e, which _reduce_dq_partials divides out with 1/log2e
            # instead of multiplying by sm_scale.
            for nt in range_constexpr(NT):
                for ks in range_constexpr(K_STEPS_QK):
                    Vec(k_b_packs[nt][ks]).store(lds, [_kv_lds_idx(G3K_BASE, nt, ks)])
                    if const_expr(V_LDS):
                        Vec(v_b_packs[nt][ks]).store(lds, [_kv_lds_idx(G3V_BASE, nt, ks)])
            if const_expr(not K_REG):
                k_b_packs = G3K_BASE
            if const_expr(V_LDS):
                v_b_packs = G3V_BASE

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

        def _a_pin(a_base):
            """The (mt=0, ks=0) A-fragment address of one LDS tile (D128 only).

            Every other fragment of the tile is this address XOR a compile-time column
            term and PLUS a compile-time row term: ks*K_STEP_QK occupies bits 5-6, the
            swizzle mask occupies bits 4-6 and nothing else in the address reaches that
            field, so the XOR reproduces (col ^ mask) exactly. Holding the tile base in
            the register rather than in the ds_read offset immediate is what lets the
            slot ring grow past the 16-bit offset field (a base >= 64 KB otherwise forces
            a separate live address per slot, which this kernel has no registers for).
            """
            return _opaque_idx(
                a_base + lane16 * fx.Index(PBLK) + (kg * MFMA_LANE_K ^ a_swz_mask)
            )

        def _a_idx(a_base, mt, ks, pin=None):
            if const_expr(pin is not None):
                base = pin if const_expr(ks == 0) else pin ^ fx.Index(ks * K_STEP_QK)
                return base + fx.Index(mt * M_TILE * PBLK)
            row = fx.Index(mt * M_TILE) + lane16
            col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
            return a_base + _pblk(row) * fx.Index(PBLK) + (col ^ a_swz_mask)

        def _keepalive_v4(v4list):
            """Pin the -lse C-init registers live past GEMM1a.

            Without a later use the RA may reuse them as a later nt's MFMA output D while
            an earlier nt still reads them as C -- a WAR the hardware cannot guard. Empty
            side-effecting asm: no instruction emitted, liveness constraint only.

            The operands are "v"-constrained, so on the accumulator path each one costs a
            v_accvgpr_read. Naming one element instead of four already pins the whole
            4-aligned tuple against reuse, but only trims 55 of 1534 v_accvgpr moves and
            measures -0.2%, so all four are named.
            """
            for v4 in v4list:
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [_raw(fx.Float32(Vec(v4)[t])) for t in range_constexpr(4)],
                    "",
                    "=v,v,v,v,v",
                    has_side_effects=True,
                )

        def _gemm_qk(a_base, b_packs, inits=None, mts=None, pin=None):
            """S[mt][nt] (v4f32) = A(Q/dO)[mt] @ B(owned K/V)[nt]^T over D. inits[mt]
            optionally pre-loads the accumulator (folds -delta into the dP GEMM for free).
            mts restricts work to a subset of the MT q-tiles (per-half GEMM1); the
            output is keyed by mt so [2,3] halves index correctly.

            b_packs is either a register list or, for a tile staged in LDS, its base: the
            fragments are then re-read per head-step so they are live only across this GEMM
            rather than across the whole kernel."""
            _mts = list(range_constexpr(MT)) if mts is None else list(mts)
            if const_expr(isinstance(b_packs, int)):
                b_packs = [
                    [
                        Vec.load(mfma_pack_type, lds, [_kv_lds_idx(b_packs, nt, ks)])
                        for ks in range_constexpr(K_STEPS_QK)
                    ]
                    for nt in range_constexpr(NT)
                ]
            a = {}
            for mt in _mts:
                a[mt] = [
                    Vec.load(mfma_pack_type, lds, [_a_idx(a_base, mt, ks, pin)])
                    for ks in range_constexpr(K_STEPS_QK)
                ]
            out = {mt: [None] * NT for mt in _mts}
            if const_expr(G1_KS_OUTER):
                # Emit the D-contraction outermost so the len(_mts)*NT accumulator chains
                # interleave: consecutive MFMAs are independent instead of being the next
                # link of the same chain. At one wave per SIMD there is no sibling wave to
                # cover an MFMA's result latency, so the chains have to cover each other.
                # Each accumulator still sees ks in order -> bit-identical.
                for mt in _mts:
                    for nt in range_constexpr(NT):
                        out[mt][nt] = c_zero_v4f32 if inits is None else inits[mt]
                for ks in range_constexpr(K_STEPS_QK):
                    for mt in _mts:
                        for nt in range_constexpr(NT):
                            out[mt][nt] = mfma_acc(a[mt][ks], b_packs[nt][ks], out[mt][nt])
            else:
                for mt in _mts:
                    for nt in range_constexpr(NT):
                        acc = c_zero_v4f32 if inits is None else inits[mt]
                        for ks in range_constexpr(K_STEPS_QK):
                            acc = mfma_acc(a[mt][ks], b_packs[nt][ks], acc)
                        out[mt][nt] = acc
            return out

        def _opaque_idx(v):
            """Identity that LICM cannot hoist (empty asm, output tied to input).

            The transpose-read addresses are q-loop invariant, so the whole (dt, pks, side)
            set -- 64 values -- is hoisted into the preheader and kept live for the entire
            loop; the allocator parks it in the AGPR file and reads it back per use. Pinning
            the four bases inside the loop makes every address a short-lived XOR off a live
            base instead. Only worth it from NT=2 up, where the dK/dV accumulators leave no
            room for the hoisted set (at NT=2: 512 VGPR / 1305 AGPR moves -> 331 / 111, and
            it is what lets NT=3 fit at all); at NT=1 the set fits and the recompute is a
            pure cost (measured -1.7% on the short-Skv tile).
            """
            if const_expr(NT < 2):
                return v
            r = llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [_raw(fx.Int32(v))],
                "",
                "=v,0",
                has_side_effects=True,
            )
            return fx.Index(r)

        # A transpose read is keyed by (dt, pks, row-half); with PV_K_STEP == 2*N_TILE its
        # row is i*N_TILE for i = 2*pks + row-half, so the four (pks, row-half) variants sit
        # a compile-time row stride apart. N_TILE is a multiple of 8, hence row&7 -- the
        # swizzle mask -- is the same for all four and the stride survives as a pure element
        # offset that the backend folds into the ds_read offset field.
        assert PV_K_STEP == 2 * N_TILE

        def _tr_off(i):
            return i * N_TILE * PBLK

        def _tr_base(a_base):
            """The (dt=0, pks=0, row-half=0) transpose-read address (D128 only).

            Every other dt is this base XOR (dt*D_TILE): the swizzle mask (row&7)<<4 and
            the column term dt*16 occupy the same bit field, while the row stride (128),
            the tile base (multiple of BLOCK_Q*128) and the lane column (bits 2-3) all
            avoid it -- so bits 4-6 of the base are exactly row&7 and XORing dt in
            reproduces col ^ mask. The other (pks, row-half) reads ride _tr_off as ds_read
            offset immediates, so one XOR per (tile, dt) feeds all four reads and a single
            loop-invariant address per tile stays live instead of one per (dt, pks).
            """
            row = kg * fx.Index(4) + (lane16 // fx.Index(4))
            return _opaque_idx(
                a_base
                + _pblk(row) * fx.Index(PBLK)
                + ((row & fx.Index(7)) << fx.Index(4))
                + (lane % fx.Index(4)) * fx.Index(4)
            )

        def _read_tr(a_base, dt, pks, base=None):
            """Transpose-read Q/dO -> GEMM2 A-operand [m=D=dt*16+lane16][k=q=kg*8+s].
            Two ds_read_tr16 (4 q each): read0->s0..3 (q=pks*32+kg*4+j), read1->s4..7
            (q=pks*32+16+kg*4+j)."""
            if const_expr(HEAD_DIM == 128):
                b_dt = base ^ fx.Index(dt * D_TILE)
                v0 = ds_read_tr_v4f16(b_dt, _tr_off(2 * pks))
                v1 = ds_read_tr_v4f16(b_dt, _tr_off(2 * pks + 1))
                return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            col = fx.Index(dt * D_TILE) + (lane % fx.Index(4)) * fx.Index(4)
            row0 = fx.Index(pks * PV_K_STEP) + kg * fx.Index(4) + (lane16 // fx.Index(4))
            row1 = row0 + fx.Index(N_TILE)
            v0 = ds_read_tr_v4f16(a_base + _pblk(row0) * fx.Index(PBLK) + _swizzle(row0, col))
            v1 = ds_read_tr_v4f16(a_base + _pblk(row1) * fx.Index(PBLK) + _swizzle(row1, col))
            return Vec(v0).shuffle(Vec(v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # ---- GEMM3 (dQ) operands. Both are transpose-reads over the kv axis, so the kv
        # permutation ds_read_tr16 imposes is identical on the two sides and cancels in the
        # contraction. Wave w owns a G3_DT x G3_QT patch of the DT x MT output and contracts
        # the whole band -> no cross-wave reduction. ----
        def _g3_wave_tiles():
            return (
                (wave_id // fx.Index(G3_QGRP)) * fx.Index(G3_DT),
                (wave_id % fx.Index(G3_QGRP)) * fx.Index(G3_QT),
            )

        # Every GEMM3 transpose-read address is q-loop invariant, so left alone LICM hoists
        # the whole (kstep, tile, row-half) set -- 48 values at BLOCK_KV=256 -- into the
        # preheader and keeps it live across the loop body. Instead one base per operand
        # family is pinned inside the body (see _opaque_idx) and every read is reached from
        # it by a compile-time element offset plus, for the tile index, one XOR:
        #   row = kk*PV_K_STEP + kg*4 + lane16//4, and both layouts are linear in kk*32 with
        #   a row mask (row&7 or row&15) that kk cannot reach, so kk and the row half stay
        #   pure offsets; the tile index lands in bits 4-5 of the swizzled column while the
        #   lane column occupies bits 2-3, so column + tile == column XOR tile.
        def _g3_row0():
            return kg * fx.Index(4) + (lane16 // fx.Index(4))

        def _g3_kbase(tile0):
            """Pinned (kk=0, row-half=0, tile=tile0) K/V transpose-read address."""
            _r = _g3_row0()
            return _opaque_idx(
                fx.Index(G3K_BASE)
                + _pblk(_r) * fx.Index(PBLK)
                + (
                    (tile0 * fx.Index(D_TILE))
                    ^ ((lane % fx.Index(4)) * fx.Index(4))
                    ^ ((_r & fx.Index(7)) << fx.Index(4))
                )
            )

        def _g3_tr(base, tile, kk, row_stride, off=0):
            """Transpose-read a [kv][col] LDS tile -> operand [m/n=col=tile*16+lane16][k=kv].

            base is a pinned family base, tile the compile-time index within the family.
            """
            _b = base ^ fx.Index(tile * D_TILE) if const_expr(tile) else base
            _o = off + kk * PV_K_STEP * row_stride
            _v0 = ds_read_tr_v4f16(_b, _o)
            _v1 = ds_read_tr_v4f16(_b, _o + N_TILE * row_stride)
            return Vec(_v0).shuffle(Vec(_v1), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # dS staging layout [kv][q] with a q ^= 4*(kv&15) swizzle. dS arrives in the MFMA
        # C-layout, so one quarter-wave writes 16 DIFFERENT kv rows at ONE q run: under the
        # Q/dO tiles' (kv&7)<<4 mask (period 4 in banks, and a row-pack stride that is 0 mod
        # 32 banks) all 16 rows land on the same banks -> 2-way conflict on a kernel whose
        # baseline conflict count is 0. A 4*(kv&15) mask spreads them over 16 even banks,
        # each 8 B write covering (b, b+1) => all 32, and keeps the tr16 read's 4-row x
        # 4-column group conflict-free (measured 6.5% -> 2.2%, -0.194 ms).
        def _g3s_wbase():
            """Pinned (nt=0, q-run=0, slot=0) dS write address for this lane's kv row.

            The C-layout write is the same family as the reads above: the kv row's swizzle
            mask is 4*lane16, the q run occupies bits 4-5 of the column and the lane's kg
            bits 2-3, so the run index is one XOR off the base and (nt, slot) are element
            offsets folded into the ds_write offset field.
            """
            _r = wave_id * fx.Index(ROWS_PER_WAVE_KV) + lane16
            return _opaque_idx(
                fx.Index(G3S_BASE)
                + _r * fx.Index(BLOCK_Q)
                + ((kg * fx.Index(4)) ^ (lane16 * fx.Index(4)))
            )

        def _if_wave(cond, vals, then_fn, else_fn):
            """scf.if threading ``vals`` through both arms, for a wave-uniform ``cond``.

            Built directly rather than with a traced ``if``: the tracer only carries plain
            named variables across a dynamic branch, and these are lists of accumulators.
            """
            from flydsl._mlir.dialects import scf

            _v = [_raw(v) for v in vals]
            op = scf.IfOp(_raw(cond), [x.type for x in _v], has_else=True)
            with ir.InsertionPoint(op.regions[0].blocks[0]):
                scf.YieldOp([_raw(x) for x in then_fn()])
            if not op.regions[1].blocks:
                op.regions[1].blocks.append()
            with ir.InsertionPoint(op.regions[1].blocks[0]):
                else_fn()
                scf.YieldOp(_v)
            return list(op.results)

        def _ds_write_v4f16(lds_elem_idx, const_elem_off, val):
            """LDS store reached through a pinned base + compile-time element offset."""
            ptr = buffer_ops.create_llvm_ptr(fx.Int64(lds_elem_idx * 2 + lds_off), address_space=3)
            if const_expr(const_elem_off != 0):
                ptr = buffer_ops.get_element_ptr(ptr, fx.Int64(const_elem_off), elem_type=elem_type)
            llvm.StoreOp(_raw(val), ptr)

        def _g3_sbase(tile0):
            """Pinned (kk=0, row-half=0, slot=0, tile=tile0) dS transpose-read address."""
            _r = _g3_row0()
            return _opaque_idx(
                fx.Index(G3S_BASE)
                + _r * fx.Index(BLOCK_Q)
                + (
                    (tile0 * fx.Index(D_TILE))
                    ^ ((lane % fx.Index(4)) * fx.Index(4))
                    ^ (_r * fx.Index(4))
                )
            )

        def _gemm3(q_start, head_local, slot, drain=None):
            """Run the dQ pass on its carrier waves (see G3_WAVES).

            The guard is wave-uniform, so it costs one s_cbranch and leaves the carriers'
            MFMA count per SIMD unchanged; only the LDS traffic drops. drain, when given,
            is the enclosing rendezvous' wait: it is emitted after the MFMA run but BEFORE
            the dQ partial stores, so the fetch it waits on gets GEMM3 as its shadow while
            the stores stay out of the wait (see G3_SHADOW). The non-carriers owe the same
            wait, hence the complementary guard rather than an else.
            """
            if const_expr(G3_WAVES < NUM_WAVES):
                if wave_id < fx.Index(G3_WAVES):
                    _gemm3_tiles(q_start, head_local, slot, drain)
                if const_expr(drain is not None):
                    if wave_id >= fx.Index(G3_WAVES):
                        drain()
            else:
                _gemm3_tiles(q_start, head_local, slot, drain)

        def _gemm3_tiles(q_start, head_local, slot, drain=None):
            """dQ^T[m=D][n=q] += K^T . dS^T over this band's kv rows, for ONE head.

            Both operands are transpose-reads over kv, so the kv permutation ds_read_tr16
            imposes is identical on the two sides and cancels in the contraction. K^T stays
            in LDS: hoisting it into registers measured neutral at BLOCK_KV=64 and +0.44 ms
            / 413 spill at 128. The caller owns the fence -- see G3S_SLOTS.
            """
            _g3d0, _g3q0 = _g3_wave_tiles()
            _kb, _sb = _g3_kbase(_g3d0), _g3_sbase(_g3q0)
            _soff = slot * G3S_SLOT_ELEMS
            _g3 = [[c_zero_v4f32 for _ in range_constexpr(G3_QT)] for _ in range_constexpr(G3_DT)]

            def _g3_frags(kk):
                # GEMM3's transpose reads are free, like GEMM1's and GEMM2's: a probe that
                # pairs the ksteps so the odd one's reads CSE onto the even one's (wrong dQ,
                # but 1536 -> 1024 tr at an untouched MFMA count) measures 6/11 -- the last
                # read family this body had not priced. Do not spend a round on read count.
                return (
                    [_g3_tr(_kb, i, kk, PBLK // 2) for i in range_constexpr(G3_DT)],
                    [_g3_tr(_sb, j, kk, BLOCK_Q, _soff) for j in range_constexpr(G3_QT)],
                )

            # kstep prefetch ring, depth G3D: kk+G3D's transpose-reads are issued before
            # kk's MFMAs so the ds_read_tr16 latency lands in the MFMA shadow instead of at
            # every kstep's first MFMA -- the same trade GEMM2's g2d ring makes, paid for by
            # the registers the pinned bases above freed.
            # An s_setprio(1) pair around this MFMA run -- the same trade the GEMM2 pair
            # makes -- is 6/11 and costs 63 spill dwords / 208 B of scratch, because
            # s_setprio does not split a scheduling region but does pin the ring's live
            # ranges across it.
            _ring = [_g3_frags(kk) for kk in range_constexpr(min(G3D, G3_KSTEPS))]
            for _kk in range_constexpr(G3_KSTEPS):
                _g3k, _g3s = _ring[_kk % G3D]
                if const_expr(_kk + G3D < G3_KSTEPS):
                    _ring[_kk % G3D] = _g3_frags(_kk + G3D)
                for i in range_constexpr(G3_DT):
                    for j in range_constexpr(G3_QT):
                        _g3[i][j] = mfma_acc(_g3k[i], _g3s[j], _g3[i][j])
            if const_expr(drain is not None):
                drain()
            # Store the partial straight out of the dQ^T C-layout: 4 contiguous D per
            # lane, 32 B per 4 lanes. Routing it through an LDS CShuffle to get 128 B
            # contiguous measured 0.00 ms once GEMM3 is present, and dwordx4 +0.115 ms.
            _g3qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            for i in range_constexpr(G3_DT):
                for j in range_constexpr(G3_QT):
                    _g3p = bf16_trunc_scored_v4(_g3[i][j])
                    _g3row = q_start + (_g3q0 + fx.Index(j)) * fx.Index(M_TILE) + lane16
                    _g3col = (_g3d0 + fx.Index(i)) * fx.Index(D_TILE) + kg * fx.Index(4)
                    buffer_ops.buffer_store(
                        _g3p.ir_value(),
                        wsq_rsrc,
                        ((_g3row * fx.Index(NUM_HEADS_Q) + _g3qh) * fx.Index(HEAD_DIM) + _g3col)
                        * fx.Index(2),
                        offset_is_bytes=True,
                    )

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
            _v = [
                buffer_ops.buffer_load(rsrc, _g, vec_width=LD_VEC, dtype=fx.Float32)
                for rsrc in (delta_rsrc, lse_rsrc)
            ]
            if const_expr(LD_VEC == 1):
                # An 8-wave group leaves one element per thread; vec_width=1 lowers to a
                # scalar, and the LDS commit below stores vectors.
                _v = [Vec.from_elements([fx.Float32(x)], fx.Float32).ir_value() for x in _v]
            return _v

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
                [
                    fx.Index(arr * LD_ARR_ELEMS + head_local * LD_HEAD_ELEMS + mt * M_TILE)
                    + kg * fx.Index(4)
                ],
            ).ir_value()

        def _dma_head(head_local, bases):
            """Issue (no wait) the Q/dO DMA for head_local into its LDS slot."""
            sl = head_local % LDS_SLOTS
            _qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            if const_expr(DMA_SHARED_PTR):
                coop_dma_tile(q_rsrc, sl * LDS_TOTAL * 2, bases, _qh)
                coop_dma_tile(do_rsrc, (sl * LDS_TOTAL + LDS_DO_BASE) * 2, bases, _qh)
            else:
                coop_dma_tile(q_rsrc, q_lds_ptrs[sl], bases, _qh)
                coop_dma_tile(do_rsrc, do_lds_ptrs[sl], bases, _qh)

        def _qdo_src_elem(q_start, head_local, d):
            """Element index of this thread's 16 B slice of copy batch d, DMA lane mapping.

            Mirrors _dma_bases exactly, so the LDS image -- and therefore every reader and
            the kernel's output -- is unchanged; only the transport differs.
            """
            _blk = tid // fx.Index(16) + fx.Index(d * ROWS_PER_DMA_BATCH)
            _lib = tid % fx.Index(16)
            if const_expr(PACK_2ROW):
                _row = (
                    fx.Index(8) * (_blk >> fx.Index(2))
                    + (_blk & fx.Index(3))
                    + (_lib // fx.Index(8)) * fx.Index(4)
                )
            else:
                _row = _blk
            _col = (_lib * fx.Index(8)) ^ ((_row & fx.Index(7)) << fx.Index(4))
            _qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            return global_idx_q(q_start + _row, _col, _qh)

        def _qdo_issue(q_start, head_local):
            """Issue (no wait) head_local's Q/dO tile into VGPRs."""
            return [
                buffer_ops.buffer_load(
                    rsrc, _qdo_src_elem(q_start, head_local, d), vec_width=8, dtype=elem_dtype
                )
                for d in range_constexpr(NUM_DMA_Q)
                for rsrc in (q_rsrc, do_rsrc)
            ]

        def _qdo_commit(vals, slot):
            """Publish a prefetched Q/dO tile pair into the LDS slot."""
            for d in range_constexpr(NUM_DMA_Q):
                _i = slot + fx.Index(d * (DMA_BATCH_BYTES // 2)) + tid * fx.Index(8)
                Vec(vals[2 * d]).store(lds, [_i])
                Vec(vals[2 * d + 1]).store(lds, [fx.Index(LDS_DO_BASE) + _i])

        def _vgpr_load_head(head_local, q_start):
            """VGPR-staged fallback for _dma_head (ENABLE_DMA off)."""
            sl = fx.Index((head_local % LDS_SLOTS) * LDS_TOTAL)
            _qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            _coop_load(q_ptr, sl, q_start, _qh)
            _coop_load(do_ptr, sl + fx.Index(LDS_DO_BASE), q_start, _qh)

        def _q_prologue(q_start, bases):
            """Fill slot 0 with head 0's tile and stage the group's (-delta, lse).

            Slot 0 was last read by head GQA-2 of the previous q-block, which the head
            GQA-1 barrier already fenced, so the DMA can be issued before this barrier.
            """
            _ldv = _stage_ld_issue(q_start)
            _dma_head(0, bases)
            gpu.barrier()  # WAR: every head of the previous q-block read the lse staging
            _stage_ld_commit(_ldv)
            rocdl.s_waitcnt(0)

        def _head_step_lds(
            q_start,
            apply_mask,
            head_local,
            dv_cur,
            dk_cur,
            bases=None,
            stage_heads=None,
            mid_pf=None,
            qdo=None,
            ldv=None,
        ):
            sb_bulk = not apply_mask  # exps only exist on these paths
            # The next head's Q/dO fetch: the earlier it is issued the more of this step
            # covers it, and the longer its 16 B per tensor stay live over the body's
            # register peak (GEMM2's accumulators). QPF_AT picks that trade.
            # [0] = Q/dO for the next head-step, [1] = (-delta, lse) for the next q-block.
            _qdo_next = [None, None]

            def _qdo_pf(at=0):
                if const_expr(not Q_PREF or at != QPF_AT):
                    return
                if const_expr(head_local + 1 < GQA_GROUP_SIZE):
                    _qdo_next[0] = _qdo_issue(q_start, head_local + 1)
                elif const_expr(PF_QB):
                    # Same issue point, next q-block's head 0. Rows past the sequence end
                    # are clamped by the slice's num_records (they read 0 and the block
                    # they belong to never runs), so the tail iteration needs no guard.
                    _nq = q_start + fx.Index(_step)
                    _qdo_next[0] = _qdo_issue(_nq, 0)
                    _qdo_next[1] = _stage_ld_issue(_nq)

            q_start_i32 = fx.Int32(q_start)
            kg_off_i32 = fx.Int32(kg) * fx.Int32(4)
            q_lds = fx.Index((head_local % LDS_SLOTS) * LDS_TOTAL)
            do_lds = q_lds + fx.Index(LDS_DO_BASE)
            if const_expr(q_dbuf):
                # This head's tile landed during the previous step; the barrier publishes it
                # and simultaneously fences the slot that the prefetch below overwrites.
                gpu.barrier()
                if const_expr(head_local + 1 < GQA_GROUP_SIZE):
                    _dma_head(head_local + 1, bases)
            elif const_expr(Q_PREF):
                # qdo already holds this head's tile, fetched one head-step ago. The
                # ds_write is the only point that waits on it, and the next head's fetch
                # is issued right after so it gets this whole step as its shadow.
                _ldv = None
                if const_expr(head_local == 0):
                    _ldv = ldv if const_expr(PF_QB) else _stage_ld_issue(q_start)
                gpu.barrier()  # WAR: the previous head's GEMM2 still read this slot
                _qdo_commit(qdo, q_lds)
                qdo = None
                _qdo_pf(0)
                if const_expr(head_local == 0):
                    _stage_ld_commit(_ldv)
                rocdl.s_waitcnt(WAIT_LGKM)  # retire the ds_writes; the loads stay in flight
                gpu.barrier()  # Q/dO + ld_lds commit visible before GEMM1 reads
            elif const_expr(PF_RING):
                pass  # the rendezvous sits inside the GEMM2 loop below
            elif const_expr(stage_heads is not None):
                # Group leader: stage this group's whole set of Q/dO tiles in one shot.
                # The rendezvous (WAR barrier + drain + publish barrier) is then paid once
                # per DMA_GRP heads instead of per head, and the group's tiles are in flight
                # together so their HBM latencies overlap instead of serialising. Followers
                # read an already-published slot and need no fence at all.
                # (-delta, lse) for the whole GQA group rides head 0's barrier pair;
                # heads 1..7 re-read straight from LDS.
                _ldv = None
                if const_expr(head_local == 0):
                    _ldv = _stage_ld_issue(q_start)
                _shadow = G3_SHADOW and head_local > 0
                if const_expr(_shadow):
                    # The barrier below now publishes the dS tile GEMM3 is about to read,
                    # so the write side has to retire first; the same wait also fences the
                    # slot's last GEMM2 reads against the DMA that overwrites it.
                    rocdl.s_waitcnt(WAIT_LGKM)
                gpu.barrier()  # WAR: the slots this group overwrites were read last group
                if const_expr(ENABLE_DMA):
                    for _sh in stage_heads:
                        _dma_head(_sh, bases)
                    if const_expr(head_local == 0):
                        _stage_ld_commit(_ldv)

                    def _rdv_drain():
                        rocdl.s_waitcnt(0)

                    if const_expr(_shadow):
                        _gemm3(q_start, head_local - 1, (head_local - 1) % G3S_SLOTS, _rdv_drain)
                    else:
                        _rdv_drain()
                else:
                    for _sh in stage_heads:
                        _vgpr_load_head(_sh, q_start)
                    if const_expr(head_local == 0):
                        _stage_ld_commit(_ldv)
                gpu.barrier()  # DMA + ld_lds commit visible before GEMM1 reads

            if const_expr(G3_DEFER and head_local > 0 and not G3_SHADOW):
                # The PREVIOUS head's dQ, emitted at the TOP of this head-step. Its dS tile
                # was published by the staging pair above (ds_write -> WAR barrier -> drain
                # -> publish barrier), so GEMM3 needs no fence of its own, and the same pair
                # one step later fences the read against the head that reuses the slot.
                # Two reasons for the position. (1) The partial stores: gfx950 shares one
                # vmcnt between loads and stores and has no subset wait, so the NEXT
                # head-step's DMA drain also retires these stores. From here they get a
                # whole GEMM1+GEMM2 of slack instead of the ~10 instructions they had at the
                # end of the step. (2) Registers: GEMM3's ring and accumulators now die
                # before GEMM1a's fragments are live, so the two peaks no longer add --
                # 256 vgpr / 36 spill -> 227 vgpr / 0 spill, which is what pays for G3D.
                _gemm3(q_start, head_local - 1, (head_local - 1) % G3S_SLOTS)

            _qdo_pf(1)

            # GEMM1a/exp2/GEMM1b/dS/pack per q-HALF (one pks = two mt packing into one
            # GEMM2 K=32 step): processing 2 of the MT q-tiles at a time halves the live
            # S/dP/P/dS transient that pinned dkdv at spill, so the kernel fits spill-free.
            # lse/-delta are pulled from LDS at their use points (only the 2 v4f32 this
            # half consumes are ever live). Pure re-ordering -> bit-identical, det-neutral.
            p_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            ds_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            _g3wb = _g3s_wbase() if const_expr(FUSE_DQ) else None

            def _flat_accs():
                return [
                    dv_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)
                ] + [dk_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]

            def _set_accs(vals):
                for dt in range_constexpr(DT):
                    for nt in range_constexpr(NT):
                        dv_cur[dt][nt] = vals[dt * NT + nt]
                        dk_cur[dt][nt] = vals[DT * NT + dt * NT + nt]

            def _gemm2(pk_list, do_ring, q_ring, carry_rdv):
                """GEMM2a dV^T += dO_tr @ P ; GEMM2b dK^T += Q_tr @ dS over the DT d-tiles.

                pk_list selects which q-halves this pass consumes; a depth-g2d dt prefetch
                ring issues dt+g2d's transpose-reads before dt's MFMAs so the ds_read_tr16
                LDS latency hides in the MFMA shadow. g2d=1 -> depth-1 baseline.
                """
                _nk = len(pk_list)
                # PF_RING rendezvous, parked on the LAST GEMM2 step rather than at the head
                # boundary. By here the head has issued every read of its own slot (the
                # transpose-read ring runs g2d ahead and stops at DT-1-g2d), so the drain
                # retires them and the slots refilled below -- last read a group ago -- are
                # free. Earlier dt is not legal (reads still to come) and was also slower;
                # dropping the explicit drain costs 0.6% and dropping the s_setprio pair
                # around it costs 2.0%, so both stay.
                # Hoisting the last dt's transpose reads ahead of the rendezvous to move it
                # off DT-1 (giving its DMA a longer MFMA run before the next head's first
                # LDS read) is a measured loss at every depth -- 1 -> +1.7%, 2 -> +0.8% --
                # because the hoisted reads' live range crosses it on a full register file.
                _mid_dt = (DT - 1) if const_expr(carry_rdv) else -1
                _n_out = 2  # sched-hint scale: 1 op-stream per output (dV + dK)
                # The priority pair is what de-phases the two waves of a SIMD: the one in
                # GEMM2 wins issue until it drops out, so its sibling's exp chain drifts
                # into this MFMA run instead of contending with it. Dropping it (both
                # GEMM2 calls under G2_HALF) is 9/9 losses at -2.8%, even though the ISA
                # then interleaves MFMA and TRANS far more (28 of 32 exps within +-3
                # instructions of an MFMA against 31) -- more interleaving on paper, less
                # throughput. KB pitfalls/12 judges s_setprio negative for sparse-MLA
                # attention; on this body it is strongly positive.
                rocdl.s_setprio(1)
                for dt in range_constexpr(DT):
                    if const_expr(dt == _mid_dt):
                        rocdl.s_setprio(0)
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        for _sh in mid_pf:
                            _dma_head(_sh, bases)
                        rocdl.s_setprio(1)
                    if const_expr(dt == 1 and pk_list[-1] == PV_K_STEPS - 1):
                        _qdo_pf(3)
                    _slot = dt % g2d
                    do_tr = do_ring[_slot]
                    q_tr = q_ring[_slot]
                    _rd_next = dt + g2d < DT
                    if const_expr(_rd_next):
                        do_tr_n = [
                            _read_tr(do_lds, dt + g2d, pk_list[i], _do_trb)
                            for i in range_constexpr(_nk)
                        ]
                    for i in range_constexpr(_nk):
                        for nt in range_constexpr(NT):
                            dv_cur[dt][nt] = mfma_acc(do_tr[i], p_pack[pk_list[i]][nt], dv_cur[dt][nt])
                    if const_expr(NT >= 3):
                        # NT>=3 pins the packs' liveness hard enough that the RA sinks the
                        # pack next to the MFMA that reads it as SrcB. Pinning the dV group
                        # live past its MFMAs blocks that sinking, which is worth 8 spill
                        # dwords even now that the scored pack makes the sink itself legal.
                        # Naming fewer than all four elements of each tuple saves v_accvgpr
                        # reads but measures neutral (-0.2% at one element), so all four stay.
                        # The pin is dV-only: adding the same on dK costs 4.6%.
                        _keepalive_v4([dv_cur[dt][nt] for nt in range_constexpr(NT)])
                    if const_expr(_rd_next):
                        q_tr_n = [
                            _read_tr(q_lds, dt + g2d, pk_list[i], _q_trb)
                            for i in range_constexpr(_nk)
                        ]
                    for i in range_constexpr(_nk):
                        for nt in range_constexpr(NT):
                            dk_cur[dt][nt] = mfma_acc(q_tr[i], ds_pack[pk_list[i]][nt], dk_cur[dt][nt])
                    if const_expr(_rd_next):
                        for _ in range_constexpr(_n_out * _nk * NT):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(1)
                        do_ring[_slot] = do_tr_n
                        q_ring[_slot] = q_tr_n
                rocdl.s_setprio(0)

            _q_trb = _tr_base(q_lds) if const_expr(HEAD_DIM == 128) else None
            _do_trb = _tr_base(do_lds) if const_expr(HEAD_DIM == 128) else None
            _q_apin = _a_pin(q_lds) if const_expr(A_PIN) else None
            _do_apin = _a_pin(do_lds) if const_expr(A_PIN) else None

            def _gemm_dp(half):
                return _gemm_qk(
                    do_lds,
                    v_b_packs,
                    inits={mt: _ld_read(head_local, mt, 0) for mt in half},
                    mts=half,
                    pin=_do_apin,
                )

            def _half_gemm1(half):
                """The MFMA-only front of a q-half: S = Q@K^T and, fused, dP = dO@V^T.

                dP does not depend on P, so at D128 it is issued FIRST: its MFMA run then
                covers the quarter-rate exp2 chain that GEMM1a's accumulators feed, instead
                of trailing it. D128 is occ=1 (no sibling wave to hide the exps) and PMC puts
                it at MFMA 51% / VALU 29%, so that overlap is worth having. The split D64
                body runs at occ=2 and keeps the legacy order -> byte-identical. The FUSED
                body is occ=2 as well but all 8 waves are barrier-locked into the same
                head-step, so the siblings reach their exp chain together and cover nothing
                for each other -- it takes the D128 order (arithmetic unchanged either way).
                Pipelining the dS/pack block that follows against MFMA is a measured loss in
                both directions: splitting dP per kv 16-tile so each tile's VALU trails the
                next tile's MFMAs costs 4.0% (two accumulator chains cannot cover an MFMA's
                result latency), and deferring a whole half's block into the next half's
                GEMM1a costs 0.9% (its P/dP stay live across those 24 MFMAs).
                """
                if const_expr(fold_lse and not apply_mask):
                    # FOLD unmasked: prescaled -log2e*lse is GEMM1a's C-init, so the
                    # accumulator already IS the base-2 softmax exponent.
                    _st = _gemm_qk(
                        q_lds,
                        k_b_packs,
                        inits={mt: _ld_read(head_local, mt, 1) for mt in half},
                        mts=half,
                        pin=_q_apin,
                    )
                else:
                    _st = _gemm_qk(
                        q_lds,
                        k_b_packs,
                        mts=half,
                        pin=_q_apin,
                    )
                if const_expr(sb_bulk and not exp_intrin):
                    rocdl.sched_barrier(SCHED_TRANS)
                _dpt = _gemm_dp(half) if const_expr(HEAD_DIM == 128 or FUSE_DQ) else None
                # Extending the GEMM2 s_setprio(1) pair over this run too (so a SIMD's two
                # waves also de-phase across GEMM1) is 7/11 then 6/11 = noise, even though it
                # halves the hazard nops (198 -> 102): the pair only pays where one wave has
                # an MFMA run its sibling does not, which is GEMM2 and the carriers' GEMM3.
                return _st, _dpt

            def _half_soft(pks, half, s_tiles, dp_tiles):
                """softmax -> dS -> bf16 pack (-> dS publish) for one q-half.

                Returns the GEMM2 transpose-read ring this half primed, or None.
                """
                ma, mb = half
                P = [[None] * NT for _ in range_constexpr(MT)]
                if const_expr(fold_lse and not apply_mask):
                    for mt in half:
                        for nt in range_constexpr(NT):
                            s_v = Vec(s_tiles[mt][nt])
                            if const_expr(exp_intrin):
                                P[mt][nt] = [
                                    _vexp_intrin(fx.Float32(s_v[t])) for t in range_constexpr(4)
                                ]
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
                                    if const_expr(window_left >= 0):
                                        # keep kv >= q+off-W (W+1 keys), matching the fwd
                                        # SWA edge; strict '<' -> the boundary key q+off-W stays.
                                        _lo = ArithValue(
                                            kv_row_i32_of(nt)
                                            < q_slot + causal_off_i32 - fx.Int32(window_left)
                                        )
                                        _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                                    else:
                                        _mm = _up
                                    s_r = _mm.select(c_neg_inf, s_r)
                                p_vals.append(_p_of(s_r, fx.Float32(Vec(lse_v)[t]), apply_mask))
                            P[mt][nt] = p_vals

                if const_expr(HEAD_DIM != 128 and not FUSE_DQ):
                    dp_tiles = _gemm_dp(half)

                # Hoist the first g2d dt's GEMM2 transpose-reads into the LAST half's
                # dS/pack shadow: the ds_read_tr16 LDS latency overlaps that VALU block
                # instead of exposing at GEMM2's first MFMA. dV reads dO_tr, dK reads Q_tr.
                _pk_seg = [pks] if const_expr(G2_HALF) else list(range_constexpr(PV_K_STEPS))
                _rings = None
                if const_expr(G2_HALF or pks == PV_K_STEPS - 1):
                    _rings = (
                        [
                            [_read_tr(do_lds, _d, _p, _do_trb) for _p in _pk_seg]
                            for _d in range_constexpr(g2d)
                        ],
                        [
                            [_read_tr(q_lds, _d, _p, _q_trb) for _p in _pk_seg]
                            for _d in range_constexpr(g2d)
                        ],
                    )

                for nt in range_constexpr(NT):
                    _ds = [
                        [_fmul(P[mt][nt][t], Vec(dp_tiles[mt][nt])[t]) for t in range_constexpr(4)]
                        for mt in half
                    ]
                    p_pack[pks][nt] = bf16_trunc_pack_v8(P[ma][nt] + P[mb][nt])
                    ds_pack[pks][nt] = bf16_trunc_pack_v8(_ds[0] + _ds[1])
                    if const_expr(FUSE_DQ):
                        # Publish dS as [kv][q] for GEMM3's transpose-read. The v8 pack is
                        # q = {ma,mb}*16 + kg*4 + t of ONE kv row, so its two halves are two
                        # 4-wide q runs -> two ds_write_b64 into the swizzled dS tile.
                        _g3si = Vec(ds_pack[pks][nt]).bitcast(fx.Int32)
                        _g3wo = nt * N_TILE * BLOCK_Q + (head_local % G3S_SLOTS) * G3S_SLOT_ELEMS
                        for _hh in range_constexpr(2):
                            _ds_write_v4f16(
                                _g3wb ^ fx.Index((2 * pks + _hh) * M_TILE),
                                _g3wo,
                                Vec.from_elements(
                                    [fx.Int32(_g3si[2 * _hh]), fx.Int32(_g3si[2 * _hh + 1])], fx.Int32
                                ).bitcast(elem_dtype),
                            )

                return _rings

            # GEMM2 per q-half, consuming a half's packs as soon as they exist. The
            # per-accumulator half order stays pks-ascending -> bit-identical, and the read
            # and MFMA counts are untouched. Two things move: the packs die a half earlier,
            # which takes 16 dwords off the next half's GEMM1 peak (256 vgpr / 28 B scratch
            # / 6 spilled -> 249 / 0 / 0), and a half's 16 GEMM2 MFMAs land next to the NEXT
            # half's GEMM1a and quarter-rate exp chain, which the ISA shows running in a bare
            # VALU window (29 of 32 exps have no MFMA within +-3 instructions). Paired wins
            # in three separate sessions: 8/9, 7/9 and 11/15, at +0.5% / +0.2% / +0.5%.
            # Flushing that GEMM2 one step later still -- between the next half's GEMM1 and
            # its exps, so the exps issue into an already-full MFMA pipe -- puts the next
            # half's S and dP tiles across those 32 MFMAs and is 9/9 losses at -2.0% (256
            # vgpr / 20 B scratch / 4 spilled). Every other way of forcing that overlap loses
            # too, so ADJACENCY plus the register relief is what pays, not interleaving:
            # sched_group_barrier MFMA:TRANS pipelines over the GEMM1 pair and its exps cost
            # 0.2% at both 1:2 and 2:2 (spill 6 -> 11 dwords), and see the s_setprio note in
            # _gemm2.
            def _pks_chain(pf=True):
                _rings = None
                for pks in range_constexpr(PV_K_STEPS):
                    half = [2 * pks, 2 * pks + 1]
                    _st, _dpt = _half_gemm1(half)
                    _rings = _half_soft(pks, half, _st, _dpt)
                    if const_expr(G2_HALF):
                        _last = const_expr(pks == PV_K_STEPS - 1)
                        if const_expr(_last and pf):
                            _qdo_pf(2)
                        _gemm2(
                            [pks],
                            _rings[0],
                            _rings[1],
                            const_expr(PF_RING and mid_pf is not None and _last),
                        )

                if const_expr(not G2_HALF):
                    if const_expr(pf):
                        _qdo_pf(2)
                    _gemm2(
                        list(range_constexpr(PV_K_STEPS)),
                        _rings[0],
                        _rings[1],
                        const_expr(PF_RING and mid_pf is not None),
                    )

            if const_expr(MASK_SKIP and apply_mask):
                # Diagonal q-block: a wave whose whole ROWS_PER_WAVE_KV kv rows sit above
                # this block's causal edge has P = dS = 0 for every head, so it skips the
                # entire GEMM chain behind a wave-uniform branch and only publishes zeros
                # into its dS rows (GEMM3 contracts the WHOLE band, so they must be
                # defined -- and on the first q-block of a band the slot is still
                # uninitialised LDS). The live waves' arithmetic is untouched and the
                # skipped contributions are exact zeros, so every output stays bitwise
                # identical. What pays is the machine: with q_split=4 the four diagonal
                # blocks skip 6/4/2/0 of the 8 waves, and waves are round-robin over the
                # SIMDs, so the surviving wave usually gets its SIMD to itself. Worth
                # +0.22% / +0.24% / +0.33% in three separate interleaved-A/B sessions.
                # That is most of what is there to take: forcing EVERY wave to skip these
                # blocks (wrong results, diagnostic only) measures +0.37%, i.e. the whole
                # diagonal family costs 0.4% of the wall even though it is 6% of the
                # q-block visits -- do not budget more than that for masked-block work.
                # The next tile's fetch is every wave's own lanes, so it stays outside.
                _qdo_pf(2)

                def _live():
                    _pks_chain(pf=False)
                    return _flat_accs()

                def _dead():
                    _z = Vec.from_elements(
                        [fx.Int32(0), fx.Int32(0)], fx.Int32
                    ).bitcast(elem_dtype)
                    for nt in range_constexpr(NT):
                        _zo = nt * N_TILE * BLOCK_Q + (head_local % G3S_SLOTS) * G3S_SLOT_ELEMS
                        for pks in range_constexpr(PV_K_STEPS):
                            for _hh in range_constexpr(2):
                                _ds_write_v4f16(_g3wb ^ fx.Index((2 * pks + _hh) * M_TILE), _zo, _z)

                _q_last = q_start + fx.Index(BLOCK_Q - 1) + causal_offset
                _cond = ArithValue(kv_row_wave <= _q_last)
                _set_accs(_if_wave(_cond, _flat_accs(), _live, _dead))
            else:
                _pks_chain()
            if const_expr(FUSE_DQ and not G3_DEFER):
                # Undeferred: dS is read in the head-step that wrote it, so this head-step
                # pays its own RAW fence. gpu.barrier() alone is not a fence -- retire the
                # ds_writes first. Only lgkmcnt is needed: a full drain would also wait on
                # the previous head-step's dQ partial stores, which nothing here reads
                # (gfx950 shares one vmcnt between loads and stores). Emitting GEMM3 here
                # rather than before GEMM2 keeps the transpose-reads' live ranges off
                # GEMM2, which measured -2.4% on a full register file.
                rocdl.s_waitcnt(WAIT_LGKM)
                gpu.barrier()  # RAW: every wave's dS rows feed every wave's GEMM3
                _gemm3(q_start, head_local, 0)
            if const_expr(q_dbuf and head_local + 1 < GQA_GROUP_SIZE):
                rocdl.s_waitcnt(0)  # prefetch landed; the next step's barrier publishes it
            return dv_cur, dk_cur, (_qdo_next if const_expr(Q_PREF) else [qdo, None])

        def _q_body(q_start, inner, apply_mask):
            # inner (loop-carried) = [dv accs][dk accs] (+ [Q/dO][-delta, lse] under PF_QB).
            _dk_base = DT * NT
            dv_cur = [[inner[dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)]
            dk_cur = [
                [inner[_dk_base + dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)
            ]
            # Head-invariant DMA offsets: computed once per q-block, reused by all heads.
            _bases = _dma_bases(q_start) if const_expr(ENABLE_DMA and not Q_PREF) else None
            _ldv = None
            if const_expr(PF_QB):
                _pfb = 2 * DT * NT
                _qdo = list(inner[_pfb : _pfb + 2 * NUM_DMA_Q])
                _ldv = list(inner[_pfb + 2 * NUM_DMA_Q :])
            else:
                _qdo = _qdo_issue(q_start, 0) if const_expr(Q_PREF) else None
            if const_expr(q_dbuf):
                _q_prologue(q_start, _bases)
            if const_expr(PF_RING):
                # Prime the whole ring up front; every later refill rides a rendezvous
                # parked inside a GEMM2 run (see _head_step_lds).
                _ldv = _stage_ld_issue(q_start)
                gpu.barrier()  # WAR: the previous q-block read these slots and the lse staging
                for _sh in range_constexpr(LDS_SLOTS):
                    _dma_head(_sh, _bases)
                _stage_ld_commit(_ldv)
                rocdl.s_waitcnt(0)
                gpu.barrier()
            for head_local in range_constexpr(GQA_GROUP_SIZE):
                # Only the leader of each DMA_GRP-sized head group stages tiles; the rest
                # consume slots this group already published.
                _sh = None
                if const_expr(head_local % DMA_GRP == 0 and not PF_RING):
                    _sh = list(range_constexpr(head_local, head_local + DMA_GRP))
                _mid = None
                if const_expr(PF_RING and head_local % DMA_GRP == DMA_GRP - 1):
                    # The last head of a group carries the rendezvous: it publishes the
                    # NEXT group (already in flight) and refills the slots the group before
                    # this one vacated, which is one full group of slack on each edge.
                    _first = head_local + 1 + (LDS_SLOTS - DMA_GRP)
                    if const_expr(head_local + 1 < GQA_GROUP_SIZE):
                        # The tail groups have nothing left to refill, but their barrier is
                        # still what publishes the previous group's tiles (empty mid_pf).
                        _mid = list(range_constexpr(min(_first, GQA_GROUP_SIZE), min(_first + DMA_GRP, GQA_GROUP_SIZE)))
                dv_cur, dk_cur, _pf = _head_step_lds(
                    q_start,
                    apply_mask,
                    head_local,
                    dv_cur,
                    dk_cur,
                    bases=_bases,
                    stage_heads=_sh,
                    mid_pf=_mid,
                    qdo=_qdo,
                    ldv=_ldv,
                )
                _qdo = _pf[0]
                if const_expr(_pf[1] is not None):
                    _ldv = _pf[1]
            if const_expr(G3_DEFER):
                # The last head has no successor head-step to ride, so it pays the only
                # explicit dS fence left in the kernel: one per q-block instead of one per
                # head-step. gpu.barrier() alone is not a fence -- retire the ds_writes.
                rocdl.s_waitcnt(WAIT_LGKM)
                gpu.barrier()  # RAW: every wave's dS rows feed every wave's GEMM3
                _gemm3(q_start, GQA_GROUP_SIZE - 1, (GQA_GROUP_SIZE - 1) % G3S_SLOTS)
            out = [dv_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]
            out += [dk_cur[dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)]
            if const_expr(PF_QB):
                out += list(_qdo) + list(_ldv)
            return out

        _carry = dv_accs + dk_accs
        if const_expr(PF_QB):
            # Prologue fetch for the first q-block; every later one is issued a head-step
            # early inside the body. The masked loop hands its pending fetch to the
            # unmasked loop: _unmask_start is exactly the last masked q_start + _step (and
            # _q_loop_start itself when the masked loop is empty), so the carry stays valid.
            _carry = _carry + _qdo_issue(_q_loop_start, 0) + _stage_ld_issue(_q_loop_start)
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
        _dk_base = DT * NT
        dv_accs = [loop_results[i] for i in range_constexpr(DT * NT)]
        dk_accs = [loop_results[_dk_base + i] for i in range_constexpr(DT * NT)]

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
        WSQ: fx.Tensor,
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
            WSQ,
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
    if sched_strategy is not None:
        _hints["llvm_options"]["amdgpu-sched-strategy"] = sched_strategy

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
    block_m=192,  # q rows per work-group (owned); must be a multiple of 64
    # g2d: GEMM2 transpose-read read-ahead in d-tiles (even, >= 2). Depth hides the
    # ds_read_tr16 latency behind more MFMA, at one live transpose-read per extra tile.
    g2d=2,
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

    BLOCK_M = block_m  # q rows per work-group (owned)
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
    G2A = g2d
    assert G2A >= 2 and G2A % 2 == 0

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

    # D128 stages the next K/V tile through VGPRs instead of buffer_load_lds (see the
    # loop body); D64 keeps the DMA path.
    KV_REG_PF = ENABLE_DMA and HEAD_DIM == 128

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

        # D64 packs 2 real rows into one 128-wide LDS block (low r&4=0 -> [0,64),
        # high -> [64,128)); D128 is already 128-wide, so one row == one block.
        PACK_2ROW = HEAD_DIM == 64  # host bool; gate tracer branches with const_expr()
        PBLK = 128 if PACK_2ROW else HEAD_DIM

        def _pblk(row_idx):
            if const_expr(PACK_2ROW):
                return ((row_idx >> fx.Index(3)) << fx.Index(2)) | (row_idx & fx.Index(3))
            return row_idx

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
            # D64: (BLOCK_KV/2) blocks, 2 rows each. D128: BLOCK_KV blocks, 1 row each.
            KV_TILE_BYTES = BLOCK_KV * HEAD_DIM * 2
            NUM_DMA_KV = KV_TILE_BYTES // DMA_BATCH_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (128 * 2)  # 128-wide blocks per batch
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def _kv_src_elem(tile_start, d):
                """Element index of this thread's 16 B slice of copy batch d.

                Address math is recomputed per tile on purpose: keeping the offsets live
                across the k_tr peak pushes VGPRs past the occ-2 boundary.
                """
                block = tid // fx.Index(16) + fx.Index(d * ROWS_PER_DMA_BATCH)
                lane_in_block = tid % fx.Index(16)
                position = lane_in_block * fx.Index(8)  # swiz col within 128-block
                if const_expr(PACK_2ROW):
                    # D64: block holds 2 rows; 8 lanes/half, real col in [0,64).
                    half = lane_in_block // fx.Index(8)
                    row_in_tile = (
                        fx.Index(8) * (block >> fx.Index(2)) + (block & fx.Index(3)) + half * fx.Index(4)
                    )
                else:
                    # D128: block == row; 16 lanes span the full 128-wide row.
                    row_in_tile = block
                xor_mask = (row_in_tile & fx.Index(7)) << fx.Index(4)
                unsw_col_f16 = position ^ xor_mask  # real col (1x HBM)
                return (
                    (tile_start + row_in_tile) * fx.Index(RD_STRIDE_KV)
                    + kv_head_idx * fx.Index(HEAD_DIM)
                    + unsw_col_f16
                )

            def coop_dma_tile(src_rsrc, lds_byte_base, tile_start):
                """DMA a tile into the swizzled LDS layout."""
                for d in range_constexpr(NUM_DMA_KV):
                    lds_addr = (
                        lds_byte_base
                        + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_addr))
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
                    # Byte-offset arithmetic kept inline here (not via _kv_src_elem) so the
                    # traced address IR is bit-identical to the pre-g2d dq kernel -- D64 zero
                    # regression. The D128 register-prefetch path (coop_load_tile_regs) uses
                    # the _kv_src_elem element-index form instead; the two are equal.
                    block = tid // fx.Index(16) + fx.Index(d * ROWS_PER_DMA_BATCH)
                    lane_in_block = tid % fx.Index(16)
                    position = lane_in_block * fx.Index(8)  # swiz col within 128-block
                    if const_expr(PACK_2ROW):
                        # D64: block holds 2 rows; 8 lanes/half, real col in [0,64).
                        half = lane_in_block // fx.Index(8)
                        row_in_tile = (
                            fx.Index(8) * (block >> fx.Index(2)) + (block & fx.Index(3)) + half * fx.Index(4)
                        )
                    else:
                        # D128: block == row; 16 lanes span the full 128-wide row.
                        row_in_tile = block
                    xor_mask = (row_in_tile & fx.Index(7)) << fx.Index(4)
                    unsw_col_f16 = position ^ xor_mask  # real col (1x HBM)
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

            def coop_load_tile_regs(tile_start):
                """Issue (no wait) the K and V global loads for one tile into VGPRs."""
                return [
                    buffer_ops.buffer_load(
                        _rsrc, _kv_src_elem(tile_start, d), vec_width=DMA_BYTES // 2, dtype=elem_dtype
                    )
                    for _rsrc in (k_rsrc, v_rsrc)
                    for d in range_constexpr(NUM_DMA_KV)
                ]

            def coop_store_tile_lds(regs):
                """Write a register-staged tile into the swizzled LDS layout.

                The destination is exactly where coop_dma_tile would have put it: the
                hardware spreads a DMA batch over the wave 16 B per lane, so thread tid
                owns element tid*(DMA_BYTES/2) of the batch.
                """
                for i, (_base, d) in enumerate(
                    [(b, d) for b in (0, LDS_V_BASE) for d in range_constexpr(NUM_DMA_KV)]
                ):
                    Vec(regs[i]).store(
                        lds,
                        [fx.Index(_base + d * (DMA_BATCH_BYTES // 2)) + tid * fx.Index(DMA_BYTES // 2)],
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
            # Emission order is ks-innermost on purpose: the ks-outer form that wins in
            # dkdv (see g1_ks_outer) costs 1.5% here, because dq runs two waves per SIMD
            # and the sibling wave already covers an MFMA's result latency, so the wider
            # live accumulator set buys nothing.
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

        # The K/V global loads are issued at the top of the body, so the whole tile's
        # compute covers their HBM latency and the only LDS traffic is the write at the
        # very end. That lets the WAR barrier leave the middle of GEMM2 -- measured: the
        # barrier's position inside GEMM2, not the barrier count, is what this loop was
        # paying for.

        def _issue_dma(kv_start):
            """Issue (no wait) the K/V DMA for the tile after kv_start.

            One tile past the causal range on the tail iteration: the SRD bounds it,
            so it lands as zeros with no memory traffic.
            """
            if const_expr(ENABLE_DMA):
                _kv_next = kv_start + fx.Index(BLOCK_KV)
                coop_dma_tile(k_rsrc, lds_base_idx, _kv_next)
                coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), _kv_next)

        def _kv_body(kv_start, inner, apply_mask):
            # The LDS tile for kv_start is already resident (prologue for the first
            # iteration, the previous body for the rest): this body consumes it and
            # leaves the tile for kv_start+BLOCK_KV behind (see the hand-over below).
            A_cur = [[inner[dt * QT + qt] for qt in range_constexpr(QT)] for dt in range_constexpr(DT)]
            sb_bulk = not apply_mask  # exps only exist on these paths
            _pf = coop_load_tile_regs(kv_start + fx.Index(BLOCK_KV)) if const_expr(KV_REG_PF) else None

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
                                # keep kv >= q+off-W (W+1 keys), matching the fwd SWA edge:
                                # '>=' keeps the boundary key kv == _thr == q+off-W.
                                _thr = q_slot_i32[qt] + causal_off_i32 - fx.Int32(window_left)
                                _kvb = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4)
                                p4 = Vec.from_elements(
                                    [
                                        ArithValue(_kvb + fx.Int32(t) >= _thr).select(Vec(p4)[t], c_zero_f)
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
                                    # keep kv >= q+off-W (W+1 keys), matching the fwd SWA edge.
                                    _lo = ArithValue(
                                        kv_slot < q_slot_i32[qt] + causal_off_i32 - fx.Int32(window_left)
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
                for d in range_constexpr(min(G2A, DT))
            ]
            rocdl.s_setprio(2)
            for d0 in range_constexpr(0, DT, 2):
                if const_expr(d0 + G2A < DT):
                    for _dn in range_constexpr(d0 + G2A, d0 + G2A + 2):
                        kts.append(
                            [_read_tr(fx.Index(0), _dn, pks) for pks in range_constexpr(PV_K_STEPS)]
                        )
                for pks in range_constexpr(PV_K_STEPS):
                    for dd in range_constexpr(d0, min(d0 + 2, DT)):
                        for qt in range_constexpr(QT):
                            A_cur[dd][qt] = mfma_acc(kts[dd][pks], c_pack[pks][qt], A_cur[dd][qt])
                # Interleave the next-pair prefetch ds_read_tr16 1:1 with the pair MFMAs.
                if const_expr(d0 + G2A < DT):
                    for _ in range_constexpr(2 * PV_K_STEPS * QT):
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
                # LDS hand-over (DMA path): this pair issues the tile's last k_tr read, so the
                # next tile is DMA'd into the SAME buffer while the remaining register-only
                # GEMM2 pair covers the transfer. Unlike an LDS double buffer (measured
                # net-negative) the in-flight writes never contend with LDS reads and no second
                # buffer is needed. Issuing it here also keeps the DMA address registers off the
                # all-DT k_tr peak.
                if const_expr(d0 == max(0, DT - 4) and not KV_REG_PF):
                    gpu.barrier()
                    _issue_dma(kv_start)  # noqa: B023
            rocdl.s_setprio(0)
            if const_expr(KV_REG_PF):
                # This pair of barriers is the whole per-tile rendezvous and prices at only
                # 0.67% of the kernel (WAR 0.62%, publish 0.10%), so an LDS double buffer --
                # which would remove the WAR half at 2x LDS plus a x2 loop unroll to keep the
                # slot bases compile-time constants -- is not worth its register cost.
                gpu.barrier()  # WAR: fence this tile's LDS reads before the rewrite
                coop_store_tile_lds(_pf)
            elif const_expr(ENABLE_DMA):
                rocdl.s_waitcnt(0)
            gpu.barrier()

            out = [A_cur[dt][qt] for dt in range_constexpr(DT) for qt in range_constexpr(QT)]
            return out

        # Split the causal kv-loop: [0, q_start) below the diagonal (no mask),
        # [q_start, kv_upper) straddles it (mask).
        _carry = A_accs
        loop_results = _carry
        if const_expr(window_left >= 0):
            # fx.Index is unsigned: guard the subtract (W may exceed q+off) to
            # avoid underflow-to-huge. _wlo skips fully-out-of-window kv tiles; the
            # first in-window kv is q+off-W (W+1-key window), so start from there.
            _wlo = fx.Index(
                ArithValue(q_start + causal_offset >= fx.Index(window_left)).select(
                    q_start + causal_offset - fx.Index(window_left), fx.Index(0)
                )
            )
            _wlo = (_wlo // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
        else:
            _wlo = fx.Index(0)
        # Prologue for the software-pipelined body: it expects its own tile already in
        # LDS and leaves the next one there. _wlo is the first kv tile of whichever of
        # the two loops below runs first (they are contiguous).
        if const_expr(ENABLE_DMA):
            coop_dma_tile(k_rsrc, lds_base_idx, _wlo)
            coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), _wlo)
            rocdl.s_waitcnt(0)
        gpu.barrier()
        for kv_start, inner in range(_wlo, q_start + causal_offset, BLOCK_KV, init=_carry):
            loop_results = yield _kv_body(kv_start, inner, False)
        for kv_start, inner in range(q_start + causal_offset, kv_upper, BLOCK_KV, init=loop_results):
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


def _blockkv_for(Skv, head_dim=64):
    # dkdv is KV-outer, so Skv (not Sq) sets its grid: a short Skv needs BLOCK_KV=64
    # to fill the CU array, a long one wants 128 to amortise the per-tile cost. Keying
    # this on Sq costs 19% on rectangular shapes such as Sq=2048, Skv=16384.
    # D128 scales the dK/dV accumulators with BLOCK_KV, so a wide tile only pays once the
    # grid is long enough to amortise it: it needs the full 512-register file
    # (waves_per_eu=1, see _get_bwd) and a short Skv would then run one wave per SIMD on a
    # sparse grid. Every LDS read feeds BLOCK_KV/64 MFMAs, so 192 (NT=3) drops the kernel's
    # reads/MFMA to 0.54 -- below the D64 tile's 0.875 -- and is the widest tile that still
    # fits the register file (256 spills catastrophically).
    # NOTE: the bench's SNR/det gate runs at a short S_REF, which lands in the 64 tier and
    # therefore never exercises this one. Re-verify dQ/dK/dV SNR and det at Skv >= 8192
    # directly whenever these tiers move.
    if head_dim >= 128:
        return 192 if Skv >= 8192 else 64
    return 64 if Skv <= 2048 else 128


def _fuse_blockkv_for(Skv):
    """kv band for the fused path. The dQ split-K traffic is (Skv/BLOCK_KV)/2 * |dQ| in
    each direction, so unlike the split path -- where BLOCK_KV only trades grid width
    against per-tile cost -- the fused path pays for a narrow band in DRAM bytes and
    wants the widest band the register file takes.
    """
    # 256 only fits because the K/V B-operands live in LDS (see _kv_lds_idx): NT goes
    # 2 -> 4, which doubles the dK/dV accumulators, and the packs it displaces are exactly
    # what pays for them. With the packs still in registers this was 512 vgpr / 325 spill
    # / occ 2 -> 1 and the bench fell 562 -> 433 TF.
    if Skv >= 4096:
        return 256
    return 64 if Skv <= 1024 else 128


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


_DQ_WS: dict = {}
_DQRED_CACHE: dict = {}
_CU_PH: dict = {}


def _cu_placeholder(device):
    """Unused cu_seqlens argument slot (read only under ``const_expr(varlen)``).

    Cached per device: a fresh one costs a fill kernel launch inside the timed backward.
    """
    ph = _CU_PH.get(device)
    if ph is None:
        ph = torch.zeros(1, device=device, dtype=torch.int32)
        _CU_PH[device] = ph
    return ph


def _dq_partial_ws(nb, B, Sq, hd, device, dtype):
    """dQ split-K workspace [bands, B, Sq, Hq*D] for the fused KV-outer kernel.

    One slot per kv band, so a band's contribution to a q row is written by exactly one
    work-group (no atomics, bitwise-reproducible). Cached: at the scored shape this is
    12.9 GB, far too much to reallocate per call.
    """
    key = (nb, B, Sq, hd, device, dtype)
    ws = _DQ_WS.get(key)
    if ws is None:
        _DQ_WS.clear()
        ws = torch.empty(nb, B, Sq, hd, device=device, dtype=dtype)
        _DQ_WS[key] = ws
    return ws


_SLOTRED_CACHE: dict = {}
# Elements one slot-reduce work-group folds per tensor (BLOCK*UC*VEC); a workspace whose
# per-group element count is not a multiple of this keeps torch's reduction.
_SLOTRED_TILE = 256 * 2 * 8


def _reduce_dkdv_slots(ws_dk, ws_dv, n_slots, n_groups, stream):
    """dk/dv = Sum over the q_split slot axis, in one FlyDSL pass over both tensors.

    ``ws_*`` are viewed as [n_groups, n_slots, n_elems]; the returned tensors are
    [n_groups, n_elems] and the caller reshapes them to the layout the workspace was
    built for (THD [B,q_split,Skv,Hkv,D] -> [B*Skv,Hkv,D], SBHD [q_split,...] with
    n_groups=1). Falls back to torch when the element count does not tile.
    """
    n_elems = ws_dk.numel() // (n_slots * n_groups)
    if n_elems % _SLOTRED_TILE:
        axis = 1 if n_groups > 1 else 0
        return ws_dk.sum(dim=axis), ws_dv.sum(dim=axis)
    dk = torch.empty(n_groups * n_elems, device=ws_dk.device, dtype=ws_dk.dtype)
    dv = torch.empty(n_groups * n_elems, device=ws_dv.device, dtype=ws_dv.dtype)
    key = (n_slots, n_groups, n_elems)
    launcher = _SLOTRED_CACHE.get(key)
    if launcher is None:
        if len(_SLOTRED_CACHE) >= 32:
            _SLOTRED_CACHE.clear()
        launcher = build_flash_attn_bwd_slotred_module(
            n_slots=n_slots, n_groups=n_groups, n_elems=n_elems
        )
        _SLOTRED_CACHE[key] = launcher
    launcher(ws_dk.reshape(-1), dk, ws_dv.reshape(-1), dv, stream)
    return dk, dv


def _reduce_dq_partials(ws, dq, block_kv, num_heads, head_dim, scale, stream):
    """dQ[q] = scale * Sum_{b : b*BLOCK_KV <= q} ws[b][q], in ascending band order.

    A kv band only writes the q rows that causally see it, so the bands ABOVE q's own
    band hold stale data and are skipped -- which is also what keeps the traffic at the
    causal half. Fixed band order and fp32 accumulation -> bitwise deterministic.

    ``scale`` is 1/log2e, not sm_scale: the fused kernel's fifth GEMM contracts against
    the LDS K tile, which is staged already prescaled by sm*log2e for GEMM1a.

    Band count is what this costs: at Skv=8192/BLOCK_KV=256 it reads 32 bands = 3.32 GB
    at 6 TB/s. Telling it 16 bands (half the reads, wrong dQ -- diagnostic only) is
    +6.0% of the WHOLE backward on 11/11 paired trials, so folding band pairs into one
    workspace slot is worth roughly +6% on the read side alone before counting the
    matching halving of the kernel's 3.32 GB of partial stores.
    """
    _, B, Sq, _ = ws.shape
    key = (num_heads, head_dim, B, Sq, block_kv, scale)
    launcher = _DQRED_CACHE.get(key)
    if launcher is None:
        if len(_DQRED_CACHE) >= 32:
            _DQRED_CACHE.clear()
        launcher = build_flash_attn_bwd_dqred_module(
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=B,
            seq_len_q=Sq,
            block_kv=block_kv,
            sm_scale=scale,
        )
        _DQRED_CACHE[key] = launcher
    # Pass ONE band slice: the descriptor is rebased per band with a 64-bit offset, and
    # the whole workspace overflows a flat memref's i32 element count.
    launcher(ws[0].reshape(-1), dq.reshape(-1), stream)


def _get_bwd(
    Hq, Hkv, D, scale, window_left, q_split, block_kv, dq_block_kv=64, batch_size=None, sbhd=False,
    varlen=False, fuse_dq=False,
):
    key = (Hq, Hkv, D, scale, window_left, q_split, block_kv, dq_block_kv, batch_size, sbhd, varlen, fuse_dq)
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
        # dq is Q-outer: a WG owns block_m q rows and streams ALL kv. D64's body is lean
        # enough that the widest tile (192) still leaves two waves per SIMD. D128's body is
        # twice as wide, so block_m=256 needs 511 registers = one wave per SIMD and loses all
        # sibling-wave latency hiding; halving the tile and asking for two waves (which also
        # needs the narrower kv tile below) lands on 2 waves per SIMD spill-free.
        # g2d: D128's GEMM2 reads a transpose per d-tile and only had the next pair in
        # flight, so every second MFMA pair waited on ds_read_tr16 latency. Reading four
        # d-tiles ahead covers it within the 6 registers the occ-2 budget still had spare;
        # six is already too deep (measured slower). D64 keeps 2 -> byte-identical.
        dq_block_m = 128 if D == 128 else 192
        dq_l = (
            None
            if fuse_dq
            else build_flash_attn_bwd_dq_module(
                block_kv=32 if D == 128 else dq_block_kv,
                waves_per_eu=2 if D == 128 else 1,
                g2d=4 if D == 128 else 2,
                batch_size=batch_size,
                sbhd=sbhd,
                fuse_delta=_FUSE_DELTA,
                block_m=dq_block_m,
                varlen=varlen,
                **common,
            )
        )
        # dkdv reads one LDS operand per MFMA per kv 16-tile, so the read/MFMA ratio is 1/NT
        # with NT = BLOCK_KV/64. D64 runs BLOCK_KV=128 (NT=2); D128 at BLOCK_KV=64 (NT=1)
        # issues twice the LDS reads per MFMA and is stalled on them. A wide tile restores
        # the reuse, but its accumulators (DT=8 halves per NT) only fit spill-free in the
        # full 512-register file: waves_per_eu=1 plus a depth-1 GEMM2 prefetch ring (a deeper
        # ring spills). At BLOCK_KV=64 D128 keeps the occ-2 / depth-3 pairing.
        # g2d (GEMM2 transpose-read prefetch depth): depth hides ds_read_tr16 latency but
        # costs live transpose-reads, so it trades against the register budget above.
        # dma_grp: heads per Q/dO staging round-trip. At D128 the kernel is one wave per
        # SIMD, so the Q/dO DMA latency and the barrier pair around it are fully exposed
        # once per head; staging two heads together pays that rendezvous half as often and
        # overlaps the two tiles' HBM latency, at 2x LDS which occ=1 has spare. Three-deep
        # and wider needs so many live slot addresses that the kernel starts using scratch.
        # pf_ring: with a 2x-deep slot ring the group rendezvous collapses to one barrier
        # and moves off the head boundary into the last GEMM2 step, so head h+1's GEMM1
        # and exp2 chain are free to schedule into head h's GEMM2 shadow. It doubles LDS
        # to 132 KB, which only the wide-tile (waves_per_eu=1, one work-group per CU)
        # configuration has spare -- at BLOCK_KV=64 the second work-group would no longer
        # fit in the 160 KB LDS and occ would fall from 2 to 1.
        if D == 128:
            dkdv_wpe = 1 if block_kv >= 128 else 2
            dkdv_g2d = 1 if block_kv >= 128 else 3
            dkdv_dma_grp = 2
            dkdv_pf_ring = block_kv >= 128
        else:
            dkdv_wpe, dkdv_g2d, dkdv_dma_grp = 2, 1, 1
            dkdv_pf_ring = False
        # The fused body keeps V's B-operand in registers; spending the 32 KB that frees on a
        # second Q/dO slot loses both ways it can be spent. As a prefetch slot (q_dbuf: head
        # h+1's tiles DMA'd at the top of step h, drained at its tail, so the barrier PAIR
        # collapses to one) it is 753 vs 780 at equal GEMM3 ring depth. As a second staging
        # slot (dma_grp=2: both heads fetched in one rendezvous, so a full drain and a
        # barrier pair are paid per two heads instead of per head) it is 767, and the four
        # live slot addresses it needs cost enough registers to miss occupancy 2. Only 11% of
        # this kernel's wave stalls are non-issue waits (barrier/waitcnt); the other 89% are
        # instruction-issue waits, so trading instructions for fewer fences loses.
        dkdv_l = build_flash_attn_bwd_dkdv_module(
            q_split=q_split,
            block_kv=block_kv,
            batch_size=batch_size,
            sbhd=sbhd,
            waves_per_eu=dkdv_wpe,
            g2d=dkdv_g2d,
            dma_grp=dkdv_dma_grp,
            pf_ring=dkdv_pf_ring,
            varlen=varlen,
            fuse_dq=fuse_dq,
            q_pref=fuse_dq,
            flat_wg=512 if fuse_dq and block_kv > 128 else 256,
            **common,
        )
        if fuse_dq:
            # The fifth GEMM replaces the whole dq kernel, so DELTA has no producer left:
            # the standalone odo pass comes back (measured +0.106 ms).
            odo_l = build_flash_attn_bwd_odo_module(
                num_heads=Hq, head_dim=D, num_kv_heads=Hkv, sm_scale=scale, sbhd=sbhd
            )
        elif _FUSE_DELTA:
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


_LSET_TILE = 32
_LSET_CACHE: dict = {}


def _prescale_lse(lse_bhsq):
    """Fold -log2e into lse host-side so the kernel's exp2 argument is a bare fma.

    The uniform path hands over a [B,Sq,Hq] -> [B,Hq,Sq] view, so this pass is a transpose
    as well as a scale. torch fuses the two but reduces the strided axis with 4 B accesses
    and runs at 1.1 TB/s; the LDS-tiled kernel below makes both sides 128 B contiguous.
    Other layouts (the packed ragged lse, native SBHD [B,Hq,Sq]) need no transpose and keep
    the plain ``mul`` -- which must still write a fresh contiguous buffer, since ``mul``
    would otherwise propagate the input's stride order to its output.
    """
    src = lse_bhsq.float()
    if src.dim() == 3:
        B, Hq, Sq = src.shape
        if src.stride() == (Sq * Hq, 1, Hq) and Sq % _LSET_TILE == 0 and Hq % _LSET_TILE == 0:
            out = torch.empty(B, Hq, Sq, device=src.device, dtype=src.dtype)
            key = (B, Sq, Hq)
            launcher = _LSET_CACHE.get(key)
            if launcher is None:
                if len(_LSET_CACHE) >= 32:
                    _LSET_CACHE.clear()
                launcher = build_flash_attn_bwd_lset_module(B=B, Sq=Sq, Hq=Hq, scale=-_LOG2E)
                _LSET_CACHE[key] = launcher
            # permute back to the contiguous [B,Sq,Hq] storage this view is built on
            launcher(src.permute(0, 2, 1).reshape(-1), out.reshape(-1), torch.cuda.current_stream())
            return out
    return torch.mul(src, -_LOG2E, out=torch.empty(src.shape, device=src.device, dtype=src.dtype))


# ============================================================================
# kernel: dsink (attention-sink gradient)
# ============================================================================

_DSINK_THREADS = 256


def build_flash_dsink_module(B, Sq, Hq):
    """d_sink[h] = sum over all (b, s) of exp(sink_h - lse[b,h,s]) * delta[b,h,s].

    LSE is the raw sink-inclusive natural-log softmax LSE and DELTA is the flash
    identity delta = -sum_d O_s[b,s,h,d]*dO[b,s,h,d] (already negated by the dq kernel),
    both fp32 [B,Hq,Sq] with the same flat layout (b*Hq+h)*Sq+s. Because delta carries
    the negation, no final negate is applied (unlike sparse's build_dsink_reduce).

    One WG per q-head (grid=(Hq,1,1)); the WG's 256 threads stride the head's B*Sq
    scalars, accumulate in fp32, then thread 0 sums the LDS partials and writes d_sink[h].
    Deterministic (fixed fp32 reduction order, no atomics)."""
    THREADS = _DSINK_THREADS
    NCHUNK = (Sq + THREADS - 1) // THREADS
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="flash_attn_bwd_dsink_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + THREADS * 4

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(SINK: fx.Tensor, LSE: fx.Tensor, DELTA: fx.Tensor, DSINK: fx.Tensor):
        lds = SmemPtr(allocator.get_base(), lds_off, fx.Float32.ir_type, shape=(THREADS,)).get()
        h = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        Hqn = fx.Index(Hq)
        Sqn = fx.Index(Sq)
        total_elems = fx.Index(B) * Hqn * Sqn

        sink_rsrc = buffer_ops.create_buffer_resource(
            SINK, max_size=False, num_records_bytes=_raw(Hqn * fx.Index(4))
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(total_elems * fx.Index(4))
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(total_elems * fx.Index(4))
        )
        dsink_rsrc = buffer_ops.create_buffer_resource(
            DSINK, max_size=False, num_records_bytes=_raw(Hqn * fx.Index(4))
        )
        c_log2e = fx.Float32(_LOG2E)
        c_zero = fx.Float32(0.0)
        sink_h = fx.Float32(buffer_ops.buffer_load(sink_rsrc, h, vec_width=1, dtype=fx.Float32))

        acc = fx.Float32(0.0)
        for b in range_constexpr(B):
            head_base = (fx.Index(b) * Hqn + h) * Sqn  # first scalar of (b, h) row
            for c in range_constexpr(NCHUNK):
                s = fx.Index(c * THREADS) + tid
                in_range = ArithValue(s < Sqn)
                # clamp OOB tail to element 0 of the row (in-buffer, contribution masked)
                g = head_base + fx.Index(in_range.select(s, fx.Index(0)))
                lse_g = fx.Float32(buffer_ops.buffer_load(lse_rsrc, g, vec_width=1, dtype=fx.Float32))
                delta_g = fx.Float32(buffer_ops.buffer_load(delta_rsrc, g, vec_width=1, dtype=fx.Float32))
                e = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sink_h - lse_g) * c_log2e)))
                term = e * delta_g
                acc = fx.Float32(
                    arith.AddFOp(_raw(acc), _raw(fx.Float32(in_range.select(term, c_zero)))).result
                )

        Vec.from_elements([acc], fx.Float32).store(lds, [tid])
        gpu.barrier()
        # thread 0 sums the 256 partials serially (one WG per head; tiny, deterministic).
        total = fx.Float32(0.0)
        for j in range_constexpr(THREADS):
            total = fx.Float32(
                arith.AddFOp(
                    _raw(total), _raw(Vec.load(Vec.make_type(1, fx.Float32), lds, [fx.Index(j)])[0])
                ).result
            )
        buffer_ops.buffer_store(
            total,
            dsink_rsrc,
            h * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.eq, _raw(tid), _raw(fx.Index(0))).result),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch(SINK, LSE, DELTA, DSINK, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(SINK, LSE, DELTA, DSINK).launch(grid=(fx.Index(Hq), 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


_DSINK_CACHE: dict = {}


def _flash_dsink(sink, lse_bhsq, delta, B, Hq, Sq, stream):
    """Launch the dsink reduction. ``sink``:[Hq] f32, ``lse_bhsq``/``delta``:[B,Hq,Sq] f32
    (raw sink-inclusive natural-log LSE and the already-negated identity delta). Returns
    d_sink:[Hq] f32."""
    d_sink = torch.empty(Hq, device=sink.device, dtype=torch.float32)
    args = (
        sink.reshape(-1),
        lse_bhsq.reshape(-1).contiguous(),
        delta.reshape(-1),
        d_sink,
        stream,
    )
    key = (B, Hq, Sq)
    compiled = _DSINK_CACHE.get(key)
    if compiled is None:
        if len(_DSINK_CACHE) >= 64:
            _DSINK_CACHE.clear()
        compiled = flyc.compile(build_flash_dsink_module(B, Sq, Hq), *args)
        _DSINK_CACHE[key] = compiled
    compiled(*args)
    return d_sink


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
    sink=None,
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
    ``sink`` (optional [Hq] f32): learned per-q-head attention sink. dQ/dK/dV are
    sink-agnostic (lse_bhsq is already sink-inclusive from the forward); when given, a
    dedicated reduction kernel also returns dsink[h]=Sum_i exp(sink_h-lse_i)*delta_flash
    (delta_flash is already -rowsum(O_s.dO), so no final negate), and the result is the
    4-tuple (dq,dk,dv,dsink) instead of (dq,dk,dv).

    Ragged / block-causal (cu_seqlens_q given, THD only): q/k/v/dq/dk/dv and lse_bhsq
    all packed; each segment [cu[i],cu[i+1]) is an independent document (per-segment
    bottom-right causal + cross-segment masking). Grid tiles by max_seqlen_q/kv. D in
    {64,128}; no learned sink on this path."""
    varlen = cu_seqlens_q is not None
    st = torch.cuda.current_stream()
    lse_s = _prescale_lse(lse_bhsq)
    qf, kf, vf, dof = q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1)
    o16 = out.to(q.dtype).reshape(-1)

    if varlen:
        assert not sbhd, "ragged / block-causal backward is THD only"
        assert sink is None, "ragged / block-causal backward does not support learned sink"
        assert _FUSE_DELTA, "ragged bwd fuses DELTA into dq (no odo launch)"
        num_seg = cu_seqlens_q.numel() - 1
        total_q, total_kv = q.shape[0], k.shape[0]
        max_sq = int(max_seqlen_q) if max_seqlen_q is not None else Sq
        max_skv = int(max_seqlen_kv) if max_seqlen_kv is not None else Skv
        q_split = _qsplit_for(max_sq)
        dq_l, dkdv_l, _ = _get_bwd(
            Hq, Hkv, D, scale, window_left, q_split,
            _blockkv_for(max_skv, D), _dq_block_kv(max_sq),
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
            cu_seqlens_q, cu_seqlens_kv, o16[:1], num_seg, max_sq, max_skv, total_kv, st,
        )
        dk = ws_dk.sum(dim=0)
        dv = ws_dv.sum(dim=0)
        return dq, dk, dv

    q_split = _qsplit_for(Sq)
    block_kv = _blockkv_for(Skv, D)
    # Fused KV-outer path: dkdv also emits dQ, so S/dP/softmax are computed once instead
    # of twice (5 GEMMs, not 7). It needs a per-band dQ workspace whose q axis is the kv
    # band axis, hence square + block-aligned shapes; SWA and D128 keep the split pair.
    fuse_kv = _fuse_blockkv_for(Skv)
    fuse_dq = (
        _FUSE_DQ and not sbhd and window_left < 0 and D == 64 and Sq == Skv and Sq % fuse_kv == 0
    )
    if fuse_dq:
        block_kv = fuse_kv
    dq_l, dkdv_l, odo_l = _get_bwd(
        Hq,
        Hkv,
        D,
        scale,
        window_left,
        q_split,
        block_kv,
        _dq_block_kv(Sq),
        batch_size=B,
        sbhd=sbhd,
        fuse_dq=fuse_dq,
    )
    # identity delta = -rowsum(O.dO); both kernels center dP by it (exact). dq owns the
    # reduce (it already holds dO in registers) and stores DELTA for dkdv when
    # _FUSE_DELTA is on, so no odo launch is needed; O is cast to bf16 (no-op when out
    # is already bf16) and passed into dq's freed slot via _defer_delta.
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    if fuse_dq or not _FUSE_DELTA:
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
    cu_ph = _cu_placeholder(q.device)
    if fuse_dq:
        ws_dq = _dq_partial_ws(Skv // block_kv, B, Sq, Hq * D, q.device, q.dtype)
        # Pass ONE (band, batch) slice: the kernel rebases the SRD to its own slice with a
        # 64-bit offset, and the whole workspace overflows a flat memref's i32 element count.
        dkdv_l(
            qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1),
            cu_ph, cu_ph, ws_dq[0, 0].reshape(-1), B, Sq, Skv, 0, st,
        )
        _reduce_dq_partials(ws_dq, dq, block_kv, Hq, D, 1.0 / _LOG2E, st)
    else:
        dq_l(qf, kf, vf, dof, lsef, df, dq.reshape(-1), o16, cu_ph, cu_ph, B, Sq, Skv, st)
        dkdv_l(
            qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1),
            cu_ph, cu_ph, o16[:1], B, Sq, Skv, 0, st,
        )
    if sbhd:
        dk, dv = _reduce_dkdv_slots(ws_dk, ws_dv, q_split, 1, st)
        dk = dk.reshape(Skv, B, Hkv, D)  # SBHD contiguous
        dv = dv.reshape(Skv, B, Hkv, D)
    else:
        dk, dv = _reduce_dkdv_slots(ws_dk, ws_dv, q_split, B, st)
        dk = dk.reshape(B * Skv, Hkv, D)
        dv = dv.reshape(B * Skv, Hkv, D)
    if sink is not None:
        # dsink[h] = Sum_i exp(sink_h - lse_i) * delta_flash[b,h,i], with delta already
        # -rowsum(O_s.dO) (negated) and lse_bhsq the raw sink-inclusive natural-log LSE.
        # Both are [B,Hq,Sq] with the same flat layout (b*Hq+h)*Sq+s.
        d_sink = _flash_dsink(sink, lse_bhsq, delta, B, Hq, Sq, st)
        return dq, dk, dv, d_sink
    return dq, dk, dv
