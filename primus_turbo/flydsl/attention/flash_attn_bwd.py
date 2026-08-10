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
# Warp specialisation (splitting the head-step across a wave pair by role) is
# register-walled: each role's live set needs more than half the 512-dword pool, so the
# pair cannot co-reside two waves per SIMD, and no register donor closes the gap.
# q rows one dkdv work-group folds per q-loop step. 128 is spill-free and cheaper per
# unit of work, but loses on the wall due to the wider band's dS ring rendezvous cost.
_BWD_BLOCK_Q = 64

# dkdv MFMA-accumulator AGPR forcing (amdgpu-agpr-alloc): only pays off once the body
# is VGPR-lean, so it is disabled here. On the four-wave fused body it is not even a
# knob -- the compiler's own split is already byte-identical across the range tried.
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
    # [B,Hq,S], so the natural flat-row tiling (one q, ROWS_PER_WG heads) writes scalars a
    # full S floats apart -- one cache line touched per 4 B. Trading heads for q instead
    # makes each work-group write SPW*4 contiguous DELTA bytes per head while the O/dO
    # side keeps SPW runs of HPW*D*2 contiguous bytes; past SPW=8 the shrinking O/dO run
    # costs more than the extra DELTA coalescing pays for.
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
    uc=None,  # vector chunks per thread (None: one work-group per q group, see below)
    vec=8,  # elements per load: 8 = buffer_load_dwordx4, 4 = dwordx2
    lpt=True,
    bat_lo=0,
    n_bat=None,  # None: the whole batch; else this launch owns batches [bat_lo, bat_lo+n_bat)
    # q_split/qsp_lo/n_qsp/block_q mirror the fused kernel's q-loop split: split s owns the
    # q blocks with (q/block_q) % q_split == s in every band, so once the dkdv launch for
    # splits [qsp_lo, qsp_lo+n_qsp) retires, exactly those q blocks are complete and can be
    # reduced while the rest of the band still runs. Same rows, same ascending band order,
    # so dQ is bitwise identical however the q blocks are partitioned across launches.
    q_split=1,
    qsp_lo=0,
    n_qsp=None,  # None: every q block (no q-split sub-range)
    block_q=_BWD_BLOCK_Q,
    causal_offset=0,  # Skv-Sq for a bottom-right-causal rectangular shape; 0 for square.
):
    """Fold the fused kernel's dQ split-K partials: DQ[b,q] = sm * Sum_b' WSQ[b',b,q].

    Only the bands a q row causally sees (b' <= q/BLOCK_KV) are read, in ascending
    order and with an fp32 accumulator, so the result is bitwise reproducible without
    atomics. One pass replaces torch's sum -> mul_ -> cast chain, which materialises an
    fp32 [B, BLOCK_KV*Hq*D] temporary per q group and touches it three more times.

    A work-group owns ``rows_per_wg`` q rows (one q group, hence one band count) and
    every thread carries ``uc`` independent 16 B chunks, so that many loads per band are
    in flight -- the band loop is dynamic and cannot be unrolled. ``lpt`` matters for the
    work-group width; every (block, uc, lpt) combination returns bit-identical dQ, so
    these are pure rate knobs -- re-sweep with _r5_dqred.py if the layout changes.

    ITS OWN RATE IS NOT WHAT TO TUNE FOR, THOUGH: this kernel runs CO-RESIDENT with the
    fused body (see _fused_pipelined), so what it really costs is the registers it takes
    out of the shared 512-dword pool. ``uc`` is that price -- a thread holds uc fp32
    accumulators of VEC plus uc loads in flight, so uc directly sets the per-wave
    allocation. Whether a smaller uc is worth its slower per-call rate depends entirely
    on whether the dwords it frees let a SECOND reduce wave co-reside per SIMD, which is
    a hard threshold effect (one more wave fitting is a step function, not a smooth
    trade), so the reduce's own latency barely matters: it runs mostly hidden inside the
    much longer fused kernel regardless of shape.

    The deciding equation is ceil8(alloc_body) + n*alloc_reduce <= 512, where alloc_body
    is whatever the fused kernel (see build_flash_attn_bwd_fused / g3_kreg / g1_ks_outer)
    currently allocates. Every uc/vec choice here must be re-verified against that
    equation whenever the fused body's own allocation changes, since a verdict taken at
    one alloc_body does not transfer to another -- crossing the 512 boundary evicts a
    wave outright rather than degrading gradually. The currently deployed uc favors
    fitting one FAST co-resident wave over one SLOW one, which measured strictly better
    than either the extra-wave attempt or the pre-donation baseline; re-run the sweep
    (interleaved against a baseline, not a lone number) before trusting a change here.

    Only the LAST pipeline chunk's reduce runs with nothing beside it, so that one may
    legally take a different (wider, faster-standalone) shape -- a chunk owns a disjoint
    element slice, and uc/block partition elements rather than reordering a slot's band
    sum, so mixing shapes across launches is still bitwise identical. Measured: not a
    gain, because the tail dispatch is a small fraction of the whole call regardless of
    per-thread load count, so its co-resident slowdown (which does not apply to it in
    the first place, since nothing else runs beside it) was never the real cost.

    Both sides are non-temporal: a partial is read exactly once by exactly this kernel
    and dQ is not read again in the backward, so keeping either in L2/MALL only evicts
    the dO the fused kernel re-reads once per band. The same hint is a LOSS on the odo
    kernel's O/dO reads and on the fused kernel's partial store, so it is not a general
    lever -- see the sign flip in memory.md.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "dq reduce kernel targets gfx950"
    elem_dtype = dtype_to_elem_type(dtype_str)
    HD = num_heads * head_dim
    SQ = seq_len_q
    VEC = vec
    RPW = rows_per_wg
    if block is None:
        cands = [b for b in (512, 256, 128, 64, 32) if (RPW * HD) % (b * VEC) == 0]
        assert cands, f"cannot tile {RPW}*{HD} elements into {VEC}-element lanes"
        block = cands[0]
    BLOCK = block
    LPT = lpt
    # ``uc`` splits a q group's RPW*Hq*D elements over N_CHUNK work-groups instead of
    # giving all of them to one, which is what prices this kernel's REGISTER footprint:
    # a thread carries UC fp32 accumulators of VEC plus UC loads in flight, so UC is the
    # per-wave allocation and the allocation is what decides how much of the 512-register
    # pool the fused kernel may keep for itself while a reduce work-group co-resides
    # beside it (see _dq_partial_ws / _fused_pipelined).
    UC = RPW * HD // (BLOCK * VEC) if uc is None else uc
    CHUNK_ELEMS = BLOCK * VEC * UC
    assert (RPW * HD) % CHUNK_ELEMS == 0, "rows_per_wg*Hq*D must tile the work-group"
    N_CHUNK = RPW * HD // CHUNK_ELEMS
    # ``batch_size`` is the workspace's band stride and stays whole-batch even when this
    # launch owns a batch slice, so a slice addresses exactly the rows the whole-batch
    # launch would have handed it and dQ comes out bitwise identical.
    NB = batch_size if n_bat is None else n_bat
    assert block_kv % RPW == 0 and (NB * SQ) % RPW == 0 and 0 <= bat_lo <= batch_size - NB
    BAND_BYTES = batch_size * SQ * HD * 2
    ROW0 = bat_lo * SQ

    QSP = q_split
    NQ = QSP if n_qsp is None else n_qsp
    BQ = block_q
    QSP_SUB = NQ < QSP  # this launch owns a strided subset of the q blocks
    assert 1 <= NQ <= QSP and 0 <= qsp_lo <= QSP - NQ
    if QSP_SUB:
        assert SQ % (BQ * QSP) == 0 and BQ % RPW == 0
    WG_PER_BLK = BQ // RPW  # work-groups per q block
    BLK_SEL = SQ // BQ // QSP * NQ  # q blocks this launch owns, per batch

    NGRP = NB * BLK_SEL * WG_PER_BLK if QSP_SUB else NB * SQ // RPW
    NWG = NGRP * N_CHUNK

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_dqred_kernel(WSQ: fx.Tensor, DQ: fx.Tensor):
        bid = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        chunk = fx.Index(0)
        if const_expr(LPT):
            # Longest-processing-time-first: work per work-group ramps along the grid
            # (band count = q/BLOCK_KV), so walking the grid backwards front-loads the
            # heaviest rows instead of leaving them to drain alone at the tail, with the
            # DRAM stream still sequential and the output bit-identical.
            bid = fx.Index(NWG - 1) - bid
        if const_expr(N_CHUNK > 1):
            # Chunk is the fastest grid axis, so neighbouring work-groups still read
            # neighbouring addresses (forwards or, under lpt, backwards).
            chunk = bid % fx.Index(N_CHUNK)
            bid = bid // fx.Index(N_CHUNK)
        if const_expr(QSP_SUB):
            # Grid is dense over the owned q blocks; spread it back onto the strided ones.
            _blk = bid // fx.Index(WG_PER_BLK)
            _qb = _blk % fx.Index(BLK_SEL)
            _qb = (_qb // fx.Index(NQ)) * fx.Index(QSP) + fx.Index(qsp_lo) + _qb % fx.Index(NQ)
            row0 = (
                fx.Index(ROW0)
                + (_blk // fx.Index(BLK_SEL)) * fx.Index(SQ)
                + _qb * fx.Index(BQ)
                + (bid % fx.Index(WG_PER_BLK)) * fx.Index(RPW)
            )
        else:
            row0 = fx.Index(ROW0) + bid * fx.Index(RPW)  # b*SQ + q of this group's first row
        # Topmost band this q group sees: with bottom-right causal masking a q row sees
        # keys kv <= q + causal_offset, so on a rectangular shape (Skv>Sq) it reaches
        # causal_offset//block_kv bands higher than its own. causal_offset==0 for square.
        g = ((row0 % fx.Index(SQ)) + fx.Index(causal_offset)) // fx.Index(block_kv)
        base = row0 * fx.Index(HD) + chunk * fx.Index(CHUNK_ELEMS) + tid * fx.Index(VEC)
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
        if const_expr(UC == 1):
            acc = [acc]  # a lone iter_arg comes back bare rather than as a 1-element list

        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=True)
        sm_vec = Vec.filled(VEC, sm_scale, fx.Float32)
        for c in range_constexpr(UC):
            _v = (Vec(acc[c]) * sm_vec).to(elem_dtype)
            # The workspace's D axis is permuted so the fused kernel can write a q row's
            # partial 64 B at a time (see the store in `_gemm3_tiles`): bit 4 of the real
            # D index sits at bit 2 of the permuted one, so a chunk's VEC elements are read
            # contiguously here and written back as VEC/4 runs of 4 at their un-permuted
            # address -- cheap on this read-bandwidth-bound kernel, and it halves the
            # partial store's request count on the fused kernel's critical path instead.
            for s in range_constexpr(VEC // 4):
                _o = offs[c] + fx.Index(4 * s)
                _dq_off = (
                    (_o - (_o & fx.Index(31)))
                    + ((_o & fx.Index(24)) >> fx.Index(1))
                    + (((_o >> fx.Index(2)) & fx.Index(1)) << fx.Index(4))
                )
                buffer_ops.buffer_store(
                    _v.shuffle(_v, [4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3]).ir_value(),
                    dq_rsrc,
                    _dq_off * fx.Index(2),
                    cache_modifier=2,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch_flash_attn_bwd_dqred(WSQ: fx.Tensor, DQ: fx.Tensor, stream: fx.Stream):
        flash_attn_bwd_dqred_kernel(
            WSQ,
            DQ,
            value_attrs={"rocdl.flat_work_group_size": f"{int(BLOCK)},{int(BLOCK)}"},
        ).launch(grid=(fx.Index(NWG), 1, 1), block=(BLOCK, 1, 1), stream=stream)

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
    # waves_per_eu is a request, not a cap: on the fused body LLVM already lands at 244
    # VGPR, so waves_per_eu=1 compiles byte-identical ISA (244 / 0 spill / 118,784 B) and
    # occupancy stays at 2 waves per SIMD either way. It is not a lever here.
    waves_per_eu=2,
    block_kv=128,
    # block_q: q rows staged per q-loop trip (None = _BWD_BLOCK_Q). Under a left window a
    # band's q extent is exactly BLOCK_KV + W rows, so a block_q that divides it walks the
    # SAME (kv,q) area in fewer staging trips -- 96 covers 64+128 in two instead of three.
    block_q=None,
    num_kv_heads=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    q_split=2,
    enable_dma=True,
    window_left=-1,
    # q_dbuf: stage the GQA group's Q/dO tiles into two alternating LDS slots so head h's
    # step issues head h+1's DMA up front and drains it only at its own tail, halving the
    # rendezvous count. Costs LDS_TOTAL extra bytes (only affordable at waves_per_eu=1);
    # on the fused body the DMA's LDS writes then interleave with the GEMMs' LDS reads for
    # a whole head-step, which costs more than the drain it hides.
    q_dbuf=False,
    fold_lse=None,  # None = fold on the hw-exp path only (see below)
    batch_size=None,  # compile-time B; required for SBHD seq-step stride bake
    sbhd=False,  # SBHD [S,B,H,D] native layout (seq-step = B*H*D)
    agpr=_DKDV_AGPR,  # force N MFMA accumulators into AGPRs (0 disables); layout-agnostic
    # exp_intrin: FOLD bulk exp via exp2 intrinsic anchor (vs v_min+dead-op asm) -- a
    # win once the body is VGPR-lean (spill-free). Layout-agnostic.
    exp_intrin=True,
    # exp_iglp: hand the head-step's scheduling region to LLVM's MFMAExpInterleave
    # strategy (`iglp_opt(2)`), which is the pipeline built for exactly this shape --
    # an MFMA chain feeding a quarter-rate transcendental chain. See EXP_IGLP.
    exp_iglp=True,
    # g2d: GEMM2 transpose-read prefetch depth (ring across dt). Depth-1 wins even once
    # deeper rings are register-free, because the read-ahead burst displaces MFMA issue
    # more than it saves in fences -- cutting fences is not this body's currency,
    # MFMA-run density is.
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
    # On for D128 and for the fused body; forcing it off there is 881.6 / 877.6 (-1.1%).
    g1_ks_outer=None,  # None = on for D128
    varlen=False,  # ragged / block-causal: per-segment [tok_base,tok_end) from cu_seqlens
    square=True,  # caller guarantees Sq==Skv (causal_offset==0); gates the Q_PAIR windowed
    # optimization, which is only correct for square shapes (see Q_PAIR).
    # fuse_dq: also emit dQ (the fifth GEMM) into a split-K workspace, replacing the
    # separate Q-outer dq kernel. See _FUSE_DQ and _gemm3 below.
    fuse_dq=False,
    # v_lds: stage the owned V rows in LDS as GEMM1b's B operand instead of keeping them
    # in registers. K has to be staged either way (GEMM3 transpose-reads it), V does not.
    # Registers win: this body stalls on LDS instruction issue, not on LDS capacity, and
    # the LDS a V stage would free stays unspent since every other use of it also loses.
    # Also useful as a probe for whether the power budget is spent on LDS->VGPR reads: at
    # the same power draw the staged-in-LDS arm clocks higher but is still slower, so a
    # denser instruction mix (fewer LDS reads) is what DVFS rewards, not less LDS traffic.
    v_lds=False,
    k_reg=True,  # feed GEMM1a's B from the K register packs, not the LDS tile (see K_REG)
    # g3_kreg: hold GEMM3's whole K^T fragment set live for the band instead of reading it
    # back per head-step. K^T is head-invariant, so this is pure read removal. See G3_KREG.
    g3_kreg=False,
    # g3_qt/g3_krt: GEMM3's output patch shape and how much of its K^T set is resident.
    # See G3_QT and G3_KRT -- both are priced against the co-residency budget, not the
    # read count, so neither is a free knob.
    g3_qt=2,
    g3_krt=None,  # None = all G3_DT D-tiles resident
    # qdo_tail: publish head h+1's Q/dO tile on the SAME fence that publishes head h's dS,
    # halving the head-step's rendezvous count. See QDO_TAIL.
    qdo_tail=False,
    # q_pref: stage the Q/dO tiles through VGPRs and issue head h+1's fetch at the top of
    # head-step h, so a whole head-step covers it. See Q_PREF.
    q_pref=False,
    # g3_defer: run GEMM3 one head-step late off a second dS slot. See G3_DEFER.
    g3_defer=True,
    # g2_half: flush GEMM2 per q-half instead of once per q-loop trip (None = fused only).
    # It shortens the pack live ranges, which is what lets BLOCK_Q grow past 64. See G2_HALF.
    g2_half=None,
    # qsp_lo/n_qsp: dispatch only the q_split sub-range [qsp_lo, qsp_lo+n_qsp) instead of
    # all q_split subsets. A split owns the q blocks with (q/BLOCK_Q) % q_split == split
    # in EVERY band, so a sub-range launch completes those q rows' dQ partials outright
    # and its reduce can start while the remaining splits still run. Grid, slot indices
    # and per-work-group work are otherwise untouched, so dQ/dK/dV stay bitwise identical.
    qsp_lo=0,
    n_qsp=None,  # None = all q_split subsets (single whole-band dispatch)
    # flat_wg: work-group size, and on this part it is really a REGISTER-FILE choice.
    # 512 (8 waves) puts two waves on every SIMD, capping each at 256 architected
    # registers; 256 (4 waves, one per SIMD) opens the whole 512-register file to a
    # single wave via AGPRs. The wide fused band needs the second form: its wider
    # accumulator set only fits spill-free there, and one wave per SIMD also doubles
    # ROWS_PER_WAVE_KV so each transpose-read feeds more MFMAs. The two forms trade
    # per-cycle efficiency against clock roughly evenly on their own; what makes the
    # four-wave form the right pick is the registers per SIMD it leaves free, which is
    # what lets the dQ reduce co-reside (see `_dq_partial_ws` / `_fused_pipelined`).
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

    BLOCK_Q = _BWD_BLOCK_Q if block_q is None else int(block_q)
    WARP_SIZE = 64
    NUM_XCD = 8  # gfx950 XCDs; the dispatcher hands block_id to xcd = block_id % NUM_XCD
    BLOCK_KV = block_kv
    Q_SPLIT = q_split
    assert q_split >= 1
    N_QSP = Q_SPLIT if n_qsp is None else n_qsp
    QSP_LO = qsp_lo
    assert 1 <= N_QSP <= Q_SPLIT and 0 <= QSP_LO <= Q_SPLIT - N_QSP
    flat_work_group_size = flat_wg
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE_KV = BLOCK_KV // NUM_WAVES

    # ---- 16x16x32 bf16 MFMA tiling (M=N=16, K=32): four independent 16x16 accumulator
    # chains at the same accumulator VGPR total (dkdv is MFMA dep-wait bound). Lane layout:
    # lane%16 = M/N index, lane//16 = K-subgroup (4 x 8 = K32) and, on the C output, the
    # M-block ((lane//16)*4 + t, t in 0..3 -> 4 f32/lane).
    # 32x32x16 buys nothing here: LDS traffic per carrier is the same either way, so the
    # wider MFMA only halves instruction count where issue is nowhere near a limit, and
    # GEMM2 cannot even take it -- P/dS sit in the GEMM1 C-layout, which is not
    # lane-uniform the way 32x32x16 needs without a cross-lane shuffle or a second LDS trip. ----
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
    # A wave's output patch is the squarest G3_TILES-tile rectangle: transpose reads per
    # MFMA go as (G3_DT + G3_QT) / (G3_DT * G3_QT), so a square patch minimizes LDS
    # traffic per MFMA. GEMM3 runs on only the first G3_WAVES waves (MFMA-neutral, since
    # waves round-robin over the SIMDs); amortising K^T across a head pair or spreading
    # one head over all waves both lose, on the register cliff or on read-burst density.
    # Ring depth is priced by *where* GEMM3 sits -- right after the head-step's RAW
    # barrier, where the carrier wave has no other MFMA to issue, so a deeper ring hides
    # latency for free instead of displacing MFMA issue like GEMM2's g2d does mid-stream.
    # Depth 6 of the 8 available ksteps measures as the optimum on the fused band.
    G3D = min(6, G3_KSTEPS)
    G3_WAVES = min(NUM_WAVES, max(1, DT * MT // min(4, DT * MT)))  # waves carrying GEMM3
    G3_TILES = max(1, DT * MT // G3_WAVES)  # output tiles per carrier wave
    # Re-pricing GEMM3's patch shape once K^T is band-resident (G3_KRT) is bound by the
    # co-residency budget, not by LDS read count: growing G3_DT multiplies every wave's
    # resident set, not just its read count, so shrinking the patch only pays off if it
    # frees enough registers for the co-resident dQ-reduce wave (`_dq_partial_ws`) to fit.
    G3_QT = min(MT, int(g3_qt) if G3_TILES >= 2 else 1)  # q 16-tiles per wave
    G3_DT = G3_TILES // G3_QT  # D 16-tiles per wave
    G3_QGRP = MT // G3_QT  # q-tile groups; wave -> (D group, q group)
    # G3_SPLIT: run GEMM3 as one pass per q-half instead of once at the head-step's end,
    # to fill this body's one bare-MFMA VALU window. Disabled: LLVM hoists the early
    # pass's MFMAs to the top of its emission region instead of leaving them where aimed,
    # so the bare run grows instead of shrinking.
    G3_SPLIT = False
    G3_SPL_STRIDE = G3_QGRP if G3_SPLIT else 1  # q-tile stride within a wave's patch
    G3_SPL_AT = 1  # q-half whose GEMM1 the early pass is emitted after
    G3D_E = 3  # kstep prefetch depth of the early split pass
    if FUSE_DQ:
        assert head_dim == 64 and window_left < 0 and not sbhd and not varlen
        assert fold_lse, "fused dQ reads the prescaled K tile; see _reduce_dq_partials"
        assert DT * MT == G3_WAVES * G3_DT * G3_QT, "GEMM3 tiles must partition over carriers"
        # A wave's D-tiles must pair up (and start even) for the permuted partial layout
        # the store in _gemm3_tiles and the reduce in build_flash_attn_bwd_dqred_module
        # both assume; G3_DT even makes _g3d0 = (wave/G3_QGRP)*G3_DT even too.
        assert G3_DT % 2 == 0, "permuted dQ partial layout needs an even G3_DT"
        assert BLOCK_KV % PV_K_STEP == 0
        assert batch_size is not None, "fused dQ needs compile-time B for the workspace stride"
        # D128's occ-1 recipe (dma_grp=2 + pf_ring) does not port here even though the
        # fused body is now one wave per SIMD too: the dS ring has no fence of its own,
        # it rides the PER-HEAD Q/dO staging barrier pair, and both alternatives either
        # trip this assert (dma_grp=2, which pays that pair once per two heads instead)
        # or fail the bitwise-determinism gate (pf_ring). See QDO_TAIL for why giving the
        # ring its own fence loses on this body's fence-trading economics.
        assert dma_grp == 1, (
            "fused dQ rides the per-head Q/dO staging barriers as the dS WAR fence"
        )

    # sched_barrier(TRANS) pins MFMA/ds_read/VALU in place and frees only the softmax's
    # quarter-rate v_exp to migrate, so the exps are what fills GEMM1b's MFMA latency shadow
    # (schedule-only: opcode multiset and output unchanged). It stays off on the exp2
    # intrinsic path: forcing it on (alone or duplicated after GEMM1b) loses there, since
    # the default placement of this body's MFMA runs is already a local optimum.
    SCHED_TRANS = 0x400  # LLVM SchedGroupMask: TRANS (v_exp)
    # EXP_IGLP: at one wave/SIMD there's no sibling wave to hide exp2 latency under, so
    # hand the head-step region to LLVM's MFMAExpInterleave IGLP strategy instead of
    # hand-placed barriers. Gated to NUM_WAVES == 4 (see _dq_partial_ws for the call count).
    EXP_IGLP = bool(exp_iglp) and FUSE_DQ and NUM_WAVES == 4
    IGLP_EXP_INTERLEAVE = 2  # LLVM IGLPStrategyID::MFMAExpInterleaveID
    # G2_HALF: run GEMM2 once per q-half instead of once per head-step. Fused-only -- the
    # split bodies keep the single call so their ISA stays byte-identical; see the
    # emission point in _head_step_lds for what it buys.
    G2_HALF = FUSE_DQ if g2_half is None else bool(g2_half)
    # Q_PAIR: a windowed band's two waves attend q ranges staggered by one half-tile, so
    # walking contiguous tiles wastes a half-tile per wave. Stage the band's FIRST and LAST
    # q-half together (each wave takes its own half); trip count becomes compile-time NB.
    Q_PAIR = (
        window_left >= 0
        # Q_PAIR's compile-time NB trip count and half-tile pairing assume the band's q
        # extent starts at kv_start (causal_offset==0); on a rectangular shape (Sq!=Skv)
        # its dk/dv come out wrong (SNR~15) while the plain windowed q-loop below stays
        # correct. Only take Q_PAIR when the caller guarantees a square shape.
        and square
        and not fuse_dq
        and ENABLE_DMA
        and not q_dbuf
        and not bool(q_pref)
        and not bool(pf_ring)
        and Q_SPLIT == 1
        and NUM_WAVES == 2
        and PV_K_STEPS == 2
        and BLOCK_Q // 2 == PV_K_STEP
        # wave 1's first attended q row is the second half-tile's first row ...
        and ROWS_PER_WAVE_KV == BLOCK_Q // 2
        # ... and wave 0's last one is the last row before the extent's final half-tile.
        and (BLOCK_KV + window_left) % BLOCK_Q == 0
        and (BLOCK_KV + window_left) // BLOCK_Q >= 2
        and ROWS_PER_WAVE_KV + window_left <= BLOCK_KV + window_left - BLOCK_Q // 2
    )
    Q_PAIR_NB = ((BLOCK_KV + window_left) // BLOCK_Q) if Q_PAIR else 0
    # Q_BOUND: the windowed q-loop grid is BLOCK_KV-aligned, not BLOCK_Q-aligned, so when
    # BLOCK_Q doesn't divide BLOCK_KV (or Q_PAIR fixes the trip count at NB) the band
    # nearest the sequence end walks past real rows and needs an explicit mask term.
    Q_BOUND = window_left >= 0 and (BLOCK_KV % BLOCK_Q != 0 or Q_PAIR)
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
    # only waited on at its ds_write one head-step later. Runs at LDS_SLOTS == 1: a 2-slot
    # ring would retire the staging pair's WAR barrier too, but the register cost of a
    # second live slot outweighs that barrier's price on this body.
    Q_PREF = bool(q_pref) and ENABLE_DMA and not q_dbuf and not PF_RING and DMA_GRP == 1
    # Issue point for that fetch: earlier gives it more of the step as shadow but keeps
    # its VGPRs live over GEMM2's accumulator peak. The point right before GEMM2 is the
    # cheapest-scratch tie among the options that still fully cover the fetch.
    QPF_AT = 2
    # PF_QB: the LAST head-step of a q-block has no next head to fetch for, so it issues
    # head 0's fetch of the NEXT q-block instead -- Q/dO and the group's (-delta, lse) --
    # riding the q-loop's iter_args. The fused body runs ONE work-group per CU, so nothing
    # else is resident to cover a q-block prologue; the last head-step is also where
    # register pressure is lowest, so the extra live values land in the cheapest spot.
    PF_QB = Q_PREF and FUSE_DQ
    # MASK_SKIP: let a wave sit out a diagonal q-block whose kv rows it cannot see. Its
    # P and dS are zero there, so this only removes work -- the output is bitwise equal.
    # Full-causal only (a left window would need the lower edge too), fused only.
    MASK_SKIP = FUSE_DQ and window_left < 0
    # GEMM1 emits ks-outer: the four kv tiles of one k-step first, so consecutive MFMAs
    # write different accumulators and the next one issues without waiting on the last
    # one's result. At D128 (one wave per SIMD) there is no sibling wave to cover that
    # latency at all; at D64 it still pays, together with PF_QB.
    # The fused body emits ks-inner instead: wall-neutral on its own, but it frees 16
    # live dwords that are handed to the co-resident dQ reduce so it can widen its own
    # load (see uc in _dq_partial_ws) -- restore ks-outer here only together with that.
    G1_KS_OUTER = (HEAD_DIM == 128) if g1_ks_outer is None else bool(g1_ks_outer)
    # QDO_TAIL: publish head h+1's Q/dO tile at the END of head-step h instead of at its
    # own start, so the drain + barrier that already fences dS publishes BOTH and the
    # head boundary's own pair disappears (2 rendezvous per head-step become 1). Needs a
    # second slot on each of the Q/dO and dS rings, since the writer is now a head ahead
    # of the reader. Disabled: even with the extra slot funded back out of GEMM3's
    # resident K^T set, a barrier is still cheaper here than any structure that removes
    # one -- the same conclusion G3_DEFER reaches independently below.
    QDO_RING = bool(qdo_tail) and Q_PREF and FUSE_DQ and not g3_defer
    QDO_TAIL = QDO_RING  # the merged publish the second slots exist for
    LDS_SLOTS = 2 if (q_dbuf or QDO_RING) else ((2 * DMA_GRP) if PF_RING else DMA_GRP)
    assert GQA_GROUP_SIZE % LDS_SLOTS == 0
    # Whole-window residency (the shape dq uses) does NOT port here: keeping a head's
    # Q/dO extent resident needs LDS this body doesn't have room for without dropping a
    # work-group per CU, so the occupancy loss outweighs the fence/code-size savings.
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
    # buffer_load takes power-of-two vectors up to dwordx4, so a per-thread run that is not
    # one (block_q=96 leaves 6 floats) is issued as its greedy power-of-two pieces.
    LD_CHUNKS = []
    _ld_rem = LD_VEC
    while _ld_rem:
        _ld_c = min(4, 1 << (_ld_rem.bit_length() - 1))
        LD_CHUNKS.append((LD_VEC - _ld_rem, _ld_c))
        _ld_rem -= _ld_c

    # The Q/dO slot ring, and under FUSE_DQ the K, V and dS tiles, share one element-indexed
    # view so every reader (_a_idx / _read_tr / _kv_lds_idx / _g3s_idx) addresses them the
    # same way.
    LDS_VIEW_ELEMS = LDS_TOTAL * LDS_SLOTS
    V_LDS = FUSE_DQ and bool(v_lds)
    # K_REG: GEMM3 transpose-reads the staged K tile, but GEMM1a's B operand can come
    # from the register packs that filled it instead of being re-read from LDS once per
    # q-half (the LDS copy stays -- this is a read-side choice, not a staging one). This
    # is the one asymmetry in this body's LDS accounting: deleting reads elsewhere is
    # free, but adding these back costs, because they would land inside GEMM1's MFMA run
    # as fresh SrcB dependencies and break its issue density. What this body pays for is
    # MFMA-run density, not LDS latency or read count.
    K_REG = FUSE_DQ and bool(k_reg)
    G3_KREG = FUSE_DQ and bool(g3_kreg)
    # D-tiles of the band-resident K^T set; the rest are re-read every head-step.
    G3_KRT = min(G3_DT, G3_DT if g3_krt is None else int(g3_krt)) if G3_KREG else 0
    G3K_BASE = LDS_VIEW_ELEMS  # prescaled K [BLOCK_KV][HEAD_DIM]
    G3V_BASE = G3K_BASE + (BLOCK_KV * HEAD_DIM if FUSE_DQ else 0)  # V [BLOCK_KV][HEAD_DIM]
    G3S_BASE = G3V_BASE + (BLOCK_KV * HEAD_DIM if V_LDS else 0)  # dS [slot][BLOCK_KV][BLOCK_Q]
    G3S_SLOT_ELEMS = BLOCK_KV * BLOCK_Q
    # g3_defer: let GEMM3 lag one head-step behind the head that produced dS and read the
    # OTHER slot, so its RAW edge is covered by the head boundary's own staging barrier
    # pair (drain + publish) and its WAR edge by the pair one step later. Both dS fences
    # per head-step then disappear, for the price of a second dS slot. That trade paid
    # while the body ran eight waves per work-group and loses at four: the extra slot's
    # registers and LDS have no sibling MFMA run left to hide the retired fences under.
    G3_DEFER = FUSE_DQ and g3_defer
    G3S_SLOTS = 2 if (G3_DEFER or QDO_RING) else 1
    # G3_SHADOW: emit the deferred GEMM3 INSIDE the rendezvous, between the Q/dO DMA issue
    # and its drain, since GEMM3 is the only work that reads neither the slot being filled
    # nor GEMM1's output. Disabled: a pending buffer_load ... lds forces vmcnt(0) before
    # every later ds_read of the same LDS allocation, and LLVM plants that wait at GEMM3's
    # first transpose-read, so the shadow never materialises -- the same blocker as
    # G3_SHADOW's DMA path applies even through the VGPR-staged Q_PREF route, because
    # publishing dS at the WAR barrier instead of at the drain needs its own retire there.
    G3_SHADOW = False
    # HS_WAR_BAR: whether the head-step still needs its own leading WAR barrier before
    # overwriting the Q/dO slot. With the undeferred GEMM3 the PREVIOUS head-step already
    # ends in [lgkmcnt(0) drain, barrier] to publish dS, and that drain retires every
    # wave's GEMM2 reads of the slot, so the WAR edge is discharged before this head-step
    # begins. Keeping the barrier anyway is then pure rendezvous cost plus a scheduling
    # wall between GEMM3's MFMAs and the ds_write pair that refills the slot.
    HS_WAR_BAR = not (FUSE_DQ and not G3_DEFER) and not QDO_TAIL
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
            # The q_split axis is deliberately the FASTEST after the XCD term: all Q_SPLIT
            # work-groups of a band then walk interleaved q-blocks of the same band, so a
            # resident window covers every q-block of a whole band group with Q_SPLIT
            # readers each. Both alternative orderings (band slowest or band fastest)
            # measured worse -- whatever the dispatch order is worth here is spent on L2
            # sharing, not on makespan.
            if const_expr(N_QSP > 1):
                split_idx = _slot % fx.Index(N_QSP) + fx.Index(QSP_LO)
                _slot = _slot // fx.Index(N_QSP)
            else:
                split_idx = fx.Index(QSP_LO)
            kv_tile_idx = _slot % num_kv_tiles
            _u = _slot // num_kv_tiles
            _bkv = _u * fx.Index(NUM_XCD) + _xcd
            kv_head_idx = _bkv % NUM_HEADS_KV
            batch_idx = _bkv // NUM_HEADS_KV
        else:
            kv_head_idx = block_id % NUM_HEADS_KV
            _rest = block_id // NUM_HEADS_KV
            if const_expr(N_QSP > 1):
                split_idx = _rest % fx.Index(N_QSP) + fx.Index(QSP_LO)
                _rest = _rest // fx.Index(N_QSP)
            else:
                split_idx = fx.Index(QSP_LO)
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
        seq_len_q_i32 = fx.Int32(seq_len_q_v)
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
        # field and materialises a separate live address per A-fragment family, which on
        # this register-full body spills badly. Pinning one address per tile removes that.
        # Below the limit the compile-time form is cheaper, and pinning only the
        # overflowing slots is worse than pinning all of them (mixed addressing modes
        # give the allocator two live-range shapes to juggle).
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

        # Row-blocks one N_TILE row step advances the LDS image by. A tile step is a
        # multiple of 8 rows, so _pblk splits into a tile term and a lane term --
        # _pblk(t*N_TILE + lane16) == t*ROW_BLK + _pblk(lane16) -- which is what lets a
        # pinned base reach its whole tile family by a compile-time offset.
        ROW_BLK = (N_TILE // 2) if PACK_2ROW else N_TILE
        # LDS element delta of one half-tile of q rows (Q_PAIR): whole 8-row groups, so the
        # packed image steps by ROW_BLK blocks per N_TILE rows.
        Q_PAIR_HALF = (BLOCK_Q // 2 // N_TILE) * ROW_BLK * PBLK

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

            if const_expr(Q_PAIR):
                # A copy batch covers 16 consecutive tile rows, so the batches split evenly
                # into the tile's two half-tiles -- which is what lets Q_PAIR source the
                # second half from a different q offset at no extra address cost.
                assert NUM_DMA_Q % 2 == 0 and ROWS_PER_DMA_BATCH * 16 == BLOCK_SIZE

            def _dma_bases(tile_start, poff=None):
                """Head-independent part of the Q/dO DMA byte offset, one per batch.

                Only the q_head term differs between GQA heads sharing a q-block, so hoisting
                the row/swizzle/column derivation collapses each head's DMA to a single add
                and takes the kernel's scratch spill to zero.

                poff (Q_PAIR) sources the tile's second half-tile poff rows further on
                instead of BLOCK_Q/2, pairing two non-adjacent halves in one tile.
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
                    _src_row = tile_start + row_in_tile
                    if const_expr(poff is not None and d >= NUM_DMA_Q // 2):
                        _src_row = _src_row + poff - fx.Index(BLOCK_Q // 2)
                    bases.append(_src_row * fx.Index(RD_STRIDE_Q * 2) + col_byte)
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
            # GEMM3 contracts over kv, so its A operand is K^T: stage the owned K (and, for
            # GEMM1b, V) into LDS as [kv][D] in the Q/dO tile layout and transpose-read it
            # back, once per kv-block -- a pure register->LDS repack that takes both B
            # operands off the register file for the whole kernel, avoiding the spill
            # cliff. K goes in ALREADY PRESCALED, so GEMM1a reads it directly; that leaves
            # the dQ partial scaled by sm*log2e, which `_reduce_dq_partials` divides out.
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

        # TR_PIN: reach GEMM2's transpose reads from one pinned base per tile instead of a
        # separate loop-invariant address per (dt, pks, row-half). D128 needs it because its
        # slot ring overflows the ds_read offset field; the fused D64 body wants it for the
        # registers -- it runs at NT=2 with a full file, which is exactly the regime
        # _opaque_idx describes.
        TR_PIN = HEAD_DIM == 128 or FUSE_DQ
        # See _pin_bases: hoist the pinned bases themselves out of the q-loop. Needs a
        # single Q/dO slot, so the base is not a function of the head.
        HOIST_PIN = TR_PIN and FUSE_DQ and (LDS_SLOTS == 1 or QDO_RING)

        def _tr_off(i):
            return i * ROW_BLK * PBLK

        def _tr_base(a_base):
            """The (dt=0, pks=0, row-half=0) transpose-read address.

            Every other dt is this base XOR (dt*D_TILE): the swizzle mask (row&7)<<4 and
            the column term dt*16 occupy the same bit field, while the row stride (128),
            the tile base (multiple of BLOCK_Q*128) and the lane column (bits 2-3) all
            avoid it -- so bits 4-6 of the base are exactly row&7 and XORing dt in
            reproduces col ^ mask. The other (pks, row-half) reads ride _tr_off as ds_read
            offset immediates, so one XOR per (tile, dt) feeds all four reads and a single
            loop-invariant address per tile stays live instead of one per (dt, pks).

            pks only shifts the row by PV_K_STEP, and _pblk is affine in whole 8-row groups
            (32*pks rows -> 16*pks blocks), so the D64 packed layout rides the same offsets.
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
            if const_expr(base is not None):
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
            # Under G3_SPLIT the wave's q-tiles are strided by G3_QGRP so it owns one tile
            # per q-half; otherwise they are the contiguous G3_QT run. Either way the
            # (D group, q group) product is a partition of the DT x MT output tiles.
            return (
                (wave_id // fx.Index(G3_QGRP)) * fx.Index(G3_DT),
                (wave_id % fx.Index(G3_QGRP)) * fx.Index(1 if G3_SPLIT else G3_QT),
            )

        # Every GEMM3 transpose-read address is q-loop invariant, so left alone LICM would
        # hoist the whole address set into the preheader and keep it all live across the
        # loop body. Instead one base per operand family is pinned inside the body (see
        # `_opaque_idx`), and every read reaches it by a compile-time element offset plus,
        # for the tile index, one XOR (the tile index lands in bits 4-5 of the swizzled
        # column, disjoint from the lane column's bits 2-3, so `column + tile == column
        # XOR tile`).
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

        # dS staging layout [kv][qp] with a qp ^= 8*(kv&7) swizzle to avoid the bank
        # conflict the raw MFMA C-layout would hit under the Q/dO tiles' mask. qp permutes
        # q so the eight dS values a lane packs become contiguous, turning a ds_write_b64
        # pair into one ds_write_b128:
        #   q  = [a c b1 b0 d1 d0]  (mt = 2a+c, kg = b, t = d)
        #   qp = [a b1 b0 c d1 d0]  -> qp = 32*(mt//2) + 8*kg + 4*(mt%2) + t
        # GEMM3 contracts over kv, so the permuted n axis only permutes dQ output rows;
        # `_g3_qrow` inverts it at the partial store and every value stays bit-identical.
        def _g3s_wbase():
            """Pinned (nt=0, q-run=0, slot=0) dS write address for this lane's kv row.

            The C-layout write is the same family as the reads above: the kv row's swizzle
            mask is 8*(lane16&7), the q run occupies bit 5 of the column and the lane's kg
            bits 3-4, so the run index is one XOR off the base and (nt, slot) are element
            offsets folded into the ds_write offset field.
            """
            _r = wave_id * fx.Index(ROWS_PER_WAVE_KV) + lane16
            return _opaque_idx(
                fx.Index(G3S_BASE)
                + _r * fx.Index(BLOCK_Q)
                + ((kg * fx.Index(8)) ^ ((lane16 & fx.Index(7)) * fx.Index(8)))
            )

        def _g3_qrow(tile):
            """q row of GEMM3's n index ``tile*16 + lane16`` -- the inverse of the qp
            permutation the dS staging applies (see _g3s_wbase)."""
            return (
                ((tile >> fx.Index(1)) << fx.Index(5))
                + ((tile & fx.Index(1)) << fx.Index(3))
                + (((lane16 >> fx.Index(2)) & fx.Index(1)) << fx.Index(4))
                + ((lane16 >> fx.Index(3)) << fx.Index(2))
                + (lane16 & fx.Index(3))
            )

        def _ds_write_vec(lds_elem_idx, const_elem_off, val):
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
                    ^ ((_r & fx.Index(7)) * fx.Index(8))
                )
            )

        def _gemm3(q_start, head_local, slot, drain=None, qsel=None, depth=None):
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
                    _gemm3_tiles(q_start, head_local, slot, drain, qsel, depth)
                if const_expr(drain is not None):
                    if wave_id >= fx.Index(G3_WAVES):
                        drain()
            else:
                _gemm3_tiles(q_start, head_local, slot, drain, qsel, depth)

        def _gemm3_tiles(q_start, head_local, slot, drain=None, qsel=None, depth=None):
            """dQ^T[m=D][n=q] += K^T . dS^T over this band's kv rows, for ONE head.

            Both operands are transpose-reads over kv, so the kv permutation ds_read_tr16
            imposes is identical on the two sides and cancels in the contraction. K^T stays
            in LDS: hoisting it into registers measured neutral at BLOCK_KV=64 and +0.44 ms
            / 413 spill at 128. The caller owns the fence -- see G3S_SLOTS.
            """
            _g3d0, _g3q0 = _g3_wave_tiles()
            if const_expr(HOIST_PIN):
                _kb, _sb = _pins["g3k"], _pins["g3s"]
            else:
                _kb, _sb = _g3_kbase(_g3d0), _g3_sbase(_g3q0)
            _soff = slot * G3S_SLOT_ELEMS
            # qsel picks ONE of the wave's q-tiles (its dS columns come from that q-half
            # alone), so the pass can run as soon as that half's softmax has published.
            _qs = list(range_constexpr(G3_QT)) if qsel is None else [qsel]
            _g3 = [[c_zero_v4f32 for _ in _qs] for _ in range_constexpr(G3_DT)]

            def _g3_frags(kk):
                # GEMM3's transpose reads are free, like GEMM1's and GEMM2's: a probe that
                # pairs the ksteps so the odd one's reads CSE onto the even one's (wrong dQ,
                # but 1536 -> 1024 tr at an untouched MFMA count) measures 6/11 -- the last
                # read family this body had not priced. Do not spend a round on read count.
                # K^T is head-INVARIANT, so G3_KREG holds all G3_DT*G3_KSTEPS fragments live
                # for the band and reads them once instead of once per head-step.
                if const_expr(G3_KREG):
                    _ka = list(_g3kt[kk]) + [
                        _g3_tr(_kb, i, kk, PBLK // 2) for i in range_constexpr(G3_KRT, G3_DT)
                    ]
                else:
                    _ka = [_g3_tr(_kb, i, kk, PBLK // 2) for i in range_constexpr(G3_DT)]
                return _ka, [
                    _g3_tr(_sb, j * G3_SPL_STRIDE, kk, BLOCK_Q, _soff) for j in _qs
                ]

            # kstep prefetch ring, depth G3D: kk+G3D's transpose-reads are issued before
            # kk's MFMAs so the ds_read_tr16 latency lands in the MFMA shadow instead of at
            # every kstep's first MFMA -- the same trade GEMM2's g2d ring makes, paid for by
            # the registers the pinned bases above freed. Unlike GEMM2's ring, this one
            # wants no sched_group_barrier or s_setprio around it: both cost more in spill
            # or scheduling freedom than they save, since the scheduler's own default burst
            # already keeps the run's lgkmcnt deep. An early split pass carries its ring
            # across the softmax it covers, so its depth is priced against that live range.
            _gd = const_expr(G3D if depth is None else min(depth, G3_KSTEPS))
            _ring = [_g3_frags(kk) for kk in range_constexpr(_gd)]
            for _kk in range_constexpr(G3_KSTEPS):
                _g3k, _g3s = _ring[_kk % _gd]
                if const_expr(_kk + _gd < G3_KSTEPS):
                    _ring[_kk % _gd] = _g3_frags(_kk + _gd)
                for i in range_constexpr(G3_DT):
                    for jj in range_constexpr(len(_qs)):
                        _g3[i][jj] = mfma_acc(_g3k[i], _g3s[jj], _g3[i][jj])
            if const_expr(drain is not None):
                drain()
            # Store the partial out of the dQ^T C-layout, a PAIR of D-tiles per store.
            # The C-layout alone hands a lane only 32 B of a q row per instruction -- half
            # of a 64 B request slot, doubling fabric traffic. A wave's G3_DT tiles are
            # adjacent and even-aligned, so the workspace's D axis is instead PERMUTED to
            # dperm = I*16 + kg*8 + p*4 + t for tile I+p (see `_dq_partial_ws`): a lane's 8
            # bf16 become contiguous and the 4 kg lanes of a row cover a full 64 B in ONE
            # dwordx4, with no LDS trip or barrier needed to get there.
            # The store stays CACHED: the partials are read back by the reduce, and a
            # non-temporal policy would lose the L2 write-combining the 64 B pairing sets up.
            _g3qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            for i2 in range_constexpr(G3_DT // 2):
                i = 2 * i2
                for jj in range_constexpr(len(_qs)):
                    j = _qs[jj]
                    _g3p = bf16_trunc_scored_v4(_g3[i][jj]).shuffle(
                        bf16_trunc_scored_v4(_g3[i + 1][jj]), [0, 1, 2, 3]
                    )
                    _g3row = q_start + _g3_qrow(_g3q0 + fx.Index(j * G3_SPL_STRIDE))
                    _g3col = (_g3d0 + fx.Index(i)) * fx.Index(D_TILE) + kg * fx.Index(8)
                    buffer_ops.buffer_store(
                        _g3p.ir_value(),
                        wsq_rsrc,
                        ((_g3row * fx.Index(NUM_HEADS_Q) + _g3qh) * fx.Index(HEAD_DIM) + _g3col)
                        * fx.Index(2),
                        offset_is_bytes=True,
                    )

        # HOIST_PIN: the pinned bases are functions of wave_id and lane only, invariant
        # over the whole q-loop. `_opaque_idx` stops LICM from hoisting the individual
        # read addresses, but it also re-emits the bases every head-step; emitting them
        # once ahead of the q-loop deletes those instructions at zero register cost.
        _pins = {}

        def _pin_bases():
            # One base per Q/dO ring slot: the slot offset is a whole-tile stride, far
            # above the swizzle's bit field, so each slot is just another pinned base.
            _pins["q"] = [_tr_base(fx.Index(s * LDS_TOTAL)) for s in range_constexpr(LDS_SLOTS)]
            _pins["do"] = [
                _tr_base(fx.Index(s * LDS_TOTAL + LDS_DO_BASE)) for s in range_constexpr(LDS_SLOTS)
            ]
            if const_expr(FUSE_DQ):
                _g3d0, _g3q0 = _g3_wave_tiles()
                _pins["g3w"] = _g3s_wbase()
                _pins["g3k"] = _g3_kbase(_g3d0)
                _pins["g3s"] = _g3_sbase(_g3q0)

        if const_expr(HOIST_PIN):
            _pin_bases()

        # G3_KREG: GEMM3's A operand is the band's prescaled K tile, which no head-step
        # writes, so its whole fragment set can be read ONCE per band and kept live over
        # the q-loop. The extra live registers this costs are only affordable while the
        # dQ reduce still co-resides in what is left of the 512-dword pool (see
        # `_reduce_dq_partials`). The band prologue's LDS store of the tile needs its own
        # publish barrier here -- every other read of it sits behind a head-step's fence.
        _g3kt = None
        if const_expr(G3_KREG):
            rocdl.s_waitcnt(WAIT_LGKM)
            gpu.barrier()
            _g3kb = _pins["g3k"] if const_expr(HOIST_PIN) else _g3_kbase(_g3_wave_tiles()[0])
            _g3kt = [
                [_g3_tr(_g3kb, i, kk, PBLK // 2) for i in range_constexpr(G3_KRT)]
                for kk in range_constexpr(G3_KSTEPS)
            ]

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

        if const_expr(Q_PAIR):
            # Q_PAIR needs the (-delta, lse) staging to follow the tile's row pairing, so a
            # thread's LD_VEC run must sit wholly inside one half-tile.
            assert (BLOCK_Q // 2) % LD_VEC == 0

        def _stage_ld_issue(q_start, poff=None):
            # Issued BEFORE the Q/dO DMA so both HBM streams are in flight together;
            # the LDS commit lands after the DMA, so its vmcnt wait does not serialise
            # them (gfx950 has no vmcnt subset wait, but the counter is in-order).
            if const_expr(poff is not None):
                # Second half-tile: same source shift as the Q/dO DMA (see _dma_bases).
                q_start = q_start + ArithValue(_ld_q >= fx.Index(BLOCK_Q // 2)).select(
                    poff - fx.Index(BLOCK_Q // 2), fx.Index(0)
                )
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
                            for j in range_constexpr(_o, _o + _c)
                        ],
                        fx.Float32,
                    ).ir_value()
                    for rsrc in (delta_rsrc, lse_rsrc)
                    for _o, _c in LD_CHUNKS
                ]
            _g = (kv_head_idx * fx.Index(GQA_GROUP_SIZE) + _ld_head) * seq_len_q_v + q_start + _ld_q
            _v = [
                buffer_ops.buffer_load(rsrc, _g + fx.Index(_o), vec_width=_c, dtype=fx.Float32)
                for rsrc in (delta_rsrc, lse_rsrc)
                for _o, _c in LD_CHUNKS
            ]
            if const_expr(LD_VEC == 1):
                # An 8-wave group leaves one element per thread; vec_width=1 lowers to a
                # scalar, and the LDS commit below stores vectors.
                _v = [Vec.from_elements([fx.Float32(x)], fx.Float32).ir_value() for x in _v]
            return _v

        def _stage_ld_commit(vals):
            _lds_i = _ld_head * fx.Index(LD_HEAD_ELEMS) + _ld_q
            for arr in range_constexpr(2):
                for i, (_o, _c) in enumerate(LD_CHUNKS):
                    Vec(vals[arr * len(LD_CHUNKS) + i]).store(
                        ld_lds, [fx.Index(arr * LD_ARR_ELEMS + _o) + _lds_i]
                    )

        def _ld_read(head_local, mt, arr, qoff=None):
            # v4f32 at q = head's q-block + mt*M_TILE + kg*4 (+t), matching the GEMM1
            # accumulator C layout; lane16 is absent -> a 16-way LDS broadcast.
            # arr=0 -> -delta (GEMM1b init), arr=1 -> prescaled lse (GEMM1a init/masked add).
            # qoff (Q_PAIR half-step) selects the tile half this wave is running.
            _i = (
                fx.Index(arr * LD_ARR_ELEMS + head_local * LD_HEAD_ELEMS + mt * M_TILE)
                + kg * fx.Index(4)
            )
            if const_expr(qoff is not None):
                _i = _i + qoff
            return Vec.load(v4f32_type, ld_lds, [_i]).ir_value()

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
            poff=None,
            half=False,
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
            # Q_PAIR: the paired trip's second half-tile holds rows q_start + poff instead
            # of the contiguous q_start + BLOCK_Q/2, and each wave runs only its own half,
            # so the LDS base, (-delta, lse) rows and mask q index all shift accordingly.
            _pk_list = [0] if const_expr(half) else list(range_constexpr(PV_K_STEPS))
            _hq = None
            if const_expr(half):
                _hq = wave_id * fx.Index(BLOCK_Q // 2)
                q_lds = q_lds + wave_id * fx.Index(Q_PAIR_HALF)
            do_lds = q_lds + fx.Index(LDS_DO_BASE)

            def _ld_rd(mt, arr):
                return _ld_read(head_local, mt, arr, _hq)

            def _q_slot_i32(mt):
                if const_expr(half):
                    return q_start_i32 + fx.Int32(wave_id * poff) + fx.Int32(mt * M_TILE)
                _o = fx.Int32(mt * M_TILE)
                if const_expr(poff is not None and mt >= MT // 2):
                    _o = _o + fx.Int32(poff) - fx.Int32(BLOCK_Q // 2)
                return q_start_i32 + _o
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
                if const_expr(HS_WAR_BAR):
                    gpu.barrier()  # WAR: the previous head's GEMM2 still read this slot
                # Under QDO_TAIL only head 0 publishes here; every later head's tile was
                # committed at the end of the previous head-step and published by that
                # step's dS barrier. Moving JUST the ds_write back into the previous
                # step's GEMM3 run (same two barriers, no second ring slot) loses: the
                # staged tile then has to stay live across GEMM1/GEMM2 instead.
                if const_expr(not QDO_TAIL or head_local == 0):
                    _qdo_commit(qdo, q_lds)
                    qdo = None
                    _qdo_pf(0)
                    if const_expr(head_local == 0):
                        _stage_ld_commit(_ldv)
                    rocdl.s_waitcnt(WAIT_LGKM)  # retire ds_writes; the loads stay in flight
                    gpu.barrier()  # Q/dO + ld_lds commit visible before GEMM1 reads
                else:
                    _qdo_pf(0)
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
                    _ldv = _stage_ld_issue(q_start, poff)
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
                # was published by the staging pair above, so GEMM3 needs no fence of its
                # own, and the same pair one step later fences the read against the head
                # that reuses the slot. Positioned here for two reasons: the dQ partial
                # stores get a whole GEMM1+GEMM2 of slack before the next drain retires
                # them (gfx950 shares one vmcnt between loads and stores), and GEMM3's ring
                # and accumulators die before GEMM1a's fragments go live, so the two
                # register peaks no longer add -- which is what pays for the ring depth.
                _gemm3(q_start, head_local - 1, (head_local - 1) % G3S_SLOTS)

            _qdo_pf(1)

            # GEMM1a/exp2/GEMM1b/dS/pack per q-HALF (one pks = two mt packing into one
            # GEMM2 K=32 step): processing 2 of the MT q-tiles at a time halves the live
            # S/dP/P/dS transient that pinned dkdv at spill, so the kernel fits spill-free.
            # lse/-delta are pulled from LDS at their use points (only the 2 v4f32 this
            # half consumes are ever live). Pure re-ordering -> bit-identical, det-neutral.
            p_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            ds_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            if const_expr(HOIST_PIN):
                _g3wb = _pins["g3w"]
            else:
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
                # boundary: by here the head has issued every read of its own slot (the
                # transpose-read ring runs g2d ahead and stops at DT-1-g2d), so the drain
                # retires them and the slot it refills is free. An earlier dt is not legal
                # (its reads are still to come) and hoisting the last dt's reads instead to
                # move the rendezvous off DT-1 loses, since their live range then crosses
                # it on an already-full register file.
                _mid_dt = (DT - 1) if const_expr(carry_rdv) else -1
                _n_out = 2  # sched-hint scale: 1 op-stream per output (dV + dK)
                # The priority pair de-phases the two waves of a SIMD: the one in GEMM2
                # wins issue until it drops out, so its sibling's exp chain drifts into
                # this MFMA run instead of contending with it. On the four-wave body
                # there is no such sibling any more (the co-resident dQ reduce wave is
                # DRAM-latency-bound, not issue-hungry, so winning slots from it buys
                # nothing), so the pair is inert rather than negative here -- unlike
                # pitfalls/12's s_setprio verdict for sparse-MLA attention, where it cost
                # throughput outright. Kept at the measured deployment point (prio 1).
                rocdl.s_setprio(1)
                for dt in range_constexpr(DT):
                    if const_expr(dt == _mid_dt):
                        rocdl.s_setprio(0)
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        for _sh in mid_pf:
                            _dma_head(_sh, bases)
                        rocdl.s_setprio(1)
                    if const_expr(dt == 1 and pk_list[-1] == _pk_list[-1]):
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
                            dv_cur[dt][nt] = mfma_acc(
                                do_tr[i], p_pack[pk_list[i]][nt], dv_cur[dt][nt]
                            )
                    if const_expr(NT >= 3):
                        # NT>=3 pins the packs' liveness hard enough that the RA sinks the
                        # pack next to the MFMA that reads it as SrcB. Pinning the dV group
                        # live past its MFMAs blocks that sinking, which reduces spill even
                        # now that the scored pack makes the sink itself legal. Naming fewer
                        # than all four elements of each tuple saves v_accvgpr reads but
                        # measures neutral, so all four stay; the pin is dV-only (dK regresses).
                        _keepalive_v4([dv_cur[dt][nt] for nt in range_constexpr(NT)])
                    if const_expr(_rd_next):
                        q_tr_n = [
                            _read_tr(q_lds, dt + g2d, pk_list[i], _q_trb)
                            for i in range_constexpr(_nk)
                        ]
                    for i in range_constexpr(_nk):
                        for nt in range_constexpr(NT):
                            dk_cur[dt][nt] = mfma_acc(
                                q_tr[i], ds_pack[pk_list[i]][nt], dk_cur[dt][nt]
                            )
                    if const_expr(_rd_next):
                        # Grouping the whole read set ahead of the MFMA run loses, even
                        # though it drops half the run's s_waitcnt lgkmcnt(2), because the
                        # read burst blocks MFMA issue. Dropping the hints entirely and
                        # letting the default scheduler place the run is worse still, so
                        # this pair is load-bearing, not decorative.
                        for _ in range_constexpr(_n_out * _nk * NT):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(1)
                        do_ring[_slot] = do_tr_n
                        q_ring[_slot] = q_tr_n
                rocdl.s_setprio(0)

            if const_expr(HOIST_PIN):
                _slot_pin = const_expr(head_local % LDS_SLOTS)
                _q_trb, _do_trb = _pins["q"][_slot_pin], _pins["do"][_slot_pin]
            else:
                _q_trb = _tr_base(q_lds) if const_expr(TR_PIN) else None
                _do_trb = _tr_base(do_lds) if const_expr(TR_PIN) else None
            _q_apin = _a_pin(q_lds) if const_expr(A_PIN) else None
            _do_apin = _a_pin(do_lds) if const_expr(A_PIN) else None

            def _gemm_dp(half):
                return _gemm_qk(
                    do_lds,
                    v_b_packs,
                    inits={mt: _ld_rd(mt, 0) for mt in half},
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
                if const_expr(EXP_IGLP):
                    # One call per q-half: the region's MFMA -> exp chain is per half, and
                    # two calls is where the register outcome lands right (see EXP_IGLP).
                    rocdl.iglp_opt(IGLP_EXP_INTERLEAVE)
                if const_expr(fold_lse and not apply_mask):
                    # FOLD unmasked: prescaled -log2e*lse is GEMM1a's C-init, so the
                    # accumulator already IS the base-2 softmax exponent.
                    _st = _gemm_qk(
                        q_lds,
                        k_b_packs,
                        inits={mt: _ld_rd(mt, 1) for mt in half},
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
                        lse_v = _ld_rd(mt, 1)
                        for nt in range_constexpr(NT):
                            s_v = s_tiles[mt][nt]
                            p_vals = []
                            for t in range_constexpr(4):
                                s_r = fx.Float32(Vec(s_v)[t])
                                if const_expr(apply_mask):
                                    q_slot = _q_slot_i32(mt) + kg_off_i32 + fx.Int32(t)
                                    _up = ArithValue(kv_row_i32_of(nt) > q_slot + causal_off_i32)
                                    if const_expr(window_left >= 0):
                                        # keep kv >= q+off-W (W+1 keys), matching the fwd
                                        # SWA edge; strict '<' -> the boundary key q+off-W stays.
                                        _lo = ArithValue(
                                            kv_row_i32_of(nt)
                                            < q_slot + causal_off_i32 - fx.Int32(window_left)
                                        )
                                        _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                                        if const_expr(Q_BOUND):
                                            # A BLOCK_Q not dividing BLOCK_KV lets the band's
                                            # last trip run past the sequence end into the
                                            # next batch's (real, in-range) rows.
                                            _mm = ArithValue(
                                                arith.ori(
                                                    _raw(_mm),
                                                    _raw(ArithValue(q_slot >= seq_len_q_i32)),
                                                )
                                            )
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
                _pk_seg = [pks] if const_expr(G2_HALF) else list(_pk_list)
                _rings = None
                if const_expr(G2_HALF or pks == _pk_list[-1]):
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
                        # Publish dS as [kv][qp] for GEMM3's transpose-read. The v8 pack is
                        # q = {ma,mb}*16 + kg*4 + t of ONE kv row, which the qp permutation
                        # lays out as ONE 8-wide run -> a single ds_write_b128 (see
                        # _g3s_wbase). The run index is bit 5 of the column, hence pks*32.
                        _g3wo = nt * N_TILE * BLOCK_Q + (head_local % G3S_SLOTS) * G3S_SLOT_ELEMS
                        _ds_write_vec(
                            _g3wb ^ fx.Index(pks * 2 * M_TILE), _g3wo, ds_pack[pks][nt]
                        )

                return _rings

            # GEMM2 per q-half, consuming a half's packs as soon as they exist. The
            # per-accumulator half order stays pks-ascending -> bit-identical, and the read
            # and MFMA counts are untouched. Two things move: the packs die a half earlier,
            # relieving the next half's GEMM1 register peak, and a half's GEMM2 MFMAs land
            # next to the NEXT half's GEMM1a and exp chain, filling what the ISA otherwise
            # shows as a bare VALU window. Flushing GEMM2 later still -- once that next
            # half's MFMA pipe is already full -- loses outright, so it is adjacency plus
            # the register relief that pays here, not interleaving for its own sake.
            def _pks_chain(pf=True, g3_split=False):
                _rings = None
                for pks in _pk_list:
                    half = [2 * pks, 2 * pks + 1]
                    # Half pks-1's dQ pass, emitted one q-half after its dS was published:
                    # its 16 MFMAs are the only independent matrix work available to the
                    # last half's softmax tail. AT=0 puts it ahead of this half's GEMM1,
                    # AT=1 between GEMM1 and the softmax it is meant to cover.
                    if const_expr(g3_split and pks > 0 and G3_SPL_AT == 0):
                        _gemm3(q_start, head_local, 0, qsel=pks - 1, depth=G3D_E)
                    _st, _dpt = _half_gemm1(half)
                    if const_expr(g3_split and pks > 0 and G3_SPL_AT == 1):
                        _gemm3(q_start, head_local, 0, qsel=pks - 1, depth=G3D_E)
                    _rings = _half_soft(pks, half, _st, _dpt)
                    if const_expr(g3_split and pks < PV_K_STEPS - 1):
                        rocdl.s_waitcnt(WAIT_LGKM)
                        gpu.barrier()  # RAW: this half's dS columns feed every wave
                    if const_expr(G2_HALF):
                        _last = const_expr(pks == _pk_list[-1])
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
                        list(_pk_list),
                        _rings[0],
                        _rings[1],
                        const_expr(PF_RING and mid_pf is not None),
                    )

            if const_expr(MASK_SKIP and apply_mask):
                # Diagonal q-block: a wave whose whole ROWS_PER_WAVE_KV kv rows sit above
                # this block's causal edge has P = dS = 0 for every head, so it skips the
                # entire GEMM chain behind a wave-uniform branch and only publishes zeros
                # into its dS rows (GEMM3 contracts the WHOLE band, so they must be
                # defined). The live waves' arithmetic is untouched and the skipped
                # contributions are exact zeros, so every output stays bitwise identical.
                # The next tile's fetch is every wave's own lanes, so it stays outside.
                _qdo_pf(2)

                def _live():
                    # No split here: its publish barrier would sit inside a wave-divergent
                    # region. Diagonal blocks are 6% of the visits and 0.4% of the wall.
                    _pks_chain(pf=False)
                    return _flat_accs()

                def _dead():
                    _z = Vec.from_elements(
                        [fx.Int32(0) for _ in range_constexpr(4)], fx.Int32
                    ).bitcast(elem_dtype)
                    for nt in range_constexpr(NT):
                        _zo = nt * N_TILE * BLOCK_Q + (head_local % G3S_SLOTS) * G3S_SLOT_ELEMS
                        for pks in range_constexpr(PV_K_STEPS):
                            _ds_write_vec(_g3wb ^ fx.Index(pks * 2 * M_TILE), _zo, _z)

                _q_last = q_start + fx.Index(BLOCK_Q - 1) + causal_offset
                _cond = ArithValue(kv_row_wave <= _q_last)
                _set_accs(_if_wave(_cond, _flat_accs(), _live, _dead))
            else:
                _pks_chain(g3_split=const_expr(G3_SPLIT))
            if const_expr(FUSE_DQ and not G3_DEFER):
                # Undeferred: dS is read in the head-step that wrote it, so this head-step
                # pays its own RAW fence. gpu.barrier() alone is not a fence -- retire the
                # ds_writes first with lgkmcnt only (a full drain would also wait on the
                # previous head-step's dQ partial stores, which nothing here reads).
                # Emitting GEMM3 here rather than before GEMM2 keeps its transpose-reads'
                # live ranges off GEMM2's, which loses on a full register file.
                if const_expr(
                    QDO_TAIL and head_local + 1 < GQA_GROUP_SIZE and _qdo_next[0] is not None
                ):
                    # Head h+1's tile rides this fence. Its ring slot was last read by
                    # head h-1, whose reads all precede the previous head-step's barrier.
                    _qdo_commit(
                        _qdo_next[0], fx.Index(((head_local + 1) % LDS_SLOTS) * LDS_TOTAL)
                    )
                    _qdo_next[0] = None
                rocdl.s_waitcnt(WAIT_LGKM)
                gpu.barrier()  # RAW: every wave's dS rows feed every wave's GEMM3
                _gemm3(
                    q_start,
                    head_local,
                    const_expr(head_local % G3S_SLOTS),
                    qsel=const_expr(
                        PV_K_STEPS - 1
                        if (G3_SPLIT and not (MASK_SKIP and apply_mask))
                        else None
                    ),
                )
            if const_expr(q_dbuf and head_local + 1 < GQA_GROUP_SIZE):
                rocdl.s_waitcnt(0)  # prefetch landed; the next step's barrier publishes it
            return dv_cur, dk_cur, (_qdo_next if const_expr(Q_PREF) else [qdo, None])

        def _q_body(q_start, inner, apply_mask, poff=None, half=False):
            # inner (loop-carried) = [dv accs][dk accs] (+ [Q/dO][-delta, lse] under PF_QB).
            _dk_base = DT * NT
            dv_cur = [[inner[dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)]
            dk_cur = [
                [inner[_dk_base + dt * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)
            ]
            # Head-invariant DMA offsets: computed once per q-block, reused by all heads.
            _bases = _dma_bases(q_start, poff) if const_expr(ENABLE_DMA and not Q_PREF) else None
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
                    poff=poff,
                    half=half,
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

        # The q loop walks UP from the band's own first query, which staggers band b by
        # 4b q-blocks and causes repeated cross-band re-reads of the same Q/dO tiles.
        # Walking DOWN instead (every band's range ends at seq_len, so descending puts
        # concurrent work-groups on the same q-block) does cut DRAM traffic, but loses on
        # the wall: the added cross-work-group sharing is itself a contention hotspot. Not
        # worth re-walking for bytes alone -- this kernel is nowhere near DRAM-bandwidth
        # bound; only revisit for latency, and then the access phase must be spread first
        # (e.g. rotate the GQA head order by band) to avoid a same-cycle hotspot.
        _carry = dv_accs + dk_accs
        if const_expr(PF_QB):
            # Prologue fetch for the first q-block; every later one is issued a head-step
            # early inside the body. The masked loop hands its pending fetch to the
            # unmasked loop: _unmask_start is exactly the last masked q_start + _step (and
            # _q_loop_start itself when the masked loop is empty), so the carry stays valid.
            _carry = _carry + _qdo_issue(_q_loop_start, 0) + _stage_ld_issue(_q_loop_start)
        loop_results = _carry

        if const_expr(Q_PAIR):
            # NB trips: one paired trip carries the band's two extreme q-halves (each wave
            # its own half), plus NB-1 interior tiles; the paired trip's position rotates
            # with the band index for L2 reuse (see _qrot), spelled as two interior loops.
            _poff = fx.Index((2 * Q_PAIR_NB - 1) * (BLOCK_Q // 2))
            _in0 = _q_loop_start + fx.Index(BLOCK_Q // 2)
            _ihi = _in0 + fx.Index((Q_PAIR_NB - 1) * _step)
            _ilo = _ihi - (_q_loop_start // fx.Index(_step)) % fx.Index(Q_PAIR_NB) * fx.Index(_step)
            for q_start, inner in range(_ilo, _ihi, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
            loop_results = _q_body(_q_loop_start, loop_results, True, poff=_poff, half=True)
            for q_start, inner in range(_in0, _ilo, _step, init=loop_results):
                loop_results = yield _q_body(q_start, inner, True)
        elif const_expr(window_left >= 0):
            _qhi = _kv_end_c - causal_offset + fx.Index(window_left)
            _qhi = fx.Index(ArithValue(_qhi < seq_len_q_v).select(_qhi, seq_len_q_v))
            # Phase-align the visit order across bands: every band's q range is NB blocks
            # wide and the same q-block is read by NB different bands, so rotating each
            # band's start by its own index makes trip i of every band land on the same
            # q-block (mod NB), turning cross-band reuse into an L2 hit (permutation only;
            # dk/dv accumulation order per band is unchanged, so determinism holds).
            _qnb = (_qhi - _q_loop_start + fx.Index(_step - 1)) // fx.Index(_step)
            _qnb = fx.Index(ArithValue(_qnb > fx.Index(0)).select(_qnb, fx.Index(1)))
            _qrot = (_q_loop_start // fx.Index(_step)) % _qnb
            for q_start, inner in range(_q_loop_start, _qhi, _step, init=_carry):
                _trip = (q_start - _q_loop_start) // fx.Index(_step)
                _qs = _q_loop_start + ((_trip + _qnb - _qrot) % _qnb) * fx.Index(_step)
                loop_results = yield _q_body(_qs, inner, True)
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
        grid_x = bs_idx * num_kv_tiles * NUM_HEADS_KV * N_QSP

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
        # amdgpu-mfma-vgpr-form on its own (agpr=0, i.e. the deployed 4-wave body) is inert:
        # forced true and forced false both emit BYTE-IDENTICAL ISA to the default, because
        # the accumulators are already past the 256 arch-VGPR line, so the 343 accvgpr moves
        # are the register file's shape, not this flag's choice.
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
        # post-misched is load-bearing on the four-wave fused body: dropping it measures
        # 892.5 against 918.9, and every amdgpu-sched-strategy override is worse still
        # (max-memory-clause 863.0, max-ilp 888.4) -- the hand-placed sched_mfma/sched_dsrd
        # structure is what the default scheduler is being asked to preserve.
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
    # GQA sharers per work-group: the CTA's waves split into (row groups) x (q heads of
    # one kv head). Width is unchanged, so occupancy and LDS are too.
    q_heads_per_wg=1,
    q_group=1,
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
    # GQA sharers spread ACROSS the waves of an unchanged-width CTA (low wave bits pick
    # the row group, high bits pick the q head), so sharers of a kv head share one staging
    # pass/rendezvous instead of each walking the full windowed kv extent. Unlike the
    # judged-negative GQA merge in pitfalls/13 (which widened the CTA to 8 waves), waves,
    # registers, LDS and work-group count here are unchanged -- only wave ownership moves.
    Q_HEADS_PER_WG = q_heads_per_wg
    WAVE_ROW_GROUPS = NUM_WAVES // Q_HEADS_PER_WG
    ROWS_PER_WAVE_Q = BLOCK_M // WAVE_ROW_GROUPS  # 32
    # A wave's own window is ROWS_PER_WAVE_Q + W rows inside the group's BLOCK_M + W, so
    # the overhang tiles are all-masked for it and get a wave-uniform skip. Sharing the
    # tile across heads instead of rows removes the overhang entirely.
    WAVE_OVERHANG = ROWS_PER_WAVE_Q < BLOCK_M

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

    assert NUM_WAVES % Q_HEADS_PER_WG == 0
    assert BLOCK_M % WAVE_ROW_GROUPS == 0
    assert ROWS_PER_WAVE_Q % N_TILE == 0
    assert BLOCK_KV % M_TILE == 0
    assert head_dim % 32 == 0 and head_dim >= 64

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    assert GQA_GROUP_SIZE % Q_HEADS_PER_WG == 0
    GQA_SLOTS = GQA_GROUP_SIZE // Q_HEADS_PER_WG  # sharer groups per kv head
    Q_GROUP = q_group  # q_tile run length made the fastest dispatch axis (1 = sharer first)

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
    LDS_SLOT = 2 * LDS_TILE  # K tile followed by V tile, in elements

    # Stage the next K/V tile through VGPRs instead of buffer_load_lds: D128's wider tile
    # needs it. Full-causal D64's kv prefix is long enough that the DMA path's lower
    # register pressure wins there instead; windowed D64's short kv loop gets its own
    # register-free mechanism below (WIN_RESIDENT), not this one.
    KV_REG_PF = ENABLE_DMA and HEAD_DIM == 128
    # Depth of that register pipeline (tiles ahead the global loads are issued): depth 1
    # wins -- a second live tile's registers cost more than the extra latency-hiding shadow
    # buys on a walk this short, and this held up when re-checked at five trips.
    KV_PF_D = 1
    # DMA_SEED: issue the FIRST tile's buffer_load_lds before the Q/dO/O burst so its HBM
    # latency lands under theirs (every Q/dO/O load is issued ahead of the delta reduce's
    # forced drain point) instead of adding a second serial round trip after them.
    DMA_SEED = ENABLE_DMA and window_left >= 0
    # Whole-window residency: with BLOCK_M == BLOCK_KV and a window that's a multiple of
    # it, the kv extent a work-group touches is exactly window_left/BLOCK_KV + 1 tiles at
    # compile-time offsets, so giving each its own LDS slot turns the walk into
    # straight-line code with no rendezvous, hand-over or loop-carried phi between tiles.
    WIN_RESIDENT = (
        ENABLE_DMA
        and not KV_REG_PF
        and window_left >= 0
        and BLOCK_M == BLOCK_KV
        and window_left % BLOCK_KV == 0
        and WAVE_ROW_GROUPS == 1
        and not varlen
        # Each resident tile gets its own LDS slot; a wide window (e.g. W=2048 with
        # BLOCK_KV=32 -> 65 tiles) would blow past the 65536-byte ds_read offset ceiling
        # asserted below. When it will not fit, fall back to the single-slot looped walk
        # (WIN_TILES=1), the same path W=2047 already takes.
        and (window_left // BLOCK_KV + 1) * LDS_SLOT * 2 <= 65536
    )
    WIN_TILES = (window_left // BLOCK_KV + 1) if WIN_RESIDENT else 1
    # s_setprio (raises this wave's issue priority over SIMD siblings) is a net loss on the
    # resident windowed walk -- a straight-line region with no rendezvous, so winning
    # arbitration just starves the sibling work-groups covering this wave's memory latency.
    DQ_SETPRIO = not WIN_RESIDENT
    LDS_TOTAL = WIN_TILES * LDS_SLOT
    # ds_read takes a 16-bit unsigned immediate offset, so the top of one work-group's
    # LDS must stay below 65536 bytes or every A-fragment family needs a live address
    # register (see the note on the tile read helpers).
    assert LDS_TOTAL * 2 <= 65536
    # An A/B flag has to be resolved HERE, in the builder's scope: the JIT cache key is
    # the launcher's source text plus the scalar values its closure captures, so a flag
    # read from a module global inside the traced kernel leaves the key unchanged and the
    # stale binary is served back -- an arm that silently does not switch.

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

        def _setprio(v):
            if const_expr(DQ_SETPRIO):
                rocdl.s_setprio(v)

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

        def _dot2_bf16(a_i32, b_i32, acc_f32):
            """gfx950 v_dot2_f32_bf16: acc + a.lo*b.lo + a.hi*b.hi, products in f32.
            No ROCDL op exists for the dot family, so it goes through inline asm."""
            return llvm.inline_asm(
                ir.F32Type.get(),
                [_raw(a_i32), _raw(b_i32), _raw(acc_f32)],
                "v_dot2_f32_bf16 $0, $1, $2, $3",
                "=v,v,v,v",
                has_side_effects=False,
            )

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

        def _q_head_of(_kvh, _grp):
            """q head this WAVE owns: the group's sharer base plus its own sharer slot."""
            if const_expr(Q_HEADS_PER_WG == 1):
                return _kvh * GQA_GROUP_SIZE + _grp
            # Wave-uniform so the head offset stays scalar: it feeds the Q/dO/O/LSE/dQ
            # addressing for the whole kernel, and a divergent copy would sit in a VGPR
            # across the entire kv walk.
            _sharer = fx.Index(
                rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave_id // fx.Index(WAVE_ROW_GROUPS)))
            )
            return _kvh * GQA_GROUP_SIZE + _grp * fx.Index(Q_HEADS_PER_WG) + _sharer

        if const_expr(NUM_HEADS_KV % NUM_XCD == 0):
            _xcd = block_id % fx.Index(NUM_XCD)
            _slot = block_id // fx.Index(NUM_XCD)
            # Sharer-innermost: the GQA sharer groups of one (batch, q_tile, kv-head) read
            # a byte-identical kv window, so they are the tightest reuse pair to keep
            # co-resident. Making q_tile the fastest axis instead -- consecutive q tiles
            # share BLOCK_M + W - BLOCK_M of their windows, a sliding rather than exact
            # reuse -- measures +0.45% median / -0.63% min, i.e. level.
            if const_expr(Q_GROUP > 1):
                # Group swizzle on the q_tile axis: a run of Q_GROUP consecutive q tiles
                # is the fastest axis, the GQA sharer groups the next one. Bijective for
                # any num_q_tiles because the tail run shrinks to what is left.
                _per_kv = num_q_tiles * fx.Index(GQA_SLOTS)
                _lin = _slot % _per_kv
                _run = _lin // fx.Index(Q_GROUP * GQA_SLOTS)
                _qbase = _run * fx.Index(Q_GROUP)
                _rem = num_q_tiles - _qbase
                _gcur = fx.Index(ArithValue(_rem < fx.Index(Q_GROUP)).select(_rem, fx.Index(Q_GROUP)))
                _r = _lin - _run * fx.Index(Q_GROUP * GQA_SLOTS)
                _q_in_group = _r // _gcur
                _qt_disp = _qbase + _r % _gcur
                _bkv = (_slot // _per_kv) * fx.Index(NUM_XCD) + _xcd
            else:
                _q_in_group = _slot % GQA_SLOTS
                _u = _slot // GQA_SLOTS
                _qt_disp = _u % num_q_tiles
                _bkv = (_u // num_q_tiles) * fx.Index(NUM_XCD) + _xcd
            kv_head_idx = _bkv % NUM_HEADS_KV
            batch_idx = _bkv // NUM_HEADS_KV
            q_head_idx = _q_head_of(kv_head_idx, _q_in_group)
        elif const_expr(GQA_GROUP_SIZE == 1):
            assert Q_HEADS_PER_WG == 1
            q_head_idx = block_id % NUM_HEADS_Q
            batch_q_tile_id = block_id // NUM_HEADS_Q
            kv_head_idx = q_head_idx
            _qt_disp = batch_q_tile_id % num_q_tiles
            batch_idx = batch_q_tile_id // num_q_tiles
        else:
            kv_head_idx = block_id % NUM_HEADS_KV
            _bid_rest = block_id // NUM_HEADS_KV
            _q_in_group = _bid_rest % GQA_SLOTS
            batch_q_tile_id = _bid_rest // GQA_SLOTS
            q_head_idx = _q_head_of(kv_head_idx, _q_in_group)
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
            # A None entry marks a value that is exactly zero at compile time (a patch
            # _patch_bounds proved all-masked): cvt_pk_bf16_f32 is inline asm and would
            # not fold, so emit the zero word directly.
            pairs = [
                fx.Int32(0)
                if f32_vals[j * 2] is None
                else fx.Int32(
                    _raw(rocdl.cvt_pk_bf16_f32(_raw(f32_vals[j * 2]), _raw(f32_vals[j * 2 + 1])))
                )
                for j in range_constexpr(4)
            ]
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

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
                across the k_tr peak pushes VGPRs past the occ-2 boundary. Hoisting the
                tile-invariant row/col part out of the loop is bitwise-neutral and even
                lands at vgpr 152, but measures +0.42% then -0.08% across two sessions,
                so the recompute is not what this loop pays for.
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

        # fx.Index is unsigned: guard the subtract (W may exceed q+off) to avoid
        # underflow-to-huge. _wlo skips fully-out-of-window kv tiles; the first in-window
        # kv is q+off-W (W+1-key window), so start from there.
        if const_expr(window_left >= 0):
            _wlo = fx.Index(
                ArithValue(q_start + causal_offset >= fx.Index(window_left)).select(
                    q_start + causal_offset - fx.Index(window_left), fx.Index(0)
                )
            )
            _wlo = (_wlo // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
        else:
            _wlo = fx.Index(0)
        # The first kv tiles' loads are issued before the Q/dO/O loads so their HBM latency
        # overlaps instead of adding a second serial round trip; only the issue moves up,
        # the wait/publish barrier stay at the loop head, and buffer_load_lds is VGPR-free.
        # Tile j of the resident window as a signed row index: tiles clipped by the
        # sequence start are staged from row 0 and neutralised via a -inf softmax seed
        # (same zero contribution as a skipped tile, without a branch through the carry).
        _qoff_i32 = fx.Int32(q_start) + causal_off_i32

        def _win_tile_i32(j):
            return _qoff_i32 + fx.Int32(j * BLOCK_KV - window_left)

        def _win_dma_burst():
            for _j in range_constexpr(WIN_TILES):
                _tj = _win_tile_i32(_j)
                _tj = fx.Index(ArithValue(_tj > fx.Int32(0)).select(_tj, fx.Int32(0)))
                _sb = _j * LDS_SLOT * 2
                coop_dma_tile(k_rsrc, lds_base_idx + fx.Index(_sb), _tj)
                coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(_sb + LDS_V_BASE * 2), _tj)

        _pf_carry = []
        # The burst goes ahead of the Q/dO/O loads. vmcnt retires in issue order, so the
        # O.dO reduce then drains the transfers too; issuing the burst after the O loads
        # instead, so the reduce only waits for its own operands, measures -0.91% median.
        if const_expr(WIN_RESIDENT):
            _win_dma_burst()
        elif const_expr(DMA_SEED and not WIN_RESIDENT):
            coop_dma_tile(k_rsrc, lds_base_idx, _wlo)
            coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), _wlo)
        if const_expr(KV_PF_D == 2):
            _pf_carry = coop_load_tile_regs(_wlo + fx.Index(BLOCK_KV))
        # ---- Owned Q,dO B-operand packs: B[k=D][n=q], n=lane16, k=kg*8+s. Per wave
        # QT q 16-tiles x K_STEPS_QK D-steps; q_b_packs[qt][ks] is a v8 bf16. ----
        if const_expr(WAVE_ROW_GROUPS == NUM_WAVES):
            q_row_wave = q_start + wave_id * ROWS_PER_WAVE_Q
        elif const_expr(WAVE_ROW_GROUPS == 1):
            q_row_wave = q_start  # every wave owns the whole tile, only its head differs
        else:
            q_row_wave = q_start + (wave_id % fx.Index(WAVE_ROW_GROUPS)) * ROWS_PER_WAVE_Q

        def q_row_of(qt):
            return q_row_wave + fx.Index(qt * N_TILE) + lane16

        q_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        do_b_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        o_packs = [[None] * K_STEPS_QK for _ in range_constexpr(QT)]
        d_parts = [fx.Float32(0.0) for _ in range_constexpr(QT)]

        def _q_col_of(ks):
            return fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K

        for qt in range_constexpr(QT):
            _qr = q_row_of(qt)
            for ks in range_constexpr(K_STEPS_QK):
                q_col = _q_col_of(ks)
                q_b_packs[qt][ks] = buffer_ops.buffer_load(
                    q_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
                do_b_packs[qt][ks] = buffer_ops.buffer_load(
                    do_rsrc, global_idx_q(_qr, q_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                )
        if const_expr(fuse_delta):
            # Every O load is issued before the first is consumed (interleaving the O.dO
            # reduce with the loads instead blocks the scheduler from hoisting them, adding
            # a serial round trip); holding all O packs costs VGPRs but not waves/SIMD.
            def _od_acc(qt, ks):
                # v_dot2_f32_bf16 consumes packed bf16 operands and accumulates in f32,
                # collapsing this lane's O.dO widen/multiply/add chain into 4 instructions.
                # Still exact hardware fp32 products in a fixed order, so DELTA stays
                # run-to-run deterministic.
                _o = Vec(o_packs[qt][ks]).bitcast(fx.Int32)
                _d = Vec(do_b_packs[qt][ks]).bitcast(fx.Int32)
                for i in range_constexpr(MFMA_LANE_K // 2):
                    d_parts[qt] = fx.Float32(_dot2_bf16(_o[i], _d[i], d_parts[qt]))

            for qt in range_constexpr(QT):
                for ks in range_constexpr(K_STEPS_QK):
                    o_packs[qt][ks] = buffer_ops.buffer_load(
                        o_rsrc,
                        global_idx_q(q_row_of(qt), _q_col_of(ks)),
                        vec_width=MFMA_LANE_K,
                        dtype=elem_dtype,
                    )
        # ---- Owned LSE/-delta_id per q (one scalar per qt, q = qt*16 + lane16). ----
        lse_owned = []
        delta_owned = []

        def _issue_lse():
            for qt in range_constexpr(QT):
                _lse_elem = _ld_delta_elem(q_row_of(qt))
                lse_owned.append(
                    fx.Float32(
                        buffer_ops.buffer_load(lse_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32)
                    )
                )
                if const_expr(not fuse_delta):
                    delta_owned.append(
                        fx.Float32(
                            buffer_ops.buffer_load(
                                delta_in_rsrc, _lse_elem, vec_width=1, dtype=fx.Float32
                            )
                        )
                    )

        if const_expr(fuse_delta):
            for qt in range_constexpr(QT):
                for ks in range_constexpr(K_STEPS_QK):
                    _od_acc(qt, ks)

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

        # Issuing these with the Q/dO/O burst still in flight -- ahead of the O.dO reduce
        # that drains vmcnt -- so the work-group pays one HBM round trip instead of two
        # measures +-0 (median +0.04%, min -0.62%): the reduce is already covered, and
        # the two extra live scalars per qt sit across the whole kv walk.
        _issue_lse()
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

        def _gemm1_mfma(a, b_packs, inits_q=None, kvts=None, dead=()):
            """S[kvt][qt] (v4f32) = a[kvt] @ B(owned Q/dO)[qt]^T over D, given
            already-loaded A tiles `a` (see _gemm1_load). inits_q[qt] optionally
            pre-loads the accumulator (folds -delta_id into the dP GEMM for free).
            `dead` drops the (kvt, qt) patches the caller proved all-masked."""
            if kvts is None:
                kvts = list(a.keys())
            out = [[None] * QT for _ in range_constexpr(KVT)]
            # Emission order is ks-innermost on purpose: the ks-outer form that wins in
            # dkdv (see g1_ks_outer) costs 1.5% here, because dq runs two waves per SIMD
            # and the sibling wave already covers an MFMA's result latency, so the wider
            # live accumulator set buys nothing.
            for kvt in kvts:
                for qt in range_constexpr(QT):
                    if const_expr((kvt, qt) in dead):
                        continue
                    acc = c_zero_v4f32 if inits_q is None else inits_q[qt]
                    for ks in range_constexpr(K_STEPS_QK):
                        acc = mfma_acc(a[kvt][ks], b_packs[qt][ks], acc)
                    out[kvt][qt] = acc
            return out

        def _gemm1(a_base, b_packs, inits_q=None, kvts=None, dead=()):
            """S[kvt][qt] (v4f32) = A(K/V)[kvt] @ B(owned Q/dO)[qt]^T over D. A is
            loaded once per (kvt,ks) and reused across qt. inits_q[qt] optionally
            pre-loads the accumulator (folds -delta_id into the dP GEMM for free).
            kvts restricts to a subset of kv 16-tiles (halves the live s/dp transient
            peak when the caller interleaves exp2/pack per kv-half)."""
            if kvts is None:
                kvts = list(range_constexpr(KVT))
            a = _gemm1_load(a_base, kvts)
            return _gemm1_mfma(a, b_packs, inits_q, kvts, dead)

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
        def _seed4(x):
            """MFMA accumulator seed: one per-q scalar broadcast over the 4 kv slots."""
            return Vec.from_elements([x], fx.Float32).broadcast_to(4).ir_value()

        # Rebuilding these two seeds at each tile instead of carrying them frees enough
        # registers to reach four work-groups per CU spill-free; forcing the same
        # occupancy tier via waves_per_eu without freeing registers first spills badly.
        delta_inits = [_seed4(delta_owned[qt]) for qt in range_constexpr(QT)]
        if const_expr(fold_lse):
            lse_inits = [_seed4(lse_owned[qt]) for qt in range_constexpr(QT)]
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

        def _kv_body(
            kv_start,
            inner,
            apply_mask,
            win_mask=True,
            skip=None,
            causal_sel=False,
            dead=(),
            in_range=(),
            slot=0,
            s_inits=None,
        ):
            # `slot` picks this tile's LDS pair when the whole window is resident; the
            # offsets stay compile-time so ds_read keeps its immediate form. `s_inits`
            # overrides the GEMM1a accumulator seed for the clipped tiles described above.
            _lds_k = fx.Index(slot * LDS_SLOT)
            _lds_v = fx.Index(slot * LDS_SLOT + LDS_V_BASE)
            # `dead` / `in_range` are the compile-time (kvt, qt) patch sets from
            # _patch_bounds: `dead` patches are always outside the window/causal edge (P
            # is exactly zero, so GEMM1/exp/mask/select all drop); `in_range` ones are
            # always inside (keep arithmetic, drop the select). Both are bitwise-neutral.
            # `skip` is a wave-uniform predicate for "this wave sees at least one kv row of
            # this tile"; excluded waves have P == 0 over the whole tile, so branching
            # around them is bitwise-neutral. The hand-over stays outside the branch
            # since it's work-group collective.
            sb_bulk = not apply_mask  # exps only exist on these paths
            # Tiles KV_PF_D ahead are issued here; at depth 2 the tile due for LDS at the
            # end of this body was issued by the previous one and arrives as a carry.
            # A forced-L1-hit re-staging probe capped the whole "cut K/V requests" family
            # (head-outer restructure, wider chunks, deeper prefetch) at a small gain, so
            # that's not where this kernel's time goes.
            _pf = (
                coop_load_tile_regs(kv_start + fx.Index(KV_PF_D * BLOCK_KV))
                if const_expr(KV_REG_PF)
                else None
            )
            _pf_due = _pf
            if const_expr(KV_PF_D == 2):
                _pf_due = inner[DT * QT :]
                inner = inner[: DT * QT]
            kv_start_i32 = fx.Int32(kv_start)

            def _core(carry, emit_dma):
                A_cur = [
                    [carry[dt * QT + qt] for qt in range_constexpr(QT)]
                    for dt in range_constexpr(DT)
                ]
                # C[kvt][qt]: 4 f32 at kv=kvt*16+kg*4+t, q=qt*16+lane16. C = P~*(dP-delta_id)
                # feeds GEMM2.
                C = [[None] * QT for _ in range_constexpr(KVT)]
                c_pack = [[None] * QT for _ in range_constexpr(PV_K_STEPS)]
                # Split GEMM1a/1b + exp2/C + pack per kv-half (pks = the 2 kvt of one GEMM2 K=32 step):
                # only 2 kvt of s/dP are live at a time, halving the transient VGPR peak without touching
                # the batched GEMM2 below. The next half's K ds_read is issued right after this half's
                # GEMM1 MFMAs so its latency hides in the VALU-heavy exp2/C/pack shadow (V stays in-half).
                k_a_by_half = {0: _gemm1_load(_lds_k, [0, 1])}
                for pks in range_constexpr(PV_K_STEPS):
                    ka, kb = 2 * pks, 2 * pks + 1
                    half = [ka, kb]
                    # GEMM1a S[kv,q]=K@Q^T ; GEMM1b dP[kv,q]=V@dO^T (acc init=-delta_id) for
                    # this kv-half. s_setprio(1) raises MFMA priority over ds_read/VALU;
                    # dropped to 0 for the exp2/pack/reduce VALU section so it is not starved.
                    _setprio(1)
                    rocdl.iglp_opt(0)
                    if const_expr(fold_lse and not apply_mask):
                        _s_inits = lse_inits if s_inits is None else s_inits
                    else:
                        _s_inits = None
                    s_tiles = _gemm1_mfma(
                        k_a_by_half[pks], q_b_packs, inits_q=_s_inits, kvts=half, dead=dead
                    )
                    if const_expr(sb_bulk):
                        rocdl.sched_barrier(SCHED_TRANS)
                    dp_tiles = _gemm1(
                        _lds_v, do_b_packs, delta_inits, kvts=half, dead=dead
                    )
                    _setprio(0)

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
                        _setprio(1)
                        k_a_by_half[pks + 1] = _gemm1_load(_lds_k, [nka])
                        _setprio(0)

                        def _next_kb_load():  # noqa: B023
                            _setprio(1)
                            k_a_by_half[pks + 1].update(_gemm1_load(_lds_k, [nkb]))  # noqa: B023
                            _setprio(0)

                    if const_expr(not apply_mask):
                        # Vectorized bulk (below-diagonal): exp2/C/reduce as packed v4 ops
                        # (v_pk_*), mirroring the 32x32 kernel's v8 path. exp2 and C=P*dP are
                        # strictly elementwise so C is bit-identical to the scalar branch;
                        # R re-associated in a fixed order -> deterministic (det gate holds).
                        for qt in range_constexpr(QT):
                            # QT == 1 (BLOCK_M == 64) has no qt=1 to hang the load on, so it
                            # issues at qt=0; the k_a_by_half entry is not optional.
                            if const_expr(qt == min(1, QT - 1) and _next_kb_load is not None):
                                _next_kb_load()
                            if const_expr(not fold_lse):
                                lse_v4 = Vec.from_elements([lse_owned[qt]], fx.Float32).broadcast_to(4)
                            for kvt in half:
                                if const_expr((kvt, qt) in dead):
                                    C[kvt][qt] = [None] * 4
                                    continue
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
                                _sel_win = const_expr(
                                    window_left >= 0 and win_mask and (kvt, qt) not in in_range
                                )
                                _sel_causal = const_expr(causal_sel and (kvt, qt) not in in_range)
                                if const_expr(_sel_win or _sel_causal):
                                    # keep kv >= q+off-W (W+1 keys, matching the fwd SWA
                                    # edge); causal_sel adds kv <= q+off so diagonal tiles
                                    # can run this vectorised body (zero P after exp2, same
                                    # result as feeding -inf) instead of the scalar one.
                                    _kvb = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4)
                                    _qc = q_slot_i32[qt] + causal_off_i32
                                    _thr = _qc - fx.Int32(window_left)
                                    _keep = []
                                    for t in range_constexpr(4):
                                        _kvt4 = _kvb + fx.Int32(t)
                                        _k = None
                                        if const_expr(_sel_win):
                                            _k = ArithValue(_kvt4 >= _thr)
                                        if const_expr(_sel_causal):
                                            _c = ArithValue(_kvt4 <= _qc)
                                            _k = _c if _k is None else ArithValue(
                                                arith.andi(_raw(_k), _raw(_c))
                                            )
                                        _keep.append(_k.select(Vec(p4)[t], c_zero_f))
                                    p4 = Vec.from_elements(_keep, fx.Float32)
                                c4 = p4 * Vec(dp_tiles[kvt][qt])
                                C[kvt][qt] = [c4[t] for t in range_constexpr(4)]
                    else:
                        for qt in range_constexpr(QT):
                            if const_expr(qt == min(1, QT - 1) and _next_kb_load is not None):
                                _next_kb_load()
                            lse_q = lse_owned[qt]
                            for kvt in half:
                                if const_expr((kvt, qt) in dead):
                                    C[kvt][qt] = [None] * 4
                                    continue
                                dp_v = dp_tiles[kvt][qt]
                                s_v = s_tiles[kvt][qt]
                                c_vals = []
                                for t in range_constexpr(4):
                                    kv_slot = kv_start_i32 + fx.Int32(kvt * M_TILE + kg * 4 + t)
                                    _up = ArithValue(kv_slot > q_slot_i32[qt] + causal_off_i32)
                                    if const_expr(window_left >= 0 and win_mask):
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

                # GEMM2 A^T[D,q] += K_tr @ C, processed in interleaved dt pairs so a dependent
                # MFMA is separated by 3 independent ones (covers the 16x16x32 operand
                # latency); next pair's k_tr is prefetched during the current one.
                kts = [
                    [_read_tr(_lds_k, d, pks) for pks in range_constexpr(PV_K_STEPS)]
                    for d in range_constexpr(min(G2A, DT))
                ]
                _setprio(2)
                for d0 in range_constexpr(0, DT, 2):
                    if const_expr(d0 + G2A < DT):
                        for _dn in range_constexpr(d0 + G2A, d0 + G2A + 2):
                            kts.append(
                                [_read_tr(_lds_k, _dn, pks) for pks in range_constexpr(PV_K_STEPS)]
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
                    # LDS hand-over (DMA path): this pair issues the tile's last k_tr read, so
                    # the next tile DMAs into the SAME buffer while the remaining register-only
                    # GEMM2 pair covers the transfer -- no double buffer needed since in-flight
                    # writes never contend with LDS reads once the last read has issued.
                    if const_expr(
                        emit_dma and d0 == max(0, DT - 4) and not KV_REG_PF and not WIN_RESIDENT
                    ):
                        gpu.barrier()
                        _issue_dma(kv_start)  # noqa: B023
                _setprio(0)
                return [
                    A_cur[dt][qt] for dt in range_constexpr(DT) for qt in range_constexpr(QT)
                ]

            if const_expr(skip is None):
                out = _core(inner, True)
            else:
                out = _if_wave(
                    skip,
                    [inner[i] for i in range_constexpr(DT * QT)],
                    lambda: _core(inner, False),
                    lambda: None,
                )
                # The skipped waves never read this tile, so the WAR fence and the next
                # tile's DMA -- work-group collective -- move behind the whole branch
                # instead of riding GEMM2's last pair.
                if const_expr(not KV_REG_PF and not WIN_RESIDENT):
                    gpu.barrier()
                    _issue_dma(kv_start)
            if const_expr(KV_REG_PF):
                # This pair of barriers is the whole per-tile rendezvous and prices at only
                # 0.67% of the kernel (WAR 0.62%, publish 0.10%), so an LDS double buffer --
                # which would remove the WAR half at 2x LDS plus a x2 loop unroll to keep the
                # slot bases compile-time constants -- is not worth its register cost.
                gpu.barrier()  # WAR: fence this tile's LDS reads before the rewrite
                coop_store_tile_lds(_pf_due)
            elif const_expr(WIN_RESIDENT):
                # Nothing was written to LDS during the body, so there is no WAR hazard
                # and no publish to fence: the tiles are all live from the single
                # prologue barrier to the end of the kernel.
                pass
            elif const_expr(ENABLE_DMA):
                rocdl.s_waitcnt(0)
            if const_expr(not WIN_RESIDENT):
                gpu.barrier()

            return out + _pf if const_expr(KV_PF_D == 2) else out

        # ---- 16-row (kv, q) patch bounds -----------------------------------------------
        # A BLOCK_KV x BLOCK_M tile is walked as KVT x QT patches of M_TILE x N_TILE rows,
        # so band edges resolve per patch instead of per tile: a patch entirely outside the
        # attended band (-window_left <= kv-(q+off) <= 0) has P == 0, so its GEMM1/exp2/mask
        # select/C pack all drop; one entirely inside just drops the select. Both are exact.
        _TRIM_ROWS = const_expr(
            window_left >= 0 and BLOCK_M <= BLOCK_KV and WAVE_ROW_GROUPS == 1
        )

        def _patch_bounds(kv_off):
            """(dead, in_range) patch sets for a tile at q_start + causal_offset + kv_off."""
            dead, inr = set(), set()
            if const_expr(_TRIM_ROWS):
                for kvt in range_constexpr(KVT):
                    klo, khi = kv_off + kvt * M_TILE, kv_off + kvt * M_TILE + M_TILE - 1
                    for qt in range_constexpr(QT):
                        qlo, qhi = qt * N_TILE, qt * N_TILE + N_TILE - 1
                        if const_expr(klo > qhi or khi < qlo - window_left):
                            dead.add((kvt, qt))
                        elif const_expr(khi <= qlo and klo >= qhi - window_left):
                            inr.add((kvt, qt))
            return dead, inr

        # The window-edge loop below runs at most one trip, at kv_off == -window_left: the
        # tile grid is anchored on q_start + causal_offset (a BLOCK_KV multiple), so when
        # q+off < W the whole edge range collapses onto _wlo and the loop doesn't run.
        _EDGE_ANCHORED = const_expr(
            _TRIM_ROWS and BLOCK_M == BLOCK_KV and window_left % BLOCK_KV == 0
        )
        _edge_dead, _edge_inr = (
            _patch_bounds(-window_left) if const_expr(_EDGE_ANCHORED) else (set(), set())
        )
        _diag_dead, _diag_inr = _patch_bounds(0)

        # Split the causal kv-loop: [0, q_start) below the diagonal (no mask),
        # [q_start, kv_upper) straddles it (mask).
        _carry = A_accs + _pf_carry
        loop_results = _carry
        if const_expr(window_left >= 0):
            # First kv tile the window predicate is a tautology on: every walked q row
            # (up to q_start+BLOCK_M-1) sees every kv >= _wsafe, so [_wsafe, diagonal)
            # runs a window-mask-free body. Ceil onto the tile grid _wlo already sits
            # on and clamp into [_wlo, diagonal] so the two loops stay contiguous.
            _whi = q_start + causal_offset + fx.Index(BLOCK_M - 1)
            _wsafe = fx.Index(
                ArithValue(_whi >= fx.Index(window_left)).select(
                    _whi - fx.Index(window_left), fx.Index(0)
                )
            )
            _wsafe = ((_wsafe + fx.Index(BLOCK_KV - 1)) // fx.Index(BLOCK_KV)) * fx.Index(BLOCK_KV)
            _wsafe = fx.Index(ArithValue(_wsafe > _wlo).select(_wsafe, _wlo))
            _wsafe = fx.Index(
                ArithValue(_wsafe < q_start + causal_offset).select(
                    _wsafe, q_start + causal_offset
                )
            )
        # Prologue for the software-pipelined body: it expects its own tile already in
        # LDS and leaves the next one there. _wlo is the first kv tile of whichever of
        # the two loops below runs first (they are contiguous).
        if const_expr(ENABLE_DMA):
            if const_expr(not DMA_SEED):
                coop_dma_tile(k_rsrc, lds_base_idx, _wlo)
                coop_dma_tile(v_rsrc, lds_base_idx + fx.Index(LDS_V_BASE * 2), _wlo)
            rocdl.s_waitcnt(0)
        gpu.barrier()
        # Collapsing the three kv bodies below (window edge / no edge / causal edge) into
        # one that applies both edges everywhere is bitwise-neutral but regresses: every
        # tile then pays both a window compare and a causal select where it used to pay one.
        if const_expr(WIN_RESIDENT):
            # Straight-line walk over the resident window: tile j sits at a compile-time
            # offset from q_start + causal_offset, so its patch classification is
            # compile-time too (j == 0 is the window edge, the last is the diagonal,
            # everything between needs neither compare nor select).
            for _j in range_constexpr(WIN_TILES):
                _kvj = _win_tile_i32(_j)
                if const_expr(_j == 0):
                    _flags = dict(
                        win_mask=True, dead=_edge_dead, in_range=_edge_inr
                    )
                elif const_expr(_j == WIN_TILES - 1):
                    _flags = dict(
                        win_mask=window_left < BLOCK_M - 1,
                        causal_sel=True,
                        dead=_diag_dead,
                        in_range=_diag_inr,
                    )
                else:
                    _flags = dict(win_mask=False)
                # Only the tiles the sequence start can clip need the -inf seed; the
                # diagonal tile is always in range. One scalar select per owned q 16-tile.
                _si = None
                if const_expr(_j < WIN_TILES - 1):
                    _keep = ArithValue(_kvj >= fx.Int32(0))
                    _si = [
                        Vec.from_elements(
                            [fx.Float32(_keep.select(lse_owned[qt], c_neg_inf))], fx.Float32
                        )
                        .broadcast_to(4)
                        .ir_value()
                        for qt in range_constexpr(QT)
                    ]
                loop_results = _kv_body(
                    _kvj, loop_results, False, slot=_j, s_inits=_si, **_flags
                )
        elif const_expr(window_left >= 0):
            # A wave owns only ROWS_PER_WAVE_Q of the group's BLOCK_M rows, so the group's
            # kv extent overhangs each wave's own window at both ends; those tiles are
            # all-masked (P == 0) for that wave and skipped behind a wave-uniform branch.
            # Only the two edge loops need the test -- the bulk is in-window by construction.
            _qw_i32 = fx.Int32(q_row_wave) + causal_off_i32
            for kv_start, inner in range(_wlo, _wsafe, BLOCK_KV, init=_carry):
                _live = None
                if const_expr(WAVE_OVERHANG):
                    _live = ArithValue(
                        fx.Int32(kv_start) + fx.Int32(BLOCK_KV - 1 + window_left) >= _qw_i32
                    )
                loop_results = yield _kv_body(
                    kv_start, inner, False, True, _live, False, _edge_dead, _edge_inr
                )
            for kv_start, inner in range(_wsafe, q_start + causal_offset, BLOCK_KV, init=loop_results):
                loop_results = yield _kv_body(kv_start, inner, False, False)
        else:
            for kv_start, inner in range(_wlo, q_start + causal_offset, BLOCK_KV, init=_carry):
                loop_results = yield _kv_body(kv_start, inner, False)
        # The diagonal block only needs the window test when the window is narrower than
        # the rows this group walks; at W >= BLOCK_M-1 no kv >= q_start can fall below
        # any walked row's q+off-W.
        _diag_win = const_expr(window_left < BLOCK_M - 1)
        # Under a window the diagonal runs the vectorised body with the causal edge as a
        # select instead of the scalar masked body: the window already forces a per-element
        # test here, so folding the causal edge into it costs one more compare and buys the
        # whole diagonal the bulk path's packing.
        _cs = const_expr(window_left >= 0)
        if const_expr(not WIN_RESIDENT):
            # The resident walk above already ran this tile as its last slot.
            for kv_start, inner in range(
                q_start + causal_offset, kv_upper, BLOCK_KV, init=loop_results
            ):
                _live = None
                if const_expr(_cs and WAVE_OVERHANG):
                    # Above-diagonal counterpart of the loop above: a tile starting past
                    # this wave's last owned q row is causally dead for it.
                    _live = ArithValue(
                        fx.Int32(kv_start) <= _qw_i32 + fx.Int32(ROWS_PER_WAVE_Q - 1)
                    )
                loop_results = yield _kv_body(
                    kv_start, inner, not _cs, _diag_win, _live, _cs, _diag_dead, _diag_inr
                )

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
        grid_x = bs_idx * num_q_tiles * (NUM_HEADS_Q // Q_HEADS_PER_WG)

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


def _qsplit_for(Sq, window_left=-1):
    # q_split fans the dK/dV KV-owner WGs across the CU grid; the optimum rises with
    # Sq before split-reduction overhead dominates. Re-swept once the fused path started
    # dispatching one batch at a time (fewer work-groups per dispatch, so list-scheduling
    # slack matters more): 4 still wins; wider only adds dk/dv slots for the slot reduce
    # to fold.
    if window_left >= 0:
        # Sliding window: a band's whole q range is only BLOCK_KV+W rows, so splitting it
        # hands the SAME work to q_split times as many work-groups and multiplies the
        # dk/dv workspace. One slot avoids both the redundant prologues and the slot reduce.
        return 1
    if Sq <= 8192:
        return 4
    return 3


def _blockkv_for(Skv, head_dim=64, window_left=-1):
    # dkdv is KV-outer, so Skv (not Sq) sets its grid: a short Skv needs BLOCK_KV=64
    # to fill the CU array, a long one wants a wider tile to amortise per-tile cost;
    # keying this on Sq instead costs badly on rectangular shapes. D128's wider tile
    # (192) needs the full register file (one wave per SIMD, see `_get_bwd`), so it only
    # pays once Skv is long enough that a sparse grid would not follow from picking it.
    # NOTE: the bench's SNR/det gate never exercises this tier -- re-verify dQ/dK/dV SNR
    # and determinism at Skv >= 8192 directly whenever these thresholds move.
    if head_dim >= 128:
        return 192 if Skv >= 8192 else 64
    if window_left >= 0:
        # Narrowing the band raises the attended fraction BLOCK_KV/(BLOCK_KV+W) and kills
        # more fully-outside-window tiles, but alone it costs more in Q/dO re-read than it
        # saves in MFMA. It only pays together with Q_PAIR, which recovers that MFMA
        # saving at BLOCK_KV=64 without the extra staging (see Q_PAIR).
        return 64
    return 64 if Skv <= 2048 else 128


def _fuse_blockkv_for(Skv):
    """kv band for the fused path. The dQ split-K traffic is (Skv/BLOCK_KV)/2 * |dQ| in
    each direction, so unlike the split path -- where BLOCK_KV only trades grid width
    against per-tile cost -- the fused path pays for a narrow band in DRAM bytes and
    wants the widest band the register file takes.
    """
    # 256 only fits because the K/V B-operands live in LDS (see `_kv_lds_idx`): it
    # doubles the dK/dV accumulators, and the packs it displaces are exactly what pays
    # for them (with the packs still in registers this spilled catastrophically). 512
    # would halve the band count again, but its dK/dV accumulators alone exceed even the
    # four-wave body's whole register file, so the band stays 256.
    if Skv >= 4096:
        return 256
    # Skv==1024 takes 128 (8 bands), not 64 (16 bands): on the small square the reduce's
    # (Skv/BLOCK_KV)/2*|dQ| DRAM traffic is what makes the fused path lose, and halving the
    # band count there recovers it (64/1024/1024 B=4 with pipe off: bkv=64 0.217ms ->
    # bkv=128 0.168ms; bkv=256 0.190ms as its wider tile thins the grid). Skv<1024 is not
    # bench-covered and keeps 64.
    return 64 if Skv < 1024 else 128


def _dq_block_kv(Sq, window_left=-1):
    """dq's kv tile, swept with dkdv running before it (L2 contention moves the
    optimum). Only multiples of 32 are valid -- other sizes silently produce the
    wrong dQ, so keep to the checked set {32, 64, 96, 192}.
    """
    if window_left >= 0:
        # The window bound already trims the kv loop to ceil((BLOCK_M+W)/BLOCK_KV) tiles,
        # so a narrow tile costs no extra area here (BLOCK_M and W are both multiples of
        # 32) and buys scheduling/register slack. 96 and 192 compute the wrong dQ under a
        # window (SNR ~6 dB) -- keep to {32, 64}.
        return 32
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

    Its D axis is PERMUTED, and both writer and reader must agree: within each aligned
    32-element run, element ``dperm`` holds real D index
    ``d = (dperm & ~31) | ((dperm & 24) >> 1) | ((dperm & 4) << 2) | (dperm & 3)``,
    i.e. bit 4 of D sits at bit 2 of the stored position. That is exactly the order the
    dQ^T MFMA C-layout hands a pair of D-tiles to a lane, so the fused kernel's partial
    store covers a full 64 B of a q row per instruction instead of 32 B (see the store in
    _gemm3_tiles), and the reduce below un-permutes on its own store side. Measured with
    TCP_TCC counters: this roughly halves TCP->TCC write requests at unchanged DRAM
    traffic (byte-identical reads/EA-writes), which shows up as a clock-leg gain on a
    kernel already close to the power cap rather than a per-cycle efficiency change.

    One slot per kv band, so a band's contribution to a q row is written by exactly one
    work-group (no atomics, bitwise-reproducible). Cached, since the full workspace is
    far too large to reallocate per call. The causal trim means only ~half of it is ever
    touched, and that read is bandwidth-bound, so it can only be made cheaper in BYTES,
    or hidden. Folding band pairs does NOT remove bytes: the accumulating band still has
    to read its partner's slot, just trading dqred reads for dkdv reads (net loss).

    Folding the pair IN REGISTERS instead -- one work-group owning two adjacent bands and
    writing a q row's slot once -- does halve the slots and both request counts, but it is
    priced by the dK/dV accumulators, not by dQ, and either loop ordering needs them live
    twice over. That extra live range does not fit: this body is already register-full
    (see the BLOCK_KV=512 spill case, where overflowing the file spills the fragment and
    pack chains with the accumulators), and the whole discretionary resident set here is
    far smaller than what a second dK/dV set would need. Halving these request counts
    therefore needs a fold whose unit is NOT a second dK/dV accumulator set -- e.g. two
    work-groups on the same CU trading partials through LDS, or a wider Q_SPLIT so a slot
    covers more q rows per band.

    Hiding it is what pays, and it is a REGISTER question, not a scheduling one. An
    eight-wave fused work-group fills its SIMDs solid, so no dqred wave can land beside
    it and a dependency-free two-stream arm barely overlaps at all. The four-wave
    geometry (see _fused_pipelined) leaves enough of the 512-dword pool free for the
    reduce's waves to co-reside per SIMD, and that overlap is where the real win is.

    So the fused body's register allocation is priced by what the SIBLING kernel needs,
    not by this kernel's own spill, and the budget is negotiable from BOTH sides: a body
    change that looks like a pure ISA improvement (fewer spills) can still lose if it
    pushes the allocation past the point where two reduce waves fit per SIMD, and a leaner
    reduce (fewer per-thread accumulators, see build_flash_attn_bwd_dqred_module) can buy
    back headroom the body spent elsewhere. Rule of thumb before spending a register on
    either kernel: check whether 512 - waves_per_simd(body)*alloc(body) still covers
    n*alloc(dqred) for the co-resident wave count you want -- dump both kernels' VGPR/AGPR
    counts first, since "spill == 0" alone no longer proves an arm has passed once two
    kernels share the pool. The permuted store above costs a modest AGPR
    increase on the fused body and still wins outright at the resulting allocation; buying
    a second reduce wave back after that is not worth it here since no available register
    donor is large enough to clear the threshold without a body-side regression.
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
    if n_slots == 1:
        # Nothing to fold: the single slot IS the result (one writer per element, so this
        # is bitwise what the reduce would have produced). Skips a full-workspace
        # DRAM round trip that the reduce cannot hit in L2.
        return ws_dk.reshape(-1), ws_dv.reshape(-1)
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


def _reduce_dq_partials(
    ws, dq, block_kv, num_heads, head_dim, scale, stream, bat_lo=0, n_bat=None, qsp=(1, 0, None),
    causal_offset=0,
):
    """dQ[q] = scale * Sum_{b : b*BLOCK_KV <= q} ws[b][q], in ascending band order.

    A kv band only writes the q rows that causally see it, so the bands ABOVE q's own
    band hold stale data and are skipped -- which is also what keeps the traffic at the
    causal half. Fixed band order and fp32 accumulation -> bitwise deterministic.

    ``scale`` is 1/log2e, not sm_scale: the fused kernel's fifth GEMM contracts against
    the LDS K tile, which is staged already prescaled by sm*log2e for GEMM1a.

    Band count is what this costs: halving the bands read (diagnostic-only, wrong dQ)
    measurably speeds up the whole backward, so folding band pairs into one workspace
    slot is worth pursuing on the read side alone, before counting the matching halving
    of the kernel's partial-store traffic.
    """
    _, B, Sq, _ = ws.shape
    # rows_per_wg(=2)*Hq*D must tile the reduce's block256*vec8*uc chunk. uc=2 (=4096
    # chunk) is the tuned Hq=128 footprint; keep it when it tiles, else fall to the
    # smallest tiling uc (Hq=48 -> 6144, uc=1 -> 2048 chunk). Any tiling uc yields
    # bitwise-identical dQ (see build_flash_attn_bwd_dqred_module); it only trades the
    # co-resident wave's register footprint (re-sweep per shape when perf-tuning).
    rpw_hd = 2 * num_heads * head_dim
    uc = 2 if rpw_hd % 4096 == 0 else next(u for u in (1, 3) if rpw_hd % (2048 * u) == 0)
    key = (num_heads, head_dim, B, Sq, block_kv, scale, bat_lo, n_bat, qsp, causal_offset)
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
            block=256,
            rows_per_wg=2,
            uc=uc,
            bat_lo=bat_lo,
            n_bat=n_bat,
            q_split=qsp[0],
            qsp_lo=qsp[1],
            n_qsp=qsp[2],
            causal_offset=causal_offset,
        )
        _DQRED_CACHE[key] = launcher
    # Pass ONE band slice: the descriptor is rebased per band with a 64-bit offset, and
    # the whole workspace overflows a flat memref's i32 element count.
    launcher(ws[0].reshape(-1), dq.reshape(-1), stream)


# Overlap the dQ reduce with the fused kernel by running the backward in per-batch chunks
# (see _fused_pipelined). Off keeps the single whole-batch dispatch pair, which is what
# the eight-wave geometry had to use since it leaves no room for a co-resident reduce
# wave; the pipeline is only worth its chunking overhead at the four-wave geometry,
# where the fused work-group leaves enough registers for a reduce work-group to land.
_DQ_PIPE = True
_SIDE_STREAM: dict = {}
_PIPE_EVENTS: dict = {}


def _pipe_chunks(B, q_split, block_kv):
    """Pipeline stages as (batch, qsp_lo, n_qsp), in dispatch order.

    Batch alone leaves the LAST batch's reduce with nothing behind it to hide under.
    Cutting it finer is not free: each chunk is its own dispatch, so its stragglers drain
    against an emptying machine instead of against the next chunk's work, and the grid is
    what prices that drain. With a band's work ramping from many q-blocks down to one, a
    fine-enough cut hands each CU only a single work-group and a round's makespan is set
    by the longest band against a much lower mean, so quarters lose to the drain while
    halves still hand every CU two work-groups and the list schedule pairs the long and
    short bands together, costing nothing extra. Measured: only cutting the last batch,
    and only in half, wins; re-measured after the reduce's own shape changed, in case a
    shorter reduce had changed the trade -- still no gain from cutting further, and every
    extra chunk boundary costs another dispatch bubble.

    A split owns the q blocks with (q/BLOCK_Q) % q_split == split only if every band
    starts on a q_split boundary; otherwise the split -> q-block map depends on the band
    and a sub-range launch would leave those rows half-summed (the SNR shape's narrower
    band is exactly that case), so the cut is only taken when the band divides evenly.
    """
    h = q_split // 2
    cut = q_split % 2 == 0 and block_kv % (q_split * _BWD_BLOCK_Q) == 0
    return [(b, 0, None) for b in range(B - 1)] + (
        [(B - 1, 0, h), (B - 1, h, h)] if cut else [(B - 1, 0, None)]
    )


def _fused_pipelined(dkdv_l, odo_l, bufs, ws_dq, dq, B, Sq, Skv, block_kv, Hq, D, q_split, stream):
    """Run the fused kernel in chunks and hide each chunk's dQ reduce under the next one.

    The reduce is pure DRAM at the roofline while the fused kernel only uses 2.4 of the
    6.2 TB/s, so the reduce costs wall time only while nothing else is running. Batch is
    the one axis both kernels can be cut on: a kv band owns a slice of dQ for its OWN
    batch only, so batch b's partials are complete the moment batch b's dkdv retires.
    The delta pass is per-batch for the same reason, so all but its first chunk also
    moves off the critical path and lands inside the first fused dispatch.

    Where the wall time goes, from the kernel trace: almost all of it is fused dispatches
    (the first one alone, then the rest with a reduce hidden beside them), plus a small
    uncoverable head (the lse prescale and batch 0's delta, which the first chunk reads),
    a handful of dispatch-boundary bubbles (the event that fences a chunk off from its
    reduce is an HSA barrier packet), and one exposed tail (the last reduce, with the
    dk/dv slot reduce running beside it). So the fused body dominates the wall and
    everything the host layer still owns is small, most of which is already hidden.

    Every argument is a per-batch slice of the same buffers. The kernels' workspace
    strides are build-time constants of the FULL batch (dQ partials stride by
    ``batch_size`` bands, dk/dv slots by ``Q_SPLIT``), so a slice addresses exactly the
    rows the whole-batch launch would have given it -- same bands, same ascending order,
    same fp32 accumulator, hence bitwise-identical dQ/dK/dV.

    The chunks stay on ONE fused stream even though they are independent (a chunk owns its
    own batch, and where the last batch is cut its own q_split subset, so its dk/dv slots
    and dQ partial rows are disjoint from every other chunk's). Alternating them across
    two streams to fill each dispatch's causal drain is a measured LOSS: the overlap is
    real, but it is the overlap itself that costs -- two chunks in flight means two
    batches' Q/dO/K/V resident at once, which widens the cache footprint enough (on this
    kernel and its neighbouring odo/slot-reduce dispatches) to cost more than the drain
    it was meant to hide.

    Returns the event the caller must join before reading dQ; the dk/dv slot reduce is
    independent of dQ and is meant to be enqueued in front of that join.
    """
    qb, kb, vb, dob, ob, lb, db, wk, wv, cu_ph = bufs
    # Both queues stay at the DEFAULT priority. A reduce launch is many tiny work-groups
    # against a fused chunk's few large ones, so every CU slot the fused body frees is
    # taken by a reduce work-group before the next fused chunk lands, and the two
    # co-resident kernels do measurably interfere. Skewing HSA queue priority to fix it
    # loses both ways (starving either side costs more than the current balance), so
    # the interference has to be attacked by shrinking the reduce, not by re-arbitrating.
    side = _SIDE_STREAM.get(dq.device)
    if side is None:
        side = torch.cuda.Stream(device=dq.device)
        _SIDE_STREAM[dq.device] = side
    chunks = _pipe_chunks(B, q_split, block_kv)
    nc = len(chunks)
    evs = _PIPE_EVENTS.get(nc)
    if evs is None:
        evs = [torch.cuda.Event() for _ in range(nc + 2)]
        _PIPE_EVENTS[nc] = evs
    ev_delta, ev_join = evs[nc], evs[nc + 1]
    # Batch 0's delta goes FIRST and alone: it is the only one on the critical path (the
    # first fused chunk reads it), and it is a pure-bandwidth pass, so sharing the machine
    # with the other batches' delta costs it directly. Issuing it before the side stream
    # forks makes the fork's own wait order the side passes behind it at no extra
    # synchronisation -- `wait_stream` is the barrier that was already there. Kernel-trace
    # evidence (not TF) confirms the uncoverable head shrinks and the first fused dispatch
    # starts earlier, while batches 1..B-1 still land well before the chunk that needs
    # them and the fused dispatch beside them is unchanged.
    odo_l(ob[0], dob[0], db[0], 1, Sq, stream)
    side.wait_stream(stream)
    for b in range(1, B):
        odo_l(ob[b], dob[b], db[b], 1, Sq, side)
    ev_delta.record(side)
    for i, (b, lo, n) in enumerate(chunks):
        if i == 1:
            stream.wait_event(ev_delta)
        (dkdv_l if n is None else dkdv_l.chunk(lo, n))(
            qb[b], kb[b], vb[b], dob[b], lb[b], db[b], wk[b], wv[b],
            cu_ph, cu_ph, ws_dq[0, b].reshape(-1), 1, Sq, Skv, 0, stream,
        )
        evs[i].record(stream)
        side.wait_event(evs[i])
        _reduce_dq_partials(
            ws_dq, dq, block_kv, Hq, D, 1.0 / _LOG2E, side, bat_lo=b, n_bat=1,
            qsp=(q_split, lo, n), causal_offset=Skv - Sq,
        )
    ev_join.record(side)
    return ev_join


def _get_bwd(
    Hq, Hkv, D, scale, window_left, q_split, block_kv, dq_block_kv=64, batch_size=None, sbhd=False,
    varlen=False, fuse_dq=False, square=True,
):
    key = (Hq, Hkv, D, scale, window_left, q_split, block_kv, dq_block_kv, batch_size, sbhd, varlen, fuse_dq, square)
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
        # enough that the widest tile still leaves two waves per SIMD; D128's body is
        # twice as wide, so it needs a halved tile to keep two waves per SIMD spill-free.
        # g2d (D128's GEMM2 transpose-read prefetch depth) goes deeper here to cover the
        # extra ds_read_tr16 latency within the occ-2 budget's spare registers.
        # Under a left window the walked kv extent per WG is block_m + W, not the whole
        # causal prefix, so a wide tile no longer amortises anything; g2d goes as deep as
        # D128's under a window since the shortened kv loop is too short to hide the latency.
        _swa = window_left >= 0
        # Splitting the four waves by q rows leaves most staged tiles dead for most waves
        # under a window (each wave's own window is a fraction of the group's kv extent).
        # Splitting by GQA q head instead (one sharer head per wave) makes every wave want
        # every staged tile, at unchanged rows/accumulators/LDS/occupancy/work-group count;
        # needs four sharers to spend the waves on, so D128 and odd GQA ratios fall back
        # to the row split.
        _dq_hpw = 4 if (_swa and D != 128 and (Hq // Hkv) % 4 == 0) else 1
        # Dispatch a run of 8 q tiles before advancing the GQA sharer group, so work-groups
        # that overlap most of their window rows stay co-resident.
        _dq_qgrp = 8 if _swa else 1
        dq_block_m = 32 * (4 // _dq_hpw) if (_swa and D != 128) else (128 if D == 128 else 192)
        dq_l = (
            None
            if fuse_dq
            else build_flash_attn_bwd_dq_module(
                block_kv=32 if D == 128 else dq_block_kv,
                waves_per_eu=2 if D == 128 else 1,
                g2d=4 if (D == 128 or _swa) else 2,
                batch_size=batch_size,
                sbhd=sbhd,
                fuse_delta=_FUSE_DELTA,
                block_m=dq_block_m,
                q_heads_per_wg=_dq_hpw,
                q_group=_dq_qgrp,
                varlen=varlen,
                **common,
            )
        )
        # dkdv's read/MFMA ratio favors a wide tile, but that only fits spill-free at
        # waves_per_eu=1, so D128's narrower BLOCK_KV=64 keeps occ=2 with a deeper g2d
        # ring instead; dma_grp/pf_ring amortize the Q/dO staging rendezvous, at LDS cost
        # only the wide-tile (one wave per SIMD) configuration has spare. The fused
        # wide-band body needs one wave per SIMD for its dK/dV accumulators regardless,
        # which is also what lets the dQ reduce co-reside (see `_fused_pipelined`).
        _fuse_wide = fuse_dq and block_kv > 128
        if D == 128:
            dkdv_wpe = 1 if block_kv >= 128 else 2
            dkdv_g2d = 1 if block_kv >= 128 else 3
            dkdv_dma_grp = 2
            dkdv_pf_ring = block_kv >= 128
        else:
            # A windowed band's q loop is three blocks long, so a second wave per SIMD has
            # nothing left to interleave with: asking for one buys the body the whole
            # register budget instead (the two-wave group below is already the occupancy).
            dkdv_wpe, dkdv_g2d, dkdv_dma_grp = 1 if (_fuse_wide or _swa) else 2, 1, 1
            dkdv_pf_ring = False
        # The fused body keeps V's B-operand in registers; spending the LDS that frees on
        # a second Q/dO slot (q_dbuf, or dma_grp=2) loses both ways, and fence-trading
        # loses here even more than on the split D64 body. Every knob here was
        # independently re-swept on the four-wave fused arm, since a verdict taken at two
        # waves per SIMD does not automatically survive the move to one.
        dkdv_kw = dict(
            q_split=q_split,
            block_kv=block_kv,
            batch_size=batch_size,
            sbhd=sbhd,
            waves_per_eu=dkdv_wpe,
            g2d=dkdv_g2d,
            dma_grp=dkdv_dma_grp,
            pf_ring=dkdv_pf_ring,
            varlen=varlen,
            square=square,
            fuse_dq=fuse_dq,
            q_pref=fuse_dq,
            g3_defer=False,
            g3_kreg=_fuse_wide,
            # A windowed band is only BLOCK_KV + W q rows wide, so the default four-wave
            # split gives each wave a single kv tile and a repeated Q/dO fragment read;
            # halving to two waves shares that fragment read across two tiles per wave.
            flat_wg=256 if not _swa else 128,
            # Every knob below was independently re-swept on the Q_PAIR body since its
            # register pressure differs from the 64-row band; verdicts held except
            # g2_half (now a loss) and block_q (now fails the ISA occupancy gate).
            block_q=None,
            g2_half=None,
            g1_ks_outer=None,
            agpr=_DKDV_AGPR,
            **common,
        )
        dkdv_l = build_flash_attn_bwd_dkdv_module(**dkdv_kw)
        _dkdv_subs: dict = {}

        def _dkdv_chunk(qsp_lo, n_qsp):
            """Same body, dispatching only the q_split sub-range (see _fused_pipelined)."""
            sub = _dkdv_subs.get((qsp_lo, n_qsp))
            if sub is None:
                sub = build_flash_attn_bwd_dkdv_module(qsp_lo=qsp_lo, n_qsp=n_qsp, **dkdv_kw)
                _dkdv_subs[(qsp_lo, n_qsp)] = sub
            return sub

        dkdv_l.chunk = _dkdv_chunk
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
        q_split = _qsplit_for(max_sq, window_left)
        dq_l, dkdv_l, _ = _get_bwd(
            Hq, Hkv, D, scale, window_left, q_split,
            _blockkv_for(max_skv, D, window_left), _dq_block_kv(max_sq, window_left),
            batch_size=num_seg, sbhd=False, varlen=True, square=False,
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

    # A left window at least as wide as the sequence keeps every causal key: the smallest
    # in-range key index a query can mask off is Skv-1-W, so W >= Skv-1 makes the lower
    # bound vacuous and the shape is mathematically full causal. Normalize to -1 so it
    # takes the (faster) full-causal path instead of the windowed q-loop, which pins
    # q_split=1 and cannot fuse. Bit-identical result (no key is ever outside the window).
    if window_left >= 0 and window_left >= Skv - 1:
        window_left = -1
    q_split = _qsplit_for(Sq, window_left)
    block_kv = _blockkv_for(Skv, D, window_left)
    # Fused KV-outer path: dkdv also emits dQ, so S/dP/softmax are computed once instead
    # of twice (5 GEMMs, not 7). It needs a per-band dQ workspace (Skv//block_kv bands x
    # Sq rows), so block-aligned full-causal shapes -- square or rectangular; SWA and D128
    # keep the split pair.
    fuse_kv = _fuse_blockkv_for(Skv)
    # The fused dQ reduce tiles rows_per_wg*Hq*D (=2*Hq*D) by a block256*vec8*uc chunk;
    # _reduce_dq_partials picks uc=2 (4096 chunk) when it tiles and drops to uc=1 (2048)
    # otherwise (Hq=48 -> 6144), so any Hq with 2*Hq*D % 2048 == 0 fuses. D==64 here, so
    # this is Hq % 16 == 0.
    # The fused path fuses down to the smallest bench square (64/1024/1024 B=4, area 1M):
    # once the pipeline is off and the band is 128 it ties/beats the split pair there
    # (fused 0.168ms vs split 0.170ms; see the `pipe`/`_fuse_blockkv_for` small-shape
    # cases). Gate on causal area Sq*Skv >= 1024^2 -- squares >=1024 and rectangulars like
    # 1024/16384 whose huge Skv amortizes the reduce. Below 1M there is no bench coverage,
    # so keep the floor rather than fuse un-measured tiny shapes.
    # Rectangular (Skv>Sq) bottom-right causal also fuses: the dQ workspace already carries
    # Skv//block_kv bands and the fused body's G3 dQ emission is causal_offset-aware, so the
    # only rectangular-specific piece is the reduce's band count (see _reduce_dq_partials's
    # causal_offset). Needs Skv block-aligned too so the band axis tiles exactly.
    fuse_dq = (
        _FUSE_DQ and not sbhd and window_left < 0 and D == 64
        and Sq * Skv >= 1024 * 1024
        and Sq % fuse_kv == 0 and Skv % fuse_kv == 0 and (2 * Hq * D) % 2048 == 0
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
        _dq_block_kv(Sq, window_left),
        batch_size=B,
        sbhd=sbhd,
        fuse_dq=fuse_dq,
        square=(Sq == Skv),
    )
    # identity delta = -rowsum(O.dO); both kernels center dP by it (exact). dq owns the
    # reduce (it already holds dO in registers) and stores DELTA for dkdv when
    # _FUSE_DELTA is on, so no odo launch is needed; O is cast to bf16 (no-op when out
    # is already bf16) and passed into dq's freed slot via _defer_delta.
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    # The pipeline runs the backward one batch at a time to overlap each batch's dQ reduce
    # with the next batch's compute; it only pays when per-batch compute dwarfs the extra
    # per-batch dispatch. On the small square (64/1024/1024 B=4) the 4 micro-dispatches cost
    # far more than they overlap (pipe on 0.274ms vs pipe off 0.168ms), so gate it on causal
    # area: only Sq*Skv >= 2048^2 keeps the pipeline. Every fused bench shape but 1024^2 is
    # well above this, so this flips only the small square to the single whole-batch dispatch.
    pipe = fuse_dq and _DQ_PIPE and B > 1 and not sbhd and Sq * Skv >= 2048 * 2048
    if (fuse_dq or not _FUSE_DELTA) and not pipe:
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
        if pipe:
            bufs = (
                qf.view(B, -1), kf.view(B, -1), vf.view(B, -1), dof.view(B, -1),
                o16.view(B, -1), lsef.view(B, -1), df.view(B, -1),
                ws_dk.reshape(B, -1), ws_dv.reshape(B, -1), cu_ph,
            )
            join_ev = _fused_pipelined(
                dkdv_l, odo_l, bufs, ws_dq, dq, B, Sq, Skv, block_kv, Hq, D, q_split, st
            )
        else:
            # Pass ONE (band, batch) slice: the kernel rebases the SRD to its own slice with
            # a 64-bit offset, and the whole workspace overflows a flat memref's i32 count.
            dkdv_l(
                qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1),
                cu_ph, cu_ph, ws_dq[0, 0].reshape(-1), B, Sq, Skv, 0, st,
            )
            _reduce_dq_partials(ws_dq, dq, block_kv, Hq, D, 1.0 / _LOG2E, st, causal_offset=Skv - Sq)
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
    if pipe:
        # Joined only here: the slot reduce above needs no dQ, so it runs against the
        # last batch's dQ reduce instead of behind it.
        st.wait_event(join_ev)
    if sink is not None:
        # dsink[h] = Sum_i exp(sink_h - lse_i) * delta_flash[b,h,i], with delta already
        # -rowsum(O_s.dO) (negated) and lse_bhsq the raw sink-inclusive natural-log LSE.
        # Both are [B,Hq,Sq] with the same flat layout (b*Hq+h)*Sq+s.
        d_sink = _flash_dsink(sink, lse_bhsq, delta, B, Hq, Sq, st)
        return dq, dk, dv, d_sink
    return dq, dk, dv
