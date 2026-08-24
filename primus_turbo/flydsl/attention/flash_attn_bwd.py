###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL).
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

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

from primus_turbo.flydsl.utils.gemm_helper import xcd_remap_pid

_LOG2E = host_math.log2(host_math.e)
# Warp specialisation (splitting the head-step across a wave pair by role) is
# register-walled: each role's live set needs more than half the 512-dword pool, so the
# pair cannot co-reside two waves per SIMD, and no register donor closes the gap.
# q rows one dkdv work-group folds per q-loop step. 128 halves the (band, q-block) trips
# at an identical MFMA count per unit of work, and the halving itself measures -6.5% at
# l8b_b2 -- but every register form that fits costs more than the halving pays, so the item
# is FUNDING-bound, not mechanism-bound. Probe it with `_probe_bwdcfg.py`'s `bq<N>` arm:
# `kw:block_q=` desyncs the fold from the body and reports det=false, because six host
# planners capture this constant as a def-time default.
#
#   form (all at l8b_b2, spill/scratch from 21_final_isa.s)        dw   spill   verdict
#   g3_defer=0                                                    512   268 B   over
#   g3_defer=0,g3_kreg=0                                          500     0     over
#   g3_defer=0,g3_kreg=0,g2d=1                                    498     0     g2d frees 2
#   g3_defer=0,kv_halves=2                                        512   156 B   over
#   g3_defer=0,g3_kreg=0,q_pref=0                                 438     0     donors +25%
#   g3_defer=0,g3_kreg=0,kv_halves=2                              447     0     halving = 0
#
# The last row is the only form that clears the 464 co-residency line while KEEPING the
# Q_PREF staging, and it collects none of the prize: 4.3251 ms against 4.3280 for the same
# three donors at BLOCK_Q=64 (min of 40, palindrome, same-binary control spread 0.32%), so
# the trip halving delivers -0.07% instead of -6.5%. `kv_halves=2` halves NT, which makes
# every q fragment feed two passes instead of one -- 2832 `ds_read_b64_tr_b16` against 1792
# and 13614 instructions against 11212 -- so it does not fund BLOCK_Q=128, it SPENDS it.
# => the 36 dwords still have to come from somewhere that does not touch the q-side read
# count. The one named, never-dumped source is a half-width Q_PREF stage (`_qdo_issue`
# issues NUM_DMA_Q*2 loads of 4 dwords = 32 in flight at BQ=64 and 64 at BQ=128, so
# issuing half of them a hook later is worth ~32), which lands 500 -> ~468 and is then
# inside `red:vec=4`'s 480 line at the +0.1% that arm costs a 470-dword body.
_BWD_BLOCK_Q = 64

# dkdv MFMA-accumulator AGPR forcing (amdgpu-agpr-alloc): only pays off once the body
# is VGPR-lean, so it is disabled here. On the four-wave fused body it is not even a
# knob -- the compiler's own split is already byte-identical across the range tried.
_DKDV_AGPR = 0

# Pads dQ split-K band groups off a power-of-two stride so QDESC's same-row groups do not co-alias. D128 only.
_WSQ_BAND_PAD = 1 << 20
# Bands per interleaved dQ partial row: adjacent bands share a row so their same-row groups fill
# whole DRAM pages, bounded by what keeps a group slice inside a 32-bit num_records (see _wsq_ilv).
_WSQ_BAND_ILV = 8
# DPP quad_perm:[1,0,3,2], swaps a value between the two halves of every lane pair.
_QUAD_SWAP = 0xB1
# LLVMSchedGroupMask::VALU. flydsl's rocdl module exports the MFMA / DS / VMEM masks but
# not this one; the forward kernel drives its exp and VALU blocks with the same value.
_SCHED_VALU_MASK = 0x002
# gfx950: 8 XCDs each with a private L2 slice, picked by block_id % _NUM_XCD.
_NUM_XCD = 8
# gfx950 compute units. Only used to ask whether a dispatch is narrower than the machine.
_NUM_CU = 256


def _wsq_ilv(nb, B, Sq, hd, elem_bytes=2):
    """Bands per interleaved dQ partial row: a power of two dividing nb (see _WSQ_BAND_ILV)."""
    ilv = 1
    while ilv * 2 <= _WSQ_BAND_ILV and nb % (ilv * 2) == 0 and B * Sq * hd * elem_bytes * ilv * 2 < (1 << 32):
        ilv *= 2
    return ilv


# Cap on the dQ split-K partial workspace. Its natural size is bands*|dQ|, which follows the
# SQUARE of the context length, so a long context asks for hundreds of GB where the split pair
# asks for none -- capacity, not speed, is what still needs the Q-outer dq kernel there. Past
# the cap the band axis is walked in groups (see _band_span_for) and the reduce carries the
# running fp32 sum from group to group, so the footprint follows the cap instead of the context.
# A smaller cap hands the next group a running sum to re-read one more time. That traffic does
# measure flat against the group count, but walking the axis in groups is NOT free: it costs
# 11% at (4, 64, 8, 8192, 128) and 11% at (1, 64, 8, 16384, 128), and 9-23% across a span sweep
# at both head dims, bitwise identical either way. So the cap should be as large as the card can
# actually spare, not a round number.
#
# It scales with TOTAL memory, not free memory. Free memory moves between calls, and a cap that
# moves with it flips the plan between the grouped and whole-axis paths on the same shape -- two
# guard cells were seen reading either 14 or 28 ms for exactly that reason. Total memory is a
# property of the device, so the plan a shape gets is fixed.
_WSQ_BUDGET_FRACTION = 0.15


def _wsq_budget_bytes():
    """Cap on the dQ split-K partial workspace, at least the old fixed 16 GiB."""
    fixed = 16 << 30
    try:
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    except Exception:
        return fixed
    return max(fixed, int(total * _WSQ_BUDGET_FRACTION))


_WSQ_BUDGET_BYTES = None  # set to a byte count to override; probes use it to price a span


def _wsq_budget():
    return _WSQ_BUDGET_BYTES if _WSQ_BUDGET_BYTES is not None else _wsq_budget_bytes()


def _band_span_for(n_bands, band_bytes, ilv, whole=True):
    """Bands one dkdv pass owns; 0 when the whole band axis fits the workspace budget.

    A power-of-two multiple of the band interleave, so a pass boundary is also a band-group
    boundary and every pass sees the same slot layout the single-pass workspace has. As large
    as the cap allows: the carry the groups hand each other is re-read once per group, so the
    whole cost of walking the axis in groups falls as the group grows.

    ``whole`` also asks the span to TILE the band axis, which a dense pass needs because its
    launch addresses kv by the group's own extent -- a partial last group would reach past
    the axis and write dK/dV into the next slot. A ragged launch takes its kv extent from
    cu_seqlens instead, so a partial last group's tiles sit above every segment's kv length
    and retire without touching memory, so there the span is free of the band axis and only
    has to fit the cap -- but NOT free of the dispatch: the kv tile axis sits above the XCD
    term in the dkdv decode, so the grid is walked in runs of _NUM_XCD tiles and a span that
    is not a whole number of them puts the tail of one segment and the head of the next in
    the same run, whose makespan is the longest walk in it. Rounding the span down to a run
    beats every span between two runs by more than the extra pass it costs.
    """
    if n_bands * band_bytes <= _wsq_budget():
        return 0
    if not whole:
        span = max(1, _wsq_budget() // band_bytes)
        return span - span % _NUM_XCD if span >= _NUM_XCD else span
    span = ilv
    while span * 2 < n_bands and n_bands % (span * 2) == 0 and span * 2 * band_bytes <= _wsq_budget():
        span *= 2
    return span if span < n_bands else 0


def _wsq_ring_for(n_bands, block_kv, window_left, ilv, band_bytes, block_q=_BWD_BLOCK_Q):
    """Band groups the dQ partial workspace keeps live; 0 = one group per band.

    A finite window bounds how far along the band axis a q BLOCK's dQ slot travels: the body
    takes or drops a q block whole, so the blocks one band writes reach at most (W+block_q)
    rows past it. Bands further apart than that write DISJOINT rows, so letting them share a
    slot keeps one writer per slot -- the property the fixed-order fp32 reduce rests on -- and
    changes neither what is written nor the ascending order it is read back in. Only the
    workspace shrinks, from the band axis to the window. Gated on BYTES rather than on having
    a window, so a shape whose bands already fit keeps the plain band index (see
    _wsq_budget()); the group, not the band, is the unit because interleaved bands share
    a row (see _wsq_ilv). Opening it on the window alone is a CAPACITY lever, not a speed one:
    it takes gptoss_swa's workspace from 16 GiB to 4 GiB, and paired against this gate on the
    same host it measures gptoss_swa +0.72% and swa_d128_hq64 +0.08% -- the slots a window
    skips cost neither descriptor nor page walk, so shrinking them buys no wall time.
    """
    if window_left < 0 or n_bands * band_bytes <= _wsq_budget():
        return 0
    span = (window_left + block_q - 1) // block_kv + 2
    grp = span // ilv + 2
    return 0 if grp * ilv >= n_bands else grp


# NOTE: nothing here keeps a device tensor across calls. Under torch.compile's cudagraph
# trees the allocator is pointed at the graph pool for the whole tree, so a tensor cached by
# a kernel launched in that window stays resident in the pool while being no output of any
# graph -- which is exactly what "live storage data ptrs are in the cudagraph pool but not
# accounted for" reports. Re-allocating instead costs nothing measurable (the caching
# allocator hands back the same block): the deployment table moves 874.05 -> 872.37 backward
# TFLOPS and the varlen table is flat, both inside this node's noise.


def _band_lo_table(n_bands, span, device):
    """First band of each pass, as a device scalar the kernels read.

    A pass base cannot ride a scalar argument: the launch cache keys on the scalar signature,
    so a per-pass value would recompile both kernels once per pass. A one-element SLICE of this
    table is a tensor argument instead, which that cache ignores.
    """
    return torch.arange(0, n_bands, span, device=device, dtype=torch.int32)


def _cu_band_rows(cu_kv, n_bands, span):
    """cu_seqlens_kv with each pass's first band appended, one row per pass.

    A ragged launch needs the real segment table in the argument slot a dense band group
    hands its first band in, so the band rides one entry PAST the table. Built per call
    because the table is the caller's, and read as a row rather than a scalar argument for
    the same reason the dense table is (see _band_lo_table).
    """
    lo = _band_lo_table(n_bands, span, cu_kv.device)
    return torch.cat(
        (cu_kv.to(torch.int32).unsqueeze(0).expand(lo.numel(), cu_kv.numel()), lo.unsqueeze(1)),
        dim=1,
    )


def _qsp_absolute(head_dim, block_kv, q_split, block_q=_BWD_BLOCK_Q):
    """Whether the dkdv body must key its q_split subsets on the ABSOLUTE q block index.

    A split walks the q blocks ``_kv_first_q + split*BLOCK_Q + j*q_split*BLOCK_Q``, i.e.
    band-RELATIVE, while the reduce (and the pipeline's sub-range cut) both model a split
    as owning the q blocks with ``(q/BLOCK_Q) % q_split == split`` in EVERY band. The two
    coincide only when a band is a whole number of split strides, ``q_split |
    BLOCK_KV/BLOCK_Q``. D64's 256-row band holds four q blocks and the deployed q_split=4
    divides it; D128's band is halved to 128 rows (two q blocks) for its dK/dV
    accumulators, so it does NOT -- each band shifts the map by two, and _pipe_chunks has
    to drop its last-batch cut, leaving the whole final dQ reduce exposed with nothing
    beside it. Re-phasing the walk (see QSP_ABS) restores the absolute map and with it the
    cut; it does not change WHICH blocks the splits cover between them (the starts are the
    same BLOCK_Q offsets, permuted), only which split owns which, so it is equally
    available to D64 -- see _qsplit_for for why D64 has no split count that wants it.
    """
    return head_dim == 128 and q_split > 1 and (block_kv // block_q) % q_split != 0


def _qsp_cuttable(Sq, q_split, block_q=_BWD_BLOCK_Q):
    """Whether a pipeline chunk may own a q_split SUBSET of the q blocks.

    Both the body and the reduce address a subset as the q blocks with ``(q/BLOCK_Q) %
    q_split == split``, which only partitions the rows when the splits tile the q blocks
    -- Sq=16384 over q_split=3 leaves a remainder block whose rows no chunk would reduce.
    """
    return Sq % (block_q * q_split) == 0


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


def _if_wave(cond, vals, then_fn, else_fn, else_yields=False):
    """scf.if threading ``vals`` through both arms, for a wave-uniform ``cond``.

    Built directly rather than with a traced ``if``: the tracer only carries plain
    named variables across a dynamic branch, and these are lists of accumulators.
    ``else_yields`` takes the else arm's own return list instead of ``vals``, which
    is what nesting one of these inside another's arm needs.
    """
    from flydsl._mlir.dialects import scf

    _v = [_raw(v) for v in vals]
    op = scf.IfOp(_raw(cond), [x.type for x in _v], has_else=True)
    with ir.InsertionPoint(op.regions[0].blocks[0]):
        scf.YieldOp([_raw(x) for x in then_fn()])
    if not op.regions[1].blocks:
        op.regions[1].blocks.append()
    with ir.InsertionPoint(op.regions[1].blocks[0]):
        _e = else_fn()
        scf.YieldOp([_raw(x) for x in _e] if else_yields else _v)
    return list(op.results)


def _mc_clear(mt, nt):
    """Mask class of a 16-tile on a wave whose kv rows all sit below the causal edge."""
    return 0


def _mc_diag(mt, nt):
    """Mask classes of the DIAGONAL wave, whose first kv row is the q block's first row.

    Its kv 16-tile nt then starts exactly on q 16-tile mt = nt: above that tile it is
    entirely past the causal edge, below it entirely under the edge, whatever the causal
    offset (see MASK_ALIGN). Only the one tile per row where the two coincide needs the
    compare/select set.
    """
    _d = nt - mt
    return 0 if _d < 0 else (1 if _d == 0 else 2)


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
    waves_per_eu=4,
    block=256,
    sbhd=False,  # SBHD [S,B,H,D] native O/dO layout (seq-step = B*H*D)
    token_major=False,  # True: DELTA is packed [S,Hq] (the ragged layout, see below)
    spw=8,  # q rows per work-group tile (the rest of the tile is q-heads)
    # q_split/qsp_lo/n_qsp mirror the fused body's q-loop split, so the pass that PRODUCES
    # delta can be cut on the same axis as the body that consumes it: the body's chunk for
    # splits [qsp_lo, qsp_lo+n_qsp) only reads the delta of the q blocks those splits own,
    # so only that much delta has to exist before the chunk launches and the rest can run
    # beside it (see _fused_pipelined). SBHD only -- the ragged path cuts by segment instead.
    q_split=1,
    qsp_lo=0,
    n_qsp=None,  # None: every q block (no q-split sub-range)
    block_q=_BWD_BLOCK_Q,
):
    """Identity-delta ("odo") kernel: DELTA[b,hq,s] = -sum_d O[b,s,hq,d]*dO[b,s,hq,d].

    LPR lanes cooperate on one (b,s,hq) row -- one 16 B chunk of O and of dO each -- and
    fold their partials with an xor butterfly over the low lane bits (ds_bpermute is the
    LDS crossbar only: no allocation, no barrier), then one lane stores the negated scalar
    (the dkdv/dq fold convention) to the transposed [B,Hq,S] delta -- or, under
    ``token_major``, to the packed [S,Hq] one the ragged backward reads.

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
    N_QSP = q_split if n_qsp is None else n_qsp
    assert 1 <= N_QSP <= q_split and 0 <= qsp_lo <= q_split - N_QSP
    # A q block is a whole run of s-tiles, and with the s-tile the SLOWEST decode axis a q
    # block's work-groups are a contiguous run of ids -- which is what lets a sub-range be a
    # pure grid restriction plus a remap, with no per-thread predicate.
    assert N_QSP == q_split or (sbhd and block_q % SPW == 0)
    WG_PER_STILE = NUM_HEADS_Q // HPW
    STPB = block_q // SPW  # s-tiles per q block

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
        if const_expr(N_QSP < q_split):
            # Spread this launch's dense ids back over the q blocks its splits own, the same
            # map the body and the dQ reduce use: block (q/block_q) belongs to split
            # (q/block_q) % q_split (see build).
            _wg_qb = fx.Index(STPB * WG_PER_STILE) * fx.Index(batch_size)
            _qb, _in = bid // _wg_qb, bid % _wg_qb
            bid = ((_qb // fx.Index(N_QSP)) * fx.Index(q_split) + fx.Index(qsp_lo)) * _wg_qb + _in
            bid = bid + (_qb % fx.Index(N_QSP)) * _wg_qb
        ht = bid % fx.Index(WG_PER_STILE)
        _r = bid // fx.Index(WG_PER_STILE)
        if const_expr(sbhd):
            b = _r % fx.Index(batch_size)
            st = _r // fx.Index(batch_size)
        else:
            n_stile = (sl + fx.Index(SPW - 1)) // fx.Index(SPW)
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
        # Both streams non-temporal: O/dO is touched once and the set outruns LLC (same for dQ reduce).
        off = base + chunk * fx.Index(VEC)
        ov = buffer_ops.buffer_load(o_rsrc, off, vec_width=VEC, dtype=elem_dtype_l, cache_modifier=2)
        dv = buffer_ops.buffer_load(do_rsrc, off, vec_width=VEC, dtype=elem_dtype_l, cache_modifier=2)
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

        # DELTA is transposed [B,Hq,S]: delta[b,hq,s] at (b*Hq + hq)*S + s. The ragged
        # backward instead keeps DELTA packed by token, the layout its LSE already has and
        # the fused body's per-segment gather reads (called with batch_size=1, s = token).
        if const_expr(token_major):
            delta_off = s * fx.Index(NUM_HEADS_Q) + hq
        else:
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
        n_stile = (fx.Index(seq_len) + fx.Index(SPW - 1)) // fx.Index(SPW)
        if const_expr(N_QSP < q_split):
            n_stile = n_stile // fx.Index(q_split) * fx.Index(N_QSP)
        grid_x = fx.Index(batch_size) * fx.Index(WG_PER_STILE) * n_stile
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
    vec=8,  # elements per load: 8 = buffer_load_dwordx4 (also what the paired store needs)
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
    window_left=-1,  # >=0: SWA -- also clamp the band loop's LOW edge (see g_lo). <0: full.
    sbhd=False,  # True: DQ is native SBHD [Sq,B,Hq,D]; remap the final store's row (see below).
    band_pad=0,  # padding bytes between band groups (see _WSQ_BAND_PAD); 0 = dense
    band_ilv=1,  # adjacent bands sharing one partial row (see _WSQ_BAND_ILV)
    band_ring=0,  # >0: band groups reuse this many workspace slots (see _wsq_ring_for)
    varlen=False,  # True: rows are packed q tokens of ``num_seg`` segments (see below)
    num_seg=1,
    # band_span: >0 = the workspace holds ONE GROUP of that many bands and this launch folds
    # the group, resuming the row's running fp32 sum from CARRY and leaving it there again
    # unless the group holds the row's top band (see _band_span_for). The group's first band
    # arrives as a device scalar in the CuSeqKv slot.
    band_span=0,
):
    """Fold the fused kernel's dQ split-K partials: DQ[b,q] = sm * Sum_b' WSQ[b',b,q].

    Only the bands a q row causally sees (b' <= q/BLOCK_KV) are read, in ascending
    order and with an fp32 accumulator, so the result is bitwise reproducible without
    atomics. One pass replaces torch's sum -> mul_ -> cast chain, which materialises an
    fp32 [B, BLOCK_KV*Hq*D] temporary per q group and touches it three more times.

    Under ``varlen`` the partials are packed by q TOKEN ([bands, total_q, Hq*D], built with
    batch_size=1 and seq_len_q=total_q so the band stride follows the workspace), and which
    bands a row sees comes from its OWN segment: the band window is derived per row from
    cu_seqlens rather than from one compile-time Sq and causal offset. A work-group then
    owns a SINGLE row, since a segment's token base carries no alignment and two
    consecutive packed rows can straddle a segment boundary without sharing a band window.
    That costs nothing: the work-group count is rows*Hq*D/chunk however the rows are split.

    A work-group owns ``rows_per_wg`` q rows (one q group, hence one band count) and
    every thread carries ``uc`` independent 16 B chunks, so that many loads per band are
    in flight -- the band loop is dynamic and cannot be unrolled. ``lpt`` matters for the
    work-group width; every (block, uc, lpt) combination returns bit-identical dQ, so
    these are pure rate knobs: pick one by what it costs the co-resident body below, not by
    its own rate.

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

    Adding a THIRD wave is a measured LOSS, from both sides of the equation. The body
    sits at 455 dwords, so ceil8 456 + 2*24 leaves 56 -- two waves with 8 to spare, and a
    third needs 24 more. Donating them from the body costs more than the wave returns
    (k_reg=0 -> 434 dwords and -0.49%, g3_kreg=0 -> 428 and -2.0%, both on 32-sample
    same-process A/B), and shrinking the wave instead costs the same (rows_per_wg=1
    -0.14%, and uc=2 on the g3_kreg body -7.6%). Run alone this kernel already streams at
    the machine roofline, so extra waves cannot shorten it; they only take DRAM and issue
    from the body beside them. Treat two waves as the operating point and spend any future
    donation on the band width instead (see _fuse_blockkv_for).

    Only the LAST pipeline chunk's reduce runs with nothing beside it, so that one may
    legally take a different (wider, faster-standalone) shape -- a chunk owns a disjoint
    element slice, and uc/block partition elements rather than reordering a slot's band
    sum, so mixing shapes across launches is still bitwise identical. Measured: not a
    gain, because the tail dispatch is a small fraction of the whole call regardless of
    per-thread load count, so its co-resident slowdown (which does not apply to it in
    the first place, since nothing else runs beside it) was never the real cost.

    The dQ STORE is non-temporal: dQ is not read again in the backward, so keeping it in
    L2/MALL only evicts the dO the fused kernel re-reads once per band. It is not a general
    lever: on the fused kernel's own partial store the whole CPol space measured worse than
    the default cached policy, so a stream that something downstream re-reads keeps it.

    The partial READ keeps it too, and that was re-checked rather than assumed: this kernel
    runs CO-RESIDENT with the producer that wrote those partials, so dropping nt to let the
    just-written lines stay resident looks free, but it measured D128 797.8 THD / 757.1 SBHD
    against 810.6 / 774.4 with nt (D64 arm flat) -- the partial stream is far larger than
    any cache level, so retaining it only displaces the Q/dO the producer is re-reading.
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
    ILV = band_ilv
    RING = int(band_ring)
    BAND_BYTES = batch_size * SQ * HD * 2 * ILV
    BAND_STRIDE = BAND_BYTES + band_pad
    assert BAND_BYTES < (1 << 32), "band group must fit a 32-bit num_records"
    ROW0 = bat_lo * SQ
    NSEG = num_seg
    SPAN = int(band_span)
    DQ_BYTES = batch_size * SQ * HD * 2
    CARRY_BYTES = batch_size * SQ * HD * 4
    # Both are addressed by one descriptor whose num_records also does the row's "is this my
    # group" gating, so they have to fit a 32-bit byte count. The carry moves in 16 B pieces
    # (fp32 has no 32 B store), which is what pairs a VEC of 8 into two of them.
    assert not SPAN or (CARRY_BYTES < (1 << 32) and SPAN % ILV == 0 and VEC == 8)
    # A packed row's band window is per-segment, so nothing here may assume a single Sq:
    # the whole token axis is one "batch" and one row is one work-group (see above).
    assert not varlen or (NB == 1 and RPW == 1 and not sbhd and causal_offset == 0 and NSEG >= 1)

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
    # A windowed row reads a fixed handful of bands, so consecutive work-groups re-read the same
    # K/V: hand each XCD a contiguous run of them and those re-reads stay in one L2 slice. Only
    # worth it when the chunks divide evenly across the XCDs, and only on a rectangle (a square's
    # causal_offset is 0, where the runs a row shares are already neighbours).
    XCD_CUT = (
        window_left >= 0
        and QSP == 1
        and NB == batch_size
        and causal_offset > 0
        and (N_CHUNK % _NUM_XCD == 0 or _NUM_XCD % N_CHUNK == 0)
    )

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_dqred_kernel(
        WSQ: fx.Tensor, DQ: fx.Tensor, CARRY: fx.Tensor, CuSeqQ: fx.Tensor, CuSeqKv: fx.Tensor
    ):
        bid = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        chunk = fx.Index(0)
        if const_expr(XCD_CUT):
            bid = fx.Index(xcd_remap_pid(bid, fx.Index(gpu.grid_dim.x), _NUM_XCD))
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
        # SWA also bounds the LOW band: the fused body's windowed q-loop only wrote this q's
        # dQ slot in bands whose kv range overlaps [q+off-W, q+off], so a band below
        # g_lo = floor(max(0, q_blk+off-W) / block_kv) never wrote it -- summing it would read
        # a stale slot. The fused body writes at q-BLOCK granularity (a band takes/leaves a
        # whole BLOCK_Q run of q together), so the low edge must be computed from the block's
        # first row q_blk, not the work-group's row0: when W is not a block_kv multiple (e.g.
        # W=2047), a per-row floor would vary inside one block and skip a band that did write.
        # (g's per-row and per-block value coincide -- off%block_kv==0 and a BLOCK_Q block sits
        # in one block_kv bin -- so it needs no such alignment.) Full-causal (window_left<0)
        # keeps g_lo=0, so range(0, g+1) is unchanged and the ISA stays byte-identical.
        if const_expr(varlen):
            # Ragged: this row's segment owns both edges. The segment is the largest s with
            # CuSeqQ[s] <= row0, found by a branchless binary search over the compile-time
            # segment count (every value here is work-group uniform, hence scalar).
            _cuq_rsrc = buffer_ops.create_buffer_resource(CuSeqQ, max_size=True)
            _cukv_rsrc = buffer_ops.create_buffer_resource(CuSeqKv, max_size=True)

            def _cu_at(rsrc, idx):
                return fx.Index(fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=fx.Int32)))

            _seg = fx.Index(0)
            _nstep = max(1, (NSEG - 1).bit_length())
            for _si in range_constexpr(_nstep):
                _cand = _seg + fx.Index(1 << (_nstep - 1 - _si))
                _fits = ArithValue(_cand < fx.Index(NSEG))
                _probe = fx.Index(_fits.select(_cand, fx.Index(NSEG - 1)))
                _take = _fits & ArithValue(_cu_at(_cuq_rsrc, _probe) <= row0)
                _seg = fx.Index(_take.select(_cand, _seg))
            _qbeg = _cu_at(_cuq_rsrc, _seg)
            _qlen = _cu_at(_cuq_rsrc, _seg + fx.Index(1)) - _qbeg
            _kvlen = _cu_at(_cukv_rsrc, _seg + fx.Index(1)) - _cu_at(_cukv_rsrc, _seg)
            _qloc = row0 - _qbeg
            # Segment-local causal edge q + (Skv_seg - Sq_seg), kept non-negative by
            # comparing before subtracting (a segment may have fewer keys than queries, and
            # a row below its first key attends nothing at all -> empty band range).
            _top = _qloc + _kvlen
            _att = ArithValue(_top >= _qlen)
            g = fx.Index(_att.select(_top - _qlen, fx.Index(0))) // fx.Index(block_kv)
            if const_expr(window_left >= 0):
                _lonum = _qloc - (_qloc % fx.Index(BQ)) + _kvlen
                _losub = _qlen + fx.Index(window_left)
                g_lo = fx.Index(
                    ArithValue(_lonum > _losub).select((_lonum - _losub) // fx.Index(block_kv), fx.Index(0))
                )
            else:
                g_lo = fx.Index(0)
            g_hi = fx.Index(_att.select(g + fx.Index(1), g_lo))
        else:
            _qloc = row0 % fx.Index(SQ)
            g = (_qloc + fx.Index(causal_offset)) // fx.Index(block_kv)
            if const_expr(window_left >= 0):
                _dlt = causal_offset - window_left  # constexpr int, may be negative
                _qblk = _qloc - (_qloc % fx.Index(BQ))
                if const_expr(_dlt >= 0):
                    _lonum = _qblk + fx.Index(_dlt)
                else:
                    _dsub = fx.Index(-_dlt)
                    _lonum = fx.Index(ArithValue(_qblk > _dsub).select(_qblk - _dsub, fx.Index(0)))
                g_lo = _lonum // fx.Index(block_kv)
            else:
                g_lo = fx.Index(0)
            g_hi = g + fx.Index(1)
        if const_expr(SPAN):
            # This launch owns bands [blo, blo+SPAN) and the workspace holds exactly those,
            # so the row's band range is clipped into the group and re-based onto it. A row
            # whose top band is below the group was finished by an earlier launch; a row
            # above it leaves its running sum in CARRY. Both are expressed by CLAMPING the
            # range to empty and by zeroing a descriptor's num_records rather than by
            # branching, so a work-group with nothing to do issues no memory at all.
            # The pass's first band: its own device scalar where the cu_seqlens slot is free,
            # one entry past the segment table on a ragged launch (see _cu_band_rows).
            if const_expr(varlen):
                _lo_rsrc, _lo_idx = _cukv_rsrc, fx.Index(NSEG + 1)
            else:
                _lo_rsrc = buffer_ops.create_buffer_resource(CuSeqKv, max_size=True)
                _lo_idx = fx.Index(0)
            _blo = fx.Index(fx.Int32(buffer_ops.buffer_load(_lo_rsrc, _lo_idx, vec_width=1, dtype=fx.Int32)))
            _bhi = _blo + fx.Index(SPAN)
            _abs_lo, _abs_hi = g_lo, g_hi
            _top = fx.Index(ArithValue(_abs_hi < _bhi).select(_abs_hi, _bhi))
            g_lo = fx.Index(ArithValue(g_lo > _blo).select(g_lo - _blo, fx.Index(0)))
            g_hi = fx.Index(ArithValue(_top > _blo).select(_top - _blo, fx.Index(0)))
            _live = ArithValue(g_hi > g_lo)
            _final = _live & ArithValue(_abs_hi <= _bhi)
            if const_expr(varlen):
                # A ragged row can see no key at all (its segment has fewer keys than
                # queries), which makes it live in NO group -- so the first group also owns
                # the zero store such a row gets for free when one pass folds every band.
                _final = _final | (ArithValue(_abs_hi <= _abs_lo) & ArithValue(_blo == fx.Index(0)))
            _carry_on = _live & ArithValue(_abs_hi > _bhi)
        base = row0 * fx.Index(HD) + chunk * fx.Index(CHUNK_ELEMS) + tid * fx.Index(VEC)
        offs = [base + fx.Index(c * BLOCK * VEC) for c in range_constexpr(UC)]
        if const_expr(ILV > 1):
            iloffs = [
                (o // fx.Index(head_dim)) * fx.Index(ILV * head_dim) + o % fx.Index(head_dim) for o in offs
            ]
        else:
            iloffs = offs
        c_zero_vec = Vec.filled(VEC, 0.0, fx.Float32).ir_value()

        if const_expr(SPAN):
            # Resume where the previous group left off. The first group has nothing to
            # resume: its descriptor is empty, the loads return zeros, and 0 + first band is
            # bitwise what the single-group kernel accumulates -- so however the band axis is
            # cut, dQ is the same left-to-right fp32 sum over the same ascending bands.
            _carry_rd = buffer_ops.create_buffer_resource(
                CARRY,
                max_size=False,
                num_records_bytes=_raw(
                    fx.Index(
                        (_live & ArithValue(_blo > fx.Index(0))).select(fx.Index(CARRY_BYTES), fx.Index(0))
                    )
                ),
            )
            acc = [
                Vec(buffer_ops.buffer_load(_carry_rd, o, vec_width=4, dtype=fx.Float32))
                .shuffle(
                    Vec(buffer_ops.buffer_load(_carry_rd, o + fx.Index(4), vec_width=4, dtype=fx.Float32)),
                    list(range(VEC)),
                )
                .ir_value()
                for o in offs
            ]
        else:
            acc = [c_zero_vec for _ in range_constexpr(UC)]
        for band, inner in range(g_lo, g_hi, fx.Index(1), init=acc):
            _grp = band // fx.Index(ILV) if const_expr(ILV > 1) else band
            if const_expr(RING):
                _grp = _grp % fx.Index(RING)
            band_rsrc = buffer_ops.create_buffer_resource(
                WSQ,
                max_size=False,
                num_records_bytes=_raw(fx.Index(BAND_BYTES)),
                base_byte_offset=_raw(_grp * fx.Index(BAND_STRIDE)),
            )
            _lane = (band % fx.Index(ILV)) * fx.Index(head_dim) if const_expr(ILV > 1) else None
            parts = [
                buffer_ops.buffer_load(
                    band_rsrc,
                    o if _lane is None else o + _lane,
                    vec_width=VEC,
                    dtype=elem_dtype,
                    cache_modifier=2,
                )
                for o in iloffs
            ]
            acc = yield [
                (Vec(inner[c]) + Vec(parts[c]).to(fx.Float32)).ir_value() for c in range_constexpr(UC)
            ]
        if const_expr(UC == 1):
            acc = [acc]  # a lone iter_arg comes back bare rather than as a 1-element list

        if const_expr(SPAN):
            # A row whose top band is above this group hands the fp32 sum on instead of
            # storing dQ, and the two descriptors make that choice by num_records.
            _carry_wr = buffer_ops.create_buffer_resource(
                CARRY,
                max_size=False,
                num_records_bytes=_raw(fx.Index(_carry_on.select(fx.Index(CARRY_BYTES), fx.Index(0)))),
            )
            for c in range_constexpr(UC):
                for s in range_constexpr(VEC // 4):
                    _cv = Vec(acc[c])
                    buffer_ops.buffer_store(
                        _cv.shuffle(_cv, [4 * s + i for i in range_constexpr(4)]).ir_value(),
                        _carry_wr,
                        (offs[c] + fx.Index(4 * s)) * fx.Index(4),
                        cache_modifier=2,
                        offset_is_bytes=True,
                    )
            dq_rsrc = buffer_ops.create_buffer_resource(
                DQ,
                max_size=False,
                num_records_bytes=_raw(fx.Index(_final.select(fx.Index(DQ_BYTES), fx.Index(0)))),
            )
        else:
            dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=True)
        sm_vec = Vec.filled(VEC, sm_scale, fx.Float32)
        # PAIR_ST: pair quad neighbours for a full-width store -- wins for a window, not full-causal.
        PAIR_ST = window_left >= 0
        if const_expr(PAIR_ST):
            _pair_lo = ArithValue((tid & fx.Index(1)) == fx.Index(0))
        for c in range_constexpr(UC):
            _v = (Vec(acc[c]) * sm_vec).to(elem_dtype)
            if const_expr(PAIR_ST):
                _dw = _v.bitcast(fx.Int32)
                _sw = [
                    rocdl.update_dpp(
                        fx.Int32.ir_type, _raw(_dw[i]), _raw(_dw[i]), _QUAD_SWAP, 0xF, 0xF, False
                    )
                    for i in range_constexpr(4)
                ]
                _pick = [_pair_lo.select(_dw[i], _sw[i + 2]) for i in range_constexpr(2)]
                _pick += [_pair_lo.select(_sw[i], _dw[i + 2]) for i in range_constexpr(2)]
                _o = offs[c]
                _outs = [
                    (
                        Vec.from_elements([fx.Int32(p) for p in _pick], fx.Int32).bitcast(elem_dtype),
                        (_o - (_o & fx.Index(31)))
                        + ((_o & fx.Index(8)) << fx.Index(1))
                        + ((_o & fx.Index(16)) >> fx.Index(1)),
                    )
                ]
            else:
                # The workspace's D axis is permuted so the fused kernel can write a q row's
                # partial 64 B at a time (see the store in `_gemm3_tiles`): bit 4 of the real
                # D index sits at bit 2 of the permuted one, so a chunk's VEC elements are read
                # contiguously here and written back as VEC/4 runs of 4 at their un-permuted
                # address -- cheap on this read-bandwidth-bound kernel, and it halves the
                # partial store's request count on the fused kernel's critical path instead.
                _outs = []
                for s in range_constexpr(VEC // 4):
                    _o = offs[c] + fx.Index(4 * s)
                    _outs.append(
                        (
                            _v.shuffle(_v, [4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3]),
                            (_o - (_o & fx.Index(31)))
                            + ((_o & fx.Index(24)) >> fx.Index(1))
                            + (((_o >> fx.Index(2)) & fx.Index(1)) << fx.Index(4)),
                        )
                    )
            for _val, _dq_off in _outs:
                if const_expr(sbhd):
                    _row = _dq_off // fx.Index(HD)
                    _d = _dq_off - _row * fx.Index(HD)
                    _bb = _row // fx.Index(SQ)
                    _qq = _row - _bb * fx.Index(SQ)
                    _st_off = (_qq * fx.Index(batch_size) + _bb) * fx.Index(HD) + _d
                else:
                    _st_off = _dq_off
                buffer_ops.buffer_store(
                    _val.ir_value(),
                    dq_rsrc,
                    _st_off * fx.Index(2),
                    cache_modifier=2,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch_flash_attn_bwd_dqred(
        WSQ: fx.Tensor,
        DQ: fx.Tensor,
        CARRY: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        stream: fx.Stream,
    ):
        flash_attn_bwd_dqred_kernel(
            WSQ,
            DQ,
            CARRY,
            CuSeqQ,
            CuSeqKv,
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

    ``uc`` (chunks per thread) is what sizes the grid. All the loads are issued before any is
    consumed, so it is also this kernel's memory-level parallelism -- but the rate is flat
    across the legal range and falls off a cliff once the grid narrows to the CU count, so the
    caller picks it for tiling rather than for speed (see _slotred_uc).
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "slot reduce kernel targets gfx950"
    elem_dtype = dtype_to_elem_type(dtype_str)
    VEC = _SLOTRED_VEC
    BLOCK = block
    UC = uc
    TILE = BLOCK * UC * VEC  # elements one work-group folds, per tensor
    assert n_elems % TILE == 0, "n_elems must tile the work-group"
    WPG = n_elems // TILE
    NS = n_slots

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flash_attn_bwd_slotred_kernel(WSK: fx.Tensor, DK: fx.Tensor, WSV: fx.Tensor, DV: fx.Tensor):
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
    # waves_per_eu is a request, not a cap: on the D128 fused body LLVM already lands at 244
    # VGPR, so waves_per_eu=1 compiles byte-identical ISA (244 / 0 spill / 118,784 B) and
    # occupancy stays at 2 waves per SIMD either way. It is not a lever here.
    #
    # The D64 wide-band body's own ISA allocation is 256 with 4 spills, and that does NOT cost
    # it the co-resident dQ reduce: its 84 KB of LDS puts one work-group of four waves on a CU,
    # so one body wave per SIMD leaves most of the 512-register file for the reduce's 10, and
    # the kernel trace shows three of the four pipelined reduces running wholly inside the next
    # chunk's body. What the co-residency costs is measurable and small -- the chunk with no
    # reduce beside it runs 1354 us against 1419/1425/1479 for the three that host one, and a
    # hosted reduce stretches to 756-783 us against 338 alone. Trying to widen that seat by
    # cutting the body's registers loses: k_reg=False reaches 250, q_pref=False reaches 244 but
    # costs gptoss_full +11.3%, and g3_kreg=False frees none (LLVM spends the slack on the
    # spills and stays at 256). Neither --amdgpu-num-vgpr nor waves_per_eu can move it either:
    # the binary-opts channel both travel on is inert here (byte-identical ISA).
    waves_per_eu=2,
    block_kv=128,
    # block_q: q rows staged per q-loop trip (None = _BWD_BLOCK_Q). Under a left window a
    # band's q extent is exactly BLOCK_KV + W rows, so a block_q that divides it walks the
    # SAME (kv,q) area in fewer staging trips -- 96 covers 64+128 in two instead of three.
    block_q=None,
    num_kv_heads=None,
    q_split=2,
    window_left=-1,
    batch_size=None,  # compile-time B; required for SBHD seq-step stride bake
    sbhd=False,  # SBHD [S,B,H,D] native layout (seq-step = B*H*D)
    agpr=_DKDV_AGPR,  # force N MFMA accumulators into AGPRs (0 disables); layout-agnostic
    # g2d: GEMM2 transpose-read prefetch depth (ring across dt). Depth-1 wins even once
    # deeper rings are register-free, because the read-ahead burst displaces MFMA issue
    # more than it saves in fences -- cutting fences is not this body's currency,
    # MFMA-run density is.
    g2d=1,
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
    square=True,  # caller guarantees Sq==Skv (causal_offset==0); gates the FQ_PAIR windowed
    # optimization, which is only correct for square shapes (see FQ_PAIR).
    kv_halves=1,
    wsq_pad=0,  # padding bytes between dQ partial band groups (see _WSQ_BAND_PAD)
    wsq_ilv=1,  # adjacent bands sharing one partial row (see _WSQ_BAND_ILV)
    wsq_ring=0,  # >0: band groups reuse this many workspace slots (see _wsq_ring_for)
    # wsq_nrec: PROBE (dQ wrong, deterministic), the write end of the split-K round trip.
    # Divides the partial slot's num_records by N so the hardware drops every store past
    # seq_len_q/N while the instructions, address chains and registers stay byte-identical --
    # the only arm that prices the store stream's VOLUME. Aliasing the bands onto one slab
    # (wsq_ring) shrinks the FOOTPRINT instead and so prices cache residency, not volume.
    wsq_nrec=None,
    # wsq_atomic: PROBE, and dQ IS WRONG WHILE IT IS ON -- the buffer is never zeroed, the
    # sm_scale still lives in the fold and there is no fp32->bf16 cast pass. It prices the
    # accumulation and nothing else. Non-zero replaces the bf16 partial store with the pair's
    # 8 floats as 8 consecutive-dword `buffer_atomic_add_f32` into one band-less fp32 image --
    # aiter's hd128 scheme -- so band axis, workspace and fold all disappear, and so does
    # bitwise reproducibility. The value is the atomic's CPol aux plus one; aux 0 is cheapest
    # and is not by itself correct across XCDs, since an agent-scope atomic must be performed
    # past the local L2 to see the other XCDs' adds.
    #
    # Measured, and the verdict is arithmetic rather than tuning. The ISA is clean (512
    # atomics, 486 VGPR, 0 scratch, LDS unchanged) but d128_70b_b2 reads 105.5 ms against
    # 6.34, where the whole dQ split-K round trip it would replace costs 1.229 ms. The payload
    # is why: an element takes one contribution per kv band that causally sees it, ~32 here,
    # the same count aiter pays, so either scheme moves 4.3 G element-updates -- 8.6 GB stored
    # plus 8.6 GB folded as bf16 partials, against 17.2 GB of fp32 read-modify-write, already
    # 2x at the memory side. Then this C layout hands a lane 8 floats of ONE q row, so a
    # wave's 64 lanes touch 32 lines two dwords at a time: 2.15 G line-RMWs, 275 GB, which is
    # the 105 ms. Transposing the epilogue so a wave covers 4 whole lines would recover that
    # 8x and land near 6 ms -- still 5x the leg. The lever that would re-price it is a larger
    # BLOCK_KV, and that is measured shut from the other side: +17% at D128, +65% at D64,
    # spill-free arms included (see _fuse_blockkv_for).
    wsq_atomic=0,
    # wsq_acoal: with wsq_atomic on, issue the atomics COALESCED (see _g3_atomic).
    wsq_acoal=0,
    # wsq_rmw: PROBE (dQ/dK/dV stay EXACT -- the value read is pinned live and discarded).
    # A slot-indexed dQ partial workspace (one slot per resident work-group instead of one
    # per kv band, so nsplits falls from Skv/BLOCK_KV to that over the bands a slot owns)
    # does NOT remove bytes. Every (band, q row) pair still has to be written, so the
    # writes stay put and the fold's reads fall by 1 - 1/k only because the body now READS
    # each slot back once per extra band it owns: the scheme moves (1 - 1/k) of one read
    # stream from the fold kernel into the body and its whole net is
    #     (1 - 1/k) * |partials| * (fold read ms/GB - body read ms/GB).
    # The first rate is measured and linear with no fixed part (0.086 ms/GB, see
    # _reduce_dq_partials); the second has never been measured on this body, and it is the
    # only unknown in front of the scheme. This arm issues exactly that read -- same
    # address as the store, same buffer, ahead of the MFMA run so its latency lands in the
    # same shadow a real read-modify-write C-init would use -- at k -> infinity, so the
    # slope is read off ONE arm. Read it with the fold OFF (`nored`): the loads cost
    # registers, and with the fold on that cost is a co-residency eviction, not the stream.
    wsq_rmw=False,
    # kv_refetch: PROBE (dQ/dK/dV stay EXACT), N extra re-reads of the band's owned K/V
    # packs per q block -- the one cost a q-outer loop nest would add. See KV_REFETCH.
    kv_refetch=0,
    # band_span: >0 = this launch owns ONE GROUP of that many kv bands (see _band_span_for).
    # K/V/DK/DV still span the whole kv axis; the group's first kv row arrives as a device
    # scalar in the CuSeqKv slot, and seq_len_k is the group's own extent, so the grid, the
    # band decode and the workspace all size themselves to the group.
    band_span=0,
    k_reg=True,  # feed GEMM1a's B from the K register packs, not the LDS tile (see K_REG)
    # s_ovl: overlay the dS ring on the band's K tile, whose elements are dead after the
    # prologue's one transpose read (see S_OVL). Frees the ring's whole allocation.
    s_ovl=False,
    # g3_kreg: hold GEMM3's whole K^T fragment set live for the band instead of reading it
    # back per head-step. K^T is head-invariant, so this is pure read removal. See G3_KREG.
    g3_kreg=False,
    # qdesc: walk the unmasked q blocks downwards (None = every full-causal shape, see QDESC).
    qdesc=None,
    qdesc_r=None,  # bands per QDESC phase group (None = QDESC_R)
    # num_xcd: PROBE, the XCD-major decode's fan-out (None = _NUM_XCD). It sets how many
    # (batch, kv_head) groups the resident work-groups spread over, i.e. how many bands of
    # one group share an XCD's L2 slice. See the block_id decode. Swept at l70b_b2 against
    # the deployed 8: 4 is +2.8%, 2 is +17.6%, 1 is +29.5%, and the byte counter agrees
    # (dkdv FetchSize 4.40 GB at 8 against 7.46 GB at 4, fold off) -- each XCD's Q/dO image
    # is private to its (batch, kv_head) groups, so the fan-out belongs at the physical
    # XCD count and there is nothing on this axis to tune.
    num_xcd=None,
    g3_dbat=None,
    g3d=None,  # GEMM3 kstep prefetch ring depth (None = 6, capped by G3_KSTEPS). See G3D.
    # q_pref: stage the Q/dO tiles through VGPRs and issue head h+1's fetch at the top of
    # head-step h, so a whole head-step covers it. See Q_PREF.
    q_pref=False,
    # q_alias: PROBE ONLY (dQ/dK/dV wrong, deterministic). Pin the Q/dO fetch's q row to the
    # band's own first row, keeping every instruction, address chain and LDS write in place
    # and only making the read hit. Prices what the per-band Q/dO re-read costs (see Q_ALIAS).
    q_alias=False,
    qpf_at=None,  # issue point of the next head-step's Q/dO fetch (None = per head dim, see QPF_AT)
    # win_fold: give the MASKED tiles the -log2e*lse C-init too (None = every shape, see WIN_FOLD).
    win_fold=None,
    # mask_cls: split the diagonal q-block's waves into below / diagonal / dead classes
    # (None = every shape that can prove the alignment, see MASK_ALIGN).
    mask_cls=None,
    # no_exp: PROBE ONLY (dQ/dK/dV wrong, deterministic). Drop the v_exp_f32 from the softmax
    # so P is the raw exponent. Prices the quarter-rate exp chain against the MFMA pipe.
    no_exp=False,
    # g3_defer: run GEMM3 one head-step late off a second dS slot. See G3_DEFER.
    g3_defer=True,
    g3_st_at=None,
    g3_st_n=None,
    g3_sb=None,
    g3_at=None,  # _hs_hook position of the deferred GEMM3 run. See G3_AT.
    g3_valu=None,  # VALU per MFMA in the GEMM3 co-execution pipeline. See G3_VALU.
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
    # bat_lo: dispatch only batches [bat_lo, bat_lo+batch_size), where batch_size is the
    # LAUNCH argument. A batch is a whole slab of every workspace this kernel writes (dQ
    # partials stride by band then batch, dk/dv slots carry batch as their own axis) and of
    # every tensor it reads, so restricting it is a pure grid restriction -- no tensor has to
    # be sliced, which is what an SBHD seq-major layout cannot do. dQ/dK/dV stay bitwise
    # identical to the whole-batch launch. See _fused_pipelined's tail cut.
    bat_lo=0,
    # flat_wg: work-group size, and on this part it is really a REGISTER-FILE choice.
    # 512 (8 waves) puts two waves on every SIMD, capping each at 256 architected
    # registers; 256 (4 waves, one per SIMD) opens the whole 512-register file to a
    # single wave via AGPRs. The wide fused band needs the second form: its wider
    # accumulator set only fits spill-free there, and one wave per SIMD also doubles
    # ROWS_PER_WAVE_KV so each transpose-read feeds more MFMAs. The two forms trade
    # per-cycle efficiency against clock roughly evenly on their own; what makes the
    # four-wave form the right pick is the registers per SIMD it leaves free, which is
    # what lets the dQ reduce co-reside (see `_dq_partial_ws` / `_fused_pipelined`).
    # 512 is also the only way this body can hold TWO waves per SIMD (four waves is one
    # per SIMD, so its 460 unified dwords against the 512 pool cost nothing), and it was
    # measured that way: 512 spills 72 dwords and reads gptoss_full 9.41 ms, while
    # 512 + g3_kreg=False fits at vgpr 254 / spill 0 -- a real two-wave allocation -- and
    # still reads 6.56 against 5.70. The second wave cannot pay for what splitting the band
    # costs it: the per-wave transpose-read count does not halve with ROWS_PER_WAVE_KV (the
    # q-side operands are read whole by every wave), so the CU's LDS read traffic doubles,
    # and dropping the resident K^T set to make room adds 256 more per trip.
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
    # buffer_load_dwordx4 ... lds (16B DMA-to-LDS) needs gfx950+ (gfx94x has only
    # the 4B dword variant). DMA bypasses the VGPR staging of the Q/dO tile loads,
    # relieving register pressure on this VGPR-locked (236 VGPR, occ ~2) kernel.
    ENABLE_DMA = not gpu_arch.startswith("gfx942")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    BLOCK_Q = _BWD_BLOCK_Q if block_q is None else int(block_q)
    WARP_SIZE = 64
    NUM_XCD = _NUM_XCD if num_xcd is None else int(num_xcd)
    BLOCK_KV = block_kv
    Q_SPLIT = q_split
    assert q_split >= 1
    N_QSP = Q_SPLIT if n_qsp is None else n_qsp
    QSP_LO = qsp_lo
    BAT_LO = int(bat_lo)
    assert 1 <= N_QSP <= Q_SPLIT and 0 <= QSP_LO <= Q_SPLIT - N_QSP
    assert BAT_LO == 0 or (batch_size is not None and 0 < BAT_LO < batch_size)
    flat_work_group_size = flat_wg
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    # A band is owned in KV_HALVES passes of BKV_H rows; only the dK/dV accumulators and
    # the dS staging exist per band, everything else is sized by the pass. Halving the pass
    # at the deployed band (kv_halves=2, bkv=256) halves the live S/dP/P/dS transient and
    # reads gptoss_full 6.15 against 5.70 (+7.9%): the q-side operands are re-read once per
    # pass, which is the same doubled LDS read traffic that costs the 8-wave form (flat_wg).
    KV_HALVES = max(1, int(kv_halves))
    assert KV_HALVES == 1 or BLOCK_KV % (KV_HALVES * NUM_WAVES * 16) == 0
    BKV_H = BLOCK_KV // KV_HALVES
    ROWS_PER_WAVE_KV = BKV_H // NUM_WAVES

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
    WSQ_PAD = int(wsq_pad)
    WSQ_ILV = int(wsq_ilv)
    WSQ_RING = int(wsq_ring)
    WSQ_NREC = max(1, int(wsq_nrec or 1))
    WSQ_ATOMIC = int(wsq_atomic or 0)
    WSQ_ACOAL = int(wsq_acoal or 0)
    WSQ_RMW = bool(wsq_rmw)
    KV_REFETCH = max(0, int(kv_refetch or 0))
    # A band group only shifts where this launch's kv rows sit; every other assumption of the
    # body has to still hold, so it rides the plain full-causal path only -- square SBHD, or
    # ragged (whose bands are per segment).
    BAND_SPAN = int(band_span)
    assert not BAND_SPAN or (window_left < 0 and (varlen or (sbhd and square)))
    # Where the group's kv rows are addressed FROM is what the two carriers differ in. A dense
    # launch is handed the group's own kv extent, so its rows are group-local and every causal
    # term is lifted back onto the axis. A ragged launch reads its extent from cu_seqlens and
    # addresses kv by the row's absolute position in the segment either way, so there the group
    # shifts the band INDEX alone and the rest of the body is already exact.
    BAND_LIFT = BAND_SPAN and not varlen
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
    # Depth 6 of the 8 available ksteps measures as the optimum on the fused band -- and on
    # the D64 fused path it is also the only depth that reaches the ISA: 4, 8 and 10 all
    # compile byte-identical kernels there (G3_KREG holds the whole band's K^T in registers,
    # so there is no read ring left for the depth to size). Do not read a wall difference
    # between those arms as a result; it is the same binary.
    G3D = min(6 if g3d is None else int(g3d), G3_KSTEPS)
    G3_WAVES = min(NUM_WAVES, max(1, DT * MT // min(4, DT * MT)))  # waves carrying GEMM3
    G3_TILES = max(1, DT * MT // G3_WAVES)  # output tiles per carrier wave
    # Re-pricing GEMM3's patch shape once K^T is band-resident (G3_KRT) is bound by the
    # co-residency budget, not by LDS read count: growing G3_DT multiplies every wave's
    # resident set, not just its read count, so shrinking the patch only pays off if it
    # frees enough registers for the co-resident dQ-reduce wave (`_dq_partial_ws`) to fit.
    G3_QT = min(MT, 2 if G3_TILES >= 2 else 1)  # q 16-tiles per wave
    G3_DT = G3_TILES // G3_QT  # D 16-tiles per wave
    G3_DBAT = G3_DT if g3_dbat is None else int(g3_dbat)
    G3_QGRP = MT // G3_QT  # q-tile groups; wave -> (D group, q group)
    # G3_SPLIT: run GEMM3 as one pass per q-half instead of once at the head-step's end,
    # to fill this body's one bare-MFMA VALU window. Disabled: LLVM hoists the early
    # pass's MFMAs to the top of its emission region instead of leaving them where aimed,
    # so the bare run grows instead of shrinking.
    G3_SPLIT = False
    G3_SPL_STRIDE = G3_QGRP if G3_SPLIT else 1  # q-tile stride within a wave's patch
    G3_SPL_AT = 1  # q-half whose GEMM1 the early pass is emitted after
    G3D_E = 3  # kstep prefetch depth of the early split pass
    # window_left>=0 (SWA) rides the fused path too: the G3 q-loop takes the windowed
    # upper bound (_qhi) and the reduce clamps the band range with a lower edge (g_lo).
    # Ragged varlen rides it as well: every base, descriptor and loop bound below is
    # already shadowed to the segment, so only the dQ workspace has to move from a
    # per-batch slab to rows packed by q token (see _wsq_band).
    assert head_dim in (64, 128)
    assert DT * MT == G3_WAVES * G3_DT * G3_QT, "GEMM3 tiles must partition over carriers"
    # A wave's D-tiles must pair up (and start even) for the permuted partial layout
    # the store in _gemm3_tiles and the reduce in build_flash_attn_bwd_dqred_module
    # both assume; G3_DT even makes _g3d0 = (wave/G3_QGRP)*G3_DT even too.
    assert G3_DT % 2 == 0, "permuted dQ partial layout needs an even G3_DT"
    assert G3_DBAT % 2 == 0 and G3_DT % G3_DBAT == 0, "a D-tile group holds whole pairs"
    assert BLOCK_KV % PV_K_STEP == 0
    assert batch_size is not None, "fused dQ needs compile-time B for the workspace stride"
    # D128's occ-1 recipe (dma_grp=2 + pf_ring) does not port here even though the
    # fused body is now one wave per SIMD too: the dS ring has no fence of its own,
    # it rides the PER-HEAD Q/dO staging barrier pair, and both alternatives either
    # trip this assert (dma_grp=2, which pays that pair once per two heads instead)
    # or fail the bitwise-determinism gate (pf_ring). See QDO_TAIL for why giving the
    # ring its own fence loses on this body's fence-trading economics.
    assert dma_grp == 1 or (head_dim == 128 and not g3_defer), (
        "fused dQ rides the per-head Q/dO staging barriers as the dS WAR fence"
    )

    # EXP_IGLP: at one wave/SIMD there's no sibling wave to hide exp2 latency under, so
    # hand the head-step region to LLVM's MFMAExpInterleave IGLP strategy instead of
    # hand-placed barriers. Gated to NUM_WAVES == 4 (see _dq_partial_ws for the call count).
    EXP_IGLP = NUM_WAVES == 4
    IGLP_EXP_INTERLEAVE = 2  # LLVM IGLPStrategyID::MFMAExpInterleaveID
    # G2_HALF: run GEMM2 once per q-half instead of once per head-step. Fused-only -- the
    # split bodies keep the single call so their ISA stays byte-identical; see the
    # emission point in _head_step_lds for what it buys.
    G2_HALF = True if g2_half is None else bool(g2_half)
    # FQ_PAIR: the fused body's four staggered waves attend q ranges one half-tile apart,
    # so a paired trip carries two of them in one tile (see _dma_bases's poff).
    FQ_PAIR = (
        window_left >= 0
        and square
        and head_dim == 64
        and NUM_WAVES == 4
        and Q_SPLIT == 1
        and ENABLE_DMA
        and not bool(pf_ring)
        and PV_K_STEPS == 2
        and BLOCK_Q // 2 == PV_K_STEP
        and ROWS_PER_WAVE_KV == BLOCK_Q // 2
        and BLOCK_KV == NUM_WAVES * (BLOCK_Q // 2)
        and window_left % BLOCK_Q == 0
        and window_left // (BLOCK_Q // 2) >= NUM_WAVES - 2
        and G3_QT * 2 == MT
    )
    FQ_HALF = BLOCK_Q // 2
    FQ_PAIR_POFF = ((window_left // FQ_HALF) + 1) * FQ_HALF if FQ_PAIR else 0
    FQ_PAIR_NX = (NUM_WAVES - 1) if FQ_PAIR else 0  # paired trips (one per stagger step)
    FQ_PAIR_NF = (((window_left // FQ_HALF) - NUM_WAVES + 2) // 2) if FQ_PAIR else 0
    # Q_BOUND: the windowed q-loop grid is BLOCK_KV-aligned, not BLOCK_Q-aligned, so when
    # BLOCK_Q doesn't divide BLOCK_KV (or FQ_PAIR fixes the trip count at NB) the band
    # nearest the sequence end walks past real rows and needs an explicit mask term.
    Q_BOUND = window_left >= 0 and (BLOCK_KV % BLOCK_Q != 0 or FQ_PAIR)
    # WIN_FOLD: let the masked tiles take the lse C-init too instead of the zero one. Legal only
    # with exp_intrin, whose select+exp2 is then the accumulator read that carries the MFMA
    # hazard wait the identity fma used to buy (see _p_of)..
    # Full causal takes it too: a diagonal q block is 4 of a band's q blocks, but its copy of
    # the q loop is the expensive one, and the zero C-init made every masked element pay an
    # identity fma AND the hazard nops that fma's accumulator read pulled in. Folding drops
    # the masked loop 6338 -> 5402 instructions and its s_nop waits 779 -> 222 cycles per
    # eight-head trip, leaving the unmasked loop byte-identical (+4 vgpr, 463, same 464
    # granule, spill 0). gptoss_full -0.98% over 14 palindrome pairs in three sessions,
    # 14 of 14 same sign; the windowed shapes already folded and are unchanged.
    WIN_FOLD = True if win_fold is None else bool(win_fold)
    # WIN_DIST: mask off one live band-relative distance instead of a live compare pair per element.
    WIN_DIST = window_left >= 0
    # WIN_KV_BELOW: a rectangle's low kv rows fall out of every q row's window, so give their K/V
    # a ZERO num_records to skip DRAM, and renumber kv_tile_idx among the live tiles so the
    # dispatch order stays dense over what is left. Both halves are priced by ONE rectangle:
    # the tile count they centre the renumbering on is the grid's, which on a ragged batch is
    # the PADDED count of the longest segment rather than this segment's own -- so a short
    # segment's live bands get renumbered towards the far end of its dispatch run, and the
    # machine walks that segment's empty tiles first. The ragged branch keeps the dense
    # ascending walk, where a band's neighbours are the ones re-reading its q rows.
    WIN_KV_BELOW = window_left >= 0 and not square and not varlen
    # WIN_XCD: give each XCD a contiguous dispatch run so its re-read bands stay in one L2 slice.
    WIN_XCD = window_left >= 0 and N_QSP == 1
    # s_waitcnt SIMM16 selecting lgkmcnt(0) alone: vmcnt/expcnt stay at their maxima, so the
    # wait retires the LDS traffic without also retiring in-flight global stores.
    WAIT_LGKM = 0xC07F

    assert BKV_H % NUM_WAVES == 0
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
    PF_RING = bool(pf_ring) and ENABLE_DMA
    # Q_PREF: fetch the Q/dO tile into VGPRs and ds_write it, instead of buffer_load ... lds.
    # The DMA route cannot be given a shadow at all: a pending buffer_load ... lds forces
    # vmcnt(0) before every later ds_read of the same LDS allocation, so wherever the issue
    # point is moved the drain reappears at the next LDS read (see G3_SHADOW, q_dbuf,
    # dma_grp=2 -- all measured losses). Through VGPRs the fetch is an ordinary VMEM load
    # with no LDS dependence, so head h+1's tile is issued at the top of head-step h and
    # only waited on at its ds_write one head-step later. Runs at LDS_SLOTS == 1: a 2-slot
    # ring would retire the staging pair's WAR barrier too, but the register cost of a
    # second live slot outweighs that barrier's price on this body.
    Q_PREF = bool(q_pref) and ENABLE_DMA and not PF_RING and DMA_GRP == 1
    # Q_ALIAS: probe. Every trip of a band re-fetches the SAME q rows, so the whole re-read
    # collapses to one tile per (band, head) and the difference against the real walk is the
    # re-read's price -- traffic and latency together (methodology/03 alias arm).
    Q_ALIAS = bool(q_alias) and Q_PREF
    # gfx950 has one in-order vmcnt, so at D128 this fetch issues at point 0 to keep dQ partial stores in flight.
    QPF_AT = (0 if HEAD_DIM == 128 else 2) if qpf_at is None else int(qpf_at)
    # PF_QB: the LAST head-step of a q-block has no next head to fetch for, so it issues
    # head 0's fetch of the NEXT q-block instead -- Q/dO and the group's (-delta, lse) --
    # riding the q-loop's iter_args. The fused body runs ONE work-group per CU, so nothing
    # else is resident to cover a q-block prologue; the last head-step is also where
    # register pressure is lowest, so the extra live values land in the cheapest spot.
    # A paired trip has no single next q block to prefetch, so FQ_PAIR opts out.
    PF_QB = Q_PREF and not FQ_PAIR
    # MASK_SKIP: let a wave sit out a diagonal q-block whose kv rows it cannot see. Its
    # P and dS are zero there, so this only removes work -- the output is bitwise equal.
    MASK_SKIP = window_left < 0
    # MASK_ALIGN: the diagonal q-block's wave classes, resolved per wave instead of
    # masking every live wave's whole 16-tile set.
    #
    # The masked q-block's first row and the band's first kv row both come off kv_start
    # (_q_loop_start = kv_start - causal_offset + m*BLOCK_Q, kv_row_wave = kv_start +
    # w*ROWS_PER_WAVE_KV), so their difference is m*BLOCK_Q - w*ROWS_PER_WAVE_KV with the
    # causal offset cancelling. That is an exact multiple of ROWS_PER_WAVE_KV whenever the
    # wave's kv extent DIVIDES BLOCK_Q -- not only when the two are equal -- so a wave is
    # always in exactly one of BLOCK_Q/R + 2 states, with no partially-aligned case:
    #   kvw + R <= q_first          every kv row is at or below the causal edge -> NO mask
    #   kvw == q_first + dn*R       aligned dn wave-rows above the block's first row: its
    #                               (mt, nt) 16-tiles split by nt + dn*NT vs mt alone
    #                               (fully masked above, masked on, clear below)
    #   kvw > q_last                dead (the existing zero-publishing arm)
    #
    # ★ So the R == BLOCK_Q gate below is SUFFICIENT, NOT NECESSARY -- and the general form
    # was BUILT AND MEASURED at D128 (BLOCK_Q/R = 2, so a clear arm plus TWO aligned arms)
    # and is a large LOSS, so the gate stays. l70b_b2, min of 40, one cell per process,
    # palindrome, same-binary control (spread 0.21% and 0.5%):
    #   two aligned arms, unfunded    6.8482 against 6.3283   +8.2%   (473 dwords)
    #   the same, red:vec=4 funded    6.8885 against 6.5325   +5.4%   (line moved to 480)
    # Both legs separated 3/3, and the funded leg is the one that prices the SPECIALISATION
    # rather than the co-residency it evicts: paying the reduce wave back still leaves +5.4%.
    # The cost is code REPLICATION, which the D64 reading below cannot see because one
    # aligned arm needs no extra copy of the region: three arms take the dkdv body from
    # 10220 to 14487 instructions and 2560 to 3968 static MFMA (+42% / +55%) while each wave
    # still executes ONE arm, so the DYNAMIC saving stays bounded by the 3.1% masked-trip
    # share and the static cost is not. At D128 each arm's region is twice as wide as at
    # D64, which is what inverts the sign. The next thing to try on this axis is an arm
    # count that does NOT grow with BLOCK_Q/R -- one aligned arm plus a masked fallthrough
    # for the rest -- priced on the ISA first, since this form spent 4267 instructions on a
    # <=3.1% dynamic prize.
    # At D64 this is worth an arm each because the four waves are barrier-locked into the same
    # head-step: the q-block costs the MAX over waves, and before this every live wave paid
    # the full 64-element mask compare/select set plus the MFMAs, exp2 chain and packs of
    # 16-tiles that can only be zero. At D64 the diagonal wave's in-branch MFMAs go
    # 128 -> 88 and the block's critical path 675 -> 560 instructions (560 = exactly what
    # an interior q-block costs, so the below waves set the floor), or -> 430 when the
    # diagonal wave is the only live one. dK/dV/dQ stay BITWISE equal to the single-arm
    # form (verified both head dims): every removed MFMA had an exactly-zero B operand.
    # The clamp in _kv_first_q (kv_start < causal_offset, i.e. a rectangular shape's lowest
    # bands) is the one case that breaks the exact multiple, hence square-only; BAND_LIFT
    # and the FQ_PAIR half-tile map re-base q the same way and are excluded for the same
    # reason.
    MASK_ALIGN = (
        MASK_SKIP
        and (square and not varlen and not BAND_SPAN and not FQ_PAIR)
        and ROWS_PER_WAVE_KV == BLOCK_Q
        and (True if mask_cls is None else bool(mask_cls))
    )
    # QDESC: walk the unmasked bulk of the q loop DOWNWARDS. A band's q range starts at its own
    # kv row, so ascending leaves every band on a different q block for the whole loop while
    # descending starts them all on the last one and walks them together. Paired against the
    # ascending walk, eight of eight alternating cycles over two sessions, gptoss_full -0.45%
    # on the mean of ten and -0.36% on the min of six; d64_hq32_b4 -0.61%, d64_hq64_b1 -0.64%,
    # d64_hq128_16k -2.34%, varlen_d64_u -0.27% (the wider the kv axis the more bands gather).
    # What it moves is the dQ fold, NOT the body: with the fold replaced by a fill, the
    # descending walk is 0.9% SLOWER (5.628 against 5.577 ms), while the fold's exposed cost
    # -- full minus that same probe -- drops from 409 to 331 us. The bands that share a q block
    # share its partial rows too, so they leave the fold denser rows to stream.
    # Windowed bands get their own phase rotation (see the window branch of the q loop), so
    # this is full-causal only.
    QDESC = (True if qdesc is None else bool(qdesc)) and window_left < 0
    # Bands per phase group. The phases keep the gathered bands off each OTHER's partial rows
    # and are not free to drop: R=1 (every band on the same q block) costs gptoss_full +1.6%
    # and l8b_b2 +0.8%, while widening past the concurrent sharers does nothing -- R=8 and
    # R=16 read flat to +1.0% against 4 at Hq=64, R=8 is +0.7% at Hq=32.
    #
    # What sets how many bands pile onto one q block's partial rows is the DWELL: a
    # work-group holds a q block for GQA_GROUP_SIZE head-steps, so halving the group halves
    # the pile and half the phases carry the same separation. Every R above was tuned at a
    # group of 8; at 4 the phase group is one step too wide and R=2 measures l8b_b2 -1.37% on
    # the mean and -1.55% on the min of 28 readings over three strict-palindrome batches of
    # min-of-40 (every batch negative on its own), and d64_hq32_b4 -0.1% -- flat, so the rule
    # costs the narrow-group D64 shapes nothing. Every GUARD cell has a group of 8, so they
    # all compile byte-identical to the literal 4 this replaces.
    QDESC_R = max(1, GQA_GROUP_SIZE // 2) if qdesc_r is None else int(qdesc_r)
    # GEMM1 emits ks-outer: the four kv tiles of one k-step first, so consecutive MFMAs
    # write different accumulators and the next one issues without waiting on the last
    # one's result. At D128 (one wave per SIMD) there is no sibling wave to cover that
    # latency at all; at D64 it still pays, together with PF_QB.
    # The fused body emits ks-inner instead: it frees 16 live dwords that are handed to the
    # co-resident dQ reduce so it can widen its own load (see uc in _dq_partial_ws). Not a
    # free choice any more either -- forcing ks-outer back on at D64 reads gptoss_full
    # 6.06 against 5.70 (+6.3%), so the ordering itself now costs, not just the registers.
    G1_KS_OUTER = (HEAD_DIM == 128) if g1_ks_outer is None else bool(g1_ks_outer)
    # QSP_ABS: phase the q-split onto ABSOLUTE q blocks (split s owns (q/BLOCK_Q) % Q_SPLIT == s)
    # -- the map the reduce and pipeline sub-range cut assume (see _qsp_absolute).
    QSP_ABS = _qsp_absolute(HEAD_DIM, BLOCK_KV, Q_SPLIT, BLOCK_Q)
    # QDO_TAIL: publish head h+1's Q/dO tile at the END of head-step h instead of at its
    # own start, so the drain + barrier that already fences dS publishes BOTH and the
    # head boundary's own pair disappears (2 rendezvous per head-step become 1). Needs a
    # second slot on each of the Q/dO and dS rings, since the writer is now a head ahead
    # of the reader. Disabled: even with the extra slot funded back out of GEMM3's
    # resident K^T set, a barrier is still cheaper here than any structure that removes
    # one -- the same conclusion G3_DEFER reaches independently below.
    QDO_RING = False
    QDO_TAIL = QDO_RING  # the merged publish the second slots exist for
    QDO_PP = False
    LDS_SLOTS = (2 * DMA_GRP) if PF_RING else DMA_GRP
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
    # A group with fewer than BLOCK_SIZE // BLOCK_Q heads has fewer elements to stage than
    # the work-group has threads, so cap a head's span at one element per thread. The
    # threads past LD_ACTIVE then repeat the first ones -- same source, same LDS address,
    # same value -- so the duplicate loads and stores are idempotent and cost no branch.
    # At larger groups LD_ACTIVE is BLOCK_SIZE and the wrap below is not emitted at all.
    LD_THREADS_PER_HEAD = min(BLOCK_SIZE // GQA_GROUP_SIZE, LD_HEAD_ELEMS)
    LD_VEC = LD_HEAD_ELEMS // LD_THREADS_PER_HEAD
    LD_ACTIVE = GQA_GROUP_SIZE * LD_THREADS_PER_HEAD
    assert BLOCK_SIZE % GQA_GROUP_SIZE == 0 and LD_HEAD_ELEMS % LD_THREADS_PER_HEAD == 0
    # buffer_load takes power-of-two vectors up to dwordx4, so a per-thread run that is not
    # one (block_q=96 leaves 6 floats) is issued as its greedy power-of-two pieces.
    LD_CHUNKS = []
    _ld_rem = LD_VEC
    while _ld_rem:
        _ld_c = min(4, 1 << (_ld_rem.bit_length() - 1))
        LD_CHUNKS.append((LD_VEC - _ld_rem, _ld_c))
        _ld_rem -= _ld_c

    # The Q/dO slot ring, the K, V and dS tiles all share one element-indexed
    # view so every reader (_a_idx / _read_tr / _kv_lds_idx / _g3s_idx) addresses them the
    # same way.
    LDS_VIEW_ELEMS = LDS_TOTAL * LDS_SLOTS
    # Staging GEMM1b's V B-operand through LDS instead of holding its register packs live for
    # the band was measured against the LDS that S_OVL frees: it does NOT free the architected
    # registers it is supposed to (the ISA reads 487 dwords against 462, +25) because LLVM
    # keeps the packs live to feed the ds_write anyway, and it costs +9.5% of wall at D128.
    V_LDS = False
    # K_REG: GEMM3 transpose-reads the staged K tile, but GEMM1a's B operand can come
    # from the register packs that filled it instead of being re-read from LDS once per
    # q-half (the LDS copy stays -- this is a read-side choice, not a staging one). This
    # is the one asymmetry in this body's LDS accounting: deleting reads elsewhere is
    # free, but adding these back costs, because they would land inside GEMM1's MFMA run
    # as fresh SrcB dependencies and break its issue density. What this body pays for is
    # MFMA-run density, not LDS latency or read count.
    K_REG = bool(k_reg)
    G3_KREG = bool(g3_kreg)
    # D-tiles of the band-resident K^T set; the rest are re-read every head-step.
    G3_KRT = G3_DT if G3_KREG else 0
    G3K_BASE = LDS_VIEW_ELEMS  # prescaled K [BLOCK_KV][HEAD_DIM]
    G3V_BASE = G3K_BASE + BLOCK_KV * HEAD_DIM  # V [BLOCK_KV][HEAD_DIM]
    # dS [slot][BLOCK_KV][BLOCK_Q]. Staging it over the K tile's read-once elements (S_OVL
    # below) buys NO occupancy on its own: the ISA allocates 462 unified VGPRs (256 arch
    # saturated + 206 accum, no spill), so one wave per SIMD is what the register file says
    # whatever the LDS says. The 81920 B two-work-group line is out of reach through LDS at
    # any dS packing; reaching it needs a body that fits 256 registers, which is 206
    # accumulator dwords away. The overlay is a donor for a candidate that needs LDS, not a
    # candidate on its own.
    G3S_BASE = G3V_BASE + (BLOCK_KV * HEAD_DIM if V_LDS else 0)
    G3S_SLOT_ELEMS = BKV_H * BLOCK_Q
    G3S_GRP_ELEMS = KV_HALVES * G3S_SLOT_ELEMS
    # g3_defer: let GEMM3 lag one head-step behind the head that produced dS and read the
    # OTHER slot, so its RAW edge is covered by the head boundary's own staging barrier
    # pair (drain + publish) and its WAR edge by the pair one step later. Both dS fences
    # per head-step then disappear, for the price of a second dS slot. That trade paid
    # while the body ran eight waves per work-group and loses at four: the extra slot's
    # registers and LDS have no sibling MFMA run left to hide the retired fences under.
    G3_DEFER = bool(g3_defer)
    # G3_AT: _hs_hook position of the deferred GEMM3. 0 = the head-step top; 3*pks+1 puts
    # its MFMAs on q-half pks's dS/pack window, the body's one bare-VALU run. Only the
    # DEFERRED path has a call to move, and at D64 deferring is itself gptoss_full +3.5%
    # (5.931 against 5.731, three palindrome rounds) -- moving it to position 1 on top of
    # that is +5.0%, which is why the D64 exp window is filled from the undeferred emission
    # point instead. At D128 the deferred path IS the deployed one, and there the move is
    # priced on the ISA rather than inherited from D64: see G3_VALU for what the position
    # buys and what it costs.
    G3_AT = 0 if g3_at is None else int(g3_at)
    # G3_VALU: request an MFMA<->VALU co-execution pipeline (sched_group_barrier group 1,
    # `(MFMA 1, VALU n)` repeated) over the window that holds GEMM3's MFMA run and the
    # dS/pack VALU block. GEMM3 is the only matrix work in the head-step that depends on
    # nothing this q-half computes, so it is the only run those packs can hide under; the
    # in-half alternatives are all measured losses (see _half_gemm1's docstring). 0 = off,
    # which is byte-identical. The forward kernel drives its exp/VALU blocks the same way
    # (`_sched_barrier_pairs`); the backward has only ever named MFMA and DS_READ, so its
    # VALU has always been free-floating and PMC measures it 5.6% co-executed against the
    # exp segment's 34.3%.
    #
    # BUILT AND MEASURED at D128, l70b_b2, and the pipeline WORKS while the wall does not.
    # ISA, fraction of the body's VALU that sits inside an MFMA run (gap threshold 8):
    #
    #   G3_AT=0 (deployed)  462 dw  60.3% hidden    G3_AT=4          462 dw  61.8% hidden
    #   G3_AT=1             466 dw  58.8%           G3_AT=1,G3_VALU=3 472 dw  57.7%
    #
    # so the position DOES deal the packs out under the run -- 553 of the 1182 pack/multiply
    # instructions move inside one, `SQ_INSTS_VALU` and `SQ_INSTS_MFMA` unchanged, spill 0.
    # The wall goes the other way on every arm, min of 40, strict palindrome, same-binary
    # control spread 0.53% (fold on) / 0.36% (fold off):
    #
    #   fold on   G3_AT=4 6.4837 against 6.3212   +2.6%      G3_AT=4,G3_ST_AT=5  +11.9%
    #   fold off  G3_AT=4 5.5473 against 5.4498   +1.8%      (so it is not co-residency)
    #   gptoss_full D64 (undeferred path, unreachable)       byte-identical, 5.7749
    #
    # Mechanism, from the dumps: GEMM3's run is not idle time the packs can be poured into.
    # Its 32 MFMAs are already 1:1 interleaved with the 32 `ds_read_b64_tr_b16` that feed
    # them, so a `(MFMA 1, VALU n)` request competes with the DS_READ pairing rather than
    # filling a hole, and the pack block's own producers then wait one whole GEMM3 pass.
    # The exp segment hides 34.3% because `iglp_opt(2)` interleaves it into GEMM1's run,
    # where the operands are already resident in registers and there is no LDS ladder to
    # displace. => the exposed-VALU pool (0.318 ms, R17-3) is real but POSITION does not
    # collect it, and this is the sixth position-only lever to price at <=0 on this body
    # (goal 3.4). The next thing to try on the pool is not where the block sits but how
    # many instructions it is: the 864 `v_cvt_pk_bf16_f32` are two per written dQ dword
    # pair, so the lever with a different mechanism is the dQ partial's LAYOUT (one pack
    # per 4 dwords instead of 2) rather than its schedule.
    #
    # ALSO MEASURED, and the reason to keep this knob rather than delete it: the ISA
    # hidden-VALU fraction moved 1.5 points in the right direction while the wall lost
    # 1.8-2.6%, on the same four builds. `SQ_VALU_MFMA_COEXEC_CYCLES` therefore carries no
    # sign information for an accept decision here, the same way `FetchSize` does not on
    # the dispatch-order axis (goal 3.2). Gate on it, then judge on the wall.
    G3_VALU = 0 if g3_valu is None else int(g3_valu)
    G3_MFMA = G3_DT * G3_QT * G3_KSTEPS  # MFMAs one whole GEMM3 pass emits per wave
    G3_ST_AT = -1 if g3_st_at is None else int(g3_st_at)
    G3_ST_N = G3_DT // 2 * G3_QT if g3_st_n is None else int(g3_st_n)
    assert G3_ST_AT < 0 or G3_WAVES == NUM_WAVES
    G3_SB = 0 if g3_sb is None else int(g3_sb)
    G3S_SLOTS = 2 if (G3_DEFER or QDO_RING or DMA_GRP > 1) else 1
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
    HS_WAR_BAR = G3_DEFER and not QDO_TAIL and not QDO_PP
    # S_OVL: the band's prescaled K tile is read EXACTLY once. K_REG feeds GEMM1a's B from the
    # register packs that filled the tile, and G3_KREG holds every D-tile a wave reads live for
    # the whole band, so after the prologue's transpose read no head-step touches those
    # elements again. Laying the dS ring over them costs one release rendezvous per band --
    # amortised over the band's whole q walk -- and frees the ring's entire allocation:
    # 102400 -> 69632 B at D128, the only LDS donor this body has that is not a scheduling
    # knob (the g3d / g3_dbat ring depths hold no LDS at any depth). It buys nothing by
    # itself -- the wall reads +0.4% and one wave per SIMD is what the 462-dword register
    # allocation says whatever the LDS says -- so it is off until a candidate needs the
    # 32768 B, and the price it is measured at is that one rendezvous.
    S_OVL = bool(s_ovl) and K_REG and G3_KREG and G3_KRT >= G3_DT and not V_LDS
    if S_OVL:
        G3S_BASE = G3K_BASE
    _KV_LDS_END = G3V_BASE + (BLOCK_KV * HEAD_DIM if V_LDS else 0)
    LDS_VIEW_ELEMS = max(G3S_BASE + G3S_SLOTS * G3S_GRP_ELEMS, _KV_LDS_END)

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
        WSQ: fx.Tensor,  # dQ partials [kv_band, B, Sq, Hq, D] bf16
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        total_kv: fx.Int32,  # whole kv axis: dk/dv slot stride when this launch owns a slice of it
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
        # rows stay co-resident. Bijective when B*NUM_HEADS_KV % NUM_XCD == 0; WIN_XCD gets the
        # same locality via xcd_remap_pid when it doesn't, else falls back to the plain decode.
        num_kv_tiles = (seq_len_k_v + BLOCK_KV - 1) // BLOCK_KV
        if const_expr(NUM_HEADS_KV % NUM_XCD != 0 and WIN_XCD):
            _lin = fx.Index(xcd_remap_pid(block_id, fx.Index(gpu.grid_dim.x), NUM_XCD))
            split_idx = fx.Index(QSP_LO)
            kv_tile_idx = _lin % num_kv_tiles
            _u = _lin // num_kv_tiles
            kv_head_idx = _u % NUM_HEADS_KV
            batch_idx = _u // NUM_HEADS_KV
        elif const_expr(NUM_HEADS_KV % NUM_XCD == 0):
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
        # A batch sub-range launch keeps the decode above and shifts its result: the grid
        # carries the COUNT (the launch's batch_size) and this carries the base, so every
        # address downstream is the one the whole-batch launch would have formed. See bat_lo.
        if const_expr(BAT_LO):
            batch_idx = batch_idx + fx.Index(BAT_LO)
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
            # Packed dQ partials stride by the WHOLE q token axis, whose length is the
            # last cu_seqlens entry -- one more wave-uniform scalar load, rather than an
            # argument slot every non-fused body would have to carry as well.
            total_q_v = fx.Index(
                fx.Int32(buffer_ops.buffer_load(_cuq_rsrc, fx.Index(batch_size), vec_width=1, dtype=fx.Int32))
            )
            if const_expr(BAND_SPAN):
                # This pass's first band, one entry past the segment table (see _cu_band_rows).
                band_lo_v = fx.Index(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            _cukv_rsrc, fx.Index(batch_size + 1), vec_width=1, dtype=fx.Int32
                        )
                    )
                )
        else:
            q_tok_base = batch_idx * seq_len_q_v
            kv_tok_base = batch_idx * seq_len_k_v
        causal_offset = seq_len_k_v - seq_len_q_v
        if const_expr(BAND_LIFT):
            # Band group: seq_len_k is the GROUP's kv extent, so every kv row this body sees
            # is group-local and sits _kv_lift rows further down the real kv axis. The causal
            # edge follows: a local row attends q >= local + lift. causal_offset (group extent
            # minus Sq) is negative here and unusable as an unsigned index, so the shift is
            # kept positive and added on the kv side instead, and the signed mask offset --
            # the only causal term the body evaluates per element -- is just its negation.
            _kv_lift = (
                fx.Index(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            buffer_ops.create_buffer_resource(CuSeqKv, max_size=True),
                            fx.Index(0),
                            vec_width=1,
                            dtype=fx.Int32,
                        )
                    )
                )
                * BLOCK_KV
            )
            causal_off_i32 = fx.Int32(0) - fx.Int32(_kv_lift)
        seq_len_q_i32 = fx.Int32(seq_len_q_v)
        if const_expr(WIN_KV_BELOW):
            _dead = fx.Index(
                ArithValue(causal_offset > fx.Index(window_left)).select(
                    causal_offset - fx.Index(window_left), fx.Index(0)
                )
            ) // fx.Index(BLOCK_KV)
            _live = ArithValue(kv_tile_idx >= _dead)
            _lpt_i = fx.Index(_live.select(kv_tile_idx - _dead, fx.Index(0)))
            _lpt_mid = (num_kv_tiles - _dead) // fx.Index(2)
            _lpt_h = _lpt_i // fx.Index(2)
            _lpt_j = fx.Index(
                ArithValue((_lpt_i & fx.Index(1)) == fx.Index(0)).select(
                    _lpt_mid + _lpt_h, _lpt_mid - fx.Index(1) - _lpt_h
                )
            )
            kv_tile_idx = fx.Index(_live.select(_dead + _lpt_j, kv_tile_idx))
        if const_expr(BAND_SPAN and varlen):
            # The group owns the segment's bands [lo, lo+span): only the tile index is
            # group-local, the kv rows it names are the segment's own. Its dQ partial slot
            # stays the group-local one (see the WSQ descriptor), which is what bounds the
            # workspace to the group, while anything keyed on WHICH band this is (the q-loop
            # rotation below) has to stay keyed on the band's place in the whole axis.
            kv_band_idx = kv_tile_idx + band_lo_v
        else:
            kv_band_idx = kv_tile_idx
        kv_start = kv_band_idx * BLOCK_KV
        # This wave owns ROWS_PER_WAVE_KV kv rows, split into NT 16-wide N-tiles.
        # In the 16x16 layout the owned kv row for a lane is nt*16 + lane16.
        kv_row_wave = kv_start + wave_id * ROWS_PER_WAVE_KV

        def global_idx_kv(token_idx, col):
            return token_idx * RD_STRIDE_KV + kv_head_idx * HEAD_DIM + col

        def kv_row_of(nt, h=0):
            return kv_row_wave + fx.Index(h * BKV_H + nt * N_TILE) + lane16

        def kv_row_i32_of(nt, h=0):
            return fx.Int32(kv_row_of(nt, h))

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
        SCORED_PACK = True
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
        # LDS element delta of one half-tile of q rows (FQ_PAIR): whole 8-row groups, so the
        # packed image steps by ROW_BLK blocks per N_TILE rows.
        FQ_PAIR_HALF = (BLOCK_Q // 2 // N_TILE) * ROW_BLK * PBLK
        # _g3_tr steps the packed image by ROW_BLK*PBLK per N_TILE rows: PBLK//2 at D64 (2 rows
        # per 128 block) but full PBLK at D128 (one row per block); a literal PBLK//2 halves the D128 read.
        G3_KROW_STRIDE = ROW_BLK * PBLK // N_TILE

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
        if const_expr(sbhd and BAND_LIFT):
            # The group enters K/V at its own first kv row; num_records then covers exactly
            # the group, so the loads of a band that the causal edge leaves empty fall out.
            _kv_batch_byte_off = _raw(
                (batch_idx * fx.Index(STRIDE_TOKEN_KV) + _kv_lift * fx.Index(RD_STRIDE_KV)) * fx.Index(2)
            )
        elif const_expr(sbhd):
            _kv_batch_byte_off = _raw(batch_idx * fx.Index(STRIDE_TOKEN_KV * 2))
        else:
            _kv_batch_byte_off = _raw(kv_tok_base * fx.Index(STRIDE_TOKEN_KV * 2))
        _kv_rd_nrec = _kv_nrec_bytes
        if const_expr(WIN_KV_BELOW):
            _win_dead_end = ArithValue(causal_offset > fx.Index(window_left)).select(
                causal_offset - fx.Index(window_left), fx.Index(0)
            )
            _kv_rd_nrec = _raw(
                fx.Index(
                    ArithValue(kv_start + fx.Index(BLOCK_KV) > _win_dead_end).select(
                        seq_len_k_v * fx.Index(RD_STRIDE_KV * 2), fx.Index(0)
                    )
                )
            )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=_kv_rd_nrec, base_byte_offset=_kv_batch_byte_off
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=False, num_records_bytes=_kv_rd_nrec, base_byte_offset=_kv_batch_byte_off
        )
        # DK/DV point at this split's slot of the [B, q_split, S, Hkv, D] workspace
        # (slot index = batch*q_split + split_idx); one WG writes it exactly once.
        if const_expr(sbhd and BAND_LIFT):
            # Same slot, but a band group's launch only knows its OWN kv extent, while the
            # slot stride is the whole axis -- which is what total_kv carries here.
            _dkv_ws_byte_off = _raw(
                (
                    (split_idx * fx.Index(total_kv) + _kv_lift) * fx.Index(RD_STRIDE_KV)
                    + batch_idx * fx.Index(STRIDE_TOKEN_KV)
                )
                * fx.Index(2)
            )
        elif const_expr(sbhd):
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
        # This band's slot for THIS q slice, whose num_records also clips the store of
        # the tail q-block when the slice is not a whole number of BLOCK_Q.
        # Dense: a band holds one slab per batch. Ragged: rows are packed by q token, so
        # a band spans total_q rows and the segment enters at its own token base --
        # exactly the shift dk/dv take above, and the same one-writer-per-slot property.
        _wsq_row = fx.Index(NUM_HEADS_Q * HEAD_DIM * 2 * WSQ_ILV)
        _wsq_slice = seq_len_q_v * _wsq_row
        if const_expr(varlen):
            _wsq_band = total_q_v * _wsq_row + fx.Index(WSQ_PAD)
            _wsq_off = q_tok_base * _wsq_row
        else:
            _wsq_band = _wsq_slice * fx.Index(batch_size or 1) + fx.Index(WSQ_PAD)
            _wsq_off = batch_idx * _wsq_slice
        _wsq_grp = kv_tile_idx // fx.Index(WSQ_ILV) if WSQ_ILV > 1 else kv_tile_idx
        if const_expr(WSQ_RING):
            _wsq_grp = _wsq_grp % fx.Index(WSQ_RING)
        wsq_rsrc = buffer_ops.create_buffer_resource(
            WSQ,
            max_size=False,
            num_records_bytes=_raw(_wsq_slice // fx.Index(WSQ_NREC) if WSQ_NREC > 1 else _wsq_slice),
            base_byte_offset=_raw(_wsq_grp * _wsq_band + _wsq_off),
        )
        if const_expr(WSQ_ATOMIC):
            # The fp32 dQ accumulator has NO band axis and no interleave: every band adds into
            # the same image, so one batch slice of Sq*Hq*D floats is the whole resource.
            _wsq32_slice = seq_len_q_v * fx.Index(NUM_HEADS_Q * HEAD_DIM * 4)
            wsq32_rsrc = buffer_ops.create_buffer_resource(
                WSQ,
                max_size=False,
                num_records_bytes=_raw(_wsq32_slice),
                base_byte_offset=_raw(
                    (q_tok_base * fx.Index(NUM_HEADS_Q * HEAD_DIM * 4))
                    if const_expr(varlen)
                    else (batch_idx * _wsq32_slice)
                ),
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

            def _dma_bases(tile_start, poff=None):
                """Head-independent part of the Q/dO DMA byte offset, one per batch.

                Only the q_head term differs between GQA heads sharing a q-block, so hoisting
                the row/swizzle/column derivation collapses each head's DMA to a single add
                and takes the kernel's scratch spill to zero.

                poff (FQ_PAIR) sources the tile's second half-tile poff rows further on
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
                        _dst = buffer_ops.get_element_ptr(_dma_lds_base, lds_dst + d * DMA_BATCH_BYTES)
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
        k_b_packs = [[[None] * K_STEPS_QK for _ in range_constexpr(NT)] for _ in range_constexpr(KV_HALVES)]
        v_b_packs = [[[None] * K_STEPS_QK for _ in range_constexpr(NT)] for _ in range_constexpr(KV_HALVES)]
        for h in range_constexpr(KV_HALVES):
            for nt in range_constexpr(NT):
                _kvr = kv_row_of(nt, h)
                for ks in range_constexpr(K_STEPS_QK):
                    kv_col = fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K
                    k_b_packs[h][nt][ks] = buffer_ops.buffer_load(
                        k_rsrc, global_idx_kv(_kvr, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                    )
                    v_b_packs[h][nt][ks] = buffer_ops.buffer_load(
                        v_rsrc, global_idx_kv(_kvr, kv_col), vec_width=MFMA_LANE_K, dtype=elem_dtype
                    )

        # ---- FOLD: prescale the owned K by sm*log2e once per kv-block (amortized over
        # the GQA group's heads). K feeds GEMM1a only -- dK is a separate accumulator --
        # so scaling k_b_packs is safe. Together with -log2e*lse folded into GEMM1a's
        # C-init, GEMM1a's raw output already IS the base-2 softmax exponent. ----
        if const_expr(True):
            _kscale_v8 = Vec.filled(MFMA_LANE_K, sm_scale * _LOG2E, fx.Float32)
            for h in range_constexpr(KV_HALVES):
                for nt in range_constexpr(NT):
                    for ks in range_constexpr(K_STEPS_QK):
                        k_b_packs[h][nt][ks] = (
                            (Vec(k_b_packs[h][nt][ks]).to(fx.Float32) * _kscale_v8).to(elem_dtype).ir_value()
                        )

        # GEMM3 contracts over kv, so its A operand is K^T: stage the owned K (and, for
        # GEMM1b, V) into LDS as [kv][D] in the Q/dO tile layout and transpose-read it
        # back, once per kv-block -- a pure register->LDS repack that takes both B
        # operands off the register file for the whole kernel, avoiding the spill
        # cliff. K goes in ALREADY PRESCALED, so GEMM1a reads it directly; that leaves
        # the dQ partial scaled by sm*log2e, which `_reduce_dq_partials` divides out.
        for h in range_constexpr(KV_HALVES):
            _kb_h, _vb_h = G3K_BASE + h * BKV_H * HEAD_DIM, G3V_BASE + h * BKV_H * HEAD_DIM
            for nt in range_constexpr(NT):
                for ks in range_constexpr(K_STEPS_QK):
                    Vec(k_b_packs[h][nt][ks]).store(lds, [_kv_lds_idx(_kb_h, nt, ks)])
                    if const_expr(V_LDS):
                        Vec(v_b_packs[h][nt][ks]).store(lds, [_kv_lds_idx(_vb_h, nt, ks)])
        if const_expr(not K_REG):
            k_b_packs = [G3K_BASE + h * BKV_H * HEAD_DIM for h in range_constexpr(KV_HALVES)]
        if const_expr(V_LDS):
            v_b_packs = [G3V_BASE + h * BKV_H * HEAD_DIM for h in range_constexpr(KV_HALVES)]

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
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
            if const_expr(no_exp):
                return fx.Float32(_raw(x))
            return fx.Float32(
                llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", [_raw(x)], [], [])
            )

        def _p_of(s_r, lse_t, apply_mask):
            assert apply_mask, "FOLD bulk uses the hazard-anchored path in _head_step_lds"
            s_r = fmath.fma(s_r, fx.Float32(1.0), lse_t, fastmath=fm_fast)
            return _vexp(s_r)

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
            return _opaque_idx(a_base + lane16 * fx.Index(PBLK) + (kg * MFMA_LANE_K ^ a_swz_mask))

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

        def _gemm_qk(a_base, b_packs, inits=None, mts=None, pin=None, drop=None):
            """S[mt][nt] (v4f32) = A(Q/dO)[mt] @ B(owned K/V)[nt]^T over D. inits[mt]
            optionally pre-loads the accumulator (folds -delta into the dP GEMM for free).
            mts restricts work to a subset of the MT q-tiles (per-half GEMM1); the
            output is keyed by mt so [2,3] halves index correctly.

            drop(mt, nt) marks a 16-tile whose result is not read at all (a fully masked
            tile on the diagonal wave, see MASK_ALIGN): its output stays None and its
            MFMAs are never emitted.

            b_packs is either a register list or, for a tile staged in LDS, its base: the
            fragments are then re-read per head-step so they are live only across this GEMM
            rather than across the whole kernel."""
            _mts = list(range_constexpr(MT)) if mts is None else list(mts)
            _nts = {mt: [nt for nt in range_constexpr(NT) if drop is None or not drop(mt, nt)] for mt in _mts}
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
                if const_expr(not _nts[mt]):
                    continue
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
                    for nt in _nts[mt]:
                        out[mt][nt] = c_zero_v4f32 if inits is None else inits[mt]
                for ks in range_constexpr(K_STEPS_QK):
                    for mt in _mts:
                        for nt in _nts[mt]:
                            out[mt][nt] = mfma_acc(a[mt][ks], b_packs[nt][ks], out[mt][nt])
            else:
                for mt in _mts:
                    for nt in _nts[mt]:
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
        TR_PIN = True
        # See _pin_bases: hoist the pinned bases themselves out of the q-loop. Needs a
        # single Q/dO slot, so the base is not a function of the head.
        HOIST_PIN = LDS_SLOTS == 1 or QDO_RING or QDO_PP

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

        def _gemm3(q_start, head_local, slot, drain=None, qsel=None, depth=None, st_sink=None, poff=None):
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
                    _gemm3_tiles(q_start, head_local, slot, drain, qsel, depth, poff=poff)
                if const_expr(drain is not None):
                    if wave_id >= fx.Index(G3_WAVES):
                        drain()
            else:
                _gemm3_tiles(q_start, head_local, slot, drain, qsel, depth, st_sink, poff)

        def _gemm3_tiles(
            q_start, head_local, slot, drain=None, qsel=None, depth=None, st_sink=None, poff=None
        ):
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
            _soff = slot * G3S_GRP_ELEMS
            # qsel picks ONE of the wave's q-tiles (its dS columns come from that q-half
            # alone), so the pass can run as soon as that half's softmax has published.
            _qs = list(range_constexpr(G3_QT)) if qsel is None else [qsel]

            def _g3_frags(kk, dsel):
                # GEMM3's transpose reads are free, like GEMM1's and GEMM2's: a probe that
                # pairs the ksteps so the odd one's reads CSE onto the even one's (wrong dQ,
                # but 1536 -> 1024 tr at an untouched MFMA count) measures 6/11 -- the last
                # read family this body had not priced. Do not spend a round on read count.
                # K^T is head-INVARIANT, so G3_KREG holds all G3_DT*G3_KSTEPS fragments live
                # for the band and reads them once instead of once per head-step. It is ON
                # for the D64 fused band, so the hot loop already issues NO K^T LDS read at
                # all (turning it off puts 256 ds_read back per q-block trip) -- a read-ahead
                # ring for K^T has nothing left to hide, and what is left to prefetch is dS,
                # which the depth-G3D ring below already runs 6 ksteps ahead.
                _ka = [
                    _g3kt[kk][i] if const_expr(G3_KREG and i < G3_KRT) else _g3_tr(_kb, i, kk, G3_KROW_STRIDE)
                    for i in dsel
                ]
                return _ka, [_g3_tr(_sb, j * G3_SPL_STRIDE, kk, BLOCK_Q, _soff) for j in _qs]

            # kstep prefetch ring, depth G3D: kk+G3D's transpose-reads are issued before
            # kk's MFMAs so the ds_read_tr16 latency lands in the MFMA shadow instead of at
            # every kstep's first MFMA -- the same trade GEMM2's g2d ring makes, paid for by
            # the registers the pinned bases above freed. Unlike GEMM2's ring, this one
            # wants no sched_group_barrier or s_setprio around it: both cost more in spill
            # or scheduling freedom than they save, since the scheduler's own default burst
            # already keeps the run's lgkmcnt deep. An early split pass carries its ring
            # across the softmax it covers, so its depth is priced against that live range.
            _gd = const_expr(G3D if depth is None else min(depth, G3_KSTEPS))
            _dgs = [list(range_constexpr(_g, _g + G3_DBAT)) for _g in range_constexpr(0, G3_DT, G3_DBAT)]
            # Store the partial out of the dQ^T C-layout, a PAIR of D-tiles per store.
            # The C-layout alone hands a lane only 32 B of a q row per instruction -- half
            # of a 64 B request slot, doubling fabric traffic. A wave's G3_DT tiles are
            # adjacent and even-aligned, so the workspace's D axis is instead PERMUTED to
            # dperm = I*16 + kg*8 + p*4 + t for tile I+p (see `_dq_partial_ws`): a lane's 8
            # bf16 become contiguous and the 4 kg lanes of a row cover a full 64 B in ONE
            # dwordx4, with no LDS trip or barrier needed to get there.
            # The store stays CACHED: the partials are read back by the reduce, and a
            # non-temporal policy would lose the L2 write-combining the 64 B pairing sets up.
            # Re-checked under the pipelined fold, where that reason no longer holds: every
            # CPol on this store is inside the noise channel, so the write stream is not
            # displacing the q window QDESC gathers either.
            _g3qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)

            def _g3_elem(i, j, ilv=True):
                """This store's element offset into the partial slot (see the D permutation).

                ``ilv=False`` drops the band interleave, i.e. addresses a plain dQ image --
                which is what the fp32 atomic accumulator is.
                """
                _g3t = _g3q0 + fx.Index(j * G3_SPL_STRIDE)
                _g3row = q_start + _g3_qrow(_g3t)
                if const_expr(poff is not None):
                    _g3row = _g3row + (_g3t >> fx.Index(1)) * (poff - fx.Index(BLOCK_Q // 2))
                _g3col = (_g3d0 + fx.Index(i)) * fx.Index(D_TILE) + kg * fx.Index(8)
                _g3qrow = _g3row * fx.Index(NUM_HEADS_Q) + _g3qh
                if const_expr(WSQ_ILV > 1 and ilv):
                    _g3qrow = _g3qrow * fx.Index(WSQ_ILV) + kv_tile_idx % fx.Index(WSQ_ILV)
                return _g3qrow * fx.Index(HEAD_DIM) + _g3col

            def _g3_store(_g3p, i, j):
                buffer_ops.buffer_store(
                    _g3p.ir_value(),
                    wsq_rsrc,
                    _g3_elem(i, j) * fx.Index(2),
                    offset_is_bytes=True,
                )

            def _g3_atomic(_lo, _hi, i, j):
                """The same D-tile pair, added into the fp32 dQ image instead of stored.

                The permuted D axis pays off here too: the pair's 8 floats are 8 CONSECUTIVE
                dwords, so one voffset plus an immediate soffset covers the lot and the 4 kg
                lanes of a q row still cover 128 B between them. No bf16 pack, no rounding of
                the partial -- the accumulation itself is fp32, which is what keeps dQ inside
                its 1 dB SNR margin once the fold's fixed band order is gone.
                """
                if const_expr(WSQ_ACOAL):
                    # A PROBE ON A PROBE, and dQ is wrong under either. Same eight atomics per
                    # lane, same dwords, same image -- only the address pattern moves, from 64
                    # lanes scattered over 32 lines (each taking two dwords of one) to 64 lanes
                    # covering one contiguous 256 B run. The wave's base is its own q row and
                    # D-tile with the lane terms dropped, so the footprint stays the real dQ
                    # image rather than collapsing somewhere cache-resident. This isolates the
                    # 8x line amplification from the payload, which is the whole question:
                    # the payload is arithmetic and cannot be tuned away, the amplification is
                    # a layout and can (that is what aiter's 64 v_permlane16_swap_b32 buy).
                    _t = _g3q0 + fx.Index(j * G3_SPL_STRIDE)
                    _row0 = (
                        q_start
                        + ((_t >> fx.Index(1)) << fx.Index(5))
                        + ((_t & fx.Index(1)) << fx.Index(3))
                    )
                    _e0 = (_row0 * fx.Index(NUM_HEADS_Q) + _g3qh) * fx.Index(HEAD_DIM) + (
                        _g3d0 + fx.Index(i)
                    ) * fx.Index(D_TILE)
                    _a32 = ArithValue(_raw((_e0 + lane) * fx.Index(4))).index_cast(
                        fx.Int32.ir_type
                    )
                    _step = 256
                else:
                    _a32 = ArithValue(_raw(_g3_elem(i, j, ilv=False) * fx.Index(4))).index_cast(
                        fx.Int32.ir_type
                    )
                    _step = 4
                for _w in range_constexpr(4):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        _raw(_lo[_w]), wsq32_rsrc, _a32, _step * _w, WSQ_ATOMIC - 1
                    )
                for _w in range_constexpr(4):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        _raw(_hi[_w]), wsq32_rsrc, _a32, _step * (4 + _w), WSQ_ATOMIC - 1
                    )

            for _gi in range_constexpr(len(_dgs)):
                _dg = _dgs[_gi]
                _g3 = [[c_zero_v4f32 for _ in _qs] for _ in range_constexpr(len(_dg))]
                _rmw = []
                if const_expr(WSQ_RMW):
                    # Issued here, a whole GEMM3 run ahead of the store that shares its
                    # address, and ONE PER STORE (the store packs a d-tile pair into its
                    # 16 B, so the loop is over pairs) so the arm's extra volume is exactly
                    # the partial volume. bf16 element offsets are multiples of 8, so >> 1
                    # is the same address in f32 units and the load is the store's own line.
                    _rmw = [
                        buffer_ops.buffer_load(
                            wsq_rsrc,
                            _g3_elem(_dg[2 * i2], _qs[jj]) >> fx.Index(1),
                            vec_width=4,
                            dtype=fx.Float32,
                        )
                        for i2 in range_constexpr(len(_dg) // 2)
                        for jj in range_constexpr(len(_qs))
                    ]
                _ring = [_g3_frags(kk, _dg) for kk in range_constexpr(_gd)]
                for _kk in range_constexpr(G3_KSTEPS):
                    _g3k, _g3s = _ring[_kk % _gd]
                    if const_expr(_kk + _gd < G3_KSTEPS):
                        _ring[_kk % _gd] = _g3_frags(_kk + _gd, _dg)
                    for i in range_constexpr(len(_dg)):
                        for jj in range_constexpr(len(_qs)):
                            _g3[i][jj] = mfma_acc(_g3k[i], _g3s[jj], _g3[i][jj])
                if const_expr(drain is not None and _gi == 0):
                    drain()
                for i2 in range_constexpr(len(_dg) // 2):
                    i = 2 * i2
                    for jj in range_constexpr(len(_qs)):
                        j = _qs[jj]
                        _gd0 = _dg[i]
                        if const_expr(WSQ_ATOMIC):
                            _lo, _hi = Vec(_g3[i][jj]), Vec(_g3[i + 1][jj])
                            if const_expr(st_sink is None):
                                _g3_atomic(_lo, _hi, _gd0, j)
                            else:
                                st_sink.append(
                                    lambda a=_lo, b=_hi, _i=_gd0, _j=j: _g3_atomic(a, b, _i, _j)
                                )
                            continue
                        _g3p = bf16_trunc_scored_v4(_g3[i][jj]).shuffle(
                            bf16_trunc_scored_v4(_g3[i + 1][jj]), [0, 1, 2, 3]
                        )
                        if const_expr(st_sink is None):
                            _g3_store(_g3p, _gd0, j)
                        else:
                            st_sink.append(lambda p=_g3p, _i=_gd0, _j=j: _g3_store(p, _i, _j))
                if const_expr(WSQ_RMW):
                    _keepalive_v4(_rmw)

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
                [_g3_tr(_g3kb, i, kk, G3_KROW_STRIDE) for i in range_constexpr(G3_KRT)]
                for kk in range_constexpr(G3_KSTEPS)
            ]
            if const_expr(S_OVL):
                # The dS ring lies over these elements, so every wave's read of the tile has
                # to retire before the first head-step publishes dS into them.
                rocdl.s_waitcnt(WAIT_LGKM)
                gpu.barrier()

        H_ACCS = KV_HALVES * DT * NT
        dv_accs = [c_zero_v4f32 for _ in range_constexpr(H_ACCS)]
        dk_accs = [c_zero_v4f32 for _ in range_constexpr(H_ACCS)]

        # Bottom-right causal: first query attending this kv-tile = max(0, kv_start-offset).
        # A band group's rows are square-causal against q = local row + lift (see _kv_lift).
        if const_expr(BAND_LIFT):
            _kv_first_q = kv_start + _kv_lift
        else:
            _kv_first_q = ArithValue(kv_start >= causal_offset).select(kv_start - causal_offset, fx.Index(0))
        if const_expr(QSP_ABS):
            _qsp_ph = (_kv_first_q // fx.Index(BLOCK_Q)) % fx.Index(Q_SPLIT)
            _q_loop_start = _kv_first_q + (
                (split_idx + fx.Index(Q_SPLIT) - _qsp_ph) % fx.Index(Q_SPLIT)
            ) * fx.Index(BLOCK_Q)
        else:
            _q_loop_start = _kv_first_q + split_idx * fx.Index(BLOCK_Q)
        _kv_end = kv_start + fx.Index(BLOCK_KV)
        _kv_end_c = ArithValue(_kv_end < seq_len_k_v).select(_kv_end, seq_len_k_v)
        _step = Q_SPLIT * BLOCK_Q
        if const_expr(BAND_LIFT):
            _masked_upper = _kv_end_c + _kv_lift
        else:
            _masked_upper = ArithValue(_kv_end_c >= causal_offset).select(
                _kv_end_c - causal_offset, fx.Index(0)
            )
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
        if const_expr(QDESC):
            _u_span = ArithValue(seq_len_q_v > _unmask_start).select(seq_len_q_v - _unmask_start, fx.Index(0))
            _u_trips = (_u_span + fx.Index(_step - 1)) // fx.Index(_step)
            _u_nb = fx.Index(ArithValue(_u_trips > fx.Index(0)).select(_u_trips, fx.Index(1)))
            _u_top = _unmask_start + (_u_nb - fx.Index(1)) * fx.Index(_step)
            _rot = kv_band_idx % fx.Index(QDESC_R)

            def _desc_q(trip):
                return _u_top - ((trip + _rot) % _u_nb) * fx.Index(_step)

        # The GQA head axis is unrolled INSIDE each q_start body so head h+1's GEMM1/exp2 is
        # emitted in the same straight-line block as head h's GEMM2 and schedules into its
        # MFMA shadow; accumulating dv/dk across heads is a pure reassociation (det-neutral).
        ld_lds = SmemPtr(base_ptr, ld_off, fx.Float32.ir_type, shape=(LD_ELEMS,)).get()
        # Thread t owns LD_VEC consecutive q of one GQA head.
        _ld_tid = tid if const_expr(LD_ACTIVE == BLOCK_SIZE) else (tid % fx.Index(LD_ACTIVE))
        _ld_head = _ld_tid // fx.Index(LD_THREADS_PER_HEAD)
        _ld_q = (tid % fx.Index(LD_THREADS_PER_HEAD)) * fx.Index(LD_VEC)

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
            # qoff (FQ_PAIR half-step) selects the tile half this wave is running.
            _i = fx.Index(arr * LD_ARR_ELEMS + head_local * LD_HEAD_ELEMS + mt * M_TILE) + kg * fx.Index(4)
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

        def _qdo_src_elem(q_start, head_local, d, poff=None):
            """Element index of this thread's 16 B slice of copy batch d, DMA lane mapping.

            Mirrors _dma_bases exactly, so the LDS image -- and therefore every reader and
            the kernel's output -- is unchanged; only the transport differs. poff pairs the
            tile's two half-tiles from non-adjacent q offsets, as _dma_bases does.
            """
            if const_expr(poff is not None and d >= NUM_DMA_Q // 2):
                q_start = q_start + poff - fx.Index(BLOCK_Q // 2)
            if const_expr(Q_ALIAS):
                q_start = kv_start
            _blk = tid // fx.Index(16) + fx.Index(d * ROWS_PER_DMA_BATCH)
            if const_expr(PACK_2ROW):
                _row = (
                    fx.Index(8) * (_blk >> fx.Index(2))
                    + (_blk & fx.Index(3))
                    + ((tid % fx.Index(16)) // fx.Index(8)) * fx.Index(4)
                )
            else:
                _row = _blk
            _col = ((tid % fx.Index(16)) * fx.Index(8)) ^ ((_row & fx.Index(7)) << fx.Index(4))
            _qh = kv_head_idx * fx.Index(GQA_GROUP_SIZE) + fx.Index(head_local)
            return global_idx_q(q_start + _row, _col, _qh)

        def _qdo_issue(q_start, head_local, poff=None):
            """Issue (no wait) head_local's Q/dO tile pair into VGPRs."""
            return [
                buffer_ops.buffer_load(
                    rsrc, _qdo_src_elem(q_start, head_local, d, poff), vec_width=8, dtype=elem_dtype
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
            hsel=None,
            nq=None,
        ):
            if const_expr(KV_REFETCH and head_local == 0):
                # PROBE (dK/dV/dQ stay EXACT -- the values are pinned live and discarded):
                # the ONE cost a q-outer walk would add. This body is kv-outer, so the
                # owned K/V B-operand packs are loaded once per band and reused for every
                # (q block, head) trip; q-outer reloads them per q block instead. Re-issue
                # exactly that set here, at the top of a q block ahead of the whole
                # GQA_GROUP_SIZE head loop, which is where the flip would put it and the
                # most cover it could ever get. One level adds the band's whole K+V once
                # per q block = 2 * BLOCK_KV * HEAD_DIM * 2 B per trip, so the wall slope
                # per level IS the flip's bill. `_opaque_idx` on the row keeps LICM from
                # hoisting it back out of the q loop.
                _refetch = [
                    buffer_ops.buffer_load(
                        _r,
                        global_idx_kv(
                            _opaque_idx(kv_row_of(nt, h)),
                            fx.Index(ks * K_STEP_QK) + kg * MFMA_LANE_K,
                        ),
                        vec_width=MFMA_LANE_K,
                        dtype=elem_dtype,
                    )
                    for _ in range_constexpr(KV_REFETCH)
                    for h in range_constexpr(KV_HALVES)
                    for nt in range_constexpr(NT)
                    for ks in range_constexpr(K_STEPS_QK)
                    for _r in (k_rsrc, v_rsrc)
                ]
                _keepalive_v4([Vec(_p).bitcast(fx.Float32).ir_value() for _p in _refetch])
            # The next head's Q/dO fetch: the earlier it is issued the more of this step
            # covers it, and the longer its 16 B per tensor stay live over the body's
            # register peak (GEMM2's accumulators). QPF_AT picks that trade.
            # [0] = Q/dO for the next head-step, [1] = (-delta, lse) for the next q-block.
            _qdo_next = [None, None]

            def _qdo_pf(at=0):
                if const_expr(not Q_PREF or at != QPF_AT):
                    return
                if const_expr(head_local + 1 < GQA_GROUP_SIZE):
                    _qdo_next[0] = _qdo_issue(q_start, head_local + 1, poff)
                elif const_expr(PF_QB):
                    # Same issue point, next q-block's head 0. Rows past the sequence end
                    # are clamped by the slice's num_records (they read 0 and the block
                    # they belong to never runs), so the tail iteration needs no guard.
                    _nq = q_start + fx.Index(_step) if nq is None else nq
                    _qdo_next[0] = _qdo_issue(_nq, 0)
                    _qdo_next[1] = _stage_ld_issue(_nq)

            q_start_i32 = fx.Int32(q_start)
            kg_off_i32 = fx.Int32(kg) * fx.Int32(4)
            _slot_lds = fx.Index((head_local % LDS_SLOTS) * LDS_TOTAL)
            q_lds = _slot_lds
            # FQ_PAIR: the paired trip's second half-tile holds rows q_start + poff instead
            # of the contiguous q_start + BLOCK_Q/2, and each wave runs only its own half,
            # so the LDS base, (-delta, lse) rows and mask q index all shift accordingly.
            _pk_list = [0] if const_expr(half) else list(range_constexpr(PV_K_STEPS))
            _hq = None
            _hs = None
            if const_expr(half):
                _hs = wave_id if const_expr(hsel is None) else hsel
                _hq = _hs * fx.Index(BLOCK_Q // 2)
                q_lds = q_lds + _hs * fx.Index(FQ_PAIR_HALF)
            do_lds = q_lds + fx.Index(LDS_DO_BASE)

            def _ld_rd(mt, arr):
                return _ld_read(head_local, mt, arr, _hq)

            def _q_slot_i32(mt):
                if const_expr(half):
                    return q_start_i32 + fx.Int32(_hs * poff) + fx.Int32(mt * M_TILE)
                _o = fx.Int32(mt * M_TILE)
                if const_expr(poff is not None and mt >= MT // 2):
                    _o = _o + fx.Int32(poff) - fx.Int32(BLOCK_Q // 2)
                return q_start_i32 + _o

            if const_expr(Q_PREF):
                # qdo already holds this head's tile, fetched one head-step ago. The
                # ds_write is the only point that waits on it, and the next head's fetch
                # is issued right after so it gets this whole step as its shadow.
                _ldv = None
                if const_expr(head_local == 0):
                    _ldv = ldv if const_expr(PF_QB) else _stage_ld_issue(q_start, poff)
                if const_expr(HS_WAR_BAR):
                    gpu.barrier()  # WAR: the previous head's GEMM2 still read this slot
                # Under QDO_TAIL only head 0 publishes here; every later head's tile was
                # committed at the end of the previous head-step and published by that
                # step's dS barrier. Moving JUST the ds_write back into the previous
                # step's GEMM3 run (same two barriers, no second ring slot) loses: the
                # staged tile then has to stay live across GEMM1/GEMM2 instead.
                if const_expr(not QDO_TAIL or head_local == 0):
                    _qdo_commit(qdo, _slot_lds)
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
                        _gemm3(q_start, head_local - 1, (head_local - 1) % G3S_SLOTS, _rdv_drain, poff=poff)
                    else:
                        _rdv_drain()
                else:
                    for _sh in stage_heads:
                        _vgpr_load_head(_sh, q_start)
                    if const_expr(head_local == 0):
                        _stage_ld_commit(_ldv)
                gpu.barrier()  # DMA + ld_lds commit visible before GEMM1 reads

            _g3_pend = []
            _g3_call = []

            def _hs_hook(pos):
                if const_expr(G3_AT == pos and len(_g3_call) > 0):
                    _g3_call.pop()()
                if const_expr(G3_VALU > 0 and G3_AT > 0 and pos == G3_AT + 1):
                    # GEMM3 ran one hook back and the q-half's dS/pack block sits between
                    # then and here, so both are in this scheduling region: deal the block's
                    # VALU out under GEMM3's MFMA run instead of leaving them to trail it.
                    # Group 1 -- group 0 is GEMM2's MFMA/DS_READ pipeline and appending to it
                    # would shift every one of its pairs.
                    for _ in range_constexpr(G3_MFMA):
                        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 1)
                        rocdl.sched_group_barrier(_SCHED_VALU_MASK, G3_VALU, 1)
                if const_expr(G3_ST_AT >= 0 and pos >= G3_ST_AT):
                    _hs_flush(G3_ST_N)

            def _hs_flush(n=None):
                _n = len(_g3_pend) if n is None else min(n, len(_g3_pend))
                for _i in range_constexpr(_n):
                    _g3_pend[_i]()
                del _g3_pend[:_n]

            def _hs_drain():
                if const_expr(len(_g3_call) > 0):
                    _g3_call.pop()()
                _hs_flush()

            if const_expr(G3_DEFER and head_local > 0 and not G3_SHADOW):
                # The PREVIOUS head's dQ, emitted at the TOP of this head-step. Its dS tile
                # was published by the staging pair above, so GEMM3 needs no fence of its
                # own, and the same pair one step later fences the read against the head
                # that reuses the slot. Positioned here for two reasons: the dQ partial
                # stores get a whole GEMM1+GEMM2 of slack before the next drain retires
                # them (gfx950 shares one vmcnt between loads and stores), and GEMM3's ring
                # and accumulators die before GEMM1a's fragments go live, so the two
                # register peaks no longer add -- which is what pays for the ring depth.
                _g3_call.append(
                    lambda: _gemm3(
                        q_start,
                        head_local - 1,
                        const_expr((head_local - 1) % G3S_SLOTS),
                        st_sink=_g3_pend if const_expr(G3_ST_AT >= 0) else None,
                    )
                )
                if const_expr(G3_SB & 1):
                    rocdl.sched_barrier(0)
                _hs_hook(0)
                if const_expr(G3_SB & 2):
                    rocdl.sched_barrier(0)

            _qdo_pf(1)

            # GEMM1a/exp2/GEMM1b/dS/pack per q-HALF (one pks = two mt packing into one
            # GEMM2 K=32 step): processing 2 of the MT q-tiles at a time halves the live
            # S/dP/P/dS transient that pinned dkdv at spill, so the kernel fits spill-free.
            # lse/-delta are pulled from LDS at their use points (only the 2 v4f32 this
            # half consumes are ever live). Pure re-ordering -> bit-identical, det-neutral.
            p_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            ds_pack = [[None] * NT for _ in range_constexpr(PV_K_STEPS)]
            _H = [0]
            if const_expr(HOIST_PIN):
                _g3wb = _pins["g3w"]
            else:
                _g3wb = _g3s_wbase()

            def _flat_accs():
                _h = _H[0]
                return [dv_cur[_h][dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)] + [
                    dk_cur[_h][dt][nt] for dt in range_constexpr(DT) for nt in range_constexpr(NT)
                ]

            def _set_accs(vals):
                _h = _H[0]
                for dt in range_constexpr(DT):
                    for nt in range_constexpr(NT):
                        dv_cur[_h][dt][nt] = vals[dt * NT + nt]
                        dk_cur[_h][dt][nt] = vals[DT * NT + dt * NT + nt]

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
                _dvh, _dkh = dv_cur[_H[0]], dk_cur[_H[0]]
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
                            _read_tr(do_lds, dt + g2d, pk_list[i], _do_trb) for i in range_constexpr(_nk)
                        ]
                    for i in range_constexpr(_nk):
                        for nt in range_constexpr(NT):
                            if const_expr(p_pack[pk_list[i]][nt] is None):
                                continue  # zero pack (see MASK_ALIGN)
                            _dvh[dt][nt] = mfma_acc(do_tr[i], p_pack[pk_list[i]][nt], _dvh[dt][nt])
                    if const_expr(NT >= 3):
                        # NT>=3 pins the packs' liveness hard enough that the RA sinks the
                        # pack next to the MFMA that reads it as SrcB. Pinning the dV group
                        # live past its MFMAs blocks that sinking, which reduces spill even
                        # now that the scored pack makes the sink itself legal. Naming fewer
                        # than all four elements of each tuple saves v_accvgpr reads but
                        # measures neutral, so all four stay; the pin is dV-only (dK regresses).
                        _keepalive_v4([_dvh[dt][nt] for nt in range_constexpr(NT)])
                    if const_expr(_rd_next):
                        q_tr_n = [_read_tr(q_lds, dt + g2d, pk_list[i], _q_trb) for i in range_constexpr(_nk)]
                    for i in range_constexpr(_nk):
                        for nt in range_constexpr(NT):
                            if const_expr(ds_pack[pk_list[i]][nt] is None):
                                continue  # zero pack (see MASK_ALIGN)
                            _dkh[dt][nt] = mfma_acc(q_tr[i], ds_pack[pk_list[i]][nt], _dkh[dt][nt])
                    if const_expr(_rd_next):
                        # Grouping the whole read set ahead of the MFMA run loses, even
                        # though it drops half the run's s_waitcnt lgkmcnt(2), because the
                        # read burst blocks MFMA issue. Dropping the hints entirely and
                        # letting the default scheduler place the run is worse still, so
                        # this pair is load-bearing, not decorative.
                        # Scale the hints by the MFMAs actually emitted, not by _nk*NT: a
                        # skipped zero pack has no MFMA for a read to interleave with.
                        _hn = sum(
                            1
                            for _i in range_constexpr(_nk)
                            for _n in range_constexpr(NT)
                            if p_pack[pk_list[_i]][_n] is not None
                        )
                        for _ in range_constexpr(_n_out * _hn):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(1)
                        do_ring[_slot] = do_tr_n
                        q_ring[_slot] = q_tr_n
                rocdl.s_setprio(0)

            if const_expr(HOIST_PIN):
                _slot_pin = const_expr(head_local % LDS_SLOTS)
                _q_trb, _do_trb = _pins["q"][_slot_pin], _pins["do"][_slot_pin]
                if const_expr(half):
                    _q_trb = _q_trb + _hs * fx.Index(FQ_PAIR_HALF)
                    _do_trb = _do_trb + _hs * fx.Index(FQ_PAIR_HALF)
            else:
                _q_trb = _tr_base(q_lds) if const_expr(TR_PIN) else None
                _do_trb = _tr_base(do_lds) if const_expr(TR_PIN) else None
            _q_apin = _a_pin(q_lds) if const_expr(A_PIN) else None
            _do_apin = _a_pin(do_lds) if const_expr(A_PIN) else None

            def _gemm_dp(half, drop=None):
                return _gemm_qk(
                    do_lds,
                    v_b_packs[_H[0]],
                    inits={mt: _ld_rd(mt, 0) for mt in half},
                    mts=half,
                    pin=_do_apin,
                    drop=drop,
                )

            def _half_gemm1(half, cls):
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
                    # Not a tuning hint but a load-bearing one: with the strategy off the body costs
                    # +8.9%, ID0 +10.2% and ID3 +9.3%, and ID1 core-dumps LLVM. Anything that reorders
                    # the head step has to keep it.
                    rocdl.iglp_opt(IGLP_EXP_INTERLEAVE)

                def _drop(mt, nt):
                    # A fully masked 16-tile's S and dP are read by nothing (see MASK_ALIGN).
                    return cls(mt, nt) == 2

                if const_expr(not apply_mask or WIN_FOLD):
                    _st = _gemm_qk(
                        q_lds,
                        k_b_packs[_H[0]],
                        inits={mt: _ld_rd(mt, 1) for mt in half},
                        mts=half,
                        pin=_q_apin,
                        drop=_drop,
                    )
                else:
                    _st = _gemm_qk(
                        q_lds,
                        k_b_packs[_H[0]],
                        mts=half,
                        pin=_q_apin,
                        drop=_drop,
                    )
                _dpt = _gemm_dp(half, _drop)
                # Extending the GEMM2 s_setprio(1) pair over this run too (so a SIMD's two
                # waves also de-phase across GEMM1) is 7/11 then 6/11 = noise, even though it
                # halves the hazard nops (198 -> 102): the pair only pays where one wave has
                # an MFMA run its sibling does not, which is GEMM2 and the carriers' GEMM3.
                return _st, _dpt

            # Only the masked tiles consult it (see _mask_lanes), so an interior q block does
            # not pay for the chain.
            if const_expr(WIN_DIST and apply_mask):
                _wdist = [
                    kv_row_i32_of(0, _h)
                    - _q_slot_i32(0)
                    - kg_off_i32
                    - causal_off_i32
                    + fx.Int32(window_left)
                    for _h in range_constexpr(KV_HALVES)
                ]

            def _mask_lanes(mt, nt, t):
                """Lanes of one GEMM1a accumulator element that must not contribute.

                Head-invariant, so the compares hoist out of the unrolled GQA head loop and
                only the resulting lane masks are carried through it.
                """
                if const_expr(WIN_DIST):
                    _off = fx.Int32(mt * M_TILE + t - nt * N_TILE)
                    _dd = _wdist[_H[0]] - _off
                    _mm = ArithValue(
                        arith.ori(
                            _raw(ArithValue(_dd > fx.Int32(window_left))),
                            _raw(ArithValue(_dd < fx.Int32(0))),
                        )
                    )
                    if const_expr(Q_BOUND):
                        _qs = _q_slot_i32(mt) + kg_off_i32 + fx.Int32(t)
                        _mm = ArithValue(arith.ori(_raw(_mm), _raw(ArithValue(_qs >= seq_len_q_i32))))
                    return _mm
                q_slot = _q_slot_i32(mt) + kg_off_i32 + fx.Int32(t)
                _up = ArithValue(kv_row_i32_of(nt, _H[0]) > q_slot + causal_off_i32)
                if const_expr(window_left < 0):
                    return _up
                _lo = ArithValue(kv_row_i32_of(nt, _H[0]) < q_slot + causal_off_i32 - fx.Int32(window_left))
                _mm = ArithValue(arith.ori(_raw(_up), _raw(_lo)))
                if const_expr(Q_BOUND):
                    _mm = ArithValue(arith.ori(_raw(_mm), _raw(ArithValue(q_slot >= seq_len_q_i32))))
                return _mm

            def _half_soft(pks, half, s_tiles, dp_tiles, cls):
                """softmax -> dS -> bf16 pack (-> dS publish) for one q-half.

                Returns the GEMM2 transpose-read ring this half primed, or None.
                """
                ma, mb = half
                P = [[None] * NT for _ in range_constexpr(MT)]
                if const_expr(not apply_mask or WIN_FOLD):
                    for mt in half:
                        for nt in range_constexpr(NT):
                            if const_expr(cls(mt, nt) == 2):
                                continue  # P == 0: no exp2, no dS, no pack (see MASK_ALIGN)
                            s_v = Vec(s_tiles[mt][nt])
                            if const_expr(cls(mt, nt) == 1):
                                P[mt][nt] = [
                                    _vexp_intrin(_mask_lanes(mt, nt, t).select(c_neg_inf, fx.Float32(s_v[t])))
                                    for t in range_constexpr(4)
                                ]
                            elif const_expr(True):
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
                        lse_v = _ld_rd(mt, 1)
                        for nt in range_constexpr(NT):
                            if const_expr(cls(mt, nt) == 2):
                                continue
                            s_v = s_tiles[mt][nt]
                            p_vals = []
                            for t in range_constexpr(4):
                                s_r = fx.Float32(Vec(s_v)[t])
                                if const_expr(cls(mt, nt) == 1):
                                    s_r = _mask_lanes(mt, nt, t).select(c_neg_inf, s_r)
                                p_vals.append(_p_of(s_r, fx.Float32(Vec(lse_v)[t]), apply_mask))
                            P[mt][nt] = p_vals

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
                        [[_read_tr(q_lds, _d, _p, _q_trb) for _p in _pk_seg] for _d in range_constexpr(g2d)],
                    )

                for nt in range_constexpr(NT):
                    if const_expr(P[ma][nt] is None and P[mb][nt] is None):
                        # Both q-tiles of this GEMM2 K-step are fully masked, so the pack is
                        # exactly zero: GEMM2 skips it (adding a zero B leaves dK/dV alone)
                        # and the dS row still has to be published, GEMM3 contracting the
                        # WHOLE band. Same zeros the dead-wave arm writes.
                        p_pack[pks][nt] = None
                        ds_pack[pks][nt] = None
                        _dsv = Vec.from_elements([fx.Int32(0) for _ in range_constexpr(4)], fx.Int32).bitcast(
                            elem_dtype
                        )
                    else:
                        _z4 = [c_zero_f for _ in range_constexpr(4)]
                        _ds = [
                            [_fmul(P[mt][nt][t], Vec(dp_tiles[mt][nt])[t]) for t in range_constexpr(4)]
                            if const_expr(P[mt][nt] is not None)
                            else _z4
                            for mt in half
                        ]
                        p_pack[pks][nt] = bf16_trunc_pack_v8(
                            (P[ma][nt] if const_expr(P[ma][nt] is not None) else _z4)
                            + (P[mb][nt] if const_expr(P[mb][nt] is not None) else _z4)
                        )
                        ds_pack[pks][nt] = bf16_trunc_pack_v8(_ds[0] + _ds[1])
                        _dsv = ds_pack[pks][nt]
                    # Publish dS as [kv][qp] for GEMM3's transpose-read. The v8 pack is
                    # q = {ma,mb}*16 + kg*4 + t of ONE kv row, which the qp permutation
                    # lays out as ONE 8-wide run -> a single ds_write_b128 (see
                    # _g3s_wbase). The run index is bit 5 of the column, hence pks*32.
                    _g3wo = (
                        nt * N_TILE * BLOCK_Q
                        + (head_local % G3S_SLOTS) * G3S_GRP_ELEMS
                        + _H[0] * G3S_SLOT_ELEMS
                    )
                    if const_expr(FQ_PAIR and _hs is not None):
                        _qx = _hs * fx.Index(2 * M_TILE)
                        _ds_write_vec(_g3wb ^ _qx, _g3wo, _dsv)
                        _ds_write_vec(
                            (_g3wb ^ _qx) ^ fx.Index(2 * M_TILE),
                            _g3wo,
                            Vec.from_elements([fx.Int32(0) for _ in range_constexpr(4)], fx.Int32).bitcast(
                                elem_dtype
                            ),
                        )
                    else:
                        _ds_write_vec(_g3wb ^ fx.Index(pks * 2 * M_TILE), _g3wo, _dsv)

                return _rings

            # GEMM2 per q-half, consuming a half's packs as soon as they exist. The
            # per-accumulator half order stays pks-ascending -> bit-identical, and the read
            # and MFMA counts are untouched. Two things move: the packs die a half earlier,
            # relieving the next half's GEMM1 register peak, and a half's GEMM2 MFMAs land
            # next to the NEXT half's GEMM1a and exp chain, filling what the ISA otherwise
            # shows as a bare VALU window. Flushing GEMM2 later still -- once that next
            # half's MFMA pipe is already full -- loses outright, so it is adjacency plus
            # the register relief that pays here, not interleaving for its own sake.
            def _pks_chain(pf=True, g3_split=False, hooks=False, mcls=None):
                # mcls(mt, nt) = 0 clear / 1 masked / 2 fully masked, per 16-tile. None is
                # the uniform "every tile like the q-block" default (see MASK_ALIGN).
                cls = mcls
                if const_expr(cls is None):

                    def cls(mt, nt):
                        return 1 if const_expr(apply_mask) else 0

                _rings = None
                for pks in _pk_list:
                    half = [2 * pks, 2 * pks + 1]
                    _at = 3 * pks
                    # Half pks-1's dQ pass, emitted one q-half after its dS was published:
                    # its 16 MFMAs are the only independent matrix work available to the
                    # last half's softmax tail. AT=0 puts it ahead of this half's GEMM1,
                    # AT=1 between GEMM1 and the softmax it is meant to cover.
                    if const_expr(g3_split and pks > 0 and G3_SPL_AT == 0):
                        _gemm3(q_start, head_local, 0, qsel=pks - 1, depth=G3D_E)
                    _st, _dpt = _half_gemm1(half, cls)
                    if const_expr(g3_split and pks > 0 and G3_SPL_AT == 1):
                        _gemm3(q_start, head_local, 0, qsel=pks - 1, depth=G3D_E)
                    if const_expr(hooks):
                        _hs_hook(_at + 1)
                    _rings = _half_soft(pks, half, _st, _dpt, cls)
                    if const_expr(hooks):
                        _hs_hook(_at + 2)
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
                        if const_expr(hooks):
                            _hs_hook(_at + 3)

                if const_expr(not G2_HALF):
                    if const_expr(pf):
                        _qdo_pf(2)
                    _gemm2(
                        list(_pk_list),
                        _rings[0],
                        _rings[1],
                        const_expr(PF_RING and mid_pf is not None),
                    )

            for _h in range_constexpr(KV_HALVES):
                if const_expr(_h):
                    rocdl.sched_barrier(0)
                _H[0] = _h
                _last_pass = const_expr(_h == KV_HALVES - 1)
                if const_expr(MASK_SKIP and apply_mask):
                    _hs_drain()
                    # Diagonal q-block: a wave whose kv rows all sit above the causal edge
                    # has P = dS = 0, so it publishes exact zeros into its dS rows (GEMM3
                    # contracts the WHOLE band) and the output stays bitwise identical.
                    if const_expr(_last_pass):
                        _qdo_pf(2)

                    _base = [_flat_accs()]

                    def _live():
                        _pks_chain(pf=False)
                        return _flat_accs()

                    def _arm(mcls):
                        """One wave class's chain, accumulating onto the class before it."""

                        # B023: `_base` is a per-iteration cell and every closure below is
                        # consumed by the `_if_wave` calls in this same iteration, so there is
                        # no late binding to bind. Same idiom as `_H` above.
                        def _run():
                            _set_accs(_base[0])  # noqa: B023
                            _pks_chain(pf=False, mcls=mcls)
                            return _flat_accs()

                        return _run

                    def _keep():
                        return _base[0]  # noqa: B023

                    def _dead():
                        _z = Vec.from_elements([fx.Int32(0) for _ in range_constexpr(4)], fx.Int32).bitcast(
                            elem_dtype
                        )
                        for nt in range_constexpr(NT):
                            _zo = (
                                nt * N_TILE * BLOCK_Q
                                + (head_local % G3S_SLOTS) * G3S_GRP_ELEMS
                                + _H[0] * G3S_SLOT_ELEMS
                            )
                            for pks in range_constexpr(PV_K_STEPS):
                                _ds_write_vec(_g3wb ^ fx.Index(pks * 2 * M_TILE), _zo, _z)

                    if const_expr(BAND_LIFT):
                        # q_start >= _kv_first_q >= _kv_lift, so this stays non-negative.
                        _q_first = q_start - _kv_lift
                    else:
                        _q_first = q_start + causal_offset
                    _q_last = _q_first + fx.Index(BLOCK_Q - 1)
                    _kvw = kv_row_wave if const_expr(_h == 0) else kv_row_wave + fx.Index(_h * BKV_H)
                    _cond = ArithValue(_kvw <= _q_last)
                    if const_expr(MASK_ALIGN and not half and poff is None):
                        # One if per class rather than a nested pair: each class's else arm
                        # yields its incoming accumulators unchanged, which coalesces away,
                        # where nesting made both arms produce fresh values and cost the
                        # masked loop 13 more accumulator dwords than the whole kernel has
                        # to spare (the co-resident dQ reduce owns 512 - ceil8(vgpr)).
                        # sched_barrier(0) at these three boundaries emits a BYTE-IDENTICAL
                        # kernel (vgpr, both loops' histograms, maxa): an scf.if boundary is
                        # already a scheduling region edge, so the unmasked loop's +23
                        # instructions against the single-arm form are a whole-function
                        # allocation effect, not motion a fence can pin back.
                        _bcond = ArithValue(_kvw + fx.Index(ROWS_PER_WAVE_KV - 1) <= _q_first)
                        _base[0] = _if_wave(_bcond, _base[0], _arm(_mc_clear), _keep)
                        _dcond = ArithValue(_kvw == _q_first)
                        _base[0] = _if_wave(_dcond, _base[0], _arm(_mc_diag), _keep)
                        _set_accs(_if_wave(_cond, _base[0], _keep, _dead))
                    else:
                        _set_accs(_if_wave(_cond, _base[0], _live, _dead))
                else:
                    _pks_chain(
                        pf=const_expr(_last_pass),
                        g3_split=const_expr(G3_SPLIT),
                        hooks=const_expr(_h == 0),
                    )
            _hs_drain()
            if const_expr(not G3_DEFER):
                # Undeferred: dS is read in the head-step that wrote it, so this head-step
                # pays its own RAW fence. gpu.barrier() alone is not a fence -- retire the
                # ds_writes first with lgkmcnt only (a full drain would also wait on the
                # previous head-step's dQ partial stores, which nothing here reads).
                # Emitting GEMM3 here rather than before GEMM2 keeps its transpose-reads'
                # live ranges off GEMM2's, which loses on a full register file.
                if const_expr(QDO_TAIL and head_local + 1 < GQA_GROUP_SIZE and _qdo_next[0] is not None):
                    # Head h+1's tile rides this fence. Its ring slot was last read by
                    # head h-1, whose reads all precede the previous head-step's barrier.
                    _qdo_commit(_qdo_next[0], fx.Index(((head_local + 1) % LDS_SLOTS) * LDS_TOTAL))
                    _qdo_next[0] = None
                rocdl.s_waitcnt(WAIT_LGKM)
                gpu.barrier()  # RAW: every wave's dS rows feed every wave's GEMM3
                _gemm3(
                    q_start,
                    head_local,
                    const_expr(head_local % G3S_SLOTS),
                    qsel=const_expr(
                        PV_K_STEPS - 1 if (G3_SPLIT and not (MASK_SKIP and apply_mask)) else None
                    ),
                    poff=poff,
                )
            return dv_cur, dk_cur, (_qdo_next if const_expr(Q_PREF) else [qdo, None])

        def _q_body(q_start, inner, apply_mask, poff=None, half=False, hsel=None, nq=None):
            # inner (loop-carried) = [dv accs][dk accs] (+ [Q/dO][-delta, lse] under PF_QB).
            _dk_base = H_ACCS
            dv_cur = [
                [[inner[(h * DT + dt) * NT + nt] for nt in range_constexpr(NT)] for dt in range_constexpr(DT)]
                for h in range_constexpr(KV_HALVES)
            ]
            dk_cur = [
                [
                    [inner[_dk_base + (h * DT + dt) * NT + nt] for nt in range_constexpr(NT)]
                    for dt in range_constexpr(DT)
                ]
                for h in range_constexpr(KV_HALVES)
            ]
            # Head-invariant DMA offsets: computed once per q-block, reused by all heads.
            _bases = _dma_bases(q_start, poff) if const_expr(ENABLE_DMA and not Q_PREF) else None
            _ldv = None
            if const_expr(PF_QB):
                _pfb = 2 * H_ACCS
                _qdo = list(inner[_pfb : _pfb + 2 * NUM_DMA_Q])
                _ldv = list(inner[_pfb + 2 * NUM_DMA_Q :])
            else:
                _qdo = _qdo_issue(q_start, 0, poff) if const_expr(Q_PREF) else None
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
                        _mid = list(
                            range_constexpr(
                                min(_first, GQA_GROUP_SIZE), min(_first + DMA_GRP, GQA_GROUP_SIZE)
                            )
                        )
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
                    hsel=hsel,
                    nq=nq,
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
            out = [
                dv_cur[h][dt][nt]
                for h in range_constexpr(KV_HALVES)
                for dt in range_constexpr(DT)
                for nt in range_constexpr(NT)
            ]
            out += [
                dk_cur[h][dt][nt]
                for h in range_constexpr(KV_HALVES)
                for dt in range_constexpr(DT)
                for nt in range_constexpr(NT)
            ]
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
            _pf0 = _q_loop_start
            if const_expr(QDESC):
                _pf0 = fx.Index(
                    ArithValue(_q_loop_start < _masked_upper).select(_q_loop_start, _desc_q(fx.Index(0)))
                )
            _carry = _carry + _qdo_issue(_pf0, 0) + _stage_ld_issue(_pf0)
        loop_results = _carry

        if const_expr(FQ_PAIR):
            _fp = fx.Index(FQ_PAIR_POFF)
            for _t, inner in range(fx.Index(0), fx.Index(FQ_PAIR_NX), 1, init=_carry):
                loop_results = yield _q_body(
                    _q_loop_start + _t * fx.Index(FQ_HALF),
                    inner,
                    True,
                    poff=_fp,
                    half=True,
                    hsel=fx.Index(ArithValue(wave_id > _t).select(fx.Index(1), fx.Index(0))),
                )
            _fq0 = _q_loop_start + fx.Index((NUM_WAVES - 1) * FQ_HALF)
            for q_start, inner in range(
                _fq0, _fq0 + fx.Index(FQ_PAIR_NF * BLOCK_Q), _step, init=loop_results
            ):
                loop_results = yield _q_body(q_start, inner, True)
        elif const_expr(window_left >= 0):
            # Fused SWA: three regions per band, mirroring the full-causal masked/unmask split
            # but bounded by BOTH window edges -- upper-causal-masked | interior UNMASKED |
            # lower-window-masked. A q-block is masked only where the band straddles a window
            # boundary; the interior runs the fast apply_mask=False path. (Running every block
            # apply_mask=True -- the scalar per-element exp path -- was the 15x fused-SWA
            # regression: 85% of time sat in the dkdv body's masked path while full-causal ran
            # its bulk unmasked.) Coverage is byte-identical to a single [_q_loop_start,_qhi)
            # masked walk -- only interior blocks flip to unmasked -- so the reduce's g_lo/g
            # coupling (which q wrote which band) is unchanged.
            # _qhi = min(seq_len_q, _kv_end + W - off): last q whose window still reaches this
            # band's top key. A rectangular LOW band has _kv_end + W < off (window too far
            # below), so NO q attends it; clamp the unsigned underflow to 0 (empty loop).
            _qtop = _kv_end_c + fx.Index(window_left)
            _qhi = ArithValue(_qtop >= causal_offset).select(_qtop - causal_offset, fx.Index(0))
            _qhi = fx.Index(ArithValue(_qhi < seq_len_q_v).select(_qhi, seq_len_q_v))
            # Lower window edge: q needs the lower mask once the band's first key drops out of
            # its window -- kv_start < q + off - W  <=>  q > kv_start + W - off (= _lo_edge). A
            # q-block [qs,qs+step) is lower-safe iff qs+step-1 <= _lo_edge; num_safe counts the
            # lower-safe blocks from _q_loop_start (floor((_lo_span+1)/_step) folds the +step-1
            # rounding), and _lo_masked_start is the block where masking resumes. Same unsigned
            # underflow guard as _qhi: if _lo_edge < _q_loop_start no block is lower-safe.
            _lo_top = kv_start + fx.Index(window_left)
            _lo_edge = fx.Index(
                ArithValue(_lo_top >= causal_offset).select(_lo_top - causal_offset, fx.Index(0))
            )
            _num_safe = fx.Index(
                ArithValue(_lo_edge >= _q_loop_start).select(
                    (_lo_edge - _q_loop_start + fx.Index(1)) // fx.Index(_step), fx.Index(0)
                )
            )
            _lo_masked_start = _q_loop_start + _num_safe * fx.Index(_step)
            # interior end = _lo_masked_start clamped into [_unmask_start, _qhi]: interior may
            # be empty (narrow W: the two masked edges meet) or full (wide W past seq end).
            _int_end = fx.Index(ArithValue(_lo_masked_start < _qhi).select(_lo_masked_start, _qhi))
            _int_end = fx.Index(ArithValue(_int_end > _unmask_start).select(_int_end, _unmask_start))
            for q_start, inner in range(_q_loop_start, _masked_upper, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
            for q_start, inner in range(_unmask_start, _int_end, _step, init=loop_results):
                loop_results = yield _q_body(q_start, inner, False)
            for q_start, inner in range(_int_end, _qhi, _step, init=loop_results):
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
        elif const_expr(QDESC):
            for q_start, inner in range(_q_loop_start, _masked_upper, _step, init=_carry):
                _m_nxt = q_start + fx.Index(_step)
                loop_results = yield _q_body(
                    q_start,
                    inner,
                    True,
                    nq=fx.Index(ArithValue(_m_nxt < _masked_upper).select(_m_nxt, _desc_q(fx.Index(0)))),
                )
            for q_start, inner in range(_unmask_start, seq_len_q_v, _step, init=loop_results):
                _t = (q_start - _unmask_start) // fx.Index(_step)
                loop_results = yield _q_body(_desc_q(_t), inner, False, nq=_desc_q(_t + fx.Index(1)))
        else:
            for q_start, inner in range(_q_loop_start, _masked_upper, _step, init=_carry):
                loop_results = yield _q_body(q_start, inner, True)
            for q_start, inner in range(_unmask_start, seq_len_q_v, _step, init=loop_results):
                loop_results = yield _q_body(q_start, inner, False)
        _dk_base = H_ACCS
        dv_accs = [loop_results[i] for i in range_constexpr(H_ACCS)]
        dk_accs = [loop_results[_dk_base + i] for i in range_constexpr(H_ACCS)]

        # ---- Store dV[kv,D], dK[kv,D]. The 16x16 C-layout gives each lane 4
        # CONTIGUOUS D values (D = dt*16 + kg*4 + t) at kv = nt*16 + lane16, so the
        # store is direct (no permlane32 transpose needed, unlike the 32x32 path). ----
        sm_vec4 = Vec.from_elements([fx.Float32(sm_scale)], fx.Float32).broadcast_to(4)

        def _store(accs, rsrc, scale):
            for h in range_constexpr(KV_HALVES):
                for dt in range_constexpr(DT):
                    for nt in range_constexpr(NT):
                        v = Vec(accs[(h * DT + dt) * NT + nt])
                        if const_expr(scale):
                            v = v * sm_vec4
                        lo = rocdl.cvt_pk_bf16_f32(v[0], v[1])
                        hi = rocdl.cvt_pk_bf16_f32(v[2], v[3])
                        o_pack = Vec.from_elements([fx.Int32(_raw(lo)), fx.Int32(_raw(hi))], fx.Int32)
                        d_col = fx.Index(dt * D_TILE) + kg * fx.Index(4)
                        g_idx = global_idx_kv(kv_row_of(nt, h), d_col)
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
            if const_expr(True)
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
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        # Backward is VALU/exp2-issue-bound with the MFMA pipe mostly idle; post-RA
        # misched hides the gradient-GEMM MFMAs in the exp2/reduce VALU shadow.
        # post-misched is load-bearing on the four-wave fused body: dropping it measures
        # 892.5 against 918.9, and every amdgpu-sched-strategy override is worse still
        # (max-memory-clause 863.0, max-ilp 888.4) -- the hand-placed sched_mfma/sched_dsrd
        # structure is what the default scheduler is being asked to preserve.
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


# ===========================================================================
# Host-side varlen backward orchestration (odo + dq + dkdv split-K reduce).
# Deterministic drop-in for the CK hd64 FMHA varlen backward; the build_* module
# factories above are called directly (same module).
# ===========================================================================


def _qsplit_for(Sq, window_left=-1, head_dim=64):
    # q_split fans the dK/dV KV-owner WGs across the CU grid; the optimum rises with
    # Sq before split-reduction overhead dominates. Re-swept once the fused path started
    # dispatching one batch at a time (fewer work-groups per dispatch, so list-scheduling
    # slack matters more): 4 still wins; wider only adds dk/dv slots for the slot reduce
    # to fold. It holds past 8192, which used to take a narrower 3 unmeasured: on the fused
    # D128 arm that costs several percent, since a split count that tiles the q blocks is
    # also what lets the pipeline cut them (see _qsp_cuttable).
    if window_left >= 0:
        # Sliding window: a band's whole q range is only BLOCK_KV+W rows, so splitting it
        # hands the SAME work to q_split times as many work-groups and multiplies the
        # dk/dv workspace. One slot avoids both the redundant prologues and the slot reduce.
        return 1
    # 8 was measured for the pipeline's sake and lost: it is the only way to cut the fused
    # dispatch into eight chunks that each still fill the CU array (a chunk is
    # bands*Hkv*B work-groups whatever q_split is), so it halves the one dQ reduce the
    # pipeline cannot hide -- but doubling the splits also doubles the dk/dv slots the body
    # writes and the slot reduce folds, and doubles how often each band re-stages its K/V.
    # Paired standalone cells: gptoss_full +1.32%, d64_hq32_b4 +2.3%, d64_hq64_b1 +1.7%,
    # d64_hq128_16k +0.9%, and varlen_d64_u +10.3% (it pays the slots with no pipeline at
    # all to spend them on). The exposed tail is worth less than the slots it costs.
    # 2 loses from the other side, gptoss_full +1.35%: the body does amortize a prologue
    # over half as many work-groups, but a two-chunk cut doubles the tail it cannot hide.
    #
    # That trade turns over at D=128, and it is the head dim that turns it: the amortized
    # side scales with what a work-group re-stages per q-loop trip, which is D per kv row,
    # while the tail it doubles is the fold of ONE q subset, whose bytes are the same share
    # of a dQ image that also grew with D. So halving the splits buys twice as much at D128
    # as at D64 and costs the same fraction. Measured on (2, 64, 8, 8192, 128), own process,
    # min of 40, interleaved and palindromic in one batch: 6.2017 / 6.2020 / 6.1945 at 2
    # against 6.2389 / 6.2407 / 6.2481 / 6.2510 at 4 -- all three below all four, -0.73%,
    # and dq/dk/dv stay bitwise identical (a split count moves WHICH work-group writes a
    # partial, never the fold's ascending band sum). D64 keeps 4 on its own reading above;
    # do not lift this gate without re-running that cell.
    if head_dim >= 128:
        return 2
    return 4


def _fuse_qsplit_for(max_sq, total_kv, num_kv_heads, block_kv, window_left, block_q=_BWD_BLOCK_Q):
    """q-loop splits for the FUSED ragged path (the split pair keeps _qsplit_for).

    A windowed band walks BLOCK_KV+W q rows whatever the sequence length is, so a ragged
    dispatch is only ``(kv tokens / BLOCK_KV) * Hkv`` work-groups wide -- one band per
    segment row of the packed kv axis -- where a dense shape gets a batch axis on top of
    that. Below the CU array that leaves at most one band per CU, so the wall is the
    LONGEST band's walk, and on ragged segments the longest is far above the mean (a band
    inside a short segment stops at its segment's end). Splitting the walk hands the same
    work to more work-groups, so that tail averages out instead of draining alone; it
    costs no extra MFMA, only one more staging of the band's K/V per split. So split only
    while the dispatch is narrower than the array: once it covers the array a split buys
    a round it has to pay for as well, and at the narrow bands (where a band's whole walk
    is a few q blocks) the extra stagings already outweigh the tail. That leaves a split
    on the table for a dispatch that covers the array at the WIDEST band, whose walk is
    long enough to survive being cut again. Kept to what the band's q blocks divide by,
    since a split's band-relative blocks have to be its absolute ones for the reduce to
    see one writer per slot.
    """
    if window_left < 0:
        return _qsplit_for(max_sq, window_left)
    q_split, bands = 1, max(1, total_kv // block_kv) * num_kv_heads
    while q_split * 2 <= block_kv // block_q and bands * q_split < _NUM_CU:
        q_split *= 2
    return q_split


def _assert_fusable(Hq, D):
    """Every shape that reaches this backward emits dQ from the KV-outer body.

    The dQ reduce tiles a work-group's chunk out of 2*Hq*D elements, which needs Hq*D to be
    a multiple of 128 -- D128 always is, D64 needs an even Hq. That is not a case to branch
    on: the backend admits FlyDSL only when the GQA group Hq/Hkv is a power of two in
    [8, 256] (see _gqa_group_ok), so Hq is a multiple of 8 and the condition holds for both
    head dims. Asserted rather than assumed because it is the whole reason the Q-outer dq
    kernel could be deleted -- if the backend ever admits a smaller group, this fires here
    instead of silently reducing a partial dQ.
    """
    assert (Hq * D) % 128 == 0, (
        f"the fused dQ reduce needs Hq*D % 128 == 0, got Hq={Hq} D={D}; the FlyDSL backend "
        "is supposed to admit only GQA groups that make Hq a multiple of 8"
    )


def _fuse_blockkv_for(Skv, D=64, window_left=-1):
    """kv band for the fused path. The dQ split-K traffic is (Skv/BLOCK_KV)/2 * |dQ| in
    each direction, so unlike the split path -- where BLOCK_KV only trades grid width
    against per-tile cost -- the fused path pays for a narrow band in DRAM bytes and
    wants the widest band the register file takes.
    """
    # 256 only fits because the K/V B-operands live in LDS (see `_kv_lds_idx`): it
    # doubles the dK/dV accumulators, and the packs it displaces are exactly what pays
    # for them (with the packs still in registers this spilled catastrophically). 512
    # would halve the band count again, and it does compile -- but its dK/dV accumulators
    # alone are 512 dwords across four waves, so it buys the halved traffic with scratch:
    # d64_gptoss_b4 reads 9.218 ms against the deployed 256's 5.601, and 512 with BLOCK_Q
    # dropped to 32 to fund it reads 8.427 and no longer matches the 256 build bitwise.
    # The same wall stands at D128 one step earlier: 256 reads 8.82/8.89 against 6.29 with
    # 880 scratch ops, and 256 with BLOCK_Q=32 -- the one knob that zeroes that spill --
    # still reads 7.428 against a 6.375/6.311 palindromic base. So widening the band is not
    # the way to cut the split-K leg's byte count, at either head dim.
    # Register cost is `dk+dv dwords = D*block_kv/(32*waves)`; D128's non-accumulator state caps it at 128.
    if window_left >= 0:
        # A window wants the widest FILLED band; round W down to a power of two, capped by LDS/wave count.
        cap = 128 if D == 128 else 256
        return min(cap, max(32, 1 << max(0, int(window_left).bit_length() - 1)))
    if Skv >= 4096:
        return 128 if D == 128 else 256
    # The band count is what the fused path pays for on a small shape -- the reduce's
    # (Skv/BLOCK_KV)/2*|dQ| of DRAM traffic is the whole reason it can lose to the split pair
    # there -- so take the widest band before the wider tile thins the grid past the win.
    # D64 reaches its 256 as soon as the sequence is longer than one 1024-row square; D128,
    # whose band is halved for its dK/dV accumulators, stays at 128 all the way down.
    if D == 128:
        return 128
    return 256 if Skv > 1024 else 128


_BWD_CACHE: dict = {}
_DQRED_CACHE: dict = {}
# Launchers the dQ fold keeps built. A hidden fold goes out in up to _DQ_FOLD_SLICES slices,
# each its own (qsp, batch) key, so ONE shape asks for slices*(chunks-1)+1 of these -- 49 at
# the widest plan a D128 cell reaches. A cap a single shape can reach is not a cap, it is a
# rebuild of every launcher on every call: pushed past it the target reads 1424 ms against
# 5.85, because the JIT then runs inside the timed loop.
# Holds several shapes' worth -- these are launchers, and no device tensor is cached here.
_DQRED_CACHE_MAX = 256


def _cu_placeholder(device):
    """Unused cu_seqlens argument slot (read only under ``const_expr(varlen)``)."""
    return torch.zeros(1, device=device, dtype=torch.int32)


def _dq_partial_ws(nb, B, Sq, hd, device, dtype, pad_bytes=0, ilv=1, carry=False):
    """dQ split-K workspace [bands/ilv, B, Sq, Hq*D*ilv] for the fused KV-outer kernel.

    Returns (workspace, carry): ``carry`` is the fp32 running dQ sum the reduce hands from one
    band group to the next, or None when the whole band axis fits one workspace.

    ``ilv`` adjacent bands share a row and are interleaved at D granularity rather than
    getting a slab each; ``ilv=1`` is the plain [bands, B, Sq, Hq*D]. See _WSQ_BAND_ILV
    for why, and _wsq_ilv for what bounds it.

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
    or hidden. Folding band pairs by letting one band ACCUMULATE INTO its partner's slot
    does not remove bytes: the accumulating band still has to read that slot, which only
    trades dqred reads for dkdv reads (net loss). Folding them by giving one work-group
    both bands does remove bytes, and is priced further down.

    WHAT THESE BYTES COST, measured by subtraction on the scored shape (D128, 6.78 ms):
    dropping the partial store alone takes the wall to 4.91 ms and dropping the reduce
    with it to 4.18 ms, while D64's own compute phase doubled is 4.25 ms. So the fused
    D128 body is ALREADY at D64's per-flop efficiency everywhere except this workspace --
    the whole remaining gap is these partials, and nothing else in the body is worth
    attacking until they shrink. Splitting the store cost further: writing the partials to
    a single aliased slab (wrong dQ, pricing only) recovers 0.60 ms, so ~1.06 ms of it is
    memory traffic (~0.12 ms/GB at 8.7 GB, well under the 0.161 ms/GB roofline price
    because it overlaps compute) and ~0.79 ms is GEMM3 plus store issue. Halving the
    partial count is therefore worth ~0.9 ms on the write side plus a MEASURED 0.34 ms on
    the read side (running the reduce over half as many bands), i.e. ~+18%.

    Folding the pair IN REGISTERS instead -- one work-group owning two adjacent bands and
    writing a q row's slot once -- does halve the slots and both request counts, but it is
    priced by the dK/dV accumulators, not by dQ, and either loop ordering needs them live
    twice over. The accumulator is BLOCK_KV*D*2/256 = 128 dwords at bkv=128 and is
    invariant to how the work is split across waves, q blocks or heads, so a pair costs a
    flat +128 dwords against the 52 this body has spare (460 of 512, with two 24-dword
    reduce waves in the rest). Nothing smaller than that buys the fold: q-outer needs both
    bands live across the q block, band-outer cannot share the dQ accumulator at all, and
    flushing the second band's dK/dV to LDS or to a slot per q block costs orders of
    magnitude more traffic than the partials it saves. Halving these request counts
    therefore needs a fold whose unit is NOT a second dK/dV accumulator set. The routes
    that survive that constraint all pay in RECOMPUTE -- e.g. one work-group owning the
    pair's dK and dQ while a sibling owns its dV, which duplicates only GEMM1 (+20% flops
    for -1.25 ms) -- so the next round's job is to price that recompute against the
    compute phase, not to re-attempt the register fold.

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
    assert nb % ilv == 0, "the band interleave must divide the band count"
    ng, ghd = nb // ilv, hd * ilv
    slab = B * Sq * ghd
    pad = pad_bytes // dtype.itemsize
    assert pad * dtype.itemsize == pad_bytes, "band padding must be a whole element count"
    # The band-group carry (fp32, one dQ image) is the tail of this same allocation: it
    # is the other half of the same workspace decision and has to be freed with it.
    tail = (B * Sq * hd * 4 // dtype.itemsize) if carry else 0
    if pad or tail:
        flat = torch.empty(ng * (slab + pad) + tail, device=device, dtype=dtype)
        ws = (
            flat[: ng * (slab + pad)].as_strided((ng, B, Sq, ghd), (slab + pad, Sq * ghd, ghd, 1)),
            flat[ng * (slab + pad) :].view(torch.float32) if tail else None,
        )
    else:
        ws = (torch.empty(ng, B, Sq, ghd, device=device, dtype=dtype), None)
    return ws


_SLOTRED_CACHE: dict = {}
_SLOTRED_BLOCK = 256
_SLOTRED_VEC = 8
# Chunks one thread folds, which is also what sizes the grid: 2 dispatches 4096 work-groups
# for the target's 16.8 M elements per tensor, i.e. 16 resident rounds on 256 CUs. Widening it
# reads like an obvious win -- 32 loads per thread in flight at uc=8 against 8 at uc=2 -- and
# it is not one, because the fold is already at the roofline across the whole range. Timed
# alone on the deployed slot shape it reads 60.0 / 58.5 / 57.6 / 58.4 / 60.2 us at uc
# 1 / 2 / 4 / 8 / 16 (5.6-5.8 TB/s, a 4% plateau), and 8 against 2 in situ on gptoss_full is a
# dead 9-9 split over 18 position-balanced trials. One work-group per CU is where the plateau
# ends, in the wrong direction: uc=32 keeps 128 dwords of loads live per thread and collapses
# to 121 us / 2.76 TB/s. So this is a CEILING on a plateau, not a peak -- it stays at the
# measured-neutral 2, and _slotred_uc only narrows below it to keep a workspace that cannot
# tile at 2 out of torch's strided sum.
_SLOTRED_UC_MAX = 2


def _slotred_uc(n_elems, n_groups):
    """Widest per-thread chunk count that tiles the slots without under-filling the machine.

    The rate is flat across the legal range (see _SLOTRED_UC_MAX), so what picking a width
    buys is not speed: it is that a shape whose slots do not tile at the deployed width still
    gets this fold at ~5.7 TB/s instead of falling back to sum(dim=) at 4.5 over two launches.
    """
    unit = _SLOTRED_BLOCK * _SLOTRED_VEC
    fits = [uc for uc in (16, 8, 4, 2, 1) if uc <= _SLOTRED_UC_MAX and n_elems % (unit * uc) == 0]
    if not fits:
        return None
    for uc in fits:
        if n_groups * (n_elems // (unit * uc)) >= _NUM_CU:
            return uc
    return fits[-1]


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
    uc = _slotred_uc(n_elems, n_groups)
    if uc is None:
        axis = 1 if n_groups > 1 else 0
        return ws_dk.sum(dim=axis), ws_dv.sum(dim=axis)
    dk = torch.empty(n_groups * n_elems, device=ws_dk.device, dtype=ws_dk.dtype)
    dv = torch.empty(n_groups * n_elems, device=ws_dv.device, dtype=ws_dv.dtype)
    key = (n_slots, n_groups, n_elems, uc)
    launcher = _SLOTRED_CACHE.get(key)
    if launcher is None:
        if len(_SLOTRED_CACHE) >= 32:
            _SLOTRED_CACHE.clear()
        launcher = build_flash_attn_bwd_slotred_module(
            n_slots=n_slots, n_groups=n_groups, n_elems=n_elems, block=_SLOTRED_BLOCK, uc=uc
        )
        _SLOTRED_CACHE[key] = launcher
    launcher(ws_dk.reshape(-1), dk, ws_dv.reshape(-1), dv, stream)
    return dk, dv


def _reduce_dq_partials(
    ws,
    dq,
    block_kv,
    num_heads,
    head_dim,
    scale,
    stream,
    bat_lo=0,
    n_bat=None,
    qsp=(1, 0, None),
    causal_offset=0,
    window_left=-1,
    sbhd=False,
    cu=None,  # (cu_seqlens_q, cu_seqlens_kv): ragged rows, band window per segment
    band=None,  # (band_span, cu_seqlens_kv slot, carry): fold ONE band group (see _band_span_for)
    band_ring=0,  # >0: bands share workspace slots modulo this (see _wsq_ring_for)
    ph=None,  # the caller's unused-slot placeholder, if it already has one (see below)
):
    """dQ[q] = scale * Sum_{b : b*BLOCK_KV <= q} ws[b][q], in ascending band order.

    A kv band only writes the q rows that causally see it, so the bands ABOVE q's own
    band hold stale data and are skipped -- which is also what keeps the traffic at the
    causal half. Fixed band order and fp32 accumulation -> bitwise deterministic.

    ``scale`` is 1/log2e, not sm_scale: the fused kernel's fifth GEMM contracts against
    the LDS K tile, which is staged already prescaled by sm*log2e for GEMM1a.

    Band count is what this costs, and on the fused D128 body it is ALL that is left.
    Widening the reduce's band unit N-fold so it reads 1/N of the bands (diagnostic only,
    wrong dQ) prices it exactly linear in bytes on the scored shape: 8.72 GB -> 6.569 ms,
    4.36 -> 6.186, 2.18 -> 6.002, none -> 5.749, i.e. 0.086 ms per logical GB with no
    fixed part, which is 6.0 TB/s over the ~50% of those bytes that miss cache = the
    fabric roofline. Two consequences worth keeping:
      * with the reduce off the D128 body runs 956.5 TF against the D64 fused arm's
        915.9 -- the BODY is already 1.04x D64 per flop, so no lever inside it can close
        the score gap because the gap is not there;
      * closing the remaining gap needs -0.55 of these 0.82 ms, i.e. a third fewer
        partial bytes or a third more cache absorption. Absorption is capacity-bound
        (1.07 GB pipeline chunk against a 256 MB MALL), so the byte count is the
        only remaining term -- and it is (Skv/BLOCK_KV)*|dQ|/2, set by the band width.

    The work-group SHAPE sweep above only covers a q row reading every band below it; a
    window leaves each row a fixed handful of bands, favoring more walks in flight over a
    wide load per walk (see the chunk choice below).
    """
    _, B, Sq, _ = ws.shape
    band_pad = _WSQ_BAND_PAD if head_dim == 128 else 0
    band_ilv = ws.shape[3] // (num_heads * head_dim)
    # Ragged rows are packed q tokens whose segment base has no alignment, so a work-group
    # takes ONE row there (a pair could straddle a segment boundary); see the build.
    num_seg = (cu[0].numel() - 1) if cu is not None else 1
    rpw = 1 if cu is not None else 2
    rpw_hd = rpw * num_heads * head_dim
    # rows_per_wg*Hq*D must tile the reduce's block*vec8*uc chunk; any tiling uc/block is
    # bitwise-identical (see build), trading only the co-resident register footprint. The
    # narrowest chunk that still tiles wins at both head dims (more walks in flight per CU).
    # That holds whether or not anything co-resides: raising uc to put a second 16 B load
    # per thread in flight was measured on a launch that runs ALONE (the D64 SBHD reduce,
    # where waves/SIMD and not the register pool is what caps it, so the occupancy is
    # unchanged) and it still loses -- gptoss_full +1.18%, gptoss_swa +9.4%, varlen +3.1%.
    # The stream is priced in bytes on the shared fabric, not in exposed latency, so depth
    # per thread buys nothing and the wider chunk only costs walks.
    block, uc = (256, 1) if rpw_hd % 2048 == 0 else (None, None)
    # A window leaves each q row a fixed handful of bands, which is what makes more walks in
    # flight worth a narrower load; full causal reads every band below the row and keeps the
    # swept width, so the re-tile is windowed-only.
    if block is not None and window_left >= 0:
        _n_chunk, _xcd_block = rpw_hd // (block * 8), rpw_hd // (_NUM_XCD * 8)
        if (
            _n_chunk % _NUM_XCD != 0
            and _NUM_XCD % _n_chunk != 0
            and _xcd_block % 64 == 0
            and _xcd_block <= 1024
        ):
            block, uc = _xcd_block, 1
    key = (
        num_heads,
        head_dim,
        B,
        Sq,
        block_kv,
        scale,
        bat_lo,
        n_bat,
        qsp,
        causal_offset,
        window_left,
        sbhd,
        band_pad,
        band_ilv,
        band_ring,
        num_seg if cu is not None else None,
        band[0] if band is not None else 0,
    )
    launcher = _DQRED_CACHE.get(key)
    if launcher is None:
        if len(_DQRED_CACHE) >= _DQRED_CACHE_MAX:
            _DQRED_CACHE.clear()
        launcher = build_flash_attn_bwd_dqred_module(
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=B,
            seq_len_q=Sq,
            block_kv=block_kv,
            sm_scale=scale,
            block=block,
            rows_per_wg=rpw,
            uc=uc,
            bat_lo=bat_lo,
            n_bat=n_bat,
            q_split=qsp[0],
            qsp_lo=qsp[1],
            n_qsp=qsp[2],
            causal_offset=causal_offset,
            window_left=window_left,
            sbhd=sbhd,
            band_pad=band_pad,
            band_ilv=band_ilv,
            band_ring=band_ring,
            varlen=cu is not None,
            num_seg=num_seg,
            band_span=band[0] if band is not None else 0,
        )
        _DQRED_CACHE[key] = launcher
    # Pass ONE band slice: the descriptor is rebased per band with a 64-bit offset, and
    # the whole workspace overflows a flat memref's i32 element count.
    # The unused cu_seqlens slot rides the CALLER's placeholder when it has one. Allocating
    # one here instead costs a fill kernel per launch, and the chunked path calls this once
    # per chunk ON THE FUSED STREAM: the trace shows those fills landing between two chunks
    # of the body, so each one is exposed wall time (4 of them, ~6 us each, at q_split=4).
    # Reusing the caller's is not the module-level cache that was removed on purpose -- it
    # lives and dies inside one backward, so no tensor is held across calls. A ragged launch
    # has the real table in that slot and needs no placeholder at all.
    if ph is None and cu is None:
        ph = _cu_placeholder(dq.device)
    dqf = dq.reshape(-1)
    if band is not None:
        # A group's first band arrives in the cu_seqlens_kv slot: alone on a dense launch,
        # appended to the segment table on a ragged one (see _cu_band_rows).
        launcher(ws[0].reshape(-1), dqf, band[2], cu[0] if cu is not None else ph, band[1], stream)
    else:
        launcher(ws[0].reshape(-1), dqf, dqf, *(cu or (ph, ph)), stream)


# Overlap the dQ reduce with the fused kernel by running the backward in chunks
# (see _fused_pipelined). Off keeps the single whole-batch dispatch pair, which is what
# the eight-wave geometry had to use since it leaves no room for a co-resident reduce
# wave; the pipeline is only worth its chunking overhead at the four-wave geometry,
# where the fused work-group leaves enough registers for a reduce work-group to land.
_DQ_PIPE = True
# Causal area at which chunking stops paying. At and below it a chunk's own compute no longer
# dwarfs the dispatch it costs and the overlap is a double-digit loss, whatever the batch or
# head count -- and the fewer heads, the worse, until the fixed cost is the whole backward.
# Above it the measured spread is a few percent either way with no shape term behind it
# (area, workspace size and band interleave all fail to predict the sign), so it stays on.
_DQ_PIPE_AREA_FLOOR = 2048 * 2048
# Fills of the CU array a single SBHD chunk must still be worth. The fused body runs one
# work-group per CU, so a chunk narrower than the array pays a whole round of the LONGEST
# band's walk with nothing beside it to average the tail against, and the band walks under
# causal masking span 128:1. Measured on the eight-cell ruler, chunk grid = bands*Hkv*B:
# 1024 work-groups is gptoss_full -1.11% and d64_hq128_16k -3.00% (paired against the uncut
# arm, alternating, min of 40 per reading), 512 is d64_hq32_b4 +36.4% and 256 is d64_hq64_b1
# +56.9%. The cliff is therefore somewhere in (2, 4] fills; it is taken at 4, the narrowest
# chunk measured to pay. Every chunk _pipe_chunks emits is at least one split wide, so the
# rule is checked once on a single-split chunk and holds for every plan.
# All four of those readings are at the 256-row band, and the threshold does NOT hold across
# band widths: what a chunk BUYS is the fold bytes it hides, and those go as 1/(2*BLOCK_KV)
# per unit of body work, while what it COSTS -- one drain of the longest band's walk -- is a
# body quantity that the band width leaves alone. So the same 512-work-group chunk that costs
# d64_hq32_b4 +36.4% at bkv=256 is the other sign at bkv=128: l70b_b1 (D128, B=1, the one
# cell of the seven that falls under the flat rule) reads 3.3226 / 3.3551 unpiped against
# 3.1997 / 3.1869 piped, -4.4%, palindrome 2/2, bitwise deterministic, with the four cells
# already above the threshold flat. The profiler names the mechanism rather than leaving it
# to the wall: under the flat rule B=1 issues ONE dqred launch against B=2's three, so none
# of its 0.74 ms of fold can hide under a following body and all of it is exposed.
# A chunk under the rule is not a shape without a pipeline: the rule is a chunk WIDTH, and
# merging subsets widens the chunk until it clears (see _dq_pipe_qsp). d64_hq64_b2 is the one
# scored cell that lands there -- 32 bands * 8 kv-heads * 2 batches = 512 work-groups a subset
# against the 1024 the 256-row band asks -- and it was taking the unpiped path with its WHOLE
# fold exposed. At two subsets a chunk it clears the same rule and reads 2.9060 / 2.8879
# against 2.9489 / 2.9390 unpiped (own process, min of 40, palindrome), -1.6%, bitwise equal
# to the unpiped result on dq/dk/dv.
_DQ_PIPE_FILLS = 4
# Queues the pipeline's chunks alternate across. One queue serialises them, so every chunk
# boundary drains the CU array before the next chunk starts filling it, and at one
# work-group per CU a drain is the ragged tail of a whole round. Two queues let the next
# chunk fill what the current one is draining -- the chunks are independent (disjoint dk/dv
# slots, disjoint dQ partial rows) and, being splits of one q loop, share their K/V, so
# nothing beyond the launch order changes.
# Whether the last SPLIT chunk's dQ reduce stays on the caller's stream, so the dk/dv slot
# reduce the caller enqueues next runs behind it rather than beside it. See _fused_pipelined.
_DQ_TAIL_SERIAL = True
# Fills of the CU array one piece of a BATCH-CUT tail must be worth. The last chunk's fold is
# the one with no next chunk over it, and it is measured pure exposure -- 0.347 ms on
# d128_hq64_b2, 0.275 on gptoss_full, bytes that the hidden folds pay too, so the position is
# the whole cost. Cutting that chunk on the BATCH axis hands all but its last piece a body to
# hide under, and unlike a finer q_split cut it re-stages nothing: the pieces own different
# batches, so the work-group count, the K/V each stages and the dk/dv slot traffic are what the
# uncut tail had, and the only new cost is one dispatch boundary. Priced first on the q_split
# axis, where the same structure comes with the re-staging attached: at D128 halving the tail
# alone is worth -0.097 ms (plan [2,2,2,1,1] 6.4300 against [2,2,2,2] 6.5269 at q_split=8, own
# process, min of 40, palindrome) while raising q_split 4 -> 8 to get it costs +0.213.
#
# Taken at the WIDE band only, and measured that way from both sides (own process, min of 40,
# palindrome-ordered, two readings an arm inside one batch): gptoss_full 5.6338 / 5.6913 cut
# against 5.7656 / 5.7330 uncut, -1.5% with both readings below both, while the same cut at
# the 128-row band is the other sign -- d128_hq64_b2 6.286 / 6.3857 against 6.3233 / 6.2640
# and d128_hq32_b2 3.4524 / 3.4296 against 3.3268 / 3.3829, +2.6% with both above both. The
# dispatch boundary is not what separates them: with the fold OFF, d128_hq64_b2 reads 5.4999
# at one chunk, 5.4223 at two and 5.4725 at four, i.e. a D128 boundary prices at ~0 in the
# body. What is left is the term this file measures everywhere else -- a unit of body absorbs
# 1/(2*BLOCK_KV) of partial bytes, twice as much to pay at D128 -- so moving tail bytes from
# EXPOSED to HIDDEN buys back less than it costs the body it now runs under. Same gate, and
# for the same reason, as the pair merge in _pipe_chunks.
_DQ_TAIL_CUT_FILLS = 2
# Fold bytes one piece of a UNIFORM batch cut must still carry (see _dq_grid_cut). Cutting the
# tail alone leaves the plan UNEVEN -- the chunk in front of the tail keeps its whole-batch
# fold and now has only a tail PIECE to hide it under -- and at the 128-row band that window is
# subscribed 2:1 against a body that absorbs half as many partial bytes per unit of work, which
# is why the tail-only cut measures d128_hq64_b2 +1.4%. Cutting EVERY chunk instead keeps every
# window 1:1 and halves the exposed tail with it, and it is the only member of the family that
# wins: paired A/B inside one process, 30 pairs, the two arms alternating so the host's swing is
# common mode (_probe_ab.py, whose null control reads +0.003 ms / 15 of 30), d128_hq64_b2
# -0.0844 ms with B ahead in 28 of the 30 pairs, min 6.2574 against 6.3411. Cutting only the
# head is +0.33% and cutting all but the tail +0.80% (own process, min of 40, palindrome), so
# it is the evenness that pays rather than the piece count.
# The gate is the fold a PIECE still carries, and the two D128 cells set it from both sides:
# d128_hq64_b2's piece is 1 GiB and wins, while d128_hq32_b2 -- same grid, same bands, same
# q_split, HALF the bytes a row -- puts 512 MiB in a piece and reads +0.0383 ms with A ahead in
# 27 of 30. What the cut buys goes with those bytes (the exposed tail it halves) while what it
# costs does not: the extra boundaries re-stage the same K/V and drain the same grid whatever
# Hq is. So the line is taken at the piece's fold bytes, at the widest power of two the measured
# pair allows, and not at a work-group count -- both cells cut to the same 512 work-groups, and
# both sit exactly on the fill floor their band asks for.
_DQ_CHUNK_FOLD_BYTES = 1 << 30
# Slices a HIDDEN dQ fold is issued in (see _fold_slices). What a hidden fold costs is not its
# own latency but what it does to the body it runs under, and that is paid per BURST rather
# than per byte, so the same bytes handed to the fabric in narrow slices cost the body less.
# Monotone in the slice count and it has to be taken far enough to pay for the extra launches.
# On gptoss_full, own process, min of 40, palindrome-ordered against the unsliced arm: the
# whole fold reads 5.932 mean over 17 readings, four slices 5.890 over 19 and eight 5.851 over
# 10 -- with all ten of the eight-slice readings below all seventeen unsliced ones. TWO slices
# is where it stops paying, which is what sets the floor: d128_hq64_b2 has B=2, so cutting its
# batch axis alone gives exactly two, and that measures +1.3% (6.5805 / 6.5906 / 6.6123
# against 6.4803 / 6.4928 / 6.5539) while the same cell at eight reads 6.4634. Hence a SLICE
# count with the q axis cut once the batch axis runs out, rather than a rule about the batch.
# The price is launchers: a slice is a build-time (qsp, batch), so this is slices*(chunks-1)+1
# of them per shape, 25 at q_split=4, each paying one JIT on its first call.
# A COUNT, not a burst width: under the merged plan (_pipe_chunks) the wide chunk's fold
# covers twice the rows at the same count, so the obvious follow-up is to scale it with the
# chunk width and hold the burst constant. It measures flat -- sixteen slices read 5.7316 on
# gptoss_full against 5.7334 for eight (eight readings each, own process, palindrome-ordered)
# and 0.3% down on d128_hq64_b2, and scaling by width lands between them. Four readings each
# had shown sixteen 0.8% ahead, which is the host's swing, not a result.
#
# The count alone does NOT carry across shapes, and once the equal chunk plan made every
# hidden fold one subset wide the two D128 cells say so from opposite directions. Sixteen
# slices, own process, min of 40, palindrome-ordered (four readings an arm): d128_hq64_b2
# 6.2455 / 6.2511 / 6.2628 / 6.2631 against 6.2915 / 6.2934 / 6.2972 / 6.3034 at eight, all
# four below all four, -0.65% -- while d128_hq32_b2, the SAME grid (1024 work-groups, 64
# bands, B=2, q_split=4, so the same 256 q rows a slice) at HALF the bytes a row, reads
# 3.7528 / 3.7630 against 3.3110 / 3.3321, +13.1%. Their slice counts, row counts and
# resident rounds are identical, so what separates them is the only term left: the BYTES one
# slice hands over, 134 MB against 67 MB. Below that the ramp is no longer amortized and the
# fold stops keeping up with the body it is meant to hide under.
_DQ_FOLD_SLICES = 16
# So the count is a cap and this is the floor that stops it: the fold's bytes, per slice.
# 134 MB pays and 67 MB does not, at both head dims (gptoss_full's narrow chunk sits at 134
# and its own sixteen-slice arm, which lands at 67, is the flat reading recorded above), so
# the line is taken at 128 MiB -- the widest power of two the measured pair allows.
_DQ_FOLD_SLICE_BYTES = 1 << 27
_SIDE_STREAM: dict = {}
_PIPE_EVENTS: dict = {}


def _dq_pipe_fills(block_kv):
    """Fills of the CU array one chunk must be worth, at THIS band width (see _DQ_PIPE_FILLS)."""
    return max(1, _DQ_PIPE_FILLS * block_kv // 256)


def _dq_pipe_qsp(wgs, q_split, block_kv, batch=1):
    """q_split subsets one chunk must hold to be worth the fill rule; 0 = do not pipe.

    ``wgs`` is what ONE subset dispatches over the whole batch, ``n_bands*Hkv*B``. The rule
    (_DQ_PIPE_FILLS) is a chunk WIDTH, and a shape whose one-subset chunk falls under it is not
    out of reach of the pipeline: merging subsets widens the chunk until it clears, which is a
    plan the same rule already licenses -- the alternative is not a narrow chunk but NO chunk,
    i.e. the whole fold exposed on the caller's stream. Widening stops at two chunks, below
    which there is nothing left for a fold to hide under.

    Past the fill rule, widening is what the merged pair in _pipe_chunks was avoiding: a wider
    chunk is a wider TAIL, and the tail is the one fold with nothing over it. That stops being
    true once the tail is cut on the batch axis (see _dq_tail_cut) -- the cut divides the wide
    tail back to the same exposed bytes -- and then the only thing left to choose is how the
    HIDDEN folds are spread, where equal-wide chunks subscribe every window 1:1 against the
    merged pair's one window at 2:1. So the second widening below is taken only while the cut
    keeps up with it, which leaves the merge in place wherever the cut cannot answer: B=1, and
    the 128-row band, where the cut is not taken at all.
    Measured on gptoss_full, own process, min of 40, palindrome-ordered, six readings an arm:
    5.5699 / 5.5824 / 5.5842 / 5.5849 / 5.5924 / 5.5990 on the equal-wide [2,2] plan with a
    four-piece tail against 5.6404 / 5.6638 / 5.6657 / 5.6693 / 5.6827 / 5.6848 on [1,2,1] with
    a two-piece one -- all six below all six, -1.45%. The same widening at the 128-row band,
    where no cut answers it, is d128_hq64_b2 +2.9% (6.4452 / 6.5037 / 6.5040 / 6.5054 against
    6.2939 / 6.2966 / 6.3011 / 6.3360), which is the gate below reading from the other side.
    """
    floor = _dq_pipe_fills(block_kv) * _NUM_CU
    per = 1
    while per * 2 <= q_split and wgs * per < floor:
        per *= 2
    if wgs * per < floor and per * 2 > q_split:
        return 0
    while per * 4 <= q_split and _dq_tail_cut(wgs, batch, per * 2, block_kv) >= 2 * _dq_tail_cut(
        wgs, batch, per, block_kv
    ):
        per *= 2
    return per


def _dq_tail_cut(wgs, batch, n_qsp, block_kv):
    """Batch pieces the exposed tail chunk is dispatched in (see _DQ_TAIL_CUT_FILLS).

    ``wgs`` is one subset's whole-batch grid, so a piece of an ``n_qsp``-wide tail holds
    ``wgs*n_qsp/pieces`` work-groups; halving keeps going while a piece is still worth
    _DQ_TAIL_CUT_FILLS fills, and while the batch divides EVENLY -- an uneven cut would hand
    the pieces ``batch//pieces`` batches each and leave the remainder undispatched. Two fills
    is also where it stops paying from below: at one fill gptoss_full's tail cuts into four
    256-work-group pieces and reads 6.6013 / 6.5740 against 5.6866 / 5.6465 at two, +16%.
    """
    if _DQ_TAIL_CUT_FILLS <= 0 or block_kv < 256:
        return 1
    floor = _DQ_TAIL_CUT_FILLS * _NUM_CU
    cut = 1
    while cut * 2 <= batch and batch % (cut * 2) == 0 and wgs * n_qsp // (cut * 2) >= floor:
        cut *= 2
    return cut


def _dq_grid_cut(wgs, batch, n_qsp, block_kv, fold_bytes):
    """Batch pieces EVERY chunk of the plan is dispatched in (see _DQ_CHUNK_FOLD_BYTES).

    The tail cut answers "can the last fold stop being exposed"; this answers the same question
    for the plan as a whole, and it is a different trade because a piece that is not the tail
    still has to HIDE a fold. Cutting one chunk narrows the window the chunk before it folds
    into, so the cut only pays when every chunk takes it: then the plan is a uniform grid over
    (q_split subset x batch), every window is subscribed 1:1, and the tail is one piece rather
    than one chunk. ``fold_bytes`` is what one whole chunk's fold reads, so a piece carries
    ``fold_bytes/pieces``, and halving stops while that clears _DQ_CHUNK_FOLD_BYTES, while a
    piece is still a legal chunk width (_dq_pipe_fills) and while the batch divides evenly.

    Taken at the NARROW band only, and that is the mirror of _dq_tail_cut rather than a second
    opinion about it: what a unit of body absorbs is 1/(2*BLOCK_KV) of partial bytes, so at the
    256-row band a hidden fold is cheap enough that the plan would rather stay WIDE and pay one
    boundary for the tail, while at the 128-row band the same over-subscription is twice as
    expensive and evenness is worth more than width. Measured from both sides on the two cells
    whose pieces carry the SAME 1 GiB, so bytes are not what separates them (paired A/B, 30
    pairs, arms alternating inside one process, the uniform plan against the tail-only one it
    replaces; the null control reads +0.011 ms / 18 of 30 at gptoss_full and +0.003 / 15 of 30
    at d128_hq64_b2): d128_hq64_b2 -0.0585 ms for the uniform plan with 27 of 30 pairs behind
    it, gptoss_full +0.0885 ms with 28 of 30 the other way. The same verdict on the deployed
    instrument -- own process, min of 40, palindrome-ordered inside one batch -- is 6.3238 /
    6.2244 / 6.3170 uniform against 6.3919 / 6.3811 / 6.3073 tail-only, -1.13%.
    """
    if block_kv >= 256:
        return 1
    floor = _dq_pipe_fills(block_kv) * _NUM_CU
    cut = 1
    while (
        cut * 2 <= batch
        and batch % (cut * 2) == 0
        and wgs * n_qsp // (cut * 2) >= floor
        and fold_bytes // (cut * 2) >= _DQ_CHUNK_FOLD_BYTES
    ):
        cut *= 2
    return cut


def _pipe_chunks(B, q_split, block_kv, seq_len_q, head_dim=64, sbhd=False, wgs=None):
    """Pipeline stages as (batch, qsp_lo, n_qsp), in dispatch order; batch None = all.

    ``wgs`` is what ONE q_split subset dispatches (``n_bands*Hkv*B``), which is what decides
    how many subsets a chunk has to hold to be worth the fill rule; None keeps the one-subset
    plans below. See _dq_pipe_qsp.

    A split owns the q blocks with (q/BLOCK_Q) % q_split == split only if every band
    starts on a q_split boundary; otherwise the split -> q-block map depends on the band
    and a sub-range launch would leave those rows half-summed (the SNR shape's narrower
    band is exactly that case), so the cut is only legal when the band divides evenly --
    or when the body re-phases its walk onto the absolute map itself (_qsp_absolute,
    which is how the D128 band gets its cut back).

    Cutting is not free: each chunk is its own dispatch, so its stragglers drain against
    an emptying machine instead of against the next chunk's work, and the grid is what
    prices that drain. A chunk holds ``(Skv/block_kv)*Hkv*n_qsp`` work-groups, so HALVING
    the band doubles them at equal n_qsp. That is the whole difference between the two
    head dims: on D64's 256-row band a quarter-batch chunk hands each CU a single
    work-group, the round's makespan goes to the longest band against a much lower mean,
    and only the last batch, only in half, wins; D128's 128-row band reaches the same
    work-groups-per-CU at n_qsp=1, so every batch can be cut to single splits. Measured
    at D128/bkv128/q_split=4 (D64 arm flat, 932-937 TF): n_qsp=1 on every batch 782.8 /
    781.7 TF vs 770.2 / 767.1 for the D64-shaped last-batch-half cut, and 776.9 for
    n_qsp=2 on every batch.

    SBHD's split cut (batch None) is a pure grid restriction, so unlike a TOKEN-major batch
    chunk it needs no contiguous per-batch slice of Q/dO/K/V -- which seq-major SBHD cannot
    give -- and the partials of the q blocks a split owns are complete the moment that split
    retires, in every batch at once.

    A finite window pins q_split=1, which leaves only the BATCH to cut on -- and that cut
    was measured and is not taken. Grid restriction makes it correct and bitwise identical
    (dq/dk/dv `torch.equal` against the uncut arm), but a windowed band is 52 KB of LDS
    rather than 84, so the body runs THREE work-groups per CU and the whole 2048-work-group
    grid is only 2.7 resident rounds: halving it leaves 1.33, and the third of a round the
    chunk cannot fill costs more than the reduce it hides. Paired, alternating, min of 40
    per reading: two batches per chunk 0.6619 against 0.5736 uncut, one batch 0.6451 --
    +15%. The fill rule below is therefore in RESIDENT ROUNDS, not work-groups per CU.
    """
    # SBHD MERGES THE PAIR AFTER THE FIRST CHUNK, i.e. [1,2,1] at q_split=4. The wall splits
    # into three terms, each priced by its own subtraction arm (gptoss_full, own process, min
    # of 40, the fold replaced by ONE dq fill for the body-only readings -- a fill per chunk
    # would price the chunk count into the probe): the BODY pays ~0.11 ms per chunk boundary
    # past the second, since the chunks share one queue and at one work-group per CU a
    # boundary drains the array and every chunk re-stages the whole band's K/V (body-only
    # 5.225 at two chunks, 5.338 at three, 5.453 at four); the LAST chunk's fold is exposed at
    # its full byte cost (the whole fold is 0.774 ms solo, so a one-subset tail is 0.194 and a
    # two-subset tail 0.387); and a fold that does hide still costs the body a share of its
    # bytes on the shared fabric. So a plan costs 0.113*(chunks-1) + 0.194*(tail subsets) plus
    # what its windows spill, and only the partitions with a one-subset tail are candidates.
    #
    # Fold i is issued after body i and runs under body i+1, so a plan's windows are the
    # consecutive pairs. [1,2,1] hides fold 0 (one subset) under a TWO-subset body and fold 1
    # (two subsets) under a one-subset body; [2,1,1] hides three subsets of fold under two
    # subsets of body instead. Same boundaries, same tail: the difference is only how much
    # body the same hidden bytes are spread over, and which way that trades is the partial
    # bytes per unit of body work, 1/(2*BLOCK_KV), twice as large at D128 as at D64.
    #
    # Measured on the three cells the pipeline reaches (the rest fall under _DQ_PIPE_FILLS),
    # each reading its own process, min of 40, palindrome-ordered so no arm keeps a position;
    # mean over 4-9 readings, the base row frozen and the percentages against the equal plan:
    #
    #   plan      gptoss_full        d64_hq128_16k      d128_hq64_b2
    #   base      5.9264             10.9444            6.4929
    #   [1,1,1,1] 5.8608             10.9055            6.3816
    #   [2,1,1]   5.8273             10.8704            6.4501
    #   [1,2,1]   5.7563  (-1.8%)    10.8464  (-0.5%)   6.5033  (+1.9%)
    #   [2,2]     5.8217             11.2054            6.6621
    #
    # [1,2,1] wins the target and the 16k guard and gives back part of the D128 guard's
    # margin, landing it within 0.2% of the baseline it is guarded against (min over nine
    # readings 6.4366, below it). Spreading the bytes is what buys it, not burst shape:
    # slicing the merged fold in proportion to its width, so each launch hands over the same
    # burst as an unmerged one, leaves d128_hq64_b2 where it was (6.5287 against 6.5204), and
    # moving the exposed tail to the reduce queue under this plan is level too (6.4810 against
    # 6.4894). [2,2] confirms the tail term from the other side -- one boundary fewer still,
    # and both guards lose to the doubled tail.
    #
    # The MERGE is taken at the wide band only, and the table above is what says so: the two
    # cells it wins are the two 256-row bands, and the one it loses is the 128-row one
    # ([1,1,1,1] 6.3816 against [1,2,1] 6.5033 on d128_hq64_b2 -- the "margin given back"
    # sentence above). Same term as everywhere else on this path: a merged pair hides a
    # TWO-subset fold under a one-subset body, a window subscribed 2:1, and what a unit of
    # body absorbs is 1/(2*BLOCK_KV), so halving the band doubles the over-subscription and
    # that one window stops covering. The equal cut pays one more boundary (~0.11 ms) for it
    # and keeps the same one-subset tail. Both readings above are UNCUT tails, which is why the
    # merge is now reached only where the batch cannot cut the tail (see _dq_pipe_qsp): where it
    # can, the equal-WIDE plan gets the same one-subset exposure without over-subscribing a
    # window, and wins by more than the boundary costs.
    # The SPLIT axis is what an SBHD chunk cuts, and the batch axis is not an alternative to it
    # even though bat_lo makes one dispatchable. A whole-batch-per-chunk plan gives the better
    # BODY -- one (batch, kv-head) per XCD instead of four, worth gptoss_full 5.229 against
    # 5.352 with the fold off -- and then loses all of it and more at the fold, because a chunk
    # that owns a whole batch owns a whole batch of PARTIALS too: gptoss_full 9.62 against
    # 5.68 (+70%) and d64_hq64_b2 4.23 against 2.94 (+44%), neither of which the slicing
    # explains (unsliced reads the same 4.19). Cutting BOTH axes at once to keep the subset's
    # fold cadence with the batch's locality is worse still, +15% on gptoss_full at eight
    # (batch pair, subset) chunks. The batch axis pays only where the alternative is exposure
    # rather than a different chunk -- i.e. on the tail, see _dq_tail_cut.
    if sbhd:
        # A shape whose one-subset chunk is under the fill rule takes the widest plan the rule
        # does license instead of no pipeline at all (see _dq_pipe_qsp). Equal chunks: the two
        # windows a merge trades between are the same width here, so there is nothing to trade.
        per = _dq_pipe_qsp(wgs, q_split, block_kv, B) if wgs else 1
        if per > 1:
            return [(None, s, min(per, q_split - s)) for s in range(0, q_split, per)]
        if q_split < 4 or block_kv < 256:
            return [(None, s, 1) for s in range(q_split)]
        return [(None, 0, 1), (None, 1, 2)] + [(None, s, 1) for s in range(3, q_split)]
    abs_map = block_kv % (q_split * _BWD_BLOCK_Q) == 0 or _qsp_absolute(head_dim, block_kv, q_split)
    sub = abs_map and _qsp_cuttable(seq_len_q, q_split)
    if head_dim == 128 and sub:
        return [(b, s, 1) for b in range(B) for s in range(q_split)]
    h = q_split // 2
    cut = q_split % 2 == 0 and sub
    return [(b, 0, None) for b in range(B - 1)] + (
        [(B - 1, 0, h), (B - 1, h, h)] if cut else [(B - 1, 0, None)]
    )


def _dq_fold_bytes(n_bat, Sq, Hq, D, Skv, block_kv, q_split, n_qsp):
    """What one chunk's dQ fold reads: the causal half of the bands, times the dQ image the
    chunk's batches hold, times its share of the q blocks (_reduce_dq_partials does the same
    accounting for the whole fold). It is what gates the batch cut (_dq_grid_cut) and what
    floors how small _fold_slices may cut.
    """
    return (Skv // block_kv) // 2 * (n_bat * Sq * Hq * D * 2) * (n_qsp or q_split) // q_split


def _fold_slices(B, Sq, q_split, lo, n, fold_bytes, bat_lo=0, n_bat=None):
    """One hidden dQ fold as ``(batch kwargs, qsp)`` slices, at most _DQ_FOLD_SLICES of them.

    The BATCH axis first, since a batch owns a contiguous slab of every band's partial row and
    of dQ. Then the q axis: split ``s`` of ``q_split`` is the q blocks with
    ``(q/BLOCK_Q) % q_split == s``, which is exactly splits ``{s, s+q_split}`` of
    ``2*q_split``, so re-expressing the same subset on a finer modulus splits the fold in half
    without changing WHICH rows it folds. Every slice keeps the whole ascending band walk and
    its own fp32 accumulator for the rows it owns, and no row is owned twice, so dQ is bitwise
    what a single launch produces -- checked directly against the unsliced schedule, not just
    through the determinism gate. Doubling stops where the finer splits would no longer tile
    the q blocks (see _qsp_cuttable), which would leave a remainder block unreduced.

    ``fold_bytes`` is what this whole fold reads, so the cut also stops while a slice would
    fall under _DQ_FOLD_SLICE_BYTES: past that the slices are the same rows and rounds but
    too few bytes each to pay for their own ramp (see the constants).
    """
    nb = B if n_bat is None else n_bat
    # ``n_qsp=None`` is "all q_split subsets"; name the count so a finer modulus can halve it.
    qs = [(q_split, lo, q_split if n is None else n)]
    while (
        nb * len(qs) * 2 <= _DQ_FOLD_SLICES
        and _qsp_cuttable(Sq, q_split * len(qs) * 2)
        and fold_bytes >= _DQ_FOLD_SLICE_BYTES * nb * len(qs) * 2
    ):
        qs = [(q * 2, l + j * q, m) for q, l, m in qs for j in (0, 1)]
    if nb > 1:
        bats = [dict(bat_lo=bat_lo + j, n_bat=1) for j in range(nb)]
    else:
        bats = [{}] if n_bat is None else [dict(bat_lo=bat_lo, n_bat=1)]
    return [(bat, q) for bat in bats for q in qs]


def _fused_pipelined(
    dkdv_l,
    odo_l,
    bufs,
    ws_dq,
    dq,
    B,
    Sq,
    Skv,
    block_kv,
    Hq,
    Hkv,
    D,
    q_split,
    stream,
    window_left=-1,
    sbhd=False,
    band_ring=0,
):
    """Run the fused kernel in chunks and hide each chunk's dQ reduce under the next one.

    The reduce is pure DRAM at the roofline while the fused kernel only uses 2.4 of the
    6.2 TB/s, so the reduce costs wall time only while nothing else is running. Two axes
    cut both kernels: BATCH -- a kv band owns a slice of dQ for its OWN batch only, so
    batch b's partials are complete the moment batch b's dkdv retires -- and the q_split
    SUBSET, once the body's q walk is on the absolute map (see _pipe_chunks). Either cut takes
    the delta pass with it -- a batch chunk by batch, a split chunk by q block -- so all but
    the first chunk's delta moves off the critical path into the first fused dispatch.

    Where the wall time goes, from the kernel trace (D64 SBHD, q_split=4, one backward):
    dispatches of 1354 us (the first chunk, nothing beside it) then 1419/1425/1479 us, each
    of the last three covering a whole 756-783 us reduce; then one exposed 338 us reduce with
    the 123 us dk/dv slot reduce inside it. So the body is 93% of the wall, the reduce is
    hidden except for its last chunk, and what co-residency costs the body is the difference
    between that first chunk and the others. The rest of the wall is a small uncoverable head
    (the lse prescale and the first chunk's own delta, which it reads) and a handful of
    dispatch-boundary bubbles: the event that fences a chunk off from its reduce is an HSA
    barrier packet.

    A TOKEN-major batch chunk's arguments are per-batch slices of the same buffers. The
    kernels' workspace strides are build-time constants of the FULL batch (dQ partials
    stride by ``batch_size`` bands, dk/dv slots by ``Q_SPLIT``), so a slice addresses
    exactly the rows the whole-batch launch would have given it -- same bands, same
    ascending order, same fp32 accumulator, hence bitwise-identical dQ/dK/dV.

    The chunks stay on ONE fused stream even though they are independent (a chunk owns its
    own batch, and where the last batch is cut its own q_split subset, so its dk/dv slots
    and dQ partial rows are disjoint from every other chunk's). Alternating them across two
    streams to fill each dispatch's causal drain is a measured LOSS on both cuts, and for
    two different reasons. A batch cut pays for the overlap itself: two chunks in flight
    means two batches' Q/dO/K/V resident at once, which widens the cache footprint enough
    to cost more than the drain. A SPLIT cut has no such footprint (the splits share their
    K/V), and it still loses gptoss_full +3.8% -- because what a chunk boundary costs at
    this width turns out to be near nothing, while running two chunks at a time delays every
    chunk's completion until the pair's, leaving TWO reduces at the end with nothing over
    them instead of one. The exposed tail, not the drain, is what the chunk count buys.

    Returns the event the caller must join before reading dQ; the dk/dv slot reduce is
    independent of dQ and is meant to be enqueued in front of that join.
    """
    qf, kf, vf, dof, o16, lsef, df, wk, wv, cu_ph = bufs
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
    # A stage is (batch, qsp_lo, n_qsp, bat_lo, n_bat): the first three are the plan's, the
    # last two the SBHD batch sub-range (None = the whole batch, see the builder's bat_lo).
    wgs = (Skv // block_kv) * Hkv * B
    plan = _pipe_chunks(B, q_split, block_kv, Sq, D, sbhd, wgs=wgs)
    chunks = [c if len(c) == 5 else tuple(c) + (0, None) for c in plan]
    # Cut the plan on the batch axis, so a fold gets a body over it that would otherwise be
    # exposed. Legal for SBHD without slicing a single tensor because the body takes its batch
    # range as a compile-time base plus the grid's count (see the builder's bat_lo), and every
    # piece folds only the rows its own body just completed -- same bands, same ascending
    # order, same fp32 accumulator, so dQ stays bitwise what one launch produces.
    # Two scopes, and they are alternatives rather than stages: cutting EVERY chunk keeps the
    # plan uniform and is what pays wherever a piece still carries a fold worth the boundary
    # (_dq_grid_cut); where it does not, only the tail -- the one fold with nothing over it --
    # is worth a boundary of its own (_dq_tail_cut). The plan the tail is cut out of answers to
    # that cut, not the other way round: the merged pair was chosen against an UNCUT tail, so
    # wherever the cut is available the plan widens instead of merging (see _dq_pipe_qsp).
    cuttable = sbhd and chunks[-1][0] is None
    n_tail = chunks[-1][2] or q_split
    # Only a plan that is already uniform on the SPLIT axis can stay uniform under the cut; the
    # merged pair (_pipe_chunks) is the one that is not, and it is chosen where no cut answers.
    uniform = len(chunks) > 1 and len({c[2] for c in chunks}) == 1
    grid_cut = (
        _dq_grid_cut(wgs, B, n_tail, block_kv, _dq_fold_bytes(B, Sq, Hq, D, Skv, block_kv, q_split, n_tail))
        if cuttable and uniform
        else 1
    )
    tail_cut = _dq_tail_cut(wgs, B, n_tail, block_kv) if cuttable and grid_cut == 1 else 1
    if grid_cut > 1:
        per = B // grid_cut
        chunks = [(b, lo, n, j * per, per) for b, lo, n, _, _ in chunks for j in range(grid_cut)]
    elif tail_cut > 1:
        b, lo, n, _, _ = chunks[-1]
        per = B // tail_cut
        chunks = chunks[:-1] + [(b, lo, n, j * per, per) for j in range(tail_cut)]
    nc = len(chunks)
    evs = _PIPE_EVENTS.get(nc)
    if evs is None:
        evs = [torch.cuda.Event() for _ in range(nc + 2)]
        _PIPE_EVENTS[nc] = evs
    ev_delta, ev_join = evs[nc], evs[nc + 1]
    per_batch = chunks[0][0] is not None
    cut_odo = False
    if per_batch:

        def _bat(t):
            """Batch b's contiguous block (only a token-major layout has one, see above)."""
            return list(t.view(B, -1))

        qb, dob = (_bat(t) for t in (qf, dof))
        kb, vb, wkb, wvb = (_bat(t) for t in (kf, vf, wk, wv))
        lb, db, ob = _bat(lsef), _bat(df), _bat(o16)
        odo_l(ob[0], dob[0], db[0], 1, Sq, stream)
        side.wait_stream(stream)
        for b in range(1, B):
            odo_l(ob[b], dob[b], db[b], 1, Sq, side)
        ev_delta.record(side)
    else:
        # A split chunk shares the whole batch, so the delta pass is cut on the SPLIT axis
        # instead: chunk 0 only reads the delta of the q blocks its own splits own, and the
        # rest -- three quarters of a 79 us pass that used to sit wholly in front of the
        # first chunk -- runs on the side queue underneath it, which the trace confirms it
        # finishes inside (58 us against a 1354 us first chunk). Worth gptoss_full -1.03%.
        # Only a SPLIT-axis plan leaves delta to cut: a batch chunk owns every q block of its
        # own batch, so the whole q range of delta is due before the first chunk either way.
        cut_odo = odo_l.chunk is not None and nc > 1 and chunks[0][2] not in (None, q_split)
        odo_first = odo_l.chunk(chunks[0][1], chunks[0][2]) if cut_odo else odo_l
        odo_first(o16, dof, df, B, Sq, stream)
        side.wait_stream(stream)
        if cut_odo:
            rest = chunks[0][1] + chunks[0][2]
            odo_l.chunk(rest, q_split - rest)(o16, dof, df, B, Sq, side)
            ev_delta.record(side)
    for i, (b, lo, n, blo, nbt) in enumerate(chunks):
        if i == 1 and (per_batch or cut_odo):
            stream.wait_event(ev_delta)
        if per_batch:
            args = (qb[b], kb[b], vb[b], dob[b], lb[b], db[b], wkb[b], wvb[b])
            ws_arg, nb, bat = ws_dq[0, b], 1, dict(bat_lo=b, n_bat=1)
        else:
            args = (qf, kf, vf, dof, lsef, df, wk, wv)
            ws_arg, nb = ws_dq[0, 0], B if nbt is None else nbt
            bat = {} if nbt is None else dict(bat_lo=blo, n_bat=nbt)
        # The LAST chunk's reduce is the one whose queue is a free choice: it has no next
        # chunk to hide under, so on the reduce queue it runs beside whatever the caller
        # enqueues next (the dk/dv slot fold) and on the caller's it runs in front of it.
        # A split chunk's tail reduce walks 1/q_split of every band and saturates DRAM by
        # itself, so running the slot fold alongside only halves both -- serialising them is
        # gptoss_full -0.51% and d64_hq128_16k -0.59% (six alternating pairs, own process,
        # min of 40 each; on the target all six readings fall below all six of the other
        # arm), with d128_hq64_b2 level on the min and -1.09% on the mean over twelve.
        # A per-batch chunk's tail reduce is one batch of a far smaller workspace and does
        # not saturate, so there the two folds genuinely do overlap and serialising them
        # costs varlen_d64_u +1.2%: that branch keeps the reduce queue.
        red_q = stream if (_DQ_TAIL_SERIAL and not per_batch and i == nc - 1) else side
        body = dkdv_l if (not blo and n in (None, q_split)) else dkdv_l.chunk(lo, n, blo)
        body(
            *args,
            cu_ph,
            cu_ph,
            ws_arg.reshape(-1),
            nb,
            Sq,
            Skv,
            0,
            stream,
        )
        if red_q is not stream:
            evs[i].record(stream)
            red_q.wait_event(evs[i])
        # A HIDDEN fold is issued in narrow slices (see _fold_slices), so the body it runs
        # under is not held off by one wide burst; the size of a burst shows up on its own on
        # d64_hq128_16k, where the SAME hidden bytes cost 0.32 ms delivered in three windows
        # and 0.655 ms in two. The EXPOSED tail keeps its single launch: it has nothing to run
        # under, so there is no burst to break up and slicing only costs it the extra ramps
        # (5.874 against 5.854). A per-batch chunk is already one batch of a far smaller
        # workspace and takes the plain launch too.
        #
        fold_bytes = _dq_fold_bytes(nbt or B, Sq, Hq, D, Skv, block_kv, q_split, n)
        slices = (
            _fold_slices(B, Sq, q_split, lo, n, fold_bytes, bat_lo=blo, n_bat=nbt)
            if not per_batch and (n is not None or nbt is not None) and i < nc - 1
            else [(bat, (q_split, lo, n))]
        )
        for bat_kw, qsp in slices:
            _reduce_dq_partials(
                ws_dq,
                dq,
                block_kv,
                Hq,
                D,
                1.0 / _LOG2E,
                red_q,
                qsp=qsp,
                causal_offset=Skv - Sq,
                window_left=window_left,
                sbhd=sbhd,
                band_ring=band_ring,
                ph=cu_ph,
                **bat_kw,
            )
    ev_join.record(side)
    return ev_join


def _fused_bandgroups(
    dkdv_l, bufs, ws_dq, carry, dq, B, Sq, total_kv, block_kv, Hq, D, band_span, n_bands, stream, cu=None
):
    """Run the fused kernel one BAND GROUP at a time, carrying the dQ sum between groups.

    The dQ partial workspace is bands*|dQ|, i.e. quadratic in the context, and it is the only
    thing the split pair still wins on at a long context (its Q-outer dq kernel needs no
    workspace at all). A group launch owns ``band_span`` bands, so the workspace it needs is
    the group's, not the axis's; what crosses a group boundary is one fp32 dQ image, which the
    reduce reads at the start of a group and writes back at the end unless the group holds the
    row's top band. Ascending groups, ascending bands within a group, one writer per slot and
    an fp32 accumulator that is only ever spilled and reloaded exactly -- so the result is
    bitwise what the single-group path produces, which is worth testing that way.

    ``cu`` switches the same walk onto a RAGGED batch, where the bands are per segment and the
    rows are packed q tokens. A group then owns bands [lo, lo+span) OF EVERY SEGMENT: the
    segments occupy disjoint rows of each slot, so one writer per slot survives untouched, and
    a segment whose kv length stops below the group simply has no live tile there. Because the
    kv extent comes from cu_seqlens rather than from the group, the span need not tile the band
    axis -- the last group's surplus bands sit above every segment and retire empty.

    The groups run back to back on ONE stream. A second set of slots would let a group's reduce
    run beside the next group's body, but at this context the body already has the memory system
    saturated writing its own partials, so the two only contend; the slots buy more as a longer
    group, which is what shrinks the carry.
    """
    qf, kf, vf, dof, lsef, df, wk, wv, cu_q = bufs
    # The group's first band rides a tensor slice, not a scalar argument, so the launch cache
    # does not see a new signature per group (see _band_lo_table); a ragged launch needs the
    # segment table in that same slot and appends the band to it (see _cu_band_rows).
    slots = (
        _cu_band_rows(cu[1], n_bands, band_span)
        if cu is not None
        else _band_lo_table(n_bands, band_span, dq.device).unsqueeze(1)
    )
    for g in range(slots.shape[0]):
        lo = slots[g]
        dkdv_l(
            qf,
            kf,
            vf,
            dof,
            lsef,
            df,
            wk,
            wv,
            cu_q,
            lo,
            ws_dq[0, 0].reshape(-1),
            B,
            Sq,
            band_span * block_kv,
            total_kv,
            stream,
        )
        _reduce_dq_partials(
            ws_dq,
            dq,
            block_kv,
            Hq,
            D,
            1.0 / _LOG2E,
            stream,
            causal_offset=0,
            sbhd=cu is None,
            cu=cu,
            band=(band_span, lo, carry),
            ph=cu_q,
        )


def _get_bwd(
    Hq,
    Hkv,
    D,
    scale,
    window_left,
    q_split,
    block_kv,
    batch_size=None,
    sbhd=False,
    varlen=False,
    square=True,
    wsq_ilv=1,
    wsq_ring=0,
    band_span=0,
):
    key = (
        Hq,
        Hkv,
        D,
        scale,
        window_left,
        q_split,
        block_kv,
        batch_size,
        sbhd,
        varlen,
        square,
        wsq_ilv,
        wsq_ring,
        band_span,
    )
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
        _swa = window_left >= 0
        # dkdv's read/MFMA ratio favors a wide tile, but that only fits spill-free at
        # waves_per_eu=1; dma_grp/pf_ring amortize the Q/dO staging rendezvous, at LDS cost
        # only the wide-tile (one wave per SIMD) configuration has spare. The fused
        # wide-band body needs one wave per SIMD for its dK/dV accumulators regardless,
        # which is also what lets the dQ reduce co-reside (see `_fused_pipelined`).
        _fuse_wide = D * block_kv >= 16384
        _fuse_d128 = _fuse_wide and D == 128
        _fuse_halves = max(1, D * block_kv // 16384)
        _pair = _fuse_halves > 1
        # A windowed band's q loop is three blocks long, so a second wave per SIMD has
        # nothing left to interleave with: asking for one buys the body the whole
        # register budget instead (the two-wave group below is already the occupancy).
        dkdv_wpe, dkdv_dma_grp = 1 if (_fuse_wide or _swa) else 2, 1
        # GEMM2's dt prefetch ring: depth 2 pays at D128 and LOSES at D64 -- it trades 26 of
        # the hot loop's s_waitcnt away and still measures gptoss_full +1.9% over six of six
        # palindrome pairs, the wider read burst blocking MFMA issue for more than the waits
        # were worth. Fewer waits is not the objective; MFMA-run density is.
        dkdv_g2d = 2 if _fuse_d128 else 1
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
            kv_halves=_fuse_halves,
            wsq_pad=(_WSQ_BAND_PAD if D == 128 else 0),
            wsq_ilv=wsq_ilv,
            wsq_ring=wsq_ring,
            band_span=band_span,
            q_pref=not _pair,
            g3_defer=_fuse_d128 and not _pair,
            g3_dbat=2 if (_fuse_d128 and not _pair) else None,
            g3_kreg=_fuse_wide and not _pair,
            k_reg=not _pair,
            g3d=2 if _pair else None,
            # The dQ partial store is this path's one uncovered burst: emitted inside GEMM3 it
            # lands where the carrier wave has no MFMA left to hide it, and at one wave per SIMD
            # there is no sibling wave to cover it either. Hand it to the head-step hook instead,
            # which spreads the same stores two at a time across the softmax run that follows,
            # and keep the sched_barrier so LLVM cannot sink them back into one burst.
            # All three reach the ISA only through the DEFERRED GEMM3 emission (_g3_pend /
            # st_sink), and g3_defer above is off at D64 -- so "opening them to the D64 wide
            # band" is not a knob there: forced on, the D64 full-causal dkdv kernel compiles to
            # the same bytes (md5-identical 21_final_isa.s, vgpr 460, spill 0, LDS 86016, all
            # three loop histograms equal). The gptoss_full -0.28% over six alternating pairs,
            # varlen -3.8%, d64_hq128_16k +1.39% and d64_hq64_b1 +0.74% once recorded here were
            # all noise on ONE binary, and the shape-gate argument built on them is moot: a
            # rotated re-read of the same arm still reads -0.19% over six trials, which is this
            # host's amplitude on gptoss_full, not a signal. Moving the D64 store placement
            # means moving the emission point on the UNDEFERRED path.
            g3_st_at=2 if _fuse_d128 else None,
            g3_st_n=2 if _fuse_d128 else None,
            g3_sb=1 if _fuse_d128 else None,
            # A windowed band is only BLOCK_KV + W q rows wide, so the default four-wave
            # split gives each wave a single kv tile and a repeated Q/dO fragment read;
            # halving to two waves shares that fragment read across two tiles per wave.
            flat_wg=256 if (not _swa or block_kv >= 128) else 128,
            # Every knob below was independently re-swept on the FQ_PAIR body since its
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

        def _dkdv_chunk(qsp_lo, n_qsp, bat_lo=0):
            """Same body, dispatching only the q_split / batch sub-range (see _fused_pipelined)."""
            sub = _dkdv_subs.get((qsp_lo, n_qsp, bat_lo))
            if sub is None:
                sub = build_flash_attn_bwd_dkdv_module(
                    qsp_lo=qsp_lo, n_qsp=n_qsp, bat_lo=bat_lo, **dkdv_kw
                )
                _dkdv_subs[(qsp_lo, n_qsp, bat_lo)] = sub
            return sub

        dkdv_l.chunk = _dkdv_chunk
        # DELTA has no other producer, so the standalone odo pass runs; ragged wants it
        # packed by token, the layout its LSE has.
        odo_kw = dict(num_heads=Hq, head_dim=D, sbhd=sbhd, token_major=varlen)
        odo_l = build_flash_attn_bwd_odo_module(q_split=q_split if sbhd else 1, **odo_kw)
        _odo_subs: dict = {}

        def _odo_chunk(qsp_lo, n_qsp):
            """Same delta pass, dispatching only the q_split sub-range (see _fused_pipelined)."""
            sub = _odo_subs.get((qsp_lo, n_qsp))
            if sub is None:
                sub = build_flash_attn_bwd_odo_module(q_split=q_split, qsp_lo=qsp_lo, n_qsp=n_qsp, **odo_kw)
                _odo_subs[(qsp_lo, n_qsp)] = sub
            return sub

        odo_l.chunk = _odo_chunk if sbhd and q_split > 1 else None
        launchers = (dkdv_l, odo_l)
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
        num_seg = cu_seqlens_q.numel() - 1
        total_q, total_kv = q.shape[0], k.shape[0]
        max_sq = int(max_seqlen_q) if max_seqlen_q is not None else Sq
        max_skv = int(max_seqlen_kv) if max_seqlen_kv is not None else Skv
        # The ragged body already reads its segment bounds from cu_seqlens; what the fusion
        # adds is a dQ partial workspace packed by q TOKEN, whose band a segment enters at its
        # own token base, and a reduce that derives each row's band window from that row's
        # segment.
        _assert_fusable(Hq, D)
        block_kv = _fuse_blockkv_for(max_skv, D, window_left)
        q_split = _fuse_qsplit_for(max_sq, total_kv, Hkv, block_kv, window_left)
        n_bands = (max_skv + block_kv - 1) // block_kv
        # Packed rows make the workspace bands*total_q, which a long ragged batch outgrows the
        # same way a long dense one does. A window bounds the bands a q block writes, so its
        # slots can be shared (see _wsq_ring_for); full causal has no such bound and walks the
        # band axis in groups instead (see _band_span_for). Rows are packed, so ilv stays 1.
        band_bytes = total_q * Hq * D * 2
        wsq_ring = _wsq_ring_for(n_bands, block_kv, window_left, 1, band_bytes)
        band_span = _band_span_for(n_bands, band_bytes, 1, whole=False) if window_left < 0 else 0
        dkdv_l, odo_l = _get_bwd(
            Hq,
            Hkv,
            D,
            scale,
            window_left,
            q_split,
            block_kv,
            batch_size=num_seg,
            sbhd=False,
            varlen=True,
            square=False,
            wsq_ring=wsq_ring,
            band_span=band_span,
        )
        delta = torch.empty(total_q, Hq, device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        ws_dk = torch.zeros(q_split, total_kv, Hkv, D, device=q.device, dtype=k.dtype)
        ws_dv = torch.zeros(q_split, total_kv, Hkv, D, device=q.device, dtype=v.dtype)
        lsef, df = lse_s.reshape(-1), delta.reshape(-1)
        # The grid tiles kv by max_seqlen_kv, so the band count follows it; a segment
        # neither writes nor reads the bands above its own kv length (its rows stop at
        # g = (Sq_seg-1+off)/block_kv), so those slots need no fill.
        # dQ needs no reduction over q_split either: the splits of one band own disjoint
        # q blocks, so every (band, packed row) slot still has exactly one writer and the
        # band-ascending fp32 sum below is the only accumulation dQ ever sees.
        ws_dq, ws_carry = _dq_partial_ws(
            band_span or wsq_ring or n_bands,
            1,
            total_q,
            Hq * D,
            q.device,
            q.dtype,
            _WSQ_BAND_PAD if D == 128 else 0,
            carry=bool(band_span),
        )
        odo_l(o16, dof, df, 1, total_q, st)
        if band_span:
            _fused_bandgroups(
                dkdv_l,
                (qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1), cu_seqlens_q),
                ws_dq,
                ws_carry,
                dq,
                num_seg,
                max_sq,
                total_kv,
                block_kv,
                Hq,
                D,
                band_span,
                n_bands,
                st,
                cu=(cu_seqlens_q, cu_seqlens_kv),
            )
        else:
            dkdv_l(
                qf,
                kf,
                vf,
                dof,
                lsef,
                df,
                ws_dk.reshape(-1),
                ws_dv.reshape(-1),
                cu_seqlens_q,
                cu_seqlens_kv,
                ws_dq[0, 0].reshape(-1),
                num_seg,
                max_sq,
                max_skv,
                total_kv,
                st,
            )
            _reduce_dq_partials(
                ws_dq,
                dq,
                block_kv,
                Hq,
                D,
                1.0 / _LOG2E,
                st,
                window_left=window_left,
                cu=(cu_seqlens_q, cu_seqlens_kv),
                band_ring=wsq_ring,
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
    q_split = _qsplit_for(Sq, window_left, D)
    # Every causal shape rides the fused path, down to the smallest: bands are ceil-counted so
    # a non-aligned Skv keeps its ragged top band, rectangular (Skv>Sq) bottom-right causal
    # needs nothing extra (the G3 dQ emission and the reduce are both causal_offset-aware), and
    # the reduce auto-tiles whatever Hq reaches here.
    # SWA (window_left>=0) rides it too -- the G3 q-loop stops at the windowed _qhi and the
    # reduce clamps its band range with a lower edge g_lo, both wrapped in const_expr so
    # full-causal ISA stays byte-identical.
    _assert_fusable(Hq, D)
    block_kv = _fuse_blockkv_for(Skv, D, window_left)
    # ceil so a non-aligned Skv keeps its ragged top band (the body ceil-grids kv and
    # masks OOB keys; only the workspace band count was floor).
    n_bands = (Skv + block_kv - 1) // block_kv
    # ilv packing assumes whole band groups; a non-aligned Skv's ragged top band breaks
    # it, so interleave only when Skv tiles the band exactly. D64 full-causal is excluded by
    # measurement, twice: flat before QDESC opened there (gptoss_full -0.14%) and a loss
    # after it (ten alternating readings each, mean 5.9721 against 5.9479 without, four of
    # five cycle pairs), so the plain slab keeps the simpler addressing.
    wsq_ilv = (
        _wsq_ilv(n_bands, B, Sq, Hq * D) if Skv % block_kv == 0 and (D == 128 or window_left >= 0) else 1
    )
    # Long context: the dQ partial workspace is bands*|dQ| and outgrows the card, so walk the
    # band axis in groups instead of asking for all of it at once (see _band_span_for).
    band_span = (
        _band_span_for(n_bands, B * Sq * Hq * D * 2, wsq_ilv)
        if sbhd and window_left < 0 and Sq == Skv and Skv % block_kv == 0
        else 0
    )
    # A window bounds the band span a q block writes, so the same footprint problem is
    # answered without any pass structure at all: the bands share slots (see _wsq_ring_for).
    wsq_ring = _wsq_ring_for(n_bands, block_kv, window_left, wsq_ilv, B * Sq * Hq * D * 2)
    dkdv_l, odo_l = _get_bwd(
        Hq,
        Hkv,
        D,
        scale,
        window_left,
        q_split,
        block_kv,
        batch_size=B,
        sbhd=sbhd,
        square=(Sq == Skv),
        wsq_ilv=wsq_ilv,
        wsq_ring=wsq_ring,
        band_span=band_span,
    )
    # identity delta = -rowsum(O.dO); the body centers dP by it (exact).
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    # The pipeline overlaps each chunk's dQ reduce with the next chunk's compute; it only pays
    # when the chunk's own compute dwarfs the dispatch it costs (see _DQ_PIPE_AREA_FLOOR).
    pipe = (
        _DQ_PIPE
        and not band_span  # band groups drive their own dispatch order (see _fused_bandgroups)
        # a ragged top band makes the split->q-block map band-dependent (see _pipe_chunks),
        # so a non-aligned Skv must take the single whole-batch dispatch.
        and Skv % block_kv == 0
        and Sq * (Skv if window_left < 0 else window_left + block_kv) > _DQ_PIPE_AREA_FLOOR
        # An SBHD chunk cuts the SPLIT axis: that needs q_split>1 (which a window never has)
        # and the body's band-relative q walk has to agree with the absolute map the reduce
        # and the cut both model -- either because the band is a whole number of split
        # strides (D64's 256-row band over q_split=4) or because the body re-phases onto the
        # absolute map (_qsp_absolute, which is how D128's half-width band gets its cut
        # back). The last term is the width rule: a chunk that no longer fills the CU array
        # pays a whole extra round of makespan for the reduce it hides (see _DQ_PIPE_FILLS).
        and (
            (
                q_split > 1
                and (block_kv % (q_split * _BWD_BLOCK_Q) == 0 or _qsp_absolute(D, block_kv, q_split))
                and _qsp_cuttable(Sq, q_split)
                and _dq_pipe_qsp(n_bands * Hkv * B, q_split, block_kv) > 0
            )
            if sbhd
            else B > 1
        )
    )
    if not pipe:
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
    lsef = lse_s.reshape(-1)
    df = delta.reshape(-1)
    cu_ph = _cu_placeholder(q.device)
    ws_dq, ws_carry = _dq_partial_ws(
        band_span or wsq_ring * wsq_ilv or n_bands,
        B,
        Sq,
        Hq * D,
        q.device,
        q.dtype,
        _WSQ_BAND_PAD if D == 128 else 0,
        wsq_ilv,
        carry=bool(band_span),
    )
    if band_span:
        _fused_bandgroups(
            dkdv_l,
            (qf, kf, vf, dof, lsef, df, ws_dk.reshape(-1), ws_dv.reshape(-1), cu_ph),
            ws_dq,
            ws_carry,
            dq,
            B,
            Sq,
            Skv,
            block_kv,
            Hq,
            D,
            band_span,
            n_bands,
            st,
        )
    elif pipe:
        bufs = (
            qf,
            kf,
            vf,
            dof,
            o16,
            lsef,
            df,
            ws_dk.reshape(-1),
            ws_dv.reshape(-1),
            cu_ph,
        )
        join_ev = _fused_pipelined(
            dkdv_l,
            odo_l,
            bufs,
            ws_dq,
            dq,
            B,
            Sq,
            Skv,
            block_kv,
            Hq,
            Hkv,
            D,
            q_split,
            st,
            window_left=window_left,
            sbhd=sbhd,
            band_ring=wsq_ring,
        )
    else:
        # Pass ONE (band, batch) slice: the kernel rebases the SRD to its own slice with
        # a 64-bit offset, and the whole workspace overflows a flat memref's i32 count.
        dkdv_l(
            qf,
            kf,
            vf,
            dof,
            lsef,
            df,
            ws_dk.reshape(-1),
            ws_dv.reshape(-1),
            cu_ph,
            cu_ph,
            ws_dq[0, 0].reshape(-1),
            B,
            Sq,
            Skv,
            0,
            st,
        )
        _reduce_dq_partials(
            ws_dq,
            dq,
            block_kv,
            Hq,
            D,
            1.0 / _LOG2E,
            st,
            causal_offset=Skv - Sq,
            window_left=window_left,
            sbhd=sbhd,
            band_ring=wsq_ring,
            ph=cu_ph,
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
