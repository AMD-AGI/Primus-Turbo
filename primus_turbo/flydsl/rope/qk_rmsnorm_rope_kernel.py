###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""GPT-OSS packed-QKV RMSNorm + RoPE FlyDSL kernels.

The production tensor is ``[S, B, NG, (NPG + 2) * D]`` with one packed group
laid out as ``[NPG query heads, key, value]``.  GPT-OSS-20B uses
``NG=8, NPG=8, D=64``.

Lane / row tiling (vectorized, campaign round P0a): each 64-element head row
is split into two 32-element halves (``lo`` = elements ``[0, 32)``, ``hi`` =
elements ``[32, 64)``).  ``_LANES_PER_ROW = 4`` lanes cover one row, each lane
owning ``_EPL = 8`` contiguous elements of a half, so one ``dwordx4`` BF16
load/store moves a lane's whole chunk.  A wave's 64 lanes therefore cover
``_ROWS_PER_WAVE = 16`` complete head rows per iteration instead of one row
per wave, quadrupling useful bytes per instruction and amortizing the loop
and gamma/cos/sin overhead over 16 rows at a time.

Megatron's rotary table is built as ``cat(half, half)``, so ``cos[d] ==
cos[d + 32]`` (same for ``sin``) for every row.  ``_load_f32x8`` loads one
8-element FP32 chunk per lane via two ``dwordx4`` reads and the kernels reuse
that single chunk for both the ``lo`` and ``hi`` halves ("cos/sin half-table"),
halving the rotary-table instruction count versus loading it twice.

Forward consumes packed QKV directly and emits contiguous Q/K/V.  Q and K are
RMS-normalized, explicitly rounded to BF16 (matching the materialized output of
the existing RMSNorm kernel), and then rotated.  V is copied unchanged.

Backward performs inverse RoPE, explicitly rounds that result to BF16 (matching
the materialized gradient between the existing RoPE and RMSNorm kernels), then
applies RMSNorm backward and writes directly into packed dQKV.  Each persistent
wave folds its ``_ROWS_PER_WAVE`` per-row dgamma accumulators once at the end of
its token loop and writes one FP32 dgamma partial per logical head, for a
deterministic host-side fold.

Campaign round P0b: the forward no longer materializes ``rstd``.  Its
``QRSTD``/``KRSTD`` outputs become a single cached ``(1,)`` FP32 tensor
holding ``eps`` (``_eps_tensor``, keyed on device + the Python ``eps`` value --
configuration, never a tensor ``id()``, see Rule 11); the autograd wrapper
only ever passes these two slots through from forward's return to backward's
input, so repurposing their contents changes no external contract.  The
backward loads that scalar once per wave and recomputes ``rstd`` from the
packed row it already reads for ``u = x * rstd``, using the exact same
per-row reduction (``_row_sum_f32`` over the ``_EPL`` lo/hi pairs) and the
same ``fastmath="afn"`` rsqrt the forward used, so the recomputed value is
bit-identical and ``dqg``/``dkg`` stay exactly as bit-exact as before.  This
trades one extra sum-of-squares + rsqrt per row for one fewer FP32 store
(forward) and one fewer FP32 load (backward); both kernels are
memory/occupancy-bound rather than ALU-bound, so the traffic/VGPR reduction
nets positive (campaign rounds ANALYZE-r2, OPTIMIZE-r2).

Campaign round P0c: the backward's row-to-wave mapping changed from
token-strided (one wave owns ``_ROWS_PER_WAVE`` token-strided rows of a
*single* head slot, scattered across a ~160 KB device range) to contiguous
(one wave owns ``_ROWS_PER_WAVE`` *contiguous* head rows of a single sequence
position).  A Q wave now owns one ``(batch, block of 16 consecutive query
heads)`` pair -- two 1 KB runs 1280 B apart instead of 16 cache lines spread
over 160 KB; a K or V wave owns 8 group rows x 2 batches of one sequence
position.  ``_bwd_slot_counts`` derives the new slot count (``packed_heads``
80 -> 20), so ``_BWD_GRID_CYCLES`` is re-swept for it (256 -> 2048, the best
scored point of the post-remap sweep).  This changes only *which* rows a wave
reads/writes and the shape of the dgamma partial-row tensor the un-editable
wrapper folds -- not how many bytes move, not any row's own per-element
arithmetic, and not which physical rows ultimately get summed into dgamma --
and parity holds bit-exact (``dqg``/``dkg`` maxabs 0.0, re-verified repeatedly
on this exact code; campaign ANALYZE round 3, OPTIMIZE round 3).
``_xcd_chunk`` additionally remaps the backward's workgroup id so one cycle's
~20 workgroups land on one gfx950 XCD (hence one L2 slice) instead of
round-robining across all 8; measured neutral once the contiguous row map
already collapses a cycle's cos/sin reuse to one 128 B line, but it costs
nothing extra, so it rides along unconditionally.

Host-side launches go through a small per-configuration call-state cache
(``_CALL_STATE_CACHE``, keyed on shape/dtype — never on tensor ``id()``, see
Rule 11) that re-dispatches a previously bound FlyDSL launcher directly instead
of re-binding the signature on every call.

Campaign round Q6 (host/Python issue cost): the ruler brackets the whole
Python call, so every per-call Python-side op not strictly needed on the hot
(already-cached) path is scored.  Two fixes, both preserving every external
contract: (1) ``_cached_cos_sin`` used to call ``torch.cuda.current_stream``
itself, redundant with the wrapper's own call for the kernel launch -- the
wrapper now reads the stream once and threads it in.  (2) ``_launch``'s hot
path (``state(args)``, the common case once a shape/dtype has been seen
once) never reads its ``kwargs`` dict at all -- only the cold path
(``compiled(**kwargs)`` / signature re-binding) does -- so the wrapper now
passes a zero-arg callable instead of a pre-built dict, and ``_launch`` only
calls it on the cold path; building a ~15-key dict every call just to throw
it away on every steady-state call (the overwhelming majority, in both the
benchmark loop and real training once a shape stabilizes) was pure waste.
Both the forward and backward wrapper also had the same configuration value
(``eps``/the stream) recomputed 2-3x per call; each is now computed once and
reused.  None of this changes which tensors are read, written, or cached --
only how many times an already-known Python value is re-derived per call.

Campaign round Q2 (dgamma-fold decoupling): the un-editable autograd wrapper
always does ``dqg_partial.sum(dim=0).to(q_gamma.dtype)`` (and the k
equivalent) on whatever shape/dtype ``flydsl_qkv_rmsnorm_rope_backward``
returns.  Measured in isolation, a naive in-kernel global ``atomic_add``
collapsing every wave's partial directly into one shared accumulator costs
~2.7x-31x more than an uncontended unique-address store as wave count grows
(an isolated probe at this shape's real Q-side wave count, 32768, measured
~816 us for the atomic version alone -- a >20x regression versus the ~20 us
the Q side's slice of the fold costs today), so it is not used.  Instead, a
second tiny FlyDSL kernel (``_make_fold_kernel``), launched on the same
stream right after the main backward kernel via the same ``_launch`` fast
path, fold-reduces DQG_PART/DKG_PART (unchanged: still P0c's row mapping,
still ``_BWD_GRID_CYCLES``) down to ``(R/_FOLD_ROWS_PER_WAVE, D)`` f32
tensors first.  No atomics: each wave still owns a unique, disjoint
``_FOLD_ROWS_PER_WAVE``-row range and writes a unique output row, the same
safety property the main kernel's own partial store already has, so
``dqg``/``dkg`` parity stays bit-exact (maxabs 0.0).  Measured on real
hardware (rocprofv3), the fold kernel itself costs only ~8 us of GPU time
(576 workgroups, 28 VGPRs); the win comes from replacing the wrapper's two
``sum(dim=0).to(dtype)`` calls on 8 MB + 1 MB tensors (~32.7 us combined,
dominated by per-call/launch floor rather than bandwidth -- shrinking the
row count 2048x only saved a few us) with that cheap kernel plus the same
two calls on a ~128x smaller tensor.  ``rows_per_wave`` was swept
(4/8/16/32/256): a loop-carried single accumulator chain over large
per-wave row counts was far worse than doing nothing (more serialized
dependent loads than the bandwidth it saved), and too few workgroups
under-occupies the GPU; ``_FOLD_ROWS_PER_WAVE = 16`` with a fully unrolled,
independent-accumulator reduction (``_accumulate_rows``) was the best of the
sweep (campaign OPTIMIZE round 8).
"""

from __future__ import annotations

import weakref

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, buffer_ops, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec

# CODE-REVIEW fix: `arith` (flydsl.expr.arith, imported above) already
# re-exports `_to_raw` as its own public (if deprecated) surface; reach
# through that instead of importing from the private `flydsl.expr.utils.arith`
# implementation submodule directly.
_raw = arith._to_raw

QK_RMSNORM_ROPE_HEAD_DIM = 64
_D = QK_RMSNORM_ROPE_HEAD_DIM
_HALF = _D // 2
_WARP = 64
_WAVES = 4
_BLOCK_THREADS = _WARP * _WAVES

_LANES_PER_ROW = 4
_EPL = _HALF // _LANES_PER_ROW  # 8 elements per lane per half
_ROWS_PER_WAVE = _WARP // _LANES_PER_ROW  # 16


def _check_row_tileable(B: int, NG: int, NPG: int) -> None:
    """CODE-REVIEW fix: the P0a row/lane tiling packs ``_ROWS_PER_WAVE`` (16)
    head rows into one wave and derives slot counts via plain integer
    division (``q_heads // _ROWS_PER_WAVE``, ``(B * NG) // _ROWS_PER_WAVE``
    in both ``_make_fwd_kernel`` and ``_bwd_slot_counts``).  For shapes where
    ``NG * NPG`` or ``B * NG`` is not an exact multiple of 16 -- e.g. every
    shape in ``tests/pytorch/ops/test_qk_rmsnorm_rope.py`` (NG=1 or NG=2,
    B=1 or B=4) -- that division silently truncates, in the worst case to a
    slot count of **zero**, which launches a kernel that does no work at all
    and leaves the (``torch.empty``) output buffers uninitialized instead of
    raising.  Only the production GPT-OSS-20B shape (NG=8, NPG=8, B a
    multiple of 2) was validated against this tiling; fail loudly here
    instead of silently returning garbage for anything else.
    """
    q_heads = NG * NPG
    if q_heads % _ROWS_PER_WAVE:
        raise ValueError(
            f"qk_rmsnorm_rope: NG*NPG={q_heads} must be a multiple of "
            f"_ROWS_PER_WAVE={_ROWS_PER_WAVE} for the vectorized row tiling "
            "(campaign round P0a); this shape is unsupported by the current kernel."
        )
    if (B * NG) % _ROWS_PER_WAVE:
        raise ValueError(
            f"qk_rmsnorm_rope: B*NG={B * NG} must be a multiple of "
            f"_ROWS_PER_WAVE={_ROWS_PER_WAVE} for the vectorized row tiling "
            "(campaign round P0a); this shape is unsupported by the current kernel."
        )


# One cycle is one persistent wave per logical 16-row group.  Both counts sit
# on the flat part of their measured sweep (campaign ANALYZE rounds 1-2).
_FWD_GRID_CYCLES = 4096
# Campaign round P0c re-swept this after the backward's row remap dropped
# slots/cycle from packed_heads=80 to 20 (see _bwd_slot_counts below); 2048
# is the best-scored point of that post-remap sweep (ANALYZE round 3).
_BWD_GRID_CYCLES = 2048

# Campaign round Q2 (dgamma-fold decoupling): the un-editable wrapper always
# does `dqg_partial.sum(dim=0).to(gamma.dtype)` / same for k on whatever
# shape this module returns.  A naive in-kernel global atomic_add collapsing
# every wave's partial directly into one (D,) accumulator was measured (an
# isolated contention probe, same wave counts as this shape) to cost ~800 us
# at the Q side's 32768-wave scale -- a >20x regression versus the ~20 us
# the Q side's slice of today's fold costs -- so it is not used here (see
# round 8 report for the raw numbers).  Instead, a second tiny FlyDSL kernel
# (`_make_fold_kernel`), launched right after the main backward kernel on the
# same stream via the same `_launch` fast path, fold-reduces the large
# (R, D) partials down to (R/_FOLD_ROWS_PER_WAVE, D) first.  No atomics: each
# wave still owns a unique, disjoint row range and writes a unique output
# row, the same safety property the main kernel's own partial store already
# has.  The wrapper's `.sum(dim=0)` then runs on a tensor ~128x smaller.
_FOLD_ROWS_PER_WAVE = 16

# Megatron passes the same full-sequence rotary table to every transformer
# layer.  Materialize cos/sin once and reuse it across all fused launches rather
# than evaluating transcendental functions independently for every Q/K head.
# Keep one entry per device and stream. Same-stream ordering makes reuse safe;
# another stream builds its own table rather than racing an asynchronous fill.
_ROTARY_TABLE_CACHE = {}


def _cached_cos_sin(freqs, stream):
    """`stream` is the caller's already-obtained `torch.cuda.current_stream()`
    (Q6: avoids a second, redundant current-stream query -- see module
    docstring)."""
    device = freqs.device.index
    cuda_stream = stream.cuda_stream
    version = freqs._version
    key = (device, cuda_stream)
    cached = _ROTARY_TABLE_CACHE.get(key)
    if cached is not None:
        ref, cached_version, cosine, sine = cached
        if ref() is freqs and cached_version == version:
            return cosine, sine
    cosine = freqs.cos()
    sine = freqs.sin()
    _ROTARY_TABLE_CACHE[key] = (weakref.ref(freqs), version, cosine, sine)
    return cosine, sine


# P0b: the forward's QRSTD/KRSTD outputs no longer carry per-row rstd -- they
# carry one cached FP32 scalar holding eps, so the backward can recompute
# rstd itself.  Keyed on (device, eps): both are configuration, not a tensor
# id() (Rule 11) -- eps comes from task.yaml/the call site, never from an
# activation, so this cache hits on every call in both the benchmark loop and
# real training.
_EPS_TENSOR_CACHE = {}


def _eps_tensor(device, eps):
    key = (device.type, device.index, eps)
    t = _EPS_TENSOR_CACHE.get(key)
    if t is None:
        t = torch.full((1,), eps, device=device, dtype=torch.float32)
        _EPS_TENSOR_CACHE[key] = t
    return t


def _wave_sum_f32(value):
    """Butterfly sum across one gfx950 wave64."""
    value = fx.arith.ArithValue(value)
    for distance in (1, 2, 4, 8, 16, 32):
        value = value.addf(fx.arith.ArithValue(value.shuffle_xor(distance, _WARP)))
    return value


def _row_sum_f32(value):
    """Sum across the _LANES_PER_ROW lanes that share one head row."""
    value = fx.arith.ArithValue(value)
    distance = 1
    while distance < _LANES_PER_ROW:
        value = value.addf(fx.arith.ArithValue(value.shuffle_xor(distance, _WARP)))
        distance *= 2
    return value


def _f32(v):
    """Widen a BF16 scalar (as returned by ``Vec(...)[t]``) to FP32.

    CODE-REVIEW fix: this used to hand-build the extend op via
    ``arith.ExtFOp(fx.Float32.ir_type, _raw(v)).result``, duplicating what
    ``Numeric.to()`` already does (``Float.__init__`` -> ``fp_to_fp`` ->
    the identical ``arith.extf``) and what this same file already uses
    elsewhere for narrow/widen round-trips (e.g. ``.to(fx.BFloat16).to(fx.Float32)``).
    Routing through the public ``.to()`` idiom emits the exact same IR, so
    this is a conventions-only fix -- no behavior change.
    """
    return v.to(fx.Float32)


def _load_f32x8(rsrc, offset):
    """8 contiguous f32.  A v8f32 buffer_load is not selectable (dwordx4 max),
    so issue two dwordx4 and flatten."""
    a = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=fx.Float32)
    b = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=fx.Float32, soffset_bytes=16)
    return [fx.Float32(Vec(a)[t]) for t in range_constexpr(4)] + [
        fx.Float32(Vec(b)[t]) for t in range_constexpr(4)
    ]


def _store_f32_chunks(rsrc, base, vals):
    """Store a python list of f32 as the widest legal buffer_store chunks."""
    i = 0
    n = len(vals)
    while i < n:
        w = 4 if n - i >= 4 else (2 if n - i >= 2 else 1)
        if w == 1:
            buffer_ops.buffer_store(vals[i], rsrc, base + fx.Int32(i))
        else:
            buffer_ops.buffer_store(
                _raw(Vec.from_elements(vals[i : i + w], fx.Float32)), rsrc, base + fx.Int32(i)
            )
        i += w


def _make_fwd_kernel(S: int, B: int, NG: int, NPG: int, eps: float, cycles: int):
    packed_heads = NG * (NPG + 2)
    q_heads = NG * NPG
    q_groups_per_token = q_heads // _ROWS_PER_WAVE  # 4
    q_slots = B * q_groups_per_token  # 16
    k_slots = (B * NG) // _ROWS_PER_WAVE  # 2
    slots = q_slots + k_slots  # 18

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(
        PACKED: fx.Tensor,
        QG: fx.Tensor,
        KG: fx.Tensor,
        COSINE: fx.Tensor,
        SINE: fx.Tensor,
        QOUT: fx.Tensor,
        KOUT: fx.Tensor,
        QRSTD: fx.Tensor,
        KRSTD: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        block_x, _, _ = fx.block_idx
        lane = tid % fx.Int32(_WARP)
        wave = tid // fx.Int32(_WARP)
        global_wave = block_x * fx.Int32(_WAVES) + wave
        slot = global_wave % fx.Int32(slots)
        cycle = global_wave // fx.Int32(slots)

        row = lane // fx.Int32(_LANES_PER_ROW)
        chunk = (lane % fx.Int32(_LANES_PER_ROW)) * fx.Int32(_EPL)

        is_q = slot < fx.Int32(q_slots)

        # Q slot -> (token-in-seq, which block of _ROWS_PER_WAVE query heads)
        qb = slot // fx.Int32(q_groups_per_token)
        qsub = slot % fx.Int32(q_groups_per_token)
        qh = qsub * fx.Int32(_ROWS_PER_WAVE) + row
        q_pack = (qh // fx.Int32(NPG) * fx.Int32(NPG + 2) + qh % fx.Int32(NPG)) * fx.Int32(_D) + chunk
        q_out = qh * fx.Int32(_D) + chunk

        # K slot -> _ROWS_PER_WAVE key rows spanning several tokens of the seq
        krow = (slot - fx.Int32(q_slots)) * fx.Int32(_ROWS_PER_WAVE) + row
        kb = krow // fx.Int32(NG)
        kgrp = krow % fx.Int32(NG)
        k_pack = (kgrp * fx.Int32(NPG + 2) + fx.Int32(NPG)) * fx.Int32(_D) + chunk
        k_out = kgrp * fx.Int32(_D) + chunk

        btok = fx.arith.select(is_q, qb, kb)
        pack_in_token = fx.arith.select(is_q, q_pack, k_pack)
        out_in_token = fx.arith.select(is_q, q_out, k_out)
        out_heads = fx.arith.select(is_q, fx.Int32(q_heads), fx.Int32(NG))

        # P0b: rstd is deliberately not stored -- QRSTD/KRSTD now only carry
        # the cached eps scalar the backward reads; see module docstring.
        # Forward never reads or writes QRSTD/KRSTD, so (CODE-REVIEW fix) no
        # buffer resource is created for them here -- they stay declared
        # kernel parameters only so the launch call signature is unchanged.
        packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
        qg_rsrc = buffer_ops.create_buffer_resource(QG, max_size=True)
        kg_rsrc = buffer_ops.create_buffer_resource(KG, max_size=True)
        cosine_rsrc = buffer_ops.create_buffer_resource(COSINE, max_size=True)
        sine_rsrc = buffer_ops.create_buffer_resource(SINE, max_size=True)
        qout_rsrc = buffer_ops.create_buffer_resource(QOUT, max_size=True)
        kout_rsrc = buffer_ops.create_buffer_resource(KOUT, max_size=True)

        # gamma is loop invariant: 2 x dwordx4 per gamma tensor, selected once.
        qg_lo = buffer_ops.buffer_load(qg_rsrc, chunk, vec_width=_EPL, dtype=fx.BFloat16)
        qg_hi = buffer_ops.buffer_load(qg_rsrc, chunk + fx.Int32(_HALF), vec_width=_EPL, dtype=fx.BFloat16)
        kg_lo = buffer_ops.buffer_load(kg_rsrc, chunk, vec_width=_EPL, dtype=fx.BFloat16)
        kg_hi = buffer_ops.buffer_load(kg_rsrc, chunk + fx.Int32(_HALF), vec_width=_EPL, dtype=fx.BFloat16)
        g_lo = [
            fx.arith.select(is_q, _f32(Vec(qg_lo)[t]), _f32(Vec(kg_lo)[t])) for t in range_constexpr(_EPL)
        ]
        g_hi = [
            fx.arith.select(is_q, _f32(Vec(qg_hi)[t]), _f32(Vec(kg_hi)[t])) for t in range_constexpr(_EPL)
        ]

        seq = cycle
        while seq < fx.Int32(S):
            rot = seq * fx.Int32(_D) + chunk
            cos_lo = _load_f32x8(cosine_rsrc, rot)
            cos_hi = cos_lo
            sin_lo = _load_f32x8(sine_rsrc, rot)
            sin_hi = sin_lo

            token = seq * fx.Int32(B) + btok
            src = token * fx.Int32(packed_heads * _D) + pack_in_token
            x_lo = buffer_ops.buffer_load(packed_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16)
            x_hi = buffer_ops.buffer_load(
                packed_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16, soffset_bytes=_HALF * 2
            )

            xl = [_f32(Vec(x_lo)[t]) for t in range_constexpr(_EPL)]
            xh = [_f32(Vec(x_hi)[t]) for t in range_constexpr(_EPL)]

            acc = fx.Float32(0.0)
            for t in range_constexpr(_EPL):
                acc = acc + xl[t] * xl[t]
                acc = acc + xh[t] * xh[t]
            sumsq = _row_sum_f32(acc)
            mean = fx.Float32(sumsq) / fx.Float32(float(_D))
            rstd = fx.Float32(fmath.rsqrt(mean + fx.Float32(eps), fastmath="afn"))

            nl = [(xl[t] * rstd * g_lo[t]).to(fx.BFloat16).to(fx.Float32) for t in range_constexpr(_EPL)]
            nh = [(xh[t] * rstd * g_hi[t]).to(fx.BFloat16).to(fx.Float32) for t in range_constexpr(_EPL)]

            out_lo = []
            out_hi = []
            for t in range_constexpr(_EPL):
                out_lo.append((nl[t] * cos_lo[t] - nh[t] * sin_lo[t]).to(fx.BFloat16))
                out_hi.append((nh[t] * cos_hi[t] + nl[t] * sin_hi[t]).to(fx.BFloat16))
            v_lo = Vec.from_elements(out_lo, fx.BFloat16)
            v_hi = Vec.from_elements(out_hi, fx.BFloat16)

            dst = token * out_heads * fx.Int32(_D) + out_in_token
            if is_q:
                buffer_ops.buffer_store(_raw(v_lo), qout_rsrc, dst)
                buffer_ops.buffer_store(_raw(v_hi), qout_rsrc, dst, soffset_bytes=_HALF * 2)
            else:
                buffer_ops.buffer_store(_raw(v_lo), kout_rsrc, dst)
                buffer_ops.buffer_store(_raw(v_hi), kout_rsrc, dst, soffset_bytes=_HALF * 2)
            # P0b: rstd is deliberately not stored -- QRSTD/KRSTD now only
            # carry the cached eps scalar the backward reads; see module
            # docstring.

            seq = seq + fx.Int32(cycles)

    return kernel, slots


def _rows_sum_f32(value):
    """Sum across the _ROWS_PER_WAVE rows a wave carries (lane bits 2..5)."""
    value = fx.arith.ArithValue(value)
    distance = _LANES_PER_ROW
    while distance < _WARP:
        value = value.addf(fx.arith.ArithValue(value.shuffle_xor(distance, _WARP)))
        distance *= 2
    return value


def _bwd_slot_counts(B: int, NG: int, NPG: int):
    """Derive the P0c slot layout: one wave owns _ROWS_PER_WAVE *contiguous*
    head rows of a single sequence position instead of _ROWS_PER_WAVE
    token-strided rows of a single head slot."""
    q_heads = NG * NPG
    q_blocks_per_token = q_heads // _ROWS_PER_WAVE  # 4
    q_slots = B * q_blocks_per_token  # 16
    kv_slots = (B * NG) // _ROWS_PER_WAVE  # 2
    return q_blocks_per_token, q_slots, kv_slots, q_slots + 2 * kv_slots


# gfx950 dispatches workgroup p to XCD (p % 8), each XCD having a private
# 4 MB L2.  A full inversion of that round-robin measured WORSE (forward
# +8 us, campaign ANALYZE round 1) because it destroys the natural
# contiguous device-wide walk.  This keeps the walk but gives each XCD a
# run of `chunk` consecutive logical workgroup ids -- just long enough to
# hold one cycle's slots (20 slots = 5 workgroups) -- so the cos/sin rows a
# cycle shares land in one L2 slice instead of eight.
_XCD = 8


def _xcd_chunk(block_x, nwg, chunk):
    if chunk <= 0:
        return block_x
    sb = _XCD * chunk
    if nwg % sb:
        return block_x
    base = (block_x // fx.Int32(sb)) * fx.Int32(sb)
    x = block_x % fx.Int32(_XCD)
    j = (block_x // fx.Int32(_XCD)) % fx.Int32(chunk)
    return base + x * fx.Int32(chunk) + j


def _accumulate_rows(rsrc, row0, rows_per_wave: int, lane):
    """Sum `rows_per_wave` contiguous rows of a (*, _D) f32 buffer into one
    scalar per lane, where `lane` (0.._D) owns a fixed column.  Used by the
    Q2 dgamma fold-reduce kernel: no cross-lane or cross-wave communication
    is needed since each lane already owns a distinct column and each wave
    owns a distinct, disjoint row range.  Fully unrolled with `rows_per_wave`
    independent accumulators (no loop-carried fadd chain) so the hardware
    can keep every row's load in flight at once instead of serializing on
    one dependent reduction (measured necessary: a single serial accumulator
    chain over the full rows_per_wave cost far more than it saved -- round 8
    report).  Keep rows_per_wave small (see _FOLD_ROWS_PER_WAVE) so this
    stays cheap in both registers and code size."""
    accs = [fx.Float32(0.0) for _ in range_constexpr(rows_per_wave)]
    for r in range_constexpr(rows_per_wave):
        off = (row0 + fx.Int32(r)) * fx.Int32(_D) + lane
        v = fx.Float32(buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=fx.Float32))
        accs[r] = accs[r] + v
    acc = accs[0]
    for r in range_constexpr(1, rows_per_wave):
        acc = acc + accs[r]
    return acc


def _fold_wave_counts(B: int, NG: int, NPG: int):
    """Row counts for the Q2 dgamma fold-reduce kernel at the current P0c
    slot layout / _BWD_GRID_CYCLES (both unchanged by this round -- only the
    *post*-processing of DQG_PART/DKG_PART changes)."""
    _, q_slots, kv_slots, _ = _bwd_slot_counts(B, NG, NPG)
    r_q = _BWD_GRID_CYCLES * q_slots
    r_k = _BWD_GRID_CYCLES * kv_slots
    fold_waves_q = r_q // _FOLD_ROWS_PER_WAVE
    fold_waves_k = r_k // _FOLD_ROWS_PER_WAVE
    assert fold_waves_q * _FOLD_ROWS_PER_WAVE == r_q
    assert fold_waves_k * _FOLD_ROWS_PER_WAVE == r_k
    return fold_waves_q, fold_waves_k


def _make_fold_kernel(fold_waves_q: int, fold_waves_k: int, rows_per_wave: int):
    """Q2: collapse DQG_PART (fold_waves_q*rows_per_wave, D) and DKG_PART
    (fold_waves_k*rows_per_wave, D) down to (fold_waves_q, D) / (fold_waves_k,
    D) f32 tensors in one combined launch (is_q-style wave dispatch, same
    convention _make_bwd_kernel uses to combine Q/K/V in one grid).  No
    atomics: every wave still writes a unique output row."""
    total_waves = fold_waves_q + fold_waves_k
    assert total_waves % _WAVES == 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(DQG_PART: fx.Tensor, DKG_PART: fx.Tensor, DQG_SMALL: fx.Tensor, DKG_SMALL: fx.Tensor):
        tid = fx.thread_idx.x
        block_x, _, _ = fx.block_idx
        lane = tid % fx.Int32(_WARP)
        wave = tid // fx.Int32(_WARP)
        global_wave = block_x * fx.Int32(_WAVES) + wave
        is_q = global_wave < fx.Int32(fold_waves_q)

        if is_q:
            rsrc = buffer_ops.create_buffer_resource(DQG_PART, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(DQG_SMALL, max_size=True)
            row0 = global_wave * fx.Int32(rows_per_wave)
            acc = _accumulate_rows(rsrc, row0, rows_per_wave, lane)
            buffer_ops.buffer_store(acc, out_rsrc, global_wave * fx.Int32(_D) + lane)
        else:
            k_wave = global_wave - fx.Int32(fold_waves_q)
            rsrc = buffer_ops.create_buffer_resource(DKG_PART, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(DKG_SMALL, max_size=True)
            row0 = k_wave * fx.Int32(rows_per_wave)
            acc = _accumulate_rows(rsrc, row0, rows_per_wave, lane)
            buffer_ops.buffer_store(acc, out_rsrc, k_wave * fx.Int32(_D) + lane)

    return kernel, total_waves


def _make_bwd_kernel(S: int, B: int, NG: int, NPG: int, cycles: int):
    packed_heads = NG * (NPG + 2)
    q_heads = NG * NPG
    # P0c: a wave owns _ROWS_PER_WAVE rows that are CONTIGUOUS head rows of
    # one sequence position instead of 16 token-strided rows of one head
    # slot.  Q: 16 consecutive query heads of one (seq, batch) -> two 1 KB
    # runs 1280 B apart instead of 16 cache lines spread over a 160 KB span.
    # K/V: 8 group rows x 2 batches of one seq position.  This is the
    # forward's own mapping, measured faster on an identical grid /
    # instruction / byte-count controlled copy (campaign ANALYZE round 3).
    q_blocks_per_token, q_slots, kv_slots, slots = _bwd_slot_counts(B, NG, NPG)
    nwg = cycles * slots // _WAVES
    xcd_chunk = slots // _WAVES  # one cycle = 5 workgroups

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        PACKED: fx.Tensor,
        QG: fx.Tensor,
        KG: fx.Tensor,
        COSINE: fx.Tensor,
        SINE: fx.Tensor,
        QRSTD: fx.Tensor,
        KRSTD: fx.Tensor,
        DPACKED: fx.Tensor,
        DQG_PART: fx.Tensor,
        DKG_PART: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        block_x, _, _ = fx.block_idx
        block_x = _xcd_chunk(block_x, nwg, xcd_chunk)
        lane = tid % fx.Int32(_WARP)
        wave = tid // fx.Int32(_WARP)
        global_wave = block_x * fx.Int32(_WAVES) + wave
        slot = global_wave % fx.Int32(slots)
        cycle = global_wave // fx.Int32(slots)
        # Both predicates stay wave-uniform: `is_k` is only consulted in the
        # else-region of `is_q`, where slot >= q_slots already holds.
        is_q = slot < fx.Int32(q_slots)
        is_k = slot < fx.Int32(q_slots + kv_slots)

        row = lane // fx.Int32(_LANES_PER_ROW)
        chunk = (lane % fx.Int32(_LANES_PER_ROW)) * fx.Int32(_EPL)

        # Q slot -> (batch, which block of _ROWS_PER_WAVE consecutive q heads)
        qb = slot // fx.Int32(q_blocks_per_token)
        qsub = slot % fx.Int32(q_blocks_per_token)
        qh = qsub * fx.Int32(_ROWS_PER_WAVE) + row
        q_pack_in_tok = (qh // fx.Int32(NPG) * fx.Int32(NPG + 2) + qh % fx.Int32(NPG)) * fx.Int32(_D) + chunk
        q_src_in_tok = qh * fx.Int32(_D) + chunk

        # K slot -> _ROWS_PER_WAVE (batch, group) key rows of one seq position
        krow = (slot - fx.Int32(q_slots)) * fx.Int32(_ROWS_PER_WAVE) + row
        kbat = krow // fx.Int32(NG)
        kgrp = krow % fx.Int32(NG)
        k_pack_in_tok = (kgrp * fx.Int32(NPG + 2) + fx.Int32(NPG)) * fx.Int32(_D) + chunk
        kv_src_in_tok = kgrp * fx.Int32(_D) + chunk

        # V slot -> the same decomposition against the packed V column
        vrow = (slot - fx.Int32(q_slots + kv_slots)) * fx.Int32(_ROWS_PER_WAVE) + row
        vbat = vrow // fx.Int32(NG)
        vgrp = vrow % fx.Int32(NG)
        v_pack_in_tok = (vgrp * fx.Int32(NPG + 2) + fx.Int32(NPG + 1)) * fx.Int32(_D) + chunk
        v_src_in_tok = vgrp * fx.Int32(_D) + chunk

        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=True)
        dk_rsrc = buffer_ops.create_buffer_resource(DK, max_size=True)
        dv_rsrc = buffer_ops.create_buffer_resource(DV, max_size=True)
        packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
        qg_rsrc = buffer_ops.create_buffer_resource(QG, max_size=True)
        kg_rsrc = buffer_ops.create_buffer_resource(KG, max_size=True)
        cosine_rsrc = buffer_ops.create_buffer_resource(COSINE, max_size=True)
        sine_rsrc = buffer_ops.create_buffer_resource(SINE, max_size=True)
        # P0b: KRSTD aliases the same cached eps scalar as QRSTD (the
        # autograd wrapper passes k_rstd == q_rstd through unexamined -- see
        # module docstring), so only QRSTD needs a buffer resource / load
        # below; (CODE-REVIEW fix) building one for KRSTD too was dead.
        qrstd_rsrc = buffer_ops.create_buffer_resource(QRSTD, max_size=True)
        dpacked_rsrc = buffer_ops.create_buffer_resource(DPACKED, max_size=True)
        dqg_part_rsrc = buffer_ops.create_buffer_resource(DQG_PART, max_size=True)
        dkg_part_rsrc = buffer_ops.create_buffer_resource(DKG_PART, max_size=True)

        qg_lo_v = buffer_ops.buffer_load(qg_rsrc, chunk, vec_width=_EPL, dtype=fx.BFloat16)
        qg_hi_v = buffer_ops.buffer_load(qg_rsrc, chunk + fx.Int32(_HALF), vec_width=_EPL, dtype=fx.BFloat16)
        kg_lo_v = buffer_ops.buffer_load(kg_rsrc, chunk, vec_width=_EPL, dtype=fx.BFloat16)
        kg_hi_v = buffer_ops.buffer_load(kg_rsrc, chunk + fx.Int32(_HALF), vec_width=_EPL, dtype=fx.BFloat16)

        # P0b: QRSTD no longer holds per-row rstd -- it holds one cached FP32
        # eps scalar (see module docstring / `_eps_tensor`).  Every wave reads
        # the same value, so load it once here rather than per token below.
        eps_v = fx.Float32(buffer_ops.buffer_load(qrstd_rsrc, fx.Int32(0), vec_width=1, dtype=fx.T.f32()))

        # ------------------------------------------------------------------
        # `slot` is wave-uniform, so the Q / K / V dispatch is hoisted out of
        # the block loop: each wave runs one branch-free loop for its whole
        # life instead of re-evaluating the dispatch every row.
        # ------------------------------------------------------------------
        if is_q:
            gam_lo = [fx.Float32(Vec(qg_lo_v)[t]) for t in range_constexpr(_EPL)]
            gam_hi = [fx.Float32(Vec(qg_hi_v)[t]) for t in range_constexpr(_EPL)]
            dg_lo = Vec.from_elements([fx.Float32(0.0) for _ in range_constexpr(_EPL)], fx.Float32)
            dg_hi = Vec.from_elements([fx.Float32(0.0) for _ in range_constexpr(_EPL)], fx.Float32)
            seq = cycle
            while seq < fx.Int32(S):
                # `seq` is wave-uniform under this mapping, so all 64 lanes
                # share one cos/sin row (one 128 B line) instead of the four
                # distinct rows the token-strided map needed.
                rot = seq * fx.Int32(_D) + chunk
                cos_lo = _load_f32x8(cosine_rsrc, rot)
                cos_hi = cos_lo
                sin_lo = _load_f32x8(sine_rsrc, rot)
                sin_hi = sin_lo

                token = seq * fx.Int32(B) + qb
                dst = token * fx.Int32(packed_heads * _D) + q_pack_in_tok
                src = token * fx.Int32(q_heads * _D) + q_src_in_tok
                gl_v = buffer_ops.buffer_load(dq_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16)
                gh_v = buffer_ops.buffer_load(
                    dq_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16, soffset_bytes=_HALF * 2
                )
                xl_v = buffer_ops.buffer_load(packed_rsrc, dst, vec_width=_EPL, dtype=fx.BFloat16)
                xh_v = buffer_ops.buffer_load(
                    packed_rsrc, dst, vec_width=_EPL, dtype=fx.BFloat16, soffset_bytes=_HALF * 2
                )
                # P0b: recompute rstd from the packed row instead of loading
                # it.  Identical per-row reduction order to the forward
                # (`_row_sum_f32` over the same lo/hi pairs) and the same
                # fastmath="afn" rsqrt, so the result is bit-identical.
                sq = fx.Float32(0.0)
                for t in range_constexpr(_EPL):
                    _xl = _f32(Vec(xl_v)[t])
                    _xh = _f32(Vec(xh_v)[t])
                    sq = sq + _xl * _xl
                    sq = sq + _xh * _xh
                rstd = fx.Float32(
                    fmath.rsqrt(
                        fx.Float32(_row_sum_f32(sq)) / fx.Float32(float(_D)) + eps_v,
                        fastmath="afn",
                    )
                )

                dn_lo = []
                dn_hi = []
                u_lo = []
                u_hi = []
                du_lo = []
                du_hi = []
                dot = fx.Float32(0.0)
                for t in range_constexpr(_EPL):
                    g_l = _f32(Vec(gl_v)[t])
                    g_h = _f32(Vec(gh_v)[t])
                    d_l = (g_l * cos_lo[t] + g_h * sin_lo[t]).to(fx.BFloat16).to(fx.Float32)
                    d_h = (g_h * cos_hi[t] - g_l * sin_hi[t]).to(fx.BFloat16).to(fx.Float32)
                    ul = _f32(Vec(xl_v)[t]) * rstd
                    uh = _f32(Vec(xh_v)[t]) * rstd
                    dul = d_l * gam_lo[t]
                    duh = d_h * gam_hi[t]
                    dot = dot + ul * dul
                    dot = dot + uh * duh
                    dn_lo.append(d_l)
                    dn_hi.append(d_h)
                    u_lo.append(ul)
                    u_hi.append(uh)
                    du_lo.append(dul)
                    du_hi.append(duh)

                scale = fx.Float32(_row_sum_f32(dot)) / fx.Float32(float(_D))
                dx_lo = []
                dx_hi = []
                for t in range_constexpr(_EPL):
                    dx_lo.append(((du_lo[t] - u_lo[t] * scale) * rstd).to(fx.BFloat16))
                    dx_hi.append(((du_hi[t] - u_hi[t] * scale) * rstd).to(fx.BFloat16))
                buffer_ops.buffer_store(_raw(Vec.from_elements(dx_lo, fx.BFloat16)), dpacked_rsrc, dst)
                buffer_ops.buffer_store(
                    _raw(Vec.from_elements(dx_hi, fx.BFloat16)), dpacked_rsrc, dst, soffset_bytes=_HALF * 2
                )

                dg_lo = Vec(
                    arith.AddFOp(
                        _raw(dg_lo),
                        _raw(
                            Vec.from_elements([dn_lo[t] * u_lo[t] for t in range_constexpr(_EPL)], fx.Float32)
                        ),
                    ).result
                )
                dg_hi = Vec(
                    arith.AddFOp(
                        _raw(dg_hi),
                        _raw(
                            Vec.from_elements([dn_hi[t] * u_hi[t] for t in range_constexpr(_EPL)], fx.Float32)
                        ),
                    ).result
                )
                seq = seq + fx.Int32(cycles)

            # Fold the _ROWS_PER_WAVE per-row accumulators once per wave. All
            # 16 rows are Q rows sharing the one gamma vector, so the fold is
            # the same sum it was under the token-strided map; only the
            # partial-row tensor's row count changes (cycles * q_slots here,
            # versus the old cycles * q_heads), and the un-editable wrapper's
            # `.sum(dim=0)` is agnostic to that count.
            part = (cycle * fx.Int32(q_slots) + slot) * fx.Int32(_D) + chunk
            red_lo = [fx.Float32(_rows_sum_f32(fx.Float32(dg_lo[t]))) for t in range_constexpr(_EPL)]
            red_hi = [fx.Float32(_rows_sum_f32(fx.Float32(dg_hi[t]))) for t in range_constexpr(_EPL)]
            _store_f32_chunks(dqg_part_rsrc, part, red_lo)
            _store_f32_chunks(dqg_part_rsrc, part + fx.Int32(_HALF), red_hi)
        elif is_k:
            gam_lo = [fx.Float32(Vec(kg_lo_v)[t]) for t in range_constexpr(_EPL)]
            gam_hi = [fx.Float32(Vec(kg_hi_v)[t]) for t in range_constexpr(_EPL)]
            dg_lo = Vec.from_elements([fx.Float32(0.0) for _ in range_constexpr(_EPL)], fx.Float32)
            dg_hi = Vec.from_elements([fx.Float32(0.0) for _ in range_constexpr(_EPL)], fx.Float32)
            seq = cycle
            while seq < fx.Int32(S):
                rot = seq * fx.Int32(_D) + chunk
                cos_lo = _load_f32x8(cosine_rsrc, rot)
                cos_hi = cos_lo
                sin_lo = _load_f32x8(sine_rsrc, rot)
                sin_hi = sin_lo

                token = seq * fx.Int32(B) + kbat
                dst = token * fx.Int32(packed_heads * _D) + k_pack_in_tok
                src = token * fx.Int32(NG * _D) + kv_src_in_tok
                gl_v = buffer_ops.buffer_load(dk_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16)
                gh_v = buffer_ops.buffer_load(
                    dk_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16, soffset_bytes=_HALF * 2
                )
                xl_v = buffer_ops.buffer_load(packed_rsrc, dst, vec_width=_EPL, dtype=fx.BFloat16)
                xh_v = buffer_ops.buffer_load(
                    packed_rsrc, dst, vec_width=_EPL, dtype=fx.BFloat16, soffset_bytes=_HALF * 2
                )
                # P0b: recompute rstd from the packed row instead of loading
                # it (same mechanism as the is_q branch above).
                sq = fx.Float32(0.0)
                for t in range_constexpr(_EPL):
                    _xl = _f32(Vec(xl_v)[t])
                    _xh = _f32(Vec(xh_v)[t])
                    sq = sq + _xl * _xl
                    sq = sq + _xh * _xh
                rstd = fx.Float32(
                    fmath.rsqrt(
                        fx.Float32(_row_sum_f32(sq)) / fx.Float32(float(_D)) + eps_v,
                        fastmath="afn",
                    )
                )

                dn_lo = []
                dn_hi = []
                u_lo = []
                u_hi = []
                du_lo = []
                du_hi = []
                dot = fx.Float32(0.0)
                for t in range_constexpr(_EPL):
                    g_l = _f32(Vec(gl_v)[t])
                    g_h = _f32(Vec(gh_v)[t])
                    d_l = (g_l * cos_lo[t] + g_h * sin_lo[t]).to(fx.BFloat16).to(fx.Float32)
                    d_h = (g_h * cos_hi[t] - g_l * sin_hi[t]).to(fx.BFloat16).to(fx.Float32)
                    ul = _f32(Vec(xl_v)[t]) * rstd
                    uh = _f32(Vec(xh_v)[t]) * rstd
                    dul = d_l * gam_lo[t]
                    duh = d_h * gam_hi[t]
                    dot = dot + ul * dul
                    dot = dot + uh * duh
                    dn_lo.append(d_l)
                    dn_hi.append(d_h)
                    u_lo.append(ul)
                    u_hi.append(uh)
                    du_lo.append(dul)
                    du_hi.append(duh)

                scale = fx.Float32(_row_sum_f32(dot)) / fx.Float32(float(_D))
                dx_lo = []
                dx_hi = []
                for t in range_constexpr(_EPL):
                    dx_lo.append(((du_lo[t] - u_lo[t] * scale) * rstd).to(fx.BFloat16))
                    dx_hi.append(((du_hi[t] - u_hi[t] * scale) * rstd).to(fx.BFloat16))
                buffer_ops.buffer_store(_raw(Vec.from_elements(dx_lo, fx.BFloat16)), dpacked_rsrc, dst)
                buffer_ops.buffer_store(
                    _raw(Vec.from_elements(dx_hi, fx.BFloat16)), dpacked_rsrc, dst, soffset_bytes=_HALF * 2
                )

                dg_lo = Vec(
                    arith.AddFOp(
                        _raw(dg_lo),
                        _raw(
                            Vec.from_elements([dn_lo[t] * u_lo[t] for t in range_constexpr(_EPL)], fx.Float32)
                        ),
                    ).result
                )
                dg_hi = Vec(
                    arith.AddFOp(
                        _raw(dg_hi),
                        _raw(
                            Vec.from_elements([dn_hi[t] * u_hi[t] for t in range_constexpr(_EPL)], fx.Float32)
                        ),
                    ).result
                )
                seq = seq + fx.Int32(cycles)

            part = (cycle * fx.Int32(kv_slots) + slot - fx.Int32(q_slots)) * fx.Int32(_D) + chunk
            red_lo = [fx.Float32(_rows_sum_f32(fx.Float32(dg_lo[t]))) for t in range_constexpr(_EPL)]
            red_hi = [fx.Float32(_rows_sum_f32(fx.Float32(dg_hi[t]))) for t in range_constexpr(_EPL)]
            _store_f32_chunks(dkg_part_rsrc, part, red_lo)
            _store_f32_chunks(dkg_part_rsrc, part + fx.Int32(_HALF), red_hi)
        else:
            seq = cycle
            while seq < fx.Int32(S):
                token = seq * fx.Int32(B) + vbat
                dst = token * fx.Int32(packed_heads * _D) + v_pack_in_tok
                src = token * fx.Int32(NG * _D) + v_src_in_tok
                gl_v = buffer_ops.buffer_load(dv_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16)
                gh_v = buffer_ops.buffer_load(
                    dv_rsrc, src, vec_width=_EPL, dtype=fx.BFloat16, soffset_bytes=_HALF * 2
                )
                buffer_ops.buffer_store(_raw(gl_v), dpacked_rsrc, dst)
                buffer_ops.buffer_store(_raw(gh_v), dpacked_rsrc, dst, soffset_bytes=_HALF * 2)
                seq = seq + fx.Int32(cycles)

    return kernel


@flyc.jit
def _compiled_fwd(
    PACKED,
    QG,
    KG,
    COSINE,
    SINE,
    QOUT,
    KOUT,
    QRSTD,
    KRSTD,
    S: fx.Constexpr[int],
    B: fx.Constexpr[int],
    NG: fx.Constexpr[int],
    NPG: fx.Constexpr[int],
    EPS: fx.Constexpr[float],
    stream: fx.Stream,
):
    kernel, slots = _make_fwd_kernel(S, B, NG, NPG, EPS, _FWD_GRID_CYCLES)
    assert (_FWD_GRID_CYCLES * slots) % _WAVES == 0
    grid_x = _FWD_GRID_CYCLES * slots // _WAVES
    kernel(PACKED, QG, KG, COSINE, SINE, QOUT, KOUT, QRSTD, KRSTD).launch(
        grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


@flyc.jit
def _compiled_bwd(
    DQ,
    DK,
    DV,
    PACKED,
    QG,
    KG,
    COSINE,
    SINE,
    QRSTD,
    KRSTD,
    DPACKED,
    DQG_PART,
    DKG_PART,
    S: fx.Constexpr[int],
    B: fx.Constexpr[int],
    NG: fx.Constexpr[int],
    NPG: fx.Constexpr[int],
    stream: fx.Stream,
):
    _, _, _, slots = _bwd_slot_counts(B, NG, NPG)
    assert (_BWD_GRID_CYCLES * slots) % _WAVES == 0
    grid_x = _BWD_GRID_CYCLES * slots // _WAVES
    kernel = _make_bwd_kernel(S, B, NG, NPG, _BWD_GRID_CYCLES)
    kernel(DQ, DK, DV, PACKED, QG, KG, COSINE, SINE, QRSTD, KRSTD, DPACKED, DQG_PART, DKG_PART).launch(
        grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


@flyc.jit
def _compiled_fold(
    DQG_PART,
    DKG_PART,
    DQG_SMALL,
    DKG_SMALL,
    FOLD_WAVES_Q: fx.Constexpr[int],
    FOLD_WAVES_K: fx.Constexpr[int],
    ROWS_PER_WAVE: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, total_waves = _make_fold_kernel(FOLD_WAVES_Q, FOLD_WAVES_K, ROWS_PER_WAVE)
    assert total_waves % _WAVES == 0
    grid_x = total_waves // _WAVES
    kernel(DQG_PART, DKG_PART, DQG_SMALL, DKG_SMALL).launch(
        grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


_CALL_STATE_CACHE = {}


def _launch(compiled, key, kwargs_fn, args):
    """`kwargs_fn` is a zero-arg callable building the kwargs dict, not the
    dict itself (Q6: the hot `state(args)` path below never reads kwargs at
    all -- only the two cold paths do -- so building a ~15-key dict on every
    call just to discard it on every steady-state call was pure waste; see
    module docstring)."""
    state = _CALL_STATE_CACHE.get(key)
    if state is None:
        kwargs = kwargs_fn()
        compiled(**kwargs)
        signature = compiled._sig
        bound = signature.bind(**kwargs)
        bound.apply_defaults()
        cache_key = compiled._build_full_cache_key(bound.arguments, owner_cls=None, bound_self=None)
        state = compiled._call_state_cache.get(cache_key)
        _CALL_STATE_CACHE[key] = state if state is not None else False
        return
    if state is False:
        compiled(**kwargs_fn())
        return
    state(args)


def flydsl_qkv_rmsnorm_rope_forward(qkv, q_gamma, k_gamma, freqs, split_sizes, eps):
    """Raw forward entry point.

    General input-contract validation, including the row/lane tiling guard,
    lives in the PyTorch wrapper (``qk_rmsnorm_rope_shape_error``). Keep the
    check here as defense in depth for direct callers of the raw entry point.
    """
    S, B, NG, _ = qkv.shape
    q_size, k_size, _ = split_sizes
    npg = q_size // _D
    assert k_size == _D
    _check_row_tileable(B, NG, npg)
    q = torch.empty((S, B, NG * npg, _D), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((S, B, NG, _D), device=qkv.device, dtype=qkv.dtype)
    v = qkv[..., -_D:]
    eps_f = float(eps)
    # P0b: QRSTD/KRSTD no longer materialize per-row rstd (the backward
    # recomputes it).  They now carry a single cached eps scalar that the
    # autograd wrapper passes straight through to the backward unexamined.
    q_rstd = _eps_tensor(qkv.device, eps_f)
    k_rstd = q_rstd
    # Q6: read the stream once and thread it into _cached_cos_sin (see
    # module docstring) instead of two independent current-stream queries.
    stream = torch.cuda.current_stream(qkv.device)
    cosine, sine = _cached_cos_sin(freqs, stream)
    args = (qkv, q_gamma, k_gamma, cosine, sine, q, k, q_rstd, k_rstd, S, B, NG, npg, eps_f, stream)
    _launch(
        _compiled_fwd,
        ("fwd", S, B, NG, npg, eps_f, qkv.dtype),
        lambda: dict(
            PACKED=qkv,
            QG=q_gamma,
            KG=k_gamma,
            COSINE=cosine,
            SINE=sine,
            QOUT=q,
            KOUT=k,
            QRSTD=q_rstd,
            KRSTD=k_rstd,
            S=S,
            B=B,
            NG=NG,
            NPG=npg,
            EPS=eps_f,
            stream=stream,
        ),
        args,
    )
    return q, k, v, q_rstd, k_rstd


def flydsl_qkv_rmsnorm_rope_backward(dq, dk, dv, qkv, q_gamma, k_gamma, freqs, q_rstd, k_rstd, split_sizes):
    """Raw backward entry point returning packed dQKV and dgamma partials.

    See ``flydsl_qkv_rmsnorm_rope_forward`` for why the tiling guard is also
    retained at this raw entry point.
    """
    S, B, NG, _ = qkv.shape
    q_size, k_size, _ = split_sizes
    npg = q_size // _D
    assert k_size == _D
    _check_row_tileable(B, NG, npg)
    dqkv = torch.empty_like(qkv)
    # P0c: the dgamma partial-row count is now driven by the new slot layout
    # (q_slots / kv_slots), not q_heads / NG directly -- see _bwd_slot_counts.
    _, q_slots, kv_slots, _ = _bwd_slot_counts(B, NG, npg)
    dqg_part = torch.empty((_BWD_GRID_CYCLES * q_slots, _D), device=qkv.device, dtype=torch.float32)
    dkg_part = torch.empty((_BWD_GRID_CYCLES * kv_slots, _D), device=qkv.device, dtype=torch.float32)
    # Q6: see flydsl_qkv_rmsnorm_rope_forward -- one stream query threaded
    # into _cached_cos_sin instead of two independent ones.
    stream = torch.cuda.current_stream(qkv.device)
    cosine, sine = _cached_cos_sin(freqs, stream)
    args = (
        dq,
        dk,
        dv,
        qkv,
        q_gamma,
        k_gamma,
        cosine,
        sine,
        q_rstd,
        k_rstd,
        dqkv,
        dqg_part,
        dkg_part,
        S,
        B,
        NG,
        npg,
        stream,
    )
    _launch(
        _compiled_bwd,
        ("bwd", S, B, NG, npg, qkv.dtype),
        lambda: dict(
            DQ=dq,
            DK=dk,
            DV=dv,
            PACKED=qkv,
            QG=q_gamma,
            KG=k_gamma,
            COSINE=cosine,
            SINE=sine,
            QRSTD=q_rstd,
            KRSTD=k_rstd,
            DPACKED=dqkv,
            DQG_PART=dqg_part,
            DKG_PART=dkg_part,
            S=S,
            B=B,
            NG=NG,
            NPG=npg,
            stream=stream,
        ),
        args,
    )

    # Q2: fold-reduce the large dgamma partials down to a tiny tensor before
    # handing them to the un-editable wrapper's `.sum(dim=0).to(dtype)` --
    # see module docstring / _make_fold_kernel / _FOLD_ROWS_PER_WAVE.
    fold_waves_q, fold_waves_k = _fold_wave_counts(B, NG, npg)
    dqg_small = torch.empty((fold_waves_q, _D), device=qkv.device, dtype=torch.float32)
    dkg_small = torch.empty((fold_waves_k, _D), device=qkv.device, dtype=torch.float32)
    fold_args = (
        dqg_part,
        dkg_part,
        dqg_small,
        dkg_small,
        fold_waves_q,
        fold_waves_k,
        _FOLD_ROWS_PER_WAVE,
        stream,
    )
    _launch(
        _compiled_fold,
        ("fold", B, NG, npg),
        lambda: dict(
            DQG_PART=dqg_part,
            DKG_PART=dkg_part,
            DQG_SMALL=dqg_small,
            DKG_SMALL=dkg_small,
            FOLD_WAVES_Q=fold_waves_q,
            FOLD_WAVES_K=fold_waves_k,
            ROWS_PER_WAVE=_FOLD_ROWS_PER_WAVE,
            stream=stream,
        ),
        fold_args,
    )
    return dqkv, dqg_small, dkg_small
