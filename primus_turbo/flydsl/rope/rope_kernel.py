###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Standalone FlyDSL RoPE forward kernel: split packed QKV into three contiguous
Q/K/V tensors, applying rotate-half RoPE to Q and K (V is a pure pass-through).

Matches the fused QKV rotary-embedding I/O contract Megatron calls with;
``primus_turbo.pytorch.ops.rope`` wraps this file's forward and backward in
autograd.

Layout:
  - PACKED_QKV: [S, B, NG, (NPG+2)*D] bf16 contiguous. Per group, the last dim is
    NPG Q head-slots of D columns, then 1 K head-slot, then 1 V head-slot -- i.e.
    6 contiguous D-wide "head slots" per group when NPG=4.
  - Q_FREQS / K_FREQS: [S, 1, 1, D] f32 contiguous, duplicated so
    freq[..., j] == freq[..., j + D//2] (only the low half is ever read here).
  - Q_OUT: [S, B, NG*NPG, D] bf16 contiguous, dest head = g*NPG + local_head.
  - K_OUT / V_OUT: [S, B, NG, D] bf16 contiguous, dest head = g.
  - Row index (flattened over S,B) is identical across every tensor above:
    row = s*B + b. Position for the frequency table is s = row // B.

Rotation (non-interleaved rotate-half, with cos/sin taken in f32 and the
result rounded once on the way to bf16):
    lo, hi = x[:D//2], x[D//2:]
    out_lo = lo * cos(freq) - hi * sin(freq)
    out_hi = hi * cos(freq) + lo * sin(freq)

Kernel shape (measured on the production shape S=8192,B=4,NG=8,NPG=4,D=128):
block_threads=64 is
split into 4 "row lanes" (_NROW) x 16 "p lanes" (_NPLANE), each p lane issuing
a vec_width=4 (128-bit) buffer_load/store. This is the widest vector the f32
frequency load supports (vec_width=8 f32 would need a 256-bit buffer op, which
AMDGPU buffer intrinsics don't have), and keeps all 64 lanes of the wave busy
by having each row-lane process a different row of the same (group, head)
column slice concurrently -- measured ~30% faster than one-row-per-block with
vec_width=1 or 2 (both of which leave lanes idle or under-vectorized).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, math

from primus_turbo.flydsl.mega.tune_utils import Config, autotune

_D = 128  # head dim (fixed for this model family)
_HALF = _D // 2  # 64: rotate-half split point
_VEC = 4  # vector width per lane; capped at 4xf32 = 128 bits (the freq load)
_NPLANE = _HALF // _VEC  # 16 lanes cover the 64-wide half with vec4
_NROW = 64 // _NPLANE  # 4 rows processed concurrently per block (fills the wave)
# Row lanes a block advances together. A caller must keep S*B a multiple of this:
# the kernels carry no per-row predicate and their descriptors span the whole address space.
ROPE_ROW_GROUP = _NROW

# Issue the rotate and the K+V kernels from one Python call instead of two. Same two
# kernels and the same grids either way, so the outputs are bit-identical; what it
# saves is one host-side autotune-key and JIT-cache-key resolution per call. That is
# the whole call at shapes whose device time sits under the host floor.
_MERGED_DISPATCH = True
_BLOCK_THREADS = _NROW * _NPLANE  # 64 = one full wave

assert _HALF % _VEC == 0
assert 64 % _NPLANE == 0


def _vec_type(base_scalar_type, width):
    """Scalar type when width==1, else an MLIR vector<width x base> type."""
    return base_scalar_type if width == 1 else fx.T.VectorType.get([width], base_scalar_type)


def _make_rope_kernel(
    NG: int, B: int, total_rows: int, hk_base: int, n_hk: int, do_rotate: bool, grid_x: int, npg: int = 4
):
    """Build one specialized RoPE extract kernel.

    ``hk_base``/``n_hk`` select which of the packed row's ``npg+2`` D-wide
    head slots per group this kernel reads: hk_base=0, n_hk=NPG for Q;
    hk_base=NPG, n_hk=1 for K; hk_base=NPG+1, n_hk=1 for V. ``do_rotate`` is a
    Python bool (not a traced value): it must stay a closed-over constant
    because it changes which ops get traced, not just their operands (V has
    no freq input at all).

    ``npg`` defaults to 4 (this campaign's only production value -- Llama-3.1
    8B's GQA ratio, NG=8 KV groups x NPG=4 Q heads/group = 32 Q heads) but is
    a real parameter, not hardcoded into the row-stride math: an earlier
    version of this function hardcoded ``pack_stride = NG * 6 * _D`` (i.e.
    baked in NPG+2=6), which silently produced garbage for any NPG != 4 --
    caught only by a multi-shape test, which is why the pack width is derived
    for the production shape but is cheap insurance against a future GQA
    ratio change. Fixed by deriving the packed row width from ``npg`` itself.

    grid: (grid_x, NG*n_hk, 1). block_idx.y enumerates (group, local_head) --
    uniform per block, so the per-block column bases below are computed once,
    outside the row loop.
    """
    pack_stride = NG * (npg + 2) * _D
    out_row_stride = NG * n_hk * _D

    if do_rotate:

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def rope_kernel(PACKED: fx.Tensor, FREQS: fx.Tensor, OUT: fx.Tensor):
            f32ty = _vec_type(fx.T.f32(), _VEC)
            bf16ty = _vec_type(fx.T.bf16(), _VEC)
            thread_index = fx.thread_idx.x
            block_index_x, block_index_y, _ = fx.block_idx
            row_lane = thread_index // fx.Int32(_NPLANE)
            p_lane = thread_index % fx.Int32(_NPLANE)
            p = p_lane * fx.Int32(_VEC)
            g = block_index_y // fx.Int32(n_hk)
            lh = block_index_y % fx.Int32(n_hk)
            hk = fx.Int32(hk_base) + lh

            packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
            freq_rsrc = buffer_ops.create_buffer_resource(FREQS, max_size=True)

            packed_col_base = (g * fx.Int32(npg + 2) + hk) * fx.Int32(_D)
            dest_head = g * fx.Int32(n_hk) + lh
            dest_col_base = dest_head * fx.Int32(_D)

            def compute_rows(row0):
                row = row0 + row_lane
                in_base = row * fx.Int32(pack_stride) + packed_col_base + p
                out_base = row * fx.Int32(out_row_stride) + dest_col_base + p
                in_lo = buffer_ops.buffer_load(packed_rsrc, in_base, vec_width=_VEC, dtype=fx.T.bf16())
                in_hi = buffer_ops.buffer_load(
                    packed_rsrc, in_base + fx.Int32(_HALF), vec_width=_VEC, dtype=fx.T.bf16()
                )
                # Position for the frequency table: s = row // B (B is MBS; row = s*B+b).
                s = row // fx.Int32(B)
                freq_off = s * fx.Int32(_D) + p
                freq = buffer_ops.buffer_load(freq_rsrc, freq_off, vec_width=_VEC, dtype=fx.T.f32())
                cos_p = math.cos(freq)
                sin_p = math.sin(freq)
                lo_f = fx.arith.extf(f32ty, in_lo)
                hi_f = fx.arith.extf(f32ty, in_hi)
                out_lo_f = fx.arith.subf(fx.arith.mulf(lo_f, cos_p), fx.arith.mulf(hi_f, sin_p))
                out_hi_f = fx.arith.addf(fx.arith.mulf(hi_f, cos_p), fx.arith.mulf(lo_f, sin_p))
                out_lo = fx.arith.trunc_f(bf16ty, out_lo_f)
                out_hi = fx.arith.trunc_f(bf16ty, out_hi_f)
                buffer_ops.buffer_store(out_lo, out_rsrc, out_base)
                buffer_ops.buffer_store(out_hi, out_rsrc, out_base + fx.Int32(_HALF))

            row0 = block_index_x * fx.Int32(_NROW)
            while row0 < fx.Int32(total_rows):
                compute_rows(row0)
                row0 = row0 + fx.Int32(grid_x * _NROW)

    else:

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def rope_kernel(PACKED: fx.Tensor, OUT: fx.Tensor):
            thread_index = fx.thread_idx.x
            block_index_x, block_index_y, _ = fx.block_idx
            row_lane = thread_index // fx.Int32(_NPLANE)
            p_lane = thread_index % fx.Int32(_NPLANE)
            p = p_lane * fx.Int32(_VEC)
            g = block_index_y // fx.Int32(n_hk)
            lh = block_index_y % fx.Int32(n_hk)
            hk = fx.Int32(hk_base) + lh

            packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)

            packed_col_base = (g * fx.Int32(npg + 2) + hk) * fx.Int32(_D)
            dest_head = g * fx.Int32(n_hk) + lh
            dest_col_base = dest_head * fx.Int32(_D)

            def compute_rows(row0):
                row = row0 + row_lane
                in_base = row * fx.Int32(pack_stride) + packed_col_base + p
                out_base = row * fx.Int32(out_row_stride) + dest_col_base + p
                in_lo = buffer_ops.buffer_load(packed_rsrc, in_base, vec_width=_VEC, dtype=fx.T.bf16())
                in_hi = buffer_ops.buffer_load(
                    packed_rsrc, in_base + fx.Int32(_HALF), vec_width=_VEC, dtype=fx.T.bf16()
                )
                buffer_ops.buffer_store(in_lo, out_rsrc, out_base)
                buffer_ops.buffer_store(in_hi, out_rsrc, out_base + fx.Int32(_HALF))

            row0 = block_index_x * fx.Int32(_NROW)
            while row0 < fx.Int32(total_rows):
                compute_rows(row0)
                row0 = row0 + fx.Int32(grid_x * _NROW)

    return rope_kernel


def _make_kv_merged_kernel(NG: int, B: int, total_rows: int, npg: int, grid_x: int):
    """Build ONE kernel that does both K (rotate) and V (pass-through-copy) in a
    single launch, picking its branch from a block-uniform runtime predicate.

    Why K+V and not Q+K: the two
    rotate kernels want very different total block counts for the same
    grid_x -- Q has grid_y=NG*NPG=32 and its optimum sits at a SMALL grid_x
    (~224; more blocks than that just adds grid-stride-loop overhead for no
    gain), while K has grid_y=NG=8 and needs a LARGE grid_x (~4096) to reach
    the same total block count. Sharing one grid_x between Q and K would force
    one of them off its optimum. V, however, is measured FLAT across the
    entire grid_x range the rotate kernels care about (flat, from
    192 to 8192) -- so V pays ~nothing for adopting
    K's grid_x, and the merge only removes a kernel launch + a redundant
    buffer-resource setup, never forces a compromise. Measured: 15.9% faster
    than two separate launches at the production shape, bit-exact vs the
    separate K/V kernels, with its own grid_x optimum re-swept and flat over
    the range this uses.

    block_idx.y in [0, NG*2): g = y//2, is_v = (y%2==1). is_v is block-uniform
    (every lane in a block sees the same y), so the ``if is_v`` below is a
    real per-block branch with no wavefront divergence cost -- it is exactly
    like dispatching two different kernels from one grid, not a per-lane
    conditional. Both tensors' buffer resources are created unconditionally
    (cheap SGPR-only setup, no memory traffic) so each branch's body is plain
    straight-line code once the branch is taken, matching the pattern already
    used by ``_make_rope_kernel``'s do_rotate specialization.
    """
    pack_stride = NG * (npg + 2) * _D
    out_row_stride = NG * _D
    hk_k = npg  # K's head-slot index within the (npg+2) D-wide slots per group.
    hk_v = npg + 1  # V's head-slot index (the last one).

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def rope_kv_kernel(PACKED: fx.Tensor, K_FREQS: fx.Tensor, K_OUT: fx.Tensor, V_OUT: fx.Tensor):
        f32ty = _vec_type(fx.T.f32(), _VEC)
        bf16ty = _vec_type(fx.T.bf16(), _VEC)
        thread_index = fx.thread_idx.x
        block_index_x, block_index_y, _ = fx.block_idx
        row_lane = thread_index // fx.Int32(_NPLANE)
        p_lane = thread_index % fx.Int32(_NPLANE)
        p = p_lane * fx.Int32(_VEC)

        g = block_index_y // fx.Int32(2)
        is_v = (block_index_y % fx.Int32(2)) == fx.Int32(1)

        packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
        k_out_rsrc = buffer_ops.create_buffer_resource(K_OUT, max_size=True)
        v_out_rsrc = buffer_ops.create_buffer_resource(V_OUT, max_size=True)
        freq_rsrc = buffer_ops.create_buffer_resource(K_FREQS, max_size=True)

        if is_v:
            packed_col_base = (g * fx.Int32(npg + 2) + fx.Int32(hk_v)) * fx.Int32(_D)
            dest_col_base = g * fx.Int32(_D)

            def compute_rows_v(row0):
                row = row0 + row_lane
                in_base = row * fx.Int32(pack_stride) + packed_col_base + p
                out_base = row * fx.Int32(out_row_stride) + dest_col_base + p
                in_lo = buffer_ops.buffer_load(packed_rsrc, in_base, vec_width=_VEC, dtype=fx.T.bf16())
                in_hi = buffer_ops.buffer_load(
                    packed_rsrc, in_base + fx.Int32(_HALF), vec_width=_VEC, dtype=fx.T.bf16()
                )
                buffer_ops.buffer_store(in_lo, v_out_rsrc, out_base)
                buffer_ops.buffer_store(in_hi, v_out_rsrc, out_base + fx.Int32(_HALF))

            row0 = block_index_x * fx.Int32(_NROW)
            while row0 < fx.Int32(total_rows):
                compute_rows_v(row0)
                row0 = row0 + fx.Int32(grid_x * _NROW)
        else:
            packed_col_base = (g * fx.Int32(npg + 2) + fx.Int32(hk_k)) * fx.Int32(_D)
            dest_col_base = g * fx.Int32(_D)

            def compute_rows_k(row0):
                row = row0 + row_lane
                in_base = row * fx.Int32(pack_stride) + packed_col_base + p
                out_base = row * fx.Int32(out_row_stride) + dest_col_base + p
                in_lo = buffer_ops.buffer_load(packed_rsrc, in_base, vec_width=_VEC, dtype=fx.T.bf16())
                in_hi = buffer_ops.buffer_load(
                    packed_rsrc, in_base + fx.Int32(_HALF), vec_width=_VEC, dtype=fx.T.bf16()
                )
                s = row // fx.Int32(B)
                freq_off = s * fx.Int32(_D) + p
                freq = buffer_ops.buffer_load(freq_rsrc, freq_off, vec_width=_VEC, dtype=fx.T.f32())
                cos_p = math.cos(freq)
                sin_p = math.sin(freq)
                lo_f = fx.arith.extf(f32ty, in_lo)
                hi_f = fx.arith.extf(f32ty, in_hi)
                out_lo_f = fx.arith.subf(fx.arith.mulf(lo_f, cos_p), fx.arith.mulf(hi_f, sin_p))
                out_hi_f = fx.arith.addf(fx.arith.mulf(hi_f, cos_p), fx.arith.mulf(lo_f, sin_p))
                out_lo = fx.arith.trunc_f(bf16ty, out_lo_f)
                out_hi = fx.arith.trunc_f(bf16ty, out_hi_f)
                buffer_ops.buffer_store(out_lo, k_out_rsrc, out_base)
                buffer_ops.buffer_store(out_hi, k_out_rsrc, out_base + fx.Int32(_HALF))

            row0 = block_index_x * fx.Int32(_NROW)
            while row0 < fx.Int32(total_rows):
                compute_rows_k(row0)
                row0 = row0 + fx.Int32(grid_x * _NROW)

    return rope_kv_kernel


# grid_x candidates. A wide sweep (64..16384) found the true optimum tracks
# grid_y = NG*n_hk, NOT
# do_rotate: Q (n_hk=NPG=4, grid_y=32) wants a SMALL grid_x (~160-256, total
# blocks ~5-8K; degrades past 1024 as each block's work shrinks below the
# grid-stride loop's fixed overhead), while K (n_hk=1, grid_y=8, but still a
# *rotate* kernel -- it shares Q's autotune candidate list, not V's) wants a
# LARGE grid_x (~2048-4096, because grid_y=8 alone gives only 8x the block
# count of grid_x, so it needs ~4x Q's grid_x for comparable total blocks).
# An earlier version of this file conflated "rotate vs copy" with "small vs
# large grid_x" and gave the rotate list only Q's narrow range: K (which is
# also do_rotate=True) then silently tuned inside a range 3-6x away from its
# optimum -- the in-range best is close enough to the true one that
# a ~20% regression that cost nothing to detect once measured directly, and
# nothing to fix by simply widening the shared candidate list. V (grid_y=8,
# do_rotate=False) is nearly flat from 192 to 8192, so one
# wide list safely covers Q+K+V; the one-time autotune search cost (cached to
# disk per shape key) is negligible against millions of training-step calls.
_ROTATE_GRID_X_CANDIDATES = (128, 160, 192, 224, 256, 320, 512, 768, 1024, 1536, 2048, 3072, 4096)
_COPY_GRID_X_CANDIDATES = (192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


@autotune(
    configs=[Config(grid_x=gx) for gx in _ROTATE_GRID_X_CANDIDATES],
    key=["NG", "B", "total_rows", "hk_base", "n_hk", "npg"],
)
@flyc.jit
def _compiled_rope_rotate(
    PACKED,
    FREQS,
    OUT,
    NG: fx.Constexpr[int],
    B: fx.Constexpr[int],
    total_rows: fx.Constexpr[int],
    hk_base: fx.Constexpr[int],
    n_hk: fx.Constexpr[int],
    npg: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    stream: fx.Stream,
):
    grid_y = NG * n_hk
    kernel = _make_rope_kernel(NG, B, total_rows, hk_base, n_hk, True, grid_x, npg)
    kernel(PACKED, FREQS, OUT).launch(grid=(grid_x, grid_y, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


@autotune(
    configs=[Config(grid_x=gx) for gx in _COPY_GRID_X_CANDIDATES],
    key=["NG", "B", "total_rows", "hk_base", "n_hk", "npg"],
)
@flyc.jit
def _compiled_rope_copy(
    PACKED,
    OUT,
    NG: fx.Constexpr[int],
    B: fx.Constexpr[int],
    total_rows: fx.Constexpr[int],
    hk_base: fx.Constexpr[int],
    n_hk: fx.Constexpr[int],
    npg: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    stream: fx.Stream,
):
    grid_y = NG * n_hk
    kernel = _make_rope_kernel(NG, B, total_rows, hk_base, n_hk, False, grid_x, npg)
    kernel(PACKED, OUT).launch(grid=(grid_x, grid_y, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


# Merged K+V launch (see _make_kv_merged_kernel docstring for
# the measured rationale). Reuses the ROTATE candidate list: the merged
# kernel's own re-swept optimum is flat over the range this uses,
# best 3072 -- already inside _ROTATE_GRID_X_CANDIDATES, and K (the harder of
# the two to tune) needs the same large-grid_x region the rotate list covers.
@autotune(
    configs=[Config(grid_x=gx) for gx in _ROTATE_GRID_X_CANDIDATES],
    key=["NG", "B", "total_rows", "npg"],
)
@flyc.jit
def _compiled_rope_kv(
    PACKED,
    K_FREQS,
    K_OUT,
    V_OUT,
    NG: fx.Constexpr[int],
    B: fx.Constexpr[int],
    total_rows: fx.Constexpr[int],
    npg: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_kv_merged_kernel(NG, B, total_rows, npg, grid_x)
    kernel(PACKED, K_FREQS, K_OUT, V_OUT).launch(
        grid=(grid_x, NG * 2, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


# --- Merged-dispatch launcher ---
#
# Motivation: a cProfile of the FlyDSL host
# dispatch cost inside two per-call key-resolution steps that run on EVERY
# Python-level call regardless of cache warmth -- flydsl's own
# ``Autotuner._make_key`` (~25%, string-serializes shape/dtype/stride/env
# every call) and ``JitFunction._resolve_and_make_cache_key`` (~39%, wraps
# every tensor arg into a fresh ``JitArgument`` and hashes it, *before* the
# JIT's own "reuse compiled CallState" fast path can even be consulted --
# read from ``jit_function.py``: the fast-path check is the LAST
# line of that resolution, not a bypass of it). r4 wired the standalone FlyDSL
# RoPE kernel in and measured the device-side family drop (21.770 -> 16.521
# ms/step) land as a smaller step drop (786.0 -> ~784.2-784.6) than the device
# number alone predicts -- exactly the gap this dispatch cost explains, and
# step doesn't drop as much").
#
# Reaching into ``JitFunction``/``Autotuner`` internals to memoize by
# (shape, dtype, stride) was investigated and set aside THIS round: the
# private ``_call_state_cache`` is keyed by a tuple built as a *side effect*
# of the same wrapping pass we would be trying to skip (``_resolve_and_make_
# cache_key`` mutates ``bound_args[name]`` in place while it builds the key),
# so a caller-side shortcut would still have to reconstruct fresh
# ``JitArgument`` wrappers (to carry each call's live data pointer) before it
# could even look up the cache -- i.e. it cannot skip the wrapping cost, only
# the key-hashing cost layered on top, for uncertain payoff against a real
# risk of silently serving a stale ``call_state`` if any invalidating axis
# (env fingerprint, globals snapshot) is missed. Rejected as the H3 mechanism
# in favor of the lever below, which needs no private-API access.
#
# Mechanism actually used: cut the NUMBER of Python-level JIT dispatches per
# ``flydsl_qkv_rope_forward`` call from 2 to 1 by launching Q's rotate kernel
# and the K+V merged kernel back-to-back from ONE ``@autotune``+``@flyc.jit``
# function, on the same stream. This halves the per-call
# ``_make_key``/``_resolve_and_make_cache_key`` tax (paid once per Python
# dispatch, not per GPU kernel) while reusing ``_make_rope_kernel`` and
# ``_make_kv_merged_kernel`` byte-for-byte unchanged -- neither kernel's own,
# independently-measured-optimal ``grid_x`` is touched (Q stays on its own
# small-grid autotune axis; K+V keeps the already-established 3072 from
# ``_make_kv_merged_kernel``'s own docstring), so this is a pure host-side
# dispatch-count change with zero device-kernel-shape risk, unlike the
# rejected Q+K+V *single-kernel* 3-way-branch alternative (which would force
# one of Q/K off its own optimal grid as analyzed in that function's
# docstring, and would need new correctness-sensitive branch code).
#
# Correction (measured): an earlier version of this
# function hardcoded grid_x_kv=3072 from _make_kv_merged_kernel's docstring
# ("own grid_x optimum re-swept ... best 3072"). Re-autotuning THIS kernel on
# THIS box today, side by side, shows the standalone _compiled_rope_kv
# settles on grid_x=1536, not 3072 -- the docstring's number is from a prior
# session/box and the KV kernel's optimum drifts. Hardcoding 3072 made the
# first merged-dispatch attempt measure 4.4-8.99% SLOWER than the
# two-dispatch baseline across all 8 fan-out ranks (bit-exact, so the
# regression was purely from running K+V off its current-day optimum, not
# from the dispatch merge itself) -- textbook "vgpr_count is not the full
# budget" cousin: don't trust a historical constant, autotune both axes.
_QKV_MERGED_GRID_X_Q_CANDIDATES = (160, 192, 224, 256, 320)
_QKV_MERGED_GRID_X_KV_CANDIDATES = (768, 1024, 1536, 2048, 3072)


@autotune(
    configs=[
        Config(grid_x_q=gq, grid_x_kv=gk)
        for gq in _QKV_MERGED_GRID_X_Q_CANDIDATES
        for gk in _QKV_MERGED_GRID_X_KV_CANDIDATES
    ],
    key=["NG", "B", "total_rows", "npg"],
)
@flyc.jit
def _compiled_rope_qkv_merged(
    PACKED,
    Q_FREQS,
    K_FREQS,
    Q_OUT,
    K_OUT,
    V_OUT,
    NG: fx.Constexpr[int],
    B: fx.Constexpr[int],
    total_rows: fx.Constexpr[int],
    npg: fx.Constexpr[int],
    grid_x_q: fx.Constexpr[int],
    grid_x_kv: fx.Constexpr[int],
    stream: fx.Stream,
):
    """One Python-level JIT dispatch, two kernel launches on ``stream``: Q's
    rotate kernel and the K+V merged kernel, EACH WITH ITS OWN AUTOTUNED
    grid_x (both axes searched together, after the hardcoded-KV
    version regressed -- see module comment above). Each launched kernel is
    byte-identical in its trace to what the two-dispatch path builds -- only
    the number of times ``Autotuner.__call__``/``JitFunction.__call__`` run
    on the host changes.
    """
    grid_y_q = NG * npg
    q_kernel = _make_rope_kernel(NG, B, total_rows, 0, npg, True, grid_x_q, npg)
    q_kernel(PACKED, Q_FREQS, Q_OUT).launch(
        grid=(grid_x_q, grid_y_q, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )
    kv_kernel = _make_kv_merged_kernel(NG, B, total_rows, npg, grid_x_kv)
    kv_kernel(PACKED, K_FREQS, K_OUT, V_OUT).launch(
        grid=(grid_x_kv, NG * 2, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


def flydsl_qkv_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list):
    """Raw (non-autograd) FlyDSL RoPE forward; ``primus_turbo.pytorch.ops.rope``
    carries the autograd-wrapped entry point.

    Args:
        qkv: [S, B, NG, (NPG+2)*D] bf16, contiguous.
        q_freqs / k_freqs: [S, 1, 1, D] f32, contiguous, duplicated halves.
        qkv_split_arg_list: [q_size, k_size, v_size] with k_size == v_size == D.

    Returns:
        (q_out, k_out, v_out): bf16 contiguous, q_out [S,B,NG*NPG,D] and
        k_out/v_out [S,B,NG,D].

    Launches 2 kernels: Q (rotate) on its own, K+V merged into one launch
    (``_compiled_rope_kv`` -- measured 15.9% faster than separate K and V
    launches at the production shape, bit-exact, see that function's
    docstring). Q is NOT merged with K *in the kernel*: they need incompatible
    grid_x optima (measured, t17/t18) so merging them into one kernel would
    force one off its optimum, whereas V's grid_x-flatness makes the K+V
    kernel merge free.

    How many Python-level dispatches carry those kernels is a separate axis from
    the kernel shapes; ``_MERGED_DISPATCH`` selects it.
    """
    import torch

    S, B, NG, PACK = qkv.shape
    q_size, k_size, v_size = qkv_split_arg_list
    assert k_size == _D and v_size == _D, f"expected k_size=v_size={_D}, got {k_size}/{v_size}"
    assert q_size % _D == 0, f"q_size {q_size} not a multiple of D={_D}"
    npg = q_size // _D
    assert PACK == (npg + 2) * _D, f"packed width {PACK} != (NPG+2)*D for NPG={npg}, D={_D}"
    assert qkv.is_contiguous(), "packed QKV must be contiguous"
    assert q_freqs.dtype == torch.float32 and k_freqs.dtype == torch.float32
    assert q_freqs.is_contiguous() and k_freqs.is_contiguous()

    total_rows = S * B
    q_out = torch.empty((S, B, NG * npg, _D), dtype=torch.bfloat16, device=qkv.device)
    k_out = torch.empty((S, B, NG, _D), dtype=torch.bfloat16, device=qkv.device)
    v_out = torch.empty((S, B, NG, _D), dtype=torch.bfloat16, device=qkv.device)
    stream = torch.cuda.current_stream()

    if _MERGED_DISPATCH:
        _compiled_rope_qkv_merged(
            PACKED=qkv,
            Q_FREQS=q_freqs,
            K_FREQS=k_freqs,
            Q_OUT=q_out,
            K_OUT=k_out,
            V_OUT=v_out,
            NG=NG,
            B=B,
            total_rows=total_rows,
            npg=npg,
            stream=stream,
        )
    else:
        _compiled_rope_rotate(
            PACKED=qkv,
            FREQS=q_freqs,
            OUT=q_out,
            NG=NG,
            B=B,
            total_rows=total_rows,
            hk_base=0,
            n_hk=npg,
            npg=npg,
            stream=stream,
        )
        _compiled_rope_kv(
            PACKED=qkv,
            K_FREQS=k_freqs,
            K_OUT=k_out,
            V_OUT=v_out,
            NG=NG,
            B=B,
            total_rows=total_rows,
            npg=npg,
            stream=stream,
        )

    return q_out, k_out, v_out


# ==========================================================================
# Backward (inverse RoPE) kernel + launchers. Same rotation as the forward with
# sin negated, so the packed dQKV comes back in one pass.
# Mirrors _make_rope_kernel/_make_kv_merged_kernel with source and destination
# roles swapped: reads the contiguous per-tensor gradients and writes the
# strided packed dQKV gradient.
#
# Rotation is the forward 2x2 rotate-half matrix transposed (sin negated):
#     d_lo = d_lo' * cos + d_hi' * sin
#     d_hi = d_hi' * cos - d_lo' * sin
# where (d_lo', d_hi') is the incoming (contiguous, per-tensor) gradient
# half-pair and (d_lo, d_hi) is the outgoing (packed) one. V has no rotation,
# matching the forward.
#
# Correctness is covered by tests/pytorch/ops/test_rope.py against a plain
# PyTorch reference.
#
# Launch count: 3 (Q rotate, K rotate, V copy) -- the same shape as this
# round's prototype, deliberately NOT merging K+V the way forward's
# _compiled_rope_kv does. That merge is a separate, already-measured-risky
# Merging the two launches is a host-side lever only; see _MERGED_DISPATCH.
# into one JIT dispatch cut host time ~32% but cost DEVICE time +5-6%, net
# negative in isolation for the forward K+V pair) and is not this file's
# single hypothesis -- one change per round (iteration_rules.mdc Rule 1).
# ==========================================================================


def _make_rope_bwd_kernel(
    NG: int, B: int, total_rows: int, hk_base: int, n_hk: int, do_rotate: bool, grid_x: int, npg: int = 4
):
    """Build one specialized inverse-RoPE backward kernel.

    Same parameterisation as ``_make_rope_kernel`` (``hk_base``/``n_hk`` select
    which packed head-slot this kernel targets: hk_base=0, n_hk=NPG for Q;
    hk_base=NPG, n_hk=1 for K; hk_base=NPG+1, n_hk=1 for V), but ``DSPLIT`` is
    the READ side (contiguous, one tensor per Q/K/V) and ``DPACKED`` is the
    WRITE side (strided packed dQKV) -- the opposite of forward's PACKED-in,
    split-out direction. ``do_rotate=False`` (V) skips the frequency load and
    rotate-half math entirely, same as forward's V branch.
    """
    pack_stride = NG * (npg + 2) * _D
    src_row_stride = NG * n_hk * _D

    if do_rotate:

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def rope_bwd_kernel(DSPLIT: fx.Tensor, FREQS: fx.Tensor, DPACKED: fx.Tensor):
            f32ty = _vec_type(fx.T.f32(), _VEC)
            bf16ty = _vec_type(fx.T.bf16(), _VEC)
            thread_index = fx.thread_idx.x
            block_index_x, block_index_y, _ = fx.block_idx
            row_lane = thread_index // fx.Int32(_NPLANE)
            p_lane = thread_index % fx.Int32(_NPLANE)
            p = p_lane * fx.Int32(_VEC)
            g = block_index_y // fx.Int32(n_hk)
            lh = block_index_y % fx.Int32(n_hk)
            hk = fx.Int32(hk_base) + lh

            split_rsrc = buffer_ops.create_buffer_resource(DSPLIT, max_size=True)
            packed_rsrc = buffer_ops.create_buffer_resource(DPACKED, max_size=True)
            freq_rsrc = buffer_ops.create_buffer_resource(FREQS, max_size=True)

            dst_col_base = (g * fx.Int32(npg + 2) + hk) * fx.Int32(_D)
            src_head = g * fx.Int32(n_hk) + lh
            src_col_base = src_head * fx.Int32(_D)

            def compute_rows(row0):
                row = row0 + row_lane
                src_base = row * fx.Int32(src_row_stride) + src_col_base + p
                dst_base = row * fx.Int32(pack_stride) + dst_col_base + p
                in_lo = buffer_ops.buffer_load(split_rsrc, src_base, vec_width=_VEC, dtype=fx.T.bf16())
                in_hi = buffer_ops.buffer_load(
                    split_rsrc, src_base + fx.Int32(_HALF), vec_width=_VEC, dtype=fx.T.bf16()
                )
                # Position for the frequency table: s = row // B (B is MBS; row = s*B+b).
                s = row // fx.Int32(B)
                freq_off = s * fx.Int32(_D) + p
                freq = buffer_ops.buffer_load(freq_rsrc, freq_off, vec_width=_VEC, dtype=fx.T.f32())
                cos_p = math.cos(freq)
                sin_p = math.sin(freq)
                lo_f = fx.arith.extf(f32ty, in_lo)
                hi_f = fx.arith.extf(f32ty, in_hi)
                # Transpose of the forward rotation: sin negated.
                out_lo_f = fx.arith.addf(fx.arith.mulf(lo_f, cos_p), fx.arith.mulf(hi_f, sin_p))
                out_hi_f = fx.arith.subf(fx.arith.mulf(hi_f, cos_p), fx.arith.mulf(lo_f, sin_p))
                out_lo = fx.arith.trunc_f(bf16ty, out_lo_f)
                out_hi = fx.arith.trunc_f(bf16ty, out_hi_f)
                buffer_ops.buffer_store(out_lo, packed_rsrc, dst_base)
                buffer_ops.buffer_store(out_hi, packed_rsrc, dst_base + fx.Int32(_HALF))

            row0 = block_index_x * fx.Int32(_NROW)
            while row0 < fx.Int32(total_rows):
                compute_rows(row0)
                row0 = row0 + fx.Int32(grid_x * _NROW)

    else:

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def rope_bwd_kernel(DSPLIT: fx.Tensor, DPACKED: fx.Tensor):
            thread_index = fx.thread_idx.x
            block_index_x, block_index_y, _ = fx.block_idx
            row_lane = thread_index // fx.Int32(_NPLANE)
            p_lane = thread_index % fx.Int32(_NPLANE)
            p = p_lane * fx.Int32(_VEC)
            g = block_index_y // fx.Int32(n_hk)
            lh = block_index_y % fx.Int32(n_hk)
            hk = fx.Int32(hk_base) + lh

            split_rsrc = buffer_ops.create_buffer_resource(DSPLIT, max_size=True)
            packed_rsrc = buffer_ops.create_buffer_resource(DPACKED, max_size=True)

            dst_col_base = (g * fx.Int32(npg + 2) + hk) * fx.Int32(_D)
            src_head = g * fx.Int32(n_hk) + lh
            src_col_base = src_head * fx.Int32(_D)

            def compute_rows(row0):
                row = row0 + row_lane
                src_base = row * fx.Int32(src_row_stride) + src_col_base + p
                dst_base = row * fx.Int32(pack_stride) + dst_col_base + p
                lo = buffer_ops.buffer_load(split_rsrc, src_base, vec_width=_VEC, dtype=fx.T.bf16())
                hi = buffer_ops.buffer_load(
                    split_rsrc, src_base + fx.Int32(_HALF), vec_width=_VEC, dtype=fx.T.bf16()
                )
                buffer_ops.buffer_store(lo, packed_rsrc, dst_base)
                buffer_ops.buffer_store(hi, packed_rsrc, dst_base + fx.Int32(_HALF))

            row0 = block_index_x * fx.Int32(_NROW)
            while row0 < fx.Int32(total_rows):
                compute_rows(row0)
                row0 = row0 + fx.Int32(grid_x * _NROW)

    return rope_bwd_kernel


# grid_x candidates: reuse the forward kernels' own sweep results verbatim
# (_ROTATE_GRID_X_CANDIDATES / _COPY_GRID_X_CANDIDATES, defined above) rather
# than introducing a second candidate list. The backward rotate/copy kernels
# share forward's exact (grid_y, do_rotate) shape per role -- Q here has the
# same grid_y=NG*NPG=32 as forward's Q rotate, K has the same grid_y=NG=8 as
# forward's K (inside the KV-merged kernel), V has the same grid_y=NG=8
# copy-only shape as forward's V -- so the autotuner still independently
# re-searches this range for ITS OWN function (each @autotune call below gets
# its own Autotuner instance / own on-disk cache per tune_utils.py, keyed by
# this function's identity plus the listed key fields) instead of trusting
# Autotune both axes rather than carrying the forward optimum over: the two
# kernels do not share an optimum.
# regression is exactly this mistake for a sibling kernel).
@autotune(
    configs=[Config(grid_x=gx) for gx in _ROTATE_GRID_X_CANDIDATES],
    key=["NG", "B", "total_rows", "hk_base", "n_hk", "npg"],
)
@flyc.jit
def _compiled_rope_bwd_rotate(
    DSPLIT,
    FREQS,
    DPACKED,
    NG: fx.Constexpr[int],
    B: fx.Constexpr[int],
    total_rows: fx.Constexpr[int],
    hk_base: fx.Constexpr[int],
    n_hk: fx.Constexpr[int],
    npg: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    stream: fx.Stream,
):
    grid_y = NG * n_hk
    kernel = _make_rope_bwd_kernel(NG, B, total_rows, hk_base, n_hk, True, grid_x, npg)
    kernel(DSPLIT, FREQS, DPACKED).launch(
        grid=(grid_x, grid_y, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


@autotune(
    configs=[Config(grid_x=gx) for gx in _COPY_GRID_X_CANDIDATES],
    key=["NG", "B", "total_rows", "hk_base", "n_hk", "npg"],
)
@flyc.jit
def _compiled_rope_bwd_copy(
    DSPLIT,
    DPACKED,
    NG: fx.Constexpr[int],
    B: fx.Constexpr[int],
    total_rows: fx.Constexpr[int],
    hk_base: fx.Constexpr[int],
    n_hk: fx.Constexpr[int],
    npg: fx.Constexpr[int],
    grid_x: fx.Constexpr[int],
    stream: fx.Stream,
):
    grid_y = NG * n_hk
    kernel = _make_rope_bwd_kernel(NG, B, total_rows, hk_base, n_hk, False, grid_x, npg)
    kernel(DSPLIT, DPACKED).launch(grid=(grid_x, grid_y, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def flydsl_qkv_rope_backward(dq, dk, dv, q_freqs, k_freqs, qkv_split_arg_list):
    """Inverse-RoPE backward: the packed ``dQKV`` gradient, from the per-tensor
    Q/K/V gradients. Raw (non-autograd) entry point, like the forward;
    ``primus_turbo.pytorch.ops.rope`` wraps both.

    Args:
        dq: [S, B, NG*NPG, D] bf16, contiguous (gradient w.r.t. q_out).
        dk / dv: [S, B, NG, D] bf16, contiguous (gradient w.r.t. k_out/v_out).
        q_freqs / k_freqs: [S, 1, 1, D] f32, contiguous, duplicated halves --
            the SAME tensors ``_FlyDSLFusedQKVRoPEFunc.forward`` saved
            (already float32 + contiguous by construction, see that method).
        qkv_split_arg_list: [q_size, k_size, v_size] with k_size == v_size == D.

    Returns:
        grad_qkv: [S, B, NG, (NPG+2)*D] bf16 contiguous, matching the packed
        ``qkv`` shape ``_FlyDSLFusedQKVRoPEFunc.forward`` received.

    Launches 3 kernels (Q rotate, K rotate, V copy) on the current stream --
    see the module comment above this section for why K+V are not merged
    here.
    """
    import torch

    S, B = dq.shape[0], dq.shape[1]
    NG = dk.shape[2]
    q_size, k_size, v_size = qkv_split_arg_list
    assert k_size == _D and v_size == _D, f"expected k_size=v_size={_D}, got {k_size}/{v_size}"
    assert q_size % _D == 0, f"q_size {q_size} not a multiple of D={_D}"
    npg = q_size // _D
    assert dq.shape[2] == NG * npg, f"dq head dim {dq.shape[2]} != NG*NPG ({NG}*{npg})"
    assert dq.shape[-1] == _D and dk.shape[-1] == _D and dv.shape[-1] == _D
    assert dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous(), (
        "dq/dk/dv must be contiguous -- the production trace shows no copy "
        "kernel between ck_fused_attn::dk_dv_reduce and rope_bwd (r8 "
        "forensics), so callers should already satisfy this; "
        "_te_rope_backward calls .contiguous() defensively before reaching "
        "here, so this assert should never fire in production."
    )
    assert q_freqs.dtype == torch.float32 and k_freqs.dtype == torch.float32
    assert q_freqs.is_contiguous() and k_freqs.is_contiguous()

    total_rows = S * B
    grad_qkv = torch.empty((S, B, NG, (npg + 2) * _D), dtype=torch.bfloat16, device=dq.device)
    stream = torch.cuda.current_stream()

    _compiled_rope_bwd_rotate(
        DSPLIT=dq,
        FREQS=q_freqs,
        DPACKED=grad_qkv,
        NG=NG,
        B=B,
        total_rows=total_rows,
        hk_base=0,
        n_hk=npg,
        npg=npg,
        stream=stream,
    )
    _compiled_rope_bwd_rotate(
        DSPLIT=dk,
        FREQS=k_freqs,
        DPACKED=grad_qkv,
        NG=NG,
        B=B,
        total_rows=total_rows,
        hk_base=npg,
        n_hk=1,
        npg=npg,
        stream=stream,
    )
    _compiled_rope_bwd_copy(
        DSPLIT=dv,
        DPACKED=grad_qkv,
        NG=NG,
        B=B,
        total_rows=total_rows,
        hk_base=npg + 1,
        n_hk=1,
        npg=npg,
        stream=stream,
    )

    return grad_qkv
