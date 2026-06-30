###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Small FlyDSL helpers vendored for the ported DeepSeek-V4 attention kernels.

The reference kernels (ported from the FlyDSL-amd source tree) import a single
symbol, ``dtype_to_elem_type``, from ``kernels.kernels_common``. That package is
part of the FlyDSL-amd *source* layout and is not present in the pip-installed
``flydsl`` runtime, so the one helper the kernels need is reconstructed here.

``dtype_to_elem_type`` maps a dtype string ("bf16" / "f16" / ...) to the FlyDSL
MLIR element type (``flydsl.expr.typing.T.bf16`` etc.). The ``T.*`` properties
build their MLIR type lazily, so this must be called inside an active MLIR
``Context`` (i.e. during kernel build), exactly as in the reference.
"""

from __future__ import annotations

from flydsl.expr.typing import T

__all__ = [
    "dtype_to_elem_type",
    "mfma_f32_16x16x32",
    "mfma_mv_reduce_16",
    "dgathered_atomic_elem_base",
    "dgathered_split_elem_base",
]

# Accepted spellings per logical dtype (lower-cased before lookup).
_BF16 = {"bf16", "bfloat16"}
_F16 = {"f16", "fp16", "float16", "half"}
_F32 = {"f32", "fp32", "float32", "float"}
_F8E4M3 = {"f8", "fp8", "f8e4m3", "float8_e4m3", "e4m3"}
_F8E5M2 = {"f8e5m2", "float8_e5m2", "e5m2"}


def dtype_to_elem_type(dtype_str):
    """Return the FlyDSL element type for ``dtype_str`` (call inside a Context)."""
    s = str(dtype_str).lower()
    if s in _BF16:
        return T.bf16
    if s in _F16:
        return T.f16
    if s in _F32:
        return T.f32
    if s in _F8E5M2:
        return T.f8
    if s in _F8E4M3:
        return T.f8
    raise ValueError(f"unsupported dtype_str for FlyDSL elem type: {dtype_str!r}")


def dgathered_atomic_elem_base(
    bid_i32, seq_len_i32, pid_m_safe_i32, K_topk_i32,
    k_pos_i32, lane_i32, d_per_lane_i32, head_dim_i32,
):
    """Element-index base into DGATHERED ``[B, Sq, K_topk, D]`` for one lane's
    per-``(bid, pid_m, k_pos)`` atomic_fadd RMW row.

    Returns the i32 element index of this lane's ``d_off == 0`` slot; the
    per-``d_off`` element index is simply ``base + d_off`` (one add). Folding the
    ``((bid*Sq + pid_m)*K_topk + k_pos)*D + lane*D_PER_LANE`` row/col arithmetic
    into a single reused value keeps the address live range short -- the caller
    carries one base plus a tiny per-``d_off`` add, instead of materialising a
    full independently-live absolute address for every ``d_off``. The DGATHERED
    layout and the atomic reduction algorithm are unchanged; this only moves
    address generation. Must be called inside an active MLIR Context.
    """
    from flydsl.expr import arith

    _bm = arith.AddIOp(
        arith.MulIOp(bid_i32, seq_len_i32).result, pid_m_safe_i32,
    ).result
    _bm_k = arith.AddIOp(
        arith.MulIOp(_bm, K_topk_i32).result, k_pos_i32,
    ).result
    _row_d = arith.AddIOp(
        arith.MulIOp(_bm_k, head_dim_i32).result,
        arith.MulIOp(lane_i32, d_per_lane_i32).result,
    ).result
    return _row_d


def dgathered_split_elem_base(
    bid_i32, seq_len_i32, pid_m_safe_i32, K_topk_i32,
    k_pos_i32, group_i32, num_groups_i32, lane_i32, d_per_lane_i32, head_dim_i32,
):
    """Element-index base into the WARP-DISJOINT split-K scratch
    ``DGATHERED_SPLIT[B, Sq, K_topk, num_groups, D]`` (bf16) for one head-group
    program's ``(bid, pid_m, k_pos, group_id)`` packed-bf16 store row.

    Instead of every head-group program serialising an fp32 ``atomic_fadd`` RMW
    into the SAME ``DGATHERED[b, q, k_pos, d]`` word, each of
    the ``num_groups = HQ/HEAD_BLOCK`` head-group programs owns a DISJOINT
    ``group_id`` stripe of a packed-bf16 split buffer and writes it with a plain
    (race-free) packed 2xbf16 store -- no atomics, half the accumulation store
    bytes. A separate conflict-free finalize pass sums the ``num_groups`` bf16
    stripes back into the fp32 ``DGATHERED``.

    Returns the i64 element index of this lane's ``(group_id, d_off == 0)`` slot;
    the per-``d_off`` element index is ``base + d_off``. The group axis is placed
    immediately above ``D`` so the per-lane ``D_PER_LANE`` columns stay
    contiguous (32-bit-aligned packed stores) and the base needs only ``Sq``,
    ``K_topk``, ``num_groups`` and ``D`` (all known to the kernel; no ``B``).

    The flat element index is accumulated in 64-bit ``index`` arithmetic: the
    split scratch has ``B*Sq*K_topk*num_groups*D`` elements, which exceeds the
    signed-i32 range (2^31) for wide MHA shapes (e.g. ``num_groups == HQ`` with
    ``HQ*Sq >= ~6.5e4`` at ``D == 512``); an i32 product would wrap and produce a
    wild out-of-bounds store. Must be called inside an active MLIR Context.
    """
    from flydsl.expr import arith

    bid = arith.extsi(T.i64, bid_i32)
    seq_len = arith.extsi(T.i64, seq_len_i32)
    pid_m = arith.extsi(T.i64, pid_m_safe_i32)
    K_topk = arith.extsi(T.i64, K_topk_i32)
    k_pos = arith.extsi(T.i64, k_pos_i32)
    group = arith.extsi(T.i64, group_i32)
    num_groups = arith.extsi(T.i64, num_groups_i32)
    lane = arith.extsi(T.i64, lane_i32)
    d_per_lane = arith.extsi(T.i64, d_per_lane_i32)
    head_dim = arith.extsi(T.i64, head_dim_i32)

    _bm = arith.AddIOp(arith.MulIOp(bid, seq_len).result, pid_m).result
    _bm_k = arith.AddIOp(arith.MulIOp(_bm, K_topk).result, k_pos).result
    _bm_k_g = arith.AddIOp(arith.MulIOp(_bm_k, num_groups).result, group).result
    _row_d = arith.AddIOp(
        arith.MulIOp(_bm_k_g, head_dim).result,
        arith.MulIOp(lane, d_per_lane).result,
    ).result
    return _row_d


def mfma_f32_16x16x32(a_pack, b_pack, c_acc, dtype_str="bf16"):
    """Single CDNA4 (gfx950) ``v_mfma_f32_16x16x32_{bf16,f16}``.

    Computes one matrix-core tile ``D[16, 16] += A[16, 32] @ B[32, 16]`` with
    f32 accumulation. ``a_pack`` / ``b_pack`` are the MFMA operand packs
    (``vec(8, elem)`` -- 4 VGPRs, 8 packed 16-bit items per lane, in the
    standard A/B lane layout: lane ``L`` holds row/col ``L % 16`` and K-subgroup
    ``(L // 16) * 8 + 0..7``). ``c_acc`` is the ``vec(4, f32)`` accumulator
    (column-major: lane ``L`` holds ``C[(L // 16) * 4 + k, L % 16]`` for
    ``k = 0..3``). Returns the updated ``vec(4, f32)`` accumulator.

    ``cbsz / abid / blgp`` are left at 0; operand sharing across the 16 M-rows
    (the single-query broadcast) is realised by a per-lane redundant load of the
    broadcast operand, which is functionally identical and avoids illegal
    broadcast encodings. Must be called inside an active MLIR Context.
    """
    from flydsl.expr import rocdl

    rty = T.vec(4, T.f32)
    s = str(dtype_str).lower()
    if s in _F16:
        return rocdl.mfma_f32_16x16x32_f16(rty, [a_pack, b_pack, c_acc])
    return rocdl.mfma_f32_16x16x32_bf16(rty, [a_pack, b_pack, c_acc])


def mfma_mv_reduce_16(a_packs, b_packs, num_rows, dtype_str="bf16"):
    """MFMA 16x16x32 matrix-vector reduction with a single (broadcast) B column.

    Reusable matrix-core replacement for the per-key ``vec_dot_f32`` +
    ``warp_reduce_sum`` (bpermute) reduction chains in the per-query CSA
    backward kernels. Computes, for up to 16 contraction rows against one
    broadcast vector ``q`` (or ``dO``), ``out[r] = sum_K A[r, K] * q[K]`` on the
    matrix core instead of a lane-partial dot + 6-step cross-lane butterfly.

    ``a_packs[ks]`` / ``b_packs[ks]`` are the MFMA operand packs for K-step
    ``ks`` (``vec(8, elem)`` -- 8 packed 16-bit items per lane). They MUST be
    staged in the standard MFMA 16x16x32 A/B lane layout:

        a_packs[ks]: lane L holds A[row = L % 16, K = ks*32 + (L // 16)*8 + 0..7]
        b_packs[ks]: lane L holds q[K = ks*32 + (L // 16)*8 + 0..7]  (independent
                     of L % 16 -- the single owned query broadcast across the 16
                     MFMA N-columns, realised by a per-lane redundant load).

    The K-step accumulation covers ``len(a_packs)*32`` contraction elements
    (= head_dim when ``len(a_packs) == head_dim // 32``).

    Returns a Python list of ``num_rows`` f32 scalars; entry ``r`` is the full
    reduction for row ``r``, *broadcast to every lane* via ``readlane`` so the
    caller can apply masks / bias / exp exactly as the warp-reduce path did
    (whose butterfly also left the sum on all lanes). The C-frag layout is
    ``C[m = (L // 16)*4 + ii, n = L % 16]``; because B is broadcast, every N
    column holds the same value, so row ``r`` is read from lane ``(r // 4)*16``
    (N-column 0), accumulator element ``ii = r % 4``.

    Must be called inside an active MLIR Context.
    """
    from flydsl.expr import rocdl, vector, arith

    assert num_rows <= 16, "mfma_mv_reduce_16 produces at most 16 rows per call"
    assert len(a_packs) == len(b_packs), "a_packs / b_packs K-step count mismatch"

    c = arith.constant_vector(0.0, T.vec(4, T.f32))
    for ks in range(len(a_packs)):
        c = mfma_f32_16x16x32(a_packs[ks], b_packs[ks], c, dtype_str)

    out = []
    for r in range(num_rows):
        ii = r % 4
        src_lane = arith.constant((r // 4) * 16, type=T.i32)
        val = vector.extract(c, static_position=[ii], dynamic_position=[])
        out.append(rocdl.readlane(T.f32, val, src_lane))
    return out
