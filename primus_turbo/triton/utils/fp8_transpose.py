###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Fused fp8 3-D transpose ([B, K, N] -> [B, N, K]).

Round-13 (Lever H) helper used by the FP8 grouped GEMM dA-backward
re-route (``grouped_gemm_fp8_impl.py``). The PyTorch reference path
``b.transpose(-2, -1).contiguous()`` dispatches to a generic
``elementwise_kernel_manual_unroll<12,...>`` copy that runs at ~1 TB/s
effective on a 530 MB rd+wr (``B=32 K=N=2880`` fp8) — only 14 % of MI350X
HBM peak (3.4 TB/s) because the transposed source stride defeats coalescing.

This Triton kernel stages a ``BK x BN`` tile through registers (Triton's
``tl.trans`` lowers to in-register shuffle on CDNA4) and hits ~7.6× speedup
vs PyTorch on the gpt_oss-Down B32-M2048 worst case (1056 -> 138 μs at
BK=BN=128). At the per-iter level that closes ~30 % of the bwd wall on
the 8 gpt_oss FP8 cases (the only shapes that enter the H4 reroute path,
because they are exactly the cases with ``K_RRR % 128 != 0``).

The kernel operates on raw 8-bit views (``b.view(torch.uint8)``) so it is
dtype-agnostic across ``float8_e4m3fn`` / ``float8_e4m3fnuz`` /
``float8_e5m2``. Bit-identical to the PyTorch path (verified via
``torch.equal(out_ref.view(torch.uint8), out_tri.view(torch.uint8))``).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_transpose_3d_kernel(
    src_ptr,
    dst_ptr,
    B,
    K,
    N,
    stride_b_src,
    stride_k_src,
    stride_n_src,
    stride_b_dst,
    stride_n_dst,
    stride_k_dst,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kn = tl.program_id(1)
    num_n_blocks = tl.cdiv(N, BN)
    pid_k = pid_kn // num_n_blocks
    pid_n = pid_kn % num_n_blocks

    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_n = pid_n * BN + tl.arange(0, BN)

    src_off = (
        pid_b * stride_b_src
        + offs_k[:, None] * stride_k_src
        + offs_n[None, :] * stride_n_src
    )
    src_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    block = tl.load(src_ptr + src_off, mask=src_mask)

    block_t = tl.trans(block)

    dst_off = (
        pid_b * stride_b_dst
        + offs_n[:, None] * stride_n_dst
        + offs_k[None, :] * stride_k_dst
    )
    dst_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

    tl.store(dst_ptr + dst_off, block_t, mask=dst_mask)


def _select_block_shape(K: int, N: int) -> tuple[int, int]:
    """Round-15 (Lever Q) per-shape (BK, BN) heuristic.

    Source: R14 microbench sweep (``/tmp/tri_transpose_sweep.py``) over 4
    gpt_oss reroute shapes ([4|32, 5760|2880, 2880]) at BK,BN ∈
    {64,128,256}^2. Per-shape winners cluster by K vs N relationship:

      - K > N   (e.g. gpt_oss-GateUP K=5760, N=2880): (BK=128, BN=256)
        — fewer N-blocks (12 vs 23 at BN=128) keeps the per-program
        work-set in L2.  Wall: 23.0 μs B4 / 187.7 μs B32 (vs 25.5 / 193
        at default).

      - K == N (e.g. gpt_oss-Down K=N=2880): (BK=256, BN=128) — half the
        K-blocks (12 vs 23) and N-blocks unchanged.  Wall: 13.2 μs B4 /
        129.5 μs B32 (vs 14.1 / 139.4 at default, -7.1 % B32 worst case).

      - K < N (default): (BK=128, BN=128) — the default sweep saturates
        at this block shape for K=1536-N=4096 (Qwen3-Down) etc.  No FP8
        metric shape currently triggers this branch (the H4 reroute only
        fires for gpt_oss K=2880 cases where K ≥ N).

    Static heuristic instead of @triton.autotune to avoid the per-call
    cache lookup overhead and the 30 ms first-call sweep that polluted
    R15's 5-trial metric distribution (range 958-964 with autotune vs
    961-963 with static defaults).
    """
    if K > N:
        return 128, 256
    if K == N:
        return 256, 128
    return 128, 128


def fp8_transpose_3d(b: torch.Tensor) -> torch.Tensor:
    """Fused [B, K, N] -> [B, N, K] transpose for any 8-bit dtype.

    Round-15 (Lever Q): the (BK, BN) tile shape is selected per-shape via
    ``_select_block_shape(K, N)``; see that helper for the empirical
    sweep data and rationale.  Default fallback (BK=BN=128) preserved for
    K < N inputs which the FP8 metric never triggers.

    Args:
        b: Source tensor with shape ``[B, K, N]``. Must be contiguous and
            have an 8-bit dtype (``float8_e4m3fn`` /
            ``float8_e4m3fnuz`` / ``float8_e5m2``).

    Returns:
        A new contiguous tensor with shape ``[B, N, K]`` and the same dtype
        as ``b``. Bit-identical to ``b.transpose(-2, -1).contiguous()``.
    """
    assert b.dim() == 3, f"expected 3-D tensor, got {b.dim()}-D"
    assert b.element_size() == 1, (
        f"expected 8-bit dtype, got {b.dtype} "
        f"(element_size={b.element_size()})"
    )
    assert b.is_contiguous(), "fp8_transpose_3d expects a contiguous source"
    B, K, N = b.shape
    bk, bn = _select_block_shape(K, N)
    out = torch.empty((B, N, K), dtype=b.dtype, device=b.device)
    src_v = b.view(torch.uint8)
    dst_v = out.view(torch.uint8)
    num_n_blocks = triton.cdiv(N, bn)
    grid = (B, triton.cdiv(K, bk) * num_n_blocks)
    _fp8_transpose_3d_kernel[grid](
        src_v,
        dst_v,
        B,
        K,
        N,
        K * N,
        N,
        1,
        N * K,
        K,
        1,
        BK=bk,
        BN=bn,
    )
    return out
