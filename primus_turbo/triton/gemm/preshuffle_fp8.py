###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Coalesced Triton pre-shuffle for the FlyDSL blockwise-FP8 backward operands.

The FlyDSL wgrad/dgrad GEMMs consume their second operand in the kernel's
``(16, 16)`` MFMA layout produced by ``shuffle_b(src.transpose(0, 1))``. The
launcher historically materialised this with a single ``torch.as_strided`` +
``.contiguous()`` whose innermost (``K_inner``) read stride is ``P`` -- an
*uncoalesced* gather that lowers to the generic ``elementwise_kernel`` copy and
shows up as the largest non-GEMM cost in the backward timeline.

This module produces a **byte-identical** operand with a purpose-built kernel
that (a) reads the contiguous ``[Q, P]`` source coalesced along the contiguous
``P`` axis and (b) writes the permuted output in contiguous ``K_inner`` runs,
turning the uncoalesced gather into a clean 2D transpose-store.

Layout derivation (matches ``launcher._shuffle_b_transposed`` exactly):
    ``T = src.transpose(0, 1)`` has shape ``[P, Q]``; ``shuffle_b(T)`` views it
    as ``(P//BN, BN, Q//BK, BK//K_inner, K_inner)`` and permutes the dims to
    ``(i0, i2, i3, i1, i4)`` before the final contiguous ``view(P, Q)``. For a
    logical output coordinate ``(p, q)`` (``T[p, q] == src[q, p]``):
        i0 = p // BN,  i1 = p % BN
        i2 = q // BK,  i3 = (q % BK) // K_inner,  i4 = q % K_inner
        out_flat = i4 + K_inner*(i1 + BN*(i3 + (BK//K_inner)*(i2 + (Q//BK)*i0)))
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton optional at import time
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _preshuffle_b_transposed_kernel(
        src_ptr,  # *u8  contiguous [Q, P]
        out_ptr,  # *u8  contiguous [P, Q] (shuffled order)
        P,
        Q,
        BN: tl.constexpr,
        BK: tl.constexpr,
        K_inner: tl.constexpr,
        TP: tl.constexpr,
        TQ: tl.constexpr,
    ):
        pid_p = tl.program_id(0)
        pid_q = tl.program_id(1)
        p = pid_p * TP + tl.arange(0, TP)  # [TP]  (contiguous axis of src)
        q = pid_q * TQ + tl.arange(0, TQ)  # [TQ]
        p_mask = p < P
        q_mask = q < Q

        # Coalesced load: src[q, p] at q*P + p, contiguous along p (last dim).
        src_off = q[:, None] * P + p[None, :]  # [TQ, TP]
        in_mask = q_mask[:, None] & p_mask[None, :]
        vals = tl.load(src_ptr + src_off, mask=in_mask, other=0)  # [TQ, TP]
        vals_t = tl.trans(vals)  # [TP, TQ]

        # Permuted output offset for each (p, q); writes are contiguous in
        # runs of K_inner along q (last dim), so the store coalesces.
        i0 = p // BN
        i1 = p % BN
        i2 = q // BK
        rem = q % BK
        i3 = rem // K_inner
        i4 = rem % K_inner
        out_off = i4[None, :] + K_inner * (
            i1[:, None]
            + BN * (i3[None, :] + (BK // K_inner) * (i2[None, :] + (Q // BK) * i0[:, None]))
        )  # [TP, TQ]
        out_mask = p_mask[:, None] & q_mask[None, :]
        tl.store(out_ptr + out_off, vals_t, mask=out_mask)


def preshuffle_b_transposed_triton(
    src: torch.Tensor, layout: Tuple[int, int] = (16, 16)
) -> torch.Tensor:
    """Coalesced equivalent of ``shuffle_b(src.transpose(0, 1).contiguous())``.

    ``src`` is the un-transposed, row-major ``[Q, P]`` FP8 operand. Returns the
    pre-shuffled transpose ``[P, Q]`` in the kernel's ``(16, 16)`` MFMA layout.
    Only the 1-byte (FP8) path is accelerated; the caller falls back to the
    torch strided path for any other element size.
    """
    if not src.is_contiguous():
        src = src.contiguous()
    Q, P = src.shape
    IN, IK = layout
    BK = IK * 2
    K_inner = 16 // src.element_size()
    BN = IN
    assert (
        P % BN == 0 and Q % BK == 0
    ), f"preshuffle_b_transposed_triton: P={P} Q={Q} not divisible by ({BN}, {BK})"

    out = torch.empty((P, Q), dtype=src.dtype, device=src.device)
    # Operate on raw bytes so the copy is exactly bit-preserving (no FP8
    # canonicalisation) and identical to the torch strided-contiguous result.
    src_u8 = src.view(torch.uint8)
    out_u8 = out.view(torch.uint8)

    # Bandwidth retune (round-26, K1 kernel-internal): widen the contiguous-p
    # load tile to a full 128-byte cache line (TP=128) and give the copy an
    # explicit 8-warp launch so more 16-byte vector loads/stores are in flight
    # per block to hide HBM latency. TP is a multiple of BN(16) and TQ a
    # multiple of both BK(32) and K_inner(16), so the i0..i4 permutation math is
    # untouched and the output stays byte-identical to the torch reference.
    TP = 128
    TQ = 64
    grid = (triton.cdiv(P, TP), triton.cdiv(Q, TQ))
    _preshuffle_b_transposed_kernel[grid](
        src_u8,
        out_u8,
        P,
        Q,
        BN=BN,
        BK=BK,
        K_inner=K_inner,
        TP=TP,
        TQ=TQ,
        num_warps=8,
    )
    return out


def is_available() -> bool:
    return _HAS_TRITON
