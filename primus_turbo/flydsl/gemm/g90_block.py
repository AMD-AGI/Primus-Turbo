###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""Dispatch of the 128-block (UE8M0) FP8 dense GEMMs to the gfx1250 g90 FlyDSL kernel.

Enabled by PRIMUS_TURBO_FP8_BLOCK_KERNEL=g90 inside FP8GemmBlockFunction (trans_a=False, trans_b=True,
2D operands). All three passes become NT GEMMs:
  fwd   Y[M,N]  = X[M,K]  . W[N,K]^T       A 1x128 scales, B 128x128 block scales
  dgrad dX[M,K] = dY[M,N] . (W^T)[K,N]^T   W^T is an FP8 copy of W (128x128 blocks are transpose-invariant)
  wgrad dW[N,K] = dY^T[N,M] . X^T[K,M]^T  token-major copies, 1x128 along tokens on both operands
Config choice from the per-layer sweep (g90gemm ledger rows 21-22): 256x256 (2x2 waves, 3 V_WMMA_SCALE rows +
VALU rescale) for M, N >= 2048; else 128x128 all-V_WMMA_SCALE; per-row B scales (wgrad) use 128x128.
"""

import os

import torch

from primus_turbo.flydsl.gemm.gemm_g90_kernel import gemm_g90

_CFG_256 = dict(tile=(256, 256), lead=24, tdm_mode=2, ws_rows=3)
_CFG_128 = dict(tile=(128, 128), lead=4, tdm_mode=2, wscale=True)
_CFG_128_ROWB = dict(tile=(128, 128), lead=4, tdm_mode=2)


def enabled() -> bool:
    # dsv4up rides on the g90 path (gluon/gemm/dsv4up_block.py swaps the GEMM per supported pass)
    return os.environ.get("PRIMUS_TURBO_FP8_BLOCK_KERNEL", "") in ("g90", "dsv4up")


def _cfg(M, N, K, rowb):
    if rowb:
        return _CFG_128_ROWB
    if M >= 2048 and N >= 2048 and N % 256 == 0:
        return _CFG_256
    return _CFG_128


def supported(M, N, K) -> bool:
    return M > 0 and N % 128 == 0 and K % 512 == 0


def dgrad_split_k(M, N, K) -> int:
    """Split-K factor for a dgrad NT GEMM C[M,N] = A[M,K] . B[N,K]^T (1 = off).

    PRIMUS_TURBO_WQB_SPLITK=<s> (default 1): split the K loop of deep-K / narrow-N dgrads into ``s``
    fp32 partials summed in a fixed order (deterministic). Target is the DSv4 wq_b dgrad
    dX[4096,1024] = dY[4096,32768] . W (K/N = 32): K-serial (~0.55 us per 128-K tile, 133 us at s=1 on
    128x128 tiles). Applied only when K >= 16 * N, K % (512 * s) == 0 and K / s >= 1024, so the other
    dense dgrads (K/N <= 2 in DSv4) are untouched."""
    try:
        s = int(os.environ.get("PRIMUS_TURBO_WQB_SPLITK", "1") or "1")
    except ValueError:
        return 1
    if s <= 1 or K < 16 * N or K % (512 * s) or K // s < 1024:
        return 1
    return s


def accum_supported(main_grad, N, K) -> bool:
    """Whether gemm_nt can write/accumulate the [N, K] wgrad straight into ``main_grad``."""
    return (
        isinstance(main_grad, torch.Tensor)
        and main_grad.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and tuple(main_grad.shape) == (N, K)
        and main_grad.is_contiguous()
        and main_grad.data_ptr() % 16 == 0
    )


def gemm_nt(a, b, ea, eb, out_dtype=torch.bfloat16, out=None, accumulate=False, split_k=1):
    """C[M,N] = A[M,K] . B[N,K]^T with E8M0 scales ea [M,K/128] and eb [N/128,K/128] (block) or [N,K/128] (row).

    ``split_k`` > 1: fp32 partials per K slice, summed by gemm_g90 (not with ``accumulate``)."""
    M, K = a.shape
    N = b.shape[0]
    rowb = tuple(eb.shape) == (N, K // 128)
    kw = dict(_cfg(M, N, K, rowb))
    if accumulate:
        kw.pop("wscale", None)
        kw.pop("ws_rows", None)
        kw.update(tile=(128, 128), lead=4)
        if not rowb:
            kw["wscale"] = True
    if split_k > 1:
        assert not accumulate, "split_k and accumulate are exclusive"
        kw["split_k"] = split_k
    return gemm_g90(a, b, out=out, scale_a=ea, scale_b=eb, out_dtype=out_dtype, accumulate=accumulate, **kw)
