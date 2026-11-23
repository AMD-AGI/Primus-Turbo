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

"""Dispatch of the 128-block (UE8M0) FP8 dense GEMMs to the gfx1250 g90 FlyDSL kernel.

Used by FP8GemmBlockFunction (trans_a=False, trans_b=True, 2D operands) when PRIMUS_TURBO_FP8_BLOCK_KERNEL is not
"off", for the shapes and passes the dsv4up Gluon kernel does not take. All three passes become NT GEMMs:
  fwd   Y[M,N]  = X[M,K]  . W[N,K]^T       A 1x128 scales, B 128x128 block scales
  dgrad dX[M,K] = dY[M,N] . (W^T)[K,N]^T   W^T is an FP8 copy of W (128x128 blocks are transpose-invariant)
  wgrad dW[N,K] = dY^T[N,M] . X^T[K,M]^T  token-major copies, 1x128 along tokens on both operands
Config choice: 256x256 (2x2 waves, 3 V_WMMA_SCALE rows + VALU rescale) for M, N >= 2048; else 128x128
all-V_WMMA_SCALE; per-row B scales (wgrad) use 128x128.
"""

import torch

import primus_turbo.common.envs as envs
from primus_turbo.flydsl.gemm.gemm_fp8blk_kernel import gemm_g90
from primus_turbo.pytorch.core.utils import is_gfx1250

_CFG_256 = dict(tile=(256, 256), lead=24, ws_rows=3)
_CFG_128 = dict(tile=(128, 128), lead=4, wscale=True)
_CFG_128_ROWB = dict(tile=(128, 128), lead=4)


def enabled() -> bool:
    # dsv4up rides on the g90 path (primus_turbo/pytorch/kernels/gemm/gemm_fp8_gluon_gfx1250_impl.py swaps the GEMM per supported pass)
    return envs.PRIMUS_TURBO_FP8_BLOCK_KERNEL != "off" and is_gfx1250()


def _cfg(M, N, K, rowb):
    if rowb:
        return _CFG_128_ROWB
    if M >= 2048 and N >= 2048 and N % 256 == 0:
        return _CFG_256
    return _CFG_128


def supported(M, N, K) -> bool:
    return M > 0 and N % 128 == 0 and K % 512 == 0


def accum_supported(main_grad, N, K) -> bool:
    """Whether gemm_nt can write/accumulate the [N, K] wgrad straight into ``main_grad``."""
    return (
        isinstance(main_grad, torch.Tensor)
        and main_grad.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and tuple(main_grad.shape) == (N, K)
        and main_grad.is_contiguous()
        and main_grad.data_ptr() % 16 == 0
    )


def gemm_nt(a, b, ea, eb, out_dtype=torch.bfloat16, out=None, accumulate=False):
    """C[M,N] = A[M,K] . B[N,K]^T with E8M0 scales ea [M,K/128] and eb [N/128,K/128] (block) or [N,K/128] (row)."""
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
    return gemm_g90(a, b, out=out, scale_a=ea, scale_b=eb, out_dtype=out_dtype, accumulate=accumulate, **kw)
