###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Layout / alignment / padding helpers shared by the HipKittens backends.

HipKittens' BF16 and FP8 kernels accept three logical GEMM layouts:

  * ``rcr`` - A row-major,  B col-major  (forward NT)
  * ``rrr`` - A row-major,  B row-major  (grad-X NN)
  * ``crr`` - A col-major,  B row-major  (grad-W TN)

Both precisions require M and N to be multiples of ``BLOCK_SIZE = 256``.
BF16 needs ``K % 64 == 0`` (``K_STEP``); FP8 needs ``K % 128 == 0``
(``K_BLOCK``). BF16 additionally has minimum-N (rrr) and minimum-K (crr)
floors that reflect kernel-template constraints in HipKittens.
"""
from __future__ import annotations

from typing import Literal

DType = Literal["bf16", "fp8"]
Layout = Literal["rcr", "rrr", "crr"]

_K_ALIGN = {"bf16": 64, "fp8": 128}
_BF16_RRR_MIN_N = 4096
_BF16_CRR_MIN_K = 4096


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def layout_of(trans_a: bool, trans_b: bool) -> Layout | None:
    """Map (trans_a, trans_b) flags to one of HipKittens' supported layouts.

    Returns None for the unsupported (trans_a, trans_b)=(True, True) case.
    """
    if not trans_a and trans_b:
        return "rcr"
    if not trans_a and not trans_b:
        return "rrr"
    if trans_a and not trans_b:
        return "crr"
    return None


def aligned_for(m: int, n: int, k: int, dtype: DType) -> bool:
    """True when the unpadded shape is directly callable without padding."""
    return m % 256 == 0 and n % 256 == 0 and k % _K_ALIGN[dtype] == 0


def padded_shape(m: int, n: int, k: int, layout: Layout, dtype: DType) -> tuple[int, int, int]:
    """Pad (M, N, K) up to the smallest shape HipKittens can accept.

    For BF16 the rrr / crr layouts have additional minimum-dim constraints
    (the kernel templates were tuned for non-trivial inner shapes), which
    we encode here so the per-backend padding logic stays uniform.
    """
    m_pad = round_up(m, 256)
    n_pad = round_up(n, 256)
    k_pad = round_up(k, _K_ALIGN[dtype])
    if dtype == "bf16":
        if layout == "rrr":
            n_pad = max(n_pad, _BF16_RRR_MIN_N)
        elif layout == "crr":
            k_pad = max(k_pad, _BF16_CRR_MIN_K)
    return m_pad, n_pad, k_pad
