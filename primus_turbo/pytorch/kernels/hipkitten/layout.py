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
The K alignment is **128 for both** precisions — see below.

BF16 K-alignment (subtle):
  The binding header advertises ``K_STEP = 64`` (the per-iteration K stride
  inside the inner MFMA loop), but the kernel's *dynamic fallback*
  (``ki = K / K_STEP`` when ``ki`` is not in the precompiled-template list)
  empirically returns wrong results when ``ki`` is **odd** — i.e. when
  ``K % 128 != 0``. A K-sweep at M=N=256 over ``K ∈ [64, 4096]`` shows
  every odd ``ki`` produces SNR < 18 dB and every even ``ki`` produces
  SNR ≈ 49.6 dB (probe in this round's commit). Practically the kernel
  needs *two* K_STEP iterations per accumulator pass, so we treat the
  effective K-alignment as **128** here. All the LLM shapes in
  ``benchmark/ops/config.py`` (Llama / DeepSeek / gpt_oss) have K that is
  already a multiple of 128, so this stricter alignment never rejects a
  shape that previously ran correctly. The notable beneficiary is the
  gpt_oss_20B grouped GEMM family with ``K = 2880`` (= 22.5 × 128): the
  old 64-byte alignment accepted these into the BF16 grouped path and
  silently produced wrong outputs; with the corrected 128-alignment the
  ``execute`` path either pads ``K`` up to 3072 or routes to a per-group
  fallback, both of which produce SNR > 30 dB results.

FP8 K-alignment is naturally 128 from the binding's ``K_BLOCK``.

BF16 additionally has minimum-N (rrr) and minimum-K (crr) floors that
reflect kernel-template constraints in HipKittens.
"""
from __future__ import annotations

from typing import Literal

DType = Literal["bf16", "fp8"]
Layout = Literal["rcr", "rrr", "crr"]

_K_ALIGN = {"bf16": 128, "fp8": 128}
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
