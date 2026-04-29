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
The K alignment is **128 for both** precisions (two K_STEP=64 sub-tiles
per accumulator pass for BF16; the FP8 binding's K_BLOCK).

State of host pad after Phase 4:

  * BF16 dense — kernel handles any (M, N, K) natively after the
    round-1 ``feat(bf16-dense): native non-aligned M/N/K`` commit; the
    Primus path retains the alignment gate for the smoke-test
    rejection contract but no longer pads.

  * BF16 grouped — kernel handles any non-aligned N / K natively
    (round-3 ``feat(bf16-grouped): native non-aligned N/K`` + round-7
    ``perf(tail-kernel): vec4 BF16 K-loop loads in RCR tail`` + round-9
    ``feat(bf16-grouped): LDS-staged K-tail kernel`` cooperative LDS
    staging for the dominant K-tail interior cells). After round-9 LDS
    the BF16 grouped tail runs at ~100-250 TF on gpt_oss K-tail
    (1.4-1.7× over scalar vec4); the host-pad-then-main-kernel path
    still runs at 600-900 TF on the same shapes — the gap is ~3-6×
    because the LDS tail uses scalar fp32 MAC (no MFMA) and is HBM-
    bandwidth-bound at ~6.5 TB/s on a 16×16×K_REM=64 cooperative load
    pattern. The Primus K-pad fast path therefore stays in place
    until either the tail kernel grows MFMA accumulation
    (16×16×16 BF16 over the K_REM strip, 4 instructions per tile) or
    the main kernel grows native K-tail handling (single-K_STEP epilog
    when ``ki`` is odd). Per round-9 ``/tmp/bench_drop_kpad.py``,
    dropping host K-pad on gpt_oss-Down-B4-M2048 (M=8192) goes from
    226 µs (601 TF, P1) to 583 µs (233 TF, P2) — a 0.39× regression
    from missing the main-kernel two-tile schedule. The LDS kernel is
    plumbed via a new ``m_per_group`` arg on ``grouped_{rcr,rrr,crr}``
    bindings (default 0 = scalar tail); Primus continues to call the
    binding without ``m_per_group`` so the metric path is unchanged.

  * BF16 variable-K dB CRR — per-group ``dense_run`` loop dispatches
    the BF16 dense kernel directly, which is native non-aligned, so
    the host pad is dropped from this path.

  * FP8 dense — kernel has had native non-aligned support since
    ``kernel_fp8_layouts.cpp::dispatch<L>`` shipped; the Primus path
    drops the alignment gate this round.

  * FP8 grouped — binding's ``dispatch_grouped_rcr`` has had a
    fast/tail splitter since round 6 ``feat(fp8-grouped): native
    non-aligned N/K``. Round-7 ``perf(tail-kernel): vec8 FP8 K-loop
    loads`` lifted the FP8 grouped tail from ~22 TF to ~245 TF on
    gpt_oss shapes — but the host-pad-then-main-kernel path still
    runs at ~830-1280 TF (4-5× faster) because of the same
    no-LDS-reuse bandwidth bound on the tail kernel. The Primus
    K-pad path stays for the same reason as BF16 grouped above.
    Per round-8 bench: full Primus dispatch on gpt_oss-Down-B4-M2048
    is 261 µs (520 TF), of which ~64 µs is host-pad alloc/copy and
    ~99 µs is BF16→FP8 quant + custom-op + dispatch overhead.
"""
from __future__ import annotations

from typing import Literal

DType = Literal["bf16", "fp8"]
Layout = Literal["rcr", "rrr", "crr"]

_K_ALIGN = {"bf16": 128, "fp8": 128}


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
    """True when the unpadded shape lands on the kernel's tile multiple."""
    return m % 256 == 0 and n % 256 == 0 and k % _K_ALIGN[dtype] == 0


def padded_shape(m: int, n: int, k: int, layout: Layout, dtype: DType) -> tuple[int, int, int]:
    """Pad (M, N, K) up to the smallest shape HipKittens can accept.

    Phase 4: BF16 callers (dense + grouped + variable-K) no longer
    invoke this helper — the BF16 dense and grouped kernels handle
    arbitrary (M, N, K) natively via their scalar fp32 tail kernels
    after Phase 1 / Phase 3. FP8 grouped still calls this until the
    FP8 grouped binding picks up a fast/tail splitter (separate
    HK-side change).
    """
    m_pad = round_up(m, 256)
    n_pad = round_up(n, 256)
    k_pad = round_up(k, _K_ALIGN[dtype])
    return m_pad, n_pad, k_pad
