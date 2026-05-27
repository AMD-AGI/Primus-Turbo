###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Python launcher for the FlyDSL blockscale grouped GEMM.

Wraps compile + shuffle + launch, with a per-shape compile cache so the
JIT cost is paid once per (G, N, K, tile, dtype, num_sms) combo. The
persistent variant launches a fixed (num_sms, 1, 1) grid; per-tile dispatch
is derived from group_offs on-device, so no CPU sync is required.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch

import flydsl.compiler as flyc

from primus_turbo.flydsl.grouped_gemm.blockscale_grouped_gemm_persistent import (
    compile_blockscale_grouped_gemm_persistent,
)

__all__ = [
    "grouped_gemm_fp8_blockwise_flydsl_kernel",
    "shuffle_b_batched",
]


# Cache the compiled callable per shape. flyc.compile binds tensor pointers at
# JIT time but the returned handle accepts new pointers as long as dtypes /
# shapes match, so we compile once with the first set of inputs per shape and
# reuse it on subsequent calls.
_compiled_cache: dict = {}
_cu_num_cache: int | None = None


def _get_num_cus() -> int:
    global _cu_num_cache
    if _cu_num_cache is None:
        _cu_num_cache = torch.cuda.get_device_properties(0).multi_processor_count
    return _cu_num_cache


def _select_tile(M_g_max: int, N: int, K: int) -> Tuple[int, int, int]:
    """Lightweight tile picker; mirrors the heuristic in FlyDSL's blockscale test.

    tile_n=256 gives ~25% better TF than 128 (single WG covers 4 wide N tiles
    vs 2); falls back to 128 when N isn't divisible by 256.

    tile_k=256 wins +1-9% over 128 on all hot shapes (deeper K unroll moves
    more work per prologue, better LICM amortization). Requires K % 256 == 0
    (LFM2 1792 / Qwen3 1536 / DS 2048 all qualify).

    tile_m heuristic (empirical):
    - 128: N >= 7168 AND M_g <= 4096 — DS-like fat-N + medium-M; less LICM
           cross-iter overhead amortizes the larger per-WG work. Measured
           +3-5% TF on DS B=4/16 Mt=8192.
    - 64:  default. Wins on small/medium N and very large M.
    """
    tile_n = 256 if (N % 256 == 0) else 128
    if (N >= 7168) and (M_g_max % 128 == 0) and (M_g_max <= 4096):
        tile_m = 128
    else:
        tile_m = 64
    # tile_k=256 only when (a) K divides cleanly and (b) tile_m=64 (tile_m=128
    # + tile_k=256 makes per-WG work 4x and may not fit register/LDS budget).
    tile_k = 256 if (K % 256 == 0 and tile_m == 64) else 128
    return tile_m, tile_n, tile_k


def _select_super_m(N: int, Mt: int) -> int:
    """Empirically-tuned M-super-grouping size.

    The trade-off: larger super_m gives a single WG more A-row reuse across n
    columns (good for small N where the per-super B working set fits in L2),
    but a super of `super_m × tiles_n` B tiles blows L2 when N is large. The
    benefit only shows up when total_tiles >> num_sms so the WGs actually loop
    many tiles. Below that, super_m=1 (no swizzle) costs nothing extra.
    """
    if Mt < 16384:
        return 1
    if N <= 2048:
        return 8
    if N <= 4096:
        return 4
    return 1


def shuffle_b_batched(b: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Batched preshuffle of a [G, N, K] weight; ~3x faster than per-group launches."""
    G, N, K = b.shape
    IN, IK = layout
    BK = IK * 2
    K_inner = 16 // b.element_size()
    BN = IN
    assert N % BN == 0 and K % BK == 0
    v = b.view(G, N // BN, BN, K // BK, BK // K_inner, K_inner)
    return v.permute(0, 1, 3, 4, 2, 5).contiguous().view(G, N, K)


def grouped_gemm_fp8_blockwise_flydsl_kernel(
    a_fp8: torch.Tensor,       # [Mt, K] fp8 (row-quant)
    b_fp8: torch.Tensor,       # [G, N, K] fp8 (weight-quant, NOT shuffled)
    a_scales: torch.Tensor,    # [K//128, Mt] fp32 (pshuffled)
    b_scales: torch.Tensor,    # [G, N//128, K//128] fp32
    group_offs: torch.Tensor,  # [G+1] int64
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent + CPU-sync-free grouped blockwise FP8 GEMM via FlyDSL.

    Single launch covers G groups with grid=(num_sms, 1, 1); the kernel walks
    the tile space itself, deriving per-tile (group, m_tile, n_tile) from
    group_offs on-device. Caller does NOT need to compute max(group_lens)."""
    assert a_fp8.ndim == 2 and b_fp8.ndim == 3
    assert a_scales.ndim == 2 and b_scales.ndim == 3
    assert group_offs.dtype == torch.int64
    assert out_dtype in (torch.bfloat16, torch.float16)

    Mt, K = a_fp8.shape
    G, N, K_b = b_fp8.shape
    assert K == K_b

    out_dtype_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    # Tile picker no longer depends on per-group M (kernel handles tile-space
    # walk internally); pass Mt as a coarse hint for any future heuristics.
    # Estimate per-group M for tile_m heuristic. group_offs is on-device, so
    # use the average Mt/G as a proxy — for hot balanced shapes this matches
    # the actual per-group M exactly.
    m_g_est = Mt // max(G, 1)
    tile_m, tile_n, tile_k = _select_tile(m_g_est, N, K)
    num_sms = _get_num_cus()
    super_m = _select_super_m(N, Mt)

    # waves_per_eu trade-off (locked at None after experiments):
    # - None: LLVM picks 280 VGPR / 1 wave-SIMD / no spill (current stable)
    # - 2:    Forces ≤256 VGPR for 2 wave-SIMD; LLVM spills 6 lane-dependent
    #         vgpr to scratch + hot loop reloads each iter. Measured -5% TF.
    # The 6 spilled vgpr are thread_id-derived address mids that LLVM's
    # LICM hoists to outer setup; the only way to truly free them is to
    # drop the persistent loop entirely (vetoed).
    waves_per_eu = None
    # Toggle: skip host-side shuffle_b_batched if kernel reads raw [G,N,K] B.
    # Measured (LFM2 B=4 M=2048): preshuffle kernel 0.119ms vs no-shuffle kernel
    # 0.156ms (+35%). Cache miss cost (16 lanes within klane span 16*K bytes
    # apart) exceeds the shuffle call savings (~23us) on every hot shape tested.
    # Default kept at preshuffle; env var only for special cases (e.g. B reused
    # across many GEMMs where shuffle amortizes).
    use_preshuffle_b = bool(int(os.environ.get("PRIMUS_TURBO_FLYDSL_PRESHUFFLE", "1")))
    key = (G, N, K, tile_m, tile_n, tile_k, out_dtype_str, Mt, num_sms, super_m, use_preshuffle_b)
    b_for_kernel = shuffle_b_batched(b_fp8) if use_preshuffle_b else b_fp8
    out = torch.empty((Mt, N), dtype=out_dtype, device=a_fp8.device)
    a_sc_flat = a_scales.view(-1)
    b_sc_flat = b_scales.view(-1)
    stream = torch.cuda.current_stream()

    compiled = _compiled_cache.get(key)
    if compiled is None:
        exe = compile_blockscale_grouped_gemm_persistent(
            Mt=Mt, N=N, K=K, G=G,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            num_sms=num_sms, super_m=super_m,
            waves_per_eu=waves_per_eu,
            use_preshuffle_b=use_preshuffle_b,
            # use_cshuffle_epilog=True tested: VGPR 280→294 (+14), LDS 16→32 KB,
            # 0 spill but kernel TF unchanged (avg 1.08x). Direct epilog wins
            # via lower LDS pressure.
            use_cshuffle_epilog=False,
            # use_async_copy=True tested with waves=None/1/2: VGPR 280→352 (sync→async);
            # waves=2 forces ≤256 → spill 13 (sync) or 59 (async), -5%/-23% TF.
            # waves=None (LLVM-chosen 1 wave) is global optimum; async hurts.
            use_async_copy=False,
            scale_block_k=128, out_dtype=out_dtype_str,
        )
        compiled = flyc.compile(
            exe, out, a_fp8, b_for_kernel, a_sc_flat, b_sc_flat,
            group_offs, Mt, N, stream,
        )
        _compiled_cache[key] = compiled

    compiled(out, a_fp8, b_for_kernel, a_sc_flat, b_sc_flat,
             group_offs, Mt, N, stream)
    return out
