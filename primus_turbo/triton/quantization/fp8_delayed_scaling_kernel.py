###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused delayed-scaling FP8 scale update.

One GPU dispatch over every (module, track) pair replaces a Python loop that
rolls the amax history and recomputes the FP8 scale per module. ``track``
indexes the three tensors each linear layer quantizes: 0 = input, 1 = weight,
2 = grad.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_delayed_scale_update_kernel(
    amax_history_ptr,
    staged_amaxes_ptr,
    scales_ptr,
    fp8_maxes_ptr,
    history_idx,
    N: tl.constexpr,
    H: tl.constexpr,
    use_max_algo: tl.constexpr,
    BLOCK_H: tl.constexpr,
    FP32_MAX: tl.constexpr,
    FILTER_ZEROS: tl.constexpr = False,
):
    track = tl.program_id(1)
    mod = tl.program_id(0)

    base = track * N * H + mod * H

    new_amax = tl.load(staged_amaxes_ptr + track * N + mod)
    tl.store(amax_history_ptr + base + history_idx, new_amax)

    if use_max_algo:
        amax = tl.zeros([], dtype=tl.float32)
        has_nonzero = tl.zeros([], dtype=tl.int32)
        for off in range(0, H, BLOCK_H):
            h_idx = off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            vals = tl.load(amax_history_ptr + base + h_idx, mask=mask, other=0.0)
            if FILTER_ZEROS:
                nonzero_mask = vals > 0.0
                has_nonzero = has_nonzero | tl.sum(nonzero_mask.to(tl.int32), axis=0)
                filtered = tl.where(nonzero_mask, vals, 0.0)
                amax = tl.maximum(amax, tl.max(filtered, axis=0))
            else:
                amax = tl.maximum(amax, tl.max(vals, axis=0))
    else:
        amax = new_amax
        has_nonzero = tl.where(new_amax > 0.0, 1, 0).to(tl.int32)

    fp8_max = tl.load(fp8_maxes_ptr + track)
    old_scale = tl.load(scales_ptr + track * N + mod)
    sf = fp8_max / tl.maximum(amax, 1e-12)
    if FILTER_ZEROS:
        sf = tl.where(has_nonzero > 0, sf, old_scale)
    else:
        sf = tl.where(amax > 0.0, sf, old_scale)
    sf = tl.where(amax == amax, sf, old_scale)
    sf = tl.minimum(sf, FP32_MAX)
    tl.store(scales_ptr + track * N + mod, sf)


def fused_delayed_scale_update(
    amax_history: torch.Tensor,
    staged_amaxes: torch.Tensor,
    scales: torch.Tensor,
    fp8_maxes: torch.Tensor,
    history_idx: int,
    use_max_algo: bool,
    filter_zeros: bool = False,
) -> None:
    """Roll the amax history and recompute the FP8 scales in place.

    Args:
        amax_history: ``[3, N, H]`` fp32 amax history; slot ``history_idx`` is overwritten.
        staged_amaxes: ``[3, N]`` fp32 amaxes observed this step.
        scales: ``[3, N]`` fp32 scales, updated in place.
        fp8_maxes: ``[3]`` fp32 max representable value per track.
        history_idx: history slot to write this step's amax into.
        use_max_algo: reduce the history with ``max`` (else use the newest amax).
        filter_zeros: ignore zero entries of the history when reducing with ``max``.
    """
    _, N, H = amax_history.shape
    BLOCK_H = triton.next_power_of_2(H) if H <= 1024 else 1024
    _fused_delayed_scale_update_kernel[(N, 3)](
        amax_history,
        staged_amaxes,
        scales,
        fp8_maxes,
        history_idx,
        N=N,
        H=H,
        use_max_algo=use_max_algo,
        BLOCK_H=BLOCK_H,
        FP32_MAX=torch.finfo(torch.float32).max,
        FILTER_ZEROS=filter_zeros,
    )
