###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Choose how to run a mask, separately for the forward and the backward.

A mask can be executed several ways that give identical results:

* the kernel's dedicated causal / sliding-window path (built-in masks, equal lengths);
* the block-sparse path at any block size -- partial blocks evaluate ``mask_mod`` per
  element, so every granularity is exact.

Which is fastest depends on the pass. The forward favours big tiles; the backward's
tiles are pinned to the block size and favour smaller ones. On MI300X (8K tokens,
16/2 heads, head dim 128) a band mask's backward took 10.8 ms at block 128 and 5.1 ms at
block 64, while its forward was fastest at 128. Since any correct forward produces the
same output and LSE, the two passes are chosen independently.
"""

import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

from primus_turbo.pytorch.kernels.flex_attention.flex_attention_block_mask_utils import (
    BlockPlan,
)

__all__ = ["AUTOTUNE_BLOCK_SIZES", "MaskPath", "default_paths", "fastest"]

AUTOTUNE_BLOCK_SIZES = (64, 128)
# Untuned backward: the largest block size that did not lose to a smaller one.
_DEFAULT_BWD_MAX_BLOCK = 64


@dataclass(frozen=True)
class MaskPath:
    """One way to run a mask: kernel causal/window flags, or the block-sparse path at
    ``block_size``. Decisions are cached without the plan (``plan`` is None); it is bound
    to the call's mask just before launch, so only the chosen block size is built."""

    name: str
    causal: bool = False
    window: Tuple[int, int] = (-1, -1)
    mask_mod: object = None
    block_size: Optional[int] = None
    plan: Optional[BlockPlan] = None


NO_MASK = MaskPath("none")


def default_paths(candidates: Sequence[MaskPath], block_size: int) -> Tuple[MaskPath, MaskPath]:
    """Untuned choice: the fast path if there is one, else block-sparse at ``block_size``
    for the forward and at most ``_DEFAULT_BWD_MAX_BLOCK`` for the backward."""
    fast = [c for c in candidates if c.block_size is None]
    if fast:
        return fast[0], fast[0]
    by_size = {c.block_size: c for c in candidates}
    fwd = by_size.get(block_size, candidates[0])
    bwd_size = max((s for s in by_size if s <= min(block_size, _DEFAULT_BWD_MAX_BLOCK)), default=block_size)
    return fwd, by_size.get(bwd_size, fwd)


def _median_ms(fn: Callable[[], None], warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def fastest(
    candidates: Sequence[MaskPath], run: Callable[[MaskPath], None], warmup: int = 3, reps: int = 10
) -> MaskPath:
    """Time ``run(candidate)`` for each candidate and return the fastest. A candidate the
    kernel cannot launch (e.g. a block size over the LDS budget) is skipped."""
    best, best_ms, errors = None, float("inf"), []
    for cand in candidates:
        try:
            ms = _median_ms(lambda cand=cand: run(cand), warmup, reps)
        except Exception as exc:  # noqa: BLE001 - an unlaunchable candidate is just not chosen
            errors.append(f"{cand.name}: {type(exc).__name__}: {exc}")
            continue
        if ms < best_ms:
            best, best_ms = cand, ms
    if best is None:
        raise RuntimeError("no flex-attention execution path could run:\n  " + "\n  ".join(errors))
    return best
