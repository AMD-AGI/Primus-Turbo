###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Schedule tuning for the batch-invariant RL attention.

``tune`` searches the kernel options that only schedule work (MODE, BLOCK_M, num_warps,
num_stages, waves_per_eu, SPLIT_SEGS, PACK) for one call and returns the fastest. A candidate is
kept only if its output is bitwise equal to the default schedule's, so a tuned schedule never
breaks rollout == training. The options that change bits (TILE_SIZE, SEGMENT_SIZE, BACKEND) are
held at the caller's values: they must match between the rollout and the training process.

The search is coordinate descent over one option at a time, repeated until nothing improves by
more than ``min_gain``; ``exhaustive=True`` times the full grid instead.
"""

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

__all__ = ["SCHEDULE_OPTIONS", "TuneResult", "format_options", "search_space", "tune"]

SCHEDULE_OPTIONS = ("MODE", "BLOCK_M", "num_warps", "num_stages", "waves_per_eu", "SPLIT_SEGS", "PACK")
_DEFAULTS = (0, "auto")


def search_space(backend: str, max_seqlen_q: int) -> Dict[str, list]:
    """Candidate values per schedule option for a call with this many query tokens per sequence,
    the default (0 / "auto") first."""
    decode = max_seqlen_q <= 16
    if backend.upper() == "FLYDSL":
        space = {"MODE": ["auto", "fused", "split"], "BLOCK_M": [0, 64, 128], "waves_per_eu": [0, 1, 2]}
        if decode:
            space.update(PACK=["auto", "off"], SPLIT_SEGS=[0, 1, 2, 4, 8, 16])
        return space
    return {
        "MODE": ["auto", "fused", "split"] if decode else ["auto", "fused"],
        "BLOCK_M": [0, 16, 32, 64] if decode else [0, 64, 128, 256],
        "num_warps": [0, 1, 2, 4, 8],
        "num_stages": [0, 1, 2, 3],
        "waves_per_eu": [0, 1, 2, 4],
    }


@dataclass
class TuneResult:
    options: dict  # the fastest schedule, non-default keys only
    default_us: float
    best_us: float
    tried: int
    rejected: List[Tuple[dict, str]] = field(default_factory=list)  # (options, why)

    @property
    def speedup(self) -> float:
        return self.default_us / self.best_us


def format_options(options: dict) -> str:
    """As PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS takes them: KEY=VALUE,KEY=VALUE."""
    return ",".join(f"{k}={v}" for k, v in options.items())


def _bench_us(fn: Callable[[], object]) -> float:
    import triton

    return triton.testing.do_bench(fn, warmup=10, rep=50) * 1e3


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(torch.uint8)


def tune(
    run: Callable[[dict], object],
    space: Dict[str, list],
    *,
    bench: Optional[Callable[[Callable[[], object]], float]] = None,
    min_gain: float = 0.03,
    max_rounds: int = 3,
    exhaustive: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> TuneResult:
    """Tune ``run(schedule_options) -> output`` (a tensor, or a tuple whose first item is compared)
    over ``space``. ``run`` adds the fixed options (BACKEND, TILE_SIZE, ...) itself."""
    unknown = set(space) - set(SCHEDULE_OPTIONS)
    if unknown:
        raise ValueError(f"only schedule options can be tuned, got {sorted(unknown)}")
    bench = bench or _bench_us
    reference = _bits(_first(run({}))).clone()
    times: Dict[tuple, Optional[float]] = {}
    rejected: List[Tuple[dict, str]] = []

    def measure(options: dict) -> Optional[float]:
        options = {k: v for k, v in options.items() if v not in _DEFAULTS}
        key = tuple(sorted(options.items()))
        if key not in times:
            try:
                if not torch.equal(_bits(_first(run(options))), reference):
                    times[key] = None
                    rejected.append((options, "bits differ from the default schedule"))
                else:
                    times[key] = bench(lambda: run(options))
            except Exception as e:  # an unsupported combination: skip it
                times[key] = None
                rejected.append((options, f"{type(e).__name__}: {e}"))
            if log is not None:
                t = times[key]
                log(f"  {format_options(options) or 'default':<60} {'-' if t is None else f'{t:.1f} us'}")
        return times[key]

    default_us = measure({})
    best, best_us = {}, default_us
    if exhaustive:
        names = list(space)
        for values in itertools.product(*(space[n] for n in names)):
            t = measure(dict(zip(names, values)))
            if t is not None and t < best_us * (1 - min_gain):
                best, best_us = {k: v for k, v in zip(names, values) if v not in _DEFAULTS}, t
    else:
        for _ in range(max_rounds):
            changed = False
            for name, values in space.items():
                for value in values:
                    candidate = {**best, name: value}
                    t = measure(candidate)
                    if t is not None and t < best_us * (1 - min_gain):
                        best = {k: v for k, v in candidate.items() if v not in _DEFAULTS}
                        best_us, changed = t, True
            if not changed:
                break
    return TuneResult(best, default_us, best_us, len(times), rejected)
