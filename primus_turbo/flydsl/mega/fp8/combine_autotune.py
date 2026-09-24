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

"""Online autotuner for the fp8 combine's comm/GEMM CU split.

The combine runs ``max(PUSH, GEMM) + tail`` behind a scoreboard, so ``num_combine_cu`` -- how many
of the grid's blocks run the cross-rank PUSH -- sets where that balance lands. The optimum moves
with the expert shape: DeepSeek-V3 (H=7168, I=2048) wants 32, while Qwen3-30B-A3B (H=2048, I=768)
wants 80 on the L1 dgrad and 96 on the L2 forward. Shipping one constant costs the smaller shape
~8% of its step time, which is what this module exists to stop.

Why not ``flydsl.autotune``
---------------------------
The bf16 combine tunes the same knob with the stock ``@autotune`` decorator, which ``do_bench``\\ es
each candidate: warmup plus ``rep`` **extra** launches of the kernel. That is safe for a pure GEMM
and is not safe here. This kernel is a collective: it pushes into the shared symmetric buffer and
advances ``_combine_parity`` / the combine and reduce epoch expectations on device. Replaying it
off the real call sequence moves that state without a matching dispatch, and the failure mode is
not an exception -- it is a peer whose reduce waits on an arrival count that no longer matches, or
an output assembled from another epoch's payload.

So this tunes **online** instead: every candidate is measured on a real call that the model was
going to make anyway. One launch per call, exactly as before, with a CUDA event pair around it.

Reading the elapsed time would normally mean a sync, so the pair is parked and drained on a later
call, by which point the events have long completed -- the measurement costs no stall.

Lockstep
--------
Every rank tunes the same key in the same order (the shapes are static and all MoE layers of a
model share one key), so the candidate sequence is identical everywhere without coordination.
Only the *winner* could differ, since each rank times its own launches. ``group`` makes that
choice unanimous with one small all-reduce, once per key. It is optional: the CU count only
partitions this rank's own push work -- a receiver's expectations come from the routing, not from
how many blocks the sender split its push across -- so a divergent pick is a performance wart
rather than a correctness bug.
"""

import os
from collections import defaultdict

import torch

__all__ = ["choose_combine_cu", "observe_combine_cu", "autotune_enabled", "tuning_report"]


def _env_int_list(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(v) for v in raw.replace(",", " ").split())


# Spans both regimes seen so far: 16-32 is where DSv3-sized experts land, 80-128 where Qwen3-sized
# ones do. Each candidate costs one extra kernel compile per direction on first use (``_compile``
# is lru_cached at 64 entries, so 7 x 2 fits alongside everything else).
_CANDIDATES = _env_int_list("PT_MEGA_FP8_COMBINE_CU_CANDIDATES", (16, 32, 48, 64, 80, 96, 128))

# Samples per candidate. The first use of a candidate pays its kernel compile, so that sample is a
# wild outlier -- taking the min over repeats discards it without a separate warmup phase.
_REPS = int(os.environ.get("PT_MEGA_FP8_COMBINE_TUNE_REPS", "3"))

_ENABLED = os.environ.get("PT_MEGA_FP8_COMBINE_AUTOTUNE", "1") not in ("0", "false", "False")

# One line per shape when the split locks in. It is the only evidence that tuning ran and what
# it chose, and a run makes at most a couple of these.
_VERBOSE = os.environ.get("PT_MEGA_FP8_COMBINE_AUTOTUNE_VERBOSE", "1") not in ("0", "false", "False")


def autotune_enabled() -> bool:
    return _ENABLED


class _KeyState:
    """Per-shape tuning state: a round-robin schedule, its samples, and the locked-in winner."""

    __slots__ = ("schedule", "next_idx", "samples", "pending", "winner")

    def __init__(self):
        # Round-robin rather than all repeats of one candidate back to back: a slow stretch (a
        # straggler peer, a clock dip) then lands on every candidate instead of condemning one.
        self.schedule = [cu for _ in range(_REPS) for cu in _CANDIDATES]
        self.next_idx = 0
        self.samples = defaultdict(list)
        self.pending = []  # (cu, start_event, end_event)
        self.winner = None


_STATE: dict = {}


def _drain(st, *, block=False):
    """Fold completed event pairs into ``samples``; keep the rest for a later call."""
    still_pending = []
    for cu, ev_start, ev_end in st.pending:
        if not block and not ev_end.query():
            still_pending.append((cu, ev_start, ev_end))
            continue
        ev_end.synchronize()
        st.samples[cu].append(ev_start.elapsed_time(ev_end))
    st.pending = still_pending


def _decide(key, st, group):
    """Lock the winner: lowest per-candidate min, agreed across ranks when a group is given."""
    cands = list(_CANDIDATES)
    local = torch.tensor(
        [min(st.samples[cu]) if st.samples[cu] else float("inf") for cu in cands],
        dtype=torch.float64,
    )
    if group is not None and torch.distributed.is_initialized():
        # SUM, not MIN: the split that is best on average across ranks beats one that is best on
        # the luckiest rank. inf stays inf, so a candidate no rank measured is never chosen.
        dev = local.to(torch.cuda.current_device())
        torch.distributed.all_reduce(dev, op=torch.distributed.ReduceOp.SUM, group=group)
        local = dev.cpu()
    best = int(torch.argmin(local).item())
    st.winner = cands[best]
    if _VERBOSE and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        table = " ".join(f"{cu}:{min(st.samples[cu]):.3f}" if st.samples[cu] else f"{cu}:-" for cu in cands)
        print(
            f"[mega fp8] combine CU tuned -> {st.winner} for key={key} ms(min per cu): {table}",
            flush=True,
        )
    return st.winner


def choose_combine_cu(key, *, group=None, default=None):
    """The CU split to use for this call -- a candidate while tuning, the winner once locked.

    ``default`` is returned unchanged when autotuning is off, which is how an explicit pin (the
    ``PT_MEGA_FP8_L{1,2}_COMBINE_CU`` env overrides) bypasses this entirely.
    """
    if not _ENABLED:
        return default
    st = _STATE.get(key)
    if st is None:
        _STATE[key] = st = _KeyState()
    if st.winner is not None:
        return st.winner

    _drain(st)
    if st.next_idx < len(st.schedule):
        return st.schedule[st.next_idx]
    # Schedule exhausted -- wait out the last few launches, then commit.
    _drain(st, block=True)
    return _decide(key, st, group)


def observe_combine_cu(key, cu, ev_start, ev_end):
    """Hand back the event pair that bracketed the launch ``choose_combine_cu`` picked."""
    if not _ENABLED:
        return
    st = _STATE.get(key)
    if st is None or st.winner is not None:
        return
    st.pending.append((cu, ev_start, ev_end))
    st.next_idx += 1


def tuning_report():
    """``{key: (winner, {cu: min_ms})}`` for every key seen -- for logging and for the benches."""
    return {
        k: (st.winner, {cu: (min(v) if v else None) for cu, v in st.samples.items()})
        for k, st in _STATE.items()
    }
