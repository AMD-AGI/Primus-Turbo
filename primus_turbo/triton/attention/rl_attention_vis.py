###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Visibility metadata for the FlyDSL RL attention forward.

``causal_intervals_kernel`` writes, per query token, the [lo, hi] interval (global key indices) of
bottom-right causal attention narrowed by a left window. ``vis_group_bounds_kernel`` writes, at the
first token of every group of ``TOKENS`` consecutive tokens of a sequence, the group's
(min lo, max lo, min hi, max hi); rows that see no key are left out of min lo / max hi and force
max lo = VIS_EMPTY, min hi = -1, so a tile is never treated as fully visible to them.
"""

import triton
import triton.language as tl

VIS_EMPTY = 1 << 30  # must equal rl_attn_helper.VIS_EMPTY
_VIS_EMPTY = tl.constexpr(VIS_EMPTY)


@triton.jit
def _seq_of(cu_ptr, token, num_seqs):
    """Index of the sequence holding ``token``: the last s with cu[s] <= token (binary search)."""
    lo = 0
    hi = num_seqs
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if tl.load(cu_ptr + mid) <= token:
            lo = mid
        else:
            hi = mid
    return lo


@triton.jit
def causal_intervals_kernel(cu_q_ptr, key_starts_ptr, lo_ptr, hi_ptr, num_seqs, WINDOW_LEFT: tl.constexpr):
    token = tl.program_id(0)
    s = _seq_of(cu_q_ptr, token, num_seqs)
    q0 = tl.load(cu_q_ptr + s)
    q_len = tl.load(cu_q_ptr + s + 1) - q0
    k0 = tl.load(key_starts_ptr + s)
    k_len = tl.load(key_starts_ptr + s + 1) - k0
    hi = k0 + k_len - q_len + (token - q0)
    lo = k0
    if WINDOW_LEFT >= 0:
        lo = tl.maximum(lo, hi - WINDOW_LEFT)
    tl.store(lo_ptr + token, lo)
    tl.store(hi_ptr + token, hi)


@triton.jit
def vis_group_bounds_kernel(cu_q_ptr, lo_ptr, hi_ptr, out_ptr, num_seqs, TOKENS: tl.constexpr):
    token = tl.program_id(0)
    s = _seq_of(cu_q_ptr, token, num_seqs)
    q0 = tl.load(cu_q_ptr + s)
    q1 = tl.load(cu_q_ptr + s + 1)
    if (token - q0) % TOKENS == 0:
        rows = token + tl.arange(0, TOKENS)
        valid = rows < q1
        lo = tl.load(lo_ptr + rows, mask=valid, other=1)
        hi = tl.load(hi_ptr + rows, mask=valid, other=0)
        seen = valid & (lo <= hi)
        all_seen = tl.sum(tl.where(valid & ~seen, 1, 0), axis=0) == 0
        min_lo = tl.min(tl.where(seen, lo, _VIS_EMPTY), axis=0)
        max_hi = tl.max(tl.where(seen, hi, -1), axis=0)
        max_lo = tl.where(all_seen, tl.max(tl.where(valid, lo, -1), axis=0), _VIS_EMPTY)
        min_hi = tl.where(all_seen, tl.min(tl.where(valid, hi, _VIS_EMPTY), axis=0), -1)
        base = out_ptr + token * 4
        tl.store(base + 0, min_lo)
        tl.store(base + 1, max_lo)
        tl.store(base + 2, min_hi)
        tl.store(base + 3, max_hi)
