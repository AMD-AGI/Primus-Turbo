###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Visibility metadata for the FlyDSL RL attention forward, in one launch (``rl_metadata_kernel``).

Per call it writes what the kernel needs: the key starts (exclusive cumulative sum of the key
counts, for a paged cache), each query token's [lo, hi] interval (global key indices) of
bottom-right causal attention narrowed by a left window when no mask is given, and, at the first
token of every group of ``TOKENS`` consecutive tokens of a sequence, the group's
(min lo, max lo, min hi, max hi). Rows that see no key are left out of min lo / max hi and force
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
def rl_metadata_kernel(
    cu_q_ptr,
    keys_ptr,
    lo_in_ptr,
    hi_in_ptr,
    starts_ptr,
    lo_ptr,
    hi_ptr,
    group_ptr,
    num_seqs,
    num_tokens,
    KEYS_ARE_LENS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    TOKENS: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
):
    """``keys_ptr`` is cu_seqlens_k [num_seqs + 1], or with KEYS_ARE_LENS the per-sequence key counts
    [num_seqs], whose exclusive cumulative sum goes to ``starts_ptr``. With HAS_MASK the intervals
    are read from ``lo_in`` / ``hi_in``; otherwise they are computed and written to ``lo`` / ``hi``."""
    pid = tl.program_id(0)
    seq_offs = tl.arange(0, SEQ_BLOCK)
    if KEYS_ARE_LENS:
        if pid <= num_seqs:
            lens = tl.load(keys_ptr + seq_offs, mask=seq_offs < pid, other=0)
            tl.store(starts_ptr + pid, tl.sum(lens, axis=0))
    if pid < num_tokens:
        s = _seq_of(cu_q_ptr, pid, num_seqs)
        q0 = tl.load(cu_q_ptr + s)
        q1 = tl.load(cu_q_ptr + s + 1)
        if KEYS_ARE_LENS:
            k0 = tl.sum(tl.load(keys_ptr + seq_offs, mask=seq_offs < s, other=0), axis=0)
            k_len = tl.load(keys_ptr + s)
        else:
            k0 = tl.load(keys_ptr + s)
            k_len = tl.load(keys_ptr + s + 1) - k0
        shift = k0 + k_len - (q1 - q0) - q0
        if not HAS_MASK:
            hi = shift + pid
            lo = k0
            if WINDOW_LEFT >= 0:
                lo = tl.maximum(lo, hi - WINDOW_LEFT)
            tl.store(lo_ptr + pid, lo)
            tl.store(hi_ptr + pid, hi)
        if (pid - q0) % TOKENS == 0:
            rows = pid + tl.arange(0, TOKENS)
            valid = rows < q1
            if HAS_MASK:
                row_lo = tl.load(lo_in_ptr + rows, mask=valid, other=1)
                row_hi = tl.load(hi_in_ptr + rows, mask=valid, other=0)
            else:
                row_hi = shift + rows
                row_lo = tl.zeros_like(row_hi) + k0
                if WINDOW_LEFT >= 0:
                    row_lo = tl.maximum(row_lo, row_hi - WINDOW_LEFT)
            seen = valid & (row_lo <= row_hi)
            all_seen = tl.sum(tl.where(valid & ~seen, 1, 0), axis=0) == 0
            min_lo = tl.min(tl.where(seen, row_lo, _VIS_EMPTY), axis=0)
            max_hi = tl.max(tl.where(seen, row_hi, -1), axis=0)
            max_lo = tl.where(all_seen, tl.max(tl.where(valid, row_lo, -1), axis=0), _VIS_EMPTY)
            min_hi = tl.where(all_seen, tl.min(tl.where(valid, row_hi, _VIS_EMPTY), axis=0), -1)
            base = group_ptr + pid * 4
            tl.store(base + 0, min_lo)
            tl.store(base + 1, max_lo)
            tl.store(base + 2, min_hi)
            tl.store(base + 3, max_hi)
