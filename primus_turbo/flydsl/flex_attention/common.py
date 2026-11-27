###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Device-side building blocks shared by the FlyDSL flex attention kernels, the
counterparts of FA4's ``seqlen_info``, ``block_info``, ``mask`` and ``softmax`` modules."""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import flydsl.expr as fx

LOG2E = 1.4426950408889634
LN2 = math.log(2.0)
# Masked scores get a finite floor so exp2 stays finite under fast math; a row with no
# kept key at all is detected by its running max staying at the floor.
NEG_BIG = -1.0e30


def _f32(x):
    return fx.Float32(x)


def _max(a, b):
    return (a > b).select(a, b)


def _min(a, b):
    return (a < b).select(a, b)


def _ceil_div(a, b: int):
    return (a + fx.Int32(b - 1)) // fx.Int32(b)


# --- SeqlenInfo -------------------------------------------------------------
# Per-sequence offsets and lengths, the counterpart of FA4's ``seqlen_info.SeqlenInfoQK``.


@dataclass
class SeqlenInfo:
    """Where this batch entry's queries / keys start in a packed (varlen) tensor, and how
    many there are. ``offset`` is ``seqlen_k - seqlen_q``: causal and local masks are
    bottom-right aligned, as in FA4."""

    offset_q: fx.Int32
    offset_k: fx.Int32
    seqlen_q: fx.Int32
    seqlen_k: fx.Int32

    @property
    def offset(self) -> fx.Int32:
        return self.seqlen_k - self.seqlen_q

    @staticmethod
    def create(
        batch,
        seqlen_q_static,
        seqlen_k_static,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
    ):
        """FA4's rules: the offset is ``cu_seqlens[batch]`` (0 when dense); the length is
        ``seqused[batch]`` when given, else ``cu_seqlens[batch + 1] - cu_seqlens[batch]``,
        else the static (dense) length. Absent tensors are ``None`` at trace time."""

        def side(static_len, cu, used):
            if cu is not None:
                offset = fx.Int32(cu[batch])
                length = fx.Int32(cu[batch + fx.Int32(1)]) - offset
            else:
                offset = fx.Int32(0)
                length = fx.Int32(static_len)
            if used is not None:
                length = fx.Int32(used[batch])
            return offset, length

        offset_q, seqlen_q = side(seqlen_q_static, cu_seqlens_q, seqused_q)
        offset_k, seqlen_k = side(seqlen_k_static, cu_seqlens_k, seqused_k)
        return SeqlenInfo(offset_q, offset_k, seqlen_q, seqlen_k)


# --- BlockInfo --------------------------------------------------------------
# KV block range for a query block, the counterpart of FA4's ``block_info.BlockInfo``.


@dataclass
class BlockInfo:
    """Static tile sizes and mask kind. ``is_causal``: keys up to the (bottom-right
    aligned) diagonal. ``is_local``: runtime ``window_left`` / ``window_right``, where a
    negative bound means unbounded on that side; causal with a left window is passed as
    local with ``window_right = 0``."""

    tile_m: int
    tile_n: int
    is_causal: bool
    is_local: bool

    def get_n_block_min_max(self, seqlen: SeqlenInfo, m_block, window_left, window_right):
        """``[n_block_min, n_block_max)``: the KV blocks a query block can see."""
        n_block_max = _ceil_div(seqlen.seqlen_k, self.tile_n)
        n_block_min = fx.Int32(0)
        if self.is_causal or self.is_local:
            m_start = m_block * fx.Int32(self.tile_m)
            m_end = _min(m_start + fx.Int32(self.tile_m), seqlen.seqlen_q)
            zero = fx.Int32(0)
            if self.is_causal:
                last_key = m_end - fx.Int32(1) + seqlen.offset
                n_block_max = _min(n_block_max, _ceil_div(_max(last_key + fx.Int32(1), zero), self.tile_n))
            else:
                last_key = m_end - fx.Int32(1) + seqlen.offset + window_right
                bounded_max = _min(n_block_max, _ceil_div(_max(last_key + fx.Int32(1), zero), self.tile_n))
                n_block_max = (window_right >= zero).select(bounded_max, n_block_max)
                first_key = m_start + seqlen.offset - window_left
                n_block_min = (window_left >= zero).select(
                    _max(first_key, zero) // fx.Int32(self.tile_n), zero
                )
        return n_block_min, n_block_max

    def get_n_block_interior(
        self, seqlen: SeqlenInfo, m_block, window_left, window_right, n_block_min, n_block_max
    ):
        """``[n_lo, n_hi)`` within ``[n_block_min, n_block_max)``: the KV blocks whose every key
        is kept for every query of the block by causal / local and the sequence end, so they
        need no per-element mask (without a mask_mod). The rest of the range is masked."""
        zero = fx.Int32(0)
        tn = fx.Int32(self.tile_n)
        m_start = m_block * fx.Int32(self.tile_m)
        m_last = _min(m_start + fx.Int32(self.tile_m), seqlen.seqlen_q) - fx.Int32(1)
        n_lo = zero
        n_hi = seqlen.seqlen_k // tn  # blocks wholly before the sequence end
        if self.is_causal:
            n_hi = _min(n_hi, _max(m_start + seqlen.offset + fx.Int32(1), zero) // tn)
        if self.is_local:
            lo = _ceil_div(_max(m_last + seqlen.offset - window_left, zero), self.tile_n)
            n_lo = (window_left >= zero).select(lo, zero)
            hi = _max(m_start + seqlen.offset + window_right + fx.Int32(1), zero) // tn
            n_hi = (window_right >= zero).select(_min(n_hi, hi), n_hi)
        n_lo = _min(_max(n_lo, n_block_min), n_block_max)
        n_hi = _max(_min(n_hi, n_block_max), n_lo)
        return n_lo, n_hi

    @staticmethod
    def split(n_block_min, n_block_max, split_idx, num_splits):
        """FA4's split-KV share: ``ceil(n / num_splits)`` consecutive blocks per split."""
        zero = fx.Int32(0)
        per_split = (n_block_max > n_block_min).select(
            (n_block_max - n_block_min + num_splits - fx.Int32(1)) // num_splits, zero
        )
        lo = n_block_min + split_idx * per_split
        return lo, _min(lo + per_split, n_block_max)

    def get_m_block_min_max(self, seqlen: SeqlenInfo, n_block, window_left, window_right):
        """``[m_block_min, m_block_max)``: the query blocks that can see a KV block (the
        backward's dK / dV loop). An empty range may come out with min > max."""
        m_block_max = _ceil_div(seqlen.seqlen_q, self.tile_m)
        m_block_min = fx.Int32(0)
        if self.is_causal or self.is_local:
            zero = fx.Int32(0)
            n_start = n_block * fx.Int32(self.tile_n)
            first_query = n_start - seqlen.offset  # a query sees keys up to its diagonal (+ right)
            if self.is_causal:
                m_block_min = _max(first_query, zero) // fx.Int32(self.tile_m)
            else:
                bounded_min = _max(first_query - window_right, zero) // fx.Int32(self.tile_m)
                m_block_min = (window_right >= zero).select(bounded_min, zero)
                last_query = n_start + fx.Int32(self.tile_n) - fx.Int32(1) - seqlen.offset + window_left
                bounded_max = _min(m_block_max, _ceil_div(_max(last_query + fx.Int32(1), zero), self.tile_m))
                m_block_max = (window_left >= zero).select(bounded_max, m_block_max)
        return m_block_min, m_block_max


# --- AttentionMask ----------------------------------------------------------
# Per-element masking, the counterpart of FA4's ``mask.AttentionMask``.


@dataclass
class AttentionMask:
    """Which (query, key) score elements are kept. Causal and local are bottom-right
    aligned (``key - query <= seqlen_k - seqlen_q``), as in FA4; ``window_left`` /
    ``window_right`` are runtime and negative means unbounded.

    ``mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors) -> bool``
    is FA4's signature, with sequence-local indices. Indices past a sequence's end
    (tile padding) are clamped to its last position before the call, so a mod that
    indexes an aux tensor by them stays in bounds; those positions are masked anyway.
    """

    seqlen: SeqlenInfo
    is_causal: bool
    is_local: bool
    window_left: fx.Int32
    window_right: fx.Int32
    mask_mod: Optional[Callable] = None
    batch_idx: Optional[fx.Int32] = None
    head_idx: Optional[fx.Int32] = None
    aux_tensors: Tuple = ()

    def clamp_q(self, q_pos):
        return _min(q_pos, self.seqlen.seqlen_q - fx.Int32(1))

    def clamp_kv(self, kv_pos):
        return _min(kv_pos, self.seqlen.seqlen_k - fx.Int32(1))

    def keep(self, q_pos, kv_pos, mask_seqlen: bool = True, apply_mask_mod: bool = True):
        keep = kv_pos < self.seqlen.seqlen_k if mask_seqlen else (kv_pos == kv_pos)
        if self.is_causal or self.is_local:
            rel = kv_pos - q_pos - self.seqlen.offset  # 0 on the diagonal
            if self.is_causal:
                keep = keep & (rel <= fx.Int32(0))
            else:
                zero = fx.Int32(0)
                right_ok = (self.window_right < zero) | (rel <= self.window_right)
                left_ok = (self.window_left < zero) | (rel >= -self.window_left)
                keep = keep & right_ok & left_ok
        if self.mask_mod is not None and apply_mask_mod:
            keep = keep & self.mask_mod(
                self.batch_idx,
                self.head_idx,
                self.clamp_q(q_pos),
                self.clamp_kv(kv_pos),
                self.seqlen,
                self.aux_tensors,
            )
        return keep


# --- Softmax ----------------------------------------------------------------
# Online softmax, the counterpart of FA4's ``softmax.Softmax``.
#
# Each lane owns one query row (the MFMA32 accumulator column). A KV tile's scores for
# that row are split between the lane and its partner ``lane ^ 32``, so row reductions
# are in-lane plus one ``shuffle_xor(32)``. Scores arrive in the log2 domain.


def _row_reduce(values, op):
    acc = values[0]
    for v in values[1:]:
        acc = op(acc, v)
    return op(acc, _f32(fx.gpu.shuffle_xor(acc, 32, 64)))


class Softmax:
    """Running row max ``m`` (log2 domain) and row sum ``l`` for this lane's query row."""

    @staticmethod
    def initial_state():
        return [_f32(NEG_BIG), _f32(0.0)]

    @staticmethod
    def online_softmax(row_max, row_sum, scores, keep):
        """Fold one tile in. ``scores`` / ``keep``: this lane's log2-domain scores and keep
        flags (``keep=None``: every score is kept). Returns ``(probs, new_max, new_sum,
        rescale)``; the caller scales O by ``rescale`` before accumulating ``probs @ V``."""
        if keep is None:
            masked = list(scores)
        else:
            masked = [k.select(s, _f32(NEG_BIG)) for s, k in zip(scores, keep)]
        tile_max = _row_reduce(masked, _max)
        new_max = _max(row_max, tile_max)
        rescale = fx.math.exp2(row_max - new_max)
        if keep is None:
            probs = [fx.math.exp2(s - new_max) for s in masked]
        else:
            probs = [k.select(fx.math.exp2(s - new_max), _f32(0.0)) for s, k in zip(masked, keep)]
        tile_sum = _row_reduce(probs, lambda a, b: a + b)
        return probs, new_max, row_sum * rescale + tile_sum, rescale

    @staticmethod
    def finalize(row_max, row_sum, sink_log2=None):
        """``(output scale, lse)`` with the LSE in natural log.

        ``sink_log2`` (a learnable sink logit times log2(e)) adds one more softmax term
        with no value vector, as in FA4: it joins the denominator and the LSE but not O.
        An empty row then gives O = 0 and LSE = the sink; without a sink, O = 0 and
        LSE = -inf.
        """
        if sink_log2 is None:
            has_keys = row_sum > _f32(0.0)
            scale = has_keys.select(_f32(1.0) / row_sum, _f32(0.0))
            lse = has_keys.select((row_max + fx.math.log2(row_sum)) * _f32(LN2), _f32(float("-inf")))
            return scale, lse
        new_max = _max(row_max, sink_log2)
        rescale = fx.math.exp2(row_max - new_max)
        total = row_sum * rescale + fx.math.exp2(sink_log2 - new_max)
        return rescale / total, (new_max + fx.math.log2(total)) * _f32(LN2)
