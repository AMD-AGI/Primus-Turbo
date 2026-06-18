###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL attention primitives for the DeepSeek-V4 forward kernel (design §4).

This is the attention counterpart of ``flydsl/utils/gemm_helper.py``. It adds
the primitives the online-softmax flash-attention dataflow needs on top of the
GEMM helpers (which it re-exports for reuse):

* :class:`MfmaBF16` — a bf16 / fp16 ``mfma_f32_16x16x16`` atom wrapper (the
  QK and PV matmuls run in input dtype with an fp32 accumulator; design §4.2).
* in-register online-softmax helpers — ``exp2`` pre-scaled by ``log2e`` using
  the single-issue hardware transcendental, the running-max / running-sum
  rescale, and the cross-lane *peer* reduction over the MFMA row partners
  (``lane ^ 32`` for the wave64 16x16 / 32x32 tile; design §4.4).
* SWA / causal mask folding into the QK register tile with the finite
  ``NEG_INF = -1e30`` sentinel (design §4.5 / §6).
* the per-head softmax sink virtual-column update (design §4.8) and the final
  ``O = O_acc * rcp(l_i)`` / ``LSE = m_i + log(l_i)`` normalisation.

The reusable GEMM primitives (``G2SLoader``, ``S2RLoaderTr``, ``make_value_attrs``,
``xcd_remap_pid``, ``wait_barrier``, swizzle helpers) are imported from
``gemm_helper`` so both backends share one set of G2S / transpose loaders.

NOTE (bring-up): this targets the same in-tree FlyDSL API as ``gemm_helper``
(``fx.rocdl.cdna4.*``, ``fx.struct`` / ``fx.Array``, ``Vec``). The exact
register / LDS layouts here are the design's round-0/1 starting point and must
be validated + tuned on gfx950 per the iteration plan (design §7.3). Until then
the dispatcher gates FlyDSL behind ``can_handle`` (gfx950, ``D == 512``) and
keeps Triton as the default + fallback, so nothing regresses for existing users.
"""

import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl  # noqa: F401
from flydsl.expr.typing import Vector as Vec

# Re-export the reusable GEMM primitives so the attention kernel imports them
# from one place (and so both backends share the same loaders / value-attrs).
from primus_turbo.flydsl.utils.gemm_helper import (  # noqa: F401
    G2SLoader,
    S2RLoader,
    S2RLoaderTr,
    ceildiv,
    make_value_attrs,
    swizzle_128,
    wait_barrier,
    xcd_remap_pid,
)

# Finite negative sentinel for masked / out-of-bounds keys. Using -inf would
# make exp(-inf - -inf) = NaN for all-masked rows (design §6); -1e30 keeps
# exp(NEG_INF) ~= 0 with identical algebra.
NEG_INF = -1.0e30

# log2(e): online softmax uses exp2 (single-issue v_exp_f32) instead of exp,
# pre-scaling the exponent by log2e (design §4.4 / §6).
LOG2E = 1.4426950408889634


# ───────────────────────────────────────────────────────────────────────────
# bf16 / fp16 MFMA atom (QK and PV), fp32 accumulator
# ───────────────────────────────────────────────────────────────────────────
class MfmaBF16:
    """``mfma_f32_16x16x16`` atom for bf16 / fp16 operands, fp32 accumulator.

    Mirrors :class:`gemm_helper.Mfma16x16x128` but for the 16-bit attention
    matmuls: the K-step is 16 elements (so a ``D = 512`` contraction is 32
    MFMA issues), and ``n_tiles_a`` / ``n_tiles_b`` index the output-tile
    fragment grid exactly as the fp8 atom does. ``is_fp16`` switches the ROCDL
    op between the f16 and bf16 16x16x16 instructions.
    """

    def __init__(self, n_tiles_a, n_tiles_b, is_fp16=False):
        elem = fx.Float16 if is_fp16 else fx.BFloat16
        # 16x16x16 atom: A [16, 16], B [16, 16], C/D [16, 16] f32. The bf16/fp16
        # MFMA atom lives at the top-level rocdl namespace (CDNA3 MFMA type);
        # only the fp8 ``MFMA_Scale`` form is under ``rocdl.cdna4``.
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem))
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c):
        from flydsl._mlir.dialects import fly as _fly

        return _fly.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)])
        return c


# ───────────────────────────────────────────────────────────────────────────
# cross-lane peer reduction (MFMA row partners)
# ───────────────────────────────────────────────────────────────────────────
def _peer_value(val_f32, xor_mask):
    """Read ``val_f32`` from the partner lane ``lane ^ xor_mask`` via
    ``ds_bpermute`` (byte address = src_lane * 4). The result is bit-identical
    f32; the lgkmcnt drain is emitted by the intrinsic lowering."""
    lane = fx.thread_idx.x % 64
    src_lane = lane ^ fx.Int32(xor_mask)
    addr = (src_lane * fx.Int32(4)).bitcast(fx.Int32)
    bits = val_f32.bitcast(fx.Int32)
    out = rocdl.ds_bpermute(fx.Int32.ir_type, addr, bits).result
    return out.bitcast(fx.Float32)


def peer_reduce_max(val_f32, xor_mask=32):
    """fmax of ``val`` with its MFMA row-partner lane (``lane ^ xor_mask``).

    For the wave64 16x16 attention tiles the QK score columns of one row are
    split across the ``lane ^ 32`` partner, so the per-row max / sum must be
    reduced across that partner before the softmax update (design §4.4; the XOR
    mask is tile-bound — changing the MFMA atom changes it)."""
    return val_f32.maximumf(_peer_value(val_f32, xor_mask))


def peer_reduce_sum(val_f32, xor_mask=32):
    """Sum of ``val`` with its MFMA row-partner lane (design §4.4)."""
    return val_f32 + _peer_value(val_f32, xor_mask)


# ───────────────────────────────────────────────────────────────────────────
# online-softmax math (exp2 + log2e prescale)
# ───────────────────────────────────────────────────────────────────────────
def exp2_scaled(x_minus_m):
    """``exp2(x * log2e)`` via the single-issue hardware transcendental.

    ``p = exp2((s - m) * log2e)`` and ``corr = exp2((m_old - m_new) * log2e)``
    both go through this (design §4.4 / §6)."""
    return (x_minus_m * fx.Float32(LOG2E)).exp2()


def rcp_f32(x):
    """Reciprocal for the final ``O_acc * rcp(l_i)`` divide. ``ArithValue``
    division lowers to ``arith.divf`` (``v_rcp_f32`` + Newton step); using the
    operator keeps the value an ``ArithValue`` for the surrounding math."""
    return fx.Float32(1.0) / x


def ln_f32(x):
    """Natural log via the ``math.log2`` op: ``ln(x) = log2(x) / log2(e)``.
    Used for ``LSE = m_i + ln(l_i)``."""
    from flydsl._mlir.dialects import math as _math

    log2x = _math.log2(x)
    return log2x * fx.Float32(1.0 / LOG2E)


# ───────────────────────────────────────────────────────────────────────────
# SWA / causal mask folding into the QK register tile
# ───────────────────────────────────────────────────────────────────────────
def apply_swa_mask(score_f32, row_idx, col_idx, swa_window, seqlen_k, use_causal):
    """Fold the visibility predicate into the QK score with the NEG_INF sentinel.

    Visibility (design §4.5, bit-aligned with the Triton kernel):

    * ``SWA_WINDOW > 0``: keep ``col in [row - SWA_WINDOW + 1, row]``;
    * else ``use_causal``: keep ``col <= row``;
    * always: keep ``col < seqlen_k`` (out-of-bounds keys -> NEG_INF).
    """
    in_bounds = col_idx < fx.Int32(seqlen_k)
    if swa_window > 0:
        lo = row_idx - fx.Int32(swa_window - 1)
        visible = (col_idx >= lo) & (col_idx <= row_idx) & in_bounds
    elif use_causal:
        visible = (col_idx <= row_idx) & in_bounds
    else:
        visible = in_bounds
    return arith.select(visible, score_f32, fx.Float32(NEG_INF))


# ───────────────────────────────────────────────────────────────────────────
# sink virtual-column update + normalise
# ───────────────────────────────────────────────────────────────────────────
def apply_sink(o_acc, m_i, l_i, sink_h):
    """Join the per-head learned sink as a virtual key column (design §4.8).

    ``m_new = max(m_i, sink)``; ``l_i = l_i*exp2((m_i-m_new)*log2e) +
    exp2((sink-m_new)*log2e)``; ``O_acc *= exp2((m_i-m_new)*log2e)`` (the sink
    notional value is 0, so it joins ``l_i`` but not ``O_acc``)."""
    m_new = arith.maximumf(m_i, sink_h)
    corr = exp2_scaled(m_i - m_new)
    beta = exp2_scaled(sink_h - m_new)
    l_new = l_i * corr + beta
    o_acc = [Vec(o) * corr for o in o_acc]
    return o_acc, m_new, l_new


def normalize_o(o_acc, l_i):
    """``O = O_acc * rcp(l_i)`` with the all-masked-row 0-guard (design §6)."""
    inv = (l_i > fx.Float32(0.0)).select(rcp_f32(l_i), fx.Float32(0.0))
    return [Vec(o) * inv for o in o_acc]


def lse_value(m_i, l_i):
    """``LSE = m_i + ln(l_i)`` (fp32, saved for backward). The all-masked-row
    guard maps ``l_i <= 0`` to ``ln(1) = 0`` so LSE stays finite (design §6)."""
    safe_l = (l_i > fx.Float32(0.0)).select(l_i, fx.Float32(1.0))
    return m_i + ln_f32(safe_l)
