###############################################################################
# Every A6W4 forward Linear quantizes its bf16 activation into the E2M3 fprop pack (aq, sa) and
# the MXFP4 Wgrad col pack (co, cs), then gemm_a6w4 reads (aq, sa) back. What sits in front of the
# pack depends on the Linear:
# ln_mod qkv / fc1 / single linear1: a lightweight Triton kernel (_ln_stats) reads x once and
# emits only each row's LayerNorm stats (rstd, nmr = -mean*rstd; no shift/scale, no stored y),
# then flydsl_quant_a6w4_act_dual_lnmod folds the modulate (y = (1+scale)*xn+shift) directly into
# the dual pack's own tile-load hook -- the pack reads x once (the same read it always needed for
# the fprop/wgrad packs) and computes y on the fly, so no modulated activation is ever written to
# or read back from HBM. Pack's row/col quantize math and the GEMM are untouched; only the
# producer->pack edge changed. attn double proj: a6w4_act_dual on the attention output gelu double
# fc2: a6w4_act_dual_legs on a single leg (preact,), gelus= (True,), biases=(bias,) -- the bias
# add + tanh-GELU run INSIDE the dual pack's tile-load hook (the same in-kernel path the "legs"
# kind already uses for its own GELU leg), so the Inductor GELU kernel and its intermediate bf16
# write + reread of the activation are both gone. Byte-for-byte the same GELU math as the old
# Inductor producer (`_gelu1_torch`, exact vs torch's tanh-GELU); only the launch count changes.
# legs single linear2: a6w4_act_dual_legs on (attn, gelu(preact + bias)) without the cat. byte-
# exact vs the still-frozen two-launch flydsl_quant_a6w4_act_dual_legs.
# ##############################################################################

import torch
import triton
import triton.language as tl

from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
    flydsl_quant_a6w4_act_dual,
    flydsl_quant_a6w4_act_dual_legs,
    flydsl_quant_a6w4_act_dual_legs_merged,
    flydsl_quant_a6w4_act_dual_lnmod,
)

# Static (ROWS, num_warps) per K -- "one static config per kernel chosen from the shape in Python;
# no multi-config autotune" (task.yaml). P2a's own (ROWS=4, num_warps=1) pick for K=3072 was for a
# heavier kernel body (also loading shift/scale and storing y);
_LN_STATS_CONFIG = {
    3072: dict(ROWS=1, num_warps=1),
}
_LN_STATS_DEFAULT = dict(ROWS=1, num_warps=1)


@triton.jit
def _ln_stats_kernel(
    X_ptr, RSTD_ptr, NMR_ptr,
    ROWS_PER_BATCH, K: tl.constexpr, KP: tl.constexpr, EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    """
Row LayerNorm stats only -- no modulate, no y store.

P2a's _ln_mod_kernel read x once and wrote a full bf16 y = (1+scale)*xn+shift that a6w4_act_dual then reread. This kernel keeps only the row reduction half: it emits rstd = 1/sqrt(var+eps) and nmr = -mean*rstd per row, nothing else. flydsl_quant_a6w4_act_dual_lnmod's own tile-load hook (mxfp4_quant_kernel_flux.py's _emit_dual_body, affine=True) consumes these directly from the SAME bf16 x it already has to read for the pack, computing xn = x*rstd + nmr then y = xn*(1+scale)+shift on the fly -- no y (and no xn) is ever written to or read from HBM. Same KP = next_power_of_2(K) masked-load idiom as the old _ln_mod_kernel for the non-power-of-2 K=3072 row width.
"""
    b = tl.program_id(0)
    j = tl.program_id(1)
    row0 = b * ROWS_PER_BATCH + j * ROWS
    rows = row0 + tl.arange(0, ROWS)
    k_offs = tl.arange(0, KP)
    k_mask = k_offs < K

    x_ptrs = X_ptr + rows[:, None] * K + k_offs[None, :]
    x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

    s = tl.sum(x, axis=1)
    s2 = tl.sum(x * x, axis=1)
    mean = s / K
    rstd = 1.0 / tl.sqrt(s2 / K - mean * mean + EPS)
    nmr = -mean * rstd

    tl.store(RSTD_ptr + rows, rstd)
    tl.store(NMR_ptr + rows, nmr)


def _ln_stats(x):
    """Per-row LayerNorm(eps 1e-6, no affine) stats for x [B, L, K]: returns
    (rstd, nmr), each a flat [B*L] fp32 tensor -- rstd = 1/sqrt(var+eps),
    nmr = -mean*rstd, so xn = x*rstd + nmr. Feeds
    flydsl_quant_a6w4_act_dual_lnmod's affine tile-load hook, which does the
    rest (y = xn*(1+scale)+shift) without ever materialising xn or y.
    """
    Bsz, L, K = x.shape
    M = Bsz * L
    cfg = _LN_STATS_CONFIG.get(K, _LN_STATS_DEFAULT)
    rows = cfg["ROWS"]
    if L % rows:  # grid axis 1 needs L // rows whole tiles; ROWS=1 always divides
        rows = 1
    xf = x.reshape(M, K)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    nmr = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (Bsz, L // rows)
    _ln_stats_kernel[grid](
        xf, rstd, nmr, L,
        K=K, KP=triton.next_power_of_2(K), EPS=1e-6, ROWS=rows,
        num_warps=cfg["num_warps"],
    )
    return rstd, nmr


def a6w4_fwd_pack(kind, *, x=None, shift=None, scale=None, attn=None, preact=None,
                  bias=None, wq=None, sb=None, n=None):
    """Return the packs (aq, sa, co, cs), or (y, co, cs) if the GEMM ran here.

    ln_mod: x [B, L, C], shift / scale [B, 1, C]
    attn:   x [M, K]
    gelu:   preact [M, K], bias [K]
    legs:   attn [M, leg0], preact [M, leg1], bias [leg1]
    wq, sb, n are the GEMM's pre-quantized weight, for a candidate that fuses
    the pack into gemm_a6w4 and returns y.
    """
    if kind == "ln_mod":
        Bsz, L, K = x.shape
        rstd, nmr = _ln_stats(x)
        xf = x.reshape(Bsz * L, K)
        shift2 = shift.reshape(Bsz, K)
        scale2 = scale.reshape(Bsz, K)
        return flydsl_quant_a6w4_act_dual_lnmod(xf, rstd, nmr, shift2, scale2)
    if kind == "attn":
        return flydsl_quant_a6w4_act_dual(x.contiguous())
    if kind == "gelu":
        # Single-leg legpack call: bias add + tanh-GELU fused into the dual
        # pack's own tile-load hook, instead of a separate torch.compile'd
        # GELU kernel writing a full bf16 tensor the pack then rereads.
        return flydsl_quant_a6w4_act_dual_legs((preact,), (True,), (bias,))
    if kind == "legs":
        # Byte-exact; the reference flydsl_quant_a6w4_act_dual_legs stays untouched above.
        return flydsl_quant_a6w4_act_dual_legs_merged((attn, preact), (False, True), (None, bias))
    raise ValueError(f"unknown kind {kind!r}")
