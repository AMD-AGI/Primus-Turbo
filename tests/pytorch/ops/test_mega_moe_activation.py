###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU accuracy of MegaMoE's four fused activation kernels against a torch reference.

The EP8 op tests gate the whole op on SNR, which cannot tell a one-sided gate clamp from a
symmetric one: below -7 the gate term of ``swigluoai`` is ~1e-5 either way, so the error drowns in
the rest of the tensor. Here the pool is built from 32-row bands that each sit in ONE regime the
clamps distinguish, and every regime is checked on its own:

  * ``mixed``      gate and up ~ N(0, 5^2): both halves cross +-7 on a good fraction of elements.
  * ``gate_below`` gate in [-9, -7.5]: a one-sided clamp leaves it alone and lets the gradient
                   through; a symmetric clamp would pin it to -7 and zero ``dgate``.
  * ``gate_above`` gate in [7.5, 12]: clamped to the limit, ``dgate`` exactly 0.
  * ``up_outside`` |up| in [7.5, 12]: clamped both ways, ``dup`` exactly 0.

32-row bands keep each mxfp8 block regime-pure in both directions (rowwise blocks run along I,
colwise blocks along the pool rows), so a regime's quantization is not set by another regime's amax.
"""

import math

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import check_mxfp8_support
from primus_turbo.pytorch.core.utils import is_gfx1250

if is_gfx1250():
    pytest.skip("mega_moe_fused is not supported on gfx1250", allow_module_level=True)

import primus_turbo.pytorch  # noqa: E402,F401  (pytorch before the mega kernels)
from primus_turbo.flydsl.mega.fp8 import (  # noqa: E402
    colwise_grouped_meta,
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.utils.glu_activation import GLUActivation  # noqa: E402
from primus_turbo.flydsl.utils.swiglu_kernel import (  # noqa: E402
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_fp8_weights import (  # noqa: E402
    _DW_FP8_FORMAT,
)
from tests.pytorch.test_utils import compute_snr  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and check_mxfp8_support()[0]), reason="MegaMoE kernels require gfx950"
)

_BM = 256  # pool block (symm_buffer.BLOCK_M)
_BLK = 32  # mxfp8 block
_REGIMES = ("mixed", "gate_below", "gate_above", "up_outside")
# Ragged groups like a real pool: an empty expert, and M3's 1056 tokens per expert.
_GROUP_LENS = (300, 1056, 0, 700)

# bf16 kernels: one bf16 rounding of an fp32 result, plus a few fp32 ulp in exp -> at most 1 bf16 ulp.
_BF16_RTOL = 1.6e-2
# mxfp8: E4M3 carries 3 mantissa bits, ~29 dB on a well-scaled block; a mis-clamped regime is off by
# 2-20x and lands below 0 dB.
_FP8_SNR_DB = 24.0

_ACTIVATIONS = {
    "silu_default": None,
    "swigluoai": GLUActivation.swigluoai(),
    "quick_geglu_unclamped": GLUActivation(
        alpha=1.702, glu_offset=1.0, gate_clamp_lo=None, gate_clamp_hi=None, up_clamp=None
    ),
}


def _spec(activation):
    return GLUActivation() if activation is None else activation


def _make_pool(I, seed=0):
    """``l1 [P, 2I]`` with one regime per 32-row band, plus dact / routing weight / group tables."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lens = torch.tensor(_GROUP_LENS, dtype=torch.int64, device="cuda")
    offs = torch.cat([lens.new_zeros(1), (((lens + _BM - 1) // _BM) * _BM).cumsum(0)])
    P = int(offs[-1])

    def randn(scale):
        return torch.randn(P, I, generator=g, device="cuda") * scale

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(P, I, generator=g, device="cuda")

    sign = torch.where(torch.rand(P, I, generator=g, device="cuda") < 0.5, -1.0, 1.0)
    regime = (torch.arange(P, device="cuda") // _BLK) % len(_REGIMES)
    pick = [(regime == k).unsqueeze(-1) for k in range(len(_REGIMES))]
    gate = torch.where(pick[0], randn(5.0), torch.where(pick[1], uniform(-9.0, -7.5), randn(2.0)))
    gate = torch.where(pick[2], uniform(7.5, 12.0), gate)
    up = torch.where(pick[0], randn(5.0), randn(3.0))
    up = torch.where(pick[3], sign * uniform(7.5, 12.0), up)
    l1 = torch.cat([gate, up], dim=-1).bfloat16()

    dact = torch.randn(P, I, generator=g, device="cuda").bfloat16()
    scale = torch.rand(P, generator=g, device="cuda").float() + 0.05
    ntb = torch.tensor([P // _BM], dtype=torch.int32, device="cuda")
    return l1, dact, scale, ntb, lens, offs, regime


def _clamp(x, lo, hi):
    return x if lo is None and hi is None else x.clamp(min=lo, max=hi)


def _reference(l1, dact, scale, activation):
    """fp32 forward / backward of ``g * sigmoid(alpha g) * (u + offset)``, torch.clamp semantics."""
    a = _spec(activation)
    gate, up = l1.float().chunk(2, dim=-1)
    up_lo = None if a.up_clamp is None else -a.up_clamp
    g = _clamp(gate, a.gate_clamp_lo, a.gate_clamp_hi)
    u = _clamp(up, up_lo, a.up_clamp)
    # torch.clamp's gradient passes where the input is inside the range, inclusive.
    g_mask = (g == gate).float()
    u_mask = (u == up).float()
    s = torch.sigmoid(a.alpha * g)
    uo = u + a.glu_offset
    act = g * s * uo
    w = scale.unsqueeze(-1)
    d = dact.float() * w
    dgate = d * uo * s * (1 + a.alpha * g * (1 - s)) * g_mask
    dup = d * g * s * u_mask
    return {
        "act": act,
        "act_scaled": act * w,
        "dx": torch.cat([dgate, dup], dim=-1),
        # 1 + alpha g (1 - s) has a root, where a few fp32 ulp are a large relative error; this is
        # dgate's size without that cancellation.
        "dx_mag": torch.cat([(d * uo * s).abs() * (1 + (a.alpha * g).abs()), dup.abs()], dim=-1),
        "grad_gate": (dact.float() * act).sum(-1),
        "grad_gate_abs": (dact.float() * act).abs().sum(-1),
        "act_w": act * w,
    }


def _a_sp_bytes(a_sp, M, K):
    """Invert the pack-4 ScaleS2R A-scale preshuffle: ``a_sp`` -> raw E8M0 ``[M, K // 32]``."""
    K128p = math.ceil(K // 128 / 4)
    r = torch.arange(M, device=a_sp.device).unsqueeze(-1)
    b = torch.arange(K // _BLK, device=a_sp.device).unsqueeze(0)
    ki, grp4 = b // 4, b % 4
    kkp, byte = ki // 4, ki % 4
    idx = (((r // 64) * K128p + kkp) * 64 + grp4 * 16 + r % 16) * 4 + (r % 64) // 16
    return (a_sp.long()[idx] >> (8 * byte)) & 0xFF


def _dequant_rowwise(q, e8m0):
    M, K = q.shape
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(_BLK, dim=1)
    return q.float() * scale[:, :K]


def _real_rows(lens, offs):
    return torch.cat([torch.arange(int(o), int(o) + int(n), device=lens.device) for o, n in zip(offs, lens)])


def _assert_bf16_close(actual, ref, tag, mag=None):
    """Elementwise bf16-level agreement; ``mag`` is each element's un-cancelled size, if it has one."""
    tol = _BF16_RTOL * ref.abs() + 1e-30
    if mag is not None:
        tol = tol + 1e-4 * mag
    err = (actual.float() - ref).abs()
    bad = err > tol
    if bool(bad.any()):
        i = int((err / tol).flatten().argmax())
        raise AssertionError(
            f"[{tag}] {int(bad.sum())} / {bad.numel()} elements off; worst got "
            f"{float(actual.flatten()[i]):.6e} want {float(ref.flatten()[i]):.6e}"
        )


def _assert_row_sum_close(actual, ref, ref_abs, tag):
    # Summation order differs from torch; bound the error by the magnitude of the summed terms.
    err = (actual - ref).abs()
    worst = float((err / (ref_abs + 1e-30)).max())
    assert worst < 1e-4, f"[{tag}] row-sum rel error {worst:.3e} vs sum|terms|"


def _assert_snr_per_regime(actual, ref, regime_of_row, tag):
    for k, name in enumerate(_REGIMES):
        rows = regime_of_row == k
        a, r = actual[rows], ref[rows]
        snr = compute_snr(r, a)
        assert snr >= _FP8_SNR_DB, f"[{tag}/{name}] SNR {snr:.2f} dB < {_FP8_SNR_DB}"
        # A clamped-out element carries an exact zero gradient; the quantizer must keep it exact.
        zero = r == 0
        assert bool((a[zero] == 0).all()), (
            f"[{tag}/{name}] {int((a[zero] != 0).sum())} nonzeros where ref is 0"
        )


def _assert_clamp_regimes_exercised(ref, regime, I, activation):
    """Guard the test itself: the regimes must actually hit the behaviour they are named for."""
    if activation is None or _spec(activation).gate_clamp_hi is None:
        return
    dgate, dup = ref["dx"][:, :I], ref["dx"][:, I:]
    # Not all(): u + offset is exactly 0 for ~0.1% of bf16 draws, which zeroes dgate legitimately.
    live = float((dgate[regime == 1] != 0).float().mean())
    assert live > 0.99, f"gate_below rows must keep a live gate gradient, got {live:.4f} nonzero"
    assert bool((dgate[regime == 2] == 0).all()), "gate_above rows must be clamped"
    assert bool((dup[regime == 3] == 0).all()), "up_outside rows must be clamped"


@pytest.mark.parametrize("I", [3072, 1536])
@pytest.mark.parametrize("act_name", list(_ACTIVATIONS))
def test_bf16_activation_kernels(I, act_name):
    activation = _ACTIVATIONS[act_name]
    l1, dact, scale, ntb, _, _, regime = _make_pool(I)
    ref = _reference(l1, dact, scale, activation)
    _assert_clamp_regimes_exercised(ref, regime, I, activation)

    _assert_bf16_close(swiglu_flydsl_kernel(l1, ntb, activation=activation), ref["act"], "fwd")
    _assert_bf16_close(
        swiglu_flydsl_kernel(l1, ntb, scale=scale, activation=activation), ref["act_scaled"], "fwd scaled"
    )

    dx, grad_gate, act_w = swiglu_backward_flydsl_kernel(
        dact, l1, ntb, scale=scale, return_gate=True, return_act_w=True, activation=activation
    )
    _assert_bf16_close(dx, ref["dx"], "bwd dx", mag=ref["dx_mag"])
    _assert_bf16_close(act_w, ref["act_w"], "bwd act_w")
    _assert_row_sum_close(grad_gate, ref["grad_gate"], ref["grad_gate_abs"], "bwd grad_gate")
    # The column-tiled variant (no gate reduction) is a different kernel body.
    dx_only = swiglu_backward_flydsl_kernel(dact, l1, ntb, scale=scale, activation=activation)
    _assert_bf16_close(dx_only, ref["dx"], "bwd dx only", mag=ref["dx_mag"])


@pytest.mark.parametrize("I", [3072, 1536])
@pytest.mark.parametrize("act_name", list(_ACTIVATIONS))
def test_mxfp8_activation_kernels(I, act_name):
    activation = _ACTIVATIONS[act_name]
    l1, dact, scale, ntb, lens, offs, regime = _make_pool(I)
    P = l1.shape[0]
    ref = _reference(l1, dact, scale, activation)
    _assert_clamp_regimes_exercised(ref, regime, I, activation)

    # forward: rowwise mxfp8 of the activation, scales in the GEMM's preshuffled layout
    q, a_sp = swiglu_mxfp8_flydsl_kernel(l1, ntb, activation=activation)
    act = _dequant_rowwise(q, _a_sp_bytes(a_sp, P, I))
    _assert_snr_per_regime(act, ref["act"], regime, "fwd")

    # backward: grad_l1 as rowwise (L1 dgrad) + colwise (dW1) mxfp8, plus the fp32/bf16 side outputs
    meta = colwise_grouped_meta(lens, offs, pool_rows=P)
    q_row, a_sp_row, q_col, s_col, grad_gate, act_w = swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl(
        dact, l1, scale, _DW_FP8_FORMAT, meta=meta, activation=activation
    )
    rows = _real_rows(lens, offs)
    dx_ref = ref["dx"][rows]

    dx_row = _dequant_rowwise(q_row, _a_sp_bytes(a_sp_row, P, 2 * I))[rows]
    _assert_snr_per_regime(dx_row, dx_ref, regime[rows], "bwd rowwise")

    # colwise lives in a per-group 128-padded row space; map each real pool row onto it
    lens_pc = ((lens + 127) // 128) * 128
    offs_pc = torch.cat([lens.new_zeros(1), lens_pc.cumsum(0)])
    cols = _real_rows(lens, offs_pc)
    col_scale = torch.exp2(s_col.float() - 127.0).repeat_interleave(_BLK, dim=1)
    dx_col = (q_col.float() * col_scale[:, : q_col.shape[1]])[:, cols].t()
    _assert_snr_per_regime(dx_col, dx_ref, regime[rows], "bwd colwise")

    _assert_bf16_close(act_w[rows], ref["act_w"][rows], "bwd act_w")
    _assert_row_sum_close(
        grad_gate[rows], ref["grad_gate"][rows], ref["grad_gate_abs"][rows], "bwd grad_gate"
    )
    outside = torch.ones(P, dtype=torch.bool, device=l1.device)
    outside[rows] = False
    assert bool((grad_gate[outside] == 0).all()), "rows outside every group must fold to a 0 gate gradient"
