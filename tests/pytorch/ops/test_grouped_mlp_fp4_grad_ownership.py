###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Real-kernel coverage for ``grouped_mlp_fp4``'s fused wgrad-accumulation path.

``test_grouped_mlp_fp4.py`` never passes ``fuse_wgrad_accum_pattern`` and never
uses a real ``torch.nn.Parameter`` for ``w1``/``w2``, so none of its cases can
reach the beta=0-first-write branch ``_setup_fused_grad_accum`` guards on
``isinstance(b, torch.nn.Parameter)``. The same is true of
``test_grouped_gemm_fp4.py::test_grouped_gemm_fp4_fused_grad_accum``, whose
weight is ``b.detach().clone().requires_grad_(True)`` -- a plain tensor, not a
``Parameter`` -- so it structurally only ever exercises beta=1 accumulate.

This file drives the real FlyDSL kernel (not just the bookkeeping module) to
cover the three end-to-end properties the campaign charter requires for the
expert wgrad path that ``primus/backends/megatron/patches/turbo/
grad_buffer_ownership_patches.py`` and ``primus_turbo.pytorch.core.
grad_ownership`` build their skip-zeroing decision on:

* a beta=0 write is a true overwrite -- it must not depend on whatever was in
  ``main_grad`` beforehand, which is the actual safety property that lets the
  framework skip zeroing that slice;
* a zero-token expert's beta=0 write must not retain prior/poison content --
  its slice of ``main_grad`` must come out exactly zero and finite;
* a second microbatch against the same real ``Parameter`` accumulates
  (beta=1) on top of the first microbatch's beta=0 write exactly once, rather
  than overwriting or dropping it. The beta=0 writer is selected when backward
  executes; CPU regression coverage for recompute and reversed backward order
  lives in ``test_fused_grad_overwrite_claim.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.low_precision import Float4QuantConfig, check_mxfp4_support
from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4
from tests.pytorch.test_utils import compute_snr

SNR_THRESHOLD = 10.0

# Same (M, K, I, G) as test_grouped_mlp_fp4.py's SHAPES[0]: every GEMM dim is a
# 32-multiple and K is an odd multiple of 128, which the fused GLU/dGLU quant
# epilogues require (see that file's ``glu_epi_quant_supported`` note).
M, K, I, G = 2048, 896, 512, 4


def _require_mxfp4():
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)


def _group_lens_with_zero_expert(total_m: int, num_groups: int, zero_idx: int):
    """Split ``total_m`` tokens over ``num_groups`` groups with ``zero_idx`` empty."""
    others = [i for i in range(num_groups) if i != zero_idx]
    base, rem = divmod(total_m, len(others))
    lens = [0] * num_groups
    for k, i in enumerate(others):
        lens[i] = base + (1 if k < rem else 0)
    assert sum(lens) == total_m and lens[zero_idx] == 0
    return lens


def _leaves(group_lens_values, seed):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    lens = torch.tensor(group_lens_values, device=dev, dtype=torch.int64)
    offs = torch.cat([torch.zeros(1, device=dev, dtype=torch.int64), lens.cumsum(0)])
    x = (torch.randn(M, K, device=dev, generator=gen) * 0.1).bfloat16()
    w1 = (torch.randn(G, 2 * I, K, device=dev, generator=gen) * 0.02).bfloat16()
    w2 = (torch.randn(G, K, I, device=dev, generator=gen) * 0.02).bfloat16()
    probs = torch.rand(M, device=dev, dtype=torch.float32, generator=gen) + 0.25
    return offs, lens, x, w1, w2, probs


def _fused_param(w: torch.Tensor, main_grad_init: torch.Tensor) -> torch.nn.Parameter:
    """A real ``nn.Parameter`` carrying the framework's fused-wgrad attributes.

    Only a real ``Parameter`` can ever take ``_setup_fused_grad_accum``'s
    ``first_write=True`` branch -- a plain ``requires_grad`` tensor cannot,
    regardless of ``grad_added_to_main_grad``.
    """
    p = torch.nn.Parameter(w.detach().clone())
    p.main_grad = main_grad_init.clone()
    p.grad_added_to_main_grad = False
    return p


def _run_fused(x, w1_param, w2_param, group_lens, probs, group_offs, cotangent):
    x = x.detach().clone().requires_grad_(True)
    probs = probs.detach().clone().requires_grad_(True)
    out = grouped_mlp_fp4(
        x,
        w1_param,
        w2_param,
        group_lens,
        probs,
        group_offs=group_offs,
        trans_w1=True,
        trans_w2=True,
        config=Float4QuantConfig(),
        activation="silu",
        fuse_wgrad_accum_pattern="megatron",
    )
    out.backward(cotangent.to(out.dtype))
    return out.detach()


def _ref_grad_w(x, w1, w2, probs, offs, cotangent):
    """Per-expert fp32 eager reference; returns fp32 (grad_w1, grad_w2).

    ``cotangent`` must be the exact same upstream gradient tensor the fused op
    was driven with -- a gradient reference is only meaningful against the
    same cotangent.

    A zero-length group is skipped in the loop entirely, so autograd naturally
    reports an exact-zero gradient for it -- this is the ground truth the
    fused kernel's zero-token-expert handling is checked against.
    """
    x = x.detach().clone().float().requires_grad_(True)
    w1p = w1.detach().clone().float().requires_grad_(True)
    w2p = w2.detach().clone().float().requires_grad_(True)
    probsf = probs.detach().clone().float()
    outs = []
    for g in range(w1p.shape[0]):
        lo, hi = int(offs[g]), int(offs[g + 1])
        if hi == lo:
            continue
        l1 = x[lo:hi] @ w1p[g].t()
        gate, up = torch.chunk(l1, 2, dim=-1)
        act = F.silu(gate) * up * probsf[lo:hi, None]
        outs.append(act @ w2p[g].t())
    out = torch.cat(outs, dim=0)
    out.backward(cotangent.float())
    return w1p.grad, w2p.grad


@pytest.mark.parametrize("zero_idx", [0, 1, 3])
def test_zero_token_expert_beta0_write_has_no_nan_residual(zero_idx):
    """A zero-token expert's beta=0 write must land exact-zero, not poison."""
    _require_mxfp4()
    group_lens_values = _group_lens_with_zero_expert(M, G, zero_idx)
    offs, group_lens, x, w1, w2, probs = _leaves(group_lens_values, seed=100 + zero_idx)

    poison1 = torch.full((G, 2 * I, K), float("nan"), dtype=torch.bfloat16, device="cuda")
    poison2 = torch.full((G, K, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    w1_param = _fused_param(w1, poison1)
    w2_param = _fused_param(w2, poison2)

    gen = torch.Generator(device="cuda").manual_seed(7)
    cotangent = torch.randn(M, K, device="cuda", generator=gen)
    _run_fused(x, w1_param, w2_param, group_lens, probs, offs, cotangent)

    assert w1_param.grad_added_to_main_grad is True
    assert w2_param.grad_added_to_main_grad is True

    assert torch.isfinite(w1_param.main_grad).all(), "w1.main_grad retained NaN poison"
    assert torch.isfinite(w2_param.main_grad).all(), "w2.main_grad retained NaN poison"

    zero_w1 = w1_param.main_grad[zero_idx]
    zero_w2 = w2_param.main_grad[zero_idx]
    assert zero_w1.abs().max().item() == 0.0, "zero-token expert's w1 grad must be exactly zero"
    assert zero_w2.abs().max().item() == 0.0, "zero-token expert's w2 grad must be exactly zero"

    ref_grad_w1, ref_grad_w2 = _ref_grad_w(x, w1, w2, probs, offs, cotangent)
    snr_w1 = compute_snr(ref_grad_w1, w1_param.main_grad.float())
    snr_w2 = compute_snr(ref_grad_w2, w2_param.main_grad.float())
    assert snr_w1 > SNR_THRESHOLD, f"grad_w1 snr={snr_w1:.2f} too low"
    assert snr_w2 > SNR_THRESHOLD, f"grad_w2 snr={snr_w2:.2f} too low"


def test_beta0_overwrite_ignores_main_grads_prior_contents():
    """The safety property the campaign leans on: a beta=0 write must be a true
    overwrite, bit-identical regardless of what was in ``main_grad`` before.

    This is the real-kernel counterpart of the CPU-only bookkeeping tests in
    ``test_grad_ownership.py`` / ``test_grad_buffer_ownership.py``, which only
    check that the framework *decides* to skip zeroing -- not that skipping it
    is actually safe at the kernel level.
    """
    _require_mxfp4()
    group_lens_values = _group_lens_with_zero_expert(M, G, zero_idx=1)
    offs, group_lens, x, w1, w2, probs = _leaves(group_lens_values, seed=55)
    gen = torch.Generator(device="cuda").manual_seed(9)
    cotangent = torch.randn(M, K, device="cuda", generator=gen)

    zero_init_w1 = torch.zeros(G, 2 * I, K, dtype=torch.bfloat16, device="cuda")
    zero_init_w2 = torch.zeros(G, K, I, dtype=torch.bfloat16, device="cuda")
    w1_a = _fused_param(w1, zero_init_w1)
    w2_a = _fused_param(w2, zero_init_w2)
    _run_fused(x, w1_a, w2_a, group_lens, probs, offs, cotangent)

    garbage_w1 = torch.full((G, 2 * I, K), 12345.0, dtype=torch.bfloat16, device="cuda")
    garbage_w2 = torch.full((G, K, I), -6789.0, dtype=torch.bfloat16, device="cuda")
    w1_b = _fused_param(w1, garbage_w1)
    w2_b = _fused_param(w2, garbage_w2)
    _run_fused(x, w1_b, w2_b, group_lens, probs, offs, cotangent)

    assert torch.equal(w1_a.main_grad, w1_b.main_grad), (
        "beta=0 write must not depend on main_grad's prior contents (w1)"
    )
    assert torch.equal(w2_a.main_grad, w2_b.main_grad), (
        "beta=0 write must not depend on main_grad's prior contents (w2)"
    )


def test_two_microbatches_accumulate_exactly_once():
    """Two microbatches against the same real Parameter: beta=0 then beta=1,
    and the result must equal the sum of each microbatch's own gradient --
    not zero, not double-counted, not just the second microbatch's alone."""
    _require_mxfp4()
    group_lens_values = _group_lens_with_zero_expert(M, G, zero_idx=2)
    offs, group_lens, x1, w1, w2, probs1 = _leaves(group_lens_values, seed=200)
    _, _, x2, _, _, probs2 = _leaves(group_lens_values, seed=201)

    zero_init_w1 = torch.zeros(G, 2 * I, K, dtype=torch.bfloat16, device="cuda")
    zero_init_w2 = torch.zeros(G, K, I, dtype=torch.bfloat16, device="cuda")
    w1_param = _fused_param(w1, zero_init_w1)
    w2_param = _fused_param(w2, zero_init_w2)

    gen = torch.Generator(device="cuda").manual_seed(11)
    cotangent1 = torch.randn(M, K, device="cuda", generator=gen)
    cotangent2 = torch.randn(M, K, device="cuda", generator=gen)

    # Microbatch 1: first write on this real Parameter -> beta=0.
    assert w1_param.grad_added_to_main_grad is False
    _run_fused(x1, w1_param, w2_param, group_lens, probs1, offs, cotangent1)
    assert w1_param.grad_added_to_main_grad is True, "must flag first write"
    after_mb1_w1 = w1_param.main_grad.clone()
    after_mb1_w2 = w2_param.main_grad.clone()

    # Microbatch 2: same Parameter, flag already set -> beta=1 (accumulate).
    _run_fused(x2, w1_param, w2_param, group_lens, probs2, offs, cotangent2)

    ref1_w1, ref1_w2 = _ref_grad_w(x1, w1, w2, probs1, offs, cotangent1)
    ref2_w1, ref2_w2 = _ref_grad_w(x2, w1, w2, probs2, offs, cotangent2)

    # After mb1 alone, main_grad must already match mb1's own gradient (the
    # beta=0 write, not zero and not some partial value).
    snr_mb1_w1 = compute_snr(ref1_w1, after_mb1_w1.float())
    snr_mb1_w2 = compute_snr(ref1_w2, after_mb1_w2.float())
    assert snr_mb1_w1 > SNR_THRESHOLD, f"post-mb1 grad_w1 snr={snr_mb1_w1:.2f} too low"
    assert snr_mb1_w2 > SNR_THRESHOLD, f"post-mb1 grad_w2 snr={snr_mb1_w2:.2f} too low"

    # After mb2, main_grad must equal mb1 + mb2 -- accumulated exactly once,
    # not overwritten (which would erase mb1) and not double-added.
    snr_total_w1 = compute_snr(ref1_w1 + ref2_w1, w1_param.main_grad.float())
    snr_total_w2 = compute_snr(ref1_w2 + ref2_w2, w2_param.main_grad.float())
    assert snr_total_w1 > SNR_THRESHOLD, f"post-mb2 grad_w1 snr={snr_total_w1:.2f} too low"
    assert snr_total_w2 > SNR_THRESHOLD, f"post-mb2 grad_w2 snr={snr_total_w2:.2f} too low"
