###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.ops import (
    causal_mask,
    create_block_mask_varlen,
    flex_attention_varlen,
    flex_attention_varlen_rel_bias,
    identity_score_mod_bwd,
    sliding_window_mask,
)
from tests.pytorch.ref import inkling_attention_ref as inkling
from tests.pytorch.ref.flex_attention_ref import flex_attention_varlen_ref, rel_bias_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")

DTYPE = torch.bfloat16
TOL = 2e-2


def _cu(seqlens):
    return torch.tensor([0] + list(seqlens), device="cuda").cumsum(0).to(torch.int32)


def _packed(Tq, Tk, Hq, Hkv, D, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to("cuda", DTYPE).requires_grad_(True)

    return randn(Tq, Hq, D), randn(Tk, Hkv, D), randn(Tk, Hkv, D)


def _check(out, ref, leaves):
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, leaves, dout)
    ref_grads = torch.autograd.grad(ref, leaves, dout.float())
    for i, (g, r) in enumerate(zip(grads, ref_grads)):
        scale = r.abs().max().item()
        err = (g.float() - r.float()).abs().max().item()
        assert err <= 5 * TOL * max(scale, 1.0), f"grad {i}: max err {err} vs max |ref| {scale}"


# The longest sequence is kept at 256 or 300, because the forward compiles one kernel
# per distinct maximum length.
BUILTINS = {
    "none": (None, None),
    "causal": (lambda: causal_mask, lambda b, h, q, kv: q >= kv),
    "window100": (lambda: sliding_window_mask(100), lambda b, h, q, kv: (q >= kv) & (q - kv <= 100)),
}
# One sequence, several ragged ones, and two long ones, each mask on at least two.
BUILTIN_CASES = [
    ("none", (256, 150, 106)),
    ("causal", (256,)),
    ("causal", (300, 212)),
    ("window100", (256, 150, 106)),
    ("window100", (300, 212)),
]


@pytest.mark.parametrize("name,seqlens", BUILTIN_CASES, ids=[f"{n}-{s}" for n, s in BUILTIN_CASES])
def test_builtin_masks_self_attention(name, seqlens):
    mask_fn, keep_fn = BUILTINS[name]
    cu = _cu(seqlens)
    q, k, v = _packed(cu[-1].item(), cu[-1].item(), 8, 2, 128)
    block_mask = None
    if mask_fn is not None:
        block_mask = create_block_mask_varlen(mask_fn(), cu, cu, max(seqlens), max(seqlens))
        assert block_mask._fast_path is not None
    out = flex_attention_varlen(
        q, k, v, cu, cu, max(seqlens), max(seqlens), block_mask=block_mask, enable_gqa=True
    )
    _check(out, flex_attention_varlen_ref(q, k, v, cu, cu, keep_fn=keep_fn)[0], (q, k, v))


def test_causal_mask_with_different_key_lengths():
    # Separate cu_seqlens tensors leave the fast path; the mask runs per tile instead.
    seq_q, seq_k = (128, 256), (256, 256)
    cu_q, cu_k = _cu(seq_q), _cu(seq_k)
    q, k, v = _packed(cu_q[-1].item(), cu_k[-1].item(), 4, 4, 64, seed=1)
    block_mask = create_block_mask_varlen(causal_mask, cu_q, cu_k, max(seq_q), max(seq_k), BLOCK_SIZE=64)
    assert block_mask._fast_path is None
    out = flex_attention_varlen(q, k, v, cu_q, cu_k, max(seq_q), max(seq_k), block_mask=block_mask)
    ref = flex_attention_varlen_ref(q, k, v, cu_q, cu_k, keep_fn=lambda b, h, qi, ki: qi >= ki)[0]
    _check(out, ref, (q, k, v))


@triton.jit
def _band_mask(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 80) & (d > -80)


def test_custom_block_sparse_mask():
    seqlens = (300, 212, 128)
    cu = _cu(seqlens)
    q, k, v = _packed(cu[-1].item(), cu[-1].item(), 4, 2, 64, seed=2)
    block_mask = create_block_mask_varlen(_band_mask, cu, cu, max(seqlens), max(seqlens), BLOCK_SIZE=64)
    out = flex_attention_varlen(
        q, k, v, cu, cu, max(seqlens), max(seqlens), block_mask=block_mask, enable_gqa=True
    )
    ref = flex_attention_varlen_ref(q, k, v, cu, cu, keep_fn=lambda b, h, qi, ki: (qi - ki).abs() < 80)[0]
    _check(out, ref, (q, k, v))


@triton.jit
def _per_sequence_prefix_mask(b, h, q_idx, kv_idx, PREFIX):
    return (kv_idx < tl.load(PREFIX + b)) | (q_idx >= kv_idx)


def test_mask_aux_tensor_per_sequence_prefix():
    # b is the sequence index under varlen, so a per-sequence value is PREFIX[b].
    seqlens = (300, 212, 128)
    cu = _cu(seqlens)
    prefix = torch.tensor([150, 20, 128], device="cuda", dtype=torch.int32)
    q, k, v = _packed(cu[-1].item(), cu[-1].item(), 4, 2, 64, seed=6)
    block_mask = create_block_mask_varlen(
        _per_sequence_prefix_mask, cu, cu, max(seqlens), max(seqlens), BLOCK_SIZE=64, aux_tensors=[prefix]
    )
    out = flex_attention_varlen(
        q, k, v, cu, cu, max(seqlens), max(seqlens), block_mask=block_mask, enable_gqa=True
    )
    ref = flex_attention_varlen_ref(
        q, k, v, cu, cu, keep_fn=lambda b, h, qi, ki: (ki < prefix[b]) | (qi >= ki)
    )[0]
    _check(out, ref, (q, k, v))


def test_custom_mask_lists_build_without_device_sync():
    # Varlen lists are rebuilt for every batch, so building them -- forward and backward
    # -- must not read anything back from the device.
    from primus_turbo.pytorch.kernels.flex_attention.flex_attention_block_mask_utils import (
        backward_block_plans,
    )

    create_block_mask_varlen(_band_mask, _cu((64,)), _cu((64,)), 64, 64, BLOCK_SIZE=64)  # compile
    seqlens = (300, 212, 128)
    cu = _cu(seqlens)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        block_mask = create_block_mask_varlen(_band_mask, cu, cu, max(seqlens), max(seqlens), BLOCK_SIZE=64)
        plan = block_mask.plan
        dq_plan, dkdv_plan = backward_block_plans(plan)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert plan.mask_block_idx.shape[-1] == plan.num_kv_blocks
    assert dkdv_plan.mask_block_idx.shape[-1] == dkdv_plan.num_kv_blocks

    q, k, v = _packed(cu[-1].item(), cu[-1].item(), 4, 2, 64, seed=2)
    out = flex_attention_varlen(
        q, k, v, cu, cu, max(seqlens), max(seqlens), block_mask=block_mask, enable_gqa=True
    )
    ref = flex_attention_varlen_ref(q, k, v, cu, cu, keep_fn=lambda b, h, qi, ki: (qi - ki).abs() < 80)[0]
    _check(out, ref, (q, k, v))


@triton.jit
def _packed_kv_bias_score_mod(score, b, h, q_idx, kv_idx, BIAS, CU_SEQLENS):
    return score + tl.load(BIAS + tl.load(CU_SEQLENS + b) + kv_idx)


def test_two_aux_tensors():
    seqlens = (256, 150, 106)
    cu = _cu(seqlens)
    q, k, v = _packed(cu[-1].item(), cu[-1].item(), 4, 4, 128, seed=3)
    bias = torch.randn(cu[-1].item() + 256, device="cuda") * 2  # padded for tail tiles
    out = flex_attention_varlen(
        q,
        k,
        v,
        cu,
        cu,
        max(seqlens),
        max(seqlens),
        score_mod=_packed_kv_bias_score_mod,
        score_mod_bwd=identity_score_mod_bwd,
        block_mask=create_block_mask_varlen(causal_mask, cu, cu, max(seqlens), max(seqlens)),
        aux_tensors=[bias, cu],
    )
    starts = cu[:-1]

    def score_fn(s, b, h, qi, ki):
        return s + bias[starts[b] + ki]

    ref = flex_attention_varlen_ref(
        q, k, v, cu, cu, score_fn=score_fn, keep_fn=lambda b, h, qi, ki: qi >= ki
    )[0]
    _check(out, ref, (q, k, v))


# (Hq/Hkv, seqlens, rel_extent, window_left): one tile and several, ragged tails, windows
# shorter than, equal to and longer than the extent, alternating Inkling's global (16/2)
# and local (16/4) head layouts.
REL_BIAS_CASES = [
    ((16, 2), (256,), 48, None),
    ((16, 4), (256, 96), 48, 47),
    ((16, 2), (300, 106), 130, None),
    ((16, 4), (300, 212), 130, 199),
    ((16, 2), (300, 212), 200, 129),
]


@pytest.mark.parametrize("heads,seqlens,rel_extent,window_left", REL_BIAS_CASES, ids=str)
def test_rel_bias(heads, seqlens, rel_extent, window_left):
    Hq, Hkv = heads
    cu = _cu(seqlens)
    T = cu[-1].item()
    q, k, v = _packed(T, T, Hq, Hkv, 128, seed=4)
    rel = (torch.randn(T, Hq, rel_extent, device="cuda") * 0.5).requires_grad_(True)
    out = flex_attention_varlen_rel_bias(
        q, k, v, rel, cu, max(seqlens), scale=1.0 / 128, window_left=window_left
    )
    ref = rel_bias_ref(q, k, v, rel, cu, scale=1.0 / 128, window_left=window_left)
    _check(out, ref, (q, k, v, rel))


@pytest.mark.parametrize(
    "layer",
    [layer for layer in inkling.INKLING_LAYERS if not layer.name.startswith("inkling_small")],
    ids=lambda layer: layer.name,
)
def test_inkling_matches_miles_flex(layer):
    # Inkling's heads and structure at a size the references can afford: Turbo must
    # match both the fp32 reference and Miles' own PyTorch FlexAttention formulation.
    layer = layer.shrunk(
        seqlens=(256, 96) if len(layer.seqlens) > 1 else (256,),
        rel_extent=48,
        window=48 if layer.window is not None else None,
    )
    x = inkling.make_inputs(layer)
    out = inkling.turbo_flex(layer, x)
    ref = rel_bias_ref(
        x.q, x.k, x.v, x.rel_logits, x.cu_seqlens, scale=layer.scale, window_left=layer.window_left
    )
    _check(out, ref, (x.q, x.k, x.v, x.rel_logits))

    out = inkling.turbo_flex(layer, x)
    miles = inkling.miles_flex(layer, x, inkling.miles_block_mask(layer, x, compile_mask=False))
    torch.testing.assert_close(out.float(), miles.float(), atol=TOL, rtol=TOL)
    dout = torch.randn_like(out)
    (drel,) = torch.autograd.grad(out, x.rel_logits, dout)
    (drel_miles,) = torch.autograd.grad(miles, x.rel_logits_htr, dout)
    drel_miles = drel_miles.permute(1, 0, 2)
    scale = max(drel_miles.abs().max().item(), 1.0)
    assert (drel - drel_miles).abs().max().item() <= 5 * TOL * scale


def test_rel_bias_without_grad():
    seqlens = (256, 96)
    cu = _cu(seqlens)
    q, k, v = _packed(352, 352, 8, 2, 128, seed=5)
    rel = torch.randn(352, 8, 48, device="cuda") * 0.5
    with torch.no_grad():
        out = flex_attention_varlen_rel_bias(q, k, v, rel, cu, 256, scale=1.0 / 128)
    ref = rel_bias_ref(q, k, v, rel, cu, scale=1.0 / 128)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)
