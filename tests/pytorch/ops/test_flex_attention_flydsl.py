###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FlyDSL backend of flex attention (``kernel_options={"BACKEND": "FLYDSL"}``)."""

import pytest
import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.ops import (
    AuxRequest,
    causal_mask,
    create_block_mask,
    create_block_mask_varlen,
    dual_mod,
    flex_attention,
    flex_attention_varlen,
    flex_attention_varlen_rel_bias,
    identity_score_mod_bwd,
    make_softcap_score_mod,
    noop_mask,
    sliding_window_mask,
)
from tests.pytorch.ref.flex_attention_ref import (
    flex_attention_ref,
    flex_attention_varlen_ref,
)

flydsl = pytest.importorskip("flydsl")
fx = pytest.importorskip("flydsl.expr")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")

FLYDSL = {"BACKEND": "FLYDSL"}
DTYPE = torch.bfloat16
TOL = 2e-2


def _inputs(B, Hq, Hkv, Sq, Skv, D, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to("cuda", DTYPE).requires_grad_(True)

    return randn(B, Hq, Sq, D), randn(B, Hkv, Skv, D), randn(B, Hkv, Skv, D)


def _grads(out, leaves, dout):
    return torch.autograd.grad(out, leaves, dout, retain_graph=True)


def _check(out, ref, leaves):
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)
    dout = torch.randn_like(out)
    for i, (g, r) in enumerate(zip(_grads(out, leaves, dout), _grads(ref, leaves, dout.float()))):
        scale = r.abs().max().item()
        err = (g.float() - r.float()).abs().max().item()
        assert err <= 5 * TOL * max(scale, 1.0), f"grad {i}: max err {err} vs max |ref| {scale}"


def _keep_window(left, right):
    return lambda b, h, q, kv: (q - kv <= left) & (q - kv >= -right)


# ---------------------------------------------------------------- built-in masks

BUILTIN_CASES = [
    ("none", None, None, (2, 4, 4, 256, 256, 64)),
    ("causal", causal_mask, lambda b, h, q, kv: q >= kv, (1, 8, 2, 300, 300, 128)),
    ("window64", sliding_window_mask(64), _keep_window(64, 0), (1, 8, 2, 256, 256, 128)),
    ("window32x32", sliding_window_mask(32, 32), _keep_window(32, 32), (2, 4, 4, 256, 256, 64)),
    ("noop", noop_mask, None, (1, 4, 1, 300, 300, 128)),
]


@pytest.mark.parametrize("name,mask_mod,keep_fn,shape", BUILTIN_CASES, ids=[c[0] for c in BUILTIN_CASES])
def test_builtin_masks(name, mask_mod, keep_fn, shape):
    B, Hq, Hkv, Sq, Skv, D = shape
    q, k, v = _inputs(B, Hq, Hkv, Sq, Skv, D)
    block_mask = None if mask_mod is None else create_block_mask(mask_mod, None, None, Sq, Skv)
    out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=FLYDSL)
    _check(out, flex_attention_ref(q, k, v, keep_fn=keep_fn)[0], (q, k, v))


def test_causal_unequal_lengths_is_top_left():
    # Off the fast path the mask's FlyDSL twin runs over the block lists.
    q, k, v = _inputs(1, 4, 4, 384, 640, 64)
    block_mask = create_block_mask(causal_mask, 1, 4, 384, 640, BLOCK_SIZE=64)
    out = flex_attention(q, k, v, block_mask=block_mask, kernel_options=FLYDSL)
    _check(out, flex_attention_ref(q, k, v, keep_fn=lambda b, h, qi, ki: qi >= ki)[0], (q, k, v))


# ---------------------------------------------------------------- dual mods


@triton.jit
def _head_checker_mask(b, h, q_idx, kv_idx):
    return ((q_idx // 64 + kv_idx // 64 + h) % 2) == 0


def _head_checker_mask_fly(b, h, q_idx, kv_idx, seqlen_info, aux):
    return ((q_idx // fx.Int32(64) + kv_idx // fx.Int32(64) + h) % fx.Int32(2)) == fx.Int32(0)


dual_mod(_head_checker_mask, _head_checker_mask_fly)


@triton.jit
def _doc_causal_mask(b, h, q_idx, kv_idx, DOC):
    q_doc = tl.load(DOC + b * 700 + q_idx, mask=q_idx < 700, other=-1)
    kv_doc = tl.load(DOC + b * 700 + kv_idx, mask=kv_idx < 700, other=-2)
    return (q_doc == kv_doc) & (q_idx >= kv_idx)


def _doc_causal_mask_fly(b, h, q_idx, kv_idx, seqlen_info, aux):
    docs = fx.slice(aux[0], (b, None))
    return (fx.Int32(docs[q_idx]) == fx.Int32(docs[kv_idx])) & (kv_idx <= q_idx)


dual_mod(_doc_causal_mask, _doc_causal_mask_fly)


@triton.jit
def _head_bias_score_mod(score, b, h, q_idx, kv_idx, BIAS):
    return score + tl.load(BIAS + h * 700 + kv_idx, mask=kv_idx < 700, other=0.0)


def _head_bias_score_mod_fly(score, b, h, q_idx, kv_idx, seqlen_info, aux):
    return score + fx.Float32(fx.slice(aux[0], (h, None))[kv_idx])


dual_mod(_head_bias_score_mod, _head_bias_score_mod_fly)


def test_per_head_block_sparse_mask():
    q, k, v = _inputs(2, 4, 2, 384, 640, 64, seed=1)
    block_mask = create_block_mask(_head_checker_mask, None, 4, 384, 640, BLOCK_SIZE=64)
    out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=FLYDSL)
    keep = lambda b, h, qi, ki: ((qi // 64 + ki // 64 + h) % 2) == 0  # noqa: E731
    _check(out, flex_attention_ref(q, k, v, keep_fn=keep)[0], (q, k, v))


def test_document_mask_with_aux_and_score_mod_with_aux():
    S = 700
    doc = torch.zeros(2, S, dtype=torch.int32, device="cuda")
    doc[0, 200:] = 1
    doc[0, 450:] = 2
    doc[1, 333:] = 1
    bias = torch.randn(4, S, device="cuda")
    q, k, v = _inputs(2, 4, 4, S, S, 128, seed=2)
    block_mask = create_block_mask(_doc_causal_mask, 2, None, S, S, aux_tensors=[doc])
    out = flex_attention(
        q, k, v, score_mod=_head_bias_score_mod, score_mod_bwd=identity_score_mod_bwd,
        aux_tensors=[bias], block_mask=block_mask, kernel_options=FLYDSL,
    )
    ref = flex_attention_ref(
        q, k, v, score_fn=lambda s, b, h, qi, ki: s + bias[h, ki],
        keep_fn=lambda b, h, qi, ki: (doc[b, qi] == doc[b, ki]) & (qi >= ki),
    )[0]
    _check(out, ref, (q, k, v))


def test_softcap_and_lse_gradient():
    softcap, softcap_bwd = make_softcap_score_mod(20.0)
    q, k, v = _inputs(2, 4, 4, 300, 300, 128, seed=4)
    out, aux = flex_attention(
        q, k, v, score_mod=softcap, score_mod_bwd=softcap_bwd,
        block_mask=create_block_mask(causal_mask, None, None, 300, 300),
        return_aux=AuxRequest(lse=True), kernel_options=FLYDSL,
    )
    ref, ref_lse = flex_attention_ref(
        q, k, v, score_fn=lambda s, *_: 20.0 * torch.tanh(s / 20.0), keep_fn=lambda b, h, qi, ki: qi >= ki
    )
    torch.testing.assert_close(aux.lse, ref_lse, atol=1e-3, rtol=1e-3)
    dlse = torch.randn_like(aux.lse)
    got = torch.autograd.grad((out.float() * ref.detach()).sum() + (aux.lse * dlse).sum(), (q, k, v))
    want = torch.autograd.grad((ref * ref.detach()).sum() + (ref_lse * dlse).sum(), (q, k, v))
    for g, w in zip(got, want):
        assert (g.float() - w).abs().max().item() <= 5 * TOL * max(w.abs().max().item(), 1.0)


def test_num_splits():
    q, k, v = _inputs(2, 8, 2, 4, 2048, 128, seed=5)
    out = flex_attention(q, k, v, enable_gqa=True, kernel_options={"BACKEND": "FLYDSL", "num_splits": 4})
    torch.testing.assert_close(out.float(), flex_attention_ref(q, k, v)[0], atol=TOL, rtol=TOL)


def test_backward_is_deterministic():
    q, k, v = _inputs(1, 8, 2, 512, 512, 128, seed=6)
    block_mask = create_block_mask(causal_mask, None, None, 512, 512)
    out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=FLYDSL)
    dout = torch.randn_like(out)
    first = _grads(out, (q, k, v), dout)
    second = _grads(out, (q, k, v), dout)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


# ---------------------------------------------------------------- varlen


def _cu(seqlens):
    return torch.tensor([0] + list(seqlens), device="cuda").cumsum(0).to(torch.int32)


@pytest.mark.parametrize("mask_name", ["causal", "window100", "head_checker"])
def test_varlen(mask_name):
    seqlens = (300, 212, 77)
    cu = _cu(seqlens)
    gen = torch.Generator().manual_seed(8)
    q, k, v = (
        torch.randn(sum(seqlens), h, 128, generator=gen).to("cuda", DTYPE).requires_grad_(True) for h in (8, 2, 2)
    )
    mask_mod, keep = {
        "causal": (causal_mask, lambda b, h, qi, ki: qi >= ki),
        "window100": (sliding_window_mask(100), lambda b, h, qi, ki: (qi >= ki) & (qi - ki <= 100)),
        "head_checker": (_head_checker_mask, lambda b, h, qi, ki: ((qi // 64 + ki // 64 + h) % 2) == 0),
    }[mask_name]
    block_mask = create_block_mask_varlen(mask_mod, cu, cu, max(seqlens), max(seqlens), H=8)
    out = flex_attention_varlen(
        q, k, v, cu, cu, max(seqlens), max(seqlens), block_mask=block_mask, enable_gqa=True,
        kernel_options=FLYDSL,
    )
    ref = flex_attention_varlen_ref(q, k, v, cu, cu, keep_fn=keep)[0]
    _check(out, ref, (q, k, v))


# ---------------------------------------------------------------- backend selection


@pytest.mark.parametrize("window_left", [None, 100])
def test_rel_bias_score_grad_hook_matches_triton(window_left):
    # Inkling's relative-position bias: the bias table trains through score_grad_hook.
    seqlens = (300, 129, 64)
    cu = _cu(seqlens)
    T, Hq, Hkv, D, RE = sum(seqlens), 8, 2, 64, 160
    gen = torch.Generator().manual_seed(21)
    q = torch.randn(T, Hq, D, generator=gen).to("cuda", DTYPE).requires_grad_(True)
    k, v = (torch.randn(T, Hkv, D, generator=gen).to("cuda", DTYPE).requires_grad_(True) for _ in range(2))
    rel = (torch.randn(T, Hq, RE, generator=gen) * 0.5).cuda().requires_grad_(True)
    dout = torch.randn(T, Hq, D, generator=gen).to("cuda", DTYPE)
    results = []
    for backend in ("TRITON", "FLYDSL"):
        out = flex_attention_varlen_rel_bias(q, k, v, rel, cu, max(seqlens), scale=D**-0.5, window_left=window_left,
                                             kernel_options={"BACKEND": backend})
        results.append((out, *torch.autograd.grad(out, (q, k, v, rel), dout)))
    for name, ref, got in zip(("out", "dq", "dk", "dv", "drel"), *results):
        scale = ref.abs().max().item()
        err = (got.float() - ref.float()).abs().max().item()
        assert err <= TOL * max(scale, 1.0), f"{name}: max error {err:.3e} (scale {scale:.3e})"


def test_auto_falls_back_to_triton_and_flydsl_rejects():
    @triton.jit
    def _triton_only_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx - 1

    block_mask = create_block_mask(_triton_only_mask, None, None, 256, 256)
    q2, k2, v2 = _inputs(1, 2, 2, 256, 256, 64)
    with pytest.raises(NotImplementedError, match="FlyDSL twin"):
        flex_attention(q2, k2, v2, block_mask=block_mask, kernel_options=FLYDSL)
    out = flex_attention(q2, k2, v2, block_mask=block_mask, kernel_options={"BACKEND": "AUTO"})
    keep = lambda b, h, qi, ki: qi >= ki - 1  # noqa: E731
    torch.testing.assert_close(out.float(), flex_attention_ref(q2, k2, v2, keep_fn=keep)[0], atol=TOL, rtol=TOL)
