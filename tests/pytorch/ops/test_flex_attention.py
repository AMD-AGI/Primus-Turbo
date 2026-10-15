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
    AuxRequest,
    causal_mask,
    create_block_mask,
    flex_attention,
    identity_score_mod_bwd,
    make_softcap_score_mod,
    noop_mask,
    sliding_window_mask,
)
from tests.pytorch.ref.flex_attention_ref import flex_attention_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")

DTYPE = torch.bfloat16
TOL = 2e-2


def _inputs(B, Hq, Hkv, Sq, Skv, D, Dv=None, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to("cuda", DTYPE).requires_grad_(True)

    return randn(B, Hq, Sq, D), randn(B, Hkv, Skv, D), randn(B, Hkv, Skv, Dv or D)


def _check(out, ref, leaves, extra_loss=None, ref_extra_loss=None):
    """Compare out, then every gradient of leaves, against the fp32 reference."""
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)
    dout = torch.randn_like(out)
    loss = (out * dout).sum() + (0 if extra_loss is None else extra_loss)
    ref_loss = (ref * dout.float()).sum() + (0 if ref_extra_loss is None else ref_extra_loss)
    grads = torch.autograd.grad(loss, leaves, retain_graph=True)
    ref_grads = torch.autograd.grad(ref_loss, leaves)
    for i, (g, r) in enumerate(zip(grads, ref_grads)):
        scale = r.abs().max().item()
        err = (g.float() - r.float()).abs().max().item()
        assert err <= 5 * TOL * max(scale, 1.0), f"grad {i}: max err {err} vs max |ref| {scale}"


# ---------------------------------------------------------------- built-in masks


def _keep_causal(b, h, q, kv):
    return q >= kv


def _keep_window(left, right):
    return lambda b, h, q, kv: (q - kv <= left) & (q - kv >= -right)


# Sequence lengths are kept to a few values -- 256 square, 384 / 640 unequal, 300 ragged
# -- because the forward compiles one kernel per distinct length.
MHA = (2, 4, 4, 256, 256, 64)
GQA = (1, 8, 2, 256, 256, 128)
MQA_RAGGED = (1, 4, 1, 300, 300, 128)
BUILTINS = {
    "none": (None, None),
    "causal": (causal_mask, _keep_causal),
    "window64": (sliding_window_mask(64), _keep_window(64, 0)),
    "window32x32": (sliding_window_mask(32, 32), _keep_window(32, 32)),
    "noop": (noop_mask, None),
}
# Each mask once, plus the causal and window paths on the ragged MQA shape.
BUILTIN_CASES = [
    ("none", MHA),
    ("causal", GQA),
    ("causal", MQA_RAGGED),
    ("window64", GQA),
    ("window64", MQA_RAGGED),
    ("window32x32", MHA),
    ("noop", GQA),
]


@pytest.mark.parametrize("name,shape", BUILTIN_CASES, ids=[f"{n}-{s}" for n, s in BUILTIN_CASES])
def test_builtin_masks(name, shape):
    mask_mod, keep_fn = BUILTINS[name]
    B, Hq, Hkv, Sq, Skv, D = shape
    q, k, v = _inputs(B, Hq, Hkv, Sq, Skv, D)
    block_mask = None if mask_mod is None else create_block_mask(mask_mod, None, None, Sq, Skv)
    if block_mask is not None:
        assert block_mask._fast_path is not None
    out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    ref, _ = flex_attention_ref(q, k, v, keep_fn=keep_fn)
    _check(out, ref, (q, k, v))


def test_causal_mask_unequal_lengths_is_top_left():
    # Sq != Skv leaves the fast path (its causal flag is bottom-right aligned) and runs
    # the mask itself, which keeps torch's top-left semantics.
    q, k, v = _inputs(1, 4, 4, 384, 640, 64)
    block_mask = create_block_mask(causal_mask, 1, 4, 384, 640, BLOCK_SIZE=64)
    assert block_mask._fast_path is None
    out = flex_attention(q, k, v, block_mask=block_mask)
    _check(out, flex_attention_ref(q, k, v, keep_fn=_keep_causal)[0], (q, k, v))


# ---------------------------------------------------------------- block-sparse masks


@triton.jit
def _band_mask(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 96) & (d > -96)


@triton.jit
def _prefix_lm_mask(b, h, q_idx, kv_idx):
    return (kv_idx < 64) | (q_idx >= kv_idx)


@triton.jit
def _head_checker_mask(b, h, q_idx, kv_idx):
    return ((q_idx // 64 + kv_idx // 64 + h) % 2) == 0


CUSTOM = [
    ("band", _band_mask, lambda b, h, q, kv: (q - kv).abs() < 96),
    ("prefix_lm", _prefix_lm_mask, lambda b, h, q, kv: (kv < 64) | (q >= kv)),
    ("head_checker", _head_checker_mask, lambda b, h, q, kv: ((q // 64 + kv // 64 + h) % 2) == 0),
]
# Every mask at both block sizes, and every shape (square, unequal, ragged) twice.
BLOCK_SPARSE_CASES = [
    (CUSTOM[0], 64, 256, 256),
    (CUSTOM[0], 128, 300, 300),
    (CUSTOM[1], 128, 384, 640),
    (CUSTOM[1], 64, 300, 300),
    (CUSTOM[2], 64, 384, 640),
    (CUSTOM[2], 128, 256, 256),
]


@pytest.mark.parametrize(
    "custom,block_size,Sq,Skv",
    BLOCK_SPARSE_CASES,
    ids=[f"{c[0]}-b{bs}-{sq}x{skv}" for c, bs, sq, skv in BLOCK_SPARSE_CASES],
)
def test_block_sparse_masks(custom, block_size, Sq, Skv):
    name, mask_mod, keep_fn = custom
    B, H = 2, 4
    q, k, v = _inputs(B, H, 2, Sq, Skv, 64)
    per_head = name == "head_checker"
    block_mask = create_block_mask(mask_mod, None, H if per_head else None, Sq, Skv, BLOCK_SIZE=block_size)

    # The visited tiles are exactly those with at least one kept position.
    dev = q.device
    keep = keep_fn(
        torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.long),
        torch.arange(block_mask.shape[1], device=dev).view(1, -1, 1, 1),
        torch.arange(Sq, device=dev).view(1, 1, Sq, 1),
        torch.arange(Skv, device=dev).view(1, 1, 1, Skv),
    ).expand(1, block_mask.shape[1], Sq, Skv)
    nq, nk = triton.cdiv(Sq, block_size), triton.cdiv(Skv, block_size)
    padded = torch.zeros(1, keep.shape[1], nq * block_size, nk * block_size, dtype=torch.bool, device=dev)
    padded[..., :Sq, :Skv] = keep
    visited = padded.view(1, keep.shape[1], nq, block_size, nk, block_size).any(dim=(3, 5))
    assert torch.equal(block_mask.to_dense(), visited)

    out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    _check(out, flex_attention_ref(q, k, v, keep_fn=keep_fn)[0], (q, k, v))


def test_create_block_mask_is_memoized(monkeypatch):
    from primus_turbo.pytorch.ops.attention import flex_attention_masks as fm

    mask = create_block_mask(_band_mask, None, None, 256, 256, BLOCK_SIZE=64)
    assert create_block_mask(_band_mask, None, None, 256, 256, BLOCK_SIZE=64) is mask
    device = f"cuda:{torch.cuda.current_device()}"
    assert create_block_mask(_band_mask, None, None, 256, 256, device=device, BLOCK_SIZE=64) is mask
    others = [
        create_block_mask(_band_mask, None, None, 256, 256, BLOCK_SIZE=128),
        create_block_mask(_band_mask, None, None, 320, 320, BLOCK_SIZE=64),
        create_block_mask(_band_mask, None, 2, 256, 256, BLOCK_SIZE=64),
        create_block_mask(causal_mask, None, None, 256, 256, BLOCK_SIZE=64),
    ]
    assert all(other is not mask for other in others)

    # A cached mask holds the same lists as a fresh build.
    fm._DENSE_MASKS.clear()
    fresh = create_block_mask(_band_mask, None, None, 256, 256, BLOCK_SIZE=64)
    assert fresh is not mask
    assert torch.equal(fresh.to_dense(), mask.to_dense())

    # Bounded, least recently used out first.
    monkeypatch.setattr(fm, "_DENSE_MASK_CACHE_SIZE", 2)
    fm._DENSE_MASKS.clear()
    a = create_block_mask(_band_mask, None, None, 128, 128, BLOCK_SIZE=64)
    b = create_block_mask(_band_mask, None, None, 192, 192, BLOCK_SIZE=64)
    assert create_block_mask(_band_mask, None, None, 128, 128, BLOCK_SIZE=64) is a
    create_block_mask(_band_mask, None, None, 256, 256, BLOCK_SIZE=64)  # evicts b
    assert create_block_mask(_band_mask, None, None, 128, 128, BLOCK_SIZE=64) is a
    assert create_block_mask(_band_mask, None, None, 192, 192, BLOCK_SIZE=64) is not b


# ---------------------------------------------------------------- mask_mod aux tensors

AUX_S = 256


@triton.jit
def _doc_causal_aux_mask(b, h, q_idx, kv_idx, DOC):
    # q_idx / kv_idx run past the sequence end in the last tile: guard the loads.
    q_doc = tl.load(DOC + b * 256 + q_idx, mask=q_idx < 256, other=-1)
    kv_doc = tl.load(DOC + b * 256 + kv_idx, mask=kv_idx < 256, other=-2)
    return (q_doc == kv_doc) & (q_idx >= kv_idx)


@triton.jit
def _prefix_window_aux_mask(b, h, q_idx, kv_idx, PREFIX, WINDOW):
    prefix = tl.load(PREFIX + b)
    window = tl.load(WINDOW + b)
    return (kv_idx < prefix) | ((q_idx >= kv_idx) & (q_idx - kv_idx <= window))


def _doc_ids(doc_lengths):
    """int32 ``[B, AUX_S]`` document ids, one row of document lengths per batch."""
    rows = [torch.repeat_interleave(torch.arange(len(ls)), torch.tensor(ls)) for ls in doc_lengths]
    return torch.stack(rows).to("cuda", torch.int32).contiguous()


@pytest.mark.parametrize("block_size", [64, 128])
def test_mask_aux_tensor_document_mask(block_size):
    # Per-batch document boundaries, not tile-aligned: the mask reads them from a tensor.
    doc = _doc_ids([[100, 156], [40, 90, 126]])
    q, k, v = _inputs(2, 4, 2, AUX_S, AUX_S, 64, seed=7)
    block_mask = create_block_mask(
        _doc_causal_aux_mask, 2, None, AUX_S, AUX_S, BLOCK_SIZE=block_size, aux_tensors=[doc]
    )
    out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    ref = flex_attention_ref(q, k, v, keep_fn=lambda b, h, qi, ki: (doc[b, qi] == doc[b, ki]) & (qi >= ki))[0]
    _check(out, ref, (q, k, v))


def test_mask_aux_tensors_two_per_batch_scalars():
    prefix = torch.tensor([32, 100], device="cuda", dtype=torch.int32)
    window = torch.tensor([200, 48], device="cuda", dtype=torch.int32)
    q, k, v = _inputs(2, 4, 4, AUX_S, AUX_S, 64, seed=8)
    block_mask = create_block_mask(
        _prefix_window_aux_mask, 2, None, AUX_S, AUX_S, BLOCK_SIZE=64, aux_tensors=[prefix, window]
    )
    out = flex_attention(q, k, v, block_mask=block_mask)

    def keep(b, h, qi, ki):
        return (ki < prefix[b]) | ((qi >= ki) & (qi - ki <= window[b]))

    _check(out, flex_attention_ref(q, k, v, keep_fn=keep)[0], (q, k, v))


def test_mask_aux_masks_are_rebuilt_not_memoized():
    doc_a = _doc_ids([[128, 128], [128, 128]])
    doc_b = _doc_ids([[64, 192], [192, 64]])
    a = create_block_mask(_doc_causal_aux_mask, 2, None, AUX_S, AUX_S, BLOCK_SIZE=64, aux_tensors=[doc_a])
    a2 = create_block_mask(_doc_causal_aux_mask, 2, None, AUX_S, AUX_S, BLOCK_SIZE=64, aux_tensors=[doc_a])
    b = create_block_mask(_doc_causal_aux_mask, 2, None, AUX_S, AUX_S, BLOCK_SIZE=64, aux_tensors=[doc_b])
    assert a is not a2 and torch.equal(a.to_dense(), a2.to_dense())
    assert not torch.equal(a.to_dense(), b.to_dense())


def test_mask_aux_validation():
    doc = _doc_ids([[128, 128]])
    with pytest.raises(ValueError, match="built-in"):
        create_block_mask(causal_mask, None, None, AUX_S, AUX_S, aux_tensors=[doc])
    with pytest.raises(ValueError, match="not differentiable"):
        create_block_mask(
            _doc_causal_aux_mask, None, None, AUX_S, AUX_S, aux_tensors=[doc.float().requires_grad_(True)]
        )
    with pytest.raises(ValueError, match="contiguous"):
        create_block_mask(_doc_causal_aux_mask, None, None, AUX_S, AUX_S, aux_tensors=[doc.expand(2, -1)[:, ::2]])


# ---------------------------------------------------------------- execution-path selection


@pytest.fixture
def path_tuning(monkeypatch):
    """Path tuning on, though the suite runs with autotuning off (see tests/conftest.py)."""
    from primus_turbo.pytorch.ops.attention import flex_attention_interface as fi

    monkeypatch.setattr(fi, "AUTOTUNE", "on")


@pytest.mark.parametrize(
    "name,mask_mod,keep_fn",
    [("window64", sliding_window_mask(64), _keep_window(64, 0)), CUSTOM[1]],
    ids=["builtin_window", "custom_prefix_lm"],
)
def test_every_forward_backward_path_pair(path_tuning, name, mask_mod, keep_fn):
    # The forward and backward are tuned independently, so any pairing must be exact.
    from primus_turbo.pytorch.ops.attention import flex_attention_interface as fi

    q, k, v = _inputs(1, 4, 2, 300, 300, 64)
    block_mask = create_block_mask(mask_mod, None, None, 300, 300)
    cands = fi._candidates(block_mask, num_splits=1)
    assert len(cands) >= 2
    key = fi._tune_key(block_mask, q, k, v, 1, None, False)
    try:
        for fwd in cands:
            for bwd in cands:
                fi._TUNED[key] = (fwd, bwd)
                out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
                _check(out, flex_attention_ref(q, k, v, keep_fn=keep_fn)[0], (q, k, v))
    finally:
        fi._TUNED.pop(key, None)


@triton.jit
def _band_mask_for_tuning(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 64) & (d > -64)


def test_first_call_tunes_and_caches_both_passes(path_tuning):
    from primus_turbo.pytorch.ops.attention import flex_attention_interface as fi
    from primus_turbo.pytorch.ops.attention import flex_attention_masks as fm

    q, k, v = _inputs(1, 4, 4, 256, 256, 64)
    block_mask = create_block_mask(_band_mask_for_tuning, None, None, 256, 256)
    key = fi._tune_key(block_mask, q, k, v, 1, None, False)
    with torch.no_grad():
        flex_attention(q, k, v, block_mask=block_mask)
    fwd, bwd = fi._TUNED[key]
    assert fwd is not None and bwd is None  # no gradient needed yet
    # A rebuilt mask (a new object, past the create_block_mask cache) with the same
    # mask_mod and shapes reuses the decision.
    fm._DENSE_MASKS.clear()
    rebuilt = create_block_mask(_band_mask_for_tuning, None, None, 256, 256)
    assert rebuilt is not block_mask
    flex_attention(q, k, v, block_mask=rebuilt).sum().backward()
    fwd2, bwd2 = fi._TUNED[key]
    assert fwd2 is fwd and bwd2 is not None


def test_untuned_default_paths():
    from primus_turbo.pytorch.kernels.flex_attention.flex_attention_heuristic import (
        MaskPath,
        default_paths,
    )

    sparse = [MaskPath(f"bs{s}", block_size=s) for s in (64, 128)]
    fwd, bwd = default_paths(sparse, 128)
    assert (fwd.block_size, bwd.block_size) == (128, 64)
    fast = MaskPath("fast", causal=True)
    assert default_paths([fast] + sparse, 128) == (fast, fast)


# ---------------------------------------------------------------- score mods


@triton.jit
def _alibi_score_mod(score, b, h, q_idx, kv_idx):
    return score - (h + 1) * 0.05 * (q_idx - kv_idx).to(tl.float32)


def test_additive_score_mod_with_causal():
    q, k, v = _inputs(2, 4, 4, 256, 256, 64)
    out = flex_attention(
        q,
        k,
        v,
        score_mod=_alibi_score_mod,
        score_mod_bwd=identity_score_mod_bwd,
        block_mask=create_block_mask(causal_mask, None, None, 256, 256),
    )

    def score_fn(s, b, h, qi, ki):
        return s - (h + 1) * 0.05 * (qi - ki).float()

    ref, _ = flex_attention_ref(q, k, v, score_fn=score_fn, keep_fn=_keep_causal)
    _check(out, ref, (q, k, v))


def test_softcap_score_mod():
    q, k, v = _inputs(1, 4, 4, 256, 256, 128)
    cap = 2.0
    score_mod, score_mod_bwd = make_softcap_score_mod(cap)
    out = flex_attention(q, k, v, score_mod=score_mod, score_mod_bwd=score_mod_bwd, scale=0.5)
    ref, _ = flex_attention_ref(q, k, v, score_fn=lambda s, *_: cap * torch.tanh(s / cap), scale=0.5)
    _check(out, ref, (q, k, v))


# ---------------------------------------------------------------- aux outputs / options


def test_lse_is_returned_and_differentiable():
    q, k, v = _inputs(2, 4, 2, 256, 256, 64)
    block_mask = create_block_mask(causal_mask, None, None, 256, 256)
    out, aux = flex_attention(
        q, k, v, block_mask=block_mask, enable_gqa=True, return_aux=AuxRequest(lse=True)
    )
    ref, ref_lse = flex_attention_ref(q, k, v, keep_fn=_keep_causal)
    torch.testing.assert_close(aux.lse, ref_lse, atol=TOL, rtol=TOL)
    w = torch.randn_like(ref_lse)
    _check(out, ref, (q, k, v), extra_loss=(aux.lse * w).sum(), ref_extra_loss=(ref_lse * w).sum())


def test_num_splits_short_query_long_context():
    q, k, v = _inputs(1, 8, 8, 16, 4096, 128)
    out = flex_attention(q, k, v, kernel_options={"num_splits": 8})
    _check(out, flex_attention_ref(q, k, v)[0], (q, k, v))


def test_mla_head_dims():
    q, k, v = _inputs(1, 4, 4, 256, 256, 192, Dv=128)
    out = flex_attention(q, k, v, block_mask=create_block_mask(causal_mask, None, None, 256, 256))
    _check(out, flex_attention_ref(q, k, v, keep_fn=_keep_causal)[0], (q, k, v))


# ---------------------------------------------------------------- aux tensors + gradient hook

PAIR = 384  # per-pair bias table [B, H, PAIR, PAIR], for sequences up to 300


@triton.jit
def _pair_bias_score_mod(score, b, h, q_idx, kv_idx, BIAS):
    offset = ((b * 2 + h) * 384 + q_idx) * 384 + kv_idx
    return score + tl.load(BIAS + offset, mask=(q_idx < 300) & (kv_idx < 300), other=0.0)


@triton.jit
def _pair_bias_grad_hook(dscore, valid, b, h, q_idx, kv_idx, DBIAS, BIAS):
    offset = ((b * 2 + h) * 384 + q_idx) * 384 + kv_idx
    tl.store(DBIAS + offset, dscore, mask=valid)


def test_score_grad_hook_returns_captured_tensor_gradient():
    # Every (b, h, q, kv) has its own bias entry, so the hook's output must equal
    # autograd's gradient of the bias, zeros at masked and padded positions included.
    B, H, S = 2, 2, 300
    q, k, v = _inputs(B, H, H, S, S, 64, seed=3)
    bias = (torch.randn(B, H, PAIR, PAIR, device="cuda") * 2).requires_grad_(True)
    out = flex_attention(
        q,
        k,
        v,
        score_mod=_pair_bias_score_mod,
        score_mod_bwd=identity_score_mod_bwd,
        block_mask=create_block_mask(causal_mask, None, None, S, S),
        aux_tensors=[bias.detach()],
        score_grad_hook=_pair_bias_grad_hook,
        score_grad_target=bias,
    )
    ref, _ = flex_attention_ref(q, k, v, score_fn=lambda s, *_: s + bias[:, :, :S, :S], keep_fn=_keep_causal)
    _check(out, ref, (q, k, v, bias))


# ---------------------------------------------------------------- validation


def test_validation_errors():
    q, k, v = _inputs(1, 4, 2, 128, 128, 64)
    with pytest.raises(ValueError, match="enable_gqa"):
        flex_attention(q, k, v)
    with pytest.raises(NotImplementedError, match="max_scores"):
        flex_attention(q, k, v, enable_gqa=True, return_aux=AuxRequest(max_scores=True))
    with pytest.raises(ValueError, match="score_mod_bwd"):
        flex_attention(q, k, v, enable_gqa=True, score_mod=_alibi_score_mod)
    with pytest.raises(ValueError, match="kernel_options"):
        flex_attention(q, k, v, enable_gqa=True, kernel_options={"BLOCK_M": 64})
    with pytest.raises(ValueError, match="does not match"):
        flex_attention(
            q, k, v, enable_gqa=True, block_mask=create_block_mask(causal_mask, None, None, 256, 256)
        )
    with pytest.raises(ValueError, match="square"):
        create_block_mask(_band_mask, None, None, 128, 128, BLOCK_SIZE=(64, 128))
    bias = torch.zeros(4, 128, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="requires grad"):
        flex_attention(
            q,
            k,
            v,
            enable_gqa=True,
            score_mod=_alibi_score_mod,
            score_mod_bwd=identity_score_mod_bwd,
            aux_tensors=[bias],
        )
