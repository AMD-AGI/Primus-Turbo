###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sparse MLA with optional sink: independent attention oracle and fake outputs."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention.sparse_mla_impl import sparse_mla_bwd_impl, sparse_mla_fwd_impl
from primus_turbo.pytorch.ops.attention.sparse_mla_interface import sparse_mla_func
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = [
    pytest.mark.usefixtures("caller_thread_backward"),
    pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950"),
]


@pytest.mark.parametrize("sink_enabled", [False, True])
@pytest.mark.parametrize("backend", [BackendType.TRITON, BackendType.FLYDSL, None])
def test_sparse_mla_reference_and_fake(backend, sink_enabled, monkeypatch):
    monkeypatch.setattr(GlobalBackendManager, "_sparse_attn_backend", None)
    torch.manual_seed(819)
    GlobalBackendManager.set_sparse_attn_backend(backend)
    GlobalBackendManager.set_auto_tune(backend is None)
    q = torch.randn(64, 64, 576, device="cuda", dtype=torch.bfloat16)
    q.requires_grad_()
    kv = torch.randn(64, 576, device="cuda", dtype=torch.bfloat16)
    kv.requires_grad_()
    indices = torch.stack([torch.randperm(64, device="cuda")[:64] for _ in range(64)]).int()
    indices[:, -4:] = -1
    sink = torch.randn(64, device="cuda", requires_grad=True) if sink_enabled else None
    qr, kr = q.detach().float().requires_grad_(), kv.detach().float().requires_grad_()
    sr = sink.detach().clone().requires_grad_() if sink_enabled else None
    gathered = kr[indices.clamp_min(0).long()]
    scale = 512**-0.5
    scores = torch.einsum("thd,tkd->thk", qr, gathered) * scale
    scores = scores.masked_fill(indices[:, None, :] < 0, -torch.inf)
    if sink_enabled:
        scores = torch.cat((scores, sr[None, :, None].expand(64, -1, -1)), dim=-1)
    lse_ref = scores.logsumexp(-1)
    probs = scores.softmax(-1)[..., :64]
    ref = torch.einsum("thk,tkd->thd", probs, gathered[..., :512])
    actual = sparse_mla_func(q, kv, indices, attn_sink=sink, kv_lora_rank=512, scale=scale)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    ref.backward(grad.float())
    pairs = [(actual, ref), (q.grad, qr.grad), (kv.grad, kr.grad)]
    for result, expected in [
        (q.grad[..., 512:], qr.grad[..., 512:]),
        (kv.grad[..., 512:], kr.grad[..., 512:]),
    ]:
        assert expected.abs().max() > 0
        assert compute_snr(expected, result.float()) > 35
    if sink_enabled:
        pairs.append((sink.grad, sr.grad))
    for result, expected in pairs:
        assert result.shape == expected.shape
        assert torch.isfinite(result).all()
        assert compute_snr(expected, result.float()) > 35
    args = (
        q.detach(),
        kv.detach(),
        indices,
        sink.detach() if sink_enabled else None,
        512,
        scale,
        BackendType.FLYDSL.value,
    )
    out, lse = sparse_mla_fwd_impl(*args)
    torch.testing.assert_close(lse, lse_ref, rtol=0.002, atol=0.02)
    torch.library.opcheck(sparse_mla_fwd_impl, args, test_utils=("test_faketensor",))
    bwd_args = (args[0], args[1], out, grad, indices, lse, *args[3:])
    torch.library.opcheck(sparse_mla_bwd_impl, bwd_args, test_utils=("test_faketensor",))


def test_32_heads_uses_triton_fallback(monkeypatch):
    from primus_turbo.pytorch.kernels.attention.sparse_mla_impl import (
        SparseMlaFwdDispatcher,
        SparseMlaFwdFlydslBackend,
    )

    monkeypatch.setattr(GlobalBackendManager, "_sparse_attn_backend", None)
    q = torch.zeros(64, 32, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.ones(64, 576, device="cuda", dtype=torch.bfloat16)
    kv[..., 512:] = 0
    kv.requires_grad_()
    indices = torch.arange(64, device="cuda", dtype=torch.int32).expand(64, -1).contiguous()
    assert not SparseMlaFwdFlydslBackend.can_handle(q=q, topk_indices=indices)
    assert (
        SparseMlaFwdDispatcher.resolve(BackendType.FLYDSL, None, q=q, kv=kv, topk_indices=indices)
        is BackendType.TRITON
    )
    out = sparse_mla_func(q, kv, indices)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
    out.sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), atol=1e-3, rtol=0)
    expected_dkv = torch.full_like(kv, 32)
    expected_dkv[..., 512:] = 0
    torch.testing.assert_close(kv.grad, expected_dkv, atol=1e-3, rtol=0)
    GlobalBackendManager.set_sparse_attn_backend(BackendType.FLYDSL)
    with pytest.raises(ValueError, match="cannot handle"):
        sparse_mla_func(q, kv, indices)
