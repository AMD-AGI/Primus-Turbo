###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Numerical checks for frozen inputs, sink gradients and ragged attention."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.ops import flash_attn_func, flash_attn_varlen_func
from tests.pytorch.ops.gradient_contract_helpers import (
    attention_reference,
    caller_thread_backward,  # noqa: F401
)
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


def check(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
    assert compute_snr(expected.float(), actual.float()) > 35


@pytest.mark.parametrize("backend", [BackendType.AITER, BackendType.FLYDSL])
@pytest.mark.parametrize("grad_inputs", ["q", "kv", "qkvs", "s"])
@pytest.mark.parametrize("window", [-1, 64])
def test_dense_partial_gradients(backend, grad_inputs, window):
    if backend is BackendType.FLYDSL and get_device_compute_capability() != (9, 5):
        pytest.skip("This FlyDSL backward regression exercises gfx950")
    torch.manual_seed(713)
    GlobalBackendManager.set_attn_backend(backend)
    # Permuted BSHD views preserve SBHD storage, including a nontrivial batch stride.
    inputs = [
        torch.randn(128, 2, h, 64, device="cuda", dtype=torch.bfloat16)
        .transpose(0, 1)
        .detach()
        .requires_grad_(name in grad_inputs)
        for name, h in zip("qkv", (4, 2, 2))
    ]
    sink = torch.randn(4, device="cuda", requires_grad="s" in grad_inputs)
    refs = [x.detach().float().requires_grad_(x.requires_grad) for x in inputs + [sink]]
    out = flash_attn_func(*inputs, causal=True, window_size=(window, 0), sink=sink)
    ref = attention_reference(*refs[:3], window=window, sink=refs[3])
    check(out, ref)
    # Noncontiguous cotangent catches backward paths which require a dense buffer.
    grad = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    out.backward(grad)
    ref.backward(grad.float())
    for actual, expected in zip(inputs + [sink], refs):
        if actual.requires_grad:
            check(actual.grad, expected.grad)
        else:
            assert actual.grad is None


@pytest.mark.parametrize(
    "backend,causal", [(BackendType.AITER, True), (BackendType.AITER, False), (BackendType.FLYDSL, True)]
)
@pytest.mark.parametrize("grad_inputs", ["q", "kv", "qkv"])
def test_varlen_partial_gradients(backend, grad_inputs, causal):
    if backend is BackendType.FLYDSL:
        if get_device_compute_capability() != (9, 5) or not causal:
            pytest.skip("gfx950 FlyDSL varlen requires causal attention")
    torch.manual_seed(714)
    GlobalBackendManager.set_attn_backend(backend)
    # FlyDSL supports uniform packed sequences; AITER also exercises ragged ones.
    lengths = [128, 128] if backend is BackendType.FLYDSL else [65, 127]
    total = sum(lengths)
    cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    inputs = [
        torch.randn(total, h, 64, device="cuda", dtype=torch.bfloat16).requires_grad_(name in grad_inputs)
        for name, h in zip("qkv", (4, 2, 2))
    ]
    refs = [x.detach().float().requires_grad_(x.requires_grad) for x in inputs]
    out = flash_attn_varlen_func(*inputs, cu, cu, max(lengths), max(lengths), causal=causal)
    ref = torch.cat(
        [
            attention_reference(*(x[start : start + n][None] for x in refs), causal=causal)[0]
            for start, n in zip([0, lengths[0]], lengths)
        ]
    )
    check(out, ref)
    grad = torch.randn(4, total, 64, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    out.backward(grad)
    ref.backward(grad.float())
    for actual, expected in zip(inputs, refs):
        if actual.requires_grad:
            check(actual.grad, expected.grad)
        else:
            assert actual.grad is None
