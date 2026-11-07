###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fixtures and an independent FP32 oracle for partial-gradient regression tests."""

import copy

import pytest
import torch

from primus_turbo.pytorch.core.backend import GlobalBackendManager
from tests.pytorch.test_utils import compute_snr, get_tolerances


@pytest.fixture
def caller_thread_backward(monkeypatch):
    # Run backward on the caller thread so coverage.py can trace Python callbacks.
    # Restore backend settings and autograd scheduling after each test.
    for name in (
        "_attn_backend",
        "_sparse_attn_backend",
        "_gemm_backend",
        "_grouped_gemm_backend",
        "_moe_dispatch_combine_backend",
        "_auto_tune",
    ):
        monkeypatch.setattr(GlobalBackendManager, name, copy.deepcopy(getattr(GlobalBackendManager, name)))
    GlobalBackendManager.set_auto_tune(False)
    with torch.autograd.set_multithreading_enabled(False):
        yield


def attention_reference(q, k, v, causal=True, window=-1, sink=None):
    """BSHD GQA, bottom-right causal/window mask, optional softmax sink."""
    repeats = q.shape[2] // k.shape[2]
    q, k, v = [t.float().transpose(1, 2) for t in (q, k, v)]
    k, v = [t.repeat_interleave(repeats, dim=1) for t in (k, v)]
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    nq, nk = scores.shape[-2:]
    qi = torch.arange(nq, device=q.device)[:, None] + nk - nq
    ki = torch.arange(nk, device=q.device)[None, :]
    mask = torch.ones((nq, nk), device=q.device, dtype=torch.bool)
    if causal:
        mask &= ki <= qi
    if window >= 0:
        mask &= ki >= qi - window
    scores = scores.masked_fill(~mask, float("-inf"))
    if sink is not None:
        extra = sink.float()[None, :, None, None].expand(scores.shape[:-1] + (1,))
        probs = torch.cat((scores, extra), dim=-1).softmax(-1)[..., :-1]
    else:
        probs = scores.softmax(-1)
    return (probs @ v).transpose(1, 2)


def assert_bf16_close(actual, expected, snr_floor=40):
    expected = expected.to(actual.dtype)
    torch.testing.assert_close(actual, expected, **get_tolerances(actual.dtype))
    assert compute_snr(expected, actual) > snr_floor
