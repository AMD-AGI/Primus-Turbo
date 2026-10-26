###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""pytest fixtures for the flex compat-layer unit tests.

These live here rather than in ``flex_test_utils`` because pytest only resolves a
fixture that is visible in a ``conftest`` or in the test module itself. Importing
them into each test module instead would work, but every test that takes one as a
parameter would then shadow the imported name (ruff ``F811``).

The autouse reset is deliberately scoped to the flex modules: ``tests/pytorch/ops``
also holds the heavily parametrised kernel suites, and there is no reason to run
flex bookkeeping around those.
"""

import pytest
import torch

_FLEX_TEST_PREFIX = "test_flex_"


def _is_flex_test(request) -> bool:
    return request.node.module.__name__.rpartition(".")[2].startswith(_FLEX_TEST_PREFIX)


@pytest.fixture(autouse=True)
def _reset_backend_overrides(request):
    """The override registry and classification caches are module-global; keep
    tests independent (cold cache each test)."""
    if not _is_flex_test(request):
        yield
        return

    from primus_turbo.pytorch.ops.attention.flex._cache import clear_classification_cache
    from primus_turbo.pytorch.ops.attention.flex._routing import clear_backend_overrides

    clear_backend_overrides()
    clear_classification_cache()
    yield
    clear_backend_overrides()
    clear_classification_cache()


@pytest.fixture
def capture_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_func(q, k, v, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["q_shape"] = tuple(q.shape)
        if kwargs.get("return_lse"):
            b, s, h, d = q.shape
            return q.clone(), torch.zeros((b, h, s), dtype=torch.float32)
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
        fake_flash_attn_func,
        raising=True,
    )
    return captured


@pytest.fixture
def capture_varlen_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
    ):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["cu_q"] = cu_seqlens_q
        captured["cu_k"] = cu_seqlens_k
        captured["max_q"] = max_seqlen_q
        captured["max_k"] = max_seqlen_k
        captured["q_shape"] = tuple(q.shape)
        captured["k_shape"] = tuple(k.shape)
        captured["v_shape"] = tuple(v.shape)
        if kwargs.get("return_lse"):
            return q.clone(), torch.zeros((q.shape[1], q.shape[0]), dtype=torch.float32)
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=True,
    )
    return captured


@pytest.fixture
def capture_fp8_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_fp8_func(q, k, v, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["q_shape"] = tuple(q.shape)
        captured["q_is_contiguous"] = q.is_contiguous()
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_fp8_func",
        fake_flash_attn_fp8_func,
        raising=True,
    )
    return captured
