###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus_turbo.pytorch.core.backend import AutoKernelDispatcher, TuneCache
from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_utils as utils


@pytest.mark.parametrize(
    "lengths, total, expected",
    [([], 10, []), ([0, 9, 0], 9, [3, 3, 3]), ([0, 10, 0], 10, [4, 3, 3]), ([7, 1, 0], 0, [0, 0, 0])],
)
def test_balanced_groups_preserve_total_and_input(lengths, total, expected):
    lens = torch.tensor(lengths, dtype=torch.int64)
    balanced = utils._lb_group_lens(lens, total)
    assert balanced.tolist() == expected
    assert lens.tolist() == lengths
    if lengths:
        assert balanced.sum().item() == total
        assert balanced.max() - balanced.min() <= 1
    else:
        assert balanced is lens


@pytest.fixture(
    params=[utils.BaseGroupedGEMMKernelDispatcher, utils.BaseGroupedGEMMVariableKKernelDispatcher]
)
def dispatcher(request, monkeypatch):
    # Avoid leaving test dispatcher classes in the process-wide production registry.
    monkeypatch.setattr(AutoKernelDispatcher, "_subclasses", [])

    class Dispatcher(request.param):
        @classmethod
        def make_key(cls, **kwargs):
            return "shape"

    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(
        utils, "group_offs_from_lens", lambda lens: torch.cat([lens.new_zeros(1), lens.cumsum(0)])
    )
    return Dispatcher


def operands():
    return dict(
        a=torch.empty(10, 10),
        b=torch.empty(10, 10),
        group_lens=torch.tensor([0, 10, 0]),
        group_offs=torch.tensor([0, 0, 10, 10]),
        group_offs_out=torch.tensor([0, 0, 10, 10]),
        trans_a=True,
        trans_b=False,
        trans_c=False,
    )


def test_tuning_uses_balanced_copies_but_checks_original_workload(dispatcher):
    args = operands()
    candidates = [Mock(name=name) for name in ("slow", "fast", "broken", "unsupported", "disabled")]
    slow, fast, broken, unsupported, disabled = candidates
    for impl in candidates:
        impl.can_handle.return_value = impl is not unsupported
    dispatcher._backends = {
        i: SimpleNamespace(impl=impl, autotune=impl is not disabled) for i, impl in enumerate(candidates)
    }
    observed = []

    def profile(impl, **kwargs):
        observed.append(kwargs)
        assert kwargs["group_lens"].tolist() == [4, 3, 3]
        assert kwargs["group_offs"].tolist() == [0, 4, 7, 10]
        if issubclass(dispatcher, utils.BaseGroupedGEMMKernelDispatcher):
            assert kwargs["group_offs_out"].tolist() == [0, 4, 7, 10]
        if impl is broken:
            raise RuntimeError("unavailable backend")
        return 1 if impl is fast else 5

    dispatcher.profile = staticmethod(profile)
    assert dispatcher.tune(**args) is fast
    assert len(observed) == 3
    for impl in (slow, fast, broken, unsupported):
        assert impl.can_handle.call_args.kwargs["group_lens"] is args["group_lens"]
    disabled.can_handle.assert_not_called()
    assert args["group_lens"].tolist() == [0, 10, 0]
    assert dispatcher.tune(**args) is fast
    assert len(observed) == 3  # cached winner avoids profiling


@pytest.mark.parametrize("cache", [True, False])
def test_tuning_failure_does_not_cache_a_backend(dispatcher, cache):
    dispatcher._cache = TuneCache() if cache else None
    impl = Mock()
    impl.can_handle.return_value = True
    dispatcher._backends = {0: SimpleNamespace(impl=impl, autotune=True)}
    dispatcher.profile = Mock(side_effect=RuntimeError("profile failed"))
    assert dispatcher.tune(**operands()) is None
    assert dispatcher.tune(**operands()) is None
    assert dispatcher.profile.call_count == 2


@pytest.mark.parametrize(
    "trans_a, trans_b, trans_c, total",
    [(True, False, False, 10), (False, False, False, 7), (True, False, True, 11), (True, True, True, 13)],
)
def test_variable_k_tuning_respects_transposed_output(monkeypatch, trans_a, trans_b, trans_c, total):
    monkeypatch.setattr(AutoKernelDispatcher, "_subclasses", [])

    class Dispatcher(utils.BaseGroupedGEMMVariableKKernelDispatcher):
        @classmethod
        def make_key(cls, **kwargs):
            return "test"

    balanced = Mock(wraps=utils._lb_group_lens)
    monkeypatch.setattr(utils, "_lb_group_lens", balanced)
    monkeypatch.setattr(utils, "group_offs_from_lens", lambda lens: lens.cumsum(0))
    args = operands()
    args.update(
        a=torch.empty(10, 7), b=torch.empty(11, 13), trans_a=trans_a, trans_b=trans_b, trans_c=trans_c
    )
    assert Dispatcher.tune(**args) is None
    assert balanced.call_args.args[1] == total
