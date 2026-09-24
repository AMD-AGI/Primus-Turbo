###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU regression tests for beta=0 selection in actual backward order."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from primus_turbo.pytorch.ops.utils import _setup_fused_grad_accum


def _parameter():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    parameter.main_grad = torch.zeros_like(parameter)
    parameter.grad_added_to_main_grad = False
    return parameter


def _claim(parameter, *, supports_overwrite=True):
    enabled, main_grad, claim = _setup_fused_grad_accum(
        parameter,
        "megatron",
        supports_overwrite=supports_overwrite,
    )
    assert enabled is True
    assert main_grad is parameter.main_grad
    return claim


def _execute(claim):
    return claim.execute(lambda overwrite: overwrite)


def test_staged_forwards_assign_beta0_in_reverse_backward_order():
    parameter = _parameter()
    first_forward = _claim(parameter)
    second_forward = _claim(parameter)

    # Pipeline schedules may stage both forwards and run the second backward
    # first. The first actual write, not the first forward, must overwrite.
    assert _execute(second_forward) is True
    assert _execute(first_forward) is False


def test_concurrent_backward_contexts_serialize_selection_through_launch(monkeypatch):
    parameter = _parameter()
    winner_claim = _claim(parameter)
    loser_claim = _claim(parameter)
    winner_launched = threading.Event()
    loser_attempting = threading.Event()
    loser_launched = threading.Event()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def launch_winner(overwrite):
        assert overwrite is True
        winner_launched.set()
        assert loser_attempting.wait(timeout=2)
        assert not loser_launched.is_set()
        return overwrite

    def run_winner():
        return winner_claim.execute(launch_winner)

    def run_loser():
        assert winner_launched.wait(timeout=2)
        loser_attempting.set()

        def launch(overwrite):
            loser_launched.set()
            return overwrite

        return loser_claim.execute(launch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(run_winner)
        loser = executor.submit(run_loser)

    assert winner.result() is True
    assert loser.result() is False
    assert loser_launched.is_set()


def test_checkpoint_recompute_can_claim_after_original_forward_is_discarded():
    parameter = _parameter()
    _discarded_forward = _claim(parameter)
    recomputed_forward = _claim(parameter)

    assert _execute(recomputed_forward) is True


def test_retained_graph_is_rejected_after_framework_reset():
    parameter = _parameter()
    retained_claim = _claim(parameter)
    assert _execute(retained_claim) is True

    parameter.grad_added_to_main_grad = False

    with pytest.raises(RuntimeError, match="stale write epoch"):
        _execute(retained_claim)


def test_old_claim_is_rejected_after_new_epoch_forward():
    parameter = _parameter()
    old_claim = _claim(parameter)

    parameter.grad_added_to_main_grad = False
    new_claim = _claim(parameter)

    with pytest.raises(RuntimeError, match="stale write epoch"):
        _execute(old_claim)
    assert _execute(new_claim) is True


def test_cuda_graph_capture_disables_overwrite_for_epoch(monkeypatch):
    parameter = _parameter()
    capture_claim = _claim(parameter)
    later_claim = _claim(parameter)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert _execute(capture_claim) is False

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert _execute(later_claim) is False


@pytest.mark.parametrize("exception", [RuntimeError, AssertionError, AttributeError])
def test_cuda_graph_capture_probe_tolerates_unavailable_runtime(monkeypatch, exception):
    parameter = _parameter()
    capture_claim = _claim(parameter)

    def unavailable():
        raise exception("CUDA unavailable")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unavailable)

    assert _execute(capture_claim) is True


def test_beta1_only_tied_weight_producer_rejects_mixed_epoch():
    parameter = _parameter()
    _claim(parameter)

    with pytest.raises(RuntimeError, match="mixing beta=0-capable and beta=1-only"):
        _claim(parameter, supports_overwrite=False)


def test_overwrite_producer_rejects_epoch_registered_by_beta1_only_producer():
    parameter = _parameter()
    assert _claim(parameter, supports_overwrite=False) is None

    with pytest.raises(RuntimeError, match="mixing beta=0-capable and beta=1-only"):
        _claim(parameter)


def test_beta1_only_producer_rejects_slice_skipped_from_previous_epoch():
    parameter = _parameter()
    previous_claim = _claim(parameter)
    assert _execute(previous_claim) is True

    parameter.grad_added_to_main_grad = False

    with pytest.raises(RuntimeError, match="may have skipped its clear"):
        _claim(parameter, supports_overwrite=False)


def test_tensor_alias_never_receives_an_overwrite_claim():
    weight = torch.ones(4, 4, requires_grad=True)
    weight.main_grad = torch.zeros_like(weight)
    weight.grad_added_to_main_grad = False

    enabled, main_grad, claim = _setup_fused_grad_accum(
        weight,
        "megatron",
        supports_overwrite=True,
    )

    assert enabled is True
    assert main_grad is weight.main_grad
    assert claim is None
