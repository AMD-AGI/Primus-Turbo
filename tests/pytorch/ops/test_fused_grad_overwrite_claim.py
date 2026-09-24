###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU regression tests for beta=0 selection in actual backward order."""

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


def test_staged_forwards_assign_beta0_in_reverse_backward_order():
    parameter = _parameter()
    first_forward = _claim(parameter)
    second_forward = _claim(parameter)

    # Pipeline schedules may stage both forwards and run the second backward
    # first. The first actual write, not the first forward, must overwrite.
    assert second_forward.claim() is True
    assert first_forward.claim() is False


def test_checkpoint_recompute_can_claim_after_original_forward_is_discarded():
    parameter = _parameter()
    _discarded_forward = _claim(parameter)
    recomputed_forward = _claim(parameter)

    assert recomputed_forward.claim() is True


def test_retained_graph_falls_back_to_beta1_after_framework_reset():
    parameter = _parameter()
    retained_claim = _claim(parameter)
    assert retained_claim.claim() is True

    parameter.grad_added_to_main_grad = False

    assert retained_claim.claim() is False


def test_beta1_only_tied_weight_producer_disables_overwrite_for_epoch():
    parameter = _parameter()
    overwrite_capable = _claim(parameter)
    assert _claim(parameter, supports_overwrite=False) is None

    assert overwrite_capable.claim() is False


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
