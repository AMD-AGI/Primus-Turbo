###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType

gemm_impl_module = importlib.import_module("primus_turbo.pytorch.kernels.gemm.gemm_impl")


def _bf16(shape):
    return torch.empty(shape, dtype=torch.bfloat16, device="meta")


@pytest.mark.parametrize(
    "a_shape,trans_a,b_shape,trans_b,trans_c,expected_layout",
    [
        ((32768, 2880), False, (128256, 2880), True, False, "nt"),
        ((32768, 128256), False, (128256, 2880), False, False, "nn"),
        # Autograd requests (activations.T @ grad_out).T. The backend must
        # canonicalize it to grad_out.T @ activations, the tuned TN contract.
        ((32768, 2880), True, (32768, 128256), False, True, "tn"),
    ],
)
def test_gptoss_bf16_lm_head_roles_are_flydsl_eligible(
    monkeypatch,
    a_shape,
    trans_a,
    b_shape,
    trans_b,
    trans_c,
    expected_layout,
):
    monkeypatch.setattr(gemm_impl_module, "is_gfx950", lambda: True)
    a = _bf16(a_shape)
    b = _bf16(b_shape)
    out = _bf16((128256, 2880)) if expected_layout == "tn" else None
    inplace = expected_layout == "tn"

    assert gemm_impl_module._is_gptoss_bf16_lm_head_flydsl_case(
        a,
        trans_a,
        b,
        trans_b,
        torch.bfloat16,
        trans_c,
        inplace,
        out,
    )

    a, trans_a, b, trans_b = gemm_impl_module._canonicalize_transposed_output(a, trans_a, b, trans_b, trans_c)
    layout = ("t" if trans_a else "n") + ("t" if trans_b else "n")
    assert layout == expected_layout


def test_flydsl_backend_registered_without_autotune():
    entry = gemm_impl_module._GEMM_BACKENDS[BackendType.FLYDSL]
    assert entry.impl is gemm_impl_module.GEMMFlyDSLBackend
    assert entry.autotune is False


def test_non_lm_head_shape_stays_on_hipblaslt(monkeypatch):
    monkeypatch.setattr(gemm_impl_module, "is_gfx950", lambda: True)
    a = _bf16((256, 256))
    b = _bf16((256, 256))
    sentinel = object()
    calls = []

    def fake_hipblaslt(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(gemm_impl_module.GEMMHipBLASLtBackend, "execute", fake_hipblaslt)
    result = gemm_impl_module.GEMMFlyDSLBackend.execute(
        a=a,
        trans_a=False,
        b=b,
        trans_b=False,
        out_dtype=torch.bfloat16,
        trans_c=False,
    )

    assert result is sentinel
    assert len(calls) == 1


def test_wgrad_transposed_output_canonicalizes_to_tn(monkeypatch):
    monkeypatch.setattr(gemm_impl_module, "is_gfx950", lambda: True)
    activations = _bf16((32768, 2880))
    grad_out = _bf16((32768, 128256))
    main_grad = _bf16((128256, 2880))
    sentinel = object()
    calls = []

    def fake_flydsl(a, b, **kwargs):
        calls.append((a, b, kwargs))
        return sentinel

    kernel_module = importlib.import_module("primus_turbo.flydsl.gemm.gemm_bf16_kernel")
    monkeypatch.setattr(kernel_module, "gemm_bf16_flydsl_kernel", fake_flydsl)
    result = gemm_impl_module.GEMMFlyDSLBackend.execute(
        a=activations,
        trans_a=True,
        b=grad_out,
        trans_b=False,
        out_dtype=torch.bfloat16,
        trans_c=True,
        inplace_add_to_out=True,
        out=main_grad,
    )

    assert result is sentinel
    assert len(calls) == 1
    a, b, kwargs = calls[0]
    assert a is grad_out and b is activations
    assert kwargs == {
        "trans_a": True,
        "trans_b": False,
        "out_dtype": torch.bfloat16,
        "trans_c": False,
        "beta": 1.0,
        "out": main_grad,
    }
