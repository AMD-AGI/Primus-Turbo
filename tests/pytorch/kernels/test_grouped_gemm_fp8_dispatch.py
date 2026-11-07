###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_fp8_impl as impl


def operands():
    return dict(
        a=torch.empty(128, 32, device="meta", dtype=float8_e4m3),
        b=torch.empty(128, 48, device="meta", dtype=float8_e4m3),
        a_scales=torch.empty((), device="meta"),
        b_scales=torch.empty((), device="meta"),
        group_lens=torch.empty(3, device="meta", dtype=torch.int64),
        group_offs=torch.empty(4, device="meta", dtype=torch.int64),
        trans_a=True,
        trans_b=False,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE,
        num_cu=None,
    )


@pytest.mark.parametrize(
    "backend", [impl.GroupedGEMMFP8VariableKCKBackend, impl.GroupedGEMMFP8VariableKHipblasltBackend]
)
@pytest.mark.parametrize("trans_c, m, n", [(False, 32, 48), (True, 48, 32)])
def test_full_pitch_backends_reject_tight_output(monkeypatch, backend, trans_c, m, n):
    monkeypatch.setattr(impl, "build_ck", lambda: True)
    monkeypatch.setattr(impl, "is_gfx1250", lambda: False)
    args = operands()
    args["trans_c"] = trans_c
    assert backend.can_handle(**args, m_real=m, n_real=n)
    assert not backend.can_handle(**args, m_real=m - 1, n_real=n)
    assert not backend.can_handle(**args, m_real=m, n_real=n - 1)


def test_ck_requires_compiled_backend(monkeypatch):
    monkeypatch.setattr(impl, "build_ck", lambda: False)
    assert not impl.GroupedGEMMFP8VariableKCKBackend.can_handle(**operands())


@pytest.mark.parametrize(
    "output, expected",
    [
        (None, False),
        (torch.empty(3, 32, 48, device="meta"), True),
        (torch.empty(3, 48, 32, device="meta").transpose(1, 2), False),
        (torch.empty(3, 32, 48, device="meta", dtype=torch.int32), False),
    ],
)
def test_hipblaslt_accumulation_requires_valid_output(output, expected):
    assert (
        impl.GroupedGEMMFP8VariableKHipblasltBackend.can_handle(
            **operands(), inplace_add_to_out=True, out=output
        )
        == expected
    )


@pytest.mark.parametrize(
    "granularity, accumulate, supported",
    [
        (ScalingGranularity.TENSORWISE, True, True),
        (ScalingGranularity.ROWWISE, False, True),
        (ScalingGranularity.ROWWISE, True, False),
        (ScalingGranularity.BLOCKWISE, False, True),
        (ScalingGranularity.BLOCKWISE, True, False),
    ],
)
def test_triton_variable_k_scaling_and_accumulation(granularity, accumulate, supported):
    args = operands()
    args["granularity"] = granularity
    assert (
        impl.GroupedGEMMFP8VariableKTritonBackend.can_handle(**args, inplace_add_to_out=accumulate)
        == supported
    )


@pytest.mark.parametrize(
    "gfx942, trans_a, expected", [(False, False, True), (False, True, False), (True, False, False)]
)
def test_triton_mx_requires_supported_arch_and_feature_major_layout(monkeypatch, gfx942, trans_a, expected):
    monkeypatch.setattr(impl, "is_gfx942", lambda: gfx942)
    args = operands()
    args.update(
        a=torch.empty(32, 128, device="meta", dtype=float8_e4m3),
        b=torch.empty(48, 128, device="meta", dtype=float8_e4m3),
        granularity=ScalingGranularity.MX_BLOCKWISE,
        trans_a=trans_a,
    )
    assert impl.GroupedGEMMFP8VariableKTritonBackend.can_handle(**args) == expected


@pytest.mark.parametrize(
    "supported_arch, contracted, expected", [(True, 128, True), (True, 127, False), (False, 128, False)]
)
def test_flydsl_mx_contract_alignment_and_arch(monkeypatch, supported_arch, contracted, expected):
    monkeypatch.setattr(impl, "is_gfx950", lambda: supported_arch)
    args = operands()
    args.update(
        a=torch.empty(32, contracted, device="meta", dtype=float8_e4m3),
        b=torch.empty(48, contracted, device="meta", dtype=float8_e4m3),
        granularity=ScalingGranularity.MX_BLOCKWISE,
        trans_a=False,
    )
    assert impl.GroupedGEMMFP8VariableKFlyDSLBackend.can_handle(**args) == expected


@pytest.mark.parametrize(
    "dtype, expected", [(torch.float32, False), (torch.float16, True), (torch.bfloat16, True)]
)
def test_flydsl_mx_accumulation_output_dtype(monkeypatch, dtype, expected):
    monkeypatch.setattr(impl, "is_gfx950", lambda: True)
    args = operands()
    args.update(
        a=torch.empty(32, 128, device="meta", dtype=float8_e4m3),
        b=torch.empty(48, 128, device="meta", dtype=float8_e4m3),
        granularity=ScalingGranularity.MX_BLOCKWISE,
        trans_a=False,
    )
    out = torch.empty(3, 32, 48, device="meta", dtype=dtype)
    assert (
        impl.GroupedGEMMFP8VariableKFlyDSLBackend.can_handle(**args, inplace_add_to_out=True, out=out)
        == expected
    )
