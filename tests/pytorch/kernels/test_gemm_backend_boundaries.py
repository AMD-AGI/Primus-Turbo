###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Host dispatch contracts, including architectures unavailable to the GPU runner.

Device entry points are replaced only in routing tests; these assert argument
identity and output orientation, and do not claim device numerical validation.
"""

import importlib
from unittest.mock import Mock

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    check_mxfp4_support,
    float4_e2m1fn_x2,
    float8_e4m3,
)
from primus_turbo.pytorch.kernels.gemm import gemm_fp4_impl as fp4
from primus_turbo.pytorch.kernels.gemm import gemm_fp8_impl as fp8
from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_fp8_impl as group8
from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_impl as grouped


def fp4_args():
    return dict(
        a=torch.empty(128, 128, dtype=float4_e2m1fn_x2),
        b=torch.empty(192, 128, dtype=float4_e2m1fn_x2),
        a_scale_inv=torch.empty(128, 8, dtype=torch.uint8),
        b_scale_inv=torch.empty(192, 8, dtype=torch.uint8),
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE,
    )


@pytest.mark.skipif(not check_mxfp4_support()[0], reason="MXFP4 requires supported hardware")
def test_fp4_preshuffle_layout_and_accumulation_gates(monkeypatch):
    monkeypatch.setattr(fp4, "is_gfx950", lambda: True)
    monkeypatch.setattr(fp4, "is_gfx942", lambda: False)
    args = fp4_args()
    assert not fp4.GEMMFP4HipBLASLtBackend.can_handle(**args, preshuffled=True)
    for out in [None, torch.empty(128, 192, dtype=torch.int32), torch.empty(192, 128).T]:
        assert not fp4.GEMMFP4HipBLASLtBackend.can_handle(**args, inplace_add_to_out=True, out=out)
    assert fp4.GEMMFP4HipBLASLtBackend.can_handle(**args, inplace_add_to_out=True, out=torch.empty(128, 192))
    assert not fp4.GEMMFP4FlyDSLBackend.can_handle(**args, preshuffled=True)
    assert not fp4.GEMMFP4FlyDSLBackend.can_handle(**{**args, "trans_a": True})
    packed = {
        **args,
        "a_scale_inv": torch.empty(512, dtype=torch.int32),
        "b_scale_inv": torch.empty(512, dtype=torch.int32),
        "preshuffled": True,
    }
    assert fp4.GEMMFP4FlyDSLBackend.can_handle(**packed)
    assert not fp4.GEMMFP4AITERBackend.can_handle(**packed)
    assert not fp4.GEMMFP4FlyDSLBackend.can_handle(
        **{**packed, "b_scale_inv": torch.empty(511, dtype=torch.int32)}
    )
    assert fp4.GEMMFP4FlyDSLBackend.tuning_kwargs(**packed) == packed
    assert [fp4.ceil_div(x, 32) for x in (0, 1, 32, 33)] == [0, 1, 1, 2]


@pytest.mark.parametrize("n,k,expected", [(128, 128, True), (96, 128, False), (128, 96, False)])
def test_ck_blockwise_alignment(n, k, expected, monkeypatch):
    monkeypatch.setattr(fp8, "build_ck", lambda: True)
    monkeypatch.setattr(fp8, "is_gfx1250", lambda: False)
    args = dict(
        a=torch.empty(64, k, dtype=float8_e4m3),
        b=torch.empty(n, k, dtype=float8_e4m3),
        a_scale_inv=torch.ones(1),
        b_scale_inv=torch.ones(1),
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.BLOCKWISE,
    )
    assert fp8.GEMMFP8CKBackend.can_handle(**args) == expected
    with pytest.raises(ValueError, match="Unsupported granularity"):
        fp8.GEMMFP8TritonBackend.execute(**{**args, "granularity": "invalid"})


@pytest.mark.parametrize("backend", [group8.GroupedGEMMFP8CKBackend, group8.GroupedGEMMFP8HipblasltBackend])
@pytest.mark.parametrize("trans_b", [False, True])
def test_pitched_fp8_backend_declines_trimmed_output(backend, trans_b, monkeypatch):
    monkeypatch.setattr(group8, "build_ck", lambda: True)
    monkeypatch.setattr(group8, "is_gfx1250", lambda: False)
    args = dict(
        a=torch.empty(64, 128, dtype=float8_e4m3),
        b=torch.empty((2, 256, 128) if trans_b else (2, 128, 256), dtype=float8_e4m3),
        a_scales=torch.ones(1),
        b_scales=torch.ones(1),
        group_lens=torch.tensor([32, 32]),
        group_offs=torch.tensor([0, 32, 64]),
        trans_a=False,
        trans_b=trans_b,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE,
        num_cu=None,
    )
    assert backend.can_handle(**args, n_real=256)
    assert not backend.can_handle(**args, n_real=255)


@pytest.mark.parametrize("variable_k", [False, True])
def test_ck_work_steal_requires_cdna4(variable_k, monkeypatch):
    monkeypatch.setattr(grouped, "is_gfx950", lambda: False)
    args = dict(
        a=torch.empty(128, 128, dtype=torch.bfloat16),
        b=torch.empty((128, 256) if variable_k else (2, 128, 256), dtype=torch.bfloat16),
        group_lens=torch.tensor([64, 64]),
        group_offs=torch.tensor([0, 64, 128]),
        trans_a=variable_k,
        trans_b=False,
        num_cu=None,
        schedule="work_steal",
    )
    backend = grouped.GroupedGEMMVariableKCKBackend if variable_k else grouped.GroupedGEMMCKBackend
    if variable_k:
        args["trans_c"] = False
    assert not backend.can_handle(**args)


@pytest.mark.parametrize("trans_b", [False, True])
def test_rdna_grouped_forward_routes_layout_and_rejects_strides(trans_b, monkeypatch):
    device = importlib.import_module("primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250")
    monkeypatch.setattr(grouped, "is_gfx950", lambda: False)
    monkeypatch.setattr(grouped, "is_gfx1250", lambda: True)
    monkeypatch.setattr(grouped, "_num_cus", lambda device: 256)
    a = torch.empty(512, 1024, dtype=torch.bfloat16)
    b = torch.empty((2, 768, 1024) if trans_b else (2, 1024, 768), dtype=a.dtype)
    args = dict(
        a=a,
        b=b,
        group_lens=torch.tensor([256, 256]),
        group_offs=torch.tensor([0, 256, 512]),
        trans_a=False,
        trans_b=trans_b,
        num_cu=0,
    )
    gate = device.grouped_gemm_bf16_nt_supported if trans_b else device.grouped_gemm_bf16_nn_supported
    assert grouped.GroupedGEMMFlyDSLBackend.can_handle(**args) == gate(768, 1024, 256, 2)
    assert not grouped._gfx1250_grouped_gemm_shapes_ok(a.T, b, trans_b)
    assert not grouped.GroupedGEMMFlyDSLBackend.can_handle(**{**args, "num_cu": 1})
    name = "grouped_gemm_bf16_" + ("nt" if trans_b else "nn") + "_flydsl_kernel"
    result = object()
    launch = Mock(return_value=result)
    monkeypatch.setattr(device, name, launch)
    assert grouped.GroupedGEMMFlyDSLBackend.execute(**args) is result
    launch.assert_called_once_with(a, b, args["group_offs"], out_dtype=a.dtype)


@pytest.mark.parametrize("trans_c", [False, True])
def test_rdna_variable_k_routes_transpose_by_swapping_operands(trans_c, monkeypatch):
    device = importlib.import_module("primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250")
    monkeypatch.setattr(grouped, "is_gfx950", lambda: False)
    monkeypatch.setattr(grouped, "is_gfx1250", lambda: True)
    monkeypatch.setattr(grouped, "_num_cus", lambda device: 256)
    a = torch.empty(512, 768, dtype=torch.bfloat16)
    b = torch.empty(512, 1024, dtype=a.dtype)
    args = dict(
        a=a,
        b=b,
        group_lens=torch.tensor([256, 256]),
        group_offs=torch.tensor([0, 256, 512]),
        trans_a=True,
        trans_b=False,
        trans_c=trans_c,
        num_cu=0,
    )
    lhs, rhs = (b, a) if trans_c else (a, b)
    expected = device.grouped_gemm_bf16_variable_k_supported(lhs.shape[1], rhs.shape[1], 256)
    assert grouped.GroupedGEMMVariableKFlyDSLBackend.can_handle(**args) == expected
    assert not grouped.GroupedGEMMVariableKFlyDSLBackend.can_handle(**{**args, "num_cu": 1})
    result = object()
    launch = Mock(return_value=result)
    monkeypatch.setattr(device, "grouped_gemm_bf16_variable_k_flydsl_kernel", launch)
    assert grouped.GroupedGEMMVariableKFlyDSLBackend.execute(**args) is result
    launch.assert_called_once_with(
        lhs, rhs, args["group_offs"], masked_k=args["group_lens"], out_dtype=a.dtype
    )


def test_gfx942_backward_layout_stays_untransposed(monkeypatch):
    mod = importlib.import_module("primus_turbo.pytorch.ops.gemm_fp8")
    monkeypatch.setattr(mod, "is_gfx942", lambda: True)
    assert mod._deter_use_nt_layout_gemm_in_bwd(False, True) is False
