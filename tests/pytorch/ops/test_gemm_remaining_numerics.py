###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Numerical default-config, cached-FP4 and fused-MLP gradient regressions."""

import importlib

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, QuantizedTensorPair
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8TurboBackend
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_accum_impl
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = [
    pytest.mark.usefixtures("caller_thread_backward"),
    pytest.mark.skipif(not is_gfx950(), reason="Requires gfx950"),
]


@pytest.mark.parametrize("name", ["gemm", "gemm_fp4", "gemm_fp8", "grouped_gemm_fp4", "grouped_gemm_fp8"])
def test_default_gemm_config_against_fp32(name):
    torch.manual_seed(920)
    grouped = "grouped" in name
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16).requires_grad_()
    b = torch.randn(
        (2, 128, 256) if grouped else (128, 256), device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    fn = getattr(importlib.import_module("primus_turbo.pytorch.ops." + name), name)
    args = (a, b, torch.tensor([128, 128], device="cuda")) if grouped else (a, b)
    out = fn(*args, trans_b=True)
    ref = (
        torch.cat([x.float() @ w.float().T for x, w in zip(a.split(128), b)])
        if grouped
        else a.float() @ b.float().T
    )
    assert compute_snr(ref, out.float()) > (10 if "fp4" in name else 25)
    out.sum().backward()
    if name == "gemm":
        expected_a = b.float().sum(0).expand_as(a)
        expected_b = a.float().sum(0).expand_as(b)
        torch.testing.assert_close(a.grad.float(), expected_a, atol=0.15, rtol=0.02)
        torch.testing.assert_close(b.grad.float(), expected_b, atol=0.15, rtol=0.02)
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


@pytest.mark.parametrize("grouped", [False, True])
def test_cached_fp4_transposes_match_reconstructed_transposes(grouped):
    torch.manual_seed(921)
    config = Float4QuantConfig(use_gradient_sr=False)
    lens = torch.tensor([128, 128], device="cuda")
    values = [
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16),
        torch.randn((2, 128, 256) if grouped else (128, 256), device="cuda", dtype=torch.bfloat16),
    ]
    leaves = []
    pairs = []
    for i, value in enumerate(values):
        extra = {"group_lens": lens} if grouped and i == 0 else {}
        recipe = ScalingRecipe(use_2d_block=i == 1)
        q = QuantizedTensor.quantize(
            value,
            float4_e2m1fn_x2,
            config.granularity,
            axis=-1,
            block_size=32,
            scaling_recipe=recipe,
            **extra,
        ).requires_grad_()
        qt = QuantizedTensor.quantize(
            q.dequantize(),
            float4_e2m1fn_x2,
            config.granularity,
            axis=-2,
            block_size=32,
            scaling_recipe=ScalingRecipe(use_2d_block=i == 1, use_rht=i == 0),
            **extra,
        )
        leaves.append(q)
        pairs.append(QuantizedTensorPair(q, qt))
    name = "grouped_gemm_fp4" if grouped else "gemm_fp4"
    fn = getattr(importlib.import_module("primus_turbo.pytorch.ops." + name), name)

    def call(inputs):
        return fn(*inputs, *([lens] if grouped else []), trans_b=True, config=config)

    ref = call(leaves)
    out = call(pairs)
    grad = torch.randn_like(out)
    rg = torch.autograd.grad(ref, leaves, grad)
    ag = torch.autograd.grad(out, leaves, grad)
    for actual, expected in zip((out, *ag), (ref, *rg)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("trans_c", [False, True])
def test_dense_bf16_accum_autotune_and_fake_contract(trans_c):
    from primus_turbo.pytorch.kernels.gemm.gemm_impl import GEMMKernelDispatcher

    GlobalBackendManager.set_gemm_backend(None)
    GlobalBackendManager.set_auto_tune(True)
    GEMMKernelDispatcher._cache.clear()
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 192, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ b.float()
    if trans_c:
        ref = ref.T.contiguous()
    out = torch.randn_like(ref)
    initial = out.clone()
    args = (a, False, b, False, torch.float32, trans_c, out, BackendType.HIPBLASLT.value)
    gemm_accum_impl(*args)
    torch.testing.assert_close(out, initial + ref, atol=0.02, rtol=0.002)
    torch.library.opcheck(gemm_accum_impl, args, test_utils=("test_faketensor",))


def test_turbo_fp8_backend_matches_scaled_fp32():
    torch.manual_seed(924)
    values = [torch.randn(128, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    qa, qb = [
        QuantizedTensor.quantize(
            x,
            float8_e4m3,
            ScalingGranularity.MX_BLOCKWISE,
            axis=-1,
            block_size=32,
            scaling_recipe=ScalingRecipe(),
        )
        for x in values
    ]
    args = dict(
        a=qa.qdata,
        b=qb.qdata,
        a_scale_inv=qa.scale_inv,
        b_scale_inv=qb.scale_inv,
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE,
    )
    assert GEMMFP8TurboBackend.can_handle(**args)
    out = GEMMFP8TurboBackend.execute(**args)
    ref = qa.dequantize().float() @ qb.dequantize().float().T
    torch.testing.assert_close(out.float(), ref, atol=0.15, rtol=0.01)


@pytest.mark.parametrize("kind", ["dense_fp4", "grouped_fp4", "grouped_fp8"])
def test_mlp_main_grad_accumulates_two_microbatches(kind):
    torch.manual_seed(922)
    grouped = kind.startswith("grouped")
    name = kind.replace("dense_", "mlp_").replace("grouped_", "grouped_mlp_")
    fn = getattr(importlib.import_module("primus_turbo.pytorch.ops." + name), name)
    k = 896 if kind == "grouped_fp4" else 1024
    shape1 = (2, 2048, k) if grouped else (2048, k)
    shape2 = (2, k, 1024) if grouped else (k, 1024)
    w1 = (torch.randn(shape1, device="cuda") * 0.03).bfloat16().requires_grad_()
    w2 = (torch.randn(shape2, device="cuda") * 0.03).bfloat16().requires_grad_()
    originals = []
    for w in (w1, w2):
        w.main_grad = torch.randn_like(w) * 0.01
        originals.append(w.main_grad.clone())
        w.grad_added_to_main_grad = False
    expected = [v.float() for v in originals]
    for step in range(2):
        x = (torch.randn(512, k, device="cuda") * 0.1).bfloat16().requires_grad_()
        refs = [v.detach().clone().requires_grad_() for v in (x, w1, w2)]
        args = [torch.tensor([256, 256], device="cuda")] if grouped else []
        kw = (
            {
                "probs": torch.ones(512, device="cuda"),
                "activation": "silu",
                "trans_w1": True,
                "trans_w2": True,
            }
            if grouped
            else {}
        )
        if "fp4" in kind:
            kw["config"] = Float4QuantConfig(use_gradient_sr=False)
        out = fn(x, w1, w2, *args, fuse_wgrad_accum_pattern="megatron", **kw)
        ref = fn(*refs, *args, **kw)
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)
        torch.testing.assert_close(out, ref)
        assert compute_snr(refs[0].grad.float(), x.grad.float()) > 20
        for i, w in enumerate((w1, w2)):
            expected[i] = (expected[i] + refs[i + 1].grad.float()).to(w.main_grad.dtype).float()
            assert compute_snr(expected[i], w.main_grad.float()) > 20
            assert w.grad_added_to_main_grad
            w.grad = None


def test_grouped_fp8_consumes_prequantized_output_gradient():
    from primus_turbo.pytorch.core.low_precision import Format
    from tests.pytorch.ops.test_grouped_mlp_fp8 import test_grouped_mlp_fp8_prequantized_input_and_grad

    test_grouped_mlp_fp8_prequantized_input_and_grad(Format.E4M3)


def test_grouped_fp4_mlp_lazy_permutation_matches_explicit():
    from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4
    from primus_turbo.pytorch.ops.moe_gather import register_permuted_activation_seam

    torch.manual_seed(923)
    src = (torch.randn(256, 896, device="cuda") * 0.1).bfloat16()
    mapping = torch.arange(512, device="cuda", dtype=torch.int32).remainder(256).flip(0)
    probs = torch.ones(512, device="cuda")
    probs[::7] = 0
    explicit = src[mapping.long()].clone()
    explicit[probs == 0] = 0
    placeholder = torch.empty_like(explicit)
    register_permuted_activation_seam(placeholder, src, mapping, probs)
    w1 = (torch.randn(2, 1024, 896, device="cuda") * 0.03).bfloat16()
    w2 = (torch.randn(2, 896, 512, device="cuda") * 0.03).bfloat16()
    lens = torch.tensor([256, 256], device="cuda")
    expected = grouped_mlp_fp4(
        explicit, w1, w2, lens, probs=probs, activation="silu", trans_w1=True, trans_w2=True
    )
    actual = grouped_mlp_fp4(
        placeholder, w1, w2, lens, probs=probs, activation="silu", trans_w1=True, trans_w2=True
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", [BackendType.TRITON, BackendType.FLYDSL, None])
def test_grouped_fp4_transposed_variable_k_and_autotune(backend):
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import (
        GroupedGEMMFP4VariableKKernelDispatcher,
        grouped_gemm_fp4_variable_k_accum_impl,
        grouped_gemm_fp4_variable_k_impl,
    )
    from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp4_with_trans

    torch.manual_seed(925)
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(backend is None)
    GroupedGEMMFP4VariableKKernelDispatcher._cache.clear()
    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([128, 128], device="cuda")
    offs = torch.tensor([0, 128, 256], device="cuda")

    def quant(x):
        return grouped_quantize_fp4_with_trans(
            x,
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            lens,
            offs,
            block_size=32,
            scaling_recipe=ScalingRecipe(),
            scaling_recipe_for_trans=ScalingRecipe(use_rht=True),
        )

    aq, bq = quant(a), quant(b)
    args = (
        aq[2],
        bq[2],
        aq[3],
        bq[3],
        aq[6],
        aq[7],
        False,
        True,
        True,
        torch.bfloat16,
        ScalingGranularity.MX_BLOCKWISE.value,
        None,
        BackendType.FLYDSL.value,
    )
    actual = grouped_gemm_fp4_variable_k_impl(*args)
    ref = torch.stack([y.float().T @ x.float() for x, y in zip(a.split(128), b.split(128))])
    assert actual.shape == ref.shape and compute_snr(ref, actual.float()) > 8
    if backend is None:
        out = torch.randn_like(actual)
        initial = out.clone()
        grouped_gemm_fp4_variable_k_accum_impl(*args, out=out)
        assert compute_snr(ref, out.float() - initial.float()) > 8
        assert torch.isfinite(out).all()


def test_dense_fp4_pinned_accumulation_with_global_autotune_enabled():
    from tests.pytorch.ops.test_gemm_accumulation_gradients import test_existing_main_grad_accumulates

    GlobalBackendManager.set_auto_tune(True)
    test_existing_main_grad_accumulates("fp4", grouped=False, activation_grad=True)


def test_grouped_tensorwise_function_infers_missing_offsets():
    from primus_turbo.pytorch.core.low_precision import Float8QuantConfig
    from primus_turbo.pytorch.ops.grouped_gemm_fp8 import FP8GroupedGemmTensorFunc, grouped_gemm_fp8

    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([128, 128], device="cuda")
    config = Float8QuantConfig()
    actual = FP8GroupedGemmTensorFunc.apply(a, b, None, None, lens, None, True, a.dtype, config, None, None)
    expected = grouped_gemm_fp8(a, b, lens, trans_b=True, config=config)
    torch.testing.assert_close(actual, expected)


def test_grouped_blockwise_rejects_prequantized_activation():
    from primus_turbo.pytorch.core.low_precision import Float8QuantConfig
    from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8

    lens = torch.tensor([128, 128], device="cuda")
    q = QuantizedTensor.quantize(
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16),
        float8_e4m3,
        ScalingGranularity.TENSORWISE,
        axis=-1,
        group_lens=lens,
    )
    b = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="pre-quantized activation"):
        grouped_gemm_fp8(
            q,
            b,
            lens,
            trans_b=True,
            config=Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128),
        )


def test_dense_bf16_main_grad_accumulation_regression():
    from tests.pytorch.ops.test_gemm_accumulation_gradients import test_existing_main_grad_accumulates

    test_existing_main_grad_accumulates("bf16", grouped=False, activation_grad=True)
