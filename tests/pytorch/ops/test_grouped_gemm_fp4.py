###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    Float4QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import (
    GroupedGEMMFP4VariableKFlyDSLBackend,
    GroupedGEMMFP4VariableKKernelDispatcher,
    GroupedGEMMFP4VariableKTritonBackend,
    grouped_gemm_fp4_variable_k_impl_meta,
)
from primus_turbo.pytorch.ops.grouped_gemm_fp4 import grouped_gemm_fp4
from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp4_with_trans
from tests.pytorch.ops.determinism_helper import (  # noqa: F401  (_det_seed: fixture)
    ACCUM_GROUPED_SHAPE,
    DET_DTYPES,
    _det_seed,
    check_determinism,
    check_fused_grad_accum,
    fp4_config,
    grouped_ref,
    make_generator,
    randn,
    require_gfx950,
    require_mxfp4,
    skip_flydsl_on_gfx942,
    unbalanced_group_lens,
)
from tests.pytorch.ops.gemm_shapes_helper import (
    GROUPED_GEMM_SHAPES,
    GROUPED_GEMM_SHAPES_SMALL,
)
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)

# Sweep parameters. MXFP4 is NT-only (trans_b=True), single E2M1 format, Triton
# backend only, so we drop those axes and sweep (B, M, N, K) tuples / dtype.
# N, K need only be multiples of MXFP4_BLOCK_SIZE (=32); the quantizer zero-pads
# the contraction dims up to 128. M is grouped along rows; the FlyDSL wgrad
# operand uses a compact 256-aligned span per group.

# 32-multiples that are NOT 128-multiples (exercises the padded-contraction +
# free-dim c_mask path): covers N-unaligned, K-unaligned, and both-unaligned.
SHAPES_UNALIGNED = [
    (2, 128, 96, 160),
    (4, 256, 160, 96),
    (8, 512, 288, 256),
    (2, 256, 256, 288),
    (4, 512, 1568, 2080),
    (8, 128, 1568, 2080),
]
DTYPE_VALUES = [torch.bfloat16, torch.float16]

# E2M1 (1-bit mantissa) is intrinsically lossy, so the SNR bar is lower than the
# FP8 suite's 20-25 dB. 8 dB cleanly separates "correct" from "broken layout".
SNR_THRESHOLD = 8.0


def _check_hit_int32_limit(B, M, N, K):
    """Skip shapes whose largest operand would overflow int32 element indexing."""
    a_elems = B * M * K
    b_elems = B * N * K
    out_elems = B * M * N
    return max(a_elems, b_elems, out_elems) >= 2**31


def _make_config():
    return Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )


def test_grouped_gemm_fp4_variable_k_dispatch_keys():
    fp4_dtype = float4_e2m1fn_x2 if float4_e2m1fn_x2 is not None else torch.uint8
    a = torch.empty((512, 2048), device="meta", dtype=fp4_dtype)
    group_lens = torch.empty((8,), device="meta", dtype=torch.int64)
    common = dict(
        a=a,
        a_scales=None,
        b_scales=None,
        group_lens=group_lens,
        group_offs=None,
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        num_cu=None,
    )

    key_n3072 = GroupedGEMMFP4VariableKKernelDispatcher.make_key(
        b=torch.empty((3072, 2048), device="meta", dtype=fp4_dtype), **common
    )
    key_n4096 = GroupedGEMMFP4VariableKKernelDispatcher.make_key(
        b=torch.empty((4096, 2048), device="meta", dtype=fp4_dtype), **common
    )
    accumulation_key = GroupedGEMMFP4VariableKKernelDispatcher.make_key(
        b=torch.empty((3072, 2048), device="meta", dtype=fp4_dtype),
        inplace_add_to_out=True,
        **common,
    )

    assert key_n3072[1:4] == (512, 3072, 2048)
    assert key_n4096[1:4] == (512, 4096, 2048)
    assert key_n3072 != key_n4096
    assert key_n3072 != accumulation_key


@pytest.mark.parametrize(
    (
        "inplace_add_to_out",
        "overwrite_out",
        "allow_overwrite",
        "graph_capturing",
        "record_ownership",
        "expected_beta",
    ),
    [
        (False, False, False, False, True, 0.0),
        (True, False, True, False, True, 1.0),
        (True, True, False, False, True, 1.0),
        (True, True, True, False, True, 0.0),
        (True, True, True, False, False, 0.0),
        (True, True, True, True, True, 1.0),
    ],
)
def test_grouped_gemm_fp4_variable_k_flydsl_beta(
    monkeypatch,
    inplace_add_to_out,
    overwrite_out,
    allow_overwrite,
    graph_capturing,
    record_ownership,
    expected_beta,
):
    from primus_turbo.pytorch.core import grad_ownership

    kernel_module_name = "primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_kernel"
    kernel_module = type(sys)(kernel_module_name)
    captured = {}

    def fake_kernel(*args, **kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    kernel_module.grouped_gemm_mxfp4_variable_k_flydsl_kernel = fake_kernel
    monkeypatch.setitem(sys.modules, kernel_module_name, kernel_module)
    recorded = []
    monkeypatch.setattr(grad_ownership, "record_overwrite", recorded.append)
    monkeypatch.setattr(
        GroupedGEMMFP4VariableKKernelDispatcher,
        "_is_graph_capturing",
        staticmethod(lambda: graph_capturing),
    )
    if overwrite_out:
        monkeypatch.setenv("PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT", "1")
    else:
        monkeypatch.delenv("PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT", raising=False)

    out = torch.empty((4, 128, 64), device="meta", dtype=torch.bfloat16) if inplace_add_to_out else None
    result = GroupedGEMMFP4VariableKFlyDSLBackend.execute(
        a=torch.empty((128, 256), device="meta", dtype=torch.uint8),
        b=torch.empty((64, 256), device="meta", dtype=torch.uint8),
        a_scales=torch.empty((128, 16), device="meta", dtype=torch.uint8),
        b_scales=torch.empty((64, 16), device="meta", dtype=torch.uint8),
        group_lens=torch.empty((4,), device="meta", dtype=torch.int64),
        group_offs=torch.empty((5,), device="meta", dtype=torch.int64),
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        num_cu=None,
        inplace_add_to_out=inplace_add_to_out,
        out=out,
        record_ownership=record_ownership,
        allow_overwrite=allow_overwrite,
    )

    assert captured["beta"] == expected_beta
    assert captured["out"] is out
    assert result is out
    should_record = (
        inplace_add_to_out and overwrite_out and allow_overwrite and not graph_capturing and record_ownership
    )
    assert recorded == ([out] if should_record else [])


def test_grouped_gemm_fp4_variable_k_dispatch_contract(monkeypatch):
    fp4_dtype = float4_e2m1fn_x2 if float4_e2m1fn_x2 is not None else torch.uint8
    monkeypatch.setattr(
        "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl.float4_e2m1fn_x2",
        fp4_dtype,
    )
    monkeypatch.setattr(
        "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl.is_gfx942", lambda: False
    )
    monkeypatch.setattr(
        "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl.is_gfx950", lambda: True
    )

    a = torch.empty((128, 256), device="meta", dtype=fp4_dtype)
    b = torch.empty((64, 256), device="meta", dtype=fp4_dtype)
    kwargs = dict(
        a=a,
        b=b,
        a_scales=torch.empty((128, 16), device="meta", dtype=torch.uint8),
        b_scales=torch.empty((64, 16), device="meta", dtype=torch.uint8),
        group_lens=torch.empty((4,), device="meta", dtype=torch.int64),
        group_offs=torch.empty((5,), device="meta", dtype=torch.int64),
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        num_cu=None,
    )

    for backend in (GroupedGEMMFP4VariableKTritonBackend, GroupedGEMMFP4VariableKFlyDSLBackend):
        assert backend.can_handle(**kwargs)
        assert not backend.can_handle(**{**kwargs, "trans_b": False})

    out = grouped_gemm_fp4_variable_k_impl_meta(
        **{
            **kwargs,
            "granularity": ScalingGranularity.MX_BLOCKWISE.value,
            "default_backend": BackendType.FLYDSL.value,
        }
    )
    assert out.shape == (4, 128, 64)
    with pytest.raises(AssertionError, match="NT only"):
        grouped_gemm_fp4_variable_k_impl_meta(
            **{
                **kwargs,
                "trans_b": False,
                "granularity": ScalingGranularity.MX_BLOCKWISE.value,
                "default_backend": BackendType.FLYDSL.value,
            }
        )


def _run(B, M, N, K, dtype):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    device = "cuda:0"
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=False).to(device)
    print(f"\nB={B}, M={M}, N={N}, K={K}, dtype={dtype}")

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = _make_config()
    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > SNR_THRESHOLD, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"


# ----------------------------------------------------------------------------
# Main sweep: fwd + dgrad + wgrad on the (B, M, N, K) tuples x dtype.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("B, M, N, K", GROUPED_GEMM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
def test_grouped_gemm_fp4_mx_blockwise(B, M, N, K, dtype):
    """MXFP4 grouped GEMM fwd + dgrad + wgrad on the Triton backend."""
    _run(B, M, N, K, dtype)


# ----------------------------------------------------------------------------
# Pre-quantized QuantizedTensor inputs.
# ----------------------------------------------------------------------------
def _run_grouped_gemm_fp4_quantized_tensor_test(B, M, N, K, dtype):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    device = "cuda:0"
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=False).to(device)
    print(f"\n[QT] B={B}, M={M}, N={N}, K={K}, dtype={dtype}")

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Match the row-wise recipes grouped_gemm_fp4 applies internally so the
    # pre-quantized operands pass check_quantized_tensor.
    a_scaling_recipe = ScalingRecipe()
    b_scaling_recipe = ScalingRecipe(use_2d_block=True)

    qt_a = QuantizedTensor.quantize(
        a,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        scaling_recipe=a_scaling_recipe,
        block_size=MXFP4_BLOCK_SIZE,
        group_lens=group_lens,
        axis=-1,
    )
    qt_b = QuantizedTensor.quantize(
        b,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        scaling_recipe=b_scaling_recipe,
        block_size=MXFP4_BLOCK_SIZE,
        axis=-1,
    )

    # Reference
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = _make_config()
    out = grouped_gemm_fp4(qt_a, qt_b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    # Check Shape
    assert out.shape == out_ref.shape
    assert qt_a.grad is not None and qt_a.grad.shape == a.shape
    assert qt_b.grad is not None and qt_b.grad.shape == b.shape

    # Check Results
    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, qt_a.grad)
    b_grad_snr = compute_snr(b_ref.grad, qt_b.grad)
    print(f"[QT] Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > SNR_THRESHOLD, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"


@pytest.mark.parametrize("B, M, N, K", GROUPED_GEMM_SHAPES_SMALL)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
def test_grouped_gemm_fp4_mx_blockwise_quantized_tensor(B, M, N, K, dtype):
    """MXFP4 grouped GEMM with pre-quantized grouped/regular QuantizedTensor inputs."""
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    _run_grouped_gemm_fp4_quantized_tensor_test(B, M, N, K, dtype)


# ----------------------------------------------------------------------------
# 32-but-not-128 N/K (padded-contraction path) + unbalanced groups.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("B, M, N, K", SHAPES_UNALIGNED)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
def test_grouped_gemm_fp4_unaligned_nk(B, M, N, K, dtype):
    """N/K are 32-multiples but not 128-multiples (+ unbalanced groups).

    Validates the padded-contraction path: the quantizer zero-pads the
    contraction to 128 (data + self-consistent E8M0 scales), the GEMM runs over
    that padded length so the zero tail contributes 0, and the free dim is masked
    by the kernel. Passing SNR also confirms the padding-region scales are not
    NaN (a 0*NaN would poison the dot)."""
    _run(B, M, N, K, dtype)


# ----------------------------------------------------------------------------
# Zero-length groups (MoE routing where some experts get no tokens).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize(
    "group_lens_values,N,K",
    [
        ((8192, 8192, 0, 0, 0, 0, 0, 0), 8192, 2048),
        ((0, 1, 255, 257, 511, 513), 256, 256),
    ],
    ids=["empty-experts", "k256-zero-even-odd"],
)
def test_grouped_gemm_fp4_zero_group_lens(dtype, group_lens_values, N, K):
    """group_lens containing zeros must not crash fwd/bwd (illegal-memory-access
    regression guard) and must stay correct on the non-empty groups.

    The K256 routing case also guards the compact producer/consumer contract:
    after 256-row alignment its groups exercise zero, one, two, and three wgrad
    K-blocks, covering both the even-pair loop and the odd-tail path.
    """
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    device = "cuda:0"

    E = len(group_lens_values)
    group_lens = torch.tensor(group_lens_values, dtype=torch.int64, device=device)
    group_offs = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), group_lens.cumsum(0)])
    total_m = int(group_lens.sum().item())
    print(f"\ngroup_lens={group_lens_values}, total_M={total_m}, N={N}, K={K}, dtype={dtype}")

    a = torch.randn((total_m, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((E, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    if N == K == 256:
        (
            row,
            row_scale,
            col,
            col_scale,
            row_lens,
            row_offs,
            col_lens,
            col_offs,
        ) = grouped_quantize_fp4_with_trans(
            a.detach(),
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            group_lens,
            group_offs,
            block_size=MXFP4_BLOCK_SIZE,
        )
        expected_col_lens = ((group_lens + 255) // 256) * 256
        expected_col_offs = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), expected_col_lens.cumsum(0)]
        )
        expected_alloc_m = (total_m + E * 256 + 255) // 256 * 256

        torch.testing.assert_close(row_lens, group_lens, rtol=0, atol=0)
        torch.testing.assert_close(row_offs, group_offs, rtol=0, atol=0)
        torch.testing.assert_close(col_lens, expected_col_lens, rtol=0, atol=0)
        torch.testing.assert_close(col_offs, expected_col_offs, rtol=0, atol=0)
        assert row.shape == (total_m, K // 2)
        assert row_scale.shape == (total_m, K // MXFP4_BLOCK_SIZE)
        assert col.shape == (K, expected_alloc_m // 2)
        assert col_scale.shape == (K, expected_alloc_m // MXFP4_BLOCK_SIZE)

    config = _make_config()
    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    # b_grad of empty experts is 0 in both ref and turbo; SNR over the full
    # tensor still reflects the populated experts.
    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > SNR_THRESHOLD, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"


# ----------------------------------------------------------------------------
# Fused gradient accumulation (Megatron main_grad written from the wgrad epilogue).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
def test_grouped_gemm_fp4_fused_grad_accum(dtype, request):
    """``fuse_bgrad_accum_pattern`` must leave ``main_grad`` holding previous + wgrad.

    FlyDSL is the only FP4 variable-K backend with the accumulate epilogue and its store
    is 16-bit, so ``main_grad`` is allocated in the weight's dtype, not Megatron's fp32.
    """
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    per_op_autotune = dtype == torch.bfloat16
    if per_op_autotune:
        GlobalBackendManager.reset()
        request.addfinalizer(GlobalBackendManager.reset)
        GlobalBackendManager.set_grouped_gemm_backend(
            precision=PrecisionType.FP4,
            auto_tune=True,
        )

    device = "cuda:0"
    B, M, N, K = 4, 256, 512, 256
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=False).to(device)
    print(f"\nB={B}, M={M}, N={N}, K={K}, dtype={dtype}")

    config = _make_config()

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn((B * M, N), dtype=dtype, device=device)
    a_fused = a.detach().clone().requires_grad_(True)
    b_fused = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Baseline: ordinary autograd, b.grad holds the weight gradient.
    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    # Fused: the wgrad is accumulated into a pre-seeded main_grad buffer.
    previous = torch.randn(b_fused.shape, dtype=dtype, device=device)
    b_fused.main_grad = previous.clone()
    b_fused.grad_added_to_main_grad = False

    out_fused = grouped_gemm_fp4(
        a_fused,
        b_fused,
        group_lens,
        trans_b=True,
        config=config,
        fuse_bgrad_accum_pattern="megatron",
    )
    out_fused.backward(grad_out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_fused, out)
    assert b_fused.grad_added_to_main_grad is True, "weight must be flagged during forward"
    assert b_fused.grad.shape == b_fused.shape, "dummy wgrad must keep the weight's shape"
    assert b_fused.grad.dtype == b_fused.dtype, "dummy wgrad must keep the weight's dtype"

    a_grad_snr = compute_snr(a.grad, a_fused.grad)
    accumulated = b_fused.main_grad.float() - previous.float()
    b_grad_snr = compute_snr(b.grad.float(), accumulated)
    print(f"AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"

    if per_op_autotune:
        # Re-run the accumulation key to cover the cache-hit path as well.
        before_cache_hit = b_fused.main_grad.clone()
        a_fused.grad = None
        b_fused.grad = None
        out_fused = grouped_gemm_fp4(
            a_fused,
            b_fused,
            group_lens,
            trans_b=True,
            config=config,
            fuse_bgrad_accum_pattern="megatron",
        )
        out_fused.backward(grad_out)
        torch.cuda.synchronize()
        cache_hit_delta = b_fused.main_grad.float() - before_cache_hit.float()
        cache_hit_snr = compute_snr(b.grad.float(), cache_hit_delta)
        assert cache_hit_snr > SNR_THRESHOLD, f"cache_hit_snr={cache_hit_snr:.2f} too low"


# CUDA-graph capturability (forward). The forward uses no D2H sync, so it is
# graph-capturable and a replay with in-place-updated group_lens must re-route.
# Only the forward is captured: fwd+bwd through autograd segfaults at capture_end
# (AccumulateGrad-on-default-stream, reproducible identically on the MXFP8 path).
@pytest.mark.parametrize("NK", [(2048, 1536), (4096, 4096), (160, 96)])
def test_grouped_gemm_fp4_cuda_graph(NK):
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    B, M = 4, 1024
    N, K = NK
    device = "cuda:0"
    torch.manual_seed(0)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=False).to(device)
    group_lens2 = generate_grouped_gemm_group_lens(B, M, balance=False).to(device)

    a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device)
    config = _make_config()

    out_ref = grouped_gemm_ref(a, b, group_lens, trans_b=True)
    out_ref2 = grouped_gemm_ref(a, b, group_lens2, trans_b=True)

    # Warmup (JIT-compiles Triton kernels; graphs can't capture compilation).
    with torch.no_grad():
        grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    g.replay()
    torch.cuda.synchronize()
    assert out.shape == out_ref.shape
    assert compute_snr(out_ref, out) > SNR_THRESHOLD

    # Replay with a different group distribution (same total M) bound in-place.
    group_lens.copy_(group_lens2)
    g.replay()
    torch.cuda.synchronize()
    assert compute_snr(out_ref2, out) > SNR_THRESHOLD


# ---------------------------------------------------------------------------
# Determinism (--deterministic-only): bit-exact across repeats on determinism_helper's harness.
# ---------------------------------------------------------------------------
# A single group, a deep K, 64- but not 256-aligned free dims (2880 / 320) that need scale
# padding.
DET_SHAPES = [(1, 1024, 2048, 1536), (8, 256, 4096, 7168), (8, 1024, 2880, 2048), (8, 256, 256, 320)]


@pytest.mark.parametrize("B, M, N, K", DET_SHAPES)
@pytest.mark.parametrize("dtype", DET_DTYPES)
@pytest.mark.parametrize(
    "backend", [None, BackendType.TRITON, BackendType.FLYDSL], ids=["default", "TRITON", "FLYDSL"]
)
@pytest.mark.deterministic
def test_det_grouped_gemm_fp4(request, det_seed, B, M, N, K, dtype, backend):
    skip_flydsl_on_gfx942(backend)
    require_mxfp4()
    if backend == BackendType.FLYDSL:
        require_gfx950("FlyDSL grouped MXFP4")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    gen = make_generator(det_seed)
    group_lens = unbalanced_group_lens(B, M, det_seed)
    a0 = randn((B * M, K), dtype, gen, normalize=True)
    b0 = randn((B, N, K), dtype, gen, normalize=True)
    g = randn((B * M, N), dtype, gen)
    config = fp4_config()

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
        mid()
        out.backward(g)
        return {"out": out, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(request.node.nodeid, run_once, grouped_ref(a0, b0, group_lens, True, g), snr_floor=8)


@pytest.mark.parametrize("B, M, N, K", DET_SHAPES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL], ids=["default", "FLYDSL"])
@pytest.mark.deterministic
def test_det_grouped_gemm_fp4_empty_expert(request, det_seed, B, M, N, K, dtype, backend):
    """The leading expert empty (a real MoE occurrence), plus the zeroed over-allocated tail.

    An empty leading expert shifts every downstream per-expert scale offset; with the 64- but
    not 256-aligned free dims -- (2880, 2048) for forward, (2048, 2880) for dgrad -- that is
    what exposed the FlyDSL packed B-scale workspace leaving its 256-row padding unwritten
    (#427). The workspace is poisoned per run like every other, so the bug breaks bit-exactness.
    """
    require_mxfp4()
    if backend == BackendType.FLYDSL and N % 64 != 0:
        pytest.skip("FlyDSL grouped MXFP4 backend requires N % 64 == 0")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    gen = make_generator(det_seed)
    group_lens = unbalanced_group_lens(B, M, det_seed)
    if B > 1:
        group_lens[-1] += group_lens[0]
        group_lens[0] = 0
    a0 = randn((B * M, K), dtype, gen, normalize=True)
    b0 = randn((B, N, K), dtype, gen, normalize=True)
    g = randn((B * M, N), dtype, gen)
    config = fp4_config()

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
        mid()
        out.backward(g)
        return {"out": out, "dgrad": a.grad, "wgrad": b.grad}

    # An empty expert's wgrad slice has no defined value to compare with the reference's zeros,
    # so its SNR covers the non-empty experts only; bit-exactness still covers all of it.
    nonempty = group_lens > 0
    ref = grouped_ref(a0, b0, group_lens, True, g)
    ref["wgrad"] = ref["wgrad"][nonempty]
    views = {"wgrad": lambda res: res["wgrad"][nonempty]}
    check_determinism(request.node.nodeid, run_once, ref, snr_floor=SNR_THRESHOLD, snr_views=views)

    # Rows past sum(group_lens) of an over-allocated input (a fixed-capacity permute) must come
    # out zero, not as whatever the caching allocator last held there.
    S, pad = int(group_lens.sum()), 224
    a_pad = randn((S + pad, K), dtype, gen)
    blocks = [torch.full((S + pad, N), 12288.0, dtype=dtype, device="cuda") for _ in range(8)]
    del blocks
    tail = grouped_gemm_fp4(a_pad, b0, group_lens, trans_b=True, config=config)[S:]
    torch.cuda.synchronize()
    assert torch.isfinite(tail).all(), "over-allocated tail non-finite"
    assert int(torch.count_nonzero(tail)) == 0, f"over-allocated tail [{S}:{S + pad}] not zeroed"


@pytest.mark.parametrize("microbatches", [1, 2], ids=["mb1", "mb2"])
@pytest.mark.parametrize("backend", [pytest.param(BackendType.FLYDSL, id="grouped_fp4-FLYDSL")])
@pytest.mark.deterministic
def test_det_fused_grad_accum(request, det_seed, backend, microbatches):
    require_mxfp4()
    require_gfx950("FlyDSL")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    config = fp4_config()
    check_fused_grad_accum(
        request.node.nodeid,
        det_seed,
        lambda a, b, group_lens: grouped_gemm_fp4(
            a, b, group_lens, trans_b=True, config=config, fuse_bgrad_accum_pattern="megatron"
        ),
        ACCUM_GROUPED_SHAPE,
        torch.bfloat16,
        8,
        microbatches,
    )
