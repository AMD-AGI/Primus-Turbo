###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)

# Common test parameters
B_VALUES = [1, 2, 3, 8, 16, 32, 64]
M_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192]
NK_VALUES = [
    (2048, 1536),
    (2048, 1408),
    (1408, 2048),
    (2816, 2048),
    (3072, 5120),
    (5120, 1536),
    (4096, 7168),
    (7168, 2048),
]
ORI_DTYPE_VALUES = [torch.bfloat16, torch.float16]
FORMAT_VALUES = [Format.E4M3, Format.E5M2]
TRANS_B_VALUES = [True, False]
BALANCE_VALUES = [True, False]


def _check_hit_int32_limit(B, M, N, K):
    a_elems = B * M * K
    b_elems = B * N * K
    out_elems = B * M * N
    return max(a_elems, out_elems, b_elems) >= 2**31


def _run_grouped_gemm_fp8_test(
    B: int,
    M: int,
    N: int,
    K: int,
    ori_dtype: torch.dtype,
    format: Format,
    granularity: ScalingGranularity,
    trans_b: bool,
    balance: bool,
    block_size: int | None = None,
    backend: BackendType | None = None,
    auto_tune: bool = False,
):
    """Common test logic for grouped_gemm_fp8 with different scaling granularities."""
    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    # Skip invalid granularity/block_size combinations
    if granularity == ScalingGranularity.BLOCKWISE and block_size is None:
        pytest.skip("BLOCKWISE granularity requires block_size to be set.")
    if granularity != ScalingGranularity.BLOCKWISE and block_size is not None:
        pytest.skip("Only BLOCKWISE granularity supports block_size.")
    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    # Set backend and auto_tune config
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(auto_tune)

    device = "cuda:0"

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, format={format}, "
        f"granularity={granularity}, block_size={block_size}, trans_b={trans_b}, "
        f"balance={balance}, backend={backend}, auto_tune={auto_tune}"
    )

    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Ref
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    # Turbo
    config = Float8QuantConfig(format=format, granularity=granularity, block_size=block_size)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    out.backward(grad_out)

    # Check Shape
    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    # Check Results
    snr_threshold = 25 if format == Format.E4M3 else 20

    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, "out_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"

    # Reset config and caches
    GlobalBackendManager.reset()


@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("ori_dtype", ORI_DTYPE_VALUES)
@pytest.mark.parametrize("format", FORMAT_VALUES)
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_grouped_gemm_fp8_tensorwise(B, M, NK, ori_dtype, format, trans_b, balance, backend, auto_tune):

    # TODO(xiaobochen-amd): When auto tune is enabled and M < 512 (e.g. 128 or 256), autotune hangs on
    # hipblaslt backend in some cases. Root cause not yet identified, further investigation needed.
    # Note: This hang only occurs when running via pytest with auto_tune enabled and M < 512;
    # standalone execution works fine. This issue is observed on MI325, but MI355 works normally.
    if backend is None and auto_tune and M < 512:
        pytest.skip("autotune with small M hangs on hipblaslt backend")

    N, K = NK
    _run_grouped_gemm_fp8_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=ori_dtype,
        format=format,
        granularity=ScalingGranularity.TENSORWISE,
        trans_b=trans_b,
        balance=balance,
        backend=backend,
        auto_tune=auto_tune,
    )


@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("ori_dtype", ORI_DTYPE_VALUES)
@pytest.mark.parametrize("format", FORMAT_VALUES)
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
def test_grouped_gemm_fp8_rowwise(B, M, NK, ori_dtype, format, trans_b, balance):
    N, K = NK
    _run_grouped_gemm_fp8_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=ori_dtype,
        format=format,
        granularity=ScalingGranularity.ROWWISE,
        trans_b=trans_b,
        balance=balance,
    )


@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("ori_dtype", ORI_DTYPE_VALUES)
@pytest.mark.parametrize("format", FORMAT_VALUES)
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
def test_grouped_gemm_fp8_blockwise(B, M, NK, ori_dtype, format, block_size, trans_b, balance):
    N, K = NK
    _run_grouped_gemm_fp8_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=ori_dtype,
        format=format,
        granularity=ScalingGranularity.BLOCKWISE,
        trans_b=trans_b,
        balance=balance,
        block_size=block_size,
    )


# Test case for group_lens containing zeros (MoE scenario where some experts receive no tokens)
# This matches the actual bug scenario from primus_turbo_ut.py:
#   E=8, in_features=2048, out_features=8192, group_lens=[8192, 8192, 0, 0, 0, 0, 0, 0]
def test_grouped_gemm_fp8_blockwise_zero_group_lens():
    """
    Test block-wise scaling FP8 group GEMM with group_lens containing zeros.

    This reproduces the crash that occurs in MoE scenarios where some experts
    receive no tokens during routing.

    Bug: backward pass crashes with illegal memory access when group_lens contains 0.
    """
    device = "cuda:0"
    ori_dtype = torch.bfloat16

    # Match the actual bug scenario
    E = 8  # Number of experts
    K = 2048  # in_features
    N = 8192  # out_features

    # MoE routing: only first 2 experts receive tokens, rest get 0
    group_lens_list = [8192, 8192, 0, 0, 0, 0, 0, 0]
    group_lens = torch.tensor(group_lens_list, dtype=torch.int64, device=device)
    total_m = group_lens.sum().item()  # 16384

    print(f"\ngroup_lens={group_lens_list}, total_M={total_m}, N={N}, K={K}")

    B = E
    b_shape = (B, N, K)  # trans_b=True

    a = torch.randn((total_m, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Ref
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    # Turbo with BLOCKWISE scaling
    config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)  # This crashes without the fix

    # Check Shape
    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    # Check Results
    snr_threshold = 25

    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, "out_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"
