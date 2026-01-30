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
from primus_turbo.pytorch.core.utils import get_device_compute_capability
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
    cuda_graph: bool = False,
):
    """Common test logic for grouped_gemm_fp8 with different scaling granularities."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    if backend != BackendType.HIPBLASLT:
        group_lens = group_lens.to(device)
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, format={format}, "
        f"granularity={granularity}, block_size={block_size}, trans_b={trans_b}, "
        f"balance={balance}, backend={backend}, auto_tune={auto_tune}, cuda_graph={cuda_graph}"
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

    if cuda_graph:
        # CUDA graph mode: warmup -> capture -> replay
        # Warmup is REQUIRED to JIT compile Triton kernels before graph capture
        # (CUDA graphs cannot capture kernel compilation)
        out_warmup = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
        out_warmup.backward(grad_out)
        del out_warmup

        a.grad.zero_()
        b.grad.zero_()
        torch.cuda.synchronize()

        # Capture the CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
            out.backward(grad_out)

        # Replay the graph
        g.replay()
        torch.cuda.synchronize()
        del g
    else:
        # Normal mode: direct execution
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
@pytest.mark.parametrize("format", FORMAT_VALUES + [Format.HYBRID])
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_grouped_gemm_fp8_tensorwise(B, M, NK, ori_dtype, format, trans_b, balance, backend, auto_tune):
    if format == Format.HYBRID:
        # TODO(ruibin): Remove skip after CK backend supports hybrid format.
        if backend != BackendType.HIPBLASLT or auto_tune:
            pytest.skip("HYBRID format requires HIPBLASLt backend")

    # TODO(xiaobochen-amd): On gfx942, the hipBLASLt path can hang/flake when M <= 512.
    # This has been observed under pytest; root cause not yet identified. MI355 works normally.
    # Skip also when auto_tune=True because the tuner may select hipBLASLt.
    if (
        get_device_compute_capability() == (9, 4)
        and M <= 512
        and (backend is BackendType.HIPBLASLT or auto_tune is True)
    ):
        pytest.skip("gfx942: hipBLASLt path can hang/flake when M <= 512")

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


def _test_grouped_gemm_fp8_hipgraph_test(
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
):
    """Common test logic for grouped_gemm_fp8 hipgraph with different scaling granularities."""
    # Skip invalid granularity/block_size combinations
    if granularity == ScalingGranularity.BLOCKWISE and block_size is None:
        pytest.skip("BLOCKWISE granularity requires block_size to be set.")
    if granularity != ScalingGranularity.BLOCKWISE and block_size is not None:
        pytest.skip("Only BLOCKWISE granularity supports block_size.")

    seed = 33
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda:0"

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, format={format}, "
        f"granularity={granularity}, block_size={block_size}, trans_b={trans_b}, "
        f"balance={balance}"
    )

    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Ref for group_lens
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    # Generate group_lens2 with different seed (same total M, different distribution)
    seed += 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    group_lens2 = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)

    # Ref for group_lens2
    a_ref2 = a.detach().clone().requires_grad_(True)
    b_ref2 = b.detach().clone().requires_grad_(True)
    out_ref2 = grouped_gemm_ref(a_ref2, b_ref2, group_lens2, trans_b)
    out_ref2.backward(grad_out)
    torch.cuda.synchronize()

    # Turbo
    config = Float8QuantConfig(format=format, granularity=granularity, block_size=block_size)

    # Warmup both group_lens to compile all kernels
    out_warmup = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    out_warmup.backward(grad_out)
    out_warmup2 = grouped_gemm_fp8(a, b, group_lens2, trans_b=trans_b, config=config)
    out_warmup2.backward(grad_out)
    del out_warmup, out_warmup2

    a.grad.zero_()
    b.grad.zero_()
    torch.cuda.synchronize()

    # Capture the CUDA graph with ONE grouped_gemm_fp8 operation
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
        out.backward(grad_out)

    # first run
    g.replay()
    torch.cuda.synchronize()

    snr_threshold = 25 if format == Format.E4M3 else 20

    # Verify out with group_lens
    out_snr = compute_snr(out_ref, out)
    print(f"[group_lens] Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, "out_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"[group_lens] AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"[group_lens] BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"

    group_lens.copy_(group_lens2)  # In-place update

    # Reset gradients for second replay
    a.grad.zero_()
    b.grad.zero_()

    # second run, replay the same graph with updated group_lens
    g.replay()
    torch.cuda.synchronize()

    # Verify out with group_lens2
    out2_snr = compute_snr(out_ref2, out)
    print(f"[group_lens2] Out-SNR: {out2_snr:.2f} dB")
    assert out2_snr > snr_threshold, f"out2_snr too low"

    a_grad_snr2 = compute_snr(a_ref2.grad, a.grad)
    print(f"[group_lens2] AGrad-SNR: {a_grad_snr2:.2f} dB")
    assert a_grad_snr2 > snr_threshold, f"a_grad_snr2 too low"

    b_grad_snr2 = compute_snr(b_ref2.grad, b.grad)
    print(f"[group_lens2] BGrad-SNR: {b_grad_snr2:.2f} dB")
    assert b_grad_snr2 > snr_threshold, f"b_grad_snr2 too low"

    del g
    torch.cuda.synchronize()

    # Reset config and caches
    GlobalBackendManager.reset()


# NOTE: HIPGraph tests are temporarily skipped due to hipgraph issue.
# These tests require a PyTorch version upgrade to work properly with HIPGraph.
@pytest.mark.skip(reason="Requires PyTorch version upgrade for HIPGraph support")
@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("ori_dtype", ORI_DTYPE_VALUES)
@pytest.mark.parametrize("format", FORMAT_VALUES)
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", [False])
def test_grouped_gemm_fp8_tensorwise_hipgraph(B, M, NK, ori_dtype, format, trans_b, balance):
    N, K = NK
    _test_grouped_gemm_fp8_hipgraph_test(
        B=B,
        M=M,
        N=N,
        K=K,
        ori_dtype=ori_dtype,
        format=format,
        granularity=ScalingGranularity.TENSORWISE,
        trans_b=trans_b,
        balance=balance,
    )


@pytest.mark.skip(reason="Requires PyTorch version upgrade for HIPGraph support")
@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("ori_dtype", ORI_DTYPE_VALUES)
@pytest.mark.parametrize("format", FORMAT_VALUES)
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", [False])
def test_grouped_gemm_fp8_rowwise_hipgraph(B, M, NK, ori_dtype, format, trans_b, balance):
    N, K = NK
    _test_grouped_gemm_fp8_hipgraph_test(
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


@pytest.mark.skip(reason="Requires PyTorch version upgrade for HIPGraph support")
@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("ori_dtype", ORI_DTYPE_VALUES)
@pytest.mark.parametrize("format", FORMAT_VALUES)
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("trans_b", TRANS_B_VALUES)
@pytest.mark.parametrize("balance", [False])
def test_grouped_gemm_fp8_blockwise_hipgraph(B, M, NK, ori_dtype, format, block_size, trans_b, balance):
    N, K = NK
    _test_grouped_gemm_fp8_hipgraph_test(
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
