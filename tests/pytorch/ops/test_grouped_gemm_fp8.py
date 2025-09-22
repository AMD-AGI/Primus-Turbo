###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8, grouped_gemm_fp8_blockwise
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.test_utils import compute_snr


def _check_hit_int32_limit(B, M, N, K):
    a_elems = B * M * K
    b_elems = B * N * K
    out_elems = B * M * N
    return max(a_elems, out_elems, b_elems) >= 2**31


@pytest.mark.parametrize("B", [1, 2, 3, 8, 16, 32, 64])
@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "NK",
    [
        (2048, 1536),
        (2048, 1408),
        (1408, 2048),
        (2816, 2048),
        (3072, 5120),
        (5120, 1536),
        (4096, 7168),
        (7168, 2048),
    ],
)
@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("balance", [True, False])
def test_grouped_gemm_fp8(B, M, NK, ori_dtype, format, granularity, trans_b, balance):
    N, K = NK
    device = "cuda:0"

    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, format={format}, granularity={granularity}, trans_b={trans_b}, balance={balance}"
    )

    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # Ref
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    # Turbo
    config = Float8QuantConfig(format=format, granularity=granularity)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    out.backward(grad_out)

    # Check
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


@pytest.mark.parametrize("B", [1, 2, 3, 32])
@pytest.mark.parametrize("M", [32, 256, 2048])
@pytest.mark.parametrize("NK", [(4096, 7168)])
@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("block_size", [128])  # TODO: 256 error
def test_blockwise_fp8_grouped_gemm_func(B, M, NK, ori_dtype, dtype, block_size):
    N, K = NK
    device = "cuda:0"

    print(f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, dtype={dtype}, block_size={block_size}")

    #
    dist = 0.2 + 0.8 * torch.rand(B)
    dist /= dist.sum()
    seg_lens = (dist * M * B).to(torch.long).to(device)
    error = M * B - seg_lens.sum()
    seg_lens[-1] += error
    #
    x = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    # print(x.shape, x.dtype)
    # print(w.shape, w.dtype)
    # print(seg_lens)
    # print(seg_indptr)

    # Ref
    out_ref = grouped_gemm_ref(x_ref, w_ref, seg_lens, True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    x_grad_ref = x_ref.grad
    w_grad_ref = w_ref.grad

    # Turbo
    out = grouped_gemm_fp8_blockwise(x, w, seg_lens, block_size, dtype)
    out.backward(grad_out)
    x_grad = x.grad
    w_grad = w.grad

    out_snr = compute_snr(out_ref, out)

    print("fwd")
    # print(out, out.shape)
    # print(out_ref, out.shape)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    print("dgrad")
    # print(x_grad, x_grad.shape)
    # print(x_grad_ref, x_grad_ref.shape)
    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    print("wgrad")
    # print(w_grad, w_grad.shape)
    # print(w_grad_ref, w_grad_ref.shape)
    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"
