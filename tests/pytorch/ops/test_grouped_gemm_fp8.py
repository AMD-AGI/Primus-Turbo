import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.ops import grouped_gemm_fp8_blockwise
from tests.test_utils import compute_snr
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref

@pytest.mark.parametrize("B", [1, 2, 3, 32])
@pytest.mark.parametrize("M", [32, 256, 2048])
@pytest.mark.parametrize("NK", [(4096, 7168)])
@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("block_size", [128, 256])
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
