import torch

from primus_turbo.pytorch.ops import grouped_gemm_fp8_blockwise
from tests.test_utils import compute_snr

# def seg_lens_to_seg_indptr(seg_lens: torch.Tensor) -> torch.Tensor:
#     return torch.cat([torch.tensor([0], device=seg_lens.device), seg_lens.cumsum(0)])


def grouped_gemm_ref(a, b, seg_lens, trans_b=True):
    seg_lens = seg_lens.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(seg_lens):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


def test_blockwise_fp8_grouped_gemm_func():
    ori_dtype = torch.bfloat16
    device = "cuda:0"
    trans_b = True
    block_size = 128
    dtype = torch.float8_e4m3fnuz

    # DeepSeek-V3
    # E = 32
    # M = 256
    # N = 4096
    # K = 7168

    # Simply
    E = 2
    M = 256
    N = 1024
    K = 2048

    dist = 0.2 + 0.8 * torch.rand(E)
    dist /= dist.sum()
    seg_lens = (dist * M * E).to(torch.long).to(device)
    error = M * E - seg_lens.sum()
    seg_lens[-1] += error

    x = torch.randn((E * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    if trans_b:
        w = torch.randn((E, N, K), dtype=ori_dtype, device=device, requires_grad=True)
    else:
        w = torch.randn((E, K, N), dtype=ori_dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    # print(x.shape, x.dtype)
    # print(w.shape, w.dtype)
    # print(seg_lens)
    # print(seg_indptr)

    # Ref
    out_ref = grouped_gemm_ref(x_ref, w_ref, seg_lens, trans_b)
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

    print("\nfwd")
    print(out, out.shape)
    print(out_ref, out.shape)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    print("dgrad")
    print(x_grad, x_grad.shape)
    print(x_grad_ref, x_grad_ref.shape)
    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    print("wgrad")
    print(w_grad, w_grad.shape)
    print(w_grad_ref, w_grad_ref.shape)
    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"
