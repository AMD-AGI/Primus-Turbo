import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(x, DTYPE_FP8, axis, False, False, False)


def dq(q, s):
    f = s.view(torch.uint8).to(torch.int32) - 127
    f = (2.0 ** f.float()).repeat_interleave(32, dim=-1)
    return q.float() * f


def test(G, lens_list, n=8192, k=2048, N=200):
    torch.manual_seed(2)
    total_m = sum(lens_list)
    a_hp = torch.randn(total_m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(G, n, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    oa = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = oa[0], oa[1]
    ob = quantize_mx(b_hp.reshape(G * n, k), axis=1)
    b_fp8 = ob[0].reshape(G, n, k)
    b_s = ob[1].reshape(G, n, -1)
    lens_t = torch.tensor(lens_list, dtype=torch.int64, device=DEVICE)
    offs_t = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(lens_t)
    a_f = dq(a_fp8, a_s)
    b_f = dq(b_fp8.reshape(-1, k), b_s.reshape(-1, k // 32)).reshape(b_fp8.shape)
    ref = torch.empty(total_m, n, dtype=DTYPE_OUT, device=DEVICE)
    cum = 0
    for g in range(G):
        Mg = int(lens_t[g].item())
        ref[cum : cum + Mg].copy_((a_f[cum : cum + Mg] @ b_f[g].T).to(DTYPE_OUT))
        cum += Mg
    bad = 0
    for t in range(N):
        out = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
            a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
        )
        torch.cuda.synchronize()
        d = (ref.float() - out.float()).abs()
        if d.max().item() > 1.0:
            bad += 1
    print(f"G={G} {lens_list}: {bad}/{N} BAD")


for cfg in [
    (4, [256] * 4),
    (4, [512] * 4),
    (4, [1024] * 4),
    (4, [2048] * 4),
    (4, [4096] * 4),
    (4, [8192] * 4),
    (8, [2048] * 8),
    (2, [8192] * 2),
    (4, [4096, 2048, 1024, 512]),
]:
    test(*cfg)
