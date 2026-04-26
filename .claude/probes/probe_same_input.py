import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(x, DTYPE_FP8, axis, False, False, False)


def dequantize_mxfp8(q, s):
    s_f = s.view(torch.uint8).to(torch.int32) - 127
    s_f = (2.0 ** s_f.float()).repeat_interleave(32, dim=-1)
    return q.float() * s_f


torch.manual_seed(2)
G = 4
n, k = 8192, 2048
total_m = 4096
a_hp = torch.randn(total_m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
b_hp = torch.randn(G, n, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
out_a = quantize_mx(a_hp, axis=1)
a_fp8, a_s = out_a[0], out_a[1]
out_b = quantize_mx(b_hp.reshape(G * n, k), axis=1)
b_fp8 = out_b[0].reshape(G, n, k)
b_s = out_b[1].reshape(G, n, -1)
lens_t = torch.tensor([1024] * 4, dtype=torch.int64, device=DEVICE)
offs_t = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(lens_t)

a_f = dequantize_mxfp8(a_fp8, a_s)
b_f = dequantize_mxfp8(b_fp8.reshape(-1, k), b_s.reshape(-1, k // 32)).reshape(b_fp8.shape)
ref = torch.empty(total_m, n, dtype=DTYPE_OUT, device=DEVICE)
cum = 0
for g in range(G):
    Mg = int(lens_t[g].item())
    ref[cum : cum + Mg].copy_((a_f[cum : cum + Mg] @ b_f[g].T).to(DTYPE_OUT))
    cum += Mg

N_T = 100
bad = 0
for t in range(N_T):
    out = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
        a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
    )
    torch.cuda.synchronize()
    d = (ref.float() - out.float()).abs()
    if d.max().item() > 1.0:
        bad += 1
        print(f"trial {t}: BAD max_d={d.max().item():.3f}")
print(f"\nResult: {bad}/{N_T} BAD")
