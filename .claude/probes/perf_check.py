import time

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(x, DTYPE_FP8, axis, False, False, False)


def bench(G, lens_list, n=8192, k=2048, iters=200):
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
    # warmup
    for _ in range(20):
        torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
            a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
        )
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(iters):
        torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
            a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - s) / iters * 1e6  # us
    flops = (
        2 * total_m * n * k
    )  # one GEMM per group, sum of m*n*k across groups; with single weight per group, =sum(m_g)*n*k = total_m*n*k
    tflops = flops / elapsed / 1e6
    print(
        f"G={G} {lens_list[:3]}{'…' if len(lens_list)>3 else ''} (total_m={total_m},n={n},k={k}): {elapsed:.1f} us  {tflops:.1f} TFLOPS"
    )


for cfg in [
    (1, [8192]),
    (2, [8192] * 2),
    (4, [8192] * 4),
    (4, [2048] * 4),
    (4, [1024] * 4),
    (4, [512] * 4),
    (8, [2048] * 8),
]:
    bench(*cfg)
