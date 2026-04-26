"""Single-GEMM deep stress: larger K, multiple seeds, more iters."""

import sys

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
        x, DTYPE_FP8, axis, False, False, False
    )


def dq(q, s):
    f = s.view(torch.uint8).to(torch.int32) - 127
    f = (2.0 ** f.float()).repeat_interleave(32, dim=-1)
    return q.float() * f


def test(M, N, K, N_ITERS=200, seed=2):
    torch.manual_seed(seed)
    a_hp = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    oa = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = oa[0], oa[1]
    ob = quantize_mx(b_hp, axis=1)
    b_fp8, b_s = ob[0], ob[1]
    a_f = dq(a_fp8, a_s)
    b_f = dq(b_fp8, b_s)
    ref = (a_f @ b_f.T).to(DTYPE_OUT)
    for _ in range(5):
        torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
            a_fp8, a_s, b_fp8, b_s, DTYPE_OUT, False, True, False, "MX_BLOCKWISE"
        )
    torch.cuda.synchronize()
    bad = 0
    for _ in range(N_ITERS):
        out = torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
            a_fp8, a_s, b_fp8, b_s, DTYPE_OUT, False, True, False, "MX_BLOCKWISE"
        )
        torch.cuda.synchronize()
        d = (ref.float() - out.float()).abs()
        if d.max().item() > 1.0:
            bad += 1
    print(f"M={M:5d} N={N:5d} K={K:5d} seed={seed}: {bad}/{N_ITERS} BAD")
    return bad


cfgs = [
    # (M, N, K)
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    (8192, 8192, 8192),
    (4096, 8192, 4096),
    (4096, 8192, 8192),
    (2048, 8192, 4096),
    (2048, 8192, 8192),
    (4096, 4096, 8192),
    (2048, 2048, 8192),
    (1024, 1024, 8192),
    # asymmetric
    (4096, 8192, 16384),
    (2048, 8192, 16384),
]

total_bad = 0
for M, N, K in cfgs:
    for seed in [2, 3, 5]:
        total_bad += test(M, N, K, seed=seed)

print(f"\nTotal BAD: {total_bad}")
sys.exit(1 if total_bad else 0)
