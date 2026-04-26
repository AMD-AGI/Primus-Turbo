"""1000-iter race stress for both single and grouped MXFP8 GEMM.

Covers the historically-racy configs:
  - Grouped: G=4 [512]×4, [8192]×4, G=8 [2048]×8, uneven [4096,2048,1024,512]
  - Single:  large square (M=N=K=8192) + asymmetric (large M, small N or K)
"""

import sys

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16
N_ITERS = 1000


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
        x, DTYPE_FP8, axis, False, False, False
    )


def dq(q, s):
    f = s.view(torch.uint8).to(torch.int32) - 127
    f = (2.0 ** f.float()).repeat_interleave(32, dim=-1)
    return q.float() * f


def grouped(G, lens_list, n=8192, k=2048, seed=2):
    torch.manual_seed(seed)
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
    for _ in range(N_ITERS):
        out = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
            a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
        )
        torch.cuda.synchronize()
        d = (ref.float() - out.float()).abs()
        if d.max().item() > 1.0:
            bad += 1
    print(f"GG  G={G} {str(lens_list)[:30]:30s} k={k:5d} seed={seed}: {bad:4d}/{N_ITERS} BAD")
    return bad


def single(M, N, K, seed=2):
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
    print(f"S   M={M:5d} N={N:5d} K={K:5d}              seed={seed}: {bad:4d}/{N_ITERS} BAD")
    return bad


total = 0

print("=== Grouped GEMM (1000 iters each) ===")
for cfg in [
    (4, [512] * 4),                # historically the racey-est small-M_g
    (4, [1024] * 4),
    (4, [2048] * 4),
    (4, [4096] * 4),
    (4, [8192] * 4),               # historically worst large-K race
    (8, [2048] * 8),               # historically worst many-group race
    (4, [4096, 2048, 1024, 512]),  # uneven
    (2, [8192] * 2),
]:
    total += grouped(*cfg)

print("\n=== Single GEMM (1000 iters each) ===")
for cfg in [
    (8192, 8192, 2048),
    (8192, 8192, 8192),  # max work per kernel
    (4096, 8192, 8192),
    (2048, 8192, 8192),
    (4096, 8192, 16384),  # large K = many K-iter barriers
]:
    total += single(*cfg)

print(f"\n=== TOTAL: {total} BAD across all configs ===")
sys.exit(1 if total else 0)
