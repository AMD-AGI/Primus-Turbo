"""Bisect: is G the trigger, or m-tiles per group?

If G=4 with M_g=8192 (large) is also broken → it's about G (concurrent blocks)
If G=4 only fails when M_g <=4096 → it's about per-group tile count
"""

import math

import torch

import primus_turbo  # noqa
import primus_turbo.pytorch  # noqa

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def snr_db(ref, out):
    diff = ref.float() - out.float()
    return 10.0 * math.log10(((ref.float() ** 2).mean().item() + 1e-30) / ((diff**2).mean().item() + 1e-30))


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(x, DTYPE_FP8, axis, False, False, False)


def dequantize_mxfp8(q, s):
    s_f = s.view(torch.uint8).to(torch.int32) - 127
    s_f = (2.0 ** s_f.float()).repeat_interleave(32, dim=-1)
    return q.float() * s_f


def reference(a_fp8, b_fp8, a_s, b_s, lens, total_m, n, k):
    a_f = dequantize_mxfp8(a_fp8, a_s)
    b_f = dequantize_mxfp8(b_fp8.reshape(-1, k), b_s.reshape(-1, k // 32)).reshape(b_fp8.shape)
    out = torch.empty(total_m, n, dtype=DTYPE_OUT, device=DEVICE)
    cum = 0
    for g in range(len(lens)):
        Mg = int(lens[g].item())
        if Mg == 0:
            continue
        out[cum : cum + Mg].copy_((a_f[cum : cum + Mg] @ b_f[g].T).to(DTYPE_OUT))
        cum += Mg
    return out


def run_one(group_lens, n, k, seed=0):
    torch.manual_seed(seed)
    G = len(group_lens)
    total_m = sum(group_lens)
    a_hp = torch.randn(total_m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(G, n, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    out_a = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = out_a[0], out_a[1]
    out_b = quantize_mx(b_hp.reshape(G * n, k), axis=1)
    b_fp8 = out_b[0].reshape(G, n, k)
    b_s = out_b[1].reshape(G, n, -1)
    lens_t = torch.tensor(group_lens, dtype=torch.int64, device=DEVICE)
    offs_t = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(lens_t)
    out = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
        a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
    )
    torch.cuda.synchronize()
    ref = reference(a_fp8, b_fp8, a_s, b_s, lens_t, total_m, n, k)
    return snr_db(ref, out)


def repeat(label, lens, reps=5):
    snrs = [run_one(lens, 8192, 2048, seed=s) for s in range(reps)]
    bad = sum(1 for s in snrs if s < 70)
    print(f"  {label:48s} {[f'{s:5.1f}' for s in snrs]}  bad={bad}/{reps}")


if __name__ == "__main__":
    print("=== G vs M_g per group ===\n")
    print("Vary G with FIXED M_g=8192 per group (large):")
    repeat("G=1 [8192]", [8192])
    repeat("G=2 [8192]*2", [8192] * 2)
    repeat("G=3 [8192]*3", [8192] * 3)
    repeat("G=4 [8192]*4", [8192] * 4)
    print()
    print("Vary M_g with FIXED G=4:")
    repeat("G=4 [256]*4", [256] * 4)
    repeat("G=4 [512]*4", [512] * 4)
    repeat("G=4 [1024]*4", [1024] * 4)
    repeat("G=4 [2048]*4", [2048] * 4)
    repeat("G=4 [4096]*4", [4096] * 4)
    repeat("G=4 [8192]*4", [8192] * 4)
