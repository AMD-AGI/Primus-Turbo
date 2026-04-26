"""Same seed, multiple repeats: is the bad SNR deterministic per input,
or does the SAME input flip pass/fail across runs?

If deterministic per input → input-triggered (specific scale combos crash MFMA)
If non-deterministic → true race (timing-dependent, between launches/blocks)
"""

import math

import torch

import primus_turbo  # noqa
import primus_turbo.pytorch  # noqa

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(x, DTYPE_FP8, axis, False, False, False)


def dequantize_mxfp8(q, s):
    s_f = s.view(torch.uint8).to(torch.int32) - 127
    s_f = (2.0 ** s_f.float()).repeat_interleave(32, dim=-1)
    return q.float() * s_f


def snr_db(ref, out):
    diff = ref.float() - out.float()
    return 10.0 * math.log10(((ref.float() ** 2).mean().item() + 1e-30) / ((diff**2).mean().item() + 1e-30))


def gemm_only(a_fp8, b_fp8, a_s, b_s):
    """Just call the GEMM op, no input regen."""
    return torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
        a_fp8, a_s, b_fp8, b_s, DTYPE_OUT, False, True, False, "MX_BLOCKWISE"
    )


def test_seed(m, n, k, seed, reps=5):
    """For ONE seed (one input), call GEMM `reps` times and check SNR each time.
    If SNR varies across reps → true race.
    """
    torch.manual_seed(seed)
    a_hp = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    out_a = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = out_a[0], out_a[1]
    out_b = quantize_mx(b_hp, axis=1)
    b_fp8, b_s = out_b[0], out_b[1]
    a_f = dequantize_mxfp8(a_fp8, a_s)
    b_f = dequantize_mxfp8(b_fp8, b_s)
    ref = (a_f @ b_f.T).to(DTYPE_OUT)
    snrs = []
    for _ in range(reps):
        out = gemm_only(a_fp8, b_fp8, a_s, b_s)
        torch.cuda.synchronize()
        snrs.append(snr_db(ref, out))
    return snrs


if __name__ == "__main__":
    print("=== Same seed, repeated GEMM calls — is per-input output deterministic? ===\n")
    for m in [2048, 4096, 8192]:
        print(f"M={m} N=8192 K=2048:")
        for seed in range(5):
            snrs = test_seed(m, 8192, 2048, seed, reps=5)
            uniq = len(set(round(s, 1) for s in snrs))
            print(f"  seed={seed} 5x calls: {[f'{s:5.1f}' for s in snrs]}  uniq={uniq}")
        print()
