###############################################################################
# Test the hybrid HIP+Triton autograd on real production gpt_oss_20B MoE
# tokens-per-expert distributions from gem_shape_summary.txt.
#
# All entries share: M_total=65536, K=2880, N=5760, G=32, trans_b=False.
# Token distributions vary from balanced to highly unbalanced.
###############################################################################
from __future__ import annotations
import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8.autograd import (
    grouped_gemm_mxfp8_hip,
    prequant_mxfp8_weights_hip,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_rowwise


# Real tokens-per-expert from /mnt/vast/john/rocm-dynamo/turbo/gem_shape_summary.txt
SHAPES = {
    "balanced (uniform M_g=2048)": [2048] * 32,
    "Entry -10 (real, includes M_g=16053)": [
        327, 105, 1843, 2724, 1150, 1798, 769, 646, 711, 462, 2019, 645, 961,
        697, 1391, 16053, 3452, 575, 693, 252, 956, 1120, 1856, 352, 899, 234,
        14682, 2483, 415, 4292, 606, 368,
    ],
    "Entry -9 (real, very unbalanced)": [
        20, 262, 137, 967, 8954, 216, 183, 336, 32, 350, 95, 4549, 15903, 922,
        328, 323, 2047, 278, 6100, 201, 407, 2420, 1011, 9594, 417, 8682, 77,
        57, 289, 134, 77, 168,
    ],
    "Entry -8 (real, very unbalanced)": [
        124, 53, 701, 313, 790, 13623, 604, 490, 9173, 286, 3808, 11600, 220,
        654, 77, 212, 1933, 44, 626, 91, 1235, 268, 543, 15, 2589, 94, 2177,
        1890, 169, 10388, 147, 599,
    ],
    "Entry -7 (real, more uniform)": [
        3851, 3535, 3804, 468, 1460, 11248, 451, 769, 2452, 2190, 1693, 2117,
        1204, 534, 2160, 1960, 3732, 2438, 1627, 2638, 1147, 1359, 1870, 116,
        2155, 1467, 130, 2721, 439, 642, 1041, 2118,
    ],
    "Entry -6 (real, more uniform)": [
        3213, 2872, 3070, 688, 1574, 9224, 643, 1138, 3433, 3816, 1305, 2367,
        1403, 896, 3147, 2092, 3430, 2477, 1351, 2310, 787, 1086, 1391, 264,
        2362, 1563, 205, 2654, 493, 969, 1169, 2144,
    ],
    "Entry -5 (real, more uniform)": [
        3407, 3789, 4091, 808, 1523, 8829, 659, 1184, 2952, 2079, 1752, 1771,
        1852, 914, 1940, 2421, 3507, 1838, 1592, 2662, 1101, 1181, 1648, 328,
        2740, 1756, 209, 2293, 465, 835, 1126, 2284,
    ],
    "Entry warmup (catastrophic — 4 experts hold all)": [
        16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16384, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 16384, 16384, 0, 0, 0,
    ],
}


def snr(ref, out):
    ref_f = ref.float(); out_f = out.float()
    n = (ref_f - out_f).pow(2).mean().item()
    s = ref_f.pow(2).mean().item()
    return 10 * math.log10(s / max(n, 1e-30)) if n else float("inf")


def all_m_g_aligned(tokens_per_expert: list[int], align: int = 16) -> bool:
    return all(t > 0 and (t % align == 0) for t in tokens_per_expert)


def bf16_ref_step(a_bf, b_bf, group_offs, grad_out):
    """Per-expert bf16 fwd + bwd reference."""
    g = b_bf.shape[0]
    out = torch.zeros(a_bf.shape[0], b_bf.shape[1], device=a_bf.device, dtype=torch.bfloat16)
    grad_a = torch.zeros_like(a_bf)
    grad_b = torch.zeros_like(b_bf)
    for gi in range(g):
        s_, e_ = int(group_offs[gi]), int(group_offs[gi + 1])
        if e_ <= s_:
            continue
        a_g = a_bf[s_:e_].float()
        b_g = b_bf[gi].float()
        dy_g = grad_out[s_:e_].float()
        out[s_:e_]   = (a_g @ b_g.T).to(torch.bfloat16)
        grad_a[s_:e_] = (dy_g @ b_g).to(torch.bfloat16)
        grad_b[gi]   = (dy_g.T @ a_g).to(torch.bfloat16)
    return out, grad_a, grad_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--no-correctness", action="store_true",
                    help="skip the bf16 reference (slow on big shapes)")
    args = ap.parse_args()

    M_total, K, N, G = 65536, 2880, 5760, 32
    device = "cuda"
    torch.manual_seed(0)

    # Shared inputs across shapes (same M_total/K/N).
    a_bf = torch.randn(M_total, K, device=device, dtype=torch.bfloat16)
    b_bf = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)

    print(f"Shape: M={M_total}, K={K}, N={N}, G={G}, trans_b=False (B layout [G, N, K])")
    print(f"{'config':50s} {'aligned':>8s} {'out_dB':>8s} {'gA_dB':>8s} {'gB_dB':>8s} "
          f"{'hip_ms':>8s} {'pre_ms':>8s} {'tri_ms':>8s} {'pre_x':>6s}")
    print("-" * 120)

    for name, tokens in SHAPES.items():
        assert sum(tokens) == M_total, f"{name}: tokens sum {sum(tokens)} != {M_total}"
        group_lens = torch.tensor(tokens, dtype=torch.int64, device=device)
        # group_offs prefix sum
        cumsum = [0]
        for t in tokens:
            cumsum.append(cumsum[-1] + t)
        group_offs = torch.tensor(cumsum, dtype=torch.int64, device=device)
        aligned = all_m_g_aligned(tokens, 16)

        # Run hybrid autograd
        a_run = a_bf.detach().requires_grad_(True)
        b_run = b_bf.detach().requires_grad_(True)
        try:
            out = grouped_gemm_mxfp8_hip(a_run, b_run, group_lens, group_offs)
            grad_out = torch.randn_like(out)
            grad_a, grad_b = torch.autograd.grad(out, [a_run, b_run], grad_out)
            ok_run = True
        except Exception as e:
            print(f"{name:50s} {str(aligned):>8s} {'EXC':>8s}  {type(e).__name__}: {e}")
            continue

        # Correctness vs bf16 ref (only for moderate shapes; can be skipped via --no-correctness)
        if not args.no_correctness:
            out_bf, ga_bf, gb_bf = bf16_ref_step(
                a_bf, b_bf, group_offs.cpu(), grad_out
            )
            s_out = snr(out_bf, out)
            s_ga  = snr(ga_bf, grad_a)
            s_gb  = snr(gb_bf, grad_b)
        else:
            s_out = s_ga = s_gb = float("nan")

        # Bench: hybrid (HIP+Triton) step / hybrid+prequant / pure Triton step
        if args.no_bench:
            t_hip = t_pre = t_tri = float("nan")
        else:
            def step_hip():
                a2 = a_bf.detach().requires_grad_(True)
                b2 = b_bf.detach().requires_grad_(True)
                o = grouped_gemm_mxfp8_hip(a2, b2, group_lens, group_offs)
                torch.autograd.grad(o, [a2, b2], grad_out)

            prequant = prequant_mxfp8_weights_hip(b_bf.detach())
            def step_prequant():
                a2 = a_bf.detach().requires_grad_(True)
                o = grouped_gemm_mxfp8_hip(a2, prequant, group_lens, group_offs)
                torch.autograd.grad(o, [a2], grad_out)

            # Pure Triton step for comparison: use the existing Primus-Turbo MX
            # autograd Function (FP8GroupedGemmMXFunc) which is all-Triton.
            from primus_turbo.pytorch.ops.grouped_gemm_fp8 import FP8GroupedGemmMXFunc
            from primus_turbo.pytorch.core.low_precision import (
                Float8QuantConfig, ScalingGranularity, ScaleDtype,
            )
            cfg = Float8QuantConfig(granularity=ScalingGranularity.MX_BLOCKWISE,
                                    block_size=32, scale_dtype=ScaleDtype.E8M0)
            # Triton path uses B in [G, K, N] (trans_b=False); transpose b_bf.
            b_bf_tri = b_bf.transpose(1, 2).contiguous()
            def step_triton():
                a2 = a_bf.detach().requires_grad_(True)
                b2 = b_bf_tri.detach().requires_grad_(True)
                o = FP8GroupedGemmMXFunc.apply(a2, b2, group_lens, group_offs,
                                                False, cfg, None,
                                                None, None, None, None)
                torch.autograd.grad(o, [a2, b2], grad_out)

            for _ in range(3):
                step_hip(); step_prequant(); step_triton()
            torch.cuda.synchronize()
            t_hip = tbench.Timer(stmt="f()", globals={"f": step_hip}).timeit(args.iters).mean * 1e3
            t_pre = tbench.Timer(stmt="f()", globals={"f": step_prequant}).timeit(args.iters).mean * 1e3
            t_tri = tbench.Timer(stmt="f()", globals={"f": step_triton}).timeit(args.iters).mean * 1e3

        speedup = (t_tri / t_pre) if t_pre and t_pre > 0 else float("nan")
        print(f"{name:50s} {str(aligned):>8s} {s_out:8.2f} {s_ga:8.2f} {s_gb:8.2f} "
              f"{t_hip:8.3f} {t_pre:8.3f} {t_tri:8.3f} {speedup:6.2f}")

    print()
    print("Legend:")
    print("  aligned = all per-expert M_g % 16 == 0  (HIP fwd/dgrad path; else Triton fallback)")
    print("  hip_ms  = hybrid step (HIP+Triton or full Triton fallback)")
    print("  pre_ms  = hybrid step with MXFP8WeightPrequantHip (B quant lifted out of fwd)")
    print("  tri_ms  = pure-Triton FP8GroupedGemmMXFunc step (baseline)")
    print("  pre_x   = tri_ms / pre_ms  (>1 = hybrid+prequant beats pure Triton)")
    print("  Wgrad: always Triton (HIP wgrad v1 needs balanced; jagged scales handle unbalanced).")


if __name__ == "__main__":
    main()
