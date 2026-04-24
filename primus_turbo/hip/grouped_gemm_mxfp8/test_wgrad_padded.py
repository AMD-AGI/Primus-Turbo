###############################################################################
# Correctness + bench for grouped_gemm_mxfp8_hip_variable_k_padded — the
# unbalanced-MoE HIP wgrad fallback via per-expert padded calls.
#
# Decision matrix:
#   balanced / normal-unbalanced  → use Triton variable-K (faster)
#   all-HIP code path mandate     → use padded variant (~4-5× slower)
#   few non-zero experts (<=8)    → padded variant wins (skip-empty saves G launches)
###############################################################################
from __future__ import annotations
import math
import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_variable_k_padded
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_dual_jagged


def snr(ref, out):
    ref_f = ref.float(); out_f = out.float()
    n = (ref_f - out_f).pow(2).mean().item()
    s = ref_f.pow(2).mean().item()
    return 10 * math.log10(s / max(n, 1e-30)) if n else float("inf")


SHAPES = {
    "balanced (M_g=2048)": [2048] * 32,
    "Entry -10 (max=16053)": [
        327, 105, 1843, 2724, 1150, 1798, 769, 646, 711, 462, 2019, 645, 961,
        697, 1391, 16053, 3452, 575, 693, 252, 956, 1120, 1856, 352, 899, 234,
        14682, 2483, 415, 4292, 606, 368,
    ],
    "warmup (4 non-zero experts)": [
        16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16384, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 16384, 16384, 0, 0, 0,
    ],
}


def main():
    M_total = 65536
    k, n, g = 2880, 5760, 32
    device = "cuda"
    torch.manual_seed(0)
    a_bf  = torch.randn(M_total, k, device=device, dtype=torch.bfloat16)
    go_bf = torch.randn(M_total, n, device=device, dtype=torch.bfloat16)

    print(f"{'shape':40s}  {'SNR dB':>7s}  {'HIP pad':>9s}  {'Triton':>9s}  {'pad/Tri':>9s}")
    print("-" * 85)

    for name, tokens in SHAPES.items():
        assert sum(tokens) == M_total
        group_lens = torch.tensor(tokens, dtype=torch.int64, device=device)
        cumsum = [0]
        for t in tokens: cumsum.append(cumsum[-1] + t)
        group_offs = torch.tensor(cumsum, dtype=torch.int64, device=device)

        _, _, a_col,  a_sc,  sc_offs  = quant_mxfp8_dual_jagged(a_bf, group_offs, group_lens)
        _, _, go_col, go_sc, _        = quant_mxfp8_dual_jagged(go_bf, group_offs, group_lens)

        # bf16 ref  [G, N, K]
        dB_bf = torch.zeros(g, n, k, device=device, dtype=torch.bfloat16)
        for gi in range(g):
            s_, e_ = int(group_offs[gi]), int(group_offs[gi + 1])
            if e_ > s_:
                dB_bf[gi] = (go_bf[s_:e_].float().T @ a_bf[s_:e_].float()).to(torch.bfloat16)

        def run_hip():
            return grouped_gemm_mxfp8_hip_variable_k_padded(
                a_col, go_col, a_sc, go_sc, group_offs, sc_offs,
                out_dtype=torch.bfloat16, trans_c=True,
            )
        def run_tri():
            return grouped_gemm_mxfp8_variable_k_triton_kernel(
                go_col, a_col, go_sc, a_sc, group_offs, sc_offs,
                out_dtype=torch.bfloat16,
            )
        dB_hip = run_hip()
        s_hip = snr(dB_bf, dB_hip)
        assert s_hip >= 25.0, f"correctness FAIL on {name}: {s_hip:.2f} dB"

        for _ in range(3): run_hip(); run_tri()
        torch.cuda.synchronize()
        t_hip = tbench.Timer(stmt="f()", globals={"f": run_hip}).timeit(20).mean * 1e3
        t_tri = tbench.Timer(stmt="f()", globals={"f": run_tri}).timeit(20).mean * 1e3
        print(f"{name:40s}  {s_hip:7.2f}  {t_hip:9.3f}  {t_tri:9.3f}  {t_tri/t_hip:9.3f}x")


if __name__ == "__main__":
    main()
