###############################################################################
# End-to-end fwd + bwd perf comparison on gpt_oss_20B MoE gate_up shape.
#
# Covers:
#   - Kernel-only: fwd, dgrad, wgrad (HIP vs Triton)
#   - Per-step: full autograd with and without MXFP8WeightPrequant
#   - Pure-Triton baseline vs hybrid (HIP fwd+dgrad+wgrad with Triton fallback)
###############################################################################
from __future__ import annotations
import argparse, math
import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import (
    grouped_gemm_mxfp8_hip_fwd,
    grouped_gemm_mxfp8_hip_dgrad,
    grouped_gemm_mxfp8_hip_variable_k,
)
from primus_turbo.hip.grouped_gemm_mxfp8.autograd import (
    grouped_gemm_mxfp8_hip, prequant_mxfp8_weights_hip,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    quant_mxfp8_dual_jagged, quant_mxfp8_rowwise, quant_mxfp8_weight_dgrad,
)


def bench_ms(fn, iters=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    return tbench.Timer(stmt="f()", globals={"f": fn}).timeit(iters).mean * 1e3


def snr(ref, out):
    ref_f = ref.float(); out_f = out.float()
    n = (ref_f - out_f).pow(2).mean().item()
    s = ref_f.pow(2).mean().item()
    return 10 * math.log10(s / max(n, 1e-30)) if n else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=65536)
    ap.add_argument("--k", type=int, default=2880)
    ap.add_argument("--n", type=int, default=5760)
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    m, k, n, g = args.m, args.k, args.n, args.g
    m_per = m // g
    assert m % g == 0
    device = "cuda"
    torch.manual_seed(0)

    print(f"Shape: M={m}, K={k}, N={n}, G={g}, M_g={m_per}, balanced")
    print(f"FLOPs per GEMM (fwd/dgrad/wgrad): 2*{m}*{k}*{n} = {2*m*k*n/1e12:.3f} TFLOP\n")

    a_bf = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_bf = torch.randn(g, n, k, device=device, dtype=torch.bfloat16)   # NT layout
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)
    group_lens = torch.full((g,), m_per, dtype=torch.int64, device=device)

    # ── Pre-quant all operands for kernel-only bench ──
    a_fp8, a_scale = quant_mxfp8_rowwise(a_bf)
    b_flat_fp8, b_flat_scale = quant_mxfp8_rowwise(b_bf.reshape(g * n, k))
    b_fp8 = b_flat_fp8.view(g, n, k)              # [G, N, K] NT for HIP fwd
    b_scale = b_flat_scale.view(g, n, k // 32)
    # dgrad layout: B as [G, K, N]
    b_dgrad_in = b_bf.transpose(1, 2).contiguous()
    b_fp8_dgrad, b_scale_dgrad = quant_mxfp8_weight_dgrad(b_dgrad_in)
    # For Triton dgrad: takes grad_out_fp8 + b_fp8_dgrad; we'll make grad_out_fp8 here
    grad_out_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    go_fp8, go_scale = quant_mxfp8_rowwise(grad_out_bf)
    # Dual-jagged for wgrad inputs
    _, _, a_col, a_scale_col, a_sc_offs = quant_mxfp8_dual_jagged(a_bf, group_offs, group_lens)
    _, _, go_col, go_scale_col, _       = quant_mxfp8_dual_jagged(grad_out_bf, group_offs, group_lens)

    flops = 2.0 * m * k * n

    # ═══════ Kernel-only bench ═══════
    print("═" * 85)
    print(f"{'Kernel-only bench':40s}  {'ms':>8s}  {'TFLOPS':>8s}  {'vs Triton':>10s}")
    print("─" * 85)

    # fwd: HIP vs Triton
    def hip_fwd():
        return grouped_gemm_mxfp8_hip_fwd(a_fp8, b_fp8, a_scale, b_scale, group_offs)
    def tri_fwd():
        return grouped_gemm_mxfp8_triton_kernel(
            a_fp8, b_fp8, a_scale, b_scale, group_offs,
            trans_b=True, out_dtype=torch.bfloat16)

    t = bench_ms(tri_fwd, args.iters); print(f"{'Triton fwd':40s}  {t:8.3f}  {flops/t/1e9:8.1f}  {1.0:10.3f}")
    t_tri_fwd = t
    t = bench_ms(hip_fwd, args.iters); print(f"{'HIP    fwd (256 tile)':40s}  {t:8.3f}  {flops/t/1e9:8.1f}  {t_tri_fwd/t:10.3f}x")

    # dgrad: HIP reuses fwd with dgrad-layout B [G, K, N]
    def hip_dgrad():
        return grouped_gemm_mxfp8_hip_dgrad(
            go_fp8, b_fp8_dgrad, go_scale, b_scale_dgrad, group_offs,
            out_dtype=torch.bfloat16)
    def tri_dgrad():
        return grouped_gemm_mxfp8_triton_kernel(
            go_fp8, b_fp8_dgrad, go_scale, b_scale_dgrad, group_offs,
            trans_b=True, out_dtype=torch.bfloat16)

    t = bench_ms(tri_dgrad, args.iters); print(f"{'Triton dgrad':40s}  {t:8.3f}  {flops/t/1e9:8.1f}  {1.0:10.3f}")
    t_tri_dgrad = t
    t = bench_ms(hip_dgrad, args.iters); print(f"{'HIP    dgrad (fwd reuse)':40s}  {t:8.3f}  {flops/t/1e9:8.1f}  {t_tri_dgrad/t:10.3f}x")

    # wgrad: HIP variable-K (v1.3 fast-permute) vs Triton
    def hip_wgrad():
        return grouped_gemm_mxfp8_hip_variable_k(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16, trans_c=False)
    def tri_wgrad():
        return grouped_gemm_mxfp8_variable_k_triton_kernel(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16)

    t = bench_ms(tri_wgrad, args.iters); print(f"{'Triton wgrad (variable-K)':40s}  {t:8.3f}  {flops/t/1e9:8.1f}  {1.0:10.3f}")
    t_tri_wgrad = t
    t = bench_ms(hip_wgrad, args.iters); print(f"{'HIP    wgrad (v1.3 fast permute)':40s}  {t:8.3f}  {flops/t/1e9:8.1f}  {t_tri_wgrad/t:10.3f}x")

    # ═══════ Autograd step bench ═══════
    print()
    print("═" * 85)
    print(f"{'Autograd full-step bench':40s}  {'ms':>8s}  {'vs Tri step':>12s}")
    print("─" * 85)

    # Pure Triton step (existing FP8GroupedGemmMXFunc)
    from primus_turbo.pytorch.ops.grouped_gemm_fp8 import FP8GroupedGemmMXFunc
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig, ScalingGranularity, ScaleDtype,
    )
    cfg = Float8QuantConfig(granularity=ScalingGranularity.MX_BLOCKWISE,
                            block_size=32, scale_dtype=ScaleDtype.E8M0)
    # Triton MX path uses B in [G, K, N] (trans_b=False).
    b_bf_triton = b_bf.transpose(1, 2).contiguous()

    def step_tri():
        a2 = a_bf.detach().requires_grad_(True)
        b2 = b_bf_triton.detach().requires_grad_(True)
        o = FP8GroupedGemmMXFunc.apply(a2, b2, group_lens, group_offs,
                                        False, cfg, None,
                                        None, None, None, None)
        torch.autograd.grad(o, [a2, b2], grad_out_bf)

    def step_hybrid():
        a2 = a_bf.detach().requires_grad_(True)
        b2 = b_bf.detach().requires_grad_(True)
        o = grouped_gemm_mxfp8_hip(a2, b2, group_lens, group_offs)
        torch.autograd.grad(o, [a2, b2], grad_out_bf)

    prequant = prequant_mxfp8_weights_hip(b_bf.detach())
    def step_hybrid_prequant():
        a2 = a_bf.detach().requires_grad_(True)
        o = grouped_gemm_mxfp8_hip(a2, prequant, group_lens, group_offs)
        torch.autograd.grad(o, [a2], grad_out_bf)

    t_tri = bench_ms(step_tri, args.iters)
    print(f"{'pure Triton step':40s}  {t_tri:8.3f}  {1.00:12.3f}")
    t_hy = bench_ms(step_hybrid, args.iters)
    print(f"{'hybrid (HIP fwd+dgrad+wgrad) step':40s}  {t_hy:8.3f}  {t_tri/t_hy:12.3f}x")
    t_hp = bench_ms(step_hybrid_prequant, args.iters)
    print(f"{'hybrid + MXFP8WeightPrequantHip':40s}  {t_hp:8.3f}  {t_tri/t_hp:12.3f}x")

    # ── Correctness gate ──
    print()
    a2 = a_bf.detach().requires_grad_(True)
    b2 = b_bf.detach().requires_grad_(True)
    out_hy = grouped_gemm_mxfp8_hip(a2, b2, group_lens, group_offs)
    grad_a_hy, grad_b_hy = torch.autograd.grad(out_hy, [a2, b2], grad_out_bf)
    out_bf = torch.zeros_like(out_hy)
    grad_a_bf = torch.zeros_like(a_bf)
    grad_b_bf = torch.zeros_like(b_bf)
    for gi in range(g):
        s, e = int(group_offs[gi]), int(group_offs[gi+1])
        a_g = a_bf[s:e].float(); b_g = b_bf[gi].float(); dy = grad_out_bf[s:e].float()
        out_bf[s:e]    = (a_g @ b_g.T).to(torch.bfloat16)
        grad_a_bf[s:e] = (dy @ b_g).to(torch.bfloat16)
        grad_b_bf[gi]  = (dy.T @ a_g).to(torch.bfloat16)
    print(f"Correctness (hybrid autograd vs bf16 ref):")
    print(f"  out    SNR: {snr(out_bf, out_hy):.2f} dB  (fp8 floor ~28.5 dB)")
    print(f"  grad_a SNR: {snr(grad_a_bf, grad_a_hy):.2f} dB")
    print(f"  grad_b SNR: {snr(grad_b_bf, grad_b_hy):.2f} dB")


if __name__ == "__main__":
    main()
