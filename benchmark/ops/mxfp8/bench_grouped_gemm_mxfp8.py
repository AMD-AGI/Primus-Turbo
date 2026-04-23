###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MX-FP8 grouped GEMM benchmark suite (MI355X / gfx950).

Consolidates correctness + perf measurements covered in docs/mxfp8_perf.md:

  --mode kernel     Isolated fwd / dgrad / wgrad kernel TFLOPS (pre-quantised)
  --mode step       End-to-end training step (autograd, fwd + bwd)
  --mode prequant   Gradient-accumulation savings with MXFP8WeightPrequant
  --mode unbalanced Real MoE trace (entry -10 from gpt_oss_20B gem_shape_summary)
  --mode swiglu     Fused SwiGLU + MX-FP8 quant vs separate silu_mul + quant
  --mode all        Run everything (default)

Default shape mirrors gpt_oss_20B gate_up:
  a = [65536, 2880]  bf16
  b = [32, 2880, 5760] bf16
  G = 32  (balanced or real unbalanced distribution)
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig, Format, ScaleDtype, ScalingGranularity,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm as grouped_gemm_bf16
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import float8_e4m3
from primus_turbo.triton.activation.swiglu_quant_kernel import swiglu_quant_mxfp8_fwd
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    grouped_gemm_fp8_tensorwise_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import grouped_gemm_triton_kernel
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import grouped_gemm_mxfp8_triton_kernel
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    MXFP8WeightPrequant,
    prequant_mxfp8_weights,
    quant_mxfp8_dual_jagged,
    quant_mxfp8_rowwise,
    quant_mxfp8_weight_dgrad,
    quant_mxfp8_weight_fwd,
)


# ─── defaults ──────────────────────────────────────────────────────────────
G_DEFAULT = 32
TOTAL_DEFAULT = 65536
HIDDEN_DEFAULT = 2880
INTER_DEFAULT = 5760

# Real gpt_oss_20B unbalanced trace (entry -10 from gem_shape_summary.txt).
UNBALANCED_LENS = [
    327, 105, 1843, 2724, 1150, 1798, 769, 646, 711, 462,
    2019, 645, 961, 697, 1391, 16053, 3452, 575, 693, 252,
    956, 1120, 1856, 352, 899, 234, 14682, 2483, 415, 4292,
    606, 368,
]

DTYPE = torch.bfloat16
DEVICE = "cuda"


# ─── helpers ───────────────────────────────────────────────────────────────


def timer(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return tbench.Timer(stmt="fn()", globals={"fn": fn}).timeit(iters).mean * 1e3


def snr(ref, out):
    diff = (out.float() - ref.float())
    sig = (ref.float() ** 2).mean().item()
    noise = (diff ** 2).mean().item() + 1e-30
    return 10 * math.log10(sig / noise)


def make_inputs(total_m, hidden, inter, lens_list, requires_grad=True, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((total_m, hidden), dtype=DTYPE, device=DEVICE) * 0.3
    b = torch.randn((len(lens_list), hidden, inter), dtype=DTYPE, device=DEVICE) * 0.3
    lens = torch.tensor(lens_list, dtype=torch.int64, device=DEVICE)
    offs = grouped_gemm_compute_offs(lens)
    if requires_grad:
        a.requires_grad_(True)
        b.requires_grad_(True)
    return a, b, lens, offs


def mx_config():
    return Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        scale_dtype=ScaleDtype.E8M0,
        block_size=32,
    )


# ─── modes ─────────────────────────────────────────────────────────────────


def mode_kernel(total_m, hidden, inter, lens_list):
    """Isolated fwd/dgrad/wgrad kernel TFLOPS with pre-quantised inputs."""
    print(f"\n=== KERNEL-ONLY  M={total_m}  K={hidden}  N={inter}  G={len(lens_list)} ===")
    a, b, lens, offs = make_inputs(total_m, hidden, inter, lens_list, requires_grad=False)
    grad_out = torch.randn((total_m, inter), dtype=DTYPE, device=DEVICE) * 0.3

    bf_out = grouped_gemm_triton_kernel(a, b, offs, trans_b=False)
    bf_ms = timer(lambda: grouped_gemm_triton_kernel(a, b, offs, trans_b=False))

    a_tw, as_tw = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
    b_tw, bs_tw = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
    tw_ms = timer(lambda: grouped_gemm_fp8_tensorwise_triton_kernel(
        a_tw, b_tw, as_tw, bs_tw, offs, trans_b=False, out_dtype=DTYPE,
    ))

    a_row, a_scale_row, a_col, a_scale_col, sc_offs = quant_mxfp8_dual_jagged(a, offs, lens)
    b_fwd, b_scale_fwd = quant_mxfp8_weight_fwd(b)
    b_dgrad, b_scale_dgrad = quant_mxfp8_weight_dgrad(b)
    go_row, go_scale_row, go_col, go_scale_col, _ = quant_mxfp8_dual_jagged(grad_out, offs, lens)

    fwd_fn = lambda: grouped_gemm_mxfp8_triton_kernel(
        a_row, b_fwd, a_scale_row, b_scale_fwd, offs, trans_b=False, out_dtype=DTYPE,
    )
    fwd_out = fwd_fn()
    fwd_ms = timer(fwd_fn)

    dgrad_fn = lambda: grouped_gemm_mxfp8_triton_kernel(
        go_row, b_dgrad, go_scale_row, b_scale_dgrad, offs, trans_b=True, out_dtype=DTYPE,
    )
    dgrad_fn()
    dgrad_ms = timer(dgrad_fn)

    wgrad_fn = lambda: grouped_gemm_mxfp8_variable_k_triton_kernel(
        a_col, go_col, a_scale_col, go_scale_col, offs, sc_offs, out_dtype=DTYPE,
    )
    wgrad_fn()
    wgrad_ms = timer(wgrad_fn)

    flops = 2.0 * total_m * hidden * inter
    def tf(ms): return flops / (ms * 1e-3) / 1e12
    print(f"  {'bf16 grouped_gemm':<38} {bf_ms:7.3f} ms  {tf(bf_ms):7.1f} TFLOPS  {1.0:5.2f}×")
    print(f"  {'fp8 tensorwise (ceiling)':<38} {tw_ms:7.3f} ms  {tf(tw_ms):7.1f} TFLOPS  {tf(tw_ms)/tf(bf_ms):5.2f}×")
    print(f"  {'MX-FP8 forward':<38} {fwd_ms:7.3f} ms  {tf(fwd_ms):7.1f} TFLOPS  {tf(fwd_ms)/tf(bf_ms):5.2f}×")
    print(f"  {'MX-FP8 dgrad (fwd kernel, trans_b=T)':<38} {dgrad_ms:7.3f} ms  {tf(dgrad_ms):7.1f} TFLOPS  {tf(dgrad_ms)/tf(bf_ms):5.2f}×")
    print(f"  {'MX-FP8 wgrad (variable-K)':<38} {wgrad_ms:7.3f} ms  {tf(wgrad_ms):7.1f} TFLOPS  {tf(wgrad_ms)/tf(bf_ms):5.2f}×")
    print(f"  SNR fwd vs bf16: {snr(bf_out, fwd_out):.2f} dB")


def mode_step(total_m, hidden, inter, lens_list, label="balanced"):
    """End-to-end training step (fwd + dgrad + wgrad via autograd)."""
    print(f"\n=== STEP ({label})  M={total_m}  K={hidden}  N={inter}  G={len(lens_list)} ===")
    cfg = mx_config()
    a, b, lens, offs = make_inputs(total_m, hidden, inter, lens_list)
    out_bf = grouped_gemm_bf16(a, b, lens, offs, trans_b=False)
    grad_out = torch.randn_like(out_bf)
    out_bf.backward(grad_out)
    grad_a_ref, grad_b_ref = a.grad.detach().clone(), b.grad.detach().clone()

    a2, b2, _, _ = make_inputs(total_m, hidden, inter, lens_list)
    out_mx = grouped_gemm_fp8(a2, b2, lens, offs, trans_b=False, config=cfg)
    out_mx.backward(grad_out)
    print(f"  SNR: out {snr(out_bf, out_mx):.2f}  grad_a {snr(grad_a_ref, a2.grad):.2f}  grad_b {snr(grad_b_ref, b2.grad):.2f}")

    a3, b3, _, _ = make_inputs(total_m, hidden, inter, lens_list, requires_grad=False)
    bf_fwd = timer(lambda: grouped_gemm_bf16(a3, b3, lens, offs, trans_b=False))
    def bf_step():
        a_, b_, _, _ = make_inputs(total_m, hidden, inter, lens_list)
        grouped_gemm_bf16(a_, b_, lens, offs, trans_b=False).backward(grad_out)
    bf_step_ms = timer(bf_step, iters=20, warmup=5)

    mx_fwd = timer(lambda: grouped_gemm_fp8(a3, b3, lens, offs, trans_b=False, config=cfg))
    def mx_step():
        a_, b_, _, _ = make_inputs(total_m, hidden, inter, lens_list)
        grouped_gemm_fp8(a_, b_, lens, offs, trans_b=False, config=cfg).backward(grad_out)
    mx_step_ms = timer(mx_step, iters=20, warmup=5)

    print(f"  {'Path':<10} {'fwd':>8} {'bwd':>8} {'step':>8}")
    print(f"  {'bf16':<10} {bf_fwd:8.3f} {bf_step_ms - bf_fwd:8.3f} {bf_step_ms:8.3f}")
    print(f"  {'MX-FP8':<10} {mx_fwd:8.3f} {mx_step_ms - mx_fwd:8.3f} {mx_step_ms:8.3f}")
    print(f"  vs bf16:  fwd {bf_fwd/mx_fwd:.2f}×   bwd {(bf_step_ms-bf_fwd)/(mx_step_ms-mx_fwd):.2f}×   step {bf_step_ms/mx_step_ms:.2f}×")


def mode_prequant(total_m, hidden, inter, lens_list, k_accum=8):
    """Gradient-accumulation savings with prequant_mxfp8_weights."""
    print(f"\n=== PREQUANT  k={k_accum} accum  M={total_m}  K={hidden}  N={inter} ===")
    cfg = mx_config()
    a, b, lens, offs = make_inputs(total_m, hidden, inter, lens_list)
    out_ref = grouped_gemm_fp8(a, b, lens, offs, trans_b=False, config=cfg)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    grad_a_ref, grad_b_ref = a.grad.detach().clone(), b.grad.detach().clone()

    a2, b2, _, _ = make_inputs(total_m, hidden, inter, lens_list)
    pq = prequant_mxfp8_weights(b2)
    out_pq = grouped_gemm_fp8(a2, pq, lens, offs, trans_b=False, config=cfg)
    out_pq.backward(grad_out)
    print(f"  SNR (prequant vs non): out {snr(out_ref, out_pq):.2f}  "
          f"grad_a {snr(grad_a_ref, a2.grad):.2f}  grad_b {snr(grad_b_ref, b2.grad):.2f}")

    def accum_norm():
        a_, b_, _, _ = make_inputs(total_m, hidden, inter, lens_list)
        for _k in range(k_accum):
            grouped_gemm_fp8(a_, b_, lens, offs, trans_b=False, config=cfg).backward(
                grad_out, retain_graph=(_k < k_accum - 1)
            )

    def accum_pq():
        a_, b_, _, _ = make_inputs(total_m, hidden, inter, lens_list)
        pq_ = prequant_mxfp8_weights(b_)
        for _k in range(k_accum):
            grouped_gemm_fp8(a_, pq_, lens, offs, trans_b=False, config=cfg).backward(
                grad_out, retain_graph=(_k < k_accum - 1)
            )

    t_norm = timer(accum_norm, iters=5, warmup=2)
    t_pq = timer(accum_pq, iters=5, warmup=2)
    print(f"  {'Path':<28} {'total ms':>10} {'per-step':>10}")
    print(f"  {'bf16 b (quant ×k)':<28} {t_norm:10.3f} {t_norm / k_accum:10.3f}")
    print(f"  {'prequant (quant ×1)':<28} {t_pq:10.3f} {t_pq / k_accum:10.3f}")
    print(f"  Speedup: {t_norm / t_pq:.2f}×   per-step saved: {(t_norm - t_pq) / k_accum:.3f} ms")


def mode_unbalanced(hidden, inter):
    """Real gpt_oss_20B unbalanced trace — correctness + perf."""
    mode_step(sum(UNBALANCED_LENS), hidden, inter, UNBALANCED_LENS, label="real unbalanced trace")


def mode_swiglu(total_m, inter):
    """Fused SwiGLU + MX-FP8 quant vs separate silu_mul + quant_mxfp8_rowwise."""
    print(f"\n=== SWIGLU FUSION  M={total_m}  N={inter} ===")
    torch.manual_seed(0)
    gate_up = torch.randn((total_m, 2 * inter), dtype=DTYPE, device=DEVICE) * 0.3
    N = inter

    y_ref = torch.nn.functional.silu(gate_up[:, :N]) * gate_up[:, N:]
    y_ref_fp8, y_ref_scale = quant_mxfp8_rowwise(y_ref)
    y_fp8, y_scale = swiglu_quant_mxfp8_fwd(gate_up)

    def dequant(fp8, scale):
        fp32 = fp8.float()
        s = torch.pow(2.0, (scale.int() - 127).float()).unsqueeze(-1)
        return (fp32.reshape(total_m, N // 32, 32) * s).reshape(total_m, N)

    ref_deq = dequant(y_ref_fp8, y_ref_scale)
    fused_deq = dequant(y_fp8, y_scale)
    print(f"  SNR (fused vs torch ref)      : {snr(y_ref, fused_deq):.2f} dB")
    print(f"  SNR (fused vs separate kernel): {snr(ref_deq, fused_deq):.2f} dB")

    ref_ms = timer(lambda: quant_mxfp8_rowwise(
        torch.nn.functional.silu(gate_up[:, :N]) * gate_up[:, N:]
    ))
    fused_ms = timer(lambda: swiglu_quant_mxfp8_fwd(gate_up))
    print(f"  {'Path':<45} {'ms':>8}")
    print(f"  {'separate torch silu_mul + quant_mxfp8_rowwise':<45} {ref_ms:8.3f}")
    print(f"  {'fused swiglu_quant_mxfp8_fwd':<45} {fused_ms:8.3f}")
    print(f"  Speedup: {ref_ms / fused_ms:.2f}×")


# ─── entry ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="MX-FP8 grouped GEMM bench suite")
    p.add_argument("--mode", default="all",
                   choices=["all", "kernel", "step", "prequant", "unbalanced", "swiglu"])
    p.add_argument("--total-m", type=int, default=TOTAL_DEFAULT)
    p.add_argument("--hidden", type=int, default=HIDDEN_DEFAULT)
    p.add_argument("--inter", type=int, default=INTER_DEFAULT)
    p.add_argument("--g", type=int, default=G_DEFAULT)
    p.add_argument("--k-accum", type=int, default=8, help="micro-batches per optim step (prequant mode)")
    args = p.parse_args()

    lens_balanced = [args.total_m // args.g] * args.g
    assert sum(lens_balanced) == args.total_m, "total-m must divide g evenly"

    if args.mode in ("all", "kernel"):
        mode_kernel(args.total_m, args.hidden, args.inter, lens_balanced)
    if args.mode in ("all", "step"):
        mode_step(args.total_m, args.hidden, args.inter, lens_balanced, label="balanced")
    if args.mode in ("all", "unbalanced"):
        mode_unbalanced(args.hidden, args.inter)
    if args.mode in ("all", "prequant"):
        mode_prequant(args.total_m, args.hidden, args.inter, lens_balanced, k_accum=args.k_accum)
    if args.mode in ("all", "swiglu"):
        mode_swiglu(args.total_m, args.inter)


if __name__ == "__main__":
    main()
