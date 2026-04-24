###############################################################################
# Test the hipBLASLt MX-FP8 wgrad path via the patched C++ binding.
# The C++ binding now accepts granularity="MX_BLOCKWISE" and sets
# scale_mode=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0.
#
# hipBLASLt MX-FP8 constraints (from ROCm docs api-reference.md):
#   opA=T, opB=N, K%128==0, m%16==0, n%16==0
#   Scale layout: 1 uint8 e8m0 per 32-element block along the INNERMOST dim
#   of each operand (same memory order as the operand).
#
# Wgrad mapping (per expert g):
#   dB[g][K_out, N_out] = sum_m a[m, K_out] * grad_out[m, N_out]
#
# hipBLASLt wants shapes (m_hbl, n_hbl, K_hbl). Primus-Turbo VariableK backend
# convention: a=lhs[M,K], b=rhs[M,N], trans_a=True, trans_b=False, trans_c
# selects output orientation.
###############################################################################
from __future__ import annotations
import math
import torch
import torch.utils.benchmark as tbench

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    GroupedGEMMFP8VariableKHipblasltBackend,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    quant_mxfp8_dual_jagged,
    quant_mxfp8_rowwise,
)


def snr(ref, out):
    ref_f = ref.float(); out_f = out.float()
    n = (ref_f - out_f).pow(2).mean().item()
    s = ref_f.pow(2).mean().item()
    return 10 * math.log10(s / max(n, 1e-30)) if n else float("inf")


def main():
    m, k, n, g = 65536, 2880, 5760, 32
    m_per = m // g
    device = "cuda"
    torch.manual_seed(0)

    a_bf  = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    go_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)
    group_lens = torch.full((g,), m_per, dtype=torch.int64, device=device)

    # ── bf16 ref ──
    dB_bf = torch.zeros(g, k, n, device=device, dtype=torch.bfloat16)
    for gi in range(g):
        s_, e_ = int(group_offs[gi]), int(group_offs[gi + 1])
        dB_bf[gi] = (a_bf[s_:e_].float().T @ go_bf[s_:e_].float()).to(torch.bfloat16)

    # ── Triton reference (uses col-quant + jagged scale offs) ──
    _, _, a_col,  a_scale_col,  a_sc_offs  = quant_mxfp8_dual_jagged(a_bf,  group_offs, group_lens)
    _, _, go_col, go_scale_col, _          = quant_mxfp8_dual_jagged(go_bf, group_offs, group_lens)
    dB_tri = grouped_gemm_mxfp8_variable_k_triton_kernel(
        a_col, go_col, a_scale_col, go_scale_col,
        group_offs, a_sc_offs, out_dtype=torch.bfloat16,
    )  # [G, K, N]

    # ── hipBLASLt MX path ──
    # hipBLASLt's VEC32_UE8M0 groups scales along the INNERMOST dim of each
    # operand. Our tensors a [M_total, K] and go [M_total, N] are row-major
    # with K / N innermost. So we need rowwise MX-FP8 quant (groups along the
    # last axis), NOT the col-quant used by Triton wgrad. This is a DIFFERENT
    # quant from Triton's — we re-quantize here.
    a_fp8_row,  a_scale_row  = quant_mxfp8_rowwise(a_bf)    # [M, K] fp8, [M, K/32] u8
    go_fp8_row, go_scale_row = quant_mxfp8_rowwise(go_bf)   # [M, N] fp8, [M, N/32] u8

    # Cast uint8 scales to torch.float8_e8m0fnu (same bits) — the C++ binding
    # checks 8-bit floating-point dtype.
    # Actually the hipblaslt_grouped_gemm_fp8 binding checks is_8bit_floating_point_dtype
    # on a and b, but passes scales as raw void*. So uint8 scale tensors work
    # as-is — the binding doesn't validate scale dtype.

    # Call the backend with trans_a=True, trans_b=False, trans_c=True so the
    # output layout is [G, K, N] (matching Triton's output).
    dB_hbl = GroupedGEMMFP8VariableKHipblasltBackend.execute(
        a=a_fp8_row, b=go_fp8_row,
        a_scales=a_scale_row, b_scales=go_scale_row,
        group_lens=group_lens, group_offs=group_offs,
        trans_a=True, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        num_cu=None,
    )

    print("\n─── Correctness ───")
    print(f"  Triton vs bf16  : {snr(dB_bf, dB_tri):.2f} dB")
    print(f"  hipBLASLt vs bf16: {snr(dB_bf, dB_hbl):.2f} dB")
    print(f"  hipBLASLt vs Triton: {snr(dB_tri, dB_hbl):.2f} dB")

    ok = snr(dB_bf, dB_hbl) >= 25.0
    print(f"  Gate (>=25 dB vs bf16): {'PASS' if ok else 'FAIL'}")

    if not ok:
        return

    # ── Bench ──
    def run_hbl():
        return GroupedGEMMFP8VariableKHipblasltBackend.execute(
            a=a_fp8_row, b=go_fp8_row,
            a_scales=a_scale_row, b_scales=go_scale_row,
            group_lens=group_lens, group_offs=group_offs,
            trans_a=True, trans_b=False, trans_c=False,
            out_dtype=torch.bfloat16,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            num_cu=None,
        )

    def run_tri():
        return grouped_gemm_mxfp8_variable_k_triton_kernel(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16,
        )

    for _ in range(3):
        run_hbl(); run_tri()
    torch.cuda.synchronize()
    t_hbl = tbench.Timer(stmt="f()", globals={"f": run_hbl}).timeit(30).mean
    t_tri = tbench.Timer(stmt="f()", globals={"f": run_tri}).timeit(30).mean
    flops = 2.0 * m * k * n
    print(f"\n─── Bench (kernel-only) ───")
    print(f"  Triton (variable-K)   : {t_tri*1e3:7.3f} ms  {flops/t_tri/1e12:7.1f} TFLOPS")
    print(f"  hipBLASLt (MX-FP8 grouped): {t_hbl*1e3:7.3f} ms  {flops/t_hbl/1e12:7.1f} TFLOPS  ({t_tri/t_hbl:.3f}x)")


if __name__ == "__main__":
    main()
