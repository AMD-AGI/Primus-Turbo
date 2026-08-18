###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL MXFP8 kernels for the fused mega MoE (forward + partial-fp8 backward).

Self-contained port of the Primus-Turbo mega MXFP8 stack. MegaMoE's bf16 path runs on a
different symmetric-memory design (``SymBuffer`` + ``Workspace`` + flag/parity epochs), while the
fp8 kernels were written against the ``SymLayout`` + scoreboard + two-heap design. To avoid
touching the bf16 stack, that whole foundation is VENDORED here under this package
(``prims`` / ``barrier`` / ``symm_buffer`` / ``dispatch_prologue`` /
``gemm_helper``), and all fp8 modules import from ``primus_turbo.flydsl.mega.fp8.*`` only. It
shares nothing with the bf16 files except ``primus_turbo.pytorch.core`` (SymmetricMemory,
low_precision) and the external ``flydsl`` package.
"""

# --- fused mxfp8 dispatch PUSH + preshuffle + grouped mxfp8 NT GEMM ---
#   forward L1     : dispatch(x) PUSH + fc1 GEMM; owns the routing and returns the step's handle
#   backward L2 dgrad: dispatch(dy) PUSH + fc2 dgrad over that handle; same NT op, own CU split
from .dispatch_grouped_gemm_mxfp8_kernel import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_l1_fwd_mxfp8_flydsl_kernel,
    dispatch_l2_dgrad_mxfp8_flydsl_kernel,
    extend_handle,
)

# --- symmetric workspace (SymLayout + scoreboard + two-heap) ---
from .dispatch_prologue import dispatch_prologue

# --- unified fp8 combine (ONE entry, role inferred from topk_weights/grad_gate; mirrors bf16) ---
#   forward L2      : fp8 GEMM + combine PUSH + weighted top-k reduce (bf16 out)
#   backward L1 dgrad : fp8 fc1-dgrad + combine PUSH + unweighted reduce (+ gate scatter)
from .grouped_gemm_combine_fp8_kernel import (
    combine_l1_dgrad_mxfp8_flydsl_kernel,
    combine_l2_fwd_mxfp8_flydsl_kernel,
)

# --- mxfp8 quantization: rowwise (weights / activations) + colwise-transpose (backward
#     variable-K wgrad operands: dW2 / dW1) ---
from .quant import (
    colwise_grouped_meta,
    colwise_requant_fp8in_and_quant_bf16_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    preshuffle_b_scale,
    quantize_grouped_weight_mxfp8_flydsl,
    quantize_rowwise_mxfp8_flydsl,
)
from .swiglu_mxfp8_kernel import (
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
    swiglu_mxfp8_flydsl_kernel,
)
from .symm_buffer import SymLayout, get_symm_buffer_for_mega_moe

__all__ = [
    "dispatch_grouped_gemm_mxfp8",
    "dispatch_l1_fwd_mxfp8_flydsl_kernel",
    "dispatch_l2_dgrad_mxfp8_flydsl_kernel",
    "extend_handle",
    "combine_l1_dgrad_mxfp8_flydsl_kernel",
    "combine_l2_fwd_mxfp8_flydsl_kernel",
    "swiglu_mxfp8_flydsl_kernel",
    "dispatch_prologue",
    "SymLayout",
    "get_symm_buffer_for_mega_moe",
    "quantize_grouped_weight_mxfp8_flydsl",
    "quantize_rowwise_mxfp8_flydsl",
    "preshuffle_b_scale",
    "colwise_grouped_meta",
    "colwise_requant_mxfp8_grouped_fp8in_flydsl",
    "colwise_requant_fp8in_and_quant_bf16_grouped_flydsl",
    "swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl",
]
