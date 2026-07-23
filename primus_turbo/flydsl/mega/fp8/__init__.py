###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL MXFP8 kernels for the fused mega MoE (forward + partial-fp8 backward).

Self-contained port of the Primus-Turbo mega MXFP8 stack. MegaMoE's bf16 path runs on a
different symmetric-memory design (``SymBuffer`` + ``Workspace`` + flag/parity epochs), while the
fp8 kernels were written against the ``SymLayout`` + scoreboard + two-heap design. To avoid
touching the bf16 stack, that whole foundation is VENDORED here under this package
(``prims`` / ``sym_layout`` / ``barrier`` / ``symm_buffer`` / ``dispatch_prologue`` /
``gemm_helper``), and all fp8 modules import from ``primus_turbo.flydsl.mega.fp8.*`` only. It
shares nothing with the bf16 files except ``primus_turbo.pytorch.core`` (SymmetricMemory,
low_precision) and the external ``flydsl`` package.

Currently wired: the L1 forward path (fused mxfp8 dispatch + fc1). L2 combine + backward are
ported in later steps.
"""

# --- fused mxfp8 dispatch PUSH + preshuffle + grouped mxfp8 NT GEMM ---
# (generic: forward L1 = dispatch(x)+fc1; backward STEP1 = dispatch(dy)+fc2 dgrad reuses it with a
# different CU split -- no separate bwd kernel)
from .dispatch_grouped_gemm_mxfp8_kernel import (
    _host_rendezvous,
    dispatch_grouped_gemm_mxfp8,
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
)

# --- unified fp8 combine (ONE entry, role inferred from topk_weights/grad_gate; mirrors bf16) ---
#   forward L2      : fp8 GEMM + combine PUSH + weighted top-k reduce (bf16 out)
#   backward STEP3  : fp8 fc1-dgrad + combine PUSH + unweighted reduce (+ gate scatter)
from .grouped_gemm_combine_fp8_kernel import (
    grouped_gemm_combine_mxfp8_flydsl_kernel,
)

# --- symmetric workspace (SymLayout + scoreboard + two-heap) ---
from .dispatch_prologue import dispatch_prologue
from .sym_layout import SymLayout
from .symm_buffer import get_symm_buffer_for_mega_moe

# --- quantization (weights: grouped mxfp8; activations: FlyDSL rowwise mxfp8) ---
from .quant import (
    quantize_grouped_weight_mxfp8,
    quantize_grouped_weight_mxfp8_flydsl,
    quantize_rowwise_mxfp8,
)
from .quant_flydsl import preshuffle_b_scale, quantize_rowwise_mxfp8_flydsl

# --- colwise-transpose mxfp8 quant (backward variable-K wgrad operands: dW2 / dW1) ---
from .quant_colwise_trans_flydsl import (
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
)

__all__ = [
    "dispatch_grouped_gemm_mxfp8",
    "dispatch_grouped_gemm_mxfp8_flydsl_kernel",
    "grouped_gemm_combine_mxfp8_flydsl_kernel",
    "dispatch_prologue",
    "SymLayout",
    "get_symm_buffer_for_mega_moe",
    "quantize_grouped_weight_mxfp8",
    "quantize_grouped_weight_mxfp8_flydsl",
    "quantize_rowwise_mxfp8",
    "quantize_rowwise_mxfp8_flydsl",
    "preshuffle_b_scale",
    "colwise_grouped_meta",
    "colwise_quant_mxfp8_grouped_flydsl",
    "colwise_requant_mxfp8_grouped_fp8in_flydsl",
]
