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

# --- L1 forward: fused mxfp8 dispatch PUSH + preshuffle + grouped mxfp8 NT GEMM ---
from .dispatch_grouped_gemm_mxfp8_kernel import dispatch_grouped_gemm_mxfp8

# --- L2 forward: fp8 GEMM + combine PUSH + weighted top-k reduce (bf16 out) ---
from .grouped_gemm_combine_fp8_kernel import grouped_gemm_combine_fp8, prepare_w2_fp8

# --- SwiGLU (bf16, between L1 and L2) ---
from .swiglu_kernel import swiglu, swiglu_backward

# --- symmetric workspace (SymLayout + scoreboard + two-heap) ---
from .dispatch_prologue import dispatch_prologue
from .sym_layout import SymLayout
from .symm_buffer import get_symm_buffer_for_mega_moe

# --- quantization (weights: grouped mxfp8; activations: FlyDSL rowwise mxfp8) ---
from .quant import (
    quantize_grouped_weight_mxfp8,
    quantize_grouped_weight_mxfp8_cached,
    quantize_grouped_weight_mxfp8_flydsl,
    quantize_rowwise_mxfp8,
)
from .quant_flydsl import preshuffle_b_scale, quantize_rowwise_mxfp8_flydsl

__all__ = [
    "dispatch_grouped_gemm_mxfp8",
    "grouped_gemm_combine_fp8",
    "prepare_w2_fp8",
    "swiglu",
    "swiglu_backward",
    "dispatch_prologue",
    "SymLayout",
    "get_symm_buffer_for_mega_moe",
    "quantize_grouped_weight_mxfp8",
    "quantize_grouped_weight_mxfp8_cached",
    "quantize_grouped_weight_mxfp8_flydsl",
    "quantize_rowwise_mxfp8",
    "quantize_rowwise_mxfp8_flydsl",
    "preshuffle_b_scale",
]
