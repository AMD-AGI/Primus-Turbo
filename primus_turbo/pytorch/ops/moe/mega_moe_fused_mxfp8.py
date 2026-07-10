###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trainable mega MoE with MXFP8 forward + partial-fp8 backward (autograd Function).

Forward: the fused mxfp8 forward (`mega_moe_fused_mxfp8_forward(comm="fp8_fused")`).
Backward (conjugate via the Dispatch<->Combine duality, mirrors the bf16
`MegaMoEFusedFunction.backward`):
  * STEP1 (dispatch(dy) + fc2 dgrad, NN) and STEP3 (fc1 dgrad + combine) + dW1: bf16.
  * dW2 (fc2 weight grad, variable-K over the pool): **MXFP8** -- quantize
    ``dispatch_l2_grad`` + ``act_weighted`` colwise (along the pool contraction) and run
    the mxfp8 variable-K wgrad (`grouped_gemm_fp8_variable_k_impl`, MX_BLOCKWISE), mirroring
    the validated `MXFP8GroupedGEMMFunction.backward` grad_b. dW2 is a large GEMM
    (H*I*pool), so fp8 there is the main backward win; the dgrad/combine stay bf16 per the
    L2-combine analysis (Increment A). A later increment fp8-izes STEP1 dgrad too.

``_DW2_FP8_FORMAT`` selects the dW2 wgrad fp8 encoding (default E5M2 for gradient range;
E4M3 measured higher-SNR at DSv3 magnitudes -- gated by dW2 SNR + few-step training loss)."""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.fp8.mega_moe_fused_mxfp8 import mega_moe_fused_mxfp8_forward
from primus_turbo.flydsl.mega.swiglu_kernel import swiglu_backward
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,  # noqa: F401  (alternative dW2 encoding; see _DW2_FP8_FORMAT)
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
    dispatch_grouped_gemm_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.grouped_gemm_combine_impl import (
    grouped_gemm_combine_impl,
)
from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp8_with_trans

__all__ = ["MegaMoEFusedMxfp8Function", "mega_moe_fused_mxfp8"]

_DW2_FP8_FORMAT = float8_e5m2  # dW2 wgrad encoding (E5M2 = grad range; flip to float8_e4m3 to compare)
# fp8 dW2 default OFF: MEASURED NET-NEGATIVE at DSv3 EP8 shapes. The mega variable-K wgrad is
# small-K / output-write bound (tokens/expert ~256, contraction = pool), so the in-tree Triton
# mxfp8 variable-K wgrad is NOT faster than bf16 (fp8 gemm ~= bf16, even slightly slower), and
# the colwise quant of both operands adds ~0.1-0.3 ms -> net +0.08..0.21 ms SLOWER (probe), for
# no benefit + fp8 accuracy loss. Correctness is fine (fp8 vs bf16 dW2 = 22.51 dB); it's a perf
# loss, not a win. Kept behind the flag: a compute-bound regime (large tokens/expert) or a
# faster native FlyDSL mxfp8 wgrad could flip this. Enable only after re-measuring a speedup.
_USE_FP8_DW2 = False
_MXFP8_BLOCK = 32
_HANDLE_GROUP_LENS = 9
_HANDLE_GROUP_OFFS = 10


def _mxfp8_variable_k_wgrad(a_bf16, b_bf16, group_lens, group_offs):
    """dW = a^T @ b (variable-K over the pool/contraction axis) in MXFP8. Quantizes both
    operands colwise (transposed) via the grouped dual-quant, then the mxfp8 variable-K
    grouped GEMM. Returns [G, a.shape[1], b.shape[1]] bf16. Mirrors MXFP8GroupedGEMMFunction
    grad_b (colwise operands, trans_a=trans_b=False)."""
    lens64 = group_lens.to(torch.int64)
    offs64 = group_offs.to(torch.int64)
    (_, _, a_t, a_ts, _, _, lens_pc, offs_pc) = grouped_quantize_fp8_with_trans(
        a_bf16, _DW2_FP8_FORMAT, ScalingGranularity.MX_BLOCKWISE, lens64, offs64, block_size=_MXFP8_BLOCK
    )
    (_, _, b_t, b_ts, _, _, _, _) = grouped_quantize_fp8_with_trans(
        b_bf16, _DW2_FP8_FORMAT, ScalingGranularity.MX_BLOCKWISE, lens64, offs64, block_size=_MXFP8_BLOCK
    )
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t, a_ts, b_ts, lens_pc, offs_pc,
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.TRITON.value,
    )


class MegaMoEFusedMxfp8Function(torch.autograd.Function):
    """Fused mega MoE, MXFP8 forward + fp8-dW2 backward. Joins the autograd graph."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group: ProcessGroup,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        num_topk = topk_idx.shape[-1]
        topk_idx = topk_idx.to(torch.int64)
        ctx.set_materialize_grads(False)

        y, aux = mega_moe_fused_mxfp8_forward(
            x, topk_idx, topk_weights, w1, w2, group,
            block_m=block_m, block_n=block_n, comm="fp8_fused", return_aux=True,
        )

        if any(ctx.needs_input_grad):
            handle = tuple(aux["handle"])
            ctx.group = group
            ctx.num_tokens = num_tokens
            ctx.num_topk = num_topk
            ctx.block_m = block_m
            ctx.block_n = block_n
            ctx.handle_len = len(handle)
            ctx.save_for_backward(
                *handle,
                x,
                aux["l1"],
                aux["dispatch_weights"],
                w1,
                w2,
                topk_idx,
            )
        return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        """Conjugate of the mxfp8 forward. STEP1/STEP3/dW1 bf16; dW2 MXFP8."""
        if grad_y is None:
            return (None,) * 8
        saved = ctx.saved_tensors
        handle = tuple(saved[: ctx.handle_len])
        (saved_x, l1_out, dispatch_weights_in_buf, w1, w2, topk_idx) = saved[ctx.handle_len :]

        group_lens = handle[_HANDLE_GROUP_LENS]
        group_offs = handle[_HANDLE_GROUP_OFFS]
        num_tokens, num_topk = ctx.num_tokens, ctx.num_topk
        dy = grad_y.contiguous().to(torch.bfloat16)
        triton_be = BackendType.TRITON.value

        # STEP 1 (bf16): dispatch dy + L2 dgrad (grad_swiglu = dispatch_l2_grad @ w2, NN)
        grad_swiglu, dispatch_l2_grad, _, _ = dispatch_grouped_gemm_impl(
            dy, w2, ctx.group, BackendType.FLYDSL.value, handle=handle, layout="nn", num_dispatch_cu=16,
        )

        # STEP 2 (bf16): SwiGLU^T (re-inject routing weight) + gate grad
        grad_l1, grad_gate, act_weighted = swiglu_backward(
            grad_swiglu, l1_out, scale=dispatch_weights_in_buf, return_gate=True, return_act_w=True,
        )

        # dW2: dispatch_l2_grad^T @ act_weighted (variable-K wgrad). MXFP8 (fp8 colwise) by
        # default; bf16 path kept for isolating the fp8-dW2 effect (_USE_FP8_DW2=False).
        if _USE_FP8_DW2:
            dW2 = _mxfp8_variable_k_wgrad(dispatch_l2_grad, act_weighted, group_lens, group_offs)
        else:
            dW2 = grouped_gemm_variable_k_impl(
                dispatch_l2_grad, act_weighted, group_lens, group_offs,
                trans_a=True, trans_b=False, trans_c=False, num_cu=None, default_backend=triton_be,
            )

        # STEP 3 (bf16): L1 dgrad + combine PUSH + dx reduce + grad_gate scatter
        dx, grad_topk_weights_flat = grouped_gemm_combine_impl(
            grad_l1, w1, list(handle), BackendType.FLYDSL.value,
            topk_indices=topk_idx.contiguous().view(-1), topk_weights=None,
            grad_gate=grad_gate, num_combine_cu=16, num_reduce_cu=0,
            layout="nn", BM=ctx.block_m, BN=ctx.block_n,
        )

        # dW1 (bf16): pool(x)^T @ grad_l1 (variable-K TN wgrad; re-dispatch saved x)
        dW1, _, _, _ = dispatch_grouped_gemm_impl(
            saved_x, grad_l1, ctx.group, BackendType.FLYDSL.value,
            handle=handle, layout="tn", trans_c=True, num_dispatch_cu=16,
        )

        grad_topk_weights = grad_topk_weights_flat.view(num_tokens, num_topk)
        return (
            dx,
            None,
            grad_topk_weights,
            dW1.to(w1.dtype),
            dW2.to(w2.dtype),
            None,
            None,
            None,
        )


def mega_moe_fused_mxfp8(
    group: ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    block_m: int = 256,
    block_n: int = 256,
) -> torch.Tensor:
    """One fully fused mega MoE forward (MXFP8) that joins autograd; backward fp8-izes dW2."""
    return MegaMoEFusedMxfp8Function.apply(x, topk_idx, topk_weights, w1, w2, group, block_m, block_n)
