###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fully fused mega MoE forward/backward (FlyDSL) over a single symmetric buffer."""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.swiglu_kernel import swiglu, swiglu_backward
from primus_turbo.flydsl.mega.symm_buffer import SymmBuffer
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
    dispatch_grouped_gemm_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.grouped_gemm_combine_impl import (
    grouped_gemm_combine_impl,
)

__all__ = [
    "SymmBuffer",
    "MegaMoEFusedFunction",
    "mega_moe_fused",
]


class MegaMoEFusedFunction(torch.autograd.Function):
    """Wraps the fused mega MoE forward so its output joins the autograd graph."""

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
        with torch.profiler.record_function("mega_moe_forward"):
            num_tokens = x.shape[0]
            num_topk = topk_idx.shape[-1]
            # int64 end-to-end (combine reads topk i64)
            topk_idx = topk_idx.to(torch.int64)

            ctx.set_materialize_grads(False)

            # fused prologue + cross-rank dispatch PUSH + grouped L1 GEMM (NT)
            l1_out, _, dispatch_weights_in_buf, handle = dispatch_grouped_gemm_impl(
                x,
                w1,
                group,
                BackendType.FLYDSL.value,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                layout="nt",
            )

            act = swiglu(l1_out)

            y, _ = grouped_gemm_combine_impl(
                act,
                w2,
                list(handle),
                BackendType.FLYDSL.value,
                topk_indices=topk_idx.contiguous().view(-1),
                topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
                # fwd(nt): combine is GEMM-hidden -> MORE cu hides the drain tail. Stable-box
                # sweep: 64=2.811ms best (16=3.32, 32=2.85, 96=2.85). Multiple of 8 (XCD count).
                num_combine_cu=64,
                num_reduce_cu=0,
                layout="nt",
                BM=block_m,
                BN=block_n,
            )

            # stash everything backward needs
            if any(ctx.needs_input_grad):
                # clone before a later layer overwrites the shared buffer
                dispatch_weights_in_buf = dispatch_weights_in_buf.clone()

                ctx.group = group
                ctx.num_tokens = num_tokens
                ctx.num_topk = num_topk
                ctx.block_m = block_m
                ctx.block_n = block_n
                ctx.handle_len = len(handle)
                ctx.save_for_backward(
                    *handle,
                    x,
                    l1_out,
                    dispatch_weights_in_buf,
                    w1,
                    w2,
                    topk_idx,
                )
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        """Conjugate of forward via Dispatch<->Combine duality; grads for x / w1 / w2 / topk_w."""
        with torch.profiler.record_function("mega_moe_backward"):
            # grad_y is None when the output got no grad
            if grad_y is None:
                return (None,) * 8
            saved = ctx.saved_tensors
            handle = tuple(saved[: ctx.handle_len])
            (
                saved_x,
                l1_out,
                dispatch_weights_in_buf,
                w1,
                w2,
                topk_idx,
            ) = saved[ctx.handle_len :]

            # handle contract (dispatch_prologue return order):
            # [7]=num_tokens_per_expert (group_lens), [8]=prefix-sum (group_offs)
            group_lens, group_offs = handle[7], handle[8]
            num_tokens, num_topk = ctx.num_tokens, ctx.num_topk
            dy = grad_y.contiguous().to(torch.bfloat16)

            # STEP 1: dispatch dy + L2 dgrad (grad_swiglu = dispatch_l2_grad @ w2, NN)
            # dispatch_l2_grad aliases the symm token pool: must be consumed by
            # dW2 below BEFORE STEP 3 / dW1 re-dispatch overwrite the pool.
            grad_swiglu, dispatch_l2_grad, _, _ = dispatch_grouped_gemm_impl(
                dy,
                w2,
                ctx.group,
                BackendType.FLYDSL.value,
                handle=handle,
                layout="nn",
            )

            # STEP 2: SwiGLU^T (re-inject routing weight) + gate grad
            grad_l1, grad_gate, act_weighted = swiglu_backward(
                grad_swiglu,
                l1_out,
                scale=dispatch_weights_in_buf,
                return_gate=True,
                return_act_w=True,
            )

            # dW2 = dispatch_l2_grad^T @ act_weighted (variable-K wgrad)
            dW2 = grouped_gemm_variable_k_impl(
                dispatch_l2_grad,
                act_weighted,
                group_lens,
                group_offs,
                trans_a=True,
                trans_b=False,
                trans_c=False,
                num_cu=None,
                default_backend=BackendType.TRITON.value,
            )

            # STEP 3: L1 dgrad (grad_l1 @ w1, NN) + combine PUSH + dx reduce + grad_gate scatter
            dx, grad_topk_weights_flat = grouped_gemm_combine_impl(
                grad_l1,
                w1,
                list(handle),
                BackendType.FLYDSL.value,
                topk_indices=topk_idx.contiguous().view(-1),
                topk_weights=None,
                grad_gate=grad_gate,
                # bwd(nn): GEMM-bound -> FEWER combine cu frees GEMM. Stable-box sweep:
                # 16=4.122ms best (8=5.55, 24=4.14, 32=4.13, 48=4.33, 64=4.36). Mult of 8.
                num_combine_cu=16,
                num_reduce_cu=0,
                layout="nn",
                BM=ctx.block_m,
                BN=ctx.block_n,
            )

            # dW1 = pool(x)^T @ grad_l1 (variable-K TN wgrad; re-dispatch saved x)
            dW1, _, _, _ = dispatch_grouped_gemm_impl(
                saved_x,
                grad_l1,
                ctx.group,
                BackendType.FLYDSL.value,
                handle=handle,
                layout="tn",
                trans_c=True,
                num_dispatch_cu=16,
            )

            # reshape the combine-reduce gate output to [num_tokens, num_topk]
            grad_topk_weights = grad_topk_weights_flat.view(num_tokens, num_topk)
            # grads for (x, topk_idx, topk_weights, w1, w2, group, block_m, block_n)
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


def mega_moe_fused(
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
    """One fully fused mega MoE forward; the symmetric buffer is fetched (and cached) internally."""
    return MegaMoEFusedFunction.apply(x, topk_idx, topk_weights, w1, w2, group, block_m, block_n)
