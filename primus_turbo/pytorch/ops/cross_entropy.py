# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Single-rank vocabulary cross entropy with deferred gradient construction."""

import torch
import triton
from torch.autograd.function import once_differentiable

from primus_turbo.triton.cross_entropy import (
    turbo_cross_entropy_backward_kernel,
    turbo_cross_entropy_forward_kernel,
)

__all__ = ["cross_entropy"]


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index, label_smoothing, overwrite_input):
        vocab = logits.shape[-1]
        rows = target.numel()
        row_width = logits.shape[1] if logits.ndim == 3 else 1
        stride_middle = logits.stride(1) if logits.ndim == 3 else 0
        target = target.contiguous()
        saved_logits = (
            logits if overwrite_input else torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
        )
        stats = torch.empty((rows, 2), device=logits.device, dtype=torch.float32)
        loss = torch.empty(target.shape, device=logits.device, dtype=torch.float32)
        block = min(32768, triton.next_power_of_2(vocab))
        turbo_cross_entropy_forward_kernel[(rows,)](
            logits,
            logits.stride(0),
            stride_middle,
            logits.stride(-1),
            saved_logits,
            target,
            loss,
            stats,
            stats,
            vocab,
            row_width,
            ignore_index,
            label_smoothing,
            COUNT_NON_IGNORE=False,
            COPY_INPUT=not overwrite_input,
            BLOCK_SIZE=block,
            num_warps=16,
        )
        ctx.save_for_backward(saved_logits.detach(), stats, target)
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.overwrite_input = overwrite_input
        ctx.did_backward = False
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.did_backward:
            raise RuntimeError("cross_entropy supports only one backward pass per forward")
        ctx.did_backward = True
        saved_logits, stats, target = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        vocab = saved_logits.shape[-1]
        block = min(32768, triton.next_power_of_2(vocab))
        turbo_cross_entropy_backward_kernel[(target.numel(),)](
            saved_logits,
            target,
            stats,
            stats,
            grad_output,
            1,
            0,
            1,
            vocab,
            ctx.ignore_index,
            False,
            ctx.label_smoothing,
            BLOCK_SIZE=block,
            num_warps=16,
        )
        if ctx.overwrite_input:
            torch.autograd.graph.increment_version(saved_logits)
        return saved_logits, None, None, None, None


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    overwrite_input: bool = False,
) -> torch.Tensor:
    """Compute unreduced, single-rank cross entropy over the final dimension.

    ``logits`` is a BF16 or FP32 GPU tensor of shape ``[tokens, vocab]`` or
    ``[batch, sequence, vocab]`` (sequence-first layout is also supported).
    ``target`` is int64 on the same device with shape ``logits.shape[:-1]``;
    entries must be valid vocabulary indices or ``ignore_index``. The returned
    per-token loss is FP32. Masking/reduction and distributed normalization
    remain the caller's responsibility. There is no vocabulary-sharded TP path.

    Forward saves logits plus two FP32 softmax statistics per token. Backward
    reconstructs probabilities and applies the incoming gradient in FP32, then
    stores the logits gradient in the input dtype. This follows current TE
    arithmetic, not older TE's intermediate BF16 probability rounding.

    By default logits are copied and remain unchanged. ``overwrite_input=True``
    requires contiguous logits and reuses their storage during backward. The
    caller must own this storage exclusively: do not read it or use it in
    another autograd branch after backward begins. Both modes support only one
    backward per forward and do not support higher-order gradients.

    This operator has no Transformer Engine dependency. The kernel port and
    upstream attribution are in ``primus_turbo/triton/cross_entropy.py``.
    """
    if logits.ndim not in (2, 3) or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have a nonempty 2D or 3D shape")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("logits must be BF16 or FP32")
    if not logits.is_cuda or target.device != logits.device:
        raise ValueError("logits and target must be on the same GPU")
    if target.dtype != torch.int64:
        raise TypeError("target must have dtype int64")
    if target.shape != logits.shape[:-1]:
        raise ValueError("target.shape must equal logits.shape[:-1]")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be between zero and one")
    if overwrite_input and not logits.is_contiguous():
        raise ValueError("overwrite_input=True requires contiguous logits")
    return _CrossEntropy.apply(logits, target, ignore_index, label_smoothing, overwrite_input)
