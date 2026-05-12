###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "moe_permute",
    "moe_unpermute",
]


class _MoEPermute(torch.autograd.Function):
    """Forward = ``permute_preprocessing`` + ``permute``.

    The forward saves ``row_id_map`` (and ``num_dispatched_token_tensor``) so
    the backward can run a single ``unpermute`` over ``grad_permuted_tokens``
    to recover the gradient of the dispatched ``tokens`` (and, when ``probs``
    was provided, the gradient of ``probs`` via ``permuted_probs_grad``).

    Returns
    -------
    permuted_tokens : torch.Tensor
        ``[num_permuted, hidden]`` (same dtype as ``tokens``).
    row_id_map : torch.Tensor
        ``int32 [max_num_dispatched_tokens + pad_multiple, 2 * E + 1]`` map
        emitted by preprocessing — re-usable by a downstream ``_MoEUnpermute``.
    tokens_per_expert : torch.Tensor
        ``int32 [num_local_experts]``.
    overflow_flag : torch.Tensor
        ``int32 [1]`` — non-zero if preprocessing dropped tokens.
    permuted_scaling_factor : Optional[torch.Tensor]
        Only populated when ``use_fp8`` and ``scaling_factor`` is given.
    permuted_probs : Optional[torch.Tensor]
        Only populated when ``probs`` is given.
    """

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        expert_map: torch.Tensor,
        num_dispatched_token_tensor: torch.Tensor,
        num_local_experts: int,
        num_topk: int,
        pad_multiple: int = 0,
        num_permuted_tokens: int = -1,
        scaling_factor: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        scales_per_token: int = 0,
        use_fp8: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        device = tokens.device
        hidden_size = int(tokens.shape[-1])
        num_dispatched = int(tokens.shape[0])
        # The host-side cap on dispatched rows is now read directly from
        # ``expert_map`` — the preprocessing launcher uses ``expert_map.size(0)``
        # to allocate ``row_id_map``, so we mirror that here.
        max_num_dispatched_tokens = int(expert_map.shape[0])

        # State shared by the fast path and the regular path; backward needs
        # all of these whether or not we actually launched a kernel.
        ctx.num_dispatched = num_dispatched
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.use_fp8 = use_fp8
        ctx.with_probs = probs is not None
        ctx.probs_dtype = probs.dtype if probs is not None else None

        # Fast path: no dispatched tokens or zero reserved capacity. The
        # preprocessing kernel hard-asserts ``max_num_dispatched_tokens > 0``,
        # so we have to short-circuit before reaching it. Mirrors the
        # Triton ``TokenPermuteMaskMap`` empty-input behavior.
        if num_dispatched == 0 or max_num_dispatched_tokens == 0:
            int_opts = dict(dtype=torch.int32, device=device)
            row_id_map_rows = max(0, max_num_dispatched_tokens + int(pad_multiple))
            # Zero-fill so an unrelated consumer that inspects the map without
            # going through ``_MoEUnpermute`` doesn't observe garbage. The map
            # is small so the cost is negligible.
            row_id_map = torch.zeros((row_id_map_rows, 2 * num_local_experts + 1), **int_opts)
            tokens_per_expert = torch.zeros((num_local_experts,), **int_opts)
            overflow_flag = torch.zeros((1,), **int_opts)
            permuted_tokens = tokens.new_empty((0, hidden_size))
            if use_fp8 and scaling_factor is not None:
                assert scales_per_token > 0, "_MoEPermute: scales_per_token must be > 0 when use_fp8=True"
                permuted_scaling_factor: Optional[torch.Tensor] = scaling_factor.new_empty(
                    (0, scales_per_token)
                )
            else:
                permuted_scaling_factor = None
            permuted_probs: Optional[torch.Tensor] = probs.new_zeros((0,)) if probs is not None else None

            ctx.save_for_backward(row_id_map, num_dispatched_token_tensor)
            return (
                permuted_tokens,
                row_id_map,
                tokens_per_expert,
                overflow_flag,
                permuted_scaling_factor,
                permuted_probs,
            )

        # 1) Preprocessing: row_id_map / tokens_per_expert / overflow_flag.
        row_id_map, tokens_per_expert, overflow_flag = (
            torch.ops.primus_turbo_cpp_extension.permute_preprocessing(
                expert_map,
                num_dispatched_token_tensor,
                num_local_experts,
                num_topk,
                pad_multiple,
                num_permuted_tokens,
            )
        )

        # 2) Pick the permuted-output row count.
        # Prefer the caller-provided cap to avoid a host sync. Fall back to a
        # ``.item()`` pull only when the cap is unknown (``num_permuted_tokens < 0``).
        if num_permuted_tokens is not None and num_permuted_tokens >= 0:
            num_permuted_alloc = int(num_permuted_tokens)
        else:
            num_permuted_alloc = int(tokens_per_expert.sum().item())

        # 3) Allocate outputs.
        permuted_tokens = torch.empty((num_permuted_alloc, hidden_size), dtype=tokens.dtype, device=device)
        if use_fp8 and scaling_factor is not None:
            assert scales_per_token > 0, "_MoEPermute: scales_per_token must be > 0 when use_fp8=True"
            permuted_scaling_factor = torch.empty(
                (num_permuted_alloc, scales_per_token),
                dtype=scaling_factor.dtype,
                device=device,
            )
        else:
            permuted_scaling_factor = None
        if probs is not None:
            # ``torch.zeros`` so the trailing ``[Σ real + pad, num_permuted_alloc)``
            # range — which the kernel never writes when the caller-provided cap
            # over-allocates — stays at 0 instead of holding undefined memory.
            permuted_probs = torch.zeros((num_permuted_alloc,), dtype=probs.dtype, device=device)
        else:
            permuted_probs = None
        with_probs = (probs is not None) and (permuted_probs is not None)

        # 4) Run the permute kernel.
        torch.ops.primus_turbo_cpp_extension.permute(
            tokens,
            permuted_tokens,
            scaling_factor,
            permuted_scaling_factor,
            probs,
            permuted_probs,
            row_id_map,
            num_dispatched_token_tensor,
            pad_multiple,
            num_local_experts,
            hidden_size,
            scales_per_token,
            use_fp8,
            with_probs,
            num_permuted_alloc,
        )

        # 5) Save state for backward (unpermute).
        ctx.save_for_backward(row_id_map, num_dispatched_token_tensor)
        return (
            permuted_tokens,
            row_id_map,
            tokens_per_expert,
            overflow_flag,
            permuted_scaling_factor,
            permuted_probs,
        )

    @staticmethod
    def backward(
        ctx,
        grad_permuted_tokens: torch.Tensor,
        row_id_map_grad: Optional[torch.Tensor],
        tokens_per_expert_grad: Optional[torch.Tensor],
        overflow_flag_grad: Optional[torch.Tensor],
        permuted_scaling_factor_grad: Optional[torch.Tensor],
        permuted_probs_grad: Optional[torch.Tensor],
    ):
        # The HIP unpermute kernel accepts bfloat16 / float16 only; we don't
        # currently support FP8 backward (gradients are typically bf16 anyway).
        assert not ctx.use_fp8, (
            "_MoEPermute.backward: FP8 backward is not supported "
            "(unpermute kernel only accepts bfloat16/float16 inputs)."
        )

        row_id_map, num_dispatched_token_tensor = ctx.saved_tensors
        grad_permuted_tokens = grad_permuted_tokens.contiguous()
        device = grad_permuted_tokens.device
        grad_tokens = torch.empty(
            (ctx.num_dispatched, ctx.hidden_size),
            dtype=grad_permuted_tokens.dtype,
            device=device,
        )

        # Probability gradient: route ``permuted_probs_grad`` through unpermute
        # so the scatter writes ``probs.grad`` at ``[token_id, expert_idx]``.
        # The kernel zeros every ``probs[t, :]`` slot up front, so allocating
        # with ``torch.empty`` is safe.
        if ctx.with_probs and permuted_probs_grad is not None:
            permuted_probs_grad = permuted_probs_grad.contiguous()
            grad_probs: Optional[torch.Tensor] = torch.empty(
                (ctx.num_dispatched, ctx.num_local_experts),
                dtype=ctx.probs_dtype,
                device=device,
            )
            unpermute_with_probs = True
        else:
            permuted_probs_grad = None
            grad_probs = None
            unpermute_with_probs = False

        # Fast path: the forward short-circuited (or the kernel got nothing to
        # scatter from). Zero out the output buffers so callers don't observe
        # uninitialized memory — only the "matched" rows would otherwise be
        # written by the kernel.
        if ctx.num_dispatched == 0 or grad_permuted_tokens.shape[0] == 0:
            grad_tokens.zero_()
            if grad_probs is not None:
                grad_probs.zero_()
        else:
            torch.ops.primus_turbo_cpp_extension.unpermute(
                grad_permuted_tokens,
                grad_tokens,
                permuted_probs_grad,
                grad_probs,
                row_id_map,
                num_dispatched_token_tensor,
                ctx.num_local_experts,
                ctx.hidden_size,
                unpermute_with_probs,
            )

        # Match the 11 forward inputs; ``tokens`` and ``probs`` may receive a
        # gradient — everything else is a non-differentiable hyper-parameter.
        return (
            grad_tokens,
            None,  # expert_map
            None,  # num_dispatched_token_tensor
            None,  # num_local_experts
            None,  # num_topk
            None,  # pad_multiple
            None,  # num_permuted_tokens
            None,  # scaling_factor
            grad_probs,  # probs
            None,  # scales_per_token
            None,  # use_fp8
        )


class _MoEUnpermute(torch.autograd.Function):
    """Forward = ``unpermute``.

    Saves ``row_id_map`` so the backward can re-run ``permute`` on
    ``grad_unpermuted_tokens`` (and, when ``permuted_probs`` was provided, on
    ``unpermuted_probs_grad``).
    """

    @staticmethod
    def forward(
        ctx,
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        num_dispatched_tokens_tensor: torch.Tensor,
        num_local_experts: int,
        permuted_probs: Optional[torch.Tensor],
        pad_multiple: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = permuted_tokens.device
        hidden_size = int(permuted_tokens.shape[-1])
        num_permuted = int(permuted_tokens.shape[0])

        # ``row_id_map`` was allocated by ``_MoEPermute.forward`` with shape
        # ``[max_num_dispatched_tokens + pad_multiple, 2*E + 1]``. Strip the
        # padding rows to recover the dispatched-token count for the output
        # buffer's shape — combine consumers expect [num_dispatched, hidden].
        assert row_id_map.dim() == 2, "row_id_map must be 2D"
        num_dispatched = max(0, int(row_id_map.shape[0]) - int(pad_multiple))

        ctx.num_permuted = num_permuted
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.with_probs = permuted_probs is not None
        ctx.permuted_probs_dtype = permuted_probs.dtype if permuted_probs is not None else None

        unpermuted_tokens = torch.empty(
            (num_dispatched, hidden_size),
            dtype=permuted_tokens.dtype,
            device=device,
        )
        unpermuted_probs: Optional[torch.Tensor] = None
        if permuted_probs is not None:
            # ``torch.zeros`` so rows that the kernel never visits (e.g. when
            # ``num_dispatched > Σ row_id_map`` due to padding) stay at 0.
            unpermuted_probs = torch.zeros(
                (num_dispatched, num_local_experts),
                dtype=permuted_probs.dtype,
                device=device,
            )

        # Fast path: nothing to scatter. Skip the kernel and just hand back
        # the (zero-filled) buffers.
        if num_permuted == 0 or num_dispatched == 0:
            unpermuted_tokens.zero_()
            ctx.save_for_backward(row_id_map, num_dispatched_tokens_tensor)
            return unpermuted_tokens, unpermuted_probs

        with_probs = (permuted_probs is not None) and (unpermuted_probs is not None)
        torch.ops.primus_turbo_cpp_extension.unpermute(
            permuted_tokens,
            unpermuted_tokens,
            permuted_probs,
            unpermuted_probs,
            row_id_map,
            num_dispatched_tokens_tensor,
            num_local_experts,
            hidden_size,
            with_probs,
        )

        ctx.save_for_backward(row_id_map, num_dispatched_tokens_tensor)
        return unpermuted_tokens, unpermuted_probs

    @staticmethod
    def backward(
        ctx,
        grad_unpermuted_tokens: torch.Tensor,
        unpermuted_probs_grad: Optional[torch.Tensor],
    ):
        row_id_map, num_dispatched_tokens_tensor = ctx.saved_tensors
        grad_unpermuted_tokens = grad_unpermuted_tokens.contiguous()
        device = grad_unpermuted_tokens.device
        # ``torch.zeros`` (not ``empty``): the inner permute kernel call below
        # passes ``pad_multiple=0`` so padding-token rows (token_id ≥ N) are
        # early-returned. That leaves the per-expert padded slots — referenced
        # only by negative dst_row entries in the padding-row range of
        # ``row_id_map`` — without an explicit kernel write. Pre-zeroing keeps
        # those slots as 0 (matching the forward-emitted padded data), so the
        # gradient handed back to the upstream op is well-defined.
        grad_permuted = torch.zeros(
            (ctx.num_permuted, ctx.hidden_size),
            dtype=grad_unpermuted_tokens.dtype,
            device=device,
        )

        # Probability gradient: route ``unpermuted_probs_grad`` through the
        # permute kernel to gather ``permuted_probs.grad`` at the dispatched
        # rows. ``torch.zeros`` so the trailing ``[Σ real + pad, num_permuted)``
        # capacity stays at 0 (the kernel only writes the matched/padded rows).
        if ctx.with_probs and unpermuted_probs_grad is not None:
            unpermuted_probs_grad = unpermuted_probs_grad.contiguous()
            grad_permuted_probs: Optional[torch.Tensor] = torch.zeros(
                (ctx.num_permuted,),
                dtype=ctx.permuted_probs_dtype,
                device=device,
            )
            permute_with_probs = True
        else:
            unpermuted_probs_grad = None
            grad_permuted_probs = None
            permute_with_probs = False

        # Fast path: nothing was permuted in forward (or the upstream gave us
        # zero rows to scatter from). Zero out the output buffers so callers
        # don't see uninitialized memory.
        if ctx.num_permuted == 0 or grad_unpermuted_tokens.shape[0] == 0:
            grad_permuted.zero_()
            if grad_permuted_probs is not None:
                grad_permuted_probs.zero_()
        else:
            torch.ops.primus_turbo_cpp_extension.permute(
                grad_unpermuted_tokens,
                grad_permuted,
                None,  # scaling_factor
                None,  # output_scaling_factor
                unpermuted_probs_grad,
                grad_permuted_probs,
                row_id_map,
                num_dispatched_tokens_tensor,
                0,
                ctx.num_local_experts,
                ctx.hidden_size,
                0,  # scales_per_token
                False,  # use_fp8
                permute_with_probs,
                ctx.num_permuted,
            )

        # Match the 6 forward inputs; ``permuted_tokens`` and ``permuted_probs``
        # may receive a gradient.
        return (
            grad_permuted,
            None,  # row_id_map
            None,  # num_dispatched_tokens_tensor
            None,  # num_local_experts
            grad_permuted_probs,  # permuted_probs
            None,  # pad_multiple
        )


# -----------------------------------------------------------------------------
# User-facing autograd-aware wrappers.
# -----------------------------------------------------------------------------


def moe_permute(
    tokens: torch.Tensor,
    expert_map: torch.Tensor,
    num_dispatched_token_tensor: torch.Tensor,
    *,
    num_local_experts: int,
    num_topk: int = 0,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    scaling_factor: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    scales_per_token: int = 0,
    use_fp8: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Autograd-aware fused ``preprocessing + permute``.
    Returns ``(permuted_tokens, row_id_map, tokens_per_expert, overflow_flag,
    permuted_scaling_factor, permuted_probs)``. The backward routes the
    activation gradient (``permuted_tokens.grad``) back to ``tokens`` and,
    when ``probs`` was supplied, ``permuted_probs.grad`` back to ``probs``.
    """
    return _MoEPermute.apply(
        tokens,
        expert_map,
        num_dispatched_token_tensor,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
    )


def moe_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens_tensor: torch.Tensor,
    *,
    num_local_experts: int,
    permuted_probs: Optional[torch.Tensor] = None,
    pad_multiple: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Autograd-aware ``unpermute`` (forward) + ``permute`` (backward).

    Returns ``(unpermuted_tokens, unpermuted_probs)``. When ``permuted_probs``
    is given, the backward also routes ``unpermuted_probs.grad`` back to
    ``permuted_probs``.

    ``pad_multiple`` must match the value passed to the upstream ``moe_permute``;
    it is used to strip the padding rows from ``row_id_map`` so the output is
    sized to the real ``num_dispatched`` rather than the padded capacity.
    """
    return _MoEUnpermute.apply(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens_tensor,
        num_local_experts,
        permuted_probs,
        pad_multiple,
    )
