###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers shared across the GEMM / grouped-GEMM op implementations."""

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    Format,
    ScalingRecipe,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.utils import is_gfx1250

# gfx1250's MX GEMM layout, the one its FlyDSL MXFP8 GEMMs read and its quantizers write for these recipes: packed
# E8M0 scales for every operand (an A operand: MX_A) and, for the B operands, codes in 16x16 blocks (an activation's
# or gradient's column copy as the wgrad's B: MX_B; a weight, both copies: MX_W). Only gfx1250 reads the shuffle
# flags this way.
MX_A, MX_B = ScalingRecipe(shuffle_scale=True), ScalingRecipe(shuffle_scale=True, shuffle_out=True)
MX_W = ScalingRecipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)


def _mx1250_choice(grouped: bool):
    """The FP8 (grouped) GEMM backend choice set for the MXFP8 ops of this kind, or None."""
    get = GlobalBackendManager.get_grouped_gemm_backend if grouped else GlobalBackendManager.get_gemm_backend
    return get(PrecisionType.FP8)


def mx1250_main_grad_ok(grouped: bool, main_grad) -> bool:
    """Whether gfx1250's MXFP8 wgrad adds into ``main_grad`` in its epilogue: a tensor its FlyDSL kernels take
    (``mx1250_out_ok``: bf16 / fp32, contiguous, 16-byte aligned), with FlyDSL the backend that runs (no other
    pinned) and no per-op auto-tune request, whose tuner would time the accumulating GEMM on ``main_grad`` itself.
    """
    from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import mx1250_out_ok

    choice = _mx1250_choice(grouped)
    ok = choice is None or (choice.backend in (None, BackendType.FLYDSL) and not choice.auto_tune)
    return (
        ok and isinstance(main_grad, torch.Tensor) and mx1250_out_ok(main_grad.shape, None, True, main_grad)
    )


def mx1250_recipes(who: str, config, out_dtype, a, a_t, *weights, fuse=None, glu_width=None, strict=True):
    """The scaling recipes ``(MX_A, MX_B, MX_W)`` of the MXFP8 op ``who`` on gfx1250: the row and column copies of
    its activation ``a`` / ``a_t``, the copies of its ``weights`` (each followed by its transposed copy or None).

    gfx1250's FlyDSL kernels run MXFP8 in that one layout. A call they do not take (format, dtypes, weight shape,
    ``glu_width``: the fused MLP's I, a pinned backend, a pre-quantized operand in another layout, or, with a
    ``fuse`` pattern, a ``main_grad`` the wgrad cannot add into) raises NotImplementedError, before any weight is
    marked; without ``strict`` it returns None (the caller has another path).
    """
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

    grouped = weights[0].dim() == 3
    align, choice = 32 if grouped else 16, _mx1250_choice(grouped)
    pinned = getattr(choice, "backend", None) not in (None, BackendType.FLYDSL)
    ok = config.format == Format.E4M3 and out_dtype == torch.bfloat16 and not pinned
    ok = ok and (glu_width is None or glu_width % 128 == 0)
    for t, recipe in ((a, MX_A), (a_t, MX_B), *((w, MX_W) for w in weights)):
        if t is None:
            continue
        if isinstance(t, QuantizedTensor):
            ok = ok and t.scaling_recipe == recipe
        else:
            ok = ok and t.dtype == torch.bfloat16
        if recipe is MX_W:
            ok = ok and t.shape[-2] % align == 0 and t.shape[-1] % align == 0
    main_grads = [] if fuse is None else [getattr(w, "main_grad", None) for w in weights[::2]]
    adds = all(g is None or mx1250_main_grad_ok(grouped, g) for g in main_grads)
    if not (ok and adds) and not strict:
        return None
    if not ok:
        operands = [t for t in (a, a_t, *weights) if t is not None]
        got = [
            f"{type(t).__name__}{tuple(t.shape)} {getattr(t, 'scaling_recipe', t.dtype)}" for t in operands
        ]
        raise NotImplementedError(
            f"{who}: MXFP8 on gfx1250 runs on its FlyDSL kernels (no other FP8 GEMM backend pinned), for E4M3 with "
            f"bf16 operands and a bf16 output, weight dims multiples of {align}"
            f"{'' if glu_width is None else ' (the fused MLP: its intermediate width I of 128)'}, and pre-quantized "
            f"operands in gfx1250's MX GEMM layout (create_quantized_weight's; ops.utils MX_A / MX_B / MX_W). Got "
            f"format {config.format}, out_dtype {out_dtype}, operands {got}"
            f"{f', pinned backend {choice.backend.name}' if pinned else ''}."
        )
    if not adds:
        got = [
            (g.dtype, g.is_contiguous(), g.data_ptr() % 16) for g in main_grads if isinstance(g, torch.Tensor)
        ]
        raise NotImplementedError(
            f"{who}: gfx1250's MXFP8 wgrad adds into a main_grad that is bf16 or fp32, contiguous and 16-byte "
            f"aligned, on its FlyDSL kernels without a per-op auto-tune request; got (dtype, contiguous, "
            f"data_ptr % 16) {got}. mx_wgrad_accum_supported tells; without the fuse pattern the wgrad is returned."
        )
    return MX_A, MX_B, MX_W


def mx_wgrad_accum_supported(grouped: bool, main_grad: torch.Tensor) -> bool:
    """Whether the MXFP8 wgrad of ``gemm_fp8`` / (``grouped``) ``grouped_gemm_fp8`` and ``grouped_mlp_fp8`` adds
    into this ``main_grad`` in its epilogue (``fuse_bgrad_accum_pattern`` / ``fuse_wgrad_accum_pattern``) on
    gfx1250, see ``mx1250_main_grad_ok``. A framework asks before it passes the pattern, and adds the returned
    wgrad itself where this is False. Answered for gfx1250 only: False on the other archs, where frameworks keep
    their own rules.
    """
    return is_gfx1250() and mx1250_main_grad_ok(grouped, main_grad)


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


def _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern: Optional[str]):
    """Resolve the weight's gradient-accumulation buffer for the fused wgrad path.

    Returns ``(enabled, main_grad)``. When enabled, the wgrad GEMM writes straight
    into ``main_grad`` and the Function must return no gradient for ``b``.

    Call this at the top of ``forward``: the flag has to be set while the weight
    object is still in hand, because the backward pass only sees the saved tensors.
    """
    if fuse_bgrad_accum_pattern is None:
        return False, None

    assert fuse_bgrad_accum_pattern in ["megatron"], (
        "Only megatron support gradient accumulation fusion currently"
    )

    assert hasattr(b, "grad_added_to_main_grad"), (
        "b.grad_added_to_main_grad must be set up before the backward pass."
    )
    assert hasattr(b, "main_grad"), "b.main_grad must be set up before the backward pass."
    assert isinstance(b.main_grad, torch.Tensor) and (b.main_grad.shape == b.shape), (
        "b.main_grad must be a tensor with the same shape as b"
    )

    # Set in forward, not backward: autograd hands backward the saved tensors, not
    # the parameter object that carries this attribute.
    b.grad_added_to_main_grad = True
    return True, b.main_grad


_dummy_wgrads = {}


def _get_dummy_wgrad(shape: list, dtype: torch.dtype, zero=False) -> torch.Tensor:
    """Returns a dummy tensor of given shape.

    Supports arbitrary rank (2D for plain Linear weights, 3D for stacked
    grouped-linear weights ``(num_gemms, out_features, in_features)``, etc.).
    Tensors are cached by ``(shape, dtype)`` so each distinct weight layout
    only allocates one persistent buffer that gets reused across steps.
    """
    global _dummy_wgrads
    key = (tuple(shape), dtype)
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(
            shape,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[key].fill_(0)
    return _dummy_wgrads[key].detach()
