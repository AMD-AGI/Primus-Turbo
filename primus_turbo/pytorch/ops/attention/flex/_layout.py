###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bhsd <-> backend-layout plumbing for the dense flex entry.

torch flex speaks ``bhsd`` (``[B, H, S, D]``); ``flash_attn_func`` takes tensors whose
*logical* shape is ``[B, S, H, D]`` and reads the actual memory order back out of the
strides (``attention_utils._infer_qkv_format``). ``transpose(1, 2)`` supplies that
logical shape as a **view** -- the copy only happens if we additionally ask for
``.contiguous()``.

The compat layer used to always ask. It did not need to: ``_infer_qkv_format``
recognises three memory orders as first class -- ``bshd``, ``sbhd`` and ``bhsd`` -- the
aiter backend lists all three in ``_SUPPORTED_QKV_FORMATS``, and both the forward and
the backward allocate their outputs/gradients in whichever order they were handed
(``attention_aiter_impl.execute``, ``FlashAttnFunc.backward``). So a caller whose bytes
are in any of those orders can be handed straight through, and the result transposes
back for free. That removes 3 input copies + 1 output copy per forward, and the
mirror-image copies on the gradients in backward.

Passing through only ``bhsd``-contiguous inputs was not enough
--------------------------------------------------------------
The first version of this gate asked ``t.is_contiguous()``, which is true only for a
genuinely ``bhsd``-contiguous tensor. That covers a synthetic ``bshd`` caller going
through :func:`flex_attention_bshd` -- the double transpose lands back on its own
contiguous buffer -- but it does not cover the caller that matters. Megatron holds
activations ``sbhd``-contiguous and hands attention a permuted *view*
(``PrimusTurboAttention.forward``: ``query.contiguous().permute(1, 0, 2, 3)``), whose
strides are ``sbhd``. ``is_contiguous()`` is False there, so all three inputs were
materialised on every layer of every step -- while the direct ``flash_attn_func`` path
that this layer is benchmarked against read the same strides and copied nothing.

Measured on MI355 (gfx950) at the Llama-70B training shape (B=4, S=8192, Hq=32, Hkv=8,
D=128, bf16), flex minus direct, per attention call:

    input layout            extra live memory     extra time
    bshd-contiguous              0 MB              +0.09 ms
    sbhd-contiguous (Megatron) +384 MB             +0.36 ms

384 MB is exactly q + k + v (256 + 64 + 64), i.e. all three copies, retained for
backward -- 12 GB across 32 layers. That is the whole of the end-to-end gap the flex
arm showed against the turbo arm, and none of it was visible to a benchmark that
allocated its inputs ``bshd``-contiguous.

So the gate asks the real question instead of a proxy for it: *can the backend address
these bytes as they lie?*

The one case that still materialises
-----------------------------------
Two dense backends (FlyDSL, HipKittens) are compiled to address ``sbhd`` and are gated
on ``attention_impl._sbhd_layout``, which accepts ``bshd`` only when ``B == 1`` (at a
batch of one the two orders are the same bytes). A ``B == 1`` tensor handed through in
some other order would therefore lose access to those kernels. Since attention at
``B == 1`` is the long-sequence case -- ``O(S**2)`` work against an ``O(S*H*D)`` copy --
keeping the faster kernel is worth the copy, so ``B == 1`` still materialises exactly as
before. Numerics are unaffected either way.
"""

from typing import Optional, Tuple

import torch

# Batch size from which handing the backend a strided view is unambiguously better.
# Below it (i.e. B == 1) a bshd-contiguous copy keeps the sbhd-only FlyDSL / HipKittens
# backends eligible -- see the module docstring.
_LAYOUT_PASSTHROUGH_MIN_BATCH = 2


def _backend_format(t: torch.Tensor) -> Optional[str]:
    """The memory order the backend would read out of a ``[B, S, H, D]``-shaped ``t``.

    Mirrors ``attention_utils._infer_qkv_format._infer_format``, with one deliberate
    difference: that function *asserts* on anything it cannot classify, because by then
    the caller has committed to the kernel. Here the answer feeds a decision -- pass
    through, or materialise -- so an unrecognised layout is a ``None`` to fall back on,
    not an error. Keep the three orderings in step with ``_infer_qkv_format``; a layout
    accepted here but rejected there would turn a copy we skipped into an assertion
    inside the kernel call.
    """
    if t.ndim != 4:
        return None
    s0, s1, s2, s3 = t.stride()
    if s3 != 1:
        return None
    if s0 >= s1 >= s2:
        return "bshd"
    if s1 >= s0 >= s2:
        return "sbhd"
    if s0 >= s2 >= s1:
        return "bhsd"
    return None


def to_backend_layout(t: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` -> a ``[B, S, H, D]``-shaped tensor ``flash_attn_func`` can address.

    Single-tensor form, kept for callers that hold only one tensor. Prefer
    :func:`to_backend_layout_qkv` for a q/k/v triple: the backend requires all three to
    agree on one memory order, and deciding per tensor cannot guarantee that.
    """
    view = t.transpose(1, 2)
    if t.shape[0] >= _LAYOUT_PASSTHROUGH_MIN_BATCH and _backend_format(view) is not None:
        return view
    return view.contiguous()


def to_backend_layout_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a bhsd q/k/v triple together, passing through only if all three agree.

    ``_infer_qkv_format`` classifies each tensor and then asserts they match, so the
    passthrough decision has to be made for the triple at once. Deciding per tensor
    could hand the backend a ``sbhd`` q beside a ``bshd`` k -- each individually
    addressable, the combination an assertion failure inside the kernel. When the three
    do not agree, materialise all of them, which lands every one in ``bshd``.
    """
    views = tuple(t.transpose(1, 2) for t in (query, key, value))
    if query.shape[0] >= _LAYOUT_PASSTHROUGH_MIN_BATCH:
        formats = {_backend_format(v) for v in views}
        if len(formats) == 1 and None not in formats:
            return views
    return tuple(v.contiguous() for v in views)


def from_backend_layout(out: torch.Tensor) -> torch.Tensor:
    """``[B, S, H, D]`` backend output -> ``[B, H, S, D]`` contiguous, copying only if needed.

    When the backend ran on a ``bhsd`` passthrough it allocated its output in bhsd order,
    so the transpose back is already contiguous and nothing is copied. When it ran on a
    materialised bshd buffer -- or on a ``sbhd`` passthrough, whose output comes back in
    that same order -- this copies, exactly as before. Callers that want the backend's
    own buffer untouched use ``_return_bshd`` and never reach this.
    """
    res = out.transpose(1, 2)
    return res if res.is_contiguous() else res.contiguous()
