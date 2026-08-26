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
recognises ``bhsd`` as a first-class layout (``s0 >= s2 >= s1``), the aiter backend
lists it in ``_SUPPORTED_QKV_FORMATS``, and both the forward and the backward allocate
their outputs/gradients in that same order
(``attention_aiter_impl.execute``, ``FlashAttnFunc.backward``). So a bhsd caller can be
handed straight through, and the result transposes back to a contiguous ``[B, H, S, D]``
for free. That removes 3 input copies + 1 output copy per forward, and the mirror-image
copies on the gradients in backward.

The one case that still materialises
-----------------------------------
Two dense backends (FlyDSL, HipKittens) are compiled to address ``sbhd`` and are gated
on ``attention_impl._sbhd_layout``, which accepts ``bshd`` only when ``B == 1`` (at a
batch of one the two orders are the same bytes). A ``B == 1`` bhsd tensor handed through
as ``bhsd`` would therefore lose access to those kernels. Since attention at ``B == 1``
is the long-sequence case -- ``O(S**2)`` work against an ``O(S*H*D)`` copy -- keeping the
faster kernel is worth the copy, so ``B == 1`` still materialises exactly as before. This
is the *only* behavioural difference between the two branches; numerics are unaffected
either way.
"""

import torch

# Batch size from which handing the backend a bhsd view is unambiguously better. Below it
# (i.e. B == 1) a bshd-contiguous copy keeps the sbhd-only FlyDSL / HipKittens backends
# eligible -- see the module docstring.
_LAYOUT_PASSTHROUGH_MIN_BATCH = 2


def to_backend_layout(t: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` -> a ``[B, S, H, D]``-shaped tensor ``flash_attn_func`` can address.

    Returns a zero-copy view when the backend can read the bhsd bytes directly, and a
    bshd-contiguous copy otherwise (``B == 1``, or any input whose strides are not plain
    bhsd -- e.g. the transposed view ``flex_attention_bshd`` passes in, where
    ``.contiguous()`` collapses back onto the caller's original buffer and copies nothing).
    """
    view = t.transpose(1, 2)
    if t.is_contiguous() and t.shape[0] >= _LAYOUT_PASSTHROUGH_MIN_BATCH:
        return view
    return view.contiguous()


def from_backend_layout(out: torch.Tensor) -> torch.Tensor:
    """``[B, S, H, D]`` backend output -> ``[B, H, S, D]`` contiguous, copying only if needed.

    When the backend ran on the bhsd passthrough it allocated its output in bhsd order,
    so the transpose back is already contiguous and nothing is copied. When it ran on a
    materialised bshd buffer this copies, exactly as before.
    """
    res = out.transpose(1, 2)
    return res if res.is_contiguous() else res.contiguous()
