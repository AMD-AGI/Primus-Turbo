"""Baseline: Primus-Turbo Triton attention, forward + fused one-kernel backward.

Exposed as a callable PyTorch op with the signature every implementation in this
job shares:

    attention(q, k, v, causal=True, softmax_scale=None) -> out

    q       [B, Sq,  Hq,  D]  bf16, bshd
    k, v    [B, Skv, Hkv, D]  bf16, bshd
    out     [B, Sq,  Hq,  D]  bf16

``out.backward(dout)`` fills ``q.grad``, ``k.grad``, ``v.grad``.

THE BACKEND IS PINNED STRUCTURALLY. The spec's `op.reference.api` asks for
``GlobalBackendManager.set_attn_backend(BackendType.TRITON, ...)``. That module
does not exist in the pinned image (see PROVENANCE.md), so the pin is achieved a
stronger way instead: the kernels are vendored under ``vendor/`` and imported
from there, and no dispatcher is reachable at all. There is exactly one kernel
this file can call. See PROVENANCE.md.

Kernels are built by Triton at first call and cached by Triton's own JIT cache.
Nothing here compiles inside a timing loop.
"""

import os
import sys

import torch

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    # insert(0): shadow any primus_turbo installed in the image. The installed
    # copy is an older commit (f857e429) and differs materially; the baseline is
    # frozen, so it must not drift with the image.
    sys.path.insert(0, _VENDOR)

if "primus_turbo" in sys.modules and not getattr(
    sys.modules["primus_turbo"], "__file__", ""
).startswith(_VENDOR):
    raise RuntimeError(
        "an installed primus_turbo was imported before this module; the vendored "
        "baseline would not be the code under measurement"
    )

from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (  # noqa: E402
    dense_fused_backward,
    fused_backward_eligible,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (  # noqa: E402
    dense_forward,
)

NAME = "baseline"


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale):
        out, lse = dense_forward(q, k, v, softmax_scale, causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        if not fused_backward_eligible(q, k.shape[1], None):
            # The gate is a performance gate upstream, but here it is a
            # correctness gate: the fused kernel is the thing being measured.
            # Silently taking another path would report the wrong kernel.
            raise RuntimeError(
                "fused backward is not eligible for this shape/dtype; the "
                "baseline measures the fused kernel and nothing else"
            )
        dq, dk, dv = dense_fused_backward(
            dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal
        )
        return dq, dk, dv, None, None


def attention(q, k, v, causal=True, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _Attention.apply(q, k, v, causal, softmax_scale)


def fingerprint():
    """What actually ran, asserted before any timing is believed.

    `op.reference.api` is explicit that "the config that was asked for must be
    ASSERTED to be the config that ran, before any timing", and gives the failure
    signature: a sweep where every candidate returns the same time to within
    noise, which reads as "this knob does nothing" rather than as a broken
    harness. The tile fields below are the ones `dense_fused_backward` overrides
    per call, so the static config table is NOT sufficient on its own.
    """
    import hashlib

    from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (
        fused_backward_tile,
    )
    from primus_turbo.triton.attention.fused_mha_bwd_kernel import get_fused_bwd_config

    digests = {}
    for rel in ("primus_turbo/triton/attention/attention_kernel.py",
                "primus_turbo/triton/attention/fused_mha_bwd_kernel.py"):
        with open(os.path.join(_VENDOR, rel), "rb") as fh:
            digests[os.path.basename(rel)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    cfg = dict(get_fused_bwd_config()["onekernel"])
    return {"onekernel": cfg, "tile_for_8192": fused_backward_tile(8192), "sha256": digests}
