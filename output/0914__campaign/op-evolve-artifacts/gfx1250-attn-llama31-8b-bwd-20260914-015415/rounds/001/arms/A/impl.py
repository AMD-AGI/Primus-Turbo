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

import builtins
import os
import sys

import torch

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def _is_pt(name):
    return name == "primus_turbo" or name.startswith("primus_turbo.")


# Process-wide, keyed by this directory: `validation.py` execs each impl.py
# several times in one process and the vendored package registers a torch
# opaque type at import, which raises on a second registration.
_CACHE = builtins.__dict__.setdefault("_op_evolve_vendor_cache", {})


def _allow_reregistering_opaque_types():
    """Second half of the same harness fix.

    `primus_turbo.pytorch.core.low_precision` calls `register_opaque_type` at
    import, and torch's opaque-type registry is process-global and keyed by
    qualname. So the SECOND primus_turbo tree imported into this process --
    ours and then `op/baseline`'s -- dies with "Type '...Float8QuantConfig' is
    already registered as an opaque type", whichever order they come in.

    Make the C-level registration idempotent instead of unregistering ours
    afterwards: unregistering perturbs a registry that the dispatcher also
    reads, and it cost this round two runs that died later, elsewhere, with
    `schema_.has_value() INTERNAL ASSERT FAILED`. The types are fp8
    quantisation configs; the two registrations are the same class from two
    identical copies of the same file, and nothing in the bf16 attention path
    under measurement touches them.
    """
    if getattr(torch._C._register_opaque_type, "_op_evolve_shim", False):
        return
    _orig = torch._C._register_opaque_type

    def _shim(name):
        if torch._C._is_opaque_type_registered(name):
            return
        return _orig(name)

    _shim._op_evolve_shim = True
    torch._C._register_opaque_type = _shim


def _import_vendored():
    """Import THIS directory's vendored primus_turbo, leaving sys.modules as found.

    CHANGED round 1, and it is a harness fix rather than an optimisation.
    `validation.py` loads several impl.py files into ONE process (candidate,
    beat, baseline, and back again). Each has its own `vendor/` tree, and
    `op/baseline/impl.py` raises if a `primus_turbo` rooted anywhere else is
    already in `sys.modules`. The original module-level import left ours there,
    so the first arm measured after the candidate died on that guard and NO
    candidate outside `op/baseline/` could be validated at all -- independent of
    what it changed. Importing under a stripped-and-restored `sys.modules`, once
    per directory, gives every arm a private copy while leaving the guard intact
    and the vendor pin exactly as strong as before: there is still only one tree
    this file can import from, no dispatcher is reachable, and the assert below
    fails loudly if anything else answered the import.
    """
    if _VENDOR in _CACHE:
        return _CACHE[_VENDOR]
    saved = {k: v for k, v in sys.modules.items() if _is_pt(k)}
    for k in saved:
        del sys.modules[k]
    _allow_reregistering_opaque_types()
    sys.path.insert(0, _VENDOR)
    try:
        import primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl as fused
        import primus_turbo.pytorch.kernels.attention.attention_triton_impl as tri
        import primus_turbo.triton.attention.fused_mha_bwd_kernel as bwd_kernel
        for mod in (fused, tri, bwd_kernel):
            assert mod.__file__.startswith(_VENDOR), mod.__file__
    finally:
        sys.path.remove(_VENDOR)
        # Hold a reference to EVERY module the import created, not just the
        # three named above. The vendored package defines its forward as a
        # `torch.library.custom_op` in a module nothing else keeps alive; once
        # that module is dropped from `sys.modules` its `Library` destructor
        # deregisters the schema, and the next call dies at a whim of the
        # collector with `schema_.has_value() INTERNAL ASSERT FAILED ... Tried
        # to access the schema for .` -- which is exactly what it did.
        # Move, do not delete. The tree stays registered in `sys.modules` under a
        # private prefix, so no module object is ever torn down and no
        # `torch.library` registration it owns is ever destroyed; the name
        # `primus_turbo` is simply free again, which is all `op/baseline`'s guard
        # looks at. Deleting the modules and holding them in a list instead was
        # not enough: two runs died later with
        # `schema_.has_value() INTERNAL ASSERT FAILED ... Tried to access the
        # schema for`, the dispatcher having lost an operator whose defining
        # module Python had begun to tear down.
        _prefix = "_op_evolve_%d." % len(_CACHE)
        for k in [k for k in sys.modules if _is_pt(k)]:
            sys.modules[_prefix + k] = sys.modules.pop(k)
        sys.modules.update(saved)
    _CACHE[_VENDOR] = (fused, tri, bwd_kernel)
    return _CACHE[_VENDOR]


_FUSED, _TRI, _BWD_KERNEL = _import_vendored()
dense_fused_backward = _FUSED.dense_fused_backward
fused_backward_eligible = _FUSED.fused_backward_eligible
dense_forward = _TRI.dense_forward

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

    fused_backward_tile = _FUSED.fused_backward_tile
    get_fused_bwd_config = _BWD_KERNEL.get_fused_bwd_config

    digests = {}
    for rel in ("primus_turbo/triton/attention/attention_kernel.py",
                "primus_turbo/triton/attention/fused_mha_bwd_kernel.py"):
        with open(os.path.join(_VENDOR, rel), "rb") as fh:
            digests[os.path.basename(rel)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    cfg = dict(get_fused_bwd_config()["onekernel"])
    return {"onekernel": cfg, "tile_for_8192": fused_backward_tile(8192), "sha256": digests}
