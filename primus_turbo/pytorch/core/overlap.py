###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""PRIMUS_OVERLAP_*: run independent kernel chains of the training step on side streams.

Two mechanisms, both bit-identical to the serial step (same kernels, same inputs, same reduction orders):

1. ``defer(fn, *keep)`` (stream "wgrad"): weight gradients are only read by the optimizer after the backward, so a
   wgrad that writes ``main_grad`` can run on a side stream while the main stream continues with the dgrad chain.
   ``defer`` forks the side stream off the current stream (so the side work depends on everything issued so far,
   e.g. the dY quantization), runs ``fn`` there and keeps ``keep`` (every tensor ``fn`` touches) alive until
   ``join()``. Only work whose outputs nobody reads before ``join()`` may be deferred; each weight's ``main_grad``
   is written by its own wgrad only, and all deferred work is issued in program order on one stream, so the side
   and main streams never write the same buffer.

2. ``branch()`` / ``ForkGrad`` (stream "branch"): an independent forward branch (e.g. the MoE shared expert) is
   issued on a side stream; autograd then runs its backward on that stream too. ``ForkGrad`` hands the branch input
   to the consumers as aliases and sums their gradients in a fixed order (the order the serial autograd engine
   accumulates them), so the input gradient keeps its bits even though the branch's nodes are created earlier.

``join()`` makes the current stream wait for every side stream that has un-joined work; the framework calls it after
each microbatch's backward (before grad-norm / optimizer / the next microbatch), so inside a captured CUDA graph the
forks and joins are edges of the same capture.

Switches (default off; read once at import):
  PRIMUS_OVERLAP_WGRAD=1       dense block-FP8 Linear wgrads (dsv4up / g90, fused main_grad) on the side stream
  PRIMUS_OVERLAP_MOE_WGRAD=1   MegaMoE fp8_block expert wgrads (vkp / vk) on the side stream
  PRIMUS_OVERLAP_SHARED=1      (Primus) MoE shared expert forward (and so its backward) on the branch stream
  PRIMUS_OVERLAP_INDEXER=1     (Primus) CSA indexer (forward only) on the "indexer" side stream
"""

import os

import torch

__all__ = ["enabled", "side_stream", "defer", "join", "pending", "branch", "ForkGrad"]

_FLAGS = {k: os.environ.get(f"PRIMUS_OVERLAP_{k}", "0") == "1" for k in ("WGRAD", "MOE_WGRAD", "SHARED", "INDEXER")}
_side = {}
_used = {}  # (device index, name) -> side stream with un-joined work
_keep = []  # tensors read/written by un-joined deferred work


def enabled(name: str) -> bool:
    """Switch ``PRIMUS_OVERLAP_<name>`` (read once at import; tests may set ``_FLAGS``)."""
    return _FLAGS.get(name, False)


def side_stream(name: str = "wgrad", device=None) -> torch.cuda.Stream:
    """One lazily created side stream per (device, name); first created eagerly, before any graph capture."""
    dev = torch.cuda.current_device() if device is None else torch.device(device).index
    if dev is None:
        dev = torch.cuda.current_device()
    s = _side.get((dev, name))
    if s is None:
        s = _side[(dev, name)] = torch.cuda.Stream(device=dev)
    return s


def defer(fn, *keep, device=None):
    """Run ``fn()`` on the "wgrad" side stream after everything issued so far on the current stream."""
    s = side_stream("wgrad", device)
    cur = torch.cuda.current_stream(s.device)
    s.wait_stream(cur)
    with torch.cuda.stream(s):
        out = fn()
    _used[(s.device.index, "wgrad")] = s
    _keep.extend(t for t in keep if t is not None)
    if out is not None:
        _keep.append(out)
    return out


class branch:
    """``with branch(name) as br:`` issue the body on the side stream ``name`` after the current stream's work;
    call ``done(t)`` on the branch outputs after the main stream's own work to make the main stream wait for them."""

    def __init__(self, name: str = "branch", device=None):
        self.name = name
        self.side = side_stream(name, device)
        self.main = torch.cuda.current_stream(self.side.device)

    def __enter__(self):
        self.side.wait_stream(self.main)
        _used[(self.side.device.index, self.name)] = self.side
        self._ctx = torch.cuda.stream(self.side)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False

    def done(self, *outs):
        """Main stream waits for the branch; the outputs (allocated on the branch stream) are marked as used there."""
        self.main.wait_stream(self.side)
        for t in outs:
            if isinstance(t, torch.Tensor):
                t.record_stream(self.main)
        return outs[0] if len(outs) == 1 else outs


class ForkGrad(torch.autograd.Function):
    """``a_0, ..., a_{n-1} = ForkGrad.apply(x, n, order)``: n aliases of ``x``; backward returns the sum of their
    gradients accumulated in ``order`` (left to right: ((g[o0] + g[o1]) + g[o2]) ...), i.e. the order the serial
    autograd engine would add them in. Missing (None) gradients are skipped."""

    @staticmethod
    def forward(ctx, x, n, order):
        ctx.order = tuple(order)
        return tuple(x.view_as(x) for _ in range(n))

    @staticmethod
    def backward(ctx, *grads):
        acc = None
        for i in ctx.order:
            g = grads[i]
            if g is None:
                continue
            acc = g if acc is None else acc + g
        return acc, None, None


def pending() -> bool:
    return bool(_used)


def join():
    """Make the current stream wait for all side streams with un-joined work, then release the kept tensors."""
    if not _used:
        return
    for (dev, _), s in list(_used.items()):
        torch.cuda.current_stream(s.device).wait_stream(s)
    _used.clear()
    _keep.clear()
