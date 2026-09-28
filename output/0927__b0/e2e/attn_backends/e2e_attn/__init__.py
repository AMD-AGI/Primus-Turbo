"""Swappable attention backend for the B0 Llama-3.1-8B e2e runs. ONE env var picks the arm:

    E2E_ATTN=turbo            stock Primus-Turbo (wt-bakeoff @1cb2e183) TurboAttention -> Triton.
                              Needs the real primus_turbo on PYTHONPATH; process "P1".
    E2E_ATTN=asm              aiter ASM fwd + hand-launched ASM bwd.                  process "P2"
    E2E_ATTN=fly              FlyDSL fwd r6 + bwd r20 (flydsl 0.3.4.1).               process "P2"
    E2E_ATTN=asm/fly          fwd from one arm, bwd from another (lse conventions match).
    E2E_ATTN="W;C"            per-TRAINING-STEP schedule: comma-separated tokens W played once,
                              then C cycled, e.g. "asm,fly,asm,fly;asm,fly,fly,asm" (ABBA blocks).

The step is each module instance's own forward-call count (no activation checkpointing, so
every attention module runs exactly once per training step); layer 0 logs every arm switch.

Every call is wrapped in profiler ranges `e2e::attn_fwd[<fwd-arm>]` / `e2e::attn_bwd[<bwd-arm>]`
so a trace attributes EVERY GPU kernel launched under them -- adapter copies, GQA sums, zeroing,
dq converts -- to the FA path, not just the named attention kernels.
"""
from __future__ import annotations

import math
import os
import sys

import torch

ENV = "E2E_ATTN"


def _log(msg):
    print(f"[e2e_attn] {msg}", file=sys.stderr, flush=True)


def parse_schedule(spec: str):
    """-> (warm list, cycle list) of (fwd_arm, bwd_arm) tuples."""
    spec = (spec or "").strip()
    if not spec:
        raise RuntimeError(f"{ENV} is not set -- refusing to guess an attention arm")

    def tok(t):
        t = t.strip()
        f, _, b = t.partition("/")
        return (f, b or f)

    warm, _, cyc = spec.rpartition(";")
    warm_l = [tok(t) for t in warm.split(",") if t.strip()] if warm else []
    cyc_l = [tok(t) for t in cyc.split(",") if t.strip()]
    if not cyc_l:
        raise RuntimeError(f"{ENV}={spec!r}: empty cycle")
    return warm_l, cyc_l


def arm_for_step(sched, step):
    warm, cyc = sched
    if step < len(warm):
        return warm[step]
    return cyc[(step - len(warm)) % len(cyc)]


# --------------------------------------------------------------------------- stats
STATS = {"copies": 0, "copy_bytes": 0}


def _contig(t, name):
    if t.is_contiguous():
        return t
    STATS["copies"] += 1
    STATS["copy_bytes"] += t.numel() * t.element_size()
    if STATS["copies"] <= 4:
        _log(f"adapter copy: {name} {tuple(t.shape)} stride {t.stride()} -> contiguous()")
    with torch.profiler.record_function(f"e2e::adapter_contiguous[{name}]"):
        return t.contiguous()


# --------------------------------------------------------------------------- asm / fly
class E2EAttnFunc(torch.autograd.Function):
    """q/k/v BSHD [B,S,H,D] bf16 -> o BSHD. The bwd arm may differ from the fwd arm."""

    @staticmethod
    def forward(ctx, q, k, v, scale, fwd_arm, bwd_arm):
        from . import arms
        with torch.profiler.record_function(f"e2e::attn_fwd[{fwd_arm}]"):
            q, k, v = _contig(q, "q"), _contig(k, "k"), _contig(v, "v")
            o, lse = arms.get_fwd(fwd_arm)(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale, ctx.bwd_arm = scale, bwd_arm
        return o

    @staticmethod
    def backward(ctx, do):
        from . import arms
        q, k, v, o, lse = ctx.saved_tensors
        with torch.profiler.record_function(f"e2e::attn_bwd[{ctx.bwd_arm}]"):
            do = _contig(do, "do")
            dq, dk, dv = arms.get_bwd(ctx.bwd_arm)(do, q, k, v, o, lse, ctx.scale)
        return dq, dk, dv, None, None, None


# --------------------------------------------------------------------------- turbo
class E2ETurboFunc(torch.autograd.Function):
    """Runs the REAL product path (primus_turbo flash_attn_func -> FlashAttnFunc -> Triton) as an
    inner autograd graph, only to put the same e2e::attn_fwd/bwd ranges around it. The inner
    tensors are detached views (no copy); backward is torch.autograd.grad over that graph."""

    @staticmethod
    def forward(ctx, q, k, v, scale, attn_fn):
        with torch.profiler.record_function("e2e::attn_fwd[turbo]"):
            qi, ki, vi = (t.detach().requires_grad_(t.requires_grad) for t in (q, k, v))
            with torch.enable_grad():
                o = attn_fn(qi, ki, vi, softmax_scale=scale, causal=True)
        ctx.inner = (qi, ki, vi, o)
        return o.detach()

    @staticmethod
    def backward(ctx, do):
        qi, ki, vi, o = ctx.inner
        ctx.inner = None
        with torch.profiler.record_function("e2e::attn_bwd[turbo]"):
            dq, dk, dv = torch.autograd.grad(o, (qi, ki, vi), do)
        return dq, dk, dv, None, None


_turbo_fn = None


def turbo_pair():
    """The real product entry point. Imported lazily: only legal in a P1 process."""
    global _turbo_fn
    if _turbo_fn is None:
        from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func
        _turbo_fn = flash_attn_func
    return _turbo_fn


# --------------------------------------------------------------------------- module
_instances = 0
_blas_logged = False


def _log_blas_once():
    global _blas_logged
    if _blas_logged:
        return
    _blas_logged = True
    try:
        lib = torch.backends.cuda.preferred_blas_library()
    except Exception as exc:  # pragma: no cover
        lib = f"? ({exc})"
    _log("BLAS TORCH_BLAS_PREFER_HIPBLASLT=%s HIPBLASLT_TENSILE_LIBPATH=%s preferred_blas_library=%s"
         % (os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
            os.environ.get("HIPBLASLT_TENSILE_LIBPATH"), lib))


class E2EAttention(torch.nn.Module):
    """Drop-in for primus_turbo.pytorch.modules.TurboAttention (the converter's constructor
    signature), used for every arm."""

    def __init__(self, dropout_p=0.0, softmax_scale=None, causal=True, fp8_config=None, **kw):
        super().__init__()
        global _instances
        if not causal or dropout_p or fp8_config is not None:
            raise NotImplementedError(
                f"e2e attention: causal bf16 no-dropout only (causal={causal}, "
                f"dropout_p={dropout_p}, fp8_config={fp8_config})")
        self.softmax_scale = softmax_scale
        self.layer = _instances
        _instances += 1
        self.calls = 0
        self.sched = parse_schedule(os.environ.get(ENV, ""))
        self.is_turbo = self.sched == ([], [("turbo", "turbo")])
        from . import arms
        for f, b in self.sched[0] + self.sched[1]:
            if "turbo" in (f, b) and not self.is_turbo:
                raise RuntimeError(f"{ENV}: turbo cannot be scheduled with other arms "
                                   "(it needs the real primus_turbo, which cannot share a "
                                   "process with flydsl 0.3.4.1)")
            if not self.is_turbo and not (arms.known(f) and arms.known(b)):
                raise RuntimeError(f"{ENV}: unknown arm in {(f, b)}")
        if self.layer == 0:
            _log(f"{ENV}={os.environ.get(ENV)!r} schedule warm={self.sched[0]} "
                 f"cycle={self.sched[1]}")

    def forward(self, q, k, v, bias=None, **kw):
        if bias is not None:
            raise NotImplementedError("e2e attention takes no bias")
        step = self.calls
        self.calls += 1
        scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        if self.is_turbo:
            arm = ("turbo", "turbo")
            out = E2ETurboFunc.apply(q, k, v, scale, turbo_pair())
        else:
            arm = arm_for_step(self.sched, step)
            out = E2EAttnFunc.apply(q, k, v, scale, arm[0], arm[1])
        if self.layer == 0:
            if step == 0:
                _log_blas_once()
                if not self.is_turbo:
                    from . import arms
                    _log(arms.check_versions())
                _log(f"layout q{tuple(q.shape)} {q.dtype} contig={q.is_contiguous()} "
                     f"k{tuple(k.shape)} contig={k.is_contiguous()} "
                     f"v{tuple(v.shape)} contig={v.is_contiguous()}")
            if step == 0 or arm != arm_for_step(self.sched, step - 1) or step < 3:
                _log(f"step_idx={step} arm fwd={arm[0]} bwd={arm[1]} "
                     f"adapter_copies_so_far={STATS['copies']} ({STATS['copy_bytes'] >> 20} MiB)")
        return out
