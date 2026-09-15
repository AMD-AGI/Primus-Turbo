"""Gate and adapter for aiter's prebuilt gfx1250 ASM backward.

Same shape as attention_asm_fwd_impl: a capability check that declines loudly under a trace
flag, and a thin adapter. Anything declined falls through to the path that was already
shipping, so this file cannot make a working configuration worse.

Three kernel launches per backward, not one:

    bwd_hd128_odo_bf16            delta = rowsum(dO * O)
    bwd_hd128_bf16_causal_br_a32_pssk   the body (mask=0 object when non-causal, and the x
                                        grid is halved for causal only); 514
                                        buffer_atomic_add_f32 into an fp32 dq_acc
    bwd_hd128_dq_convert_bf16     dq_acc fp32 -> dq bf16

WHY dk/dv ARE ALLOCATED PER Q HEAD. The body's grid is (kv_tiles, nhead_q, batch), so under
GQA `ratio` workgroups own the same dk/dv tile. Writing them per kv head is an out-of-bounds
write that shows up two different ways depending on where it lands: measured on 0915 it
faulted outright at seqlen 256 (page not present) and silently corrupted at seqlen 1024,
giving dk/dv about -0.3 dB while dq stayed correct at 52.2. Per q head plus a host-side
reduction is correct at every seqlen tested from 128 to 8192.

WHY THERE IS A SEQUENCE-LENGTH FLOOR. Three launches, a zeroed fp32 dq_acc and a host-side
reduction are fixed costs, and the kernel wants one resident workgroup per CU across 320 KB
of LDS. Its own time barely moves with the work -- 0.88 / 0.91 / 1.00 / 1.34 ms from seqlen
1024 to 8192 -- so below roughly 2048 there is not enough work to amortise it. The table and
the reasoning are on _MIN_SEQLEN_ASM below.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

__all__ = ["asm_backward_eligible", "asm_dense_backward"]

_TRACE = os.environ.get("PRIMUS_TURBO_ASM_BWD_TRACE", "") not in ("", "0")
# A path writes the trace there instead of stdout. Under a training launcher stdout goes
# through capture layers that demonstrably swallow lines -- aiter's own load banner never
# appeared in any e2e log on 0914 -- and then "no line" and "never called" look identical.
_TRACE_FILE = os.environ.get("PRIMUS_TURBO_ASM_BWD_TRACE_FILE", "")
# Hard off switch, mirroring PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD. Without one, the only way to
# measure what this path is worth is to make aiter unimportable, which changes the forward
# at the same time and makes the comparison meaningless.
# DEFAULT OFF as of 0915. Opt in with PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1.
#
# The kernel is 1.74x the vendored fused backward at the operator level, measured n=5 with
# bit-identical SQNR, and that number is real. It is also the wrong number to ship on:
#
#   20 training steps, llama-3.1-8B b=4 s=8192, only this switch differs
#     ASM backward ON    1858 tps   mfu 34.49%
#     ASM backward OFF   1984 tps   mfu 36.83%     <- 6.78% FASTER, at 30x the noise floor
#
# The cause is memory, not the kernel. This backward needs an fp32 dq_acc that the fused one
# does not (that one has no atomics) plus dk/dv per q head rather than per kv head: 1.07 GB
# per layer per step, ~34 GB of allocator churn over 32 layers. Operator-level timing never
# sees it because the caching allocator reuses the same blocks every iteration.
#
# Caching the scratch removes the churn and is NOT a fix: it trades churn for peak, and this
# config has no peak headroom. Memory at step 3 went 380.06 GiB (87.98%) without the cache
# to 381.44 GiB (88.30%) with it, and the run took SIGBUS -- then leaked its KFD context and
# 411 GB of VRAM, leaving the card unable to create new contexts until a power cycle.
#
# So: correct, fast in isolation, and not shippable by default on this configuration. The
# measurement entry points stay (--impl asmbwd, the harness switches) so the operator-level
# result remains reproducible.
_ENABLED = os.environ.get("PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD", "") not in ("", "0")
_DISABLED = not _ENABLED

# The gate is on SEQUENCE LENGTH, not parallelism. Measured head-to-head against
# dense_fused_backward, both arms in one process (ratios are fused/asm, >1 means the ASM
# kernel wins):
#
#   seqlen  b*nhead_q=8   =32     =128
#     1024      0.660x   0.632x  0.537x     <-- loses at every parallelism, including
#     1536      0.869x                          the production value of 128
#     2048      1.748x   1.549x
#     4096      2.812x   2.110x
#     8192      3.954x            2.002x
#
# The ASM backward's own time is nearly constant -- 0.88 / 0.91 / 1.00 / 1.34 ms across a
# 16x range of work -- so it has a fixed floor of roughly 0.85 ms and wins once there is
# enough work to amortise it. dense_fused_backward at seqlen 1024 costs 0.5806 / 0.5893 /
# 0.5785 ms at b*nhead_q of 8 / 32 / 128, i.e. flat: at that size it is latency-bound and
# more parallelism buys nothing. That is why batch*nhead_q does not predict the crossover
# and seqlen does.
#
# An earlier version of this file gated on batch*nhead_q >= 32, extrapolated from a single
# losing measurement. It was wrong in both directions: it would have declined seqlen 8192 at
# b*nhead_q=8, which wins 3.95x, and admitted seqlen 1024 at b*nhead_q=128, which loses.
_MIN_SEQLEN_ASM = int(os.environ.get("PRIMUS_TURBO_ASM_BWD_MIN_SEQLEN", "2048"))

_SEEN: set = set()
_LAUNCH = None

# Scratch buffers, cached per (shape, device). Measured 0915: without this the op-level win
# of 1.74x on the backward turned into a 6.8% end-to-end LOSS (1858 vs 1984 tps over 20
# training steps). The kernel needs an fp32 dq_acc of batch*nhead_q*seqlen*head_dim -- 536 MB
# at the production shape -- plus dk/dv allocated per q head rather than per kv head, another
# 268 MB each. That is 1.07 GB per layer per step, about 34 GB of allocator churn across 32
# layers, and the op-level harness never sees it because the caching allocator reuses the
# same blocks across its iterations.
#
# Only SCRATCH is cached. dq and the reduced dk/dv are returned to autograd and must be
# fresh: handing back a reused buffer would let the next layer's backward overwrite a
# gradient that has not been accumulated yet.
_SCRATCH: dict = {}

# One HipModule for the process. asm_backward does `hip = hip or HipModule()`, and
# asm_dense_backward was not passing one -- so every backward built a fresh instance with an
# empty module cache and re-ran hipModuleLoad on all three .co files. 3 loads per layer per
# step, 96 per step across 32 layers, and nothing ever calls hipModuleUnload, so they
# accumulate for the life of the process.
#
# Found by review on 0915, after the scratch-reuse fix failed to recover the 6.78% e2e
# regression. Allocation churn was the hypothesis; this is a second, independent per-call
# cost that the hypothesis missed entirely, and the operator-level harness hides it the same
# way it hides the allocation -- 20 iterations of one call site reload the same three modules
# 20 times, which is nothing next to a 32-layer training step.
_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        m = _launcher()
        if m is None:
            return None
        _HIP = m.HipModule()
    return _HIP


def _scratch(key, shape, dtype, device, zero=False):
    buf = _SCRATCH.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    if zero:
        # The non-scratch path allocates dk/dv with torch.ZEROS, not empty. Whether the
        # kernel writes every element of a per-q-head dk/dv slice has not been established,
        # and a reused buffer carries the previous layer's gradient rather than the fresh
        # garbage a new allocation would -- which is worse, because it is plausible-looking.
        # Match the path being replaced until there is evidence full coverage holds.
        buf.zero_()
    return buf


def _say(msg: str, key: str) -> None:
    if not _TRACE or key in _SEEN:
        return
    _SEEN.add(key)
    line = f"[primus_turbo asm_bwd pid={os.getpid()}] {msg}"
    if _TRACE_FILE:
        try:
            with open(_TRACE_FILE, "a") as fh:
                fh.write(line + "\n")
            return
        except Exception:
            pass
    print(line, flush=True)


def _no(reason: str) -> bool:
    _say(f"declined: {reason}", reason)
    return False


def _launcher():
    """Tri-state, resolved once: None = not tried, False = unavailable, module = ready."""
    global _LAUNCH
    if _LAUNCH is None:
        try:
            from primus_turbo.pytorch.kernels.attention import _asm_bwd_kernargs as m

            missing = [n for n in m.CO.values() if not (m.ASM_DIR / n).is_file()]
            _LAUNCH = False if missing else m
            if missing:
                _say(f"prebuilt objects missing: {missing}", "missing")
        except Exception as exc:  # pragma: no cover - environment dependent
            _say(f"launcher unavailable: {exc}", "import")
            _LAUNCH = False
    return _LAUNCH or None


def asm_backward_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    dropout_p: float = 0.0,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
) -> bool:
    """Whether the prebuilt ASM backward can serve this call.

    Deliberately narrow: hd128 bf16 only, no sink, window, bias, alibi or dropout. Anything
    not validated on hardware is declined rather than guessed at, because a wrong kernarg on
    this path does not raise, it reads the wrong memory.
    """
    if _DISABLED:
        return _no("off by default; set PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1 to opt in "
                   "(6.78% e2e regression from its memory footprint -- see module docstring)")
    if not is_gfx1250():
        return _no("not gfx1250")
    if _launcher() is None:
        return _no("prebuilt ASM objects unavailable")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return _no("dtype not bf16")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return _no("not 4-D")
    # Non-causal uses the mask=0 object with the x grid NOT halved; validated on 0915 at
    # three shapes, dq 52.48-52.52 / dk 50.77-50.82 / dv 50.68-50.91 dB.
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return _no("head_dim != 128; the .co set covers nothing else on this arch")
    if q.shape[1] != k.shape[1]:
        return _no("seqlen_q != seqlen_k; bottom-right alignment is untested here")
    if sink is not None:
        return _no("sink unsupported")
    if window_size != (-1, -1):
        return _no("sliding window unsupported")
    if dropout_p != 0.0:
        return _no("dropout unsupported")
    if bias is not None or alibi_slopes is not None:
        return _no("bias/alibi unsupported")
    nhead_q, nhead_k = q.shape[2], k.shape[2]
    if nhead_q % nhead_k:
        return _no("nhead_q not a multiple of nhead_k")
    if q.shape[1] < _MIN_SEQLEN_ASM:
        return _no(f"seqlen < {_MIN_SEQLEN_ASM}; the fixed cost is not amortised below it")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return _no("q/k/v not contiguous; the byte strides assume it")
    # Says so when it FIRES, not only when it declines. Silence from a decline-only trace
    # cannot be told apart from "the gate was never reached", and under a training launcher
    # that is exactly the question being asked -- the campaign has already lost a day to
    # inferring that a path was live because nothing complained.
    _say(f"engaged: b={q.shape[0]} s={q.shape[1]} hq={nhead_q} hkv={nhead_k} d={q.shape[3]}",
         "engaged")
    return True


def asm_dense_backward(dout, q, k, v, out, lse, softmax_scale, causal=True):
    """Adapter. lse may be the plain [B, Hq, Sq] the ASM forward returns or turbo's packed
    [B, Hq, 2*Sq] scratch; the launcher unpacks the second form itself rather than carrying
    a second copy of that layout knowledge."""
    m = _launcher()
    if m is None:  # pragma: no cover - guarded by the gate
        raise RuntimeError("asm_dense_backward called without the prebuilt objects")
    b, s, hq, d = q.shape
    hk = k.shape[2]
    rep = hq // hk
    dev = q.device
    # dq_acc is pure scratch and always safe to reuse -- it is consumed by dq_convert and
    # never returned. dk/dv scratch is only safe when rep > 1, because the reduction below
    # then produces fresh output tensors; at rep == 1 there is no reduction and the buffers
    # would be handed straight to autograd, where the next layer's backward would overwrite
    # a gradient that has not been accumulated yet.
    scratch = {"dq_acc": _scratch(("dq_acc", b, hq, s, d, dev), (b, hq, s, d),
                                  torch.float32, dev)}
    if rep > 1:
        scratch["dk"] = _scratch(("dk", b, s, hq, d, dev), (b, s, hq, d), k.dtype, dev, zero=True)
        scratch["dv"] = _scratch(("dv", b, s, hq, d, dev), (b, s, hq, d), v.dtype, dev, zero=True)
    dq, dk, dv = m.asm_backward(
        q, k, v, out, dout.contiguous(), lse, softmax_scale, hip=_hip(), dkdv_heads="q",
        causal=causal, scratch=scratch,
    )
    if rep > 1:
        # fp32 accumulate: the partials are bf16 and summing four of them in bf16 costs
        # about 1.5 dB of SQNR for nothing.
        dk = dk.view(b, s, hk, rep, d).float().sum(3).to(k.dtype)
        dv = dv.view(b, s, hk, rep, d).float().sum(3).to(v.dtype)
    return dq, dk, dv
