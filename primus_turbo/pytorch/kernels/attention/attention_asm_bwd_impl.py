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
# The kernel beats the vendored fused backward at the operator level and that is not in
# dispute. It is still the wrong number to ship on, but the REASON changed on 0915 evening
# once the two fixes below were finally verified on hardware. The old reason recorded here --
# "6.78% regression from its memory footprint" -- was WRONG on both halves. Corrected:
#
# WHAT THE REPLICATED MEASUREMENT SAYS (8-layer config, seed pinned, n=9 per arm).
# The earlier single 32-layer pair (ON 1858 / OFF 1984 tps) and the first 8-layer pair
# (ON +3.42%) are both unusable: no seed was set (torchtitan's set_determinism returns
# WITHOUT seeding at world_size==1 when debug.seed is None, so weight init was fresh-random
# per run), n=1 per arm, and the arms ran cold-then-hot in fixed order.
#
# With debug.seed pinned and nine runs per arm, both arms turn out to be TRIMODAL:
#
#     ON   5901 x3 | 6011 x3 | 6420 x3      mean 6110.8   between-run sd 3.88%
#     OFF  5822 x2 | 5936 x2 | 6327 x5      mean 6128.1   between-run sd 3.91%
#
# The variance is the SAME in both arms, so the multimodality belongs to e2e measurement on
# this box, not to this kernel. Junction temp is 44.0 C and sclk 1100 MHz on every run, so it
# is neither thermal nor clock; within a run throughput is flat (0.37%) and the scatter lives
# entirely between processes. Mechanism unknown -- address/bank layout per process is the
# leading guess and is NOT confirmed.
#
# Consequence for anyone measuring here: the run-to-run noise floor is ~3.9% and multimodal.
# A single A/B pair cannot resolve anything smaller, and every e2e conclusion in this campaign
# before 0915 evening was n=1. (An earlier draft of this comment claimed the ON arm was the
# unstable one, sd 4.53% against OFF's 0.06% -- that was n=3, and those three OFF runs simply
# happened to land in one mode. Same error, one level down.)
#
# Mean to mean the arms are indistinguishable: -0.28%, 0.15x the sem of the difference.
# Mode to mode, however, ON leads consistently -- 6420/6327 = +1.5%, 6011/5936 = +1.3%,
# 5901/5822 = +1.0% -- against an operator-level prediction of +1.36% (8 layers x 9.556 ms
# saved over a 5.6 s step). That agreement is suggestive, NOT established: it assumes the two
# arms' modes correspond to the same underlying system states, and the mode positions are not
# in fact equal between arms. The overall mean washes out because the modes are sampled at
# different rates (OFF hits its top mode 5/9, ON 3/9), a difference n=9 cannot call.
#
# So the default stays OFF because the end-to-end gain is UNMEASURABLE at n=9, while the path
# costs 1 GiB of resident scratch, an eligibility gate and three .co files to maintain -- not
# because it regresses. If the per-mode +1.2% can be nailed down (n=18 per arm, or by finding
# and controlling the physical cause of the modes), this decision is worth revisiting.
#
# THE TWO PER-CALL COSTS, both invisible to operator-level timing. Neither is the kernel.
#
#   1. Module reload. asm_dense_backward was not passing a HipModule, so every backward built
#      a fresh one with an empty cache and re-ran hipModuleLoad on all three .co files: 96
#      loads per training step across 32 layers, never unloaded. Fixed (_hip below).
#   2. Allocation. This backward needs an fp32 dq_acc the fused one does not -- that one has
#      no atomics -- plus dk/dv per q head rather than per kv head. Fixed (_SCRATCH below).
#
# CORRECTION to what (2) costs. The old note said 1.07 GB per layer per step, ~34 GB of churn
# over 32 layers, and that the scratch cache traded churn for peak. The per-LAYER part is
# wrong: the _SCRATCH keys are shape-only (no layer index), so every identically-shaped layer
# shares ONE buffer set -- dq_acc 0.500 GiB + dk/dv 0.250 GiB x2 = exactly 1.000 GiB
# process-wide, independent of layer count. The 32-layer logs show it: ab-on and ab-off both
# report 380.06 GiB to the byte (the ASM path cost zero extra reserved memory), and fix-on
# with the cache reports 381.44 GiB -- a delta of 1.38 GiB, not 34. Note also that the
# "memory:" figure in a step line is reserved_bytes.all.peak (allocator pool high-water,
# reset each log), not live bytes, which is why the 8-layer delta (15.65 GiB at 36% occupancy,
# where the allocator grows freely) is LARGER than the 32-layer one (1.38 GiB at 88%, where it
# reuses) -- impossible for anything that scales with layers.
#
# So the SIGBUS on the 32-layer config was 1.38 GiB landing on a run already at 88.30%, not a
# tens-of-GB footprint. Real, and a reason to keep headroom, but not the reason to default off.
#
# CAUTION FOR WHOEVER TOUCHES THE OPERATOR-LEVEL NUMBERS NEXT. All four measurement entry
# points -- tune_attention.py:529, t2_bringup.py:224/266, perfdq_diag.py:18 -- call
# asm_backward WITHOUT hip= and WITHOUT scratch=, i.e. they still measure the UNFIXED path;
# only the product call site below passes both. Two consequences, neither hardware-verified:
# the recorded 1.74x understates the fixed kernel (re-measured 8.13-8.68 ms against 17.686,
# ~2.18x), and the seqlen >= 2048 gate was derived from a ~0.85 ms fixed floor that partly
# CONSISTED of the per-call hipModuleLoad -- so the floor should now be lower and the
# threshold may be set too high. The nine measurements behind the gate need re-taking.
#
# So: correct, faster in isolation than recorded, and not shippable by default because it is
# not reproducible run to run. The measurement entry points stay (--impl asmbwd, the harness
# switches) so the operator-level result remains reproducible -- but see the caution above
# before comparing their output to anything the product path produces.
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
                   "(no measurable e2e gain at n=9: -0.28%, 0.15x sem -- see module docstring)")
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
