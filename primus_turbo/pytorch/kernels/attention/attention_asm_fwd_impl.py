###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""aiter's prebuilt gfx1250 ASM forward, as an optional accelerator for the Triton path.

This is a *substitution*, not a new backend: the dense Triton path keeps its dispatch, its
backward and its LSE contract, and this only swaps the forward kernel when every condition
below holds. Anything it declines falls through to ``triton_dense_forward`` unchanged, so a
wrong "no" costs performance and a wrong "yes" is the only thing that could cost correctness
-- which is why the gate is narrow by construction.

Measured on gfx1250 at b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal, median of 3 under an
exclusive GPU window, interleaved with the in-tree forward:

    forward       1.410 ms  vs  2.607 ms   (1.85x)
    fwd+bwd      11.147 ms  vs 12.406 ms   (1.11x)

and the pairing is bitwise deterministic over 200 reps on all four tensors. Output SQNR
against an fp32 reference is 53.62 dB where the in-tree forward gives 53.67 -- the same
band, not a numerics trade.

Why it pairs only with the fused backward
-----------------------------------------
The ASM kernel returns LSE as plain ``[B, Hq, Sq]`` fp32 in natural log.
``dense_fused_backward`` accepts exactly that. The in-tree two-kernel backward does NOT: it
consumes Primus-Turbo's packed ``[B, Hq, 2*Sq]`` lse/delta scratch, where LSE and delta
interleave every ``FIXED_BLOCK_M`` rows. So this gate REQUIRES that the fused backward will
also be taken; otherwise the backward would receive an LSE in a layout it cannot read, and
it would not fault -- it would return smoothly wrong gradients.

aiter is an optional dependency here
------------------------------------
On gfx1250 the AITER dense backend is deliberately gated OFF (see
``DenseAttnFwdAiterBackend.can_handle``) so the Triton backend is reachable at all, and the
package is not part of the gfx1250 image. The import is therefore attempted once, lazily,
and a failure permanently disables this path rather than raising -- a build without aiter
behaves exactly as it did before this file existed.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

__all__ = ["asm_forward_eligible", "asm_dense_forward"]

# Opt-in tracing. The gate is a chain of silent early returns, which is right for a hot path
# and useless when the question is "why did this not fire in training". One line per distinct
# reason, once each, behind an env var.
_TRACE = os.environ.get("PRIMUS_TURBO_ASM_FWD_TRACE", "") not in ("", "0")
# Hard off switch, read at import like the trace flags above. Without it there is no way to
# measure the Triton forward at all: the dispatcher takes the ASM forward whenever it is
# eligible, so --impl turbo, fused and asm all reach the same kernel and differ by 0.4% --
# the noise floor, not an A/B. Attributing anything to the ASM forward requires a run with
# this set. Off-only on purpose: eligibility is a capability question, and forcing "on" past
# a declined check would just move the failure later.
_DISABLED = os.environ.get("PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD", "") not in ("", "0")
# A path here writes the trace to that file instead of stdout. Under a training launcher the
# process's stdout goes through capture layers that demonstrably swallow lines -- aiter's own
# load banner never appeared in any e2e log -- and then "no line" cannot be told apart from
# "the gate was never called", which is the one thing the trace exists to answer.
_TRACE_FILE = os.environ.get("PRIMUS_TURBO_ASM_FWD_TRACE_FILE", "")
# Resolved at import: os.getpid() is a posix builtin dynamo cannot trace, and this
# module is reached from inside an autograd.Function that torch.compile traces.
_PID = os.getpid()
_SEEN: set = set()



def _say(msg: str, key: str) -> None:
    # Folded away at trace time; see the note above _TRACE.
    if torch.compiler.is_compiling():
        return
    if not _TRACE or key in _SEEN:
        return
    _SEEN.add(key)
    line = f"[asm_fwd] {msg} pid={_PID}"
    if _TRACE_FILE:
        try:
            with open(_TRACE_FILE, "a") as fh:
                fh.write(line + "\n")
            return
        except OSError:
            pass  # fall through to stdout rather than losing the line entirely
    print(line, flush=True)


def _no(reason: str) -> bool:
    _say(f"declined: {reason}", reason)
    return False

# Tri-state: None = not yet attempted, False = unavailable, callable = ready.
_ASM_FWD = None


def _asm_entry():
    """The aiter entry point, or None if aiter is not installed.

    Resolved once. Import errors are cached as "unavailable" rather than retried, because a
    missing optional dependency does not become present later in the process and retrying it
    on every forward would put an exception on the hot path.
    """
    global _ASM_FWD
    if _ASM_FWD is None:
        try:
            from aiter.ops.mha import fmha_fwd_with_sink_asm

            _ASM_FWD = fmha_fwd_with_sink_asm
        except Exception:  # noqa: BLE001 -- any import failure means "not available"
            _ASM_FWD = False
    return _ASM_FWD or None


def asm_forward_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
) -> bool:
    """Whether the prebuilt ASM forward can serve this call.

    Mirrors aiter's own ``can_impl_fmha_fwd_with_sink_asm`` and adds the two constraints
    this integration imposes: gfx1250 only (the .co files are built for it and nothing
    else), and no sliding window (only the dense causal and non-causal variants have been
    validated here).

    The caller must ALSO confirm ``fused_backward_eligible`` -- see the module docstring.
    """
    if _DISABLED:
        return _no("disabled by PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD")
    if not is_gfx1250():
        return _no("not gfx1250")
    if _asm_entry() is None:
        return _no("aiter unavailable")
    # bf16 only. The .co set covers fp16 nowhere on this arch, and a silent dtype promotion
    # would change the numerics the SQNR gate was calibrated against.
    if q.dtype is not torch.bfloat16 or k.dtype is not q.dtype or v.dtype is not q.dtype:
        return _no("dtype not bf16")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return _no("not 4-D")
    if not (q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1):
        return _no("last dim not contiguous")
    # No sink on this path: hdim 128 has no sink-carrying .co, and returning None for a
    # gradient the caller asked for is worse than being slower.
    if sink is not None:
        return _no("sink present")
    if dropout_p != 0.0 or bias is not None or alibi_slopes is not None:
        return _no("dropout/bias/alibi")
    if window_size != (-1, -1):
        return _no("sliding window")
    head_dim_qk = q.shape[-1]
    if head_dim_qk not in (64, 128) or v.shape[-1] != head_dim_qk:
        return _no("head_dim")
    hq, hkv = q.shape[2], k.shape[2]
    if hkv == 0 or hq % hkv != 0:
        return _no("hq % hkv")
    # Self-attention only: cross-attention shapes are not in the CSV rows these .co files
    # were built from, and an unmatched row returns without computing rather than raising.
    if q.shape[1] != k.shape[1] or k.shape[1] != v.shape[1]:
        return _no("not self-attention")
    # Trace the ACCEPT too, once. Absence of a "declined" line is not evidence the path was
    # taken: aiter's own load banner is the obvious signal and Primus exports
    # AITER_LOG_LEVEL=ERROR, which suppresses it -- so "no aiter lines in the log" was read
    # as "the ASM kernel never ran" when it meant nothing at all.
    _say(f"ACCEPTED: q={tuple(q.shape)} k={tuple(k.shape)} dtype={q.dtype}", "accepted")
    return True


def asm_dense_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the prebuilt ASM forward. Returns ``(out, lse)`` with lse ``[B, Hq, Sq]`` fp32.

    The LSE is in NATURAL log, confirmed empirically against ``torch.logsumexp`` of an fp32
    reference at 140.5 dB / 4.8e-6 max abs error -- not merely inferred from aiter's tests.
    """
    entry = _asm_entry()
    if entry is None:  # pragma: no cover -- callers gate on asm_forward_eligible first
        raise RuntimeError("aiter is not available; asm_dense_forward should not have been reached")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, lse = entry(q, k, v, softmax_scale, causal, True)
    return out, lse
