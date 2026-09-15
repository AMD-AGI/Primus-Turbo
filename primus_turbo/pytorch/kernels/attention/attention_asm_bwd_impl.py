"""Gate and adapter for aiter's prebuilt gfx1250 ASM backward.

Same shape as attention_asm_fwd_impl: a capability check that declines loudly under a trace
flag, and a thin adapter. Anything declined falls through to the path that was already
shipping, so this file cannot make a working configuration worse.

Three kernel launches per backward, not one:

    bwd_hd128_odo_bf16            delta = rowsum(dO * O)
    bwd_hd128_bf16_causal_br_a32_pssk   the body; 514 buffer_atomic_add_f32 into an fp32 dq_acc
    bwd_hd128_dq_convert_bf16     dq_acc fp32 -> dq bf16

WHY dk/dv ARE ALLOCATED PER Q HEAD. The body's grid is (kv_tiles, nhead_q, batch), so under
GQA `ratio` workgroups own the same dk/dv tile. Writing them per kv head is an out-of-bounds
write that shows up two different ways depending on where it lands: measured on 0915 it
faulted outright at seqlen 256 (page not present) and silently corrupted at seqlen 1024,
giving dk/dv about -0.3 dB while dq stayed correct at 52.2. Per q head plus a host-side
reduction is correct at every seqlen tested from 128 to 8192.

WHY THERE IS A PARALLELISM FLOOR. The reduction and the zeroed fp32 dq_acc are fixed costs,
and the kernel wants one resident workgroup per CU across 320 KB of LDS. Measured backward
ratios against the vendored fused backward at 1100 MHz:

    b=4 s=8192 hq=32   17.675 -> 10.098 ms   1.750x
    b=2 s=8192 hq=32    9.007 ->  6.141 ms   1.467x
    b=4 s=4096 hq=32    5.538 ->  4.405 ms   1.257x
    b=1 s=4096 hq=32    3.201 ->  2.497 ms   1.282x
    b=4 s=4096 hq=8     3.232 ->  2.518 ms   1.284x
    b=1 s=1024 hq=8     0.743 ->  2.218 ms   0.335x   <-- a 3x LOSS

Every winner has batch * nhead_q >= 32 and the only loser is at 8, which is the same
threshold _MIN_PARALLEL_WORK already applies to the fused backward. That is a coincidence
worth stating rather than relying on: the number here is measured independently.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

__all__ = ["asm_backward_eligible", "asm_dense_backward"]

_TRACE = os.environ.get("PRIMUS_TURBO_ASM_BWD_TRACE", "") not in ("", "0")
# Hard off switch, mirroring PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD. Without one, the only way to
# measure what this path is worth is to make aiter unimportable, which changes the forward
# at the same time and makes the comparison meaningless.
_DISABLED = os.environ.get("PRIMUS_TURBO_ATTN_DISABLE_ASM_BWD", "") not in ("", "0")

# batch * nhead_q below this loses to the vendored fused backward -- see the module docstring.
_MIN_PARALLEL_WORK_ASM = int(os.environ.get("PRIMUS_TURBO_ASM_BWD_MIN_WORK", "32"))

_SEEN: set = set()
_LAUNCH = None


def _say(msg: str, key: str) -> None:
    if _TRACE and key not in _SEEN:
        _SEEN.add(key)
        print(f"[primus_turbo asm_bwd] {msg}", flush=True)


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

    Deliberately narrow. Only the causal bottom-right hd128 bf16 variant has been validated
    on hardware; everything else is declined rather than guessed at, because a wrong kernarg
    here does not raise, it reads the wrong memory.
    """
    if _DISABLED:
        return _no("disabled by PRIMUS_TURBO_ATTN_DISABLE_ASM_BWD")
    if not is_gfx1250():
        return _no("not gfx1250")
    if _launcher() is None:
        return _no("prebuilt ASM objects unavailable")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return _no("dtype not bf16")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return _no("not 4-D")
    if not causal:
        # bwd_hd128_bf16_a32_pssk (the non-causal object) exists and is unvalidated here.
        return _no("only the causal variant is validated")
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
    if q.shape[0] * nhead_q < _MIN_PARALLEL_WORK_ASM:
        return _no(f"batch*nhead_q < {_MIN_PARALLEL_WORK_ASM}; measured a 3x loss there")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return _no("q/k/v not contiguous; the byte strides assume it")
    return True


def asm_dense_backward(dout, q, k, v, out, lse, softmax_scale, causal=True):
    """Adapter. lse may be the plain [B, Hq, Sq] the ASM forward returns or turbo's packed
    [B, Hq, 2*Sq] scratch; the launcher unpacks the second form itself rather than carrying
    a second copy of that layout knowledge."""
    m = _launcher()
    if m is None:  # pragma: no cover - guarded by the gate
        raise RuntimeError("asm_dense_backward called without the prebuilt objects")
    dq, dk, dv = m.asm_backward(
        q, k, v, out, dout.contiguous(), lse, softmax_scale, dkdv_heads="q"
    )
    rep = q.shape[2] // k.shape[2]
    if rep > 1:
        b, s, _, d = dk.shape
        hk = k.shape[2]
        # fp32 accumulate: the partials are bf16 and summing four of them in bf16 costs
        # about 1.5 dB of SQNR for nothing.
        dk = dk.view(b, s, hk, rep, d).float().sum(3).to(k.dtype)
        dv = dv.view(b, s, hk, rep, d).float().sum(3).to(v.dtype)
    return dq, dk, dv
