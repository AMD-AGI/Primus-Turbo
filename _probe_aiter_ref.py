#!/usr/bin/env python3
"""aiter's backward on OUR ruler, so the reference number is one we measured rather than one
we were told. Calls aiter.mha_bwd directly -- going through out.backward() adds autograd
dispatch and leaf .grad accumulation, which is what made an earlier attempt read 10.7 ms and
get dismissed as overhead.

Both arms: same shape, same clock, min of 40 in this process, forward run once to make o/lse.
aiter wants bshd and its own lse layout; flydsl wants sbhd. Each is given the layout it wants,
so this compares implementations, not transposes.

usage: _probe_aiter_ref.py <B> <Hq> <Hkv> <S> <D> [reps] [--det]
"""
import json
import sys

import torch

DEV, DT = "cuda", torch.bfloat16
WARMS = 5


def _time(fn, reps):
    for _ in range(WARMS):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def main():
    B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
    reps = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("-") else 40
    det = "--det" in sys.argv
    out = {"B": B, "Hq": Hq, "Hkv": Hkv, "S": S, "D": D, "deterministic": det}
    flops = 5 * B * Hq * S * S * D  # bwd = 2.5x fwd, halved for causal

    import aiter

    torch.manual_seed(11)
    q = torch.randn(B, S, Hq, D, device=DEV, dtype=DT)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=DT)
    v = torch.randn(B, S, Hkv, D, device=DEV, dtype=DT)
    do = torch.randn_like(q)
    from aiter.ops.mha import _flash_attn_backward

    o, lse = aiter.flash_attn_func(q, k, v, causal=True, return_lse=True)[:2]
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    # aiter's own autograd node calls this entry, in this order; mha_bwd's optional args are
    # not positional-safe through the op registration.
    fn = lambda: _flash_attn_backward(
        do, q, k, v, o, lse, dq, dk, dv, None, 0.0, D**-0.5, True, -1, -1,
        None, None, det, None, True, 0, sink_ptr=None,
    )
    ms = _time(fn, reps)
    out["aiter_ms"] = round(ms, 4)
    out["aiter_tfs"] = round(flops / ms / 1e9, 1)

    del q, k, v, do, o, lse, dq, dk, dv
    torch.cuda.empty_cache()

    sys.path.insert(0, "/workspace/code/tensorwise/Primus-Turbo")
    from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
        flash_attn_sbhd_flydsl_backward_impl as fbwd,
        flash_attn_sbhd_flydsl_forward_impl as ffwd,
    )

    torch.manual_seed(11)
    mk = lambda H: torch.randn(S, B, H, D, device=DEV, dtype=DT)
    q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
    do = torch.randn_like(q)
    o, lse = ffwd(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    fn = lambda: fbwd(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))
    ms = _time(fn, reps)
    out["flydsl_ms"] = round(ms, 4)
    out["flydsl_tfs"] = round(flops / ms / 1e9, 1)
    out["aiter_over_flydsl"] = round(out["flydsl_ms"] / out["aiter_ms"], 4)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
