#!/usr/bin/env python3
"""aiter against flydsl through each one's PUBLIC autograd API, so both pay one autograd node
and the overhead cancels. Writes its result to a file -- piping a long run through tail
buffers the whole batch until it ends, which has cost two runs already.
usage: _probe_ab_autograd.py <B> <Hq> <Hkv> <S> <D> <reps> <out.json>"""
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


def bench(make_out, B, S, Hq, Hkv, D, sbhd, reps):
    shp = (lambda H: (S, B, H, D)) if sbhd else (lambda H: (B, S, H, D))
    torch.manual_seed(11)
    q, k, v = (torch.randn(*shp(H), device=DEV, dtype=DT, requires_grad=True) for H in (Hq, Hkv, Hkv))
    out = make_out(q, k, v)
    go = torch.randn_like(out)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(go, retain_graph=True)

    return _time(bwd, reps)


def main():
    B, Hq, Hkv, S, D, reps = (int(x) for x in sys.argv[1:7])
    dst = sys.argv[7]
    flops = 5 * B * Hq * S * S * D
    res = {"B": B, "Hq": Hq, "Hkv": Hkv, "S": S, "D": D, "reps": reps}

    import aiter

    ms = bench(lambda q, k, v: aiter.flash_attn_func(q, k, v, causal=True, return_lse=True)[0], B, S, Hq, Hkv, D, False, reps)
    res["aiter_ms"], res["aiter_tfs"] = round(ms, 4), round(flops / ms / 1e9, 1)
    torch.cuda.empty_cache()

    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func as pt_fa

    # Same bshd tensors aiter got: this compares implementations, not layouts.
    ms = bench(lambda q, k, v: pt_fa(q, k, v, causal=True), B, S, Hq, Hkv, D, False, reps)
    res["flydsl_ms"], res["flydsl_tfs"] = round(ms, 4), round(flops / ms / 1e9, 1)
    res["flydsl_over_aiter"] = round(res["flydsl_ms"] / res["aiter_ms"], 4)
    with open(dst, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
