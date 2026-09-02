#!/usr/bin/env python3
"""Sweep the RMSNorm launcher configs for the four widths the proj unit runs.

Every arm is built (one untimed call) before any timing, then ranked by an interleaved
palindrome, so the JIT gap does not put the DVFS ramp inside the comparison
(pitfalls/02 §power-capped ramp). Isolated ruler: candidates found here are re-ranked
inside the proj unit before anything ships.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import primus_turbo.pytorch.kernels.normalization.rmsnorm_impl as RI  # noqa: E402

DEV, BF = "cuda", torch.bfloat16
S, B4, H = 8192, 4, 2880
NTOK = S * B4
HQ, HKV, HD = 64, 8, 64
QKV = HQ * HD + 2 * HKV * HD
AO = HQ * HD


def _time(fn, iters=10, reps=2):
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best


def race(name, arms, mk_fn):
    """arms = [(label, patch_fn)]; mk_fn() -> callable running one launch."""
    fns = []
    for label, patch in arms:
        patch()
        f = mk_fn()
        f()
        torch.cuda.synchronize()
        fns.append((label, f))
    ts = [float("inf")] * len(fns)
    order = list(range(len(fns)))
    for _ in range(20):
        fns[0][1]()
    torch.cuda.synchronize()
    for r in range(4):
        for i in order if r % 2 == 0 else order[::-1]:
            ts[i] = min(ts[i], _time(fns[i][1]))
    print(f"== {name}")
    for t, (label, _) in sorted(zip(ts, fns), key=lambda kv: kv[0]):
        print(f"   {t * 1000:9.2f} us   {label}")


def _pin_fwd(rows, warps, stages):
    def go():
        RI._pick_config = lambda Hh, Bb: (RI._next_pow2(Hh), rows, warps, stages)

    return go


def _pin_bwd(mode, gr, warps, stages):
    def go():
        RI._pick_bwd_config = lambda Hh, Bb: (mode, RI._next_pow2(Hh), gr, warps, stages)

    return go


def wide_fwd(residual):
    x = torch.randn(NTOK, H, device=DEV, dtype=BF)
    r = torch.randn(NTOK, H, device=DEV, dtype=BF)
    g = torch.randn(H, device=DEV, dtype=BF)
    if residual:
        return lambda: RI.rmsnorm_fwd_residual_impl(x, r, g, 1e-6, amax_out=True)
    return lambda: RI.rmsnorm_fwd_impl(x, g, 1e-6)


def head_fwd(nh):
    qkv = torch.randn(NTOK, QKV, device=DEV, dtype=BF)
    x = qkv[:, :AO].view(S, B4, HQ, HD) if nh == HQ else qkv[:, AO : AO + HKV * HD].view(S, B4, HKV, HD)
    g = torch.randn(HD, device=DEV, dtype=BF)
    return lambda: RI.rmsnorm_fwd_impl(x, g, 1e-6)


def head_bwd(nh):
    qkv = torch.randn(NTOK, QKV, device=DEV, dtype=BF)
    x = qkv[:, :AO].view(S, B4, HQ, HD) if nh == HQ else qkv[:, AO : AO + HKV * HD].view(S, B4, HKV, HD)
    g = torch.randn(HD, device=DEV, dtype=BF)
    y, x2, rstd, bh, rows, nw, ns = RI.rmsnorm_fwd_impl(x, g, 1e-6)[:7]
    dy = torch.randn_like(y)
    return lambda: RI.rmsnorm_bwd_impl(dy, x2, g, rstd, bh, rows, nw, ns)


def wide_bwd():
    x = torch.randn(NTOK, H, device=DEV, dtype=BF)
    g = torch.randn(H, device=DEV, dtype=BF)
    y, x2, rstd, bh, rows, nw, ns = RI.rmsnorm_fwd_impl(x, g, 1e-6)[:7]
    dy = torch.randn_like(y)
    return lambda: RI.rmsnorm_bwd_impl(dy, x2, g, rstd, bh, rows, nw, ns)


def main():
    what = sys.argv[1]
    if what == "wfwd":
        arms = [
            (f"rows{r} w{w} s{s}", _pin_fwd(r, w, s))
            for r in (1, 2, 4)
            for w in (4, 8, 16)
            for s in (1, 2)
        ]
        race("plain fwd H2880 B32768", arms, lambda: wide_fwd(False))
        race("residual fwd H2880 B32768", arms, lambda: wide_fwd(True))
    elif what == "hfwd":
        arms = [
            (f"rows{r} w{w} s{s}", _pin_fwd(r, w, s))
            for r in (4, 8, 16, 32, 64, 128)
            for w in (1, 2, 4, 8)
            for s in (1, 2)
        ]
        race("q fwd H64 B2097152", arms, lambda: head_fwd(HQ))
    elif what == "hbwd":
        arms = [
            (f"multi rows{r} w{w} s{s}", _pin_bwd("multi", r, w, s))
            for r in (16, 32, 64, 128, 256)
            for w in (2, 4, 8)
            for s in (1, 2, 3)
        ]
        race("q bwd H64 B2097152", arms, lambda: head_bwd(HQ))
    elif what == "wbwd":
        arms = [(f"grid g{g}", _pin_bwd("grid", g, 0, 0)) for g in (256, 512, 1024, 2048)]
        race("plain bwd H2880 B32768", arms, lambda: wide_bwd())
    return 0


if __name__ == "__main__":
    sys.exit(main())
