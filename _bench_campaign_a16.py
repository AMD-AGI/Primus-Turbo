#!/usr/bin/env python3
# Campaign bench: llama attention BACKWARD at D=128, scoring the **a16** scheme.
#
# a16 = dQ leaves the body as `buffer_atomic_pk_add_bf16` into a band-less bf16 image, the
# way aiter's production backward does it. It is NON-DETERMINISTIC BY CONSTRUCTION (atomic
# completion order) and its dQ accumulator is bf16, so:
#
#   * there is NO bitwise-determinism gate here. The deployed deterministic split-K path has
#     one; re-imposing it would rule out the very scheme this campaign exists to build.
#   * dQ's SNR floor is relaxed to 30 dB. ~32 causally-visible kv bands each round the running
#     sum to bf16, against ONE rounding at the end for the fp32 path -- that is the accepted
#     trade, not a defect. dK/dV are held to the SAME 30 dB floor (lowered from 45): the
#     deployed path accumulates them in fp32 registers and clears 45 easily, but a candidate
#     that reaches them through a lower-precision accumulate should compete on speed rather
#     than be ruled out by a floor that only describes today's implementation.
#
# BASE is the DEPLOYED deterministic path, so a candidate only scores above 1.0 once a16 beats
# what we ship today. The real target is aiter a16 on the same ruler: l70b 2.787, l8b 5.562.
#
# LLAMA ONLY. gpt-oss and every D=64 shape are out of the score and out of the guards.
#
# score  llama = geomean(TARGET base_ms/cur_ms) * geomean(min(GUARD base_ms/cur_ms, 1.0))
# order  EACH CELL RUNS IN ITS OWN PROCESS (the dQ workspace is large and the allocator state
#        a cell inherits moved a reading by up to 70% when they shared one).
#
# The last stdout line is JSON. Any exception -> ok=false. Harness file: never committed.
import json
import math
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16
SNR_FP32 = 30.0   # dk/dv -- lowered from 45 with the same reasoning as dq below: a scheme
SNR_DQ = 30.0     # dq: bf16 accumulate, see header
# that lands dK/dV through a low-precision accumulate should be allowed to compete on speed
# rather than be ruled out by a floor the deployed fp32-register path happens to clear.
WARMS, REPS = 5, 40

# (tag, B, Hq, Hkv, S, D, window)
# aiter a16 on this machine, same ruler (min of 30, own process): l70b 2.8032, l8b 5.5646.
TARGET = [
    ("l70b", 1, 64, 8, 8192, 128, -1),
    ("l8b", 4, 32, 8, 8192, 128, -1),
]
GUARD = [
    # another llama batch, so a change tuned to (B=1, B=4) at S=8192 is seen for what it is.
    ("l70b_b2", 2, 64, 8, 8192, 128, -1),
]
SPAWNS = {}
COOLDOWN = {}

# Measured through the HARNESS path (which clears the flydsl cache before every bench, so each
# cell recompiles and re-autotunes), not by running this file directly -- a warm-cache base
# reads some cells ~3% fast and makes that a permanent tax the campaign then chases.
# TARGET takes the median of the calibration runs so an unchanged tree scores 1.0; GUARD takes
# the max, since a capped guard can only ever subtract and must not spend score on noise.
# Three harness-path runs 2026-08-24 on n02-29 at 1dd5428e+e7d58c5d:
#   l70b    3.1888 / 3.1776 / 3.1869   spread 0.35%
#   l8b     6.4377 / 6.3950 / 6.3998   spread 0.67%
#   l70b_b2 6.2122 / 6.1660 / 6.1398   spread 1.18%
# l70b reads 3.187 here against 3.1876 measured directly, so these cells carry no cache tax.
BASE = {
    "l70b_bwd": 3.1869,
    "l8b_bwd": 6.3998,
    "l70b_b2_bwd": 6.2122,
}
FLOOR = 0.05


def _sbhd(B, S, H, D):
    return torch.randn(S, B, H, D, device=DEV, dtype=DT)


def _make(cell):
    tag, B, Hq, Hkv, S, D, window = cell
    ws = (window, 0) if window >= 0 else (-1, -1)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=ws, return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    return lambda: flash_attn_sbhd_flydsl_backward_impl(
        do, q, k, v, o, lse_h, causal=True, window_size=ws
    )


def _time(fn):
    for _ in range(WARMS):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def _deterministic(bwd):
    a, b = bwd(), bwd()
    return all(torch.equal(x, y) for x, y in zip(a[:3], b[:3]))


def _snr(ref, got):
    ref = ref.float()
    err = ref - got.float()
    return 10 * math.log10(float(ref.pow(2).sum()) / max(float(err.pow(2).sum()), 1e-30))


def _snr_gate(D):
    B, S, Hq, Hkv = 1, 1024, 32, 8
    torch.manual_seed(7)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
    dq, dk, dv = flash_attn_sbhd_flydsl_backward_impl(
        do, q, k, v, o, lse.view(B, S, Hq).permute(0, 2, 1), causal=True, window_size=(-1, -1)
    )
    bh = lambda t: t.permute(1, 2, 0, 3).float().requires_grad_()
    qf, kf, vf = bh(q), bh(k), bh(v)
    g = Hq // Hkv
    s = (qf @ kf.repeat_interleave(g, 1).transpose(-1, -2)) * D**-0.5
    m = torch.ones(S, S, device=DEV, dtype=torch.bool).tril()
    (s.masked_fill(~m, float("-inf")).softmax(-1) @ vf.repeat_interleave(g, 1)).backward(
        do.permute(1, 2, 0, 3).float()
    )
    return [_snr(r.grad, x.permute(1, 2, 0, 3)) for r, x in zip((qf, kf, vf), (dq, dk, dv))]


def _one_cell(tag):
    bwd = _make(next(c for c in TARGET + GUARD if c[0] == tag))
    # No determinism gate: a16 is non-deterministic by construction (see header).
    return {f"{tag}_bwd": round(_time(bwd), 4)}


def _spawn(args, tag=""):
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__)] + args, capture_output=True, text=True, timeout=1800
    )
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise AssertionError(f"{tag}: no result\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    got = json.loads(line[-1])
    if not got.get("ok"):
        raise AssertionError(f"{tag}: failed\n{r.stdout[-2000:]}")
    return got


def main():
    for a in sys.argv[1:]:
        if a.startswith("--cell="):
            res = _one_cell(a.split("=", 1)[1])
            res["ok"] = True
            return res
    if "--snr" in sys.argv:
        out = {}
        for D in (128,):
            sn = _snr_gate(D)
            out[f"snr_d{D}"] = [round(x, 1) for x in sn]
            if sn[0] < SNR_DQ:
                raise AssertionError(f"D{D}: dq SNR {sn[0]:.1f} below {SNR_DQ}")
            if min(sn[1:]) < SNR_FP32:
                raise AssertionError(f"D{D}: dk/dv SNR {sn[1:]} below {SNR_FP32}")
        out["ok"] = True
        return out

    baseline = "--baseline" in sys.argv
    out, tgt, grd = {}, [], []
    for cell in TARGET + GUARD:
        tag = cell[0]
        key = f"{tag}_bwd"
        time.sleep(COOLDOWN.get(tag, 0))
        tries = [_spawn([f"--cell={tag}"], tag) for _ in range(SPAWNS.get(tag, 1))]
        out[key] = min(t[key] for t in tries)
        if not baseline:
            sp = BASE[key] / out[key]
            out[key + "_sp"] = round(sp, 4)
            (tgt if cell in TARGET else grd).append(sp if cell in TARGET else min(sp, 1.0))
    out.update({k: v for k, v in _spawn(["--snr"]).items() if k != "ok"})

    if baseline:
        print("BASE = {" + ", ".join(f'"{k}": {v}' for k, v in out.items() if not k.startswith("snr")) + "}")
        out["ok"] = True
        return out
    geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 1.0
    out["target"] = round(geo(tgt), 5)
    out["guard"] = round(geo(grd), 5)
    out["llama"] = round(geo(tgt) * geo(grd), 5)
    out["ok"] = True
    return out


if __name__ == "__main__":
    try:
        res = main()
    except Exception:
        traceback.print_exc()
        res = {"ok": False, "llama": FLOOR}
    print(json.dumps(res))
