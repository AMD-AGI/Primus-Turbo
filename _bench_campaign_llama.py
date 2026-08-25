#!/usr/bin/env python3
# Campaign bench: the llama attention BACKWARD, where flydsl trails x-Attention 1.39-1.62x.
#
# The three llama rows of the 7/23 comparison. All are D=128, which is the structural
# difference from gpt-oss: at D=64 this kernel beats the same x-Attention build and at D=128
# it loses. Backward only -- the campaign's target file is the backward kernel, so a forward
# number in the score would be a third of the weight on something no edit can move.
#
# score  llama = geomean(TARGET base_ms/cur_ms) * geomean(min(GUARD base_ms/cur_ms, 1.0))
#        The GUARDs are capped at 1.0 -- they cannot earn, only take away. gpt-oss is in
#        there because it shares the kernel and is the shape that currently wins.
#
# order  EACH CELL RUNS IN ITS OWN PROCESS: the dQ partial workspace is gigabytes and the
#        allocator state a cell inherits moved a reading by up to 70% when they shared one.
#
# gates  every cell bitwise deterministic, and dq/dk/dv >= 47 dB against a chunked fp32
#        reference at D=128 and D=64.
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
SNR_FP32 = 47.0
WARMS, REPS = 5, 40

# (tag, B, Hq, Hkv, S, D, window)
TARGET = [
    ("l70b_b2", 2, 64, 8, 8192, 128, -1),
    ("l8b_b2", 2, 32, 8, 8192, 128, -1),
]
GUARD = [
    # gpt-oss: the shape this kernel currently wins on, and it shares every line of the body.
    ("gptoss_full", 4, 64, 8, 8192, 64, -1),
    ("gptoss_swa", 4, 64, 8, 8192, 64, 128),
    # llama off the two scored points -- another batch, another context length -- so a change
    # that only works at (B=2, S=8192) is seen for what it is.
    ("l70b_b1", 1, 64, 8, 8192, 128, -1),
    ("l70b_b4", 4, 64, 8, 8192, 128, -1),
    ("l70b_16k_b1", 1, 64, 8, 16384, 128, -1),
]
# gptoss_swa is 0.57 ms and reads bimodally -- 0.568 or 0.626, a 10% coin flip. The guess was
# that it inherits the thermal tail of the 5.7 ms gptoss_full ahead of it, so it gets more
# spawns and a cooldown first. Measured afterwards: it still flips, so that guess was wrong
# and the cause is still unknown. Both knobs are harmless and stay; what actually stops the
# flip from spending score is calibrating this guard's base to the slow mode (see BASE).
SPAWNS = {"gptoss_swa": 5}
COOLDOWN = {"gptoss_swa": 20}

# Measured 2026-08-21 on n06-33, on the round-2 tree, so 1.0 means "as good as round 2".
#
# Taken from three runs through the HARNESS path, not from running this file directly. That
# distinction is the whole point: the harness clears the flydsl cache before every bench, so
# each cell recompiles and re-autotunes, and it reads l70b_b4 and l70b_16k_b1 about 3% slower
# than a warm-cache run does. A base measured the warm way makes that 3% a permanent tax on a
# capped guard -- no optimisation can pay it off, and the harness reverts good work chasing it.
# The ruler has to be read the same way the campaign reads it.
#
# The two roles take different statistics, because their noise is not symmetric:
#   TARGET -- median, so an unchanged tree scores 1.0. A min here reads about 1% low on every
#             run, and with revert-patience 10 that is enough to make real sub-1% gains never
#             register as a new best and have the harness throw the round's work away.
#   GUARD  -- max, so run-to-run noise cannot spend score. A guard is capped at 1.0 and can
#             only ever subtract, so calibrating it to a fast read taxes every candidate.
#             The cost is that a guard now only catches a regression larger than the cell's
#             own spread; for gptoss_swa, which still flips 10% between 0.568 and 0.626
#             despite the cooldown above, that means it only catches regressions past 10%.
BASE = {
    "l70b_b2_bwd": 6.3424,
    "l8b_b2_bwd": 3.4345,
    "gptoss_full_bwd": 5.7544,
    "gptoss_swa_bwd": 0.6262,
    "l70b_b1_bwd": 3.2602,
    "l70b_b4_bwd": 14.2835,
    "l70b_16k_b1_bwd": 13.6705,
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
    if not _deterministic(bwd):
        raise AssertionError(f"{tag}: backward is not bitwise deterministic")
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
        for D in (128, 64):
            sn = _snr_gate(D)
            out[f"snr_d{D}"] = [round(x, 1) for x in sn]
            if min(sn) < SNR_FP32:
                raise AssertionError(f"D{D}: dq/dk/dv fp32 SNR {sn} below {SNR_FP32}")
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
