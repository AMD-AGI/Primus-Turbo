#!/usr/bin/env python3
# Campaign bench: a NON-DETERMINISTIC flash-attn backward, at both head dims.
#
# The deterministic kernel's dQ split-K leg was measured at 2.041 ms on llama2-70B b2, against
# the 2.034 ms that shape needs to reach the aiter build -- the gap and the price of bitwise
# determinism are the same number. aiter's hd128 winner pays it differently: 72 fp32 atomic
# adds straight into dQ, no partial workspace and no fold pass at all. This campaign builds
# that, keeps the accuracy gate, and drops only the bitwise-reproducibility gate.
#
# score  nd = geomean(TARGET base_ms/cur_ms) * geomean(min(GUARD base_ms/cur_ms, 1.0))
#        Four targets, two per head dim, because the fold is paid at BOTH -- D=64 carries it
#        too, and D=64 is the shape this kernel currently wins on, so it must not be traded
#        away for D=128. The GUARDs are capped at 1.0: they cannot earn, only take away.
#
# order  EACH CELL RUNS IN ITS OWN PROCESS: the allocator state a cell inherits moved a
#        reading by up to 70% when they shared one.
#
# gates  - dq/dk/dv >= 47 dB against a chunked fp32 reference at D=128 and D=64 (UNCHANGED).
#        - run-to-run agreement >= 60 dB between two calls on identical inputs. This replaces
#          the bitwise `torch.equal` gate: fp32 atomics reorder, so bits may differ, but a
#          race that corrupts rather than reorders will not clear 60 dB.
#
# --ref-aiter times aiter's backward on these same shapes, but NOT yet on the same terms: it
# goes through `out.backward()`, so it carries autograd dispatch, leaf .grad accumulation and
# a fresh gradient allocation per call, and it reads ~1.7x our raw-impl timing -- the opposite
# of what the published comparison says. Do not use it as a target until it calls aiter's raw
# backward op the way the flydsl arm calls ours.
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
SNR_RERUN = 60.0
WARMS, REPS = 5, 40

# (tag, B, Hq, Hkv, S, D, window)
TARGET = [
    ("d128_70b_b2", 2, 64, 8, 8192, 128, -1),
    ("d128_8b_b2", 2, 32, 8, 8192, 128, -1),
    ("d64_gptoss_b4", 4, 64, 8, 8192, 64, -1),
    ("d64_b2", 2, 64, 8, 8192, 64, -1),
]
GUARD = [
    ("d64_swa_b4", 4, 64, 8, 8192, 64, 128),
    ("d128_b1", 1, 64, 8, 8192, 128, -1),
]
# (4,64,8,8192,128) and (1,64,8,16384,128) were guards too and were dropped: on identical code
# they read either ~14 ms or ~28 ms, a clean factor of two, which is a plan switch and not
# noise -- their dQ partial workspace sits near the 16 GiB budget that decides between the
# pipelined and the fused-bandgroup path. A guard capped at 1.0 that halves at random can only
# ever spend score. Worth diagnosing later; not worth blocking the campaign on.
# d64_swa_b4 is sub-millisecond and reads bimodally; more spawns and a cooldown before it.
SPAWNS = {"d64_swa_b4": 5}
COOLDOWN = {"d64_swa_b4": 20}

# Filled by --baseline, measured through the harness path (bench.sh, which clears the flydsl
# cache) rather than by running this file directly -- a warm-cache base reads the two largest
# cells about 3% fast and turns that into a permanent tax on a capped guard.
# Re-measured on n02-29 after the campaign moved hosts, two whole-bench runs with the flydsl
# cache cleared before each the way the harness does it (the new host reads ~2% faster, and a
# base carried across machines hands every candidate that 2% for free). TARGET takes the mean of the two so an unchanged tree scores 1.0;
# GUARD takes the max, because a capped guard can only subtract and calibrating it to a fast
# read taxes every candidate. d64_swa_b4 is set above even its own max: it reads bimodally,
# either ~0.565 or ~0.626, and one slow read took 4.9% off a round whose real gain was 1.8%.
# Calibrated to the top of the slow mode it can no longer spend score, at the cost of only
# catching regressions past about 10% on that cell.
BASE = {
    "d128_70b_b2_bwd": 6.2706,
    "d128_8b_b2_bwd": 3.3791,
    "d64_gptoss_b4_bwd": 5.5565,
    "d64_b2_bwd": 2.8121,
    "d64_swa_b4_bwd": 0.6124,
    "d128_b1_bwd": 3.2330,
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


def _make_aiter(cell):
    """aiter's backward on the same shape, timed by the same clock. bshd in, so the tensors
    are built in aiter's own layout -- this is a reference point, not a layout comparison."""
    import aiter

    tag, B, Hq, Hkv, S, D, window = cell
    ws = (window, 0) if window >= 0 else (-1, -1)
    mk = lambda H: torch.randn(B, S, H, D, device=DEV, dtype=DT, requires_grad=True)
    q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
    # return_lse is asserted True inside aiter's autograd Function; it returns (out, lse) then.
    out = aiter.flash_attn_func(q, k, v, causal=True, window_size=ws, return_lse=True)
    out = out[0] if isinstance(out, tuple) else out
    go = torch.randn_like(out)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(go, retain_graph=True)

    return bwd


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


def _snr(ref, got):
    ref = ref.float()
    err = ref - got.float()
    return 10 * math.log10(float(ref.pow(2).sum()) / max(float(err.pow(2).sum()), 1e-30))


def _rerun_agrees(bwd):
    """Two calls on identical inputs must agree to 60 dB. Bits may differ -- fp32 atomics
    reorder -- but a race that corrupts rather than reorders will not clear this."""
    a, b = bwd(), bwd()
    return min(_snr(x, y) for x, y in zip(a[:3], b[:3]))


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


def _one_cell(tag, aiter_ref=False):
    cell = next(c for c in TARGET + GUARD if c[0] == tag)
    if aiter_ref:
        return {f"{tag}_bwd": round(_time(_make_aiter(cell)), 4)}
    bwd = _make(cell)
    agree = _rerun_agrees(bwd)
    if agree < SNR_RERUN:
        raise AssertionError(f"{tag}: two runs disagree at {agree:.1f} dB, below {SNR_RERUN}")
    return {f"{tag}_bwd": round(_time(bwd), 4), f"{tag}_agree": round(agree, 1)}


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
    aiter_ref = "--ref-aiter" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--cell="):
            res = _one_cell(a.split("=", 1)[1], aiter_ref)
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
        extra = ["--ref-aiter"] if aiter_ref else []
        tries = [_spawn([f"--cell={tag}"] + extra, tag) for _ in range(SPAWNS.get(tag, 1))]
        out[key] = min(t[key] for t in tries)
        # Backward FLOPs, flash-attn convention: 2.5x the forward, halved for causal.
        B_, Hq_, _, S_, D_ = cell[1], cell[2], cell[3], cell[4], cell[5]
        out[f"{tag}_tfs"] = round(5 * B_ * Hq_ * S_ * S_ * D_ / out[key] / 1e9, 1)
        if f"{tag}_agree" in tries[0]:
            out[f"{tag}_agree"] = min(t[f"{tag}_agree"] for t in tries)
        if not (baseline or aiter_ref):
            sp = BASE[key] / out[key]
            out[key + "_sp"] = round(sp, 4)
            (tgt if cell in TARGET else grd).append(sp if cell in TARGET else min(sp, 1.0))
    if aiter_ref:
        print("AITER = {" + ", ".join(f'"{k}": {v}' for k, v in out.items()) + "}")
        out["ok"] = True
        return out
    out.update({k: v for k, v in _spawn(["--snr"]).items() if k != "ok"})

    if baseline:
        print(
            "BASE = {"
            + ", ".join(f'"{k}": {v}' for k, v in out.items() if k.endswith("_bwd"))
            + "}"
        )
        out["ok"] = True
        return out
    geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 1.0
    out["target"] = round(geo(tgt), 5)
    out["guard"] = round(geo(grd), 5)
    out["nd"] = round(geo(tgt) * geo(grd), 5)
    out["ok"] = True
    return out


if __name__ == "__main__":
    try:
        res = main()
    except Exception:
        traceback.print_exc()
        res = {"ok": False, "nd": FLOOR}
    print(json.dumps(res))
