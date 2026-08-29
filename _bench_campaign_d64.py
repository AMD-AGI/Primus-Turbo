#!/usr/bin/env python3
"""Score for the gpt-oss D=64 attention-backward campaign.

    d64 = geomean(TARGET base_ms/cur_ms) * geomean(min(GUARD base_ms/cur_ms, 1.0))

TARGET is gpt-oss D=64. GUARD is llama D=128, capped at 1.0, and it is there for one reason:
D=64 and D=128 share ONE file (`assert head_dim in (64, 128)`), and the D=128 path was just
taken to 1022 TF/s over forty rounds. A capped guard can only ever subtract, so it cannot be
farmed for score -- it only makes a D=128 regression worthless. Do not remove it, do not raise
the cap, do not swap the shapes: that is the scoring rubric and it is off limits.

BASE rows are the DEPLOYED tree at 701e9ae0 (the D=128 campaign's final squash), median of
three calibration runs, own process per cell, min-of-40 after 5 warms.

    bwd flops = 5*B*Hq*S^2*D
    d64_gptoss_b4 (4/64/8/8192/64) = 5.4976e12  =>  1200 TF/s = 4.5813 ms
    d64_b2        (2/64/8/8192/64) = 2.7488e12  =>  1200 TF/s = 2.2907 ms

Both SNR gates run every time, at D=64 AND at D=128, floors 40 dB on dq/dK/dV. There is no
determinism gate: a16 accumulates dQ with atomics by construction.

usage: _bench_campaign_d64.py            # print {"d64": ..., "ok": true, ...}
       _bench_campaign_d64.py --calibrate # re-measure BASE (3 runs/cell, prints medians)
"""
import json, math, os, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16
WARMS, REPS = 5, 40
SNR_FP32 = 40.0
SNR_DQ = 40.0
FLOOR = 0.05

#         tag              B  Hq  Hkv    S    D  window
TARGET = [
    ("d64_gptoss_b4", 4, 64, 8, 8192, 64, -1),
    ("d64_b2",        2, 64, 8, 8192, 64, -1),
]
GUARD = [
    ("l70b", 1, 64, 8, 8192, 128, -1),
    ("l8b",  4, 32, 8, 8192, 128, -1),
]

# Calibrated on 701e9ae0, three runs per cell, own process each:
#   d64_gptoss_b4  5.5707 / 5.5109 / 5.5745   spread 1.15%
#   d64_b2         2.8805 / 2.8557 / 2.8413   spread 1.38%
#   l70b           2.6047 / 2.6347 / 2.6418   spread 1.42%
#   l8b            5.4621 / 5.4907 / 5.4928   spread 0.56%
# TARGET takes the MEDIAN so an unchanged tree scores 1.0. GUARD takes the MAX: a capped guard
# can only ever subtract, so it must not spend score on noise.
BASE = {
    "d64_gptoss_b4_bwd": 5.5707,
    "d64_b2_bwd": 2.8557,
    "l70b_bwd": 2.6418,
    "l8b_bwd": 5.4928,
}


def _sbhd(B, S, H, D):
    return torch.randn(S, B, H, D, device=DEV, dtype=DT)


def _make(cell):
    tag, B, Hq, Hkv, S, D, window = cell
    ws = (window, 0) if window >= 0 else (-1, -1)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=ws, return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    return lambda: flash_attn_sbhd_flydsl_backward_impl(do, q, k, v, o, lse_h, causal=True, window_size=ws)


def _time(fn):
    for _ in range(WARMS):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def _snr(ref, got):
    err = ref.float() - got.float()
    return 10 * math.log10(float(ref.float().pow(2).sum()) / max(float(err.pow(2).sum()), 1e-30))


def _snr_gate(D):
    B, S, Hq, Hkv = 1, 1024, 32, 8
    torch.manual_seed(7)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
    dq, dk, dv = flash_attn_sbhd_flydsl_backward_impl(
        do, q, k, v, o, lse.view(B, S, Hq).permute(0, 2, 1), causal=True, window_size=(-1, -1))
    bh = lambda t: t.permute(1, 2, 0, 3).float().requires_grad_()
    qf, kf, vf = bh(q), bh(k), bh(v)
    g = Hq // Hkv
    s = (qf @ kf.repeat_interleave(g, 1).transpose(-1, -2)) * D ** -0.5
    m = torch.ones(S, S, device=DEV, dtype=torch.bool).tril()
    (s.masked_fill(~m, float("-inf")).softmax(-1) @ vf.repeat_interleave(g, 1)).backward(
        do.permute(1, 2, 0, 3).float())
    return [_snr(r.grad, x.permute(1, 2, 0, 3)) for r, x in zip((qf, kf, vf), (dq, dk, dv))]


def _one_cell(tag):
    bwd = _make(next(c for c in TARGET + GUARD if c[0] == tag))
    return {f"{tag}_bwd": round(_time(bwd), 4)}


def _spawn(args, tag=""):
    r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                       capture_output=True, text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise AssertionError(f"{tag}: no result\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    got = json.loads(line[-1])
    if not got.get("ok"):
        raise AssertionError(f"{tag}: failed\n{r.stdout[-2000:]}")
    return got


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    for a in sys.argv[1:]:
        if a.startswith("--cell="):
            res = _one_cell(a.split("=", 1)[1])
            res["ok"] = True
            return res
    if "--snr" in sys.argv:
        out = {}
        for D in (64, 128):
            sn = _snr_gate(D)
            out[f"snr_d{D}"] = [round(x, 1) for x in sn]
            if sn[0] < SNR_DQ:
                raise AssertionError(f"D{D}: dq SNR {sn[0]:.1f} below {SNR_DQ}")
            if min(sn[1:]) < SNR_FP32:
                raise AssertionError(f"D{D}: dk/dv SNR {sn[1:]} below {SNR_FP32}")
        out["ok"] = True
        return out
    if "--calibrate" in sys.argv:
        out = {}
        for c in TARGET + GUARD:
            runs = [_spawn([f"--cell={c[0]}"], c[0])[f"{c[0]}_bwd"] for _ in range(3)]
            out[f"{c[0]}_bwd"] = statistics.median(runs)
            out[f"{c[0]}_runs"] = runs
        out["ok"] = True
        return out

    got = {}
    for c in TARGET + GUARD:
        got.update(_spawn([f"--cell={c[0]}"], c[0]))
    got.update(_spawn(["--snr"], "snr"))

    tgt = [BASE[f"{c[0]}_bwd"] / max(got[f"{c[0]}_bwd"], FLOOR) for c in TARGET]
    grd = [min(BASE[f"{c[0]}_bwd"] / max(got[f"{c[0]}_bwd"], FLOOR), 1.0) for c in GUARD]
    got["d64"] = round(_geomean(tgt) * _geomean(grd), 5)
    got["target_legs"] = {c[0]: round(t, 4) for c, t in zip(TARGET, tgt)}
    got["guard_legs"] = {c[0]: round(g, 4) for c, g in zip(GUARD, grd)}
    got["tflops"] = {
        "d64_gptoss_b4": round(5.4976e12 / (got["d64_gptoss_b4_bwd"] * 1e-3) / 1e12, 1),
        "d64_b2": round(2.7488e12 / (got["d64_b2_bwd"] * 1e-3) / 1e12, 1),
    }
    got["ok"] = True
    return got


if __name__ == "__main__":
    print(json.dumps(main()))
