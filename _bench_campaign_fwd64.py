#!/usr/bin/env python3
"""Score for the gpt-oss D=64 attention-FORWARD campaign.

    fwd64 = geomean(TARGET base_ms/cur_ms) * geomean(guard_factor)

TARGET is gpt-oss D=64 forward. GUARD is llama D=128 forward, free until 2% under base
(GUARD_TOL) and penalised proportionally past that -- the two head dims share
`flash_attn_fwd.py`, and a hard 1.0 cap was measured on the sister bwd campaign to be a
one-sided noise tax rather than a regression detector.

    fwd causal flops = 2 * B*Hq*S^2*D   (two GEMMs, 2 flop/MAC, halved for causal)
    cross-check: bwd uses 5*B*Hq*S^2*D (five GEMMs) => bwd/fwd = 2.5, as expected.
    d64_gptoss_b4 (4/64/8/8192/64) = 2.1990e12 => 1300 TF/s = 1.6916 ms, 1400 = 1.5707 ms
    d64_b2        (2/64/8/8192/64) = 1.0995e12 => 1300 TF/s = 0.8458 ms, 1400 = 0.7854 ms

★★ TWO CORRECTNESS GATES, both mandatory every round:

  1. `snr_fwd`  -- O against an fp32 reference. Floor 45 dB (deployed reads 51.7).
  2. `snr_e2e`  -- ★ the forward's own O and LSE fed into the BACKWARD, checking dq/dK/dV.
     Floor 45 dB. This exists because fwd and bwd SHARE THE LSE: an edit that changes the LSE
     layout or semantics to speed the forward up leaves `snr_fwd` perfect and silently destroys
     the backward. Gate 1 cannot see that; gate 2 can. Do not remove it.

There is NO determinism gate -- the user has explicitly authorised a non-deterministic forward
so long as the SNR gates pass. That is NOT licence to reduce precision: bf16 in, fp32 accumulate,
always.

usage: _bench_campaign_fwd64.py             # {"fwd64": ..., "ok": true, ...}
       _bench_campaign_fwd64.py --calibrate # re-measure BASE (5 runs/cell, first discarded)
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
SNR_FWD = 45.0
SNR_E2E = 45.0
FLOOR = 0.05
GUARD_TOL = 0.98

#         tag              B  Hq  Hkv    S    D
TARGET = [
    ("d64_gptoss_b4", 4, 64, 8, 8192, 64),
    ("d64_b2",        2, 64, 8, 8192, 64),
]
GUARD = [
    ("l70b_fwd", 1, 64, 8, 8192, 128),
    ("l8b_fwd",  4, 32, 8, 8192, 128),
]

# Calibrated on 1f71799e, five runs per cell, own process each, FIRST DISCARDED.
#   d64_gptoss_b4  1.8903 1.8799 1.8743 1.8889 1.8825   spread 0.78%   = 1168.9 TF/s
#   d64_b2         0.9522 0.9490 0.9507 0.9540 0.9527   spread 0.53%   = 1155.3 TF/s
#   l70b_fwd       0.9511 0.9519 0.9515 0.9471 0.9513   spread 0.51%
#   l8b_fwd        1.8895 1.8966 1.8992 1.8932 1.8948   spread 0.32%
# ★ This ruler is 2-4x tighter than the bwd campaign's (~1%), so a 0.3% gain is measurable here
#   where it was invisible there. Do not import the bwd campaign's "bundle to >=1.5%" reflex.
BASE = {
    "d64_gptoss_b4_fwd": 1.8812,
    "d64_b2_fwd": 0.9517,
    "l70b_fwd_fwd": 0.9514,
    "l8b_fwd_fwd": 1.8957,
}


def _sbhd(B, S, H, D):
    return torch.randn(S, B, H, D, device=DEV, dtype=DT)


def _flops(cell):
    _, B, Hq, _, S, D = cell
    return 2 * B * Hq * S * S * D


def _make(cell):
    tag, B, Hq, Hkv, S, D = cell
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    return lambda: flash_attn_sbhd_flydsl_forward_impl(
        q, k, v, causal=True, window_size=(-1, -1), return_lse=True)


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


def _ref_fwd(q, k, v, D, Hq, Hkv, S):
    """fp32 reference forward, and the tensors the backward reference needs."""
    bh = lambda t: t.permute(1, 2, 0, 3).float().requires_grad_()
    qf, kf, vf = bh(q), bh(k), bh(v)
    g = Hq // Hkv
    s = (qf @ kf.repeat_interleave(g, 1).transpose(-1, -2)) * D ** -0.5
    m = torch.ones(S, S, device=DEV, dtype=torch.bool).tril()
    p = s.masked_fill(~m, float("-inf")).softmax(-1)
    return p @ vf.repeat_interleave(g, 1), (qf, kf, vf)


def _gates(D):
    """Gate 1: O vs fp32.  Gate 2: this forward's O+LSE driven through the BACKWARD."""
    B, S, Hq, Hkv = 1, 1024, 32, 8
    torch.manual_seed(7)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(
        q, k, v, causal=True, window_size=(-1, -1), return_lse=True)

    ref_o, (qf, kf, vf) = _ref_fwd(q, k, v, D, Hq, Hkv, S)
    snr_fwd = _snr(ref_o, o.permute(1, 2, 0, 3))

    # ★ the forward's OWN o/lse go into the backward -- a broken LSE shows up here and nowhere else
    dq, dk, dv = flash_attn_sbhd_flydsl_backward_impl(
        do, q, k, v, o, lse.view(B, S, Hq).permute(0, 2, 1), causal=True, window_size=(-1, -1))
    ref_o.backward(do.permute(1, 2, 0, 3).float())
    snr_e2e = [_snr(r.grad, x.permute(1, 2, 0, 3)) for r, x in zip((qf, kf, vf), (dq, dk, dv))]
    return snr_fwd, snr_e2e


def _one_cell(tag):
    cell = next(c for c in TARGET + GUARD if c[0] == tag)
    return {f"{tag}_fwd": round(_time(_make(cell)), 4)}


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
            res = _one_cell(a.split("=", 1)[1]); res["ok"] = True
            return res
    if "--snr" in sys.argv:
        out = {}
        for D in (64, 128):
            sf, se = _gates(D)
            out[f"snr_fwd_d{D}"] = round(sf, 1)
            out[f"snr_e2e_d{D}"] = [round(x, 1) for x in se]
            if sf < SNR_FWD:
                raise AssertionError(f"D{D}: forward O SNR {sf:.1f} below {SNR_FWD}")
            if min(se) < SNR_E2E:
                raise AssertionError(
                    f"D{D}: end-to-end dq/dK/dV SNR {[round(x,1) for x in se]} below {SNR_E2E} "
                    f"-- the forward's LSE or O is no longer usable by the backward")
        out["ok"] = True
        return out
    if "--calibrate" in sys.argv:
        out = {}
        for c in TARGET + GUARD:
            runs = [_spawn([f"--cell={c[0]}"], c[0])[f"{c[0]}_fwd"] for _ in range(5)]
            warm = runs[1:]
            out[f"{c[0]}_fwd"] = round(statistics.median(warm), 4)
            out[f"{c[0]}_runs"] = runs
            out[f"{c[0]}_spread_pct"] = round(
                (max(warm) - min(warm)) / statistics.median(warm) * 100, 3)
            out[f"{c[0]}_tflops"] = round(_flops(c) / (statistics.median(warm) * 1e-3) / 1e12, 1)
        out["ok"] = True
        return out

    got = {}
    for c in TARGET + GUARD:
        got.update(_spawn([f"--cell={c[0]}"], c[0]))
    got.update(_spawn(["--snr"], "snr"))

    tgt = [BASE[f"{c[0]}_fwd"] / max(got[f"{c[0]}_fwd"], FLOOR) for c in TARGET]
    grd = []
    for c in GUARD:
        r = BASE[f"{c[0]}_fwd"] / max(got[f"{c[0]}_fwd"], FLOOR)
        grd.append(1.0 if r >= GUARD_TOL else r / GUARD_TOL)
    got["fwd64"] = round(_geomean(tgt) * _geomean(grd), 5)
    got["target_legs"] = {c[0]: round(t, 4) for c, t in zip(TARGET, tgt)}
    got["guard_legs"] = {c[0]: round(g, 4) for c, g in zip(GUARD, grd)}
    got["tflops"] = {c[0]: round(_flops(c) / (got[f"{c[0]}_fwd"] * 1e-3) / 1e12, 1) for c in TARGET}
    got["ok"] = True
    return got


if __name__ == "__main__":
    print(json.dumps(main()))
