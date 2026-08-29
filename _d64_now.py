#!/usr/bin/env python3
"""Where D=64 actually stands after the D=128 a16 campaign.

`_bench_campaign_a16.py` says in its own header: "LLAMA ONLY. gpt-oss and every D=64 shape are
out of the score and out of the guards." Forty rounds therefore neither rewarded nor protected
D=64, and it shares ONE file with D=128 (`assert head_dim in (64, 128)`, a single D64-specific
branch). So D=64 may have gained for free, or regressed silently. This measures it instead of
assuming, on the same ruler the campaign used: own process per cell, WARMS/REPS from the scorer,
min-of-N, plus the scorer's own SNR gate at D=64.

usage: _d64_now.py [--cell=TAG] [--snr]      (no args = spawn every cell + snr, print JSON)
"""
import json, math, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16
WARMS, REPS = 5, 40                      # identical to _bench_campaign_a16.py
SNR_FP32, SNR_DQ = 40.0, 40.0            # identical floors

# gpt-oss deployment shape is the first row; the rest bracket it so a win can be told apart
# from a batch/head artefact.        tag           B  Hq  Hkv    S    D  window
CELLS = [
    ("d64_gptoss_b4", 4, 64, 8, 8192, 64, -1),   # ★ the deployed one
    ("d64_b1",        1, 64, 8, 8192, 64, -1),
    ("d64_b2",        2, 64, 8, 8192, 64, -1),
]


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


def _tflops(cell, ms):
    _, B, Hq, _, S, D, _ = cell
    return 5 * B * Hq * S * S * D / (ms * 1e-3) / 1e12


def _spawn(args):
    r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                       capture_output=True, text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise AssertionError(f"no result\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(line[-1])


def main():
    for a in sys.argv[1:]:
        if a.startswith("--cell="):
            cell = next(c for c in CELLS if c[0] == a.split("=", 1)[1])
            ms = round(_time(_make(cell)), 4)
            return {"ms": ms, "tflops": round(_tflops(cell, ms), 1), "ok": True}
    if "--snr" in sys.argv:
        sn = _snr_gate(64)
        return {"snr_d64": [round(x, 2) for x in sn],
                "gate": "PASS" if (sn[0] >= SNR_DQ and min(sn[1:]) >= SNR_FP32) else "FAILED gate",
                "ok": True}
    out = {}
    for c in CELLS:
        out[c[0]] = _spawn([f"--cell={c[0]}"])
    out["snr"] = _spawn(["--snr"])
    out["ok"] = True
    return out


if __name__ == "__main__":
    print(json.dumps(main()))
