"""Quick correctness + benchmark for the FlyDSL 192/128 backward (DSV3 MLA), sbhd causal bf16.

Correctness: dQ/dK/dV SNR against a chunked fp32 reference on sampled (batch, head) slices,
gated at the Primus-Turbo attention test bar (> 40 dB), plus finite on the full tensors and
dK/dV bitwise-repeatable. Timing: FlyDSL forward (mla_fwd_sbhd) and backward (mla_bwd_sbhd)
and Turbo AITER forward / autograd backward, rotated in one session on the same GPU.

Primary metric: Step TFLOPS = (fwd FLOPs + bwd FLOPs) / (fwd ms + bwd ms), geomean over PASS
shapes, with fwd = 2*(dqk+dv) and bwd = 2*(3*dqk+2*dv) per causal (q, k) pair.

Usage:
    HIP_VISIBLE_DEVICES=0 python quick_test_bench.py --set quick|full
        [--repeats N] [--budget-s S] [--summary-csv PATH] [--no-aiter]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.environ.get("MLA_PKG_ROOT", "/perf_apps/xiaoming/megaattention_overlap_bench"))
# The aiter reference is the a16 dQ kernel, as Primus training runs it (runner/helpers/envs/base_env.sh
# defaults this to 0; Turbo's own default of 1 picks the a32 kernel). Read once, at the first forward.
os.environ.setdefault("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "0")

import torch  # noqa: E402
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402

import primus_turbo.pytorch as pt  # noqa: E402
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager  # noqa: E402

HQ, HKV, DQK, DV = 128, 128, 192, 128
SCALE = DQK**-0.5
TARGET_SHAPES: list[dict[str, Any]] = [
    {"label": "b1_s4k", "B": 1, "S": 4096},
    {"label": "b1_s8k", "B": 1, "S": 8192},
    {"label": "b1_s16k", "B": 1, "S": 16384},
    {"label": "b1_s32k", "B": 1, "S": 32768},
    {"label": "b1_s64k", "B": 1, "S": 65536},
    {"label": "b2_s4k", "B": 2, "S": 4096},
    {"label": "b2_s16k", "B": 2, "S": 16384},
]
REPRESENTATIVE_SHAPES = [s for s in TARGET_SHAPES if s["label"] in ("b1_s4k", "b1_s16k", "b1_s64k", "b2_s4k")]

SNR_MIN_DB = 40.0
CHECK_HEADS = (0, 77, HQ - 1)
REF_CHUNK = 2048

SUMMARY_CSV_HEADER: list[str] = [
    "label",
    "B",
    "S",
    "HQ",
    "HKV",
    "DQK",
    "DV",
    "Check",
    "Forward TFLOPS",
    "Backward TFLOPS",
    "Backward TFLOPS_stddev",
    "Backward Time (ms)",
    "Step TFLOPS",
    "aiter Forward TFLOPS",
    "aiter Backward TFLOPS",
    "aiter Backward TFLOPS_stddev",
    "aiter Step TFLOPS",
    "bwd_ratio_vs_aiter",
    "step_ratio_vs_aiter",
    "dq_snr",
    "dk_snr",
    "dv_snr",
    "dkdv_bitwise_repeat",
    "aiter_dq_snr",
    "aiter_dk_snr",
    "aiter_dv_snr",
]


def pairs(b, s):
    return b * s * (s + 1) // 2 * HQ


def fwd_flops(b, s):
    return pairs(b, s) * 2 * (DQK + DV)


def bwd_flops(b, s):
    return pairs(b, s) * 2 * (3 * DQK + 2 * DV)


def ref_head(q, k, v, do):
    """Exact fp32 causal attention gradients for one head; q/k [S, DQK], v/do [S, DV]."""
    q, k, v, do = (t.float() for t in (q, k, v, do))
    s_len = q.shape[0]
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for r0 in range(0, s_len, REF_CHUNK):
        r1 = min(s_len, r0 + REF_CHUNK)
        qc, doc = q[r0:r1], do[r0:r1]
        s = qc @ k[:r1].T * SCALE
        rows = torch.arange(r0, r1, device=q.device)[:, None]
        s.masked_fill_(torch.arange(r1, device=q.device)[None, :] > rows, float("-inf"))
        p = torch.softmax(s, -1)
        del s
        delta = (doc * (p @ v[:r1])).sum(-1, keepdim=True)
        ds = p * (doc @ v[:r1].T - delta)
        dv[:r1] += p.T @ doc
        del p
        dq[r0:r1] = ds @ k[:r1] * SCALE
        dk[:r1] += ds.T @ qc * SCALE
        del ds
    return dq, dk, dv


def snr_db(x, ref):
    x, ref = x.double(), ref.double()
    return 10 * math.log10(ref.pow(2).sum().item() / (x - ref).pow(2).sum().clamp_min(1e-300).item())


def check_slices(b):
    return [(bi, h) for bi in sorted({0, b - 1}) for h in CHECK_HEADS]


def grads_snr(grads, refs, slices):
    out = []
    for gi in range(3):
        x = torch.cat([grads[gi][:, bi, h].reshape(-1) for bi, h in slices])
        r = torch.cat([refs[(bi, h)][gi].reshape(-1) for bi, h in slices])
        out.append(snr_db(x, r))
    return out


def timed(fn, repeats, budget_s):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    est = time.perf_counter() - t0
    iters = max(3, min(200, round(budget_s / repeats / max(est, 1e-6))))

    def sample():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    return sample


def run_one(shape, args):
    b, s = shape["B"], shape["S"]
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn((s, b, h, d), device="cuda", dtype=torch.bfloat16, generator=g)
        for h, d in ((HQ, DQK), (HKV, DQK), (HKV, DV))
    )
    dout = torch.randn((s, b, HQ, DV), device="cuda", dtype=torch.bfloat16, generator=g)

    out, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    grads = mla_bwd_sbhd(dout, q, k, v, out, lse)
    again = mla_bwd_sbhd(dout, q, k, v, out, lse)
    torch.cuda.synchronize()
    finite = all(bool(torch.isfinite(t).all().item()) for t in grads)
    rep = torch.equal(grads[1], again[1]) and torch.equal(grads[2], again[2])
    del again

    slices = check_slices(b)
    refs = {(bi, h): ref_head(q[:, bi, h], k[:, bi, h], v[:, bi, h], dout[:, bi, h]) for bi, h in slices}
    snrs = grads_snr(grads, refs, slices)
    ok = finite and rep and min(snrs) > SNR_MIN_DB
    del grads
    torch.cuda.empty_cache()

    fns = {
        "fwd": lambda: mla_fwd_sbhd(q, k, v, causal=True, return_lse=True),
        "bwd": lambda: mla_bwd_sbhd(dout, q, k, v, out, lse),
    }
    row = {
        "label": shape["label"],
        "B": b,
        "S": s,
        "HQ": HQ,
        "HKV": HKV,
        "DQK": DQK,
        "DV": DV,
        "Check": "PASS" if ok else "FAIL",
        "dq_snr": snrs[0],
        "dk_snr": snrs[1],
        "dv_snr": snrs[2],
        "dkdv_bitwise_repeat": rep,
    }
    if not args.no_aiter:
        GlobalBackendManager.set_attn_backend(BackendType.AITER)
        leaves = [t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v)]
        dout_b = dout.permute(1, 0, 2, 3)
        out_a = pt.ops.flash_attn_func(*leaves, softmax_scale=SCALE, causal=True)
        ga = torch.autograd.grad(out_a, leaves, dout_b, retain_graph=True)
        asnr = grads_snr([t.permute(1, 0, 2, 3) for t in ga], refs, slices)
        row.update({"aiter_dq_snr": asnr[0], "aiter_dk_snr": asnr[1], "aiter_dv_snr": asnr[2]})
        del ga

        def aiter_fwd():
            GlobalBackendManager.set_attn_backend(BackendType.AITER)
            pt.ops.flash_attn_func(*leaves, softmax_scale=SCALE, causal=True)

        def aiter_bwd():
            GlobalBackendManager.set_attn_backend(BackendType.AITER)
            torch.autograd.grad(out_a, leaves, dout_b, retain_graph=True)

        fns.update({"aiter_fwd": aiter_fwd, "aiter_bwd": aiter_bwd})
    del refs

    samplers = {name: timed(fn, args.repeats, args.budget_s) for name, fn in fns.items()}
    samples = {name: [] for name in fns}
    order = list(fns)
    for r in range(args.repeats):
        for name in order[r % len(order) :] + order[: r % len(order)]:
            samples[name].append(samplers[name]())

    ff, fb = fwd_flops(b, s), bwd_flops(b, s)

    def tf(name, flops):
        xs = [flops / (t * 1e-3) / 1e12 for t in samples[name]]
        return statistics.fmean(xs), statistics.pstdev(xs)

    def step(f, bw):
        return (ff + fb) / ((statistics.fmean(samples[f]) + statistics.fmean(samples[bw])) * 1e-3) / 1e12

    row["Forward TFLOPS"] = tf("fwd", ff)[0]
    row["Backward TFLOPS"], row["Backward TFLOPS_stddev"] = tf("bwd", fb)
    row["Backward Time (ms)"] = statistics.fmean(samples["bwd"])
    row["Step TFLOPS"] = step("fwd", "bwd")
    if not args.no_aiter:
        row["aiter Forward TFLOPS"] = tf("aiter_fwd", ff)[0]
        row["aiter Backward TFLOPS"], row["aiter Backward TFLOPS_stddev"] = tf("aiter_bwd", fb)
        row["aiter Step TFLOPS"] = step("aiter_fwd", "aiter_bwd")
        row["bwd_ratio_vs_aiter"] = row["Backward TFLOPS"] / row["aiter Backward TFLOPS"]
        row["step_ratio_vs_aiter"] = row["Step TFLOPS"] / row["aiter Step TFLOPS"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["quick", "full"], default="quick")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--budget-s", type=float, default=3.0)
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--no-aiter", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    shapes = TARGET_SHAPES if args.set == "full" else REPRESENTATIVE_SHAPES
    rows = []
    for shape in shapes:
        try:
            rows.append(run_one(shape, args))
        except Exception as e:  # noqa: BLE001
            print(f"{shape['label']}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            rows.append({"label": shape["label"], "B": shape["B"], "S": shape["S"], "Check": "FAIL"})
        r = rows[-1]
        nan = float("nan")
        print(
            f"{r['label']:<8} {r['Check']:<4} bwd {r.get('Backward TFLOPS', nan):7.1f} +- {r.get('Backward TFLOPS_stddev', nan):4.1f}"
            f" ({r.get('Backward Time (ms)', nan):8.3f} ms) aiter {r.get('aiter Backward TFLOPS', nan):7.1f}"
            f" r {r.get('bwd_ratio_vs_aiter', nan):.3f} | step {r.get('Step TFLOPS', nan):7.1f}"
            f" aiter {r.get('aiter Step TFLOPS', nan):7.1f} r {r.get('step_ratio_vs_aiter', nan):.3f}"
            f" | snr {r.get('dq_snr', nan):.2f}/{r.get('dk_snr', nan):.2f}/{r.get('dv_snr', nan):.2f}"
            f" aiter {r.get('aiter_dq_snr', nan):.2f}/{r.get('aiter_dk_snr', nan):.2f}/{r.get('aiter_dv_snr', nan):.2f}"
            f" rep {r.get('dkdv_bitwise_repeat', '-')}",
            flush=True,
        )
        torch.cuda.empty_cache()

    def geomean(xs):
        xs = [x for x in xs if x and x > 0 and not math.isnan(x)]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0

    passed = [r for r in rows if r["Check"] == "PASS"]
    print(f"Geomean Step TFLOPS: {geomean([r['Step TFLOPS'] for r in passed]):.2f}")
    print(f"Geomean Bwd TFLOPS: {geomean([r['Backward TFLOPS'] for r in passed]):.2f}")
    if not args.no_aiter:
        print(f"Geomean aiter Step TFLOPS: {geomean([r.get('aiter Step TFLOPS') for r in passed]):.2f}")
        print(f"Geomean aiter Bwd TFLOPS: {geomean([r.get('aiter Backward TFLOPS') for r in passed]):.2f}")
        print(f"Geomean step ratio vs aiter: {geomean([r.get('step_ratio_vs_aiter') for r in passed]):.4f}")
        print(f"Geomean bwd ratio vs aiter: {geomean([r.get('bwd_ratio_vs_aiter') for r in passed]):.4f}")
    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADER)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in SUMMARY_CSV_HEADER})
        print(f"Summary CSV: {args.summary_csv}")
    return 0 if all(r["Check"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
