"""Quick correctness + benchmark for the FlyDSL 192/128 forward prototype (DSV3 MLA).

Adapted from optimize-handoff/quick_test_bench.py for an attention forward:
the GEMM columns are replaced by B / S / heads, the SNR gate by an fp32-row gate
on O and LSE, and Turbo AITER is timed back to back on the same GPU as the
per-round reference.

Fresh-tensor note: every timed call reads the same q / k / v, but the kernel has
no wrapper cache, so the benchmark idiom cannot inflate it (Rule 11 K1).

Usage:
    HIP_VISIBLE_DEVICES=0 python quick_test_bench.py --set quick|full [--impl r7|swp8]
        [--repeats N] [--budget-s S] [--summary-csv PATH] [--no-aiter]
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")

import torch  # noqa: E402
from flydsl_mla import mla_fwd_sbhd as mla_fwd_r7  # noqa: E402
from flydsl_mla import mla_fwd_sbhd_swp8  # noqa: E402

mla_fwd_sbhd = mla_fwd_r7

import primus_turbo.pytorch as pt  # noqa: E402
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager  # noqa: E402

HQ, HKV, DQK, DV = 128, 128, 192, 128
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

O_REL_MAX = 3e-3
LSE_ABS_MAX = 1e-2
CHECK_ROWS = 256

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
    "Forward TFLOPS_stddev",
    "Forward Time (ms)",
    "aiter Forward TFLOPS",
    "aiter Forward TFLOPS_stddev",
    "ratio_vs_aiter",
    "o_rel_fp32",
    "lse_max_fp32",
    "o_rel_aiter",
]


def fwd_flops(b, s):
    return b * s * (s + 1) // 2 * HQ * 2 * (DQK + DV)


def sample_rows(s, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randperm(s, generator=g)[:CHECK_ROWS].tolist()
    edge = [0, 1, 31, 32, 63, 64, 127, 128, 255, 256, s - 257, s - 256, s - 65, s - 64, s - 2, s - 1]
    return torch.tensor(sorted({r for r in picks + edge if 0 <= r < s}), device="cuda")


def exact_rows(q, k, v, rows, batch):
    scale = DQK**-0.5
    mask = torch.arange(k.shape[0], device=q.device)[None, :] > rows[:, None]
    outs, lses = [], []
    for h in range(HQ):
        s = q[rows, batch, h].float() @ k[:, batch, h].float().T * scale
        s.masked_fill_(mask, float("-inf"))
        lses.append(torch.logsumexp(s, -1))
        outs.append(torch.softmax(s, -1) @ v[:, batch, h].float())
    return torch.stack(outs, 1), torch.stack(lses, 1)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def aiter_fwd(q, k, v):
    GlobalBackendManager.set_attn_backend(BackendType.AITER)
    qb, kb, vb = (t.permute(1, 0, 2, 3) for t in (q, k, v))
    return pt.ops.flash_attn_func(qb, kb, vb, softmax_scale=DQK**-0.5, causal=True).permute(1, 0, 2, 3)


def timed(fn, repeats, budget_s):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    est = time.perf_counter() - t0
    iters = max(3, min(200, round(budget_s / repeats / max(est, 1e-6))))
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    return samples


def run_one(shape, args):
    b, s = shape["B"], shape["S"]
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn((s, b, h, d), device="cuda", dtype=torch.bfloat16, generator=g)
        for h, d in ((HQ, DQK), (HKV, DQK), (HKV, DV))
    )
    o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    torch.cuda.synchronize()
    lse = lse.view(b, s, HQ)
    o_rel, lse_max = 0.0, 0.0
    for bi in sorted({0, b - 1}):
        idx = sample_rows(s, bi)
        ref_o, ref_lse = exact_rows(q, k, v, idx, bi)
        o_rel = max(o_rel, rel(o[idx, bi], ref_o))
        lse_max = max(lse_max, (lse[bi, idx] - ref_lse).abs().max().item())
    finite = bool(torch.isfinite(o).all().item() and torch.isfinite(lse).all().item())
    ok = finite and o_rel <= O_REL_MAX and lse_max <= LSE_ABS_MAX
    o_rel_aiter = rel(o, aiter_fwd(q, k, v))
    del o, lse

    flops = fwd_flops(b, s)
    ms = timed(lambda: mla_fwd_sbhd(q, k, v, causal=True, return_lse=True), args.repeats, args.budget_s)
    tf = [flops / (t * 1e-3) / 1e12 for t in ms]
    row = {
        "label": shape["label"],
        "B": b,
        "S": s,
        "HQ": HQ,
        "HKV": HKV,
        "DQK": DQK,
        "DV": DV,
        "Check": "PASS" if ok else "FAIL",
        "Forward TFLOPS": statistics.fmean(tf),
        "Forward TFLOPS_stddev": statistics.pstdev(tf),
        "Forward Time (ms)": statistics.fmean(ms),
        "o_rel_fp32": o_rel,
        "lse_max_fp32": lse_max,
        "o_rel_aiter": o_rel_aiter,
    }
    if not args.no_aiter:
        leaves = [t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v)]

        def aiter_call():
            GlobalBackendManager.set_attn_backend(BackendType.AITER)
            pt.ops.flash_attn_func(*leaves, softmax_scale=DQK**-0.5, causal=True)

        atf = [flops / (t * 1e-3) / 1e12 for t in timed(aiter_call, args.repeats, args.budget_s)]
        row["aiter Forward TFLOPS"] = statistics.fmean(atf)
        row["aiter Forward TFLOPS_stddev"] = statistics.pstdev(atf)
        row["ratio_vs_aiter"] = row["Forward TFLOPS"] / row["aiter Forward TFLOPS"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["quick", "full"], default="quick")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--budget-s", type=float, default=3.0)
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--no-aiter", action="store_true")
    ap.add_argument("--impl", choices=["r7", "swp8"], default="swp8")
    ap.add_argument("--kw", default="", help="swp8 build knobs, e.g. stagger=0,sched_hints=1")
    args = ap.parse_args()
    global mla_fwd_sbhd
    if args.impl == "r7":
        mla_fwd_sbhd = mla_fwd_r7
    else:
        kw = {k: bool(int(v)) for k, v in (p.split("=") for p in args.kw.split(",") if p)}
        mla_fwd_sbhd = functools.partial(mla_fwd_sbhd_swp8, **kw)

    shapes = TARGET_SHAPES if args.set == "full" else REPRESENTATIVE_SHAPES
    rows = []
    for shape in shapes:
        try:
            rows.append(run_one(shape, args))
        except Exception as e:  # noqa: BLE001
            print(f"{shape['label']}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({"label": shape["label"], "B": shape["B"], "S": shape["S"], "Check": "FAIL"})
        r = rows[-1]
        print(
            f"{r['label']:<10} {r['Check']:<5} fwd {r.get('Forward TFLOPS', float('nan')):8.1f}"
            f" +- {r.get('Forward TFLOPS_stddev', float('nan')):5.1f}"
            f" aiter {r.get('aiter Forward TFLOPS', float('nan')):8.1f}"
            f" ratio {r.get('ratio_vs_aiter', float('nan')):.3f}"
            f" o_rel {r.get('o_rel_fp32', float('nan')):.4f} lse {r.get('lse_max_fp32', float('nan')):.4f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    def geomean(xs):
        xs = [x for x in xs if x and x > 0 and not math.isnan(x)]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0

    passed = [r for r in rows if r["Check"] == "PASS"]
    print(f"Geomean Fwd TFLOPS: {geomean([r['Forward TFLOPS'] for r in passed]):.2f}")
    if not args.no_aiter:
        print(f"Geomean aiter Fwd TFLOPS: {geomean([r.get('aiter Forward TFLOPS') for r in passed]):.2f}")
        print(f"Geomean ratio vs aiter: {geomean([r.get('ratio_vs_aiter') for r in passed]):.4f}")
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
