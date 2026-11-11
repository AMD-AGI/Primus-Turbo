"""MLA 192/128 causal SBHD forward: correctness gate + same-session A/B benchmark.

Kernels: "cand" (package dir under test), optional "base" (last accepted package dir) and
aiter (Turbo AITER backend, training forward with requires_grad leaves).
Correctness: cand O relative error and LSE max abs error on sampled rows (all heads, first and
last batch) vs fp32 (O rel <= 3e-3, LSE <= 1e-2), plus bitwise O / LSE comparison vs base.
Timing: per shape, `--rot` rotations of cand base aiter aiter base cand; per-kernel median.
A package dir holds the gfx950 MLA package files (__init__.py, ops.py, ...). Every candidate
must carry its own CODEGEN_REV so cand and base never share a FlyDSL cache entry.

Usage: python ab_bench.py --cand DIR [--base DIR] [--set full|quick] [--rot N] [--csv PATH]
                          [--no-aiter] [--no-check]
"""

import argparse
import csv
import importlib.util
import math
import os
import statistics
import sys

import torch

from primus_turbo.flydsl.utils import buffer_ops

if not hasattr(buffer_ops, "_get_buffer_flags"):
    # gfx9 (CDNA) V# word 3: DATA_FORMAT 7, NUM_FORMAT 4.
    buffer_ops._get_buffer_flags = lambda arch=None: (7 << 12) | (4 << 15)

import aiter  # noqa: E402

import primus_turbo.pytorch as pt  # noqa: E402
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType  # noqa: E402

DQK, DV = 192, 128
FULL = [
    ("dsv3", 2, 4096, 128),
    ("dsv2lite", 8, 4096, 16),
    ("kimik2", 4, 4096, 64),
    ("kimik3", 2, 7168, 96),
    ("b1_s4k", 1, 4096, 128),
    ("b1_s8k", 1, 8192, 128),
    ("b1_s16k", 1, 16384, 128),
    ("b1_s32k", 1, 32768, 128),
    ("b1_s64k", 1, 65536, 128),
    ("b2_s4k", 2, 4096, 128),
    ("b2_s16k", 2, 16384, 128),
]
QUICK = ["dsv3", "dsv2lite", "kimik3", "b1_s16k", "b1_s64k"]


def load_pkg(path, name):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sample_rows(s, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randperm(s, generator=g)[:256].tolist()
    edge = [0, 1, 31, 32, 63, 64, 127, 128, 255, 256, s - 257, s - 256, s - 65, s - 64, s - 2, s - 1]
    return torch.tensor(sorted({r for r in picks + edge if 0 <= r < s}), device="cuda")


def exact_rows(q, k, v, rows, b):
    mask = torch.arange(k.shape[0], device=q.device)[None, :] > rows[:, None]
    outs, lses = [], []
    for h in range(q.shape[2]):
        s = q[rows, b, h].float() @ k[:, b, h].float().T * DQK**-0.5
        s.masked_fill_(mask, float("-inf"))
        lses.append(torch.logsumexp(s, -1))
        outs.append(torch.softmax(s, -1) @ v[:, b, h].float())
    return torch.stack(outs, 1), torch.stack(lses, 1)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def time_ms(fn, budget_s=0.3, reps=3):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    fn()
    en.record()
    torch.cuda.synchronize()
    iters = max(3, min(200, round(budget_s * 1e3 / max(st.elapsed_time(en), 1e-3))))
    out = []
    for _ in range(reps):
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en) / iters)
    return statistics.median(out)


def run(shape, pkgs, with_aiter, check, rot):
    label, B, S, H = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (DQK, DQK, DV))
    fns = {n: (lambda m=m: m.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)) for n, m in pkgs.items()}
    if with_aiter:
        leaves = [t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v)]
        fns["aiter"] = lambda: pt.ops.flash_attn_func(*leaves, softmax_scale=DQK**-0.5, causal=True)
    row = {"label": label, "B": B, "S": S, "H": H}

    o_c, lse_c = fns["cand"]()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(o_c).all().item() and torch.isfinite(lse_c).all().item())
    o_rel = lse_max = float("nan")
    if check:
        o_rel, lse_max = 0.0, 0.0
        lv = lse_c.view(B, S, H)
        for bi in sorted({0, B - 1}):
            idx = sample_rows(S, bi)
            ref_o, ref_lse = exact_rows(q, k, v, idx, bi)
            o_rel = max(o_rel, rel(o_c[idx, bi], ref_o))
            lse_max = max(lse_max, (lv[bi, idx] - ref_lse).abs().max().item())
    ok = finite and (not check or (o_rel <= 3e-3 and lse_max <= 1e-2))
    row.update(Check="PASS" if ok else "FAIL", o_rel_fp32=f"{o_rel:.3e}", lse_max_fp32=f"{lse_max:.3e}")
    if "base" in fns:
        o_b, lse_b = fns["base"]()
        torch.cuda.synchronize()
        row["bitwise_vs_base"] = bool(torch.equal(o_b, o_c) and torch.equal(lse_b, lse_c))
        row["o_maxdiff_vs_base"] = f"{(o_b.float() - o_c.float()).abs().max().item():.3e}"
        del o_b, lse_b
    del o_c, lse_c

    names = list(fns)
    order = names + names[::-1]
    times = {n: [] for n in names}
    for _ in range(rot):
        for n in order:
            times[n].append(time_ms(fns[n]))
    flops = B * S * (S + 1) * H * (DQK + DV)
    for n in names:
        ms = statistics.median(times[n])
        row[f"{n}_ms"] = round(ms, 4)
        row[f"{n}_tflops"] = round(flops / ms / 1e9, 1)
        row[f"{n}_spread_pct"] = round((max(times[n]) - min(times[n])) / ms * 100, 2)
    msg = f"{label:9s} {row['Check']} o_rel {row['o_rel_fp32']} lse {row['lse_max_fp32']}  cand {row['cand_tflops']:7.1f}"
    if "base" in fns:
        msg += f"  base {row['base_tflops']:7.1f}  cand/base {row['cand_tflops'] / row['base_tflops']:.4f}"
        msg += f"  bitwise {row['bitwise_vs_base']}"
    if with_aiter:
        msg += f"  aiter {row['aiter_tflops']:7.1f}  cand/aiter {row['cand_tflops'] / row['aiter_tflops']:.4f}"
    msg += "  spread% " + "/".join(str(row[f"{n}_spread_pct"]) for n in names)
    print(msg, flush=True)
    return row


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base")
    ap.add_argument("--set", choices=["full", "quick"], default="full")
    ap.add_argument("--rot", type=int, default=2)
    ap.add_argument("--csv")
    ap.add_argument("--no-aiter", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    pkgs = {"cand": load_pkg(args.cand, "mla_cand")}
    if args.base:
        pkgs["base"] = load_pkg(args.base, "mla_base")
    GlobalBackendManager.set_attn_backend(BackendType.AITER, PrecisionType.BF16_FP16_FP32)
    print("aiter", aiter.__file__, "| cand", os.path.abspath(args.cand), "| base", args.base, flush=True)
    shapes = FULL if args.set == "full" else [s for s in FULL if s[0] in QUICK]
    rows = []
    for shape in shapes:
        rows.append(run(shape, pkgs, not args.no_aiter, not args.no_check, args.rot))
        torch.cuda.empty_cache()

    passed = [r for r in rows if r["Check"] == "PASS"]
    names = [n for n in ("cand", "base", "aiter") if f"{n}_tflops" in rows[0]]
    g = {n: gm([r[f"{n}_tflops"] for r in passed]) for n in names}
    line = f"PASS {len(passed)}/{len(rows)}  geomean " + "  ".join(f"{n} {g[n]:.2f}" for n in names)
    if "base" in g:
        line += f"  cand/base {g['cand'] / g['base']:.4f}"
        line += f"  bitwise {sum(bool(r['bitwise_vs_base']) for r in rows)}/{len(rows)}"
    if "aiter" in g:
        line += f"  cand/aiter {g['cand'] / g['aiter']:.4f}"
        if "base" in g:
            line += f"  base/aiter {g['base'] / g['aiter']:.4f}"
    print(line, flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    sys.exit(0 if len(passed) == len(rows) else 1)


if __name__ == "__main__":
    main()
