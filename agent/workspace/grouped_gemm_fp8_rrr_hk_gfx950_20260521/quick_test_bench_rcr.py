"""RCR variant of quick_test_bench.py — measures hk_grouped_rcr_fp8 vs dense rcr.
For RCR: a is [M, K], b is [G, N, K] (col-major view, trans_b=True semantically).
Dense: hk_gemm_fp8 "rcr" at [M_g, K] × [N, K] trans_b → [M_g, N].
"""
from __future__ import annotations
import argparse, csv, math, os, sys

_REPO_ROOT = os.environ.get("PT_REPO", "/workspace/code/Primus-Turbo")
sys.path.insert(0, _REPO_ROOT)

import torch
import primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3

torch.manual_seed(42)
DEV = "cuda"

hk_gemm = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8
hk_grp  = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_fp8

DENSE_GMS = [1, 2, 4, 8, 16]
GRP_CANDS = [(2, 4), (4, 4), (4, 16), (4, 32), (8, 32), (8, 16), (4, 8),
             (16, 4), (1, 4), (8, 4), (1, 0), (4, 0), (8, 0), (16, 0)]

MODELS = {
    "dsv3":     dict(hidden=7168, inter=2048),
    "qwen235b": dict(hidden=4096, inter=1536),
    "gpt_oss":  dict(hidden=2880, inter=2880),
}
GATED_FACTOR = 2

def shape_of(model: str, op: str):
    p = MODELS[model]
    if op == "up":   return (GATED_FACTOR * p["inter"], p["hidden"])
    if op == "down": return (p["hidden"],               p["inter"])
    raise ValueError(op)

def all_target_shapes():
    out = []
    for model in MODELS:
        for op in ("up", "down"):
            if model == "gpt_oss" and op == "down":
                continue
            N, K = shape_of(model, op)
            for B in (4, 16):
                for M_g in (4096, 8192):
                    short_model = {"dsv3": "dsv3", "qwen235b": "qwen", "gpt_oss": "gpt"}[model]
                    out.append({
                        "label": f"{short_model}-{op}-B{B}-M{M_g}",
                        "model": model, "op": op, "B": B, "M_g": M_g, "N": N, "K": K,
                    })
    return out

def _time_us(f, w=10, n=200):
    for _ in range(w): f(); torch.cuda.synchronize()
    e = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
    e[0].record()
    for i in range(n): f(); e[i + 1].record()
    torch.cuda.synchronize()
    return sum(e[i].elapsed_time(e[i + 1]) for i in range(n)) / n * 1000.0

def _quant(t):
    return quantize_fp8(t, float8_e4m3, ScalingGranularity.TENSORWISE)

_dense_cache = {}
def measure_dense(M_g, N, K):
    key = (M_g, N, K)
    if key in _dense_cache: return _dense_cache[key]
    if N % 256 == 0 and K % 128 == 0:
        a = (torch.randn((M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        # RCR: b is [N, K] (col-major B, trans_b=True semantically)
        b = (torch.randn((N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        af, asc = _quant(a); bf, bsc = _quant(b)
        best_us = math.inf
        for gm in DENSE_GMS:
            try:
                us = _time_us(lambda: hk_gemm(af, bf, asc, bsc, "rcr", gm, torch.bfloat16))
                best_us = min(best_us, us)
            except Exception: pass
        backend = "HK"
    else:
        from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8
        from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format
        from primus_turbo.pytorch.core.backend import GlobalBackendManager
        GlobalBackendManager.reset()
        a = (torch.randn((M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        b_nk = (torch.randn((N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
        best_us = _time_us(lambda: gemm_fp8(a, b_nk, trans_b=True, config=cfg))
        backend = "PT-default"
    T = 2.0 * M_g * N * K / (best_us * 1e6)
    _dense_cache[key] = (T, best_us, backend)
    return T, best_us, backend

def measure_grouped(B, M_g, N, K, bn):
    a = (torch.randn((B * M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    # RCR: b is [G, N, K] (col-major view)
    b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    af, asc = _quant(a); bf, bsc = _quant(b)
    g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
    best_us, best_cfg = math.inf, None
    for gm, xcds in GRP_CANDS:
        try:
            us = _time_us(lambda: hk_grp(af, bf, asc, bsc, g_offs, gm, M_g, xcds,
                                          torch.bfloat16, bn))
            if us < best_us: best_us, best_cfg = us, (gm, xcds)
        except Exception: pass
    if best_us == math.inf:
        return 0.0, 0.0, None
    flops = 2.0 * B * M_g * N * K
    T = flops / (best_us * 1e6)
    return T, best_us, best_cfg

def run_one(case):
    B, M_g, N, K = case["B"], case["M_g"], case["N"], case["K"]
    T_d, us_d, d_backend = measure_dense(M_g, N, K)
    bns = [0, 128, -128] if N % 128 == 0 else [0]
    results = []
    for bn in bns:
        # RCR doesn't support bn=-128 cleanly perhaps; skip if errors
        T, us, cfg = measure_grouped(B, M_g, N, K, bn)
        if T > 0:
            results.append((bn, T, us, cfg))
    if not results: return None
    best = max(results, key=lambda r: r[1])
    bn, T_g, us_g, cfg = best
    ratio = T_g / T_d if T_d > 0 else float("nan")
    return {
        "label": case["label"], "model": case["model"], "op": case["op"],
        "B": B, "M": M_g, "N": N, "K": K,
        "Forward TFLOPS": T_g,
        "dense_TFLOPS": T_d, "dense_time_us": us_d, "dense_backend": d_backend,
        "best_grouped_TFLOPS": T_g, "best_grouped_time_us": us_g,
        "best_grouped_route": f"bn{bn}" if bn != 0 else "bn256",
        "best_grouped_cfg": str(cfg),
        "ratio_grouped_over_dense": ratio,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="full")
    ap.add_argument("--summary-csv", default=None)
    args = ap.parse_args()
    cases = all_target_shapes()
    print(f"Running {len(cases)} RCR case(s). GPU: {torch.cuda.get_device_name()}")
    rows = []
    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {c['label']}...")
        r = run_one(c)
        if r: rows.append(r)
    print(f'{"label":22} {"dense":>5} {"grp":>5}  {"route":6} {"ratio":>7} {"gap%":>7}')
    print("-" * 60)
    geom = 1.0
    min_r = float("inf")
    for r in rows:
        ratio = r["ratio_grouped_over_dense"]
        gap = (1 - ratio) * 100
        print(f'{r["label"]:22} {r["dense_TFLOPS"]:5.0f} {r["Forward TFLOPS"]:5.0f}  '
              f'{r["best_grouped_route"]:6} {ratio:7.4f} {gap:6.2f}%')
        geom *= ratio
        min_r = min(min_r, ratio)
    geom = geom ** (1 / len(rows))
    print(f"\nprimary_score = {geom:.4f}  | min_ratio = {min_r:.4f}  | n = {len(rows)}")
    if args.summary_csv and rows:
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
