"""Quick correctness + benchmark for grouped_gemm_fp8_rrr campaign (36-case version).

3 model (dsv3 / qwen235b / gpt_oss) × {up, down} × B ∈ {4,16} × M_g ∈ {2048,4096,8192} = 36 cases.

For each (B, M_g, N, K) the script measures:
- dense baseline: hk_gemm_fp8 "rrr" at [M_g, K] × [K, N] (smallest gm chosen by autotune sweep)
- grouped bn128 / bn256: hk_grouped_rrr_fp8 sweep over (group_m, xcds)
- best_grouped_T = max(bn128_T, bn256_T)
- ratio = best_grouped_T / dense_T   (target: >= 0.95 everywhere)

Outputs canonical SUMMARY_CSV_HEADER columns + project-specific dense/ratio columns.
Correctness: torch SNR vs bf16 reference, threshold 25 dB (FP8 E4M3).

Usage:
    python quick_test_bench.py --shapes {representative|full}
        [--summary-csv PATH] [--n-iter N] [--warmup N]
"""
from __future__ import annotations
import argparse, csv, math, os, sys

_REPO_ROOT = os.environ.get("PT_REPO", "/workspace/code/Primus-Turbo")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch  # noqa: E402
import primus_turbo  # noqa: E402  (registers cpp ops)
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs  # noqa: E402
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8  # noqa: E402
from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8  # noqa: E402
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig, Format, ScalingGranularity, float8_e4m3,
)
from primus_turbo.pytorch.core.backend import GlobalBackendManager, BackendType  # noqa: E402
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref  # noqa: E402
from tests.pytorch.test_utils import compute_snr  # noqa: E402

torch.manual_seed(42)  # aligned with PT tests
DEV = "cuda"
SNR_THRESHOLD = 28.0  # per user 2026-05-21: real-FP8-E4M3 grouped should pass 28 dB

hk_gemm = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8
hk_grp  = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8

DENSE_GMS = [1, 2, 4, 8, 16]
GRP_CANDS = [(2, 4), (4, 4), (4, 16), (4, 32), (8, 32), (8, 16), (4, 8),
             (16, 4), (1, 4), (8, 4), (1, 0), (4, 0), (8, 0), (16, 0)]

MODELS = {
    "dsv3":     dict(hidden=7168, inter=2048),
    "qwen235b": dict(hidden=4096, inter=1536),
    "gpt_oss":  dict(hidden=2880, inter=2880),
}
GATED_FACTOR = 2  # up proj is gated x2 (SwiGLU style)

def shape_of(model: str, op: str):
    p = MODELS[model]
    if op == "up":   return (GATED_FACTOR * p["inter"], p["hidden"])  # N, K
    if op == "down": return (p["hidden"],               p["inter"])
    raise ValueError(op)

def all_target_shapes():
    """30 cases: dsv3 (12) + qwen235b (12) + gpt_oss-up only (6).
    gpt_oss-down dropped because N=2880 % 256 != 0 (HK kernel unsupported).
    gpt_oss-up has N=5760, 256∤N but 128|N — grouped uses BN=128 path; dense
    falls back to PT high-level gemm_fp8 (auto-routes to best available backend)."""
    out = []
    for model in MODELS:
        for op in ("up", "down"):
            if model == "gpt_oss" and op == "down":
                continue  # N=2880 unsupported by HK (256∤N and 128∤N as 2880/128=22.5)
            N, K = shape_of(model, op)
            for B in (4, 16):
                for M_g in (4096, 8192):  # M_g=2048 dropped 2026-05-21: dense underutilized at small M inflates ratio>>1
                    short_model = {"dsv3": "dsv3", "qwen235b": "qwen", "gpt_oss": "gpt"}[model]
                    out.append({
                        "label": f"{short_model}-{op}-B{B}-M{M_g}",
                        "model": model, "op": op, "B": B, "M_g": M_g, "N": N, "K": K,
                    })
    return out

# Representative subset (matches manifest representative_shapes).
REPRESENTATIVE_LABELS = {
    "qwen-down-B4-M4096",     # smallest K=1536 down
    "qwen-down-B16-M8192",    # B=16 + K=1536 — current SNR FAIL & gap 14.76% target
    "dsv3-up-B4-M4096",       # worst gap 17.77%
    "dsv3-up-B16-M8192",      # B=16 + K=7168 worst dsv3 case
    "gpt-up-B4-M8192",        # M_max + fallback dense (ratio 1.23)
    "gpt-up-B16-M4096",       # B=16 + N=5760 fallback dense (ratio 2.23)
}

SUMMARY_CSV_HEADER = [
    "label", "model", "op", "B", "M", "N", "K", "Check",
    "Forward TFLOPS", "Forward TFLOPS_stddev",
    "Backward TFLOPS", "Backward TFLOPS_stddev",
    "Forward Time (ms)", "Backward Time (ms)",
    "out_snr", "da_snr", "db_snr",
    "dense_TFLOPS", "dense_time_us", "dense_backend",
    "best_grouped_TFLOPS", "best_grouped_time_us", "best_grouped_route", "best_grouped_cfg",
    "ratio_grouped_over_dense",
]


def _time_us(f, w=10, n=200):
    for _ in range(w):
        f(); torch.cuda.synchronize()
    e = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
    e[0].record()
    for i in range(n):
        f(); e[i + 1].record()
    torch.cuda.synchronize()
    ms = sum(e[i].elapsed_time(e[i + 1]) for i in range(n)) / n
    return ms * 1000.0


def _quant(t):
    return quantize_fp8(t, float8_e4m3, ScalingGranularity.TENSORWISE)


_dense_cache = {}
def measure_dense(M_g, N, K):
    """Try HK dense (best when N%256==0); else fall back to PT high-level gemm_fp8
    which auto-routes to best available backend (typically hipBLASLt for non-256N)."""
    key = (M_g, N, K)
    if key in _dense_cache: return _dense_cache[key]
    if N % 256 == 0 and K % 128 == 0:
        a = (torch.randn((M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        b = (torch.randn((K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        af, asc = _quant(a); bf, bsc = _quant(b)
        best_us = math.inf
        for gm in DENSE_GMS:
            try:
                us = _time_us(lambda: hk_gemm(af, bf, asc, bsc, "rrr", gm, torch.bfloat16))
                best_us = min(best_us, us)
            except Exception:
                pass
        backend = "HK"
    else:
        # PT high-level fallback for unaligned-N shapes (gpt-up N=5760).
        # Reset backend to default so PT picks best avail (hipBLASLt for unaligned).
        GlobalBackendManager.reset()
        a = (torch.randn((M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        b_nk = (torch.randn((N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()  # trans_b=True
        cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
        best_us = _time_us(lambda: gemm_fp8(a, b_nk, trans_b=True, config=cfg))
        backend = "PT-default"
    T = 2.0 * M_g * N * K / (best_us * 1e6)
    _dense_cache[key] = (T, best_us, backend)
    return T, best_us, backend


def measure_grouped(B, M_g, N, K, bn):
    a = (torch.randn((B * M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    b = (torch.randn((B, K, N), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    af, asc = _quant(a); bf, bsc = _quant(b)
    g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
    best_us, best_cfg = math.inf, None
    for gm, xcds in GRP_CANDS:
        try:
            us = _time_us(lambda: hk_grp(af, bf, asc, bsc, g_offs, gm, M_g, xcds,
                                          torch.bfloat16, bn))
            if us < best_us: best_us, best_cfg = us, (gm, xcds)
        except Exception:
            pass
    flops = 2.0 * B * M_g * N * K
    T = flops / (best_us * 1e6)
    return T, best_us, best_cfg


_FP8_CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)

def correctness_check(B, M_g, N, K):
    """Use PT high-level API + PT grouped_gemm_ref + PT compute_snr.
    Forces HipKitten backend via GlobalBackendManager. Returns snr (float)."""
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    GlobalBackendManager.set_auto_tune(False)
    # PT convention: b shape [B, N, K] when trans_b=True
    a = (torch.randn((B * M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    a.requires_grad_(True); b.requires_grad_(True)
    group_lens = torch.full((B,), M_g, dtype=torch.int64, device=DEV)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=_FP8_CFG)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    return float(compute_snr(out_ref, out))


def run_one(case):
    B, M_g, N, K = case["B"], case["M_g"], case["N"], case["K"]
    T_d, us_d, d_backend = measure_dense(M_g, N, K)
    T_128, us_128, cfg_128 = measure_grouped(B, M_g, N, K, 128)
    # bn256 only valid when N%256==0
    if N % 256 == 0:
        T_256, us_256, cfg_256 = measure_grouped(B, M_g, N, K, 256)
    else:
        T_256, us_256, cfg_256 = 0.0, math.inf, None
    if T_128 >= T_256:
        best_T, best_us, route, best_cfg = T_128, us_128, "bn128", cfg_128
    else:
        best_T, best_us, route, best_cfg = T_256, us_256, "bn256", cfg_256
    snr = correctness_check(B, M_g, N, K)
    check = "PASS" if snr >= SNR_THRESHOLD else "FAIL"
    ratio = best_T / T_d if T_d > 0 else float("nan")
    return {
        "label": case["label"], "model": case["model"], "op": case["op"],
        "B": B, "M": M_g, "N": N, "K": K, "Check": check,
        "Forward TFLOPS": best_T, "Forward TFLOPS_stddev": 0.0,
        "Backward TFLOPS": float("nan"), "Backward TFLOPS_stddev": float("nan"),
        "Forward Time (ms)": best_us / 1000.0, "Backward Time (ms)": float("nan"),
        "out_snr": snr, "da_snr": float("nan"), "db_snr": float("nan"),
        "dense_TFLOPS": T_d, "dense_time_us": us_d, "dense_backend": d_backend,
        "best_grouped_TFLOPS": best_T, "best_grouped_time_us": best_us,
        "best_grouped_route": route, "best_grouped_cfg": str(best_cfg),
        "ratio_grouped_over_dense": ratio,
    }


def _empty_row(case, reason: str):
    return {
        "label": case["label"], "model": case["model"], "op": case["op"],
        "B": case["B"], "M": case["M_g"], "N": case["N"], "K": case["K"], "Check": "FAIL",
        **{k: float("nan") for k in SUMMARY_CSV_HEADER if k not in
            ("label","model","op","B","M","N","K","Check","best_grouped_route","best_grouped_cfg")},
        "best_grouped_route": "?", "best_grouped_cfg": f"ERR:{reason[:40]}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", choices=("representative", "full"), default="representative")
    ap.add_argument("--summary-csv", type=str, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA / ROCm device required.", file=sys.stderr); return 2

    cases = all_target_shapes()
    if args.shapes == "representative":
        cases = [c for c in cases if c["label"] in REPRESENTATIVE_LABELS]
    print(f"Running {len(cases)} case(s) ({args.shapes}). GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for i, c in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {c['label']}...", flush=True)
        try:
            rows.append(run_one(c))
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}", file=sys.stderr)
            rows.append(_empty_row(c, str(e)))

    hdr = (f"{'label':<22} {'Check':<5} {'dense':>7} {'grp':>7} "
           f"{'route':>6} {'ratio':>7} {'gap%':>7}")
    print(hdr); print('-' * len(hdr))
    ratios = []
    for r in rows:
        gap_pct = (1 - r["ratio_grouped_over_dense"]) * 100 if not math.isnan(r["ratio_grouped_over_dense"]) else float("nan")
        print(f"{r['label']:<22} {r['Check']:<5} {r['dense_TFLOPS']:>7.0f} "
              f"{r['best_grouped_TFLOPS']:>7.0f} {r['best_grouped_route']:>6} "
              f"{r['ratio_grouped_over_dense']:>7.4f} {gap_pct:>6.2f}%")
        if r["Check"] == "PASS" and not math.isnan(r["ratio_grouped_over_dense"]):
            ratios.append(r["ratio_grouped_over_dense"])
    if ratios:
        gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        print(f"\nprimary_score (geomean ratio) = {gm:.4f}  | min_ratio = {min(ratios):.4f}  "
              f"| n_pass = {len(ratios)}/{len(rows)}")
        worst = min(rows, key=lambda r: r["ratio_grouped_over_dense"] if not math.isnan(r["ratio_grouped_over_dense"]) else 99)
        print(f"worst case: {worst['label']} ratio={worst['ratio_grouped_over_dense']:.4f}")

    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADER)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in SUMMARY_CSV_HEADER})
        print(f"Summary CSV: {args.summary_csv}")

    return 0 if all(r["Check"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
