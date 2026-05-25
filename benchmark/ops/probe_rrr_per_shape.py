"""Per-shape RRR dgrad probe: for each of 24 MoE shapes, sweep all
(group_m, num_xcds, bn_block) candidates + a chunk_size override sweep,
report best HK time + HK/Triton ratio. Median-of-3 trials.

Used by Session 7 to validate / extend the autotune candidate list.
Output: stdout table + /tmp/probe_rrr_per_shape.json (raw timing matrix).
"""
import os, sys, json, statistics, torch

torch.ops.load_library(os.path.abspath("primus_turbo/lib/libprimus_turbo_kernels.so"))
_pyver = f"_C.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
torch.ops.load_library(os.path.abspath(f"primus_turbo/pytorch/{_pyver}"))

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs, grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

torch.manual_seed(0)
DEV = "cuda"
GRAN = ScalingGranularity.TENSORWISE
FP8 = torch.float8_e4m3fn

MODELS = {
    "gpt_oss":  dict(hidden=2880, inter=2880, gated=2),
    "dsv3":     dict(hidden=7168, inter=2048, gated=2),
    "qwen235b": dict(hidden=4096, inter=1536, gated=2),
}
B_VALUES = [4, 16]
M_VALUES = [2048, 4096]

# Current autotune candidates from _HK_FP8_RRR_CANDIDATES
RRR_CANDIDATES = [
    (1, 0), (1, 4), (2, 4), (4, 0), (4, 4), (8, 0), (8, 4),
    (4, 8), (4, 16), (4, 32),
    (8, 16), (8, 32),
    (16, 0), (16, 4),
    (24, 0),
    (12, 4),
]

# Session 10: chunk_size sweep. 0 = use dispatcher heuristic.
CHUNK_CHOICES = [0, 32, 48, 64, 96]


def make_shapes():
    out = []
    for model, p in MODELS.items():
        N_up = p["gated"] * p["inter"]; K_up = p["hidden"]
        N_down = p["hidden"];           K_down = p["inter"]
        for B in B_VALUES:
            for M in M_VALUES:
                out.append((model, "up",   B, M, N_up,   K_up))
                out.append((model, "down", B, M, N_down, K_down))
    return out


def time_call(call, n_warmup=5, n_iter=30):
    for _ in range(n_warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        call()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter


def time_call_median(call, n_trials=3, n_warmup=5, n_iter=30):
    ts = []
    for _ in range(n_trials):
        ts.append(time_call(call, n_warmup, n_iter))
    return statistics.median(ts)


def setup_dgrad(B, Mg, N, K):
    M = B * Mg
    g_lens = torch.full((B,), Mg, dtype=torch.int64, device=DEV)
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((M, N), dtype=torch.bfloat16, device=DEV)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)
    go_fp8, go_s = quantize_fp8(grad_out, FP8, GRAN)
    b_fp8, b_s = quantize_fp8(b, FP8, GRAN)
    return M, g_lens, g_offs, go_fp8, b_fp8, go_s, b_s


def probe_one_shape(B, Mg, N, K):
    """Returns dict with raw timings + best cfg + Triton ratio."""
    M, g_lens, g_offs, go_fp8, b_fp8, go_s, b_s = setup_dgrad(B, Mg, N, K)
    avg_m = max(M // B, 1)

    # Time Triton baseline (auto, no probe)
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    GlobalBackendManager.set_auto_tune(False)
    triton_call = lambda: grouped_gemm_fp8_impl(
        go_fp8, b_fp8, go_s, b_s, g_lens, g_offs,
        trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
        granularity=GRAN.value, num_cu=None,
        default_backend=BackendType.TRITON.value,
    )
    t_triton = time_call_median(triton_call)

    # Probe HK candidates directly via op (bypass autotune)
    op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8
    bn_choices = (0, 128) if (N % 128 == 0) else (0,)

    times = {}
    for gm, xcds in RRR_CANDIDATES:
        for bn in bn_choices:
            for ck in CHUNK_CHOICES:
                call = lambda gm=gm, xcds=xcds, bn=bn, ck=ck: op(
                    go_fp8, b_fp8, go_s, b_s, g_offs,
                    gm, avg_m, xcds, torch.bfloat16, bn, ck)
                try:
                    t = time_call_median(call)
                except Exception as ex:
                    t = float("inf")
                times[(gm, xcds, bn, ck)] = t

    best_cfg, best_t = min(times.items(), key=lambda kv: kv[1])
    return {
        "B": B, "Mg": Mg, "N": N, "K": K,
        "triton_ms": t_triton,
        "best_cfg": best_cfg,
        "best_hk_ms": best_t,
        "ratio_best": t_triton / best_t,
        "all_times": {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in times.items()},
    }


def main():
    shapes = make_shapes()
    results = []
    hdr = f"{'idx':>3} {'model':<10} {'op':<5} {'B':>3} {'M':>5} {'N':>5} {'K':>5} | {'best (gm,xc,bn,ck)':>22} | {'HK ms':>8} {'TR ms':>8} {'ratio':>6} | {'pass≥1.15':>8}"
    print(hdr); print('-' * len(hdr))
    n_pass = 0
    for i, (model, op, B, Mg, N, K) in enumerate(shapes):
        r = probe_one_shape(B, Mg, N, K)
        r["model"] = model; r["op"] = op
        results.append(r)
        ratio = r["ratio_best"]
        passed = ratio >= 1.15
        n_pass += int(passed)
        print(f"{i:>3} {model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5} | "
              f"{str(r['best_cfg']):>22} | {r['best_hk_ms']:>8.3f} {r['triton_ms']:>8.3f} {ratio:>6.3f}x | "
              f"{'PASS' if passed else 'fail':>8}")
    print('-' * len(hdr))
    ratios = [r["ratio_best"] for r in results]
    print(f"24-shape best-of-candidates: geomean {statistics.geometric_mean(ratios):.3f}x  min {min(ratios):.3f}  max {max(ratios):.3f}")
    print(f"Pass count (ratio ≥ 1.15): {n_pass}/{len(results)}")
    with open("/tmp/probe_rrr_per_shape.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Raw timing matrix → /tmp/probe_rrr_per_shape.json")


if __name__ == "__main__":
    main()
