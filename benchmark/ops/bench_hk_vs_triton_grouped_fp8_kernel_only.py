"""Kernel-only HK vs Triton fp8 grouped fwd/dgrad/wgrad TFLOPS.

Pre-quantize OUTSIDE the timer; time only the single grouped GEMM
GPU dispatch (no autograd, no BF16->FP8 quant tax). This is the
methodology behind the historical "wgrad 1.83x Triton" claim
(scripts/GOAL_fp8_grouped.md). The end-to-end bench
(bench_hk_vs_triton_grouped_fp8_all.py) includes autograd + per-step
quantize and shows a different ratio.
"""
import os, sys, torch

torch.ops.load_library(os.path.abspath("primus_turbo/lib/libprimus_turbo_kernels.so"))
_pyver = f"_C.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
torch.ops.load_library(os.path.abspath(f"primus_turbo/pytorch/{_pyver}"))

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs, grouped_gemm_fp8_impl, grouped_gemm_fp8_variable_k_impl,
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


def time_loop(fn, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter


def bench_section(B, Mg, N, K, section, backend):
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    M = B * Mg
    g_lens = torch.full((B,), Mg, dtype=torch.int64, device=DEV)
    g_offs = grouped_gemm_compute_offs(g_lens)

    if section == "fwd":  # RCR: a(M,K), b(B,N,K) trans_b=True
        a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
        b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)
        a_fp8, a_s = quantize_fp8(a, FP8, GRAN)
        b_fp8, b_s = quantize_fp8(b, FP8, GRAN)
        call = lambda: grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=GRAN.value, num_cu=None,
            default_backend=backend.value,
        )
    elif section == "dgrad":  # RRR: grad_out(M,N), b(B,N,K) trans_b=False
        grad_out = torch.randn((M, N), dtype=torch.bfloat16, device=DEV)
        b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)
        go_fp8, go_s = quantize_fp8(grad_out, FP8, GRAN)
        b_fp8, b_s = quantize_fp8(b, FP8, GRAN)
        call = lambda: grouped_gemm_fp8_impl(
            go_fp8, b_fp8, go_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=GRAN.value, num_cu=None,
            default_backend=backend.value,
        )
    elif section == "wgrad":  # CRR var-K: a(M,K) col, grad(M,N) col, trans_a=True
        a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
        grad = torch.randn((M, N), dtype=torch.bfloat16, device=DEV)
        a_col, a_s_col = quantize_fp8(a, FP8, GRAN, axis=-2)
        g_col, g_s_col = quantize_fp8(grad, FP8, GRAN, axis=-2)
        call = lambda: grouped_gemm_fp8_variable_k_impl(
            a_col, g_col, a_s_col, g_s_col, g_lens, g_offs,
            trans_a=True, trans_b=False, trans_c=True,
            out_dtype=torch.bfloat16, granularity=GRAN.value, num_cu=None,
            default_backend=backend.value,
        )
    else:
        raise ValueError(section)

    try:
        out = call()
        if torch.isnan(out).any() or torch.isinf(out).any():
            return 0.0, float("nan")
    except Exception as ex:
        return 0.0, float("nan")
    ms = time_loop(call)
    tflops = 2.0 * M * N * K / (ms * 1e9)
    return ms, tflops


def main():
    hdr = (f"{'model':<10} {'op':<5} {'B':>3} {'M':>5} {'N':>5} {'K':>5} | "
           f"{'fwd HK T':>9} {'fwd TR T':>9} {'fwd x':>6} | "
           f"{'dg HK T':>9} {'dg TR T':>9} {'dg x':>6} | "
           f"{'wg HK T':>9} {'wg TR T':>9} {'wg x':>6}")
    print(hdr); print('-' * len(hdr))
    fs, ds, ws = [], [], []
    for model, op, B, Mg, N, K in make_shapes():
        _, hf = bench_section(B, Mg, N, K, "fwd",   BackendType.HIPKITTEN)
        _, tf = bench_section(B, Mg, N, K, "fwd",   BackendType.TRITON)
        _, hd = bench_section(B, Mg, N, K, "dgrad", BackendType.HIPKITTEN)
        _, td = bench_section(B, Mg, N, K, "dgrad", BackendType.TRITON)
        _, hw = bench_section(B, Mg, N, K, "wgrad", BackendType.HIPKITTEN)
        _, tw = bench_section(B, Mg, N, K, "wgrad", BackendType.TRITON)
        rf = hf / tf if tf else 0
        rd = hd / td if td else 0
        rw = hw / tw if tw else 0
        if rf: fs.append(rf)
        if rd: ds.append(rd)
        if rw: ws.append(rw)
        print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5} | "
              f"{hf:>9.0f} {tf:>9.0f} {rf:>5.2f}x | "
              f"{hd:>9.0f} {td:>9.0f} {rd:>5.2f}x | "
              f"{hw:>9.0f} {tw:>9.0f} {rw:>5.2f}x")
    if fs:
        import statistics
        gm = statistics.geometric_mean
        print('-' * len(hdr))
        print(f"fwd   HK/Triton kernel-only TFLOPS ratio: geomean {gm(fs):.3f}x  (min {min(fs):.2f}, max {max(fs):.2f})")
        print(f"dgrad HK/Triton kernel-only TFLOPS ratio: geomean {gm(ds):.3f}x  (min {min(ds):.2f}, max {max(ds):.2f})")
        print(f"wgrad HK/Triton kernel-only TFLOPS ratio: geomean {gm(ws):.3f}x  (min {min(ws):.2f}, max {max(ws):.2f})")


if __name__ == "__main__":
    main()
