"""HipKitten vs Triton — fp8 tensorwise grouped GEMM forward perf.

Restricted to gpt_oss / DeepSeek-V3 / Qwen3-235B-A22B MoE shapes.
B in {4, 16}, M_per_group in {2048, 4096}, Up + Down projections.
"""
import os, sys, torch

torch.ops.load_library(os.path.abspath("primus_turbo/lib/libprimus_turbo_kernels.so"))
_pyver = f"_C.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
torch.ops.load_library(os.path.abspath(f"primus_turbo/pytorch/{_pyver}"))

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8

torch.manual_seed(0)
DEV = "cuda"

# Model MoE shapes — (hidden, inter, gated_up_factor) for each model.
# Up proj   : A[M_total, hidden]      × B[E, factor*inter, hidden]^T → C[M_total, factor*inter]
# Down proj : A[M_total, inter]       × B[E, hidden,        inter ]^T → C[M_total, hidden]
MODELS = {
    "gpt_oss":   dict(hidden=2880, inter=2880, gated=2),
    "dsv3":      dict(hidden=7168, inter=2048, gated=2),
    "qwen235b":  dict(hidden=4096, inter=1536, gated=2),
}

B_VALUES = [4, 16]
M_VALUES = [2048, 4096]

CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)


def make_shapes():
    out = []
    for model, p in MODELS.items():
        N_up   = p["gated"] * p["inter"]
        K_up   = p["hidden"]
        N_down = p["hidden"]
        K_down = p["inter"]
        for B in B_VALUES:
            for M in M_VALUES:
                out.append((model, "up",   B, M, N_up,   K_up))
                out.append((model, "down", B, M, N_down, K_down))
    return out


def bench_fwd_bwd(B, Mg, N, K, backend, n_warmup=10, n_iter=50):
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    M = B * Mg
    a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV, requires_grad=True)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV, requires_grad=True)
    group_lens = torch.full((B,), Mg, dtype=torch.int64, device=DEV)
    out0 = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=CFG)
    grad_out = torch.randn_like(out0)

    # Forward
    for _ in range(n_warmup):
        _ = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=CFG)
    torch.cuda.synchronize()
    sf, ef = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    sf.record()
    for _ in range(n_iter):
        _ = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=CFG)
    ef.record()
    torch.cuda.synchronize()
    t_fwd = sf.elapsed_time(ef) / n_iter

    # Backward (re-run fwd each iter so the graph is fresh for backward)
    for _ in range(n_warmup):
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=CFG)
        out.backward(grad_out, retain_graph=False)
    torch.cuda.synchronize()
    sb, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    sb.record()
    for _ in range(n_iter):
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=CFG)
        out.backward(grad_out, retain_graph=False)
    eb.record()
    torch.cuda.synchronize()
    t_fb = sb.elapsed_time(eb) / n_iter
    t_bwd = t_fb - t_fwd  # subtract fwd cost to isolate bwd
    return t_fwd, t_bwd


def main():
    hdr = f"{'model':<10} {'op':<5} {'B':>3} {'M':>5} {'N':>5} {'K':>5} " \
          f"{'fwd HK':>8} {'fwd TR':>8} {'fwd ratio':>10} " \
          f"{'bwd HK':>8} {'bwd TR':>8} {'bwd ratio':>10}"
    print(hdr)
    print('-' * len(hdr))
    fwd_speedups, bwd_speedups = [], []
    for model, op, B, Mg, N, K in make_shapes():
        try:
            t_hk_fwd, t_hk_bwd = bench_fwd_bwd(B, Mg, N, K, BackendType.HIPKITTEN)
            t_tr_fwd, t_tr_bwd = bench_fwd_bwd(B, Mg, N, K, BackendType.TRITON)
        except Exception as ex:
            print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5}  failed: {ex.__class__.__name__}: {ex}")
            continue
        fwd_speedups.append(t_tr_fwd / t_hk_fwd)
        bwd_speedups.append(t_tr_bwd / t_hk_bwd)
        print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5} "
              f"{t_hk_fwd:>8.3f} {t_tr_fwd:>8.3f} {t_hk_fwd/t_tr_fwd:>10.3f} "
              f"{t_hk_bwd:>8.3f} {t_tr_bwd:>8.3f} {t_hk_bwd/t_tr_bwd:>10.3f}")
    if fwd_speedups:
        import statistics
        print('-' * len(hdr))
        print(f"fwd HK speedup vs Triton: geomean {statistics.geometric_mean(fwd_speedups):.3f}x "
              f"(min {min(fwd_speedups):.3f}, max {max(fwd_speedups):.3f})")
        print(f"bwd HK speedup vs Triton: geomean {statistics.geometric_mean(bwd_speedups):.3f}x "
              f"(min {min(bwd_speedups):.3f}, max {max(bwd_speedups):.3f})")


if __name__ == "__main__":
    main()
