"""HipKitten vs Triton — fp8 tensorwise grouped GEMM forward + dgrad-only perf.
Restricted to gpt_oss / DeepSeek-V3 / Qwen3-235B-A22B MoE shapes.

dgrad = gradient w.r.t. A only (b.requires_grad=False).
fwd is measured separately; full bwd would include wgrad.
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
        N_up   = p["gated"] * p["inter"]; K_up   = p["hidden"]
        N_down = p["hidden"];              K_down = p["inter"]
        for B in B_VALUES:
            for M in M_VALUES:
                out.append((model, "up",   B, M, N_up,   K_up))
                out.append((model, "down", B, M, N_down, K_down))
    return out


def bench_fwd_and_dgrad(B, Mg, N, K, backend, n_warmup=10, n_iter=50):
    """fwd: standalone forward pass. dgrad: backward with only A.requires_grad=True
    (so wgrad doesn't run). Returns (fwd_ms, dgrad_ms)."""
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    M = B * Mg
    group_lens = torch.full((B,), Mg, dtype=torch.int64, device=DEV)

    # FWD timing — both tensors requires_grad=False (pure fwd)
    a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)
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

    # DGRAD timing: A.requires_grad=True only; B detached
    a_g = torch.randn((M, K), dtype=torch.bfloat16, device=DEV, requires_grad=True)
    b_d = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)  # no grad
    out0 = grouped_gemm_fp8(a_g, b_d, group_lens, trans_b=True, config=CFG)
    grad_out = torch.randn_like(out0)

    for _ in range(n_warmup):
        out = grouped_gemm_fp8(a_g, b_d, group_lens, trans_b=True, config=CFG)
        out.backward(grad_out, retain_graph=False)
        a_g.grad = None
    torch.cuda.synchronize()
    sb, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    sb.record()
    for _ in range(n_iter):
        out = grouped_gemm_fp8(a_g, b_d, group_lens, trans_b=True, config=CFG)
        out.backward(grad_out, retain_graph=False)
        a_g.grad = None
    eb.record()
    torch.cuda.synchronize()
    t_fb = sb.elapsed_time(eb) / n_iter
    t_dgrad = t_fb - t_fwd  # subtract fwd time
    return t_fwd, max(t_dgrad, 0.001)  # avoid div-by-0 if measurement noise


def main():
    hdr = (f"{'model':<10} {'op':<5} {'B':>3} {'M':>5} {'N':>5} {'K':>5} | "
           f"{'fwd HK':>7} {'fwd TR':>7} {'fwd ratio':>10} | "
           f"{'dgrad HK':>9} {'dgrad TR':>9} {'dgr ratio':>10}")
    print(hdr); print('-' * len(hdr))
    fwd_speedups, dgrad_speedups = [], []
    for model, op, B, Mg, N, K in make_shapes():
        try:
            hf, hd = bench_fwd_and_dgrad(B, Mg, N, K, BackendType.HIPKITTEN)
            tf, td = bench_fwd_and_dgrad(B, Mg, N, K, BackendType.TRITON)
        except Exception as ex:
            print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5}  failed: {ex.__class__.__name__}: {ex}")
            continue
        fwd_speedups.append(tf / hf); dgrad_speedups.append(td / hd)
        print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5} | "
              f"{hf:>7.3f} {tf:>7.3f} {tf/hf:>9.3f}x | "
              f"{hd:>9.3f} {td:>9.3f} {td/hd:>9.3f}x")
    if fwd_speedups:
        import statistics
        print('-' * len(hdr))
        print(f"fwd HK speedup vs Triton:   geomean {statistics.geometric_mean(fwd_speedups):.3f}x  "
              f"(min {min(fwd_speedups):.3f}, max {max(fwd_speedups):.3f})")
        print(f"dgrad HK speedup vs Triton: geomean {statistics.geometric_mean(dgrad_speedups):.3f}x  "
              f"(min {min(dgrad_speedups):.3f}, max {max(dgrad_speedups):.3f})")


if __name__ == "__main__":
    main()
