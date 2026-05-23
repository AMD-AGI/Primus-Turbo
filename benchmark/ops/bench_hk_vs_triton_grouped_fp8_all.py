"""HipKitten vs Triton — fp8 tensorwise grouped GEMM fwd / dgrad / wgrad separate.
gpt_oss / DeepSeek-V3 / Qwen3-235B-A22B MoE shapes.

dgrad = backward with A.requires_grad=True only (b detached).
wgrad = backward with B.requires_grad=True only (a detached).
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
B_VALUES = [4, 16]; M_VALUES = [2048, 4096]
CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)

def make_shapes():
    out = []
    for model, p in MODELS.items():
        N_up=p["gated"]*p["inter"]; K_up=p["hidden"]
        N_down=p["hidden"]; K_down=p["inter"]
        for B in B_VALUES:
            for M in M_VALUES:
                out.append((model,"up",B,M,N_up,K_up))
                out.append((model,"down",B,M,N_down,K_down))
    return out

def time_loop(fn, n_warmup=10, n_iter=50):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter

def bench_three(B, Mg, N, K, backend):
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    M = B * Mg
    group_lens = torch.full((B,), Mg, dtype=torch.int64, device=DEV)

    # FWD: no grad
    a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)
    t_fwd = time_loop(lambda: grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=CFG))

    # DGRAD: a.requires_grad=True only
    a_g = torch.randn((M, K), dtype=torch.bfloat16, device=DEV, requires_grad=True)
    b_d = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV)
    out0 = grouped_gemm_fp8(a_g, b_d, group_lens, trans_b=True, config=CFG)
    grad_out = torch.randn_like(out0)
    def step_dgrad():
        out = grouped_gemm_fp8(a_g, b_d, group_lens, trans_b=True, config=CFG)
        out.backward(grad_out, retain_graph=False); a_g.grad = None
    t_fb_d = time_loop(step_dgrad)
    t_dgrad = max(t_fb_d - t_fwd, 0.001)

    # WGRAD: b.requires_grad=True only
    a_d2 = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
    b_g  = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV, requires_grad=True)
    out0 = grouped_gemm_fp8(a_d2, b_g, group_lens, trans_b=True, config=CFG)
    grad_out2 = torch.randn_like(out0)
    def step_wgrad():
        out = grouped_gemm_fp8(a_d2, b_g, group_lens, trans_b=True, config=CFG)
        out.backward(grad_out2, retain_graph=False); b_g.grad = None
    t_fb_w = time_loop(step_wgrad)
    t_wgrad = max(t_fb_w - t_fwd, 0.001)

    return t_fwd, t_dgrad, t_wgrad

def main():
    hdr = (f"{'model':<10} {'op':<5} {'B':>3} {'M':>5} {'N':>5} {'K':>5} | "
           f"{'fwd HK':>7} {'fwd TR':>7} {'fwd':>7} | "
           f"{'dg HK':>7} {'dg TR':>7} {'dg':>7} | "
           f"{'wg HK':>7} {'wg TR':>7} {'wg':>7}")
    print(hdr); print('-' * len(hdr))
    fs, ds, ws = [], [], []
    for model, op, B, Mg, N, K in make_shapes():
        try:
            hf, hd, hw = bench_three(B, Mg, N, K, BackendType.HIPKITTEN)
            tf, td, tw = bench_three(B, Mg, N, K, BackendType.TRITON)
        except Exception as ex:
            print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5}  failed: {ex}")
            continue
        fs.append(tf/hf); ds.append(td/hd); ws.append(tw/hw)
        print(f"{model:<10} {op:<5} {B:>3} {Mg:>5} {N:>5} {K:>5} | "
              f"{hf:>7.3f} {tf:>7.3f} {tf/hf:>6.3f}x | "
              f"{hd:>7.3f} {td:>7.3f} {td/hd:>6.3f}x | "
              f"{hw:>7.3f} {tw:>7.3f} {tw/hw:>6.3f}x")
    if fs:
        import statistics
        gm = statistics.geometric_mean
        print('-' * len(hdr))
        print(f"fwd   HK vs Triton: geomean {gm(fs):.3f}x  (min {min(fs):.3f}, max {max(fs):.3f})")
        print(f"dgrad HK vs Triton: geomean {gm(ds):.3f}x  (min {min(ds):.3f}, max {max(ds):.3f})")
        print(f"wgrad HK vs Triton: geomean {gm(ws):.3f}x  (min {min(ws):.3f}, max {max(ws):.3f})")

if __name__ == "__main__":
    main()
