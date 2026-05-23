"""Ad-hoc perf check: HK fp8 grouped RCR — bn=0 vs bn=128.

Times the cpp op directly (bypasses autotune) for representative shapes.
Pass means: bn=128 must be either faster than bn=0, OR if slower, autotune
will pick bn=0 (so no regression possible). We just report the ratio.
"""
import os, sys, time, torch
# Load libs BEFORE `import primus_turbo` — the primus_turbo package may
# auto-load a stale _C that prevents subsequent reloads.
torch.ops.load_library(os.path.abspath("primus_turbo/lib/libprimus_turbo_kernels.so"))
_pyver = f"_C.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
torch.ops.load_library(os.path.abspath(f"primus_turbo/pytorch/{_pyver}"))
import primus_turbo  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

torch.manual_seed(0)
DEV = "cuda"
DTYPE_OUT = torch.bfloat16

# (B, M_per_group, N, K) — covers small/medium/large M_g, varied N
SHAPES = [
    (8,  512, 2048, 1536),   # NK0-ish, M_g aligned
    (8,  256, 2048, 1536),   # M_g == BLOCK_SIZE
    (8,  128, 2048, 1536),   # M_g == HB (partial-tile path)
    (16, 1024, 3072, 5120),  # bigger N (NK4)
    (32, 1024, 4096, 7168),  # gpt_oss-ish
    (8,  2048, 2048, 1536),
]

def bench_once(a, b, sa, sb, offs, gm, mg, xcds, bn, n_iter=50):
    op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_fp8
    # warmup
    for _ in range(5):
        op(a, b, sa, sb, offs, gm, mg, xcds, DTYPE_OUT, bn)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        op(a, b, sa, sb, offs, gm, mg, xcds, DTYPE_OUT, bn)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter  # ms

def main():
    print(f"{'shape (B,Mg,N,K)':<28} {'bn=0 ms':>10} {'bn=128 ms':>11} {'ratio':>7}")
    print('-'*60)
    for B, Mg, N, K in SHAPES:
        M = B * Mg
        a = (torch.randn(M, K, device=DEV, dtype=torch.bfloat16) / 8).to(float8_e4m3)
        b = (torch.randn(B, N, K, device=DEV, dtype=torch.bfloat16) / 8).to(float8_e4m3)
        sa = torch.ones(1, device=DEV)
        sb = torch.ones(1, device=DEV)
        offs = torch.arange(0, M+1, Mg, device=DEV, dtype=torch.int64)
        # gm=8, xcds=8 — common autotune winner
        t0 = bench_once(a, b, sa, sb, offs, 8, Mg, 8, 0)
        t1 = bench_once(a, b, sa, sb, offs, 8, Mg, 8, 128)
        print(f"({B:>2},{Mg:>4},{N:>4},{K:>4})  {t0:>10.3f} {t1:>11.3f} {t1/t0:>7.3f}")

if __name__ == "__main__":
    main()
