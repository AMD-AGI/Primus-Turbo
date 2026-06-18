#!/usr/bin/env python3
"""Fair kernel-only comparison: block-wise FP8 GEMM (forward NT).

Compares:
  * Primus-Turbo Triton backend  -> gemm_fp8_blockwise_triton_kernel (NT)
  * FlyDSL                        -> blockscale_preshuffle_gemm
  * (optional) aiter             -> gemm_a8w8_blockscale_bpreshuffle

All paths compute   C[M,N] = (A_fp8 . diag-blockscale) @ (B_fp8 . blockscale)^T
with 128x128 block scaling (A: [M, K/128], B: [ceil(N/128), K/128]); scales are
dequant multipliers. Inputs are byte-identical across backends; only the
backend-internal layout transforms (weight preshuffle / scale transpose) differ.

Timing is GEMM-kernel-only (scales precomputed, weight preshuffled once).
"""

import argparse
import os
import sys
import types

import torch
import torch.nn.functional as F

PT_ROOT = "/apps/tas/yaoc/agent_work/mi355x/Primus-Turbo"
FLY_ROOT = "/apps/tas/yaoc/agent_work/mi355x/FlyDSL"

# ── Import Primus-Turbo Triton blockwise kernel WITHOUT the C++ extension ──
# primus_turbo.pytorch.__init__ (and core.__init__) import the _C extension,
# whose libprimus_turbo_kernels.so is not built in this checkout. The Triton
# kernel itself is pure Python/Triton, so we stub those two packages to skip
# their __init__ and import the submodule files directly.
sys.path.insert(0, PT_ROOT)
import primus_turbo  # noqa: E402

for _sub in ["primus_turbo.pytorch", "primus_turbo.pytorch.core"]:
    _m = types.ModuleType(_sub)
    _m.__path__ = [os.path.join(PT_ROOT, *_sub.split("."))]
    sys.modules[_sub] = _m

from primus_turbo.triton.gemm.gemm_fp8_kernel import (  # noqa: E402
    gemm_fp8_blockwise_triton_kernel,
)

# ── Import FlyDSL kernel + helpers ──
# Prefer the freshly-built HEAD package over the older installed egg.
FLY_BUILD_PKG = "/root/flydsl-llvm/build-fly/python_packages"
if os.path.isdir(FLY_BUILD_PKG):
    sys.path.insert(0, FLY_ROOT)
    sys.path.insert(0, FLY_BUILD_PKG)
    _lib = os.path.join(FLY_BUILD_PKG, "flydsl", "_mlir", "_mlir_libs")
    os.environ["LD_LIBRARY_PATH"] = _lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
else:
    sys.path.insert(0, FLY_ROOT)
import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.blockscale_preshuffle_gemm import (  # noqa: E402
    compile_blockscale_preshuffle_gemm,
)
from tests.utils import shuffle_weight  # noqa: E402

try:
    import aiter  # noqa: E402
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight  # noqa: E402

    HAS_AITER = True
except Exception:
    HAS_AITER = False

ARCH = get_rocm_arch()
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz
BLOCK_N, BLOCK_K = 128, 128


# tile selector copied from FlyDSL test_blockscale_preshuffle_gemm.py
def select_tile_config(M, N, K, scale_block_k=128):
    candidates = [
        (16, 64, 256), (16, 128, 256), (32, 64, 128), (32, 64, 256),
        (32, 128, 128), (32, 128, 256), (64, 64, 128), (64, 64, 256),
        (64, 128, 128), (64, 128, 256), (64, 256, 128),
    ]

    def _valid(tm, tn, tk):
        return N % tn == 0 and K % tk == 0 and tk % scale_block_k == 0 and tm * tk // 256 >= 16

    valid = [(tm, tn, tk) for tm, tn, tk in candidates if _valid(tm, tn, tk)]
    if not valid:
        return (64, 128, 128)

    def _score(tm, tn, tk):
        s = 0
        total_blocks = ((M + tm - 1) // tm) * (N // tn)
        s += 15 if total_blocks >= 256 else (10 if total_blocks >= 128 else (5 if total_blocks >= 64 else 0))
        if M <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif M <= 128:
            s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
        elif M <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        if M <= 128:
            s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
        else:
            s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))
        s += 6 if tk == 128 else 3
        return s

    return max(valid, key=lambda t: _score(*t))


def run_torch_blockscale(x, weight, x_scale, w_scale, dtype=torch.float32):
    """fp32 reference (matches FlyDSL test)."""
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + BLOCK_N - 1) // BLOCK_N
    scale_k = (k + BLOCK_K - 1) // BLOCK_K
    x_f32 = x.to(torch.float32).view(m, k // BLOCK_K, BLOCK_K) * x_scale.unsqueeze(-1)
    x_f32 = x_f32.view(m, k)
    w_scale_expanded = (
        w_scale.view(-1, 1)
        .repeat(1, BLOCK_N * BLOCK_K)
        .view(scale_n, scale_k, BLOCK_N, BLOCK_K)
        .permute(0, 2, 1, 3)
        .reshape(scale_n * BLOCK_N, scale_k * BLOCK_K)
    )[:n, :k]
    weight_f32 = weight.to(torch.float32) * w_scale_expanded
    out = F.linear(x_f32, weight_f32)
    return out.to(dtype)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def bench_shape(M, N, K, iters, warmup):
    scale_k = (K + BLOCK_K - 1) // BLOCK_K
    scale_n = (N + BLOCK_N - 1) // BLOCK_N
    dev = torch.device("cuda")

    x = (torch.rand((M, K), dtype=torch.float16, device=dev) / 10).to(DTYPE_FP8)
    weight = (torch.rand((N, K), dtype=torch.float16, device=dev) / 10).to(DTYPE_FP8)
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device=dev)
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=dev)

    ref = run_torch_blockscale(x, weight, x_scale, w_scale)
    flops = 2 * M * N * K

    results = {}

    # ── Primus-Turbo Triton (NT forward): a=[M,K], b=[N,K], trans_b=True ──
    try:
        c_tri = gemm_fp8_blockwise_triton_kernel(
            x, x_scale, weight, w_scale,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
        )
        diff_tri = calc_diff(c_tri.float(), ref)
        us_tri = cuda_time(
            lambda: gemm_fp8_blockwise_triton_kernel(
                x, x_scale, weight, w_scale,
                trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            ),
            iters, warmup,
        )
        results["triton"] = (us_tri, flops / (us_tri / 1e6) / 1e12, diff_tri)
    except Exception as e:
        results["triton"] = ("ERR", str(e).splitlines()[0], None)

    # ── FlyDSL blockscale_preshuffle_gemm ──
    try:
        tm, tn, tk = select_tile_config(M, N, K, BLOCK_K)
        exe = compile_blockscale_preshuffle_gemm(
            M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
            scale_block_k=BLOCK_K, out_dtype="bf16", use_async_copy=False,
        )
        b_shuf = shuffle_weight(weight, layout=(16, 16))
        x_scale_t = x_scale.transpose(0, 1).contiguous().view(-1)
        w_scale_flat = w_scale.contiguous().view(-1)
        c_fly = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
        stream = torch.cuda.current_stream()
        compiled = flyc.compile(exe, c_fly, x, b_shuf, x_scale_t, w_scale_flat, M, N, stream)
        compiled(c_fly, x, b_shuf, x_scale_t, w_scale_flat, M, N, stream)
        torch.cuda.synchronize()
        diff_fly = calc_diff(c_fly.float(), ref)
        us_fly = cuda_time(
            lambda: compiled(c_fly, x, b_shuf, x_scale_t, w_scale_flat, M, N, stream),
            iters, warmup,
        )
        results["flydsl"] = (us_fly, flops / (us_fly / 1e6) / 1e12, diff_fly, (tm, tn, tk))
    except Exception as e:
        results["flydsl"] = ("ERR", str(e).splitlines()[0], None, None)

    # ── aiter (reference upper bound) ──
    if HAS_AITER:
        try:
            x_scale_t2d = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
            b_aiter = aiter_shuffle_weight(weight, layout=(16, 16))
            aiter.gemm_a8w8_blockscale_bpreshuffle(x, b_aiter, x_scale_t2d, w_scale, torch.bfloat16)
            us_ai = cuda_time(
                lambda: aiter.gemm_a8w8_blockscale_bpreshuffle(
                    x, b_aiter, x_scale_t2d, w_scale, torch.bfloat16
                ),
                iters, warmup,
            )
            results["aiter"] = (us_ai, flops / (us_ai / 1e6) / 1e12, None)
        except Exception as e:
            results["aiter"] = ("ERR", str(e).splitlines()[0], None)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--m-list", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    args = ap.parse_args()

    torch.set_default_device("cuda")
    # DeepSeek-V3 (N, K) GEMM shapes from FlyDSL blockscale test
    nk_shapes = [(7168, 2304), (2112, 7168), (3072, 1536)]

    print(f"Arch={ARCH}  fp8={DTYPE_FP8}  iters={args.iters} warmup={args.warmup}")
    hdr = f"{'M':>6} {'N':>6} {'K':>6} | {'Triton us':>10} {'Triton TF':>10} | {'FlyDSL us':>10} {'FlyDSL TF':>10} {'tile':>14} | {'Fly/Tri':>8}"
    if HAS_AITER:
        hdr += f" | {'aiter us':>9} {'aiter TF':>9}"
    print(hdr)
    print("-" * len(hdr))

    for N, K in nk_shapes:
        for M in args.m_list:
            r = bench_shape(M, N, K, args.iters, args.warmup)
            tri = r.get("triton")
            fly = r.get("flydsl")

            def fmt_us(v):
                return f"{v:10.2f}" if isinstance(v, (int, float)) else f"{str(v):>10}"

            def fmt_tf(v):
                return f"{v:10.1f}" if isinstance(v, (int, float)) else f"{'':>10}"

            tri_us = tri[0]
            fly_us = fly[0]
            speedup = ""
            if isinstance(tri_us, (int, float)) and isinstance(fly_us, (int, float)):
                speedup = f"{tri_us / fly_us:7.2f}x"
            tile_str = str(fly[3]) if fly[3] else ""
            line = (
                f"{M:>6} {N:>6} {K:>6} | {fmt_us(tri_us)} {fmt_tf(tri[1] if isinstance(tri_us,(int,float)) else None)} "
                f"| {fmt_us(fly_us)} {fmt_tf(fly[1] if isinstance(fly_us,(int,float)) else None)} {tile_str:>14} | {speedup:>8}"
            )
            if HAS_AITER and "aiter" in r:
                ai = r["aiter"]
                line += f" | {fmt_us(ai[0])} {fmt_tf(ai[1] if isinstance(ai[0],(int,float)) else None)}"
            print(line)
            # accuracy note
            tdiff = tri[2] if len(tri) > 2 else None
            fdiff = fly[2] if len(fly) > 2 else None
            notes = []
            if isinstance(tri_us, str):
                notes.append(f"triton ERR: {tri[1]}")
            elif tdiff is not None and tdiff > 1e-3:
                notes.append(f"triton diff={tdiff:.2e}")
            if isinstance(fly_us, str):
                notes.append(f"flydsl ERR: {fly[1]}")
            elif fdiff is not None and fdiff > 1e-3:
                notes.append(f"flydsl diff={fdiff:.2e}")
            if notes:
                print("        ! " + "; ".join(notes))


if __name__ == "__main__":
    main()
