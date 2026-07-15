###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped-GEMM compute baseline targeting the PRODUCTION NT kernel file.

Same shapes/timing as bench_grouped_gemm_mxfp8.py, but the mxfp8 leg calls the
kernel from ``primus_turbo/flydsl/grouped_gemm/mxfp8_grouped_kernel.py`` (the file
wired into grouped_gemm_fp8_impl.py), which has the (N, K) explicit signature and
the _GG_SCALE_PACK=4 packed-scale path. Compares:

  * bf16        -- turbo.ops.grouped_gemm (bf16 MFMA)                     [ceiling ref]
  * fp8 (tw)    -- grouped_gemm_fp8_tensorwise_flydsl_kernel (per-tensor scalar scale)
  * mxfp8       -- grouped_gemm_mxfp8_flydsl_kernel (grouped_gemm/, per-1x32 E8M0)

Run (1 GPU):
  LD_LIBRARY_PATH=<venv>/primus_turbo/lib:$LD_LIBRARY_PATH \
  PYTHONPATH=<repo> python benchmark/ops/bench_grouped_gemm_mxfp8_prod.py
"""

import argparse

import torch

from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    grouped_gemm_fp8_tensorwise_flydsl_kernel,
)
from primus_turbo.flydsl.mega.gemm_bf16_kernel import gemm_bf16_flydsl_kernel

import primus_turbo.pytorch as turbo
from primus_turbo.flydsl.mega.fp8.quant import quantize_rowwise_mxfp8

_L2_BUF = None


def _l2_flush():
    global _L2_BUF
    if _L2_BUF is None:
        _L2_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_BUF.zero_()


def _bench(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    _l2_flush()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        es[i].record()
        fn()
        ee[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(es, ee))
    return ts[len(ts) // 2]  # median ms


def _snr(ref, act):
    ref, act = ref.float(), act.float()
    noise = ref - act
    return 10.0 * torch.log10((ref * ref).mean() / (noise * noise).mean() + 1e-20).item()


def _quant_tw(x):
    """Per-tensor fp8 E4M3: (fp8 tensor, scalar amax/448 scale). scale = amax/448."""
    amax = x.abs().amax().clamp(min=1e-8)
    scale = (amax / 448.0).float()
    q = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def _flydsl_bf16_grouped(a, b, seg, out):
    """flydsl bf16 grouped NT via per-group dense NT calls (no grouped NT wrapper exists).
    Same total FLOPs / grouping as the mxfp8 kernel; pays G kernel-launch overheads.
    ``seg`` is a precomputed host list of (start, end) row ranges (no D2H in the loop)."""
    for g, (s, e) in enumerate(seg):
        if e > s:
            out[s:e] = gemm_bf16_flydsl_kernel(a[s:e], b[g], trans_b=True, out_dtype=out.dtype)
    return out


def profile(G, M_per, N, K, dtype=torch.bfloat16):
    dev = "cuda"
    M = G * M_per
    a = torch.randn(M, K, device=dev, dtype=dtype) / (K**0.25)
    b = torch.randn(G, N, K, device=dev, dtype=dtype) / (K**0.25)
    group_lens = torch.full((G,), M_per, device=dev, dtype=torch.int64)
    group_offs = torch.zeros(G + 1, device=dev, dtype=torch.int64)
    group_offs[1:] = group_lens.cumsum(0)
    flops = 2.0 * M * N * K

    # bf16 reference (turbo grouped_gemm = Triton backend) — the compute ceiling
    out_bf16 = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)
    t_bf16 = _bench(lambda: turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True))

    # flydsl bf16 (per-group dense NT) — same-family grouped-equivalent baseline
    # (no grouped bf16 NT flydsl kernel exists -> G serial dense launches).
    seg = [(g * M_per, (g + 1) * M_per) for g in range(G)]
    out_fb = torch.empty_like(out_bf16)
    _flydsl_bf16_grouped(a, b, seg, out_fb)
    t_fb = _bench(lambda: _flydsl_bf16_grouped(a, b, seg, out_fb))

    # flydsl bf16 DENSE single-launch (a[M,K] @ b[0][N,K]^T) — the flydsl bf16 tile
    # CEILING at this shape without the G-launch / underutilization penalty. Same
    # M*N*K FLOPs; different math (single B) so SNR is not meaningful here.
    b0 = b[0].contiguous()
    _ = gemm_bf16_flydsl_kernel(a, b0, trans_b=True, out_dtype=dtype)
    t_fbd = _bench(lambda: gemm_bf16_flydsl_kernel(a, b0, trans_b=True, out_dtype=dtype))

    # per-tensor fp8 (FlyDSL tensorwise)
    aq_tw, a_sc = _quant_tw(a)
    bq_tw2, b_sc = _quant_tw(b)
    out_tw = grouped_gemm_fp8_tensorwise_flydsl_kernel(
        aq_tw, bq_tw2, a_sc, b_sc, group_offs, trans_b=True, out_dtype=dtype
    )
    t_tw = _bench(
        lambda: grouped_gemm_fp8_tensorwise_flydsl_kernel(
            aq_tw, bq_tw2, a_sc, b_sc, group_offs, trans_b=True, out_dtype=dtype
        )
    )

    # mxfp8 (per-1x32 E8M0 block scale) — PRODUCTION grouped_gemm/ kernel, (N,K) signature
    aq_mx, a_smx = quantize_rowwise_mxfp8(a)
    bq_list, bs_list = zip(*(quantize_rowwise_mxfp8(b[g]) for g in range(G)))
    bq_mx, bs_mx = torch.stack(bq_list, 0), torch.stack(bs_list, 0)
    out_mx = grouped_gemm_mxfp8_flydsl_kernel(
        aq_mx, a_smx, bq_mx, bs_mx, group_offs, N, K, out_dtype=dtype
    )
    t_mx = _bench(
        lambda: grouped_gemm_mxfp8_flydsl_kernel(
            aq_mx, a_smx, bq_mx, bs_mx, group_offs, N, K, out_dtype=dtype
        )
    )

    tf = lambda ms: flops / (ms * 1e-3) / 1e12
    return {
        "M": M, "N": N, "K": K, "G": G, "M_per": M_per,
        "bf16_ms": t_bf16, "bf16_tf": tf(t_bf16),
        "fb_ms": t_fb, "fb_tf": tf(t_fb), "fb_snr": _snr(out_bf16, out_fb),
        "fbd_ms": t_fbd, "fbd_tf": tf(t_fbd),
        "tw_ms": t_tw, "tw_tf": tf(t_tw), "tw_snr": _snr(out_bf16, out_tw),
        "mx_ms": t_mx, "mx_tf": tf(t_mx), "mx_snr": _snr(out_bf16, out_mx),
    }


def main():
    ap = argparse.ArgumentParser(description="grouped GEMM bf16 vs fp8-tw vs mxfp8 (prod grouped_gemm/)")
    ap.add_argument("--experts", type=int, default=32, help="G (experts per rank at EP8)")
    ap.add_argument("--tokens-per-expert", type=int, nargs="+", default=[256, 512, 2048])
    ap.add_argument("--stage", choices=["l1", "l2", "dgrad", "both", "all"], default="both")
    args = ap.parse_args()

    # DSv3: L1 up/gate N=2I=4096 K=H=7168 ; L2 down N=H=7168 K=I=2048
    # dgrad = fc2 backward dgrad (NT via static w2 transpose): grad_swiglu[P,I] = dL2Y[P,H] @ w2T[G,I,H]^T
    #   -> N=I=2048, K=H=7168 (grouped mxfp8 NT).
    stages = {"l1": (4096, 7168), "l2": (7168, 2048), "dgrad": (2048, 7168)}
    todo = {"both": ["l1", "l2"], "all": ["l1", "l2", "dgrad"]}.get(args.stage, [args.stage])

    print(f"\n{'='*96}")
    print(f"grouped GEMM (NT) FlyDSL [grouped_gemm/mxfp8_grouped_kernel.py]: bf16 vs fp8-tw vs mxfp8   "
          f"G={args.experts}   {torch.cuda.get_device_name(0)}")
    print(f"{'='*96}")
    header = (f"{'stage':<5} {'M_per':>6} {'M':>7} {'N':>6} {'K':>6} | "
             f"{'triton-bf16 ms/TF':>18} | {'flydsl-bf16 ms/TF':>18} | "
             f"{'fp8-tw ms/TF/SNR':>24} | {'mxfp8 ms/TF/SNR':>24} | {'mx/fbf16':>8}")
    print(header)
    print("-" * len(header))
    for stage in todo:
        N, K = stages[stage]
        for M_per in args.tokens_per_expert:
            r = profile(args.experts, M_per, N, K)
            print(
                f"{stage:<5} {M_per:>6} {r['M']:>7} {N:>6} {K:>6} | "
                f"{r['bf16_ms']:>8.3f}/{r['bf16_tf']:>7.1f} | "
                f"{r['fb_ms']:>8.3f}/{r['fb_tf']:>7.1f} | "
                f"{r['tw_ms']:>7.3f}/{r['tw_tf']:>6.1f}/{r['tw_snr']:>5.1f} | "
                f"{r['mx_ms']:>7.3f}/{r['mx_tf']:>6.1f}/{r['mx_snr']:>5.1f} | "
                f"{r['fb_ms'] / r['mx_ms']:>6.2f}x"
            )


if __name__ == "__main__":
    main()
