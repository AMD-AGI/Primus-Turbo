###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU grouped-GEMM compute baseline: bf16 vs per-tensor fp8 vs mxfp8 (FlyDSL).

Directly times the FlyDSL grouped-GEMM kernels (NT forward, out = a @ b^T) on the
same shape so we can see how much fp8 compute headroom exists and how much the mxfp8
per-1x32 E8M0 block scaling costs vs plain per-tensor fp8:

  * bf16        -- turbo.ops.grouped_gemm (bf16 MFMA)                     [ceiling ref]
  * fp8 (tw)    -- grouped_gemm_fp8_tensorwise_flydsl_kernel (per-tensor scalar scale)
  * mxfp8       -- grouped_gemm_mxfp8_flydsl_kernel (per-1x32 E8M0 block scale)

Shapes default to the DeepSeek-V3 MoE L1 (N=2I=4096, K=H=7168) and L2 (N=H=7168,
K=I=2048) with G=32 experts/rank (EP8), swept over tokens-per-expert. Reports
ms / TFLOPS / SNR-vs-bf16 for each.

Run (1 GPU):
  PYTHONPATH=<repo>:<repo>/benchmark/ops:<repo>/benchmark/ops/training \
      python benchmark/ops/bench_grouped_gemm_mxfp8.py
"""

import argparse

import torch

from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    grouped_gemm_fp8_tensorwise_flydsl_kernel,
)

try:
    import primus_turbo.pytorch as turbo
    from primus_turbo.flydsl.mega.fp8.quant import quantize_rowwise_mxfp8

    _HAS_TURBO_PYTORCH = True
except Exception:  # turbo _C (deep_ep) unavailable -> torch bf16 ref + pure-torch mxfp8 quant
    turbo = None
    _HAS_TURBO_PYTORCH = False

    def quantize_rowwise_mxfp8(x, block: int = 32):
        M, K = x.shape
        xf = x.float().reshape(M, K // block, block)
        amax = xf.abs().amax(-1).clamp(min=1e-30)
        e = (torch.floor(torch.log2(amax)) - 8.0).clamp(-127, 127)
        q = (xf / torch.exp2(e).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        e8m0 = (e + 127).to(torch.uint8)
        return q.reshape(M, K).contiguous(), e8m0.reshape(M, K // block).contiguous()

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


def profile(G, M_per, N, K, dtype=torch.bfloat16):
    dev = "cuda"
    M = G * M_per
    a = torch.randn(M, K, device=dev, dtype=dtype) / (K**0.25)
    b = torch.randn(G, N, K, device=dev, dtype=dtype) / (K**0.25)
    group_lens = torch.full((G,), M_per, device=dev, dtype=torch.int64)
    group_offs = torch.zeros(G + 1, device=dev, dtype=torch.int64)
    group_offs[1:] = group_lens.cumsum(0)
    flops = 2.0 * M * N * K

    # bf16 reference (turbo grouped_gemm) — the compute ceiling
    out_bf16 = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)
    t_bf16 = _bench(lambda: turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True))

    # per-tensor fp8 (FlyDSL tensorwise)
    aq_tw, a_sc = _quant_tw(a)
    bq_tw = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    b_scs = []
    for g in range(G):
        bq_tw[g], sc = _quant_tw(b[g])
        b_scs.append(sc)
    # per-tensor kernel takes ONE scalar b_scale; use a single global b scale for timing
    bq_tw2, b_sc = _quant_tw(b)
    out_tw = grouped_gemm_fp8_tensorwise_flydsl_kernel(
        aq_tw, bq_tw2, a_sc, b_sc, group_offs, trans_b=True, out_dtype=dtype
    )
    t_tw = _bench(
        lambda: grouped_gemm_fp8_tensorwise_flydsl_kernel(
            aq_tw, bq_tw2, a_sc, b_sc, group_offs, trans_b=True, out_dtype=dtype
        )
    )

    # mxfp8 (per-1x32 E8M0 block scale)
    aq_mx, a_smx = quantize_rowwise_mxfp8(a)
    bq_list, bs_list = zip(*(quantize_rowwise_mxfp8(b[g]) for g in range(G)))
    bq_mx, bs_mx = torch.stack(bq_list, 0), torch.stack(bs_list, 0)
    out_mx = grouped_gemm_mxfp8_flydsl_kernel(
        aq_mx, a_smx, bq_mx, bs_mx, group_offs, out_dtype=dtype
    )
    t_mx = _bench(
        lambda: grouped_gemm_mxfp8_flydsl_kernel(
            aq_mx, a_smx, bq_mx, bs_mx, group_offs, out_dtype=dtype
        )
    )

    tf = lambda ms: flops / (ms * 1e-3) / 1e12
    return {
        "M": M, "N": N, "K": K, "G": G, "M_per": M_per,
        "bf16_ms": t_bf16, "bf16_tf": tf(t_bf16),
        "tw_ms": t_tw, "tw_tf": tf(t_tw), "tw_snr": _snr(out_bf16, out_tw),
        "mx_ms": t_mx, "mx_tf": tf(t_mx), "mx_snr": _snr(out_bf16, out_mx),
    }


def main():
    ap = argparse.ArgumentParser(description="grouped GEMM bf16 vs fp8-tw vs mxfp8 (FlyDSL)")
    ap.add_argument("--experts", type=int, default=32, help="G (experts per rank at EP8)")
    ap.add_argument("--tokens-per-expert", type=int, nargs="+", default=[256, 512, 2048])
    ap.add_argument("--stage", choices=["l1", "l2", "both"], default="both")
    args = ap.parse_args()

    # DSv3: L1 up/gate N=2I=4096 K=H=7168 ; L2 down N=H=7168 K=I=2048
    stages = {"l1": (4096, 7168), "l2": (7168, 2048)}
    todo = ["l1", "l2"] if args.stage == "both" else [args.stage]

    print(f"\n{'='*96}")
    print(f"grouped GEMM (NT) FlyDSL: bf16 vs per-tensor fp8 vs mxfp8   G={args.experts}   "
          f"{torch.cuda.get_device_name(0)}")
    print(f"{'='*96}")
    header = (f"{'stage':<5} {'M_per':>6} {'M':>7} {'N':>6} {'K':>6} | "
             f"{'bf16 ms/TF':>16} | {'fp8-tw ms/TF/SNR':>24} | {'mxfp8 ms/TF/SNR':>24} | {'mx/tw':>6}")
    print(header)
    print("-" * len(header))
    for stage in todo:
        N, K = stages[stage]
        for M_per in args.tokens_per_expert:
            r = profile(args.experts, M_per, N, K)
            print(
                f"{stage:<5} {M_per:>6} {r['M']:>7} {N:>6} {K:>6} | "
                f"{r['bf16_ms']:>7.3f}/{r['bf16_tf']:>7.1f} | "
                f"{r['tw_ms']:>7.3f}/{r['tw_tf']:>6.1f}/{r['tw_snr']:>5.1f} | "
                f"{r['mx_ms']:>7.3f}/{r['mx_tf']:>6.1f}/{r['mx_snr']:>5.1f} | "
                f"{r['tw_ms'] / r['mx_ms']:>5.2f}x"
            )


if __name__ == "__main__":
    main()
