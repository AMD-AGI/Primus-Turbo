###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU DENSE GEMM baseline: bf16 vs per-tensor fp8 vs the standard mxfp8 (FlyDSL).

Times the standard dense mxfp8 kernel (``gemm_mxfp8_flydsl_kernel``, per-1x32 E8M0
block scale, with its per-shape autotune) against:

  * bf16          -- torch.matmul (hipBLASLt)                              [ref]
  * fp8 per-tensor -- torch._scaled_mm (hipBLASLt)      [per-tensor fp8 hardware ceiling]
  * mxfp8         -- gemm_mxfp8_flydsl_kernel (FlyDSL block-scaled)

NT (out = a @ b^T). This isolates whether the per-1x32 block scaling itself costs the
fp8 advantage on the DENSE path (vs the grouped path). Same FLOPs (2*M*N*K) for all.

Run (1 GPU):
  PYTHONPATH=<repo> python benchmark/ops/bench_gemm_mxfp8_dense.py
"""

import argparse

import torch

from primus_turbo.flydsl.gemm.mxfp8_gemm_kernel import gemm_mxfp8_flydsl_kernel

try:
    from primus_turbo.flydsl.mega.fp8.quant import quantize_rowwise_mxfp8
except Exception:  # turbo _C (deep_ep) not available -> pure-torch mxfp8 rowwise quant

    def quantize_rowwise_mxfp8(x, block: int = 32):
        """Pure-torch rowwise MXFP8 (per-1x32 E8M0) quant: returns (fp8 [M,K], e8m0 uint8
        [M,K//32]) with dequant = fp8 * 2^(e8m0-127). Matches the kernel's raw-scale ABI."""
        M, K = x.shape
        xf = x.float().reshape(M, K // block, block)
        amax = xf.abs().amax(-1).clamp(min=1e-30)
        e = (torch.floor(torch.log2(amax)) - 8.0).clamp(-127, 127)  # scale 2^e -> block ~2^8
        q = (xf / torch.exp2(e).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        e8m0 = (e + 127).to(torch.uint8)
        return q.reshape(M, K).contiguous(), e8m0.reshape(M, K // block).contiguous()

_L2 = None


def _l2_flush():
    global _L2
    if _L2 is None:
        _L2 = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2.zero_()


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
    return sorted(s.elapsed_time(e) for s, e in zip(es, ee))[iters // 2]


def _snr(ref, act):
    ref, act = ref.float(), act.float()
    n = ref - act
    return 10.0 * torch.log10((ref * ref).mean() / (n * n).mean() + 1e-20).item()


def _quant_tw(x):
    amax = x.abs().amax().clamp(min=1e-8)
    scale = (amax / 448.0).float().reshape(1)
    q = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def profile(M, N, K):
    dev = "cuda"
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) / (K**0.25)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / (K**0.25)
    flops = 2.0 * M * N * K

    out_bf16 = a @ b.T
    t_bf16 = _bench(lambda: a @ b.T)

    # per-tensor fp8 via torch._scaled_mm (hipBLASLt) — the per-tensor fp8 HW ceiling
    aq, a_sc = _quant_tw(a)
    bq, b_sc = _quant_tw(b)
    try:
        out_tw = torch._scaled_mm(aq, bq.T, scale_a=a_sc, scale_b=b_sc, out_dtype=torch.bfloat16)
        t_tw = _bench(lambda: torch._scaled_mm(aq, bq.T, scale_a=a_sc, scale_b=b_sc, out_dtype=torch.bfloat16))
        tw_snr = _snr(out_bf16, out_tw)
    except Exception as e:
        t_tw, tw_snr = float("nan"), float("nan")
        print(f"  (per-tensor _scaled_mm failed: {e})")

    # standard mxfp8 (FlyDSL block-scaled, autotuned)
    aq_mx, a_smx = quantize_rowwise_mxfp8(a)
    bq_mx, b_smx = quantize_rowwise_mxfp8(b)
    out_mx = gemm_mxfp8_flydsl_kernel(aq_mx, a_smx, bq_mx, b_smx, trans_b=True, out_dtype=torch.bfloat16)
    t_mx = _bench(
        lambda: gemm_mxfp8_flydsl_kernel(aq_mx, a_smx, bq_mx, b_smx, trans_b=True, out_dtype=torch.bfloat16)
    )
    mx_snr = _snr(out_bf16, out_mx)

    tf = lambda ms: flops / (ms * 1e-3) / 1e12
    return {
        "bf16_ms": t_bf16, "bf16_tf": tf(t_bf16),
        "tw_ms": t_tw, "tw_tf": tf(t_tw) if t_tw == t_tw else float("nan"), "tw_snr": tw_snr,
        "mx_ms": t_mx, "mx_tf": tf(t_mx), "mx_snr": mx_snr,
    }


def main():
    ap = argparse.ArgumentParser(description="dense GEMM bf16 vs per-tensor fp8 vs mxfp8 (FlyDSL)")
    ap.add_argument("--shapes", type=str, default=None,
                    help="semicolon list of M,N,K (default: DSv3 L1/L2 + square)")
    args = ap.parse_args()

    if args.shapes:
        shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes.split(";")]
    else:
        shapes = [
            (8192, 4096, 7168), (16384, 4096, 7168), (65536, 4096, 7168),  # DSv3 L1
            (8192, 7168, 2048), (16384, 7168, 2048), (65536, 7168, 2048),  # DSv3 L2
            (8192, 8192, 8192),  # square
        ]

    print(f"\n{'='*94}")
    print(f"DENSE GEMM (NT): bf16 vs per-tensor fp8 (_scaled_mm) vs mxfp8 (FlyDSL)   "
          f"{torch.cuda.get_device_name(0)}")
    print(f"{'='*94}")
    hdr = (f"{'M':>7} {'N':>6} {'K':>6} | {'bf16 ms/TF':>16} | "
           f"{'fp8-tw ms/TF/SNR':>24} | {'mxfp8 ms/TF/SNR':>24} | {'mx/tw':>6}")
    print(hdr)
    print("-" * len(hdr))
    for M, N, K in shapes:
        r = profile(M, N, K)
        mxtw = (r["tw_ms"] / r["mx_ms"]) if r["tw_ms"] == r["tw_ms"] else float("nan")
        print(
            f"{M:>7} {N:>6} {K:>6} | {r['bf16_ms']:>7.3f}/{r['bf16_tf']:>7.1f} | "
            f"{r['tw_ms']:>7.3f}/{r['tw_tf']:>6.1f}/{r['tw_snr']:>5.1f} | "
            f"{r['mx_ms']:>7.3f}/{r['mx_tf']:>6.1f}/{r['mx_snr']:>5.1f} | {mxtw:>5.2f}x"
        )


if __name__ == "__main__":
    main()
