###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# FlyDSL flash-attention varlen (THD, block-diagonal / document-packing) backward
# perf + correctness (gfx950/MI355X, causal, bf16, D=64, GQA G=8). The high-level
# dispatch only routes uniform cu_seqlens to FlyDSL, so ragged layouts are driven
# through the impl layer directly (which is what the sparse/ragged path exercises).
# Same framework as bench_attention_turbo.py: per-segment SDPA reference SNR,
# pandas/tabulate table, CSV export; no external hardware reference.
#
#   HIP_VISIBLE_DEVICES=0 python3 bench_flydsl_flash_attn_varlen_bwd.py

import argparse
import math
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import compute_snr, get_platform_info
from tabulate import tabulate
from torch.nn.attention import SDPBackend, sdpa_kernel

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_varlen_flydsl_backward_impl,
    flash_attn_varlen_flydsl_forward_impl,
)

DEV = "cuda"
DT = torch.bfloat16
D = 64
HQ, HKV = 64, 8  # GQA G=8 (flydsl requires G a power of two >= 8)
SNR_THRESHOLD = 40.0  # bf16

# (tag, segment layout): uniform is the zero-waste baseline; the rest are ragged
# document packing with increasing segment-length skew. Each cu_seqlens_q==_k.
CONFIGS = [
    ("uniform", [2048, 2048, 2048, 2048]),
    ("mild", [1024, 2048, 4096, 1024]),
    ("skew", [512, 2048, 1024, 4096]),
    ("longtail", [4096, 512, 256, 128]),
]
WINDOWS = [-1, 2048]  # full causal, then SWA left window


def _bottom_right_mask(Sq, Skv, window_left, device):
    """Bool [Sq, Skv] mask: bottom-right causal, optionally left-windowed (SWA)."""
    i = torch.arange(Sq, device=device).view(Sq, 1)
    j = torch.arange(Skv, device=device).view(1, Skv)
    offset = Skv - Sq
    keep = j <= i + offset
    if window_left >= 0:
        keep &= j >= (i + offset - window_left)  # flydsl keeps the left-edge column
    return keep


def _attended_frac(S, window_left):
    """Fraction of the SxS score block a query row attends (square segment)."""
    i = torch.arange(S)
    hi = i + 1
    lo = torch.zeros_like(i) if window_left < 0 else torch.clamp(i - window_left, min=0)
    return float((hi - lo).clamp(min=0).sum()) / (S * S)


def _build_cu(segs):
    cu = torch.zeros(len(segs) + 1, device=DEV, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(segs, device=DEV, dtype=torch.int32), 0)
    return cu, max(segs), int(cu[-1].item())


def _seg_ref(qs, ks, vs, dos, sm_scale, window_left):
    """Per-segment torch SDPA reference; returns (dq, dk, dv) for one square segment."""
    s = qs.shape[0]
    mask = _bottom_right_mask(s, s, window_left, qs.device)
    qh, kh, vh = (t.unsqueeze(0).transpose(1, 2) for t in (qs, ks, vs))
    n_rep = HQ // HKV
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        o = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=mask, scale=sm_scale, enable_gqa=n_rep > 1
        )
    o = o.transpose(1, 2).squeeze(0)
    o.backward(dos)
    return qs.grad, ks.grad, vs.grad


def profile_case(segs, window_left):
    """Backward perf + grad SNR for one ragged layout. Returns (ms, tf, check, backend)."""
    sm_scale = D ** (-0.5)
    ws = (window_left, 0) if window_left >= 0 else (-1, -1)
    cu, maxs, total = _build_cu(segs)
    torch.manual_seed(0)
    q = torch.randn(total, HQ, D, device=DEV, dtype=DT)
    k = torch.randn(total, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(total, HKV, D, device=DEV, dtype=DT)
    do = torch.randn(total, HQ, D, device=DEV, dtype=DT)
    out, lse = flash_attn_varlen_flydsl_forward_impl(
        q, k, v, cu, cu, maxs, maxs, softmax_scale=sm_scale, causal=True, window_size=ws, return_lse=True
    )
    dq, dk, dv = flash_attn_varlen_flydsl_backward_impl(
        do, q, k, v, out, lse, cu, cu, maxs, maxs, softmax_scale=sm_scale, causal=True, window_size=ws
    )

    # Correctness: grad SNR vs per-segment torch ref; may OOM on wide segments -> SKIP.
    check = "SKIP"
    try:
        rdq, rdk, rdv = (torch.zeros_like(t) for t in (q, k, v))
        off = 0
        for s in segs:
            qs = q[off : off + s].detach().clone().requires_grad_()
            ks = k[off : off + s].detach().clone().requires_grad_()
            vs = v[off : off + s].detach().clone().requires_grad_()
            gq, gk, gv = _seg_ref(qs, ks, vs, do[off : off + s], sm_scale, window_left)
            rdq[off : off + s], rdk[off : off + s], rdv[off : off + s] = gq, gk, gv
            off += s
        snrs = [compute_snr(r, x) for r, x in ((rdq, dq), (rdk, dk), (rdv, dv))]
        check = "PASS" if all(s > SNR_THRESHOLD for s in snrs) else f"FAIL({min(snrs):.0f})"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    bwd = lambda: flash_attn_varlen_flydsl_backward_impl(
        do, q, k, v, out, lse, cu, cu, maxs, maxs, softmax_scale=sm_scale, causal=True, window_size=ws
    )
    for _ in range(10):
        bwd()
    torch.cuda.synchronize()
    ms = benchmark.Timer(stmt="fn()", globals={"fn": bwd}).timeit(50).mean * 1e3

    # block-diagonal: 5 backward GEMMs (2.5x fwd) at D wide, summed over segments.
    flop = sum(10.0 * HQ * s * s * D * _attended_frac(s, window_left) for s in segs)
    return ms, flop / (ms * 1e-3) / 1e12, check, "FLYDSL"


def main(output_csv=None):
    platform, gpu_name = get_platform_info()
    print(f"platform={platform} gpu={gpu_name} backend=FLYDSL D={D} HQ/HKV={HQ}/{HKV}", flush=True)
    rows = []
    for tag, segs in CONFIGS:
        for window_left in WINDOWS:
            row = {"Tag": tag, "Segs": str(segs), "Total": sum(segs), "Win": window_left if window_left >= 0 else "full"}
            try:
                ms, tf, check, backend = profile_case(segs, window_left)
                row.update({"Backend": backend, "Check": check, "Bwd ms": f"{ms:.3f}", "Bwd TFLOPS": f"{tf:.1f}"})
            except Exception as e:  # noqa: BLE001
                print(f"Failed {tag} w={window_left}: {e}", flush=True)
                row.update({"Backend": "ERROR", "Check": "ERROR", "Bwd ms": "ERROR", "Bwd TFLOPS": "0"})
            print(row, flush=True)
            rows.append(row)

    results = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
    print("\nnote: ragged tiles by max_seqlen -> skewed layouts pay early-exit waste;")
    print("TFLOPS is effective (block-diagonal attended-fraction), not peak utilization.")

    filename = output_csv or f"flydsl_flash_attn_varlen_bwd_result_{datetime.now():%Y%m%d}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlyDSL flash-attention varlen backward")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV filename.")
    args = parser.parse_args()
    main(output_csv=args.output)
