###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# FlyDSL flash-attention forward perf + correctness (SBHD dense, gfx950/MI355X,
# causal, bf16, D=64). Same shape as bench_attention_turbo.py: SNR vs torch SDPA
# reference, pandas/tabulate table, CSV export; no external hardware reference.
#
#   HIP_VISIBLE_DEVICES=0 python3 bench_flydsl_flash_attn_fwd.py

import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import compute_snr, get_platform_info
from tabulate import tabulate
from torch.nn.attention import SDPBackend, sdpa_kernel

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    resolve_flash_attn_backend,
)

# Force the FlyDSL attention backend so every case exercises the flydsl kernels.
GlobalBackendManager.set_attn_backend(BackendType.FLYDSL)

DEV = "cuda"
DT = torch.bfloat16
D = 64
SNR_THRESHOLD = 40.0  # bf16

# (B, Hq, Hkv, Sq, Skv, window_left); window_left < 0 means full causal.
# B>=2: with B=1 the sbhd storage is byte-identical to bshd and _infer_qkv_format
# cannot tell them apart, so it would fall back off FlyDSL.
SQUARE = [(2, 128, 16, s, s, -1) for s in (2048, 4096, 8192, 16384)]
# Meta's rectangular configs (B=4), each run full-causal and with its SWA window.
_META = [
    (128, 16, 2048, 16384, 2048),
    (128, 16, 4096, 16384, 2048),
    (128, 16, 8192, 16384, 2048),
    (128, 16, 16384, 16384, 2048),
    (48, 6, 4096, 4096, 2047),
    (48, 6, 4096, 8192, 2047),
    (48, 6, 4096, 12288, 2047),
    (48, 6, 4096, 16384, 2047),
    (64, 8, 1024, 1024, 2047),
    (64, 8, 1024, 16384, 2047),
]
META = []
for hq, hkv, sq, skv, w in _META:
    META.append((4, hq, hkv, sq, skv, -1))
    META.append((4, hq, hkv, sq, skv, w))


def _bottom_right_mask(Sq, Skv, window_left, device):
    """Bool [Sq, Skv] mask: bottom-right causal, optionally left-windowed (SWA)."""
    i = torch.arange(Sq, device=device).view(Sq, 1)
    j = torch.arange(Skv, device=device).view(1, Skv)
    offset = Skv - Sq
    keep = j <= i + offset
    if window_left >= 0:
        keep &= j >= (i + offset - window_left)  # flydsl keeps the left-edge column
    return keep


def _attended_frac(Sq, Skv, window_left):
    """Fraction of the Sq x Skv score block that survives the mask (for TFLOPS)."""
    i = torch.arange(Sq)
    offset = Skv - Sq
    hi = torch.clamp(i + offset + 1, max=Skv)
    lo = torch.zeros_like(i) if window_left < 0 else torch.clamp(i + offset - window_left, min=0)
    return float((hi - lo).clamp(min=0).sum()) / (Sq * Skv)


def attention_ref(q, k, v, sm_scale, window_left):
    """Torch SDPA reference (bshd in/out). Bottom-right causal to match flydsl."""
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    n_rep = qt.shape[1] // kt.shape[1]
    mask = _bottom_right_mask(qt.shape[2], kt.shape[2], window_left, qt.device)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=mask, scale=sm_scale, enable_gqa=n_rep > 1
        )
    return o_ref.transpose(1, 2)


def profile_case(B, Hq, Hkv, Sq, Skv, window_left):
    """Forward perf + output SNR for one shape. Returns (fwd_ms, fwd_tf, check, backend)."""
    sm_scale = D ** (-0.5)
    window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
    torch.manual_seed(0)
    # FlyDSL dense is SBHD-native: store [S,B,H,D] and view as logical [B,S,H,D]
    # so _infer_qkv_format sees sbhd stride (else it falls back to aiter).
    q = torch.randn((Sq, B, Hq, D), device=DEV, dtype=DT).permute(1, 0, 2, 3)
    k = torch.randn((Skv, B, Hkv, D), device=DEV, dtype=DT).permute(1, 0, 2, 3)
    v = torch.randn((Skv, B, Hkv, D), device=DEV, dtype=DT).permute(1, 0, 2, 3)

    # Confirm the case actually routes to FlyDSL (not the aiter fallback).
    backend = resolve_flash_attn_backend(
        varlen=False, user_backend=BackendType.FLYDSL, q=q, k=k, v=v,
        dropout_p=0.0, softmax_scale=sm_scale, causal=True, window_size=window_size,
        bias=None, alibi_slopes=None, sink=None, qkv_format="sbhd",
    ).name

    fwd = lambda: turbo.ops.flash_attn_func(
        q, k, v, softmax_scale=sm_scale, causal=True, window_size=window_size
    )

    # Correctness: output SNR vs torch ref; may OOM on the largest shapes -> SKIP.
    check = "SKIP"
    try:
        with torch.no_grad():
            out = fwd()
            o_ref = attention_ref(q, k, v, sm_scale, window_left)
        snr = compute_snr(o_ref, out)
        check = "PASS" if snr > SNR_THRESHOLD else f"FAIL({snr:.0f})"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    for _ in range(10):
        fwd()
    torch.cuda.synchronize()
    fwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd}).timeit(50).mean * 1e3

    frac = _attended_frac(Sq, Skv, window_left)
    fwd_flops = 4 * B * Sq * Skv * Hq * D * frac
    return fwd_ms, fwd_flops / (fwd_ms * 1e-3) / 1e12, check, backend


def main(output_csv=None):
    platform, gpu_name = get_platform_info()
    print(f"platform={platform} gpu={gpu_name} backend=FLYDSL D={D}", flush=True)
    rows = []
    for tag, cases in (("square", SQUARE), ("meta", META)):
        for B, Hq, Hkv, Sq, Skv, w in cases:
            row = {
                "Set": tag, "B": B, "Hq": Hq, "Hkv": Hkv,
                "Sq": Sq, "Skv": Skv, "Win": w if w >= 0 else "full",
            }
            try:
                ms, tf, check, backend = profile_case(B, Hq, Hkv, Sq, Skv, w)
                row.update(
                    {"Backend": backend, "Check": check, "Fwd ms": f"{ms:.3f}", "Fwd TFLOPS": f"{tf:.1f}"}
                )
            except Exception as e:  # noqa: BLE001
                print(f"Failed {tag} {Hq}/{Hkv} {Sq}x{Skv} w={w}: {e}", flush=True)
                row.update({"Backend": "ERROR", "Check": "ERROR", "Fwd ms": "ERROR", "Fwd TFLOPS": "0"})
            print(row, flush=True)
            rows.append(row)

    results = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
    tf = results["Fwd TFLOPS"].astype(float)
    print(f"\nAverage Forward TFLOPS: {tf[tf > 0].mean():.1f}")

    filename = output_csv or f"flydsl_flash_attn_fwd_result_{datetime.now():%Y%m%d}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlyDSL flash-attention forward")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV filename.")
    args = parser.parse_args()
    main(output_csv=args.output)
