"""Capture the EXACT Triton kernel-config (blk_m, blk_n, blk_k, num_stages,
group_m, num_warps) for `grouped_gemm_fp8` forward + var-K dB on every
metric shape.

Why this script exists (R24): Round-23 noise-floor analysis recommended a
multi-round HK kernel surgery using ``BLOCK_K=64`` template specialization
based on the heuristic "shallow K means BK=128 wastes pipeline depth".
This script falsifies that recommendation by showing what BK Triton
actually picks for the same shapes. If Triton picks BK=128 for forward
(it does), then HK's BK=128 isn't the wedge — the wedge is elsewhere
(MFMA cell shape, prefetch pattern). If Triton picks BK=64 for var-K
(it does), then HK's var-K BK=128 single-template IS a candidate lever.

Usage:
  python3 scripts/_probe_triton_grouped_fp8_cfg.py

Output: a table of (shape, fwd_BK, fwd_stages, fwd_gm, dB_BK, dB_stages,
dB_gm) for every metric shape. Compare against
``/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``
for the corresponding HK constants (``K_BLOCK = 128`` for fwd + var-K).
"""
import torch

import primus_turbo  # noqa: F401  (registers torch.library custom_ops)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    _get_gg_fp8_tw_fwd_config,
    _get_gg_fp8_tw_vk_config,
)


def _device_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _fwd_cfg(B, M, N, K):
    """Forward: a [M_total, K]  @  b [G, N, K]^T  -> c [M_total, N] (NT)."""
    return _get_gg_fp8_tw_fwd_config(
        M, N, K,
        torch.bfloat16,
        torch.float8_e4m3fn, torch.float8_e4m3fn,
        True,                       # trans_b (forward NT)
        B, _device_sms(), B * M,
        K, K,                       # stride_ak / stride_bk for trans_b
    )


def _vk_cfg(B, M, N, K):
    """Backward dB: a^T @ grad_out -> grad_b[G, N, K]. Triton's var-K
    config keys on (OUT_M=N, OUT_N=K, avg_k=M)."""
    return _get_gg_fp8_tw_vk_config(
        N, K, M,
        torch.float8_e4m3fn, torch.float8_e4m3fn, B, _device_sms(),
    )


def _row(B, M, N, K, name):
    blk_m, blk_n, blk_k, group_m, _ca, _cb, num_stages, _cs, _gs = _fwd_cfg(B, M, N, K)
    fwd = f"BM={blk_m:>3} BN={blk_n:>3} BK={blk_k:>3} stages={num_stages} gm={group_m}"
    blk_m, blk_n, blk_k, group_m, _ca, _cb, num_stages, _cs = _vk_cfg(B, M, N, K)
    db = f"BM={blk_m:>3} BN={blk_n:>3} BK={blk_k:>3} stages={num_stages} gm={group_m}"
    print(f"  {name:<48}  fwd: {fwd} | dB(varK): {db}")


# 24 metric shapes (matches scripts/_metric_grouped_fused_wall.py suite).
SHAPES = [
    (16, 2048, 7168, 4096, "DSV3-GateUP-B16-M2048"),
    (16, 4096, 7168, 4096, "DSV3-GateUP-B16-M4096"),
    (32, 2048, 7168, 4096, "DSV3-GateUP-B32-M2048"),
    (32, 4096, 7168, 4096, "DSV3-GateUP-B32-M4096"),
    (16, 2048, 4096, 7168, "DSV3-Down-B16-M2048"),
    (16, 4096, 4096, 7168, "DSV3-Down-B16-M4096"),
    (32, 2048, 4096, 7168, "DSV3-Down-B32-M2048"),
    (32, 4096, 4096, 7168, "DSV3-Down-B32-M4096"),
    (4,  2048, 5760, 2880, "gpt_oss-GateUP-B4-M2048"),
    (4,  4096, 5760, 2880, "gpt_oss-GateUP-B4-M4096"),
    (32, 2048, 5760, 2880, "gpt_oss-GateUP-B32-M2048"),
    (32, 4096, 5760, 2880, "gpt_oss-GateUP-B32-M4096"),
    (4,  2048, 2880, 2880, "gpt_oss-Down-B4-M2048"),
    (4,  4096, 2880, 2880, "gpt_oss-Down-B4-M4096"),
    (32, 2048, 2880, 2880, "gpt_oss-Down-B32-M2048"),
    (32, 4096, 2880, 2880, "gpt_oss-Down-B32-M4096"),
    (16, 2048, 1536, 4096, "Qwen3-GateUP-B16-M2048"),
    (16, 4096, 1536, 4096, "Qwen3-GateUP-B16-M4096"),
    (32, 2048, 1536, 4096, "Qwen3-GateUP-B32-M2048"),
    (32, 4096, 1536, 4096, "Qwen3-GateUP-B32-M4096"),
    (16, 2048, 4096, 1536, "Qwen3-Down-B16-M2048"),
    (16, 4096, 4096, 1536, "Qwen3-Down-B16-M4096"),
    (32, 2048, 4096, 1536, "Qwen3-Down-B32-M2048"),
    (32, 4096, 4096, 1536, "Qwen3-Down-B32-M4096"),
]


if __name__ == "__main__":
    print(f"# device SMS = {_device_sms()}")
    print()
    print("=== Triton FP8 grouped FORWARD (NT) + VAR-K BACKWARD dB (TN) configs ===")
    print(f"# HK comparison: K_BLOCK = 128 for both fwd + var-K (single-template, see")
    print(f"# /workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:12).")
    print()
    for s in SHAPES:
        _row(*s)
