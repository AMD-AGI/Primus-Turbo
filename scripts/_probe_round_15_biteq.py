#!/usr/bin/env python3
"""R15 bit-equivalence verify: chunk_size=24 vs default for GateUP-B4-M2048
dgrad-via-H4."""
import os, sys
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
import _metric_hk_ratio as hk_ratio

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _patch_rcr(chunk_size):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def run(B, M, N, K, cs):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    orig = _patch_rcr(cs)
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        out = grouped_gemm_fp8_impl(
            g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )
    _restore(orig)
    return out.clone()


for seed in (42, 137, 2024, 99, 100):
    torch.manual_seed(seed)
    out_def = run(4, 2048, 5760, 2880, 0)  # default → cs=64
    torch.manual_seed(seed)
    out_24 = run(4, 2048, 5760, 2880, 24)
    diff = (out_def - out_24).abs().max().item()
    eq = torch.equal(out_def, out_24)
    print(f"  seed={seed}: max_abs_diff={diff:.6e}, bit_eq={eq}")
