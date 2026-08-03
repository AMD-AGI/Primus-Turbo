"""Production gfx950 blockwise FP8 GEMM kernels and dispatch."""

from .dispatch import (
    _compiled_cache,
    flydsl_blockwise_4wave_dgrad_supported,
    flydsl_blockwise_4wave_forward_supported,
    flydsl_blockwise_4wave_wgrad_supported,
    gemm_fp8_blockwise_4wave_dgrad,
    gemm_fp8_blockwise_4wave_wgrad,
    gemm_fp8_blockwise_4wave_wgrad_normalized,
    gemm_fp8_blockwise_dgrad,
    gemm_fp8_blockwise_forward,
    gemm_fp8_blockwise_wgrad,
    select_blockscale_fp8_forward_kernel,
)
from .eight_wave_blockwise_fp8_gemm_kernel import compile_blockscale_fp8_gemm_8w_3stage
from .four_wave_blockwise_fp8_gemm_kernel import (
    compile_blockscale_fp8_gemm_4w,
    compile_blockscale_fp8_gemm_nn_4w,
    compile_blockscale_fp8_gemm_nn_physical_4w,
    compile_blockscale_fp8_gemm_tn_4w,
    compile_blockscale_fp8_gemm_tn_physical_4w,
)
from .utils import compile_fp8_transpose_32x32

__all__ = [
    "compile_blockscale_fp8_gemm_4w",
    "compile_blockscale_fp8_gemm_8w_3stage",
    "compile_blockscale_fp8_gemm_nn_4w",
    "compile_blockscale_fp8_gemm_nn_physical_4w",
    "compile_blockscale_fp8_gemm_tn_4w",
    "compile_blockscale_fp8_gemm_tn_physical_4w",
    "compile_fp8_transpose_32x32",
    "flydsl_blockwise_4wave_dgrad_supported",
    "flydsl_blockwise_4wave_forward_supported",
    "flydsl_blockwise_4wave_wgrad_supported",
    "gemm_fp8_blockwise_4wave_dgrad",
    "gemm_fp8_blockwise_4wave_wgrad",
    "gemm_fp8_blockwise_4wave_wgrad_normalized",
    "gemm_fp8_blockwise_dgrad",
    "gemm_fp8_blockwise_forward",
    "gemm_fp8_blockwise_wgrad",
    "select_blockscale_fp8_forward_kernel",
]
