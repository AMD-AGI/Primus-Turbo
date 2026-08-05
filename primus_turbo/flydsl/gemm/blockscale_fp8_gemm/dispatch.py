###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); see LICENSE-APACHE for the Apache-2.0 terms.
#
# See LICENSE for license information.
###############################################################################

"""Shape validation, measured kernel selection, and cached host launchers."""

import flydsl.compiler as flyc
import torch
from flydsl.runtime.device import get_rocm_arch

from primus_turbo.flydsl.gemm.blockscale_fp8_gemm.four_wave_blockwise_fp8_gemm_kernel import (
    compile_blockscale_fp8_gemm_4w,
    compile_blockscale_fp8_gemm_nn_physical_4w,
    compile_blockscale_fp8_gemm_tn_4w,
    compile_blockscale_fp8_gemm_tn_physical_4w,
)
from primus_turbo.flydsl.utils.gemm_helper import ceildiv

_compiled_cache = {}
_MAX_BUFFER_BYTES = 0xFFFFFFFF
_DGRAD_8WAVE_MAX_CONTRACTION = 131072


def _shape_supported(M: int, N: int, K: int) -> bool:
    """Return whether dimensions satisfy the 4-wave tiling constraints."""

    return (
        str(get_rocm_arch()).startswith("gfx95")
        and M >= 128
        and N >= 128
        and K >= 128
        and M % 128 == 0
        and N % 128 == 0
        and K % 128 == 0
    )


def _buffers_fit(*byte_counts: int) -> bool:
    return all(0 < count <= _MAX_BUFFER_BYTES for count in byte_counts)


def flydsl_blockwise_4wave_dgrad_supported(M: int, N: int, K: int) -> bool:
    """Check ``grad[M,N] @ weight[N,K] -> dA[M,K]`` support."""

    return _shape_supported(M, K, N) and _buffers_fit(N * K)


def flydsl_blockwise_4wave_forward_supported(M: int, N: int, K: int) -> bool:
    """Check ``activation[M,K] @ weight[N,K].T`` support."""

    return (
        str(get_rocm_arch()).startswith("gfx95")
        and M >= 128
        and N >= 16
        and K >= 128
        and M % 128 == 0
        and N % 16 == 0
        and K % 128 == 0
        and 256 * K <= _MAX_BUFFER_BYTES
    )


def flydsl_blockwise_4wave_wgrad_supported(M: int, N: int, K: int) -> bool:
    """Check ``grad[M,N].T @ activation[M,K] -> dW[N,K]`` support."""

    return (
        str(get_rocm_arch()).startswith("gfx95")
        and M >= 128
        and N >= 16
        and K >= 128
        and M % 128 == 0
        and N % 16 == 0
        and K % 128 == 0
        and 192 * M <= _MAX_BUFFER_BYTES
    )


def select_blockscale_fp8_forward_kernel(M: int, N: int, K: int):
    """Select a measured gfx950 forward kernel and compile-time schedule."""

    if N == 202048 and K == 5120:
        return {
            "family": "8wave_3stage",
            "fold_group_size": 6,
            "interleave_width": 1,
            "wait_delay_thunks": 0,
            "scale_a_k_major": True,
            "group_m": 4,
        }

    if N % 128 != 0:
        return {
            "family": "4wave",
            "block_m": 128,
            "fold_group_size": 4,
            "k_loop_unroll": 6,
            "scale_a_k_major": True,
        }

    if K == 29568:
        return {
            "family": "4wave",
            "block_m": 128,
            "fold_group_size": 6,
            "k_loop_unroll": 6,
            "scale_a_k_major": True,
        }

    prefer_bm192 = (
        (M >= 8192 and N <= 57344 and K in (4096, 5120, 8192, 18944, 28672, 53248))
        or (K == 3584 and (N == 37888 or (M == 16384 and N == 152064) or M >= 163840))
        or (
            M == 4096
            and ((K == 5120 and N in (5120, 7168, 32768)) or (K == 4096 and N in (22016, 28672, 32000)))
        )
        or (K == 14336 and M >= 16384)
    )
    if prefer_bm192:
        return {
            "family": "4wave",
            "block_m": 192,
            "fold_group_size": 4,
            "k_loop_unroll": 2,
            "scale_a_k_major": True,
        }

    k_blocks = K // 128
    prefer_8wave = (
        M >= 4096
        and K >= 3584
        and k_blocks % 3 != 0
        and (
            K >= 32768
            or (N >= 65536 and K >= 8192)
            or (M <= 32768 and M % 192 != 0)
            or (M <= 8192 and N <= 32768)
            or (K == 4096 and N == 28672 and M % 192 != 0)
        )
    )
    if prefer_8wave:
        if K >= 32768:
            return {
                "family": "8wave_3stage",
                "fold_group_size": 5,
                "interleave_width": 2,
                "wait_delay_thunks": 8,
                "scale_a_k_major": False,
            }
        kmajor_fold_by_k = {
            3584: 4,
            11008: 4,
            14336: 4,
            16384: 12,
            18944: 4,
            28672: 8,
        }
        fold_group_size = kmajor_fold_by_k.get(K, 6)
        if (K == 4096 and N >= 28672) or (K == 5120 and N == 32768):
            fold_group_size = 8
        group_m = 4
        if K == 3584 and N == 37888:
            group_m = 1
        elif K == 4096 and N == 28672 and M > 32768:
            group_m = 2
        return {
            "family": "8wave_3stage",
            "fold_group_size": fold_group_size,
            "interleave_width": 1,
            "wait_delay_thunks": 0,
            "scale_a_k_major": True,
            "group_m": group_m,
        }

    return {
        "family": "4wave",
        "block_m": 192 if M % 192 == 0 else 128,
        "fold_group_size": 4,
        "k_loop_unroll": 2,
        "scale_a_k_major": True,
    }


def select_blockscale_fp8_dgrad_kernel(M: int, N: int, K: int):
    """Select dgrad for ``dY[M,N] @ W[N,K]`` using forward NT geometry."""

    config = select_blockscale_fp8_forward_kernel(M, K, N)
    if config["family"] == "8wave_3stage" and N <= _DGRAD_8WAVE_MAX_CONTRACTION:
        config = dict(config)
        config["scale_a_k_major"] = False
        return config
    return {
        "family": "4wave",
        "block_m": 192 if M % 192 == 0 else 128,
        "fold_group_size": 4,
        "k_loop_unroll": 2,
        "scale_a_k_major": False,
    }


def select_blockscale_fp8_wgrad_kernel(M: int, N: int, K: int):
    """Select wgrad for ``dY.T[N,M] @ A[M,K]``."""

    block_m = 192 if N % 192 == 0 else 128
    return {
        "family": "4wave",
        "block_m": block_m,
        "fold_group_size": 6 if block_m == 128 and M <= 65536 else 4,
    }


def _out_dtype_name(out_dtype: torch.dtype) -> str:
    if out_dtype == torch.bfloat16:
        return "bf16"
    if out_dtype == torch.float16:
        return "fp16"
    raise TypeError(f"4-wave blockscale GEMM output must be bf16 or fp16, got {out_dtype}")


def _run_cached(key, build, args):
    entry = _compiled_cache.get(key)
    if entry is None:
        raw = build()
        entry = raw, flyc.compile(raw, *args)
        _compiled_cache[key] = entry
    raw, compiled = entry
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        compiled(*args)


def _gemm_fp8_blockwise_dgrad(
    grad_out_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    use_selector: bool,
) -> torch.Tensor:
    """Physical NN dgrad: ``grad_out[M,N] @ weight[N,K]``."""

    M, N = grad_out_fp8.shape
    N_weight, K = weight_fp8.shape
    if N != N_weight or not flydsl_blockwise_4wave_dgrad_supported(M, N, K):
        raise ValueError(f"unsupported dgrad shape: grad={grad_out_fp8.shape}, weight={weight_fp8.shape}")
    if not grad_out_fp8.is_contiguous() or not weight_fp8.is_contiguous():
        raise ValueError("dgrad inputs must be contiguous")
    if tuple(grad_out_scale_inv.shape) != (M, N // 128):
        raise ValueError(f"invalid dgrad A-scale shape {grad_out_scale_inv.shape}")
    if tuple(weight_scale_inv.shape) != (N // 128, K // 128):
        raise ValueError(f"invalid dgrad B-scale shape {weight_scale_inv.shape}")

    config = (
        select_blockscale_fp8_dgrad_kernel(M, N, K)
        if use_selector
        else {
            "family": "4wave",
            "block_m": 192 if M % 192 == 0 else 128,
            "fold_group_size": 4,
            "k_loop_unroll": 2,
            "scale_a_k_major": False,
        }
    )
    use_8wave = config["family"] == "8wave_3stage"
    fuse_8wave_transpose = use_8wave and N * K <= _MAX_BUFFER_BYTES
    if use_8wave:
        weight_scale_arg = weight_scale_inv.T.contiguous()
    else:
        weight_scale_arg = weight_scale_inv

    out = torch.empty((M, K), dtype=out_dtype, device=grad_out_fp8.device)
    stream = torch.cuda.current_stream(grad_out_fp8.device)
    if fuse_8wave_transpose:
        args = (
            grad_out_fp8.view(torch.int8),
            weight_fp8.view(torch.int8),
            out,
            grad_out_scale_inv,
            weight_scale_arg,
            M,
            K,
            stream,
        )
    else:
        weight_t = torch.empty((K, N), dtype=weight_fp8.dtype, device=weight_fp8.device)
        args = (
            grad_out_fp8.view(torch.int8),
            weight_fp8.view(torch.int8),
            out,
            grad_out_scale_inv,
            weight_scale_arg,
            weight_t.view(torch.int8),
            stream,
        )
    out_dtype_name = _out_dtype_name(out_dtype)
    key = (
        "dgrad",
        config["family"],
        str(get_rocm_arch()),
        M,
        K,
        N,
        out_dtype_name,
        fuse_8wave_transpose,
        tuple(sorted(config.items())),
    )
    if use_8wave:
        from primus_turbo.flydsl.gemm.blockscale_fp8_gemm.eight_wave_blockwise_fp8_gemm_kernel import (
            compile_blockscale_fp8_gemm_nn_fused_8w,
            compile_blockscale_fp8_gemm_nn_physical_8w,
        )

        compile_8wave = (
            compile_blockscale_fp8_gemm_nn_fused_8w
            if fuse_8wave_transpose
            else compile_blockscale_fp8_gemm_nn_physical_8w
        )
        build = lambda: compile_8wave(
            M=M,
            N=K,
            K=N,
            out_dtype=out_dtype_name,
            fold_group_size=config["fold_group_size"],
            interleave_width=config["interleave_width"],
            wait_delay_thunks=config["wait_delay_thunks"],
            group_m=config.get("group_m", 4),
        )
    else:
        build = lambda: compile_blockscale_fp8_gemm_nn_physical_4w(
            M=M,
            N=K,
            K=N,
            out_dtype=out_dtype_name,
        )
    _run_cached(key, build, args)
    return out


def gemm_fp8_blockwise_4wave_dgrad(
    grad_out_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run physical NN dgrad with the 4-wave kernel."""

    return _gemm_fp8_blockwise_dgrad(
        grad_out_fp8,
        weight_fp8,
        grad_out_scale_inv,
        weight_scale_inv,
        out_dtype,
        use_selector=False,
    )


def gemm_fp8_blockwise_dgrad(
    grad_out_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch physical NN dgrad by its normalized GEMM shape."""

    return _gemm_fp8_blockwise_dgrad(
        grad_out_fp8,
        weight_fp8,
        grad_out_scale_inv,
        weight_scale_inv,
        out_dtype,
        use_selector=True,
    )


def gemm_fp8_blockwise_forward(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Forward NT using plain row-major FP8 operands."""

    M, K = activation_fp8.shape
    N, K_weight = weight_fp8.shape
    if K != K_weight or not flydsl_blockwise_4wave_forward_supported(M, N, K):
        raise ValueError(
            f"unsupported 4-wave forward shape: activation={activation_fp8.shape}, weight={weight_fp8.shape}"
        )
    if not activation_fp8.is_contiguous() or not weight_fp8.is_contiguous():
        raise ValueError("4-wave forward inputs must be contiguous")
    config = select_blockscale_fp8_forward_kernel(M, N, K)
    if config["scale_a_k_major"] and tuple(activation_scale_inv.shape) == (M, K // 128):
        activation_scale_inv = activation_scale_inv.T.contiguous()
    activation_scale_shape = (K // 128, M) if config["scale_a_k_major"] else (M, K // 128)
    if tuple(activation_scale_inv.shape) != activation_scale_shape:
        raise ValueError(f"invalid forward activation-scale shape {activation_scale_inv.shape}")
    if tuple(weight_scale_inv.shape) != (ceildiv(N, 128), K // 128):
        raise ValueError(f"invalid forward weight-scale shape {weight_scale_inv.shape}")

    out = torch.empty((M, N), dtype=out_dtype, device=activation_fp8.device)
    stream = torch.cuda.current_stream(activation_fp8.device)
    args = (
        activation_fp8.view(torch.int8),
        weight_fp8.view(torch.int8),
        out,
        activation_scale_inv,
        weight_scale_inv,
        M,
        N,
        stream,
    )
    out_dtype_name = _out_dtype_name(out_dtype)
    key = (
        "forward",
        config["family"],
        str(get_rocm_arch()),
        M,
        N,
        K,
        out_dtype_name,
        tuple(sorted(config.items())),
    )
    if config["family"] == "8wave_3stage":
        from primus_turbo.flydsl.gemm.blockscale_fp8_gemm.eight_wave_blockwise_fp8_gemm_kernel import (
            compile_blockscale_fp8_gemm_8w_3stage,
        )

        build = lambda: compile_blockscale_fp8_gemm_8w_3stage(
            K=K,
            M=M,
            N=N,
            out_dtype=out_dtype_name,
            fold_group_size=config["fold_group_size"],
            interleave_width=config["interleave_width"],
            wait_delay_thunks=config["wait_delay_thunks"],
            scale_a_k_major=config["scale_a_k_major"],
            group_m=config.get("group_m", 4),
        )
    else:
        build = lambda: compile_blockscale_fp8_gemm_4w(
            K=K,
            M=M,
            N=N,
            BLOCK_M=config["block_m"],
            out_dtype=out_dtype_name,
            scale_a_k_major=config["scale_a_k_major"],
            scale_b_k_major=False,
            fold_group_size=config["fold_group_size"],
            k_loop_unroll=config["k_loop_unroll"],
        )
    _run_cached(key, build, args)
    return out


def gemm_fp8_blockwise_4wave_wgrad(
    activation_fp8: torch.Tensor,
    grad_out_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Physical TN wgrad: ``grad_out[M,N].T @ activation[M,K]``."""

    M, K = activation_fp8.shape
    M_grad, N = grad_out_fp8.shape
    if M != M_grad or not flydsl_blockwise_4wave_wgrad_supported(M, N, K):
        raise ValueError(
            f"unsupported 4-wave wgrad shape: activation={activation_fp8.shape}, grad={grad_out_fp8.shape}"
        )
    if not activation_fp8.is_contiguous() or not grad_out_fp8.is_contiguous():
        raise ValueError("4-wave wgrad inputs must be contiguous")
    if tuple(activation_scale_inv.shape) != (M // 128, K):
        raise ValueError(f"invalid wgrad activation-scale shape {activation_scale_inv.shape}")
    if tuple(grad_out_scale_inv.shape) != (M // 128, N):
        raise ValueError(f"invalid wgrad gradient-scale shape {grad_out_scale_inv.shape}")

    out = torch.empty((N, K), dtype=out_dtype, device=activation_fp8.device)
    grad_t = torch.empty((N, M), dtype=grad_out_fp8.dtype, device=grad_out_fp8.device)
    activation_t = torch.empty((K, M), dtype=activation_fp8.dtype, device=activation_fp8.device)
    stream = torch.cuda.current_stream(activation_fp8.device)
    args = (
        grad_out_fp8.view(torch.int8),
        activation_fp8.view(torch.int8),
        out,
        grad_out_scale_inv,
        activation_scale_inv,
        grad_t.view(torch.int8),
        activation_t.view(torch.int8),
        stream,
    )
    out_dtype_name = _out_dtype_name(out_dtype)
    config = select_blockscale_fp8_wgrad_kernel(M, N, K)
    key = ("wgrad", str(get_rocm_arch()), N, K, M, out_dtype_name, tuple(sorted(config.items())))
    _run_cached(
        key,
        lambda: compile_blockscale_fp8_gemm_tn_physical_4w(
            M=N,
            N=K,
            K=M,
            out_dtype=out_dtype_name,
            BLOCK_M=config["block_m"],
            fold_group_size=config["fold_group_size"],
        ),
        args,
    )
    return out


def gemm_fp8_blockwise_4wave_wgrad_normalized(
    activation_fp8: torch.Tensor,
    grad_out_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """TN wgrad consuming transpose views emitted by dual-layout quantization."""

    M, K = activation_fp8.shape
    M_grad, N = grad_out_fp8.shape
    if M != M_grad or not flydsl_blockwise_4wave_wgrad_supported(M, N, K):
        raise ValueError(
            f"unsupported 4-wave wgrad shape: activation={activation_fp8.shape}, grad={grad_out_fp8.shape}"
        )
    activation_t = activation_fp8.T
    grad_t = grad_out_fp8.T
    if not activation_t.is_contiguous() or not grad_t.is_contiguous():
        raise ValueError("normalized 4-wave wgrad requires transpose views over contiguous storage")
    if tuple(activation_scale_inv.shape) != (M // 128, K):
        raise ValueError(f"invalid wgrad activation-scale shape {activation_scale_inv.shape}")
    if tuple(grad_out_scale_inv.shape) != (M // 128, N):
        raise ValueError(f"invalid wgrad gradient-scale shape {grad_out_scale_inv.shape}")

    out = torch.empty((N, K), dtype=out_dtype, device=activation_fp8.device)
    stream = torch.cuda.current_stream(activation_fp8.device)
    args = (
        grad_t.view(torch.int8),
        activation_t.view(torch.int8),
        out,
        grad_out_scale_inv,
        activation_scale_inv,
        N,
        K,
        stream,
    )
    out_dtype_name = _out_dtype_name(out_dtype)
    config = select_blockscale_fp8_wgrad_kernel(M, N, K)
    key = ("wgrad_normalized", str(get_rocm_arch()), N, K, M, out_dtype_name, tuple(sorted(config.items())))
    _run_cached(
        key,
        lambda: compile_blockscale_fp8_gemm_tn_4w(
            K=M,
            M=N,
            N=K,
            out_dtype=out_dtype_name,
            BLOCK_M=config["block_m"],
            scale_a_k_major=True,
            scale_b_k_major=True,
            fold_group_size=config["fold_group_size"],
        ),
        args,
    )
    return out


def gemm_fp8_blockwise_wgrad(
    activation_fp8: torch.Tensor,
    grad_out_fp8: torch.Tensor,
    activation_scale_inv: torch.Tensor,
    grad_out_scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch TN wgrad by the physical storage of its transpose views."""

    if activation_fp8.T.is_contiguous() and grad_out_fp8.T.is_contiguous():
        return gemm_fp8_blockwise_4wave_wgrad_normalized(
            activation_fp8,
            grad_out_fp8,
            activation_scale_inv,
            grad_out_scale_inv,
            out_dtype,
        )
    return gemm_fp8_blockwise_4wave_wgrad(
        activation_fp8,
        grad_out_fp8,
        activation_scale_inv,
        grad_out_scale_inv,
        out_dtype,
    )


__all__ = [
    "flydsl_blockwise_4wave_dgrad_supported",
    "flydsl_blockwise_4wave_forward_supported",
    "flydsl_blockwise_4wave_wgrad_supported",
    "gemm_fp8_blockwise_4wave_dgrad",
    "gemm_fp8_blockwise_4wave_wgrad",
    "gemm_fp8_blockwise_4wave_wgrad_normalized",
    "gemm_fp8_blockwise_dgrad",
    "gemm_fp8_blockwise_forward",
    "gemm_fp8_blockwise_wgrad",
    "select_blockscale_fp8_dgrad_kernel",
    "select_blockscale_fp8_forward_kernel",
    "select_blockscale_fp8_wgrad_kernel",
]
