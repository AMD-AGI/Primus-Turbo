###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Kernel launch functions for grouped-GEMM FP8 ASM .co/.hsaco kernels.

These functions pack the kernel arguments, call the appropriate loader, and
dispatch via :func:`primus_turbo.asm_co.hip_utils.asm_co_module_launch`.
"""

import ctypes
import struct

import torch

from primus_turbo.asm_co.hip_utils import asm_co_module_launch
from primus_turbo.asm_co.grouped_gemm.loader import (
    get_asm_co_dgrad_func,
    get_asm_co_fwd_hsaco_func,
    get_asm_co_wgrad_beta1_func,
    get_asm_co_wgrad_func,
)

__all__ = [
    "launch_asm_co_fwd",
    "launch_asm_co_fwd_dgrad",
    "launch_asm_co_wgrad_variable_k",
    "launch_asm_co_wgrad_variable_k_beta1",
]

# ── Shared launch parameters ──────────────────────────────────────────────────
_ASM_CO_THREADS   = 512
_ASM_CO_LDS_BYTES = 65536

_ASM_CO_FWD_LDS_BYTES   = 131072
_ASM_CO_DGRAD_LDS_BYTES = 131072
_ASM_CO_WGRAD_THREADS   = 1024


def launch_asm_co_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> torch.Tensor:
    """Launch the hand-tuned FWD .hsaco kernel (trans_b=False) via HIP module API.

    Kernarg layout (88 bytes packed into a 96-byte buffer):
      7 pointers: A, B, C, A_scale, B_scale, group_offs, tile_cumsum
      6 int32s:   G, N, K, stride_am, stride_bg, stride_cm
    """
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import _compute_tile_cumsum_kernel

    m = a.shape[0]
    k = a.shape[1]
    g = b.shape[0]
    n = b.shape[2]  # trans_b=False → N is last dim

    out = torch.empty((m, n), device=a.device, dtype=out_dtype)

    blk_m = 256
    blk_n = 256
    num_pid_n = (n + blk_n - 1) // blk_n
    tile_cumsum = torch.empty(g + 1, device=a.device, dtype=torch.int32)
    _compute_tile_cumsum_kernel[(1,)](group_offs, tile_cumsum, g, num_pid_n, BLOCK_SIZE_M=blk_m)

    func = get_asm_co_fwd_hsaco_func(n)

    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQQ", buf, 0,
        a.data_ptr(), b.data_ptr(), out.data_ptr(),
        a_scales.data_ptr(), b_scales.data_ptr(),
        group_offs.data_ptr(), tile_cumsum.data_ptr(),
    )
    struct.pack_into(
        "<iiiiii", buf, 56,
        g, n, k,
        a.stride(0),    # stride_am = K
        b.stride(0),    # stride_bg = K*N
        out.stride(0),  # stride_cm = N
    )

    grid_x = (
        num_cu if num_cu is not None
        else torch.cuda.get_device_properties(a.device).multi_processor_count
    )
    asm_co_module_launch(
        func, buf, 96, grid_x, _ASM_CO_THREADS, _ASM_CO_FWD_LDS_BYTES, a.device,
        f"fwd K={k}, N={n}",
    )
    return out


def launch_asm_co_fwd_dgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> torch.Tensor:
    """Launch the optimized DGRAD .hsaco kernel (trans_b=True) via HIP module API.

    Uses per-shape optimized Triton kernels with tile_cumsum persistent scheduling.
    Kernarg layout (84 bytes packed into a 96-byte buffer):
      7 pointers: A, B, C, A_scale, B_scale, group_offs, tile_cumsum
      7 int32s:   G, N, K, stride_am, stride_bg, stride_bn, stride_cm
    """
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import _compute_tile_cumsum_kernel

    m = a.shape[0]
    k = a.shape[1]
    g = b.shape[0]
    n = b.shape[1]  # trans_b=True → N is dim 1

    out = torch.empty((m, n), device=a.device, dtype=out_dtype)

    blk_m = 256
    blk_n = 256
    num_pid_n = (n + blk_n - 1) // blk_n
    tile_cumsum = torch.empty(g + 1, device=a.device, dtype=torch.int32)
    _compute_tile_cumsum_kernel[(1,)](group_offs, tile_cumsum, g, num_pid_n, BLOCK_SIZE_M=blk_m)

    func = get_asm_co_dgrad_func(k)

    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQQ", buf, 0,
        a.data_ptr(), b.data_ptr(), out.data_ptr(),
        a_scales.data_ptr(), b_scales.data_ptr(),
        group_offs.data_ptr(), tile_cumsum.data_ptr(),
    )
    struct.pack_into(
        "<iiiiiii", buf, 56,
        g, n, k,
        a.stride(0),    # stride_am = K
        b.stride(0),    # stride_bg = N*K
        b.stride(1),    # stride_bn = K  (trans_b=True, B is [G,N,K])
        out.stride(0),  # stride_cm = N
    )

    grid_x = (
        num_cu if num_cu is not None
        else torch.cuda.get_device_properties(a.device).multi_processor_count
    )
    asm_co_module_launch(
        func, buf, 96, grid_x, _ASM_CO_THREADS, _ASM_CO_DGRAD_LDS_BYTES, a.device,
        f"dgrad K={k}, N={n}",
    )
    return out


def launch_asm_co_wgrad_variable_k(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> torch.Tensor:
    """Launch the hand-tuned variable-K wgrad .co kernel (beta=0) via HIP module API.

    Kernarg layout (96 bytes) matches ``KernArgs`` in asm_co_grouped_gemm.cpp.
    """
    out_m = lhs.shape[1]
    out_n = rhs.shape[1]
    g = group_lens.shape[0]
    out = torch.empty((g, out_m, out_n), device=lhs.device, dtype=out_dtype)

    func = get_asm_co_wgrad_func()

    # The v_mfma_f32_32x32x64_f8f6f4 instruction on gfx950 applies an implicit
    # 0.25x (2^-1 per operand) to OCP E4M3 data; compensate with 2x per operand.
    lhs_scale_adj = lhs_scale * 2.0
    rhs_scale_adj = rhs_scale * 2.0

    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQ", buf, 0,
        lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(),
        lhs_scale_adj.data_ptr(), rhs_scale_adj.data_ptr(),
        group_offs.data_ptr(),
    )
    struct.pack_into(
        "<iiiiiiii", buf, 48,
        g, out_m, out_n,
        lhs.stride(0), rhs.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
    )

    grid_x = (
        num_cu if num_cu is not None
        else torch.cuda.get_device_properties(lhs.device).multi_processor_count
    )
    asm_co_module_launch(
        func, buf, 96, grid_x, _ASM_CO_WGRAD_THREADS, _ASM_CO_LDS_BYTES, lhs.device,
        f"wgrad OUT_M={out_m}, OUT_N={out_n}",
    )
    return out


def launch_asm_co_wgrad_variable_k_beta1(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    out: torch.Tensor,
    num_cu: int | None,
) -> None:
    """Launch the beta=1 wgrad .co: ``out += A^T @ B * scale`` (fused accumulation).

    The beta=1 kernel loads the previous C tile and adds to the scaled MFMA
    output before storing.  ``out`` is modified in-place.
    """
    out_m = lhs.shape[1]
    out_n = rhs.shape[1]

    func = get_asm_co_wgrad_beta1_func()

    lhs_scale_adj = lhs_scale * 2.0
    rhs_scale_adj = rhs_scale * 2.0

    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQ", buf, 0,
        lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(),
        lhs_scale_adj.data_ptr(), rhs_scale_adj.data_ptr(),
        group_offs.data_ptr(),
    )
    struct.pack_into(
        "<iiiiiiii", buf, 48,
        group_lens.shape[0], out_m, out_n,
        lhs.stride(0), rhs.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
    )

    grid_x = (
        num_cu if num_cu is not None
        else torch.cuda.get_device_properties(lhs.device).multi_processor_count
    )
    asm_co_module_launch(
        func, buf, 96, grid_x, _ASM_CO_WGRAD_THREADS, _ASM_CO_LDS_BYTES, lhs.device,
        f"wgrad_beta1 OUT_M={out_m}, OUT_N={out_n}",
    )
