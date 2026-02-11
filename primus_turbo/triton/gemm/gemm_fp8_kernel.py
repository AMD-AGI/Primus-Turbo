###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FP8 GEMM Triton persistent kernels.

Contains:
  Tensorwise (per-tensor) scaling:
    - _fp8_persistent_gemm_kernel: Core persistent kernel
    - gemm_fp8_tensorwise_triton_kernel: Public API

  Blockwise scaling:
    - _blockwise_fp8_autotune_kernel: Core autotuned persistent kernel
    - gemm_fp8_blockwise_triton_kernel: Unified public API (NT/NN/TN)

Environment variable: PRIMUS_TURBO_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import itertools

import torch
import triton
import triton.language as tl

from primus_turbo.triton.gemm.gemm_kernel import (
    _AUTOTUNE_MN_THRESHOLD,
    NUM_XCDS,
    _chiplet_transform_chunked,
    _get_num_cus,
    compute_sk_grid,
)

# ═══════════════════════════════════════════════════════════════════════════════
# AMD knobs helper (blockwise-specific)
# ═══════════════════════════════════════════════════════════════════════════════


def _set_amd_knobs(enable: bool = True):
    """Set AMD-specific Triton knobs.
    NOTE: use_async_copy and scalarize_packed_fops HURT CRR/TN performance
    by ~5-8% on MI300X. Only enable for NT/RCR and NN/RRR.
    """
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = enable
        triton.knobs.amd.scalarize_packed_fops = enable


# ###########################################################################
#
#  PART 1 — TENSORWISE (per-tensor) FP8 GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Tensorwise FP8 autotune configs
# ═══════════════════════════════════════════════════════════════════════════════


def _get_fp8_autotune_configs():
    """FP8 autotune configs — 4 total, pruned to 2 by layout.

    Hardcoded from 98-entry exhaustive tuning:
      - waves_per_eu=0 (85% of cases)
      - GROUP_SIZE_M: computed by heuristic, passed directly

    Autotuned:
      - BLOCK_SIZE_K ∈ {64, 128}
      - CHUNK_SIZE ∈ {32, 64}

    Total: 4 configs.
    """
    configs = []
    for blk_k in [64, 128]:
        for chunk in [32, 64]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_K": blk_k,
                        "CHUNK_SIZE": chunk,
                        "waves_per_eu": 0,
                        "matrix_instr_nonkdim": 16,
                        "kpack": 1,
                    },
                    num_warps=8,
                    num_stages=2,
                )
            )
    return configs


def _select_group_size_m_fp8(M, N, stride_ak, stride_bk):
    """Deterministic GROUP_SIZE_M from FP8 tuning data.

    FP8 patterns:
    - min_tile < 16 (non-standard dims like 3584): GROUP=8
    - Default (including TN): GROUP=4
      (29/32 FP8 TN entries use GROUP=4)
    """
    tiles_m = (M + 255) // 256
    tiles_n = (N + 255) // 256
    min_tile = min(tiles_m, tiles_n)

    if min_tile < 16:
        return 8
    else:
        return 4


def _prune_fp8_configs(configs, named_args, **kwargs):
    """Prune FP8 configs — placeholder for future layout-specific pruning."""
    return configs


@triton.jit()
def _fp8_persistent_gemm_kernel(
    A,
    B,
    C,
    A_scale_ptr,
    B_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Load per-tensor scales once (scalar)
    scale_a = tl.load(A_scale_ptr)
    scale_b = tl.load(B_scale_ptr)
    scale = scale_a * scale_b

    acc_dtype = tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # Use int64 offsets for pointer arithmetic to prevent int32 overflow with large matrices
        A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
        B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, input_precision="ieee")
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak
            B_BASE = B + rk[:, None].to(tl.int64) * stride_bk + rn[None, :].to(tl.int64) * stride_bn
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))
            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, input_precision="ieee")

        # Apply per-tensor scale
        acc *= scale
        c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :].to(tl.int64) * stride_cn
        tl.store(C_, c, c_mask)


_fp8_persistent_gemm_kernel_autotuned = triton.autotune(
    configs=_get_fp8_autotune_configs(),
    key=["M", "N", "K", "GROUP_SIZE_M", "NUM_SMS", "EVEN_K", "stride_ak", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_fp8_configs},
    warmup=20,
    rep=80,
)(_fp8_persistent_gemm_kernel)


# ═══════════════════════════════════════════════════════════════════════════════
# Tensorwise FP8 GEMM Public API
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_fp8_tensorwise_triton_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """General-purpose FP8 GEMM with per-tensor scaling.

    Computes: C = op(A) @ op(B) * a_scale_inv * b_scale_inv
    If trans_c=True, returns C^T (contiguous, shape N×M).

    Args:
        a: FP8 input matrix.
        a_scale_inv: Per-tensor dequantization scale for A, shape (1,), fp32.
        b: FP8 input matrix.
        b_scale_inv: Per-tensor dequantization scale for B, shape (1,), fp32.
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output C^T (shape N×M).

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    if trans_a:
        K, M = a.shape
        A_view = a.T
    else:
        M, K = a.shape
        A_view = a

    if trans_b:
        N, K2 = b.shape
        B_view = b.T
    else:
        K2, N = b.shape
        B_view = b

    assert K == K2, f"K mismatch: A gives K={K}, B gives K={K2}"

    # Ensure views have proper strides (no broadcast/expand zeros from autograd)
    if A_view.stride(0) == 0 or A_view.stride(1) == 0:
        A_view = A_view.contiguous()
    if B_view.stride(0) == 0 or B_view.stride(1) == 0:
        B_view = B_view.contiguous()

    # Handle trans_c by writing to (N, M) buffer with swapped strides
    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm = out.stride(1)  # = 1
        stride_cn = out.stride(0)  # = M
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm = out.stride(0)  # = N
        stride_cn = out.stride(1)  # = 1

    # Stride constexprs for compiler optimisation
    s_ak = A_view.stride(1)
    s_bk = B_view.stride(0)

    # Heuristic GROUP_SIZE_M (FP8-specific, not autotuned)
    group_m = _select_group_size_m_fp8(M, N, s_ak, s_bk)

    num_cus = _get_num_cus()
    num_sms = compute_sk_grid(M, N, K, 256, 256, 64, num_cus)

    args = (
        A_view,
        B_view,
        out,
        a_scale_inv,
        b_scale_inv,
        M,
        N,
        K,
        A_view.stride(0),
        B_view.stride(1),
        stride_cm,
        stride_cn,
    )

    if M * N > _AUTOTUNE_MN_THRESHOLD:
        # Large problem: skip autotune to avoid OOM, use default config
        blk_k = 128 if (s_ak == 1 and s_bk == 1) else 64
        even_k = K % blk_k == 0
        _fp8_persistent_gemm_kernel[(num_sms,)](
            *args,
            stride_ak=s_ak,
            stride_bk=s_bk,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=blk_k,
            GROUP_SIZE_M=group_m,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            CHUNK_SIZE=32,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=".ca",
            CACHE_MODIFIER_B=".ca",
            num_warps=8,
            num_stages=2,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=1,
        )
    else:
        even_k = K % 128 == 0  # Safe for both BLK_K=64 and 128
        _fp8_persistent_gemm_kernel_autotuned[(num_sms,)](
            *args,
            stride_ak=s_ak,
            stride_bk=s_bk,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=256,
            GROUP_SIZE_M=group_m,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=".ca",
            CACHE_MODIFIER_B=".ca",
        )
    return out


# ###########################################################################
#
#  PART 2 — BLOCKWISE FP8 GEMM
#
# ###########################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# Blockwise autotune configs (32 total)
# Fixed: BLOCK_N=128, BLOCK_K=128, wpe=2, mfma=16
# Variable: BLOCK_M, kpack, GROUP_M, CHUNK, num_stages
# ═══════════════════════════════════════════════════════════════════════════════
def _get_blockwise_autotune_configs():
    configs = []
    for block_m, kp, gm, chunk, ns in itertools.product(
        [128, 256],
        [1, 2],
        [4, 8],
        [32, 64],
        [1, 2],
    ):
        nw = 4 if block_m == 128 else 8
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": 128,
                    "BLOCK_K": 128,
                    "GROUP_M": gm,
                    "NUM_XCDS": 8,
                    "CHUNK": chunk,
                },
                num_warps=nw,
                num_stages=ns,
                pre_hook=None,
            )
        )
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Autotuned block-wise FP8 GEMM kernel
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=_get_blockwise_autotune_configs(),
    key=["M", "N", "K", "A_K_CONTIGUOUS", "B_K_CONTIGUOUS", "SCALE_2D_B"],
)
@triton.jit
def _blockwise_fp8_autotune_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scales_ptr,
    B_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak_val,
    stride_bk_val,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_k,
    stride_as_m,
    stride_bs_0,
    stride_bs_1,
    NUM_SMS,
    NUM_K_BLOCKS,
    A_K_CONTIGUOUS: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
    SCALE_2D_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)

    # XCD-aware PID transform (inlined to avoid constexpr propagation issues)
    if NUM_XCDS != 1:
        chunk_id = pid // CHUNK
        chunk_off = pid % CHUNK
        cta_per_xcd = NUM_SMS // NUM_XCDS
        pid = (chunk_id % NUM_XCDS) * cta_per_xcd + (chunk_id // NUM_XCDS) * CHUNK + chunk_off

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    total = num_m * num_n
    grp = GROUP_M * num_n

    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for tid in range(pid, total, NUM_SMS):
        gid = tid // grp
        fm = gid * GROUP_M
        gs = min(num_m - fm, GROUP_M)
        pm = fm + (tid % grp) % gs
        pn = (tid % grp) // gs
        tl.assume(pm >= 0)
        tl.assume(pn >= 0)

        rm = tl.max_contiguous(tl.multiple_of((pm * BLOCK_M + tl.arange(0, BLOCK_M)) % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of((pn * BLOCK_N + tl.arange(0, BLOCK_N)) % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        if A_K_CONTIGUOUS:
            a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :]
        else:
            a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak_val

        if B_K_CONTIGUOUS:
            b_ptrs = B_ptr + rk[:, None] + rn[None, :] * stride_bn
        else:
            b_ptrs = B_ptr + rk[:, None] * stride_bk_val + rn[None, :] * stride_bn

        as_ptrs = A_scales_ptr + rm * stride_as_m

        if SCALE_2D_B:
            bs_ptr_base = B_scales_ptr + pn * stride_bs_0
        else:
            bs_ptrs = B_scales_ptr + rn * stride_bs_0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ki in range(NUM_K_BLOCKS):
            if A_K_CONTIGUOUS:
                a = tl.load(tl.multiple_of(a_ptrs, (1, 16)), cache_modifier=".ca")
            else:
                a = tl.load(tl.multiple_of(a_ptrs, (16, 1)), cache_modifier=".ca")

            if B_K_CONTIGUOUS:
                b = tl.load(tl.multiple_of(b_ptrs, (16, 1)), cache_modifier=".ca")
            else:
                b = tl.load(tl.multiple_of(b_ptrs, (1, 16)), cache_modifier=".ca")

            partial = tl.dot(a, b, input_precision="ieee")

            a_s = tl.load(as_ptrs + ki * stride_as_k)

            if SCALE_2D_B:
                b_s = tl.load(bs_ptr_base + ki * stride_bs_1)
                acc += partial * (a_s * b_s)[:, None]
            else:
                b_s = tl.load(bs_ptrs + ki * stride_bs_1)
                acc += partial * a_s[:, None] * b_s[None, :]

            if A_K_CONTIGUOUS:
                a_ptrs += BLOCK_K
            else:
                a_ptrs += BLOCK_K * stride_ak_val

            if B_K_CONTIGUOUS:
                b_ptrs += BLOCK_K
            else:
                b_ptrs += BLOCK_K * stride_bk_val

        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(C_ptr.type.element_ty), mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Public API — Block-wise FP8 GEMM
# Interface consistent with CK blockwise backend.
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_fp8_blockwise_triton_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Unified block-wise FP8 GEMM Triton kernel.

    Interface consistent with the CK blockwise backend.
    Dispatches internally by (trans_a, trans_b) to the optimal layout.

    Supported layouts:
      NT/RCR (forward):  trans_a=False, trans_b=True
        A: [M, K], A_scales: [M, K//128]
        B: [N, K], B_scales: [N//128, K//128]  (2D block scaling)

      NN/RRR (grad_X):   trans_a=False, trans_b=False
        A: [M, K], A_scales: [M, K//128]
        B: [N, K], B_scales: [N//128, K//128]  (2D, transposed internally)

      TN/CRR (grad_W):   trans_a=True, trans_b=False
        A: [K, M], A_scales: [K//128, M]
        B: [K, N], B_scales: [K//128, N]

    Args:
        a: FP8 input matrix.
        a_scale_inv: Block-wise scale for A, shape depends on layout.
        trans_a: Whether A is transposed.
        b: FP8 input matrix.
        b_scale_inv: Block-wise scale for B, shape depends on layout.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output.

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    if not trans_a and trans_b:
        return _blockwise_nt(a, a_scale_inv, b, b_scale_inv, out_dtype, trans_c)
    elif not trans_a and not trans_b:
        return _blockwise_nn(a, a_scale_inv, b, b_scale_inv, out_dtype, trans_c)
    elif trans_a and not trans_b:
        return _blockwise_tn(a, a_scale_inv, b, b_scale_inv, out_dtype, trans_c)
    else:
        raise ValueError(f"Unsupported layout for blockwise FP8 Triton: trans_a={trans_a}, trans_b={trans_b}")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal layout-specific helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _blockwise_nt(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    trans_c: bool,
) -> torch.Tensor:
    """NT/RCR forward: C = A[M,K] @ B[N,K].T, 2D B_scales."""
    _set_amd_knobs(enable=True)

    M, K = a.shape
    N = b.shape[0]
    num_k = K // 128

    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    A_scales_t = a_scale_inv.T.contiguous()  # [K//128, M]
    B_t = b.T  # view [K, N]

    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    NUM_SMS = num_m * num_n

    _blockwise_fp8_autotune_kernel[(NUM_SMS,)](
        a,
        B_t,
        out,
        A_scales_t,
        b_scale_inv,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        B_t.stride(0),
        B_t.stride(1),
        stride_cm,
        stride_cn,
        A_scales_t.stride(0),
        A_scales_t.stride(1),
        b_scale_inv.stride(0),
        b_scale_inv.stride(1),
        NUM_SMS,
        num_k,
        A_K_CONTIGUOUS=True,
        B_K_CONTIGUOUS=True,
        SCALE_2D_B=True,
    )
    return out


def _blockwise_nn(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    trans_c: bool,
) -> torch.Tensor:
    """NN/RRR grad_X: C = A[M,K] @ B[K,N], 2D B_scales (transposed internally)."""
    _set_amd_knobs(enable=True)

    M, K = a.shape
    _, N = b.shape
    num_k = K // 128

    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    A_scales_t = a_scale_inv.T.contiguous()  # [K//128, M]
    # B_scales from quantization: [dim0_blocks, dim1_blocks] for weight stored as [N_fwd, K_fwd].
    # Kernel expects [N_output_blocks, K_inner_blocks] indexing → transpose.
    b_scale_inv_t = b_scale_inv.T.contiguous()

    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    NUM_SMS = num_m * num_n

    _blockwise_fp8_autotune_kernel[(NUM_SMS,)](
        a,
        b,
        out,
        A_scales_t,
        b_scale_inv_t,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_cm,
        stride_cn,
        A_scales_t.stride(0),
        A_scales_t.stride(1),
        b_scale_inv_t.stride(0),
        b_scale_inv_t.stride(1),
        NUM_SMS,
        num_k,
        A_K_CONTIGUOUS=True,
        B_K_CONTIGUOUS=False,
        SCALE_2D_B=True,
    )
    return out


def _blockwise_tn(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    trans_c: bool,
) -> torch.Tensor:
    """TN/CRR grad_W: C = A[K,M].T @ B[K,N], 1D+1D B_scales.

    Scale layouts (from axis=0 / column-wise quantization):
      a_scale_inv: [K//128, M]
      b_scale_inv: [K//128, N]
    """
    _set_amd_knobs(enable=False)

    K, M = a.shape
    _, N = b.shape
    num_k = K // 128

    if trans_c:
        out = torch.empty((N, M), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(1), out.stride(0)
    else:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        stride_cm, stride_cn = out.stride(0), out.stride(1)

    A_view = a.T  # [M, K] view with strided K

    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    NUM_SMS = num_m * num_n

    _blockwise_fp8_autotune_kernel[(NUM_SMS,)](
        A_view,
        b,
        out,
        a_scale_inv,
        b_scale_inv,
        M,
        N,
        K,
        A_view.stride(0),
        A_view.stride(1),
        b.stride(0),
        b.stride(1),
        stride_cm,
        stride_cn,
        a_scale_inv.stride(0),
        a_scale_inv.stride(1),  # [K//128, M]: stride_as_k=M, stride_as_m=1
        b_scale_inv.stride(1),
        b_scale_inv.stride(0),  # [K//128, N]: stride_bs_0=1(rn), stride_bs_1=N(ki)
        NUM_SMS,
        num_k,
        A_K_CONTIGUOUS=False,
        B_K_CONTIGUOUS=False,
        SCALE_2D_B=False,
    )
    return out
