###############################################################################
# This file is based on code from the TorchAO prototype for Blockwise FP8:
#   https://github.com/pytorch/ao/blob/main/torchao/prototype/blockwise_fp8_training/kernels.py
#
# Reference:
#   @software{torchao,
#     title   = {TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
#     author  = {torchao},
#     url     = {https://github.com/pytorch/ao},
#     license = {BSD-3-Clause},
#     month   = {oct},
#     year    = {2024}
#   }
#
# License:
#   BSD 3-Clause License
#   https://github.com/pytorch/ao/blob/main/LICENSE
#
# Modifications:
#   Modified by Primus-Turbo team for FP8 blockwise and
#   integration into primus_turbo/triton/gemm.
###############################################################################

import triton
import triton.language as tl
from triton import Config

fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_size_mn[0], "BLOCK_SIZE_N": block_size_mn[1]},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_mn in [(128, 128), (128, 256)]
    for num_stages in [2]
    for num_warps in [4]
]


# For FWD NT
# a is act, b is weight
@triton.autotune(configs=fp8_gemm_configs, key=["N", "K", "BLOCK_SIZE_K"])
@triton.jit
def gemm_fp8_blockwise_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    k_tile_nums = tl.cdiv(K, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k_tile_nums
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k_tile_nums

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Main loop
    for i in range(k_tile_nums):
        a_tile = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        a_s_tile = tl.load(a_s_ptrs)
        b_s_tile = tl.load(b_s_ptrs)
        accumulator += tl.dot(a_tile, b_tile) * a_s_tile[:, None] * b_s_tile[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        offs_k += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1

    # Store
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


# For DGrad NN
# Computes: dX = dY @ W^T
# [m, n] = [m, k] * [k, n]
# a_scales = [m, K//block_size]
# b_scales = [k//block_size, n//block_size]
@triton.autotune(configs=fp8_gemm_configs, key=["N", "K", "BLOCK_SIZE_K"])
@triton.jit
def gemm_fp8_blockwise_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = tl.cast((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, tl.int64)
    offs_n = tl.cast((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N, tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    k_tile_nums = tl.cdiv(K, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    a_s_ptrs = a_s_ptr + offs_m * k_tile_nums
    b_s_ptrs = b_s_ptr + offs_n // BLOCK_SIZE_K
    b_ptrs_stride = BLOCK_SIZE_K * N
    b_s_ptrs_stride = tl.cdiv(N, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_tile_nums):
        a_tile = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        a_s_tile = tl.load(a_s_ptrs)
        b_s_tile = tl.load(b_s_ptrs)
        accumulator += tl.dot(a_tile, b_tile) * a_s_tile[:, None] * b_s_tile[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_ptrs_stride
        a_s_ptrs += 1
        b_s_ptrs += b_s_ptrs_stride
        offs_k += BLOCK_SIZE_K

    # Store
    offs_m = tl.cast(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), tl.int64)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


# For WGrad TN
# Computes: dW = dY^T @ X
# [m, n] = [k, m] * [k, n]
# dY: a [k, m]FP8, a_s[k//block_size, m]FP32
# X : b [k, n]FP8, b_s[k//block_size, n]FP32
# dW: c [m, n]
@triton.autotune(configs=fp8_gemm_configs, key=["N", "K", "BLOCK_SIZE_K"])
@triton.jit
def gemm_fp8_blockwise_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    k_tile_nums = tl.cdiv(K, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_k[None, :] * M + offs_m[:, None]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    a_ptrs_stride = M * BLOCK_SIZE_K
    b_ptrs_stride = N * BLOCK_SIZE_K
    a_s_ptrs = a_s_ptr + offs_m
    b_s_ptrs = b_s_ptr + offs_n
    a_s_ptrs_stride = M
    b_s_ptrs_stride = N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_tile_nums):
        a_tile = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        a_s_tile = tl.load(a_s_ptrs)
        b_s_tile = tl.load(b_s_ptrs)
        accumulator += tl.dot(a_tile, b_tile) * a_s_tile[:, None] * b_s_tile[None, :]
        a_ptrs += a_ptrs_stride
        b_ptrs += b_ptrs_stride
        a_s_ptrs += a_s_ptrs_stride
        b_s_ptrs += b_s_ptrs_stride
        offs_k += BLOCK_SIZE_K

    # Store
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


"""
@triton.jit
def gemm_fp8_blockwise_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsk,
    stride_bsn,
    SCALE_GROUP_SIZE_M: tl.constexpr,
    SCALE_GROUP_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_asm = offs_am // SCALE_GROUP_SIZE_M
    offs_bsn = offs_bn // SCALE_GROUP_SIZE_N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_s_ptrs = a_s_ptr + offs_asm[:, None] * stride_asm
    b_s_ptrs = b_s_ptr + offs_bsn[None, :] * stride_bsn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kid in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(a_ptrs, mask=offs_k[None, :] < (K - kid * BLOCK_SIZE_K), other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_k[:, None] < (K - kid * BLOCK_SIZE_K), other=0.0)
        a_s_tile = tl.load(a_s_ptrs)
        b_s_tile = tl.load(b_s_ptrs)

        accumulator += tl.dot(a_tile, b_tile) * a_s_tile * b_s_tile

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_s_ptrs += stride_ask
        b_s_ptrs += stride_bsk

    offs_cm = offs_am
    offs_cn = offs_bn
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N),
    )
"""
