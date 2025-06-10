# This file is based on code from https://github.com/pytorch/ao/tree/main/torchao/prototype/blockwise_fp8
# Modified by Primus-Turbo for FP8 blockwise quantization

import triton
import triton.language as tl
from triton import Config

fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_size_mn[0], "BLOCK_SIZE_N": block_size_mn[1]},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_mn in [(64, 64), (128, 128), (128, 256), (256, 256)]
    for num_stages in [2]
    for num_warps in [4, 8]
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

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
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
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
