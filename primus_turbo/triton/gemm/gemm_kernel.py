###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Acknowledgement:
#   The persistent GEMM kernels in this file are adapted from tritonBLAS
#   (https://github.com/ROCm/tritonBLAS). We thank the tritonBLAS authors
#   for their high-quality Triton kernel implementations on AMD GPUs.
###############################################################################

"""
GEMM Triton persistent kernels — BF16/FP16.

Contains:
  - _bf16_persistent_gemm_kernel: BF16/FP16 persistent kernel (data-parallel grid)

Public API:
  - gemm_triton_kernel  — BF16/FP16 GEMM

FP8 kernels (tensorwise + blockwise) are in gemm_fp8_kernel.py.

Environment variable: PRIMUS_TURBO_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

try:
    import origami

    _HAS_ORIGAMI = not os.environ.get("PRIMUS_TURBO_DISABLE_ORIGAMI", "")
except ModuleNotFoundError:
    _HAS_ORIGAMI = False

_ORIGAMI_UNAVAILABLE_LOGGED = False

# Map torch dtypes to origami string (for problem_t). Align with TensorAtlas heuristics/selector.py.
_ORIGAMI_DTYPE_TO_STR = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
for _k in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if hasattr(torch, _k):
        _ORIGAMI_DTYPE_TO_STR[getattr(torch, _k)] = "f8"

# FP8 dtypes: torch.finfo can be unsupported/buggy, so we treat them explicitly.
_ORIGAMI_FP8_DTYPES = tuple(
    d
    for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if d is not None
)


def _dtype_bits(dtype):
    """Element bits for LDS/MI dim; safe for FP8 (finfo not fully supported)."""
    if _ORIGAMI_FP8_DTYPES and dtype in _ORIGAMI_FP8_DTYPES:
        return 8
    try:
        if dtype.is_floating_point:
            return torch.finfo(dtype).bits
        return torch.iinfo(dtype).bits
    except (TypeError, AttributeError):
        return 16


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware constants & chiplet transform
# ═══════════════════════════════════════════════════════════════════════════════

NUM_XCDS = 8


@triton.jit
def _chiplet_transform_chunked(
    pid,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    if pid > (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    xcd = pid % NUM_XCDS
    return chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk


# ═══════════════════════════════════════════════════════════════════════════════
# BF16 Persistent GEMM Kernel
# ═══════════════════════════════════════════════════════════════════════════════


def _select_group_size_m_bf16(M, N, stride_ak, stride_bk):
    """Fallback heuristic GROUP_SIZE_M (used when origami is unavailable).

    Patterns from 94-entry BF16 tuning data:
    - min_tile < 16 (non-standard dims like 3584): GROUP=8
    - TN layout (stride_ak!=1, stride_bk!=1) + both tiles >= 32: GROUP=5
    - Default: 4
    """
    tiles_m = (M + 255) // 256
    tiles_n = (N + 255) // 256
    min_tile = min(tiles_m, tiles_n)

    is_tn = (stride_ak != 1) and (stride_bk != 1)

    if min_tile < 16:
        return 8
    elif is_tn and tiles_m >= 32 and tiles_n >= 32:
        return 5
    else:
        return 4


# ─── Origami analytical config selection (aligned with TensorAtlas / tritonBLAS) ───


def _estimate_lds_bytes(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages=2):
    """LDS usage for Triton matmul tile; num_stages=2. Matches TensorAtlas without async_copy."""
    lds_a = block_m * block_k * elem_bytes_a
    lds_b = block_k * block_n * elem_bytes_b
    base_buffers = max(1, num_stages - 1)
    return (lds_a + lds_b) * base_buffers


def _infer_mi_dim(hardware, element_size_a, element_size_b):
    """Infer matrix instruction dimensions from hardware and dtypes. Align with TensorAtlas."""
    n_cu = hardware.N_CU
    max_bits = max(element_size_a, element_size_b)
    # gfx950
    if n_cu == 256:
        if max_bits == 32:
            return [16, 16, 4]
        if max_bits == 16:
            return [16, 16, 32]
        if max_bits <= 8:
            return [16, 16, 128]
    # gfx942 (304, 80, 64 CUs)
    if n_cu in (304, 80, 64):
        if max_bits == 32:
            return [16, 16, 4]
        if max_bits == 16:
            return [16, 16, 16]
        if max_bits == 8:
            return [16, 16, 32]
    return [16, 16, 16]


def _get_valid_tiles(hardware, block_mn_range, block_k_range, mi_dim, elem_bytes_a, elem_bytes_b):
    """Valid (blk_m, blk_n, blk_k, mi_m, mi_n, mi_k, occ) passing LDS check."""
    lds_cap = hardware.lds_capacity
    valid = []
    for bm, bn, bk in (
        (bm, bn, bk) for bm in block_mn_range for bn in block_mn_range for bk in block_k_range
    ):
        lds = _estimate_lds_bytes(bm, bn, bk, elem_bytes_a, elem_bytes_b, num_stages=2)
        if lds <= lds_cap:
            valid.append((bm, bn, bk, mi_dim[0], mi_dim[1], mi_dim[2], 1))
    return valid


def _make_problem(M, N, K, a_dtype, b_dtype, c_dtype, mi_dtype_str, trans_a, trans_b, mx_block_size=0):
    """Build origami problem_t for rank_configs / select_workgroup_mapping.

    trans_a, trans_b: logical op(A) @ op(B). NT = (False, True), TN/CRR = (True, False), NN/RRR = (False, False).
    """
    problem = origami.problem_t()
    problem.size = origami.dim3_t(M, N, K)
    problem.batch = 1
    # Per your convention: trans_a=True -> origami N, trans_a=False -> origami T
    problem.a_transpose = origami.transpose_t.N if trans_a else origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N if trans_b else origami.transpose_t.T
    problem.a_dtype = origami.string_to_datatype(_ORIGAMI_DTYPE_TO_STR.get(a_dtype, "bf16"))
    problem.b_dtype = origami.string_to_datatype(_ORIGAMI_DTYPE_TO_STR.get(b_dtype, "bf16"))
    problem.c_dtype = origami.string_to_datatype(_ORIGAMI_DTYPE_TO_STR.get(c_dtype, "bf16"))
    problem.d_dtype = problem.c_dtype
    problem.mi_dtype = origami.string_to_datatype(mi_dtype_str)
    problem.a_mx_block_size = mx_block_size
    problem.b_mx_block_size = mx_block_size
    return problem


def _tiles_to_configs(valid_tiles, streamk=True):
    """Convert valid_tiles to origami config_t list."""
    grid_sel = origami.grid_selection_t.k_split_aware if streamk else origami.grid_selection_t.data_parallel
    configs = []
    for blk_m, blk_n, blk_k, mi_m, mi_n, mi_k, occ in valid_tiles:
        cfg = origami.config_t()
        cfg.mt = origami.dim3_t(blk_m, blk_n, blk_k)
        cfg.mi = origami.dim3_t(mi_m, mi_n, mi_k)
        cfg.occupancy = occ
        cfg.grid_selection = grid_sel
        configs.append(cfg)
    return configs


@functools.lru_cache(maxsize=4096)
def _select_params_origami(M, N, K, out_dtype, a_dtype=None, b_dtype=None, trans_a=False, trans_b=True):
    """Use origami rank_configs + select_workgroup_mapping (align with TensorAtlas selector.py).

    trans_a, trans_b: logical layout (op(A) @ op(B)). Forward NT = (False, True);
    backward grad_a (NN) = (False, False); backward grad_b (TN) = (True, False).
    Returns (block_m, block_n, block_k, group_size_m, cache_a, cache_b) or None.
    """
    global _ORIGAMI_UNAVAILABLE_LOGGED
    if not _HAS_ORIGAMI:
        if not _ORIGAMI_UNAVAILABLE_LOGGED:
            _ORIGAMI_UNAVAILABLE_LOGGED = True
            print("[gemm/origami] origami not installed or disabled, using heuristic params")
        return None

    a_dtype = a_dtype if a_dtype is not None else out_dtype
    b_dtype = b_dtype if b_dtype is not None else out_dtype

    try:
        device_id = torch.cuda.current_device()
        hardware = origami.get_hardware_for_device(device_id)
    except Exception as e:
        print(f"[gemm/origami] get_hardware_for_device: {e}")
        return None

    try:
        elem_bits_a = _dtype_bits(a_dtype)
        elem_bits_b = _dtype_bits(b_dtype)
        elem_bytes_a = elem_bits_a // 8
        elem_bytes_b = elem_bits_b // 8

        # mi_dtype: use the smaller input dtype (align with TensorAtlas selector.py)
        input_dtype_for_mi = a_dtype if elem_bits_a <= elem_bits_b else b_dtype
        mi_dtype_str = _ORIGAMI_DTYPE_TO_STR.get(
            input_dtype_for_mi, _ORIGAMI_DTYPE_TO_STR.get(out_dtype, "bf16")
        )

        mi_dim = _infer_mi_dim(hardware, elem_bits_a, elem_bits_b)
        block_mn_range = [64, 128, 256]
        block_k_range = [64, 128, 256]
        valid_tiles = _get_valid_tiles(
            hardware, block_mn_range, block_k_range, mi_dim, elem_bytes_a, elem_bytes_b
        )
        if not valid_tiles:
            return None

        problem = _make_problem(M, N, K, a_dtype, b_dtype, out_dtype, mi_dtype_str, trans_a, trans_b)
        configs = _tiles_to_configs(valid_tiles, streamk=True)

        best_result = origami.select_config(problem, hardware, configs)
        # origami may return items with .config (TensorAtlas) or the config itself
        best_cfg = best_result.config if hasattr(best_result, "config") else best_result
        BLK_M = best_cfg.mt.m
        BLK_N = best_cfg.mt.n
        BLK_K = best_cfg.mt.k

        total_tiles = ((M + BLK_M - 1) // BLK_M) * ((N + BLK_N - 1) // BLK_N)
        wgm_result = origami.select_workgroup_mapping(problem, hardware, best_cfg, total_tiles)
        gsize_m = abs(wgm_result.wgm)

        cache_hint_to_modifier = {0: ".ca", 1: ".cg", 2: ".cv"}
        cache_a = cache_hint_to_modifier.get(getattr(best_cfg, "cache_hints_a", 0), None)
        cache_b = cache_hint_to_modifier.get(getattr(best_cfg, "cache_hints_b", 0), None)

        print(
            f"BLK_M: {BLK_M}, BLK_N: {BLK_N}, BLK_K: {BLK_K}, gsize_m: {gsize_m}, cache_a: {cache_a}, cache_b: {cache_b}"
        )
        return BLK_M, BLK_N, BLK_K, gsize_m, cache_a, cache_b
    except Exception as e:
        print(f"[gemm/origami] ERROR M={M} N={N} K={K}: {e}")
        return None


@triton.jit()
def _bf16_persistent_gemm_kernel(
    A,
    B,
    C,
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
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
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

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
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
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :].to(tl.int64) * stride_cn
        tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — BF16 GEMM
# ═══════════════════════════════════════════════════════════════════════════════


def gemm_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """General-purpose BF16/FP16 GEMM using optimized persistent kernel.

    Uses origami analytical model for config selection (GROUP_SIZE_M, CHUNK_SIZE)
    when available, otherwise falls back to heuristic.

    Computes: C = op(A) @ op(B), where op(X) = X^T if trans else X.
    If trans_c=True, returns C^T (contiguous, shape N×M).

    Args:
        a: Input matrix (BF16 or FP16).
        b: Input matrix (BF16 or FP16).
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        out_dtype: Output dtype (default bfloat16).
        trans_c: If True, return transposed output C^T (shape N×M).

    Returns:
        C of shape (M, N) if trans_c=False, or (N, M) if trans_c=True.
    """
    assert a.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {a.dtype}"
    assert b.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {b.dtype}"
    # Determine logical (M, K) and (K, N) views
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

    # Handle trans_c by writing to a (N, M) buffer with swapped strides
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

    # Config selection: origami or heuristic (align with TensorAtlas)
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    group_m = _select_group_size_m_bf16(M, N, s_ak, s_bk)
    cache_a, cache_b = None, None
    origami_params = _select_params_origami(
        M,
        N,
        K,
        out_dtype,
        A_view.dtype,
        B_view.dtype,
        trans_a=trans_a,
        trans_b=trans_b,
    )
    if origami_params is not None:
        om, on, ok, ogm, cache_a, cache_b = origami_params
        if (om, on, ok) == (BLOCK_M, BLOCK_N, BLOCK_K):
            group_m = ogm

    # Data-parallel launch (align with TensorAtlas ops/matmul.py)
    total_blocks_M = (M + BLOCK_M - 1) // BLOCK_M
    total_blocks_N = (N + BLOCK_N - 1) // BLOCK_N
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    chunk_size = group_m * group_m
    chunk_size = min(chunk_size, max(1, total_programs // NUM_XCDS))
    even_k = K % BLOCK_K == 0

    args = (A_view, B_view, out, M, N, K, A_view.stride(0), B_view.stride(1), stride_cm, stride_cn)

    _bf16_persistent_gemm_kernel[(total_tiles,)](
        *args,
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=group_m,
        NUM_SMS=total_programs,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        # origami config_t does not expose num_warps/num_stages; use fixed values like TensorAtlas (ops/matmul.py)
        num_warps=8,
        num_stages=2,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
