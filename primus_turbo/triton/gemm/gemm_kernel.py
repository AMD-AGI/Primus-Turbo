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

import atexit
import functools
import math
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

try:
    import origami
except ModuleNotFoundError:
    origami = None

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

# Per-architecture defaults for LDS capacity (bytes) and max compute clock (KHz).
# Used by _get_hardware to avoid calling origami.get_hardware_for_device() which
# internally invokes HIP C APIs that can segfault in certain Docker / distributed
# training environments due to HIP runtime double-initialization conflicts.
_ARCH_HW_DEFAULTS: dict[str, tuple[int, int]] = {
    "gfx950": (163840, 2400000),  # MI350X / MI355X  — 160 KB LDS, 2.4 GHz
    "gfx942": (65536, 2100000),  # MI300X / MI300A  —  64 KB LDS, 2.1 GHz
}


@dataclass(frozen=True)
class _FallbackHardware:
    N_CU: int
    lds_capacity: int
    l2_cache_size: int = 0
    clock_khz: int = 0


@functools.lru_cache(maxsize=8)
def _get_hardware(device_id=None):
    """Cached hardware descriptor for Triton GEMM config selection.

    When origami is available, return its hardware_t object and keep the
    existing analytical selector behavior. Otherwise, fall back to a lightweight
    descriptor built from torch device properties so offline heuristics still
    work without the optional origami dependency.
    """
    if device_id is None:
        device_id = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device_id)
    arch_full = getattr(props, "gcnArchName", "")  # e.g. "gfx950:sramecc+:xnack-"
    arch_base = arch_full.split(":")[0]  # e.g. "gfx950"
    default_lds_capacity, default_clock_khz = _ARCH_HW_DEFAULTS.get(
        arch_base,
        (getattr(props, "shared_memory_per_block", 65536), getattr(props, "clock_rate", 0)),
    )

    if origami is None:
        return _FallbackHardware(
            N_CU=props.multi_processor_count,
            lds_capacity=default_lds_capacity,
            l2_cache_size=getattr(props, "L2_cache_size", 0),
            clock_khz=default_clock_khz,
        )

    arch_enum = getattr(origami.architecture_t, arch_base, None)

    if arch_enum is not None and arch_base in _ARCH_HW_DEFAULTS:
        return origami.get_hardware_for_arch(
            arch_enum,
            props.multi_processor_count,
            default_lds_capacity,
            props.L2_cache_size,
            default_clock_khz,
        )

    return origami.get_hardware_for_device(device_id)


def clear_origami_caches() -> None:
    """Release cached nanobind-backed origami objects before interpreter shutdown."""
    _get_hardware.cache_clear()
    _select_params_origami.cache_clear()


_SK_TILE_FRACTIONS = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
_SK_SPLIT_FACTORS = [8, 6, 4, 3, 2, 1]
_SK_MAX_WORKSPACE = 128 * 1024 * 1024


def _compute_sk_grid(M, N, K, BLK_M, BLK_N, BLK_K, cu_count, elem_bytes_out=2):
    tiles = math.ceil(M / BLK_M) * math.ceil(N / BLK_N)
    sk_grid = tiles
    iters_per_tile = max(1, math.ceil(K / BLK_K))

    if tiles > cu_count:
        min_even_tiles = tiles / cu_count
        for frac in _SK_TILE_FRACTIONS:
            frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)
            partial_size = BLK_M * BLK_N * elem_bytes_out * frac_grid
            if tiles % frac_grid != 0 and partial_size > _SK_MAX_WORKSPACE:
                continue
            if frac_grid <= cu_count:
                sk_grid = frac_grid
                break
    elif tiles < cu_count:
        for factor in _SK_SPLIT_FACTORS:
            split_grid = tiles * factor
            iters_per_cu = iters_per_tile // factor
            if split_grid <= cu_count and iters_per_cu >= 8:
                sk_grid = split_grid
                break

    if tiles % sk_grid != 0:
        sk_grid = tiles

    if tiles >= cu_count and cu_count in (304, 80, 64):
        last_wave_remainder = tiles % cu_count
        if 0 < last_wave_remainder < 128:
            sk_grid = 256 if cu_count == 304 else 64

    return sk_grid


@functools.lru_cache(maxsize=1)
def _is_gfx950() -> bool:
    """Check if current GPU is gfx950 (CDNA4 / MI350X / MI355X)."""
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target is not None and target.backend == "hip" and target.arch == "gfx950"
    except (AttributeError, TypeError):
        return False


_KNOBS_SET = False


def _set_knobs_gfx950():
    """Enable AMD compiler knobs for gfx950 (async_copy, block_pingpong, scalarize)."""
    global _KNOBS_SET
    if _KNOBS_SET:
        return
    _KNOBS_SET = True
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "amd"):
        triton.knobs.amd.use_async_copy = True
        triton.knobs.amd.scalarize_packed_fops = True
        triton.knobs.amd.use_block_pingpong = True
    else:
        os.environ.setdefault("TRITON_HIP_USE_ASYNC_COPY", "1")
        os.environ.setdefault("AMDGCN_SCALARIZE_PACKED_FOPS", "1")
        os.environ.setdefault("TRITON_HIP_USE_BLOCK_PINGPONG", "1")


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


def offline_select_bf16(M, N, K, s_ak, s_bk):
    """BF16 config selection from MI300X bench data (out_bf16_gemm.yaml, 186 entries).

    Stride → layout:
      NT (trans_a=False, trans_b=True):  s_ak=1, s_bk=1   → C = A @ B^T
      NN (trans_a=False, trans_b=False): s_ak=1, s_bk≠1   → C = A @ B
      TN (trans_a=True,  trans_b=False): s_ak≠1, s_bk≠1   → C = A^T @ B
      TT (trans_a=True,  trans_b=True):  s_ak≠1, s_bk=1   → C = A^T @ B^T

    Returns (BM, BN, BK, GM, NUM_SMS, CHUNK, CA, CB).
    """
    # ── Block sizes (256×256×64 covers ~93% of bench entries) ──
    BM, BN, BK = 256, 256, 64

    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN
    total_tiles = tiles_m * tiles_n

    cu_count = _get_hardware().N_CU

    # ── NUM_SMS ──
    # Small grids: sk_grid for wave efficiency (persistent, NUM_SMS=256/304)
    # Large grids: data-parallel (NUM_SMS=total_tiles) to keep all CUs busy
    if total_tiles <= cu_count * 4:
        num_sms = _compute_sk_grid(M, N, K, BM, BN, BK, cu_count)
    else:
        num_sms = total_tiles

    # ── GROUP_SIZE_M ──
    if min(tiles_m, tiles_n) < 16:
        group_m = 8
    else:
        group_m = 4

    # ── CHUNK_SIZE ──
    # persistent mode: small chunks for XCD load-balance
    # data-parallel: 64 for large tile counts, 32 for small
    if num_sms < total_tiles:
        chunk = min(32, max(1, num_sms // NUM_XCDS))
    else:
        chunk = 64 if total_tiles > 1024 else 32

    return BM, BN, BK, group_m, num_sms, chunk, ".ca", ".ca"


# ─── Origami analytical config selection (aligned with TensorAtlas / tritonBLAS) ───


def _estimate_lds_bytes(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages=2):
    """LDS usage for Triton matmul tile without async_copy."""
    lds_a = block_m * block_k * elem_bytes_a
    lds_b = block_k * block_n * elem_bytes_b
    base_buffers = max(1, num_stages - 1)
    return (lds_a + lds_b) * base_buffers


def _padded_size_32_4(unpadded_size):
    """Triton [[32, 4]] PaddedSharedEncoding — bank-conflict avoidance padding."""
    block_padding = (unpadded_size >> 5) << 2
    if (unpadded_size & 31) == 0 and block_padding >= 4:
        block_padding -= 4
    return unpadded_size + block_padding


def _padded_size_pow2(unpadded_size, interval, padding):
    """Triton PaddedSharedEncodingAttr.getPaddedSize for a single (interval, padding) pair."""
    log2_interval = (interval - 1).bit_length()
    log2_padding = (padding - 1).bit_length() if padding else 0
    bp = (unpadded_size >> log2_interval) << log2_padding
    if unpadded_size % interval == 0 and bp >= padding:
        bp -= padding
    return unpadded_size + bp


def _estimate_lds_bytes_async_copy(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages):
    """LDS usage with async_copy (PaddedSharedEncoding + num_stages buffers).

    Matches tritonBLAS origami.estimate_triton_lds_bytes / triton_bench calculate_lds_usage.
    """
    elem_a = block_m * block_k
    elem_b = block_k * block_n
    padded_a = _padded_size_32_4(elem_a)
    padded_b = _padded_size_32_4(elem_b)
    if block_k & (block_k - 1) == 0:
        pa = _padded_size_pow2(elem_a, block_k, 8)
        if pa > padded_a:
            padded_a = pa
    if block_n & (block_n - 1) == 0:
        pb = _padded_size_pow2(elem_b, block_n, 8)
        if pb > padded_b:
            padded_b = pb
    return num_stages * (padded_a * elem_bytes_a + padded_b * elem_bytes_b)


def _calculate_lds_usage(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages):
    """LDS usage with auto-detection of async_copy mode."""
    if _is_gfx950():
        return _estimate_lds_bytes_async_copy(
            block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages
        )
    return _estimate_lds_bytes(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages)


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
    """Valid (blk_m, blk_n, blk_k, mi_m, mi_n, mi_k, occ) passing LDS check.

    Uses async_copy-aware LDS estimate on gfx950 with num_stages=2.
    Tiles passing here may still exceed LDS at higher num_stages; callers
    should verify with _calculate_lds_usage for their actual num_stages.
    """
    lds_cap = hardware.lds_capacity
    use_async = _is_gfx950()
    valid = []
    for bm, bn, bk in (
        (bm, bn, bk) for bm in block_mn_range for bn in block_mn_range for bk in block_k_range
    ):
        if use_async:
            lds = _estimate_lds_bytes_async_copy(bm, bn, bk, elem_bytes_a, elem_bytes_b, num_stages=2)
        else:
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


def _safe_rank_configs(problem, hardware, configs):
    """rank_configs that returns [] instead of raising on unsupported problems."""
    try:
        return origami.rank_configs(problem, hardware, configs)
    except RuntimeError:
        return []


@functools.lru_cache(maxsize=4096)
def _select_params_origami(M, N, K, out_dtype, a_dtype=None, b_dtype=None, trans_a=False, trans_b=True):
    """Use origami rank_configs + select_workgroup_mapping (align with TensorAtlas selector.py).

    trans_a, trans_b: logical layout (op(A) @ op(B)). Forward NT = (False, True);
    backward grad_a (NN) = (False, False); backward grad_b (TN) = (True, False).
    Returns (block_m, block_n, block_k, group_size_m, cache_a, cache_b) or None.
    """
    if origami is None:
        return None

    a_dtype = a_dtype if a_dtype is not None else out_dtype
    b_dtype = b_dtype if b_dtype is not None else out_dtype

    hardware = _get_hardware()

    elem_bits_a = _dtype_bits(a_dtype)
    elem_bits_b = _dtype_bits(b_dtype)
    elem_bytes_a = elem_bits_a // 8
    elem_bytes_b = elem_bits_b // 8

    input_dtype_for_mi = a_dtype if elem_bits_a <= elem_bits_b else b_dtype
    mi_dtype_str = _ORIGAMI_DTYPE_TO_STR.get(input_dtype_for_mi, _ORIGAMI_DTYPE_TO_STR.get(out_dtype, "bf16"))

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

    ranked = _safe_rank_configs(problem, hardware, configs)
    if not ranked:
        return None
    best_result = ranked[0]
    best_cfg = best_result.config if hasattr(best_result, "config") else best_result
    BLK_M = best_cfg.mt.m
    BLK_N = best_cfg.mt.n
    BLK_K = best_cfg.mt.k

    elem_bytes_out = _dtype_bits(out_dtype) // 8
    sk_grid = _compute_sk_grid(M, N, K, BLK_M, BLK_N, BLK_K, hardware.N_CU, elem_bytes_out)
    wgm_result = origami.select_workgroup_mapping(problem, hardware, best_cfg, sk_grid)
    gsize_m = abs(wgm_result.wgm)

    _CACHE_HINT_TO_MODIFIER = {0: ".ca", 1: ".cg", 2: ".cv"}
    cache_a = _CACHE_HINT_TO_MODIFIER.get(getattr(best_cfg, "cache_hints_a", 0), None)
    cache_b = _CACHE_HINT_TO_MODIFIER.get(getattr(best_cfg, "cache_hints_b", 0), None)
    # print(
    #     f"BLK_M: {BLK_M}, BLK_N: {BLK_N}, BLK_K: {BLK_K}, gsize_m: {gsize_m}, cache_a: {cache_a}, cache_b: {cache_b}"
    # )
    return BLK_M, BLK_N, BLK_K, gsize_m, cache_a, cache_b


atexit.register(clear_origami_caches)


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

    Uses offline heuristic for block sizes / NUM_SMS, then origami analytical
    model to override GROUP_SIZE_M and cache modifiers.

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

    if _is_gfx950():
        _set_knobs_gfx950()

        # gfx950 BF16 config from 164-entry tuning data.
        # TN layout with large K → BLK_K=64, stages=2; all other cases → 32/3.
        # Small TN (K≤3584, dims≤16384, min dim≤4608) stays on 32/3.
        is_tn = (s_ak == 1) and (s_bk == 1)
        use_bk64 = is_tn and (K > 3584 or min(M, N) > 4608 or max(M, N) > 16384)

        BLOCK_M, BLOCK_N = 256, 256
        BLOCK_K, num_stages = (64, 2) if use_bk64 else (32, 3)
        chunk_size, waves_per_eu = 32, 0
        cache_a, cache_b = ".ca", ".ca"

        tiles_m = (M + BLOCK_M - 1) // BLOCK_M
        tiles_n = (N + BLOCK_N - 1) // BLOCK_N
        min_tile = min(tiles_m, tiles_n)
        group_m = 7 if min_tile < 16 else 4

        cu_count = _get_hardware().N_CU

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
            om, on, ok, ogm, oca, ocb = origami_params
            if min(om, on) >= 128 and ok == BLOCK_K:
                BLOCK_M, BLOCK_N, group_m = om, on, ogm

        # Occupancy: when TN BLK_K=64 tiles land in 1–2 wave zone, halve
        # BLOCK_N for better CU utilisation (keeps BLOCK_M=256 for A locality).
        if use_bk64:
            tm = (M + BLOCK_M - 1) // BLOCK_M
            tn = (N + BLOCK_N - 1) // BLOCK_N
            if cu_count < tm * tn < 2 * cu_count and tn >= tm:
                new_tn = (N + 127) // 128
                if tm * new_tn >= 2 * cu_count:
                    BLOCK_N, group_m = 128, 8

        num_sms = _compute_sk_grid(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, cu_count)
    else:
        # ── gfx942 path (unchanged) ──────────────────────────────────────────
        BLOCK_M, BLOCK_N, BLOCK_K, group_m, num_sms, chunk_size, cache_a, cache_b = offline_select_bf16(
            M, N, K, s_ak, s_bk
        )
        num_stages, waves_per_eu = 2, 0
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
            om, on, ok, ogm, oca, ocb = origami_params
            if (om, on, ok) == (BLOCK_M, BLOCK_N, BLOCK_K):
                group_m = ogm
                cache_a, cache_b = oca, ocb

    even_k = K % BLOCK_K == 0
    args = (A_view, B_view, out, M, N, K, A_view.stride(0), B_view.stride(1), stride_cm, stride_cn)

    _bf16_persistent_gemm_kernel[(num_sms,)](
        *args,
        stride_ak=s_ak,
        stride_bk=s_bk,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
