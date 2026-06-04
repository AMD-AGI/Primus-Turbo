###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Origami analytical GEMM config selection utilities (shared across Triton kernels).

The optional ``origami`` dependency is imported lazily; when it is unavailable the
selector returns ``None`` and a lightweight fallback hardware descriptor is used so
the offline heuristics keep working.
"""

from __future__ import annotations

import atexit
import functools
import math
from dataclasses import dataclass
from typing import Any

import torch

from primus_turbo.pytorch.core.utils import is_gfx950

try:
    import origami
except ModuleNotFoundError:
    origami = None


__all__ = [
    "origama_select_params",
    "origama_hardware_info",
    "origama_calculate_lds_usage",
    "origama_compute_sk_grid",
    "origami_clear_caches",
]


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


def _dtype_bits(dtype: torch.dtype) -> int:
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
# Hardware descriptors
# ═══════════════════════════════════════════════════════════════════════════════

# Per-architecture defaults for LDS capacity (bytes) and max compute clock (KHz).
# Used by origama_hardware_info to avoid calling origami.get_hardware_for_device() which
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
def origama_hardware_info(device_id: int | None = None) -> "_FallbackHardware | Any":
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


def origami_clear_caches() -> None:
    """Release cached nanobind-backed origami objects before interpreter shutdown."""
    origama_hardware_info.cache_clear()
    origama_select_params.cache_clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Stream-K grid heuristic
# ═══════════════════════════════════════════════════════════════════════════════

_SK_TILE_FRACTIONS = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
_SK_SPLIT_FACTORS = [8, 6, 4, 3, 2, 1]
_SK_MAX_WORKSPACE = 128 * 1024 * 1024


def origama_compute_sk_grid(
    M: int,
    N: int,
    K: int,
    BLK_M: int,
    BLK_N: int,
    BLK_K: int,
    cu_count: int,
    elem_bytes_out: int = 2,
) -> int:
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


# ═══════════════════════════════════════════════════════════════════════════════
# LDS usage estimation
# ═══════════════════════════════════════════════════════════════════════════════


def _estimate_lds_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    elem_bytes_a: int,
    elem_bytes_b: int,
    num_stages: int = 2,
) -> int:
    """LDS usage for Triton matmul tile without async_copy."""
    lds_a = block_m * block_k * elem_bytes_a
    lds_b = block_k * block_n * elem_bytes_b
    base_buffers = max(1, num_stages - 1)
    return (lds_a + lds_b) * base_buffers


def _padded_size_32_4(unpadded_size: int) -> int:
    """Triton [[32, 4]] PaddedSharedEncoding — bank-conflict avoidance padding."""
    block_padding = (unpadded_size >> 5) << 2
    if (unpadded_size & 31) == 0 and block_padding >= 4:
        block_padding -= 4
    return unpadded_size + block_padding


def _padded_size_pow2(unpadded_size: int, interval: int, padding: int) -> int:
    """Triton PaddedSharedEncodingAttr.getPaddedSize for a single (interval, padding) pair."""
    log2_interval = (interval - 1).bit_length()
    log2_padding = (padding - 1).bit_length() if padding else 0
    bp = (unpadded_size >> log2_interval) << log2_padding
    if unpadded_size % interval == 0 and bp >= padding:
        bp -= padding
    return unpadded_size + bp


def _estimate_lds_bytes_async_copy(
    block_m: int,
    block_n: int,
    block_k: int,
    elem_bytes_a: int,
    elem_bytes_b: int,
    num_stages: int,
) -> int:
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


def origama_calculate_lds_usage(
    block_m: int,
    block_n: int,
    block_k: int,
    elem_bytes_a: int,
    elem_bytes_b: int,
    num_stages: int,
) -> int:
    """LDS usage with auto-detection of async_copy mode."""
    if is_gfx950():
        return _estimate_lds_bytes_async_copy(
            block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages
        )
    return _estimate_lds_bytes(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages)


# ═══════════════════════════════════════════════════════════════════════════════
# Origami analytical config selection (aligned with TensorAtlas / tritonBLAS)
# ═══════════════════════════════════════════════════════════════════════════════


def _infer_mi_dim(hardware: Any, element_size_a: int, element_size_b: int) -> list[int]:
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


def _get_valid_tiles(
    hardware: Any,
    block_mn_range: list[int],
    block_k_range: list[int],
    mi_dim: list[int],
    elem_bytes_a: int,
    elem_bytes_b: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    """Valid (blk_m, blk_n, blk_k, mi_m, mi_n, mi_k, occ) passing LDS check.

    Uses async_copy-aware LDS estimate on gfx950 with num_stages=2.
    Tiles passing here may still exceed LDS at higher num_stages; callers
    should verify with origama_calculate_lds_usage for their actual num_stages.
    """
    lds_cap = hardware.lds_capacity
    use_async = is_gfx950()
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


def _make_problem(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    mi_dtype_str: str,
    trans_a: bool,
    trans_b: bool,
    mx_block_size: int = 0,
) -> Any:
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


def _tiles_to_configs(
    valid_tiles: list[tuple[int, int, int, int, int, int, int]], streamk: bool = True
) -> list[Any]:
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


def _safe_rank_configs(problem: Any, hardware: Any, configs: list[Any]) -> list[Any]:
    """rank_configs that returns [] instead of raising on unsupported problems."""
    try:
        return origami.rank_configs(problem, hardware, configs)
    except Exception as e:
        from primus_turbo.common.logger import logger

        logger.warning(f"Found Error: {e} in Origama rank_config. Fallback to default config.", once=True)

        return []


@functools.lru_cache(maxsize=4096)
def origama_select_params(
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
    a_dtype: torch.dtype | None = None,
    b_dtype: torch.dtype | None = None,
    trans_a: bool = False,
    trans_b: bool = True,
) -> tuple[int, int, int, int, str | None, str | None] | None:
    """Use origami rank_configs + select_workgroup_mapping (align with TensorAtlas selector.py).

    trans_a, trans_b: logical layout (op(A) @ op(B)). Forward NT = (False, True);
    backward grad_a (NN) = (False, False); backward grad_b (TN) = (True, False).
    Returns (block_m, block_n, block_k, group_size_m, cache_a, cache_b) or None.
    """
    if origami is None:
        return None

    a_dtype = a_dtype if a_dtype is not None else out_dtype
    b_dtype = b_dtype if b_dtype is not None else out_dtype

    hardware = origama_hardware_info()

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
    sk_grid = origama_compute_sk_grid(M, N, K, BLK_M, BLK_N, BLK_K, hardware.N_CU, elem_bytes_out)
    wgm_result = origami.select_workgroup_mapping(problem, hardware, best_cfg, sk_grid)
    gsize_m = abs(wgm_result.wgm)

    _CACHE_HINT_TO_MODIFIER = {0: ".ca", 1: ".cg", 2: ".cv"}
    cache_a = _CACHE_HINT_TO_MODIFIER.get(getattr(best_cfg, "cache_hints_a", 0), None)
    cache_b = _CACHE_HINT_TO_MODIFIER.get(getattr(best_cfg, "cache_hints_b", 0), None)
    # print(
    #     f"BLK_M: {BLK_M}, BLK_N: {BLK_N}, BLK_K: {BLK_K}, gsize_m: {gsize_m}, cache_a: {cache_a}, cache_b: {cache_b}"
    # )
    return BLK_M, BLK_N, BLK_K, gsize_m, cache_a, cache_b


atexit.register(origami_clear_caches)
