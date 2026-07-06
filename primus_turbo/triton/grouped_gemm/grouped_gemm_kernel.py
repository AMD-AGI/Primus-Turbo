###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped GEMM Triton persistent kernels (CPU-sync-free) -- BF16/FP16.

Static-stride and work-stealing variants share per-tile helpers and live
in the same module.

Contains:
  - _process_grouped_gemm_tile             -- per-tile compute body (forward)
  - _grouped_bf16_persistent_gemm_kernel   -- BF16/FP16 forward, static stride
  - _grouped_bf16_persistent_gemm_kernel_ws -- BF16/FP16 forward, work stealing
  - _process_variable_k_tile               -- per-tile compute body (variable-K)
  - _grouped_variable_k_gemm_kernel        -- variable-K backward, static stride
  - _grouped_variable_k_gemm_kernel_ws     -- variable-K backward, work stealing

Public API:
  - grouped_gemm_triton_kernel            -- BF16/FP16 forward (work_steal kwarg)
  - grouped_gemm_variable_k_triton_kernel -- variable-K backward (work_steal kwarg)

FP8 kernels (tensorwise + blockwise) are in grouped_gemm_fp8_kernel.py.

Environment variable: PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON activates these kernels.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import get_num_cus, is_gfx950
from primus_turbo.triton.utils.origami import origama_select_params
from primus_turbo.triton.utils.triton_knobs_helper import scoped_amd_knobs

# ===============================================================================
# Hardware constants (lazy init)
# ===============================================================================

NUM_XCDS = 8
# Padding for the work-stealing per-XCD atomic counter slots: 64 * 4B = 256 B
# = one MI355X L2 line per slot, so the eight XCDs do not false-share a cache
# line. Used by the WS variants of the persistent kernels below.
COUNTER_STRIDE = 64

# ----------------------------------------------------------------------------
# Per-device cached counter buffer for the work-stealing kernels.
#
# Layout: [xcd0_slot, ..., xcd7_slot, global_slot], each slot
# COUNTER_STRIDE int32 elements wide. The kernel zeros the buffer before each
# launch on the active stream.
#
# Caveat: a singleton per device is not stream-safe. Concurrent WS launches
# on different streams of the same device would race on the same slots. Safe
# under the typical single-stream autograd graph; multi-stream callers should
# pass an explicit ``ws_counter`` to the kernel entry points.
# ----------------------------------------------------------------------------
_triton_ws_counters: dict[torch.device, torch.Tensor] = {}


def _get_triton_ws_counter(device: torch.device) -> torch.Tensor:
    """Return the per-device WS counter buffer, allocating on first use."""
    buf = _triton_ws_counters.get(device)
    if buf is None:
        buf = torch.zeros(
            (NUM_XCDS + 1) * COUNTER_STRIDE,
            dtype=torch.int32,
            device=device,
        )
        _triton_ws_counters[device] = buf
    return buf


# ===============================================================================
# Cached config selection (avoids per-call origami / LDS overhead)
# ===============================================================================


@functools.lru_cache(maxsize=256)
def _get_gg_bf16_fwd_config(avg_m, N, K, dtype_a, dtype_b, trans_b, G, num_sms):
    """Cached kernel config for BF16 grouped GEMM forward."""
    if is_gfx950():
        is_tn = not trans_b
        BLOCK_M, BLOCK_N = 256, 256
        if is_tn:
            BLOCK_K, num_stages_val = 64, 2
        else:
            BLOCK_K, num_stages_val = 32, 3
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            dtype_a,
            dtype_a,
            dtype_b,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            if min(om, on) >= 128 and ok == BLOCK_K:
                BLOCK_M, BLOCK_N, group_m = om, on, ogm
                cache_a, cache_b = oc_a, oc_b
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
        group_m = 4
        num_stages_val = 2
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 64 if num_sms >= NUM_XCDS * 64 else 32

        origami_params = origama_select_params(
            avg_m,
            N,
            K,
            dtype_a,
            dtype_a,
            dtype_b,
            trans_a=False,
            trans_b=trans_b,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            if ogm >= 2 and om * on >= 256 * 256:
                BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b = (om, on, ok, ogm, oc_a, oc_b)

    return BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size


@functools.lru_cache(maxsize=256)
def _get_gg_bf16_vk_config(OUT_M, OUT_N, avg_k, dtype_lhs, dtype_rhs, G, num_sms):
    """Cached kernel config for BF16 grouped GEMM variable-K backward."""
    if is_gfx950():
        BLOCK_M, BLOCK_N = 256, 256
        BLOCK_K, num_stages_val = 32, 3
        group_m = 4
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 32

        origami_params = origama_select_params(
            OUT_M,
            OUT_N,
            avg_k,
            dtype_lhs,
            dtype_lhs,
            dtype_rhs,
            trans_a=True,
            trans_b=False,
        )
        if origami_params is not None:
            om, on, ok, ogm, oc_a, oc_b = origami_params
            if min(om, on) >= 128 and ok == BLOCK_K:
                BLOCK_M, BLOCK_N, group_m = om, on, ogm
                cache_a, cache_b = oc_a, oc_b
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
        group_m = 4
        num_stages_val = 2
        cache_a, cache_b = ".ca", ".ca"
        chunk_size = 64 if num_sms >= NUM_XCDS * 64 else 32

    return BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size


# ===============================================================================
# Chiplet Transform (shared helper)
# ===============================================================================


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


# ===============================================================================
# Grouped GEMM -- Persistent Kernel (CPU-sync-free)
#
# Computes: out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g]
#   for g = 0, 1, ..., G-1
#
# Design:
#   Single persistent kernel processes ALL groups x ALL tiles.
#   Each CU computes total_tiles and group mapping on-the-fly via O(G)
#   linear scan of group_offs (G is small, <=256, data cached in L2).
#   Zero CPU synchronization -- group_offs read entirely on GPU.
# ===============================================================================


# -----------------------------------------------------------------------------
# Per-tile compute body -- lifted into a @triton.jit helper so both the static
# and work-stealing persistent loops can call it.
# -----------------------------------------------------------------------------


@triton.jit
def _process_grouped_gemm_tile(
    global_tile_id,
    A,
    B,
    C,
    group_offs_ptr,
    G,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    stride_cn,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Compute one output tile given its global tile id within the persistent loop."""
    # -- Find group via linear scan (O(G)) --
    group_idx: tl.int32 = 0
    tile_start: tl.int32 = 0
    cumsum: tl.int32 = 0
    for _g in range(G):
        m_g_i = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
        new_cumsum = cumsum + tiles_g
        if global_tile_id >= new_cumsum:
            group_idx = _g + 1
            tile_start = new_cumsum
        cumsum = new_cumsum

    # Defensive bound: when global_tile_id >= total_tiles (cumsum after the
    # loop), group_idx would have walked off the end of group_offs and the
    # subsequent loads at group_offs_ptr + group_idx + 1 would be OOB. All
    # current callers bound-check, so this is a guard against future misuse.
    if global_tile_id >= cumsum:
        return

    # -- Group-local tile -> (pid_m, pid_n) with GROUP_SIZE_M swizzle --
    local_tile = global_tile_id - tile_start
    m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
    M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
    tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    swizzle_group = local_tile // num_pid_in_group
    first_pid_m = swizzle_group * GROUP_SIZE_M
    group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
    pid_n = (local_tile % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # -- Address computation --
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Cast group_idx to int64 to prevent overflow in B group offset
    group_offset_b = group_idx.to(tl.int64) * stride_bg

    A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # -- K-loop --
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    # ``tl.assume(loop_k > 1)`` would be a false assertion for shapes where
    # K <= BLOCK_SIZE_K (loop_k == 1 when EVEN_K, 0 when not). Relax to a
    # condition that always holds, so the compiler can still know the loop
    # count is non-negative without risking miscompilation on tiny K.
    tl.assume(loop_k >= 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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
        rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
        B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
        if stride_ak == 1:
            A_LAST = tl.multiple_of(A_LAST, (1, 16))
        else:
            A_LAST = tl.multiple_of(A_LAST, (16, 1))
        if stride_bk == 1:
            B_LAST = tl.multiple_of(B_LAST, (16, 1))
        else:
            B_LAST = tl.multiple_of(B_LAST, (1, 16))
        a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # -- Store --
    c = acc.to(C.type.element_ty)
    rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
    C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
    tl.store(C_, c, c_mask)


@triton.jit()
def _grouped_bf16_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K]
    B,  # [G, ?, ?]  -- (K,N) or (N,K) depending on trans_b
    C,  # [M_total, N]
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,  # number of groups (runtime)
    N,
    K,
    # Strides
    stride_am,  # A row stride
    stride_bg,  # B group stride: b.stride(0)
    stride_bn,  # B N-stride (within a group)
    stride_cm,  # C row stride
    stride_cn,  # C col stride
    # Constexpr strides (for compiler optimisation)
    stride_ak: tl.constexpr,  # A K-stride (=1 when trans_a=False, contiguous)
    stride_bk: tl.constexpr,  # B K-stride (=1 when trans_b=True)
    # Tile config
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
    """Persistent grouped GEMM kernel (CPU-sync-free) -- static stride.

    One kernel launch processes ALL groups x ALL tiles.
    Each persistent CU strides through global tile IDs (block_id += grid_size)
    and dispatches per-tile work via ``_process_grouped_gemm_tile``.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # -- Compute total tiles across all groups (O(G) per CU) --
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        _process_grouped_gemm_tile(
            global_tile_id,
            A,
            B,
            C,
            group_offs_ptr,
            G,
            N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_cm,
            stride_cn,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            ALLOW_TF32=ALLOW_TF32,
        )


# -----------------------------------------------------------------------------
# Work-stealing persistent kernel.
#
# Replaces the static stride (block_id += grid_size) with an atomicAdd-based
# tile claim. Phase 1: each XCD has a dedicated counter and CTAs on that XCD
# race to claim up to ``local_per_xcd`` tiles in the contiguous slice
# ``[xcd_id * local_per_xcd, (xcd_id + 1) * local_per_xcd)`` -- preserves L2
# locality (consecutive tiles share B[g] data). Phase 2: tiles past the
# per-XCD budget are claimed from a single global counter by whichever CU
# finishes phase 1 first -- cross-XCD work stealing.
#
# Special cases of ``local_per_xcd``:
#   = ceil(total_tiles / NUM_XCDS) -> per-XCD only (phase 2 empty)
#   = 0                            -> global only (phase 1 empty)
#   = anything in between          -> hierarchical
#
# Counter buffer layout (caller-allocated int32, length
# ``(NUM_XCDS + 1) * COUNTER_STRIDE``):
#   tile_counter_ptr   = base
#   tile_counter_ptr + xcd_id * COUNTER_STRIDE -> per-XCD slots
#   global_counter_ptr = tile_counter_ptr + NUM_XCDS * COUNTER_STRIDE
# The host wrapper zeros it on the active stream before each launch.
# -----------------------------------------------------------------------------


@triton.jit()
def _grouped_bf16_persistent_gemm_kernel_ws(
    A,
    B,
    C,
    group_offs_ptr,
    tile_counter_ptr,
    global_counter_ptr,
    G,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    stride_cn,
    local_per_xcd,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent grouped GEMM with per-XCD + global-fallback work stealing."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # -- Compute total tiles across all groups (O(G) per CU) --
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # AMD pid -> XCD mapping is round-robin. When the persistent grid is
    # capped to fewer CUs than there are XCDs (NUM_SMS < NUM_XCDS), only
    # XCD ids [0, NUM_SMS) ever issue phase-1 claims, so both the per-XCD
    # slot count AND the phase-1 ID span must use min(NUM_SMS, NUM_XCDS).
    # Otherwise phase 2 starts past where phase 1 actually ended and the
    # tiles in the gap are silently dropped. The public ``grouped_gemm``
    # API rejects num_cu != None + schedule="work_steal" so callers should
    # never hit this branch from the high-level op, but the kernel-level
    # entry point exposes num_cu directly -- belt-and-braces.
    ACTIVE_XCDS: tl.constexpr = min(NUM_SMS, NUM_XCDS)
    xcd_id = pid % ACTIVE_XCDS
    local_counter = tile_counter_ptr + xcd_id * COUNTER_STRIDE
    per_xcd = local_per_xcd.to(tl.int32)
    phase1_total = (per_xcd * ACTIVE_XCDS).to(tl.int32)

    # Single unified loop with one call site for the per-tile body.
    # (Inlining the per-tile body twice -- once per phase -- produced phase-2
    # NaN on the Triton-AMD backend even with no register spilling reported.
    # Folding both phases into one while loop sidesteps the issue.)
    local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
    in_phase2 = local_idx >= per_xcd
    if in_phase2:
        g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
        tile_id = phase1_total + g_idx
    else:
        tile_id = xcd_id * per_xcd + local_idx

    while tile_id < total_tiles:
        _process_grouped_gemm_tile(
            tile_id,
            A,
            B,
            C,
            group_offs_ptr,
            G,
            N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_cm,
            stride_cn,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            ALLOW_TF32=ALLOW_TF32,
        )
        if in_phase2:
            g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
            tile_id = phase1_total + g_idx
        else:
            local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
            if local_idx >= per_xcd:
                in_phase2 = True
                g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
                tile_id = phase1_total + g_idx
            else:
                tile_id = xcd_id * per_xcd + local_idx


@scoped_amd_knobs
def grouped_gemm_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    *,
    num_cu: int | None = None,
    work_steal: bool = False,
    ws_mode: str = "auto",
    ws_counter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persistent grouped GEMM (CPU-sync-free) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g]
    for g = 0, ..., G-1, where B_view[g] is b[g] or b[g]^T depending on trans_b.

    Single kernel launch, zero CPU synchronization.

    Args:
        a: [M_total, K] BF16/FP16 input (trans_a=False always).
        b: [G, K, N] or [G, N, K] (if trans_b) BF16/FP16 weights.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).
        num_cu: Cap the persistent-grid size at this many CUs (clamped to the
            device's CU count). ``None`` uses every CU on the device.
        work_steal: When True, dispatch to the WS variant of the persistent
            kernel -- CTAs claim tiles via ``tl.atomic_add`` so fast CUs absorb
            work that slow CUs would have done (straggler tolerance under
            collective overlap). Default False (static stride).
        ws_mode: WS scheduling policy when ``work_steal=True`` -- one of
            ``"auto"`` (default, kernel-aware heuristic), ``"per-xcd"``,
            ``"global"``, ``"hierarchical"``. Ignored when ``work_steal=False``.
        ws_counter: Optional int32 buffer of length
            ``(NUM_XCDS + 1) * COUNTER_STRIDE``. When None and ``work_steal``
            is True, a per-device singleton buffer is used (see the wrapper
            in ``grouped_gemm_impl.py``). The singleton is **not stream-safe**:
            concurrent WS launches on different streams of the same device
            would race on the same counter slots. Safe under the typical
            single-stream autograd graph; for multi-stream concurrent use,
            pass an explicit per-stream ``ws_counter``.

    Returns:
        [M_total, N] BF16/FP16 output.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {a.dtype}"
    assert b.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {b.dtype}"

    M_total, K_a = a.shape
    G = b.shape[0]

    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)  # K is the fast dimension (=1 for contiguous)
        stride_bn = b.stride(1)  # N-stride
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)  # K-stride
        stride_bn = b.stride(2)  # N is the fast dimension (=1 for contiguous)

    assert K_a == K_b, f"K mismatch: a has K={K_a}, b has K={K_b}"
    K = K_a

    stride_bg = b.stride(0)  # Group stride
    stride_ak = a.stride(1)  # =1 for contiguous a

    # Output
    out = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

    # Kernel config (cached -- origami + LDS check run only on first call per shape)
    device_num_cus = get_num_cus()
    num_sms = min(num_cu, device_num_cus) if num_cu is not None and num_cu > 0 else device_num_cus
    avg_m = max(M_total // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size = (
        _get_gg_bf16_fwd_config(avg_m, N, K, a.dtype, b.dtype, trans_b, G, num_sms)
    )
    even_k = K % BLOCK_K == 0

    if work_steal:
        # Resolve ws_mode -> numeric local_per_xcd via the sync-free metadata
        # heuristic (uses a.size(0) and group_offs shape -- no tensor reads).
        from primus_turbo.triton.grouped_gemm.ws_triton_heuristic import (
            approximate_triton_total_tiles,
            resolve_triton_ws_local_per_xcd,
        )

        if ws_counter is None:
            # Default: per-device singleton (not stream-safe; see module docstring).
            ws_counter = _get_triton_ws_counter(a.device)
        expected = (NUM_XCDS + 1) * COUNTER_STRIDE
        if ws_counter.numel() < expected or ws_counter.dtype != torch.int32 or not ws_counter.is_cuda:
            raise ValueError(
                f"work_steal=True requires an int32 ws_counter on CUDA with at least "
                f"{expected} elements; got numel={ws_counter.numel()}, "
                f"dtype={ws_counter.dtype}, device={ws_counter.device}"
            )
        ws_counter.zero_()
        total_tiles = approximate_triton_total_tiles(
            a.shape[0],
            group_offs.shape[0] - 1,
            N,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
        )
        local_per_xcd = resolve_triton_ws_local_per_xcd(
            ws_mode,
            total_tiles,
            num_sms,
            num_xcds=NUM_XCDS,
        )
        _grouped_bf16_persistent_gemm_kernel_ws[(num_sms,)](
            a,
            b,
            out,
            group_offs,
            ws_counter,
            ws_counter[NUM_XCDS * COUNTER_STRIDE :],
            G,
            N,
            K,
            a.stride(0),
            stride_bg,
            stride_bn,
            out.stride(0),
            out.stride(1),
            local_per_xcd,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=group_m,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=cache_a,
            CACHE_MODIFIER_B=cache_b,
            COUNTER_STRIDE=COUNTER_STRIDE,
            num_warps=8,
            num_stages=num_stages_val,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=1,
        )
        return out

    _grouped_bf16_persistent_gemm_kernel[(num_sms,)](
        a,
        b,
        out,
        group_offs,
        G,
        N,
        K,
        a.stride(0),
        stride_bg,
        stride_bn,
        out.stride(0),
        out.stride(1),
        stride_ak=stride_ak,
        stride_bk=stride_bk,
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
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


# ===============================================================================
# Grouped Variable-K GEMM -- Persistent Kernel (backward pass, CPU-sync-free)
#
# Computes: C[g] = LHS[offs[g]:offs[g+1]]^T @ RHS[offs[g]:offs[g+1]] [* scale]
#   for g = 0, 1, ..., G-1
#
# Used in backward pass where both LHS and RHS are 2D tensors sliced by groups.
# Output is 3D: [G, OUT_M, OUT_N].
# All groups share the same output dimensions; only the inner product dim (M_g)
# varies per group, making group->tile mapping a simple div/mod.
# ===============================================================================


# -----------------------------------------------------------------------------
# Per-tile compute body for the variable-K kernel -- shared between the static
# and work-stealing persistent loops.
# -----------------------------------------------------------------------------


@triton.jit
def _process_variable_k_tile(
    global_tile,
    LHS,
    RHS,
    C,
    scale,  # per-CTA constant; ignored when IS_FP8=False
    group_offs_ptr,
    OUT_M,
    OUT_N,
    stride_lhs_m,
    stride_rhs_m,
    stride_cg,
    stride_cm,
    stride_cn,
    tiles_m,
    tiles_n,
    tiles_per_group,
    stride_lhs_n: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_FP8: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Compute one variable-K output tile given its global tile id."""
    # -- Map to (group, local_tile) -- simple div/mod, O(1) --
    group_idx = global_tile // tiles_per_group
    local_tile = global_tile - group_idx * tiles_per_group

    # -- GROUP_SIZE_M swizzle on local_tile -> (pid_m, pid_n) --
    num_pid_in_group = GROUP_SIZE_M * tiles_n
    swizzle_group = local_tile // num_pid_in_group
    first_pid_m = swizzle_group * GROUP_SIZE_M
    group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
    pid_n = (local_tile % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # -- Group boundaries --
    m_start = tl.load(group_offs_ptr + group_idx)
    M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

    # -- Output indices --
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # -- Base pointers (LHS^T[rm, rk] = LHS[m_start + rk, rm]) --
    LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
    RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

    # -- K-loop over M_g (variable per group, always masked) --
    loop_k = tl.cdiv(M_g, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(loop_k):
        k_start = k * BLOCK_SIZE_K
        mask_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) < M_g

        if stride_lhs_n == 1:
            a = tl.load(
                tl.multiple_of(LHS_BASE, (16, 1)),
                mask=mask_k[None, :],
                other=0.0,
                cache_modifier=CACHE_MODIFIER_A,
            )
        else:
            a = tl.load(
                tl.multiple_of(LHS_BASE, (1, 16)),
                mask=mask_k[None, :],
                other=0.0,
                cache_modifier=CACHE_MODIFIER_A,
            )
        if stride_rhs_n == 1:
            b = tl.load(
                tl.multiple_of(RHS_BASE, (1, 16)),
                mask=mask_k[:, None],
                other=0.0,
                cache_modifier=CACHE_MODIFIER_B,
            )
        else:
            b = tl.load(
                tl.multiple_of(RHS_BASE, (16, 1)),
                mask=mask_k[:, None],
                other=0.0,
                cache_modifier=CACHE_MODIFIER_B,
            )

        if IS_FP8:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
        RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

    # -- Apply scaling and store --
    if IS_FP8:
        acc *= scale
    c = acc.to(C.type.element_ty)
    rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
    C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
    tl.store(C_, c, c_mask)


@triton.jit()
def _grouped_variable_k_gemm_kernel(
    # C[g] = LHS_g^T @ RHS_g [* scale if IS_FP8]
    LHS,
    RHS,
    C,
    LHS_scale_ptr,
    RHS_scale_ptr,  # only used if IS_FP8
    group_offs_ptr,  # [G+1] int64
    G,  # number of groups
    OUT_M,
    OUT_N,  # output dimensions (fixed across groups)
    # Strides
    stride_lhs_m,
    stride_rhs_m,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_lhs_n: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent grouped variable-K GEMM kernel for backward (static stride)."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_n > 0)
    tl.assume(stride_rhs_m > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    scale = tl.zeros((), dtype=tl.float32)
    if IS_FP8:
        scale = tl.load(LHS_scale_ptr) * tl.load(RHS_scale_ptr)

    for global_tile in range(pid, total_tiles, NUM_SMS):
        _process_variable_k_tile(
            global_tile,
            LHS,
            RHS,
            C,
            scale,
            group_offs_ptr,
            OUT_M,
            OUT_N,
            stride_lhs_m,
            stride_rhs_m,
            stride_cg,
            stride_cm,
            stride_cn,
            tiles_m,
            tiles_n,
            tiles_per_group,
            stride_lhs_n=stride_lhs_n,
            stride_rhs_n=stride_rhs_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            IS_FP8=IS_FP8,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            ALLOW_TF32=ALLOW_TF32,
        )


# -----------------------------------------------------------------------------
# Work-stealing variant of the variable-K kernel. Same per-XCD + global
# fallback scheme as the forward WS kernel above.
# -----------------------------------------------------------------------------


@triton.jit()
def _grouped_variable_k_gemm_kernel_ws(
    LHS,
    RHS,
    C,
    LHS_scale_ptr,
    RHS_scale_ptr,  # only used if IS_FP8
    group_offs_ptr,
    tile_counter_ptr,
    global_counter_ptr,
    G,
    OUT_M,
    OUT_N,
    stride_lhs_m,
    stride_rhs_m,
    stride_cg,
    stride_cm,
    stride_cn,
    local_per_xcd,
    stride_lhs_n: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    IS_FP8: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Variable-K persistent GEMM with per-XCD + global-fallback work stealing."""
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_n > 0)
    tl.assume(stride_rhs_m > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    scale = tl.zeros((), dtype=tl.float32)
    if IS_FP8:
        scale = tl.load(LHS_scale_ptr) * tl.load(RHS_scale_ptr)

    # See the forward WS kernel for the ACTIVE_XCDS rationale: when
    # NUM_SMS < NUM_XCDS, both the per-XCD slot count and the phase-1 ID
    # span must be capped at the active XCD count, or phase 2 skips past
    # the gap and silently drops tiles.
    ACTIVE_XCDS: tl.constexpr = min(NUM_SMS, NUM_XCDS)
    xcd_id = pid % ACTIVE_XCDS
    local_counter = tile_counter_ptr + xcd_id * COUNTER_STRIDE
    per_xcd = local_per_xcd.to(tl.int32)
    phase1_total = (per_xcd * ACTIVE_XCDS).to(tl.int32)

    local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
    in_phase2 = local_idx >= per_xcd
    if in_phase2:
        g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
        tile_id = phase1_total + g_idx
    else:
        tile_id = xcd_id * per_xcd + local_idx

    while tile_id < total_tiles:
        _process_variable_k_tile(
            tile_id,
            LHS,
            RHS,
            C,
            scale,
            group_offs_ptr,
            OUT_M,
            OUT_N,
            stride_lhs_m,
            stride_rhs_m,
            stride_cg,
            stride_cm,
            stride_cn,
            tiles_m,
            tiles_n,
            tiles_per_group,
            stride_lhs_n=stride_lhs_n,
            stride_rhs_n=stride_rhs_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            IS_FP8=IS_FP8,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            ALLOW_TF32=ALLOW_TF32,
        )
        if in_phase2:
            g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
            tile_id = phase1_total + g_idx
        else:
            local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
            if local_idx >= per_xcd:
                in_phase2 = True
                g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
                tile_id = phase1_total + g_idx
            else:
                tile_id = xcd_id * per_xcd + local_idx


# -- Public API -- Variable-K BF16 grouped GEMM (backward) --


@scoped_amd_knobs
def grouped_gemm_variable_k_triton_kernel(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    num_cu: int | None = None,
    work_steal: bool = False,
    ws_mode: str = "auto",
    ws_counter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-K grouped BF16/FP16 GEMM (backward) using Triton.

    Computes C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]]
    Output: [G, OUT_M, OUT_N].

    Args:
        lhs: [M_total, OUT_M] BF16/FP16 (after trans_c swap, this is grad_out).
        rhs: [M_total, OUT_N] BF16/FP16 (after trans_c swap, this is a).
        group_offs: [G+1] int64 prefix sum.
        num_cu: Cap the persistent-grid size at this many CUs (clamped to the
            device's CU count). ``None`` uses every CU on the device.
        work_steal: When True, dispatch to the WS variant. See
            ``grouped_gemm_triton_kernel`` for the API contract.
        ws_mode: ``"auto"`` / ``"per-xcd"`` / ``"global"`` / ``"hierarchical"``.
        ws_counter: Optional int32 buffer (per-device singleton managed by
            the higher-level wrapper when None). See
            ``grouped_gemm_triton_kernel`` for the single-stream caveat.

    Returns:
        [G, OUT_M, OUT_N] output.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0]
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=lhs.dtype)
    device_num_cus = get_num_cus()
    num_sms = min(num_cu, device_num_cus) if num_cu is not None and num_cu > 0 else device_num_cus
    dummy_scale = torch.empty(1, device=lhs.device, dtype=torch.float32)

    avg_m_g = max(lhs.shape[0] // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size = _get_gg_bf16_vk_config(
        OUT_M, OUT_N, avg_m_g, lhs.dtype, rhs.dtype, G, num_sms
    )

    if work_steal:
        from primus_turbo.triton.grouped_gemm.ws_triton_heuristic import (
            compute_triton_variable_k_total_tiles,
            resolve_triton_ws_local_per_xcd,
        )

        if ws_counter is None:
            ws_counter = _get_triton_ws_counter(lhs.device)
        expected = (NUM_XCDS + 1) * COUNTER_STRIDE
        if ws_counter.numel() < expected or ws_counter.dtype != torch.int32 or not ws_counter.is_cuda:
            raise ValueError(
                f"work_steal=True requires an int32 ws_counter on CUDA with at least "
                f"{expected} elements; got numel={ws_counter.numel()}, "
                f"dtype={ws_counter.dtype}, device={ws_counter.device}"
            )
        ws_counter.zero_()
        # Variable-K total_tiles is exact from integer shapes -- no sync.
        total_tiles = compute_triton_variable_k_total_tiles(
            G,
            OUT_M,
            OUT_N,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
        )
        local_per_xcd = resolve_triton_ws_local_per_xcd(
            ws_mode,
            total_tiles,
            num_sms,
            num_xcds=NUM_XCDS,
            kernel_kind="variable_k",
        )
        _grouped_variable_k_gemm_kernel_ws[(num_sms,)](
            lhs,
            rhs,
            out,
            dummy_scale,
            dummy_scale,
            group_offs,
            ws_counter,
            ws_counter[NUM_XCDS * COUNTER_STRIDE :],
            G,
            OUT_M,
            OUT_N,
            lhs.stride(0),
            rhs.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            local_per_xcd,
            stride_lhs_n=lhs.stride(1),
            stride_rhs_n=rhs.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=group_m,
            NUM_SMS=num_sms,
            NUM_XCDS=NUM_XCDS,
            IS_FP8=False,
            CACHE_MODIFIER_A=cache_a,
            CACHE_MODIFIER_B=cache_b,
            COUNTER_STRIDE=COUNTER_STRIDE,
            num_warps=8,
            num_stages=num_stages_val,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=1,
        )
        return out

    _grouped_variable_k_gemm_kernel[(num_sms,)](
        lhs,
        rhs,
        out,
        dummy_scale,
        dummy_scale,  # unused for BF16
        group_offs,
        G,
        OUT_M,
        OUT_N,
        lhs.stride(0),
        rhs.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_lhs_n=lhs.stride(1),
        stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        IS_FP8=False,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out
