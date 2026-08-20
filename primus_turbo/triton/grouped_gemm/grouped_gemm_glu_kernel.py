###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped GEMM Triton persistent kernel with a fused GLU-activation epilogue -- BF16/FP16.

Computes the MoE fc1 (gate-up) projection and its gated activation in one
launch::

    l1[g] = a[g] @ B_view[g]                  # [M_g, 2I], gate||up chunked
    act[g] = f(l1[g][:, :I]) * l1[g][:, I:]   # [M_g, I]

where ``f`` is selected per call via the ``activation`` argument. Only SwiGLU
(``f = silu``) is implemented today; the epilogue is parameterised because every
member of the GLU family transforms just the gate half and so shares this exact
shape -- see ``_glu_activation`` for what adding one involves.

``l1`` stays in registers for the activation epilogue and is also written to HBM
when callers need it for :func:`grouped_gemm_dgrad_dglu_triton_kernel`.

Column pairing
--------------
A gated activation needs columns ``j`` and ``j + I`` of the same row, but the
plain grouped GEMM hands each program one contiguous ``BLOCK_N`` slice of N,
which splits a gate/up pair across two programs (and, when ``I % BLOCK_N != 0``,
across misaligned tile boundaries). This kernel instead gives each program
``HALF = BLOCK_N // 2`` *pairs*, by remapping the tile's N indices to the gate
slice ``[p, p+HALF)`` followed by the up slice ``[I+p, I+p+HALF)``.

The halves are then teased apart in the epilogue with
``reshape``/``permute``/``split``, keeping one ``(BLOCK_M, BLOCK_N)`` ``tl.dot``
with the same MFMA shape and software pipeline as the tuned unfused kernel.

Alternatives that were measured and rejected, all on an MI355X grouped GEMM of
M x N x K = 131072 x 5760 x 2880:

* Two ``(BLOCK_M, HALF)`` accumulators fed by separately addressed B slices, so
  no cross-lane shuffle is needed: 87% vs 96% of the unfused grouped GEMM.
  Splitting costs more than the shuffle it avoids because neither half-width
  accumulator can live in AGPRs, pinning the kernel at the 512-VGPR ceiling.
* Anything that removes the epilogue's permute. It looks like a cross-lane
  shuffle but is not one, and the blocked gate||up layout it operates on is
  what lets the halves store as wide contiguous writes -- see
  :func:`_split_pair_cols` for the two variants and their numbers.
* An IEEE-exact divide in the SwiGLU sigmoid instead of the exp2 +
  raw-reciprocal form below: -0.5%, i.e. the epilogue's transcendentals are not
  the bottleneck.
* Raising occupancy: every 2-wave/SIMD config measured 1.2-11x *slower* than
  the 1-wave one. The MFMA knobs (``matrix_instr_nonkdim``, ``kpack``) are
  within noise or worse.
* Wider or deeper tiles: ``BLOCK_M=512`` with ``BLOCK_K=128`` exceeds the LDS
  limit outright, and every other combination of the two collapses to 0.1-0.2x
  under register pressure.
* Capping the persistent grid below the full CU count (``num_cu``): within
  noise, so L2 contention between the grid's programs is not a factor.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import get_num_cus
from primus_turbo.triton.grouped_gemm.grouped_gemm_helper import (
    NUM_XCDS,
    SUPPORTED_ACTIVATIONS,
    PairTileLaunch,
    _chiplet_transform_chunked,
    _chunk_rows,
    _count_group_tiles,
    _gemm_tile_dot,
    _get_gg_bf16_fwd_config,
    _glu_activation,
    _glu_activation_grad,
    _glu_activation_probs,
    _locate_group,
    _pair_tile_cols,
    _pair_tile_dot,
    _pair_tile_locate,
    _split_pair_cols,
    _swizzle_tile,
    resolve_pair_tile_shape,
    resolve_probs_launch,
)

# Operand dtypes this module's kernels accept. FP8 operands go through
# ``grouped_gemm_fp8_glu_kernel``, which adds the dequantisation step these
# kernels have no place for.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _process_grouped_gemm_glu_tile(
    global_tile_id,
    A,
    B,
    ACT,  # activation out [M_total, I]
    L1,  # fc1 pre-activation [M_total, 2I], gate||up
    PROBS,  # [M_total] fp32 routing probs, or dummy when USE_PROBS is False
    group_offs_ptr,
    G,
    INTER_N,  # I -- half of the fc1 output width
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_actm,
    stride_actn,
    stride_l1m,
    stride_l1n,
    stride_probs,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_PROBS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_ACT: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Compute one pair-tile of ``HALF = BLOCK_N // 2`` gate/up columns."""
    HALF: tl.constexpr = BLOCK_SIZE_N // 2

    in_range, m_start_g, M_g, pid_m, pid_n, group_idx = _pair_tile_locate(
        global_tile_id, group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M, GROUP_SIZE_M
    )
    if not in_range:
        return

    acc, rn_g, pair = _pair_tile_dot(
        A,
        B,
        m_start_g,
        M_g,
        group_idx,
        pid_m,
        pid_n,
        INTER_N,
        K,
        stride_am,
        stride_bg,
        stride_bn,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        EVEN_K=EVEN_K,
        N_ALIGNED=N_ALIGNED,
        PAIR_CONTIG=PAIR_CONTIG,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        ALLOW_TF32=ALLOW_TF32,
    )

    # -- Epilogue (row-chunked) --
    # gate/up/act tiles fill ~256 VGPRs; chunking the stores keeps the epilogue
    # from pinning a full-width act on top and spilling when l1 is also written.
    # The gate/up peel happens per chunk for the same reason -- not because the
    # peel itself is expensive (it is not; see _split_pair_cols) but because
    # peeling the full width doubles what stays live across the stores.
    if N_ALIGNED:
        pair_ok = tl.full((1, HALF), True, tl.int1)
    else:
        pair_ok = (pair < INTER_N)[None, :]

    l1_ty = L1.type.element_ty
    act_ty = ACT.type.element_ty
    R: tl.constexpr = BLOCK_SIZE_M // EPI_CHUNKS

    for c in tl.static_range(EPI_CHUNKS):
        acc_c = _chunk_rows(acc, c, BLOCK_SIZE_M, BLOCK_SIZE_N, EPI_CHUNKS)
        gate_c, up_c = _split_pair_cols(acc_c, R, HALF)
        raw_m = pid_m * BLOCK_SIZE_M + c * R + tl.arange(0, R)
        rm_c = tl.minimum(raw_m, M_g - 1)
        mask_c = (raw_m < M_g)[:, None] & pair_ok

        if USE_PROBS:
            probs_c = tl.load(
                PROBS + (m_start_g + rm_c) * stride_probs,
                mask=raw_m < M_g,
                other=1.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            act_c = _glu_activation_probs(gate_c, up_c, probs_c, ACTIVATION)
        else:
            act_c = _glu_activation(gate_c, up_c, ACTIVATION)

        L1_ = L1 + (m_start_g + rm_c[:, None]) * stride_l1m
        tl.store(
            L1_ + rn_g[None, :] * stride_l1n,
            gate_c.to(l1_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_L1,
        )
        tl.store(
            L1_ + (rn_g + INTER_N)[None, :] * stride_l1n,
            up_c.to(l1_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_L1,
        )
        ACT_ = ACT + (m_start_g + rm_c[:, None]) * stride_actm + rn_g[None, :] * stride_actn
        tl.store(ACT_, act_c.to(act_ty), mask_c, cache_modifier=CACHE_MODIFIER_ACT)


@triton.jit()
def _grouped_bf16_glu_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K]
    B,  # [G, ?, ?]  -- (K, 2I) or (2I, K) depending on trans_b
    ACT,  # [M_total, I]
    L1,  # [M_total, 2I]
    PROBS,  # [M_total] fp32
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,
    INTER_N,  # I
    K,
    # Strides
    stride_am,
    stride_bg,
    stride_bn,
    stride_actm,
    stride_actn,
    stride_l1m,
    stride_l1n,
    stride_probs,
    # Constexpr strides
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_PROBS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_ACT: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent grouped GEMM + fused GLU activation (CPU-sync-free) -- static stride.

    Tiles are counted over gate/up *pairs*, so ``num_pid_n`` spans I rather
    than the 2I output width. The epilogue writes both ``act`` [M, I] and the
    pre-activation ``l1`` [M, 2I] (gate||up, before the activation and probs).
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(INTER_N, BLOCK_SIZE_N // 2)
    total_tiles = _count_group_tiles(group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_actm > 0)
    tl.assume(stride_actn > 0)
    tl.assume(stride_l1m > 0)
    tl.assume(stride_l1n > 0)
    if USE_PROBS:
        tl.assume(stride_probs > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        _process_grouped_gemm_glu_tile(
            global_tile_id,
            A,
            B,
            ACT,
            L1,
            PROBS,
            group_offs_ptr,
            G,
            INTER_N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_actm,
            stride_actn,
            stride_l1m,
            stride_l1n,
            stride_probs,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            N_ALIGNED=N_ALIGNED,
            PAIR_CONTIG=PAIR_CONTIG,
            ACTIVATION=ACTIVATION,
            USE_PROBS=USE_PROBS,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            CACHE_MODIFIER_ACT=CACHE_MODIFIER_ACT,
            CACHE_MODIFIER_L1=CACHE_MODIFIER_L1,
            EPI_CHUNKS=EPI_CHUNKS,
            ALLOW_TF32=ALLOW_TF32,
        )


def _resolve_pair_tile_launch(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_b: bool,
    activation: str,
    num_cu: int | None,
    config: dict | None,
) -> PairTileLaunch:
    """Validate the fc1 operands and pick the tile config for a pair-tile launch.

    Shared by the forward and backward entry points: both run the same GEMM
    over the same pair-tile grid, so they resolve shapes and tuning identically
    and differ only in their epilogue.
    """
    assert a.dtype in _SUPPORTED_DTYPES, f"Unsupported dtype: {a.dtype}"
    assert b.dtype in _SUPPORTED_DTYPES, f"Unsupported dtype: {b.dtype}"
    shape = resolve_pair_tile_shape(a, b, trans_b, activation)
    M_total, G, K, inter_n = shape.M_total, shape.G, shape.K, shape.inter_n
    N = 2 * inter_n

    device_num_cus = get_num_cus()
    num_sms = min(num_cu, device_num_cus) if num_cu is not None and num_cu > 0 else device_num_cus
    avg_m = max(M_total // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size = (
        _get_gg_bf16_fwd_config(avg_m, N, K, a.dtype, b.dtype, trans_b, G, num_sms)
    )
    # _get_gg_bf16_fwd_config is tuned against the unfused epilogue and picks
    # BLOCK_K=32 / 3 stages; the fused epilogue moves the optimum to a deeper K
    # step with shallower pipelining (96% vs 82% of the unfused grouped GEMM on
    # 131072 x 5760 x 2880). Only override what was actually re-measured.
    if K % 64 == 0:
        BLOCK_K, num_stages_val = 64, 2
    # Origami often picks BLOCK_N=512 for the unfused 2I-wide GEMM, but the
    # pair-tile epilogue (optionally with probs) lands at 256 on large fc1
    # shapes -- 512 reg-spills badly (measured 1.01x -> 1.08x on
    # 131072 x 5760 x 2880).
    if BLOCK_N > 256:
        BLOCK_N = 256
    # The pair-tile keeps a BLOCK_M x BLOCK_N fp32 accumulator live, which lands
    # at ~1 wave/SIMD. That is the fast point -- num_warps=8 spreads the dot
    # across more waves and codegen degrades badly (measured 2-11x slower), and
    # the higher-occupancy configs it unlocks are all slower.
    num_warps_val = 4
    waves_per_eu_val, mfma_dim_val, kpack_val = 1, 16, 1
    # The epilogue output is a pure write stream nothing re-reads. .cg keeps it
    # out of the way of B in L2; .wt costs 4%.
    cache_out = ".cg"
    cache_l1 = ".cg"
    epi_chunks = 4
    # Halving BLOCK_N halves the pair-tile grid too, so the swizzle the unfused
    # config picked is sized for the wrong tile count. Re-derive it: wide grids
    # want a short swizzle for L2 reuse, small ones want the opposite.
    est_tiles = -(-M_total // BLOCK_M) * -(-inter_n // (BLOCK_N // 2))
    group_m, chunk_size = (2, 32) if est_tiles >= 512 else (4, 64)
    if config is not None:
        BLOCK_M = config.get("BLOCK_M", BLOCK_M)
        BLOCK_N = config.get("BLOCK_N", BLOCK_N)
        BLOCK_K = config.get("BLOCK_K", BLOCK_K)
        group_m = config.get("GROUP_M", group_m)
        num_warps_val = config.get("num_warps", num_warps_val)
        num_stages_val = config.get("num_stages", num_stages_val)
        waves_per_eu_val = config.get("waves_per_eu", waves_per_eu_val)
        mfma_dim_val = config.get("matrix_instr_nonkdim", mfma_dim_val)
        kpack_val = config.get("kpack", kpack_val)
        chunk_size = config.get("CHUNK_SIZE", chunk_size)
        cache_out = config.get("cache_act", cache_out)
        cache_l1 = config.get("cache_l1", cache_l1)
        epi_chunks = config.get("epi_chunks", epi_chunks)
    # A pair-tile is BLOCK_N // 2 wide, and tl.dot needs at least 16 columns.
    while BLOCK_N > 32 and inter_n * 2 < BLOCK_N:
        BLOCK_N //= 2

    return PairTileLaunch(
        inter_n=inter_n,
        K=K,
        num_sms=num_sms,
        stride_bg=shape.stride_bg,
        stride_bn=shape.stride_bn,
        stride_ak=shape.stride_ak,
        stride_bk=shape.stride_bk,
        even_k=(K % BLOCK_K == 0),
        n_aligned=(inter_n % (BLOCK_N // 2) == 0),
        pair_contig=math.gcd(BLOCK_N // 2, inter_n),
        out_dtype=a.dtype,
        grid=dict(
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=group_m,
            CHUNK_SIZE=chunk_size,
        ),
        knobs=dict(
            num_warps=num_warps_val,
            num_stages=num_stages_val,
            waves_per_eu=waves_per_eu_val,
            matrix_instr_nonkdim=mfma_dim_val,
            kpack=kpack_val,
        ),
        cache_a=cache_a,
        cache_b=cache_b,
        cache_out=cache_out,
        cache_l1=cache_l1,
        epi_chunks=epi_chunks,
    )


def grouped_gemm_glu_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    *,
    activation: str = "swiglu",
    probs: torch.Tensor | None = None,
    intermediate_out: torch.Tensor | None = None,
    act_out: torch.Tensor | None = None,
    num_cu: int | None = None,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent grouped GEMM with a fused GLU-family activation epilogue.

    Computes ``act[g] = f(gate) * up`` (optionally scaled by ``probs``) where
    ``l1 = [gate|up] = a[g] @ B_view[g]``, in a single launch. Both the
    pre-activation ``l1`` [M, 2I] and the activation ``act`` [M, I] are written:
    ``l1`` is the raw fc1 GEMM output (before the activation and probs), for use by
    :func:`grouped_gemm_dgrad_dglu_triton_kernel` in backward.

    Args:
        a: [M_total, K] BF16/FP16 input.
        b: [G, K, 2I] or [G, 2I, K] (if trans_b) BF16/FP16 fc1 weights.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [2I, K] (transposed).
        activation: Which gate activation to fuse. Must be one of
            ``SUPPORTED_ACTIVATIONS``, currently ``"swiglu"`` only.
        probs: Optional [M_total] float32 routing probabilities. When provided,
            the epilogue scales the activation output by ``probs`` per row,
            matching :func:`swiglu_fwd_kernel`.
        intermediate_out: Optional [M_total, 2I] output buffer for the
            pre-activation ``l1``. Allocated when ``None``.
        act_out: Optional [M_total, I] output buffer for the activation.
            Allocated when ``None``.
        num_cu: Cap the persistent grid at this many CUs. None uses every CU.
        config: Override the tile config (BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M/
            num_warps/num_stages/waves_per_eu/matrix_instr_nonkdim/kpack/
            CHUNK_SIZE/cache_act/cache_l1). For tuning experiments.

    Returns:
        ``(act, l1)`` where ``act`` is [M_total, I] and ``l1`` is [M_total, 2I].
    """
    plan = _resolve_pair_tile_launch(a, b, trans_b, activation, num_cu, config)
    if act_out is None:
        act = torch.empty((a.shape[0], plan.inter_n), device=a.device, dtype=a.dtype)
    else:
        act = act_out
        assert act.shape == (a.shape[0], plan.inter_n), (
            f"act_out must be [{a.shape[0]}, {plan.inter_n}], got {tuple(act.shape)}"
        )
        assert act.device == a.device and act.dtype == a.dtype
    if intermediate_out is None:
        l1 = torch.empty((a.shape[0], 2 * plan.inter_n), device=a.device, dtype=a.dtype)
    else:
        l1 = intermediate_out
        assert l1.shape == (a.shape[0], 2 * plan.inter_n), (
            f"intermediate_out must be [{a.shape[0]}, {2 * plan.inter_n}], got {tuple(l1.shape)}"
        )
        assert l1.device == a.device and l1.dtype == a.dtype
    probs_ptr, stride_probs, use_probs = resolve_probs_launch(probs, a.shape[0], a.device)

    _grouped_bf16_glu_persistent_gemm_kernel[(plan.num_sms,)](
        a,
        b,
        act,
        l1,
        probs_ptr,
        group_offs,
        b.shape[0],
        plan.inter_n,
        plan.K,
        a.stride(0),
        plan.stride_bg,
        plan.stride_bn,
        act.stride(0),
        act.stride(1),
        l1.stride(0),
        l1.stride(1),
        stride_probs,
        stride_ak=plan.stride_ak,
        stride_bk=plan.stride_bk,
        NUM_SMS=plan.num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=plan.even_k,
        N_ALIGNED=plan.n_aligned,
        PAIR_CONTIG=plan.pair_contig,
        ACTIVATION=activation,
        USE_PROBS=use_probs,
        CACHE_MODIFIER_A=plan.cache_a,
        CACHE_MODIFIER_B=plan.cache_b,
        CACHE_MODIFIER_ACT=plan.cache_out,
        CACHE_MODIFIER_L1=plan.cache_l1,
        EPI_CHUNKS=plan.epi_chunks,
        **plan.grid,
        **plan.knobs,
    )
    return act, l1


# ===============================================================================
# Backward: fc1 recompute + fused activation gradient
#
# For callers that do not save l1 in the forward, backward has to recompute it
# before it can differentiate the activation. Recomputing it *inside* the
# gradient epilogue keeps l1 in registers exactly like the forward does, which
# is where the win is: the unfused path writes l1 (2I), reads it back (2I) and
# reads dact (I) to write dl1 (2I), while the fused kernel reads dact (I) and
# writes dl1 (2I) with the same GEMM cost.
#
# When l1 *is* saved, prefer grouped_gemm_dgrad_dglu_triton_kernel below: it
# hangs the same epilogue off fc2's dgrad and skips this recompute GEMM.
# ===============================================================================


@triton.jit
def _process_grouped_gemm_dglu_tile(
    global_tile_id,
    A,
    B,
    DACT,  # incoming grad wrt the activation output [M_total, I]
    DL1,  # grad wrt the fc1 output [M_total, 2I], gate||up
    PROBS,  # [M_total] fp32 routing probs, or dummy when USE_PROBS is False
    group_offs_ptr,
    G,
    INTER_N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_dactm,
    stride_dactn,
    stride_dl1m,
    stride_dl1n,
    stride_probs,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_PROBS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_ACT: tl.constexpr,
    CACHE_MODIFIER_DACT: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Recompute one pair-tile of fc1 and emit both halves of its gradient."""
    HALF: tl.constexpr = BLOCK_SIZE_N // 2

    in_range, m_start_g, M_g, pid_m, pid_n, group_idx = _pair_tile_locate(
        global_tile_id, group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M, GROUP_SIZE_M
    )
    if not in_range:
        return

    acc, _rn_g, _pair = _pair_tile_dot(
        A,
        B,
        m_start_g,
        M_g,
        group_idx,
        pid_m,
        pid_n,
        INTER_N,
        K,
        stride_am,
        stride_bg,
        stride_bn,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        EVEN_K=EVEN_K,
        N_ALIGNED=N_ALIGNED,
        PAIR_CONTIG=PAIR_CONTIG,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        ALLOW_TF32=ALLOW_TF32,
    )

    # -- Epilogue --
    # Walk the tile in EPI_CHUNKS row blocks. The dot's fp32 accumulator already
    # fills ~256 VGPRs, so materialising a full dact tile on top of it either
    # spills (~128 regs at BLOCK_N=256) or forces a narrower BLOCK_N; both cost
    # ~1.1 ms on 131072 x 5760 x 2880, i.e. the whole fusion win. Chunking keeps
    # dact and the products live a row block at a time, so the wide tile fits.
    #
    # Two variants measured worse and are not offered: chunking columns instead
    # of rows narrows the stores below a cache line (1.01x vs 1.04x), and
    # hoisting the dact load above the K-loop to hide its latency costs more in
    # K-loop register pressure than the overlap wins back (0.97x).
    out_ty = DL1.type.element_ty
    R: tl.constexpr = BLOCK_SIZE_M // EPI_CHUNKS
    # A clamp would be cheaper than the wrap, but it costs the compiler its
    # contiguity proof for the column index and the coalesced stores degrade
    # into a scatter (measured 2x slower).
    cols, pair = _pair_tile_cols(pid_n, INTER_N, BLOCK_SIZE_N, N_ALIGNED, PAIR_CONTIG)
    if N_ALIGNED:
        col_ok = tl.full((1, HALF), True, tl.int1)
    else:
        col_ok = (pair < INTER_N)[None, :]

    for c in tl.static_range(EPI_CHUNKS):
        acc_c = _chunk_rows(acc, c, BLOCK_SIZE_M, BLOCK_SIZE_N, EPI_CHUNKS)
        gate_c, up_c = _split_pair_cols(acc_c, R, HALF)

        # Clamp for addressing and mask on the raw index, rather than the
        # `% M_g` wrap the dot uses: a runtime modulo is a ~20-op reciprocal
        # sequence and this runs per chunk, not once per tile. The wrap is only
        # safe because a wrapped lane recomputes the value that row already
        # holds; a clamped lane does not, so the mask has to be real here (with
        # the wrap it is trivially all-true).
        raw_m = pid_m * BLOCK_SIZE_M + c * R + tl.arange(0, R)
        rm_c = tl.minimum(raw_m, M_g - 1)
        mask_c = (raw_m < M_g)[:, None] & col_ok

        # dact is I-wide, so the tile's columns index it exactly like the
        # forward's output. Read once, hence the streaming cache hint.
        dout_c = tl.load(
            DACT + (m_start_g + rm_c[:, None]) * stride_dactm + cols[None, :] * stride_dactn,
            mask=mask_c,
            other=0.0,
            cache_modifier=CACHE_MODIFIER_DACT,
        )
        dout_fp32 = dout_c.to(tl.float32)
        if USE_PROBS:
            probs_c = tl.load(
                PROBS + (m_start_g + rm_c) * stride_probs,
                mask=raw_m < M_g,
                other=1.0,
            ).to(tl.float32)
            dout_fp32 = dout_fp32 * probs_c[:, None]
        dgate_c, dup_c = _glu_activation_grad(gate_c, up_c, dout_fp32, ACTIVATION)

        DL1_ = DL1 + (m_start_g + rm_c[:, None]) * stride_dl1m

        # dl1 is 2I-wide: the gate gradient lands at the tile's own columns and
        # the up gradient I columns further along.
        tl.store(
            DL1_ + cols[None, :] * stride_dl1n,
            dgate_c.to(out_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_ACT,
        )
        tl.store(
            DL1_ + (cols + INTER_N)[None, :] * stride_dl1n,
            dup_c.to(out_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_ACT,
        )


@triton.jit()
def _grouped_bf16_dglu_persistent_gemm_kernel(
    # Pointers
    A,  # [M_total, K]
    B,  # [G, ?, ?]  -- (K, 2I) or (2I, K) depending on trans_b
    DACT,  # [M_total, I]
    DL1,  # [M_total, 2I]
    PROBS,  # [M_total] fp32
    group_offs_ptr,  # [G+1] int64
    # Dimensions
    G,
    INTER_N,  # I
    K,
    # Strides
    stride_am,
    stride_bg,
    stride_bn,
    stride_dactm,
    stride_dactn,
    stride_dl1m,
    stride_dl1n,
    stride_probs,
    # Constexpr strides
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    PAIR_CONTIG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_PROBS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_ACT: tl.constexpr,
    CACHE_MODIFIER_DACT: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent fc1 recompute + fused activation gradient (CPU-sync-free).

    Same pair-tile grid as the forward kernel, so ``num_pid_n`` spans I rather
    than the 2I gradient width.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(INTER_N, BLOCK_SIZE_N // 2)
    total_tiles = _count_group_tiles(group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_dactm > 0)
    tl.assume(stride_dactn > 0)
    tl.assume(stride_dl1m > 0)
    tl.assume(stride_dl1n > 0)
    if USE_PROBS:
        tl.assume(stride_probs > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        _process_grouped_gemm_dglu_tile(
            global_tile_id,
            A,
            B,
            DACT,
            DL1,
            PROBS,
            group_offs_ptr,
            G,
            INTER_N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_dactm,
            stride_dactn,
            stride_dl1m,
            stride_dl1n,
            stride_probs,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            N_ALIGNED=N_ALIGNED,
            PAIR_CONTIG=PAIR_CONTIG,
            ACTIVATION=ACTIVATION,
            USE_PROBS=USE_PROBS,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            CACHE_MODIFIER_ACT=CACHE_MODIFIER_ACT,
            CACHE_MODIFIER_DACT=CACHE_MODIFIER_DACT,
            EPI_CHUNKS=EPI_CHUNKS,
            ALLOW_TF32=ALLOW_TF32,
        )


def grouped_gemm_dglu_triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    dact: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    *,
    activation: str = "swiglu",
    probs: torch.Tensor | None = None,
    num_cu: int | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Recompute fc1 and apply the activation gradient in one launch.

    Backward counterpart of :func:`grouped_gemm_glu_triton_kernel`. That
    kernel does not keep the ``2I``-wide pre-activation, so backward must
    recompute ``[gate|up] = a[g] @ B_view[g]``; doing it here means l1 never
    reaches HBM, exactly as in the forward. The result is
    ``dl1 = [f'(gate) * up * (dact * probs) | f(gate) * (dact * probs)]`` when
    ``probs`` is provided (matching :func:`swiglu_bwd_kernel`), which the fc1
    dgrad and wgrad GEMMs then consume.

    ``grad_probs`` is not computed here: it needs a full row sum over I that a
    pair-tile cannot complete on its own.

    Args:
        a: [M_total, K] BF16/FP16 fc1 input (the same one the forward saw).
        b: [G, K, 2I] or [G, 2I, K] (if trans_b) BF16/FP16 fc1 weights.
        dact: [M_total, I] incoming gradient wrt the probs-scaled activation.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [2I, K] (transposed).
        activation: Must be one of ``SUPPORTED_ACTIVATIONS``.
        probs: Optional [M_total] float32 routing probabilities from forward.
        num_cu: Cap the persistent grid at this many CUs. None uses every CU.
        config: Override the tile config; see the forward kernel.

    Returns:
        ``dl1`` [M_total, 2I], gate gradient in [:, :I] and up in [:, I:].
    """
    plan = _resolve_pair_tile_launch(a, b, trans_b, activation, num_cu, config)
    M_total = a.shape[0]
    assert dact.shape == (M_total, plan.inter_n), (
        f"dact must be [{M_total}, {plan.inter_n}], got {tuple(dact.shape)}"
    )

    dl1 = torch.empty((M_total, 2 * plan.inter_n), device=a.device, dtype=a.dtype)
    probs_ptr, stride_probs, use_probs = resolve_probs_launch(probs, M_total, a.device)

    _grouped_bf16_dglu_persistent_gemm_kernel[(plan.num_sms,)](
        a,
        b,
        dact,
        dl1,
        probs_ptr,
        group_offs,
        b.shape[0],
        plan.inter_n,
        plan.K,
        a.stride(0),
        plan.stride_bg,
        plan.stride_bn,
        dact.stride(0),
        dact.stride(1),
        dl1.stride(0),
        dl1.stride(1),
        stride_probs,
        stride_ak=plan.stride_ak,
        stride_bk=plan.stride_bk,
        NUM_SMS=plan.num_sms,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=plan.even_k,
        N_ALIGNED=plan.n_aligned,
        PAIR_CONTIG=plan.pair_contig,
        ACTIVATION=activation,
        USE_PROBS=use_probs,
        CACHE_MODIFIER_A=plan.cache_a,
        CACHE_MODIFIER_B=plan.cache_b,
        CACHE_MODIFIER_ACT=plan.cache_out,
        CACHE_MODIFIER_DACT=(config or {}).get("cache_dact", ".cg"),
        EPI_CHUNKS=(config or {}).get("epi_chunks", 4),
        **plan.grid,
        **plan.knobs,
    )
    return dl1


# ===============================================================================
# fc2 dgrad + GLU-gradient epilogue -- Persistent Kernel (backward, CPU-sync-free)
#
# Computes: dl1[g] = dact_grad(l1[g], dout[g] @ B_view[g])
#
# This is the other place the activation gradient can live. Rather than
# recomputing fc1 (see grouped_gemm_dglu_triton_kernel), it hangs the
# gradient off the GEMM the backward has to run anyway -- fc2's dgrad,
# ``dact = dout @ W2^T`` -- and reads the pre-activation the forward saved.
# That trades a saved 2I-wide l1 in the forward for dropping a full M x 2I x K
# recompute GEMM from the backward.
#
# Unlike the pair-tile kernels above, the GEMM here is an ordinary grouped GEMM
# over an I-wide output; only the epilogue knows about the gate/up split.
# ===============================================================================


@triton.jit
def _process_grouped_dgrad_dglu_tile(
    global_tile_id,
    A,  # [M_total, K]   incoming gradient wrt the fc2 output
    B,  # [G, ?, ?]      fc2 weights, already oriented so the dot yields dact
    L1,  # [M_total, 2N]  saved fc1 pre-activation
    DL1,  # [M_total, 2N]
    PROBS,  # [M_total] fp32 routing probs, or dummy when USE_PROBS is False
    group_offs_ptr,
    G,
    N,  # I
    K,  # fc2 output width
    stride_am,
    stride_bg,
    stride_bn,
    stride_l1m,
    stride_l1n,
    stride_dl1m,
    stride_dl1n,
    stride_probs,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_PROBS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    CACHE_MODIFIER_DL1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """One I-wide dact tile, turned into both halves of dl1 before it leaves registers."""
    group_idx, tile_start, total_tiles = _locate_group(
        global_tile_id, group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M
    )
    if global_tile_id >= total_tiles:
        return

    local_tile = global_tile_id - tile_start
    m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
    M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
    pid_m, pid_n = _swizzle_tile(local_tile, tl.cdiv(M_g, BLOCK_SIZE_M), num_pid_n, GROUP_SIZE_M)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    acc = _gemm_tile_dot(
        A,
        B,
        m_start_g,
        group_idx,
        rm,
        rn,
        K,
        stride_am,
        stride_bg,
        stride_bn,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        EVEN_K=EVEN_K,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        ALLOW_TF32=ALLOW_TF32,
    )

    # -- Epilogue --
    # Row-blocked for the same reason as the recompute kernel: the accumulator
    # alone fills ~256 VGPRs and this epilogue needs *two* more input tiles
    # (gate and up), so a full-width version spills hard. Rows keep every access
    # BLOCK_N-wide; see grouped_gemm_dglu_triton_kernel for the column-slicing
    # and load-hoisting variants that measured worse.
    out_ty = DL1.type.element_ty
    R: tl.constexpr = BLOCK_SIZE_M // EPI_CHUNKS

    # probs scales the whole accumulator before chunking. Folding it into each
    # chunk instead measured slower (1.08x vs 1.09x): the narrower multiply does
    # not pay for the extra per-chunk index math.
    if USE_PROBS:
        rm_probs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        probs_tile = tl.load(
            PROBS + (m_start_g + tl.minimum(rm_probs, M_g - 1)) * stride_probs,
            mask=rm_probs < M_g,
            other=1.0,
        ).to(tl.float32)
        acc = acc * probs_tile[:, None]

    for c in tl.static_range(EPI_CHUNKS):
        acc_c = _chunk_rows(acc, c, BLOCK_SIZE_M, BLOCK_SIZE_N, EPI_CHUNKS)

        # Clamp for addressing, mask on the raw index: a runtime `% M_g` per
        # chunk is a ~20-op reciprocal sequence. Branching to an unmasked path
        # for tiles that sit wholly inside a group measured no faster.
        raw_m = pid_m * BLOCK_SIZE_M + c * R + tl.arange(0, R)
        rm_c = tl.minimum(raw_m, M_g - 1)
        mask_c = (raw_m < M_g)[:, None]

        # gate at l1[:, n], up at l1[:, n + I]: two BLOCK_N-wide reads, both
        # streaming (nothing re-reads l1 after this).
        L1_ = L1 + (m_start_g + rm_c[:, None]) * stride_l1m
        gate = tl.load(
            L1_ + rn[None, :] * stride_l1n,
            mask=mask_c,
            other=0.0,
            cache_modifier=CACHE_MODIFIER_L1,
        )
        up = tl.load(
            L1_ + (rn + N)[None, :] * stride_l1n,
            mask=mask_c,
            other=0.0,
            cache_modifier=CACHE_MODIFIER_L1,
        )

        dgate_c, dup_c = _glu_activation_grad(gate.to(tl.float32), up.to(tl.float32), acc_c, ACTIVATION)

        DL1_ = DL1 + (m_start_g + rm_c[:, None]) * stride_dl1m
        tl.store(
            DL1_ + rn[None, :] * stride_dl1n,
            dgate_c.to(out_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_DL1,
        )
        tl.store(
            DL1_ + (rn + N)[None, :] * stride_dl1n,
            dup_c.to(out_ty),
            mask_c,
            cache_modifier=CACHE_MODIFIER_DL1,
        )


@triton.jit()
def _grouped_bf16_dgrad_dglu_persistent_gemm_kernel(
    A,
    B,
    L1,
    DL1,
    PROBS,
    group_offs_ptr,
    G,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_l1m,
    stride_l1n,
    stride_dl1m,
    stride_dl1n,
    stride_probs,
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
    ACTIVATION: tl.constexpr,
    USE_PROBS: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    CACHE_MODIFIER_L1: tl.constexpr,
    CACHE_MODIFIER_DL1: tl.constexpr,
    EPI_CHUNKS: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent fc2 dgrad with a fused activation-gradient epilogue."""
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = _count_group_tiles(group_offs_ptr, G, num_pid_n, BLOCK_SIZE_M)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_l1m > 0)
    tl.assume(stride_l1n > 0)
    tl.assume(stride_dl1m > 0)
    tl.assume(stride_dl1n > 0)
    if USE_PROBS:
        tl.assume(stride_probs > 0)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        _process_grouped_dgrad_dglu_tile(
            global_tile_id,
            A,
            B,
            L1,
            DL1,
            PROBS,
            group_offs_ptr,
            G,
            N,
            K,
            stride_am,
            stride_bg,
            stride_bn,
            stride_l1m,
            stride_l1n,
            stride_dl1m,
            stride_dl1n,
            stride_probs,
            num_pid_n,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EVEN_K=EVEN_K,
            ACTIVATION=ACTIVATION,
            USE_PROBS=USE_PROBS,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            CACHE_MODIFIER_L1=CACHE_MODIFIER_L1,
            CACHE_MODIFIER_DL1=CACHE_MODIFIER_DL1,
            EPI_CHUNKS=EPI_CHUNKS,
            ALLOW_TF32=ALLOW_TF32,
        )


def grouped_gemm_dgrad_dglu_triton_kernel(
    dout: torch.Tensor,
    b: torch.Tensor,
    intermediate: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    *,
    activation: str = "swiglu",
    probs: torch.Tensor | None = None,
    dintermediate_out: torch.Tensor | None = None,
    num_cu: int | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """fc2 dgrad with the activation gradient fused into its epilogue.

    Computes ``dact[g] = dout[g] @ B_view[g]`` -- the gradient wrt the probs-scaled
    fc2 input -- and consumes it in registers to produce
    ``dl1 = [f'(gate) * up * (dact * probs) | f(gate) * (dact * probs)]`` when
    ``probs`` is provided, so ``dact`` never reaches HBM. ``b`` is the fc2 weight
    already oriented for the dgrad, i.e. callers pass the ``trans_b`` they used
    in the forward, flipped.

    The alternative placement is
    :func:`grouped_gemm_dglu_triton_kernel`, which recomputes fc1 instead of
    reading ``intermediate``. Pick between them on whether the forward saved
    the pre-activation: this kernel needs it, and in exchange the backward
    drops a full ``M x 2I x K`` recompute GEMM.

    ``grad_probs`` is not computed here: it needs a full row sum over I that a
    single tile cannot complete on its own.

    Args:
        dout: [M_total, K] BF16/FP16 gradient wrt the fc2 output.
        b: [G, K, I] or [G, I, K] (if trans_b) BF16/FP16 fc2 weights.
        intermediate: [M_total, 2I] saved fc1 pre-activation, gate in [:, :I],
            up in [:, I:].
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [I, K] (transposed).
        activation: Must be one of ``SUPPORTED_ACTIVATIONS``.
        probs: Optional [M_total] float32 routing probabilities from forward.
        dintermediate_out: Optional [M_total, 2I] output buffer. Allocated when
            ``None``.
        num_cu: Cap the persistent grid at this many CUs. None uses every CU.
        config: Override tile config / knobs (BLOCK_M, BLOCK_N, BLOCK_K,
            GROUP_M, num_warps, num_stages, waves_per_eu, epi_chunks,
            cache_l1, cache_dl1). For tuning experiments.

    Returns:
        ``dl1`` [M_total, 2I], gate gradient in [:, :I] and up in [:, I:].
    """
    assert dout.ndim == 2, f"dout must be 2D, got {dout.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert intermediate.ndim == 2, f"intermediate must be 2D, got {intermediate.shape}"
    assert dout.dtype in _SUPPORTED_DTYPES, f"Unsupported dtype: {dout.dtype}"
    assert b.dtype in _SUPPORTED_DTYPES, f"Unsupported dtype: {b.dtype}"
    assert activation in SUPPORTED_ACTIVATIONS, (
        f"Unsupported activation: {activation!r}, expected one of {SUPPORTED_ACTIVATIONS}"
    )

    M_total, K_a = dout.shape
    G = b.shape[0]
    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk, stride_bn = b.stride(2), b.stride(1)
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk, stride_bn = b.stride(1), b.stride(2)

    assert K_a == K_b, f"K mismatch: dout has K={K_a}, b has K={K_b}"
    assert intermediate.shape == (M_total, 2 * N), (
        f"intermediate must be [{M_total}, {2 * N}], got {tuple(intermediate.shape)}"
    )
    K = K_a

    device_num_cus = get_num_cus()
    num_sms = min(num_cu, device_num_cus) if num_cu is not None and num_cu > 0 else device_num_cus
    avg_m = max(M_total // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size = (
        _get_gg_bf16_fwd_config(avg_m, N, K, dout.dtype, b.dtype, trans_b, G, num_sms)
    )
    num_warps_val, waves_per_eu_val, mfma_dim_val, kpack_val = 4, 1, 16, 1
    cache_l1, cache_dl1 = ".cg", ".cg"
    epi_chunks = 4
    if K % 64 == 0:
        BLOCK_K, num_stages_val = 64, 2
    if BLOCK_N > 256:
        BLOCK_N = 256
    if M_total >= 65536 and N >= 2048:
        BLOCK_M = 256
        BLOCK_N = 256
    est_tiles = -(-M_total // BLOCK_M) * -(-N // BLOCK_N)
    group_m, chunk_size = (2, 16) if est_tiles >= 512 else (4, 64)
    if config is not None:
        BLOCK_M = config.get("BLOCK_M", BLOCK_M)
        BLOCK_N = config.get("BLOCK_N", BLOCK_N)
        BLOCK_K = config.get("BLOCK_K", BLOCK_K)
        group_m = config.get("GROUP_M", group_m)
        chunk_size = config.get("CHUNK_SIZE", chunk_size)
        num_warps_val = config.get("num_warps", num_warps_val)
        num_stages_val = config.get("num_stages", num_stages_val)
        waves_per_eu_val = config.get("waves_per_eu", waves_per_eu_val)
        mfma_dim_val = config.get("matrix_instr_nonkdim", mfma_dim_val)
        kpack_val = config.get("kpack", kpack_val)
        cache_l1 = config.get("cache_l1", config.get("cache_act", cache_l1))
        cache_dl1 = config.get("cache_dl1", config.get("cache_act", cache_dl1))
        epi_chunks = config.get("epi_chunks", epi_chunks)

    dl1 = (
        dintermediate_out
        if dintermediate_out is not None
        else torch.empty((M_total, 2 * N), device=dout.device, dtype=dout.dtype)
    )
    if dintermediate_out is not None:
        assert dl1.shape == (M_total, 2 * N), (
            f"dintermediate_out must be [{M_total}, {2 * N}], got {tuple(dl1.shape)}"
        )
        assert dl1.device == dout.device and dl1.dtype == dout.dtype
    probs_ptr, stride_probs, use_probs = resolve_probs_launch(probs, M_total, dout.device)

    _grouped_bf16_dgrad_dglu_persistent_gemm_kernel[(num_sms,)](
        dout,
        b,
        intermediate,
        dl1,
        probs_ptr,
        group_offs,
        G,
        N,
        K,
        dout.stride(0),
        b.stride(0),
        stride_bn,
        intermediate.stride(0),
        intermediate.stride(1),
        dl1.stride(0),
        dl1.stride(1),
        stride_probs,
        stride_ak=dout.stride(1),
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=(K % BLOCK_K == 0),
        ACTIVATION=activation,
        USE_PROBS=use_probs,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        CACHE_MODIFIER_L1=cache_l1,
        CACHE_MODIFIER_DL1=cache_dl1,
        EPI_CHUNKS=epi_chunks,
        num_warps=num_warps_val,
        num_stages=num_stages_val,
        waves_per_eu=waves_per_eu_val,
        matrix_instr_nonkdim=mfma_dim_val,
        kpack=kpack_val,
    )
    return dl1
