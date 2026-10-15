###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Adapted from AITER (https://github.com/ROCm/aiter, MIT License),
# aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py.
# Modified by the Primus-Turbo team; see NOTICE.md in this directory.
###############################################################################

from typing import Literal

import torch
import triton
import triton.language as tl

from .flex_attention_utils import (
    AUTOTUNE,
    FWD_CONF_OVERRIDE,
    AutotuneMode,
    apply_mask_mod,
    apply_score_mod,
    aux_kernel_args,
    bs_tensor_strides,
    get_arch,
    mask_aux_kernel_args,
    max_block_for_lds,
    pick_knobs,
    remap_xcd,
)

# Kernel knobs for the block-sparse path. The tile sizes are forced to the sparsity
# granularity, but these are free -- the stock autotuner is bypassed here only because it
# would pick its own tiles. See pick_knobs() in utils.py for how the winner is chosen.
#
# Exposed as a dict so bench/sweep_sparse_knobs.py can sweep them.
SPARSE_FWD_KNOBS = dict(waves_per_eu=2, PRE_LOAD_V=False, num_stages=1, num_warps=4)


def sparse_fwd_default(kv_block: int) -> dict:
    """Block-sparse forward knobs, keyed on architecture and block size.

    Used as-is when autotuning is off, and as the first candidate otherwise.

    waves_per_eu is strongly per-shape, and on gfx950 the split falls cleanly on block
    size: measured across sequence length and head dim, a block-64 forward is 1.10x
    faster at seqlen 4096 and 1.15x at 8192 with waves_per_eu=3, while a block-128
    forward is 1.19-1.33x *slower* with it. Block size is known for free at launch, so
    that case needs no measurement.

    Only gfx950/block-64 is special-cased -- the one split measured unambiguously, with
    no shape where it loses (seqlen 2048 and head_dim 128 come out level). gfx942 shows
    only ~4% for the same change and is left alone.
    """
    if get_arch().name == "gfx950" and kv_block <= 64:
        return dict(waves_per_eu=3, PRE_LOAD_V=False, num_stages=1, num_warps=4)
    return SPARSE_FWD_KNOBS


def sparse_fwd_candidates(kv_block: int) -> list:
    """Knob sets to measure when autotuning is on. The default goes first so it wins ties."""
    default = sparse_fwd_default(kv_block)
    others = [
        dict(waves_per_eu=w, PRE_LOAD_V=pv, num_stages=st, num_warps=4)
        for w, pv, st in ((2, False, 1), (3, False, 1), (3, False, 2), (1, False, 1))
    ]
    return [default] + [c for c in others if c != default]


# Keyed on the same shape properties as FWD_PREFILL_AUTOTUNE_KEYS, plus the block size.
_SPARSE_FWD_CHOICE: dict = {}

FWD_PREFILL_AUTOTUNE_KEYS = [
    "IS_CAUSAL",
    "MAX_SEQLENS_Q",
    "MAX_SEQLENS_K",
    "ACTUAL_BLOCK_DMODEL_QK",
    "ACTUAL_BLOCK_DMODEL_V",
    "IS_VARLEN",
    "HQ",
    "HK",
]


# See the note on _bwd_cfg in bwd.py: tuned tables are kept as one-line rows.
def _fwd_cfg(m, n, waves, preload_v, *, stages, warps):
    """Forward-prefill config. Columns: BLOCK_M, BLOCK_N, waves_per_eu, PRE_LOAD_V."""
    return triton.Config(
        {"BLOCK_M": m, "BLOCK_N": n, "waves_per_eu": waves, "PRE_LOAD_V": preload_v},
        num_stages=stages,
        num_warps=warps,
    )


def _fwd_configs_full(arch) -> list:
    """Every candidate config for ``arch``, most-tuned first.

    "off" mode is always exactly this list truncated to its first entry, since a
    single fast default is what "off" means -- see get_fwd_prefill_configs.
    """
    if FWD_CONF_OVERRIDE:
        return [FWD_CONF_OVERRIDE]
    if arch.name == "gfx950":
        return [
            _fwd_cfg(128, 64, 2, False, stages=1, warps=4),
            _fwd_cfg(128, 64, 2, False, stages=1, warps=2),
            _fwd_cfg(128, 64, 2, False, stages=2, warps=4),
            _fwd_cfg(128, 128, 2, False, stages=2, warps=4),
        ]
    if arch.name == "gfx942":
        return [_fwd_cfg(128, 64, 2, False, stages=1, warps=4)]
    if arch.is_rdna:
        # gfx1151 (Strix Halo / RDNA3.5): tuned for the Qwen3-Omni ViT prefill shape
        # (B=1, S=3200, H=16, head_dim=72, fp16).
        if arch.name == "gfx1151":
            return [_fwd_cfg(128, 64, 2, False, stages=1, warps=8)]
        # NOTE: Tests expect BLOCK_N=32 for RDNA (except gfx1100); see
        # _get_block_size_n_triton() in test_flash_attn_triton_amd.py
        BLOCK_N = 64 if arch.name == "gfx1100" else 32
        return [
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": BLOCK_N, "PRE_LOAD_V": False, "waves_per_eu": 6},
                num_stages=1,
                num_warps=8,
            ),
        ]
    return [_fwd_cfg(64, 64, 2, False, stages=1, warps=4)]


def get_fwd_prefill_configs(mode: AutotuneMode):
    if mode == "off":
        return _fwd_configs_full(get_arch())[:1]
    elif mode == "on":
        return _fwd_configs_full(get_arch())
    else:  # sweep
        configs = []
        BLOCK_M_OPTIONS = [128, 64, 32, 16]
        BLOCK_N_OPTIONS = [128, 64, 32, 16]
        NUM_WARPS_OPTIONS = [2, 4, 8]
        NUM_STAGES_OPTIONS = [1, 2]
        WAVES_PER_EU_OPTIONS = [4, 2, 1]
        PRE_LOAD_V_OPTIONS = [False]
        for bm in BLOCK_M_OPTIONS:
            for bn in BLOCK_N_OPTIONS:
                for waves in WAVES_PER_EU_OPTIONS:
                    for nw in NUM_WARPS_OPTIONS:
                        for ns in NUM_STAGES_OPTIONS:
                            for preload_v in PRE_LOAD_V_OPTIONS:
                                configs.append(
                                    triton.Config(
                                        {
                                            "BLOCK_M": bm,
                                            "BLOCK_N": bn,
                                            "waves_per_eu": waves,
                                            "PRE_LOAD_V": preload_v,
                                        },
                                        num_stages=ns,
                                        num_warps=nw,
                                    )
                                )
        return configs


fwd_prefill_autotune_configs = get_fwd_prefill_configs(AUTOTUNE)


def _extend_fwd_configs(configs):
    """Give the forward an actual autotune space.

    The base table ships exactly ONE forward config even with AUTOTUNE=on, so nothing
    was being tuned. A sweep of 144 configs on MI300X (BLOCK_M/N, waves_per_eu,
    PRE_LOAD_V, num_warps, num_stages) found that config already optimal on its tile
    size -- but PRE_LOAD_V=True and num_stages=2 each won on some shapes, worth 2-6%.
    Add only those few variants: every extra config is benchmarked on the first call
    for each new shape, so a large space would cost more in warm-up than it returns.

    Only applied when autotuning is enabled; AUTOTUNE=off keeps the single base config.
    """
    if AUTOTUNE != "on" or FWD_CONF_OVERRIDE is not None or len(configs) != 1:
        return configs
    base = configs[0]
    extra = []
    for pre_load_v in (True,):
        for num_stages in (1, 2):
            kwargs = dict(base.kwargs)
            kwargs["PRE_LOAD_V"] = pre_load_v
            extra.append(triton.Config(kwargs, num_stages=num_stages, num_warps=base.num_warps))
    return configs + extra


fwd_prefill_autotune_configs = _extend_fwd_configs(fwd_prefill_autotune_configs)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    stride_kn,
    stride_vk,
    start_m,
    seqlen_k,
    seqlen_q,
    off_z,
    off_h_q,
    offs_m,
    offs_n,
    offs_d_qk,
    offs_d_v,
    block_min,
    block_max,
    n_extra_tokens,
    APPLY_MASK: tl.constexpr,  # True for masked blocks, False for full blocks
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    SM_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    ACCUMULATOR_TYPE,
    SCORE_MOD: tl.constexpr = None,
    MASK_MOD: tl.constexpr = None,
    SPARSE_IDX=None,
    BLOCK_SPARSE: tl.constexpr = False,
    BS_ALIGNED: tl.constexpr = False,
    AUX0=None,
    AUX1=None,
    AUX2=None,
    AUX3=None,
    NUM_AUX: tl.constexpr = 0,
    MASK_AUX0=None,
    MASK_AUX1=None,
    MASK_AUX2=None,
    MASK_AUX3=None,
    NUM_MASK_AUX: tl.constexpr = 0,
):
    """
    Unified attention forward inner loop.

    APPLY_MASK controls whether causal/window masking is applied:
    - False: Fast path for full blocks (no masking overhead)
    - True: Masked path with causal/window masking support

    BS_ALIGNED (block-sparse only) asserts every listed KV block lies wholly inside
    seqlen_k, so no block can straddle the end of the sequence and the per-block bounds
    masking is provably redundant. It does NOT relax mask_mod or causal/window masking,
    which are applied independently of the bounds checks below.
    """
    # A contiguous range identifies its last, partially-filled block by offset; a block
    # list cannot, so absent the alignment guarantee every sparse block has to be
    # bounded against seqlen_k. When it holds, all of that work drops out.
    SKIP_SEQ_BOUNDS: tl.constexpr = BLOCK_SPARSE and BS_ALIGNED
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # seqlen diff (only used when APPLY_MASK=True)
    seqlen_delta_qk = seqlen_k - seqlen_q

    # loop over k, v, and update accumulator.
    # BLOCK_SPARSE walks an explicit list of KV block indices instead of a contiguous
    # range; block_min/block_max then carry the iteration count rather than offsets.
    if BLOCK_SPARSE:
        n_iters = block_max
    else:
        n_iters = (block_max - block_min + BLOCK_N - 1) // BLOCK_N

    for blk_i in tl.range(0, n_iters, num_stages=1):
        if BLOCK_SPARSE:
            start_n = tl.load(SPARSE_IDX + blk_i).to(tl.int32) * BLOCK_N
        else:
            start_n = block_min + blk_i * BLOCK_N
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K - different masking for APPLY_MASK vs non-masked. An aligned
        # block-sparse list needs no seqlen bound, so it takes the unmasked path even
        # when APPLY_MASK is set for mask_mod's sake.
        if APPLY_MASK and not SKIP_SEQ_BOUNDS:
            # For masked blocks, check seqlen bounds
            k_mask = kv_offs_n[None, :] < seqlen_k
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_QK:
                k_mask = k_mask & (offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK)
            if PADDED_HEAD_V:
                v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            if PRE_LOAD_V:
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        else:
            # For full blocks, only check head dimension padding
            if PADDED_HEAD_QK:
                k_mask = offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK
                k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            else:
                k = tl.load(k_ptrs)
            if PRE_LOAD_V:
                if PADDED_HEAD_V:
                    v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                    v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                else:
                    v = tl.load(v_ptrs)

        # setup qk accumulator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # Apply extra token masking for partial blocks (only when APPLY_MASK=True).
        # -inf seeded into the accumulator survives the dot below (-inf + x = -inf).
        if BLOCK_SPARSE and not SKIP_SEQ_BOUNDS:
            # A sparse block list has no notion of "last block", so bound every block.
            qk = tl.where(kv_offs_n[None, :] < seqlen_k, qk, float("-inf"))
        elif (
            (not BLOCK_SPARSE) and APPLY_MASK and ((n_extra_tokens != 0) and (start_n + BLOCK_N == block_max))
        ):
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + offs_n[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        qk = tl.dot(q, k, acc=qk)
        qk_scaled = qk * SM_SCALE

        # score_mod is applied PRE-MASK, matching upstream FA-4 (its softcap is itself a
        # "scoremod_premask_fn"). Order matters: applied after masking, a non-additive mod
        # such as softcap would map the -inf of a masked position to a finite value
        # (tanh(-inf) = -1) and silently un-mask it.
        if SCORE_MOD is not None:
            qk_scaled = apply_score_mod(
                SCORE_MOD,
                qk_scaled,
                off_z,
                off_h_q,
                offs_m[:, None],
                kv_offs_n[None, :],
                AUX0,
                AUX1,
                AUX2,
                AUX3,
                NUM_AUX,
            )

        # Apply causal/sliding window masking (only when APPLY_MASK=True)
        if APPLY_MASK:
            if USE_SLIDING_WINDOW:
                if IS_CAUSAL:
                    # ========== CAUSAL SLIDING WINDOW MASKING ==========
                    row_idx = offs_m
                    col_idx = kv_offs_n
                    row_idx_expanded = row_idx[:, None]
                    col_idx_expanded = col_idx[None, :]

                    causal_offset = seqlen_k - seqlen_q
                    causal_mask = col_idx_expanded > (row_idx_expanded + causal_offset)

                    if WINDOW_SIZE_LEFT < 0 and WINDOW_SIZE_RIGHT < 0:
                        # both edges unbounded: only the causal cap constrains cols
                        window_mask = causal_mask
                    elif WINDOW_SIZE_LEFT < 0:
                        # unbounded left, finite right
                        window_mask = col_idx_expanded > (
                            row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT
                        )
                    elif WINDOW_SIZE_RIGHT < 0:
                        # unbounded right: the causal cap already bounds the right,
                        # so only the finite left edge masks (mirror of infinite-left)
                        left_bound = row_idx_expanded + causal_offset - WINDOW_SIZE_LEFT
                        window_mask = col_idx_expanded < left_bound
                    else:
                        left_bound = row_idx_expanded + causal_offset - WINDOW_SIZE_LEFT
                        right_bound = row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT
                        window_mask = (col_idx_expanded < left_bound) | (col_idx_expanded > right_bound)

                    mask = causal_mask | window_mask
                    qk_scaled = tl.where(mask, float("-inf"), qk_scaled)
                else:
                    # ========== NON-CAUSAL SLIDING WINDOW MASKING ==========
                    row_idx = offs_m
                    col_idx = kv_offs_n
                    sk = seqlen_k
                    sq = seqlen_q
                    row_idx_expanded = row_idx[:, None]
                    col_idx_expanded = col_idx[None, :]

                    if WINDOW_SIZE_LEFT < 0 and WINDOW_SIZE_RIGHT < 0:
                        # both edges unbounded: full (non-causal) attention
                        mask = (row_idx_expanded < 0) | (col_idx_expanded < 0)
                    elif WINDOW_SIZE_LEFT < 0:
                        # unbounded left, finite right
                        mask = col_idx_expanded > (row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT)
                    elif WINDOW_SIZE_RIGHT < 0:
                        # unbounded right, finite left (mirror of infinite-left)
                        mask = col_idx_expanded < (row_idx_expanded + sk - sq - WINDOW_SIZE_LEFT)
                    else:
                        sk_full = tl.full((1, BLOCK_N), sk, dtype=tl.int32)
                        right_bound_val = row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT
                        right_bound = tl.minimum(right_bound_val, sk_full)
                        left_bound = row_idx_expanded + sk - sq - WINDOW_SIZE_LEFT
                        mask = (col_idx_expanded > right_bound) | (col_idx_expanded < left_bound)

                    qk_scaled = tl.where(mask, float("-inf"), qk_scaled)
            else:
                if IS_CAUSAL:
                    causal_boundary = start_n + offs_n - seqlen_delta_qk
                    causal_mask = offs_m[:, None] >= causal_boundary[None, :]
                    qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))

        # mask_mod applied unconditionally (not gated by APPLY_MASK): it is user-defined
        # and may mask positions the block-skip logic, which only knows about
        # IS_CAUSAL/WINDOW, still visits.
        if MASK_MOD is not None:
            keep = apply_mask_mod(
                MASK_MOD,
                off_z,
                off_h_q,
                offs_m[:, None],
                kv_offs_n[None, :],
                MASK_AUX0,
                MASK_AUX1,
                MASK_AUX2,
                MASK_AUX3,
                NUM_MASK_AUX,
            )
            qk_scaled = tl.where(keep, qk_scaled, float("-inf"))

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        # Handle the case where all values are -inf
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk_scaled - m_ij[:, None])

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)

        # -- update output accumulator --
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]

        # Load V if not preloaded
        if not PRE_LOAD_V:
            if APPLY_MASK and not SKIP_SEQ_BOUNDS:
                v_mask = kv_offs_n[:, None] < seqlen_k
                if PADDED_HEAD_V:
                    v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                if PADDED_HEAD_V:
                    v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                    v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                else:
                    v = tl.load(v_ptrs)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc = tl.dot(p.to(v.type.element_ty), v, acc=acc)

    return acc, l_i, m_i


@triton.jit
def compute_window_bounds(
    q_start,
    q_end,
    diag,
    seqlen_k,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Calculate the window boundaries for a query block."""
    # Left boundary
    if WINDOW_SIZE_LEFT < 0:
        left_min = 0
        left_max = 0
    else:
        left_min = tl.maximum(0, q_start + diag - WINDOW_SIZE_LEFT)
        left_max = tl.maximum(0, q_end + diag - WINDOW_SIZE_LEFT)

    # Right boundary
    if IS_CAUSAL:
        # Causal cap: col <= row + diag
        right_min = tl.minimum(seqlen_k - 1, q_start + diag)
        right_max = tl.minimum(seqlen_k - 1, q_end + diag)
    else:
        if WINDOW_SIZE_RIGHT < 0:
            # Unbounded right edge: mirror the infinite-left collapse
            # (WINDOW_SIZE_LEFT < 0 -> left bound collapses to 0). Here the right
            # bound saturates at the last K index so classify_window_blocks
            # produces no back-skip / back-masked blocks.
            right_min = seqlen_k - 1
            right_max = seqlen_k - 1
        else:
            # Non-causal doesn't have the diagonal constraint
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)

    return left_min, left_max, right_min, right_max


@triton.jit
def classify_window_blocks(left_min, left_max, right_min, right_max, BLOCK_N: tl.constexpr):
    """Classify blocks based on window boundaries."""
    # First and last blocks that have ANY overlap with window
    first_block = left_min // BLOCK_N
    last_block = right_max // BLOCK_N

    # First block that is FULLY visible for all rows in Q block
    full_left_block = left_max // BLOCK_N + (left_max % BLOCK_N != 0)
    clipped_left = tl.minimum(full_left_block, last_block + 1)

    # Last block that is FULLY visible for all rows in Q block
    last_full_block_candidate = right_min // BLOCK_N
    if (last_full_block_candidate + 1) * BLOCK_N - 1 > right_min:
        last_full_block_candidate -= 1
    full_right_block = tl.maximum(last_full_block_candidate, clipped_left - 1)

    # Calculate counts
    n_front_skip_blocks = first_block
    n_front_masked_blocks = tl.maximum(0, clipped_left - first_block)
    n_full_blocks = tl.maximum(0, full_right_block - clipped_left + 1)
    n_back_masked_blocks = tl.maximum(0, last_block - full_right_block)

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        clipped_left,
    )  # Return clipped_left for padded block handling


@triton.jit
def handle_padded_last_block(
    n_extra_tokens,
    last_block,
    total_k_blocks,
    clipped_left,
    n_front_masked_blocks,
    n_full_blocks,
    n_back_masked_blocks,
):
    """Ensure a padded last K-block is never classified as 'full'.

    We move the padded last block (if visible) into the back-masked bucket.
    If it's already back-masked, we do nothing.  If it was counted in the
    front-masked range, we decrement front-masked; if it was counted as full,
    we decrement full.  Then we increment back-masked.
    """
    padded_last_k = (n_extra_tokens != 0) & (last_block == total_k_blocks - 1)

    if padded_last_k:
        # current 'full' range right edge
        full_right_block = clipped_left + n_full_blocks - 1

        # If last_block is already beyond full_right_block, it's already in back-masked -> nothing to do
        last_already_back_masked = last_block > full_right_block
        if not last_already_back_masked:
            # If the window starts past last_block, it was counted in front-masked
            if clipped_left > last_block:
                n_front_masked_blocks = tl.maximum(0, n_front_masked_blocks - 1)
            else:
                # Otherwise it was counted 'full' -> move it out of full
                n_full_blocks = tl.maximum(0, n_full_blocks - 1)
            # In both cases we need one more back-masked block
            n_back_masked_blocks = n_back_masked_blocks + 1

    return n_front_masked_blocks, n_full_blocks, n_back_masked_blocks


@triton.jit
def compute_padding_info(seqlen_k, BLOCK_N: tl.constexpr):
    """Calculate padding information for the last K block."""
    # check if we will need to do masking due either BLOCK_N being bigger than seqlen_k or seqlen_k not being a factor of BLOCK_N
    # n_extra_tokens = 10 % 4 = 2
    # This means the last K block has 2 valid tokens and 2 padding positions
    # K blocks visualization:
    #         Block 0         Block 1         Block 2 (last)
    #         K0 K1 K2 K3    K4 K5 K6 K7     K8 K9 ?? ??
    #         ?---------?    ?---------?     ?---? ?---?
    #         full block     full block      valid  pad
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0
    return n_extra_tokens


@triton.jit
def compute_block_masking(
    seqlen_k,
    seqlen_q,
    start_m,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Classify K blocks for attention computation with sliding window support.

    Returns:
        - n_front_skip_blocks: Blocks completely before the window
        - n_front_masked_blocks: Blocks partially overlapping window front
        - n_full_blocks: Blocks completely inside the window
        - n_back_masked_blocks: Blocks partially overlapping window back
        - n_extra_tokens: Padding tokens in last K block
    """

    # common
    q_start = start_m * BLOCK_M
    q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
    diag = seqlen_k - seqlen_q
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)

    if USE_SLIDING_WINDOW:
        # get window bounds
        left_min, left_max, right_min, right_max = compute_window_bounds(
            q_start,
            q_end,
            diag,
            seqlen_k,
            WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT,
            IS_CAUSAL,
        )

        # window vanishes -> early exit
        if right_max < left_min:
            return 0, 0, 0, 0, n_extra_tokens

        # classify blocks
        (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            clipped_left,
        ) = classify_window_blocks(left_min, left_max, right_min, right_max, BLOCK_N)

        # handle padded last block if needed
        if n_extra_tokens != 0:
            last_block = right_max // BLOCK_N
            n_front_masked_blocks, n_full_blocks, n_back_masked_blocks = handle_padded_last_block(
                n_extra_tokens,
                last_block,
                total_k_blocks,
                clipped_left,
                n_front_masked_blocks,
                n_full_blocks,
                n_back_masked_blocks,
            )
        return (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            n_extra_tokens,
        )
    else:
        if IS_CAUSAL:
            # ========== CAUSAL MODE: Classify K Blocks ==========
            # Calculate causal boundary for this Q block
            #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
            # Q0-Q3:   [ 1  0  0  0] [ 0  0  0  0] [ 0  0 -- --]  <- Q0
            #          [ 1  1  0  0] [ 0  0  0  0] [ 0  0 -- --]  <- Q1
            #          [ 1  1  1  0] [ 0  0  0  0] [ 0  0 -- --]  <- Q2
            #          [ 1  1  1  1] [ 1  1  0  0] [ 0  0 -- --]  <- Q3
            #                           /\ can see up to K5
            #
            # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  0] [ 0  0 -- --]  <- Q4
            #          [ 1  1  1  1] [ 1  1  1  1] [ 0  0 -- --]  <- Q5
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  0 -- --]  <- Q6
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -- --]  <- Q7

            # ------------------------------------------------------------
            # 1. figure out, in tokens, the right-most K position
            #    this Q-block may attend to
            # ------------------------------------------------------------
            k_max_token = q_end + diag  # last visible K index

            # this Q-block is entirely above the diagonal => nothing to do
            if k_max_token < 0:
                return 0, 0, 0, 0, n_extra_tokens

            k_max_token = tl.minimum(k_max_token, seqlen_k - 1)

            # ------------------------------------------------------------
            # 2. translate token indices into K-block indices
            # ------------------------------------------------------------
            last_visible_k_block = k_max_token // BLOCK_N
            n_visible_k_blocks = tl.minimum(last_visible_k_block + 1, total_k_blocks)

            # ------------------------------------------------------------
            # 3. classify those visible blocks
            #    - we *never* skip or mask blocks in front, because causal
            #      attention always starts at K0
            #    - the back side can require several masked blocks:
            #         o intersection of the causal diagonal with K-grid
            #           (at most  ?BLOCK_M / BLOCK_N? blocks)
            #         o plus one for partial K blocks at the causal boundary
            # ------------------------------------------------------------
            n_back_masked_blocks = BLOCK_M // BLOCK_N + 1
            n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_visible_k_blocks)

            n_front_skip_blocks = 0  # causal never skips the left side
            n_front_masked_blocks = 0  # ditto
            n_full_blocks = n_visible_k_blocks - n_back_masked_blocks
        else:
            # ========== NON-CAUSAL MODE ==========
            # Without causal mask, all positions can attend to all positions
            # Only need to handle the padding in the last block
            #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9  ??  ??]
            # Q0-Q3:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #
            # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -inf -inf]

            n_front_skip_blocks = 0  # never skips the left side
            n_front_masked_blocks = 0  # ditto
            if n_extra_tokens != 0:
                n_back_masked_blocks = 1  # Last block needs padding mask
                n_full_blocks = total_k_blocks - 1
            else:
                n_back_masked_blocks = 0  # All blocks are aligned
                n_full_blocks = total_k_blocks

        return (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            n_extra_tokens,
        )


@triton.autotune(
    configs=fwd_prefill_autotune_configs,
    key=FWD_PREFILL_AUTOTUNE_KEYS,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    LSE,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_EXP2: tl.constexpr,
    FORCE_MASKING: tl.constexpr,
    NUM_XCD: tl.constexpr = 1,
    HEAD_STRIDE_ALIGNED_8: tl.constexpr = False,
    SCORE_MOD: tl.constexpr = None,
    MASK_MOD: tl.constexpr = None,
    SINK=None,
    stride_sink_h=0,
    USE_SINK: tl.constexpr = False,
    BS_MASK_CNT=None,
    BS_MASK_IDX=None,
    BS_FULL_CNT=None,
    BS_FULL_IDX=None,
    stride_bs_cnt_b=0,
    stride_bs_cnt_h=0,
    stride_bs_cnt_m=0,
    stride_bs_idx_b=0,
    stride_bs_idx_h=0,
    stride_bs_idx_m=0,
    stride_bs_fidx_b=0,
    stride_bs_fidx_h=0,
    stride_bs_fidx_m=0,
    BLOCK_SPARSE: tl.constexpr = False,
    BS_HAS_FULL: tl.constexpr = False,
    BS_ALIGNED: tl.constexpr = False,
    NUM_SPLITS: tl.constexpr = 1,
    stride_o_split=0,
    stride_lse_split=0,
    AUX0=None,
    AUX1=None,
    AUX2=None,
    AUX3=None,
    NUM_AUX: tl.constexpr = 0,
    MASK_AUX0=None,
    MASK_AUX1=None,
    MASK_AUX2=None,
    MASK_AUX3=None,
    NUM_MASK_AUX: tl.constexpr = 0,
):
    # set params
    ACCUMULATOR_TYPE = tl.float32

    # compute offsets
    off_h_q = tl.program_id(0)
    # apply the xcd remapping for the hq dim
    off_h_q = remap_xcd(off_h_q, HQ, NUM_XCD)

    start_m = tl.program_id(1)
    # With NUM_SPLITS > 1 the KV loop is sliced across programs to raise occupancy; the
    # split index is folded into the batch grid dimension (triton grids are 3-D).
    if NUM_SPLITS > 1:
        off_z = tl.program_id(2) // NUM_SPLITS
        split_id = tl.program_id(2) % NUM_SPLITS
    else:
        off_z = tl.program_id(2)
        split_id = 0
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q
    # Determine if we need to mask the heads
    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    # handle seqlen
    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

        # we have a one-size-fits-all grid in id(0). Some seqlens might be too small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Block-sparse: the visited KV blocks come from an explicit list, so the
    # causal/window block geometry below is bypassed entirely.
    bs_mask_cnt = 0
    bs_full_cnt = 0
    bs_mask_idx_ptr = BS_MASK_IDX
    bs_full_idx_ptr = BS_FULL_IDX
    if BLOCK_SPARSE:
        cnt_off = off_z * stride_bs_cnt_b + off_h_q * stride_bs_cnt_h + start_m * stride_bs_cnt_m
        idx_off = off_z * stride_bs_idx_b + off_h_q * stride_bs_idx_h + start_m * stride_bs_idx_m
        bs_mask_cnt = tl.load(BS_MASK_CNT + cnt_off).to(tl.int32)
        bs_mask_idx_ptr = BS_MASK_IDX + idx_off
        if BS_HAS_FULL:
            # The full list has its own max-entries dimension, so its strides differ
            # from the partial list's; they cannot share an offset.
            fidx_off = off_z * stride_bs_fidx_b + off_h_q * stride_bs_fidx_h + start_m * stride_bs_fidx_m
            bs_full_cnt = tl.load(BS_FULL_CNT + cnt_off).to(tl.int32)
            bs_full_idx_ptr = BS_FULL_IDX + fidx_off
        if NUM_SPLITS > 1:
            # Split-KV over an explicit block list: hand each split a contiguous slice
            # of the list. The contiguous path splits by KV *offset* (split_lo/split_hi),
            # which a jumping index list cannot use -- without this every split would
            # walk the whole list and recompute the entire result, costing NUM_SPLITS
            # times the work for a bitwise-identical answer.
            full_lo = (bs_full_cnt * split_id) // NUM_SPLITS
            bs_full_cnt = (bs_full_cnt * (split_id + 1)) // NUM_SPLITS - full_lo
            bs_full_idx_ptr = bs_full_idx_ptr + full_lo
            mask_lo = (bs_mask_cnt * split_id) // NUM_SPLITS
            bs_mask_cnt = (bs_mask_cnt * (split_id + 1)) // NUM_SPLITS - mask_lo
            bs_mask_idx_ptr = bs_mask_idx_ptr + mask_lo

    # figure out masking pattern
    (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    ) = compute_block_masking(
        seqlen_k,
        seqlen_q,
        start_m,
        IS_CAUSAL,
        USE_SLIDING_WINDOW,
        WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT,
        BLOCK_M,
        BLOCK_N,
    )

    # ============================================================
    #          PROGRAM EARLY EXIT (All K Blocks Skipped)
    # ============================================================
    # KV window owned by this split, in elements. Splits carve the key axis into
    # NUM_SPLITS contiguous chunks aligned to BLOCK_N so no block straddles two splits.
    if NUM_SPLITS > 1:
        total_k_blocks_split = tl.cdiv(seqlen_k, BLOCK_N)
        blocks_per_split = tl.cdiv(total_k_blocks_split, NUM_SPLITS)
        split_lo = split_id * blocks_per_split * BLOCK_N
        split_hi = tl.minimum((split_id + 1) * blocks_per_split * BLOCK_N, seqlen_k)
    else:
        split_lo = 0
        split_hi = seqlen_k

    if BLOCK_SPARSE:
        total_visible_blocks = bs_mask_cnt + bs_full_cnt
    else:
        total_visible_blocks = n_front_masked_blocks + n_full_blocks + n_back_masked_blocks
    # split_lo/split_hi partition the KV *offset* range, which only the contiguous path
    # uses; block-sparse splits its index list instead (above), so an empty offset range
    # says nothing about whether this split has list entries.
    if (not BLOCK_SPARSE) and NUM_SPLITS > 1 and split_hi <= split_lo:
        total_visible_blocks = 0
    if total_visible_blocks == 0:
        """
        No K blocks visible - write zeros and exit.
        """
        # Write zeros to output
        o_offset = (
            Out
            + split_id * stride_o_split
            + off_z * stride_oz
            + off_h_q * stride_oh
            + cu_seqlens_q_start * stride_om
        )
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
        o_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_V:
            o_mask = o_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
        tl.store(
            o_ptrs,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty),
            mask=o_mask,
        )

        # Write zeros to LSE
        l_ptrs = (
            LSE
            + split_id * stride_lse_split
            + off_z * stride_lse_z
            + off_h_q * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + offs_m * stride_lse_m
        )
        # A split that owns no visible blocks must report -inf, not 0, or the combine
        # would fold in a spurious weight-1 contribution. The non-split path keeps the
        # historical 0 fill, which downstream code expects for a fully-masked row.
        if NUM_SPLITS > 1:
            empty_lse = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        else:
            empty_lse = tl.zeros([BLOCK_M], dtype=tl.float32)
        tl.store(l_ptrs, empty_lse, mask=offs_m < seqlen_q)
        return

    # ============================================================
    #         NORMAL PROCESSING (Some K Blocks Visible)
    # ============================================================
    """
    This program has visible K blocks to process.
    We'll use two calls to handle different block types efficiently.
    """

    # Initialize for processing
    # Compute pointers for all the tensors used in this kernel.
    # When the caller guarantees that the head-axis strides of Q/K/V are
    # multiples of 8 elements (set via HEAD_STRIDE_ALIGNED_8), the head-axis
    # offset is a multiple of 8 *elements*. The resulting byte alignment is
    # element-size dependent: 16 bytes for 16-bit types (fp16/bf16), 8 bytes
    # for fp8, 32 bytes for fp32. The 16-byte case is the one that lets
    # AxisInfo widen the K/V global load to a 128-bit (buffer_load_b128)
    # access; for the other dtypes the hint is still sound but yields a
    # different (smaller or larger) vector width. Auto-specialization only
    # fires at the 16-element threshold, so hint the smaller multiple
    # explicitly.
    qh_off = off_h_q * stride_qh
    kh_off = off_h_k * stride_kh
    vh_off = off_h_k * stride_vh
    if HEAD_STRIDE_ALIGNED_8:
        qh_off = tl.multiple_of(qh_off, 8)
        kh_off = tl.multiple_of(kh_off, 8)
        vh_off = tl.multiple_of(vh_off, 8)

    q_offset = Q + off_z * stride_qz + qh_off + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d_qk[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vn

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACCUMULATOR_TYPE)

    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk[None, :] < ACTUAL_BLOCK_DMODEL_QK)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # ========== Process MASKED K Blocks in the front ==========
    # NOTE: we use USE_SLIDING_WINDOW as guard because the compiler will crash other wise. front masking is only for sliding window so that is fine.
    if BLOCK_SPARSE:
        # Two passes over the explicit block lists: fully-unmasked blocks first (no
        # masking work at all), then partial blocks with masking + mask_mod applied.
        # block_min is unused in sparse mode; block_max carries the iteration count.
        if BS_HAS_FULL:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                stride_kn,
                stride_vk,
                start_m,
                seqlen_k,
                seqlen_q,
                off_z,
                off_h_q,
                offs_m,
                offs_n,
                offs_d_qk,
                offs_d_v,
                0,
                bs_full_cnt,
                0,
                APPLY_MASK=False,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
                BLOCK_DMODEL_V=BLOCK_DMODEL_V,
                BLOCK_N=BLOCK_N,
                PRE_LOAD_V=PRE_LOAD_V,
                PADDED_HEAD_QK=PADDED_HEAD_QK,
                PADDED_HEAD_V=PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK=ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V=ACTUAL_BLOCK_DMODEL_V,
                SM_SCALE=SM_SCALE,
                USE_EXP2=USE_EXP2,
                USE_SLIDING_WINDOW=False,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
                SCORE_MOD=SCORE_MOD,
                AUX0=AUX0,
                AUX1=AUX1,
                AUX2=AUX2,
                AUX3=AUX3,
                NUM_AUX=NUM_AUX,
                MASK_AUX0=MASK_AUX0,
                MASK_AUX1=MASK_AUX1,
                MASK_AUX2=MASK_AUX2,
                MASK_AUX3=MASK_AUX3,
                NUM_MASK_AUX=NUM_MASK_AUX,
                MASK_MOD=None,
                SPARSE_IDX=bs_full_idx_ptr,
                BLOCK_SPARSE=True,
                BS_ALIGNED=BS_ALIGNED,
            )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            stride_kn,
            stride_vk,
            start_m,
            seqlen_k,
            seqlen_q,
            off_z,
            off_h_q,
            offs_m,
            offs_n,
            offs_d_qk,
            offs_d_v,
            0,
            bs_mask_cnt,
            0,
            APPLY_MASK=True,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V=BLOCK_DMODEL_V,
            BLOCK_N=BLOCK_N,
            PRE_LOAD_V=PRE_LOAD_V,
            PADDED_HEAD_QK=PADDED_HEAD_QK,
            PADDED_HEAD_V=PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK=ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V=ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE=SM_SCALE,
            USE_EXP2=USE_EXP2,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
            SCORE_MOD=SCORE_MOD,
            AUX0=AUX0,
            AUX1=AUX1,
            AUX2=AUX2,
            AUX3=AUX3,
            NUM_AUX=NUM_AUX,
            MASK_AUX0=MASK_AUX0,
            MASK_AUX1=MASK_AUX1,
            MASK_AUX2=MASK_AUX2,
            MASK_AUX3=MASK_AUX3,
            NUM_MASK_AUX=NUM_MASK_AUX,
            MASK_MOD=MASK_MOD,
            SPARSE_IDX=bs_mask_idx_ptr,
            BLOCK_SPARSE=True,
            BS_ALIGNED=BS_ALIGNED,
        )
    else:
        if n_front_masked_blocks > 0 and USE_SLIDING_WINDOW:
            block_min = n_front_skip_blocks * BLOCK_N
            block_max = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N

            # Restrict this phase to the split's KV window.
            if NUM_SPLITS > 1:
                block_min = tl.maximum(block_min, split_lo)
                block_max = tl.minimum(block_max, split_hi)
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                stride_kn,
                stride_vk,
                start_m,
                seqlen_k,
                seqlen_q,
                off_z,
                off_h_q,
                offs_m,
                offs_n,
                offs_d_qk,
                offs_d_v,
                block_min,  # Start of front masked blocks
                block_max,  # End of front masked blocks
                0,  # n_extra_tokens (0 for front blocks, only relevant for last block)
                APPLY_MASK=True,  # Masked blocks
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
                BLOCK_DMODEL_V=BLOCK_DMODEL_V,
                BLOCK_N=BLOCK_N,
                PRE_LOAD_V=PRE_LOAD_V,
                PADDED_HEAD_QK=PADDED_HEAD_QK,
                PADDED_HEAD_V=PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK=ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V=ACTUAL_BLOCK_DMODEL_V,
                SM_SCALE=SM_SCALE,
                USE_EXP2=USE_EXP2,
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
                SCORE_MOD=SCORE_MOD,
                AUX0=AUX0,
                AUX1=AUX1,
                AUX2=AUX2,
                AUX3=AUX3,
                NUM_AUX=NUM_AUX,
                MASK_AUX0=MASK_AUX0,
                MASK_AUX1=MASK_AUX1,
                MASK_AUX2=MASK_AUX2,
                MASK_AUX3=MASK_AUX3,
                NUM_MASK_AUX=NUM_MASK_AUX,
                MASK_MOD=MASK_MOD,
            )

        # ========== Process FULL K Blocks (Fast Path) ==========
        if n_full_blocks > 0:
            block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
            block_max = (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks) * BLOCK_N

            # Restrict this phase to the split's KV window.
            if NUM_SPLITS > 1:
                block_min = tl.maximum(block_min, split_lo)
                block_max = tl.minimum(block_max, split_hi)
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                stride_kn,
                stride_vk,
                start_m,
                seqlen_k,
                seqlen_q,
                off_z,
                off_h_q,
                offs_m,
                offs_n,
                offs_d_qk,
                offs_d_v,
                block_min,  # Start of range: 0
                block_max,  # End of range: n_full_blocks * BLOCK_N
                0,  # n_extra_tokens (not used for full blocks)
                APPLY_MASK=FORCE_MASKING,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
                BLOCK_DMODEL_V=BLOCK_DMODEL_V,
                BLOCK_N=BLOCK_N,
                PRE_LOAD_V=PRE_LOAD_V,
                PADDED_HEAD_QK=PADDED_HEAD_QK,
                PADDED_HEAD_V=PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK=ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V=ACTUAL_BLOCK_DMODEL_V,
                SM_SCALE=SM_SCALE,
                USE_EXP2=USE_EXP2,
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
                SCORE_MOD=SCORE_MOD,
                AUX0=AUX0,
                AUX1=AUX1,
                AUX2=AUX2,
                AUX3=AUX3,
                NUM_AUX=NUM_AUX,
                MASK_AUX0=MASK_AUX0,
                MASK_AUX1=MASK_AUX1,
                MASK_AUX2=MASK_AUX2,
                MASK_AUX3=MASK_AUX3,
                NUM_MASK_AUX=NUM_MASK_AUX,
                MASK_MOD=MASK_MOD,
            )

        # ========== Process MASKED K Blocks in the back ==========
        if n_back_masked_blocks > 0:
            block_min = (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks) * BLOCK_N
            block_max = (
                n_front_skip_blocks + n_front_masked_blocks + n_full_blocks + n_back_masked_blocks
            ) * BLOCK_N

            # Restrict this phase to the split's KV window.
            if NUM_SPLITS > 1:
                block_min = tl.maximum(block_min, split_lo)
                block_max = tl.minimum(block_max, split_hi)
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                stride_kn,
                stride_vk,
                start_m,
                seqlen_k,
                seqlen_q,
                off_z,
                off_h_q,
                offs_m,
                offs_n,
                offs_d_qk,
                offs_d_v,
                block_min,  # Start of range: n_full_blocks * BLOCK_N
                block_max,  # End of range: n_visible_k_blocks * BLOCK_N
                n_extra_tokens,  # Padding tokens in last block
                APPLY_MASK=True,  # Masked blocks
                IS_CAUSAL=IS_CAUSAL,  # Use actual causal flag
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
                BLOCK_DMODEL_V=BLOCK_DMODEL_V,
                BLOCK_N=BLOCK_N,
                PRE_LOAD_V=PRE_LOAD_V,
                PADDED_HEAD_QK=PADDED_HEAD_QK,
                PADDED_HEAD_V=PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK=ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V=ACTUAL_BLOCK_DMODEL_V,
                SM_SCALE=SM_SCALE,
                USE_EXP2=USE_EXP2,
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
                SCORE_MOD=SCORE_MOD,
                AUX0=AUX0,
                AUX1=AUX1,
                AUX2=AUX2,
                AUX3=AUX3,
                NUM_AUX=NUM_AUX,
                MASK_AUX0=MASK_AUX0,
                MASK_AUX1=MASK_AUX1,
                MASK_AUX2=MASK_AUX2,
                MASK_AUX3=MASK_AUX3,
                NUM_MASK_AUX=NUM_MASK_AUX,
                MASK_MOD=MASK_MOD,
            )

    # ============================================================
    #                        EPILOGUE
    # ============================================================
    # Handle invalid rows: rows with no valid keys to attend to.
    # This occurs with sliding window or causal attention (when seqlen_q > seqlen_k).
    # For invalid rows: m_i = -inf, l_i = 0, acc = 0.
    # We set l_i = 1.0 to avoid division by zero and ensure LSE = -inf.
    invalid_mask = m_i == float("-inf")

    # Learnable attention sink. The sink is one extra
    # per-head logit competing in the softmax with no value vector behind it, so it only
    # enters the denominator. Folding it in here -- after the KV loop, against the final
    # row max -- rather than inside the loop leaves the online rescale untouched and
    # fixes both the output and the LSE below in one place. A fully-masked row keeps
    # m_i = -inf and must stay zero, hence the invalid_mask guard.
    if USE_SINK:
        sink_logit = tl.load(SINK + off_h_q * stride_sink_h).to(tl.float32)
        m_i_safe_sink = tl.where(invalid_mask, 0.0, m_i)
        if USE_EXP2:
            RCP_LN2_SINK: tl.constexpr = 1.4426950408889634
            sink_term = tl.math.exp2((sink_logit - m_i_safe_sink) * RCP_LN2_SINK)
        else:
            sink_term = tl.math.exp(sink_logit - m_i_safe_sink)
        l_i = l_i + tl.where(invalid_mask, 0.0, sink_term)

    l_i_safe = tl.where(invalid_mask, 1.0, l_i)
    l_recip = 1 / l_i_safe[:, None]
    acc = acc * l_recip

    # compute log-sum-exp
    # NOTE: l_i here already includes the sink term when USE_SINK, so the LSE written
    # out is the sink-inclusive normalizer. The backward relies on that: it recovers
    # p = exp(qk - lse), which is then automatically sink-normalized, so dq/dk/dv need
    # no sink-specific kernel change at all.
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        softmax_lse = (m_i * RCP_LN2 + tl.math.log2(l_i_safe)) * LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i_safe)

    # Ensure invalid rows have LSE = -inf
    softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = (
        LSE
        + split_id * stride_lse_split
        + off_z * stride_lse_z
        + off_h_q * stride_lse_h
        + cu_seqlens_q_start * stride_lse_m
    )
    l_ptrs = l_offset + offs_m * stride_lse_m

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last Q block. For others, overflow_size will be -ve
    end_m_idx = (start_m + 1) * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, softmax_lse)

    # write back O
    o_offset = (
        Out
        + split_id * stride_o_split
        + off_z * stride_oz
        + off_h_q * stride_oh
        + cu_seqlens_q_start * stride_om
    )
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL_V], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD_V:
        o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


def attention_forward_prefill_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    sm_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    layout: Literal["bshd", "bhsd", "thd"],
    # varlen
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlens_q: int,
    max_seqlens_k: int,
    # misc
    use_exp2: bool,
    # score_mod / mask_mod and learnable_sink:
    score_mod=None,
    mask_mod=None,
    learnable_sink=None,
    block_sparse=None,
    num_splits: int = 1,
    out_partial=None,
    lse_partial=None,
    aux_tensors=None,
    mask_aux_tensors=None,
):
    # get params, strides and shape
    IS_VARLEN = layout == "thd"

    # common assertions
    assert q.device == k.device == v.device == o.device, (
        f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}"
    )
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    current_device = torch.cuda.current_device()
    assert q.is_cuda and q.device.index == current_device, (
        f"Device mismatch: Kernel will launch on cuda:{current_device}, but tensors are on {q.device}"
    )

    # get shapes and strides
    if IS_VARLEN:
        # shape
        total_seqlen_q, nheads_q, head_size_q = q.shape
        _total_seqlen_k, nheads_k, head_size_k = k.shape
        _total_seqlen_v, nheads_v, head_size_v = v.shape

        # assert shapes
        assert cu_seqlens_q is not None, "cu_seqlens_q must be provided for varlen layout"
        assert cu_seqlens_k is not None, "cu_seqlens_k must be provided for varlen layout"
        assert max_seqlens_q is not None and max_seqlens_q > 0, (
            "max_seqlens_q must be provided and positive for varlen layout"
        )
        assert max_seqlens_k is not None and max_seqlens_k > 0, (
            "max_seqlens_k must be provided and positive for varlen layout"
        )

        # assert head dimensions
        assert head_size_q == head_size_k, f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, (
            f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        )

        # assert output shapes
        assert o.shape == (
            total_seqlen_q,
            nheads_q,
            head_size_v,
        ), f"o shape {o.shape} != expected {(total_seqlen_q, nheads_q, head_size_v)}"

        # assert cu_seqlens
        assert cu_seqlens_q.dtype == torch.int32, f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}"
        assert cu_seqlens_k.dtype == torch.int32, f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}"

        # set vars
        batch = len(cu_seqlens_q) - 1
        head_size_qk = head_size_q

        # Assert softmax_lse tensor is large enough
        assert softmax_lse.shape[0] >= nheads_q, (
            f"softmax_lse.shape[0]={softmax_lse.shape[0]} must be >= nheads_q={nheads_q}"
        )
        assert softmax_lse.shape[1] >= total_seqlen_q, (
            f"softmax_lse.shape[1]={softmax_lse.shape[1]} must be >= total_seqlen_q={total_seqlen_q}"
        )
        assert softmax_lse.dtype == torch.float32, f"softmax_lse must be float32, got {softmax_lse.dtype}"
        assert softmax_lse.device == q.device, "softmax_lse must be on same device as q"

        # strides -- variable order is (b, h, m, d); varlen tensors store (m, h, d)
        # (no batch axis; cu_seqlens carries the per-sequence offsets), hence the swap.
        stride_qb, stride_qh, stride_qm, stride_qd = 0, q.stride(1), q.stride(0), q.stride(2)
        stride_kb, stride_kh, stride_kn, stride_kd = 0, k.stride(1), k.stride(0), k.stride(2)
        stride_vb, stride_vh, stride_vn, stride_vd = 0, v.stride(1), v.stride(0), v.stride(2)
        stride_ob, stride_oh, stride_om, stride_od = 0, o.stride(1), o.stride(0), o.stride(2)
        stride_lse_z, stride_lse_h, stride_lse_m = 0, *softmax_lse.stride()
    else:
        # shapes
        batch_q, seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, seqlen_k, nheads_k, head_size_k = k.shape
        batch_v, seqlen_v, nheads_v, head_size_v = v.shape

        # assert batch dimensions
        assert batch_q == batch_k == batch_v, f"batch sizes must match: q={batch_q}, k={batch_k}, v={batch_v}"

        # assert head dimensions
        assert head_size_q == head_size_k, f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, (
            f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        )

        # assert sequence lengths
        assert seqlen_k == seqlen_v, f"k and v sequence lengths must match: k={seqlen_k}, v={seqlen_v}"

        # assert output shapes
        assert o.shape == (
            batch_q,
            seqlen_q,
            nheads_q,
            head_size_v,
        ), f"o shape {o.shape} != expected {(batch_q, seqlen_q, nheads_q, head_size_v)}"

        # set vars
        batch = batch_q
        head_size_qk = head_size_q
        max_seqlens_q = seqlen_q
        max_seqlens_k = seqlen_k

        # Assert softmax_lse tensor is large enough
        assert softmax_lse.shape[0] >= batch, (
            f"softmax_lse.shape[0]={softmax_lse.shape[0]} must be >= batch={batch}"
        )
        assert softmax_lse.shape[1] >= nheads_q, (
            f"softmax_lse.shape[1]={softmax_lse.shape[1]} must be >= nheads_q={nheads_q}"
        )
        assert softmax_lse.shape[2] >= seqlen_q, (
            f"softmax_lse.shape[2]={softmax_lse.shape[2]} must be >= seqlen_q={seqlen_q}"
        )
        assert softmax_lse.dtype == torch.float32, f"softmax_lse must be float32, got {softmax_lse.dtype}"
        assert softmax_lse.device == q.device, "softmax_lse must be on same device as q"

        # strides -- variable order is (b, h, m, d); bshd tensors store (b, m, h, d),
        # hence the h/m swap.
        stride_qb, stride_qh, stride_qm, stride_qd = q.stride(0), q.stride(2), q.stride(1), q.stride(3)
        stride_kb, stride_kh, stride_kn, stride_kd = k.stride(0), k.stride(2), k.stride(1), k.stride(3)
        stride_vb, stride_vh, stride_vn, stride_vd = v.stride(0), v.stride(2), v.stride(1), v.stride(3)
        stride_ob, stride_oh, stride_om, stride_od = o.stride(0), o.stride(2), o.stride(1), o.stride(3)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # check output dtype matches input dtype
    assert o.dtype == q.dtype, f"Output dtype {o.dtype} must match input dtype {q.dtype}"

    # check features
    use_sliding_window = window_size_left != -1 or window_size_right != -1
    # Either edge may be unbounded and is handled uniformly: WINDOW_SIZE_LEFT < 0
    # collapses the left bound to 0, WINDOW_SIZE_RIGHT < 0 saturates the right bound
    # at the last K index (compute_window_bounds + the per-element masks mirror the
    # two edges). (-1, -1) is the only "off" sentinel (leaves use_sliding_window
    # False), so no negative-right guard is needed.

    # Get closest power of 2 over or equal to 32 for both QK and V dimensions
    padded_d_model_qk = 1 << (head_size_qk - 1).bit_length()
    padded_d_model_v = 1 << (head_size_v - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model_qk = max(padded_d_model_qk, 16)
    padded_d_model_v = max(padded_d_model_v, 16)

    # Detect if we need to force masking for all blocks (required on some architectures)
    arch = get_arch()
    force_masking = arch.is_rdna

    num_xcd = 1 if arch.is_rdna else 8

    # Soundness precondition for the `tl.multiple_of` head-stride hint inside
    # `attn_fwd`: only enable it when every Q/K/V head-axis stride is a
    # multiple of 8 elements. With a non-contiguous input (e.g. a transposed
    # view) stride_*h need not equal head_dim, so the head_dim constexpr
    # alone is not enough.
    head_stride_aligned_8 = stride_qh % 8 == 0 and stride_kh % 8 == 0 and stride_vh % 8 == 0

    # launch kernel. With num_splits > 1 the kernel writes per-split partials, which the
    # caller reduces; the split index rides in the batch grid dimension.
    if num_splits > 1:
        out_target = out_partial
        lse_target = lse_partial
        stride_o_split = out_partial.stride(0)
        stride_lse_split = lse_partial.stride(0)
    else:
        out_target = o
        lse_target = softmax_lse
        stride_o_split = 0
        stride_lse_split = 0

    def grid(META):
        return (
            nheads_q,
            triton.cdiv(max_seqlens_q, META["BLOCK_M"]),
            batch * num_splits,
        )

    # MLA / large head_dim: the tuned configs use BLOCK_M=128, whose Q tile overflows
    # CDNA3's 64 KiB LDS once padded_d_model_qk >= 512. Rather
    # than filter the autotuner's config list (triton only runs early_config_prune when
    # more than one config is present, so that would silently miss AUTOTUNE=off), bypass
    # the autotuner and launch the raw JITFunction with a block size that fits. Normal
    # head dims never take this path, so the tuned fast path is untouched.
    # Block-sparse requires the kernel tile to match the sparsity granularity exactly,
    # so the autotuner is bypassed and the block sizes are pinned to block_size.
    bs_aligned = False
    if block_sparse is not None:
        bs_q, bs_kv = block_sparse.block_size
        lds_cap = max_block_for_lds(padded_d_model_qk, q.element_size())
        if lds_cap < bs_q:
            raise ValueError(
                f"block_sparse q block {bs_q} exceeds the LDS budget for head_dim "
                f"{head_size_qk} on this GPU (max {lds_cap})"
            )
        # Every KV block is a whole BLOCK_N tile inside the sequence, so no block the
        # kernel visits can straddle its end and the per-block bounds masking is
        # redundant. Left False under varlen: each sequence ends at its own offset, and
        # the block grid is padded to the longest, so short sequences do have partial
        # trailing blocks. Deciding this from the grid width costs no device sync.
        if not IS_VARLEN:
            bs_aligned = block_sparse.blocks_fit_within(max_seqlens_k)

    cap_block_m = max_block_for_lds(padded_d_model_qk, q.element_size())
    tuned_block_m = max((c.kwargs.get("BLOCK_M", 0) for c in fwd_prefill_autotune_configs), default=0)
    if block_sparse is not None:
        launcher = attn_fwd.fn[grid]
        block_overrides = dict(BLOCK_M=bs_q, BLOCK_N=bs_kv, **sparse_fwd_default(bs_kv))
    elif cap_block_m < tuned_block_m:
        if cap_block_m == 0:
            raise ValueError(
                f"head_dim {head_size_qk} (padded to {padded_d_model_qk}) is too large for "
                f"this GPU's LDS budget in the forward kernel"
            )
        launcher = attn_fwd.fn[grid]
        block_overrides = dict(
            BLOCK_M=cap_block_m,
            BLOCK_N=min(64, cap_block_m),
            waves_per_eu=2,
            PRE_LOAD_V=False,
            num_stages=1,
            num_warps=4,
        )
    else:
        launcher = attn_fwd[grid]
        block_overrides = {}

    def _launch(_overrides):
        """The kernel launch, parameterised by the tile/knob overrides.

        Factored out so the block-sparse path can time candidate knob sets before
        committing to one -- see pick_knobs() in utils.py.
        """
        launcher(
            q,
            k,
            v,
            lse_target,
            out_target,
            stride_qb,
            stride_qh,
            stride_qm,
            stride_qd,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_vd,
            stride_ob,
            stride_oh,
            stride_om,
            stride_od,
            stride_lse_z,
            stride_lse_h,
            stride_lse_m,
            cu_seqlens_q,
            cu_seqlens_k,
            HQ=nheads_q,
            HK=nheads_k,
            ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
            ACTUAL_BLOCK_DMODEL_V=head_size_v,
            MAX_SEQLENS_Q=max_seqlens_q,
            MAX_SEQLENS_K=max_seqlens_k,
            SM_SCALE=sm_scale,
            IS_CAUSAL=causal,
            USE_SLIDING_WINDOW=use_sliding_window,
            WINDOW_SIZE_LEFT=window_size_left,
            WINDOW_SIZE_RIGHT=window_size_right,
            IS_VARLEN=IS_VARLEN,
            BLOCK_DMODEL_QK=padded_d_model_qk,
            BLOCK_DMODEL_V=padded_d_model_v,
            USE_EXP2=use_exp2,
            FORCE_MASKING=force_masking,
            NUM_XCD=num_xcd,
            HEAD_STRIDE_ALIGNED_8=head_stride_aligned_8,
            SCORE_MOD=score_mod,
            MASK_MOD=mask_mod,
            SINK=learnable_sink,
            stride_sink_h=0 if learnable_sink is None else learnable_sink.stride(0),
            USE_SINK=learnable_sink is not None,
            BS_MASK_CNT=None if block_sparse is None else block_sparse.mask_block_cnt,
            BS_MASK_IDX=None if block_sparse is None else block_sparse.mask_block_idx,
            BS_FULL_CNT=None if block_sparse is None else block_sparse.full_block_cnt,
            BS_FULL_IDX=None if block_sparse is None else block_sparse.full_block_idx,
            **bs_tensor_strides("bs_cnt", None if block_sparse is None else block_sparse.mask_block_cnt),
            **bs_tensor_strides("bs_idx", None if block_sparse is None else block_sparse.mask_block_idx),
            **bs_tensor_strides("bs_fidx", None if block_sparse is None else block_sparse.full_block_idx),
            BLOCK_SPARSE=block_sparse is not None,
            BS_HAS_FULL=block_sparse is not None and block_sparse.full_block_cnt is not None,
            BS_ALIGNED=bs_aligned,
            NUM_SPLITS=num_splits,
            stride_o_split=stride_o_split,
            stride_lse_split=stride_lse_split,
            **aux_kernel_args(aux_tensors),
            **mask_aux_kernel_args(mask_aux_tensors),
            **_overrides,
        )

    if block_sparse is not None and AUTOTUNE != "off":
        # Choose once per shape, then launch normally. The key mirrors
        # FWD_PREFILL_AUTOTUNE_KEYS, plus the block size the tiles are pinned to.
        tune_key = (
            bs_q,
            bs_kv,
            causal,
            max_seqlens_q,
            max_seqlens_k,
            head_size_qk,
            head_size_v,
            IS_VARLEN,
            nheads_q,
            nheads_k,
        )
        candidates = [dict(BLOCK_M=bs_q, BLOCK_N=bs_kv, **knobs) for knobs in sparse_fwd_candidates(bs_kv)]
        block_overrides = pick_knobs(_SPARSE_FWD_CHOICE, tune_key, candidates, _launch)

    _launch(block_overrides)
