###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Adapted from AITER (https://github.com/ROCm/aiter, MIT License),
# aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py.
# Modified by the Primus-Turbo team; see NOTICE.md in this directory.
###############################################################################

from typing import Literal

import torch
import triton
import triton.language as tl

from .flex_attention_utils import (
    AUTOTUNE,
    DEBUG,
    AutotuneMode,
    apply_mask_mod,
    apply_score_grad_hook,
    apply_score_mod,
    aux_kernel_args,
    bs_tensor_strides,
    get_arch,
    mask_aux_kernel_args,
    max_block_for_lds,
    pick_knobs,
    remap_xcd,
)

PREPROCESS_AUTOTUNE_KEYS = [
    "max_seqlen_q",
    "ACTUAL_HEAD_DIM_V",
    "IS_VARLEN",
]

# See SPARSE_FWD_KNOBS in fwd_prefill.py. BLK_SLICE_FACTOR is inert here: the non-causal
# fused kernel (the only one block-sparse uses) declares it but never reads it -- the
# masked-diagonal sub-blocking it controls is causal-kernel-only.
#
# matrix_instr_nonkdim=16 is worth 1.1-1.6x on the block-sparse backward depending on
# shape (measured across seqlen 2048-8192, block 64/128, head_dim 64/128 on MI300X) and
# was simply never set here -- the tuned dense configs use it, but block-sparse bypasses
# the autotuner to pin its tiles and so inherited none of that tuning. It is dropped
# automatically for blocks below _MFMA_ACC_MIN_K at the launch site, mirroring
# _sanitize_nonkdim.
#
# waves_per_eu stays 1 deliberately. 2 is faster at block 64 / long sequences but much
# slower elsewhere (1.8x at seqlen 8192 block 128, 1.5x at head_dim 128), so it is a
# genuinely per-shape choice, not a better default.
#
# Unlike the forward (see sparse_fwd_default), there is no free rule to key this on:
# on gfx950 a block-64 backward wants waves_per_eu=2 at head_dim 64, by 1.18-1.32x, but
# is 1.41x slower with it at head_dim 128; on gfx942 the same setting only wins at the
# longest sequence. Capturing that needs per-shape measurement -- which is what
# pick_knobs does.
SPARSE_BWD_KNOBS = dict(
    BLK_SLICE_FACTOR=1,
    waves_per_eu=1,
    num_stages=1,
    num_warps=4,
    matrix_instr_nonkdim=16,
)


def sparse_bwd_default(kv_block: int) -> dict:
    """The knobs used when autotuning is off. Deliberately not keyed on block size.

    Unlike the forward, the backward's best waves_per_eu does not split cleanly: on
    gfx950 a block-64 backward prefers 2 by 1.18-1.32x at head_dim 64 but is 1.41x
    *slower* with it at head_dim 128, and on gfx942 the same setting is best only at the
    longest sequence. There is no free rule that is right everywhere, so the fallback
    stays at the shape-independent value and per-shape selection is left to pick_knobs.
    """
    return SPARSE_BWD_KNOBS


def sparse_bwd_candidates(kv_block: int) -> list:
    """Knob sets to measure when autotuning is on. The default goes first so it wins ties."""
    default = sparse_bwd_default(kv_block)
    others = [
        dict(BLK_SLICE_FACTOR=1, waves_per_eu=w, num_stages=st, num_warps=4, matrix_instr_nonkdim=16)
        for w, st in ((1, 1), (2, 1), (2, 2), (1, 2))
    ]
    return [default] + [c for c in others if c != default]


# Keyed on the same shape properties as NONCAUSAL_AUTOTUNE_KEYS, plus the block size.
_SPARSE_BWD_CHOICE: dict = {}

CAUSAL_AUTOTUNE_KEYS = [
    "max_seqlen_q",
    "max_seqlen_k",
    "ACTUAL_HEAD_DIM_QK",
    "ACTUAL_HEAD_DIM_V",
    "IS_VARLEN",
    "HQ",
    "HK",
]

NONCAUSAL_AUTOTUNE_KEYS = [
    "max_seqlen_q",
    "max_seqlen_k",
    "ACTUAL_HEAD_DIM_QK",
    "ACTUAL_HEAD_DIM_V",
    "IS_VARLEN",
    "HQ",
    "HK",
]


# Autotune configs are written as compact positional rows rather than multi-line dict
# literals: the tables carry a few hundred tuned entries, and one line each keeps them
# readable as a table. Trailing arguments are optional, and an omitted one leaves its
# key out of the config entirely (it is not the same as passing a default).
def _bwd_cfg(m1, n1, m2, n2, slice_factor, waves=None, nonkdim=None, *, stages, warps):
    """dK/dV + dQ config. Columns: BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2,
    BLK_SLICE_FACTOR, [waves_per_eu], [matrix_instr_nonkdim]."""
    kw = {
        "BLOCK_M1": m1,
        "BLOCK_N1": n1,
        "BLOCK_M2": m2,
        "BLOCK_N2": n2,
        "BLK_SLICE_FACTOR": slice_factor,
    }
    if waves is not None:
        kw["waves_per_eu"] = waves
    if nonkdim is not None:
        kw["matrix_instr_nonkdim"] = nonkdim
    return triton.Config(kw, num_stages=stages, num_warps=warps)


def _pre_cfg(pre_block, waves=None, *, stages, warps):
    """Preprocess-kernel config. Columns: PRE_BLOCK, [waves_per_eu]."""
    kw = {"PRE_BLOCK": pre_block}
    if waves is not None:
        kw["waves_per_eu"] = waves
    return triton.Config(kw, num_stages=stages, num_warps=warps)


def _bwd_configs_full(arch):
    """Every candidate (preprocess, causal, noncausal) config list for ``arch``, most-tuned
    first. "off" mode is always exactly these lists truncated to their first entry -- see
    get_bwd_configs.
    """
    if arch.name == "gfx942" and arch.cu_count < 304:
        preprocess = [
            _pre_cfg(64, 1, stages=1, warps=8),
            _pre_cfg(64, 2, stages=2, warps=8),
            _pre_cfg(128, 2, stages=1, warps=4),
        ]
        noncausal = [
            _bwd_cfg(32, 128, 128, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(64, 128, 128, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(32, 128, 128, 32, 2, 2, 16, stages=1, warps=8),
            _bwd_cfg(32, 128, 128, 32, 2, 1, 16, stages=1, warps=8),
        ]
        causal = [
            _bwd_cfg(32, 128, 128, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(64, 64, 64, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(32, 64, 64, 64, 2, 1, 16, stages=1, warps=4),
        ]
    elif arch.name == "gfx942":  # cu_count >= 304
        preprocess = [
            _pre_cfg(64, 2, stages=2, warps=8),
            _pre_cfg(64, 1, stages=1, warps=4),
        ]
        noncausal = [
            _bwd_cfg(32, 128, 128, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(64, 64, 64, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(32, 64, 64, 64, 2, 2, 16, stages=1, warps=4),
            # Two-stage variants, absent upstream: every tuned entry above pins
            # stages=1. Found by letting the autotuner loose on the (unreachable)
            # sweep space, worth 1.17x at head_dim 64 / seqlen 4096 and 1.10x at
            # 8192, neutral elsewhere. Added as candidates rather than as a new
            # default -- the autotuner benchmarks them per shape, so a shape they
            # do not suit just keeps the config it already had.
            _bwd_cfg(32, 128, 128, 64, 2, 2, stages=2, warps=4),
            _bwd_cfg(32, 128, 128, 64, 2, 1, stages=2, warps=4),
            _bwd_cfg(64, 128, 128, 64, 2, 2, stages=2, warps=4),
            _bwd_cfg(64, 128, 128, 64, 2, 1, stages=2, warps=4),
        ]
        causal = [
            _bwd_cfg(32, 128, 128, 64, 2, 1, 16, stages=1, warps=4),
            _bwd_cfg(32, 64, 64, 64, 2, 1, 16, stages=1, warps=4),
        ]
    elif arch.name == "gfx950":
        preprocess = [
            _pre_cfg(64, 2, stages=2, warps=8),
            _pre_cfg(64, 2, stages=1, warps=8),
            _pre_cfg(64, 2, stages=2, warps=4),
        ]
        noncausal = [
            _bwd_cfg(64, 128, 128, 64, 2, 1, stages=1, warps=4),
            _bwd_cfg(64, 128, 128, 128, 2, 1, stages=1, warps=4),
            _bwd_cfg(64, 64, 64, 64, 2, 1, stages=1, warps=4),
            _bwd_cfg(16, 64, 64, 64, 2, 2, stages=1, warps=4),
            _bwd_cfg(32, 256, 256, 64, 2, 1, stages=2, warps=8),
            _bwd_cfg(32, 256, 256, 64, 2, 2, stages=2, warps=8),
            # mid-tile, 2-stage variant
            _bwd_cfg(32, 128, 128, 64, 2, 2, stages=2, warps=4),
        ]
        causal = [
            _bwd_cfg(32, 128, 128, 64, 2, 1, stages=1, warps=4),
            _bwd_cfg(64, 64, 64, 64, 2, 1, stages=1, warps=4),
            _bwd_cfg(32, 128, 128, 64, 2, 1, stages=2, warps=4),
            _bwd_cfg(32, 128, 128, 64, 2, 2, stages=2, warps=4),
            # larger-tile variant (helps long-seq throughput; noncausal-proven)
            _bwd_cfg(32, 256, 256, 64, 2, 2, stages=2, warps=8),
            # small-tile variant (helps short seqlen / wide-window cases)
            _bwd_cfg(16, 64, 64, 64, 2, 2, stages=1, warps=4),
        ]
    elif arch.is_rdna:
        preprocess = [_pre_cfg(32, stages=1, warps=4)]
        noncausal = [_bwd_cfg(32, 32, 32, 32, 2, stages=1, warps=4)]
        causal = [_bwd_cfg(32, 32, 32, 32, 2, stages=1, warps=4)]
    else:
        preprocess = [_pre_cfg(64, 2, stages=2, warps=8)]
        noncausal = [_bwd_cfg(32, 128, 128, 64, 2, 1, stages=1, warps=4)]
        causal = [_bwd_cfg(32, 128, 128, 64, 2, 1, stages=1, warps=4)]

    for cfg in (*noncausal, *causal):
        kw = cfg.all_kwargs()
        assert kw["BLOCK_N1"] == kw["BLOCK_M2"], (
            f"BLOCK_N1 ({kw['BLOCK_N1']}) must equal BLOCK_M2 ({kw['BLOCK_M2']})"
        )
    return preprocess, causal, noncausal


def get_bwd_configs(mode: AutotuneMode):
    if mode in ("off", "on"):
        preprocess, causal, noncausal = _bwd_configs_full(get_arch())
        if mode == "off":
            return preprocess[:1], causal[:1], noncausal[:1]
        return preprocess, causal, noncausal

    else:  # sweep
        PRE_BLOCK_OPTIONS = [64, 128]
        PRE_WAVES_PER_EU_OPTIONS = [1, 2]
        PRE_NUM_STAGES_OPTIONS = [1, 2]
        PRE_NUM_WARPS_OPTIONS = [4, 8]
        NUM_STAGES_OPTIONS = [1, 2]
        NUM_WARPS_OPTIONS = [4, 8]
        WAVES_PER_EU_OPTIONS = [1, 2]
        NON_CAUSAL_BLOCK_M1_OPTIONS = [16, 32, 64, 128]
        NON_CAUSAL_BLOCK_N1_M2_OPTIONS = [32, 64, 128, 256]
        NON_CAUSAL_BLOCK_N2_OPTIONS = [16, 32, 64, 128]
        CAUSAL_BLOCK_M1_OPTIONS = [32, 64]
        CAUSAL_BLOCK_N1_M2_OPTIONS = [32, 64, 128]
        CAUSAL_BLOCK_N2_OPTIONS = [32, 64]
        BLK_SLICE_FACTOR_OPTIONS = [2]

        preprocess_configs = []
        for pre_num_warps in PRE_NUM_WARPS_OPTIONS:
            for pre_num_stages in PRE_NUM_STAGES_OPTIONS:
                for pre_waves in PRE_WAVES_PER_EU_OPTIONS:
                    for pre_block in PRE_BLOCK_OPTIONS:
                        preprocess_configs.append(
                            triton.Config(
                                {
                                    "PRE_BLOCK": pre_block,
                                    "waves_per_eu": pre_waves,
                                },
                                num_stages=pre_num_stages,
                                num_warps=pre_num_warps,
                            )
                        )

        causal_configs = []
        for num_warps in NUM_WARPS_OPTIONS:
            for num_stages in NUM_STAGES_OPTIONS:
                for waves in WAVES_PER_EU_OPTIONS:
                    for m1 in CAUSAL_BLOCK_M1_OPTIONS:
                        for n1 in CAUSAL_BLOCK_N1_M2_OPTIONS:
                            m2 = n1
                            for n2 in CAUSAL_BLOCK_N2_OPTIONS:
                                assert n1 == m2, f"BLOCK_N1 ({n1}) must equal BLOCK_M2 ({m2})"
                                if m2 % n2 != 0:
                                    continue
                                if n1 % m1 != 0:
                                    continue
                                for blk_slice in BLK_SLICE_FACTOR_OPTIONS:
                                    causal_configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M1": m1,
                                                "BLOCK_N1": n1,
                                                "BLOCK_M2": m2,
                                                "BLOCK_N2": n2,
                                                "BLK_SLICE_FACTOR": blk_slice,
                                                "waves_per_eu": waves,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                        )
                                    )

        noncausal_configs = []
        for num_warps in NUM_WARPS_OPTIONS:
            for num_stages in NUM_STAGES_OPTIONS:
                for waves in WAVES_PER_EU_OPTIONS:
                    for m1 in NON_CAUSAL_BLOCK_M1_OPTIONS:
                        for n1 in NON_CAUSAL_BLOCK_N1_M2_OPTIONS:
                            m2 = n1
                            for n2 in NON_CAUSAL_BLOCK_N2_OPTIONS:
                                assert n1 == m2, f"BLOCK_N1 ({n1}) must equal BLOCK_M2 ({m2})"
                                if m2 % n2 != 0:
                                    continue
                                if n1 % m1 != 0:
                                    continue
                                for blk_slice in BLK_SLICE_FACTOR_OPTIONS:
                                    noncausal_configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M1": m1,
                                                "BLOCK_N1": n1,
                                                "BLOCK_M2": m2,
                                                "BLOCK_N2": n2,
                                                "BLK_SLICE_FACTOR": blk_slice,
                                                "waves_per_eu": waves,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                        )
                                    )

        return (preprocess_configs, causal_configs, noncausal_configs)


# matrix_instr_nonkdim miscompile workaround -------------------------------------
# The tuned backward configs all set matrix_instr_nonkdim=16. On gfx942 that forces the
# 16x16x16 MFMA, and the AMD Triton backend then miscompiles the accumulating
# `tl.dot(..., acc=...)` in _bwd_dkdv_inner / _bwd_dq_inner when the dot's K dimension
# is <= 16: every loop iteration except the last is silently dropped from dK/dV.
#
# Only the CAUSAL backward hits this, which is why the bug looked causal-specific: the
# diagonal ("masked") blocks are swept with BLOCK_M1 // BLK_SLICE_FACTOR (32 // 2 = 16),
# while the non-causal path uses the full BLOCK_M1 = 32 and is unaffected. Measured on
# MI300X: with nonkdim=16 and a masked block of 16, only the final 16 key positions of
# each K block get correct dV; setting nonkdim to 32, or dropping it so Triton picks the
# instruction itself, makes every position correct. The same applies to the masked dQ
# sweep, which uses BLOCK_N2 // BLK_SLICE_FACTOR.
#
# So drop the hint on any config whose masked sub-block would be <= 16. Configs whose
# sub-blocks stay >= 32 keep it and their tuning.
_MFMA_ACC_MIN_K = 32


def _sanitize_nonkdim(configs):
    for cfg in configs:
        kw = cfg.kwargs
        if "matrix_instr_nonkdim" not in kw:
            continue
        slice_factor = kw.get("BLK_SLICE_FACTOR", 1)
        masked_dims = [kw[name] // slice_factor for name in ("BLOCK_M1", "BLOCK_N2") if name in kw]
        if masked_dims and min(masked_dims) < _MFMA_ACC_MIN_K:
            del kw["matrix_instr_nonkdim"]
    return configs


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
(
    preprocess_autotune_configs,
    causal_autotune_configs,
    noncausal_autotune_configs,
) = get_bwd_configs(AUTOTUNE)


# Causal only: the non-causal kernel sweeps with the full BLOCK_M1 / BLOCK_N2 (there are
# no diagonal blocks to slice), so its dots never hit the small-K miscompile and it keeps
# the tuning hint. Applying this to both cost ~5% on non-causal for no correctness gain.
def _extend_bwd_configs(configs):
    """Two extra backward tile shapes for the autotuner.

    The base table has 3 non-causal / 2 causal configs. A 216-point sweep on MI300X found these
    two beat every shipped config on some shapes, but only by 1.02-1.05x -- the backward
    is close to its config-tuning ceiling, and the remaining gap to peak is not reachable
    by retiling.

    (An earlier sweep appeared to show 1.33-1.39x. It was wrong: it used do=ones as the
    upstream gradient, which makes dp - delta nearly cancel so dQ is tiny, and its
    absolute error threshold then accepted configs that silently skipped most of the dQ
    work -- the very grid bug fixed in the `grid` closure below. Re-run with a random do
    and a relative check, the real headroom is a few percent. Kept as a warning: benchmark
    a backward with a random upstream gradient, never a constant one.)
    """
    extra = [
        # best non-causal and long-causal in the corrected sweep
        (64, 128, 128, 64, 1, 1),
        # best short-causal; BLOCK_M2 > BLOCK_N1 here, which only became legal once the
        # grid covered both phases
        (16, 64, 128, 32, 1, 2),
    ]
    out = list(configs)
    for m1, n1, m2, n2, sf, wpe in extra:
        out.append(
            triton.Config(
                {
                    "BLOCK_M1": m1,
                    "BLOCK_N1": n1,
                    "BLOCK_M2": m2,
                    "BLOCK_N2": n2,
                    "BLK_SLICE_FACTOR": sf,
                    "waves_per_eu": wpe,
                },
                num_stages=1,
                num_warps=4,
            )
        )
    return out


causal_autotune_configs = _extend_bwd_configs(causal_autotune_configs)
noncausal_autotune_configs = _extend_bwd_configs(noncausal_autotune_configs)
causal_autotune_configs = _sanitize_nonkdim(causal_autotune_configs)


@triton.autotune(
    configs=preprocess_autotune_configs,
    key=PREPROCESS_AUTOTUNE_KEYS,
)
@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    cu_seqlens_q,
    max_seqlen_q,
    PRE_BLOCK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    # Handle varlen
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    else:
        q_start = 0
        seqlen_q = max_seqlen_q

    # Compute offsets
    offs_m = pid_m * PRE_BLOCK + tl.arange(0, PRE_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM_V)
    # pointer offsets for O & DO
    off_o = (
        bid * stride_ob
        + hid * stride_oh
        + q_start * stride_om
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    off_do = (
        bid * stride_dob
        + hid * stride_doh
        + q_start * stride_dom
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )

    # create masks
    mask_m = offs_m < seqlen_q
    mask_md = mask_m[:, None]
    PADDED_HEAD_V: tl.constexpr = ACTUAL_HEAD_DIM_V != HEAD_DIM_V
    if PADDED_HEAD_V:
        mask_md &= offs_d[None, :] < ACTUAL_HEAD_DIM_V
    # load
    o = tl.load(Out + off_o, mask=mask_md, other=0.0)
    do = tl.load(DO + off_do, mask=mask_md, other=0.0)
    # compute and write-back to delta
    # NOTE: Both o and do are FP32
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    off_delta = (
        bid * stride_delta_b + hid * stride_delta_h + q_start * stride_delta_m + offs_m * stride_delta_m
    )
    tl.store(Delta + off_delta, delta, mask=mask_m)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _bwd_dkdv_inner(
    dk,
    dv,  # output
    Q,
    k,
    v,
    DO,
    M,
    D,
    sm_scale,  # input tensor
    stride_qm,
    stride_qk,
    stride_dom,
    stride_dok,
    stride_lse_m,
    stride_delta_m,
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    seqlen_q,
    seqlen_k,  # max sequence length for q and k
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  # iteration numbers
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    MASK: tl.constexpr,  # causal masking, only apply to tiles on mask diagonal
    USE_SLIDING_WINDOW: tl.constexpr,
    USE_EXP2: tl.constexpr,  # activate exp2
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    off_z=None,
    off_h_q=None,
    SCORE_MOD: tl.constexpr = None,
    MASK_MOD: tl.constexpr = None,
    SCORE_MOD_BWD: tl.constexpr = None,
    SPARSE_IDX=None,
    BLOCK_SPARSE: tl.constexpr = False,
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
    # if HEAD_DIM is padded
    PADDED_HEAD_QK: tl.constexpr = ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_HEAD_DIM_V != HEAD_DIM_V
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k_qk = tl.arange(0, HEAD_DIM_QK)
    offs_k_v = tl.arange(0, HEAD_DIM_V)
    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM_QK, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k_qk[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM_V), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k_v[None, :] * stride_dok
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    step_m = BLOCK_M
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)

    for blk_idx in tl.range(num_steps, num_stages=1):
        # BLOCK_SPARSE walks an explicit list of Q-block indices for this KV block
        # (the transpose of the forward list) instead of a contiguous run.
        if BLOCK_SPARSE:
            curr_m = tl.load(SPARSE_IDX + blk_idx).to(tl.int32) * BLOCK_M
        else:
            curr_m = start_m + blk_idx * step_m
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_m = {curr_m}")
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        if BLOCK_SPARSE:
            # Pointers are incremented linearly in the dense path; a jumping index needs
            # them recomputed from the current block instead.
            qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k_qk[:, None] * stride_qk
            do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k_v[None, :] * stride_dok
        # update the mask because offs_m advanced
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD_QK:
            mask_qT &= offs_k_qk[:, None] < ACTUAL_HEAD_DIM_QK
        if PADDED_HEAD_V:
            mask_do &= offs_k_v[None, :] < ACTUAL_HEAD_DIM_V
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m * stride_lse_m, mask=mask_m, other=0.0)
        qkT = tl.dot(k, qT)
        qkT_scaled = qkT * sm_scale

        # score_mod / mask_mod. NOTE: everything here is TRANSPOSED relative to the
        # forward kernel -- qkT_scaled is (BLOCK_N, BLOCK_M), so the query index varies
        # along columns and the key index along rows. The user's mod callables are
        # written against (q_idx, kv_idx) pairs and are elementwise, so passing
        # q_idx=offs_m[None, :] / kv_idx=offs_n[:, None] gives them the same logical
        # (q, kv) pairing the forward kernel passes, just in the transposed layout.
        qkT_premod = qkT_scaled
        if SCORE_MOD is not None:
            qkT_scaled = apply_score_mod(
                SCORE_MOD,
                qkT_scaled,
                off_z,
                off_h_q,
                offs_m[None, :],
                offs_n[:, None],
                AUX0,
                AUX1,
                AUX2,
                AUX3,
                NUM_AUX,
            )
        if MASK_MOD is not None:
            keep_mod = apply_mask_mod(
                MASK_MOD,
                off_z,
                off_h_q,
                offs_m[None, :],
                offs_n[:, None],
                MASK_AUX0,
                MASK_AUX1,
                MASK_AUX2,
                MASK_AUX3,
                NUM_MASK_AUX,
            )
            qkT_scaled = tl.where(keep_mod, qkT_scaled, float("-inf"))

        if DEBUG_TRITON_DETAIL and start_n == 256:
            print(f"qT: {qT.shape}\n", qT)
            print(f"k: {k.shape}\n", k)
            print(f"qkT scaled: {qkT.shape}\n", qkT_scaled)

        # Compute probabilities - handle invalid rows where m is -inf
        # For rows where m is -inf, no keys were valid, so pT should be 0
        # We shift qkT by m to avoid numerical issues
        qkT_shifted = tl.where(m[None, :] == float("-inf"), float("-inf"), qkT_scaled - m[None, :])

        if USE_EXP2:
            pT = tl.math.exp2(qkT_shifted * RCP_LN2)
        else:
            pT = tl.math.exp(qkT_shifted)

        # Causal and sliding-window masking.
        if MASK or USE_SLIDING_WINDOW:
            mask = mask_nm
            if MASK:
                # offset offs_m with delta_qk since the causal mask starts at
                # bottom right of the (seqlen_q, seqlen_k) matrix
                causal_mask = (offs_m[None, :] - delta_qk) >= offs_n[:, None]
                mask = causal_mask & mask
            if USE_SLIDING_WINDOW:
                # Per-element form of the band a query m attends:
                #   m + (seqlen_k - seqlen_q) - L <= n <= m + (seqlen_k - seqlen_q) + R.
                # _sliding_window_q_bounds() is the block-range inversion of this same
                # inequality (it bounds m for a fixed K block); keep the two in sync.
                causal_offset = seqlen_k - seqlen_q
                if WINDOW_SIZE_LEFT < 0 and WINDOW_SIZE_RIGHT < 0:
                    # both edges unbounded -> window keeps every key (the causal
                    # cap, if any, is applied separately via MASK above)
                    window_mask = offs_n[:, None] >= 0
                elif WINDOW_SIZE_LEFT < 0:
                    rel = offs_n[:, None] - offs_m[None, :] - causal_offset
                    window_mask = rel <= WINDOW_SIZE_RIGHT
                elif WINDOW_SIZE_RIGHT < 0:
                    # unbounded right, finite left (mirror of infinite-left)
                    rel = offs_n[:, None] - offs_m[None, :] - causal_offset
                    window_mask = rel >= -WINDOW_SIZE_LEFT
                else:
                    # Keep the relative-distance form:
                    # broadcasting explicit left/right bound tiles potentially makes the
                    # gfx950 backend spill the buffer descriptors and return silently wrong dK/dV.
                    rel = offs_n[:, None] - offs_m[None, :] - causal_offset
                    window_mask = (rel >= -WINDOW_SIZE_LEFT) & (rel <= WINDOW_SIZE_RIGHT)
                mask = window_mask & mask
            if DEBUG_TRITON_DETAIL and start_n == 256:
                print(f"mask: {mask.shape}\n", mask)
                print(
                    f"pT after mask: {pT.shape}\n",
                    tl.where(mask, pT, 0.0),
                )
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        # Compute dV.
        dv = tl.dot(pT.to(do.type.element_ty), do, acc=dv)

        if DEBUG_TRITON_DETAIL and start_n == 256:
            print(f"pT: {pT.shape}\n", pT)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m * stride_delta_m, mask=mask_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do))
        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        # score_mod VJP, then re-zero anything mask_mod dropped: a user's
        # score_mod_bwd formula is not required to map a zero input gradient to a zero
        # output, so rely on the mask rather than on dsT already being 0 there.
        if SCORE_MOD_BWD is not None:
            dsT = SCORE_MOD_BWD(dsT, qkT_premod, off_z, off_h_q, offs_m[None, :], offs_n[:, None])
        if MASK_MOD is not None:
            dsT = tl.where(keep_mod, dsT, 0.0)
        dk = tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT), acc=dk)
        # Increment pointers (dense path only; the sparse path recomputes them above).
        if not BLOCK_SPARSE:
            qT_ptrs += step_m * stride_qm
            do_ptrs += step_m * stride_dom
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _bwd_dq_inner(
    dq,  # output
    q,
    K,
    V,
    do,
    m,
    Delta,
    sm_scale,  # input
    # shared by Q/K/V.
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_lse_m,
    stride_delta_m,
    seqlen_q,
    seqlen_k,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    end_n,
    num_steps,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    MASK: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    USE_EXP2: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    off_z=None,
    off_h_q=None,
    SCORE_MOD: tl.constexpr = None,
    MASK_MOD: tl.constexpr = None,
    SCORE_MOD_BWD: tl.constexpr = None,
    SPARSE_IDX=None,
    BLOCK_SPARSE: tl.constexpr = False,
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
    SCORE_GRAD_HOOK: tl.constexpr = None,
    SCORE_GRAD=None,
):
    # if HEAD_DIM is padded
    PADDED_HEAD_QK: tl.constexpr = ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_HEAD_DIM_V != HEAD_DIM_V
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k_qk = tl.arange(0, HEAD_DIM_QK)
    offs_k_v = tl.arange(0, HEAD_DIM_V)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k_v[:, None] * stride_vk
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_m * stride_delta_m, mask=mask_m, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    step_n = BLOCK_N2
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    for blk_idx in tl.range(num_steps, num_stages=1):
        # BLOCK_SPARSE walks an explicit list of KV-block indices for this Q block.
        if BLOCK_SPARSE:
            curr_n = tl.load(SPARSE_IDX + blk_idx).to(tl.int32) * BLOCK_N2
        else:
            curr_n = start_n + blk_idx * step_n
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_n = {curr_n}")
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        if BLOCK_SPARSE:
            kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
            vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k_v[:, None] * stride_vk
            # A sparse block list has no contiguous end; bound by seqlen instead.
            mask_n = offs_n < seqlen_k
        else:
            # end_n is needed because the end of causal True might not be perfectly
            # aligned with the end of the block
            mask_n = offs_n < end_n
        if DEBUG_TRITON_DETAIL:
            print(f"start_n = {start_n}, end_n = {end_n}, offs_n: {offs_n.shape}\n{offs_n}")
        if DEBUG_TRITON_DETAIL:
            print(f"mask_n: {mask_n.shape}\n{mask_n}")
        mask_kT = mask_n[None, :]
        mask_vT = mask_n[None, :]
        if BLOCK_SPARSE:
            mask_mn = mask_m[:, None] & (offs_n[None, :] < seqlen_k)
        else:
            mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD_QK:
            mask_kT &= offs_k_qk[:, None] < ACTUAL_HEAD_DIM_QK
        if PADDED_HEAD_V:
            mask_vT &= offs_k_v[:, None] < ACTUAL_HEAD_DIM_V

        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_vT, other=0.0)

        qk = tl.dot(q, kT)
        qk_scaled = qk * sm_scale

        # score_mod / mask_mod (non-transposed here, unlike the dK/dV inner loop).
        qk_premod = qk_scaled
        if SCORE_MOD is not None:
            qk_scaled = apply_score_mod(
                SCORE_MOD,
                qk_scaled,
                off_z,
                off_h_q,
                offs_m[:, None],
                offs_n[None, :],
                AUX0,
                AUX1,
                AUX2,
                AUX3,
                NUM_AUX,
            )
        if MASK_MOD is not None:
            keep_mod = apply_mask_mod(
                MASK_MOD,
                off_z,
                off_h_q,
                offs_m[:, None],
                offs_n[None, :],
                MASK_AUX0,
                MASK_AUX1,
                MASK_AUX2,
                MASK_AUX3,
                NUM_MASK_AUX,
            )
            qk_scaled = tl.where(keep_mod, qk_scaled, float("-inf"))

        if DEBUG_TRITON_DETAIL:
            print(f"qk scaled: {qk.shape}\n", qk_scaled)

        # Compute probabilities - handle invalid rows where m is -inf
        # For rows where m is -inf, no keys were valid, so p should be 0
        # We shift qk by m to avoid numerical issues
        qk_shifted = tl.where(m == float("-inf"), float("-inf"), qk_scaled - m)

        if USE_EXP2:
            p = tl.math.exp2(qk_shifted * RCP_LN2)
        else:
            p = tl.math.exp(qk_shifted)

        # Causal and sliding-window masking.
        if MASK or USE_SLIDING_WINDOW:
            mask = mask_mn
            if MASK:
                causal_mask = (offs_m[:, None] - delta_qk) >= offs_n[None, :]
                mask = causal_mask & mask
            if USE_SLIDING_WINDOW:
                # Per-element form of the band a query m attends:
                #   m + (seqlen_k - seqlen_q) - L <= n <= m + (seqlen_k - seqlen_q) + R.
                # _sliding_window_k_bounds() is the block-range inversion of this same
                # inequality (it bounds n for a fixed Q block); keep the two in sync.
                causal_offset = seqlen_k - seqlen_q
                if WINDOW_SIZE_LEFT < 0 and WINDOW_SIZE_RIGHT < 0:
                    # both edges unbounded -> window keeps every key (the causal
                    # cap, if any, is applied separately via MASK above)
                    window_mask = offs_n[None, :] >= 0
                elif WINDOW_SIZE_LEFT < 0:
                    window_mask = offs_n[None, :] <= (offs_m[:, None] + causal_offset + WINDOW_SIZE_RIGHT)
                elif WINDOW_SIZE_RIGHT < 0:
                    # unbounded right, finite left (mirror of infinite-left)
                    window_mask = offs_n[None, :] >= (offs_m[:, None] + causal_offset - WINDOW_SIZE_LEFT)
                else:
                    # Relative-distance form, as in _bwd_dkdv_inner: explicit broadcast
                    # left/right bound tiles turned dQ to NaN on MI300X once the kernel
                    # also carried a score_grad_hook.
                    rel = offs_n[None, :] - offs_m[:, None] - causal_offset
                    window_mask = (rel >= -WINDOW_SIZE_LEFT) & (rel <= WINDOW_SIZE_RIGHT)
                mask = window_mask & mask
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT)
        delta_i = Di[:, None]
        ds = p * (dp - delta_i)
        # mask_mn is exactly the pairs this tile owns: the dQ sweep's phases can share a
        # tile (an unaligned unmasked range starts inside the masked strip), so the hook
        # must store only there, or it would overwrite the other phase's values.
        if SCORE_GRAD_HOOK is not None:
            apply_score_grad_hook(
                SCORE_GRAD_HOOK,
                tl.where(mask_mn, ds, 0.0),
                mask_mn,
                off_z,
                off_h_q,
                offs_m[:, None],
                offs_n[None, :],
                SCORE_GRAD,
                AUX0,
                AUX1,
                AUX2,
                AUX3,
                NUM_AUX,
            )
        # score_mod VJP, then re-zero anything mask_mod dropped (see the matching
        # comment in _bwd_dkdv_inner for why the mask is re-applied explicitly).
        if SCORE_MOD_BWD is not None:
            ds = SCORE_MOD_BWD(ds, qk_premod, off_z, off_h_q, offs_m[:, None], offs_n[None, :])
        if MASK_MOD is not None:
            ds = tl.where(keep_mod, ds, 0.0)
        # Compute dQ.
        dq = tl.dot(ds.to(kT.type.element_ty), tl.trans(kT), acc=dq)
        # Increment pointers (dense path only; sparse recomputes them at loop top).
        if not BLOCK_SPARSE:
            kT_ptrs += step_n * stride_kn
            vT_ptrs += step_n * stride_vn
    return dq


@triton.jit
def _sliding_window_q_bounds(
    start_n,
    seqlen_q,
    seqlen_k,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Query-block range that overlaps the sliding window for a fixed K block.

    dKdV sweeps query blocks for the fixed key block [start_n, start_n+BLOCK_N).
    A query m attends key n iff
        m + (seqlen_k - seqlen_q) - L <= n <= m + (seqlen_k - seqlen_q) + R,
    so the queries that touch any key in this block span
        [start_n - R - off, (start_n + BLOCK_N - 1) + L - off]   (off = seqlen_k - seqlen_q).
    An unbounded edge drops its bound: L < 0 -> m_hi = seqlen_q - 1 (queries reach
    forward without limit); R < 0 -> m_lo = 0 (queries reach back without limit).
    Returns (start_m, num_steps) aligned to BLOCK_M; num_steps == 0 means the
    block is entirely outside the window and can be skipped.
    """
    causal_offset = seqlen_k - seqlen_q
    if WINDOW_SIZE_RIGHT < 0:  # unbounded right -> no lower limit on contributing queries
        m_lo = 0
    else:
        m_lo = start_n - WINDOW_SIZE_RIGHT - causal_offset
    if WINDOW_SIZE_LEFT < 0:  # unbounded left -> no upper limit on contributing queries
        m_hi = seqlen_q - 1
    else:
        m_hi = (start_n + BLOCK_N - 1) + WINDOW_SIZE_LEFT - causal_offset
    m_lo = tl.maximum(m_lo, 0)
    m_hi = tl.minimum(m_hi, seqlen_q - 1)
    start_m = (m_lo // BLOCK_M) * BLOCK_M
    if m_hi < m_lo:
        num_steps = 0
    else:
        num_steps = tl.cdiv(m_hi + 1 - start_m, BLOCK_M)
    return start_m, num_steps


@triton.jit
def _sliding_window_k_bounds(
    start_m,
    seqlen_q,
    seqlen_k,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Key-block range that overlaps the sliding window for a fixed Q block.

    dQ sweeps key blocks for the fixed query block [start_m, start_m+BLOCK_M).
    The keys query m attends are [m + off - L, m + off + R] (off = seqlen_k - seqlen_q),
    so across the block they span [start_m + off - L, (start_m + BLOCK_M - 1) + off + R].
    An unbounded edge drops its bound: L < 0 -> n_lo = 0 (keys reach back to 0);
    R < 0 -> n_hi = seqlen_k - 1 (keys reach forward to the last key).
    Returns (start_n, num_steps) aligned to BLOCK_N; num_steps == 0 means skip.
    """
    causal_offset = seqlen_k - seqlen_q
    if WINDOW_SIZE_LEFT < 0:  # unbounded left -> keys reach back to 0
        n_lo = 0
    else:
        n_lo = start_m + causal_offset - WINDOW_SIZE_LEFT
    if WINDOW_SIZE_RIGHT < 0:  # unbounded right -> keys reach forward to seqlen_k - 1
        n_hi = seqlen_k - 1
    else:
        n_hi = (start_m + BLOCK_M - 1) + causal_offset + WINDOW_SIZE_RIGHT
    n_lo = tl.maximum(n_lo, 0)
    n_hi = tl.minimum(n_hi, seqlen_k - 1)
    start_n = (n_lo // BLOCK_N) * BLOCK_N
    if n_hi < n_lo:
        num_steps = 0
    else:
        num_steps = tl.cdiv(n_hi + 1 - start_n, BLOCK_N)
    return start_n, num_steps


@triton.autotune(
    configs=causal_autotune_configs,
    key=CAUSAL_AUTOTUNE_KEYS,
)
@triton.jit
def bwd_kernel_fused_causal(  # grid = (nheads_k, tl.cdiv(max_seqlen_q // BLOCK_M2), batch)
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    Delta,
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
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_lse_b,
    stride_lse_h,
    stride_lse_m,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    NUM_XCD: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    MASK_MOD: tl.constexpr = None,
    SCORE_MOD_BWD: tl.constexpr = None,
    BS_DKDV_CNT=None,
    BS_DKDV_IDX=None,
    BS_DQ_CNT=None,
    BS_DQ_IDX=None,
    stride_bs_dkdv_cnt_b=0,
    stride_bs_dkdv_cnt_h=0,
    stride_bs_dkdv_cnt_m=0,
    stride_bs_dkdv_idx_b=0,
    stride_bs_dkdv_idx_h=0,
    stride_bs_dkdv_idx_m=0,
    stride_bs_dq_cnt_b=0,
    stride_bs_dq_cnt_h=0,
    stride_bs_dq_cnt_m=0,
    stride_bs_dq_idx_b=0,
    stride_bs_dq_idx_h=0,
    stride_bs_dq_idx_m=0,
    BLOCK_SPARSE: tl.constexpr = False,
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
    SCORE_GRAD_HOOK: tl.constexpr = None,
    SCORE_GRAD=None,
):
    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)

    # apply the xcd remapping for the hq dim
    hkid = remap_xcd(hkid, HK, NUM_XCD)

    if DEBUG_TRITON:
        print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)

        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON:
        print(f"delta_qk = {delta_qk}")
    PADDED_HEAD_QK: tl.constexpr = ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_HEAD_DIM_V != HEAD_DIM_V
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    GROUP_SIZE: tl.constexpr = HQ // HK
    # align the delta_qk
    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        # This section does dk and dv
        dk = tl.zeros([BLOCK_N1, HEAD_DIM_QK], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM_V], dtype=tl.float32)

        # q > k: diretcly skip all the way until the start of causal block
        start_delta_q_gt_k = delta_qk
        # q < k: some blocks will have no Masked block, other needs to re-calc
        # starting position
        # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
        # masked op
        num_blocks_skip = -delta_qk // BLOCK_N1
        delta_aligned = (num_blocks_skip + 1) * BLOCK_N1 + delta_qk
        start_delta_q_lt_k = delta_aligned // BLOCK_M1 * BLOCK_M1
        if delta_qk >= 0:
            start_delta = delta_qk
            if DEBUG_TRITON:
                print(f"q >= k: start_delta = delta_qk aligned to BLOCK_M = {start_delta_q_gt_k}")
        else:
            start_delta = start_delta_q_lt_k
            if DEBUG_TRITON:
                print(
                    f"q < k: start_delta = residue btw multiple BLOCK_N and delta_qk = {delta_aligned} = aligned to BLOCK_M = {start_delta_q_lt_k}"
                )

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_k = offs_n[:, None] < seqlen_k
        mask_v = offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_k &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_v &= mask_d_v[None, :]

        # K/V tensors not changed for the group
        adj_k = (
            bid * stride_kb
            + hkid * stride_kh
            + k_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_d_qk[None, :] * stride_kd
        )
        adj_v = (
            bid * stride_vb
            + hkid * stride_vh
            + k_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_d_v[None, :] * stride_vd
        )
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_k)
        v = tl.load(V + adj_v, mask=mask_v)
        # If MQA / GQA, set the K and V head offsets appropriately.
        # hqid = hkid
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            if delta_qk >= 0:
                start_m = start_n + start_delta
                len_m = BLOCK_N1
            else:
                start_m = max(start_n + delta_qk, 0)
                start_m = start_m // BLOCK_M1 * BLOCK_M1
                # because we might shift the masked blocks up, we are deeper into
                # the masked out region, so we would potentially increase the total
                # steps with masked operation to get out of it
                residue_m = max(start_n + delta_qk - start_m, 0)
                len_m = BLOCK_N1 + residue_m
                if DEBUG_TRITON:
                    print(f"residue_m = {residue_m}")

            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m
            M_ptr = M + adj_m

            MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
            # bound the masked operation to q len so it does not have to wast cycles
            len_m = min(len_m, seqlen_q)
            num_steps = tl.cdiv(len_m, MASK_BLOCK_M1)
            # when q < k, we may skip the initial masked op
            if pid < num_blocks_skip:
                num_steps = 0

            # if start_m is negative, the current N-tile has no block on the
            #   diagonal of causal mask, so everything have no causal mask
            if DEBUG_TRITON:
                print(f"Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}")
            dk, dv = _bwd_dkdv_inner(
                dk,
                dv,  # output tensors
                Q_ptr,
                k,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_lse_m,
                stride_delta_m,
                MASK_BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM_QK,
                HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,  # head dim
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT,
                MASK=True,  # causal masking
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                off_z=bid,
                off_h_q=hqid,
                SCORE_MOD=SCORE_MOD,
                MASK_MOD=MASK_MOD,
                SCORE_MOD_BWD=SCORE_MOD_BWD,
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
            )
            start_m += num_steps * MASK_BLOCK_M1
            # The unmasked region runs from the diagonal to seqlen_q. With a
            # finite left window, queries more than WINDOW_SIZE_LEFT past this
            # K block attend nothing in it, so cap the sweep at that upper bound.
            if USE_SLIDING_WINDOW and WINDOW_SIZE_LEFT >= 0:
                # m_hi = (n_hi + WINDOW_SIZE_LEFT) - (seqlen_k - seqlen_q)
                m_hi = (start_n + BLOCK_N1 - 1) + WINDOW_SIZE_LEFT + delta_qk
                m_hi = tl.minimum(m_hi, seqlen_q - 1)
                if m_hi < start_m:
                    num_steps = 0
                else:
                    num_steps = tl.cdiv(m_hi + 1 - start_m, BLOCK_M1)
            else:
                num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M1)
            end_m = start_m + num_steps * BLOCK_M1

            if DEBUG_TRITON:
                print(f"start_m after Masked step: {start_m}; num_steps: {num_steps}")
            if DEBUG_TRITON:
                print(
                    f"unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}"
                )
            if DEBUG_TRITON:
                print("unMasked")
            dk, dv = _bwd_dkdv_inner(
                dk,
                dv,  # output tensors
                Q_ptr,
                k,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_lse_m,
                stride_delta_m,
                BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM_QK,
                HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,  # head dim
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT,
                MASK=False,  # causal masking
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                off_z=bid,
                off_h_q=hqid,
                SCORE_MOD=SCORE_MOD,
                MASK_MOD=MASK_MOD,
                SCORE_MOD_BWD=SCORE_MOD_BWD,
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
            )
        # end of GQA/MQA of dkdv
        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d_v[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_v)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d_qk[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_k)

    # This part does dq
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        # seqlen_q > seqlen_k, no need to process these tile for dq
        if DEBUG_TRITON:
            print(f"end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2}")
        if start_m + BLOCK_M2 < delta_qk:
            if DEBUG_TRITON:
                print(
                    f"start_m + BLOCK_M2 = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2} < delta_qk of {delta_qk}"
                )
            return

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        mask_do = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_q &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_do &= mask_d_v[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dod
        # NOTE: don't assume that the strides for k and v are the same!
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn

        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # seqlen_q < seqlen_k: delta_qk more kv tokens are added at the front
            #   for every M-tile
            end_n = start_m + BLOCK_M2 - delta_qk
            # clamp end_n at [0, seqlen_k]
            end_n = max(min(end_n, seqlen_k), 0)
            if DEBUG_TRITON:
                print(f"delta_qk: {delta_qk}; end_n: {end_n}")
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m
            M_ptr = M + adj_m

            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            do = tl.load(DO + adj_do + offs_do, mask=mask_do, other=0.0)
            m = tl.load(M + adj_m + offs_m * stride_lse_m, mask=offs_m < seqlen_q)
            m = m[:, None]

            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
            # start can only be 0 at minimum
            start_n = max(end_n - BLOCK_M2, 0)
            num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N2)

            dq = tl.zeros([BLOCK_M2, HEAD_DIM_QK], dtype=tl.float32)
            dq = _bwd_dq_inner(
                dq,
                q,
                K,
                V,
                do,
                m,
                Delta_ptr,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_lse_m,
                stride_delta_m,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                MASK_BLOCK_N2,
                HEAD_DIM_QK,
                HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,
                start_m,
                start_n,
                end_n,
                num_steps,
                WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT,
                MASK=True,
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                off_z=bid,
                off_h_q=hqid,
                SCORE_MOD=SCORE_MOD,
                MASK_MOD=MASK_MOD,
                SCORE_MOD_BWD=SCORE_MOD_BWD,
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
                SCORE_GRAD_HOOK=SCORE_GRAD_HOOK,
                SCORE_GRAD=SCORE_GRAD,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            # The unmasked region runs from 0 up to the diagonal (end_n). With a
            # finite left window, keys more than WINDOW_SIZE_LEFT before this Q
            # block fall outside the band, so raise the lower bound.
            if USE_SLIDING_WINDOW and WINDOW_SIZE_LEFT >= 0:
                # n_lo = start_m - (seqlen_q - seqlen_k) - WINDOW_SIZE_LEFT
                n_lo = start_m - delta_qk - WINDOW_SIZE_LEFT
                n_lo = tl.maximum(n_lo, 0)
                n_lo = (n_lo // BLOCK_N2) * BLOCK_N2
                if n_lo >= end_n:
                    num_steps = 0
                else:
                    num_steps = tl.cdiv(end_n - n_lo, BLOCK_N2)
                start_n = n_lo
            else:
                num_steps = tl.cdiv(end_n, BLOCK_N2)
                start_n = max(end_n - num_steps * BLOCK_N2, 0)
            if DEBUG_TRITON:
                print(
                    f"unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}"
                )
            dq = _bwd_dq_inner(
                dq,
                q,
                K,
                V,
                do,
                m,
                Delta_ptr,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_lse_m,
                stride_delta_m,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM_QK,
                HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,
                start_m,
                start_n,
                end_n,
                num_steps,
                WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT,
                MASK=False,
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                off_z=bid,
                off_h_q=hqid,
                SCORE_MOD=SCORE_MOD,
                MASK_MOD=MASK_MOD,
                SCORE_MOD_BWD=SCORE_MOD_BWD,
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
                SCORE_GRAD_HOOK=SCORE_GRAD_HOOK,
                SCORE_GRAD=SCORE_GRAD,
            )
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_d_qk[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            # end of GQA/MQA of dq


@triton.autotune(
    configs=noncausal_autotune_configs,
    key=NONCAUSAL_AUTOTUNE_KEYS,
)
@triton.jit
def bwd_kernel_fused_noncausal(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    Delta,
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
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_lse_b,
    stride_lse_h,
    stride_lse_m,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M1: tl.constexpr,  # 32
    BLOCK_N1: tl.constexpr,  # 128
    BLOCK_M2: tl.constexpr,  # 128
    BLOCK_N2: tl.constexpr,  # 32
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    NUM_XCD: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    MASK_MOD: tl.constexpr = None,
    SCORE_MOD_BWD: tl.constexpr = None,
    BS_DKDV_CNT=None,
    BS_DKDV_IDX=None,
    BS_DQ_CNT=None,
    BS_DQ_IDX=None,
    stride_bs_dkdv_cnt_b=0,
    stride_bs_dkdv_cnt_h=0,
    stride_bs_dkdv_cnt_m=0,
    stride_bs_dkdv_idx_b=0,
    stride_bs_dkdv_idx_h=0,
    stride_bs_dkdv_idx_m=0,
    stride_bs_dq_cnt_b=0,
    stride_bs_dq_cnt_h=0,
    stride_bs_dq_cnt_m=0,
    stride_bs_dq_idx_b=0,
    stride_bs_dq_idx_h=0,
    stride_bs_dq_idx_m=0,
    BLOCK_SPARSE: tl.constexpr = False,
    # Block-sparse only. BlockPlan.with_causal has already dropped every KV block past
    # the diagonal and forced the straddling ones onto the masked pass, so all that is
    # left for the kernel is the per-element causal mask on those blocks.
    IS_CAUSAL: tl.constexpr = False,
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
    SCORE_GRAD_HOOK: tl.constexpr = None,
    SCORE_GRAD=None,
):
    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)

    # apply the xcd remapping for the hq dim
    hkid = remap_xcd(hkid, HK, NUM_XCD)

    if DEBUG_TRITON:
        print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)

        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_HEAD_DIM_V != HEAD_DIM_V
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    GROUP_SIZE: tl.constexpr = HQ // HK

    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        dk = tl.zeros([BLOCK_N1, HEAD_DIM_QK], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM_V], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_k = offs_n[:, None] < seqlen_k
        mask_v = offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_k &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_v &= mask_d_v[None, :]
        # NOTE: don't assume that the strides for k and v are the same!
        # K/V tensors not changed for the group
        adj_k = (
            bid * stride_kb
            + hkid * stride_kh
            + k_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_d_qk[None, :] * stride_kd
        )
        adj_v = (
            bid * stride_vb
            + hkid * stride_vh
            + k_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_d_v[None, :] * stride_vd
        )
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_k)
        v = tl.load(V + adj_v, mask=mask_v)
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m
            M_ptr = M + adj_m

            # because there is no causal, we sweep all query blocks -- unless a
            # sliding window lets us skip query blocks entirely outside the band.
            if USE_SLIDING_WINDOW:
                start_m, num_steps = _sliding_window_q_bounds(
                    start_n,
                    seqlen_q,
                    seqlen_k,
                    WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT,
                    BLOCK_N1,
                    BLOCK_M1,
                )
            else:
                start_m = 0
                num_steps = tl.cdiv(seqlen_q, BLOCK_M1)
            # Block-sparse: this KV block only sees the Q blocks listed for it (the
            # transpose of the forward list). pid indexes the KV block.
            bs_dkdv_idx_ptr = BS_DKDV_IDX
            if BLOCK_SPARSE:
                _c = bid * stride_bs_dkdv_cnt_b + hqid * stride_bs_dkdv_cnt_h + pid * stride_bs_dkdv_cnt_m
                _i = bid * stride_bs_dkdv_idx_b + hqid * stride_bs_dkdv_idx_h + pid * stride_bs_dkdv_idx_m
                num_steps = tl.load(BS_DKDV_CNT + _c).to(tl.int32)
                bs_dkdv_idx_ptr = BS_DKDV_IDX + _i
                start_m = 0
            dk, dv = _bwd_dkdv_inner(
                dk,
                dv,  # output tensors
                Q_ptr,
                k,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_lse_m,
                stride_delta_m,
                BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM_QK,
                HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,  # head dim
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT,
                MASK=IS_CAUSAL,  # causal masking
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                off_z=bid,
                off_h_q=hqid,
                SCORE_MOD=SCORE_MOD,
                MASK_MOD=MASK_MOD,
                SCORE_MOD_BWD=SCORE_MOD_BWD,
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
                SPARSE_IDX=bs_dkdv_idx_ptr,
                BLOCK_SPARSE=BLOCK_SPARSE,
            )

        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d_v[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_v)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d_qk[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_k)

    # THIS PART DOES DQ
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        mask_do = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_q &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_do &= mask_d_v[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dod
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m
            M_ptr = M + adj_m

            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            do = tl.load(DO + adj_do + offs_do, mask=mask_do, other=0.0)
            m = tl.load(M + adj_m + offs_m * stride_lse_m, mask=offs_m < seqlen_q)
            m = m[:, None]

            # start can only be 0 at minimum
            end_n = seqlen_k
            if USE_SLIDING_WINDOW:
                start_n, num_steps = _sliding_window_k_bounds(
                    start_m,
                    seqlen_q,
                    seqlen_k,
                    WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT,
                    BLOCK_M2,
                    BLOCK_N2,
                )
            else:
                start_n = 0
                num_steps = tl.cdiv(seqlen_k, BLOCK_N2)

            # Block-sparse: this Q block only sees the KV blocks listed for it.
            bs_dq_idx_ptr = BS_DQ_IDX
            if BLOCK_SPARSE:
                _c = bid * stride_bs_dq_cnt_b + hqid * stride_bs_dq_cnt_h + pid * stride_bs_dq_cnt_m
                _i = bid * stride_bs_dq_idx_b + hqid * stride_bs_dq_idx_h + pid * stride_bs_dq_idx_m
                num_steps = tl.load(BS_DQ_CNT + _c).to(tl.int32)
                bs_dq_idx_ptr = BS_DQ_IDX + _i
                start_n = 0

            dq = tl.zeros([BLOCK_M2, HEAD_DIM_QK], dtype=tl.float32)
            dq = _bwd_dq_inner(  # noncausal fused dQ (score_mod/mask_mod path)
                dq,
                q,
                K,
                V,
                do,
                m,
                Delta_ptr,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_lse_m,
                stride_delta_m,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM_QK,
                HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,
                start_m,
                start_n,
                end_n,
                num_steps,
                WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT,
                MASK=IS_CAUSAL,
                USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                off_z=bid,
                off_h_q=hqid,
                SCORE_MOD=SCORE_MOD,
                MASK_MOD=MASK_MOD,
                SCORE_MOD_BWD=SCORE_MOD_BWD,
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
                SCORE_GRAD_HOOK=SCORE_GRAD_HOOK,
                SCORE_GRAD=SCORE_GRAD,
                SPARSE_IDX=bs_dq_idx_ptr,
                BLOCK_SPARSE=BLOCK_SPARSE,
            )
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_d_qk[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


def is_contiguous(x, name):
    if x.is_contiguous():
        return x
    else:
        print(f"{name} is not contiguous")
        return x.contiguous()


# Triton kernel debug flags derived from DEBUG level.
# Level 1: basic kernel debug prints (iteration info)
# Level 2: detailed kernel debug prints (tensor values)
# Requires TRITON_INTERPRET=1 to actually print inside kernels.
DEBUG_TRITON: bool = DEBUG >= 1
DEBUG_TRITON_DETAIL: bool = DEBUG >= 2


def attention_backward_triton_impl(
    *,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    delta: torch.Tensor,
    sm_scale: float,
    causal: bool,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    use_exp2: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
    # score_mod / mask_mod: only supported on the non-causal "fused" path, see the
    # guard below.
    score_mod=None,
    mask_mod=None,
    score_mod_bwd=None,
    block_sparse_dkdv=None,
    block_sparse_dq=None,
    aux_tensors=None,
    mask_aux_tensors=None,
    score_grad_hook=None,
    score_grad=None,
    dlse=None,
):
    # get params, strides and shape
    IS_VARLEN = layout == "thd"

    # common assertions
    assert q.device == k.device == v.device == o.device == do.device == softmax_lse.device, (
        f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}, do={do.device}, softmax_lse={softmax_lse.device}"
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
        total_seqlen_k, nheads_k, head_size_k = k.shape
        _total_seqlen_v, nheads_v, head_size_v = v.shape
        nheads_lse, total_seqlen_lse = softmax_lse.shape

        # assert shapes
        assert total_seqlen_lse == total_seqlen_q, (
            f"softmax_lse seqlen {total_seqlen_lse} != q seqlen {total_seqlen_q}"
        )
        assert cu_seqlens_q is not None, "cu_seqlens_q must be provided for varlen layout"
        assert cu_seqlens_k is not None, "cu_seqlens_k must be provided for varlen layout"
        assert max_seqlen_q is not None, "max_seqlen_q must be provided for varlen layout"
        assert max_seqlen_k is not None, "max_seqlen_k must be provided for varlen layout"

        # assert head dimensions
        assert head_size_q == head_size_k, f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, (
            f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        )
        assert nheads_lse == nheads_q, f"softmax_lse heads {nheads_lse} != q heads {nheads_q}"

        # assert output shapes
        assert o.shape == (
            total_seqlen_q,
            nheads_q,
            head_size_v,
        ), f"o shape {o.shape} != expected {(total_seqlen_q, nheads_q, head_size_v)}"
        assert do.shape == o.shape, f"do shape {do.shape} != o shape {o.shape}"
        assert dq.shape == q.shape, f"dq shape {dq.shape} != q shape {q.shape}"
        assert dk.shape == k.shape, f"dk shape {dk.shape} != k shape {k.shape}"
        assert dv.shape == v.shape, f"dv shape {dv.shape} != v shape {v.shape}"

        # assert cu_seqlens
        assert cu_seqlens_q.dtype == torch.int32, f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}"
        assert cu_seqlens_k.dtype == torch.int32, f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}"
        assert cu_seqlens_q[0] == 0, "cu_seqlens_q must start with 0"
        assert cu_seqlens_k[0] == 0, "cu_seqlens_k must start with 0"
        assert cu_seqlens_q[-1] == total_seqlen_q, (
            f"cu_seqlens_q[-1] {cu_seqlens_q[-1]} != total_seqlen_q {total_seqlen_q}"
        )
        assert cu_seqlens_k[-1] == total_seqlen_k, (
            f"cu_seqlens_k[-1] {cu_seqlens_k[-1]} != total_seqlen_k {total_seqlen_k}"
        )

        # set vars
        batch = len(cu_seqlens_q) - 1
        head_size_qk = head_size_q

        # strides -- leading 0 stands in for the batch stride varlen has no true axis
        # for (all sequences are packed along dim 0; cu_seqlens carries the offsets).
        stride_qb, stride_qm, stride_qh, stride_qd = (0, *q.stride())
        stride_kb, stride_kn, stride_kh, stride_kd = (0, *k.stride())
        stride_vb, stride_vn, stride_vh, stride_vd = (0, *v.stride())
        stride_ob, stride_om, stride_oh, stride_od = (0, *o.stride())
        stride_dqb, stride_dqm, stride_dqh, stride_dqd = (0, *dq.stride())
        stride_dkb, stride_dkn, stride_dkh, stride_dkd = (0, *dk.stride())
        stride_dvb, stride_dvn, stride_dvh, stride_dvd = (0, *dv.stride())
        stride_dob, stride_dom, stride_doh, stride_dod = (0, *do.stride())
        stride_lse_b, stride_lse_h, stride_lse_m = (0, *softmax_lse.stride())
    else:
        # shapes
        batch_q, seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, seqlen_k, nheads_k, head_size_k = k.shape
        batch_v, seqlen_v, nheads_v, head_size_v = v.shape
        _batch_lse, nheads_lse, _seqlen_lse = softmax_lse.shape

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
        ), f"o shape {o.shape} != expected"
        assert do.shape == o.shape, f"do shape {do.shape} != o shape {o.shape}"
        assert dq.shape == q.shape, f"dq shape {dq.shape} != q shape {q.shape}"
        assert dk.shape == k.shape, f"dk shape {dk.shape} != k shape {k.shape}"
        assert dv.shape == v.shape, f"dv shape {dv.shape} != v shape {v.shape}"

        # assert softmax_lse shape
        assert softmax_lse.shape == (
            batch_q,
            nheads_q,
            seqlen_q,
        ), f"softmax_lse shape {softmax_lse.shape} != expected"

        # set vars
        batch = batch_q
        head_size_qk = head_size_q
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

        # strides
        stride_qb, stride_qm, stride_qh, stride_qd = q.stride()
        stride_kb, stride_kn, stride_kh, stride_kd = k.stride()
        stride_vb, stride_vn, stride_vh, stride_vd = v.stride()
        stride_ob, stride_om, stride_oh, stride_od = o.stride()
        stride_dqb, stride_dqm, stride_dqh, stride_dqd = dq.stride()
        stride_dkb, stride_dkn, stride_dkh, stride_dkd = dk.stride()
        stride_dvb, stride_dvn, stride_dvh, stride_dvd = dv.stride()
        stride_dob, stride_dom, stride_doh, stride_dod = do.stride()
        stride_lse_b, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # "Active" iff either edge differs from the -1 "off" sentinel. This mirrors
    # the forward kernels (fwd_prefill.py / fwd_decode.py both test `!= -1`); the
    # interface guards removed in this change used `>= 0`, which is equivalent for
    # every valid input (-1 is the only negative either edge ever takes).
    use_sliding_window = window_size_left != -1 or window_size_right != -1

    # score_mod/mask_mod are threaded through both fused kernels (causal and non-causal)
    # via _bwd_dkdv_inner / _bwd_dq_inner.
    block_sparse = block_sparse_dkdv is not None
    # Block-sparse always takes the non-causal fused kernel: its block geometry comes
    # from the index lists, not from the causal block arithmetic the causal kernel is
    # built around. Causality is still honoured -- BlockPlan.with_causal drops the
    # blocks past the diagonal and forces the straddling ones onto the masked pass,
    # and IS_CAUSAL below applies the per-element mask there.
    use_causal_kernel = causal and not block_sparse
    # Either edge may be unbounded and is handled uniformly (mirroring the forward
    # kernels): WINDOW_SIZE_LEFT < 0 lets keys reach back to 0 / queries have no
    # upper limit, and WINDOW_SIZE_RIGHT < 0 lets keys reach forward to seqlen_k - 1
    # / queries have no lower limit. The per-element window mask and the
    # _sliding_window_{q,k}_bounds block-range helpers both special-case each edge,
    # so no negative-right guard is needed. (-1, -1) is the only "off" sentinel.

    # get closest power of 2 over or equal to 32.
    padded_d_model_qk = 1 << (head_size_qk - 1).bit_length()
    padded_d_model_qk = max(padded_d_model_qk, 32)
    padded_d_model_v = 1 << (head_size_v - 1).bit_length()
    padded_d_model_v = max(padded_d_model_v, 32)
    HEAD_DIM_QK = padded_d_model_qk
    HEAD_DIM_V = padded_d_model_v
    ACTUAL_HEAD_DIM_QK = head_size_qk
    ACTUAL_HEAD_DIM_V = head_size_v

    # Validate pre-allocated delta tensor
    if IS_VARLEN:
        # Shape expected by interface varlen backward: (Hq, Total_Q)
        total_q, _, _ = q.shape
        assert delta.shape[0] == nheads_q, (
            f"delta.shape[0] ({delta.shape[0]}) must equal nheads_q ({nheads_q})"
        )
        assert delta.shape[1] >= total_q, f"delta.shape[1] ({delta.shape[1]}) must be >= total_q ({total_q})"
        assert delta.dtype == torch.float32, f"delta must be float32, got {delta.dtype}"
        assert delta.device == q.device, "delta must be on same device as q"
        stride_delta_b, stride_delta_h, stride_delta_m = (
            0,
            delta.stride(0),
            delta.stride(1),
        )
    else:
        # Shape expected by dense backward: (B, Hq, Sq)
        seqlen_q = q.shape[1]
        assert delta.shape[0] == batch, f"delta.shape[0] ({delta.shape[0]}) must equal batch ({batch})"
        assert delta.shape[1] == nheads_q, (
            f"delta.shape[1] ({delta.shape[1]}) must equal nheads_q ({nheads_q})"
        )
        assert delta.shape[2] >= seqlen_q, (
            f"delta.shape[2] ({delta.shape[2]}) must be >= seqlen_q ({seqlen_q})"
        )
        assert delta.dtype == torch.float32, f"delta must be float32, got {delta.dtype}"
        assert delta.device == q.device, "delta must be on same device as q"
        stride_delta_b, stride_delta_h, stride_delta_m = delta.stride()

    def pre_grid(META):
        return (
            triton.cdiv(max_seqlen_q, META["PRE_BLOCK"]),
            batch,
            nheads_q,
        )

    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dod,
        stride_delta_b,
        stride_delta_h,
        stride_delta_m,
        cu_seqlens_q,
        max_seqlen_q,
        HEAD_DIM_V=HEAD_DIM_V,
        ACTUAL_HEAD_DIM_V=ACTUAL_HEAD_DIM_V,
        IS_VARLEN=IS_VARLEN,
    )
    # d(lse)/d(score) is the softmax row P, so a gradient on the returned LSE adds
    # P * dlse to dS = P * (dP - delta): fold it into delta. delta and lse share a layout.
    if dlse is not None:
        delta.sub_(dlse)

    if DEBUG:
        print("delta:", delta, delta.shape)

    # Same 64 KiB LDS cap as the forward (see max_block_for_lds), but here the
    # dominant tiles are sized by BLOCK_N1 (dK/dV) and BLOCK_M2 (dQ). The tuned
    # configs use 128 for both, which overflows once padded_d_model_qk >= 512. See
    # fwd_prefill.attention_forward_prefill_triton_impl for why this bypasses the
    # autotuner rather than pruning its config list.
    cap_block = max_block_for_lds(padded_d_model_qk, q.element_size())
    tuned_block = max(
        (
            max(c.kwargs.get("BLOCK_N1", 0), c.kwargs.get("BLOCK_M2", 0))
            for c in (noncausal_autotune_configs + causal_autotune_configs)
        ),
        default=0,
    )
    if block_sparse:
        bs_q, bs_kv = block_sparse_dkdv.block_size
        bwd_block_overrides = dict(
            BLOCK_M1=bs_q,
            BLOCK_N1=bs_kv,
            BLOCK_M2=bs_q,
            BLOCK_N2=bs_kv,
            **sparse_bwd_default(bs_kv),
        )
        # Same rule as _sanitize_nonkdim: the accumulating tl.dot miscompiles when the
        # masked sub-block (here just the block size, BLK_SLICE_FACTOR being 1) is
        # under _MFMA_ACC_MIN_K.
        if min(bs_q, bs_kv) < _MFMA_ACC_MIN_K:
            bwd_block_overrides.pop("matrix_instr_nonkdim", None)
    elif cap_block < tuned_block:
        if cap_block == 0:
            raise ValueError(
                f"head_dim {head_size_qk} (padded to {padded_d_model_qk}) is too large for "
                f"this GPU's LDS budget in the backward kernel"
            )
        bwd_block_overrides = dict(
            BLOCK_M1=min(32, cap_block),
            BLOCK_N1=cap_block,
            BLOCK_M2=cap_block,
            BLOCK_N2=min(64, cap_block),
            BLK_SLICE_FACTOR=2,
            waves_per_eu=1,
            # deliberately no matrix_instr_nonkdim: these capped blocks halve to <= 16
            # for the causal masked sweep, which miscompiles -- see _sanitize_nonkdim.
            num_stages=1,
            num_warps=4,
        )
    else:
        bwd_block_overrides = {}

    seqlen = max(max_seqlen_q, max_seqlen_k)

    arch = get_arch()
    num_xcd = 1 if arch.is_rdna else 8

    if bwd_block_overrides:
        fixed_block_n1 = bwd_block_overrides["BLOCK_N1"]

        def grid(META):
            return (
                nheads_k,
                ((seqlen + fixed_block_n1 - 1) // fixed_block_n1),
                batch,
            )

    else:

        def grid(META):
            # The fused backward runs two phases off the same program id --
            # dK/dV strides by BLOCK_N1, dQ by BLOCK_M2 -- so the grid
            # must cover whichever needs more programs. Upstream sized it by BLOCK_N1
            # alone, which silently computes only the first
            # (seqlen/BLOCK_N1)*BLOCK_M2 rows of dQ whenever BLOCK_M2 < BLOCK_N1.
            # Every shipped config happens to satisfy BLOCK_M2 >= BLOCK_N1, so the
            # latent bug never fired for them; it blocks otherwise-faster tile shapes
            # from the autotune space. Both phases already guard their own program
            # id, so over-provisioning is safe.
            step = min(META["BLOCK_N1"], META["BLOCK_M2"])
            return (nheads_k, ((seqlen + step - 1) // step), batch)

    if use_causal_kernel:
        if DEBUG_TRITON:
            print(f"bwd_kernel: grid = {grid}")
        causal_launcher = (
            bwd_kernel_fused_causal.fn[grid] if bwd_block_overrides else bwd_kernel_fused_causal[grid]
        )
        causal_launcher(
            q,
            k,
            v,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            softmax_lse,
            delta,
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
            stride_dqb,
            stride_dqh,
            stride_dqm,
            stride_dqd,
            stride_dkb,
            stride_dkh,
            stride_dkn,
            stride_dkd,
            stride_dvb,
            stride_dvh,
            stride_dvn,
            stride_dvd,
            stride_lse_b,
            stride_lse_h,
            stride_lse_m,
            stride_delta_b,
            stride_delta_h,
            stride_delta_m,
            stride_dob,
            stride_doh,
            stride_dom,
            stride_dod,
            nheads_q,
            nheads_k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            HEAD_DIM_QK=HEAD_DIM_QK,
            HEAD_DIM_V=HEAD_DIM_V,
            ACTUAL_HEAD_DIM_QK=ACTUAL_HEAD_DIM_QK,
            ACTUAL_HEAD_DIM_V=ACTUAL_HEAD_DIM_V,
            IS_VARLEN=IS_VARLEN,
            USE_EXP2=use_exp2,
            USE_SLIDING_WINDOW=use_sliding_window,
            WINDOW_SIZE_LEFT=window_size_left,
            WINDOW_SIZE_RIGHT=window_size_right,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            NUM_XCD=num_xcd,
            SCORE_MOD=score_mod,
            MASK_MOD=mask_mod,
            SCORE_MOD_BWD=score_mod_bwd,
            SCORE_GRAD_HOOK=score_grad_hook,
            SCORE_GRAD=score_grad,
            **aux_kernel_args(aux_tensors),
            **mask_aux_kernel_args(mask_aux_tensors),
            **bwd_block_overrides,
        )
    else:
        noncausal_launcher = (
            bwd_kernel_fused_noncausal.fn[grid] if bwd_block_overrides else bwd_kernel_fused_noncausal[grid]
        )

        def _launch_noncausal(_overrides):
            """The non-causal launch, parameterised by tile/knob overrides, so the
            block-sparse path can time candidates before committing (see pick_knobs)."""
            noncausal_launcher(
                q,
                k,
                v,
                sm_scale,
                do,
                dq,
                dk,
                dv,
                softmax_lse,
                delta,
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
                stride_dqb,
                stride_dqh,
                stride_dqm,
                stride_dqd,
                stride_dkb,
                stride_dkh,
                stride_dkn,
                stride_dkd,
                stride_dvb,
                stride_dvh,
                stride_dvn,
                stride_dvd,
                stride_lse_b,
                stride_lse_h,
                stride_lse_m,
                stride_delta_b,
                stride_delta_h,
                stride_delta_m,
                stride_dob,
                stride_doh,
                stride_dom,
                stride_dod,
                nheads_q,
                nheads_k,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                HEAD_DIM_QK=HEAD_DIM_QK,
                HEAD_DIM_V=HEAD_DIM_V,
                ACTUAL_HEAD_DIM_QK=ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V=ACTUAL_HEAD_DIM_V,
                IS_VARLEN=IS_VARLEN,
                USE_EXP2=use_exp2,
                USE_SLIDING_WINDOW=use_sliding_window,
                WINDOW_SIZE_LEFT=window_size_left,
                WINDOW_SIZE_RIGHT=window_size_right,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                NUM_XCD=num_xcd,
                SCORE_MOD=score_mod,
                MASK_MOD=mask_mod,
                SCORE_MOD_BWD=score_mod_bwd,
                BS_DKDV_CNT=None if not block_sparse else block_sparse_dkdv.mask_block_cnt,
                BS_DKDV_IDX=None if not block_sparse else block_sparse_dkdv.mask_block_idx,
                BS_DQ_CNT=None if not block_sparse else block_sparse_dq.mask_block_cnt,
                BS_DQ_IDX=None if not block_sparse else block_sparse_dq.mask_block_idx,
                **bs_tensor_strides(
                    "bs_dkdv_cnt", None if not block_sparse else block_sparse_dkdv.mask_block_cnt
                ),
                **bs_tensor_strides(
                    "bs_dkdv_idx", None if not block_sparse else block_sparse_dkdv.mask_block_idx
                ),
                **bs_tensor_strides(
                    "bs_dq_cnt", None if not block_sparse else block_sparse_dq.mask_block_cnt
                ),
                **bs_tensor_strides(
                    "bs_dq_idx", None if not block_sparse else block_sparse_dq.mask_block_idx
                ),
                BLOCK_SPARSE=block_sparse,
                IS_CAUSAL=causal,
                SCORE_GRAD_HOOK=score_grad_hook,
                SCORE_GRAD=score_grad,
                **aux_kernel_args(aux_tensors),
                **mask_aux_kernel_args(mask_aux_tensors),
                **_overrides,
            )

        if block_sparse and AUTOTUNE != "off":
            # Choose once per shape, then launch normally. Key mirrors
            # NONCAUSAL_AUTOTUNE_KEYS, plus the block size the tiles are pinned to.
            tune_key = (
                bs_q,
                bs_kv,
                max_seqlen_q,
                max_seqlen_k,
                ACTUAL_HEAD_DIM_QK,
                ACTUAL_HEAD_DIM_V,
                IS_VARLEN,
                nheads_q,
                nheads_k,
            )
            candidates = [
                dict(BLOCK_M1=bs_q, BLOCK_N1=bs_kv, BLOCK_M2=bs_q, BLOCK_N2=bs_kv, **knobs)
                for knobs in sparse_bwd_candidates(bs_kv)
            ]
            bwd_block_overrides = pick_knobs(_SPARSE_BWD_CHOICE, tune_key, candidates, _launch_noncausal)

        _launch_noncausal(bwd_block_overrides)
