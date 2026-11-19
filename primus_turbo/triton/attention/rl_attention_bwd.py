###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deterministic attention backward for RL attention (see rl_attention_fwd.py).

Vendored from ROCm/aiter @ 892e5d003df529516c864f90b2fc0e0f47f093a2 (MIT):
aiter/ops/triton/_triton_kernels/attention/mha_onekernel_bwd.py (kernels),
aiter/ops/triton/attention/mha_onekernel_bwd.py (flash_attn_onekernel_backward),
utils/_triton/kernel_repr.py and _triton_kernels/common/mha_kernel_utils.py (helpers below).

Changes from upstream: the gfx950 config is inlined instead of read from aiter's JSON; fp8 is off;
visibility intervals (HAS_VIS) mask the non-causal kernel, and per-tile query ranges prune the
dK/dV and dQ walks to the intervals. dQ, dK and dV are computed in separate passes without atomics.
"""

import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    x_amax = tl.max(tl.abs(x))
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value).replace("-", "NEG_")
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value)).replace("-", "NEG_")
        return str(value).replace("-", "NEG_").replace("+", "POS_").replace(".", "_")
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        joined = "_".join(_sanitize_constexpr_value(item) for item in items)
        return joined if joined else "NONE"
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned.upper() if cleaned else "NONE"


def make_kernel_repr(base_name, config_keys, name_key=None):
    def _repr(specialization):
        constants = specialization.constants
        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(override))
                if cleaned:
                    name = cleaned
        parts = [f"{key}_{_sanitize_constexpr_value(constants.get(key, None))}" for key in config_keys]
        return f"{name}_{'_'.join(parts)}" if parts else name

    return _repr


# NOTE: triton fails to import tl.constexprs so create them here for the file
DROPOUT_USE_PYTORCH = False
DROPOUT_DUMP = False

tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)


# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q), same as softmax_lse defined at
_bwd_preprocess_repr = make_kernel_repr(
    "_bwd_preprocess",
    [
        "BLOCK_M",
        "BLOCK_D_MODEL",
        "IS_VARLEN",
        "IS_FP8",
    ],
)


@triton.jit(repr=_bwd_preprocess_repr)
def _bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    stride_o_b,
    stride_o_h,
    stride_o_m,
    stride_o_k,
    stride_do_b,
    stride_do_h,
    stride_do_m,
    stride_do_k,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    stride_descale_do_z,
    cu_seqlens_q,
    max_seqlen_q,
    descale_do_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)  # seqlen
    bid = tl.program_id(1)  # batch
    hid = tl.program_id(2)  # head

    # Handle varlen
    q_start = 0
    seqlen_q = max_seqlen_q
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    else:
        q_start = 0
        seqlen_q = max_seqlen_q

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # O and DO may have different strides (e.g. BSHD vs SBHD memory layout),
    # so address each with its own strides.
    offs_o = (
        bid * stride_o_b
        + hid * stride_o_h
        + q_start * stride_o_m
        + offs_m[:, None] * stride_o_m
        + offs_k[None, :] * stride_o_k
    )
    offs_do = (
        bid * stride_do_b
        + hid * stride_do_h
        + q_start * stride_do_m
        + offs_m[:, None] * stride_do_m
        + offs_k[None, :] * stride_do_k
    )

    # create masks
    mask_m = offs_m < seqlen_q
    mask = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask &= offs_k[None, :] < BLOCK_D_MODEL

    # load [BLOCK_M, BLOCK_D_MODEL_POW2]
    o = tl.load(o_ptr + offs_o, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs_do, mask=mask, other=0.0)

    # compute and write-back to delta
    if IS_FP8:
        descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hid)

        # NOTE: do is in the fp8 range and o is not in fp8
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do), axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    offs_delta = (
        bid * stride_delta_b + hid * stride_delta_h + q_start * stride_delta_m + offs_m * stride_delta_m
    )
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _bwd_dkdv_inner(
    dk,  # output
    dk_pe,  # optional output, pass None for non-PE case
    dv,  # output
    Q,
    k,
    k_pe,
    v,
    DO,
    M,
    D,
    sm_scale,  # input tensor
    stride_qm,
    stride_qk,
    stride_dom,
    stride_dok,
    stride_dropoutm,
    stride_dropoutn,
    stride_deltam,
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    PE_HEAD_DIM: tl.constexpr,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    dropout_offset,
    alibi_slope,
    seqlen_q,
    seqlen_k,  # max sequence length for q and k
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  # iteration numbers
    descale_q,
    descale_k,
    descale_v,
    descale_do,  # fp8 descale factors from user
    MASK: tl.constexpr,  # causal masking, only apply to tiles on mask diagonal
    ENABLE_DROPOUT: tl.constexpr,  # activate dropout
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,  # activate exp2
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    VIS_LO=None,
    VIS_HI=None,
    HAS_VIS: tl.constexpr = False,
    VIS_BASE=0,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_k_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    if HAS_PE:
        qT_pe_ptrs = Q + offs_m[None, :] * stride_qm + offs_k_pe[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)

    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_m = {curr_m}")
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        # update the mask because offs_m advanced
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
            mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        if HAS_PE:
            qT_pe = tl.load(qT_pe_ptrs, mask=mask_qT, other=0.0)
        # generate dropout mask
        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (
                curr_philox_offset + offs_m[None, :] * stride_dropoutm + offs_n[:, None] * stride_dropoutn
            )
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[None, :] * stride_dropoutm + offs_n[:, None] * stride_dropoutn
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs, mask=mask_nm)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        if IS_FP8:
            qkT = tl.dot(k, qT) * descale_q * descale_k
        else:
            qkT = tl.dot(k, qT)
            if HAS_PE:
                qkT = tl.dot(k_pe, qT_pe, acc=qkT)
        qkT_scaled = qkT * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_n[:, None] + seqlen_q - seqlen_k - offs_m[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qkT_scaled += alibi_block

        if DEBUG_TRITON_DETAIL and start_n == 256:
            print(f"qT: {qT.shape}\n", qT)
            print(f"k: {k.shape}\n", k)
            print(f"qkT scaled: {qkT.shape}\n", qkT_scaled)
        # TODO: remove the scaling of m later when we removed re-scaling in fwd
        if USE_EXP2:
            pT = tl.math.exp2(qkT_scaled * RCP_LN2 - m[None, :] * RCP_LN2)
        else:
            pT = tl.math.exp(qkT_scaled - m[None, :])

        # Autoregressive masking.
        if MASK:
            # offset offs_m with delta_qk since the causal mask starts at
            # bottom right of the (seqlen_q, seqlen_k) matrix
            causal_mask = (offs_m[None, :] - delta_qk) >= offs_n[:, None]
            mask = causal_mask & mask_nm
            if DEBUG_TRITON_DETAIL and start_n == 256:
                print(f"causal_mask: {causal_mask.shape}\n", causal_mask)
                print(
                    f"qkT after causal: {qkT.shape}\n",
                    tl.where(causal_mask, qkT * sm_scale, 0.0),
                )
            pT = tl.where(mask, pT, 0.0)
        if SLIDING_WINDOW > 0:
            window_mask = offs_n[:, None] >= (offs_m[None, :] - delta_qk - SLIDING_WINDOW)
            pT = tl.where(window_mask, pT, 0.0)
        if HAS_VIS:
            vis_lo = tl.load(VIS_LO + offs_m, mask=mask_m, other=1) - VIS_BASE
            vis_hi = tl.load(VIS_HI + offs_m, mask=mask_m, other=0) - VIS_BASE
            vis = (offs_n[:, None] >= vis_lo[None, :]) & (offs_n[:, None] <= vis_hi[None, :]) & mask_nm
            pT = tl.where(vis, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        # Compute dV.
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = _compute_fp8_scaling_factors(pT_dropout, FP8_MAX)
                dv += (
                    tl.dot((pT_dropout * scale_p_dropout).to(do.type.element_ty), do)
                    * descale_p_dropout
                    * descale_do
                )
            else:
                dv = tl.dot(pT_dropout.to(do.type.element_ty), do, acc=dv)
        else:
            if IS_FP8:
                scale_pT, descale_pT = _compute_fp8_scaling_factors(pT, FP8_MAX)
                dv += tl.dot((pT * scale_pT).to(do.type.element_ty), do) * descale_pT * descale_do
            else:
                dv = tl.dot(pT.to(do.type.element_ty), do, acc=dv)

        if DEBUG_TRITON_DETAIL and start_n == 256:
            print(f"pT: {pT.shape}\n", pT)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)
        # Compute dP and dS.
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))
        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale
        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        if IS_FP8:
            scale_dsT, descale_dsT = _compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)) * descale_dsT * descale_q
        else:
            dk = tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT), acc=dk)
            if HAS_PE:
                dk_pe = tl.dot(dsT.to(qT_pe.type.element_ty), tl.trans(qT_pe), acc=dk_pe)
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        if HAS_PE:
            qT_pe_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_dom
    return dk, dk_pe, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _bwd_dq_inner(
    dq,  # output
    dq_pe,  # optional output, pass None for non-PE case
    q,
    q_pe,
    K,
    V,
    do,
    m,
    Di,  # D (= delta) is pre-divided by ds_scale.
    sm_scale,  # input
    # shared by Q/K/V.
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_dropoutm,
    stride_dropoutn,  # stride for dropout
    seqlen_q,
    seqlen_k,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    PE_HEAD_DIM: tl.constexpr,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    dropout_offset,
    alibi_slope,
    # Filled in by the wrapper.
    start_m,
    start_n,
    end_n,
    num_steps,
    descale_q,
    descale_k,
    descale_v,
    descale_do,  # fp8 descale factors from user
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    ENABLE_SINK: tl.constexpr,
    VIS_LO=None,
    VIS_HI=None,
    HAS_VIS: tl.constexpr = False,
    VIS_BASE=0,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_k_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    if HAS_PE:
        kT_pe_ptrs = K + offs_n[None, :] * stride_kn + offs_k_pe[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k[:, None] * stride_vk
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    delta_recomp = tl.zeros([BLOCK_M2], dtype=tl.float32)
    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_n = {curr_n}")
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        # end_n is needed because the end of causal True might not be perfectly
        # aligned with the end of the block
        mask_n = offs_n < end_n
        if DEBUG_TRITON_DETAIL:
            print(f"start_n = {start_n}, end_n = {end_n}, offs_n: {offs_n.shape}\n{offs_n}")
        if DEBUG_TRITON_DETAIL:
            print(f"mask_n: {mask_n.shape}\n{mask_n}")
        mask_kT = mask_n[None, :]
        mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD:
            mask_kT &= offs_k[:, None] < ACTUAL_HEAD_DIM

        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        if HAS_PE:
            kT_pe = tl.load(kT_pe_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_kT, other=0.0)

        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (
                curr_philox_offset + offs_m[:, None] * stride_dropoutm + offs_n[None, :] * stride_dropoutn
            )
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[:, None] * stride_dropoutm + offs_n[None, :] * stride_dropoutn
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs, mask=mask_mn)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)

        if IS_FP8:
            qk = tl.dot(q, kT) * descale_q * descale_k
        else:
            qk = tl.dot(q, kT)
            if HAS_PE:
                qk = tl.dot(q_pe, kT_pe, acc=qk)
        qk_scaled = qk * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qk_scaled += alibi_block

        if DEBUG_TRITON_DETAIL:
            print(f"qk scaled: {qk.shape}\n", qk_scaled)
        if USE_EXP2:
            p = tl.math.exp2(qk_scaled * RCP_LN2 - m * RCP_LN2)
        else:
            p = tl.math.exp(qk_scaled - m)

        # Autoregressive masking.
        if MASK:
            causal_mask = (offs_m[:, None] - delta_qk) >= offs_n[None, :]
            mask = causal_mask & mask_mn
            p = tl.where(mask, p, 0.0)
        else:
            # Keys past end_n are loaded as 0, so p = exp(-m) there, which overflows to inf when every
            # logit of the row is very negative (m < -88). inf * kT(=0) would then put NaN in dq.
            p = tl.where(mask_n[None, :], p, 0.0)
        if SLIDING_WINDOW > 0:
            window_mask = offs_n[None, :] >= (offs_m[:, None] - delta_qk - SLIDING_WINDOW)
            p = tl.where(window_mask, p, 0.0)
        if HAS_VIS:
            vis_lo = tl.load(VIS_LO + offs_m, mask=mask_m, other=1) - VIS_BASE
            vis_hi = tl.load(VIS_HI + offs_m, mask=mask_m, other=0) - VIS_BASE
            vis = (
                (offs_n[None, :] >= vis_lo[:, None])
                & (offs_n[None, :] <= vis_hi[:, None])
                & (offs_n[None, :] < seqlen_k)
            )
            p = tl.where(vis, p, 0.0)
        # Compute dP and dS.
        if IS_FP8:
            dp = tl.dot(do, vT) * descale_do * descale_v
        else:
            dp = tl.dot(do, vT)
        if ENABLE_DROPOUT:
            dp = tl.where(dropout_mask, dp, 0.0) * dropout_scale
        if ENABLE_SINK:
            # Equal to do_i . o_i but skips the bf16 round-trip through forward `o`.
            delta_recomp += tl.sum(p * dp, axis=1)
        delta_i = Di[:, None]
        ds = p * (dp - delta_i)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        if IS_FP8:
            scale_ds, descale_ds = _compute_fp8_scaling_factors(ds, FP8_MAX)
            dq += tl.dot((ds * scale_ds).to(kT.type.element_ty), tl.trans(kT)) * descale_ds * descale_k
        else:
            dq = tl.dot(ds.to(kT.type.element_ty), tl.trans(kT), acc=dq)
            if HAS_PE:
                dq_pe = tl.dot(ds.to(kT_pe.type.element_ty), tl.trans(kT_pe), acc=dq_pe)
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        if HAS_PE:
            kT_pe_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_vn
    return dq, dq_pe, delta_recomp


_bwd_kernel_causal_repr = make_kernel_repr(
    "bwd_kernel_causal",
    [
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_M2",
        "BLOCK_N2",
        "BLK_SLICE_FACTOR",
        "HEAD_DIM",
        "ENABLE_DROPOUT",
        "IS_VARLEN",
        "USE_ALIBI",
        "USE_EXP2",
        "IS_FP8",
        "USE_INT64_STRIDES",
        "ENABLE_SINK",
        "SLIDING_WINDOW",
    ],
)


@triton.jit(repr=_bwd_kernel_causal_repr)
def bwd_kernel_causal(  # grid = (tl.cdiv(max_seqlen_q // BLOCK_M2), batch, nheads_q)
    Q,
    K,
    V,
    Sink,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    DSink,
    M,
    Delta,
    stride_qb_in,
    stride_qh_in,
    stride_qm_in,
    stride_qd_in,
    stride_kb_in,
    stride_kh_in,
    stride_kn_in,
    stride_kd_in,
    stride_vb_in,
    stride_vh_in,
    stride_vn_in,
    stride_vd_in,
    stride_dqb_in,
    stride_dqh_in,
    stride_dqm_in,
    stride_dqd_in,
    stride_dkb_in,
    stride_dkh_in,
    stride_dkn_in,
    stride_dkd_in,
    stride_dvb_in,
    stride_dvh_in,
    stride_dvn_in,
    stride_dvd_in,
    stride_deltab_in,
    stride_deltah_in,
    stride_deltam_in,
    stride_dob_in,
    stride_doh_in,
    stride_dom_in,
    stride_dod_in,
    stride_dropoutb_in,
    stride_dropouth_in,
    stride_dropoutm_in,
    stride_dropoutn_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_descale_do_z_in,
    stride_az_in,
    stride_ah_in,
    HQ,
    HK,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    Dropout_mask,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    Alibi_slopes,
    Descale_q,
    Descale_k,
    Descale_v,
    Descale_do,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    PE_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
    ENABLE_SINK: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    VisLo=None,
    VisHi=None,
    HAS_VIS: tl.constexpr = False,
):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qd = tl.cast(stride_qd_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kd = tl.cast(stride_kd_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vd = tl.cast(stride_vd_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqd = tl.cast(stride_dqd_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkd = tl.cast(stride_dkd_in, tl.int64)
        stride_dvb = tl.cast(stride_dvb_in, tl.int64)
        stride_dvh = tl.cast(stride_dvh_in, tl.int64)
        stride_dvn = tl.cast(stride_dvn_in, tl.int64)
        stride_dvd = tl.cast(stride_dvd_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dod = tl.cast(stride_dod_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
        stride_az = tl.cast(stride_az_in, tl.int64)
        stride_ah = tl.cast(stride_ah_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qd = stride_qd_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kd = stride_kd_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vd = stride_vd_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqd = stride_dqd_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkd = stride_dkd_in
        stride_dvb = stride_dvb_in
        stride_dvh = stride_dvh_in
        stride_dvn = stride_dvn_in
        stride_dvd = stride_dvd_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dod = stride_dod_in
        philox_offset_base = philox_offset_base_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
        stride_az = stride_az_in
        stride_ah = stride_ah_in

    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
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
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    offs_d = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_d_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    GROUP_SIZE: tl.constexpr = HQ // HK

    # align the delta_qk
    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        # This section does dk and dv
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        if HAS_PE:
            dk_pe = tl.zeros([BLOCK_N1, PE_HEAD_DIM], dtype=tl.float32)
        else:
            # Couldn't assign None to dk_pe because _bwd_dkdv_inner can't return None.
            dk_pe = dk
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

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
        mask_kv = offs_n[:, None] < seqlen_k
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_kv &= mask_d[None, :]

        # K/V tensors not changed for the group
        adj_k = (
            bid * stride_kb
            + hkid * stride_kh
            + k_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        if HAS_PE:
            adj_k_pe = (
                bid * stride_kb
                + hkid * stride_kh
                + k_start * stride_kn
                + offs_n[:, None] * stride_kn
                + offs_d_pe[None, :] * stride_kd
            )
        adj_v = (
            bid * stride_vb
            + hkid * stride_vh
            + k_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
        if HAS_PE:
            k_pe = tl.load(K + adj_k_pe, mask=mask_kv, other=0.0)
        else:
            k_pe = None
        v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
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
            adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            M_ptr = M + adj_delta
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                dropout_offset = Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

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
            dk, dk_pe, dv = _bwd_dkdv_inner(
                dk,  # output tensor
                dk_pe,  # optional output tensor
                dv,  # output tensor
                Q_ptr,
                k,
                k_pe,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_dropoutm,
                stride_dropoutn,  # strides for dropout
                stride_deltam,
                MASK_BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM,
                ACTUAL_HEAD_DIM,  # head dim
                PE_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=True,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                SLIDING_WINDOW=SLIDING_WINDOW,
            )
            start_m += num_steps * MASK_BLOCK_M1
            end_m = seqlen_q
            if SLIDING_WINDOW > 0:
                end_m = min(start_n + BLOCK_N1 + delta_qk + SLIDING_WINDOW, seqlen_q)
            num_steps = tl.cdiv(max(end_m - start_m, 0), BLOCK_M1)
            end_m = start_m + num_steps * BLOCK_M1

            if DEBUG_TRITON:
                print(f"start_m after Masked step: {start_m}; num_steps: {num_steps}")
            if DEBUG_TRITON:
                print(
                    f"unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}"
                )
            if DEBUG_TRITON:
                print("unMasked")
            dk, dk_pe, dv = _bwd_dkdv_inner(
                dk,  # output tensor
                dk_pe,  # optional output tensor
                dv,  # output tensor
                Q_ptr,
                k,
                k_pe,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_dropoutm,
                stride_dropoutn,  # strides for dropout
                stride_deltam,
                BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM,
                ACTUAL_HEAD_DIM,  # head dim
                PE_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=False,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                SLIDING_WINDOW=SLIDING_WINDOW,
            )
        # end of GQA/MQA of dkdv
        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_kv)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_kv)
        if HAS_PE:
            offs_dk_pe = offs_n[:, None] * stride_dkn + offs_d_pe[None, :] * stride_dkd
            dk_pe *= sm_scale
            tl.store(DK + adj_dk + offs_dk_pe, dk_pe, mask=mask_kv)

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
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_q &= mask_d[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        if HAS_PE:
            offs_q_pe = offs_m[:, None] * stride_qm + offs_d_pe[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
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
            adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                dropout_offset = Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth
            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            if HAS_PE:
                q_pe = tl.load(Q + adj_q + offs_q_pe, mask=mask_q, other=0.0)
            else:
                q_pe = None
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            mask_m = offs_m < seqlen_q
            m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=mask_m, other=0.0)
            m = m[:, None]
            delta = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
            # start can only be 0 at minimum
            start_n = max(end_n - BLOCK_M2, 0)
            num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N2)

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            if HAS_PE:
                dq_pe = tl.zeros([BLOCK_M2, PE_HEAD_DIM], dtype=tl.float32)
            else:
                dq_pe = dq  # Couldn't assign None to dq_pe because _bwd_dq_inner can't return None.
            dq, dq_pe, delta_recomp_masked = _bwd_dq_inner(
                dq,  # output tensor
                dq_pe,  # optional output tensor
                q,
                q_pe,
                K,
                V,
                do,
                m,
                delta,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_dropoutm,
                stride_dropoutn,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                MASK_BLOCK_N2,
                HEAD_DIM,
                ACTUAL_HEAD_DIM,
                PE_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                start_m,
                start_n,
                end_n,
                num_steps,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=True,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                SLIDING_WINDOW=SLIDING_WINDOW,
                ENABLE_SINK=ENABLE_SINK,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            window_start_n = 0
            if SLIDING_WINDOW > 0:
                window_start_n = max(start_m - delta_qk - SLIDING_WINDOW, 0)
            start_n = window_start_n // BLOCK_N2 * BLOCK_N2
            num_steps = tl.cdiv(max(end_n - start_n, 0), BLOCK_N2)
            if DEBUG_TRITON:
                print(
                    f"unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}"
                )
            dq, dq_pe, delta_recomp_unmasked = _bwd_dq_inner(
                dq,  # output tensor
                dq_pe,  # optional output tensor
                q,
                q_pe,
                K,
                V,
                do,
                m,
                delta,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_dropoutm,
                stride_dropoutn,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM,
                ACTUAL_HEAD_DIM,
                PE_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                start_m,
                start_n,
                end_n,
                num_steps,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                SLIDING_WINDOW=SLIDING_WINDOW,
                ENABLE_SINK=ENABLE_SINK,
            )
            if ENABLE_SINK:
                sink = tl.load(Sink + hqid).to(tl.float32)
                if USE_EXP2:
                    RCP_LN2: tl.constexpr = 1.4426950408889634
                    psink = tl.math.exp2(sink * RCP_LN2 - m * RCP_LN2)
                else:
                    psink = tl.math.exp(sink - m)
                delta_recomp = delta_recomp_masked + delta_recomp_unmasked
                dsink = tl.sum(-psink * delta_recomp[:, None])
                tl.atomic_add(DSink + hqid, dsink, sem="relaxed")
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            if HAS_PE:
                offs_dq_pe = offs_m[:, None] * stride_dqm + offs_d_pe[None, :] * stride_dqd
                dq_pe *= sm_scale
                tl.store(DQ + adj_dq + offs_dq_pe, dq_pe, mask=mask_q)
            # end of GQA/MQA of dq


_bwd_kernel_noncausal_repr = make_kernel_repr(
    "bwd_kernel_noncausal",
    [
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_M2",
        "BLOCK_N2",
        "BLK_SLICE_FACTOR",
        "HEAD_DIM",
        "ENABLE_DROPOUT",
        "IS_VARLEN",
        "USE_ALIBI",
        "USE_EXP2",
        "IS_FP8",
        "USE_INT64_STRIDES",
        "ENABLE_SINK",
        "SLIDING_WINDOW",
    ],
)


@triton.jit(repr=_bwd_kernel_noncausal_repr)
def bwd_kernel_noncausal(
    Q,
    K,
    V,
    Sink,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    DSink,
    M,
    Delta,
    stride_qb_in,
    stride_qh_in,
    stride_qm_in,
    stride_qd_in,
    stride_kb_in,
    stride_kh_in,
    stride_kn_in,
    stride_kd_in,
    stride_vb_in,
    stride_vh_in,
    stride_vn_in,
    stride_vd_in,
    stride_dqb_in,
    stride_dqh_in,
    stride_dqm_in,
    stride_dqd_in,
    stride_dkb_in,
    stride_dkh_in,
    stride_dkn_in,
    stride_dkd_in,
    stride_dvb_in,
    stride_dvh_in,
    stride_dvn_in,
    stride_dvd_in,
    stride_deltab_in,
    stride_deltah_in,
    stride_deltam_in,
    stride_dob_in,
    stride_doh_in,
    stride_dom_in,
    stride_dod_in,
    stride_dropoutb_in,
    stride_dropouth_in,
    stride_dropoutm_in,
    stride_dropoutn_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_descale_do_z_in,
    stride_az_in,
    stride_ah_in,
    HQ,
    HK,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    Dropout_mask,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    Alibi_slopes,
    Descale_q,
    Descale_k,
    Descale_v,
    Descale_do,
    BLOCK_M1: tl.constexpr,  # 32
    BLOCK_N1: tl.constexpr,  # 128
    BLOCK_M2: tl.constexpr,  # 128
    BLOCK_N2: tl.constexpr,  # 32
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    PE_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
    ENABLE_SINK: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    VisLo=None,
    VisHi=None,
    HAS_VIS: tl.constexpr = False,
    VisQMin=None,
    VisQMax=None,
    vis_num_kv_tiles=0,
):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qd = tl.cast(stride_qd_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kd = tl.cast(stride_kd_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vd = tl.cast(stride_vd_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqd = tl.cast(stride_dqd_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkd = tl.cast(stride_dkd_in, tl.int64)
        stride_dvb = tl.cast(stride_dvb_in, tl.int64)
        stride_dvh = tl.cast(stride_dvh_in, tl.int64)
        stride_dvn = tl.cast(stride_dvn_in, tl.int64)
        stride_dvd = tl.cast(stride_dvd_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dod = tl.cast(stride_dod_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
        stride_az = tl.cast(stride_az_in, tl.int64)
        stride_ah = tl.cast(stride_ah_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qd = stride_qd_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kd = stride_kd_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vd = stride_vd_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqd = stride_dqd_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkd = stride_dkd_in
        stride_dvb = stride_dvb_in
        stride_dvh = stride_dvh_in
        stride_dvn = stride_dvn_in
        stride_dvd = stride_dvd_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dod = stride_dod_in
        philox_offset_base = philox_offset_base_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
        stride_az = stride_az_in
        stride_ah = stride_ah_in

    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
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
    if HAS_VIS:
        vis_lo_seq = VisLo + q_start
        vis_hi_seq = VisHi + q_start
    else:
        vis_lo_seq = VisLo
        vis_hi_seq = VisHi

    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    offs_d = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_d_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    GROUP_SIZE: tl.constexpr = HQ // HK

    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        if HAS_PE:
            dk_pe = tl.zeros([BLOCK_N1, PE_HEAD_DIM], dtype=tl.float32)
        else:
            # Couldn't assign None to dk_pe because _bwd_dkdv_inner can't return None.
            dk_pe = dk
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_kv = offs_n[:, None] < seqlen_k
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_kv &= mask_d[None, :]
        # NOTE: don't assume that the strides for k and v are the same!
        # K/V tensors not changed for the group
        adj_k = (
            bid * stride_kb
            + hkid * stride_kh
            + k_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        if HAS_PE:
            adj_k_pe = (
                bid * stride_kb
                + hkid * stride_kh
                + k_start * stride_kn
                + offs_n[:, None] * stride_kn
                + offs_d_pe[None, :] * stride_kd
            )
        adj_v = (
            bid * stride_vb
            + hkid * stride_vh
            + k_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
        if HAS_PE:
            k_pe = tl.load(K + adj_k_pe, mask=mask_kv, other=0.0)
        else:
            k_pe = None
        v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            M_ptr = M + adj_delta
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                dropout_offset = Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            # because there is no causal, we always start from the beginning
            start_m = 0
            num_steps = tl.cdiv(seqlen_q, BLOCK_M1)
            if HAS_VIS:
                # only the query rows whose interval touches this KV tile (precomputed)
                q_first = tl.load(VisQMin + bid * vis_num_kv_tiles + pid)
                q_last = tl.load(VisQMax + bid * vis_num_kv_tiles + pid)
                start_m = tl.where(q_last >= q_first, (q_first // BLOCK_M1) * BLOCK_M1, 0)
                num_steps = tl.where(q_last >= q_first, tl.cdiv(q_last + 1 - start_m, BLOCK_M1), 0)
            dk, dk_pe, dv = _bwd_dkdv_inner(
                dk,  # output tensor
                dk_pe,  # optional output tensor
                dv,  # output tensor
                Q_ptr,
                k,
                k_pe,
                v,
                DO_ptr,
                M_ptr,
                Delta_ptr,
                sm_scale,  # input tensors
                stride_qm,
                stride_qd,  # strides for q
                stride_dom,
                stride_dod,  # strides for o
                stride_dropoutm,
                stride_dropoutn,  # strides for dropout
                stride_deltam,
                BLOCK_M1,
                BLOCK_N1,  # block dim
                HEAD_DIM,
                ACTUAL_HEAD_DIM,  # head dim
                PE_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                seqlen_q,
                seqlen_k,  # max sequence length for q and k
                start_n,
                start_m,
                num_steps,  # iteration numbers
                descale_q,
                descale_k,
                descale_v,
                descale_do,  # fp8 descale factors from user
                MASK=False,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                SLIDING_WINDOW=SLIDING_WINDOW,
                VIS_LO=vis_lo_seq,
                VIS_HI=vis_hi_seq,
                HAS_VIS=HAS_VIS,
                VIS_BASE=k_start,
            )

        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_kv)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_kv)
        if HAS_PE:
            offs_dk_pe = offs_n[:, None] * stride_dkn + offs_d_pe[None, :] * stride_dkd
            dk_pe *= sm_scale
            tl.store(DK + adj_dk + offs_dk_pe, dk_pe, mask=mask_kv)

    # THIS PART DOES DQ
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_q &= mask_d[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        if HAS_PE:
            offs_q_pe = offs_m[:, None] * stride_qm + offs_d_pe[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            Delta_ptr = Delta + adj_delta

            if USE_ALIBI:
                alibi_offset = bid * stride_az + hqid * stride_ah
                alibi_slope = tl.load(Alibi_slopes + alibi_offset)
            else:
                alibi_slope = None

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + bid * stride_dropoutb + hqid * stride_dropouth
                dropout_offset = Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            if HAS_PE:
                q_pe = tl.load(Q + adj_q + offs_q_pe, mask=mask_q, other=0.0)
            else:
                q_pe = None
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            mask_m = offs_m < seqlen_q
            m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=mask_m, other=0.0)
            m = m[:, None]
            delta = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            # start can only be 0 at minimum
            start_n = 0
            end_n = seqlen_k
            num_steps = tl.cdiv(seqlen_k, BLOCK_N2)
            if HAS_VIS:
                # only the KV tiles some row of this query block may see
                row_lo = tl.load(vis_lo_seq + offs_m, mask=mask_m, other=1) - k_start
                row_hi = tl.load(vis_hi_seq + offs_m, mask=mask_m, other=0) - k_start
                row_has = mask_m & (row_lo <= row_hi)
                key_first = tl.min(tl.where(row_has, row_lo, 2147483647), axis=0)
                key_last = tl.max(tl.where(row_has, row_hi, -1), axis=0)
                start_n = tl.where(key_last >= 0, (key_first // BLOCK_N2) * BLOCK_N2, 0)
                end_n = tl.where(key_last >= 0, key_last + 1, 0)
                num_steps = tl.where(key_last >= 0, tl.cdiv(end_n - start_n, BLOCK_N2), 0)

            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            if HAS_PE:
                dq_pe = tl.zeros([BLOCK_M2, PE_HEAD_DIM], dtype=tl.float32)
            else:
                dq_pe = dq  # Couldn't assign None to dq_pe because _bwd_dq_inner can't return None.
            dq, dq_pe, delta_recomp = _bwd_dq_inner(
                dq,  # output tensor
                dq_pe,  # optional output tensor
                q,
                q_pe,
                K,
                V,
                do,
                m,
                delta,
                sm_scale,
                stride_qm,
                stride_qd,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                stride_dropoutm,
                stride_dropoutn,
                seqlen_q,
                seqlen_k,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM,
                ACTUAL_HEAD_DIM,
                PE_HEAD_DIM,
                dropout_p,
                philox_seed,
                batch_philox_offset,
                dropout_offset,
                alibi_slope,
                start_m,
                start_n,
                end_n,
                num_steps,
                descale_q,
                descale_k,
                descale_v,
                descale_do,
                MASK=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
                SLIDING_WINDOW=SLIDING_WINDOW,
                ENABLE_SINK=ENABLE_SINK,
                VIS_LO=vis_lo_seq,
                VIS_HI=vis_hi_seq,
                HAS_VIS=HAS_VIS,
                VIS_BASE=k_start,
            )
            if ENABLE_SINK:
                sink = tl.load(Sink + hqid).to(tl.float32)
                if USE_EXP2:
                    RCP_LN2: tl.constexpr = 1.4426950408889634
                    psink = tl.math.exp2(sink * RCP_LN2 - m * RCP_LN2)
                else:
                    psink = tl.math.exp(sink - m)
                dsink = tl.sum(-psink * delta_recomp[:, None])
                tl.atomic_add(DSink + hqid, dsink, sem="relaxed")
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            if HAS_PE:
                offs_dq_pe = offs_m[:, None] * stride_dqm + offs_d_pe[None, :] * stride_dqd
                dq_pe *= sm_scale
                tl.store(DQ + adj_dq + offs_dq_pe, dq_pe, mask=mask_q)


@triton.jit
def vis_kv_tile_q_range_kernel(
    VisLo,
    VisHi,
    cu_seqlens_q,
    cu_seqlens_k,
    QMin,
    QMax,
    num_kv_tiles,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    """For each (sequence, KV tile of BLOCK_N keys): the first and last query row (sequence-relative)
    whose visibility interval [lo, hi] (global key indices) touches the tile; first > last if none."""
    bid = tl.program_id(0)
    tile = tl.program_id(1)
    q_start = tl.load(cu_seqlens_q + bid)
    q_end = tl.load(cu_seqlens_q + bid + 1)
    k_start = tl.load(cu_seqlens_k + bid)
    key_lo = tile * BLOCK_N
    key_hi = key_lo + BLOCK_N - 1
    first = tl.full([1], 2147483647, dtype=tl.int32)
    last = tl.full([1], -1, dtype=tl.int32)
    for s in range(q_start, q_end, BLOCK_Q):
        rows = s + tl.arange(0, BLOCK_Q)
        valid = rows < q_end
        lo = tl.load(VisLo + rows, mask=valid, other=1) - k_start
        hi = tl.load(VisHi + rows, mask=valid, other=0) - k_start
        touch = valid & (lo <= hi) & (lo <= key_hi) & (hi >= key_lo)
        rel = rows - q_start
        first = tl.minimum(first, tl.min(tl.where(touch, rel, 2147483647), axis=0))
        last = tl.maximum(last, tl.max(tl.where(touch, rel, -1), axis=0))
    out = bid * num_kv_tiles + tile + tl.arange(0, 1)
    tl.store(QMin + out, first)
    tl.store(QMax + out, last)


import torch  # noqa: E402


def _get_config() -> dict:
    """aiter's gfx950 `bkwd_onekernel` config (configs/gfx950/triton/attention/MHA/DEFAULT.json)."""
    return {
        "preprocess_kernel": {"PRE_BLOCK": 128},
        "onekernel": {
            "BLOCK_M1": 32,
            "BLOCK_N1": 128,
            "BLOCK_M2": 128,
            "BLOCK_N2": 32,
            "BLK_SLICE_FACTOR": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 1,
        },
    }


def flash_attn_onekernel_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: torch.Tensor,
    sm_scale: float,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: int | None = 0,
    philox_offset: int | None = 0,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
    descale_v: torch.Tensor | None = None,
    descale_do: torch.Tensor | None = None,
    USE_INT64_STRIDES: bool | None = False,
    sink: torch.Tensor | None = None,
    dsink: torch.Tensor | None = None,
    config: dict[str, any] | None = None,
    sliding_window: int = 0,
    visibility=None,
):
    """
    Flash Attention one-kernel backward pass with positional encoding support.
    Computes dQ, dK, dV in separate passes without atomics. Supports Q/K head dimensions
    larger than V for positional encoding.

    Args:
        do (torch.Tensor): Output gradient. Shape (batch, seqlen_q, num_q_heads, v_head_dim)
            or (total_tokens, num_q_heads, v_head_dim) for varlen.
        q (torch.Tensor): Query tensor with shape (batch, seqlen_q, num_q_heads, qk_head_dim).
            qk_head_dim may be larger than v_head_dim for positional encoding.
        k (torch.Tensor): Key tensor with shape (batch, seqlen_k, num_k_heads, qk_head_dim).
        v (torch.Tensor): Value tensor with shape (batch, seqlen_k, num_k_heads, v_head_dim).
        o (torch.Tensor): Output from forward pass with same shape as do.
        softmax_lse (torch.Tensor): Log-sum-exp from forward pass with shape
            (batch, num_q_heads, seqlen_q) or (total_tokens, num_q_heads) for varlen.
        dq (torch.Tensor): Pre-allocated query gradient with same shape as q.
        dk (torch.Tensor): Pre-allocated key gradient with same shape as k.
        dv (torch.Tensor): Pre-allocated value gradient with same shape as v.
        dbias (torch.Tensor): Bias gradient (not supported, must be None).
        sm_scale (float): Softmax scale, typically 1/sqrt(head_dim).
        alibi_slopes (Optional[torch.Tensor]): ALiBi position bias slopes with shape (num_q_heads,).
        causal (bool): Apply causal masking.
        cu_seqlens_q (Optional[torch.Tensor]): Cumulative sequence lengths for query with shape
            (batch + 1,). Enables variable-length mode.
        cu_seqlens_k (Optional[torch.Tensor]): Cumulative sequence lengths for key with shape
            (batch + 1,).
        max_seqlen_q (int): Maximum query sequence length in batch.
        max_seqlen_k (int): Maximum key sequence length in batch.
        dropout_p (float): Dropout probability. 0.0 disables dropout.
        philox_seed (Optional[int]): Random seed for dropout.
        philox_offset (Optional[int]): Random offset for dropout.
        descale_q (Optional[torch.Tensor]): FP8 descaling factor for q.
        descale_k (Optional[torch.Tensor]): FP8 descaling factor for k.
        descale_v (Optional[torch.Tensor]): FP8 descaling factor for v.
        descale_do (Optional[torch.Tensor]): FP8 descaling factor for do.
        USE_INT64_STRIDES (Optional[bool]): Use 64-bit stride indexing for large tensors.
        sink (Optional[torch.Tensor]): Attention sink scores (one per Q head). Shape (num_q_heads,).
        dsink (Optional[torch.Tensor]): Pre-allocated sink gradient with same shape as sink.
        config (Optional[Dict[str, any]]): Kernel tuning parameters (preprocess_kernel,
            onekernel, onekernel_pe).

    Returns:
        torch.Tensor: Delta tensor (element-wise product of do and o) with shape matching softmax_lse.
    """
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    IS_FP8 = False
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = (
            descale_q.stride(0),
            descale_k.stride(0),
            descale_v.stride(0),
            descale_do.stride(0),
        )
    else:
        FP8_MAX = None
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = stride_descale_do_z = None
        descale_strides = (
            stride_descale_q_z,
            stride_descale_k_z,
            stride_descale_v_z,
            stride_descale_do_z,
        )

    IS_VARLEN = cu_seqlens_q is not None

    # get strides and shape
    if IS_VARLEN:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, _seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        dv_strides = (0, dv.stride(1), dv.stride(0), dv.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v]
        batch, _seqlen_q, num_q_heads = q.shape[:-1]
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        dv_strides = (dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_D_MODEL_PE_POW2 = 0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    assert (pe_head_dim == 0 and BLOCK_D_MODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_D_MODEL_POW2 and pe_head_dim == BLOCK_D_MODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."
    assert (not IS_FP8) or (IS_FP8 and pe_head_dim == 0), "Positional encoding doesn't support FP8."

    assert (sink is None) or (sink is not None and sink.dim() == 1 and sink.shape[0] == num_q_heads), (
        "Sink must be 1D and have one element per query head."
    )
    assert (dsink is None) or (dsink is not None and dsink.dim() == 1 and dsink.shape[0] == num_q_heads), (
        "Sink gradient must be 1D and have one element per query head."
    )
    assert (sink is None) == (dsink is None), "Sink and its gradient must be both present or absent."

    # Configs
    if config is None:
        config = _get_config()

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        *o_strides,
        *do_strides,
        *delta_strides,
        descale_strides[3],
        cu_seqlens_q,
        max_seqlen_q,
        descale_do,
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=v_head_dim,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8,
        enable_fp_fusion=True,
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    seqlen = max(max_seqlen_q, max_seqlen_k)

    # "onekernel_pe" is for Positional Encoding (PE) causal case, it's going to be
    # used if present. Otherwise, fallback to default "onekernel" config.
    config_onekernel = (
        config["onekernel_pe"]
        if (pe_head_dim > 0 and causal and "onekernel_pe" in config)
        else config["onekernel"]
    )
    grid = (
        num_k_heads,
        triton.cdiv(seqlen, config_onekernel["BLOCK_N1"]),
        batch,
    )

    # Visibility intervals replace causal/window masking; the non-causal kernel walks every block
    # and the interval test in the inner loops zeroes what a row may not see.
    vis_kwargs = {}
    if visibility is not None:
        assert IS_VARLEN, "visibility masks use packed (varlen) layouts"
        causal = False
        sliding_window = 0
        block_n1 = config["onekernel"]["BLOCK_N1"]
        num_kv_tiles = triton.cdiv(max_seqlen_k, block_n1)
        q_first = torch.empty((batch, num_kv_tiles), dtype=torch.int32, device=q.device)
        q_last = torch.empty_like(q_first)
        vis_kv_tile_q_range_kernel[(batch, num_kv_tiles)](
            visibility[0],
            visibility[1],
            cu_seqlens_q,
            cu_seqlens_k,
            q_first,
            q_last,
            num_kv_tiles,
            BLOCK_N=block_n1,
            BLOCK_Q=1024,
        )
        vis_kwargs = dict(
            VisLo=visibility[0],
            VisHi=visibility[1],
            HAS_VIS=True,
            VisQMin=q_first,
            VisQMax=q_last,
            vis_num_kv_tiles=num_kv_tiles,
        )

    if causal:
        bwd_kernel_causal[grid](
            q,
            k,
            v,
            sink,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            dsink,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            HEAD_DIM=BLOCK_D_MODEL_POW2,
            ACTUAL_HEAD_DIM=v_head_dim,
            PE_HEAD_DIM=pe_head_dim,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            ENABLE_SINK=sink is not None,
            SLIDING_WINDOW=sliding_window,
            **config_onekernel,
            enable_fp_fusion=True,
        )
    else:
        bwd_kernel_noncausal[grid](
            q,
            k,
            v,
            sink,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            dsink,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            descale_q,
            descale_k,
            descale_v,
            descale_do,
            HEAD_DIM=BLOCK_D_MODEL_POW2,
            ACTUAL_HEAD_DIM=v_head_dim,
            PE_HEAD_DIM=pe_head_dim,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            ENABLE_SINK=sink is not None,
            SLIDING_WINDOW=sliding_window,
            **vis_kwargs,
            **config_onekernel,
            enable_fp_fusion=True,
        )

    return delta
