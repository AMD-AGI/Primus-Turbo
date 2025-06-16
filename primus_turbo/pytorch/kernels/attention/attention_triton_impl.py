import math

import torch
import triton
import triton.language as tl

_torch_custom_op_wrapper = torch.library.custom_op
from typing import Optional, Tuple

from torch._library import wrap_triton

from primus_turbo.triton.attention.attention_kernel import (
    DEBUG,
    FIXED_BLOCK_M,
    FIXED_BLOCK_N,
    _bwd_kernel_dkdv,
    _bwd_kernel_dq,
    _bwd_preprocess_use_o,
    attn_fwd,
    get_padded_head_dim,
    get_shape_from_layout,
    get_strides_from_layout,
    get_tl_f8_bwd_dtype,
    is_hip,
    philox_offset,
    philox_seed,
)

fwd_torch_dtype: tl.constexpr = torch.bfloat16
bwd_torch_dtype: tl.constexpr = torch.float32


def get_f8_fwd_dtype():
    if is_hip():
        return torch.float8_e4m3fnuz
    else:
        return torch.float8_e4m3fn


def get_f8_bwd_dtype():
    if is_hip():
        return torch.float8_e5m2fnuz
    else:
        return torch.float8_e5m2


F8_FWD_MAX: tl.constexpr = torch.finfo(get_f8_fwd_dtype()).max
F8_BWD_MAX: tl.constexpr = torch.finfo(get_f8_bwd_dtype()).max


@_torch_custom_op_wrapper("primus_turbo::attention_triton_forward_impl", mutates_args=(), device_types="cuda")
def attention_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    layout = "bshd"
    cu_seqlens_q = 0
    cu_seqlens_k = 0
    max_seqlens_q = q.shape[1]
    max_seqlens_k = k.shape[1]
    return_scores = return_softmax
    use_exp2 = True
    if DEBUG:
        print()
        print("attention_forward_triton_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_scale:", softmax_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)
        print("use_fp8:", use_fp8)

    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_fp8 else q.dtype,
        requires_grad=True,
    )

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if bias is not None:
        assert bias.numel() < 2**31

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k
    )
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    grid = (triton.cdiv(max_seqlens_q, FIXED_BLOCK_M), nheads_q, batch)

    if return_scores:
        scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
        scores_scaled_shifted = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3))
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0] * 2, nheads_q), device=q.device, dtype=torch.float32)
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q * 2), device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    if use_fp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = (
            q_descale.stride(0),
            q_descale.stride(1),
            q_descale.stride(2),
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = (
            k_descale.stride(0),
            k_descale.stride(1),
            k_descale.stride(2),
        )
        padded_kscale_block_num = 1 << (stride_kdescale_h - 1).bit_length()

    else:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = None, None, None
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = None, None, None
        padded_kscale_block_num = None

    wrap_triton(attn_fwd)[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_descale,
        k_descale,
        v_scale,
        use_fp8,
        softmax_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        padded_kscale_block_num,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=False if alibi_slopes is None else True,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
    )

    return o, softmax_lse, exp_scores


@attention_triton_forward_impl.register_fake
def _attention_triton_forward_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=torch.bfloat16 if use_fp8 else q.dtype,
        requires_grad=True,
    )

    batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
    batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape

    if return_softmax:
        exp_scores = torch.zeros(
            (batch_q, nheads_q, max_seqlen_q, max_seqlen_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    softmax_lse = torch.empty((batch_q, nheads_q, max_seqlen_q * 2), device=q.device, dtype=torch.float32)

    return o, softmax_lse, exp_scores


@_torch_custom_op_wrapper(
    "primus_turbo::attention_triton_backward_impl", mutates_args=("dq", "dk", "dv"), device_types="cuda"
)
def attention_triton_backward_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse_delta: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    use_exp2 = True
    layout = "bshd"
    sequence_parallel = True
    if DEBUG:
        print("####################################################")
        print("attention_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse_delta, softmax_lse_delta.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("softmax_scale:", softmax_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)
        print("sequence_parallel:", sequence_parallel)

    # make contigious
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse_delta = softmax_lse_delta.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    )
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)
    do_strides = get_strides_from_layout(do, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    stride_doz, stride_doh, stride_dom, stride_dok = do_strides
    batch_headsize_q = batch * nheads_q
    batch_headsize_k = batch * nheads_k
    is_varlen = layout == "thd"

    assert head_size_qk >= 32 and head_size_v >= 32
    assert head_size_qk % 2 == 0 and head_size_v % 2 == 0
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    do = do.contiguous()
    # NOTE: we might need to copy the output tensor if they are not continuous or have other issues
    copy_back = {"dq": False, "dk": False, "dv": False}

    # deal with dq
    if dq is None:
        dq = torch.zeros(q.shape, device=q.device, dtype=bwd_torch_dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True

        dq.zero_()
    stride_dq_all = dq.stride()[0]

    # deal with dk, dv
    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
        dv = torch.zeros_like(v, dtype=bwd_torch_dtype)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True

        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    if DEBUG:
        print("copy_back:", copy_back)

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse_delta.is_contiguous()
    # # init delta
    # delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_lse_delta_m, stride_lse_delta_h = softmax_lse_delta.stride()
        stride_lse_delta_z = 0
    else:
        stride_lse_delta_z, stride_lse_delta_h, stride_lse_delta_m = softmax_lse_delta.stride()

    if use_fp8:
        do_fp8 = torch.empty(do.shape, dtype=get_f8_bwd_dtype(), device=q.device)
        _shape = (batch, nheads_q, triton.cdiv(max_seqlen_q, FIXED_BLOCK_M))
        do_scale = torch.empty(_shape, dtype=torch.float32, device=q.device)
        stride_descalez, stride_descaleh, stride_descalem = do_scale.stride()
        stride_qscalez, stride_qscaleh, stride_qscalem = q_scale.stride()
        stride_kscalez, stride_kscaleh, stride_kscalem = k_scale.stride()

        padded_doscale_block_num = 1 << (stride_descaleh - 1).bit_length()
        padded_qscale_block_num = 1 << (stride_qscaleh - 1).bit_length()
        padded_kscale_block_num = 1 << (stride_kscaleh - 1).bit_length()

    else:
        do_fp8 = None
        do_scale = None
        stride_descalez, stride_descaleh, stride_descalem = None, None, None
        stride_qscalez, stride_qscaleh, stride_qscalem = None, None, None
        stride_kscalez, stride_kscaleh, stride_kscalem = None, None, None
        padded_doscale_block_num, padded_qscale_block_num, padded_kscale_block_num = None, None, None

    grid_prebwd = (triton.cdiv(max_seqlen_q, FIXED_BLOCK_M), batch_headsize_q)
    wrap_triton(_bwd_preprocess_use_o)[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        softmax_lse_delta,
        use_fp8,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        BLOCK_M=FIXED_BLOCK_M,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        F8_BWD_MAX=F8_BWD_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_scale", softmax_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse_delta, softmax_lse_delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:", stride_qz, stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:", stride_kz, stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:", stride_vz, stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch)
        print("heads_q:", nheads_q)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("BLOCK_DMODEL_QK:", padded_d_model_qk)
        print("BLOCK_DMODEL_V:", padded_d_model_v)
        print("SEQUENCE_PARALLEL:", sequence_parallel)
        print("CAUSAL:", causal)
        print("USE_EXP2:", use_exp2)

    log_p_scale = math.log(p_scale)
    num_block_m = triton.cdiv(max_seqlen_q, FIXED_BLOCK_M)
    grid_bwd = (
        batch_headsize_q,
        triton.cdiv(max_seqlen_q, FIXED_BLOCK_M) if sequence_parallel else 1,
    )
    wrap_triton(_bwd_kernel_dq)[grid_bwd](
        q,
        k,
        v,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse_delta,
        stride_dq_all,
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
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        padded_doscale_block_num,
        padded_qscale_block_num,
        padded_kscale_block_num,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=F8_FWD_MAX,
    )

    if use_fp8:
        n_groups = nheads_q // nheads_k
        padded_doscale_block_num = 1 << (stride_descaleh - 1).bit_length()
        padded_qscale_block_num = 1 << (stride_qscaleh * n_groups - 1).bit_length()
        padded_kscale_block_num = 1 << (stride_kscaleh * n_groups - 1).bit_length()
    else:
        padded_doscale_block_num = None
        padded_qscale_block_num = None
        padded_kscale_block_num = None

    grid_bwd_dkdv = (
        batch_headsize_k,
        triton.cdiv(max_seqlen_k, FIXED_BLOCK_N) if sequence_parallel else 1,
    )
    wrap_triton(_bwd_kernel_dkdv)[grid_bwd_dkdv](
        q,
        k,
        v,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse_delta,
        stride_dq_all,
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
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        padded_doscale_block_num,
        padded_qscale_block_num,
        padded_kscale_block_num,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=F8_FWD_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        # print("delta:", delta, delta.shape)

    if DEBUG:
        print("####################################################")
        print("attention_prefill_backward_triton_new_impl outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        # print("delta:", delta, delta.shape)
        print("copy_back:", copy_back)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq.to(fwd_torch_dtype), dk.to(fwd_torch_dtype), dv.to(fwd_torch_dtype)


@attention_triton_backward_impl.register_fake
def _attention_triton_backward_impl_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq_out, dk_out, dv_out = (
        torch.empty_like(q, dtype=torch.bfloat16),
        torch.empty_like(k, dtype=torch.bfloat16),
        torch.empty_like(v, dtype=torch.bfloat16),
    )
    return dq_out, dk_out, dv_out
