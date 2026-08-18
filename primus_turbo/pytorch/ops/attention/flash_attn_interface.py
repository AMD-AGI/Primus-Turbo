###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import (
    attention_aiter_backward_impl,
    attention_aiter_forward_impl,
    attention_aiter_varlen_backward_impl,
    attention_aiter_varlen_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
    flash_attn_varlen_flydsl_backward_impl,
    flash_attn_varlen_flydsl_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_hipkittens_impl import (
    flash_attn_sbhd_hipkittens_backward_impl,
    flash_attn_sbhd_hipkittens_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    _sbhd_layout,
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.attention.attention_utils import (
    _infer_qkv_format,
    _resolve_is_v3_atomic_fp32_from_env,
    block_scaling_node,
    get_p_scale,
)

# Flash-attn selects its backend on the bf16/fp16/fp32 precision bucket.
_ATTN_PRECISION = PrecisionType.BF16_FP16_FP32

__all__ = ["flash_attn_func", "flash_attn_fp8_func", "flash_attn_varlen_func"]


def _flash_attn_grads(dq, dk, dv, dbias, dsink):
    """One gradient per FlashAttnFunc.forward argument; only these five ever take one."""
    return (dq, dk, dv) + (None,) * 4 + (dbias,) + (None,) * 6 + (dsink, None, None)


def _flash_attn_varlen_grads(dq, dk, dv, dsink):
    """Same, for FlashAttnVarlenFunc.forward."""
    return (dq, dk, dv) + (None,) * 14 + (dsink, None)


def _any_requires_grad(*tensors) -> bool:
    """A sink is a learned parameter, so it can be the only input asking for a grad -- and
    then the backward still runs and needs everything ctx would have saved."""
    return any(t is not None and t.requires_grad for t in tensors)


class FlashAttnFunc(torch.autograd.Function):
    """Dense flash attention; ``backend`` picks the implementation and ctx carries it so the
    backward takes the same one. q/k/v arrive ``[b, s, h, d]``-shaped, so FlyDSL -- the one
    sbhd-native backend -- takes their ``[s, b, h, d]`` permute (the original tensor, since
    only sbhd storage is eligible) and permutes its results back."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        how_v3_bf16_cvt: Optional[int] = 1,
        sink: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = "bshd",
        backend: BackendType = BackendType.AITER,
    ):
        ctx.backend = backend
        if backend == BackendType.HIPKITTENS:
            # Same sbhd precondition as FlyDSL below, and for the same reason.
            assert _sbhd_layout(
                q, qkv_format
            ), f"hipkittens dense attention is sbhd only, got {qkv_format}"
            q_s, k_s, v_s = (t.permute(1, 0, 2, 3) for t in (q, k, v))
            out_s, lse = flash_attn_sbhd_hipkittens_forward_impl(
                q_s,
                k_s,
                v_s,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                return_lse=True,
            )
            if is_grad_enabled and _any_requires_grad(q, k, v, sink):
                ctx.save_for_backward(q_s, k_s, v_s, out_s, lse)
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
            out = out_s.permute(1, 0, 2, 3)
            # These kernels already return lse as [B, Hq, 1, Sq]; drop the singleton axis so it
            # matches the [B, Hq, Sq] every other backend hands back.
            return (out, lse.squeeze(2)) if return_lse else out

        if backend == BackendType.FLYDSL:
            # Only sbhd bytes make the [s,b,h,d] view a relabel rather than a reinterpretation.
            # The dispatcher checked that before naming this backend, but apply() can be called
            # straight, and then reading bshd bytes as sbhd would just return the wrong answer.
            assert _sbhd_layout(q, qkv_format), f"flydsl dense attention is sbhd only, got {qkv_format}"
            q_s, k_s, v_s = (t.permute(1, 0, 2, 3) for t in (q, k, v))
            out_s, lse = flash_attn_sbhd_flydsl_forward_impl(
                q_s,
                k_s,
                v_s,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                return_lse=True,
                sink=sink,
            )
            B, Sq, Hq = q.shape[0], q.shape[1], q.shape[2]
            if is_grad_enabled and _any_requires_grad(q, k, v, sink):
                ctx.save_for_backward(q_s, k_s, v_s, out_s, lse)
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
                ctx.sink = sink
                ctx.lse_shape = (B, Sq, Hq)
            out = out_s.permute(1, 0, 2, 3)
            # kernel LSE is [B*Sq, Hq] (batch-major) -> [B, Hq, Sq]
            return (out, lse.view(B, Sq, Hq).permute(0, 2, 1)) if return_lse else out

        assert not (deterministic and sink is not None), (
            "deterministic and sink cannot be enabled together currently; "
            "please set deterministic=False or sink=None"
        )
        # MI355 (gfx950): better perf when is_v3_atomic_fp32=False
        # Controlled by env var PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32
        is_v3_atomic_fp32 = _resolve_is_v3_atomic_fp32_from_env()

        # Avoid aiter print warning when how_v3_bf16_cvt!=0 in gfx950.
        if get_device_compute_capability() >= (9, 5):
            how_v3_bf16_cvt = 0

        is_grad = is_grad_enabled and _any_requires_grad(q, k, v, sink)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        head_size_q_og = q.size(3)
        head_size_v_og = v.size(3)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = attention_aiter_forward_impl(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=return_softmax and dropout_p > 0,
            max_seqlen_q=q.size(1),
            max_seqlen_k=k.size(1),
            sink=sink,
            qkv_format=qkv_format,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_q_og = head_size_q_og
            ctx.head_size_v_og = head_size_v_og
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
            ctx.sink = sink
            ctx.qkv_format = qkv_format

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.backend == BackendType.HIPKITTENS:
            q_s, k_s, v_s, out_s, lse = ctx.saved_tensors
            dq, dk, dv = flash_attn_sbhd_hipkittens_backward_impl(
                dout.permute(1, 0, 2, 3).contiguous(),
                q_s,
                k_s,
                v_s,
                out_s,
                lse,  # already [B, Hq, 1, Sq], which is the layout the backward wants
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                window_size=ctx.window_size,
            )
            dq, dk, dv = (g.permute(1, 0, 2, 3) for g in (dq, dk, dv))
            return _flash_attn_grads(dq, dk, dv, None, None)

        if ctx.backend == BackendType.FLYDSL:
            q_s, k_s, v_s, out_s, lse = ctx.saved_tensors
            B, Sq, Hq = ctx.lse_shape
            grads = flash_attn_sbhd_flydsl_backward_impl(
                dout.permute(1, 0, 2, 3),
                q_s,
                k_s,
                v_s,
                out_s,
                # [B, Hq, Sq], left as the view: the backward's -log2e prescale folds this
                # transpose into its own pass, and materialising it here both pays for a
                # torch transpose and puts that kernel off its stride fast path.
                lse.view(B, Sq, Hq).permute(0, 2, 1),
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                window_size=ctx.window_size,
                sink=ctx.sink,
            )
            dq, dk, dv = (g.permute(1, 0, 2, 3) for g in grads[:3])
            # backward returns (dq,dk,dv), plus dsink when a sink was given.
            return _flash_attn_grads(dq, dk, dv, None, grads[3] if len(grads) > 3 else None)

        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = ctx.head_size_v_og

        q, k, v, out_padded, softmax_lse, rng_state = ctx.saved_tensors
        qkv_format = ctx.qkv_format

        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])

        # dk/dv span the KEY sequence, which is only q's on a square shape.
        batch_size, seq_len_q, num_heads_q, head_dim_qk = q.size()
        _, seq_len_kv, num_heads_k, head_dim_k = k.size()
        _, _, num_heads_v, head_dim_v = v.size()

        if qkv_format == "sbhd":
            dq = torch.ones(
                (seq_len_q, batch_size, num_heads_q, head_dim_qk), dtype=q.dtype, device=q.device
            ).permute(1, 0, 2, 3)
            dk = torch.empty(
                (seq_len_kv, batch_size, num_heads_k, head_dim_k), dtype=k.dtype, device=k.device
            ).permute(1, 0, 2, 3)
            dv_padded = torch.empty(
                (seq_len_kv, batch_size, num_heads_v, head_dim_v), dtype=v.dtype, device=v.device
            ).permute(1, 0, 2, 3)
        elif qkv_format == "bhsd":
            dq = torch.ones(
                (batch_size, num_heads_q, seq_len_q, head_dim_qk), dtype=q.dtype, device=q.device
            ).permute(0, 2, 1, 3)
            dk = torch.empty(
                (batch_size, num_heads_k, seq_len_kv, head_dim_k), dtype=k.dtype, device=k.device
            ).permute(0, 2, 1, 3)
            dv_padded = torch.empty(
                (batch_size, num_heads_v, seq_len_kv, head_dim_v), dtype=v.dtype, device=v.device
            ).permute(0, 2, 1, 3)
        else:
            dq = torch.ones((batch_size, seq_len_q, num_heads_q, head_dim_qk), dtype=q.dtype, device=q.device)
            dk = torch.empty(
                (batch_size, seq_len_kv, num_heads_k, head_dim_k), dtype=k.dtype, device=k.device
            )
            dv_padded = torch.empty(
                (batch_size, seq_len_kv, num_heads_v, head_dim_v), dtype=v.dtype, device=v.device
            )
        dbias = torch.empty_like(ctx.bias) if ctx.bias is not None else None
        dsink = torch.zeros_like(ctx.sink, dtype=torch.float32) if ctx.sink is not None else None

        _ = attention_aiter_backward_impl(
            dout=dout_padded,
            q=q,
            k=k,
            v=v,
            out=out_padded,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv_padded,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=int(ctx.window_size[0]),
            window_size_right=int(ctx.window_size[1]),
            bias=ctx.bias,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
            is_v3_atomic_fp32=ctx.is_v3_atomic_fp32,
            how_v3_bf16_cvt=ctx.how_v3_bf16_cvt,
            dbias=dbias,
            dsink=dsink,
            sink=ctx.sink,
            qkv_format=qkv_format,
        )

        dq = dq[..., :head_size_q_og]
        dk = dk[..., :head_size_q_og]
        dv = dv_padded[..., :head_size_v_og]

        return _flash_attn_grads(dq, dk, dv, dbias, dsink)


class TritonFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        use_fp8,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        q, q_descale = block_scaling_node(q, use_fp8)
        k, k_descale = block_scaling_node(k, use_fp8)
        v, v_descale = block_scaling_node(v, use_fp8)
        p_scale = get_p_scale(use_fp8)

        output, softmax_lse, exp_scores = attention_triton_forward_impl(
            q,
            k,
            v,
            p_scale,
            q_descale,
            k_descale,
            v_descale,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            bias,
            alibi_slopes,
            return_softmax,
            use_fp8,
        )

        if is_grad:
            # q, k, v should be fp8 when set use_fp8 to True
            ctx.save_for_backward(
                q, k, v, output, softmax_lse, alibi_slopes, bias, q_descale, k_descale, v_descale
            )

            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = None
            ctx.cu_seqlens_k = None
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, o, softmax_lse, alibi_slopes, bias, q_descale, k_descale, v_descale = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert do.dtype is torch.bfloat16, f"do should be bfloat16 but get {do.dtype}"

        dq, dk, dv = attention_triton_backward_impl(
            do,
            q,
            k,
            v,
            o,
            q_descale,
            k_descale,
            v_descale,
            ctx.p_scale,
            softmax_lse,
            None,
            None,
            None,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            -1,
            -1,
            alibi_slopes,
            ctx.use_fp8,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    sink: Optional[torch.Tensor] = None,
):
    """q/k/v are ``[b, s, h, d]``-shaped; an sbhd caller passes a permuted view and permutes
    the result back. aiter reads the memory layout to allocate outputs and grads matching it;
    FlyDSL is sbhd-native and takes that layout only.
    """
    qkv_format = _infer_qkv_format(q, k, v)

    backend = resolve_flash_attn_backend(
        varlen=False,
        user_backend=GlobalBackendManager.get_attn_backend(_ATTN_PRECISION),
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        sink=sink,
        qkv_format=qkv_format,
    )
    # FlyDSL emits no dropout softmax matrix; keep return_attn_probs on aiter.
    if return_attn_probs:
        backend = BackendType.AITER

    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        1,  # how_v3_bf16_cvt
        sink,
        qkv_format,
        backend,
    )


def flash_attn_fp8_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    fp8_config: Optional[Float8QuantConfig] = None,
):
    qkv_format = _infer_qkv_format(q, k, v)

    if qkv_format == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    elif qkv_format == "sbhd":
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

    # Default config: blockwise with block_size=64
    if fp8_config is None:
        fp8_config = Float8QuantConfig(
            granularity=ScalingGranularity.BLOCKWISE,
            block_size=64,
        )

    # Check if config is supported
    if fp8_config.granularity != ScalingGranularity.BLOCKWISE:
        raise ValueError(
            f"flash_attn_fp8_func only supports BLOCKWISE granularity, but got {fp8_config.granularity}"
        )
    if fp8_config.block_size != 64:
        raise ValueError(
            f"flash_attn_fp8_func only supports block_size=64, but got block_size={fp8_config.block_size}"
        )

    o = TritonFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        True,
    )

    if qkv_format == "sbhd":
        o = o.permute(1, 0, 2, 3).contiguous()
    elif qkv_format == "bhsd":
        o = o.permute(0, 2, 1, 3).contiguous()

    return o


# =============================================================================
# Varlen Attention (thd layout)
# =============================================================================


class FlashAttnVarlenFunc(torch.autograd.Function):
    """Varlen (THD) flash attention, dispatching the same way FlashAttnFunc does. Sequences
    are packed along dim 0, a layout every backend shares, so nothing here permutes."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        sink: Optional[torch.Tensor] = None,
        backend: BackendType = BackendType.AITER,
    ):
        ctx.backend = backend
        if backend == BackendType.FLYDSL:
            out, lse = flash_attn_varlen_flydsl_forward_impl(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                return_lse=True,
                sink=sink,
            )
            B = cu_seqlens_q.numel() - 1
            Hq = q.shape[1]
            Sq = q.shape[0] // B  # uniform seqlens (enforced by the flydsl eligibility gate)
            if is_grad_enabled and _any_requires_grad(q, k, v, sink):
                ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
                ctx.max_seqlen_q = max_seqlen_q
                ctx.max_seqlen_k = max_seqlen_k
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
                ctx.sink = sink
            # kernel LSE is [B*Sq, Hq] (batch-major) -> [B, Hq, Sq]
            return (out, lse.view(B, Sq, Hq).permute(0, 2, 1)) if return_lse else out

        is_v3_atomic_fp32 = _resolve_is_v3_atomic_fp32_from_env()
        # Avoid aiter print warning when how_v3_bf16_cvt!=0 in gfx950.
        how_v3_bf16_cvt = 0 if get_device_compute_capability() >= (9, 5) else 1

        is_grad = is_grad_enabled and _any_requires_grad(q, k, v, sink)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        head_size_q_og = q.size(-1)
        head_size_v_og = v.size(-1)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = attention_aiter_varlen_forward_impl(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=return_softmax and dropout_p > 0,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_q_og = head_size_q_og
            ctx.head_size_v_og = head_size_v_og
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.backend == BackendType.FLYDSL:
            q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
            grads = flash_attn_varlen_flydsl_backward_impl(
                dout,
                q,
                k,
                v,
                out,
                lse,  # packed [total_q, Hq]; the impl transposes internally for the uniform path
                cu_seqlens_q,
                cu_seqlens_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                window_size=ctx.window_size,
                sink=ctx.sink,
            )
            # backward returns (dq,dk,dv), plus dsink when a sink was given.
            return _flash_attn_varlen_grads(*grads[:3], grads[3] if len(grads) > 3 else None)

        q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = ctx.head_size_v_og

        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv_padded = torch.empty_like(v)

        attention_aiter_varlen_backward_impl(
            dout=dout_padded,
            q=q,
            k=k,
            v=v,
            out=out_padded,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv_padded,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=int(ctx.window_size[0]),
            window_size_right=int(ctx.window_size[1]),
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
            is_v3_atomic_fp32=ctx.is_v3_atomic_fp32,
            how_v3_bf16_cvt=ctx.how_v3_bf16_cvt,
        )

        dq = dq[..., :head_size_q_og]
        dk = dk[..., :head_size_q_og]
        dv = dv_padded[..., :head_size_v_og]

        return _flash_attn_varlen_grads(dq, dk, dv, None)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    sink: Optional[torch.Tensor] = None,
):
    """Variable-length flash attention (THD layout).

    Mirrors flash-attention's `flash_attn_varlen_func` API. Sequences are
    packed back-to-back along dim 0.

    Arguments:
        q: (total_q, nheads_q, headdim_q)
        k: (total_k, nheads_k, headdim_q)
        v: (total_k, nheads_k, headdim_v)
        cu_seqlens_q: (batch + 1,) int32 cumulative query lengths
        cu_seqlens_k: (batch + 1,) int32 cumulative key lengths
        max_seqlen_q: maximum query sequence length in the batch
        max_seqlen_k: maximum key sequence length in the batch
        dropout_p: dropout probability (set 0.0 during eval).
        softmax_scale: QK^T scale; defaults to 1 / sqrt(headdim_q).
        causal: bottom-right aligned causal mask.
        window_size: (left, right) sliding window. (-1, -1) = full attention.
        bias: optional attention bias.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32.
        deterministic: use the deterministic backward (slower, more memory).
        return_lse: also return softmax_lse of shape (nheads, total_q).
        return_attn_probs: testing-only; return the dropout-encoded softmax output.
    """
    backend = resolve_flash_attn_backend(
        varlen=True,
        user_backend=GlobalBackendManager.get_attn_backend(_ATTN_PRECISION),
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        sink=sink,
    )
    # FlyDSL emits no dropout softmax matrix; keep return_attn_probs on aiter.
    if return_attn_probs:
        backend = BackendType.AITER

    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        sink,
        backend,
    )
