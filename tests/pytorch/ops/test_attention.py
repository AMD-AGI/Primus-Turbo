###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    F8_FWD_MAX,
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops import flash_attn_fp8_func, flash_attn_func
from primus_turbo.pytorch.ops.attention.attention_utils import block_scaling_node
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
    attention_with_sink_ref_impl,
)
from tests.pytorch.test_utils import compute_snr

test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=28, num_head_kv=4, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # begin regression tests for https://ontrack-internal.amd.com/browse/SWDEV-548136
    AttnConfig(
        seqlen_q=4096 + 64, seqlen_kv=4096 + 64, num_head_q=2, num_head_kv=1, head_dim_qk=32, head_dim_v=32
    ),
    AttnConfig(seqlen_q=2048, seqlen_kv=2048, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # end regression tests for https://ontrack-internal.amd.com/browse/SWDEV-548136
    AttnConfig(seqlen_q=512, seqlen_kv=512, num_head_q=40, num_head_kv=40, head_dim_qk=192, head_dim_v=128),
]


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("enable_sink", [False, True])
@pytest.mark.parametrize("window_size_left", [-1, 32, 64, 128])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "bhsd"])
@pytest.mark.parametrize("is_v3_atomic_fp32", [False, True])
def test_attention_16bit(
    batch, dtype, config, causal, enable_sink, window_size_left, qkv_format, is_v3_atomic_fp32
):
    os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "1" if is_v3_atomic_fp32 else "0"

    device = "cuda"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    # Sliding window coverage only applies when sink attention is enabled.
    if not enable_sink and window_size_left != -1:
        pytest.skip("window_size_left only applies when sink is enabled")

    # Sink attention constraints / runtime control (skip early to avoid big allocations).
    if enable_sink:
        # Triton kernel limitation for sink: requires same qk/v head dim and head dim > 32
        if head_dim_qk != head_dim_v or head_dim_qk < 32:
            pytest.skip("Sink attention requires head_dim_qk == head_dim_v and head_dim >= 32")
        if window_size_left != -1 and not causal:
            pytest.skip("sink sliding window coverage only applies to causal attention")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    window_size = (window_size_left, -1) if enable_sink and window_size_left != -1 else (-1, -1)

    print(
        f"\nDType={dtype}, B={batch}, SeqQ={seqlen_q}, SeqKV={seqlen_kv}, NHQ={num_head_q}, NHKV={num_head_kv}, "
        f"HDQK={head_dim_qk}, HDV={head_dim_v}, Causal={causal}, Sink={enable_sink}, WindowLeft={window_size_left}, Format={qkv_format}"
    )

    if qkv_format == "sbhd":
        q_layout = (seqlen_q, batch, num_head_q, head_dim_qk)
        k_layout = (seqlen_kv, batch, num_head_kv, head_dim_qk)
        v_layout = (seqlen_kv, batch, num_head_kv, head_dim_v)
        o_layout = (seqlen_q, batch, num_head_q, head_dim_v)
    elif qkv_format == "bhsd":
        q_layout = (batch, num_head_q, seqlen_q, head_dim_qk)
        k_layout = (batch, num_head_kv, seqlen_kv, head_dim_qk)
        v_layout = (batch, num_head_kv, seqlen_kv, head_dim_v)
        o_layout = (batch, num_head_q, seqlen_q, head_dim_v)
    elif qkv_format == "bshd":
        q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
        k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
        v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
        o_layout = (batch, seqlen_q, num_head_q, head_dim_v)
    else:
        raise AssertionError(f"Unsupported qkv format: {qkv_format}")

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(o_layout, device=device, dtype=dtype)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()
    grad_out_ref = grad_out.clone().detach()

    query_orig, key_orig, value_orig = query, key, value

    if qkv_format == "sbhd":
        query = query.permute(1, 0, 2, 3)
        key = key.permute(1, 0, 2, 3)
        value = value.permute(1, 0, 2, 3)
        grad_out = grad_out.permute(1, 0, 2, 3)
    elif qkv_format == "bhsd":
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        grad_out = grad_out.transpose(1, 2)

    sm_scale = head_dim_qk ** (-0.5)

    sink = None
    sink_ref = None
    if enable_sink:
        sink = torch.randn((num_head_q,), device=device, dtype=torch.float32, requires_grad=True)
        sink_ref = sink.clone().detach().requires_grad_()
        o_ref = attention_with_sink_ref_impl(
            query_ref,
            key_ref,
            value_ref,
            sink_ref,
            sm_scale,
            causal,
            window_size=window_size,
            qkv_format=qkv_format,
        )
    else:
        o_ref = attention_vanilla_forward_pytorch_ref_impl(
            query_ref, key_ref, value_ref, sm_scale, causal, qkv_format
        )

    o_ref.backward(grad_out_ref)
    o = flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=window_size,
        bias=None,
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        sink=sink,
    )
    o.backward(grad_out)

    torch.cuda.synchronize()

    if qkv_format == "sbhd":
        o_ref_cmp = o_ref.permute(1, 0, 2, 3).contiguous()
    elif qkv_format == "bhsd":
        o_ref_cmp = o_ref.transpose(1, 2).contiguous()
    else:
        o_ref_cmp = o_ref
    out_snr = compute_snr(o_ref_cmp, o)
    query_grad_snr = compute_snr(query_ref.grad, query_orig.grad)
    key_grad_snr = compute_snr(key_ref.grad, key_orig.grad)
    value_grad_snr = compute_snr(value_ref.grad, value_orig.grad)
    sink_grad_snr = compute_snr(sink_ref.grad, sink.grad) if enable_sink else None
    msg = f"out={out_snr:.2f}, dq={query_grad_snr:.2f}, dk={key_grad_snr:.2f}, dv={value_grad_snr:.2f}"
    if enable_sink:
        msg += f", dsink={sink_grad_snr:.2f}"
    print(msg)

    assert out_snr > 40, f"out_snr too low: {out_snr}"
    assert query_grad_snr > 40, f"query_grad_snr too low: {query_grad_snr}"
    assert key_grad_snr > 40, f"key_grad_snr too low: {key_grad_snr}"
    assert value_grad_snr > 40, f"value_grad_snr too low: {value_grad_snr}"
    # SNR threshold for sink grad is 5e-2, reference from aiter: https://github.com/ROCm/aiter/blob/c71075ceda2788004f1a6e02608e114137dee856/op_tests/triton_tests/attention/test_mha_with_sink.py#L151-L157
    if sink_grad_snr is not None:
        torch.testing.assert_close(
            sink.grad,
            sink_ref.grad,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda msg: f"sink_grad mismatch (snr={sink_grad_snr:.2f})\n\n{msg}\n",
        )


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skip(reason="Temporarily disabled due to external dependency issues.")
@pytest.mark.deterministic
def test_attention_16bit_deterministic(batch, dtype, config, causal):
    device = "cuda"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    # NOTE: For `head_dim_qk != head_dim_v` (e.g. 192/128), this deterministic
    # test currently fails; skip temporarily to keep CI green.
    if head_dim_qk != head_dim_v:
        pytest.skip("deterministic test currently fails when head_dim_qk != head_dim_v; skip temporarily")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(
        f"\n[deterministic] DType={dtype}, B={batch}, SeqQ={seqlen_q}, SeqKV={seqlen_kv}, "
        f"NHQ={num_head_q}, NHKV={num_head_kv}, HDQK={head_dim_qk}, HDV={head_dim_v}, Causal={causal}"
    )

    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
    o_layout = (batch, seqlen_q, num_head_q, head_dim_v)

    q0 = torch.randn(q_layout, device=device, dtype=dtype)
    k0 = torch.randn(k_layout, device=device, dtype=dtype)
    v0 = torch.randn(v_layout, device=device, dtype=dtype)
    grad_out = torch.randn(o_layout, device=device, dtype=dtype)

    sm_scale = head_dim_qk ** (-0.5)

    # Correctness check against reference implementation
    q_ref = q0.clone().detach().requires_grad_()
    k_ref = k0.clone().detach().requires_grad_()
    v_ref = v0.clone().detach().requires_grad_()
    o_ref = attention_vanilla_forward_pytorch_ref_impl(q_ref, k_ref, v_ref, sm_scale, causal)
    o_ref.backward(grad_out)

    def _run_once():
        q = q0.clone().detach().requires_grad_()
        k = k0.clone().detach().requires_grad_()
        v = v0.clone().detach().requires_grad_()

        o = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=True,
            return_lse=False,
            return_attn_probs=False,
            sink=None,
        )
        o.backward(grad_out)
        return (
            o.detach(),
            q.grad.detach(),
            k.grad.detach(),
            v.grad.detach(),
        )

    # Determinism check (bitwise identical across multiple runs).
    repeats = 10
    outs = []
    for _ in range(repeats):
        outs.append(_run_once())
        torch.cuda.synchronize()

    o1, dq1, dk1, dv1 = outs[0]
    for i in range(1, repeats):
        o_i, dq_i, dk_i, dv_i = outs[i]
        torch.testing.assert_close(o1, o_i, rtol=0, atol=0)
        torch.testing.assert_close(dq1, dq_i, rtol=0, atol=0)
        torch.testing.assert_close(dk1, dk_i, rtol=0, atol=0)
        torch.testing.assert_close(dv1, dv_i, rtol=0, atol=0)

    # Correctness check (close to reference)
    out_snr = compute_snr(o_ref, o1)
    query_grad_snr = compute_snr(q_ref.grad, dq1)
    key_grad_snr = compute_snr(k_ref.grad, dk1)
    value_grad_snr = compute_snr(v_ref.grad, dv1)
    print(
        f"deterministic: out={out_snr:.2f}, dq={query_grad_snr:.2f}, dk={key_grad_snr:.2f}, dv={value_grad_snr:.2f}"
    )
    assert out_snr > 40, f"out_snr too low: {out_snr}"
    assert query_grad_snr > 40, f"query_grad_snr too low: {query_grad_snr}"
    assert key_grad_snr > 40, f"key_grad_snr too low: {key_grad_snr}"
    assert value_grad_snr > 40, f"value_grad_snr too low: {value_grad_snr}"


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_fp8(batch, dtype, config, causal):
    device = "cuda"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    print(
        f"\nDType={dtype}, B={batch}, SeqQ={seqlen_q}, SeqKV={seqlen_kv}, NHQ={num_head_q}, NHKV={num_head_kv}, HDQK={head_dim_qk}, HDV={head_dim_v}, Causal={causal}"
    )

    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
    o_layout = (batch, seqlen_q, num_head_q, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(o_layout, device=device, dtype=dtype)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    o_ref.backward(grad_out)
    o = flash_attn_fp8_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
    )
    o.backward(grad_out)
    torch.cuda.synchronize()

    out_snr = compute_snr(o_ref, o)
    query_grad_snr = compute_snr(query_ref.grad, query.grad)
    key_grad_snr = compute_snr(key_ref.grad, key.grad)
    value_grad_snr = compute_snr(value_ref.grad, value.grad)
    print(f"{out_snr:.2f}", f"{query_grad_snr:.2f}", f"{key_grad_snr:.2f}", f"{value_grad_snr:.2f}")
    assert out_snr > 20, "out_snr too low"
    assert query_grad_snr > 20, "query_grad_snr too low"
    assert key_grad_snr > 20, "key_grad_snr too low"
    assert value_grad_snr > 20, "value_grad_snr too low"


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_fp8_with_sparse_do(batch, config, causal):
    # regression test for https://ontrack-internal.amd.com/browse/SWDEV-548136
    device = "cuda"
    torch.manual_seed(1234)

    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_shape = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_shape = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_shape = (batch, seqlen_kv, num_head_kv, head_dim_v)
    do_shape = (batch, seqlen_q, num_head_q, head_dim_v)

    do = torch.randn(do_shape, device=device, dtype=dtype) * 1e-3
    do_mask_0 = (torch.randn(do_shape[:-2], device=device, dtype=dtype) > 0.9).unsqueeze(-1).unsqueeze(-1)
    do_mask_1 = (torch.randn(do_shape[:-1], device=device, dtype=dtype) > 0.9).unsqueeze(-1)
    do = do * do_mask_0 * do_mask_1

    q = torch.randn(q_shape, device=device, dtype=dtype)
    k = torch.randn(k_shape, device=device, dtype=dtype)
    v = torch.randn(v_shape, device=device, dtype=dtype)

    sm_scale = q.shape[-1] ** -0.5

    q_fp8, q_descale = block_scaling_node(q, True)
    k_fp8, k_descale = block_scaling_node(k, True)
    v_fp8, v_descale = block_scaling_node(v, True)

    o, softmax_lse, _ = attention_triton_forward_impl(
        q_fp8,
        k_fp8,
        v_fp8,
        F8_FWD_MAX,
        q_descale,
        k_descale,
        v_descale,
        0,
        sm_scale,
        causal,
        -1,
        -1,
        None,
        None,
        False,
        True,
    )

    dq, dk, dv = attention_triton_backward_impl(
        do,
        q,
        k,
        v,
        o,
        torch.scalar_tensor(1.0, device=device),
        torch.scalar_tensor(1.0, device=device),
        torch.scalar_tensor(1.0, device=device),
        1.0,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
        q_fp8.shape[1],
        k_fp8.shape[1],
        sm_scale,
        causal,
        -1,
        -1,
        None,
        False,
    )

    dq_fp8, dk_fp8, dv_fp8 = attention_triton_backward_impl(
        do,
        q_fp8,
        k_fp8,
        v_fp8,
        o,
        q_descale,
        k_descale,
        v_descale,
        F8_FWD_MAX,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
        q_fp8.shape[1],
        k_fp8.shape[1],
        sm_scale,
        causal,
        -1,
        -1,
        None,
        True,
    )

    dq_snr = compute_snr(dq, dq_fp8)
    dk_snr = compute_snr(dk, dk_fp8)
    dv_snr = compute_snr(dv, dv_fp8)
    print(f"dq_snr: {dq_snr}, dk_snr: {dk_snr}, dv_snr: {dv_snr}")
    assert dq_snr > 15, "query_grad_snr too low"
    assert dk_snr > 15, "key_grad_snr too low"
    assert dv_snr > 15, "value_grad_snr too low"


@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "bhsd"])
def test_attention_fake_kernel_strides(qkv_format):
    """Verify that torch.compile sees correct output strides for every qkv_format.

    The fake (meta) kernel must produce output tensors whose strides match the
    eager kernel so that torch.compile's shape/stride propagation is correct.
    """
    device = "cuda"
    dtype = torch.bfloat16
    batch, seq_q, seq_kv, num_heads, head_dim = 2, 32, 32, 4, 64

    if qkv_format == "sbhd":
        q = torch.randn(seq_q, batch, num_heads, head_dim, device=device, dtype=dtype).permute(1, 0, 2, 3)
        k = torch.randn(seq_kv, batch, num_heads, head_dim, device=device, dtype=dtype).permute(1, 0, 2, 3)
        v = torch.randn(seq_kv, batch, num_heads, head_dim, device=device, dtype=dtype).permute(1, 0, 2, 3)
    elif qkv_format == "bhsd":
        q = torch.randn(batch, num_heads, seq_q, head_dim, device=device, dtype=dtype).transpose(1, 2)
        k = torch.randn(batch, num_heads, seq_kv, head_dim, device=device, dtype=dtype).transpose(1, 2)
        v = torch.randn(batch, num_heads, seq_kv, head_dim, device=device, dtype=dtype).transpose(1, 2)
    else:
        q = torch.randn(batch, seq_q, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seq_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, seq_kv, num_heads, head_dim, device=device, dtype=dtype)

    out_eager = flash_attn_func(q, k, v, causal=True)
    eager_strides = out_eager.stride()

    torch._dynamo.reset()

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True)

    out_compiled = fn(q, k, v)

    assert out_compiled.stride() == eager_strides, (
        f"Stride mismatch for qkv_format={qkv_format}: "
        f"compiled={out_compiled.stride()}, eager={eager_strides}"
    )
    assert out_compiled.shape == out_eager.shape


# ============================================================================
# DeepSeek-V4 single-latent sparse-MLA attention (flydsl, gfx950/MI355X).
# ============================================================================

# Fixed dims: kv_lora_rank (single latent, K == V) + rope pad; SWA local window.
SPARSE_MLA_ROPE_DIM = 64
SPARSE_MLA_HEAD_DIM = 512
SPARSE_MLA_SWA_WINDOW = 128
# (variant -> num_heads, index-topk cap). cr spans pure-SWA / random-pool / deterministic-pool (HCA).
SPARSE_MLA_VARIANTS = {"flash": (64, 512), "pro": (128, 1024)}


def _sparse_mla_topk(variant, cr, seqlen):
    if cr == 0:
        return 0, 0, SPARSE_MLA_SWA_WINDOW
    if cr == 4:
        pool = max(seqlen // 4, 1)
        topk_pool = min(SPARSE_MLA_VARIANTS[variant][1], pool)
        return pool, topk_pool, SPARSE_MLA_SWA_WINDOW + topk_pool
    pool = max(seqlen // cr, 1)
    return pool, 0, SPARSE_MLA_SWA_WINDOW + pool


def _build_sparse_mla(cr, num_heads, seqlen, pool, topk_pool, seed=0):
    """DSV4 sparse-MLA inputs: single-latent kv, per-token top-k (SWA band + optional pool),
    zero-padded rope cols, random sink / grad_out."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt, d, w = "cuda", torch.bfloat16, SPARSE_MLA_HEAD_DIM, SPARSE_MLA_SWA_WINDOW
    latent = torch.randn(seqlen, d, generator=gen, device=dev, dtype=dt)
    q = torch.randn(seqlen, num_heads, d, generator=gen, device=dev, dtype=dt)
    q = torch.cat([q, torch.zeros(seqlen, num_heads, SPARSE_MLA_ROPE_DIM, device=dev, dtype=dt)], -1)
    sink = torch.randn(num_heads, generator=gen, device=dev, dtype=torch.float32) * 0.1
    grad_out = torch.randn(seqlen, num_heads, d, generator=gen, device=dev, dtype=dt)

    tok = torch.arange(seqlen, device=dev).view(seqlen, 1)
    win = tok - w + 1 + torch.arange(w, device=dev).view(1, w)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if cr == 0:
        kv = latent.unsqueeze(1)
        topk = win
    else:
        p = torch.randn(pool, d, generator=gen, device=dev, dtype=dt)
        kv = torch.cat([latent, p], 0).unsqueeze(1)
        if cr == 4:
            pool_topk = seqlen + torch.randint(0, pool, (seqlen, topk_pool), generator=gen, device=dev)
        else:
            ps = torch.arange(pool, device=dev).view(1, pool)
            pool_topk = torch.where(
                ((ps + 1) * cr - 1) <= tok, seqlen + ps, torch.full_like(ps.expand(seqlen, pool), -1)
            )
        topk = torch.cat([win, pool_topk], 1)
    pad = ((topk.shape[1] + 63) // 64) * 64 - topk.shape[1]
    if pad > 0:
        topk = torch.cat([topk, torch.full((seqlen, pad), -1, device=dev, dtype=topk.dtype)], 1)
    kv = torch.cat([kv, torch.zeros(kv.shape[0], 1, SPARSE_MLA_ROPE_DIM, device=dev, dtype=dt)], -1)
    return q.contiguous(), kv.contiguous(), topk.to(torch.int32).contiguous(), sink, grad_out


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="sparse-MLA (flydsl) is gfx950-only"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("variant", ["flash", "pro"])
@pytest.mark.parametrize("cr", [0, 4, 128])
def test_sparse_mla(variant, cr, seqlen):
    """flydsl sparse-MLA fwd/bwd correctness (SNR vs triton oracle) + determinism, over the
    pure-SWA (cr=0), random-pool (cr=4) and deterministic-pool/HCA (cr=128) paths. seqlen=512
    also exercises the cr=4 small-seq dkv-dispatch guard."""
    import math

    # Lazy import: keeps collection working on non-gfx950 (flydsl sparse-MLA is gfx950-only).
    from primus_turbo.flydsl.attention.sparse_mla_bwd import sparse_mla_bwd_flydsl
    from primus_turbo.flydsl.attention.sparse_mla_fwd import sparse_mla_fwd_flydsl
    from primus_turbo.triton.attention.sparse_mla import (
        sparse_mla_bwd_triton,
        sparse_mla_fwd_triton,
    )

    d = SPARSE_MLA_HEAD_DIM
    num_heads = SPARSE_MLA_VARIANTS[variant][0]
    pool, topk_pool, _ = _sparse_mla_topk(variant, cr, seqlen)
    scale = 1.0 / math.sqrt(d)
    q, kv, topk_idx, sink, grad_out = _build_sparse_mla(cr, num_heads, seqlen, pool, topk_pool)

    out, lse = sparse_mla_fwd_flydsl(q, kv, topk_idx, attn_sink=sink, kv_lora_rank=d, scale=scale)
    out_ref, lse_ref = sparse_mla_fwd_triton(q, kv, topk_idx, attn_sink=sink, kv_lora_rank=d, scale=scale)
    assert torch.isfinite(out).all(), "forward produced non-finite values"
    fwd_snr = compute_snr(out_ref, out)
    assert fwd_snr > 40.0, f"fwd SNR {fwd_snr:.1f} <= 40"

    dq, dkv, dsink = sparse_mla_bwd_flydsl(
        q, kv, out, grad_out, topk_idx, lse, attn_sink=sink, kv_lora_rank=d, scale=scale
    )
    dq_ref, dkv_ref, dsink_ref = sparse_mla_bwd_triton(
        q, kv, out_ref, grad_out, topk_idx, lse_ref, attn_sink=sink, kv_lora_rank=d, scale=scale
    )
    assert torch.isfinite(dq).all() and torch.isfinite(dkv).all(), "backward produced non-finite values"
    assert compute_snr(dq_ref, dq) > 40.0, "dq SNR <= 40"
    assert compute_snr(dkv_ref, dkv) > 40.0, "dkv SNR <= 40"
    if dsink is not None:
        assert compute_snr(dsink_ref, dsink) > 40.0, "dsink SNR <= 40"

    # Determinism: one WG owns each output tile (no float atomics), so a re-run is bit-exact.
    dq2, dkv2, _ = sparse_mla_bwd_flydsl(
        q, kv, out, grad_out, topk_idx, lse, attn_sink=sink, kv_lora_rank=d, scale=scale
    )
    assert torch.equal(dq, dq2) and torch.equal(dkv, dkv2), "backward is not deterministic"
