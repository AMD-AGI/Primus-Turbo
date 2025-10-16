###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch._dynamo.config

from primus_turbo.pytorch.modules import TurboAttention
from tests.pytorch.ref.attention_ref import AttnConfig, TurboAttentionRef
from tests.test_utils import compute_snr

test_cases = [
    # MHA
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    # GQA
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # 192/128
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=192, head_dim_v=128),
    # 192/192
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=192, head_dim_v=192),
]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seq", [4096])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend_type", ["ck", "triton"])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_attention_fp16(batch, seq, config, causal, backend_type, enable_torch_compile):
    device = "cuda:0"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        seq,
        seq,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    if head_dim_qk == 192 and head_dim_v == 128 and seq > 4096:
        pytest.skip()

    # print(f"\n ", seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v, causal, backend_type, enable_torch_compile)

    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)

    torch.cuda.synchronize()
    primus_attention_ck = TurboAttention(
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=False,
        backend_type=backend_type,
    )
    attention_ref = TurboAttentionRef(softmax_scale=sm_scale, causal=causal)
    if enable_torch_compile:
        torch._dynamo.reset()
        primus_attention_ck = torch.compile(primus_attention_ck, fullgraph=True, mode="max-autotune")
    torch.cuda.synchronize()

    # Test
    out = primus_attention_ck(query, key, value)
    out_ref = attention_ref(query_ref, key_ref, value_ref)
    out_snr = compute_snr(out_ref, out)
    assert out_snr > 20, "out_snr too low"

    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    out_ref.backward(grad_output)
    query_grad_snr = compute_snr(query.grad, query_ref.grad)
    key_grad_snr = compute_snr(key.grad, key_ref.grad)
    value_grad_snr = compute_snr(value.grad, value_ref.grad)
    assert query_grad_snr > 15, "query_grad_snr too low"
    assert key_grad_snr > 15, "key_grad_snr too low"
    assert value_grad_snr > 15, "value_grad_snr too low"
    torch.cuda.synchronize()


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend_type", ["triton"])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_attention_fp8(batch, config, causal, backend_type, enable_torch_compile):

    device = "cuda:0"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)

    torch.cuda.synchronize()
    primus_attention_triton = TurboAttention(
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=True,
        backend_type=backend_type,
    )
    attention_ref = TurboAttentionRef(softmax_scale=sm_scale, causal=causal)
    if enable_torch_compile:
        torch._dynamo.reset()
        primus_attention_triton = torch.compile(primus_attention_triton, fullgraph=True, mode="max-autotune")
    torch.cuda.synchronize()

    # Test
    output = primus_attention_triton(query, key, value)
    out_ref = attention_ref(query_ref, key_ref, value_ref)
    out_snr = compute_snr(out_ref, output)
    assert out_snr > 20, "out_snr too low"

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    out_ref.backward(grad_output)
    query_grad_snr = compute_snr(query.grad, query_ref.grad)
    key_grad_snr = compute_snr(key.grad, key_ref.grad)
    value_grad_snr = compute_snr(value.grad, value_ref.grad)
    assert query_grad_snr > 15, "query_grad_snr too low"
    assert key_grad_snr > 15, "key_grad_snr too low"
    assert value_grad_snr > 15, "value_grad_snr too low"
