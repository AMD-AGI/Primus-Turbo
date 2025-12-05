###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch._dynamo.config

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.modules import TurboAttention
from tests.pytorch.ref.attention_ref import AttnConfig, TurboAttentionRef
from tests.pytorch.test_utils import compute_snr

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
@pytest.mark.parametrize("seq", [4096, 8192])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_16bit(batch, seq, config, causal, enable_torch_compile, dtype):
    print(
        f"\nB={batch}, Seq={seq}, NHQ={config.num_head_q}, NHKV={config.num_head_kv}, "
        f"HDQK={config.head_dim_qk}, HDV={config.head_dim_v}, Causal={causal}, "
        f"Compile={enable_torch_compile}, DType={dtype}"
    )
    device = "cuda:0"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        seq,
        seq,
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
    primus_attention_ck = TurboAttention(
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
    )
    attention_ref = TurboAttentionRef(softmax_scale=sm_scale, causal=causal)
    if enable_torch_compile:
        torch._dynamo.reset()
        primus_attention_ck = torch.compile(primus_attention_ck, fullgraph=True, mode="max-autotune")
    torch.cuda.synchronize()

    # Test
    out = primus_attention_ck(query, key, value)
    out_ref = attention_ref(query_ref, key_ref, value_ref)
    torch.cuda.synchronize()

    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    out_ref.backward(grad_output)
    torch.cuda.synchronize()

    out_snr = compute_snr(out_ref, out)
    query_grad_snr = compute_snr(query.grad, query_ref.grad)
    key_grad_snr = compute_snr(key.grad, key_ref.grad)
    value_grad_snr = compute_snr(value.grad, value_ref.grad)

    print(f"{out_snr:.2f}", f"{query_grad_snr:.2f}", f"{key_grad_snr:.2f}", f"{value_grad_snr:.2f}")
    assert out_snr > 40, "out_snr too low"
    assert query_grad_snr > 40, "query_grad_snr too low"
    assert key_grad_snr > 40, "key_grad_snr too low"
    assert value_grad_snr > 40, "value_grad_snr too low"


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_fp8(batch, config, causal, enable_torch_compile, dtype):
    print(
        f"\nB={batch}, Seq={config.seqlen_q}, NHQ={config.num_head_q}, NHKV={config.num_head_kv}, "
        f"HDQK={config.head_dim_qk}, HDV={config.head_dim_v}, Causal={causal}, "
        f"Compile={enable_torch_compile}, DType={dtype}"
    )
    device = "cuda:0"
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
        fp8_config=Float8QuantConfig(
            granularity=ScalingGranularity.BLOCKWISE,
            block_size=64,
        ),
    )
    attention_ref = TurboAttentionRef(softmax_scale=sm_scale, causal=causal)
    if enable_torch_compile:
        torch._dynamo.reset()
        primus_attention_triton = torch.compile(primus_attention_triton, fullgraph=True, mode="max-autotune")
    torch.cuda.synchronize()

    # Test
    output = primus_attention_triton(query, key, value)
    out_ref = attention_ref(query_ref, key_ref, value_ref)
    torch.cuda.synchronize()

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    out_ref.backward(grad_output)
    torch.cuda.synchronize()

    out_snr = compute_snr(out_ref, output)
    query_grad_snr = compute_snr(query.grad, query_ref.grad)
    key_grad_snr = compute_snr(key.grad, key_ref.grad)
    value_grad_snr = compute_snr(value.grad, value_ref.grad)

    print(f"{out_snr:.2f}", f"{query_grad_snr:.2f}", f"{key_grad_snr:.2f}", f"{value_grad_snr:.2f}")

    assert out_snr > 20, "out_snr too low"
    assert query_grad_snr > 20, "query_grad_snr too low"
    assert key_grad_snr > 20, "key_grad_snr too low"
    assert value_grad_snr > 20, "value_grad_snr too low"
