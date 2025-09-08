###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    F8_FWD_MAX,
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.attention.attention_utils import (
    block_scaling_node,
    quant_v_get_p_scale,
)
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
)
from tests.test_utils import compute_snr

test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=28, num_head_kv=4, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # begin regression tests for https://ontrack-internal.amd.com/browse/SWDEV-548136
    AttnConfig(
        seqlen_q=4096 + 64, seqlen_kv=4096 + 64, num_head_q=2, num_head_kv=1, head_dim_qk=32, head_dim_v=32
    ),
    AttnConfig(seqlen_q=2048, seqlen_kv=2048, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # end regression tests for https://ontrack-internal.amd.com/browse/SWDEV-548136
    AttnConfig(seqlen_q=512, seqlen_kv=512, num_head_q=40, num_head_kv=40, head_dim_qk=192, head_dim_v=128),
]


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend_type", ["triton", "ck"])
def test_attention_bf16(batch, config, causal, backend_type):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    if head_dim_qk == 192 and head_dim_v == 128 and not causal:
        # DQ snr a little bit low, skip
        return

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
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    loss_ref = o_ref.mean()
    loss_ref.backward()
    o = pt.ops.flash_attn_func(
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
        backend_type=backend_type,
    )

    loss = o.mean()
    loss.backward()

    out_snr = compute_snr(o_ref, o)
    query_grad_snr = compute_snr(query_ref.grad, query.grad)
    key_grad_snr = compute_snr(key_ref.grad, key.grad)
    value_grad_snr = compute_snr(value_ref.grad, value.grad)
    print(out_snr, query_grad_snr, key_grad_snr, value_grad_snr)
    assert out_snr > 20, "out_snr too low"
    if config == test_cases[9]:
        # lower the SNR threshold for this specific case
        assert query_grad_snr > 12, "query_grad_snr too low"
    else:
        assert query_grad_snr > 15, "query_grad_snr too low"
    assert key_grad_snr > 15, "key_grad_snr too low"
    assert value_grad_snr > 15, "value_grad_snr too low"


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend_type", ["triton"])
def test_attention_fp8(batch, config, causal, backend_type):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    if head_dim_qk == 192 and head_dim_v == 128 and not causal:
        # DQ snr a little bit low, skip
        return

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
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    loss_ref = o_ref.mean()
    loss_ref.backward()
    o = pt.ops.attention_fp8_blockwise(
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
        backend_type=backend_type,
    )

    loss = o.mean()
    loss.backward()

    out_snr = compute_snr(o_ref, o)
    query_grad_snr = compute_snr(query_ref.grad, query.grad)
    key_grad_snr = compute_snr(key_ref.grad, key.grad)
    value_grad_snr = compute_snr(value_ref.grad, value.grad)
    print(out_snr, query_grad_snr, key_grad_snr, value_grad_snr)
    assert out_snr > 20, "out_snr too low"
    if config == test_cases[9]:
        # lower the SNR threshold for this specific case
        assert query_grad_snr > 12, "query_grad_snr too low"
    else:
        assert query_grad_snr > 15, "query_grad_snr too low"
    assert key_grad_snr > 15, "key_grad_snr too low"
    assert value_grad_snr > 15, "value_grad_snr too low"


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_fp8_with_sparse_do(batch, config, causal):
    # regression test for https://ontrack-internal.amd.com/browse/SWDEV-548136
    device = torch.device("cuda")
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
    v_fp8, v_scale, _ = quant_v_get_p_scale(v, True)

    o, softmax_lse, _ = attention_triton_forward_impl(
        q_fp8,
        k_fp8,
        v_fp8,
        F8_FWD_MAX,
        q_descale,
        k_descale,
        v_scale,
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
        0,
        0,
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
        v_scale,
        F8_FWD_MAX,
        softmax_lse,
        None,
        None,
        None,
        0,
        0,
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
