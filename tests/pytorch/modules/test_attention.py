import pytest
import torch
import torch._dynamo.config

from primus_turbo.pytorch.modules import CoreAttention
from tests.test_utils import compute_snr

torch._dynamo.config.cache_size_limit = 100
torch._dynamo.config.recompile_limit = 100


class Config:
    def __init__(self, seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v):
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout="bshd"):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown layout {layout}")

    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1
    )
    if layout == "bshd":
        o_ref = o_ref.transpose(1, 2)
    return o_ref


class CoreAttentionRef(torch.nn.Module):
    def __init__(
        self,
        softmax_scale=None,
        causal=False,
    ):
        super().__init__()

        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        return attention_vanilla_forward_pytorch_ref_impl(
            q,
            k,
            v,
            sm_scale=self.softmax_scale,
            causal=self.causal,
        )


test_cases = [
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    Config(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend_type", ["ck", "triton"])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_attention_fp16(batch, config, causal, backend_type, enable_torch_compile):

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

    primus_attention_ck = CoreAttention(
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
    attention_ref = CoreAttentionRef(softmax_scale=sm_scale, causal=causal)
    if enable_torch_compile:
        primus_attention_ck = torch.compile(primus_attention_ck, fullgraph=True, mode="max-autotune")
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


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend_type", ["triton"])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_attention_fp8(batch, config, causal, backend_type, enable_torch_compile):

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

    primus_attention_triton = CoreAttention(
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=True,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=True,
        backend_type="triton",
    )
    attention_ref = CoreAttentionRef(softmax_scale=sm_scale, causal=causal)
    if enable_torch_compile:
        primus_attention_triton = torch.compile(primus_attention_triton, fullgraph=True, mode="max-autotune")
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
