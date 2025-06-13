import pytest
import torch

from primus_turbo.pytorch.modules import CoreAttention
from tests.test_utils import compute_snr


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


@pytest.mark.parametrize("q_layout", [(4, 1024, 32, 128)])
@pytest.mark.parametrize("k_layout", [(4, 1024, 32, 128)])
@pytest.mark.parametrize("v_layout", [(4, 1024, 32, 128)])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_ck(q_layout, k_layout, v_layout, causal):

    device = "cuda"
    dtype = torch.bfloat16
    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)

    primus_attention_ck = CoreAttention(
        attention_type="ck",
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=False,
    )
    attention_ref = CoreAttentionRef(softmax_scale=sm_scale, causal=causal)
    primus_attention_ck = torch.compile(primus_attention_ck, fullgraph=True, mode="max-autotune")
    # attention_ref = torch.compile(attention_ref, fullgraph=True, mode="max-autotune")
    out = primus_attention_ck(query, key, value)
    out_ref = attention_ref(query_ref, key_ref, value_ref)
    print(compute_snr(out_ref, out))
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    out_ref.backward(grad_output)
    print(compute_snr(query.grad, query_ref.grad))
    print(compute_snr(key.grad, key_ref.grad))
    print(compute_snr(value.grad, value_ref.grad))


@pytest.mark.parametrize("q_layout", [(4, 1024, 32, 128)])
@pytest.mark.parametrize("k_layout", [(4, 1024, 32, 128)])
@pytest.mark.parametrize("v_layout", [(4, 1024, 32, 128)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("is_fp8", [True, False])
def test_attention_triton(q_layout, k_layout, v_layout, causal, is_fp8):

    device = "cuda"
    dtype = torch.bfloat16
    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)

    primus_attention_triton = CoreAttention(
        attention_type="triton",
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=True,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=is_fp8,
    )
    attention_ref = CoreAttentionRef(softmax_scale=sm_scale, causal=causal)
    primus_attention_triton = torch.compile(primus_attention_triton, fullgraph=True, mode="max-autotune")
    # attention_ref = torch.compile(attention_ref, fullgraph=True, mode="max-autotune")
    output = primus_attention_triton(query, key, value)
    out_ref = attention_ref(query_ref, key_ref, value_ref)
    # print(output)
    print(compute_snr(out_ref, output))
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    out_ref.backward(grad_output)
    # # print(query.grad)
    print(compute_snr(query.grad, query_ref.grad))
    print(compute_snr(key.grad, key_ref.grad))
    print(compute_snr(value.grad, value_ref.grad))


if __name__ == "__main__":
    torch.manual_seed(123)
    batch, seq_len, num_heads, head_size = 4, 1024, 32, 128
    q_layout = (batch, seq_len, num_heads, head_size)
    k_layout = (batch, seq_len, num_heads, head_size)
    v_layout = (batch, seq_len, num_heads, head_size)
    # test_attention_ck(q_layout, k_layout, v_layout, causal=True)
    test_attention_triton(q_layout, k_layout, v_layout, causal=True, is_fp8=False)
