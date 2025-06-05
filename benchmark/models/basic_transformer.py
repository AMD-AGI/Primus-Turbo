import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ===== Basic Transformer Layer (Torch Native) =====
class BasicAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_kv_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # TODO: ROPE

        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        attn_output = flash_attn_func(xq, xk, xv, dropout_p=0.0, causal=True)

        attn_output = attn_output.view(bs, seqlen, -1)
        return self.wo(attn_output)


class BasicMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BasicTransformerBlock(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_heads, n_kv_heads, norm_eps=1e-5):
        super().__init__()
        self.attention = BasicAttention(hidden_size, n_heads, n_kv_heads)
        self.mlp = BasicMLP(hidden_size, intermediate_size)

        self.attention_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, x: torch.Tensor):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.mlp(self.mlp_norm(h))
        return out


class BasicTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 8192,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.block = BasicTransformerBlock(hidden_size, intermediate_size, n_heads, n_kv_heads)
        self.norm = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (bs, seqlen, hidden)
        x = self.block(x)
        x = self.norm(x)
        logits = self.head(x)  # (bs, seqlen, vocab)
        return logits
