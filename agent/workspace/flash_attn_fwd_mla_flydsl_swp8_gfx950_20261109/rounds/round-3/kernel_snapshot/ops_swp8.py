import functools

import torch

from .flash_attn_fwd_mla_swp8 import build_flash_attn_swp8_module


@functools.lru_cache(maxsize=64)
def _fwd_module(hq, hkv, d_qk, d_v, emit_lse, waves_per_eu, stagger, sched_hints):
    return build_flash_attn_swp8_module(
        num_heads=hq,
        head_dim=d_qk,
        head_dim_v=d_v,
        num_kv_heads=hkv,
        emit_lse=emit_lse,
        waves_per_eu=waves_per_eu,
        stagger=stagger,
        sched_hints=sched_hints,
    )


def mla_fwd_sbhd_swp8(q, k, v, causal=True, return_lse=False, waves_per_eu=2, stagger=True, sched_hints=True):
    """Eight-stage staggered forward; same contract as mla_fwd_sbhd (causal 192/128 only)."""
    Sq, B, Hq, d_qk = q.shape
    Skv, Bk, Hkv, d_k = k.shape
    d_v = v.shape[-1]
    assert causal, "swp8 covers causal only"
    assert Bk == B and d_k == d_qk and v.shape[:3] == k.shape[:3]
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert Sq == Skv, "prototype covers self-attention only"
    mod = _fwd_module(Hq, Hkv, d_qk, d_v, bool(return_lse), waves_per_eu, bool(stagger), bool(sched_hints))
    out = q.new_empty((Sq, B, Hq, d_v))
    lse = torch.empty((B * Sq, Hq) if return_lse else (0,), device=q.device, dtype=torch.float32)
    mod(
        q,
        k,
        v,
        out,
        B,
        Sq,
        stride_q_n=B * Hq * d_qk,
        stride_kv_n=B * Hkv * d_qk,
        seq_len_kv=Skv,
        lse=lse if return_lse else None,
        stream=torch.cuda.current_stream(q.device),
    )
    return (out, lse) if return_lse else out
