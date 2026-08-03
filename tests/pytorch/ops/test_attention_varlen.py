###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math
from typing import List, Tuple

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.ops import flash_attn_varlen_func
from tests.pytorch.ref.attention_ref import attention_varlen_forward_pytorch_ref_impl
from tests.pytorch.test_utils import compute_snr


def _build_cu_seqlens(seqlens: List[int], device: str) -> Tuple[torch.Tensor, int, int]:
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    return cu, max(seqlens), int(cu[-1].item())


# (seqlens_q, seqlens_k)
SEQLEN_PATTERNS = [
    pytest.param(([512, 512, 512, 512], [512, 512, 512, 512])),
    pytest.param(([1024], [1024])),
    pytest.param(([128, 256, 512, 1024], [128, 256, 512, 1024])),
    pytest.param(([57, 311, 800, 173], [57, 311, 800, 173])),
    pytest.param(([2048, 64, 64, 64], [2048, 64, 64, 64])),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seqlens", SEQLEN_PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "num_head_q,num_head_kv",
    [(8, 8), (16, 4)],  # MHA and GQA
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_flash_attn_varlen(dtype, seqlens, causal, num_head_q, num_head_kv, head_dim):
    seqlens_q, seqlens_k = seqlens

    # Causal varlen requires per-batch q_len == k_len(bottom-right aligned mask)
    if causal and seqlens_q != seqlens_k:
        pytest.skip("Causal varlen requires matching q/k seqlens per batch")

    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cu_seqlens_q, max_seqlen_q, total_q = _build_cu_seqlens(seqlens_q, device)
    cu_seqlens_k, max_seqlen_k, total_k = _build_cu_seqlens(seqlens_k, device)

    q = torch.randn((total_q, num_head_q, head_dim), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((total_k, num_head_kv, head_dim), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((total_k, num_head_kv, head_dim), device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn((total_q, num_head_q, head_dim), device=device, dtype=dtype)

    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()

    sm_scale = head_dim ** (-0.5)

    o_ref = attention_varlen_forward_pytorch_ref_impl(
        q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_k, sm_scale, causal
    )
    o_ref.backward(grad_out)

    o = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
    )
    o.backward(grad_out)

    torch.cuda.synchronize()

    out_snr = compute_snr(o_ref, o)
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    print(
        f"\ndtype={dtype}, causal={causal}, hq={num_head_q}, hkv={num_head_kv}, "
        f"hd={head_dim}, seqlens_q={seqlens_q}, seqlens_k={seqlens_k}\n"
        f"  out={out_snr:.2f} dq={dq_snr:.2f} dk={dk_snr:.2f} dv={dv_snr:.2f}"
    )

    assert out_snr > 40, f"out_snr too low: {out_snr}"
    assert dq_snr > 40, f"dq_snr too low: {dq_snr}"
    assert dk_snr > 40, f"dk_snr too low: {dk_snr}"
    assert dv_snr > 40, f"dv_snr too low: {dv_snr}"


# --- flydsl hd64 THD ragged / block-causal (document-masking) fwd+bwd, gfx950-only ---

# Packed segment layouts: two ragged (unequal segments) + one uniform (rect16 fast path).
FLYDSL_SEGS = [
    [512, 2048, 1024, 4096],
    [256, 512, 128, 384],
    [2048, 2048, 2048, 2048],
]
# window_size: full bottom-right causal, then causal-SWA at two widths.
FLYDSL_WINDOWS = [(-1, -1), (1024, 0), (256, 0)]


def _flydsl_block_diag_ref(q, k, v, do, cu, scale, win):
    """fp32 block-diagonal reference (per-segment bottom-right causal + optional left
    window) for o and dq/dk/dv, matching the flydsl ragged document-masking contract."""
    total_q, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    dev = q.device
    o = torch.empty(total_q, Hq, D, device=dev, dtype=torch.float32)
    dq = torch.empty_like(o)
    dk = torch.empty(total_q, Hkv, D, device=dev, dtype=torch.float32)
    dv = torch.empty_like(dk)
    for i in range(cu.numel() - 1):
        b, e = int(cu[i].item()), int(cu[i + 1].item())
        S = e - b
        qe = q[b:e].permute(1, 0, 2).float().detach().requires_grad_(True)
        ke = k[b:e].permute(1, 0, 2).float().detach().requires_grad_(True)
        ve = v[b:e].permute(1, 0, 2).float().detach().requires_grad_(True)
        sc = torch.matmul(qe, ke.repeat_interleave(G, 0).transpose(-1, -2)) * scale
        qi = torch.arange(S, device=dev).view(1, S, 1)
        kj = torch.arange(S, device=dev).view(1, 1, S)
        mask = kj > qi
        if win >= 0:
            mask = mask | (kj < qi - win)
        p = torch.softmax(sc.masked_fill(mask, float("-inf")), dim=-1)
        os_ = torch.matmul(p, ve.repeat_interleave(G, 0))
        os_.backward(do[b:e].permute(1, 0, 2).float())
        o[b:e] = os_.permute(1, 0, 2).detach()
        dq[b:e] = qe.grad.permute(1, 0, 2)
        dk[b:e] = ke.grad.permute(1, 0, 2)
        dv[b:e] = ve.grad.permute(1, 0, 2)
    return o, dq, dk, dv


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="flydsl flash-attn is gfx950-only"
)
@pytest.mark.parametrize("segs", FLYDSL_SEGS)
@pytest.mark.parametrize("window", FLYDSL_WINDOWS)
def test_flydsl_flash_attn_varlen_blockcausal(segs, window):
    """flydsl hd64 THD ragged / block-causal (packed document masking) fwd+bwd: SNR vs a
    fp32 block-diagonal reference + bitwise determinism, over ragged and uniform segment
    layouts x {full-causal, causal-SWA}. GQA G=8 (flydsl requires G a power of two >= 8)."""
    from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
        flash_attn_varlen_flydsl_backward_impl,
        flash_attn_varlen_flydsl_forward_impl,
    )

    dev = "cuda"
    torch.manual_seed(0)
    D, Hq, Hkv = 64, 64, 8
    scale = 1.0 / math.sqrt(D)
    cu, maxs, total = _build_cu_seqlens(segs, dev)
    q = (torch.randn(total, Hq, D, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    k = (torch.randn(total, Hkv, D, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    v = (torch.randn(total, Hkv, D, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    do = (torch.randn(total, Hq, D, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()

    out, lse = flash_attn_varlen_flydsl_forward_impl(
        q, k, v, cu, cu, maxs, maxs, softmax_scale=scale, causal=True, window_size=window, return_lse=True
    )
    dq, dk, dv = flash_attn_varlen_flydsl_backward_impl(
        do, q, k, v, out, lse, cu, cu, maxs, maxs, softmax_scale=scale, causal=True, window_size=window
    )

    o_ref, dq_ref, dk_ref, dv_ref = _flydsl_block_diag_ref(q, k, v, do, cu, scale, window[0])
    o_snr = compute_snr(o_ref, out.float())
    dq_snr = compute_snr(dq_ref, dq.float())
    dk_snr = compute_snr(dk_ref, dk.float())
    dv_snr = compute_snr(dv_ref, dv.float())
    print(
        f"\nsegs={segs} win={window} o={o_snr:.1f} dq={dq_snr:.1f} dk={dk_snr:.1f} dv={dv_snr:.1f}"
    )
    assert o_snr > 40, f"o SNR too low: {o_snr:.1f}"
    assert dq_snr > 40, f"dq SNR too low: {dq_snr:.1f}"
    assert dk_snr > 40, f"dk SNR too low: {dk_snr:.1f}"
    assert dv_snr > 40, f"dv SNR too low: {dv_snr:.1f}"

    # Determinism: one WG owns each output tile (split-K reduce is host-side sum), bit-exact re-run.
    dq2, dk2, dv2 = flash_attn_varlen_flydsl_backward_impl(
        do, q, k, v, out, lse, cu, cu, maxs, maxs, softmax_scale=scale, causal=True, window_size=window
    )
    assert torch.equal(dq, dq2) and torch.equal(dk, dk2) and torch.equal(dv, dv2), "bwd not deterministic"


def test_flash_attn_varlen_no_grad():
    """Smoke test: forward-only (inference) path."""
    device = "cuda"
    torch.manual_seed(0)
    seqlens = [256, 128, 384]
    cu, max_s, total = _build_cu_seqlens(seqlens, device)

    nh, hd = 8, 64
    q = torch.randn((total, nh, hd), device=device, dtype=torch.bfloat16)
    k = torch.randn((total, nh, hd), device=device, dtype=torch.bfloat16)
    v = torch.randn((total, nh, hd), device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        o = flash_attn_varlen_func(q, k, v, cu, cu, max_s, max_s, causal=True)

    assert o.shape == (total, nh, hd)
    assert o.dtype == torch.bfloat16
