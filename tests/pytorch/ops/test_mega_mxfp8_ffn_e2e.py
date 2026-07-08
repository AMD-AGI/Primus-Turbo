###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU end-to-end validation of the mega MoE FFN *compute* pipeline in mxfp8:

    permute tokens per expert -> L1 grouped mxfp8 GEMM -> SwiGLU -> quantize act
    -> L2 grouped mxfp8 GEMM -> weighted top-k combine.

This exercises the exact compute chain the distributed mega mxfp8 forward will run
(the two grouped mxfp8 GEMMs from Phase 0, the SwiGLU, and the mxfp8 act re-quant),
just with the cross-rank dispatch/combine emulated in plain torch on one GPU. The
reference is the same routing/permutation in fp32. Gate: SNR >= 20 dB (two chained
fp8 GEMMs + the SwiGLU nonlinearity), matching the fused mega MoE module's e2e gate.
"""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import check_mxfp8_support

torch.manual_seed(0)


def _compute_snr(ref, act):
    ref = ref.float()
    act = act.float()
    noise = ref - act
    return 10.0 * torch.log10((ref * ref).mean() / (noise * noise).mean()).item()


def _swiglu_ref(x, I):
    g = x[:, :I].float().clamp(-10.0, 10.0)
    u = x[:, I:].float().clamp(-10.0, 10.0)
    return (g / (1.0 + torch.exp(-g))) * u


def _build_permutation(topk_idx, G, block_m=256):
    """Group (token, k) pairs by expert, pad each group to a block_m multiple.
    Returns (perm_src [M_total] int (>=0 valid, -1 pad), group_offs [G+1] int64)."""
    T, K = topk_idx.shape
    dev = topk_idx.device
    expert_of = topk_idx.reshape(-1)  # [T*K]
    flat = torch.arange(T * K, device=dev)
    counts = torch.bincount(expert_of, minlength=G)
    padded = ((counts + block_m - 1) // block_m) * block_m
    offs = torch.zeros(G + 1, dtype=torch.int64, device=dev)
    offs[1:] = padded.cumsum(0)
    M_total = int(offs[-1].item())
    perm_src = torch.full((M_total,), -1, dtype=torch.int64, device=dev)
    for g in range(G):
        sel = flat[expert_of == g]
        s = int(offs[g].item())
        perm_src[s : s + sel.numel()] = sel
    return perm_src, offs


@pytest.mark.skipif(not check_mxfp8_support(), reason="mega mxfp8 FFN requires gfx950")
@pytest.mark.parametrize("T,H,I,G,K", [(512, 1024, 1024, 4, 2), (768, 2048, 2048, 8, 2)])
def test_mega_mxfp8_ffn_e2e(T, H, I, G, K):
    from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (
        grouped_gemm_mxfp8_flydsl_kernel,
    )
    from primus_turbo.flydsl.mega.fp8.quant import (
        quantize_grouped_weight_mxfp8,
        quantize_rowwise_mxfp8,
    )
    from primus_turbo.flydsl.mega.swiglu_kernel import swiglu

    dev = "cuda:0"
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w1 = (torch.randn(G, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.05)
    w2 = (torch.randn(G, H, I, device=dev, dtype=torch.bfloat16) * 0.05)

    # router: random gate -> top-k experts + softmax weights
    gate = torch.randn(T, G, device=dev)
    topk_w, topk_idx = torch.topk(gate.softmax(-1), K, dim=-1)  # [T,K]

    perm_src, group_offs = _build_permutation(topk_idx, G, block_m=256)
    M = perm_src.numel()
    valid = perm_src >= 0
    tok_of_row = torch.where(valid, perm_src // K, torch.zeros_like(perm_src))
    k_of_row = torch.where(valid, perm_src % K, torch.zeros_like(perm_src))

    # gather permuted tokens (pad rows = 0)
    xe = torch.zeros(M, H, device=dev, dtype=torch.bfloat16)
    xe[valid] = x[tok_of_row[valid]]

    # ---- mxfp8 compute path ----
    xq, xs = quantize_rowwise_mxfp8(xe)
    w1q, w1s = quantize_grouped_weight_mxfp8(w1)
    l1 = grouped_gemm_mxfp8_flydsl_kernel(xq, xs, w1q, w1s, group_offs, out_dtype=torch.bfloat16)
    act = swiglu(l1)  # [M, I] bf16
    aq, as_ = quantize_rowwise_mxfp8(act)
    w2q, w2s = quantize_grouped_weight_mxfp8(w2)
    l2 = grouped_gemm_mxfp8_flydsl_kernel(aq, as_, w2q, w2s, group_offs, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    y = torch.zeros(T, H, device=dev, dtype=torch.float32)
    w_row = topk_w[tok_of_row, k_of_row].float()  # [M]
    contrib = l2.float() * w_row.unsqueeze(1)
    y.index_add_(0, tok_of_row[valid], contrib[valid])

    # ---- fp32 reference (same routing/permutation, no quant) ----
    y_ref = torch.zeros(T, H, device=dev, dtype=torch.float32)
    xe_f = xe.float()
    l2_ref = torch.zeros(M, H, device=dev, dtype=torch.float32)
    for g in range(G):
        s, e = int(group_offs[g].item()), int(group_offs[g + 1].item())
        if e <= s:
            continue
        l1_g = xe_f[s:e] @ w1[g].float().T  # [., 2I]
        act_g = _swiglu_ref(l1_g, I)
        l2_ref[s:e] = act_g @ w2[g].float().T  # [., H]
    contrib_ref = l2_ref * w_row.unsqueeze(1)
    y_ref.index_add_(0, tok_of_row[valid], contrib_ref[valid])

    snr = _compute_snr(y_ref, y)
    print(f"mega mxfp8 FFN e2e: T={T} H={H} I={I} G={G} K={K} M={M}  SNR={snr:.2f} dB")
    assert snr > 20.0, f"mega mxfp8 FFN e2e SNR too low: {snr:.2f} dB"
