###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Tuple

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_varlen_flydsl_forward_impl,
)
from primus_turbo.pytorch.ops import flash_attn_varlen_func
from tests.pytorch.ref.attention_ref import attention_varlen_forward_pytorch_ref_impl
from tests.pytorch.test_utils import compute_snr, pinned_backend_takes


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
    # Odd lengths: the FlyDSL dQ reduce folds odd rows one per work-group; a batch of one with
    # Hkv=4 also takes the dK/dV slot fold's torch fallback.
    pytest.param(([777], [777])),
    pytest.param(([777, 777], [777, 777])),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seqlens", SEQLEN_PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "num_head_q,num_head_kv",
    # MHA, GQA, and a GQA group of 8 -- the smallest the FlyDSL varlen backend admits, so
    # without it the pinned-FLYDSL half of this test has nothing to run.
    [(8, 8), (16, 4), (64, 8)],
)
@pytest.mark.parametrize("head_dim", [64, 128])
# Per-segment sliding window; -1 is the plain block-causal document mask.
@pytest.mark.parametrize("window_size_left", [-1, 256])
# None is whatever resolves; the rest pin one backend so its own path stays covered.
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL])
def test_flash_attn_varlen(
    dtype, seqlens, causal, num_head_q, num_head_kv, head_dim, window_size_left, backend
):
    seqlens_q, seqlens_k = seqlens
    if window_size_left >= 0 and not causal:
        pytest.skip("a left window is only defined against a causal mask")

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
    window_size = (window_size_left, 0) if window_size_left >= 0 else (-1, -1)

    # Ahead of the reference, which is the expensive part and pointless for a combo the
    # pinned backend does not implement (that case is covered by the refusal assert inside).
    if not pinned_backend_takes(
        backend,
        varlen=True,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=window_size,
        bias=None,
        alibi_slopes=None,
        sink=None,
    ):
        return

    o_ref = attention_varlen_forward_pytorch_ref_impl(
        q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_k, sm_scale, causal, window_size=window_size
    )
    o_ref.backward(grad_out)

    GlobalBackendManager.set_attn_backend(backend, PrecisionType.BF16_FP16_FP32)
    try:
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
            window_size=window_size,
        )
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)
    o.backward(grad_out)

    torch.cuda.synchronize()

    out_snr = compute_snr(o_ref, o)
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    print(
        f"\ndtype={dtype}, causal={causal}, hq={num_head_q}, hkv={num_head_kv}, "
        f"hd={head_dim}, window={window_size_left}, backend={backend.name if backend else 'auto'}, "
        f"seqlens_q={seqlens_q}, seqlens_k={seqlens_k}\n"
        f"  out={out_snr:.2f} dq={dq_snr:.2f} dk={dk_snr:.2f} dv={dv_snr:.2f}"
    )

    assert out_snr > 40, f"out_snr too low: {out_snr}"
    assert dq_snr > 40, f"dq_snr too low: {dq_snr}"
    assert dk_snr > 40, f"dk_snr too low: {dk_snr}"
    assert dv_snr > 40, f"dv_snr too low: {dv_snr}"


@pytest.mark.skipif(not (torch.cuda.is_available() and is_gfx950()), reason="FlyDSL attention is gfx950-only")
@pytest.mark.parametrize("head_dim", [64, 128])
# Each window leaves rows whose prologue KV tile lies wholly left of it (at BLOCK_M 64 and 128 alike);
# 300 sits off the 64-key tile grid.
@pytest.mark.parametrize("window_size_left", [64, 127, 300])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k,num_head_q,num_head_kv,with_sink",
    [
        ([2048], [2048], 8, 1, False),  # GQA 8, merged CTA
        ([1000, 1000], [1000, 1000], 8, 8, False),  # MHA, two segments ending in a partial q block
        ([512], [1536], 16, 2, True),  # bottom-right cross-seqlen, with a sink
    ],
    ids=["gqa8", "mha-2seg", "cross-sink"],
)
def test_flydsl_swa_forward_lse(
    head_dim, window_size_left, seqlens_q, seqlens_k, num_head_q, num_head_kv, with_sink
):
    """A left window can leave a row's prologue KV tile wholly masked; its first live tile must
    still be scored exactly. Checks the LSE and every row: the defect this guards (head dim 128
    has no score-bound floor) hit only such rows, and showed in the LSE before the SNR."""
    device = "cuda"
    torch.manual_seed(10007)
    cu_seqlens_q, max_seqlen_q, total_q = _build_cu_seqlens(seqlens_q, device)
    cu_seqlens_k, max_seqlen_k, total_k = _build_cu_seqlens(seqlens_k, device)
    q = torch.randn((total_q, num_head_q, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((total_k, num_head_kv, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.randn((total_k, num_head_kv, head_dim), device=device, dtype=torch.bfloat16)
    sink = torch.randn((num_head_q,), device=device, dtype=torch.float32) if with_sink else None

    out, lse = flash_attn_varlen_flydsl_forward_impl(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=True,
        window_size=(window_size_left, 0),
        return_lse=True,
        sink=sink,
    )

    group = num_head_q // num_head_kv
    out_ref = torch.empty_like(out, dtype=torch.float32)
    lse_ref = torch.empty_like(lse)
    q0 = k0 = 0
    for sq, skv in zip(seqlens_q, seqlens_k):
        qh = q[q0 : q0 + sq].float().transpose(0, 1)
        kh = k[k0 : k0 + skv].float().transpose(0, 1).repeat_interleave(group, 0)
        vh = v[k0 : k0 + skv].float().transpose(0, 1).repeat_interleave(group, 0)
        scores = qh @ kh.transpose(-1, -2) * head_dim ** (-0.5)
        i = torch.arange(sq, device=device)[:, None] + (skv - sq)
        j = torch.arange(skv, device=device)[None, :]
        scores.masked_fill_(~((j <= i) & (j >= i - window_size_left)), float("-inf"))
        if sink is not None:
            # The sink is a key with logit sink[h] and a zero value.
            scores = torch.cat([scores, sink[:, None, None].expand(-1, sq, 1)], dim=-1)
            vh = torch.cat([vh, vh.new_zeros((num_head_q, 1, head_dim))], dim=1)
        out_ref[q0 : q0 + sq] = (torch.softmax(scores, dim=-1) @ vh).transpose(0, 1)
        lse_ref[q0 : q0 + sq] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
        q0, k0 = q0 + sq, k0 + skv

    # Measured on these sets: the kernel's own floor (bf16-rounded scaled Q) stays under 0.006 LSE /
    # 0.01 per row over 20 seeds, while the guarded defect gave 0.25-1.2 LSE and ~1 per row.
    lse_err = (lse - lse_ref).abs()
    row_err = (out.float() - out_ref).norm(dim=-1) / out_ref.norm(dim=-1)
    worst_lse, worst_row = (
        divmod(int(lse_err.argmax()), num_head_q),
        divmod(int(row_err.argmax()), num_head_q),
    )
    assert lse_err.max() < 1e-2, f"lse max err {lse_err.max():.4g} at (token, head) {worst_lse}"
    assert row_err.max() < 2e-2, f"row rel err {row_err.max():.4g} at (token, head) {worst_row}"
    out_snr = compute_snr(out_ref, out)
    assert out_snr > 40, f"out_snr too low: {out_snr}"


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
