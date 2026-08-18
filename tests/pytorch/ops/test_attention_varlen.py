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
