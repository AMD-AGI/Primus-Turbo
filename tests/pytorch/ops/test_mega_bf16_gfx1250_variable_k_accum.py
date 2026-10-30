###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""beta = 1 (``accumulate``) on the gfx1250 Mega variable-K wgrad, the kernel behind dW1 / dW2.

The pool is laid out the way Mega's wgrads see it: each expert's rows start on a 256-row
boundary (``group_k_offsets``), only the first ``masked_k[g]`` of them are valid, and one operand
is read through a row index into a token buffer. ``accumulate=2`` (round to bf16, then add) must
equal a separate ``out.add_(dW)`` bit for bit; ``accumulate=1`` (one rounding) must be at least
as close to an fp32 reference.
"""

import pytest
import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

_POOL_BLOCK = 256


def _vk():
    from primus_turbo.flydsl.mega.gfx1250.bf16 import variable_k

    return variable_k


def _problem(out_m, out_n, counts, gather, seed=0, num_src=3000):
    """Operands, tables and an fp32 reference of ``sum_g a[rows_g].T @ b[rows_g]``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    padded = [(c + _POOL_BLOCK - 1) // _POOL_BLOCK * _POOL_BLOCK for c in counts]
    offsets = torch.tensor([0] + padded, dtype=torch.int64).cumsum(0)
    m_total = int(offsets[-1])
    masked_k = torch.tensor(counts, dtype=torch.int32, device="cuda")
    group_k_offsets = offsets.to(torch.int32).cuda()

    def rnd(*shape):
        return torch.randn(*shape, device="cuda", generator=g).to(torch.bfloat16)

    # pool row -> source row; rows past masked_k are never read, so their entries are left at 0
    index = torch.zeros(m_total + _POOL_BLOCK, dtype=torch.int32, device="cuda")
    for e, c in enumerate(counts):
        start = int(offsets[e])
        index[start : start + c] = torch.randint(0, num_src, (c,), device="cuda", generator=g, dtype=torch.int32)
    kwargs = {}
    if gather == "a":
        a, b = rnd(num_src, out_m), rnd(m_total, out_n)
        kwargs["a_index"] = index
        a_rows, b_rows = a[index[:m_total].long()], b
    elif gather == "b":
        a, b = rnd(m_total, out_m), rnd(num_src, out_n)
        kwargs["b_index"] = index
        a_rows, b_rows = a, b[index[:m_total].long()]
    else:
        a, b = rnd(m_total, out_m), rnd(m_total, out_n)
        a_rows, b_rows = a, b
    ref = torch.zeros(len(counts), out_m, out_n, dtype=torch.float32, device="cuda")
    for e, c in enumerate(counts):
        rows = slice(int(offsets[e]), int(offsets[e]) + c)
        ref[e] = a_rows[rows].float().T @ b_rows[rows].float()
    return a, b, group_k_offsets, masked_k, kwargs, ref


def _old_out(shape, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=g).to(torch.bfloat16)


CASES = [
    # (OUT_M, OUT_N, rows per expert, gathered operand): DeepSeek-V3 dW1 / dW2 shapes, with
    # empty and ragged experts
    (4096, 7168, [300, 0, 1024, 37], "b"),  # dW1: grad_l1 dense, x through the row index
    (7168, 2048, [256, 700, 0, 1], "a"),  # dW2: dy through the row index, act_w dense
    (512, 768, [2048, 2048], "none"),
    (512, 512, [100, 0, 50, 0], "a"),  # 128 pool rows per expert: 64-deep k-tiles, C loaded after the k-loop
]


@pytest.mark.parametrize("out_m, out_n, counts, gather", CASES)
def test_round_first_matches_separate_add(out_m, out_n, counts, gather):
    vk = _vk()
    a, b, offs, masked_k, kw, _ = _problem(out_m, out_n, counts, gather)
    old = _old_out((len(counts), out_m, out_n))
    separate = old.clone()
    separate.add_(vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, **kw))
    fused = old.clone()
    vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, out=fused, accumulate=2, **kw)
    torch.testing.assert_close(fused, separate, rtol=0, atol=0)


@pytest.mark.parametrize("out_m, out_n, counts, gather", CASES)
def test_single_rounding_accuracy_and_repeats(out_m, out_n, counts, gather):
    vk = _vk()
    a, b, offs, masked_k, kw, ref = _problem(out_m, out_n, counts, gather)
    old = _old_out((len(counts), out_m, out_n))
    out = old.clone()
    for _ in range(3):
        vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, out=out, accumulate=1, **kw)
    expected = old.float() + 3 * ref
    # each call rounds once, so three calls are within three bf16 roundings of the exact sum
    scale = expected.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    assert ((out.float() - expected).abs() / scale).max() < 3 * 2**-8
    # experts with no valid row keep their old value exactly
    for e, c in enumerate(counts):
        if c == 0:
            assert torch.equal(out[e], old[e])
    # one call: at least as close to fp32 as the separate path (bf16 dW, then a bf16 add)
    single = old.clone()
    vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, out=single, accumulate=1, **kw)
    separate = old.clone()
    separate.add_(vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, **kw))
    exact = old.float() + ref
    err_single = (single.float() - exact).abs().sum()
    err_separate = (separate.float() - exact).abs().sum()
    assert err_single <= err_separate * 1.001


def test_refuses_partial_tiles_and_bad_out():
    vk = _vk()
    assert vk.grouped_gemm_bf16_variable_k_accum_supported(4096, 7168, 2048)
    assert vk.grouped_gemm_bf16_variable_k_accum_supported(7168, 2048, 2048)
    assert not vk.grouped_gemm_bf16_variable_k_accum_supported(5760, 2880, 2048)  # GPT-OSS: ragged
    a, b, offs, masked_k, kw, _ = _problem(512, 320, [300], "none")
    out = torch.zeros(1, 512, 320, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="whole output tiles"):
        vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, out=out, accumulate=1)
    a, b, offs, masked_k, kw, _ = _problem(512, 512, [300], "none")
    with pytest.raises(ValueError, match="needs out="):
        vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, accumulate=1)
    out32 = torch.zeros(1, 512, 512, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        vk.grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=masked_k, out=out32, accumulate=1)
