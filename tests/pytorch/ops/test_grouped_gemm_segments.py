###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped GEMM with the weights given as segments: ``grouped_gemm(a, [b0, b1, ...], group_lens)``."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ops.test_grouped_gemm import (
    _poison_alloc_pool,
    generate_grouped_gemm_group_lens_with_zeros,
)
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import get_tolerances

# Groups per segment: an MoE layer's experts followed by replica slots, an empty trailing segment, and the
# maximum of four segments.
SEGMENTS = [(4, 2), (6, 0), (3, 1, 2, 2)]
# (schedule, (rows per group, N, K)): N=320 is off the tile grid. work_steal stays on a shape with enough tiles
# per XCD: the WS kernels do not compile when the heuristic returns local_per_xcd == 1, which Triton turns into
# a constant (#670).
SCHEDULE_SHAPES = [
    ("static", (512, 2048, 1536)),
    ("static", (256, 320, 2880)),
    ("work_steal", (512, 2048, 1536)),
]


@pytest.fixture(autouse=True)
def _triton_backend(request):
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    GlobalBackendManager.set_auto_tune(False)
    request.addfinalizer(GlobalBackendManager.reset)


def _make_inputs(segments, M, N, K, dtype, trans_b, num_zero=0):
    G = sum(segments)
    if num_zero:
        group_lens = generate_grouped_gemm_group_lens_with_zeros(G, M, num_zero=num_zero)
    else:
        group_lens = generate_grouped_gemm_group_lens(G, M, balance=False)
    group_lens = group_lens.to("cuda")
    a = torch.randn((int(group_lens.sum()), K), dtype=dtype, device="cuda")
    b = torch.randn((G, N, K) if trans_b else (G, K, N), dtype=dtype, device="cuda")
    return a, b, group_lens


def _split(b, segments):
    """Separate allocations, as the segments of a real layer are."""
    return [s.clone().requires_grad_(True) for s in b.split(segments)]


def _wgrad_fp32(a, grad_out, group_lens, trans_b):
    out, start = [], 0
    for n in group_lens.tolist():
        x, dy = a[start : start + n].float(), grad_out[start : start + n].float()
        out.append(dy.T @ x if trans_b else x.T @ dy)
        start += n
    return torch.stack(out)


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("schedule, shape", SCHEDULE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("num_zero", [0, 2])
def test_grouped_gemm_segments_func(segments, schedule, shape, dtype, trans_b, num_zero):
    """Forward, dgrad and wgrad equal those of the same weights in one tensor bit for bit (same kernels,
    tiles and accumulation order), and the fp32 reference within tolerance."""
    torch.manual_seed(42)
    a0, b0, group_lens = _make_inputs(segments, *shape, dtype, trans_b, num_zero)

    a_ref = a0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    out_ref = grouped_gemm(a_ref, b_ref, group_lens, trans_b=trans_b, schedule=schedule)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    a = a0.clone().requires_grad_(True)
    segs = _split(b0, segments)
    out = grouped_gemm(a, segs, group_lens, trans_b=trans_b, schedule=schedule)
    out.backward(grad_out)

    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a.grad, a_ref.grad, rtol=0.0, atol=0.0)
    for s, g in zip(segs, b_ref.grad.split(segments)):
        torch.testing.assert_close(s.grad, g, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out, grouped_gemm_ref(a0, b0, group_lens, trans_b), **get_tolerances(dtype))


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("schedule", ["static", "work_steal"])
def test_grouped_gemm_segments_fused_grad_accum(segments, main_grad_dtype, trans_b, schedule):
    """Every segment's own ``main_grad`` moves by exactly its wgrad, groups without tokens leave it
    untouched, and every segment is flagged during forward."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    a0, b0, group_lens = _make_inputs(segments, 512, 2048, 1536, dtype, trans_b, num_zero=2)

    a_ref = a0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    out_ref = grouped_gemm(a_ref, b_ref, group_lens, trans_b=trans_b, schedule=schedule)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    a = a0.clone().requires_grad_(True)
    segs = _split(b0, segments)
    previous = [torch.randn(s.shape, device="cuda").to(main_grad_dtype) for s in segs]
    for s, p in zip(segs, previous):
        s.main_grad = p.clone()
        s.grad_added_to_main_grad = False
    out = grouped_gemm(
        a, segs, group_lens, trans_b=trans_b, schedule=schedule, fuse_bgrad_accum_pattern="megatron"
    )
    out.backward(grad_out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a.grad, a_ref.grad, rtol=0.0, atol=0.0)
    # The kernel adds the fp32 wgrad before its one rounding, so the expectation must not round it first.
    wgrad = _wgrad_fp32(a0, grad_out, group_lens, trans_b)
    empty = (group_lens == 0).tolist()
    first = 0
    for s, p, g in zip(segs, previous, wgrad.split(segments)):
        assert s.grad_added_to_main_grad is True, "every segment must be flagged during forward"
        expected = (p.float() + g).to(main_grad_dtype)
        torch.testing.assert_close(s.main_grad, expected, **get_tolerances(main_grad_dtype))
        for local in range(s.shape[0]):
            if empty[first + local]:
                torch.testing.assert_close(s.main_grad[local], p[local], rtol=0.0, atol=0.0)
        first += s.shape[0]


@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm_segments_padded_tail_zeroed(trans_b):
    """Rows past sum(group_lens) are zeroed, as for a single weight tensor."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    segments = (5, 3)
    K, N, PAD = 2048, 2880, 224
    group_lens = torch.tensor([4096, 0, 3072, 0, 0, 5120, 0, 0], dtype=torch.int64, device="cuda")
    S = int(group_lens.sum())
    a = torch.randn((S + PAD, K), dtype=dtype, device="cuda")
    b = torch.randn((sum(segments), N, K) if trans_b else (sum(segments), K, N), dtype=dtype, device="cuda")
    segs = [s.clone() for s in b.split(segments)]

    grouped_gemm(a, segs, group_lens, trans_b=trans_b)
    torch.cuda.synchronize()
    _poison_alloc_pool((S + PAD, N), dtype, "cuda", 12288.0)

    out = grouped_gemm(a, segs, group_lens, trans_b=trans_b)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[S:], torch.zeros_like(out[S:]), rtol=0.0, atol=0.0)


def test_grouped_gemm_segments_single_segment():
    """A one-element list takes the single-tensor path."""
    torch.manual_seed(42)
    a, b, group_lens = _make_inputs((4,), 256, 512, 256, torch.bfloat16, trans_b=True)
    torch.testing.assert_close(
        grouped_gemm(a, [b], group_lens, trans_b=True),
        grouped_gemm(a, b, group_lens, trans_b=True),
        rtol=0.0,
        atol=0.0,
    )


def test_grouped_gemm_segments_rejects():
    torch.manual_seed(42)
    a, b, group_lens = _make_inputs((2, 2, 2, 2, 2), 128, 256, 256, torch.bfloat16, trans_b=True)

    with pytest.raises(ValueError, match="cannot handle"):
        grouped_gemm(a, list(b.split(2)), group_lens, trans_b=True)
    with pytest.raises(AssertionError, match="groups"):
        grouped_gemm(a, list(b.split(4)[:2]), group_lens, trans_b=True)
    with pytest.raises(AssertionError, match="segment shapes"):
        grouped_gemm(a, [b[:5], b[5:, :128]], group_lens, trans_b=True)

    segs = _split(b, (5, 5))
    for s, main_grad_dtype in zip(segs, (torch.float32, torch.bfloat16)):
        s.main_grad = torch.zeros_like(s, dtype=main_grad_dtype)
        s.grad_added_to_main_grad = False
    with pytest.raises(AssertionError, match="one dtype"):
        grouped_gemm(a, segs, group_lens, trans_b=True, fuse_bgrad_accum_pattern="megatron")

    # Segments run on Triton only: pinning another backend is an error, not a silent fallback.
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.CK)
    with pytest.raises(ValueError, match="not registered"):
        grouped_gemm(a, list(b.split(5)), group_lens, trans_b=True)
