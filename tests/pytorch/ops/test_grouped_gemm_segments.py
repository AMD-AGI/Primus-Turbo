###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped GEMM with the weights given as segments: ``grouped_gemm(a, [b0, b1, ...], group_lens)``."""

import warnings

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.utils import is_gfx1250
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
# (rows per group, N, K) that the gfx1250 FlyDSL forward and native dgrad both run: a small shape and the
# DeepSeek-V3 gate/up projection.
FLYDSL_SHAPES = [(512, 2048, 1536), (1024, 4096, 7168)]

needs_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="the FlyDSL segments backend is gfx1250 only"
)


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


def _split_reversed(b, segments):
    """``_split`` with the later segments allocated first, so they may sit below segment 0."""
    return [s.clone().requires_grad_(True) for s in reversed(b.split(segments))][::-1]


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

    # Segments run on Triton and (gfx1250) FlyDSL only: pinning another backend is an error, not a silent
    # fallback.
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.CK)
    with pytest.raises(ValueError, match="not registered"):
        grouped_gemm(a, list(b.split(5)), group_lens, trans_b=True)


@needs_gfx1250
@pytest.mark.parametrize("segments", SEGMENTS + [(0, 5, 1)])
@pytest.mark.parametrize("shape", FLYDSL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("num_zero", [0, 2])
def test_grouped_gemm_segments_flydsl(segments, shape, dtype, trans_b, num_zero):
    """FlyDSL pinned: forward and dgrad on segments, and the unfused wgrad split into them, equal those of the
    same weights in one tensor bit for bit (same tile config, tiles and accumulation order)."""
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.FLYDSL)
    torch.manual_seed(42)
    a0, b0, group_lens = _make_inputs(segments, *shape, dtype, trans_b, num_zero)

    a_ref = a0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    out_ref = grouped_gemm(a_ref, b_ref, group_lens, trans_b=trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    a = a0.clone().requires_grad_(True)
    segs = _split_reversed(b0, segments)
    out = grouped_gemm(a, segs, group_lens, trans_b=trans_b)
    out.backward(grad_out)

    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a.grad, a_ref.grad, rtol=0.0, atol=0.0)
    for s, g in zip(segs, b_ref.grad.split(segments)):
        torch.testing.assert_close(s.grad, g, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out, grouped_gemm_ref(a0, b0, group_lens, trans_b), **get_tolerances(dtype))


@needs_gfx1250
@pytest.mark.parametrize("segments", SEGMENTS)
def test_grouped_gemm_segments_flydsl_nn_transposed_copy(segments):
    """The NN entry's fallback through a transposed weight copy takes segments too (one copy per segment)."""
    pytest.importorskip("flydsl")
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
        grouped_gemm_bf16_nn_flydsl_kernel,
    )

    torch.manual_seed(42)
    a, b, group_lens = _make_inputs(segments, 512, 2048, 1536, torch.bfloat16, trans_b=False)
    group_offs = torch.cat([group_lens.new_zeros(1), group_lens.cumsum(0)])
    segs = [s.detach() for s in _split_reversed(b, segments)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = grouped_gemm_bf16_nn_flydsl_kernel(a, b, group_offs, native=0)
        out = grouped_gemm_bf16_nn_flydsl_kernel(a, segs, group_offs, native=0)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


@needs_gfx1250
@pytest.mark.parametrize("segments", SEGMENTS + [(0, 5, 1)])
@pytest.mark.parametrize("shape", FLYDSL_SHAPES)
@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm_segments_flydsl_fused_grad_accum(segments, shape, trans_b):
    """FlyDSL pinned, fused wgrad: every segment's fp32 ``main_grad`` ends bit for bit where the one tensor's
    ``main_grad`` does, and the forward and dgrad match too."""
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.FLYDSL)
    torch.manual_seed(42)
    a0, b0, group_lens = _make_inputs(segments, *shape, torch.bfloat16, trans_b, num_zero=2)
    previous = torch.randn(b0.shape, device="cuda")

    a_ref = a0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    b_ref.main_grad = previous.clone()
    b_ref.grad_added_to_main_grad = False
    out_ref = grouped_gemm(a_ref, b_ref, group_lens, trans_b=trans_b, fuse_bgrad_accum_pattern="megatron")
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    a = a0.clone().requires_grad_(True)
    segs = _split_reversed(b0, segments)
    for s, p in zip(segs, previous.split(segments)):
        s.main_grad = p.clone()
        s.grad_added_to_main_grad = False
    out = grouped_gemm(a, segs, group_lens, trans_b=trans_b, fuse_bgrad_accum_pattern="megatron")
    out.backward(grad_out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a.grad, a_ref.grad, rtol=0.0, atol=0.0)
    for s, g in zip(segs, b_ref.main_grad.split(segments)):
        assert s.grad_added_to_main_grad is True
        torch.testing.assert_close(s.main_grad, g, rtol=0.0, atol=0.0)


def _variable_k_inputs(G, avg_m, out_m, out_n):
    """Operands with exactly G * avg_m rows (the shape keys look at avg_m), split unevenly between groups."""
    lens = torch.full((G,), avg_m, dtype=torch.int64)
    shift = torch.randint(-avg_m // 2, avg_m // 2, (G // 2,)) // 16 * 16
    lens[0::2][: G // 2] += shift
    lens[1::2][: G // 2] -= shift
    offs = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).cuda()
    a = torch.randn(G * avg_m, out_m, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(G * avg_m, out_n, device="cuda", dtype=torch.bfloat16)
    return a, b, offs


# (G, avg_m, OUT_M, OUT_N, kwargs): the plain epilogue; the DeepSeek-V3 persistent-cluster stack key; the
# MiniMax-M3 kwargs key (its vk_seam stores C through the vector memory path, bit 7); that set with the C staging
# stored per block instead (bit 9). The keys apply by themselves through the public entry.
_VK_SEAM_BIT9 = {
    "BLOCK_K": 64,
    "epi_chunk": 4,
    "head_pipe": 8,
    "lds_cosched": 2,
    "num_buffers": 4,
    "persist": 256,
    "split_bar": 1,
    "vk_ovl": 7,
    "vk_seam": 1 | 512,
}
VARIABLE_K_CASES = [
    (8, 512, 2048, 1536, {}),
    (32, 2048, 4096, 7168, {}),
    (32, 512, 6144, 3072, {}),
    (32, 512, 6144, 3072, _VK_SEAM_BIT9),
]


@needs_gfx1250
@pytest.mark.parametrize("G, avg_m, out_m, out_n, kwargs", VARIABLE_K_CASES)
@pytest.mark.parametrize("cut", [(-4,), (0, -2)])
def test_grouped_gemm_segments_flydsl_variable_k_out(G, avg_m, out_m, out_n, kwargs, cut):
    """The FlyDSL variable-K kernel writes output segments bit for bit as it writes one tensor, on every epilogue
    that addresses C per expert. ``cut`` places the segment boundaries (a leading empty segment for 0)."""
    pytest.importorskip("flydsl")
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
        grouped_gemm_bf16_variable_k_flydsl_kernel,
    )

    torch.manual_seed(42)
    a, b, offs = _variable_k_inputs(G, avg_m, out_m, out_n)
    segments = [G + c if c < 0 else c for c in cut]
    segments = [segments[0]] + [y - x for x, y in zip(segments, segments[1:])] + [G - segments[-1]]
    ref = grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, **kwargs)
    segs = [
        torch.full((n, out_m, out_n), float("nan"), device="cuda", dtype=torch.bfloat16)
        for n in segments[::-1]
    ]
    segs = segs[::-1]
    got = grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, out=segs, **kwargs)
    assert got is not None and all(x is y for x, y in zip(got, segs))
    torch.testing.assert_close(torch.cat(segs), ref, rtol=0.0, atol=0.0)


@needs_gfx1250
@pytest.mark.parametrize("accum_impl", ["tdm", "reg"])
@pytest.mark.parametrize("segments", [(28, 4), (0, 30, 2), (10, 0, 12, 10)])
def test_grouped_gemm_segments_flydsl_variable_k_accumulate(accum_impl, segments):
    """``accumulate=1`` into fp32 output segments equals accumulating into the one tensor, bit for bit."""
    pytest.importorskip("flydsl")
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
        grouped_gemm_bf16_variable_k_flydsl_kernel,
    )

    torch.manual_seed(42)
    G = sum(segments)
    a, b, offs = _variable_k_inputs(G, 512, 2048, 1536)
    previous = torch.randn(G, 2048, 1536, device="cuda")
    ref = previous.clone()
    grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, out=ref, accumulate=1, accum_impl=accum_impl)
    segs = [p.clone() for p in reversed(previous.split(segments))][::-1]
    grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, out=segs, accumulate=1, accum_impl=accum_impl)
    torch.testing.assert_close(torch.cat(segs), ref, rtol=0.0, atol=0.0)


def test_grouped_gemm_segments_flydsl_rejects():
    pytest.importorskip("flydsl")
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
        _segments,
    )

    b = torch.empty((6, 64, 32), dtype=torch.bfloat16)
    who = "test"
    with pytest.raises(ValueError, match="1 to 4"):
        _segments(list(b.split(1)), who, "weight")
    with pytest.raises(ValueError, match="agree in shape and strides"):
        _segments([b[:3], b[3:, :32]], who, "weight")
    with pytest.raises(ValueError, match="agree in shape and strides"):
        _segments([b[:3], b[3:].transpose(1, 2).contiguous().transpose(1, 2)], who, "weight")
    with pytest.raises(ValueError, match="dtype or device"):
        _segments([b[:3], b[3:].half()], who, "weight")
    with pytest.raises(ValueError, match="16-byte aligned"):
        _segments([b[:3], torch.empty(3 * 64 * 32 + 1, dtype=b.dtype)[1:].view(3, 64, 32)], who, "weight")

    # Group g of segment s sits seg_delta_s bytes past where it would in one tensor with segment 0's strides.
    segs = [b[:2].clone(), b[2:2].clone(), b[2:].clone()]
    ref, G, seg_args, n_segs = _segments(segs, who, "weight")
    grp_bytes = 64 * 32 * 2
    assert (ref is segs[0], G, n_segs) == (True, 6, 3)
    assert seg_args[:3] == (2, 2, 0)
    assert seg_args[4] == segs[2].data_ptr() - segs[0].data_ptr() - 2 * grp_bytes
    # A leading empty segment: the kernel takes the first segment with groups as its base.
    segs = [b[:0].clone(), b.clone()]
    ref, G, seg_args, n_segs = _segments(segs, who, "weight")
    assert (ref is segs[1], G, n_segs, seg_args[0], seg_args[3]) == (True, 6, 2, 0, 0)
