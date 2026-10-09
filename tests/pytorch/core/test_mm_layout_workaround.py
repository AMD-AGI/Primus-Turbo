###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""mm layout workaround: rewrite rules, chunking, exactness and the enable/disable API.

Most tests run their GEMMs on CPU tensors, which a private hook lets the mode rewrite; the host
still needs a GPU, since importing primus_turbo.pytorch queries it. These tests use operands
with entries in {-1, 0, 1} and K <= 256, so that every partial sum is exact in bf16 and fp16 and
a rewritten GEMM must match ``torch.mm`` bit for bit.

The GPU tests need a gfx1250 GPU, on which they run no fp32 GEMMs: fp32 references are computed
on the CPU, and results are compared there. The tests that run bf16 GEMMs on the GPU also need
hipBLASLt to use the image's gfx1250 kernel library, so they are skipped when torch prefers
hipBLASLt (``torch.backends.cuda.preferred_blas_library()``) and ``HIPBLASLT_TENSILE_LIBPATH``
is unset; point it at the image's own library directory, which with the ROCm Python packages is
``<site-packages>/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250``.
"""

import importlib
import os
import sys
import threading

import pytest
import torch
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode_stack,
)

from primus_turbo.pytorch.core.mm_layout_workaround import (
    disable_mm_layout_workaround,
    enable_mm_layout_workaround,
    mm_layout_workaround,
    mm_layout_workaround_enabled,
)
from primus_turbo.pytorch.core.utils import is_gfx1250

# The module itself, for its private test hooks. (It shares its name with its context manager.)
mlw = importlib.import_module("primus_turbo.pytorch.core.mm_layout_workaround")

_ON_GFX1250 = torch.cuda.is_available() and is_gfx1250()
needs_gfx1250 = pytest.mark.skipif(not _ON_GFX1250, reason="needs a gfx1250 GPU")


def _hipblaslt_library_unset() -> bool:
    """torch prefers hipBLASLt (its cuBLASLt backend on ROCm) and HIPBLASLT_TENSILE_LIBPATH is unset."""
    prefers_hipblaslt = torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublaslt
    return prefers_hipblaslt and not os.environ.get("HIPBLASLT_TENSILE_LIBPATH")


# For the GPU tests that run bf16 GEMMs; see the module docstring.
needs_hipblaslt_library = pytest.mark.skipif(
    _ON_GFX1250 and _hipblaslt_library_unset(),
    reason="torch prefers hipBLASLt but HIPBLASLT_TENSILE_LIBPATH is unset: point it at the image's "
    "gfx1250 hipBLASLt library to run the GPU GEMM tests (see the module docstring)",
)


@pytest.fixture(autouse=True)
def _rewrite_cpu_tensors(monkeypatch):
    """Let the mode rewrite CPU tensors; never leak an enabled mode into another test."""
    monkeypatch.setattr(mlw, "_REWRITE_DEVICE_TYPES", ("cuda", "cpu"))
    yield
    disable_mm_layout_workaround()
    assert not mm_layout_workaround_enabled()


@pytest.fixture
def rewrites(monkeypatch):
    """The kinds of the rewritten mm calls, in call order."""
    calls = []
    chunked_mm = mlw._chunked_mm

    def spy(a, b, copy_a, chunk_bytes, pool):
        calls.append("wgrad" if copy_a else "dgrad")
        return chunked_mm(a, b, copy_a, chunk_bytes, pool)

    monkeypatch.setattr(mlw, "_chunked_mm", spy)
    return calls


def _ints(*shape, dtype=torch.bfloat16, device="cpu"):
    """Entries in {-1, 0, 1}: with K <= 256 every partial sum is exact in bf16 and fp16."""
    return torch.randint(-1, 2, shape, device=device).to(dtype)


def _operands(layout, m, k, n, dtype=torch.bfloat16, device="cpu"):
    """``(a, b)`` of an (m, k) x (k, n) GEMM stored in the layout of a Linear GEMM."""
    if layout == "dgrad":  # dY @ W: A and B row-major
        return _ints(m, k, dtype=dtype, device=device), _ints(k, n, dtype=dtype, device=device)
    if layout == "wgrad":  # dY.t() @ X: A a transposed view
        return _ints(k, m, dtype=dtype, device=device).t(), _ints(k, n, dtype=dtype, device=device)
    if layout == "forward":  # X @ W.t(): B a transposed view
        return _ints(m, k, dtype=dtype, device=device), _ints(n, k, dtype=dtype, device=device).t()
    if layout == "both_transposed":
        return _ints(k, m, dtype=dtype, device=device).t(), _ints(n, k, dtype=dtype, device=device).t()
    raise ValueError(layout)


class _PassThroughMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


# ---------------------------------------------------------------------------
# Which calls are rewritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout, expected",
    [("dgrad", ["dgrad"]), ("wgrad", ["wgrad"]), ("forward", []), ("both_transposed", [])],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rewrites_the_backward_layouts_only(rewrites, layout, expected, dtype):
    a, b = _operands(layout, 96, 64, 80, dtype)
    with mm_layout_workaround(min_bytes=0):
        out = torch.mm(a, b)
    assert rewrites == expected
    assert torch.equal(out, torch.mm(a, b))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["dgrad", "wgrad"])
def test_leaves_other_dtypes_alone(rewrites, layout, dtype):
    a, b = _operands(layout, 96, 64, 80, dtype)
    with mm_layout_workaround(min_bytes=0):
        torch.mm(a, b)
    assert rewrites == []


def test_leaves_other_strides_alone(rewrites):
    b = _ints(64, 80)
    with mm_layout_workaround(min_bytes=0):
        torch.mm(_ints(96, 128)[:, :64], b)  # A a column slice: neither row- nor column-major
        torch.mm(_ints(96, 64), _ints(64, 160)[:, ::2])  # B strided
        torch.mm(_ints(96, 64), _ints(1, 80).expand(64, 80))  # B broadcast (stride 0)
    assert rewrites == []


def test_min_bytes_gates_on_the_copied_operand(rewrites):
    # dgrad copies B, so only B's size counts; wgrad copies A (and B), and A's size counts.
    a, b = _operands("dgrad", 8, 64, 512)
    b_bytes = b.numel() * b.element_size()
    for min_bytes in (b_bytes, b_bytes + 1):
        with mm_layout_workaround(min_bytes=min_bytes):
            torch.mm(a, b)
            torch.mm(b.t().contiguous(), a.t().contiguous())  # big A, small B: dgrad layout
    assert rewrites == ["dgrad"]

    rewrites.clear()
    a, b = _operands("wgrad", 512, 64, 8)
    a_bytes = a.numel() * a.element_size()
    for min_bytes in (a_bytes, a_bytes + 1):
        with mm_layout_workaround(min_bytes=min_bytes):
            torch.mm(a, b)
    assert rewrites == ["wgrad"]


@pytest.mark.parametrize("m, k, n", [(0, 64, 80), (96, 0, 80), (96, 64, 0)])
@pytest.mark.parametrize("layout", ["dgrad", "wgrad"])
def test_leaves_empty_gemms_alone(rewrites, layout, m, k, n):
    a, b = _operands(layout, m, k, n)
    with mm_layout_workaround(min_bytes=0):
        out = torch.mm(a, b)
    assert rewrites == []
    assert torch.equal(out, torch.mm(a, b))


def test_only_cuda_tensors_without_the_test_hook(rewrites, monkeypatch):
    monkeypatch.setattr(mlw, "_REWRITE_DEVICE_TYPES", ("cuda",))
    a, b = _operands("dgrad", 96, 64, 80)
    with mm_layout_workaround(min_bytes=0):
        torch.mm(a, b)
    assert rewrites == []


def test_rewrite_kind_needs_matching_2d_operands():
    a, b = _operands("dgrad", 96, 64, 80)
    assert mlw._rewrite_kind(a, b, 0) == "dgrad"
    assert mlw._rewrite_kind(a, b.half(), 0) is None
    assert mlw._rewrite_kind(a, b.to("meta"), 0) is None
    assert mlw._rewrite_kind(a.to("meta"), b.to("meta"), 0) is None
    assert mlw._rewrite_kind(a[None], b, 0) is None
    assert mlw._rewrite_kind(a.to_sparse(), b, 0) is None
    assert mlw._rewrite_kind(a, b.to_sparse(), 0) is None


# ---------------------------------------------------------------------------
# Results, chunking and scratch buffers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout, m, k, n, chunk_bytes",
    [
        ("dgrad", 96, 64, 80, 256 << 20),  # one GEMM
        ("wgrad", 96, 64, 80, 256 << 20),  # one GEMM, A copied
        ("dgrad", 40, 64, 600, 32 << 10),  # column blocks 256+256+88, B copies in scratch
        ("dgrad", 8, 64, 600, 64 << 10),  # column blocks 512+88, output blocks in scratch
        ("wgrad", 600, 64, 80, 32 << 10),  # row blocks 256+256+88, A copies in scratch
        ("wgrad", 600, 64, 600, 32 << 10),  # row and column blocks
        ("wgrad", 520, 200, 300, 1),  # every temporary over budget: fresh tensors
        ("dgrad", 300, 256, 700, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rewritten_results_are_exact(rewrites, layout, m, k, n, chunk_bytes, dtype):
    torch.manual_seed(0)
    a, b = _operands(layout, m, k, n, dtype)
    with mm_layout_workaround(min_bytes=0, chunk_bytes=chunk_bytes):
        out = torch.mm(a, b)
        again = torch.mm(a, b)  # reuses the scratch buffers
    assert rewrites == [layout, layout]
    ref = torch.mm(a, b)
    assert out.shape == ref.shape and out.dtype == ref.dtype and out.is_contiguous()
    assert torch.equal(out, ref)
    assert torch.equal(again, ref)


def test_scratch_is_reused_and_released():
    a, b = _operands("wgrad", 600, 64, 80)
    enable_mm_layout_workaround(min_bytes=0, chunk_bytes=32 << 10)
    mode = mlw._state.mode
    torch.mm(a, b)
    (pool,) = mode._scratch.values()
    ptrs = {name: buf.data_ptr() for name, buf in pool.items()}
    assert set(ptrs) == {"a", "b"}
    torch.mm(a, b)
    assert {name: buf.data_ptr() for name, buf in pool.items()} == ptrs

    enable_mm_layout_workaround(min_bytes=1, chunk_bytes=32 << 10)  # same budget: kept
    assert mode._scratch
    enable_mm_layout_workaround(min_bytes=0, chunk_bytes=64 << 10)  # new budget: dropped
    assert mode._scratch == {}
    torch.mm(a, b)
    assert mode._scratch
    disable_mm_layout_workaround()
    assert mode._scratch == {}


@pytest.mark.parametrize("chunk_bytes", [256 << 20, 16 << 10])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_grads_are_identical(rewrites, chunk_bytes, dtype):
    torch.manual_seed(0)
    # Backward reductions run over 256 output features (dgrad) and 2 x 96 tokens (wgrad).
    linear = torch.nn.Linear(384, 256, bias=False, dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(_ints(256, 384, dtype=dtype))
    x = _ints(2, 96, 384, dtype=dtype)
    grad_out = _ints(2, 96, 256, dtype=dtype)

    def grads():
        x_ = x.clone().requires_grad_()
        linear.weight.grad = None
        linear(x_).backward(grad_out)
        return x_.grad, linear.weight.grad

    ref_dx, ref_dw = grads()
    with mm_layout_workaround(min_bytes=0, chunk_bytes=chunk_bytes):
        dx, dw = grads()
    assert sorted(rewrites) == ["dgrad", "wgrad"]
    assert torch.equal(dx, ref_dx)
    assert torch.equal(dw, ref_dw)


def test_triton_unavailable_falls_back_to_torch_copies(monkeypatch):
    warnings = []
    monkeypatch.setattr(mlw.logger, "warning", lambda msg, *args, **kwargs: warnings.append(msg % args))
    monkeypatch.setitem(sys.modules, "primus_turbo.triton.utils.transpose_kernel", None)
    mlw._triton_transpose_into.cache_clear()
    try:
        enable_mm_layout_workaround()
        enable_mm_layout_workaround()
        assert mlw._triton_transpose_into() is None
    finally:
        disable_mm_layout_workaround()
        mlw._triton_transpose_into.cache_clear()
    assert len(warnings) == 1 and "Triton is unavailable" in warnings[0]


def test_transpose_into_validates_its_arguments():
    transpose_kernel = pytest.importorskip("primus_turbo.triton.utils.transpose_kernel")
    x = torch.empty(3, 2)
    for y in (torch.empty(3, 2), torch.empty(3, 2).t(), torch.empty(2, 3, dtype=torch.half)):
        with pytest.raises(ValueError):
            transpose_kernel.transpose_into(y, x)
    with pytest.raises(ValueError):
        transpose_kernel.transpose_into(torch.empty(2), torch.empty(2))
    y = torch.empty(5, 0)
    assert transpose_kernel.transpose_into(y, torch.empty(0, 5)) is y  # nothing to launch


# ---------------------------------------------------------------------------
# Enable / disable / context manager
# ---------------------------------------------------------------------------


def test_enable_is_idempotent_and_disable_undoes_it():
    assert not mm_layout_workaround_enabled()
    enable_mm_layout_workaround()
    enable_mm_layout_workaround()
    modes = [m for m in _get_current_dispatch_mode_stack() if isinstance(m, mlw._MMLayoutWorkaroundMode)]
    assert len(modes) == 1
    assert mm_layout_workaround_enabled()
    disable_mm_layout_workaround()
    assert not mm_layout_workaround_enabled()
    disable_mm_layout_workaround()  # not enabled: a no-op


def test_enable_again_updates_the_settings(rewrites):
    a, b = _operands("dgrad", 96, 64, 80)  # B: 10 KiB
    enable_mm_layout_workaround(min_bytes=1 << 20)
    torch.mm(a, b)
    enable_mm_layout_workaround(min_bytes=0)
    torch.mm(a, b)
    assert rewrites == ["dgrad"]


def test_the_workaround_is_thread_local(rewrites):
    a, b = _operands("dgrad", 96, 64, 80)
    enable_mm_layout_workaround(min_bytes=0)
    seen = []

    def other_thread():
        seen.append(mm_layout_workaround_enabled())
        torch.mm(a, b)

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert seen == [False] and rewrites == []
    torch.mm(a, b)
    assert rewrites == ["dgrad"]


def test_context_manager_restores_the_previous_state(rewrites):
    a, b = _operands("dgrad", 96, 64, 80)
    with mm_layout_workaround(min_bytes=0):
        assert mm_layout_workaround_enabled()
        torch.mm(a, b)
    assert not mm_layout_workaround_enabled()
    torch.mm(a, b)
    assert rewrites == ["dgrad"]

    rewrites.clear()
    enable_mm_layout_workaround(min_bytes=1 << 20, chunk_bytes=1 << 20)
    with mm_layout_workaround(min_bytes=0):
        torch.mm(a, b)
    assert mm_layout_workaround_enabled()
    assert mlw._state.mode.config == (1 << 20, 1 << 20)
    torch.mm(a, b)
    assert rewrites == ["dgrad"]

    with pytest.raises(KeyError):
        with mm_layout_workaround(min_bytes=0):
            raise KeyError("inside the block")
    assert mlw._state.mode.config == (1 << 20, 1 << 20)


def test_disable_refuses_to_pop_a_later_mode():
    enable_mm_layout_workaround()
    later = _PassThroughMode()
    later.__enter__()
    try:
        with pytest.raises(RuntimeError, match="exit it first"):
            disable_mm_layout_workaround()
        assert mm_layout_workaround_enabled()
    finally:
        later.__exit__(None, None, None)
    disable_mm_layout_workaround()
    assert not mm_layout_workaround_enabled()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(min_bytes=-1), ValueError),
        (dict(chunk_bytes=0), ValueError),
        (dict(chunk_bytes=1.5), TypeError),
    ],
)
def test_invalid_settings_are_rejected(kwargs, error):
    with pytest.raises(error):
        enable_mm_layout_workaround(**kwargs)
    assert not mm_layout_workaround_enabled()


# ---------------------------------------------------------------------------
# Transparency to the rest of the dispatch machinery
# ---------------------------------------------------------------------------


def test_higher_order_operators_pass_through():
    def true_fn(x):
        return x.sin()

    def false_fn(x):
        return x.cos()

    x = torch.randn(8)
    with mm_layout_workaround():
        out = torch.cond(torch.tensor(True), true_fn, false_fn, (x,))
    assert torch.equal(out, x.sin())


def test_fake_tensors_pass_through(rewrites):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with mm_layout_workaround(min_bytes=0), FakeTensorMode():
        a = torch.empty(96, 64, dtype=torch.bfloat16)
        b = torch.empty(64, 80, dtype=torch.bfloat16)
        out = torch.mm(a, b)
    assert out.shape == (96, 80)
    assert rewrites == []


def test_gemms_inside_a_tensor_subclass_are_rewritten(rewrites):
    two_tensor = pytest.importorskip("torch.testing._internal.two_tensor")
    a1, b1 = _operands("dgrad", 96, 64, 80)
    a2, b2 = _operands("dgrad", 96, 64, 80)
    with mm_layout_workaround(min_bytes=0):
        out = torch.mm(two_tensor.TwoTensor(a1, a2), two_tensor.TwoTensor(b1, b2))
    assert rewrites == ["dgrad", "dgrad"]
    assert torch.equal(out.a, torch.mm(a1, b1))
    assert torch.equal(out.b, torch.mm(a2, b2))


def test_torch_compile_still_compiles():
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    @torch.compile(backend=backend)
    def f(a, b):
        return torch.mm(a, b) + 1

    a, b = _operands("dgrad", 96, 64, 80)
    try:
        with mm_layout_workaround(min_bytes=0):
            out = f(a, b)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1
    assert torch.equal(out, torch.mm(a, b) + 1)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _cpu_float(t: torch.Tensor) -> torch.Tensor:
    """``t`` as an fp32 CPU tensor, for references computed on the CPU."""
    return t.detach().cpu().float()


def _snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    """Signal-to-noise ratio of ``out`` against ``ref`` in dB, computed on the CPU."""
    ref, out = ref.detach().cpu().double(), out.detach().cpu().double()
    err = out - ref
    return (10 * torch.log10(ref.square().sum() / err.square().sum().clamp_min(1e-30))).item()


@needs_gfx1250
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows, cols", [(1, 1), (7, 5), (65, 130), (257, 63), (300, 1000), (4096, 14336)])
@pytest.mark.parametrize("layout", ["contiguous", "transposed_view", "column_slice"])
def test_gpu_triton_transpose_is_exact(dtype, rows, cols, layout):
    transpose_kernel = pytest.importorskip("primus_turbo.triton.utils.transpose_kernel")
    if layout == "contiguous":
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    elif layout == "transposed_view":
        x = torch.randn(cols, rows, device="cuda", dtype=dtype).t()
    else:
        x = torch.randn(rows, cols + 37, device="cuda", dtype=dtype)[:, 5 : 5 + cols]
    y = torch.full((cols, rows), float("nan"), device="cuda", dtype=dtype)
    assert transpose_kernel.transpose_into(y, x) is y
    assert torch.equal(y.cpu(), x.cpu().t())


@needs_gfx1250
@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LARGE") != "1",
    reason="8 GiB on the GPU and the host; set PRIMUS_TURBO_TEST_LARGE=1",
)
def test_gpu_triton_transpose_beyond_int32_offsets():
    transpose_kernel = pytest.importorskip("primus_turbo.triton.utils.transpose_kernel")
    rows, cols = (1 << 16) + 1, (1 << 15) + 3  # 2**31 + 229,379 elements
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    y = torch.full((cols, rows), float("nan"), device="cuda", dtype=torch.bfloat16)
    transpose_kernel.transpose_into(y, x)
    assert torch.equal(y.cpu(), x.cpu().t())


@needs_gfx1250
@needs_hipblaslt_library
@pytest.mark.parametrize(
    "layout, m, k, n, chunk_bytes",
    [
        ("dgrad", 8192, 4096, 4096, 256 << 20),
        ("dgrad", 8192, 4096, 1024, 256 << 20),
        ("wgrad", 4096, 8192, 4096, 256 << 20),
        ("wgrad", 1024, 8192, 4096, 256 << 20),
        ("dgrad", 4096, 8192, 4096, 8 << 20),  # column blocks
        ("wgrad", 4096, 8192, 4096, 8 << 20),  # row and column blocks
    ],
)
def test_gpu_rewrite_matches_plain_mm_accuracy(rewrites, layout, m, k, n, chunk_bytes):
    torch.manual_seed(0)
    if layout == "dgrad":
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    else:
        a = torch.randn(k, m, device="cuda", dtype=torch.bfloat16).t()
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    plain = torch.mm(a, b)
    with mm_layout_workaround(chunk_bytes=chunk_bytes):
        out = torch.mm(a, b)
    assert rewrites == [layout]
    assert out.shape == plain.shape and out.is_contiguous()
    ref = _cpu_float(a) @ _cpu_float(b)
    snr_plain, snr_out = _snr_db(ref, plain), _snr_db(ref, out)
    assert snr_out > 45 and snr_out > snr_plain - 1.0, (snr_out, snr_plain)


@needs_gfx1250
@needs_hipblaslt_library
def test_gpu_linear_backward_accuracy(rewrites):
    torch.manual_seed(0)
    linear = torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 2048, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(2, 2048, 4096, device="cuda", dtype=torch.bfloat16)
    with mm_layout_workaround():
        linear(x).backward(grad_out)
    assert sorted(rewrites) == ["dgrad", "wgrad"]
    w, x2d, g2d = (_cpu_float(t) for t in (linear.weight, x.flatten(0, 1), grad_out.flatten(0, 1)))
    assert _snr_db(g2d @ w, x.grad.flatten(0, 1)) > 45
    assert _snr_db(g2d.t() @ x2d, linear.weight.grad) > 45


@needs_gfx1250
@needs_hipblaslt_library
def test_gpu_scratch_is_kept_per_stream():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    with mm_layout_workaround():
        main_out = torch.mm(a, b)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            side_out = torch.mm(a, b)
        torch.cuda.current_stream().wait_stream(side)
        streams = {key[1] for key in mlw._state.mode._scratch}
    assert streams == {torch.cuda.current_stream().cuda_stream, side.cuda_stream}
    torch.testing.assert_close(side_out.cpu(), main_out.cpu())


@needs_gfx1250
@needs_hipblaslt_library
def test_gpu_cuda_graph_capture_uses_fresh_temporaries():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    with mm_layout_workaround():
        eager = torch.mm(a, b)  # also warms up the kernels before capture
        scratch_keys = set(mlw._state.mode._scratch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = torch.mm(a, b)
        assert set(mlw._state.mode._scratch) == scratch_keys
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.cpu(), eager.cpu())
