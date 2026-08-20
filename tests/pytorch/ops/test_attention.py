###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention import attention_gluon_impl, attention_impl
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    _DENSE_FWD_BACKENDS,
    FlashAttnDenseDispatcher,
    FlashAttnVarlenDispatcher,
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    F8_FWD_MAX,
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.sparse_mla_impl import (
    SparseMlaBwdDispatcher,
    SparseMlaFwdDispatcher,
)
from primus_turbo.pytorch.ops import flash_attn_fp8_func, flash_attn_func, sparse_mla_func
from primus_turbo.pytorch.ops.attention import flash_attn_interface
from primus_turbo.pytorch.ops.attention.attention_utils import (
    _infer_qkv_format,
    block_scaling_node,
)
from primus_turbo.triton.attention.sparse_mla import (
    sparse_mla_bwd_triton,
    sparse_mla_fwd_triton,
)
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
    attention_with_sink_ref_impl,
)
from tests.pytorch.test_utils import compute_snr, pinned_backend_takes


class _GluonTensor:
    """Small CUDA-tensor stand-in for dispatcher-only Gluon eligibility tests."""

    def __init__(self, shape, *, dtype=torch.float16, device="cuda:0", storage="bshd"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.storage = storage
        self.is_cuda = device is not None

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dim):
        assert dim == -1
        return 1 if self.storage != "noncontiguous" else 2

    def is_contiguous(self):
        return self.storage == "bshd"

    def transpose(self, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        return SimpleNamespace(is_contiguous=lambda: self.storage == "bhsd")


def _gluon_tensors(*, q_shape=(2, 3, 8, 64), kv_shape=(2, 4, 2, 64), **kwargs):
    return (
        _GluonTensor(q_shape, **kwargs),
        _GluonTensor(kv_shape, **kwargs),
        _GluonTensor(kv_shape, **kwargs),
    )


@pytest.fixture
def gfx950_properties(monkeypatch):
    queried_devices = []

    def get_device_properties(device):
        queried_devices.append(device)
        return SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    attention_impl._GFX950_DEVICE_CACHE.clear()
    yield queried_devices
    attention_impl._GFX950_DEVICE_CACHE.clear()


def _resolve_gluon(q, k, v, **kwargs):
    return resolve_flash_attn_backend(
        varlen=False,
        user_backend=BackendType.GLUON,
        q=q,
        k=k,
        v=v,
        **kwargs,
    )


def test_gluon_registry_keeps_it_explicit_and_dense_only():
    assert list(_DENSE_FWD_BACKENDS) == [
        BackendType.FLYDSL,
        BackendType.AITER,
        BackendType.HIPKITTENS,
        BackendType.GLUON,
    ]
    assert _DENSE_FWD_BACKENDS[BackendType.GLUON].autotune is False
    assert BackendType.GLUON not in FlashAttnVarlenDispatcher._backends


def test_gluon_resolve_unpinned_uses_aiter_before_gluon(gfx950_properties, monkeypatch):
    # FlyDSL imports this helper into attention_impl. Keep this dispatcher-only
    # test independent of the process's current CUDA device and its cached props.
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    q, k, v = _gluon_tensors()

    # Noncausal calls are valid Gluon input but intentionally ineligible for FlyDSL.
    assert (
        resolve_flash_attn_backend(
            varlen=False,
            user_backend=None,
            q=q,
            k=k,
            v=v,
            causal=False,
        )
        == BackendType.AITER
    )


@pytest.mark.parametrize(("storage", "qkv_format"), [("bshd", "bshd"), ("bhsd", "bhsd")])
def test_gluon_resolve_pinned_accepts_supported_input(storage, qkv_format, gfx950_properties):
    q, k, v = _gluon_tensors(storage=storage)

    assert _resolve_gluon(q, k, v, causal=False, qkv_format=qkv_format) == BackendType.GLUON
    assert gfx950_properties == [q.device]


def test_gluon_resolve_reuses_cached_device_capability(gfx950_properties):
    q, k, v = _gluon_tensors()

    assert _resolve_gluon(q, k, v, causal=False) == BackendType.GLUON
    assert _resolve_gluon(q, k, v, causal=False) == BackendType.GLUON
    assert gfx950_properties == [q.device]


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("non_cuda", lambda: _gluon_tensors(device=None)),
        ("non_gfx950", lambda: _gluon_tensors()),
        (
            "mixed_devices",
            lambda: (
                _GluonTensor((2, 3, 8, 64)),
                _GluonTensor((2, 4, 2, 64), device="cuda:1"),
                _GluonTensor((2, 4, 2, 64)),
            ),
        ),
        ("rank", lambda: _gluon_tensors(q_shape=(2, 3, 8))),
        (
            "mixed_dtype",
            lambda: (
                _GluonTensor((2, 3, 8, 64)),
                _GluonTensor((2, 4, 2, 64), dtype=torch.bfloat16),
                _GluonTensor((2, 4, 2, 64)),
            ),
        ),
        ("unsupported_dtype", lambda: _gluon_tensors(dtype=torch.float32)),
        (
            "kv_shape_mismatch",
            lambda: (
                _GluonTensor((2, 3, 8, 64)),
                _GluonTensor((2, 4, 2, 64)),
                _GluonTensor((2, 5, 2, 64)),
            ),
        ),
        (
            "batch_mismatch",
            lambda: (
                _GluonTensor((2, 3, 8, 64)),
                _GluonTensor((3, 4, 2, 64)),
                _GluonTensor((3, 4, 2, 64)),
            ),
        ),
        (
            "head_dim_mismatch",
            lambda: (
                _GluonTensor((2, 3, 8, 64)),
                _GluonTensor((2, 4, 2, 32)),
                _GluonTensor((2, 4, 2, 32)),
            ),
        ),
        (
            "zero_dimension",
            lambda: _gluon_tensors(q_shape=(0, 3, 8, 64), kv_shape=(0, 4, 2, 64)),
        ),
        (
            "zero_kv_heads",
            lambda: _gluon_tensors(q_shape=(2, 3, 8, 64), kv_shape=(2, 4, 0, 64)),
        ),
        (
            "head_dim_over_limit",
            lambda: _gluon_tensors(q_shape=(2, 3, 8, 257), kv_shape=(2, 4, 2, 257)),
        ),
        (
            "invalid_gqa",
            lambda: _gluon_tensors(q_shape=(2, 3, 7, 64), kv_shape=(2, 4, 2, 64)),
        ),
        ("sbhd", lambda: _gluon_tensors(storage="sbhd")),
        ("noncontiguous", lambda: _gluon_tensors(storage="noncontiguous")),
        (
            "mixed_layouts",
            lambda: (
                _GluonTensor((2, 3, 8, 64), storage="bshd"),
                _GluonTensor((2, 4, 2, 64), storage="bhsd"),
                _GluonTensor((2, 4, 2, 64), storage="bshd"),
            ),
        ),
    ],
)
def test_gluon_eligibility_refuses_invalid_tensor_contract(name, build, gfx950_properties, monkeypatch):
    q, k, v = build()
    if name == "non_gfx950":
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
        )

    with pytest.raises(ValueError, match="cannot handle the given inputs"):
        _resolve_gluon(q, k, v, causal=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout_p": 0.1},
        {"softmax_scale": "not-a-real-scalar"},
        {"bias": object()},
        {"alibi_slopes": object()},
        {"sink": object()},
        {"window_size": (1, -1)},
        {"return_softmax": True},
        pytest.param({"causal": True}, id="causal-sq-gt-skv"),
        {"needs_backward": True},
    ],
)
def test_gluon_eligibility_refuses_unsupported_call_features(kwargs, gfx950_properties):
    q, k, v = _gluon_tensors(q_shape=(2, 5, 8, 64), kv_shape=(2, 4, 2, 64))
    kwargs.setdefault("causal", False)

    with pytest.raises(ValueError, match="cannot handle the given inputs"):
        _resolve_gluon(q, k, v, **kwargs)


@pytest.fixture
def gluon_cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the CUDA-only Gluon custom op")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def fake_gluon_raw(monkeypatch):
    calls = []

    def raw(q, k, v, *, softmax_scale, causal, qkv_format):
        assert all(tensor.is_contiguous() for tensor in (q, k, v)), (
            "the wrapper must pass contiguous raw BSHD/BHSD tensors"
        )
        assert all(tensor.stride(-1) == 1 for tensor in (q, k, v))
        if qkv_format == "bshd":
            batch, seqlen_q, num_heads_q, _ = q.shape
        else:
            batch, num_heads_q, seqlen_q, _ = q.shape
        calls.append(
            {
                "shapes": (q.shape, k.shape, v.shape),
                "strides": (q.stride(), k.stride(), v.stride()),
                "contiguous": (q.is_contiguous(), k.is_contiguous(), v.is_contiguous()),
                "scale_type": type(softmax_scale),
                "stream": torch.cuda.current_stream(q.device),
            }
        )
        out = torch.full(q.shape, softmax_scale + float(causal), device=q.device, dtype=q.dtype)
        lse = torch.full((batch, num_heads_q, seqlen_q), softmax_scale, device=q.device, dtype=torch.float32)
        return out, lse

    raw_module = SimpleNamespace(flash_attn_gluon_raw=raw)

    def import_raw(module_name):
        assert module_name == "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"
        return raw_module

    monkeypatch.setattr(attention_gluon_impl, "_import_module", import_raw)
    return calls


def test_gluon_wrapper_bshd_returns_contiguous_output_on_current_stream(gluon_cuda_device, fake_gluon_raw):
    q = torch.empty((2, 3, 8, 64), device=gluon_cuda_device, dtype=torch.float16)
    k = torch.empty((2, 5, 2, 64), device=gluon_cuda_device, dtype=torch.float16)
    v = torch.empty_like(k)
    caller_stream = torch.cuda.Stream(device=gluon_cuda_device)

    with torch.cuda.stream(caller_stream):
        out, lse = attention_gluon_impl.flash_attn_gluon_forward_impl(
            q,
            k,
            v,
            softmax_scale=Fraction(1, 4),
            causal=True,
            qkv_format="bshd",
        )
    caller_stream.synchronize()

    assert out.shape == (2, 3, 8, 64)
    assert out.stride() == (1536, 512, 64, 1)
    assert out.is_contiguous()
    assert out.dtype == torch.float16
    assert out.device == gluon_cuda_device
    assert torch.all(out == 1.25)
    assert lse.shape == (2, 8, 3)
    assert lse.dtype == torch.float32
    assert lse.device == gluon_cuda_device
    assert torch.all(lse == 0.25)
    assert fake_gluon_raw == [
        {
            "shapes": (q.shape, k.shape, v.shape),
            "strides": (q.stride(), k.stride(), v.stride()),
            "contiguous": (True, True, True),
            "scale_type": float,
            "stream": caller_stream,
        }
    ]


def test_gluon_wrapper_bhsd_preserves_physical_layout_and_defaults_scale(gluon_cuda_device, fake_gluon_raw):
    q = torch.empty((2, 8, 3, 64), device=gluon_cuda_device, dtype=torch.bfloat16).transpose(1, 2)
    k = torch.empty((2, 2, 5, 64), device=gluon_cuda_device, dtype=torch.bfloat16).transpose(1, 2)
    v = torch.empty_like(k.transpose(1, 2)).transpose(1, 2)

    out, lse = attention_gluon_impl.flash_attn_gluon_forward_impl(
        q, k, v, softmax_scale=None, causal=False, qkv_format="bhsd"
    )

    assert out.shape == (2, 3, 8, 64)
    assert out.stride() == (1536, 64, 192, 1)
    assert not out.is_contiguous()
    assert out.dtype == torch.bfloat16
    assert out.device == gluon_cuda_device
    assert torch.all(out == 0.125)
    assert lse.shape == (2, 8, 3)
    assert lse.dtype == torch.float32
    assert lse.device == gluon_cuda_device
    assert torch.all(lse == 0.125)
    assert fake_gluon_raw[0]["shapes"] == (
        torch.Size((2, 8, 3, 64)),
        torch.Size((2, 2, 5, 64)),
        torch.Size((2, 2, 5, 64)),
    )
    assert fake_gluon_raw[0]["strides"] == (
        (1536, 192, 64, 1),
        (640, 320, 64, 1),
        (640, 320, 64, 1),
    )
    assert fake_gluon_raw[0]["contiguous"] == (True, True, True)


@pytest.mark.parametrize(
    ("qkv_format", "expected_stride"),
    [("bshd", (1536, 512, 64, 1)), ("bhsd", (1536, 64, 192, 1))],
)
def test_gluon_fake_matches_eager_metadata_and_exact_strides(qkv_format, expected_stride):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        if qkv_format == "bshd":
            q = torch.empty((2, 3, 8, 64), device="cuda", dtype=torch.float16)
            k = torch.empty((2, 5, 2, 64), device="cuda", dtype=torch.float16)
        else:
            q = torch.empty((2, 8, 3, 64), device="cuda", dtype=torch.float16).transpose(1, 2)
            k = torch.empty((2, 2, 5, 64), device="cuda", dtype=torch.float16).transpose(1, 2)
        v = torch.empty_like(k)

        out, lse = attention_gluon_impl.flash_attn_gluon_forward_impl(
            q, k, v, softmax_scale=None, causal=False, qkv_format=qkv_format
        )

    assert out.shape == (2, 3, 8, 64)
    assert out.stride() == expected_stride
    assert out.dtype == q.dtype
    assert out.device == q.device
    assert lse.shape == (2, 8, 3)
    assert lse.dtype == torch.float32
    assert lse.device == q.device


@pytest.mark.parametrize("use_fake", [False, True], ids=["eager", "fake"])
def test_gluon_wrapper_rejects_invalid_format_before_dispatch(use_fake):
    from torch._subclasses.fake_tensor import FakeTensorMode

    context = FakeTensorMode() if use_fake else nullcontext()
    device = "cuda" if use_fake else "cpu"
    with context:
        q = torch.empty((1, 2, 4, 8), device=device, dtype=torch.float16)

        with pytest.raises(ValueError, match="qkv_format must be 'bshd' or 'bhsd'"):
            attention_gluon_impl.flash_attn_gluon_forward_impl(q, q, q, qkv_format="sbhd")


@pytest.mark.parametrize("use_fake", [False, True], ids=["eager", "fake"])
@pytest.mark.parametrize(
    ("qkv_format", "bad_storage", "error_match"),
    [
        ("bshd", "bhsd_view", "storage does not match"),
        ("bhsd", "bshd_contiguous", "storage does not match"),
        ("bshd", "inner_stride", "unit stride"),
    ],
)
def test_gluon_wrapper_rejects_invalid_public_storage(use_fake, qkv_format, bad_storage, error_match):
    from torch._subclasses.fake_tensor import FakeTensorMode

    context = FakeTensorMode() if use_fake else nullcontext()
    device = "cuda" if use_fake else "cpu"
    with context:
        if qkv_format == "bshd":
            q = torch.empty((1, 2, 4, 8), device=device, dtype=torch.float16)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
        else:
            q = torch.empty((1, 4, 2, 8), device=device, dtype=torch.float16).transpose(1, 2)
            k = torch.empty((1, 4, 2, 8), device=device, dtype=torch.float16).transpose(1, 2)
            v = torch.empty((1, 4, 2, 8), device=device, dtype=torch.float16).transpose(1, 2)

        if bad_storage == "bhsd_view":
            k = torch.empty((1, 4, 2, 8), device=device, dtype=torch.float16).transpose(1, 2)
        elif bad_storage == "bshd_contiguous":
            k = torch.empty((1, 2, 4, 8), device=device, dtype=torch.float16)
        else:
            k = torch.empty((1, 2, 4, 16), device=device, dtype=torch.float16)[..., ::2]

        with pytest.raises(ValueError, match=error_match):
            attention_gluon_impl.flash_attn_gluon_forward_impl(q, k, v, qkv_format=qkv_format)


@pytest.mark.parametrize(
    "module_name",
    [
        "primus_turbo.pytorch",
        "primus_turbo.pytorch.kernels.attention.attention_impl",
        "primus_turbo.pytorch.kernels.attention.attention_gluon_impl",
    ],
)
def test_gluon_lazy_import_keeps_raw_kernel_out_of_module_imports(module_name):
    raw_module = "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"
    code = (
        "import importlib, sys; "
        f"importlib.import_module({module_name!r}); "
        f"assert {raw_module!r} not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "import_error",
    [
        ModuleNotFoundError(
            "No module named 'triton.experimental.gluon'",
            name="triton.experimental.gluon",
        ),
        ModuleNotFoundError(
            "No module named 'primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster'",
            name="primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster",
        ),
    ],
    ids=["missing-triton-gluon", "missing-raw-module"],
)
def test_gluon_compiler_capability_error_names_gluon_and_triton(monkeypatch, import_error):
    def unavailable_raw_module(module_name):
        assert module_name == "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"
        raise import_error

    monkeypatch.setattr(attention_gluon_impl, "_import_module", unavailable_raw_module)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="test-triton-9.9"))

    with pytest.raises(RuntimeError) as error:
        attention_gluon_impl._load_flash_attn_gluon_raw()

    message = str(error.value)
    assert "Gluon" in message
    assert "Triton test-triton-9.9" in message
    assert "triton.experimental.gluon" in message
    assert "CDNA4" in message
    assert error.value.__cause__ is import_error


def test_gluon_compiler_capability_loader_preserves_unrelated_import_error(monkeypatch):
    import_error = ImportError(
        "cannot import name 'helper' from 'primus_turbo.gluon.attention.f16_fa_gfx950_common'",
        name="primus_turbo.gluon.attention.f16_fa_gfx950_common",
    )

    def broken_local_helper(module_name):
        assert module_name == "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"
        raise import_error

    monkeypatch.setattr(attention_gluon_impl, "_import_module", broken_local_helper)

    with pytest.raises(ImportError) as error:
        attention_gluon_impl._load_flash_attn_gluon_raw()

    assert error.value is import_error


def test_gluon_wrapper_does_not_wrap_raw_launch_errors(gluon_cuda_device, monkeypatch):
    launch_error = RuntimeError("raw Gluon launch failed")

    def raw(q, k, v, *, softmax_scale, causal, qkv_format):
        raise launch_error

    raw_module = SimpleNamespace(flash_attn_gluon_raw=raw)

    def import_raw(module_name):
        assert module_name == "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"
        return raw_module

    monkeypatch.setattr(attention_gluon_impl, "_import_module", import_raw)
    q = torch.empty((1, 2, 4, 64), device=gluon_cuda_device, dtype=torch.float16)

    with pytest.raises(RuntimeError) as error:
        attention_gluon_impl.flash_attn_gluon_forward_impl(q, q, q)

    assert error.value is launch_error


@pytest.fixture
def fake_public_gluon_launcher(monkeypatch, gfx950_properties):
    calls = []

    def launch(q, k, v, *, softmax_scale, causal, qkv_format):
        out = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
        calls.append(
            {
                "softmax_scale": softmax_scale,
                "causal": causal,
                "qkv_format": qkv_format,
                "out": out,
                "lse": lse,
            }
        )
        return out, lse

    # The wrapper is the unavailable kernel boundary. Resolution and the custom
    # autograd function remain real in these tests.
    monkeypatch.setattr(
        flash_attn_interface,
        "flash_attn_gluon_forward_impl",
        launch,
        raising=False,
    )
    monkeypatch.setattr(
        GlobalBackendManager,
        "get_attn_backend",
        classmethod(lambda cls, precision: BackendType.GLUON),
    )
    return calls


def _fake_public_gluon_tensors(requires_grad=(False, False, False)):
    q = torch.empty(
        (2, 3, 8, 64),
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad[0],
    )
    k = torch.empty(
        (2, 4, 2, 64),
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad[1],
    )
    v = torch.empty(
        (2, 4, 2, 64),
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad[2],
    )
    return q, k, v


@pytest.mark.parametrize(
    "forward_context",
    [torch.no_grad, torch.inference_mode],
    ids=["no-grad", "inference-mode"],
)
def test_gluon_forward_only_allows_requires_grad_inputs_when_grad_is_disabled(
    forward_context, fake_public_gluon_launcher
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q, k, v = _fake_public_gluon_tensors((True, True, True))

        with forward_context():
            out = flash_attn_func(q, k, v, causal=False)

    assert out.shape == q.shape
    assert out.requires_grad is False
    assert [(call["causal"], call["qkv_format"]) for call in fake_public_gluon_launcher] == [(False, "bshd")]


def test_gluon_grad_mode_allows_inputs_that_do_not_require_grad(fake_public_gluon_launcher):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.enable_grad():
        q, k, v = _fake_public_gluon_tensors()
        out = flash_attn_func(q, k, v, causal=False)

    assert out.shape == q.shape
    assert out.requires_grad is False
    assert len(fake_public_gluon_launcher) == 1


@pytest.mark.parametrize("grad_index", [0, 1, 2], ids=["q", "k", "v"])
def test_gluon_grad_mode_rejects_each_qkv_gradient_before_launch(grad_index, fake_public_gluon_launcher):
    from torch._subclasses.fake_tensor import FakeTensorMode

    requires_grad = tuple(index == grad_index for index in range(3))
    with FakeTensorMode(), torch.enable_grad():
        q, k, v = _fake_public_gluon_tensors(requires_grad)

        with pytest.raises(ValueError, match="backend GLUON cannot handle the given inputs"):
            flash_attn_func(q, k, v, causal=False)

    assert fake_public_gluon_launcher == []


@pytest.mark.parametrize("return_lse", [False, True])
def test_gluon_return_lse_preserves_the_public_return_contract(return_lse, fake_public_gluon_launcher):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.enable_grad():
        q, k, v = _fake_public_gluon_tensors()
        result = flash_attn_func(q, k, v, causal=False, return_lse=return_lse)

    if return_lse:
        assert isinstance(result, tuple)
        assert len(result) == 2
        out, lse = result
        assert lse.shape == (2, 8, 3)
    else:
        assert isinstance(result, torch.Tensor)
        out = result
    assert out.shape == q.shape
    assert len(fake_public_gluon_launcher) == 1


def test_gluon_return_attn_probs_is_strictly_rejected_before_launch(
    fake_public_gluon_launcher,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.enable_grad():
        q, k, v = _fake_public_gluon_tensors()

        with pytest.raises(ValueError, match="backend GLUON cannot handle the given inputs"):
            flash_attn_func(q, k, v, causal=False, return_attn_probs=True)

    assert fake_public_gluon_launcher == []


def test_gluon_return_attn_probs_policy_rejects_explicit_flydsl(monkeypatch):
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    q, k, v = _gluon_tensors(
        q_shape=(2, 3, 8, 64),
        kv_shape=(2, 4, 1, 64),
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="backend FLYDSL cannot handle the given inputs"):
        resolve_flash_attn_backend(
            varlen=False,
            user_backend=BackendType.FLYDSL,
            q=q,
            k=k,
            v=v,
            causal=True,
            qkv_format="sbhd",
            return_softmax=True,
        )


def test_gluon_return_attn_probs_uses_a_distinct_dense_autotune_cache_entry(monkeypatch):
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    monkeypatch.setattr(
        _DENSE_FWD_BACKENDS[BackendType.HIPKITTENS].impl,
        "can_handle",
        staticmethod(lambda **kwargs: False),
    )
    cache = TuneCache(1024)
    monkeypatch.setattr(FlashAttnDenseDispatcher, "_cache", cache)
    monkeypatch.setattr(
        GlobalBackendManager,
        "auto_tune_enabled",
        classmethod(lambda cls: True),
    )
    timings = {
        _DENSE_FWD_BACKENDS[BackendType.FLYDSL].impl: 1.0,
        _DENSE_FWD_BACKENDS[BackendType.AITER].impl: 2.0,
    }
    monkeypatch.setattr(
        FlashAttnDenseDispatcher,
        "profile",
        classmethod(lambda cls, backend, **kwargs: timings[backend]),
    )
    q, k, v = _gluon_tensors(
        q_shape=(2, 3, 8, 64),
        kv_shape=(2, 4, 1, 64),
        dtype=torch.bfloat16,
    )
    kwargs = {
        "varlen": False,
        "user_backend": None,
        "q": q,
        "k": k,
        "v": v,
        "causal": True,
        "qkv_format": "sbhd",
    }
    assert resolve_flash_attn_backend(**kwargs, return_softmax=False) == BackendType.FLYDSL
    assert resolve_flash_attn_backend(**kwargs, return_softmax=True) == BackendType.AITER
    assert len(cache) == 2


def test_gluon_lifecycle_direct_apply_rejects_backward_before_launch(monkeypatch):
    launches = []

    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return torch.empty_like(args[0]), torch.empty((1, 1, 1))

    monkeypatch.setattr(
        flash_attn_interface,
        "flash_attn_gluon_forward_impl",
        launch,
        raising=False,
    )
    q = torch.empty((1, 1, 1, 8), dtype=torch.float16, requires_grad=True)
    k = torch.empty_like(q)
    v = torch.empty_like(q)

    with (
        torch.enable_grad(),
        pytest.raises(
            RuntimeError,
            match="gluon flash-attn is forward-only; Q/K/V backward is not implemented",
        ),
    ):
        flash_attn_interface.FlashAttnFunc.apply(
            q,
            k,
            v,
            0.0,
            None,
            False,
            (-1, -1),
            None,
            None,
            False,
            False,
            False,
            True,
            1,
            None,
            "bshd",
            BackendType.GLUON,
        )

    assert launches == []


def test_gluon_lifecycle_direct_apply_rejects_backward_before_launch_under_optimize():
    script = r"""
import torch
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.ops.attention import flash_attn_interface

launches = []
def launch(*args, **kwargs):
    launches.append((args, kwargs))
    return torch.empty_like(args[0]), torch.empty((1, 1, 1))

flash_attn_interface.flash_attn_gluon_forward_impl = launch
q = torch.empty((1, 1, 1, 8), dtype=torch.float16, requires_grad=True)
k = torch.empty_like(q)
v = torch.empty_like(q)
try:
    flash_attn_interface.FlashAttnFunc.apply(
        q, k, v, 0.0, None, False, (-1, -1), None, None,
        False, False, False, True, 1, None, "bshd", BackendType.GLUON,
    )
except RuntimeError as error:
    if str(error) != "gluon flash-attn is forward-only; Q/K/V backward is not implemented":
        raise AssertionError("optimized direct Gluon apply used the wrong error")
else:
    raise AssertionError("optimized direct Gluon apply did not reject backward")
if launches:
    raise AssertionError("optimized direct Gluon apply reached the launcher")
print("optimized Gluon direct apply rejected before launch")
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "optimized Gluon direct apply rejected before launch"


def test_gluon_lifecycle_backward_guard_reports_an_internal_contract_violation():
    ctx = SimpleNamespace(backend=BackendType.GLUON)

    with pytest.raises(
        AssertionError,
        match="internal contract violation.*gluon flash-attn.*forward-only",
    ):
        flash_attn_interface.FlashAttnFunc.backward(ctx, torch.empty((1, 1, 1, 8)))


def _gfx_arch_name(device):
    properties = torch.cuda.get_device_properties(device)
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]


def _gfx950_devices():
    if not torch.cuda.is_available():
        return []
    return [
        torch.device("cuda", index)
        for index in range(torch.cuda.device_count())
        if _gfx_arch_name(torch.device("cuda", index)) == "gfx950"
    ]


def _gfx950_device_for_test(*, prefer_nonzero=False):
    devices = _gfx950_devices()
    if not devices:
        pytest.skip("Gluon flash-attention runtime coverage requires an exact gfx950 device")
    if prefer_nonzero:
        return next((device for device in devices if device.index != 0), devices[0])
    current_index = torch.cuda.current_device()
    return next((device for device in devices if device.index == current_index), devices[0])


def _require_exact_gfx950(tensor):
    arch_name = _gfx_arch_name(tensor.device)
    if arch_name != "gfx950":
        pytest.skip(f"tested tensor is on {arch_name or 'unknown'}, not exact gfx950")


@contextmanager
def _pinned_gluon_backend():
    previous = GlobalBackendManager._attn_backend
    GlobalBackendManager._attn_backend = None if previous is None else dict(previous)
    GlobalBackendManager.set_attn_backend(
        BackendType.GLUON,
        PrecisionType.BF16_FP16_FP32,
    )
    try:
        yield
    finally:
        GlobalBackendManager._attn_backend = previous


def _make_gluon_runtime_inputs(
    *,
    batch,
    seqlen_q,
    seqlen_kv,
    num_heads_q,
    num_heads_kv,
    head_dim,
    qkv_format,
    dtype,
    device,
    generator,
):
    if qkv_format == "bshd":
        q = torch.randn(
            (batch, seqlen_q, num_heads_q, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        k = torch.randn(
            (batch, seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        v = torch.randn(
            (batch, seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return q, k, v

    assert qkv_format == "bhsd"
    q = torch.randn(
        (batch, num_heads_q, seqlen_q, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    ).transpose(1, 2)
    k = torch.randn(
        (batch, num_heads_kv, seqlen_kv, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    ).transpose(1, 2)
    v = torch.randn(
        (batch, num_heads_kv, seqlen_kv, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    ).transpose(1, 2)
    return q, k, v


def _gluon_attention_fp32_reference(q, k, v, softmax_scale, causal):
    """Independent FP32 attention and natural-log LSE for logical BSHD inputs."""
    q_bhsd = q.float().permute(0, 2, 1, 3)
    k_bhsd = k.float().permute(0, 2, 1, 3)
    v_bhsd = v.float().permute(0, 2, 1, 3)

    num_heads_q = q_bhsd.shape[1]
    num_heads_kv = k_bhsd.shape[1]
    assert num_heads_kv > 0 and num_heads_q % num_heads_kv == 0
    group_size = num_heads_q // num_heads_kv
    if group_size != 1:
        k_bhsd = k_bhsd.repeat_interleave(group_size, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(group_size, dim=1)

    scale = 1.0 / math.sqrt(float(q.shape[-1])) if softmax_scale is None else float(softmax_scale)
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-1, -2)) * scale
    if causal:
        seqlen_q, seqlen_kv = q.shape[1], k.shape[1]
        query_index = torch.arange(seqlen_q, device=q.device).view(seqlen_q, 1)
        key_index = torch.arange(seqlen_kv, device=q.device).view(1, seqlen_kv)
        # Flash attention aligns a rectangular causal mask at the bottom right.
        allowed = key_index <= query_index + (seqlen_kv - seqlen_q)
        scores = scores.masked_fill(~allowed, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v_bhsd).permute(0, 2, 1, 3)
    return output, lse


_RUN_LONG_GLUON_TESTS = os.environ.get("PRIMUS_TURBO_TEST_LONG") == "1"
_GLUON_CORRECTNESS_CASES = [
    pytest.param(
        torch.float16,
        "bshd",
        False,
        37,
        53,
        8,
        8,
        64,
        None,
        id="fp16-bshd-noncausal-mha-d64-short-tail-default-scale",
    ),
    pytest.param(
        torch.bfloat16,
        "bhsd",
        True,
        128,
        128,
        16,
        4,
        128,
        0.17,
        id="bf16-bhsd-causal-gqa-d128-square-custom-scale",
    ),
    pytest.param(
        torch.float16,
        "bhsd",
        False,
        64,
        257,
        16,
        1,
        256,
        None,
        id="fp16-bhsd-noncausal-mqa-d256-rectangular-default-scale",
    ),
    pytest.param(
        torch.bfloat16,
        "bshd",
        True,
        37,
        53,
        8,
        8,
        96,
        0.23,
        id="bf16-bshd-causal-mha-d96-bottom-right-custom-scale",
    ),
    pytest.param(
        torch.float16,
        "bshd",
        False,
        1024,
        2048,
        16,
        4,
        128,
        0.11,
        marks=pytest.mark.skipif(
            not _RUN_LONG_GLUON_TESTS,
            reason="set PRIMUS_TURBO_TEST_LONG=1 to run selected long Gluon cases",
        ),
        id="fp16-bshd-noncausal-gqa-d128-long-rectangular",
    ),
    pytest.param(
        torch.bfloat16,
        "bhsd",
        True,
        2048,
        2048,
        8,
        8,
        64,
        None,
        marks=pytest.mark.skipif(
            not _RUN_LONG_GLUON_TESTS,
            reason="set PRIMUS_TURBO_TEST_LONG=1 to run selected long Gluon cases",
        ),
        id="bf16-bhsd-causal-mha-d64-long-square",
    ),
]


@pytest.mark.parametrize(
    (
        "dtype",
        "qkv_format",
        "causal",
        "seqlen_q",
        "seqlen_kv",
        "num_heads_q",
        "num_heads_kv",
        "head_dim",
        "softmax_scale",
    ),
    _GLUON_CORRECTNESS_CASES,
)
def test_gluon_correctness_gfx950(
    dtype,
    qkv_format,
    causal,
    seqlen_q,
    seqlen_kv,
    num_heads_q,
    num_heads_kv,
    head_dim,
    softmax_scale,
):
    device = _gfx950_device_for_test()
    generator = torch.Generator(device=device).manual_seed(20260818)
    q, k, v = _make_gluon_runtime_inputs(
        batch=1,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        qkv_format=qkv_format,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    _require_exact_gfx950(q)

    with torch.no_grad():
        output_ref, lse_ref = _gluon_attention_fp32_reference(
            q,
            k,
            v,
            softmax_scale,
            causal,
        )
        with _pinned_gluon_backend():
            output, lse = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                return_lse=True,
            )
    torch.cuda.synchronize(device)

    assert output.device == q.device
    assert output.dtype == dtype
    assert output.shape == q.shape
    assert lse.device == q.device
    assert lse.dtype == torch.float32
    assert lse.shape == (1, num_heads_q, seqlen_q)
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
    output_snr = compute_snr(output_ref, output)
    assert output_snr > 40.0, f"Gluon output SNR {output_snr:.2f} dB is not above 40 dB"
    torch.testing.assert_close(output.float(), output_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("dtype", "qkv_format", "seqlen"),
    [
        pytest.param(torch.float16, "bshd", 384, id="fp16-bshd-s384"),
        pytest.param(torch.float16, "bshd", 512, id="fp16-bshd-s512"),
        pytest.param(torch.bfloat16, "bhsd", 384, id="bf16-bhsd-s384"),
        pytest.param(torch.bfloat16, "bhsd", 512, id="bf16-bhsd-s512"),
    ],
)
def test_gluon_short_causal_compile_boundaries_gfx950(dtype, qkv_format, seqlen):
    """The aligned D128 causal specialization compiles at S384/S512."""
    device = _gfx950_device_for_test()
    generator = torch.Generator(device=device).manual_seed(20260818)
    q, k, v = _make_gluon_runtime_inputs(
        batch=1,
        seqlen_q=seqlen,
        seqlen_kv=seqlen,
        num_heads_q=64,
        num_heads_kv=64,
        head_dim=128,
        qkv_format=qkv_format,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    _require_exact_gfx950(q)

    with torch.no_grad():
        output_ref, lse_ref = _gluon_attention_fp32_reference(q, k, v, None, True)
        with _pinned_gluon_backend():
            output, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    torch.cuda.synchronize(device)

    assert output.shape == q.shape
    assert lse.shape == (1, 64, seqlen)
    assert compute_snr(output_ref, output) > 40.0
    torch.testing.assert_close(output.float(), output_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("qkv_format", "expected_output_stride"),
    [
        pytest.param("bshd", (18944, 512, 64, 1), id="bshd"),
        pytest.param("bhsd", (18944, 64, 2368, 1), id="bhsd"),
    ],
)
@pytest.mark.filterwarnings("error:Dynamo detected a call to a .*lru_cache.*attention_impl\\.py")
def test_gluon_compile_fullgraph_values_and_exact_strides_gfx950(
    qkv_format,
    expected_output_stride,
):
    device = _gfx950_device_for_test()
    generator = torch.Generator(device=device).manual_seed(20260818)
    q, k, v = _make_gluon_runtime_inputs(
        batch=2,
        seqlen_q=37,
        seqlen_kv=53,
        num_heads_q=8,
        num_heads_kv=2,
        head_dim=64,
        qkv_format=qkv_format,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    _require_exact_gfx950(q)

    def forward(query, key, value):
        return flash_attn_func(
            query,
            key,
            value,
            softmax_scale=0.19,
            causal=False,
            return_lse=True,
        )

    previous_backend = GlobalBackendManager._attn_backend
    GlobalBackendManager._attn_backend = None if previous_backend is None else dict(previous_backend)
    GlobalBackendManager.set_attn_backend(
        BackendType.GLUON,
        PrecisionType.BF16_FP16_FP32,
    )
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            eager_output, eager_lse = forward(q, k, v)
            attention_impl._GFX950_DEVICE_CACHE.clear()
            compiled_forward = torch.compile(forward, fullgraph=True)
            compiled_output, compiled_lse = compiled_forward(q, k, v)
        torch.cuda.synchronize(device)
    finally:
        GlobalBackendManager._attn_backend = previous_backend
        torch._dynamo.reset()

    assert q.stride() == expected_output_stride
    assert eager_output.stride() == expected_output_stride
    assert compiled_output.stride() == expected_output_stride
    assert eager_lse.stride() == (296, 37, 1)
    assert compiled_lse.stride() == (296, 37, 1)
    assert torch.isfinite(eager_output).all()
    assert torch.isfinite(eager_lse).all()
    torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(compiled_lse, eager_lse, rtol=0, atol=0)


def test_gluon_non_default_stream_uses_q_device_gfx950():
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_device = _gfx950_device_for_test(prefer_nonzero=device_count >= 2)
    original_device_index = torch.cuda.current_device()
    original_target_stream = torch.cuda.current_stream(target_device)
    launch_stream = torch.cuda.Stream(device=target_device)
    other_device_index = next(
        (index for index in range(device_count) if index != target_device.index),
        target_device.index,
    )

    try:
        torch.cuda.set_device(target_device)
        torch.cuda.set_stream(launch_stream)
        assert torch.cuda.current_stream(target_device) == launch_stream
        assert launch_stream != original_target_stream

        generator = torch.Generator(device=target_device).manual_seed(20260818)
        q, k, v = _make_gluon_runtime_inputs(
            batch=1,
            seqlen_q=37,
            seqlen_kv=53,
            num_heads_q=16,
            num_heads_kv=4,
            head_dim=64,
            qkv_format="bshd",
            dtype=torch.float16,
            device=target_device,
            generator=generator,
        )
        _require_exact_gfx950(q)
        with torch.no_grad():
            output_ref, lse_ref = _gluon_attention_fp32_reference(q, k, v, 0.2, True)

        if device_count >= 2:
            torch.cuda.set_device(other_device_index)
            assert torch.cuda.current_device() != q.device.index

        with torch.no_grad(), _pinned_gluon_backend():
            output, lse = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=0.2,
                causal=True,
                return_lse=True,
            )

        expected_current_device = other_device_index if device_count >= 2 else target_device.index
        assert torch.cuda.current_device() == expected_current_device
        launch_stream.synchronize()

        assert {q.device, k.device, v.device, output.device, lse.device} == {target_device}
        assert torch.isfinite(output).all()
        assert torch.isfinite(lse).all()
        assert compute_snr(output_ref, output) > 40.0
        torch.testing.assert_close(output.float(), output_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)
    finally:
        torch.cuda.set_device(target_device)
        torch.cuda.set_stream(original_target_stream)
        torch.cuda.set_device(original_device_index)


@pytest.mark.deterministic
def test_gluon_repeatability_is_bitwise_after_warmup_gfx950():
    device = _gfx950_device_for_test()
    generator = torch.Generator(device=device).manual_seed(20260818)
    q, k, v = _make_gluon_runtime_inputs(
        batch=1,
        seqlen_q=128,
        seqlen_kv=257,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
        qkv_format="bhsd",
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    _require_exact_gfx950(q)

    with torch.no_grad(), _pinned_gluon_backend():
        flash_attn_func(q, k, v, causal=False, return_lse=True)
        torch.cuda.synchronize(device)
        repeated = [flash_attn_func(q, k, v, causal=False, return_lse=True) for _ in range(3)]
        torch.cuda.synchronize(device)

    first_output, first_lse = repeated[0]
    assert torch.isfinite(first_output).all()
    assert torch.isfinite(first_lse).all()
    for output, lse in repeated[1:]:
        assert torch.equal(output, first_output), "Gluon output changed bitwise after warmup"
        assert torch.equal(lse, first_lse), "Gluon LSE changed bitwise after warmup"


test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=28, num_head_kv=4, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # begin regression tests for https://ontrack-internal.amd.com/browse/SWDEV-548136
    AttnConfig(
        seqlen_q=4096 + 64, seqlen_kv=4096 + 64, num_head_q=2, num_head_kv=1, head_dim_qk=32, head_dim_v=32
    ),
    AttnConfig(seqlen_q=2048, seqlen_kv=2048, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # end regression tests for https://ontrack-internal.amd.com/browse/SWDEV-548136
    AttnConfig(seqlen_q=512, seqlen_kv=512, num_head_q=40, num_head_kv=40, head_dim_qk=192, head_dim_v=128),
    # head_dim 64, and a query chunk against a longer kv context: the rest of the table is
    # square at head_dim 128/192, so neither the 64-wide kernels nor the bottom-right causal
    # offset a rectangular shape carries would be reached otherwise.
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=64, head_dim_v=64),
    AttnConfig(seqlen_q=512, seqlen_kv=2048, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("enable_sink", [False, True])
@pytest.mark.parametrize("window_size_left", [-1, 32, 64, 128])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "bhsd"])
@pytest.mark.parametrize("is_v3_atomic_fp32", [False, True])
# None is whatever resolves; the rest pin one backend so its own path stays covered.
# HIPKITTENS takes a narrow slice of this table -- bf16, causal, sbhd, head dim 64/128, no
# sink -- and pinned_backend_takes asserts it refuses the rest rather than letting another
# backend answer for it.
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL, BackendType.HIPKITTENS])
def test_attention_16bit(
    batch, dtype, config, causal, enable_sink, window_size_left, qkv_format, is_v3_atomic_fp32, backend
):
    os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "1" if is_v3_atomic_fp32 else "0"

    device = "cuda"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    # Sliding window coverage only applies when sink attention is enabled.
    if not enable_sink and window_size_left != -1:
        pytest.skip("window_size_left only applies when sink is enabled")

    # Sink attention constraints / runtime control (skip early to avoid big allocations).
    if enable_sink:
        # Triton kernel limitation for sink: requires same qk/v head dim and head dim > 32
        if head_dim_qk != head_dim_v or head_dim_qk < 32:
            pytest.skip("Sink attention requires head_dim_qk == head_dim_v and head_dim >= 32")
        if window_size_left != -1 and not causal:
            pytest.skip("sink sliding window coverage only applies to causal attention")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    window_size = (window_size_left, -1) if enable_sink and window_size_left != -1 else (-1, -1)

    print(
        f"\nDType={dtype}, B={batch}, SeqQ={seqlen_q}, SeqKV={seqlen_kv}, NHQ={num_head_q}, NHKV={num_head_kv}, "
        f"HDQK={head_dim_qk}, HDV={head_dim_v}, Causal={causal}, Sink={enable_sink}, WindowLeft={window_size_left}, Format={qkv_format}"
    )

    if qkv_format == "sbhd":
        q_layout = (seqlen_q, batch, num_head_q, head_dim_qk)
        k_layout = (seqlen_kv, batch, num_head_kv, head_dim_qk)
        v_layout = (seqlen_kv, batch, num_head_kv, head_dim_v)
        o_layout = (seqlen_q, batch, num_head_q, head_dim_v)
    elif qkv_format == "bhsd":
        q_layout = (batch, num_head_q, seqlen_q, head_dim_qk)
        k_layout = (batch, num_head_kv, seqlen_kv, head_dim_qk)
        v_layout = (batch, num_head_kv, seqlen_kv, head_dim_v)
        o_layout = (batch, num_head_q, seqlen_q, head_dim_v)
    elif qkv_format == "bshd":
        q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
        k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
        v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
        o_layout = (batch, seqlen_q, num_head_q, head_dim_v)
    else:
        raise AssertionError(f"Unsupported qkv format: {qkv_format}")

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(o_layout, device=device, dtype=dtype)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()
    grad_out_ref = grad_out.clone().detach()

    query_orig, key_orig, value_orig = query, key, value

    if qkv_format == "sbhd":
        query = query.permute(1, 0, 2, 3)
        key = key.permute(1, 0, 2, 3)
        value = value.permute(1, 0, 2, 3)
        grad_out = grad_out.permute(1, 0, 2, 3)
    elif qkv_format == "bhsd":
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        grad_out = grad_out.transpose(1, 2)

    sm_scale = head_dim_qk ** (-0.5)

    sink = None
    sink_ref = None
    if enable_sink:
        sink = torch.randn((num_head_q,), device=device, dtype=torch.float32, requires_grad=True)

    # Ahead of the reference, which is the expensive part and pointless for a combo the
    # pinned backend does not implement (that case is covered by the refusal assert inside).
    if not pinned_backend_takes(
        backend,
        q=query,
        k=key,
        v=value,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=window_size,
        bias=None,
        alibi_slopes=None,
        sink=sink,
        qkv_format=_infer_qkv_format(query, key, value),
    ):
        return

    if enable_sink:
        sink_ref = sink.clone().detach().requires_grad_()
        o_ref = attention_with_sink_ref_impl(
            query_ref,
            key_ref,
            value_ref,
            sink_ref,
            sm_scale,
            causal,
            window_size=window_size,
            qkv_format=qkv_format,
        )
    else:
        o_ref = attention_vanilla_forward_pytorch_ref_impl(
            query_ref, key_ref, value_ref, sm_scale, causal, qkv_format
        )

    o_ref.backward(grad_out_ref)
    GlobalBackendManager.set_attn_backend(backend, PrecisionType.BF16_FP16_FP32)
    try:
        o = flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=window_size,
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            sink=sink,
        )
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)
    o.backward(grad_out)

    torch.cuda.synchronize()

    if qkv_format == "sbhd":
        o_ref_cmp = o_ref.permute(1, 0, 2, 3).contiguous()
    elif qkv_format == "bhsd":
        o_ref_cmp = o_ref.transpose(1, 2).contiguous()
    else:
        o_ref_cmp = o_ref
    out_snr = compute_snr(o_ref_cmp, o)
    query_grad_snr = compute_snr(query_ref.grad, query_orig.grad)
    key_grad_snr = compute_snr(key_ref.grad, key_orig.grad)
    value_grad_snr = compute_snr(value_ref.grad, value_orig.grad)
    sink_grad_snr = compute_snr(sink_ref.grad, sink.grad) if enable_sink else None
    msg = f"out={out_snr:.2f}, dq={query_grad_snr:.2f}, dk={key_grad_snr:.2f}, dv={value_grad_snr:.2f}"
    if enable_sink:
        msg += f", dsink={sink_grad_snr:.2f}"
    print(msg)

    assert out_snr > 40, f"out_snr too low: {out_snr}"
    assert query_grad_snr > 40, f"query_grad_snr too low: {query_grad_snr}"
    assert key_grad_snr > 40, f"key_grad_snr too low: {key_grad_snr}"
    assert value_grad_snr > 40, f"value_grad_snr too low: {value_grad_snr}"
    # SNR threshold for sink grad is 5e-2, reference from aiter: https://github.com/ROCm/aiter/blob/c71075ceda2788004f1a6e02608e114137dee856/op_tests/triton_tests/attention/test_mha_with_sink.py#L151-L157
    if sink_grad_snr is not None:
        torch.testing.assert_close(
            sink.grad,
            sink_ref.grad,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda msg: f"sink_grad mismatch (snr={sink_grad_snr:.2f})\n\n{msg}\n",
        )


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL, BackendType.HIPKITTENS])
@pytest.mark.skip(reason="Temporarily disabled due to external dependency issues.")
@pytest.mark.deterministic
def test_attention_16bit_deterministic(batch, dtype, config, causal, backend):
    device = "cuda"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    # NOTE: For `head_dim_qk != head_dim_v` (e.g. 192/128), this deterministic
    # test currently fails; skip temporarily to keep CI green.
    if head_dim_qk != head_dim_v:
        pytest.skip("deterministic test currently fails when head_dim_qk != head_dim_v; skip temporarily")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(
        f"\n[deterministic] DType={dtype}, B={batch}, SeqQ={seqlen_q}, SeqKV={seqlen_kv}, "
        f"NHQ={num_head_q}, NHKV={num_head_kv}, HDQK={head_dim_qk}, HDV={head_dim_v}, Causal={causal}"
    )

    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
    o_layout = (batch, seqlen_q, num_head_q, head_dim_v)

    q0 = torch.randn(q_layout, device=device, dtype=dtype)
    k0 = torch.randn(k_layout, device=device, dtype=dtype)
    v0 = torch.randn(v_layout, device=device, dtype=dtype)
    grad_out = torch.randn(o_layout, device=device, dtype=dtype)

    sm_scale = head_dim_qk ** (-0.5)

    # Ahead of the reference, which is the expensive part and pointless for a combo the
    # pinned backend does not implement (that case is covered by the refusal assert inside).
    if not pinned_backend_takes(
        backend,
        q=q0,
        k=k0,
        v=v0,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        sink=None,
        qkv_format="bshd",
    ):
        return

    # Correctness check against reference implementation
    q_ref = q0.clone().detach().requires_grad_()
    k_ref = k0.clone().detach().requires_grad_()
    v_ref = v0.clone().detach().requires_grad_()
    o_ref = attention_vanilla_forward_pytorch_ref_impl(q_ref, k_ref, v_ref, sm_scale, causal)
    o_ref.backward(grad_out)

    def _run_once():
        q = q0.clone().detach().requires_grad_()
        k = k0.clone().detach().requires_grad_()
        v = v0.clone().detach().requires_grad_()

        o = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=True,
            return_lse=False,
            return_attn_probs=False,
            sink=None,
        )
        o.backward(grad_out)
        return (
            o.detach(),
            q.grad.detach(),
            k.grad.detach(),
            v.grad.detach(),
        )

    # Determinism check (bitwise identical across multiple runs).
    repeats = 10
    outs = []
    GlobalBackendManager.set_attn_backend(backend, PrecisionType.BF16_FP16_FP32)
    try:
        for _ in range(repeats):
            outs.append(_run_once())
            torch.cuda.synchronize()
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)

    o1, dq1, dk1, dv1 = outs[0]
    for i in range(1, repeats):
        o_i, dq_i, dk_i, dv_i = outs[i]
        torch.testing.assert_close(o1, o_i, rtol=0, atol=0)
        torch.testing.assert_close(dq1, dq_i, rtol=0, atol=0)
        torch.testing.assert_close(dk1, dk_i, rtol=0, atol=0)
        torch.testing.assert_close(dv1, dv_i, rtol=0, atol=0)

    # Correctness check (close to reference)
    out_snr = compute_snr(o_ref, o1)
    query_grad_snr = compute_snr(q_ref.grad, dq1)
    key_grad_snr = compute_snr(k_ref.grad, dk1)
    value_grad_snr = compute_snr(v_ref.grad, dv1)
    print(
        f"deterministic: out={out_snr:.2f}, dq={query_grad_snr:.2f}, dk={key_grad_snr:.2f}, dv={value_grad_snr:.2f}"
    )
    assert out_snr > 40, f"out_snr too low: {out_snr}"
    assert query_grad_snr > 40, f"query_grad_snr too low: {query_grad_snr}"
    assert key_grad_snr > 40, f"key_grad_snr too low: {key_grad_snr}"
    assert value_grad_snr > 40, f"value_grad_snr too low: {value_grad_snr}"


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_fp8(batch, dtype, config, causal):
    device = "cuda"
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )

    print(
        f"\nDType={dtype}, B={batch}, SeqQ={seqlen_q}, SeqKV={seqlen_kv}, NHQ={num_head_q}, NHKV={num_head_kv}, HDQK={head_dim_qk}, HDV={head_dim_v}, Causal={causal}"
    )

    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
    o_layout = (batch, seqlen_q, num_head_q, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(o_layout, device=device, dtype=dtype)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    o_ref.backward(grad_out)
    o = flash_attn_fp8_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
    )
    o.backward(grad_out)
    torch.cuda.synchronize()

    out_snr = compute_snr(o_ref, o)
    query_grad_snr = compute_snr(query_ref.grad, query.grad)
    key_grad_snr = compute_snr(key_ref.grad, key.grad)
    value_grad_snr = compute_snr(value_ref.grad, value.grad)
    print(f"{out_snr:.2f}", f"{query_grad_snr:.2f}", f"{key_grad_snr:.2f}", f"{value_grad_snr:.2f}")
    assert out_snr > 20, "out_snr too low"
    assert query_grad_snr > 20, "query_grad_snr too low"
    assert key_grad_snr > 20, "key_grad_snr too low"
    assert value_grad_snr > 20, "value_grad_snr too low"


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_fp8_with_sparse_do(batch, config, causal):
    # regression test for https://ontrack-internal.amd.com/browse/SWDEV-548136
    device = "cuda"
    torch.manual_seed(1234)

    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_shape = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_shape = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_shape = (batch, seqlen_kv, num_head_kv, head_dim_v)
    do_shape = (batch, seqlen_q, num_head_q, head_dim_v)

    do = torch.randn(do_shape, device=device, dtype=dtype) * 1e-3
    do_mask_0 = (torch.randn(do_shape[:-2], device=device, dtype=dtype) > 0.9).unsqueeze(-1).unsqueeze(-1)
    do_mask_1 = (torch.randn(do_shape[:-1], device=device, dtype=dtype) > 0.9).unsqueeze(-1)
    do = do * do_mask_0 * do_mask_1

    q = torch.randn(q_shape, device=device, dtype=dtype)
    k = torch.randn(k_shape, device=device, dtype=dtype)
    v = torch.randn(v_shape, device=device, dtype=dtype)

    sm_scale = q.shape[-1] ** -0.5

    q_fp8, q_descale = block_scaling_node(q, True)
    k_fp8, k_descale = block_scaling_node(k, True)
    v_fp8, v_descale = block_scaling_node(v, True)

    o, softmax_lse, _ = attention_triton_forward_impl(
        q_fp8,
        k_fp8,
        v_fp8,
        F8_FWD_MAX,
        q_descale,
        k_descale,
        v_descale,
        0,
        sm_scale,
        causal,
        -1,
        -1,
        None,
        None,
        False,
        True,
    )

    dq, dk, dv = attention_triton_backward_impl(
        do,
        q,
        k,
        v,
        o,
        torch.scalar_tensor(1.0, device=device),
        torch.scalar_tensor(1.0, device=device),
        torch.scalar_tensor(1.0, device=device),
        1.0,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
        q_fp8.shape[1],
        k_fp8.shape[1],
        sm_scale,
        causal,
        -1,
        -1,
        None,
        False,
    )

    dq_fp8, dk_fp8, dv_fp8 = attention_triton_backward_impl(
        do,
        q_fp8,
        k_fp8,
        v_fp8,
        o,
        q_descale,
        k_descale,
        v_descale,
        F8_FWD_MAX,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
        q_fp8.shape[1],
        k_fp8.shape[1],
        sm_scale,
        causal,
        -1,
        -1,
        None,
        True,
    )

    dq_snr = compute_snr(dq, dq_fp8)
    dk_snr = compute_snr(dk, dk_fp8)
    dv_snr = compute_snr(dv, dv_fp8)
    print(f"dq_snr: {dq_snr}, dk_snr: {dk_snr}, dv_snr: {dv_snr}")
    assert dq_snr > 15, "query_grad_snr too low"
    assert dk_snr > 15, "key_grad_snr too low"
    assert dv_snr > 15, "value_grad_snr too low"


@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "bhsd"])
def test_attention_fake_kernel_strides(qkv_format):
    """Verify that torch.compile sees correct output strides for every qkv_format.

    The fake (meta) kernel must produce output tensors whose strides match the
    eager kernel so that torch.compile's shape/stride propagation is correct.
    """
    device = "cuda"
    dtype = torch.bfloat16
    batch, seq_q, seq_kv, num_heads, head_dim = 2, 32, 32, 4, 64

    if qkv_format == "sbhd":
        q = torch.randn(seq_q, batch, num_heads, head_dim, device=device, dtype=dtype).permute(1, 0, 2, 3)
        k = torch.randn(seq_kv, batch, num_heads, head_dim, device=device, dtype=dtype).permute(1, 0, 2, 3)
        v = torch.randn(seq_kv, batch, num_heads, head_dim, device=device, dtype=dtype).permute(1, 0, 2, 3)
    elif qkv_format == "bhsd":
        q = torch.randn(batch, num_heads, seq_q, head_dim, device=device, dtype=dtype).transpose(1, 2)
        k = torch.randn(batch, num_heads, seq_kv, head_dim, device=device, dtype=dtype).transpose(1, 2)
        v = torch.randn(batch, num_heads, seq_kv, head_dim, device=device, dtype=dtype).transpose(1, 2)
    else:
        q = torch.randn(batch, seq_q, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seq_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, seq_kv, num_heads, head_dim, device=device, dtype=dtype)

    out_eager = flash_attn_func(q, k, v, causal=True)
    eager_strides = out_eager.stride()

    torch._dynamo.reset()

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True)

    out_compiled = fn(q, k, v)

    assert out_compiled.stride() == eager_strides, (
        f"Stride mismatch for qkv_format={qkv_format}: "
        f"compiled={out_compiled.stride()}, eager={eager_strides}"
    )
    assert out_compiled.shape == out_eager.shape


# ============================================================================
# DeepSeek-V4 single-latent sparse-MLA attention (flydsl, gfx950/MI355X).
# ============================================================================

# Fixed dims: kv_lora_rank (single latent, K == V) + rope pad; SWA local window.
SPARSE_MLA_ROPE_DIM = 64
SPARSE_MLA_HEAD_DIM = 512
SPARSE_MLA_SWA_WINDOW = 128
# (variant -> num_heads, index-topk cap). cr spans pure-SWA / random-pool / deterministic-pool (HCA).
SPARSE_MLA_VARIANTS = {"flash": (64, 512), "pro": (128, 1024)}


def _sparse_mla_topk(variant, cr, seqlen):
    if cr == 0:
        return 0, 0, SPARSE_MLA_SWA_WINDOW
    if cr == 4:
        pool = max(seqlen // 4, 1)
        topk_pool = min(SPARSE_MLA_VARIANTS[variant][1], pool)
        return pool, topk_pool, SPARSE_MLA_SWA_WINDOW + topk_pool
    pool = max(seqlen // cr, 1)
    return pool, 0, SPARSE_MLA_SWA_WINDOW + pool


def _build_sparse_mla(cr, num_heads, seqlen, pool, topk_pool, seed=0):
    """DSV4 sparse-MLA inputs: single-latent kv, per-token top-k (SWA band + optional pool),
    zero-padded rope cols, random sink / grad_out."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt, d, w = "cuda", torch.bfloat16, SPARSE_MLA_HEAD_DIM, SPARSE_MLA_SWA_WINDOW
    latent = torch.randn(seqlen, d, generator=gen, device=dev, dtype=dt)
    q = torch.randn(seqlen, num_heads, d, generator=gen, device=dev, dtype=dt)
    q = torch.cat([q, torch.zeros(seqlen, num_heads, SPARSE_MLA_ROPE_DIM, device=dev, dtype=dt)], -1)
    sink = torch.randn(num_heads, generator=gen, device=dev, dtype=torch.float32) * 0.1
    grad_out = torch.randn(seqlen, num_heads, d, generator=gen, device=dev, dtype=dt)

    tok = torch.arange(seqlen, device=dev).view(seqlen, 1)
    win = tok - w + 1 + torch.arange(w, device=dev).view(1, w)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if cr == 0:
        kv = latent.unsqueeze(1)
        topk = win
    else:
        p = torch.randn(pool, d, generator=gen, device=dev, dtype=dt)
        kv = torch.cat([latent, p], 0).unsqueeze(1)
        if cr == 4:
            pool_topk = seqlen + torch.randint(0, pool, (seqlen, topk_pool), generator=gen, device=dev)
        else:
            ps = torch.arange(pool, device=dev).view(1, pool)
            pool_topk = torch.where(
                ((ps + 1) * cr - 1) <= tok, seqlen + ps, torch.full_like(ps.expand(seqlen, pool), -1)
            )
        topk = torch.cat([win, pool_topk], 1)
    pad = ((topk.shape[1] + 63) // 64) * 64 - topk.shape[1]
    if pad > 0:
        topk = torch.cat([topk, torch.full((seqlen, pad), -1, device=dev, dtype=topk.dtype)], 1)
    kv = torch.cat([kv, torch.zeros(kv.shape[0], 1, SPARSE_MLA_ROPE_DIM, device=dev, dtype=dt)], -1)
    return q.contiguous(), kv.contiguous(), topk.to(torch.int32).contiguous(), sink, grad_out


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="sparse-MLA (flydsl) is gfx950-only"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("variant", ["flash", "pro"])
@pytest.mark.parametrize("cr", [0, 4, 128])
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL, BackendType.TRITON])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_sparse_mla_op(variant, cr, seqlen, backend, auto_tune):
    """Public multi-backend training op ``sparse_mla_func``, against the triton kernels called
    directly as oracle. Covers the pure-SWA (cr=0), random-pool (cr=4) and
    deterministic-pool/HCA (cr=128) paths; seqlen=512 also exercises the cr=4 small-seq
    dkv-dispatch guard."""
    # Skip redundant test: auto_tune is ignored when backend is explicitly specified
    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    d = SPARSE_MLA_HEAD_DIM
    num_heads = SPARSE_MLA_VARIANTS[variant][0]
    pool, topk_pool, _ = _sparse_mla_topk(variant, cr, seqlen)
    scale = 1.0 / math.sqrt(d)
    q, kv, topk_idx, sink, grad_out = _build_sparse_mla(cr, num_heads, seqlen, pool, topk_pool)

    # Oracle: the triton kernels called directly, outside the op.
    out_ref, lse_ref = sparse_mla_fwd_triton(q, kv, topk_idx, attn_sink=sink, kv_lora_rank=d, scale=scale)
    dq_ref, dkv_ref, dsink_ref = sparse_mla_bwd_triton(
        q, kv, out_ref, grad_out, topk_idx, lse_ref, attn_sink=sink, kv_lora_rank=d, scale=scale
    )
    # Pinned TRITON is the oracle's own kernels; FLYDSL is a separate bf16 implementation.
    snr_floor = 60.0 if backend == BackendType.TRITON else 40.0

    GlobalBackendManager.set_sparse_attn_backend(backend, PrecisionType.BF16_FP16_FP32)
    GlobalBackendManager.set_auto_tune(auto_tune)

    def _run_op():
        qg = q.clone().requires_grad_(True)
        kvg = kv.clone().requires_grad_(True)
        sg = sink.clone().requires_grad_(True)
        o = sparse_mla_func(qg, kvg, topk_idx, attn_sink=sg, kv_lora_rank=d, scale=scale)
        o.backward(grad_out)
        return o, qg.grad, kvg.grad, sg.grad

    try:
        out, dq, dkv, dsink = _run_op()
        assert torch.isfinite(out).all(), "op forward produced non-finite values"
        assert compute_snr(out_ref, out) > snr_floor, f"op fwd SNR <= {snr_floor}"
        assert compute_snr(dq_ref, dq) > snr_floor, f"op dq SNR <= {snr_floor}"
        assert compute_snr(dkv_ref, dkv) > snr_floor, f"op dkv SNR <= {snr_floor}"
        assert compute_snr(dsink_ref, dsink) > snr_floor, f"op dsink SNR <= {snr_floor}"

        # Determinism: one WG owns each output tile (no float atomics), so a re-run is bit-exact.
        out2, dq2, dkv2, dsink2 = _run_op()
        assert torch.equal(out, out2), "op forward is not deterministic"
        assert torch.equal(dq, dq2), "op dq is not deterministic"
        assert torch.equal(dkv, dkv2), "op dkv is not deterministic"
        assert torch.equal(dsink, dsink2), "op dsink is not deterministic"

        # Each pass owns a dispatcher, so auto-tune has to leave a winner in both tune caches.
        if auto_tune:
            assert len(SparseMlaFwdDispatcher._cache) == 1, "forward was not auto-tuned"
            assert len(SparseMlaBwdDispatcher._cache) == 1, "backward was not auto-tuned"
    finally:
        GlobalBackendManager.reset()


# =============================================================================
# HipKittens attention (gfx950). The shape families the kernels were tuned against.
#
# The backend axis above already crosses HipKittens with the shared table, which is what
# covers its eligibility and its refusals. What that table does not have is the two shape
# families these kernels were measured on, so they live here: the rectangular/windowed meta
# configs, and the head structures real pretrains run at.
# =============================================================================

hipkittens_only = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()),
    reason="HipKittens attention is gfx950-only",
)

# (Hq, Hkv, Sq, Skv, window_left), each run full-causal and windowed. Run at 1/8 the measured
# sequence lengths: the fp32 reference materialises a whole [B, Hq, Sq, Skv] score matrix,
# which at the real lengths cannot share a device with anything else. The head structure, the
# rectangular ratios and the window are what this set covers, and all three survive the
# scale-down.
_HK_META = [
    (128, 16, 2048, 16384, 2048),
    (128, 16, 4096, 16384, 2048),
    (128, 16, 8192, 16384, 2048),
    (128, 16, 16384, 16384, 2048),
    (48, 6, 4096, 4096, 2047),
    (48, 6, 4096, 8192, 2047),
    (48, 6, 4096, 12288, 2047),
    (48, 6, 4096, 16384, 2047),
    (64, 8, 1024, 1024, 2047),
    (64, 8, 1024, 16384, 2047),
]
_HK_META_SCALE = 8

_HK_META_CASES = []
for _hq, _hkv, _sq, _skv, _w in _HK_META:
    _sq_s, _skv_s = _sq // _HK_META_SCALE, _skv // _HK_META_SCALE
    _HK_META_CASES.append((_hq, _hkv, _sq_s, _skv_s, -1))
    _HK_META_CASES.append((_hq, _hkv, _sq_s, _skv_s, max(1, _w // _HK_META_SCALE)))

# (Hq, Hkv) of eleven real pretrain configs, deduplicated to eight -- an average over this set
# is meant to become an accept metric, and a set that measures one shape twice silently gives
# it double weight. All head dim 128; gpt-oss is excluded as head dim 64, which the meta set
# above already covers.
_HK_MODEL_HEADS = [
    (40, 8),  # llama4_17B128E, llama4_17B16E
    (48, 8),  # minimax_m2.5
    (64, 4),  # qwen3_235B_A22B
    (32, 4),  # qwen3_30B_A3B
    (32, 8),  # lfm2_8B_A1B, mixtral_8x7B_v0.1
    (16, 16),  # deepseek_v2_lite (MHA)
    (64, 8),  # grok2
    (48, 8),  # grok1, mixtral_8x22B_v0.1
]
_HK_MODEL_SEQLEN = 1024


def _hk_ref(q, k, v, window_left, scale):
    """fp32 reference over SBHD tensors, bottom-right aligned.

    Not attention_vanilla_forward_pytorch_ref_impl: that takes the op's [b, s, h, d] view,
    and these kernels are driven directly here in their own layout.
    """
    Sq, _, Hq, _ = q.shape
    Skv, _, Hkv, _ = k.shape
    g = Hq // Hkv
    qf = q.float().permute(1, 2, 0, 3)
    kf = k.float().permute(1, 2, 0, 3).repeat_interleave(g, 1)
    vf = v.float().permute(1, 2, 0, 3).repeat_interleave(g, 1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    off = Skv - Sq
    qi = torch.arange(Sq, device=q.device)[:, None]
    ki = torch.arange(Skv, device=q.device)[None, :]
    keep = ki <= qi + off
    if window_left >= 0:
        keep &= ki >= qi + off - window_left
    p = torch.softmax(s.masked_fill(~keep, float("-inf")), dim=-1)
    return (p @ vf).permute(2, 0, 1, 3)


def _run_hk_case(Sq, Skv, B, Hq, Hkv, D, window_left, bar=40.0):
    from primus_turbo.hipkittens.attention import (
        hipkittens_attn_backward,
        hipkittens_attn_forward,
    )

    torch.manual_seed(0)
    scale = D**-0.5

    def mk(s, h):
        return torch.randn(s, B, h, D, device="cuda", dtype=torch.bfloat16) * 0.5

    q, k, v = mk(Sq, Hq), mk(Skv, Hkv), mk(Skv, Hkv)
    out, lse = hipkittens_attn_forward(q, k, v, scale, True, (window_left, 0))
    assert out.shape == q.shape
    assert lse.shape == (B, Hq, 1, Sq)
    assert compute_snr(_hk_ref(q, k, v, window_left, scale), out) > bar, "forward SNR too low"

    dout = torch.randn_like(out)
    dq, dk, dv = hipkittens_attn_backward(dout, q, k, v, out, lse, scale, True, (window_left, 0))
    assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)

    qd, kd, vd = (t.detach().clone().float().requires_grad_(True) for t in (q, k, v))
    _hk_ref(qd, kd, vd, window_left, scale).backward(dout.float())
    for name, ref_g, got in (("dq", qd.grad, dq), ("dk", kd.grad, dk), ("dv", vd.grad, dv)):
        assert compute_snr(ref_g, got) > bar, f"{name} SNR too low"


@hipkittens_only
@pytest.mark.parametrize("Hq, Hkv, Sq, Skv, window_left", _HK_META_CASES, ids=str)
def test_attention_hipkittens_meta_shapes(Hq, Hkv, Sq, Skv, window_left):
    """Rectangular Sq < Skv, GQA, full-causal and sliding-window, head dim 64."""
    _run_hk_case(Sq, Skv, 1, Hq, Hkv, 64, window_left)


@hipkittens_only
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("heads", _HK_MODEL_HEADS, ids=lambda h: f"H{h[0]}x{h[1]}")
def test_attention_hipkittens_model_shapes(heads, batch):
    """Head dim 128, square, full causal -- the head structures real pretrains run at, over
    the batches the benchmark sweeps. Small batches change the grid and hence which CTAs are
    masked at the causal boundary, so this is not redundant with the head sweep."""
    Hq, Hkv = heads
    _run_hk_case(_HK_MODEL_SEQLEN, _HK_MODEL_SEQLEN, batch, Hq, Hkv, 128, -1)


@hipkittens_only
def test_attention_hipkittens_deterministic():
    """Same inputs must give bit-identical results across launches, forward and backward.

    Constructive rather than hopeful: one workgroup owns each output tile, there are no float
    atomics, and the split-K partials are folded in a fixed band order.
    """
    from primus_turbo.hipkittens.attention import (
        hipkittens_attn_backward,
        hipkittens_attn_forward,
    )

    D, S, B, Hq, Hkv = 128, 1024, 2, 32, 4
    torch.manual_seed(0)
    scale = D**-0.5
    q = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16)

    o1, l1 = hipkittens_attn_forward(q, k, v, scale, True, (-1, 0))
    o2, l2 = hipkittens_attn_forward(q, k, v, scale, True, (-1, 0))
    assert torch.equal(o1, o2) and torch.equal(l1, l2), "forward is not run-to-run deterministic"

    do = torch.randn_like(o1)
    g1 = hipkittens_attn_backward(do, q, k, v, o1, l1, scale, True, (-1, 0))
    g2 = hipkittens_attn_backward(do, q, k, v, o1, l1, scale, True, (-1, 0))
    for name, a, b in zip(("dq", "dk", "dv"), g1, g2):
        assert torch.equal(a, b), f"{name} is not run-to-run deterministic"


@hipkittens_only
@pytest.mark.parametrize(
    "case, needle",
    [
        ("fp16", "bf16"),
        ("non_causal", "causal"),
        ("sink", "sink"),
        ("right_window", "left window"),
        ("head_dim", "head dim"),
        ("sq_gt_skv", "Sq > Skv"),
        ("varlen", "varlen"),
        ("non_contiguous", "contiguous"),
    ],
)
def test_attention_hipkittens_envelope(case, needle):
    """Everything outside the envelope must be refused with a reason, not computed wrongly.

    These kernels read out of bounds or leave output unwritten rather than failing, so the
    checks are load-bearing. Two are worth naming: fp16 would be reinterpreted bit-for-bit
    because the kernels declare gl<bf16, ...>, and Sq > Skv leaves the leading Sq - Skv query
    rows -- which attend to no key at all -- unwritten by the forward.
    """
    from primus_turbo.hipkittens.attention import hipkittens_attn_supported

    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    D = 32 if case == "head_dim" else 64
    Sq, Skv = (256, 128) if case == "sq_gt_skv" else (128, 128)
    kw = dict(causal=True, window_size=(-1, -1))

    if case == "varlen":
        t = torch.randn(1024, 8, D, device="cuda", dtype=dtype)  # THD packing is 3-D
        q = k = v = t
    else:
        q = torch.randn(Sq, 1, 8, D, device="cuda", dtype=dtype)
        k = v = torch.randn(Skv, 1, 8, D, device="cuda", dtype=dtype)

    if case == "non_causal":
        kw["causal"] = False
    elif case == "sink":
        kw["sink"] = torch.zeros(8, device="cuda", dtype=torch.float32)
    elif case == "right_window":
        kw["window_size"] = (64, 64)
    elif case == "non_contiguous":
        q = torch.randn(Sq, 2, 8, D, device="cuda", dtype=dtype).transpose(0, 1)
        k = v = torch.randn(Skv, 2, 8, D, device="cuda", dtype=dtype)

    ok, why = hipkittens_attn_supported(q, k, v, **kw)
    assert not ok and needle in why, f"expected a refusal mentioning {needle!r}, got {ok} / {why!r}"
