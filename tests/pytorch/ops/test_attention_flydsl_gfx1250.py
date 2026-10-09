###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL dense attention on gfx1250 (MI455X): gate, routing, backward launches, packaging and
numerics.

The gate, routing, launch and packaging tests patch the arch probe, the streams or the launches
(the version-gate tests also the flydsl module) and use meta / CPU tensors, so they run no GPU
kernels; importing primus_turbo still needs a GPU host. Tests that need the gate to accept a
call need a flydsl that satisfies the kernels' FLYDSL_REQUIREMENT (>=0.3.4.1,<0.3.5) and skip
under any other release. The numeric tests
need a gfx1250 card; their fp32 reference runs on the CPU.
"""

import contextlib
import importlib
import math
import os
import re
import subprocess
import sys
import types

import pytest
import torch

import primus_turbo.pytorch.ops.attention.flash_attn_interface as flash_attn_interface
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import attention_impl
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.ops import flash_attn_func
from tests.pytorch.test_utils import compute_snr, pinned_backend_takes

_PKG = "primus_turbo.flydsl.attention.gfx1250"


def _gfx1250_modules():
    return sorted(m for m in sys.modules if m == _PKG or m.startswith(_PKG + "."))


# Taken before this module imports the package below: importing primus_turbo must not.
_GFX1250_MODULES_AT_IMPORT = _gfx1250_modules()


def _flydsl_reason():
    from primus_turbo.flydsl.attention.gfx1250.flydsl_version import (
        flydsl_unavailable_reason,
    )

    return flydsl_unavailable_reason()


_FLYDSL_REASON = _flydsl_reason()
# Tests that need the gate to accept a call, i.e. a flydsl that satisfies FLYDSL_REQUIREMENT. Under
# another flydsl the gate declines every call, which test_gate_declines_without_flydsl_0_3_4
# covers.
needs_flydsl = pytest.mark.skipif(_FLYDSL_REASON is not None, reason=str(_FLYDSL_REASON))
_ON_GFX1250 = torch.cuda.is_available() and is_gfx1250()
needs_gfx1250 = pytest.mark.skipif(
    not _ON_GFX1250 or _FLYDSL_REASON is not None,
    reason="gfx1250 FlyDSL attention kernels: needs a gfx1250 card and flydsl>=0.3.4.1,<0.3.5",
)

D = 128


def _meta(b, sq, skv, hq, hkv, d=D, dtype=torch.bfloat16):
    q = torch.empty(b, sq, hq, d, dtype=dtype, device="meta")
    k = torch.empty(b, skv, hkv, d, dtype=dtype, device="meta")
    return q, k, torch.empty_like(k)


def _gate(monkeypatch, q, k, v, **kwargs):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    kwargs.setdefault("causal", True)
    return attention_impl.DenseAttnFwdFlydslBackend.can_handle(q, k=k, v=v, **kwargs)


class _LogRecorder:
    """Stands in for the adapter's logger: records (message, once) of each warning."""

    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, once=False, **kwargs):
        self.warnings.append((msg, once))


def _unresolved_adapter(monkeypatch, import_interface, record_log=True):
    """The gfx1250 adapter with its kernel-package import not yet resolved, so the next gate
    call resolves it again (under whatever flydsl the test fakes), importing the package
    through ``import_interface`` and, with ``record_log``, logging to a recorder."""
    from primus_turbo.pytorch.kernels.attention import (
        attention_flydsl_gfx1250_impl as gfx1250_impl,
    )

    log = _LogRecorder()
    monkeypatch.setattr(gfx1250_impl, "_INTERFACE", None)
    monkeypatch.setattr(gfx1250_impl, "_import_interface", import_interface)
    if record_log:
        monkeypatch.setattr(gfx1250_impl, "logger", log)
    return gfx1250_impl, log


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@needs_flydsl
@pytest.mark.parametrize(
    "shape",
    [
        (4, 8192, 8192, 32, 8),
        (1, 64, 64, 1, 1),
        (2, 1024, 2048, 8, 2),
        (1, 256, 256, 64, 64),
        (1, 256, 256, 64, 4),
        (1, 131072, 131072, 32, 8),
    ],
    ids=["llama3.1-8b", "smallest", "sq_lt_skv", "mha", "gqa16", "q_1gib"],
)
@pytest.mark.parametrize("causal", [True, False])
def test_gate_accepts(monkeypatch, shape, causal):
    q, k, v = _meta(*shape)
    assert _gate(monkeypatch, q, k, v, causal=causal) is True


@needs_flydsl
def test_gate_accepts_causal_spelled_as_a_window(monkeypatch):
    q, k, v = _meta(1, 256, 256, 4, 2)
    assert _gate(monkeypatch, q, k, v, window_size=(-1, 0)) is True


@needs_flydsl
@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((1, 256, 256, 4, 2, 128, torch.float16), {}),
        ((0, 256, 256, 4, 2), {}),
        ((1, 256, 256, 4, 2, 64), {}),
        ((1, 256, 256, 4, 2, 256), {}),
        ((1, 256, 256, 6, 4), {}),
        ((1, 256, 256, 28, 4), {}),
        ((1, 256, 256, 32, 1), {}),
        ((1, 96, 96, 4, 2), {}),
        ((1, 256, 240, 4, 2), {}),
        ((1, 512, 256, 4, 2), {}),
        ((2, 131072, 131072, 32, 8), {}),
        ((1, 64, 1048576, 8, 8), {}),
        ((1, 256, 256, 4, 2), {"dropout_p": 0.1}),
        ((1, 256, 256, 4, 2), {"bias": torch.empty(1)}),
        ((1, 256, 256, 4, 2), {"alibi_slopes": torch.empty(4)}),
        ((1, 256, 256, 4, 2), {"sink": torch.empty(4)}),
        ((1, 256, 256, 4, 2), {"window_size": (128, 0)}),
        ((1, 256, 256, 4, 2), {"window_size": (-1, 0), "causal": False}),
        ((1, 256, 256, 4, 2), {"return_softmax": True}),
        ((1, 256, 256, 4, 2), {"softmax_scale": True}),
    ],
    ids=[
        "fp16",
        "empty_batch",
        "d64",
        "d256",
        "hq_not_multiple_of_hkv",
        "gqa_7",
        "gqa_32",
        "sq_not_multiple_of_64",
        "skv_not_multiple_of_32",
        "causal_sq_gt_skv",
        "q_over_1gib",
        "k_over_1gib",
        "dropout",
        "bias",
        "alibi",
        "sink",
        "sliding_window",
        "window_without_causal",
        "return_softmax",
        "bool_scale",
    ],
)
def test_gate_refuses(monkeypatch, shape, kwargs):
    q, k, v = _meta(*shape)
    assert _gate(monkeypatch, q, k, v, **kwargs) is False


def test_gate_refuses_missing_kv(monkeypatch):
    q, k, v = _meta(1, 256, 256, 4, 2)
    assert _gate(monkeypatch, q, None, v) is False
    assert _gate(monkeypatch, q, k, None) is False


def test_gfx950_kernels_are_not_offered_to_other_archs(monkeypatch):
    """gfx1250 compares greater than gfx950; the gfx950 gate used to accept it."""
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (12, 5))
    q, k, v = (t.permute(1, 0, 2, 3) for t in _meta(1, 256, 256, 4, 2))  # sbhd bytes
    backend = attention_impl.DenseAttnFwdFlydslBackend
    assert backend.can_handle(q, k=k, v=v, causal=True, qkv_format="sbhd") is False


# ---------------------------------------------------------------------------
# Dispatch and routing
# ---------------------------------------------------------------------------


def _resolve_kwargs(q, k, v, causal=True):
    return dict(q=q, k=k, v=v, causal=causal, window_size=(-1, -1), qkv_format="bshd", needs_backward=True)


@needs_flydsl
def test_unpinned_gfx1250_call_resolves_to_flydsl(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    q, k, v = _meta(4, 8192, 8192, 32, 8)
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v)) == BackendType.FLYDSL


@needs_flydsl
def test_pinned_flydsl_refuses_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    q, k, v = _meta(1, 96, 96, 4, 2)  # sq not a multiple of 64
    assert pinned_backend_takes(BackendType.FLYDSL, **_resolve_kwargs(q, k, v)) is False


def test_dense_dispatcher_key_tells_apart_what_decides_eligibility():
    """The autotune cache key holds Skv, dropout, bias and alibi: FlyDSL and Triton refuse some
    of these calls, so an entry tuned without them would replay a backend that cannot take the
    call."""
    make_key = attention_impl.FlashAttnDenseDispatcher.make_key
    q, k, _ = _meta(1, 256, 256, 4, 2)
    k_longer = _meta(1, 256, 512, 4, 2)[1]
    base = dict(q=q, k=k, causal=True, window_size=(-1, -1), qkv_format="bshd")
    key = make_key(**base)
    assert make_key(**base) == key
    changes = {
        "skv": {"k": k_longer},
        "dropout": {"dropout_p": 0.1},
        "bias": {"bias": torch.empty(1)},
        "alibi": {"alibi_slopes": torch.empty(4)},
    }
    for name, change in changes.items():
        assert make_key(**{**base, **change}) != key, name


@needs_flydsl
def test_forward_and_backward_route_to_the_gfx1250_kernels(monkeypatch):
    """FlashAttnFunc on "gfx1250": the gfx1250 impls are called with contiguous BSHD tensors
    and lse [B, Hq, Sq], and the gfx950 sbhd impls are never reached."""
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)
    b, sq, skv, hq, hkv = 2, 128, 256, 4, 2
    calls = []

    def fwd(q, k, v, softmax_scale=None, causal=True):
        calls.append(("fwd", q.is_contiguous(), k.is_contiguous(), softmax_scale, causal))
        return torch.zeros_like(q), torch.zeros(b, hq, sq)

    def bwd(dout, q, k, v, out, lse, softmax_scale=None, causal=True):
        calls.append(("bwd", tuple(lse.shape), tuple(out.shape)))
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v)

    def forbidden(*args, **kwargs):
        raise AssertionError("gfx950 FlyDSL impl reached on gfx1250")

    monkeypatch.setattr(flash_attn_interface, "flash_attn_flydsl_gfx1250_forward_impl", fwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_flydsl_gfx1250_backward_impl", bwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_sbhd_flydsl_forward_impl", forbidden)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_sbhd_flydsl_backward_impl", forbidden)

    # sbhd bytes on purpose: the adapter must hand the kernels contiguous BSHD.
    q = torch.randn(sq, b, hq, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    k = torch.randn(skv, b, hkv, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    v = torch.randn(skv, b, hkv, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    try:
        out = flash_attn_func(q, k, v, causal=True)
        out.sum().backward()
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)

    assert calls[0] == ("fwd", True, True, None, True)
    assert calls[1] == ("bwd", (b, hq, sq), (b, sq, hq, D))
    assert k.grad.shape == (b, skv, hkv, D) and torch.all(k.grad == 1)


@needs_flydsl
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("first_call", [False, True], ids=["resolved", "first_call"])
def test_torch_compile_traces_the_custom_ops(monkeypatch, dynamic, first_call):
    """torch.compile(fullgraph) of FlashAttnFunc on meta tensors: the gate, flydsl version check
    included, traces without a graph break -- also as the process's first gate call, which
    imports the kernel package -- and the forward and backward AOT graphs each hold the gfx1250
    custom op once, whose fake impls give the eager shapes. No kernel runs."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    if first_call:
        from primus_turbo.pytorch.kernels.attention import (
            attention_flydsl_gfx1250_impl as gfx1250_impl,
        )

        monkeypatch.setattr(gfx1250_impl, "_INTERFACE", None)
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)
    b, sq, skv, hq, hkv = 2, 128, 256, 4, 2
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
            return make_boxed_func(gm.forward)

        return compiler

    q, k, v = (t.requires_grad_() for t in _meta(b, sq, skv, hq, hkv))
    backend = aot_autograd(fw_compiler=record("fw"), bw_compiler=record("bw"))

    @torch.compile(fullgraph=True, dynamic=dynamic, backend=backend)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True, return_lse=True)

    torch._dynamo.reset()
    try:
        with _pinned_flydsl():
            out, lse = fn(q, k, v)
            (out.float().sum() + lse.sum()).backward()
    finally:
        torch._dynamo.reset()
    ops = {kind: [t for t in targets if t.startswith("primus_turbo.")] for kind, targets in seen.items()}
    assert ops == {
        "fw": ["primus_turbo.flash_attn_flydsl_gfx1250_forward.default"],
        "bw": ["primus_turbo.flash_attn_flydsl_gfx1250_backward.default"],
    }
    assert out.shape == (b, sq, hq, D) and out.dtype == torch.bfloat16
    assert lse.shape == (b, hq, sq) and lse.dtype == torch.float32
    for x, ref in ((q.grad, q), (k.grad, k), (v.grad, v)):
        assert x.shape == ref.shape and x.dtype == ref.dtype


# ---------------------------------------------------------------------------
# Packaging: lazy import and the flydsl version gate
# ---------------------------------------------------------------------------


def test_importing_primus_turbo_does_not_import_the_gfx1250_package():
    """The package is imported on the first gfx1250 gate call, never by ``import primus_turbo``:
    it is wave32 code for one arch and needs flydsl>=0.3.4.1,<0.3.5."""
    assert _GFX1250_MODULES_AT_IMPORT == []


def test_flydsl_version_module_imports_only_the_standard_library():
    """The gate asks flydsl_version before anything else, so loading it must not import flydsl
    or torch; checked in a fresh interpreter that loads the file by path."""
    from primus_turbo.flydsl.attention.gfx1250 import flydsl_version

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('flydsl_version', {flydsl_version.__file__!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print(sorted(m for m in ('flydsl', 'torch') if m in sys.modules))\n"
    )
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "[]", run.stdout + run.stderr


def _fake_flydsl(monkeypatch, version):
    """Make ``import flydsl`` fail (version None) or find a flydsl reporting ``version``."""
    if version is None:
        monkeypatch.setitem(sys.modules, "flydsl", None)
    else:
        fake = types.ModuleType("flydsl")
        fake.__version__ = version
        monkeypatch.setitem(sys.modules, "flydsl", fake)


@pytest.mark.parametrize("packaging", [True, False], ids=["packaging", "no_packaging"])
@pytest.mark.parametrize(
    ("version", "ok"),
    [
        ("0.3.4.1", True),
        ("0.3.4.1+g1234567", True),
        ("0.3.4.2", True),
        ("0.3.4", False),
        ("0.3.4.post1", False),
        (None, False),
        ("0.2.4", False),
        ("0.3.3", False),
        ("0.3.5", False),
        ("0.4.0", False),
        ("unknown", False),
    ],
)
def test_flydsl_version_requirement(monkeypatch, version, ok, packaging):
    from primus_turbo.flydsl.attention.gfx1250 import flydsl_version as ver

    if not packaging:  # the release-number fallback
        monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
        monkeypatch.setattr(ver, "_version_ok", ver._satisfies)  # uncached
    _fake_flydsl(monkeypatch, version)
    reason = ver.flydsl_unavailable_reason()
    assert (reason is None) == ok, reason
    if ok:
        ver.require_flydsl()
    else:
        assert f"flydsl{ver.FLYDSL_REQUIREMENT}" in reason
        with pytest.raises(ImportError, match=r"flydsl>=0\.3\.4\.1,<0\.3\.5"):
            ver.require_flydsl()


def test_setup_py_pins_a_flydsl_the_kernels_accept():
    """The flydsl that setup.py pins must satisfy FLYDSL_REQUIREMENT. Under any other release the
    gate declines every call and the tests that need the kernels skip, so a version bump that
    the kernels have not been checked against would otherwise go unnoticed."""
    from primus_turbo.flydsl.attention.gfx1250 import flydsl_version as ver

    setup_py = os.path.join(os.path.dirname(__file__), "..", "..", "..", "setup.py")
    if not os.path.isfile(setup_py):
        pytest.skip("no setup.py next to the tests")
    with open(setup_py) as f:
        pins = re.findall(r"""["']flydsl==([^"']+)["']""", f.read())
    assert pins, "setup.py pins no flydsl"
    for pin in pins:
        assert ver._satisfies(pin), (
            f"setup.py pins flydsl=={pin}, outside the gfx1250 attention kernels' "
            f"FLYDSL_REQUIREMENT {ver.FLYDSL_REQUIREMENT}: check the kernels on that release, then "
            "update flydsl_version.py"
        )


@pytest.mark.parametrize("version", ["0.3.4.dev1", "0.3.4.1.dev0", "0.3.4.1rc1", "0.3.5rc1"])
def test_flydsl_version_requirement_excludes_prereleases_outside(monkeypatch, version):
    pytest.importorskip("packaging")
    from primus_turbo.flydsl.attention.gfx1250 import flydsl_version as ver

    _fake_flydsl(monkeypatch, version)
    assert ver.flydsl_unavailable_reason() is not None


@pytest.mark.parametrize(
    "version", [None, "0.2.4", "0.3.4", "0.3.5"], ids=["missing", "0.2.4", "0.3.4", "0.3.5"]
)
def test_gate_declines_without_flydsl_0_3_4(monkeypatch, version):
    """Without a flydsl that satisfies FLYDSL_REQUIREMENT the gate gives that as the reason before
    importing any kernel module and logs it once, FlyDSL declines the call, and an unpinned call
    resolves to another backend."""

    def forbidden():
        raise AssertionError("gfx1250 kernel module imported without a supported flydsl")

    gfx1250_impl, log = _unresolved_adapter(monkeypatch, forbidden)
    _fake_flydsl(monkeypatch, version)
    q, k, v = _meta(4, 8192, 8192, 32, 8)
    reason = gfx1250_impl.flydsl_gfx1250_unsupported_reason(q, k, v, True)
    assert reason is not None and "flydsl>=0.3.4.1,<0.3.5" in reason, reason
    assert _gate(monkeypatch, q, k, v) is False
    assert pinned_backend_takes(BackendType.FLYDSL, **_resolve_kwargs(q, k, v)) is False
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v)) != BackendType.FLYDSL
    assert len(log.warnings) == 1 and reason in log.warnings[0][0] and log.warnings[0][1] is True


def test_gate_declines_when_the_kernel_package_fails_to_import(monkeypatch):
    """A flydsl that passes the version check but cannot load the kernels (here an
    AttributeError, as from an API renamed in a patch release) makes the gate decline instead of
    raising from can_handle or the backend resolution. The import is tried once and its reason
    logged once; the kernel entry points raise ImportError with it."""
    tries = []

    def broken():
        tries.append(None)
        raise AttributeError("module 'flydsl.expr' has no attribute 'renamed_api'")

    gfx1250_impl, log = _unresolved_adapter(monkeypatch, broken)
    _fake_flydsl(monkeypatch, "0.3.4.1")
    q, k, v = _meta(4, 8192, 8192, 32, 8)
    for _ in range(3):
        reason = gfx1250_impl.flydsl_gfx1250_unsupported_reason(q, k, v, True)
        assert reason is not None and "AttributeError" in reason and "renamed_api" in reason, reason
    assert _gate(monkeypatch, q, k, v) is False
    assert pinned_backend_takes(BackendType.FLYDSL, **_resolve_kwargs(q, k, v)) is False
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(q, k, v)) != BackendType.FLYDSL
    with pytest.raises(ImportError, match="renamed_api"):
        gfx1250_impl._interface()
    assert len(tries) == 1
    assert len(log.warnings) == 1 and reason in log.warnings[0][0] and log.warnings[0][1] is True


@pytest.mark.parametrize(
    ("version", "expected"),
    [("0.2.4", "found flydsl 0.2.4"), ("0.3.4.1", "AttributeError")],
    ids=["old_flydsl", "broken_import"],
)
def test_gate_declines_inside_torch_compile(monkeypatch, version, expected):
    """torch.compile(fullgraph) through a gate whose import is not resolved yet and fails, with
    the library's own logger: the resolution and its log line run outside Dynamo's tracer, so
    the decline traces without a graph break."""

    def broken():
        raise AttributeError("module 'flydsl.expr' has no attribute 'renamed_api'")

    gfx1250_impl, _ = _unresolved_adapter(monkeypatch, broken, record_log=False)
    _fake_flydsl(monkeypatch, version)
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)

    @torch.compile(fullgraph=True, backend="eager")
    def fn(q, k, v):
        taken = attention_impl.DenseAttnFwdFlydslBackend.can_handle(q, k=k, v=v, causal=True)
        return q * 2 if taken else q * 3

    q = torch.ones(1, 64, 1, D, dtype=torch.bfloat16)
    k = torch.ones(1, 64, 1, D, dtype=torch.bfloat16)
    torch._dynamo.reset()
    try:
        out = fn(q, k, k)
    finally:
        torch._dynamo.reset()
    assert torch.equal(out, q * 3)
    reason = gfx1250_impl._INTERFACE
    assert isinstance(reason, str) and expected in reason, reason


@pytest.mark.parametrize(
    "module",
    [
        "interface",
        "flash_attn_utils",
        "buffer_ops",
        "flash_attn_fwd_utils",
        "flash_attn_fwd_kernel",
        "flash_attn_bwd_kernel",
    ],
)
def test_package_modules_refuse_import_without_flydsl_0_3_4(monkeypatch, module):
    """Under another flydsl every module of the package that imports flydsl raises an ImportError
    naming the requirement before it loads anything (what the gate's fallback and a direct
    import see)."""
    flydsl = pytest.importorskip("flydsl")
    for m in _gfx1250_modules():
        if m not in (_PKG, f"{_PKG}.flydsl_version"):
            monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(flydsl, "__version__", "0.2.4")
    with pytest.raises(ImportError, match=r"flydsl>=0\.3\.4\.1,<0\.3\.5, found flydsl 0\.2\.4"):
        importlib.import_module(f"{_PKG}.{module}")
    left = [m for m in _gfx1250_modules() if m not in (_PKG, f"{_PKG}.flydsl_version")]
    assert left == [], left


@needs_flydsl
def test_kernel_modules_import():
    """With flydsl 0.3.4.1 the whole package imports, building no kernel, and interface exposes
    the shape rules the gate asks."""
    from primus_turbo.flydsl.attention.gfx1250 import interface

    for m in ("flash_attn_fwd_kernel", "flash_attn_bwd_kernel"):
        assert f"{_PKG}.{m}" in sys.modules
    assert interface.HEAD_DIM == D
    assert (interface.SEQLEN_Q_MULTIPLE, interface.SEQLEN_KV_MULTIPLE) == (64, 32)


# ---------------------------------------------------------------------------
# Interface argument checks
# ---------------------------------------------------------------------------


@needs_flydsl
@pytest.mark.parametrize(
    ("q_shape", "k_shape"),
    [
        ((0, 256, 4, D), (0, 256, 2, D)),
        ((1, 0, 4, D), (1, 256, 2, D)),
        ((1, 256, 0, D), (1, 256, 2, D)),
        ((1, 256, 4, 0), (1, 256, 2, 0)),
        ((1, 256, 4, D), (1, 0, 2, D)),
        ((1, 256, 4, D), (1, 256, 0, D)),
        ((256, 4, D), (256, 2, D)),
    ],
    ids=["batch", "seqlen_q", "heads_q", "head_dim", "seqlen_kv", "heads_kv", "3d"],
)
def test_empty_or_non_4d_problems_are_refused(q_shape, k_shape):
    """Every dimension must be non-empty (an empty batch would launch zero-size grids) and q / k
    4-D: unsupported_reason says so, and the forward raises it before launching anything."""
    from primus_turbo.flydsl.attention.gfx1250 import interface

    reason = interface.unsupported_reason(q_shape, k_shape, k_shape, torch.bfloat16, True)
    assert reason is not None and ("empty dimension" in reason or "4-D" in reason), reason
    q = torch.empty(q_shape, dtype=torch.bfloat16, device="meta")
    k = torch.empty(k_shape, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="empty dimension|4-D"):
        interface.flash_attn_fwd(q, k, k)


@needs_flydsl
def test_forward_refuses_tensors_on_two_devices():
    """flash_attn_fwd checks that q, k and v share a device before launching anything."""
    from primus_turbo.flydsl.attention.gfx1250 import interface

    q, _, v = _meta(1, 256, 256, 4, 2)
    k = torch.empty(1, 256, 2, D, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one device"):
        interface.flash_attn_fwd(q, k, v)


@needs_flydsl
@pytest.mark.parametrize(
    ("name", "make", "match"),
    [
        (
            "do",
            lambda: torch.empty(1, 128, 4, D, dtype=torch.bfloat16, device="meta"),
            "do must have q's shape",
        ),
        (
            "o",
            lambda: torch.empty(1, 256, 2, D, dtype=torch.bfloat16, device="meta"),
            "o must have q's shape",
        ),
        ("lse", lambda: torch.empty(1, 256, 4, device="meta"), "lse must be float32 of shape"),
        ("lse", lambda: torch.empty(1, 4, 256, dtype=torch.bfloat16, device="meta"), "lse must be float32"),
        ("lse", lambda: torch.empty(1, 4, 256), "one device"),
        ("do", lambda: torch.empty(1, 256, 4, D, dtype=torch.bfloat16), "one device"),
        ("o", lambda: torch.empty(1, 256, 4, D, dtype=torch.float16, device="meta"), "contiguous bfloat16"),
    ],
    ids=["do_shape", "o_shape", "lse_seq_major", "lse_bf16", "lse_on_cpu", "do_on_cpu", "o_fp16"],
)
def test_backward_refuses_mismatched_tensors(monkeypatch, name, make, match):
    """flash_attn_bwd checks do / o / lse against q before launching anything: the kernels take
    every shape from q and k, so a smaller tensor would be read past its end."""
    from primus_turbo.flydsl.attention.gfx1250 import interface

    def forbidden(*args):
        raise AssertionError("launched despite a mismatched tensor")

    monkeypatch.setattr(interface, "_launch", forbidden)
    q, k, v = _meta(1, 256, 256, 4, 2)
    tensors = {
        "do": torch.empty_like(q),
        "o": torch.empty_like(q),
        "lse": torch.empty(1, 4, 256, device="meta"),
    }
    tensors[name] = make()
    with pytest.raises(ValueError, match=match):
        interface.flash_attn_bwd(tensors["do"], q, k, v, tensors["o"], tensors["lse"], causal=True)


# ---------------------------------------------------------------------------
# Launches and streams
# ---------------------------------------------------------------------------


@needs_flydsl
def test_compiled_launchers_are_cached_per_launcher_device_and_signature(monkeypatch):
    """_run_compiled, the launch path of every forward and backward kernel, with flyc.compile
    faked: the first launch per launcher, device and tensor (dtype, rank) compiles (and, in
    flydsl, launches), later ones go through the cached function; a compile-only build is
    never cached, and a failed compile is re-raised with the MLIR context it left open closed."""
    from primus_turbo.flydsl.attention.gfx1250 import flash_attn_utils as utils

    calls, mode = [], {"compile": "ok"}

    def compile(exe, *args):
        calls.append(("compile", exe))
        if mode["compile"] == "fail":
            utils.ir.Context().__enter__()  # as a failed flyc.compile leaves it
            raise RuntimeError("compile failed")
        if mode["compile"] == "compile_only":
            return None
        return lambda *args: calls.append(("cached", exe))

    monkeypatch.setattr(utils, "_COMPILED", {})
    monkeypatch.setattr(utils.flyc, "compile", compile)

    def run(exe, *args):
        calls.clear()
        utils._run_compiled(exe, *args)
        return calls[:]

    bf16 = torch.empty(2, 8, dtype=torch.bfloat16, device="meta")
    assert run("a", bf16, 3) == [("compile", "a")]
    assert run("a", torch.empty(4, 4, dtype=torch.bfloat16, device="meta"), 5) == [("cached", "a")]
    assert run("b", bf16, 3) == [("compile", "b")]
    on_cpu = torch.empty(2, 8, dtype=torch.bfloat16)
    fp32 = torch.empty(2, 8, device="meta")
    rank3 = torch.empty(2, 2, 8, dtype=torch.bfloat16, device="meta")
    for other in (on_cpu, fp32, rank3):  # another device, dtype, rank
        assert run("a", other, 3) == [("compile", "a")]
    mode["compile"] = "compile_only"
    assert run("c", bf16) == [("compile", "c")]
    assert run("c", bf16) == [("compile", "c")]
    mode["compile"] = "fail"
    with pytest.raises(RuntimeError, match="compile failed"):
        utils._run_compiled("d", bf16)
    assert utils.ir.Context.current is None
    mode["compile"] = "ok"
    assert run("d", bf16) == [("compile", "d")]
    assert run("d", bf16) == [("cached", "d")]


@needs_flydsl
@pytest.mark.parametrize(
    ("shape", "dkdv_chain", "dq_chain"),
    [
        ((4, 8192, 8192, 32, 8), ["dkdv"], ["dqg"]),
        ((1, 1024, 1024, 8, 2), ["dkdv_sp", "redsp"], ["dq_sp", "redsp_q"]),
        ((1, 512, 4096, 16, 16), ["dkdv"], ["dq_sp", "redsp_q"]),
        ((2, 2048, 2048, 32, 8), ["dkdv_sp", "redsp"], ["dqg"]),
        ((2, 1024, 1024, 64, 32), ["dkdv"], ["dqg"]),
    ],
    ids=["llama3.1-8b", "small_grid", "dkdv_unsplit", "dq_unsplit", "gqa2_unsplit"],
)
def test_backward_kernels_and_streams(monkeypatch, shape, dkdv_chain, dq_chain):
    """flash_attn_bwd on meta tensors with the launches and streams faked, so no GPU kernel runs:
    k_delta_bshd on the caller's stream, the fork, the dK/dV chain on the caller's stream and
    the dQ chain on the side stream, then the join; no record_stream. The grid sizes pick each
    chain's kernels.

    Without record_stream, the side stream is safe only if every tensor its launches use is an
    input or was allocated (on the caller's stream) before the fork: a block allocated after
    the fork can be one that caller-stream work issued after the fork still uses, and the side
    stream does not wait for that work."""
    from primus_turbo.flydsl.attention.gfx1250 import interface

    b, sq, skv, hq, hkv = shape
    q, k, v = _meta(b, sq, skv, hq, hkv)
    lse = torch.empty(b, hq, sq, device="meta")
    events, allocated = [], []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait", self.name, other.name))

    def logged(alloc):
        def wrapper(*args, **kwargs):
            t = alloc(*args, **kwargs)
            allocated.append(t)  # kept alive, so no later tensor can reuse its id
            events.append(("alloc", id(t)))
            return t

        return wrapper

    def launch(name, launcher, args):
        tensors = {id(a) for a in args if isinstance(a, torch.Tensor)}
        events.append(("launch", name, args[-1].name, tensors))

    caller = FakeStream("caller")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: caller)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: FakeStream("side"))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: events.append(("record", s.name)))
    monkeypatch.setattr(torch, "empty", logged(torch.empty))
    monkeypatch.setattr(torch, "empty_like", logged(torch.empty_like))
    monkeypatch.setattr(interface, "_SIDE_STREAMS", {})
    monkeypatch.setattr(interface, "_launch", launch)
    dq, dk, dv = interface.flash_attn_bwd(q, q, k, v, q, lse, causal=True)
    assert [e[:3] for e in events if e[0] != "alloc"] == [
        ("launch", "delta", "caller"),
        ("wait", "side", "caller"),
        *[("launch", name, "caller") for name in dkdv_chain],
        *[("launch", name, "side") for name in dq_chain],
        ("wait", "caller", "side"),
    ]
    fork = events.index(("wait", "side", "caller"))
    before_fork = {id(t) for t in (q, k, v, lse)} | {e[1] for e in events[:fork] if e[0] == "alloc"}
    for e in events:
        if e[0] == "launch" and e[2] == "side":
            assert e[3] <= before_fork, f"{e[1]} on the side stream uses a tensor allocated after the fork"
    assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)
    assert dq.dtype == dk.dtype == dv.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


def _inputs(b, sq, skv, hq, hkv, layout="bshd", seed=0):
    """[b, s, h, d]-shaped bf16 tensors whose bytes are in ``layout`` order."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def make(s, h):
        if layout == "bshd":
            t = torch.randn(b, s, h, D, generator=g)
        elif layout == "sbhd":
            t = torch.randn(s, b, h, D, generator=g).permute(1, 0, 2, 3)
        else:  # bhsd
            t = torch.randn(b, h, s, D, generator=g).transpose(1, 2)
        return t.to(torch.bfloat16)

    return make(sq, hq), make(skv, hkv), make(skv, hkv), make(sq, hq)


def _reference(q, k, v, dout, causal):
    """fp32 attention and its gradients on the CPU; bottom-right causal; returns lse too."""
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    sq, skv, g = q.shape[1], k.shape[1], q.shape[2] // k.shape[2]
    s = torch.einsum("bqhd,bkhd->bhqk", q, k.repeat_interleave(g, 2)) / math.sqrt(D)
    if causal:
        mask = torch.ones(sq, skv, dtype=torch.bool).triu(skv - sq + 1)
        s = s.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(s, -1)
    out = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.repeat_interleave(g, 2))
    out.backward(dout.float())
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


@contextlib.contextmanager
def _pinned_flydsl():
    """Pin FLYDSL, so a refusal raises instead of quietly running another backend."""
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    try:
        yield
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)


def _run(q, k, v, dout, causal, **kwargs):
    q, k, v = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True, **kwargs)
    out.backward(dout.to("cuda"))
    return out, lse, q.grad, k.grad, v.grad


def _check(got, ref, min_db=40.0):
    for name, x, r in zip(("out", "lse", "dq", "dk", "dv"), got, ref):
        assert x.shape == r.shape, (name, x.shape, r.shape)
        x = x.float().cpu()
        assert torch.isfinite(x).all(), f"{name} has non-finite values"
        snr = compute_snr(r, x)
        assert snr > min_db, f"{name}: {snr:.1f} dB"


# (b, sq, skv, hq, hkv) of test_matches_reference, causal and not. Together they reach both
# forward tilings at every GQA ratio and every backward path, which
# test_numeric_shapes_reach_every_kernel_variant checks without a GPU kernel.
_NUMERIC_SHAPES = {
    "toy": (1, 128, 128, 2, 1),
    "gqa4": (2, 256, 256, 8, 2),
    "mha": (1, 512, 512, 4, 4),
    "sq_lt_skv": (1, 256, 512, 8, 2),
    "gqa16": (2, 256, 256, 16, 1),
    "gqa8_small_grid": (1, 256, 256, 64, 8),
    "llama_heads_small_grid": (1, 1024, 1024, 32, 8),
    "llama_heads_large_grid": (2, 2048, 2048, 32, 8),
    "gqa2_large_grid": (2, 1024, 1024, 64, 32),
    "gqa8_large_grid": (1, 2048, 2048, 64, 8),
    "gqa16_large_grid": (1, 1024, 1024, 128, 8),
    "dkdv_unsplit": (1, 512, 4096, 16, 16),
    "both_unsplit": (1, 1024, 1024, 128, 128),
}


@needs_flydsl
def test_numeric_shapes_reach_every_kernel_variant(monkeypatch):
    """The shapes of test_matches_reference reach both forward tilings at every GQA ratio (the
    tiling picked for a device of 256 CUs), split and unsplit dK/dV, and split and grouped dQ.
    Checked with the kernels and streams faked, so no GPU kernel runs."""
    from primus_turbo.flydsl.attention.gfx1250 import interface

    class Stream:
        def wait_stream(self, other):
            pass

    forward, backward = set(), set()

    def fwd(q, k, v, tiling, **kwargs):
        forward.add((tiling.kernel_name, q.shape[2] // k.shape[2]))

    monkeypatch.setattr(interface, "_num_cu", lambda device: 256)
    monkeypatch.setattr(interface._fwd, "flash_attn_fwd", fwd)
    monkeypatch.setattr(interface, "_launch", lambda name, launcher, args: backward.add(name))
    monkeypatch.setattr(interface, "_SIDE_STREAMS", {})
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: Stream())
    for b, sq, skv, hq, hkv in _NUMERIC_SHAPES.values():
        q, k, v = _meta(b, sq, skv, hq, hkv)
        interface.flash_attn_fwd(q, k, v, causal=True)
        interface.flash_attn_bwd(q, q, k, v, q, torch.empty(b, hq, sq, device="meta"), causal=True)
    tilings = (interface._fwd.DEFAULT, interface._fwd.SMALL_GRID)
    assert forward == {(t.kernel_name, g) for t in tilings for g in interface.GQA_RATIOS}
    assert backward == {"delta", "dkdv", "dkdv_sp", "redsp", "dqg", "dq_sp", "redsp_q"}


@needs_gfx1250
@pytest.mark.parametrize("shape", list(_NUMERIC_SHAPES.values()), ids=list(_NUMERIC_SHAPES))
@pytest.mark.parametrize("causal", [True, False])
def test_matches_reference(shape, causal):
    """Covers both forward variants (grid below / above the CU count) at every GQA ratio and the
    backward's split-K (small grids), grouped dQ (large grids) and unsplit dK/dV (dkdv_unsplit,
    whose Skv/32 * Hkv * B = 2048 kv tiles fill the machine twice; gqa2_large_grid) paths.
    both_unsplit and gqa2_large_grid run k_dkdv and k_dqg concurrently on two streams, the
    combination of the Llama-3.1-8B training shape."""
    q, k, v, dout = _inputs(*shape)
    _check(_run(q, k, v, dout, causal), _reference(q, k, v, dout, causal))


@needs_gfx1250
def test_matches_reference_with_more_queries_than_keys():
    """Non-causal sq > skv, which the gate accepts (causal refuses it: the first sq - skv
    queries would see no key)."""
    q, k, v, dout = _inputs(1, 512, 256, 8, 2)
    _check(_run(q, k, v, dout, False), _reference(q, k, v, dout, False))


@needs_gfx1250
@pytest.mark.parametrize("layout", ["sbhd", "bhsd"])
def test_matches_reference_in_other_byte_orders(layout):
    q, k, v, dout = _inputs(2, 256, 256, 8, 2, layout=layout)
    _check(_run(q, k, v, dout, True), _reference(q, k, v, dout, True))


@needs_gfx1250
def test_explicit_softmax_scale():
    """The scale is a runtime argument, not baked in: scale c on q equals 1/sqrt(D) on
    q * c * sqrt(D), whose dq is the true dq divided by c * sqrt(D)."""
    c = 0.05 * math.sqrt(D)
    q, k, v, dout = _inputs(1, 256, 256, 4, 2)
    got = _run(q, k, v, dout, True, softmax_scale=0.05)
    out, lse, dq, dk, dv = _reference(q.float() * c, k, v, dout, True)
    _check(got, (out, lse, dq * c, dk, dv))


@needs_gfx1250
@pytest.mark.deterministic
def test_backward_is_deterministic():
    q, k, v, dout = _inputs(2, 1024, 1024, 32, 8)
    first = _run(q, k, v, dout, True)
    for _ in range(3):
        again = _run(q, k, v, dout, True)
        for a, b in zip(first, again):
            assert torch.equal(a, b)


@needs_gfx1250
def test_runs_on_a_side_stream():
    q, k, v, dout = _inputs(1, 256, 256, 4, 2)
    ref = _reference(q, k, v, dout, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        got = _run(q, k, v, dout, True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _check(got, ref)


@needs_gfx1250
def test_torch_compile_fullgraph():
    q, k, v, dout = _inputs(1, 256, 256, 4, 2)
    ref = _reference(q, k, v, dout, True)

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True, return_lse=True)

    q_, k_, v_ = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out, lse = fn(q_, k_, v_)
    out.backward(dout.to("cuda"))
    _check((out, lse, q_.grad, k_.grad, v_.grad), ref)


@needs_gfx1250
@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LARGE") != "1",
    reason="Llama-3.1-8B training shape; set PRIMUS_TURBO_TEST_LARGE=1 (about a minute of CPU reference)",
)
def test_llama31_8b_training_shape():
    """b4 s8192 hq32 hkv8: every output finite on the card; batch 0, kv head 0 (q heads 0-3,
    whose dk/dv depend on nothing else) against the CPU reference."""
    b, s, hq, hkv = 4, 8192, 32, 8
    q, k, v, dout = _inputs(b, s, s, hq, hkv)
    got = _run(q, k, v, dout, True)
    for x in got:
        assert torch.isfinite(x).all()
    g = hq // hkv
    sub = (q[:1, :, :g], k[:1, :, :1], v[:1, :, :1], dout[:1, :, :g])
    ref = _reference(*sub, True)
    out, lse, dq, dk, dv = got
    _check((out[:1, :, :g], lse[:1, :g], dq[:1, :, :g], dk[:1, :, :1], dv[:1, :, :1]), ref)


@needs_gfx1250
@pytest.mark.deterministic
@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LARGE") != "1",
    reason="forward repeatability at seq 8192; set PRIMUS_TURBO_TEST_LARGE=1",
)
@pytest.mark.parametrize("causal", [True, False])
def test_forward_repeats_bitwise_at_seq_8192(causal):
    """The forward is built in expert scheduling mode 2 (ENABLE_SCHED_MODE2 of
    flash_attn_fwd_kernel), under which the upstream kernel's notes report a rare failure at
    seq 8192, non-causal. 500 forwards of the Llama-3.1-8B training shape: every output finite
    and bitwise equal to the first."""
    q, k, v, _ = _inputs(4, 8192, 8192, 32, 8)
    q, k, v = (t.to("cuda") for t in (q, k, v))
    with _pinned_flydsl():
        first = flash_attn_func(q, k, v, causal=causal, return_lse=True)
        for x in first:
            assert torch.isfinite(x).all()
        for i in range(500):
            again = flash_attn_func(q, k, v, causal=causal, return_lse=True)
            for a, b in zip(first, again):
                assert torch.equal(a, b), f"forward {i + 1} differs from the first"
