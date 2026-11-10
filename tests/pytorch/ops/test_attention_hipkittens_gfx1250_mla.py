###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens kernels of the gfx1250 (MI455X) MLA attention backward: ``attention_hipkittens_gfx1250_mla_impl``,
built from csrc/kernels/attention/hipkittens/gfx1250_mla.

The setting, the gate, the routing, the launch plan and the build run on meta / CPU tensors with stand-in kernels.
The numeric tests need a gfx1250 card and the built code objects; their fp32 reference runs on the CPU.
"""

import contextlib
import hashlib
import importlib.util
import re
from pathlib import Path

import pytest
import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import (
    attention_hipkittens_gfx1250_mla_impl as hk,
)
from primus_turbo.pytorch.ops import flash_attn_func
from tests.pytorch.ops.test_attention_flydsl_gfx1250_mla import (
    D_QK,
    D_V,
    MLA_SCALE,
    _check,
    _inputs,
    _meta,
    _nan_filled_outputs,
    _on_gfx1250,
    _pinned_flydsl,
    _reference,
    _reference_by_heads,
    _run,
    mla_impl,
)
from tests.pytorch.test_utils import compute_snr

KERNEL_DIR = (
    Path(__file__).resolve().parents[3] / "csrc" / "kernels" / "attention" / "hipkittens" / "gfx1250_mla"
)


def _hsaco():
    """The code-object build script (standard library only, loaded by path like setup.py does)."""
    spec = importlib.util.spec_from_file_location("_hk_mla_hsaco", KERNEL_DIR / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BWD_IMPL = "PRIMUS_TURBO_MLA_BWD_IMPL"


def _set_bwd_impl(monkeypatch, value):
    """PRIMUS_TURBO_MLA_BWD_IMPL through the environment (None: unset); envs caches it at the first read."""
    if value is None:
        monkeypatch.delenv(_BWD_IMPL, raising=False)
    else:
        monkeypatch.setenv(_BWD_IMPL, value)
    envs.reset_cache(_BWD_IMPL)


needs_hk = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()) or hk.hk_mla_unavailable_reason() is not None,
    reason="needs a gfx1250 (MI455X) GPU and the HipKittens MLA code objects",
)


def _record_warnings(monkeypatch):
    """The fallback warnings emitted from now on (the logger's once-per-message record starts empty)."""
    warnings = []
    monkeypatch.setattr(mla_impl.logger, "_log_once_cache", set())
    monkeypatch.setattr(
        mla_impl.logger._logger, "log", lambda level, msg, *args, **kwargs: warnings.append(msg)
    )
    return warnings


def _meta_bwd(b, s, h, skv=None):
    """Meta (dout, q, k, v, out, lse) in the kernel layout."""
    q, k, v = _meta(b, s, s if skv is None else skv, h)
    out = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="meta")
    return out, q, k, v, out, torch.empty(b, h, s, dtype=torch.float32, device="meta")


# ---------------------------------------------------------------------------
# Setting and gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (None, "flydsl"),
        ("", "flydsl"),
        ("flydsl", "flydsl"),
        ("hk", "hk"),
        (" HK ", "hk"),
        ("hipkittens", None),
    ],
)
def test_bwd_impl_setting(monkeypatch, value, want):
    """Read at the first MLA backward, not at import: a value set after primus_turbo is imported takes effect, and a
    bad one raises there instead of breaking the import."""
    _set_bwd_impl(monkeypatch, value)
    if want is None:
        with pytest.raises(ValueError, match=_BWD_IMPL):
            mla_impl.mla_bwd_impl()
    else:
        assert mla_impl.mla_bwd_impl() == want


_ACCEPTED = [
    (2, 4096, 4096, 128),
    (1, 4096, 4096, 256),
    (2, 4096, 4096, 96),
    (1, 256, 256, 2),
    (1, 256, 384, 2),
    (1, 256, 512, 2),
    (1, 4096, 4128, 4),
]
_REFUSED = [
    ((1, 512, 512, 2), "seqlen_q % 384"),
    ((1, 128, 128, 2), "seqlen_q % 384"),
    ((1, 256, 16384, 256), "exceeds the verified"),
    ((1, 256, 256, 4, 2), "MHA only"),
    ((1, 384, 256, 2), "seqlen_kv < seqlen_q"),
]


@pytest.mark.parametrize("dims", _ACCEPTED)
def test_gate_accepts(monkeypatch, dims):
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda device=None: None)
    assert hk.hk_mla_unsupported_reason(*_meta(*dims), True) is None


@pytest.mark.parametrize(("dims", "match"), _REFUSED)
def test_gate_refuses(monkeypatch, dims, match):
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda device=None: None)
    reason = hk.hk_mla_unsupported_reason(*_meta(*dims), True)
    assert reason is not None and match in reason, reason


def _stand_in_device(monkeypatch):
    """Device 0 without a GPU: the loader's device queries are stand-ins."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())


def test_gate_refuses_without_the_code_objects(monkeypatch, tmp_path):
    _stand_in_device(monkeypatch)
    monkeypatch.setattr(hk, "_MLA_HSACO_DIR", tmp_path)
    monkeypatch.setattr(hk, "_MLA_UNAVAILABLE", {})
    monkeypatch.setattr(hk, "_MLA_LOADED", {})
    q, k, v = _meta(1, 256, 256, 2)
    assert "not built" in hk.hk_mla_unsupported_reason(q, k, v, True)
    assert hk.hk_mla_unsupported_reason(q, k, v, False) == "causal only"  # the shape rules come first


def test_runtime_loads_only_pinned_code_objects(monkeypatch, tmp_path):
    """A missing, stale (not the pinned bytes, e.g. left by an older build) or unloadable code object makes the
    backward unavailable, so it runs on FlyDSL, instead of being launched with this version's kernel arguments."""
    _stand_in_device(monkeypatch)
    monkeypatch.setattr(hk, "_MLA_HSACO_DIR", tmp_path)

    def reason():
        monkeypatch.setattr(hk, "_MLA_UNAVAILABLE", {})
        monkeypatch.setattr(hk, "_MLA_LOADED", {})
        return hk.hk_mla_unavailable_reason()

    assert "k_delta.hsaco is not built" in reason()
    (tmp_path / "k_delta.hsaco").write_bytes(b"an older code object")
    assert "k_delta.hsaco is not the pinned code object" in reason()

    class RejectingHip:
        @staticmethod
        def hipModuleLoadData(module, image):
            return 200

        @staticmethod
        def hipGetErrorString(rc):
            return b"hipErrorInvalidImage"

        @staticmethod
        def hipGetLastError():
            return 200

    block, layout, _ = hk._MLA_KERNELS["k_delta"]
    pin = hashlib.sha256(b"an older code object").hexdigest()
    monkeypatch.setitem(hk._MLA_KERNELS, "k_delta", (block, layout, pin))
    monkeypatch.setattr(hk, "_hip_runtime", lambda: RejectingHip)
    assert "loading k_delta failed: hipErrorInvalidImage" in reason()


def test_entry_point_validates_its_tensors():
    do, q, k, v, o, lse = _meta_bwd(1, 256, 2)
    with pytest.raises(ValueError, match="on meta"):
        hk._mla_check(q, torch.empty_like(k, device="cpu"), v, o, do, lse)
    do2, _, _, _, o2, lse2 = _meta_bwd(2, 256, 2)
    with pytest.raises(ValueError, match="strided"):  # b > 1 SBHD storage must reach the kernels folded
        hk._mla_check(*_meta(2, 256, 256, 2, layout="sbhd"), o2, do2, lse2)
    with pytest.raises(ValueError, match="float32"):
        hk._mla_check(q, k, v, o, do, lse.to(torch.bfloat16))
    do0, q0, k0, v0, o0, lse0 = _meta_bwd(1, 0, 2)
    with pytest.raises(ValueError, match="empty problem"):
        hk._mla_check(q0, k0, v0, o0, do0, lse0)


def test_shape_helpers():
    assert [hk._mla_dkdv_blocks(s) for s in (384, 4096, 4128, 4160)] == [4, 43, 43, 44]
    assert hk._mla_dq_split(4096) == (256, 1, 10) and hk._mla_dq_split(3840) == (0, 0, 10)
    assert hk._mla_ds_bytes(2, 4096, 4096, 128, True) == 4_328_521_728
    assert hk._mla_ds_bytes(1, 256, 384, 2, False) == 2 * 24 * 8 * 1024
    for b, sq, skv, h in ((1, 256, 256, 2), (2, 4096, 4096, 3), (1, 256, 448, 1), (1, 384, 1024, 2)):
        attended = torch.ones(sq, skv, dtype=torch.bool).tril(skv - sq)
        blocks = attended.view(sq // 32, 32, skv // 16, 16).any(3).any(1).sum().item()
        assert hk._mla_ds_bytes(b, sq, skv, h, True) == b * h * blocks * 1024, (sq, skv)
    assert [hk._mla_head_group(n) for n in (2, 32, 96, 128, 256, 40, 36)] == [2, 32, 32, 32, 32, 8, 36]


# ---------------------------------------------------------------------------
# Routing and tracing
# ---------------------------------------------------------------------------


def _stub_kernels(monkeypatch, calls):
    """The MLA custom ops replaced by the CPU reference; ``calls`` records which backward ran."""

    def fwd(q, k, v, softmax_scale, causal):
        out, lse, *_ = _reference(q, k, v, torch.zeros(*q.shape[:3], D_V), softmax_scale)
        return out.to(q.dtype).contiguous(), lse.contiguous()

    def bwd(tag):
        def run(dout, q, k, v, out, lse, softmax_scale, causal):
            calls.append((tag, tuple(q.shape)))
            return tuple(g.to(q.dtype).contiguous() for g in _reference(q, k, v, dout, softmax_scale)[2:])

        return run

    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda device=None: None)
    monkeypatch.setattr(mla_impl, "_forward_op", fwd)
    monkeypatch.setattr(mla_impl, "_backward_op", bwd("flydsl"))
    monkeypatch.setattr(mla_impl, "_backward_hk_op", bwd("hk"))


@pytest.mark.parametrize(
    ("dims", "impl", "expected"),
    [
        ((256, 256), "hk", "hk"),
        ((256, 256), "flydsl", "flydsl"),
        ((256, 384), "hk", "hk"),
        ((128, 128), "hk", "flydsl"),  # seqlen_q % 384 == 128: the HipKittens kernels refuse it
    ],
)
def test_routes_and_falls_back(monkeypatch, dims, impl, expected):
    """The backward follows the setting and falls back to FlyDSL; it sees the kernel-layout tensors (b > 1 SBHD
    folded to [1, s, b*h, d])."""
    _on_gfx1250(monkeypatch)
    _set_bwd_impl(monkeypatch, impl)
    calls = []
    _stub_kernels(monkeypatch, calls)
    b, h = 2, 2
    q, k, v, dout = _inputs(b, *dims, h, layout="sbhd")
    qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out = flash_attn_func(qg, kg, vg, softmax_scale=MLA_SCALE, causal=True)
    out.backward(dout)
    assert calls == [(expected, (1, dims[0], b * h, D_QK))]
    ref = _reference(q, k, v, dout, MLA_SCALE)
    for name, x, r in zip(("out", "dq", "dk", "dv"), (out, qg.grad, kg.grad, vg.grad), ref[:1] + ref[2:]):
        assert compute_snr(r, x.detach().float()) > 40, name


def test_fallback_is_logged_once_per_reason(monkeypatch):
    """Backwards the HipKittens kernels refuse log each reason once: two calls at one refused shape log one warning,
    a call that falls back for another reason (here: no code objects) logs its own."""
    _on_gfx1250(monkeypatch)
    _set_bwd_impl(monkeypatch, "hk")
    calls = []
    _stub_kernels(monkeypatch, calls)
    warnings = _record_warnings(monkeypatch)

    def step(sq):
        q, k, v, dout = _inputs(1, sq, 256, 2, layout="bshd")
        qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
        with _pinned_flydsl():
            flash_attn_func(qg, kg, vg, softmax_scale=MLA_SCALE, causal=True).backward(dout)

    step(128)
    step(128)  # seqlen_q % 384 == 128: refused twice, logged once
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda device=None: "k_delta.hsaco is not built")
    step(256)
    assert [c[0] for c in calls] == ["flydsl"] * 3
    assert [w.split(",")[0] for w in warnings] == [f"{_BWD_IMPL}=hk"] * 2
    assert "seqlen_q % 384" in warnings[0] and "k_delta.hsaco is not built" in warnings[1]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    ("sq", "op"),
    [(256, "hk"), (640, "hk"), (128, "flydsl"), (512, "flydsl")],
    ids=["hk", "hk_s640", "falls_back", "falls_back_s512"],
)
def test_torch_compile_traces_the_hk_op(monkeypatch, sq, op, dynamic):
    """torch.compile(fullgraph=True) on meta tensors with the setting on hk: the backward graph holds the custom op
    the call routes to; a call that falls back does not break the graph, and its warning is logged while tracing. With
    dynamic shapes the gate does not specialize the seqlen (512 and 640 are not duck-sized to a head dim)."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    _on_gfx1250(monkeypatch)
    _set_bwd_impl(monkeypatch, "hk")
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda device=None: None)
    warnings = _record_warnings(monkeypatch)
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if str(n.target).startswith("primus_turbo.")]
            return make_boxed_func(gm.forward)

        return compiler

    b, h = 2, 4
    q, k, v = (t.requires_grad_() for t in _meta(b, sq, sq, h, layout="sbhd"))

    @torch.compile(
        fullgraph=True,
        dynamic=dynamic,
        backend=aot_autograd(fw_compiler=record("fw"), bw_compiler=record("bw")),
    )
    def fn(q, k, v):
        return flash_attn_func(q, k, v, softmax_scale=MLA_SCALE, causal=True, return_lse=True)

    torch._dynamo.reset()
    try:
        with _pinned_flydsl():
            out, lse = fn(q, k, v)
            (out.float().sum() + lse.sum()).backward()
    finally:
        torch._dynamo.reset()
    assert seen == {
        "fw": ["primus_turbo.flash_attn_flydsl_gfx1250_mla_forward.default"],
        "bw": [f"primus_turbo.flash_attn_{op}_gfx1250_mla_backward.default"],
    }
    assert out.shape == (b, sq, h, D_V) and lse.shape == (b, h, sq) and q.grad.shape == q.shape
    assert [w.split(",")[0] for w in warnings] == ([] if op == "hk" else [f"{_BWD_IMPL}=hk"])


# ---------------------------------------------------------------------------
# Launch plan (meta tensors)
# ---------------------------------------------------------------------------


def test_bwd_plan():
    """DeepSeek-V3's shape: grids, the dS scratch (exactly the layout), and argument tuples of the kernels' layouts."""
    launches, grads = hk._hk_mla_bwd_plan(*_meta_bwd(2, 4096, 128), MLA_SCALE, True)
    assert [(name, grid) for name, grid, _ in launches] == [
        ("k_delta", (64, 256, 1)),
        ("k_dkdv96ds", (32, 43, 8)),
        ("k_dqds_c", (32, 11, 8)),
    ]
    ds = launches[1][2][8]
    assert ds.numel() * 2 == 4_328_521_728 and launches[2][2][0] is ds
    for name, _, values in launches:
        args = hk._MLA_KERNELS[name][1]
        assert len(values) == len(re.sub(r"(\d+)(\w)", lambda m: m[2] * int(m[1]), args)) - 1, name
    assert [g.shape for g in grads] == [(2, 4096, 128, D_QK), (2, 4096, 128, D_QK), (2, 4096, 128, D_V)]
    with pytest.raises(NotImplementedError, match="seqlen_q % 384"):
        hk._hk_mla_bwd_plan(*_meta_bwd(1, 512, 2), MLA_SCALE, True)


def test_bwd_does_not_retry_a_scratch_that_did_not_fit(monkeypatch):
    """A failed allocation (the allocator has then emptied its cache and synchronized) of the dS scratch, or of the
    outputs after it, sends the call to FlyDSL, and a scratch at least as large is not tried again on that device: those
    calls go to FlyDSL at once. A smaller one is still tried."""
    attempts, empty = [], torch.empty
    full = {"oom_on": "scratch"}

    def tight_empty(*shape, **kwargs):
        is_scratch = kwargs.get("dtype") is torch.bfloat16 and len(shape) == 1 and len(shape[0]) == 1
        if is_scratch:
            attempts.append(shape[0][0] * 2)
        if (full["oom_on"] == "scratch") == is_scratch:
            raise torch.OutOfMemoryError("no memory")
        return empty(*shape, **kwargs)

    b2, b1 = _meta_bwd(2, 4096, 128), _meta_bwd(1, 4096, 128)
    large, small = hk._mla_ds_bytes(2, 4096, 4096, 128, True), hk._mla_ds_bytes(1, 4096, 4096, 128, True)
    monkeypatch.setattr(hk, "_MLA_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", tight_empty)
    for _ in range(2):
        with pytest.raises(hk._MlaScratchError, match=f"no memory for a {large}-byte dS scratch"):
            hk._hk_mla_bwd_plan(*b2, MLA_SCALE, True)
    assert attempts == [large]
    full["oom_on"] = "outputs"  # the scratch fits, NL / ND do not
    with pytest.raises(
        hk._MlaScratchError, match=f"no memory for a {small}-byte dS scratch and the gradients"
    ):
        hk._hk_mla_bwd_plan(*b1, MLA_SCALE, True)
    assert attempts == [large, small] and hk._MLA_OOM_BYTES == {torch.device("meta"): small}


# ---------------------------------------------------------------------------
# Build: manifest, sources, code objects
# ---------------------------------------------------------------------------


def _kernel_signature(source: str, name: str):
    """(workgroup size, struct format) of kernel ``name`` from its HIP source: pointers Q, float f, int i."""
    m = re.search(
        r"(?:__launch_bounds__\((\d+)\)|amdgpu_flat_work_group_size\((\d+),[^)]*\))[^{;]*?\bvoid\s+"
        + name
        + r"\s*\(([^)]*)\)",
        source,
    )
    params = [p.strip() for p in m.group(3).split(",")]
    assert all("*" in p or p.split()[0] in ("float", "int") for p in params), params
    return int(m.group(1) or m.group(2)), "".join("Q" if "*" in p else p[0] for p in params)


def test_kernel_table_matches_the_sources_and_the_pins():
    """_MLA_KERNELS has each kernel's workgroup size and argument layout as its HIP source declares them, and the
    manifest's sha256 pin; the flags are a hermetic device-only gfx1250 build (include paths inside this directory)."""
    m = _hsaco().load_manifest()
    assert m["arch"] == "gfx1250" and {"--offload-arch=gfx1250", "--cuda-device-only"} <= set(m["flags"])
    assert set(m["kernels"]) == set(hk._MLA_KERNELS)
    for name, (block, layout, pin) in hk._MLA_KERNELS.items():
        k = m["kernels"][name]
        assert pin == k["hsaco_sha256"], f"{name}: copy the manifest's hsaco_sha256 into _MLA_KERNELS"
        assert all(f == "-Iinclude" for f in m["flags"] + k["extra_flags"] if f.startswith("-I")), name
        expanded = re.sub(r"(\d+)(\w)", lambda x: x[2] * int(x[1]), layout[1:])
        assert _kernel_signature((KERNEL_DIR / k["src"]).read_text(), name) == (block, expanded), name


def test_build_reproduces_the_pinned_code_objects(tmp_path):
    """With the pinned toolchain (the amdprimus/amdprimus:tas-20261019 image), a clean build gives every pinned code
    object byte for byte. CPU only."""
    hsaco = _hsaco()
    cxx = hsaco.find_compiler()
    if cxx is None or hsaco.compiler_version(cxx) != hsaco.load_manifest()["toolchain"]:
        pytest.skip("the pinned toolchain is not installed here")
    logs = []
    assert hsaco.build(cxx, tmp_path, log=logs.append), logs
    assert hsaco.unavailable_reason(tmp_path) is None


def test_build_never_fails_and_installs_only_pinned_code_objects(monkeypatch, tmp_path):
    """Pinned code objects are kept, an unpinned one is removed, and a compiler that is missing, not the pinned
    toolchain, or emits other bytes leaves the build failed without raising or installing anything."""
    hsaco = _hsaco()
    manifest = hsaco.load_manifest()
    blobs = {n: f"code object {n}".encode() for n in manifest["kernels"]}
    pinned = {
        n: dict(k, hsaco_sha256=hashlib.sha256(blobs[n]).hexdigest()) for n, k in manifest["kernels"].items()
    }
    monkeypatch.setattr(hsaco, "load_manifest", lambda: dict(manifest, kernels=pinned))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for n, data in blobs.items():
        (bin_dir / f"{n}.hsaco").write_bytes(b"stale" if n == "k_delta" else data)
    fake = tmp_path / "fake-clang"
    fake.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && echo junk > "$2"; shift; done\n')
    fake.chmod(0o755)
    for cxx, version, want in (
        (tmp_path / "missing-clang", None, "not found"),
        (fake, None, "not the pinned toolchain"),
        (fake, manifest["toolchain"], "not built: RuntimeError"),
    ):
        if version is not None:
            monkeypatch.setattr(hsaco, "compiler_version", lambda c, v=version: v)
        logs = []
        assert hsaco.build(cxx=str(cxx), bin_dir=bin_dir, log=logs.append) is False
        assert any(want in m for m in logs), logs
        assert sorted(p.name for p in bin_dir.iterdir()) == ["k_dkdv96ds.hsaco", "k_dqds_c.hsaco"]
    assert hsaco.unavailable_reason(bin_dir) == "k_delta.hsaco is not built"
    assert hsaco.unavailable_reason(bin_dir, ["k_dkdv96ds", "k_dqds_c"]) is None


@pytest.mark.skipif(not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU")
def test_code_objects_are_built_and_load():
    """With the pinned toolchain installed the build must have produced the pinned code objects, and pinned code
    objects must load: otherwise every HipKittens numeric test here skips and the backward quietly runs on FlyDSL."""
    hsaco = _hsaco()
    cxx = hsaco.find_compiler()
    if cxx is not None and hsaco.compiler_version(cxx) == hsaco.load_manifest()["toolchain"]:
        assert hsaco.unavailable_reason(hk._MLA_HSACO_DIR) is None
    if hsaco.unavailable_reason(hk._MLA_HSACO_DIR) is not None:
        pytest.skip("the code objects are not built here")
    assert hk.hk_mla_unavailable_reason() is None


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


def _counting(monkeypatch):
    """The kernels each hk_mla_bwd call launched, one list per call. Earlier out-of-memory fallbacks are forgotten, so a
    call that runs FlyDSL instead shows up as a call without launches."""
    calls, real_bwd, real_launch = [], hk.hk_mla_bwd, hk._HipKernel.__call__

    def bwd(*args, **kwargs):
        calls.append([])
        return real_bwd(*args, **kwargs)

    def launch(self, *args, **kwargs):
        calls[-1].append(self.name)
        return real_launch(self, *args, **kwargs)

    monkeypatch.setattr(hk, "_MLA_OOM_BYTES", {})
    monkeypatch.setattr(hk, "hk_mla_bwd", bwd)
    monkeypatch.setattr(hk._HipKernel, "__call__", launch)
    return calls


def _run_hk_bwd(monkeypatch, q, k, v, dout, scale=MLA_SCALE):
    _set_bwd_impl(monkeypatch, "hk")
    with pytest.MonkeyPatch.context() as mp:
        calls = _counting(mp)
        got = _run(q, k, v, dout, scale)
    _set_bwd_impl(monkeypatch, "flydsl")
    assert calls == [list(hk._MLA_KERNELS)], f"the HipKittens backward did not run: {calls}"
    return got


@needs_hk
@pytest.mark.parametrize(
    ("dims", "layout", "scale"),
    [
        ((1, 256, 256, 2), "sbhd", MLA_SCALE),
        ((2, 256, 256, 2), "sbhd", MLA_SCALE),
        ((1, 256, 384, 2), "sbhd", MLA_SCALE),
        ((1, 256, 512, 2), "bshd", MLA_SCALE),
        ((2, 640, 704, 4), "sbhd", MLA_SCALE),
        ((1, 640, 640, 4), "bshd", MLA_SCALE),
        ((2, 256, 256, 2), "bhsd", None),
        ((2, 1024, 1024, 96), "sbhd", MLA_SCALE),
        ((1, 1536, 1536, 40), "bshd", 0.03),
    ],
    ids=[
        "toy",
        "toy_folded",
        "rect",
        "rect_two_tails",
        "rect_two_tails_folded",
        "s640",
        "bhsd_default_scale",
        "kimi_k3_heads",
        "h40_s1536",
    ],
)
def test_hk_bwd_matches_reference_and_flydsl(monkeypatch, dims, layout, scale):
    """The HipKittens backward matches the fp32 reference, is bitwise deterministic, and its gradients are bitwise
    those of the FlyDSL backward (outputs NaN-filled first, so every element is written)."""
    q, k, v, dout = _inputs(*dims, layout=layout)
    got = _run_hk_bwd(monkeypatch, q, k, v, dout, scale)
    _check(got, _reference(q, k, v, dout, D_QK**-0.5 if scale is None else scale))
    again = _run_hk_bwd(monkeypatch, q, k, v, dout, scale)
    fly = _run(q, k, v, dout, scale)
    for name, a, b, c in zip(("out", "lse", "dq", "dk", "dv"), got, again, fly):
        assert torch.equal(a, b) and torch.equal(a, c), name


@needs_hk
def test_hk_bwd_production_shape_matches_reference(monkeypatch):
    """(1, 4096, 4096, 16): 42 96-row kv blocks + the 64-row tail, 10 384-query dQ tiles + the 256-query head; outputs
    and dS scratch NaN-filled, so a block dQ reads but dK / dV never wrote fails."""
    q, k, v, dout = _inputs(1, 4096, 4096, 16, layout="bshd")
    _check(_run_hk_bwd(monkeypatch, q, k, v, dout), _reference_by_heads(q, k, v, dout, MLA_SCALE))


@needs_hk
def test_hk_bwd_rect32_matches_reference():
    """seqlen_kv - seqlen_q = 32 (the FlyDSL gate refuses it, the HipKittens rules admit it): dQ's last steady unit
    reads the first masked kv32 row, whose stored pairs start mid-row."""
    q, k, v, dout = _inputs(1, 4096, 4128, 4, layout="bshd")
    ref = _reference_by_heads(q, k, v, dout, MLA_SCALE)
    out, lse = ref[0].to(torch.bfloat16).cuda(), ref[1].float().cuda()
    args = [t.cuda().contiguous() for t in (dout, q, k, v)]
    with _nan_filled_outputs():
        got = hk.hk_mla_bwd(*args, out, lse, MLA_SCALE, True)
        again = hk.hk_mla_bwd(*args, out, lse, MLA_SCALE, True)
    for name, x, y, r in zip(("dq", "dk", "dv"), got, again, ref[2:]):
        assert torch.equal(x, y), name
        assert compute_snr(r, x.float().cpu()) >= 48.0, name


@needs_hk
def test_hk_bwd_under_cuda_graphs():
    """The backward captured into a CUDA graph after an eager warm-up (which loads the code objects), then replayed:
    the eager bits. The kernel arguments and the dS scratch are part of the graph."""
    q, k, v, dout = (t.cuda() for t in _inputs(1, 1024, 1024, 8, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    eager = hk.hk_mla_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = hk.hk_mla_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
    graph.replay()
    torch.cuda.synchronize()
    for name, a, b in zip(("dq", "dk", "dv"), got, eager):
        assert torch.equal(a, b), name


@needs_hk
def test_hk_bwd_falls_back_to_flydsl(monkeypatch):
    """Out of memory for the dS scratch: the call runs the FlyDSL backward; a shape FlyDSL cannot take either is
    refused."""
    q, k, v, dout = (t.cuda() for t in _inputs(1, 256, 256, 4, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    fly = mla_impl._backward_op(dout, q, k, v, out, lse, MLA_SCALE, True)

    def oom(*args):
        raise hk._MlaScratchError("no memory for a dS scratch")

    monkeypatch.setattr(hk, "_hk_mla_bwd_plan", oom)
    for a, b in zip(hk.hk_mla_bwd(dout, q, k, v, out, lse, MLA_SCALE, True), fly):
        assert torch.equal(a, b)
    q, k, v, dout = (t.cuda() for t in _inputs(1, 256, 288, 4, layout="bshd"))  # seqlen_kv % 64 == 32
    lse = torch.zeros(1, 4, 256, device="cuda")
    with pytest.raises(NotImplementedError, match="FlyDSL cannot take it either"):
        hk.hk_mla_bwd(dout, q, k, v, dout, lse, MLA_SCALE, True)


def _delta_inputs(b, s, h, special, seed=0):
    """Kernel-layout dO / O [b, s, h, 128] bf16 and lse [b, h, s] fp32 on the card. special: dO and O rows scaled by
    2^-k, k in [0, 80), so the bf16 products reach below the fp32 denormals; 5% bf16 denormals, 2% signed zeros;
    lse with 5% fp32 denormals, 1% signed zeros and 1% +-3e38 (NL overflows to +-inf)."""
    g = torch.Generator().manual_seed(seed)
    do, o = (torch.randn(b, s, h, D_V, generator=g) for _ in range(2))
    lse = torch.randn(b, h, s, generator=g) * 4 + 6
    if special:
        do, o = (t * torch.exp2(-torch.randint(0, 80, (b, s, h, 1), generator=g).float()) for t in (do, o))
    do, o = do.to(torch.bfloat16), o.to(torch.bfloat16)
    if special:

        def sign(shape, dtype, bit):
            return torch.randint(0, 2, shape, generator=g, dtype=dtype) * torch.tensor(bit, dtype=dtype)

        for t in (do, o):
            bits = t.view(torch.int16)
            den = torch.randint(1, 128, bits.shape, generator=g, dtype=torch.int16) | sign(
                bits.shape, torch.int16, -32768
            )
            pick = torch.rand(bits.shape, generator=g)
            bits[pick < 0.05] = den[pick < 0.05]
            zero = (pick >= 0.05) & (pick < 0.07)
            bits[zero] = sign(bits.shape, torch.int16, -32768)[zero]
        bits = lse.view(torch.int32)
        den = torch.randint(1, 1 << 23, bits.shape, generator=g, dtype=torch.int32) | sign(
            bits.shape, torch.int32, -(2**31)
        )
        pick = torch.rand(bits.shape, generator=g)
        bits[pick < 0.05] = den[pick < 0.05]
        zero = (pick >= 0.05) & (pick < 0.06)
        bits[zero] = sign(bits.shape, torch.int32, -(2**31))[zero]
        huge = pick >= 0.99
        lse[huge] = 3e38 * (1 - 2 * torch.randint(0, 2, lse.shape, generator=g).float())[huge]
    return do.cuda(), o.cuda(), lse.cuda()


@needs_hk
@pytest.mark.parametrize(
    ("dims", "special"),
    [
        ((1, 256, 2), False),
        ((2, 256, 4), False),
        ((1, 4096, 16), False),
        ((2, 256, 4), True),
        ((1, 640, 4), True),
    ],
    ids=["toy", "b2", "s4096", "b2_denormals", "s640_denormals"],
)
def test_hk_delta_matches_flydsl(dims, special):
    """k_delta's NL / ND bitwise equal the FlyDSL k_delta's, also for denormals, underflowing products, signed zeros
    and overflowing NL."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd as fly

    do, o, lse = _delta_inputs(*dims, special)
    b, s, h, _ = do.shape
    q = torch.empty(b, s, h, D_QK, dtype=torch.bfloat16, device="cuda")
    with _nan_filled_outputs():
        launches, _ = hk._hk_mla_bwd_plan(do, q, q, o, o, lse, MLA_SCALE, True)
    name, grid, values = launches[0]
    stream = torch.cuda.current_stream()
    hk._mla_kernel(name)(grid, values, stream)
    delta, nl, nd = (torch.full((b, h, s), float("nan"), device="cuda") for _ in range(3))
    fly._launch(
        fly._k.launch_delta,
        (do, o, delta, lse, nl, nd, MLA_SCALE, s, h, b * s * h, s // fly._k.ROWS_DELTA, b * h, stream),
    )
    for label, x, y in zip(("NL", "ND"), (values[3], values[4]), (nl, nd)):
        assert not torch.isnan(x).any(), label
        assert torch.equal(x.view(torch.int32), y.view(torch.int32)), label
    if special:
        tiny = (delta != 0) & (delta.abs() < torch.finfo(torch.float32).tiny)
        assert tiny.any(), "no denormal delta: the inputs do not reach the fp32 denormal range"


@needs_hk
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_hk_end_to_end(monkeypatch, compiled):
    """The backward on hk through flash_attn_func (SBHD, folded), eager and torch.compile'd: bitwise the FlyDSL
    gradients, and as close to the reference."""
    q, k, v, dout = _inputs(2, 256, 256, 8, layout="sbhd")
    calls = _counting(monkeypatch)
    attn = flash_attn_func
    if compiled:
        torch._dynamo.reset()
        attn = torch.compile(flash_attn_func, fullgraph=True, backend="aot_eager")
    try:
        _set_bwd_impl(monkeypatch, "hk")
        got = _run(q, k, v, dout, attn=attn)
        _set_bwd_impl(monkeypatch, "flydsl")
        fly = _run(q, k, v, dout)
    finally:
        torch._dynamo.reset()
    assert calls == [list(hk._MLA_KERNELS)], f"the HipKittens backward did not run: {calls}"
    _check(got, _reference(q, k, v, dout, MLA_SCALE))
    for name, a, b in zip(("out", "lse", "dq", "dk", "dv"), got, fly):
        assert torch.equal(a, b), name
