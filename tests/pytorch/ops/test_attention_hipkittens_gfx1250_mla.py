###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HIP kernels of the gfx1250 (MI455X) MLA attention: primus_turbo/hipkittens/gfx1250_mla.

The settings, gate, routing, plan, manifest, source and patch tests use meta / CPU tensors and stand-in kernels.
The numeric tests need a gfx1250 card and the built code objects (``hsaco.py --check``); their fp32 reference runs
on the CPU.
"""

import hashlib
import importlib
import re

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
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


def _hk(name):
    return importlib.import_module(f"primus_turbo.hipkittens.gfx1250_mla.{name}")


def _code_objects_reason(pass_):
    hsaco = _hk("hsaco")
    return hsaco.unavailable_reason(hsaco.FWD_KERNELS if pass_ == "fwd" else hsaco.BWD_KERNELS)


_ON_GFX1250 = torch.cuda.is_available() and is_gfx1250()
needs_hk_fwd = pytest.mark.skipif(
    not _ON_GFX1250 or _code_objects_reason("fwd") is not None,
    reason="needs a gfx1250 (MI455X) GPU and the HIP MLA forward code object",
)
needs_hk_bwd = pytest.mark.skipif(
    not _ON_GFX1250 or _code_objects_reason("bwd") is not None,
    reason="needs a gfx1250 (MI455X) GPU and the HIP MLA backward code objects",
)
needs_hk = pytest.mark.skipif(
    not _ON_GFX1250 or _code_objects_reason("fwd") is not None or _code_objects_reason("bwd") is not None,
    reason="needs a gfx1250 (MI455X) GPU and every HIP MLA code object",
)


def _settings(monkeypatch, fwd="flydsl", bwd="flydsl"):
    monkeypatch.setattr(mla_impl, "MLA_FWD_IMPL_SETTING", fwd)
    monkeypatch.setattr(mla_impl, "MLA_BWD_IMPL_SETTING", bwd)


def _meta_problem(b, s, h, skv=None):
    """Meta (q, k, v, out, lse) in the kernel layout."""
    q, k, v = _meta(b, s, s if skv is None else skv, h)
    out = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, s, dtype=torch.float32, device="meta")
    return q, k, v, out, lse


# ---------------------------------------------------------------------------
# Settings and gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pass_", ["fwd", "bwd"])
@pytest.mark.parametrize(
    ("setting", "expected"),
    [("", "flydsl"), ("flydsl", "flydsl"), ("hk", "hk"), (" HK ", "hk"), ("hipkittens", None)],
    ids=["unset", "flydsl", "hk", "hk_spelled_loosely", "unknown"],
)
def test_impl_settings(monkeypatch, pass_, setting, expected):
    monkeypatch.setattr(mla_impl, f"MLA_{pass_.upper()}_IMPL_SETTING", setting)
    impl = getattr(mla_impl, f"mla_{pass_}_impl")
    if expected is None:
        with pytest.raises(ValueError, match=f"PRIMUS_TURBO_MLA_{pass_.upper()}_IMPL"):
            impl()
    else:
        assert impl() == expected


_ACCEPTED = {
    "fwd": [(2, 4096, 4096, 128), (1, 4096, 4096, 256), (1, 256, 256, 2), (1, 640, 640, 4), (1, 64, 64, 2)],
    "bwd": [(2, 4096, 4096, 128), (1, 4096, 4096, 256), (1, 256, 256, 2), (1, 256, 384, 2)],
}
_REFUSED = {
    "fwd": [
        ((1, 256, 384, 2), "seqlen_q must equal seqlen_kv"),
        ((1, 256, 256, 4, 2), "MHA only"),
        ((1, 96, 96, 2), "multiple of 64"),
        ((1, 256, 128, 2), "seqlen_q <= seqlen_kv"),
    ],
    "bwd": [
        ((1, 512, 512, 2), "seqlen_q % 384"),
        ((1, 128, 128, 2), "seqlen_q % 384"),
        ((1, 256, 16384, 256), "verified extent"),
        ((1, 256, 256, 4, 2), "MHA only"),
        ((1, 512, 256, 2), "seqlen_q <= seqlen_kv"),
    ],
}


@pytest.mark.parametrize(("pass_", "dims"), [(p, d) for p, ds in _ACCEPTED.items() for d in ds])
def test_hk_gate_accepts(monkeypatch, pass_, dims):
    monkeypatch.setattr(mla_impl, "_hk_code_objects_reason", lambda p: None)
    gate = getattr(mla_impl, f"hk_mla_{pass_}_unsupported_reason")
    assert gate(*_meta(*dims), True) is None


@pytest.mark.parametrize(("pass_", "dims", "match"), [(p, d, m) for p, ds in _REFUSED.items() for d, m in ds])
def test_hk_gate_refuses(monkeypatch, pass_, dims, match):
    monkeypatch.setattr(mla_impl, "_hk_code_objects_reason", lambda p: None)
    reason = getattr(mla_impl, f"hk_mla_{pass_}_unsupported_reason")(*_meta(*dims), True)
    assert reason is not None and match in reason


@pytest.mark.parametrize("pass_", ["fwd", "bwd"])
def test_hk_gate_refuses_without_the_code_objects(monkeypatch, pass_):
    monkeypatch.setattr(mla_impl, "_hk_code_objects_reason", lambda p: f"{p}: k.hsaco is not built")
    gate = getattr(mla_impl, f"hk_mla_{pass_}_unsupported_reason")
    assert gate(*_meta(1, 256, 256, 2), True) == f"{pass_}: k.hsaco is not built"
    assert gate(*_meta(1, 256, 256, 2), False) == "causal only"  # the shape rules come first


@pytest.mark.parametrize("scale", [0.0, -0.1, float("inf"), float("nan")])
def test_hk_bwd_gate_refuses_scales_it_cannot_fold(monkeypatch, scale):
    """The backward folds log2(scale) into its row constants: a scale that is not positive and finite goes to
    FlyDSL (the forward takes any scale, like FlyDSL's), and a direct call refuses it."""
    monkeypatch.setattr(mla_impl, "_hk_code_objects_reason", lambda p: None)
    q, k, v = _meta(1, 256, 256, 2)
    assert "softmax_scale must be positive and finite" in mla_impl.hk_mla_bwd_unsupported_reason(
        q, k, v, True, scale
    )
    assert mla_impl.hk_mla_bwd_unsupported_reason(q, k, v, True, MLA_SCALE) is None
    assert mla_impl.hk_mla_fwd_unsupported_reason(q, k, v, True) is None
    with pytest.raises(NotImplementedError, match="softmax_scale"):
        _hk("bwd")._plan(*(_meta_problem(1, 256, 2)[i] for i in (3, 0, 1, 2, 3, 4)), scale, True)


def test_entries_validate_their_tensors():
    fwd, bwd = _hk("fwd"), _hk("bwd")
    q, k, v, o, lse = _meta_problem(1, 256, 2)
    with pytest.raises(ValueError, match="is on cpu, q on meta"):
        fwd._plan(q, torch.empty_like(k, device="cpu"), v, MLA_SCALE, True)
    with pytest.raises(ValueError, match="is on cpu, q on meta"):
        bwd._plan(o, q, k, v, o, torch.empty_like(lse, device="cpu"), MLA_SCALE, True)
    with pytest.raises(ValueError, match="empty problem"):
        fwd._plan(*_meta_problem(1, 0, 2)[:3], MLA_SCALE, True)
    with pytest.raises(_hk("hiprt").HipError, match="run on a GPU"):
        _hk("hiprt").check_device(torch.device("cpu"))


def test_limits():
    limits = _hk("limits")
    assert limits.fwd_shape_unsupported_reason(2, 4096, 4096, 128, 128, True) is None
    assert "32-bit" in limits.fwd_shape_unsupported_reason(4, 16384, 16384, 128, 128, True)
    assert "causal only" in limits.fwd_shape_unsupported_reason(1, 256, 256, 2, 2, False)
    assert limits.bwd_shape_unsupported_reason(1, 4096, 4128, 4, 4, True) is None  # (Skv - Sq) % 64 == 32
    assert "multiples of 32" in limits.bwd_shape_unsupported_reason(1, 256, 272, 2, 2, True)
    assert [(s, limits.k3_n96(s), limits.k3_n64(s), limits.k3_nblk(s)) for s in (384, 4096, 4128, 4160)] == [
        (384, 4, 0, 4),
        (4096, 42, 1, 43),
        (4128, 43, 0, 43),
        (4160, 42, 2, 44),
    ]
    assert not limits.k3_ok(32) and limits.k3_ok(64) and limits.k3_ok(96)
    assert limits.dq_split(4096) == (256, 1, 10) and limits.dq_split(3840) == (0, 0, 10)
    with pytest.raises(NotImplementedError):
        limits.dq_split(512)
    assert limits.ds_layout_bytes(2, 4096, 4096, 128, True) == 4_328_521_728
    assert limits.ds_layout_bytes(1, 256, 384, 2, False) == 2 * 24 * 8 * 1024
    for b, sq, skv, h in ((1, 256, 256, 2), (2, 4096, 4096, 3), (1, 256, 448, 1), (1, 384, 1024, 2)):
        attended = torch.ones(sq, skv, dtype=torch.bool).tril(skv - sq)
        blocks = attended.view(sq // 32, 32, skv // 16, 16).any(3).any(1).sum().item()
        assert limits.ds_layout_bytes(b, sq, skv, h, True) == b * h * blocks * 1024, (sq, skv)


@pytest.mark.parametrize(
    ("dims", "npw"),
    [
        ((1, 4096, 256), 256),  # DeepSeek-V3 MBS 2, SBHD folded
        ((2, 4096, 128), 256),  # the same, BSHD
        ((1, 1024, 128), 256),
        ((1, 4096, 16), 0),  # one step per group: odd
        ((1, 256, 8), 0),  # a group's heads are not a whole number of steps
        ((1, 640, 8), 0),  # seqlen not a multiple of 256
        ((1, 4096, 12), 0),  # heads not a multiple of 8
        ((1, 2560, 256), 0),  # 10 q blocks do not divide a group of 32 workgroups
    ],
)
def test_persistent_workgroups(dims, npw):
    fwd = _hk("fwd")
    assert fwd.persistent_workgroups(*dims) == npw
    if npw:  # the static schedule: every (q block, head-batch) exactly once, equal tiles per step pair
        b, s, h = dims
        nqb, groups = s // 256, npw // 8
        r_heads = groups // nqb
        steps = b * h // 8 // r_heads
        seen, tiles = {}, []
        for wg in range(npw):
            r, i = divmod(wg // 8, nqb)
            t = 0
            for step in range(steps):
                hb = 8 * (r_heads * step + r) + wg % 8
                qb = nqb - 1 - i if step % 2 else i
                seen[(qb, hb)] = seen.get((qb, hb), 0) + 1
                t += qb + 1
            tiles.append(t)
        assert sorted(seen) == [(qb, hb) for qb in range(nqb) for hb in range(b * h)] and set(
            seen.values()
        ) == {1}
        assert len(set(tiles)) == 1


# ---------------------------------------------------------------------------
# Routing and tracing
# ---------------------------------------------------------------------------


def _stub_kernels(monkeypatch, calls):
    """The four custom ops replaced by the CPU reference on the kernel layout; ``calls`` records which ran."""

    def fwd(tag):
        def run(q, k, v, softmax_scale, causal):
            calls.append((f"{tag}_fwd", tuple(q.shape)))
            out, lse, *_ = _reference(q, k, v, torch.zeros(*q.shape[:3], D_V), softmax_scale)
            return out.to(q.dtype).contiguous(), lse.contiguous()

        return run

    def bwd(tag):
        def run(dout, q, k, v, out, lse, softmax_scale, causal):
            calls.append((f"{tag}_bwd", tuple(q.shape)))
            return tuple(g.to(q.dtype).contiguous() for g in _reference(q, k, v, dout, softmax_scale)[2:])

        return run

    monkeypatch.setattr(mla_impl, "_hk_code_objects_reason", lambda p: None)
    monkeypatch.setattr(mla_impl, "_forward_op", fwd("flydsl"))
    monkeypatch.setattr(mla_impl, "_forward_hk_op", fwd("hk"))
    monkeypatch.setattr(mla_impl, "_backward_op", bwd("flydsl"))
    monkeypatch.setattr(mla_impl, "_backward_hk_op", bwd("hk"))


@pytest.mark.parametrize(
    ("dims", "fwd", "bwd", "expected"),
    [
        ((256, 256), "hk", "hk", ("hk_fwd", "hk_bwd")),
        ((256, 256), "hk", "flydsl", ("hk_fwd", "flydsl_bwd")),
        ((256, 256), "flydsl", "hk", ("flydsl_fwd", "hk_bwd")),
        ((256, 256), "flydsl", "flydsl", ("flydsl_fwd", "flydsl_bwd")),
        ((128, 128), "hk", "hk", ("hk_fwd", "flydsl_bwd")),  # seqlen 128: only the HIP forward takes it
        ((256, 384), "hk", "hk", ("flydsl_fwd", "hk_bwd")),  # seqlen_q < seqlen_kv: only the HIP backward
    ],
)
def test_routes_and_falls_back(monkeypatch, dims, fwd, bwd, expected):
    """Each pass takes its own setting and falls back to FlyDSL on its own; both see the same kernel-layout
    tensors (b > 1 SBHD folded to [1, s, b*h, d])."""
    _on_gfx1250(monkeypatch)
    _settings(monkeypatch, fwd, bwd)
    calls = []
    _stub_kernels(monkeypatch, calls)
    b, h = 2, 2
    q, k, v, dout = _inputs(b, *dims, h, layout="sbhd")
    qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out = flash_attn_func(qg, kg, vg, softmax_scale=MLA_SCALE, causal=True)
    out.backward(dout)
    assert calls == [(expected[0], (1, dims[0], b * h, D_QK)), (expected[1], (1, dims[0], b * h, D_QK))]
    ref = _reference(q, k, v, dout, MLA_SCALE)
    for name, x, r in zip(("out", "dq", "dk", "dv"), (out, qg.grad, kg.grad, vg.grad), ref[:1] + ref[2:]):
        assert compute_snr(r, x.detach().float()) > 40, name


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    ("dims", "ops"),
    [((256, 256), ("hk", "hk")), ((128, 128), ("hk", "flydsl")), ((256, 384), ("flydsl", "hk"))],
    ids=["hk", "bwd_falls_back", "fwd_falls_back"],
)
def test_torch_compile_traces_the_hk_ops(monkeypatch, dims, ops, dynamic):
    """torch.compile(fullgraph=True) on meta tensors with both settings on hk: each graph holds the custom op its
    pass routes to, with the eager shapes; a pass that falls back does not break the graph (its warning is not
    traced: the log-once cache starts empty)."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    _on_gfx1250(monkeypatch)
    _settings(monkeypatch, "hk", "hk")
    monkeypatch.setattr(mla_impl, "_hk_code_objects_reason", lambda p: None)
    monkeypatch.setattr(mla_impl.logger, "_log_once_cache", set())
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if str(n.target).startswith("primus_turbo.")]
            return make_boxed_func(gm.forward)

        return compiler

    b, h = 2, 4
    s = dims[0]
    q, k, v = (t.requires_grad_() for t in _meta(b, *dims, h, layout="sbhd"))

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
        "fw": [f"primus_turbo.flash_attn_{ops[0]}_gfx1250_mla_forward.default"],
        "bw": [f"primus_turbo.flash_attn_{ops[1]}_gfx1250_mla_backward.default"],
    }
    assert out.shape == (b, s, h, D_V) and lse.shape == (b, h, s) and q.grad.shape == q.shape


# ---------------------------------------------------------------------------
# Launch plans (meta tensors)
# ---------------------------------------------------------------------------


def _manifest_kernels():
    return _hk("hsaco").load_manifest()["kernels"]


def test_fwd_plan_on_meta_tensors():
    fwd = _hk("fwd")
    args = _manifest_kernels()["k_mla_fwd"]["args"]
    q, k, v, _, _ = _meta_problem(1, 4096, 256)
    grid, values, out, lse = fwd._plan(q, k, v, MLA_SCALE, True)
    assert grid == (256, 1, 1) and len(values) == len(args)
    assert values[:5] == (q, k, v, out, lse) and values[5:] == (MLA_SCALE, 4096, 4096, 256, 1, 16, 256)
    assert out.shape == (1, 4096, 256, D_V) and lse.shape == (1, 256, 4096) and lse.dtype == torch.float32
    q, k, v, _, _ = _meta_problem(2, 640, 4)
    grid, values, _, _ = fwd._plan(q, k, v, None, True)
    assert grid == (3, 4, 2) and values[5] == D_QK**-0.5 and values[-2:] == (3, 0)
    with pytest.raises(NotImplementedError, match="seqlen_q must equal seqlen_kv"):
        fwd._plan(*_meta(1, 256, 384, 2), MLA_SCALE, True)
    with pytest.raises(ValueError, match="contiguous bfloat16"):  # b > 1 SBHD storage: a strided view
        fwd._plan(*_meta(2, 256, 256, 2, layout="sbhd")[:1], *_meta(2, 256, 256, 2)[1:], MLA_SCALE, True)


def test_bwd_plan_on_meta_tensors():
    """DeepSeek-V3's shape: the delta launch, then the ticket step, dkdv and dq; grids, the dS scratch size and
    argument tuples that match the manifest's kernel signatures."""
    bwd = _hk("bwd")
    kernels = _manifest_kernels()
    b, s, h = 2, 4096, 128
    q, k, v, o, lse = _meta_problem(b, s, h)
    plan = bwd._plan(o, q, k, v, o, lse, MLA_SCALE, True)
    launches = plan["launches"]
    assert [x[0] for x in launches] == ["delta", "ticket", "dkdv", "dq"]
    grids = {name: args[0] for name, fn, args in launches if isinstance(fn, str)}
    assert grids == {"delta": (64, 256, 1), "dkdv": (32, 43, 8), "dq": (32, 11, 8)}
    assert plan["ds_layout_bytes"] == 4_328_521_728
    assert plan["ds_full"].numel() * 2 == 4_328_521_728 + 2 * bwd.DS_GUARD_B
    assert launches[1] == ("ticket", bwd._zero_ticket, (plan["ds_full"],))
    for _, name, args in launches:
        if isinstance(name, str):
            grid, block, values = args
            assert block == kernels[name]["block"] and len(values) == len(kernels[name]["args"]), name
    delta_values = launches[0][2][2]
    assert delta_values[:6] == (o, o, plan["delta"], lse, plan["nl"], plan["nd"])
    assert delta_values[6:] == (bwd.log2_scale_fp32(MLA_SCALE), s, h)
    assert launches[2][2][2][4:6] == (plan["nl"], plan["nd"])
    with pytest.raises(ValueError, match="not verified"):
        bwd._plan(o, q, k, v, o, lse, MLA_SCALE, True, head_group=48)
    with pytest.raises(NotImplementedError, match="seqlen_q % 384"):
        bwd._plan(*(_meta_problem(1, 512, 2)[i] for i in (3, 0, 1, 2, 3, 4)), MLA_SCALE, True)


def test_bwd_plan_head_chunks_under_a_scratch_budget():
    """A dS scratch budget below the whole call's splits (ticket, dkdv, dq) into head chunks of one batch that
    share one scratch; no head group fitting raises ScratchBudgetError."""
    bwd = _hk("bwd")
    b, s, h = 2, 4096, 128
    q, k, v, o, lse = _meta_problem(b, s, h)
    assert bwd._plan(o, q, k, v, o, lse, MLA_SCALE, True)["chunks"] is None
    half = bwd._scratch_bytes(1, s, s, 64, True)
    plan = bwd._plan(o, q, k, v, o, lse, MLA_SCALE, True, max_scratch_bytes=half)
    assert plan["chunks"] == [(0, 0, 64), (0, 64, 64), (1, 0, 64), (1, 64, 64)]
    assert [x[0] for x in plan["launches"]] == ["delta"] + ["ticket", "dkdv", "dq"] * 4
    assert plan["ds_full"].numel() * 2 == half
    grids = {x[0]: x[2][0] for x in plan["launches"] if x[0] in ("dkdv", "dq")}
    assert grids == {"dkdv": (32, 43, 2), "dq": (32, 11, 2)}
    with pytest.raises(bwd.ScratchBudgetError):
        bwd._plan(
            o, q, k, v, o, lse, MLA_SCALE, True, max_scratch_bytes=bwd._scratch_bytes(1, s, s, 32, True) - 1
        )


def test_zero_ticket_touches_only_the_counter():
    bwd = _hk("bwd")
    ds_full = torch.full(((2 * bwd.DS_GUARD_B + 1024) // 2,), 3.0, dtype=torch.bfloat16)
    before = ds_full.view(torch.uint8).clone()
    bwd._zero_ticket(ds_full)
    after = ds_full.view(torch.uint8)
    lo, hi = bwd.TICKET_OFFSET_B, bwd.DS_GUARD_B
    assert torch.all(after[lo:hi] == 0)
    assert torch.equal(after[:lo], before[:lo]) and torch.equal(after[hi:], before[hi:])


def test_max_scratch_bytes_setting(monkeypatch):
    bwd = _hk("bwd")
    for text, want in (("", None), ("2.2e9", 2_200_000_000), ("4096", 4096)):
        monkeypatch.setenv(bwd.MAX_SCRATCH_BYTES_ENV, text)
        assert bwd._bytes_setting(bwd.MAX_SCRATCH_BYTES_ENV) == want
    for text in ("0", "-1", "lots", "inf", "nan"):
        monkeypatch.setenv(bwd.MAX_SCRATCH_BYTES_ENV, text)
        with pytest.raises(ValueError, match="positive byte count"):
            bwd._bytes_setting(bwd.MAX_SCRATCH_BYTES_ENV)


# ---------------------------------------------------------------------------
# Build: manifest, sources, code objects, assembly patches
# ---------------------------------------------------------------------------


def test_manifest():
    """Every kernel belongs to one pass, its source is shipped, its arguments tile its kernarg segment in order,
    and the flags are a hermetic device-only gfx1250 build: relative include paths inside the package only (no
    3rdparty/hipkittens, no KITTENS_* defines)."""
    hsaco, hiprt = _hk("hsaco"), _hk("hiprt")
    m = hsaco.load_manifest()
    assert (
        m["arch"] == "gfx1250"
        and "--offload-arch=gfx1250" in m["flags"]
        and "--cuda-device-only" in m["flags"]
    )
    assert sorted(m["kernels"]) == sorted(hsaco.FWD_KERNELS + hsaco.BWD_KERNELS)
    for name, k in m["kernels"].items():
        flags = m["flags"] + k["extra_flags"]
        assert not any("KITTENS" in f or "3rdparty" in f or f.startswith("-I/") for f in flags), name
        assert all(f in ("-Iinclude", "-I.") for f in flags if f.startswith("-I")), name
        assert (hsaco.HIP_DIR / k["src"]).is_file() and re.fullmatch(r"[0-9a-f]{64}", k["hsaco_sha256"]), name
        if "asm_patch" in k:
            assert (hsaco.HIP_DIR / k["asm_patch"]).is_file() and re.fullmatch(
                r"[0-9a-f]{64}", k["asm_input_sha256"]
            )
        last = k["args"][-1]
        assert last["offset"] + last["size"] <= k["kernarg_size"], name
        values = [1 if a["kind"] == "ptr" else 0 for a in k["args"]]
        assert len(hiprt.pack_args(k["args"], values)) == last["offset"] + last["size"], name
    assert m["kernels"]["k_mla_fwd"]["block"] == _hk("fwd").BLOCK
    assert m["kernels"]["k_delta"]["block"] == _hk("bwd").DELTA_BLOCK


_SYSTEM_HEADERS = {"hip/hip_runtime.h", "stdint.h", "utility"}


def test_sources_are_self_contained():
    """The kernels include only the HIP runtime, two C++ headers and the package's own headers: nothing from
    3rdparty/hipkittens (main's gfx950 HipKittens) or any other tree can leak into them."""
    hsaco = _hk("hsaco")
    files = sorted(p for p in hsaco.HIP_DIR.rglob("*") if p.suffix in (".hip", ".h"))
    assert files
    for path in files:
        text = path.read_text()
        assert "kittens::" not in text, path
        for quoted, name in re.findall(r'#\s*include\s+(["<])([^">]+)[">]', text):
            if quoted == "<":
                assert name in _SYSTEM_HEADERS, (path, name)
            else:
                found = [d / name for d in (path.parent, hsaco.HIP_DIR / "include", hsaco.HIP_DIR)]
                assert any(f.is_file() for f in found), (path, name)
                assert all(f.resolve().is_relative_to(hsaco.HIP_DIR) for f in found if f.is_file()), (
                    path,
                    name,
                )


def test_code_object_checks(monkeypatch, tmp_path):
    """Availability is per pass: a missing or foreign code object of one pass leaves the other available, and
    hiprt refuses to load it. A stand-in manifest pins placeholder bytes (no compiler, no GPU)."""
    hsaco, hiprt = _hk("hsaco"), _hk("hiprt")
    man = hsaco.load_manifest()
    blobs = {n: f"code object {n}".encode() for n in man["kernels"]}
    pins = {n: dict(k, hsaco_sha256=hashlib.sha256(blobs[n]).hexdigest()) for n, k in man["kernels"].items()}
    monkeypatch.setattr(hsaco, "_MANIFEST", dict(man, kernels=pins))
    monkeypatch.setattr(hsaco, "_REASONS", {})

    def bin_dir(name, **override):
        d = tmp_path / name
        d.mkdir()
        for n, data in blobs.items():
            data = override.get(n, data)
            if data is not None:
                (d / f"{n}.hsaco").write_bytes(data)
        return d

    full = bin_dir("full")
    assert hsaco.unavailable_reason(hsaco.FWD_KERNELS, full) is None
    assert hsaco.unavailable_reason(hsaco.BWD_KERNELS, full) is None
    assert set(hiprt.load_kernels(hsaco.BWD_KERNELS, full)) == set(hsaco.BWD_KERNELS)
    for name, data, why in (("missing", None, "not built"), ("foreign", b"other bytes", "sha256")):
        d = bin_dir(name, k_delta=data)
        assert hsaco.unavailable_reason(hsaco.FWD_KERNELS, d) is None
        assert why in hsaco.unavailable_reason(hsaco.BWD_KERNELS, d)
        assert set(hiprt.load_kernels(hsaco.FWD_KERNELS, d)) == {"k_mla_fwd"}
        with pytest.raises(hiprt.HipError, match="unavailable"):
            hiprt.load_kernels(hsaco.BWD_KERNELS, d)


def test_build_never_fails_and_keeps_good_code_objects(monkeypatch, tmp_path):
    """hsaco.build with a compiler that is not there, or one that emits other bytes, returns False without raising,
    and leaves the code objects already in place untouched (a new one only replaces them when it is the pinned one)."""
    hsaco = _hk("hsaco")
    monkeypatch.setattr(hsaco, "_REASONS", {})
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    before = {}
    for name in hsaco.load_manifest()["kernels"]:
        before[name] = f"old {name}".encode()
        (bin_dir / f"{name}.hsaco").write_bytes(before[name])
    fake = tmp_path / "fake-clang"
    fake.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && echo junk > "$2"; shift; done\n')
    fake.chmod(0o755)
    logs = []
    for cxx in (str(tmp_path / "missing-clang"), str(fake)):
        assert hsaco.build(cxx=cxx, bin_dir=bin_dir, force=True, log=logs.append) is False
        assert {n: (bin_dir / f"{n}.hsaco").read_bytes() for n in before} == before
        assert not list(bin_dir.glob("*.tmp"))
    assert any("not found" in m for m in logs) and any("not built" in m for m in logs)


_ASM_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")
_ASM_ALLOC = re.compile(r"\s*(\.amdhsa_next_free_vgpr\s+|\.vgpr_count:\s+)(\d+)$")
# the markers each generated region goes behind, every one exactly once
_ASM_REGIONS = {
    "k_mla_fwd": (
        re.compile(r";\s*HKGEN (INIT|TILE_C|TILE_D[0-3]|FINAL_[0-3])\b"),
        ["INIT", "TILE_C"] + [f"TILE_D{i}" for i in range(4)] + [f"FINAL_{i}" for i in range(4)],
    ),
    "k_dkdv96ds": (
        re.compile(r";\s*HKGEN (INIT|LOOP|FINAL) nkv=(\d)\b"),
        [f"{r} nkv={n}" for r in ("INIT", "LOOP", "FINAL") for n in (3, 2)],
    ),
}


def _asm_patch_hunks(patch):
    """[(old start, new start, [(tag, line)])] of a one-file unified diff, each hunk's counts checked."""
    hunks = []
    for block in patch.split("\n@@ ")[1:]:
        header, *body = ("@@ " + block).rstrip("\n").split("\n")
        m = _ASM_HUNK.match(header)
        assert m and all(line[:1] in (" ", "-", "+") for line in body), header
        ops = [(line[0], line[1:]) for line in body]
        assert sum(t != "+" for t, _ in ops) == int(m.group(2) or 1), header
        assert sum(t != "-" for t, _ in ops) == int(m.group(4) or 1), header
        hunks.append((int(m.group(1)), int(m.group(3)), ops))
    return hunks


@pytest.mark.parametrize("name", sorted(_ASM_REGIONS))
def test_asm_patch(tmp_path, name):
    """The patch (tools/regen_asm_patch.py) has the generated kernel's shape: each hunk inserts one region right
    behind its marker line (every region exactly once; no s_endpgm or PC change inside, branches only to the
    region's own labels) or raises the VGPR allocation to 1024. hsaco.py applies it to its own context only: a
    changed context line is rejected, and a compiler .s with another sha256 is refused before anything is patched."""
    hsaco = _hk("hsaco")
    k = hsaco.load_manifest()["kernels"][name]
    assert k["vgpr"] == 1024
    marker_re, want_regions = _ASM_REGIONS[name]
    patch = (hsaco.HIP_DIR / k["asm_patch"]).read_text()
    src, want, shift, regions, allocs = [], [], 0, [], 0
    for old_start, new_start, ops in _asm_patch_hunks(patch):
        assert old_start > len(src) and new_start == old_start + shift, old_start
        fill = [f"\t; line {n + 1}" for n in range(len(src), old_start - 1)]
        src += fill + [t for tag, t in ops if tag != "+"]
        want += fill + [t for tag, t in ops if tag != "-"]
        shift += sum(t == "+" for t, _ in ops) - sum(t == "-" for t, _ in ops)
        old = [t for tag, t in ops if tag == "-"]
        new = [t for tag, t in ops if tag == "+"]
        if old:  # the VGPR allocation, one line
            assert len(old) == len(new) == 1, old_start
            mo, mn = _ASM_ALLOC.fullmatch(old[0]), _ASM_ALLOC.fullmatch(new[0])
            assert mo and mn and mo.group(1) == mn.group(1) and mn.group(2) == "1024", (old, new)
            allocs += 1
            continue
        first_new = next(j for j, (tag, _) in enumerate(ops) if tag == "+")
        marker = marker_re.search(ops[first_new - 1][1]) if first_new else None
        assert marker and all(tag == "+" for tag, _ in ops[first_new : first_new + len(new)]), old_start
        labels = {t.strip()[:-1] for t in new if re.fullmatch(r"\s*\.L\w+:", t)}
        for t in new:
            ins = t.split(";")[0].strip()
            assert not re.match(r"s_(endpgm|setpc|swappc|getpc|rfe)", ins), t
            if re.match(r"s_(c)?branch", ins):
                assert ins.split()[-1] in labels, t
        regions.append(
            marker.group(1) if marker.lastindex == 1 else f"{marker.group(1)} nkv={marker.group(2)}"
        )
    assert sorted(regions) == sorted(want_regions) and allocs == 2
    text = "\n".join(src + ["\ts_endpgm"]) + "\n"
    assert hsaco.apply_unified_diff(text, patch) == "\n".join(want + ["\ts_endpgm"]) + "\n"
    first = _asm_patch_hunks(patch)[0][0] - 1  # the first hunk's first (context) line
    bad = "\n".join(src[:first] + [src[first] + " "] + src[first + 1 :] + ["\ts_endpgm"]) + "\n"
    with pytest.raises(ValueError, match="does not apply"):
        hsaco.apply_unified_diff(bad, patch)
    saved = tmp_path / f"{name}-hip-amdgpu12.50-amd-amdhsa-gfx1250.s"  # as -save-temps=obj names it
    saved.write_text(text)
    with pytest.raises(RuntimeError, match="asm_input_sha256"):
        hsaco.patch_and_reassemble(name, k, [], tmp_path)
    saved.write_text(bad)
    with pytest.raises(RuntimeError, match="does not apply"):
        hsaco.patch_and_reassemble(
            name, dict(k, asm_input_sha256=hashlib.sha256(bad.encode()).hexdigest()), [], tmp_path
        )
    assert saved.read_text() == bad


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


def _kernel_inputs(b, s, h, seed=0):
    """Kernel-layout (contiguous [b, s, h, d]) q, k, v on the card."""
    q, k, v, _ = _inputs(b, s, s, h, layout="bshd", seed=seed)
    return tuple(t.cuda() for t in (q, k, v))


@needs_hk_fwd
@pytest.mark.parametrize(
    "dims",
    [(1, 64, 2), (1, 256, 2), (2, 256, 2), (1, 640, 4), (1, 1024, 128), (1, 4096, 256)],
    ids=["s64", "toy", "b2", "s640_partial_block", "persistent", "dsv3_mbs2_folded"],
)
def test_hk_fwd_bitwise_equals_flydsl(dims):
    """out and lse bitwise equal to the FlyDSL forward's, every element written (outputs NaN-filled first), and a
    rerun bitwise identical; the persistent grid included."""
    q, k, v = _kernel_inputs(*dims)
    with _nan_filled_outputs():
        got = _hk("fwd").hk_attn_fwd(q, k, v, MLA_SCALE, True)
        again = _hk("fwd").hk_attn_fwd(q, k, v, MLA_SCALE, True)
        fly = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    for name, a, b, c in zip(("out", "lse"), got, again, fly):
        assert torch.isfinite(a).all() and torch.equal(a, b) and torch.equal(a, c), name


@needs_hk_fwd
@pytest.mark.parametrize(
    ("dims", "layout", "scale"),
    [
        ((2, 256, 256, 2), "sbhd", MLA_SCALE),
        ((1, 640, 640, 4), "bshd", MLA_SCALE),
        ((1, 256, 256, 2), "bhsd", None),
    ],
    ids=["b2_sbhd_folded", "s640", "bhsd_default_scale"],
)
def test_hk_fwd_matches_reference(monkeypatch, dims, layout, scale):
    """flash_attn_func with PRIMUS_TURBO_MLA_FWD_IMPL=hk (the HIP forward is used) against the fp32 reference,
    gradients included (FlyDSL backward)."""
    _settings(monkeypatch, fwd="hk")
    calls = []
    real = mla_impl._forward_hk_op
    monkeypatch.setattr(mla_impl, "_forward_hk_op", lambda *a: calls.append(a[0].shape) or real(*a))
    q, k, v, dout = _inputs(*dims, layout=layout)
    got = _run(q, k, v, dout, scale)
    assert len(calls) == 1, "the HIP forward was not used"
    _check(got, _reference(q, k, v, dout, D_QK**-0.5 if scale is None else scale))


def _run_hk_bwd(monkeypatch, q, k, v, dout):
    _settings(monkeypatch, bwd="hk")
    calls = []
    real = mla_impl._backward_hk_op
    monkeypatch.setattr(mla_impl, "_backward_hk_op", lambda *a: calls.append(a[1].shape) or real(*a))
    got = _run(q, k, v, dout)
    assert len(calls) == 1, "the HIP backward was not used"
    return got


@needs_hk_bwd
@pytest.mark.parametrize(
    ("dims", "layout"),
    [
        ((1, 256, 256, 2), "sbhd"),
        ((2, 256, 256, 2), "sbhd"),
        ((1, 256, 384, 2), "sbhd"),
        ((1, 640, 640, 4), "bshd"),
    ],
    ids=["toy", "toy_folded", "rect", "s640"],
)
def test_hk_bwd_matches_reference_and_flydsl(monkeypatch, dims, layout):
    """The HIP backward matches the fp32 reference, is bitwise deterministic, and every gradient's SNR is within
    0.3 dB of the FlyDSL backward's (the folded ND rounds scale * P instead of P)."""
    q, k, v, dout = _inputs(*dims, layout=layout)
    got = _run_hk_bwd(monkeypatch, q, k, v, dout)
    ref = _reference(q, k, v, dout, MLA_SCALE)
    _check(got, ref)
    again = _run_hk_bwd(monkeypatch, q, k, v, dout)
    _settings(monkeypatch)
    fly = _run(q, k, v, dout)
    for name, a, b, c, r in zip(("out", "lse", "dq", "dk", "dv"), got, again, fly, ref):
        assert torch.equal(a, b), name
        if name in ("out", "lse"):
            assert torch.equal(a, c), name
        else:
            snr_hk, snr_fly = (
                compute_snr(r, a.detach().float().cpu()),
                compute_snr(r, c.detach().float().cpu()),
            )
            assert snr_hk >= snr_fly - 0.3, f"{name}: {snr_hk:.2f} dB vs FlyDSL {snr_fly:.2f} dB"


@needs_hk_bwd
def test_hk_bwd_production_shape_matches_reference(monkeypatch):
    """(1, 4096, 4096, 16): 42 96-row kv blocks + the 64-row tail, 10 384-query dQ tiles + the 256-query head;
    with the dS scratch NaN-filled, so a block dQ reads but dK/dV never wrote fails."""
    monkeypatch.setattr(_hk("bwd"), "POISON", True)
    q, k, v, dout = _inputs(1, 4096, 4096, 16, layout="bshd")
    _check(_run_hk_bwd(monkeypatch, q, k, v, dout), _reference_by_heads(q, k, v, dout, MLA_SCALE))


@needs_hk_bwd
def test_hk_bwd_rect32_matches_reference(monkeypatch):
    """hk_attn_bwd called directly at (Skv - Sq) % 64 == 32 (the FlyDSL gate refuses it, the HIP kernels' shape
    rules admit it): dQ's last steady unit reads the first masked kv32 row, whose stored pairs start mid-row. Outputs
    and scratch NaN-filled."""
    bwd = _hk("bwd")
    monkeypatch.setattr(bwd, "POISON", True)
    q, k, v, dout = _inputs(1, 4096, 4128, 4, layout="bshd")
    ref = _reference_by_heads(q, k, v, dout, MLA_SCALE)
    out, lse = ref[0].to(torch.bfloat16).cuda(), ref[1].float().cuda()
    args = [t.cuda().contiguous() for t in (dout, q, k, v)]
    got = bwd.hk_attn_bwd(*args, out, lse, MLA_SCALE, True)
    again = bwd.hk_attn_bwd(*args, out, lse, MLA_SCALE, True)
    for name, x, y, r in zip(("dq", "dk", "dv"), got, again, ref[2:]):
        assert torch.equal(x, y), name
        snr = compute_snr(r, x.float().cpu())
        assert snr >= 48.0, f"{name}: {snr:.1f} dB < 48 dB"


@needs_hk_bwd
def test_hk_bwd_head_chunks_are_bitwise_equal(monkeypatch):
    """Head-chunked calls (2 batches x 2 chunks of 32 heads: views offset in batch and head, one shared scratch)
    return exactly the unchunked gradients; outputs and scratch NaN-filled."""
    bwd = _hk("bwd")
    monkeypatch.setattr(bwd, "POISON", True)
    q, k, v, dout = (t.cuda() for t in _inputs(2, 256, 256, 64, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    full = bwd.hk_attn_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
    budget = bwd._scratch_bytes(1, 256, 256, 32, True)
    assert bwd._plan(dout, q, k, v, out, lse, MLA_SCALE, True, max_scratch_bytes=budget)["chunks"] == [
        (0, 0, 32),
        (0, 32, 32),
        (1, 0, 32),
        (1, 32, 32),
    ]
    chunked = bwd.hk_attn_bwd(dout, q, k, v, out, lse, MLA_SCALE, True, max_scratch_bytes=budget)
    for a, b in zip(full, chunked):
        assert torch.isfinite(a).all() and torch.equal(a, b)


def _delta_inputs(b, s, h, special, seed=0):
    """Kernel-layout dO / O [b, s, h, 128] bf16 and lse [b, h, s] fp32 on the card. special: every (row, head) of dO
    and of O scaled by 2^-k, k in [0, 80), so the bf16 x bf16 products range from normal to below the fp32 denormal
    range (where a fused multiply-add rounds differently from a multiply then an add); 5% bf16 denormals, 2% signed
    zeros; lse with 5% fp32 denormals, 1% signed zeros and 1% +-3e38 (NL overflows to +-inf)."""
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


@needs_hk_bwd
@pytest.mark.parametrize(
    ("dims", "special"),
    [
        ((1, 256, 2), False),
        ((2, 256, 4), False),
        ((1, 640, 4), False),
        ((1, 4096, 16), False),
        ((2, 256, 4), True),
        ((1, 640, 4), True),
    ],  # fmt: skip
    ids=["toy", "b2", "s640", "s4096", "b2_denormals", "s640_denormals"],
)
def test_hk_delta_matches_flydsl(dims, special):
    """k_delta's delta bitwise equals the FlyDSL k_delta's, and its NL / ND equal the FlyDSL ones folded on the host
    (NL + fp32 log2(scale), -delta): also for denormal inputs, products that underflow fp32, signed zeros and an lse
    whose NL overflows."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd as fly

    bwd = _hk("bwd")
    do, o, lse = _delta_inputs(*dims, special)
    b, s, h, _ = do.shape
    stream = torch.cuda.current_stream()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bwd, "POISON", True)
        q = torch.empty(b, s, h, D_QK, dtype=torch.bfloat16, device="cuda")
        v = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="cuda")
        plan = bwd._plan(do, q, q, v, o, lse, MLA_SCALE, True)
    grid, block, values = plan["launches"][0][2]
    bwd.kernels()["k_delta"].call(grid, block, values, stream)
    delta, nl, nd = (torch.full((b, h, s), float("nan"), device="cuda") for _ in range(3))
    fly._launch(
        fly._k.launch_delta,
        (do, o, delta, lse, nl, nd, MLA_SCALE, s, h, b * s * h, s // fly._k.ROWS_DELTA, b * h, stream),
    )
    want = (delta, nl + bwd.log2_scale_fp32(MLA_SCALE), -delta)
    for name, x, y in zip(("delta", "NL", "ND"), (plan["delta"], plan["nl"], plan["nd"]), want):
        assert not torch.isnan(x).any(), name
        assert torch.equal(x.view(torch.int32), y.view(torch.int32)), name
    if special:
        tiny = (plan["delta"] != 0) & (plan["delta"].abs() < torch.finfo(torch.float32).tiny)
        assert tiny.any(), "no denormal delta: the inputs do not reach the fp32 denormal range"


@needs_hk
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_hk_end_to_end(monkeypatch, compiled):
    """Both passes on the HIP kernels through flash_attn_func (SBHD, folded), eager and torch.compile'd: out / lse
    bitwise those of the FlyDSL kernels, gradients bitwise deterministic and as close to the reference."""
    q, k, v, dout = _inputs(2, 256, 256, 8, layout="sbhd")
    runs = {"fwd": 0, "bwd": 0}
    for pass_, entry in (("fwd", "hk_attn_fwd"), ("bwd", "hk_attn_bwd")):
        real = getattr(_hk(pass_), entry)
        monkeypatch.setattr(
            _hk(pass_), entry, lambda *a, p=pass_, f=real, **kw: runs.update({p: runs[p] + 1}) or f(*a, **kw)
        )
    attn = flash_attn_func
    if compiled:
        torch._dynamo.reset()
        attn = torch.compile(flash_attn_func, fullgraph=True, backend="aot_eager")
    try:
        _settings(monkeypatch, "hk", "hk")
        got = _run(q, k, v, dout, attn=attn)
        again = _run(q, k, v, dout, attn=attn)
        _settings(monkeypatch)
        fly = _run(q, k, v, dout)
    finally:
        torch._dynamo.reset()
    assert runs == {"fwd": 2, "bwd": 2}, runs
    ref = _reference(q, k, v, dout, MLA_SCALE)
    _check(got, ref)
    for name, a, b, c, r in zip(("out", "lse", "dq", "dk", "dv"), got, again, fly, ref):
        assert torch.equal(a, b), name
        if name in ("out", "lse"):
            assert torch.equal(a, c), name
        else:
            assert (
                compute_snr(r, a.detach().float().cpu()) >= compute_snr(r, c.detach().float().cpu()) - 0.3
            ), name
