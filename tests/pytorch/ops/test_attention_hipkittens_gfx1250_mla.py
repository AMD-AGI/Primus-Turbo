###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens kernels of the gfx1250 (MI455X) MLA attention: ``attention_hipkittens_impl.hk_mla_*``, built from
csrc/kernels/attention/hipkittens/gfx1250_mla.

Settings, gates, routing, launch plans and the build run on meta / CPU tensors with stand-in kernels. The numeric
tests need a gfx1250 card and the built code objects; their fp32 reference runs on the CPU.
"""

import hashlib
import importlib.util
import math
import re
import struct
from pathlib import Path

import pytest
import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import attention_hipkittens_impl as hk
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


needs_hk = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250())
    or hk.hk_mla_unavailable_reason("fwd") is not None
    or hk.hk_mla_unavailable_reason("bwd") is not None,
    reason="needs a gfx1250 (MI455X) GPU and the HipKittens MLA code objects",
)


def _settings(monkeypatch, fwd="flydsl", bwd="flydsl"):
    monkeypatch.setattr(mla_impl, "MLA_FWD_IMPL", fwd)
    monkeypatch.setattr(mla_impl, "MLA_BWD_IMPL", bwd)


def _meta_bwd(b, s, h, skv=None):
    """Meta (dout, q, k, v, out, lse) in the kernel layout."""
    q, k, v = _meta(b, s, s if skv is None else skv, h)
    out = torch.empty(b, s, h, D_V, dtype=torch.bfloat16, device="meta")
    return out, q, k, v, out, torch.empty(b, h, s, dtype=torch.float32, device="meta")


# ---------------------------------------------------------------------------
# Settings and gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["PRIMUS_TURBO_MLA_FWD_IMPL", "PRIMUS_TURBO_MLA_BWD_IMPL"])
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
def test_impl_settings(monkeypatch, name, value, want):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    envs.reset_cache(name)
    if want is None:
        with pytest.raises(ValueError, match=name):
            getattr(envs, name)
    else:
        assert getattr(envs, name) == want


def test_max_scratch_bytes_setting(monkeypatch):
    name = "PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES"
    monkeypatch.delenv(name, raising=False)
    envs.reset_cache(name)
    assert envs.PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES is None
    monkeypatch.setenv(name, "4096")
    envs.reset_cache(name)
    assert envs.PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES == 4096
    monkeypatch.setenv(name, "0")
    envs.reset_cache(name)
    with pytest.raises(ValueError, match=name):
        getattr(envs, name)


_ACCEPTED = {
    "fwd": [
        (2, 4096, 4096, 128),
        (1, 4096, 4096, 256),
        (2, 4096, 4096, 96),
        (1, 256, 256, 2),
        (1, 640, 640, 4),
    ],
    "bwd": [
        (2, 4096, 4096, 128),
        (1, 4096, 4096, 256),
        (2, 4096, 4096, 96),
        (1, 256, 256, 2),
        (1, 256, 384, 2),
    ],
}
_REFUSED = {
    "fwd": [
        ((1, 256, 384, 2), "seqlen_q must equal seqlen_kv"),
        ((1, 256, 256, 4, 2), "MHA only"),
        ((1, 96, 96, 2), "multiple of 64"),
        ((1, 256, 128, 2), "seqlen_kv 128 < seqlen_q 256"),
    ],
    "bwd": [
        ((1, 512, 512, 2), "seqlen_q % 384"),
        ((1, 128, 128, 2), "seqlen_q % 384"),
        ((1, 256, 16384, 256), "exceeds the verified"),
        ((1, 256, 256, 4, 2), "MHA only"),
    ],
}


@pytest.mark.parametrize(("pass_", "dims"), [(p, d) for p, ds in _ACCEPTED.items() for d in ds])
def test_gate_accepts(monkeypatch, pass_, dims):
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda p: None)
    assert hk.hk_mla_unsupported_reason(pass_, *_meta(*dims), True, MLA_SCALE) is None


@pytest.mark.parametrize(("pass_", "dims", "match"), [(p, d, m) for p, ds in _REFUSED.items() for d, m in ds])
def test_gate_refuses(monkeypatch, pass_, dims, match):
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda p: None)
    reason = hk.hk_mla_unsupported_reason(pass_, *_meta(*dims), True, MLA_SCALE)
    assert reason is not None and match in reason, reason


@pytest.mark.parametrize("pass_", ["fwd", "bwd"])
def test_gate_refuses_without_the_code_objects(monkeypatch, tmp_path, pass_):
    monkeypatch.setattr(hk, "_MLA_HSACO_DIR", tmp_path)
    monkeypatch.setattr(hk, "_MLA_UNAVAILABLE", {})
    q, k, v = _meta(1, 256, 256, 2)
    assert "not built" in hk.hk_mla_unsupported_reason(pass_, q, k, v, True)
    assert hk.hk_mla_unsupported_reason(pass_, q, k, v, False) == "causal only"  # the shape rules come first


@pytest.mark.parametrize("scale", [0.0, -0.1, 2.0**-101, math.inf, math.nan])
def test_bwd_gate_refuses_scales_it_cannot_fold(monkeypatch, scale):
    """The backward folds log2(scale) into its row constants and accumulates dV from bf16(scale * P); the forward
    takes any scale, like FlyDSL's."""
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda p: None)
    q, k, v = _meta(1, 256, 256, 2)
    assert "softmax_scale must be finite" in hk.hk_mla_unsupported_reason("bwd", q, k, v, True, scale)
    assert hk.hk_mla_unsupported_reason("bwd", q, k, v, True, 2.0**-100) is None
    assert hk.hk_mla_unsupported_reason("fwd", q, k, v, True, scale) is None


def test_entry_points_validate_their_tensors():
    q, k, v = _meta(1, 256, 256, 2)
    with pytest.raises(ValueError, match="on meta"):
        hk._mla_check(q, torch.empty_like(k, device="cpu"), v)
    with pytest.raises(ValueError, match="strided"):  # b > 1 SBHD storage must reach the kernels folded
        hk._mla_check(*_meta(2, 256, 256, 2, layout="sbhd"))
    do, q, k, v, o, lse = _meta_bwd(1, 256, 2)
    with pytest.raises(ValueError, match="float32"):
        hk._mla_check(q, k, v, o, do, lse.to(torch.bfloat16))
    with pytest.raises(ValueError, match="empty problem"):
        hk._mla_check(*_meta(1, 0, 0, 2))


def test_shape_helpers():
    assert [hk._mla_dkdv_blocks(s) for s in (384, 4096, 4128, 4160)] == [4, 43, 43, 44]
    assert hk._mla_dq_split(4096) == (256, 1, 10) and hk._mla_dq_split(3840) == (0, 0, 10)
    assert hk._mla_ds_bytes(2, 4096, 4096, 128, True) == 4_328_521_728
    assert hk._mla_ds_bytes(1, 256, 384, 2, False) == 2 * 24 * 8 * 1024
    for b, sq, skv, h in ((1, 256, 256, 2), (2, 4096, 4096, 3), (1, 256, 448, 1), (1, 384, 1024, 2)):
        attended = torch.ones(sq, skv, dtype=torch.bool).tril(skv - sq)
        blocks = attended.view(sq // 32, 32, skv // 16, 16).any(3).any(1).sum().item()
        assert hk._mla_ds_bytes(b, sq, skv, h, True) == b * h * blocks * 1024, (sq, skv)


@pytest.mark.parametrize(
    ("dims", "cus", "npw"),
    [
        ((1, 4096, 256), 256, 256),  # DeepSeek-V3 MBS 2, SBHD folded
        ((1, 4096, 192), 256, 256),  # Kimi K3 MBS 2 (96 heads), SBHD folded
        ((2, 4096, 128), 256, 256),
        ((1, 1024, 128), 256, 256),
        ((1, 4096, 128), 128, 128),  # a part with 128 CUs
        ((1, 4096, 128), 252, 0),  # CUs not a multiple of the 8 groups
        ((1, 4096, 16), 256, 0),  # one step per group: odd
        ((1, 640, 8), 256, 0),  # seqlen not a multiple of 256
        ((1, 4096, 12), 256, 0),  # heads not a multiple of 8
        ((1, 2560, 256), 256, 0),  # 10 q blocks do not divide a group of 32 workgroups
    ],
)
def test_persistent_workgroups(dims, cus, npw):
    """The static schedule covers every (q block, head-batch) exactly once with equal tiles per workgroup."""
    assert hk.hk_mla_persistent_workgroups(*dims, cus) == npw
    if npw:
        b, s, h = dims
        nqb = s // 256
        r_heads = npw // 8 // nqb
        seen, tiles = {}, []
        for wg in range(npw):
            r, i = divmod(wg // 8, nqb)
            t = 0
            for step in range(b * h // 8 // r_heads):
                qb = nqb - 1 - i if step % 2 else i
                key = (qb, 8 * (r_heads * step + r) + wg % 8)
                seen[key] = seen.get(key, 0) + 1
                t += qb + 1
            tiles.append(t)
        assert sorted(seen) == [(qb, hb) for qb in range(nqb) for hb in range(b * h)]
        assert set(seen.values()) == {1} and len(set(tiles)) == 1


# ---------------------------------------------------------------------------
# Routing and tracing
# ---------------------------------------------------------------------------


def _stub_kernels(monkeypatch, calls):
    """The four MLA custom ops replaced by the CPU reference; ``calls`` records which ran."""

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

    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda p: None)
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
        (
            (128, 128),
            "hk",
            "hk",
            ("hk_fwd", "flydsl_bwd"),
        ),  # seqlen 128: only the HipKittens forward takes it
        (
            (256, 384),
            "hk",
            "hk",
            ("flydsl_fwd", "hk_bwd"),
        ),  # seqlen_q < seqlen_kv: only the HipKittens backward
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
    pass routes to; a pass that falls back does not break the graph."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    _on_gfx1250(monkeypatch)
    _settings(monkeypatch, "hk", "hk")
    monkeypatch.setattr(hk, "hk_mla_unavailable_reason", lambda p: None)
    monkeypatch.setattr(mla_impl.logger, "_log_once_cache", set())
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if str(n.target).startswith("primus_turbo.")]
            return make_boxed_func(gm.forward)

        return compiler

    b, h, s = 2, 4, dims[0]
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
# Backward launch plan (meta tensors)
# ---------------------------------------------------------------------------


def test_bwd_plan():
    """DeepSeek-V3's shape: grids, the dS scratch, and argument tuples of the kernels' layouts."""
    launches, scratch, grads = hk._hk_mla_bwd_plan(*_meta_bwd(2, 4096, 128), MLA_SCALE, True, None)
    assert [(name, grid) for name, grid, _ in launches] == [
        ("k_delta", (64, 256, 1)),
        ("k_dkdv96ds", (32, 43, 8)),
        ("k_dqds_c", (32, 11, 8)),
    ]
    assert scratch.numel() * 2 == 4_328_521_728 + 2 * hk._MLA_DS_GUARD
    for name, _, values in launches:
        args = hk._MLA_KERNELS[name][1]
        assert len(values) == len(re.sub(r"(\d+)(\w)", lambda m: m[2] * int(m[1]), args)) - 1, name
    assert [g.shape for g in grads] == [(2, 4096, 128, D_QK), (2, 4096, 128, D_QK), (2, 4096, 128, D_V)]
    with pytest.raises(NotImplementedError, match="seqlen_q % 384"):
        hk._hk_mla_bwd_plan(*_meta_bwd(1, 512, 2), MLA_SCALE, True, None)


def test_bwd_plan_splits_heads_under_a_scratch_budget():
    """Over the budget, the dK / dV and dQ launches run per chunk of heads of one batch on one shared scratch: per
    batch when one batch fits, else in smaller head groups; ScratchError when not even one head group fits."""
    args = _meta_bwd(2, 4096, 128)
    guard = 2 * hk._MLA_DS_GUARD
    for heads, chunks in ((128, 2), (64, 4)):
        budget = hk._mla_ds_bytes(1, 4096, 4096, heads, True) + guard
        launches, scratch, _ = hk._hk_mla_bwd_plan(*args, MLA_SCALE, True, budget)
        assert [name for name, _, _ in launches] == ["k_delta"] + ["k_dkdv96ds", "k_dqds_c"] * chunks
        assert {name: grid for name, grid, _ in launches[1:3]} == {
            "k_dkdv96ds": (32, 43, heads // 32),
            "k_dqds_c": (32, 11, heads // 32),
        }
        assert scratch.numel() * 2 == budget
    with pytest.raises(hk._MlaScratchError):
        hk._hk_mla_bwd_plan(*args, MLA_SCALE, True, hk._mla_ds_bytes(1, 4096, 4096, 32, True) + guard - 1)


# ---------------------------------------------------------------------------
# Build: manifest, code objects, assembly patches
# ---------------------------------------------------------------------------


def test_kernel_arguments_match_the_manifest():
    """The launch layouts are the manifest's kernel signatures (packed, in order), and the flags are a hermetic
    device-only gfx1250 build: relative include paths inside the kernel directory only."""
    m = _hsaco().load_manifest()
    assert m["arch"] == "gfx1250" and {"--offload-arch=gfx1250", "--cuda-device-only"} <= set(m["flags"])
    assert set(m["kernels"]) == set(hk._MLA_KERNELS)
    kinds = {"ptr": "Q", "f32": "f", "i32": "i"}
    for name, (block, layout) in hk._MLA_KERNELS.items():
        k = m["kernels"][name]
        assert all(f == "-Iinclude" for f in m["flags"] + k["extra_flags"] if f.startswith("-I")), name
        assert (KERNEL_DIR / k["src"]).is_file() and k["block"] == block, name
        assert struct.calcsize(layout) == k["kernarg_size"], name
        expanded = "<" + "".join(kinds[a["kind"]] for a in k["args"])
        assert struct.Struct(expanded).format == "<" + re.sub(
            r"(\d+)(\w)", lambda x: x[2] * int(x[1]), layout[1:]
        )
        assert [a["offset"] for a in k["args"]] == [
            struct.calcsize(expanded[: i + 1]) for i in range(len(k["args"]))
        ]


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
        assert sorted(p.name for p in bin_dir.iterdir()) == [
            "k_dkdv96ds.hsaco",
            "k_dqds_c.hsaco",
            "k_mla_fwd.hsaco",
        ]
    assert hsaco.unavailable_reason(hsaco.FWD_KERNELS, bin_dir) is None
    assert "k_delta.hsaco is not built" == hsaco.unavailable_reason(hsaco.BWD_KERNELS, bin_dir)


_ASM_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")
_ASM_ALLOC = re.compile(r"\s*(\.amdhsa_next_free_vgpr\s+|\.vgpr_count:\s+)(\d+)$")
# the markers each generated region goes behind, every one exactly once
_ASM_REGIONS = {
    "k_mla_fwd": (
        re.compile(r";\s*HKGEN (INIT|TILE_C|TILE_D[0-3]|FINAL_A3?)\b"),
        ["INIT", "TILE_C"] + [f"TILE_D{i}" for i in range(4)] + ["FINAL_A", "FINAL_A3"],
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
    """The committed patch (tools/regen_asm_patch.py) inserts each generated region right behind its marker line
    (every region once; no s_endpgm or PC change inside, branches only to the region's own labels) and raises the
    VGPR allocation to 1024. hsaco.py applies it to its exact context only, and only to the pinned compiler .s."""
    hsaco = _hsaco()
    k = hsaco.load_manifest()["kernels"][name]
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
            mo, mn = _ASM_ALLOC.fullmatch(old[0]), _ASM_ALLOC.fullmatch(new[0])
            assert (
                len(old) == len(new) == 1
                and mo
                and mn
                and mo.group(1) == mn.group(1)
                and mn.group(2) == "1024"
            )
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


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


def _kernel_inputs(b, s, h, seed=0):
    """Kernel-layout (contiguous [b, s, h, d]) q, k, v on the card."""
    return tuple(t.cuda() for t in _inputs(b, s, s, h, layout="bshd", seed=seed)[:3])


@needs_hk
@pytest.mark.parametrize(
    "dims",
    [(1, 64, 2), (1, 256, 2), (2, 256, 2), (1, 640, 4), (1, 1024, 128), (1, 4096, 256), (2, 4096, 96)],
    ids=["s64", "toy", "b2", "s640_partial_block", "persistent", "dsv3_mbs2_folded", "kimi_k3_mbs2"],
)
def test_hk_fwd_bitwise_equals_flydsl(dims):
    """out and lse bitwise equal to the FlyDSL forward's, every element written (outputs NaN-filled first), and a
    rerun bitwise identical; the persistent grid included."""
    q, k, v = _kernel_inputs(*dims)
    with _nan_filled_outputs():
        got = hk.hk_mla_fwd(q, k, v, MLA_SCALE, True)
        again = hk.hk_mla_fwd(q, k, v, MLA_SCALE, True)
        fly = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    for name, a, b, c in zip(("out", "lse"), got, again, fly):
        assert torch.isfinite(a).all() and torch.equal(a, b) and torch.equal(a, c), name


def _counting(monkeypatch, name):
    calls = []
    real = getattr(hk, name)
    monkeypatch.setattr(hk, name, lambda *a, **kw: calls.append(a[1].shape) or real(*a, **kw))
    return calls


@needs_hk
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
    """flash_attn_func with the forward on hk against the fp32 reference, gradients included (FlyDSL backward)."""
    _settings(monkeypatch, fwd="hk")
    calls = _counting(monkeypatch, "hk_mla_fwd")
    q, k, v, dout = _inputs(*dims, layout=layout)
    got = _run(q, k, v, dout, scale)
    assert len(calls) == 1, "the HipKittens forward was not used"
    _check(got, _reference(q, k, v, dout, D_QK**-0.5 if scale is None else scale))


def _run_hk_bwd(monkeypatch, q, k, v, dout):
    _settings(monkeypatch, bwd="hk")
    with pytest.MonkeyPatch.context() as mp:
        calls = _counting(mp, "hk_mla_bwd")
        got = _run(q, k, v, dout)
    assert len(calls) == 1, "the HipKittens backward was not used"
    return got


@needs_hk
@pytest.mark.parametrize(
    ("dims", "layout"),
    [
        ((1, 256, 256, 2), "sbhd"),
        ((2, 256, 256, 2), "sbhd"),
        ((1, 256, 384, 2), "sbhd"),
        ((1, 640, 640, 4), "bshd"),
        ((2, 1024, 1024, 96), "sbhd"),
    ],
    ids=["toy", "toy_folded", "rect", "s640", "kimi_k3_heads"],
)
def test_hk_bwd_matches_reference_and_flydsl(monkeypatch, dims, layout):
    """The HipKittens backward matches the fp32 reference, is bitwise deterministic, and every gradient's SNR is
    within 0.3 dB of the FlyDSL backward's (it rounds scale * P instead of P)."""
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
def test_hk_bwd_head_chunks_are_bitwise_equal():
    """Head-chunked calls (2 batches x 2 chunks of 32 heads on one shared scratch) return exactly the unchunked
    gradients; outputs and scratch NaN-filled."""
    q, k, v, dout = (t.cuda() for t in _inputs(2, 256, 256, 64, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    budget = hk._mla_ds_bytes(1, 256, 256, 32, True) + 2 * hk._MLA_DS_GUARD
    launches, _, _ = hk._hk_mla_bwd_plan(dout, q, k, v, out, lse, MLA_SCALE, True, budget)
    assert len(launches) == 1 + 2 * 4
    with _nan_filled_outputs():
        full = hk.hk_mla_bwd(dout, q, k, v, out, lse, MLA_SCALE, True)
        chunked = hk.hk_mla_bwd(dout, q, k, v, out, lse, MLA_SCALE, True, max_scratch_bytes=budget)
    for a, b in zip(full, chunked):
        assert torch.isfinite(a).all() and torch.equal(a, b)


@needs_hk
def test_hk_bwd_falls_back_to_flydsl(monkeypatch):
    """Out of memory for the dS scratch: the call runs the FlyDSL backward; a shape FlyDSL cannot take either is
    refused."""
    q, k, v, dout = (t.cuda() for t in _inputs(1, 256, 256, 4, layout="bshd"))
    out, lse = mla_impl._forward_op(q, k, v, MLA_SCALE, True)
    fly = mla_impl._backward_op(dout, q, k, v, out, lse, MLA_SCALE, True)

    def oom(*args):
        raise torch.OutOfMemoryError("no memory for the scratch")

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
    """k_delta's delta bitwise equals the FlyDSL k_delta's, and its NL / ND the FlyDSL ones folded on the host
    (NL + fp32 log2(scale), -delta), also for denormals, underflowing products, signed zeros and overflowing NL."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd as fly

    do, o, lse = _delta_inputs(*dims, special)
    b, s, h, _ = do.shape
    q = torch.empty(b, s, h, D_QK, dtype=torch.bfloat16, device="cuda")
    with _nan_filled_outputs():
        launches, _, _ = hk._hk_mla_bwd_plan(do, q, q, o, o, lse, MLA_SCALE, True, None)
    name, grid, values = launches[0]
    stream = torch.cuda.current_stream()
    hk._mla_kernel(name)(grid, values, stream)
    delta, nl, nd = (torch.full((b, h, s), float("nan"), device="cuda") for _ in range(3))
    fly._launch(
        fly._k.launch_delta,
        (do, o, delta, lse, nl, nd, MLA_SCALE, s, h, b * s * h, s // fly._k.ROWS_DELTA, b * h, stream),
    )
    l2s = struct.unpack("f", struct.pack("f", math.log2(MLA_SCALE)))[0]
    for label, x, y in zip(
        ("delta", "NL", "ND"), (values[2], values[4], values[5]), (delta, nl + l2s, -delta)
    ):
        assert not torch.isnan(x).any(), label
        assert torch.equal(x.view(torch.int32), y.view(torch.int32)), label
    if special:
        tiny = (values[2] != 0) & (values[2].abs() < torch.finfo(torch.float32).tiny)
        assert tiny.any(), "no denormal delta: the inputs do not reach the fp32 denormal range"


@needs_hk
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_hk_end_to_end(monkeypatch, compiled):
    """Both passes on hk through flash_attn_func (SBHD, folded), eager and torch.compile'd: out / lse bitwise those of
    the FlyDSL kernels, gradients bitwise deterministic and as close to the reference."""
    q, k, v, dout = _inputs(2, 256, 256, 8, layout="sbhd")
    fwd_calls, bwd_calls = _counting(monkeypatch, "hk_mla_fwd"), _counting(monkeypatch, "hk_mla_bwd")
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
    assert (len(fwd_calls), len(bwd_calls)) == (2, 2)
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
