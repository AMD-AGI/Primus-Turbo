###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from primus_turbo.pytorch.kernels.rope.dit_qk_norm_rope_impl import (
    dit_qk_norm_rope_bwd_impl,
    dit_qk_norm_rope_fwd_impl,
)
from primus_turbo.pytorch.ops import dit_qk_norm_rope, dit_qk_norm_rope_cos_sin

_CUDA = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _CUDA, reason="needs a GPU")


def _arch():
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0] if _CUDA else ""


def _flydsl_ok():
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


requires_gfx950 = pytest.mark.skipif(
    _arch() != "gfx950" or not _flydsl_ok(), reason="FlyDSL backward: gfx950"
)
_TOL = {torch.bfloat16: 6e-3, torch.float16: 1e-3, torch.float32: 2e-5}


# --------------------------------------------------------------------------------------------------- reference
def _split(x, H, D):
    """q, k, v [T0, T1, H, D] of one QKV GEMM output (4-D per-head or 3-D head-blocked)."""
    if x.dim() == 4:
        return x[..., :D], x[..., D : 2 * D], x[..., 2 * D :]
    return tuple(x[..., i * H * D : (i + 1) * H * D].unflatten(-1, (H, D)) for i in range(3))


def _rope_ref(y, c, s, rope):
    D = y.shape[-1]
    if rope == "interleaved":
        a, b, c, s = y[..., 0::2], y[..., 1::2], c[..., 0::2], s[..., 0::2]
        return torch.stack((a * c - b * s, b * c + a * s), dim=-1).flatten(-2)
    a, b, c, s = y[..., : D // 2], y[..., D // 2 :], c[..., : D // 2], s[..., : D // 2]
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1)


def _reference(srcs, ws, biases, cos, sin, eps, layout, rope, H, D):
    """fp64 q, k, v (joint, text first) from fp64 leaves; ``biases`` are added to the sources (as the GEMM does)."""
    qs, ks, vs = [], [], []
    for x, (wq, wk), b in zip(srcs, ws, biases):
        if b is not None:
            x = x + b.reshape(x.shape[2:])
        q, k, v = _split(x, H, D)
        qs.append(q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * wq)
        ks.append(k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * wk)
        vs.append(v)
    cat = 0 if layout == "sbhd" else 1
    q, k, v = (torch.cat(t, dim=cat) for t in (qs, ks, vs))
    S_out = q.shape[cat]
    B = q.shape[1 - cat]
    if cos.shape[0] == S_out:
        c, s = cos.view(S_out, 1, 1, D), sin.view(S_out, 1, 1, D)
    else:
        c, s = cos.view(S_out, B, 1, D), sin.view(S_out, B, 1, D)
    if layout == "bshd":
        c, s = c.transpose(0, 1), s.transpose(0, 1)
    return _rope_ref(q, c, s, rope), _rope_ref(k, c, s, rope), v


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _make(layout, form, dual, H=16, D=128, S=(128, 128), B=2, dtype=torch.bfloat16, per_batch=False, seed=0,
          device="cuda"):  # fmt: skip
    g = torch.Generator(device="cpu").manual_seed(seed)

    def src(Sg):
        tok = (Sg, B) if layout == "sbhd" else (B, Sg)
        shape = tok + ((H, 3 * D) if form == 4 else (3 * H * D,))
        return torch.randn(shape, generator=g).to(dtype).to(device)

    segs = list(S) if dual else [S[1]]
    srcs = [src(sg) for sg in segs]
    ws = [tuple((1 + 0.3 * torch.randn(D, generator=g)).to(dtype).to(device) for _ in range(2)) for _ in segs]
    S_out = sum(segs)
    rows = S_out * B if per_batch else S_out
    ang = torch.randn(rows, D // 2, generator=g, dtype=torch.float64) * 3
    # Each pair's angle twice, in the pairing the table is read in.
    angs = {"interleaved": ang.repeat_interleave(2, dim=-1), "half": torch.cat((ang, ang), dim=-1)}
    return srcs, ws, angs, S_out


def _run(srcs, ws, cos, sin, eps, layout, rope, round_normed, bias, v_out="copy", seed=1):
    leaves = [x.detach().clone().requires_grad_() for x in srcs]
    wl = [tuple(w.detach().clone().requires_grad_() for w in p) for p in ws]
    bl = [torch.zeros(x.shape[2:], device=x.device, dtype=x.dtype).flatten().requires_grad_() for x in srcs] \
        if bias else [None] * len(srcs)  # fmt: skip
    kw = dict(layout=layout, rope=rope, round_normed=round_normed, v_out=v_out)
    if len(srcs) == 2:
        kw.update(txt_qkv=leaves[0], w_txt_q=wl[0][0], w_txt_k=wl[0][1], txt_bias=bl[0])
    q, k, v = dit_qk_norm_rope(leaves[-1], wl[-1][0], wl[-1][1], cos, sin, eps, bias=bl[-1], **kw)
    g = torch.Generator(device="cpu").manual_seed(seed)
    grads = [torch.randn(t.shape, generator=g).to(t.dtype).to(t.device) for t in (q, k, v)]
    torch.autograd.backward((q, k, v), grads)
    out = dict(q=q, k=k, v=v, grads=grads)
    out["d_src"] = [x.grad for x in leaves]
    out["dw"] = [(a.grad, b.grad) for a, b in wl]
    out["db"] = [b.grad if b is not None else None for b in bl]
    return out


def _check_vs_reference(srcs, ws, cos, sin, eps, layout, rope, res, H, D, bias):
    dt = srcs[0].dtype
    tol = _TOL[dt]
    leaves = [x.detach().double().requires_grad_() for x in srcs]
    wl = [tuple(w.detach().double().requires_grad_() for w in p) for p in ws]
    bl = [
        torch.zeros(x.shape[2:], dtype=torch.float64, device=x.device).flatten().requires_grad_()
        for x in srcs
    ]
    q, k, v = _reference(leaves, wl, bl, cos.double(), sin.double(), eps, layout, rope, H, D)
    torch.autograd.backward((q, k, v), [g.double() for g in res["grads"]])
    for name, a, b in (("q", res["q"], q), ("k", res["k"], k), ("v", res["v"], v)):
        assert _rel(a, b) < tol, (name, _rel(a, b))
    for i in range(len(srcs)):
        assert _rel(res["d_src"][i], leaves[i].grad) < tol, (
            "d_src",
            i,
            _rel(res["d_src"][i], leaves[i].grad),
        )
        for j in range(2):
            assert _rel(res["dw"][i][j], wl[i][j].grad) < tol, (
                "dw",
                i,
                j,
                _rel(res["dw"][i][j], wl[i][j].grad),
            )
        if bias:
            assert _rel(res["db"][i], bl[i].grad) < tol, ("db", i)


def _tables(angs, rope, dtype, device="cuda"):
    a = angs[rope]
    return torch.cos(a).to(dtype).to(device), torch.sin(a).to(dtype).to(device)


# --------------------------------------------------------------------------------------------------- GPU tests
@requires_gpu
@pytest.mark.parametrize("layout,form", [("sbhd", 4), ("bshd", 3), ("sbhd", 3), ("bshd", 4)])
@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("rope", ["interleaved", "half"])
@pytest.mark.parametrize("round_normed", [True, False])
@pytest.mark.parametrize("bias", [False, True])
def test_against_fp64(layout, form, dual, rope, round_normed, bias):
    H, D = 16, 128
    srcs, ws, angs, _ = _make(layout, form, dual, H=H, D=D)
    cos, sin = _tables(angs, rope, srcs[0].dtype)
    res = _run(srcs, ws, cos, sin, 1e-6, layout, rope, round_normed, bias)
    _check_vs_reference(srcs, ws, cos, sin, 1e-6, layout, rope, res, H, D, bias)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("H,D,S,B", [(3, 64, (5, 7), 3), (24, 128, (16, 32), 2), (1, 16, (4, 4), 1)])
def test_triton_shapes_dtypes(dtype, H, D, S, B):
    """Triton everywhere: odd head counts (masked head blocks), small D, token counts off any multiple."""
    for layout, form, dual, per_batch in (
        ("sbhd", 4, True, True),
        ("bshd", 3, True, False),
        ("sbhd", 3, False, False),
    ):
        srcs, ws, angs, _ = _make(layout, form, dual, H=H, D=D, S=S, B=B, dtype=dtype, per_batch=per_batch)
        cos, sin = _tables(angs, "interleaved", dtype)
        res = _run(srcs, ws, cos, sin, 1e-5, layout, "interleaved", False, True)
        _check_vs_reference(srcs, ws, cos, sin, 1e-5, layout, "interleaved", res, H, D, True)


@requires_gpu
@pytest.mark.parametrize("layout,form", [("sbhd", 4), ("bshd", 3)])
def test_per_batch_tables_and_strided_source(layout, form):
    """cos / sin per (position, batch element); a source read at a wider token stride (a slice)."""
    H, D = 16, 128
    srcs, ws, angs, _ = _make(layout, form, True, H=H, D=D, per_batch=True)
    wide = []
    for x in srcs:
        pad = torch.zeros(
            x.shape[:2] + ((H, 4 * D) if form == 4 else (4 * H * D,)), device=x.device, dtype=x.dtype
        )
        if form == 4:
            pad[..., : 3 * D] = x
            wide.append(pad[..., : 3 * D])
        else:
            pad[..., : 3 * H * D] = x
            wide.append(pad[..., : 3 * H * D])
    cos, sin = _tables(angs, "interleaved", srcs[0].dtype)
    for xs in (srcs, wide):
        res = _run(xs, ws, cos, sin, 1e-6, layout, "interleaved", True, form == 3)
        _check_vs_reference(xs, ws, cos, sin, 1e-6, layout, "interleaved", res, H, D, form == 3)


@requires_gpu
def test_v_out_none():
    H, D = 16, 128
    srcs, ws, angs, _ = _make("sbhd", 4, False, H=H, D=D)
    cos, sin = _tables(angs, "interleaved", srcs[0].dtype)
    a = _run(srcs, ws, cos, sin, 1e-6, "sbhd", "interleaved", True, False, v_out="copy")
    b = _run(srcs, ws, cos, sin, 1e-6, "sbhd", "interleaved", True, False, v_out="none")
    assert b["v"]._base is not None and not b["v"].is_contiguous()  # a strided view of qkv, no copy
    for key in ("q", "k", "v"):
        assert torch.equal(a[key], b[key])
    assert torch.equal(a["d_src"][0], b["d_src"][0])
    assert all(torch.equal(x, y) for x, y in zip(a["dw"][0], b["dw"][0]))
    with pytest.raises(ValueError):
        srcs2, ws2, _, _ = _make("sbhd", 4, True, H=H, D=D)
        dit_qk_norm_rope(srcs2[1], *ws2[1], cos, sin, txt_qkv=srcs2[0], w_txt_q=ws2[0][0], w_txt_k=ws2[0][1],
                         v_out="none")  # fmt: skip


def _bwd(srcs, ws, cos, sin, grads, layout, rope, round_normed, bias, backend):
    dual = len(srcs) == 2
    return dit_qk_norm_rope_bwd_impl(
        *grads, srcs[-1], ws[-1][0], ws[-1][1], cos, sin, srcs[0] if dual else None, ws[0][0] if dual else None,
        ws[0][1] if dual else None, 1e-6, layout, rope, round_normed, bias, backend=backend,
    )  # fmt: skip


@requires_gpu
@pytest.mark.parametrize("backend", ["triton", pytest.param("flydsl", marks=requires_gfx950)])
@pytest.mark.parametrize("layout,form,dual", [("sbhd", 4, False), ("sbhd", 4, True), ("bshd", 3, True)])
def test_deterministic(backend, layout, form, dual):
    srcs, ws, angs, _ = _make(layout, form, dual, H=24, D=128, S=(256, 256), B=4)
    cos, sin = _tables(angs, "interleaved", srcs[0].dtype)
    q, _, _ = dit_qk_norm_rope_fwd_impl(srcs[-1], *ws[-1], cos, sin, srcs[0] if dual else None,
                                        *(ws[0] if dual else (None, None)), 1e-6, layout, "interleaved", True, True)  # fmt: skip
    grads = [torch.randn_like(q) for _ in range(3)]
    a = _bwd(srcs, ws, cos, sin, grads, layout, "interleaved", True, True, backend)
    b = _bwd(srcs, ws, cos, sin, grads, layout, "interleaved", True, True, backend)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@requires_gfx950
@pytest.mark.parametrize("layout,form,dual", [("sbhd", 4, False), ("sbhd", 4, True), ("bshd", 3, False),
                                              ("bshd", 3, True)])  # fmt: skip
@pytest.mark.parametrize("round_normed", [True, False])
@pytest.mark.parametrize("per_batch", [False, True])
def test_flydsl_matches_triton(layout, form, dual, round_normed, per_batch):
    """The two backward backends: d(qkv) and db differ only in fp32 summation order (the 128-element row sums;
    the bias partials); dw in its partial-sum order."""
    srcs, ws, angs, _ = _make(layout, form, dual, H=24, D=128, S=(256, 256), B=2, per_batch=per_batch)
    cos, sin = _tables(angs, "interleaved", srcs[0].dtype)
    q, _, _ = dit_qk_norm_rope_fwd_impl(srcs[-1], *ws[-1], cos, sin, srcs[0] if dual else None,
                                        *(ws[0] if dual else (None, None)), 1e-6, layout, "interleaved", True, True)  # fmt: skip
    grads = [torch.randn_like(q) for _ in range(3)]
    a = _bwd(srcs, ws, cos, sin, grads, layout, "interleaved", round_normed, True, "triton")
    b = _bwd(srcs, ws, cos, sin, grads, layout, "interleaved", round_normed, True, "flydsl")
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.stride() == y.stride()
        assert _rel(y, x) < 2e-3, _rel(y, x)
    # dv is copied: bit-identical in both.
    for xa, xb, x in zip(a[:-1], b[:-1], srcs):
        va, vb = _split(xa, 24, 128)[2], _split(xb, 24, 128)[2]
        assert torch.equal(va, vb)


@requires_gpu
def test_torch_compile():
    srcs, ws, angs, _ = _make("sbhd", 4, True, H=16, D=128)
    cos, sin = _tables(angs, "interleaved", srcs[0].dtype)

    def f(x_txt, x, wq0, wk0, wq1, wk1):
        q, k, v = dit_qk_norm_rope(x, wq1, wk1, cos, sin, 1e-6, txt_qkv=x_txt, w_txt_q=wq0, w_txt_k=wk0)
        return (q.float().square().sum() + k.float().sum() * 0.5 + v.float().sum()).to(x.dtype)

    def run(fn):
        leaves = [t.detach().clone().requires_grad_() for t in (srcs[0], srcs[1], *ws[0], *ws[1])]
        fn(*leaves).backward()
        return [t.grad for t in leaves]

    eager = run(f)
    compiled = run(torch.compile(f, fullgraph=True))
    for a, b in zip(eager, compiled):
        assert torch.equal(a, b)


@requires_gpu
def test_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    srcs, ws, angs, _ = _make("bshd", 3, True, H=16, D=128)
    cos, sin = _tables(angs, "interleaved", srcs[0].dtype)
    args = (srcs[1], *ws[1], cos, sin, srcs[0], *ws[0], 1e-6, "bshd", "interleaved", True, True)
    real = torch.ops.primus_turbo.dit_qk_norm_rope(*args)
    grads = [torch.randn_like(real[0]) for _ in range(3)]
    breal = torch.ops.primus_turbo.dit_qk_norm_rope_backward(*grads, *args[:-1], True)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fargs = [mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        fake = torch.ops.primus_turbo.dit_qk_norm_rope(*fargs)
        fgrads = [mode.from_tensor(g) for g in grads]
        bfake = torch.ops.primus_turbo.dit_qk_norm_rope_backward(*fgrads, *fargs[:-1], True)
    for r, f in zip(list(real) + list(breal), list(fake) + list(bfake)):
        assert r.shape == f.shape and r.dtype == f.dtype and r.stride() == f.stride()


# --------------------------------------------------------------------------------------------------- CPU tests
def test_validation():
    D, H = 128, 4
    x = torch.zeros(8, 2, H, 3 * D, dtype=torch.bfloat16)
    w = torch.ones(D, dtype=torch.bfloat16)
    cos = torch.zeros(8, D, dtype=torch.bfloat16)
    ok = dict(layout="sbhd")
    bad = [
        dict(qkv=torch.zeros(8, 2, H, 3 * D + 1, dtype=torch.bfloat16)),  # not 3D wide
        dict(qkv=torch.zeros(8, 2, H, 3 * D, dtype=torch.bfloat16).transpose(0, 1)),  # token dims not uniform
        dict(w_q=torch.ones(D, dtype=torch.float32)),  # weight dtype
        dict(w_q=torch.ones(2 * D, dtype=torch.bfloat16)[::2]),  # weight stride
        dict(cos=torch.zeros(7, D, dtype=torch.bfloat16)),  # rows
        dict(sin=torch.zeros(8, D, dtype=torch.float32)),  # cos / sin dtypes differ
        dict(cos=torch.zeros(8, 2 * D, dtype=torch.bfloat16)[:, ::2], sin=torch.zeros(8, 2 * D)[:, ::2]),
        dict(layout="sbdh"),
        dict(rope="neox"),
        dict(v_out="view"),
        dict(txt_qkv=torch.zeros(4, 2, H, 3 * D, dtype=torch.bfloat16)),  # no text weights
        dict(
            txt_qkv=torch.zeros(4, 3, H, 3 * D, dtype=torch.bfloat16), w_txt_q=w, w_txt_k=w
        ),  # batch differs
        dict(bias=torch.zeros(3 * D, dtype=torch.bfloat16)),  # bias size
        dict(
            qkv=torch.zeros(8, 2, 3 * H * 96, dtype=torch.bfloat16), w_q=torch.ones(96, dtype=torch.bfloat16)
        ),
    ]
    for case in bad:
        kw = dict(qkv=x, w_q=w, w_k=w, cos=cos, sin=cos, **ok)
        kw.update(case)
        if "w_q" in case:
            kw["w_k"] = kw["w_q"]
        with pytest.raises(ValueError):
            dit_qk_norm_rope(kw.pop("qkv"), kw.pop("w_q"), kw.pop("w_k"), kw.pop("cos"), kw.pop("sin"), **kw)


def test_cos_sin_tables():
    f = torch.randn(6, 3, 1, 8)
    c, s = dit_qk_norm_rope_cos_sin(f, torch.bfloat16)
    assert c.shape == (18, 8) and c.is_contiguous() and s.dtype == torch.bfloat16
    assert torch.equal(c[4], torch.cos(f[1, 1, 0]).bfloat16())


# --------------------------------------------------------------------------------------------------- built ISA
_ISA_SCRIPT = textwrap.dedent(
    """
    import sys, torch
    from primus_turbo.flydsl.rope.dit_qk_norm_rope_kernel import (GRID_X, dit_qk_norm_rope_bwd_launch,
                                                                    partial_row_width)
    from primus_turbo.pytorch.kernels.rope.dit_qk_norm_rope_impl import source_layout

    class _Stream:
        cuda_stream = 0

    lay, B, dual, bias = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    H, D, S = 24, 128, 512
    segs = [S // 2, S // 2] if dual else [S]
    bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    srcs = [bf(sg, B, H, 3 * D) if lay == "sbhd" else bf(B, sg, 3 * H * D) for sg in segs]
    g = bf(S, B, H, D) if lay == "sbhd" else bf(B, S, H, D)
    ms = [source_layout(x, D, lay) for x in srcs]
    w = bf(D)
    part = torch.zeros(GRID_X, partial_row_width(H, len(srcs), bool(bias)), dtype=torch.float32)
    dit_qk_norm_rope_bwd_launch(srcs, [torch.empty_like(x) for x in srcs], g, g, g, [(w, w)] * len(srcs),
                                bf(S, D), bf(S, D), part, H=H, s_out=S, eps=1e-6, bias=bool(bias),
                                layout=dict(ms[0], segs=[(m["n_tok"], m["S"]) for m in ms]), cs_row_stride=D,
                                cs_per_batch=False, round_normed=True, stream=_Stream())
    """
)


def _hot_loop(lines):
    """The innermost backward branch's body: (start, end) line indices."""
    labels = {ln[:-1]: i for i, ln in enumerate(lines) if re.match(r"^\.LBB\d+_\d+:$", ln)}
    loops = []
    for i, ln in enumerate(lines):
        m = re.match(r"^s_cbranch_\w+\s+(\.LBB\d+_\d+)", ln)
        if m and labels.get(m.group(1), 1 << 30) < i:
            loops.append((labels[m.group(1)], i))
    return min(loops, key=lambda x: x[1] - x[0]) if loops else None


@pytest.mark.skipif(not _flydsl_ok(), reason="needs FlyDSL")
@pytest.mark.parametrize("lay,B,dual,bias", [("sbhd", 32, 0, 1), ("sbhd", 32, 0, 0), ("sbhd", 32, 1, 1),
                                             ("bshd", 32, 0, 1), ("bshd", 32, 1, 1), ("bshd", 32, 1, 0)])  # fmt: skip
def test_flydsl_backward_isa(tmp_path, lay, B, dual, bias):
    """Compile only, gfx950 (no GPU needed). The backward must keep <= 128 VGPRs (four waves per SIMD), touch no
    scratch inside its token loop, and issue the token's seven loads before waiting on any of them: the
    occupancy pass can otherwise meet the register cap by serialising the loads, which is silent and slow."""
    env = dict(os.environ, COMPILE_ONLY="1", FLYDSL_COMPILE_COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx950",
               FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0",
               )  # fmt: skip
    r = subprocess.run([sys.executable, "-c", _ISA_SCRIPT, lay, str(B), str(dual), str(bias)], env=env,
                       capture_output=True, text=True, timeout=900)  # fmt: skip
    assert r.returncode == 0, r.stderr[-4000:]
    isa = sorted(tmp_path.rglob("*final_isa.s"))
    assert isa, f"no ISA dumped under {tmp_path}"
    raw = isa[-1].read_text()
    vgpr = int(re.search(r"\.vgpr_count:\s+(\d+)", raw).group(1))
    assert vgpr <= 128, vgpr
    lines = [ln.split(";")[0].strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    loop = _hot_loop(lines)
    assert loop is not None
    body = lines[loop[0] : loop[1] + 1]
    assert not [ln for ln in body if "scratch" in ln], "scratch access inside the token loop"
    first_wait = next(i for i, ln in enumerate(body) if ln.startswith("s_waitcnt") and "vmcnt" in ln)
    loads = [i for i, ln in enumerate(body) if ln.startswith("buffer_load")]
    assert len(loads) == 7 and loads[-1] < first_wait, (loads, first_wait)
