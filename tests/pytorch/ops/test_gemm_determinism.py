###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bit-exact determinism of the GEMM-family paths the older suites leave uncovered.

Every case goes through ``determinism_helper.check_determinism``: ``DET_REPEATS`` runs in one
process (run 0 cold), poisoned allocator pools and FlyDSL workspaces, a bitwise comparison
of every output and gradient against run 0, and a separate SNR report against fp32.

Shapes are picked so the risky variants are reachable -- FlyDSL split-K tails, variable-K
wgrad, the MXFP4 beta=1 2-way split, work-stealing -- but nothing asserts which kernel ran.
Run with ``--deterministic-only``.

The cases live here rather than as ``test_<op>_..._deterministic`` next to each op's tests
because they share one poisoning harness and one set of skip rules across ops and backends.
"""

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    check_mxfp4_support,
    check_mxfp8_support,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.ops import (
    gemm,
    gemm_fp4,
    gemm_fp8,
    grouped_gemm,
    grouped_gemm_fp4,
    grouped_gemm_fp8,
)
from tests.pytorch.ops.determinism_helper import (
    DET_REPEATS,
    case_seed,
    check_determinism,
    make_generator,
    randn,
    unbalanced_group_lens,
)
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref

pytestmark = pytest.mark.deterministic

BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32
DTYPES = [pytest.param(BF16, id="bf16"), pytest.param(FP16, id="fp16")]


@pytest.fixture(autouse=True)
def _det_state():
    GlobalBackendManager.reset()
    yield
    GlobalBackendManager.reset()


@pytest.fixture
def seed(request):
    return case_seed(request.node.nodeid)


def _is_gfx950():
    return get_device_compute_capability() == (9, 5)


def _require_gfx950(what):
    if not _is_gfx950():
        pytest.skip(f"{what} is gfx950-only")


def _skip_flydsl_on_gfx942(backend):
    if backend == BackendType.FLYDSL and get_device_compute_capability() == (9, 4):
        pytest.skip("FlyDSL backends are not supported on gfx942")


def _require_mxfp8():
    ok, reason = check_mxfp8_support()
    if not ok:
        pytest.skip(reason)


def _require_mxfp4():
    ok, reason = check_mxfp4_support()
    if not ok:
        pytest.skip(reason)


def _mx8_config(fmt=Format.E4M3):
    return Float8QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=fmt,
        block_size=MXFP8_BLOCK_SIZE,
        scale_dtype=ScaleDtype.E8M0,
    )


def _fp4_config(scale_rounding_mode=0):
    return Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_gradient_sr=False,
        scale_rounding_mode=scale_rounding_mode,
    )


def _fp32_leaves(*tensors):
    return [t.detach().float().requires_grad_(True) for t in tensors]


def _dense_nt_ref(a0, b0, g):
    a, b = _fp32_leaves(a0, b0)
    out = a @ b.t()
    out.backward(g.float())
    return {"out": out.detach(), "dgrad": a.grad, "wgrad": b.grad}


def _grouped_ref(a0, b0, group_lens, trans_b, g):
    a, b = _fp32_leaves(a0, b0)
    out = grouped_gemm_ref(a, b, group_lens, trans_b)
    out.backward(g.float())
    return {"out": out.detach(), "dgrad": a.grad, "wgrad": b.grad}


# ---------------------------------------------------------------------------
# 1. Dense MXFP8: FlyDSL, Turbo and hipBLASLt-MX, fwd / dgrad / wgrad.
# ---------------------------------------------------------------------------
# (1024, 2048, 1536) fits one dispatch round. (4096, 6144, 4096) is 384 256x256 tiles = 1.5
# rounds on 256 CUs. FlyDSL has a k_split=2 candidate for that tail round in fwd and wgrad, but
# only the timed autotune (PRIMUS_TURBO_AUTO_TUNE=1) picks it; by default this runs unsplit.
DENSE_MX_SHAPES = [(1024, 2048, 1536), (4096, 6144, 4096)]


def _skip_turbo_backend(backend):
    if backend == BackendType.TURBO:
        pytest.skip("TURBO backend is not deterministic")


@pytest.mark.parametrize("m, n, k", DENSE_MX_SHAPES)
@pytest.mark.parametrize("fmt", [Format.E4M3, Format.HYBRID], ids=["E4M3", "HYBRID"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "backend",
    [None, BackendType.HIPBLASLT, BackendType.TURBO, BackendType.FLYDSL],
    ids=["default", "HIPBLASLT", "TURBO", "FLYDSL"],
)
def test_det_gemm_fp8_mx(request, seed, m, n, k, fmt, dtype, backend):
    _skip_flydsl_on_gfx942(backend)
    _skip_turbo_backend(backend)
    _require_mxfp8()
    GlobalBackendManager.set_gemm_backend(backend)
    gen = make_generator(seed)
    a0 = randn((m, k), dtype, gen, normalize=True)
    b0 = randn((n, k), dtype, gen, normalize=True)
    g = randn((m, n), dtype, gen)
    config = _mx8_config(fmt)

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        c = gemm_fp8(a, b, False, True, dtype, config)
        mid()
        c.backward(g)
        return {"out": c, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(
        request.node.nodeid,
        run_once,
        _dense_nt_ref(a0, b0, g),
        snr_floor=25 if fmt == Format.E4M3 else 20,
    )


# ---------------------------------------------------------------------------
# 2. Grouped BF16/FP16 on every backend, FlyDSL included, with the variable-K wgrad.
# ---------------------------------------------------------------------------
# Ragged groups; N=320 off the tile grid; 8 experts with a deeper K; G=64 at the expert cap.
GROUPED_SHAPES = [(4, 1024, 2048, 1536), (8, 256, 320, 2880), (8, 512, 2048, 1536), (64, 256, 2048, 1536)]

# (backend, reduce_num_cu). A reduced CU budget only where the backend acts on it: CK and
# Triton size their persistent grid from num_cu, FlyDSL turns it into cap_cu; hipBLASLt
# ignores it, and the default's behaviour depends on which backend it routes to.
_GROUPED_BACKEND_CU = [
    pytest.param(None, 0, id="default-fullcu"),
    pytest.param(BackendType.HIPBLASLT, 0, id="HIPBLASLT-fullcu"),
    pytest.param(BackendType.CK, 0, id="CK-fullcu"),
    pytest.param(BackendType.CK, 32, id="CK-cu-32"),
    pytest.param(BackendType.TRITON, 0, id="TRITON-fullcu"),
    pytest.param(BackendType.TRITON, 32, id="TRITON-cu-32"),
    pytest.param(BackendType.FLYDSL, 0, id="FLYDSL-fullcu"),
    pytest.param(BackendType.FLYDSL, 32, id="FLYDSL-cu-32"),
]


@pytest.mark.parametrize("B, M, N, K", GROUPED_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans_b", [True, False], ids=["NT", "NN"])
@pytest.mark.parametrize("backend, reduce_num_cu", _GROUPED_BACKEND_CU)
def test_det_grouped_gemm(request, seed, B, M, N, K, dtype, trans_b, backend, reduce_num_cu):
    _skip_flydsl_on_gfx942(backend)
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    num_cu = torch.cuda.get_device_properties("cuda").multi_processor_count - reduce_num_cu
    gen = make_generator(seed)
    group_lens = unbalanced_group_lens(B, M, seed)
    a0 = randn((B * M, K), dtype, gen, normalize=True)
    b0 = randn((B, N, K) if trans_b else (B, K, N), dtype, gen, normalize=True)
    g = randn((B * M, N), dtype, gen)

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm(a, b, group_lens, trans_b=trans_b, num_cu=num_cu)
        mid()
        out.backward(g)
        return {"out": out, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(
        request.node.nodeid, run_once, _grouped_ref(a0, b0, group_lens, trans_b, g), snr_floor=40
    )


# ---------------------------------------------------------------------------
# 2b. Grouped MXFP8 FlyDSL NT (fwd, dgrad) on both grids, plus its variable-K wgrad.
# ---------------------------------------------------------------------------
# num_cu=None launches one workgroup per tile; any num_cu > 0 takes the persistent grid (sized
# from the device's CU count, not from num_cu), where each workgroup loops over tiles with no
# wave_m barrier skew. A K-loop tail race on that path failed about 5-8% of runs at
# 8 x 4096 x 6144 x 4096 (3120 tiles) with no other load, so the persistent cells run enough
# repeats to catch a regression most of the time.
GROUPED_MX_SHAPES = [(8, 4096, 6144, 4096)]
GROUPED_MX_PERSISTENT_REPEATS = 32


@pytest.mark.parametrize("B, M, N, K", GROUPED_MX_SHAPES)
@pytest.mark.parametrize("fmt", [Format.E4M3, Format.HYBRID], ids=["E4M3", "HYBRID"])
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
def test_det_grouped_gemm_fp8_mx(request, seed, B, M, N, K, fmt, persistent):
    _require_gfx950("FlyDSL MXFP8 grouped GEMM")
    _require_mxfp8()
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.FLYDSL)
    num_cu = torch.cuda.get_device_properties("cuda").multi_processor_count if persistent else None
    gen = make_generator(seed)
    group_lens = unbalanced_group_lens(B, M, seed)
    a0 = randn((B * M, K), BF16, gen, normalize=True)
    b0 = randn((B, N, K), BF16, gen, normalize=True)
    g = randn((B * M, N), BF16, gen)
    config = _mx8_config(fmt)

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config, num_cu=num_cu)
        mid()
        out.backward(g)
        return {"out": out, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(
        request.node.nodeid,
        run_once,
        _grouped_ref(a0, b0, group_lens, True, g),
        snr_floor=25 if fmt == Format.E4M3 else 20,
        repeats=GROUPED_MX_PERSISTENT_REPEATS if persistent else DET_REPEATS,
    )


# ---------------------------------------------------------------------------
# 3. Fused grad-accum (beta=1) wgrad into main_grad, one and two microbatches.
# ---------------------------------------------------------------------------
_DENSE_ACCUM = (2048, 4096, 1536)  # (M, N, K), NT
_GROUPED_ACCUM = (8, 512, 2048, 1536)  # (B, M, N, K), NT
# MXFP4 FlyDSL beta=1: the wgrad contracts over M. M=1024 keeps the accumulating epilogue
# (K//256 = 4 per slice is too short to split); M=4096 gives 16 K-blocks -> 2 slices of 8 over
# 8x6 = 48 output tiles, which fits one round, so it takes the 2-way split-K + fold.
_FP4_ACCUM_EPI = (1024, 2048, 1536)
_FP4_ACCUM_SPLIT = (4096, 2048, 1536)

# (op, backend, main_grad dtype, shape, SNR floor). FlyDSL's 16-bit accumulate epilogues take
# main_grad in the output dtype; the tensorwise grouped FlyDSL one writes fp32.
_ACCUM_CELLS = [
    pytest.param("gemm", BackendType.HIPBLASLT, FP32, _DENSE_ACCUM, 40, id="gemm-bf16-HIPBLASLT"),
    pytest.param("gemm", BackendType.TRITON, FP32, _DENSE_ACCUM, 40, id="gemm-bf16-TRITON"),
    pytest.param("gemm_fp8_tw", BackendType.HIPBLASLT, FP32, _DENSE_ACCUM, 25, id="gemm_fp8-tw-HIPBLASLT"),
    pytest.param("gemm_fp8_tw", BackendType.TRITON, FP32, _DENSE_ACCUM, 25, id="gemm_fp8-tw-TRITON"),
    pytest.param("gemm_fp8_tw", BackendType.FLYDSL, BF16, _DENSE_ACCUM, 25, id="gemm_fp8-tw-FLYDSL"),
    pytest.param("gemm_fp8_mx", None, FP32, _DENSE_ACCUM, 25, id="gemm_fp8-mx-default"),
    pytest.param("gemm_fp8_mx", BackendType.FLYDSL, BF16, _DENSE_ACCUM, 25, id="gemm_fp8-mx-FLYDSL"),
    pytest.param("gemm_fp4", BackendType.HIPBLASLT, FP32, _DENSE_ACCUM, 8, id="gemm_fp4-HIPBLASLT"),
    pytest.param("gemm_fp4", BackendType.FLYDSL, BF16, _FP4_ACCUM_EPI, 8, id="gemm_fp4-FLYDSL-M1024"),
    pytest.param("gemm_fp4", BackendType.FLYDSL, BF16, _FP4_ACCUM_SPLIT, 8, id="gemm_fp4-FLYDSL-M4096"),
    pytest.param("grouped", BackendType.HIPBLASLT, FP32, _GROUPED_ACCUM, 40, id="grouped-bf16-HIPBLASLT"),
    pytest.param("grouped", BackendType.TRITON, FP32, _GROUPED_ACCUM, 40, id="grouped-bf16-TRITON"),
    pytest.param(
        "grouped_fp8_tw", BackendType.HIPBLASLT, FP32, _GROUPED_ACCUM, 25, id="grouped_fp8-tw-HIPBLASLT"
    ),
    pytest.param("grouped_fp8_tw", BackendType.TRITON, FP32, _GROUPED_ACCUM, 25, id="grouped_fp8-tw-TRITON"),
    pytest.param("grouped_fp8_tw", BackendType.FLYDSL, FP32, _GROUPED_ACCUM, 25, id="grouped_fp8-tw-FLYDSL"),
    pytest.param("grouped_fp8_mx", None, FP32, _GROUPED_ACCUM, 25, id="grouped_fp8-mx-default"),
    pytest.param("grouped_fp8_mx", BackendType.FLYDSL, BF16, _GROUPED_ACCUM, 25, id="grouped_fp8-mx-FLYDSL"),
    pytest.param("grouped_fp4", BackendType.FLYDSL, BF16, _GROUPED_ACCUM, 8, id="grouped_fp4-FLYDSL"),
]


def _accum_op(op, group_lens):
    """``f(a, b, fuse)`` for one cell, all NT with bf16 output."""
    tw = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    calls = {
        "gemm": lambda a, b, fuse: gemm(a, b, False, True, BF16, fuse_bgrad_accum_pattern=fuse),
        "gemm_fp8_tw": lambda a, b, fuse: gemm_fp8(
            a, b, False, True, BF16, tw, fuse_bgrad_accum_pattern=fuse
        ),
        "gemm_fp8_mx": lambda a, b, fuse: gemm_fp8(
            a, b, False, True, BF16, _mx8_config(), fuse_bgrad_accum_pattern=fuse
        ),
        "gemm_fp4": lambda a, b, fuse: gemm_fp4(
            a, b, trans_b=True, out_dtype=BF16, config=_fp4_config(), fuse_bgrad_accum_pattern=fuse
        ),
        "grouped": lambda a, b, fuse: grouped_gemm(
            a, b, group_lens, trans_b=True, fuse_bgrad_accum_pattern=fuse
        ),
        "grouped_fp8_tw": lambda a, b, fuse: grouped_gemm_fp8(
            a, b, group_lens, trans_b=True, config=tw, fuse_bgrad_accum_pattern=fuse
        ),
        "grouped_fp8_mx": lambda a, b, fuse: grouped_gemm_fp8(
            a, b, group_lens, trans_b=True, config=_mx8_config(), fuse_bgrad_accum_pattern=fuse
        ),
        "grouped_fp4": lambda a, b, fuse: grouped_gemm_fp4(
            a, b, group_lens, trans_b=True, config=_fp4_config(), fuse_bgrad_accum_pattern=fuse
        ),
    }
    return calls[op]


@pytest.mark.parametrize("microbatches", [1, 2], ids=["mb1", "mb2"])
@pytest.mark.parametrize("op, backend, main_grad_dtype, shape, snr_floor", _ACCUM_CELLS)
def test_det_fused_grad_accum(request, seed, op, backend, main_grad_dtype, shape, snr_floor, microbatches):
    """main_grad = previous + sum of each microbatch's wgrad, written by the beta=1 epilogue.

    Each run starts from the same non-zero ``previous``; the microbatches use different
    activations and output gradients but the same weight leaf, as in pipeline training.
    """
    grouped = op.startswith("grouped")
    if "fp8_mx" in op:
        _require_mxfp8()
    if "fp4" in op:
        _require_mxfp4()
    if backend == BackendType.FLYDSL:
        _require_gfx950("FlyDSL")
    if grouped:
        GlobalBackendManager.set_grouped_gemm_backend(backend)
    else:
        GlobalBackendManager.set_gemm_backend(backend)

    gen = make_generator(seed)
    if grouped:
        B, M, N, K = shape
        group_lens = unbalanced_group_lens(B, M, seed)
        rows, b_shape = B * M, (B, N, K)
    else:
        M, N, K = shape
        group_lens = None
        rows, b_shape = M, (N, K)
    b0 = randn(b_shape, BF16, gen, normalize=True)
    a0s = [randn((rows, K), BF16, gen, normalize=True) for _ in range(microbatches)]
    gs = [randn((rows, N), BF16, gen) for _ in range(microbatches)]
    previous = randn(b_shape, main_grad_dtype, gen)
    call = _accum_op(op, group_lens)

    def run_once(mid):
        b = b0.clone().requires_grad_(True)
        b.main_grad = previous.clone()
        b.grad_added_to_main_grad = False
        res = {}
        for i in range(microbatches):
            a = a0s[i].clone().requires_grad_(True)
            out = call(a, b, "megatron")
            mid()
            out.backward(gs[i])
            res[f"out{i}"], res[f"dgrad{i}"] = out, a.grad
        assert b.grad_added_to_main_grad is True, "weight must be flagged during forward"
        assert b.grad.shape == b.shape and b.grad.dtype == b.dtype, "dummy wgrad must match the weight"
        res["main_grad"] = b.main_grad
        return res

    ref, wgrad_sum = {}, torch.zeros(b_shape, dtype=FP32, device="cuda")
    for i in range(microbatches):
        if grouped:
            r = _grouped_ref(a0s[i], b0, group_lens, True, gs[i])
        else:
            r = _dense_nt_ref(a0s[i], b0, gs[i])
        ref[f"out{i}"], ref[f"dgrad{i}"] = r["out"], r["dgrad"]
        wgrad_sum += r["wgrad"]
    ref["main_grad_delta"] = wgrad_sum
    views = {"main_grad_delta": lambda res: res["main_grad"].float() - previous.float()}

    check_determinism(request.node.nodeid, run_once, ref, snr_floor=snr_floor, snr_views=views)


# ---------------------------------------------------------------------------
# 4. work_steal: the same configuration, run repeatedly, must reproduce itself.
# ---------------------------------------------------------------------------
WS_SHAPES = [(4, 4096, 2048, 1536), (8, 1024, 4096, 7168), (8, 512, 2048, 1536)]


@pytest.mark.parametrize("B, M, N, K", WS_SHAPES)
@pytest.mark.parametrize("trans_b", [True, False], ids=["NT", "NN"])
@pytest.mark.parametrize("backend", [BackendType.TRITON, BackendType.CK], ids=["TRITON", "CK"])
def test_det_grouped_gemm_work_steal(request, seed, B, M, N, K, trans_b, backend):
    if backend == BackendType.CK:
        _require_gfx950("CK schedule='work_steal'")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    gen = make_generator(seed)
    group_lens = unbalanced_group_lens(B, M, seed)
    a0 = randn((B * M, K), BF16, gen, normalize=True)
    b0 = randn((B, N, K) if trans_b else (B, K, N), BF16, gen, normalize=True)
    g = randn((B * M, N), BF16, gen)

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm(a, b, group_lens, trans_b=trans_b, schedule="work_steal")
        mid()
        out.backward(g)
        return {"out": out, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(
        request.node.nodeid, run_once, _grouped_ref(a0, b0, group_lens, trans_b, g), snr_floor=40
    )


# ---------------------------------------------------------------------------
# 5. Grouped MXFP4 on every backend it registers (Triton and FlyDSL) and the default.
# ---------------------------------------------------------------------------
# The grouped MXFP4 suite's shapes: a single group, a deep K, 64- but not 256-aligned free
# dims (2880 / 320) that need scale padding.
GROUPED_FP4_SHAPES = [(1, 1024, 2048, 1536), (8, 256, 4096, 7168), (8, 1024, 2880, 2048), (8, 256, 256, 320)]


@pytest.mark.parametrize("B, M, N, K", GROUPED_FP4_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "backend", [None, BackendType.TRITON, BackendType.FLYDSL], ids=["default", "TRITON", "FLYDSL"]
)
def test_det_grouped_gemm_fp4(request, seed, B, M, N, K, dtype, backend):
    _skip_flydsl_on_gfx942(backend)
    _require_mxfp4()
    if backend == BackendType.FLYDSL:
        _require_gfx950("FlyDSL grouped MXFP4")
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    gen = make_generator(seed)
    group_lens = unbalanced_group_lens(B, M, seed)
    a0 = randn((B * M, K), dtype, gen, normalize=True)
    b0 = randn((B, N, K), dtype, gen, normalize=True)
    g = randn((B * M, N), dtype, gen)
    config = _fp4_config()

    def run_once(mid):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
        mid()
        out.backward(g)
        return {"out": out, "dgrad": a.grad, "wgrad": b.grad}

    check_determinism(request.node.nodeid, run_once, _grouped_ref(a0, b0, group_lens, True, g), snr_floor=8)


# ---------------------------------------------------------------------------
# 6. GLU: dense mlp_fp4, grouped_mlp_fp8 (tensorwise and MX), grouped_mlp_fp4.
# ---------------------------------------------------------------------------
_GATES = {"silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh")}


def _glu(l1, activation, clamp_limit):
    gate, up = torch.chunk(l1, 2, dim=-1)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    return _GATES[activation](gate) * up


def _grouped_mlp_ref(x0, w10, w20, probs0, group_lens, activation, clamp_limit, cotangent):
    x, w1, w2, probs = _fp32_leaves(x0, w10, w20, probs0)
    offs = [0] + torch.cumsum(group_lens, 0).tolist()
    outs = []
    for e in range(w1.shape[0]):
        lo, hi = offs[e], offs[e + 1]
        act = _glu(x[lo:hi] @ w1[e].t(), activation, clamp_limit) * probs[lo:hi, None]
        outs.append(act @ w2[e].t())
    out = torch.cat(outs)
    out.backward(cotangent.float())
    return {
        "out": out.detach(),
        "grad_x": x.grad,
        "grad_w1": w1.grad,
        "grad_w2": w2.grad,
        "grad_probs": probs.grad,
    }


def _run_grouped_mlp(nodeid, fn, x0, w10, w20, probs0, cotangent, ref, snr_floor):
    def run_once(mid):
        x, w1, w2, probs = [t.clone().requires_grad_(True) for t in (x0, w10, w20, probs0)]
        out = fn(x, w1, w2, probs)
        mid()
        out.backward(cotangent.to(out.dtype))
        return {
            "out": out,
            "grad_x": x.grad,
            "grad_w1": w1.grad,
            "grad_w2": w2.grad,
            "grad_probs": probs.grad,
        }

    check_determinism(nodeid, run_once, ref, snr_floor=snr_floor)


# (M, K, I): the dense GLU quant epilogue needs M % 256, I % 64 and K // 256 >= 4.
MLP_FP4_SHAPES = [(512, 1024, 1024), (2048, 1024, 1024)]


@pytest.mark.parametrize("m, k, i", MLP_FP4_SHAPES)
@pytest.mark.parametrize(
    "activation, clamp_limit", [("silu", None), ("gelu", 1.0)], ids=["silu", "gelu-clamp"]
)
def test_det_mlp_fp4(request, seed, m, k, i, activation, clamp_limit):
    from primus_turbo.pytorch.ops import mlp_fp4

    _require_mxfp4()
    gen = make_generator(seed)
    x0 = randn((m, k), BF16, gen)
    w10 = randn((2 * i, k), BF16, gen, scale=0.02)
    w20 = randn((k, i), BF16, gen, scale=0.02)
    g = randn((m, k), BF16, gen, scale=0.1)

    def run_once(mid):
        x, w1, w2 = [t.clone().requires_grad_(True) for t in (x0, w10, w20)]
        out = mlp_fp4(x, w1, w2, activation=activation, clamp_limit=clamp_limit)
        mid()
        out.backward(g)
        return {"out": out, "grad_x": x.grad, "grad_w1": w1.grad, "grad_w2": w2.grad}

    x, w1, w2 = _fp32_leaves(x0, w10, w20)
    out = _glu(x @ w1.t(), activation, clamp_limit) @ w2.t()
    out.backward(g.float())
    ref = {"out": out.detach(), "grad_x": x.grad, "grad_w1": w1.grad, "grad_w2": w2.grad}
    check_determinism(request.node.nodeid, run_once, ref, snr_floor=6)


# (M, K, I, G); I=320 leaves a masked column tail.
GROUPED_MLP_FP8_SHAPES = [(2048, 1024, 512, 4), (2048, 1024, 320, 4)]
_GLU_GATES = [("silu", None), ("gelu", 1.0)]


@pytest.mark.parametrize("shape", GROUPED_MLP_FP8_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("activation, clamp_limit", _GLU_GATES, ids=["silu", "gelu-clamp"])
@pytest.mark.parametrize("granularity", ["TW", "MX"])
def test_det_grouped_mlp_fp8(request, seed, shape, activation, clamp_limit, granularity):
    from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8

    if get_device_compute_capability() == (9, 4):
        pytest.skip("grouped_mlp_fp8 is not supported on gfx942")
    if granularity == "MX":
        _require_mxfp8()
        _require_gfx950("the fused MXFP8 GLU epilogues")
    config = _mx8_config() if granularity == "MX" else Float8QuantConfig()
    M, K, I, G = shape
    gen = make_generator(seed)
    group_lens = unbalanced_group_lens(G, M // G, seed)
    x0 = randn((M, K), BF16, gen, scale=0.1)
    w10 = randn((G, 2 * I, K), BF16, gen, scale=0.3)
    w20 = randn((G, K, I), BF16, gen, scale=0.02)
    probs0 = torch.rand(M, device="cuda", dtype=FP32, generator=gen) + 0.25
    cotangent = randn((M, K), FP32, gen)

    def fn(x, w1, w2, probs):
        return grouped_mlp_fp8(
            x,
            w1,
            w2,
            group_lens,
            probs=probs,
            trans_w1=True,
            trans_w2=True,
            config=config,
            activation=activation,
            clamp_limit=clamp_limit,
        )

    ref = _grouped_mlp_ref(x0, w10, w20, probs0, group_lens, activation, clamp_limit, cotangent)
    _run_grouped_mlp(
        request.node.nodeid, fn, x0, w10, w20, probs0, cotangent, ref, 20 if clamp_limit is None else 12
    )


# (M, K, I, G); K must be an odd multiple of 128 for the fused GLU quant epilogue.
GROUPED_MLP_FP4_SHAPES = [(2048, 896, 512, 4), (2048, 1408, 320, 4)]
_GROUPED_MLP_FP4_CASES = [
    pytest.param(shape, act, clamp, 0, id=f"{'x'.join(map(str, shape))}-{act}{'-clamp' if clamp else ''}")
    for shape in GROUPED_MLP_FP4_SHAPES
    for act, clamp in [("silu", None), ("gelu", 0.10)]
] + [pytest.param(GROUPED_MLP_FP4_SHAPES[0], "silu", None, 2, id="uos-mode2")]


@pytest.mark.parametrize("shape, activation, clamp_limit, scale_rounding_mode", _GROUPED_MLP_FP4_CASES)
def test_det_grouped_mlp_fp4(request, seed, shape, activation, clamp_limit, scale_rounding_mode):
    from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4

    _require_mxfp4()
    M, K, I, G = shape
    gen = make_generator(seed)
    group_lens = unbalanced_group_lens(G, M // G, seed)
    x0 = randn((M, K), BF16, gen, scale=0.1)
    w10 = randn((G, 2 * I, K), BF16, gen, scale=0.02)
    w20 = randn((G, K, I), BF16, gen, scale=0.02)
    probs0 = torch.rand(M, device="cuda", dtype=FP32, generator=gen) + 0.25
    cotangent = randn((M, K), FP32, gen)
    config = Float4QuantConfig(scale_rounding_mode=scale_rounding_mode)

    def fn(x, w1, w2, probs):
        return grouped_mlp_fp4(
            x,
            w1,
            w2,
            group_lens,
            probs=probs,
            trans_w1=True,
            trans_w2=True,
            config=config,
            activation=activation,
            clamp_limit=clamp_limit,
        )

    ref = _grouped_mlp_ref(x0, w10, w20, probs0, group_lens, activation, clamp_limit, cotangent)
    _run_grouped_mlp(request.node.nodeid, fn, x0, w10, w20, probs0, cotangent, ref, 8)
