###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    Float4QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.ops.grouped_gemm_fp4 import grouped_gemm_fp4
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)

# Sweep parameters. MXFP4 is NT-only (trans_b=True), single E2M1 format, Triton
# backend only, so we drop those axes and keep full B / M / NK / dtype / balance.
# N, K need only be multiples of MXFP4_BLOCK_SIZE (=32); the quantizer zero-pads
# the contraction dims up to 128. M is grouped along rows (wgrad zero-pads to 128).
B_VALUES = [1, 2, 3, 8, 16, 32]
M_VALUES = [128, 256, 512, 1024, 2048]
NK_VALUES = [
    (2048, 1536),
    (2048, 1408),
    (1408, 2048),
    (2816, 2048),
    (3072, 5120),
    (5120, 1536),
    (4096, 7168),
    (7168, 2048),
]
# 32-multiples that are NOT 128-multiples (exercises the padded-contraction +
# free-dim c_mask path): covers N-unaligned, K-unaligned, and both-unaligned.
NK_UNALIGNED_VALUES = [(96, 160), (160, 96), (288, 256), (256, 288), (1568, 2080)]
DTYPE_VALUES = [torch.bfloat16, torch.float16]
BALANCE_VALUES = [True, False]

# E2M1 (1-bit mantissa) is intrinsically lossy, so the SNR bar is lower than the
# FP8 suite's 20-25 dB. 8 dB cleanly separates "correct" from "broken layout".
SNR_THRESHOLD = 8.0


def _check_hit_int32_limit(B, M, N, K):
    """Skip shapes whose largest operand would overflow int32 element indexing."""
    a_elems = B * M * K
    b_elems = B * N * K
    out_elems = B * M * N
    return max(a_elems, b_elems, out_elems) >= 2**31


def _make_config():
    return Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )


def _run(B, M, N, K, dtype, balance):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    device = "cuda:0"
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(f"\nB={B}, M={M}, N={N}, K={K}, dtype={dtype}, balance={balance}")

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = _make_config()
    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > SNR_THRESHOLD, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"


# ----------------------------------------------------------------------------
# Main sweep: fwd + dgrad + wgrad on the full B / M / NK / dtype / balance grid.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
def test_grouped_gemm_fp4_mx_blockwise(B, M, NK, dtype, balance):
    """MXFP4 grouped GEMM fwd + dgrad + wgrad on the Triton backend."""
    N, K = NK
    _run(B, M, N, K, dtype, balance)


# ----------------------------------------------------------------------------
# Pre-quantized QuantizedTensor inputs.
# ----------------------------------------------------------------------------
def _run_grouped_gemm_fp4_quantized_tensor_test(B, M, N, K, dtype, balance):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    device = "cuda:0"
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(f"\n[QT] B={B}, M={M}, N={N}, K={K}, dtype={dtype}, balance={balance}")

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    # Match the row-wise recipes grouped_gemm_fp4 applies internally so the
    # pre-quantized operands pass check_quantized_tensor.
    a_scaling_recipe = ScalingRecipe()
    b_scaling_recipe = ScalingRecipe(use_2d_block=True)

    qt_a = QuantizedTensor.quantize(
        a,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        scaling_recipe=a_scaling_recipe,
        block_size=MXFP4_BLOCK_SIZE,
        group_lens=group_lens,
        axis=-1,
    )
    qt_b = QuantizedTensor.quantize(
        b,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        scaling_recipe=b_scaling_recipe,
        block_size=MXFP4_BLOCK_SIZE,
        axis=-1,
    )

    # Reference
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = _make_config()
    out = grouped_gemm_fp4(qt_a, qt_b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    # Check Shape
    assert out.shape == out_ref.shape
    assert qt_a.grad is not None and qt_a.grad.shape == a.shape
    assert qt_b.grad is not None and qt_b.grad.shape == b.shape

    # Check Results
    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, qt_a.grad)
    b_grad_snr = compute_snr(b_ref.grad, qt_b.grad)
    print(f"[QT] Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > SNR_THRESHOLD, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"


@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("NK", NK_VALUES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
def test_grouped_gemm_fp4_mx_blockwise_quantized_tensor(B, M, NK, dtype, balance):
    """MXFP4 grouped GEMM with pre-quantized grouped/regular QuantizedTensor inputs."""
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    N, K = NK
    _run_grouped_gemm_fp4_quantized_tensor_test(B, M, N, K, dtype, balance)


# ----------------------------------------------------------------------------
# 32-but-not-128 N/K (padded-contraction path) + unbalanced groups.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("B", [2, 4, 8])
@pytest.mark.parametrize("M", [128, 256, 512])
@pytest.mark.parametrize("NK", NK_UNALIGNED_VALUES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
def test_grouped_gemm_fp4_unaligned_nk(B, M, NK, dtype, balance):
    """N/K are 32-multiples but not 128-multiples (+ unbalanced groups).

    Validates the padded-contraction path: the quantizer zero-pads the
    contraction to 128 (data + self-consistent E8M0 scales), the GEMM runs over
    that padded length so the zero tail contributes 0, and the free dim is masked
    by the kernel. Passing SNR also confirms the padding-region scales are not
    NaN (a 0*NaN would poison the dot)."""
    N, K = NK
    _run(B, M, N, K, dtype, balance)


# ----------------------------------------------------------------------------
# Zero-length groups (MoE routing where some experts get no tokens).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
def test_grouped_gemm_fp4_zero_group_lens(dtype):
    """group_lens containing zeros must not crash fwd/bwd (illegal-memory-access
    regression guard) and must stay correct on the non-empty groups."""
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    device = "cuda:0"

    E, K, N = 8, 2048, 8192
    group_lens_list = [8192, 8192, 0, 0, 0, 0, 0, 0]
    group_lens = torch.tensor(group_lens_list, dtype=torch.int64, device=device)
    total_m = int(group_lens.sum().item())
    print(f"\ngroup_lens={group_lens_list}, total_M={total_m}, N={N}, K={K}, dtype={dtype}")

    a = torch.randn((total_m, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((E, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()

    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = _make_config()
    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    assert out.shape == out_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    # b_grad of empty experts is 0 in both ref and turbo; SNR over the full
    # tensor still reflects the populated experts.
    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"Out-SNR={out_snr:.2f} dB  AGrad-SNR={a_grad_snr:.2f} dB  BGrad-SNR={b_grad_snr:.2f} dB")
    assert out_snr > SNR_THRESHOLD, f"out_snr={out_snr:.2f} too low"
    assert a_grad_snr > SNR_THRESHOLD, f"a_grad_snr={a_grad_snr:.2f} too low"
    assert b_grad_snr > SNR_THRESHOLD, f"b_grad_snr={b_grad_snr:.2f} too low"


# CUDA-graph capturability (forward). The forward uses no D2H sync, so it is
# graph-capturable and a replay with in-place-updated group_lens must re-route.
# Only the forward is captured: fwd+bwd through autograd segfaults at capture_end
# (AccumulateGrad-on-default-stream, reproducible identically on the MXFP8 path).
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("NK", [(2048, 1536), (4096, 4096), (160, 96)])
def test_grouped_gemm_fp4_cuda_graph(NK, balance):
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    B, M = 4, 1024
    N, K = NK
    device = "cuda:0"
    torch.manual_seed(0)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    group_lens2 = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)

    a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device)
    config = _make_config()

    out_ref = grouped_gemm_ref(a, b, group_lens, trans_b=True)
    out_ref2 = grouped_gemm_ref(a, b, group_lens2, trans_b=True)

    # Warmup (JIT-compiles Triton kernels; graphs can't capture compilation).
    with torch.no_grad():
        grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    g.replay()
    torch.cuda.synchronize()
    assert out.shape == out_ref.shape
    assert compute_snr(out_ref, out) > SNR_THRESHOLD

    # Replay with a different group distribution (same total M) bound in-place.
    group_lens.copy_(group_lens2)
    g.replay()
    torch.cuda.synchronize()
    assert compute_snr(out_ref2, out) > SNR_THRESHOLD


# ----------------------------------------------------------------------------
# Determinism suite (run with --deterministic-only): bit-exact across repeats.
# ----------------------------------------------------------------------------
_DET_B_VALUES = [1, 8]
_DET_M_VALUES = [256, 1024]
# (256, 320) is the FlyDSL packed-scale-workspace regression shape (#427): small dgrad
# contraction (N=256) with a 64- but NOT 256-aligned free dim (K=320, 320 % 256 == 64)
# needs 256-row scale padding; a buggy preshuffle leaves it unwritten. The poison loop
# below surfaces the leak as a cross-repeat mismatch (2880 = gpt-oss hidden).
_DET_NK_VALUES = [(2048, 1536), (4096, 7168), (2880, 2048), (256, 320)]


# Distinct-per-repeat sentinels. Bytes near 0x7f decode to moderate finite E8M0
# scales (2**(byte-127)), so a buggy kernel that reads an unwritten workspace slot
# stays finite but sentinel-dependent, surfacing as a cross-repeat mismatch.
_DET_POISON_SENTINELS = [0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82, 0x7C, 0x83, 0x7B, 0x84]


def _run_grouped_gemm_fp4_deterministic_test(B, M, N, K, dtype, balance, backend=None, repeats=10):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)
    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    from primus_turbo.flydsl.grouped_gemm import mxfp4_grouped_kernel

    device = "cuda:0"
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    # Unbalanced runs additionally empty the LEADING expert (a real MoE occurrence):
    # the zero shifts every downstream per-expert scale offset, which is what exposes
    # the FlyDSL packed-scale 256-padding bug (#427). total_M is kept fixed.
    if not balance and B > 1:
        group_lens[-1] += group_lens[0]
        group_lens[0] = 0
    print(
        f"\n[deterministic] B={B}, M={M}, N={N}, K={K}, dtype={dtype}, "
        f"balance={balance}, backend={backend}, group0={int(group_lens[0])}"
    )

    a0 = torch.randn((B * M, K), dtype=dtype, device=device)
    b0 = torch.randn((B, N, K), dtype=dtype, device=device)
    a0 = a0 / a0.abs().max()
    b0 = b0 / b0.abs().max()

    a_ref = a0.detach().clone().requires_grad_(True)
    b_ref = b0.detach().clone().requires_grad_(True)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    config = _make_config()

    def _poison(sentinel_byte):
        # Poison every FlyDSL packed B-scale workspace with a distinct-per-repeat
        # sentinel. A correct preshuffle overwrites every slot (result invariant to
        # the sentinel); the #427 bug leaves the 256-row padding unwritten and leaks
        # the sentinel into the output. On the Triton path this is a harmless no-op.
        s = sentinel_byte * 0x01010101
        if s >= 2**31:
            s -= 2**32  # to signed int32 for fill_
        # Explicitly fetch (and create) the dgrad packed-scale workspace (free dim = K,
        # contraction = N -> K128 = ceildiv(N, 128)); that is the buffer #427 under-fills,
        # and it may be absent from the cache when its dims differ from the forward.
        total_tokens = int(group_lens.sum().item())
        try:
            _, b_scale_ws, _ = mxfp4_grouped_kernel._get_grouped_mxfp4_ws(
                total_tokens, K, (N + 127) // 128, B, torch.device(device)
            )
            b_scale_ws.fill_(s)
        except Exception:
            pass
        for e in mxfp4_grouped_kernel._GMXFP4_WS_CACHE.values():
            e[1].fill_(s)

    def _run_once(sentinel_byte):
        # Force clean memory each iter so the caching allocator can't alias a
        # buffer still being written by a pending op from a prior case.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        a = a0.detach().clone().requires_grad_(True)
        b = b0.detach().clone().requires_grad_(True)
        out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
        # Poison AFTER forward populates the workspace but BEFORE dgrad reads it.
        # Poisoning earlier is futile -- forward would rewrite and erase the sentinel.
        _poison(sentinel_byte)
        out.backward(grad_out)
        return out.detach(), a.grad.detach(), b.grad.detach()

    if backend is not None:
        GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)
    mxfp4_grouped_kernel._GMXFP4_WS_CACHE.clear()
    try:
        # Warmup: compile + populate the workspace cache so the poison loop has real
        # buffers to overwrite before the first measured repeat.
        _run_once(0x7F)
        outs = []
        for i in range(repeats):
            outs.append(_run_once(_DET_POISON_SENTINELS[i % len(_DET_POISON_SENTINELS)]))
            torch.cuda.synchronize()
    finally:
        if backend is not None:
            GlobalBackendManager.set_grouped_gemm_backend(None)
        GlobalBackendManager.set_auto_tune(None)

    out0, da0_, db0_ = outs[0]
    for i in range(1, repeats):
        out_i, da_i, db_i = outs[i]
        torch.testing.assert_close(out0, out_i, rtol=0, atol=0)
        torch.testing.assert_close(da0_, da_i, rtol=0, atol=0)
        torch.testing.assert_close(db0_, db_i, rtol=0, atol=0)

    out_snr = compute_snr(out_ref, out0)
    a_grad_snr = compute_snr(a_ref.grad, da0_)
    # A 0-token expert produces no weight-grad contribution, so its wgrad slice is
    # undefined for both backends (the kernels leave it uninitialized while the ref
    # yields exact zeros). Exclude those slices from the wgrad SNR; bit-exactness
    # across repeats above still guards the full tensor, empty slices included.
    nonempty = group_lens > 0
    b_ref_g = b_ref.grad[nonempty]
    b_out_g = db0_[nonempty]
    b_grad_snr = compute_snr(b_ref_g, b_out_g)
    print(
        f"[deterministic] Out-SNR={out_snr:.2f} dB, AGrad-SNR={a_grad_snr:.2f} dB, "
        f"BGrad-SNR={b_grad_snr:.2f} dB"
    )
    assert out_snr > SNR_THRESHOLD, "out_snr too low"
    assert a_grad_snr > SNR_THRESHOLD, "a_grad_snr too low"
    assert b_grad_snr > SNR_THRESHOLD, "b_grad_snr too low"


@pytest.mark.parametrize("B", _DET_B_VALUES)
@pytest.mark.parametrize("M", _DET_M_VALUES)
@pytest.mark.parametrize("NK", _DET_NK_VALUES)
@pytest.mark.parametrize("dtype", DTYPE_VALUES)
@pytest.mark.parametrize("balance", BALANCE_VALUES)
@pytest.mark.parametrize("backend", [None, BackendType.FLYDSL], ids=["default", "FLYDSL"])
@pytest.mark.deterministic
def test_grouped_gemm_fp4_mx_blockwise_deterministic(B, M, NK, dtype, balance, backend):
    """fwd + dgrad + wgrad are bit-exact across 10 repeats (SR off).

    ``backend=None`` runs the default (Triton) dispatch; ``FLYDSL`` pins the
    packed-scale FlyDSL kernel. Every repeat poisons the cached B-scale workspace
    with a distinct sentinel, and the unbalanced (``balance=False``) runs empty the
    leading expert. Together with the 64-but-not-256 free dims -- (2880, 2048) for
    forward, (2048, 2880) for dgrad -- this makes the suite catch #427 (FlyDSL
    packed-scale 256-padding not fully overwritten): buggy code leaks the sentinel
    and breaks bit-exactness, correct code overwrites every slot.
    """
    N, K = NK
    if backend == BackendType.FLYDSL and N % 64 != 0:
        pytest.skip("FlyDSL grouped MXFP4 backend requires N % 64 == 0")
    if backend == BackendType.FLYDSL and dtype != torch.bfloat16:
        # FlyDSL variable-K wgrad is bf16-only; forcing FLYDSL for fp16 makes the
        # wgrad dispatch raise instead of falling back to Triton.
        pytest.skip("FlyDSL grouped MXFP4 wgrad is bf16-only")
    _run_grouped_gemm_fp4_deterministic_test(B, M, N, K, dtype, balance, backend=backend, repeats=10)


# ----------------------------------------------------------------------------
# Over-allocated output tail must be zeroed (mirrors the FP8 sibling test).
# ----------------------------------------------------------------------------
def _poison_alloc_pool(shape, dtype, device, sentinel, n=24):
    """Fill and free caching-allocator blocks of ``shape`` with ``sentinel`` so a
    subsequent same-shape allocation reuses a dirty (non-zero) block instead of a
    fresh, driver-zeroed page. Lets us detect output regions the kernel never wrote."""
    blocks = [torch.full(shape, sentinel, dtype=dtype, device=device) for _ in range(n)]
    for x in blocks:
        x.add_(0.0)
    del blocks


@pytest.mark.parametrize("dtype", DTYPE_VALUES)
def test_grouped_gemm_fp4_padded_tail_zeroed(dtype):
    """Over-allocated output tail [sum(group_lens):M_total] must be zeroed, not left as
    caching-allocator garbage."""
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(42)
    device = "cuda:0"
    G, K, N = 8, 2048, 2880
    group_lens = torch.tensor([4096, 0, 3072, 0, 0, 5120, 0, 0], dtype=torch.int64, device=device)
    S = int(group_lens.sum())
    PAD = 224
    M_total = S + PAD  # simulate over-allocated (fixed-capacity permute) activation rows
    sentinel = 12288.0  # exactly representable in bf16 and fp16
    print(f"\ndtype={dtype}, S={S}, M_total={M_total}")

    config = _make_config()
    a = torch.randn((M_total, K), dtype=dtype, device=device)
    b = torch.randn((G, N, K), dtype=dtype, device=device)

    # Warm up (JIT/autotune), then poison the pool so any unwritten padding rows
    # surface the sentinel instead of a fresh zeroed page.
    grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    torch.cuda.synchronize()
    _poison_alloc_pool((M_total, N), dtype, device, sentinel)

    out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    torch.cuda.synchronize()

    pad_tail = out[S:M_total]
    assert torch.isfinite(pad_tail).all(), "padding tail non-finite"
    torch.testing.assert_close(
        pad_tail,
        torch.zeros_like(pad_tail),
        rtol=0.0,
        atol=0.0,
        msg=f"padding tail [{S}:{M_total}] must be zeroed",
    )
