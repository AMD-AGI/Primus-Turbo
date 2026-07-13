###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

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

# ----------------------------------------------------------------------------
# Sweep parameters.
#
# MXFP4 is NT-only (trans_b=True), single E2M1 format, and runs on the Triton
# backend only -- so vs the MXFP8 grouped-GEMM sweep we drop the trans_b /
# format / backend axes and keep the full B / M / NK / dtype / balance coverage.
#
# N, K need only be multiples of MXFP4_BLOCK_SIZE (=32, the MX scale block);
# they are the fwd/dgrad contraction dims and the quantizer zero-pads them up to
# 128, over which the GEMM runs. M is grouped along rows (any size; the wgrad
# wrapper zero-pads each group's M up to 128).
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# CUDA-graph capturability (forward).
#
# The forward uses no D2H sync (per-group 128-padding offsets are computed
# on-GPU with a static M+G*128 buffer bound), so it is graph-capturable, and a
# replay with in-place-updated group_lens must re-route correctly.
#
# Only the forward is captured: fwd+bwd through autograd segfaults at
# capture_end due to the AccumulateGrad-on-default-stream limitation (a generic
# autograd-in-graph issue, reproducible identically on the reference MXFP8 MX
# path, which likewise has no fwd+bwd graph test).
# ----------------------------------------------------------------------------
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
_DET_NK_VALUES = [(2048, 1536), (4096, 7168)]


def _run_grouped_gemm_fp4_deterministic_test(B, M, N, K, dtype, balance, repeats=10):
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
    print(f"\n[deterministic] B={B}, M={M}, N={N}, K={K}, dtype={dtype}, balance={balance}")

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

    def _run_once():
        # Force clean memory each iter so the caching allocator can't alias a
        # buffer still being written by a pending op from a prior case.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        a = a0.detach().clone().requires_grad_(True)
        b = b0.detach().clone().requires_grad_(True)
        out = grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
        out.backward(grad_out)
        return out.detach(), a.grad.detach(), b.grad.detach()

    outs = []
    for _ in range(repeats):
        outs.append(_run_once())
        torch.cuda.synchronize()

    out0, da0_, db0_ = outs[0]
    for i in range(1, repeats):
        out_i, da_i, db_i = outs[i]
        torch.testing.assert_close(out0, out_i, rtol=0, atol=0)
        torch.testing.assert_close(da0_, da_i, rtol=0, atol=0)
        torch.testing.assert_close(db0_, db_i, rtol=0, atol=0)

    out_snr = compute_snr(out_ref, out0)
    a_grad_snr = compute_snr(a_ref.grad, da0_)
    b_grad_snr = compute_snr(b_ref.grad, db0_)
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
@pytest.mark.deterministic
def test_grouped_gemm_fp4_mx_blockwise_deterministic(B, M, NK, dtype, balance):
    """fwd + dgrad + wgrad are bit-exact across 10 repeats (SR off)."""
    N, K = NK
    _run_grouped_gemm_fp4_deterministic_test(B, M, N, K, dtype, balance, repeats=10)
