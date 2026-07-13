###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr, get_tolerances


@pytest.mark.parametrize("B", [1, 2, 3, 8, 16, 32])
@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize(
    "N_K", [(2048, 1536), (2048, 1408), (2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168), (7168, 2048)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("reduce_num_cu", [0, 16, 32])
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT, BackendType.TRITON])
@pytest.mark.parametrize("auto_tune", [False, True])
def test_grouped_gemm_func(B, M, N_K, dtype, balance, trans_b, reduce_num_cu, backend, auto_tune):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if backend is not None and auto_tune:
        pytest.skip("auto_tune is ignored when backend is explicitly specified")

    if auto_tune and reduce_num_cu > 0:
        pytest.skip(
            "skip auto_tune when reduce_num_cu > 0 because hipBLASLt does not support reduce_num_cu > 0 "
            "and the tuner may select hipBLASLt"
        )

    if backend is BackendType.HIPBLASLT and reduce_num_cu > 0:
        pytest.skip("HIPBLASLT does not support reduce_num_cu > 0")

    # TODO(xiaobochen-amd): On gfx942, the hipBLASLt path can exhibit
    # intermittent/flake failures when M <= 512. This has not been reproduced on MI355.
    # We skip for now to keep CI stable while we investigate the root cause.
    # (Also skip when auto_tune=True because the tuner may select hipBLASLt.)
    if (
        M <= 512
        and (backend is BackendType.HIPBLASLT or auto_tune is True)
        and get_device_compute_capability() == (9, 4)
    ):
        pytest.skip(
            "Intermittent flake on gfx942 with hipBLASLt when M <= 512; "
            "skipping pending root-cause investigation (not reproduced on MI355)."
        )

    # Set backend and auto_tune config
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(auto_tune)

    device = "cuda"
    props = torch.cuda.get_device_properties(device)
    num_cu = props.multi_processor_count - reduce_num_cu

    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(B, M, N, K, dtype, balance, trans_b, num_cu, backend, auto_tune)

    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = torch.randn((B * M, K), dtype=torch.float32, device=device)
    b = torch.randn(b_shape, dtype=torch.float32, device=device)
    a = a.to(dtype).requires_grad_(True)
    b = b.to(dtype).requires_grad_(True)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # FWD
    out = grouped_gemm(a, b, group_lens, trans_b=trans_b, num_cu=num_cu)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens.clone(), trans_b)
    torch.testing.assert_close(out_ref, out, **get_tolerances(dtype))

    # BWD
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out.backward(grad_out)

    # Set SNR threshold based on dtype
    snr_threshold = 45 if dtype == torch.bfloat16 else 50

    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, f"out_snr too low (threshold: {snr_threshold} dB)"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, f"a_grad_snr too low (threshold: {snr_threshold} dB)"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, f"b_grad_snr too low (threshold: {snr_threshold} dB)"
    torch.testing.assert_close(a_ref.grad, a.grad, **get_tolerances(dtype))
    torch.testing.assert_close(b_ref.grad, b.grad, **get_tolerances(dtype))

    # Reset config and caches
    GlobalBackendManager.reset()


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("M", [2048, 4096])
@pytest.mark.parametrize("N_K", [(2048, 1536), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("backend", [BackendType.CK, BackendType.HIPBLASLT, BackendType.TRITON])
@pytest.mark.deterministic
def test_grouped_gemm_deterministic(B, M, N_K, dtype, balance, trans_b, backend):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Keep CI stable: reuse known gfx942 hipBLASLt flake skip for M <= 512.
    if M <= 512 and backend is BackendType.HIPBLASLT and get_device_compute_capability() == (9, 4):
        pytest.skip("Intermittent flake on gfx942 with hipBLASLt when M <= 512; skip temporarily")

    # Keep deterministic test focused: fixed backend / no autotune / no CU reduction.
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    device = "cuda"
    props = torch.cuda.get_device_properties(device)
    num_cu = props.multi_processor_count

    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(
        f"\n[deterministic] B={B}, M={M}, N={N}, K={K}, dtype={dtype}, balance={balance}, trans_b={trans_b}, backend={backend}"
    )

    b_shape = (B, N, K) if trans_b else (B, K, N)
    a0 = torch.randn((B * M, K), dtype=torch.float32, device=device).to(dtype)
    b0 = torch.randn(b_shape, dtype=torch.float32, device=device).to(dtype)
    a0 = (a0 / a0.abs().max()).detach()
    b0 = (b0 / b0.abs().max()).detach()

    # Reference (correctness)
    a_ref = a0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens.clone(), trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    def _run_once():
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = grouped_gemm(a, b, group_lens, trans_b=trans_b, num_cu=num_cu)
        out.backward(grad_out)
        return out.detach(), a.grad.detach(), b.grad.detach()

    repeats = 10
    outs = []
    for _ in range(repeats):
        outs.append(_run_once())
        torch.cuda.synchronize()

    out0, da0, db0 = outs[0]
    # Determinism (bitwise identical across runs)
    for i in range(1, repeats):
        out_i, da_i, db_i = outs[i]
        torch.testing.assert_close(out0, out_i, rtol=0, atol=0)
        torch.testing.assert_close(da0, da_i, rtol=0, atol=0)
        torch.testing.assert_close(db0, db_i, rtol=0, atol=0)

    # Correctness
    torch.testing.assert_close(out0, out_ref.detach(), **get_tolerances(dtype))
    torch.testing.assert_close(da0, a_ref.grad.detach(), **get_tolerances(dtype))
    torch.testing.assert_close(db0, b_ref.grad.detach(), **get_tolerances(dtype))

    GlobalBackendManager.reset()


def generate_grouped_gemm_group_lens_with_zeros(b, m, num_zero):
    assert num_zero < b, f"num_zero ({num_zero}) must be less than b ({b})"

    total = b * m
    num_nonzero = b - num_zero
    group_lens = torch.zeros(b, dtype=torch.int64)

    nonzero_indices = torch.randperm(b)[:num_nonzero]

    base = total // num_nonzero
    remainder = total % num_nonzero

    group_lens[nonzero_indices] = base
    group_lens[nonzero_indices[:remainder]] += 1

    return group_lens


@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N_K", [(4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("trans_b", [True])
@pytest.mark.parametrize("backend", [BackendType.CK])
@pytest.mark.parametrize("num_zero", [1, 2])
def test_grouped_gemm_with_zero_length_groups(B, M, N_K, dtype, trans_b, backend, num_zero):
    GlobalBackendManager.set_grouped_gemm_backend(backend)

    device = "cuda"
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens_with_zeros(B, M, num_zero=num_zero).to(device)
    print(B, M, N, K, dtype, trans_b, backend, num_zero)
    print(f"group_lens: {group_lens}")

    b_shape = (B, N, K) if trans_b else (B, K, N)
    a = torch.randn((B * M, K), dtype=torch.float32, device=device)
    b = torch.randn(b_shape, dtype=torch.float32, device=device)
    a = a.to(dtype).requires_grad_(True)
    b = b.to(dtype).requires_grad_(True)

    zero_mask = group_lens == 0
    zero_count = int(zero_mask.sum().item())
    assert zero_count == num_zero, (
        f"expected num_zero={num_zero}, but got {zero_count}; group_lens={group_lens}"
    )
    zero_indices = torch.nonzero(zero_mask, as_tuple=False).flatten().tolist()
    print(f"zero_indices: {zero_indices}")

    out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    assert b.grad is not None
    for idx in zero_indices:
        torch.testing.assert_close(
            b.grad[idx],
            torch.zeros_like(b.grad[idx]),
            rtol=0.0,
            atol=0.0,
            msg=f"Expected b.grad[{idx}] to be all zeros when group_len==0 (group_lens={group_lens}).",
        )

    GlobalBackendManager.reset()


# ---------------------------------------------------------------------------
# Work-stealing tests
#
# Public API: ``schedule="work_steal"`` enables the work-stealing kernel on
# the Triton and CK backends. The integration tests use the public API
# end-to-end (forward + backward via autograd). Per-ws_mode kernel
# correctness is covered by the lower-level ``grouped_gemm_triton_kernel``
# test that follows (the public API exposes only ``"static" | "work_steal"``;
# internal modes ``"global" / "per-xcd" / "hierarchical"`` are kernel-level
# tuning knobs).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("M", [1024, 4096])
@pytest.mark.parametrize("N_K", [(2048, 1536), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("backend", [BackendType.TRITON, BackendType.CK])
def test_grouped_gemm_schedule_work_steal(B, M, N_K, dtype, balance, trans_b, backend):
    """``schedule="work_steal"`` on each WS-capable backend matches the static
    path bit-for-bit for forward and backward (per-tile accumulator order is
    identical to the static schedule; there is no cross-tile reduction)."""
    # CK WS is gfx950-only (see GroupedGEMMCKBackend.can_handle): the kernel
    # body is stubbed out on gfx942 due to the 64 KB LDS budget, and the
    # backend advertises unsupported so the dispatcher raises. Skip the
    # combination on non-gfx950 hardware.
    if backend is BackendType.CK and get_device_compute_capability() != (9, 5):
        pytest.skip("CK schedule='work_steal' is gfx950-only (LDS budget)")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    GlobalBackendManager.set_grouped_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(False)

    device = "cuda"
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)

    b_shape = (B, N, K) if trans_b else (B, K, N)
    a0 = torch.randn((B * M, K), dtype=torch.float32, device=device).to(dtype)
    b0 = torch.randn(b_shape, dtype=torch.float32, device=device).to(dtype)

    # Static reference
    a_ref = a0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    out_ref = grouped_gemm(a_ref, b_ref, group_lens, trans_b=trans_b, schedule="static")
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    # Work-stealing
    a_ws = a0.clone().requires_grad_(True)
    b_ws = b0.clone().requires_grad_(True)
    out_ws = grouped_gemm(a_ws, b_ws, group_lens, trans_b=trans_b, schedule="work_steal")
    out_ws.backward(grad_out)

    torch.testing.assert_close(out_ref, out_ws, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a_ref.grad, a_ws.grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(b_ref.grad, b_ws.grad, rtol=0.0, atol=0.0)

    GlobalBackendManager.reset()


@pytest.mark.parametrize("backend", [BackendType.TRITON, BackendType.CK])
def test_grouped_gemm_schedule_work_steal_single_group(backend):
    """Single-group degenerate case (G=1): the dispatcher special-cases this
    to call non-grouped gemm; ``schedule`` must be accepted (and ignored)
    in that branch."""
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    device = "cuda"
    torch.manual_seed(0)
    M, K, N = 1024, 1280, 2560
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(1, N, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    group_lens = torch.tensor([M], dtype=torch.int64, device=device)

    out_ws = grouped_gemm(a, b, group_lens, trans_b=True, schedule="work_steal")
    out_ref = a @ b.squeeze(0).t()
    torch.testing.assert_close(out_ref, out_ws, **get_tolerances(torch.bfloat16))
    GlobalBackendManager.reset()


def test_grouped_gemm_schedule_work_steal_rejects_hipblaslt():
    """Explicit selection of HIPBLASLT together with ``schedule="work_steal"``
    must fail at dispatch -- hipblaslt has no WS kernel and advertises only
    ``"static"`` via ``can_handle``."""
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPBLASLT)
    device = "cuda"
    torch.manual_seed(0)
    B, M, K, N = 4, 1024, 1280, 2560
    a = torch.randn(B * M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, N, K, device=device, dtype=torch.bfloat16)
    group_lens = torch.full((B,), M, dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match="cannot handle"):
        grouped_gemm(a, b, group_lens, trans_b=True, schedule="work_steal")
    GlobalBackendManager.reset()


@pytest.mark.parametrize("num_cu", [64, 128, 240, 256])
def test_grouped_gemm_schedule_work_steal_rejects_num_cu(num_cu):
    """``schedule="work_steal"`` combined with an explicit ``num_cu`` must
    raise: the WS heuristic and per-XCD slot layout assume the persistent
    grid spans every XCD, so partial-grid launches are unsupported at the
    public op layer."""
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    device = "cuda"
    torch.manual_seed(0)
    B, M, K, N = 4, 1024, 1280, 2560
    a = torch.randn(B * M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, N, K, device=device, dtype=torch.bfloat16)
    group_lens = torch.full((B,), M, dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match="num_cu=None"):
        grouped_gemm(a, b, group_lens, trans_b=True, num_cu=num_cu, schedule="work_steal")
    GlobalBackendManager.reset()


# ---------------------------------------------------------------------------
# Kernel-level WS test: cover each ws_mode (the internal tuning knob) at the
# low-level ``grouped_gemm_triton_kernel`` entry point. Not reachable from
# the public API, which exposes only ``"static" | "work_steal"``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws_mode", ["auto", "global", "per-xcd", "hierarchical"])
@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm_triton_kernel_ws_modes(ws_mode, trans_b):
    """Each WS mode produces bit-identical output to the static-stride kernel
    at the low-level Triton entry point."""
    from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
        grouped_gemm_triton_kernel,
    )

    device = "cuda"
    torch.manual_seed(0)
    B, M, N, K = 4, 4096, 2048, 1536
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
    b_shape = (B, N, K) if trans_b else (B, K, N)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)

    out_ref = grouped_gemm_triton_kernel(a, b, group_offs, trans_b=trans_b, work_steal=False)
    out_ws = grouped_gemm_triton_kernel(a, b, group_offs, trans_b=trans_b, work_steal=True, ws_mode=ws_mode)
    torch.testing.assert_close(out_ref, out_ws, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_cu", [4, 6, 7, 8, 16])
@pytest.mark.parametrize("ws_mode", ["per-xcd", "hierarchical"])
def test_grouped_gemm_triton_kernel_ws_num_cu_below_num_xcds(num_cu, ws_mode):
    """Regression: ``num_cu < NUM_XCDS`` (=8) on a per-XCD-style ws_mode used
    to silently drop tiles in the gap between ``num_cu * per_xcd`` and
    ``NUM_XCDS * per_xcd`` (phase-2 started past where phase-1 actually
    ended). With the ``ACTIVE_XCDS = min(NUM_SMS, NUM_XCDS)`` fix the kernel
    matches static at any grid size. (The public API now blocks num_cu
    together with schedule="work_steal", but the low-level kernel entry
    point still accepts the combination -- belt-and-braces defense.)"""
    from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
        grouped_gemm_triton_kernel,
    )

    device = "cuda"
    torch.manual_seed(0)
    B, M, N, K = 4, 1024, 2048, 1408
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device)
    group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)

    out_ref = grouped_gemm_triton_kernel(a, b, group_offs, trans_b=True, work_steal=False)
    out_ws = grouped_gemm_triton_kernel(
        a, b, group_offs, trans_b=True, num_cu=num_cu, work_steal=True, ws_mode=ws_mode
    )
    torch.testing.assert_close(out_ref, out_ws, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    get_device_compute_capability() != (9, 5),
    reason="CK WS kernel body is stubbed on non-gfx950 (LDS budget); the "
    "cpp op would produce a no-op result. See GroupedGEMMCKBackend.can_handle.",
)
@pytest.mark.parametrize("num_cu", [4, 6, 7, 8, 16])
def test_ck_grouped_gemm_op_ws_num_cu_below_num_xcds(num_cu):
    """Regression: ``num_cu < NUM_XCDS_WS`` (=8) on the CK WS kernel used to
    silently drop tiles in the gap between ``num_cu * per_xcd`` and
    ``NUM_XCDS_WS * per_xcd`` (phase 2 starting past where phase 1 actually
    ended). With the ``active_xcds = min(gridDim.x, NUM_XCDS_WS)`` fix the
    kernel matches static at any grid size. Mirrors the equivalent Triton
    regression in ``test_grouped_gemm_triton_kernel_ws_num_cu_below_num_xcds``.
    (The public API rejects num_cu != None + schedule="work_steal", but the
    cpp op ``ck_grouped_gemm`` still accepts the combination -- belt-and-
    braces.)"""
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
        _get_ck_ws_counter,
    )
    from primus_turbo.pytorch.kernels.grouped_gemm.ws_ck_heuristic import (
        approximate_ck_standard_total_tiles,
        resolve_ck_ws_local_per_xcd,
    )

    device = "cuda"
    torch.manual_seed(0)
    B, M, N, K = 4, 1024, 2048, 1408
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=device)
    group_lens = torch.full((B,), M, dtype=torch.int64, device=device)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)

    # Static reference (no WS).
    out_ref = torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm(
        a, b, group_lens, group_offs, False, True, num_cu, False, None, 0
    )
    # Force a per-XCD-style local_per_xcd so phase 1 actually runs and the
    # gap bug (pre-fix) would surface.
    total_tiles = approximate_ck_standard_total_tiles(a.size(0), B, N)
    local_per_xcd = resolve_ck_ws_local_per_xcd("per-xcd", total_tiles, num_cu, kernel_kind="standard")
    counter = _get_ck_ws_counter(a.device)
    out_ws = torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm(
        a, b, group_lens, group_offs, False, True, num_cu, True, counter, local_per_xcd
    )
    torch.testing.assert_close(out_ref, out_ws, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Over-allocated output tail must be zeroed (mirrors the FP8/FP4 sibling tests).
# ---------------------------------------------------------------------------
def _poison_alloc_pool(shape, dtype, device, sentinel, n=24):
    """Fill and free caching-allocator blocks of ``shape`` with ``sentinel`` so a
    subsequent same-shape allocation reuses a dirty (non-zero) block instead of a
    fresh, driver-zeroed page. Lets us detect output regions the kernel never wrote."""
    blocks = [torch.full(shape, sentinel, dtype=dtype, device=device) for _ in range(n)]
    for x in blocks:
        x.add_(0.0)
    del blocks


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("backend", [BackendType.CK, BackendType.HIPBLASLT, BackendType.TRITON])
def test_grouped_gemm_padded_tail_zeroed(dtype, trans_b, backend):
    """Over-allocated output tail [sum(group_lens):M_total] must be zeroed, not left as
    caching-allocator garbage."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda:0"
    G, K, N = 8, 2048, 2880
    group_lens = torch.tensor([4096, 0, 3072, 0, 0, 5120, 0, 0], dtype=torch.int64, device=device)
    S = int(group_lens.sum())
    PAD = 224
    M_total = S + PAD  # simulate over-allocated (fixed-capacity permute) activation rows
    sentinel = 12288.0  # exactly representable in bf16 and fp16
    print(f"\ndtype={dtype}, trans_b={trans_b}, backend={backend}, S={S}, M_total={M_total}")

    GlobalBackendManager.set_grouped_gemm_backend(backend)

    b_shape = (G, N, K) if trans_b else (G, K, N)
    a = torch.randn((M_total, K), dtype=dtype, device=device)
    b = torch.randn(b_shape, dtype=dtype, device=device)

    # Warm up (JIT/autotune), then poison the pool so any unwritten padding rows
    # surface the sentinel instead of a fresh zeroed page.
    grouped_gemm(a, b, group_lens, trans_b=trans_b)
    torch.cuda.synchronize()
    _poison_alloc_pool((M_total, N), dtype, device, sentinel)

    out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
    torch.cuda.synchronize()

    pad_tail = out[S:M_total]
    assert torch.isfinite(pad_tail).all(), f"{backend.name}: padding tail non-finite"
    torch.testing.assert_close(
        pad_tail,
        torch.zeros_like(pad_tail),
        rtol=0.0,
        atol=0.0,
        msg=f"{backend.name}: padding tail [{S}:{M_total}] must be zeroed",
    )

    GlobalBackendManager.reset()
