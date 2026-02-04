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
@pytest.mark.parametrize("backend", [None, BackendType.CK, BackendType.HIPBLASLT])
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
    assert (
        zero_count == num_zero
    ), f"expected num_zero={num_zero}, but got {zero_count}; group_lens={group_lens}"
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
