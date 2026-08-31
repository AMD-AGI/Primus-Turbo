###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for Gluon attention tests."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.kernels.attention import attention_gluon_impl, attention_impl
from primus_turbo.pytorch.kernels.attention.attention_impl import resolve_flash_attn_backend
from primus_turbo.pytorch.ops.attention import flash_attn_interface


class _GluonTensor:
    """Small CUDA-tensor stand-in for dispatcher-only Gluon eligibility tests."""

    def __init__(self, shape, *, dtype=torch.float16, device="cuda:0", storage="bshd"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.storage = storage
        self.is_cuda = device is not None

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dim):
        assert dim == -1
        return 1 if self.storage != "noncontiguous" else 2

    def is_contiguous(self):
        return self.storage == "bshd"

    def transpose(self, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        return SimpleNamespace(is_contiguous=lambda: self.storage == "bhsd")


def _gluon_tensors(*, q_shape=(2, 3, 8, 64), kv_shape=(2, 4, 2, 64), **kwargs):
    return (
        _GluonTensor(q_shape, **kwargs),
        _GluonTensor(kv_shape, **kwargs),
        _GluonTensor(kv_shape, **kwargs),
    )


def _resolve_gluon(q, k, v, **kwargs):
    return resolve_flash_attn_backend(
        varlen=False,
        user_backend=BackendType.GLUON,
        q=q,
        k=k,
        v=v,
        **kwargs,
    )


@pytest.fixture(name="mock_gfx950_properties")
def _mock_gfx950_properties(monkeypatch):
    def get_device_properties(device):
        return SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    attention_impl._GFX950_DEVICE_CACHE.clear()
    yield
    attention_impl._GFX950_DEVICE_CACHE.clear()


@pytest.fixture(name="gluon_cuda_device")
def _gluon_cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the CUDA-only Gluon custom op")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(name="fake_gluon_raw")
def _fake_gluon_raw(monkeypatch):
    calls = []

    def raw(q, k, v, *, softmax_scale, causal, qkv_format):
        assert all(tensor.is_contiguous() for tensor in (q, k, v)), (
            "the wrapper must pass contiguous raw BSHD/BHSD tensors"
        )
        assert all(tensor.stride(-1) == 1 for tensor in (q, k, v))
        if qkv_format == "bshd":
            batch, seqlen_q, num_heads_q, _ = q.shape
        else:
            batch, num_heads_q, seqlen_q, _ = q.shape
        calls.append(
            {
                "shapes": (q.shape, k.shape, v.shape),
                "strides": (q.stride(), k.stride(), v.stride()),
                "contiguous": (q.is_contiguous(), k.is_contiguous(), v.is_contiguous()),
                "scale_type": type(softmax_scale),
                "stream": torch.cuda.current_stream(q.device),
            }
        )
        out = torch.full_like(q, softmax_scale + float(causal))
        lse = torch.full(
            (batch, num_heads_q, seqlen_q),
            softmax_scale,
            device=q.device,
            dtype=torch.float32,
        )
        return out, lse

    raw_module = SimpleNamespace(flash_attn_gluon_raw=raw)

    def import_raw(module_name):
        assert module_name == "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"
        return raw_module

    monkeypatch.setattr(attention_gluon_impl, "_import_module", import_raw)
    return calls


@pytest.fixture(name="fake_public_gluon_launcher")
def _fake_public_gluon_launcher(monkeypatch, mock_gfx950_properties):
    def launch(q, k, v, *, softmax_scale, causal, qkv_format):
        out = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
        return out, lse

    # The wrapper is the unavailable kernel boundary. Resolution and the custom
    # autograd function remain real in these tests.
    monkeypatch.setattr(
        flash_attn_interface,
        "flash_attn_gluon_forward_impl",
        launch,
        raising=False,
    )
    monkeypatch.setattr(
        GlobalBackendManager,
        "get_attn_backend",
        classmethod(lambda cls, precision: BackendType.GLUON),
    )


def _fake_public_gluon_tensors(requires_grad=(False, False, False)):
    q = torch.empty(
        (2, 3, 8, 64),
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad[0],
    )
    k = torch.empty(
        (2, 4, 2, 64),
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad[1],
    )
    v = torch.empty(
        (2, 4, 2, 64),
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad[2],
    )
    return q, k, v


def _gfx_arch_name(device):
    properties = torch.cuda.get_device_properties(device)
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]


def _gfx950_devices():
    if not torch.cuda.is_available():
        return []
    return [
        torch.device("cuda", index)
        for index in range(torch.cuda.device_count())
        if _gfx_arch_name(torch.device("cuda", index)) == "gfx950"
    ]


def _has_gfx950() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return any(
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")).split(":", 1)[0] == "gfx950"
        for index in range(torch.cuda.device_count())
    )


def _gfx950_device_for_test(*, prefer_nonzero=False):
    devices = _gfx950_devices()
    if not devices:
        pytest.skip("Gluon flash-attention runtime coverage requires an exact gfx950 device")
    if prefer_nonzero:
        return next((device for device in devices if device.index != 0), devices[0])
    current_index = torch.cuda.current_device()
    return next((device for device in devices if device.index == current_index), devices[0])


def _require_exact_gfx950(tensor):
    arch_name = _gfx_arch_name(tensor.device)
    if arch_name != "gfx950":
        pytest.skip(f"tested tensor is on {arch_name or 'unknown'}, not exact gfx950")


@contextmanager
def _pinned_gluon_backend():
    previous = GlobalBackendManager._attn_backend
    GlobalBackendManager._attn_backend = None if previous is None else dict(previous)
    GlobalBackendManager.set_attn_backend(
        BackendType.GLUON,
        PrecisionType.BF16_FP16_FP32,
    )
    try:
        yield
    finally:
        GlobalBackendManager._attn_backend = previous


def _make_gluon_runtime_inputs(
    *,
    batch,
    seqlen_q,
    seqlen_kv,
    num_heads_q,
    num_heads_kv,
    head_dim,
    qkv_format,
    dtype,
    device,
    generator,
):
    if qkv_format == "bshd":
        q = torch.randn(
            (batch, seqlen_q, num_heads_q, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        k = torch.randn(
            (batch, seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        v = torch.randn(
            (batch, seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return q, k, v

    assert qkv_format == "bhsd"
    q = torch.randn(
        (batch, num_heads_q, seqlen_q, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    ).transpose(1, 2)
    k = torch.randn(
        (batch, num_heads_kv, seqlen_kv, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    ).transpose(1, 2)
    v = torch.randn(
        (batch, num_heads_kv, seqlen_kv, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    ).transpose(1, 2)
    return q, k, v


def _gluon_attention_fp32_reference(q, k, v, softmax_scale, causal):
    """Independent FP32 attention and natural-log LSE for logical BSHD inputs."""
    q_bhsd = q.float().permute(0, 2, 1, 3)
    k_bhsd = k.float().permute(0, 2, 1, 3)
    v_bhsd = v.float().permute(0, 2, 1, 3)

    num_heads_q = q_bhsd.shape[1]
    num_heads_kv = k_bhsd.shape[1]
    assert num_heads_kv > 0 and num_heads_q % num_heads_kv == 0
    group_size = num_heads_q // num_heads_kv
    if group_size != 1:
        k_bhsd = k_bhsd.repeat_interleave(group_size, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(group_size, dim=1)

    scale = 1.0 / math.sqrt(float(q.shape[-1])) if softmax_scale is None else float(softmax_scale)
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-1, -2)) * scale
    if causal:
        seqlen_q, seqlen_kv = q.shape[1], k.shape[1]
        query_index = torch.arange(seqlen_q, device=q.device).view(seqlen_q, 1)
        key_index = torch.arange(seqlen_kv, device=q.device).view(1, seqlen_kv)
        # Flash attention aligns a rectangular causal mask at the bottom right.
        allowed = key_index <= query_index + (seqlen_kv - seqlen_q)
        scores = scores.masked_fill(~allowed, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v_bhsd).permute(0, 2, 1, 3)
    return output, lse


_GLUON_CORRECTNESS_CASES = [
    pytest.param(
        torch.float16,
        "bshd",
        False,
        37,
        53,
        8,
        8,
        64,
        None,
        id="fp16-bshd-noncausal-mha-d64-short-tail-default-scale",
    ),
    pytest.param(
        torch.bfloat16,
        "bhsd",
        True,
        128,
        128,
        16,
        4,
        128,
        0.17,
        id="bf16-bhsd-causal-gqa-d128-square-custom-scale",
    ),
    pytest.param(
        torch.float16,
        "bhsd",
        False,
        64,
        257,
        16,
        1,
        256,
        None,
        id="fp16-bhsd-noncausal-mqa-d256-rectangular-default-scale",
    ),
    pytest.param(
        torch.bfloat16,
        "bshd",
        True,
        37,
        53,
        8,
        8,
        96,
        0.23,
        id="bf16-bshd-causal-mha-d96-bottom-right-custom-scale",
    ),
    pytest.param(
        torch.bfloat16,
        "bhsd",
        True,
        9,
        13,
        2,
        1,
        255,
        None,
        id="bf16-bhsd-causal-mqa-d255-padded-d256-tail",
    ),
    pytest.param(
        torch.float16,
        "bshd",
        False,
        1024,
        2048,
        16,
        4,
        128,
        0.11,
        id="fp16-bshd-noncausal-gqa-d128-long-rectangular",
    ),
]


WORKERS = (
    "balanced-mask",
    "balanced-ragged",
    "balanced-stage3",
    "balanced-stage4",
    "balanced-short",
    "forced-stage2",
    "stage2-pressure",
    "gqa-medium",
    "mqa-long",
    "prune-configs",
)


class KernelConfig(NamedTuple):
    block_m: int
    block_n: int
    pre_load_v: bool
    num_stages: int
    waves_per_eu: int
    num_warps: int
    llvm_fn_attrs: tuple[tuple[str, str], ...] = ()


_VGPR_ONLY_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)
_PRELOAD_ATTRS = _VGPR_ONLY_ATTRS + (("amdgpu-no-dispatch-id", ""),)
_MAX_ILP_ATTRS = _PRELOAD_ATTRS + (("amdgpu-sched-strategy", "max-ilp"),)

_BM256_BN64_STAGE1 = KernelConfig(256, 64, False, 1, 2, 8)
_BM256_BN64_STAGE3 = KernelConfig(256, 64, False, 3, 2, 8, _VGPR_ONLY_ATTRS)
_BM256_BN64_STAGE4 = KernelConfig(256, 64, False, 4, 2, 8, _PRELOAD_ATTRS)
_BM256_BN64_STAGE4_MAX_ILP = KernelConfig(256, 64, False, 4, 2, 8, _MAX_ILP_ATTRS)
_BM128_BN64_STAGE4 = KernelConfig(128, 64, False, 4, 1, 4)
_BM128_BN64_STAGE2 = KernelConfig(128, 64, False, 2, 0, 4)


def _worker_environment(cache: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = cache
    return env


def _run_worker(case: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix=f"primus-gluon-fa-{case}-") as cache:
        return subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", case],
            env=_worker_environment(cache),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )


def _load_fa_module():
    from primus_turbo.gluon.attention import f16_fa_gfx950_rotated_4cluster

    return f16_fa_gfx950_rotated_4cluster


def _snr_db(actual, expected) -> float:
    import torch

    if not torch.isfinite(actual).all():
        raise AssertionError("actual output contains nonfinite values")
    if not torch.isfinite(expected).all():
        raise AssertionError("reference output contains nonfinite values")
    signal = torch.linalg.vector_norm(expected.float())
    noise = torch.linalg.vector_norm(actual.float() - expected.float())
    return 20.0 * torch.log10(signal / noise).item()


def _config_signature(config) -> KernelConfig:
    return KernelConfig(
        config.kwargs["BLOCK_M"],
        config.kwargs["BLOCK_N"],
        config.kwargs["PRE_LOAD_V"],
        config.kwargs["NUM_STAGES"],
        config.kwargs["waves_per_eu"],
        config.num_warps,
        tuple(config.kwargs.get("llvm_fn_attrs", ())),
    )


def _force_config(fa, expected: KernelConfig) -> None:
    matches = [
        config for config in fa.get_gluon_cdna_autotune_configs() if _config_signature(config) == expected
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one matching Gluon config for {expected}, found {len(matches)}")
    fa.gluon_attn_fwd.configs = matches
    fa.gluon_attn_fwd.cache.clear()


def _run_main_kernel_direct(fa, q, k, v, scale: float, causal: bool):
    """Launch the autotuned main kernel without the short-class dispatcher."""
    import torch

    metadata = fa.MetaData(sm_scale=scale)
    metadata.layout = "bshd"
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    if causal:
        metadata.need_causal()

    output = torch.empty_like(q)
    lse = torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        device=q.device,
        dtype=torch.float32,
    )
    q_strides, k_strides, v_strides, o_strides = fa.get_strides_from_layout(q, k, v, output, metadata)
    arch = fa.triton.runtime.driver.active.get_current_target().arch
    mma_type = fa.get_mma_type_for_arch(arch)

    def grid(meta):
        return (
            q.shape[2],
            fa.triton.cdiv(q.shape[1], meta["BLOCK_M"]),
            q.shape[0],
        )

    fa.gluon_attn_fwd[grid](
        q,
        k,
        v,
        scale,
        lse,
        output,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        HQ=q.shape[2],
        HK=k.shape[2],
        ACTUAL_BLOCK_DMODEL=q.shape[3],
        MAX_SEQLENS_Q=q.shape[1],
        MAX_SEQLENS_K=k.shape[1],
        IS_CAUSAL=causal,
        BLOCK_DMODEL=128,
        MMA_TYPE=mma_type,
        STATIC_STRIDE_KN=-1,
        STATIC_STRIDE_QM=q_strides[2],
    )
    return output


def _run_attention_case(
    config: KernelConfig,
    n_ctx: int,
    *,
    batch: int = 1,
    query_heads: int = 1,
    kv_heads: int = 1,
    causal: bool = True,
    direct_main_kernel: bool = False,
) -> float:
    import torch

    fa = _load_fa_module()
    torch.manual_seed(0)
    _force_config(fa, config)
    q = torch.randn((batch, n_ctx, query_heads, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, n_ctx, kv_heads, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(128.0)
    if direct_main_kernel:
        output = _run_main_kernel_direct(fa, q, k, v, scale, causal)
    else:
        output, _ = fa.flash_attn_gluon_raw(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            qkv_format="bshd",
        )
    torch.cuda.synchronize()

    q_ref = q.transpose(1, 2).contiguous()
    k_ref = k.transpose(1, 2).contiguous()
    v_ref = v.transpose(1, 2).contiguous()
    if query_heads != kv_heads:
        repeats = query_heads // kv_heads
        k_ref = k_ref.repeat_interleave(repeats, dim=1)
        v_ref = v_ref.repeat_interleave(repeats, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        is_causal=causal,
        scale=scale,
    ).transpose(1, 2)
    return _snr_db(output, expected)


def _run_prune_case() -> None:
    fa = _load_fa_module()
    prune = getattr(fa, "prune_unsafe_configs", None)
    assert prune is not None, "unsafe autotune pruning is missing"

    configs = fa.get_gluon_cdna_autotune_configs()
    causal = prune(
        configs,
        {},
        IS_CAUSAL=True,
        MAX_SEQLENS_K=1024,
        ACTUAL_BLOCK_DMODEL=128,
    )
    noncausal = prune(
        configs,
        {},
        IS_CAUSAL=False,
        MAX_SEQLENS_K=1024,
        ACTUAL_BLOCK_DMODEL=128,
    )
    if fa._HAS_WARP_PREDICATE:
        assert causal == configs
    else:
        assert causal
        assert all(config.kwargs["NUM_STAGES"] != 2 for config in causal)
    assert noncausal == configs

    for actual_head_dim, padded_head_dim in ((63, 64), (127, 128), (255, 256)):
        for is_causal in (False, True):
            pruned = prune(
                configs,
                {},
                IS_CAUSAL=is_causal,
                MAX_SEQLENS_K=13,
                ACTUAL_BLOCK_DMODEL=actual_head_dim,
            )
            assert pruned
            assert all(
                not (
                    config.kwargs["NUM_STAGES"] > 1
                    and not (padded_head_dim >= 256 and config.kwargs["BLOCK_N"] >= 64)
                    and not (
                        padded_head_dim < 128 and config.kwargs["BLOCK_N"] < 64 and config.num_warps >= 8
                    )
                )
                for config in pruned
            )
            assert any(config.kwargs["NUM_STAGES"] == 1 for config in pruned)

        noncausal = prune(
            configs,
            {},
            IS_CAUSAL=False,
            MAX_SEQLENS_K=13,
            ACTUAL_BLOCK_DMODEL=actual_head_dim,
        )
        if actual_head_dim == 63:
            assert any(
                config.kwargs["BLOCK_N"] == 32 and config.kwargs["NUM_STAGES"] > 1 and config.num_warps == 8
                for config in noncausal
            )
        elif actual_head_dim == 127:
            assert all(config.kwargs["NUM_STAGES"] == 1 for config in noncausal)
        else:
            assert any(
                config.kwargs["BLOCK_N"] == 64 and config.kwargs["NUM_STAGES"] > 1 for config in noncausal
            )

    for aligned_head_dim in (64, 128, 240, 256):
        aligned = prune(
            configs,
            {},
            IS_CAUSAL=False,
            MAX_SEQLENS_K=13,
            ACTUAL_BLOCK_DMODEL=aligned_head_dim,
        )
        assert aligned == configs


def _worker(case: str) -> int:
    if case == "balanced-mask":
        snr = _run_attention_case(config=_BM256_BN64_STAGE1, n_ctx=2048)
    elif case == "balanced-ragged":
        # The outer BM256 row permutation is disabled when S is not aligned.
        # A capability-enabled predicated tail must consume that exact decision
        # instead of independently enabling balancing for equal Q/K lengths.
        snr = _run_attention_case(config=_BM256_BN64_STAGE4, n_ctx=640)
    elif case == "balanced-stage3":
        snr = _run_attention_case(config=_BM256_BN64_STAGE3, n_ctx=2048)
    elif case == "balanced-stage4":
        snr = _run_attention_case(config=_BM256_BN64_STAGE4, n_ctx=2048)
    elif case == "balanced-short":
        # The public dispatcher intentionally selects the specialized BM128
        # launcher at this length. Launch the main BM256 kernel directly so
        # this regression exercises the short-tail balance propagation fixed
        # by the pinned upstream change.
        snr = _run_attention_case(
            config=_BM256_BN64_STAGE4,
            n_ctx=256,
            direct_main_kernel=True,
        )
    elif case == "forced-stage2":
        snr = _run_attention_case(config=_BM128_BN64_STAGE2, n_ctx=1024)
    elif case == "stage2-pressure":
        snr = _run_attention_case(
            config=_BM128_BN64_STAGE2,
            n_ctx=1024,
            batch=16,
            query_heads=64,
            kv_heads=64,
        )
    elif case == "gqa-medium":
        snr = _run_attention_case(
            config=_BM256_BN64_STAGE3,
            n_ctx=2048,
            query_heads=32,
            kv_heads=8,
        )
    elif case == "mqa-long":
        snr = _run_attention_case(
            config=_BM256_BN64_STAGE4_MAX_ILP,
            n_ctx=8192,
            query_heads=32,
            kv_heads=1,
        )
    elif case == "prune-configs":
        _run_prune_case()
        print("PASS case=prune-configs", flush=True)
        return 0
    else:
        raise ValueError(f"unknown worker case: {case}")

    if not (snr > 40.0):
        print(f"FAIL case={case} snr_db={snr:.4f}", flush=True)
        return 1
    print(f"PASS case={case} snr_db={snr:.4f}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.worker is not None:
        raise SystemExit(_worker(args.worker))
    raise SystemExit("--worker is required when this file is run directly")
