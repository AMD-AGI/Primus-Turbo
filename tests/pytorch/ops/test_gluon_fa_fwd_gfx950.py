###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Isolated correctness regressions for the gfx950 Gluon FA forward kernel."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import NamedTuple

import pytest

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


class ForwardCase(NamedTuple):
    test_id: str
    batch: int
    query_heads: int
    kv_heads: int
    n_ctx: int
    causal: bool
    config: KernelConfig


_DOD_CASES = (
    ("dod-mha-h64-b32-s512", 32, 512),
    ("dod-mha-h64-b16-s1024", 16, 1024),
    ("dod-mha-h64-b8-s2048", 8, 2048),
    ("dod-mha-h64-b4-s4096", 4, 4096),
    ("dod-mha-h64-b2-s8192", 2, 8192),
    ("dod-mha-h64-b1-s16384", 1, 16384),
)
_MODEL_CASES = (
    ("mha-short", 32, 32, 128),
    ("mha-medium", 32, 32, 2048),
    ("mha-long", 32, 32, 8192),
    ("gqa-short", 32, 8, 128),
    ("gqa-medium", 32, 8, 2048),
    ("gqa-long", 32, 8, 8192),
    ("mqa-short", 32, 1, 128),
    ("mqa-medium", 32, 1, 2048),
    ("mqa-long", 32, 1, 8192),
)


def _selected_config(n_ctx: int, causal: bool) -> KernelConfig:
    if not causal:
        return _BM256_BN64_STAGE3 if n_ctx <= 512 else _BM256_BN64_STAGE4
    if n_ctx <= 1024:
        return _BM128_BN64_STAGE4
    return _BM256_BN64_STAGE3 if n_ctx == 2048 else _BM256_BN64_STAGE4_MAX_ILP


FORWARD_CASES = tuple(
    ForwardCase(
        f"{name}-{'causal' if causal else 'noncausal'}",
        batch,
        64,
        64,
        n_ctx,
        causal,
        _selected_config(n_ctx, causal),
    )
    for causal in (False, True)
    for name, batch, n_ctx in _DOD_CASES
) + tuple(
    ForwardCase(
        f"{name}-{'causal' if causal else 'noncausal'}",
        1,
        query_heads,
        kv_heads,
        n_ctx,
        causal,
        _selected_config(n_ctx, causal),
    )
    for causal in (False, True)
    for name, query_heads, kv_heads, n_ctx in _MODEL_CASES
)


def _has_gfx950() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return any(
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")).split(":", 1)[0] == "gfx950"
        for index in range(torch.cuda.device_count())
    )


pytestmark = pytest.mark.skipif(
    not _has_gfx950(),
    reason="Gluon FA forward runtime coverage requires an exact gfx950 device",
)


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


def _run_matrix_worker(index: int) -> subprocess.CompletedProcess[str]:
    case = FORWARD_CASES[index]
    with tempfile.TemporaryDirectory(prefix=f"primus-gluon-fa-{case.test_id}-") as cache:
        return subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--matrix-worker",
                str(index),
            ],
            env=_worker_environment(cache),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )


@pytest.mark.parametrize("case", WORKERS)
def test_gluon_causal_regression_in_isolated_process(case: str) -> None:
    result = _run_worker(case)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout, result.stdout


@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LONG") != "1",
    reason="set PRIMUS_TURBO_TEST_LONG=1 to run the isolated 30-row GPU suite",
)
@pytest.mark.parametrize("index", range(len(FORWARD_CASES)))
def test_gluon_forward_matrix_in_isolated_process(index: int) -> None:
    result = _run_matrix_worker(index)
    assert result.returncode == 0, result.stdout
    assert f"PASS test_id={FORWARD_CASES[index].test_id}" in result.stdout


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
    prune = getattr(fa, "prune_unsafe_causal_configs", None)
    assert prune is not None, "causal stage-2 autotune pruning is missing"

    configs = fa.get_gluon_cdna_autotune_configs()
    causal = prune(configs, {}, IS_CAUSAL=True, MAX_SEQLENS_K=1024)
    noncausal = prune(configs, {}, IS_CAUSAL=False, MAX_SEQLENS_K=1024)
    if fa._HAS_WARP_PREDICATE:
        assert causal == configs
    else:
        assert causal
        assert all(config.kwargs["NUM_STAGES"] != 2 for config in causal)
    assert noncausal == configs


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


def _matrix_worker(index: int) -> int:
    case = FORWARD_CASES[index]
    snr = _run_attention_case(
        config=case.config,
        n_ctx=case.n_ctx,
        batch=case.batch,
        query_heads=case.query_heads,
        kv_heads=case.kv_heads,
        causal=case.causal,
    )
    if not (snr > 40.0):
        print(f"FAIL test_id={case.test_id} snr_db={snr:.4f}", flush=True)
        return 1
    print(f"PASS test_id={case.test_id} snr_db={snr:.4f}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--matrix-worker", type=int, choices=range(len(FORWARD_CASES)))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.worker is not None:
        raise SystemExit(_worker(args.worker))
    if args.matrix_worker is not None:
        raise SystemExit(_matrix_worker(args.matrix_worker))
    raise SystemExit("--worker or --matrix-worker is required when this file is run directly")
