###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Benchmark for the MoE permute / unpermute HIP kernels.

Compares five preprocessing implementations side-by-side:

  * v1 — cooperative kernel with ``grid.sync()``
    (``csrc/kernels/permute/permute_v1.cu``)
  * v2 — same algorithm split into 4 ordinary kernels
    (``csrc/kernels/permute/permute_v2.cu``)
  * v3 — single ordinary kernel using a "decoupled lookback"-style scan +
    atomic barrier (no Pass 2, no ``grid.sync()``, no cooperative launch).
    Lookback step still sums all preceding blocks unconditionally.
    (``csrc/kernels/permute/permute_v3.cu``)
  * v4 — v1 re-optimised for AMD LDS bank layout + E template specialisation
    (same cooperative 4-pass structure as v1)
    (``csrc/kernels/permute/permute_v4.cu``)
  * v5 — single ordinary kernel using the textbook *decoupled lookback* scan
    with FLAG_PREFIX short-circuit (Merrill & Garland 2016; AMD GPUOpen
    "Boosting GPU Radix Sort"; rocPRIM ``device_scan``). Each (block, expert)
    publishes a 64-bit packed ``{flag, value}`` slot; successors short-circuit
    the moment they see a FLAG_PREFIX entry.
    (``csrc/kernels/permute/permute_v5.cu``)

测试 case 由 ``benchmark/ops/config.py::MoEModelConfigs`` 驱动，对每个 MoE 模型
在 ``BATCH_SIZE_LIST × GROUPED_GEMM_EP_SIZE_LIST`` 上展开。

Permute 的输入 (``recv_topk_idx`` / ``recv_num_tokens``) 模拟自 DeepEP intranode
dispatch 的输出（参考 ``benchmark/ops/deep_ep/test_intranode.py``）：

  * 每个源 rank 上有 ``N`` 个 token，每个 token 在全局 ``E`` 个 expert 中
    *不放回均匀* 选 ``K`` 个 (``topk``)。
  * DeepEP 会对 token 去重：同一 token 即使路由到本 rank 的多个 local expert，
    也只被发送一次。所以一个 token 进入本 rank 当且仅当其至少有一个 expert
    落在本 rank 的 local expert 范围 ``[0, E/R)`` 内。
  * 一个 token 不被本 rank 收到的概率是 ``C(E - E/R, K) / C(E, K)``，因此

    .. math::
        E[\\text{recv\\_num\\_tokens}] = R \\cdot N \\cdot
            \\left(1 - \\prod_{i=0}^{K-1}\\frac{(E-E/R)-i}{E-i}\\right)

    示例：``E=256, R=8, K=8, N=4096`` -> ``≈ 21622`` (≈ 0.6596 × R × N)。

Usage::

    python benchmark/ops/bench_permute.py
    python benchmark/ops/bench_permute.py --versions v1
    python benchmark/ops/bench_permute.py --versions v1,v3
    python benchmark/ops/bench_permute.py --versions v2,v3 -o out.csv

Force-JIT v1 (matches v2/v3 build settings, useful when the installed .so is
stale)::

    PRIMUS_TURBO_PERMUTE_BENCH_FORCE_JIT=1 python benchmark/ops/bench_permute.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    BATCH_SIZE_LIST,
    GROUPED_GEMM_EP_SIZE_LIST,
    MoEModelConfigs,
    get_platform_info,
)
from tabulate import tabulate

# -----------------------------------------------------------------------------
# JIT extension that exposes both v1 and v2 preprocessing kernels.
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
JIT_BUILD_DIR = REPO_ROOT / "agent" / "workspace" / "permute_port" / "_jit_build_v1_v2_v3_v4_v5"


_JIT_BIND_CPP = r"""
// Auto-generated standalone JIT binding for the permute_v1 + permute_v2 +
// permute_v3 + permute_v4 + permute_v5 kernels. Built by
// torch.utils.cpp_extension.load() on first benchmark run.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "primus_turbo/permute.h"

// permute_v2 / v3 / v4 live in primus_turbo::v2 / ::v3 / ::v4 respectively
// (defined in permute_v{2,3,4}.cu) and aren't exported through the public
// header yet. Forward-declare them so the binding can dispatch via the same
// allocator.
namespace primus_turbo { namespace v2 {
void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream);
}}

namespace primus_turbo { namespace v3 {
void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream);
}}

namespace primus_turbo { namespace v4 {
void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream);
}}

namespace primus_turbo { namespace v5 {
void permute_preprocessing_launch(bool *routing_map, int *num_dispatched_tokens_ptr,
                                  int num_of_local_experts, int *workspace_1, int rows_workspace_1,
                                  int *workspace_2, int rows_workspace_2, int pad_multiple,
                                  int32_t *tokens_per_expert, int *row_id_map, int *overflow_flag,
                                  int64_t num_permuted_tokens, hipStream_t stream);
}}

namespace py = pybind11;
using primus_turbo::dtype::bfloat16;

using PreprocLaunchFn = void (*)(bool *, int *, int, int *, int, int *, int, int,
                                 int32_t *, int *, int *, int64_t, hipStream_t);

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
jit_preprocess_dispatch(at::Tensor routing_map, at::Tensor num_dispatched_token_tensor,
                        int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                        int64_t pad_multiple, int64_t num_permuted_tokens,
                        PreprocLaunchFn launch_fn) {
    TORCH_CHECK(routing_map.is_cuda(), "routing_map must be CUDA");
    TORCH_CHECK(routing_map.scalar_type() == at::kBool, "routing_map must be bool");
    TORCH_CHECK(num_dispatched_token_tensor.scalar_type() == at::kInt,
                "num_dispatched_token_tensor must be int32");
    constexpr int block_size = primus_turbo::PermutePreprocessConfig::kBlockSize;
    auto device   = routing_map.device();
    auto int_opts = at::TensorOptions().dtype(at::kInt).device(device);
    auto row_id_map = at::empty(
        {max_num_dispatched_tokens + pad_multiple, num_of_local_experts}, int_opts);
    auto tokens_per_expert = at::empty({num_of_local_experts}, int_opts);
    auto overflow_flag     = at::empty({1}, int_opts);
    int rows_w1 = static_cast<int>((max_num_dispatched_tokens + block_size - 1) / block_size);
    int rows_w2 = (rows_w1 + block_size - 1) / block_size;
    auto w1     = at::empty({rows_w1, num_of_local_experts}, int_opts);
    auto w2     = at::empty({rows_w2, num_of_local_experts}, int_opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_fn(
        reinterpret_cast<bool *>(routing_map.data_ptr()),
        num_dispatched_token_tensor.data_ptr<int>(), static_cast<int>(num_of_local_experts),
        w1.data_ptr<int>(), rows_w1, w2.data_ptr<int>(), rows_w2,
        static_cast<int>(pad_multiple), tokens_per_expert.data_ptr<int>(),
        row_id_map.data_ptr<int>(), overflow_flag.data_ptr<int>(), num_permuted_tokens, stream);
    return {row_id_map, tokens_per_expert, overflow_flag};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
jit_permute_preprocessing(at::Tensor routing_map, at::Tensor num_dispatched_token_tensor,
                          int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                          int64_t pad_multiple, int64_t num_permuted_tokens) {
    return jit_preprocess_dispatch(routing_map, num_dispatched_token_tensor,
                                   max_num_dispatched_tokens, num_of_local_experts,
                                   pad_multiple, num_permuted_tokens,
                                   &primus_turbo::permute_preprocessing_launch);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
jit_permute_preprocessing_v2(at::Tensor routing_map, at::Tensor num_dispatched_token_tensor,
                             int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                             int64_t pad_multiple, int64_t num_permuted_tokens) {
    return jit_preprocess_dispatch(routing_map, num_dispatched_token_tensor,
                                   max_num_dispatched_tokens, num_of_local_experts,
                                   pad_multiple, num_permuted_tokens,
                                   &primus_turbo::v2::permute_preprocessing_launch);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
jit_permute_preprocessing_v3(at::Tensor routing_map, at::Tensor num_dispatched_token_tensor,
                             int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                             int64_t pad_multiple, int64_t num_permuted_tokens) {
    return jit_preprocess_dispatch(routing_map, num_dispatched_token_tensor,
                                   max_num_dispatched_tokens, num_of_local_experts,
                                   pad_multiple, num_permuted_tokens,
                                   &primus_turbo::v3::permute_preprocessing_launch);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
jit_permute_preprocessing_v4(at::Tensor routing_map, at::Tensor num_dispatched_token_tensor,
                             int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                             int64_t pad_multiple, int64_t num_permuted_tokens) {
    return jit_preprocess_dispatch(routing_map, num_dispatched_token_tensor,
                                   max_num_dispatched_tokens, num_of_local_experts,
                                   pad_multiple, num_permuted_tokens,
                                   &primus_turbo::v4::permute_preprocessing_launch);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
jit_permute_preprocessing_v5(at::Tensor routing_map, at::Tensor num_dispatched_token_tensor,
                             int64_t max_num_dispatched_tokens, int64_t num_of_local_experts,
                             int64_t pad_multiple, int64_t num_permuted_tokens) {
    return jit_preprocess_dispatch(routing_map, num_dispatched_token_tensor,
                                   max_num_dispatched_tokens, num_of_local_experts,
                                   pad_multiple, num_permuted_tokens,
                                   &primus_turbo::v5::permute_preprocessing_launch);
}

static void jit_permute_launcher(at::Tensor tokens, at::Tensor output_tokens,
                                 at::Tensor row_id_map, at::Tensor num_dispatched_token_tensor,
                                 int64_t pad_multiple, int64_t num_of_local_experts,
                                 int64_t hidden_size, int64_t num_permuted_token,
                                 int64_t num_of_blocks, bool use_fp8) {
    if (num_permuted_token == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    int  grid   = num_of_blocks > 0 ? static_cast<int>(num_of_blocks) : 304;
    if (use_fp8) {
        TORCH_CHECK(hidden_size % 16 == 0, "fp8 requires hidden_size % 16 == 0");
        primus_turbo::permute_impl<uint8_t, float, float>(
            reinterpret_cast<const uint8_t *>(tokens.data_ptr()),
            reinterpret_cast<uint8_t *>(output_tokens.data_ptr()), nullptr, nullptr, nullptr,
            nullptr, row_id_map.data_ptr<int>(), num_dispatched_token_tensor.data_ptr<int>(),
            static_cast<int>(pad_multiple), static_cast<int>(num_of_local_experts),
            static_cast<int>(hidden_size), 0, 0, 1, grid, stream);
    } else {
        TORCH_CHECK(hidden_size % 8 == 0, "16-bit requires hidden_size % 8 == 0");
        primus_turbo::permute_impl<uint16_t, float, float>(
            reinterpret_cast<const uint16_t *>(tokens.data_ptr()),
            reinterpret_cast<uint16_t *>(output_tokens.data_ptr()), nullptr, nullptr, nullptr,
            nullptr, row_id_map.data_ptr<int>(), num_dispatched_token_tensor.data_ptr<int>(),
            static_cast<int>(pad_multiple), static_cast<int>(num_of_local_experts),
            static_cast<int>(hidden_size), 0, 0, 1, grid, stream);
    }
}

static void jit_unpermute_launcher(at::Tensor permuted_tokens, at::Tensor output_tokens,
                                   at::Tensor row_id_map,
                                   at::Tensor num_dispatched_tokens_tensor,
                                   int64_t num_of_local_experts, int64_t hidden_size,
                                   int64_t num_of_blocks) {
    TORCH_CHECK(permuted_tokens.scalar_type() == at::kBFloat16,
                "unpermute: permuted_tokens must be bfloat16");
    TORCH_CHECK(output_tokens.scalar_type() == at::kBFloat16,
                "unpermute: output_tokens must be bfloat16");
    auto stream = at::cuda::getCurrentCUDAStream();
    int  grid   = num_of_blocks > 0 ? static_cast<int>(num_of_blocks) : 304;
    primus_turbo::unpermute_impl<bfloat16, float>(
        reinterpret_cast<const bfloat16 *>(permuted_tokens.data_ptr()),
        reinterpret_cast<bfloat16 *>(output_tokens.data_ptr()), nullptr, nullptr,
        row_id_map.data_ptr<int>(), num_dispatched_tokens_tensor.data_ptr<int>(),
        static_cast<int>(num_of_local_experts), static_cast<int>(hidden_size), 0, 1, grid, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("permute_preprocessing", &jit_permute_preprocessing, py::arg("routing_map"),
          py::arg("num_dispatched_token_tensor"), py::arg("max_num_dispatched_tokens"),
          py::arg("num_of_local_experts"), py::arg("pad_multiple"), py::arg("num_permuted_tokens"));
    m.def("permute_preprocessing_v2", &jit_permute_preprocessing_v2, py::arg("routing_map"),
          py::arg("num_dispatched_token_tensor"), py::arg("max_num_dispatched_tokens"),
          py::arg("num_of_local_experts"), py::arg("pad_multiple"), py::arg("num_permuted_tokens"));
    m.def("permute_preprocessing_v3", &jit_permute_preprocessing_v3, py::arg("routing_map"),
          py::arg("num_dispatched_token_tensor"), py::arg("max_num_dispatched_tokens"),
          py::arg("num_of_local_experts"), py::arg("pad_multiple"), py::arg("num_permuted_tokens"));
    m.def("permute_preprocessing_v4", &jit_permute_preprocessing_v4, py::arg("routing_map"),
          py::arg("num_dispatched_token_tensor"), py::arg("max_num_dispatched_tokens"),
          py::arg("num_of_local_experts"), py::arg("pad_multiple"), py::arg("num_permuted_tokens"));
    m.def("permute_preprocessing_v5", &jit_permute_preprocessing_v5, py::arg("routing_map"),
          py::arg("num_dispatched_token_tensor"), py::arg("max_num_dispatched_tokens"),
          py::arg("num_of_local_experts"), py::arg("pad_multiple"), py::arg("num_permuted_tokens"));
    m.def("permute_launcher", &jit_permute_launcher, py::arg("tokens"), py::arg("output_tokens"),
          py::arg("row_id_map"), py::arg("num_dispatched_token_tensor"), py::arg("pad_multiple"),
          py::arg("num_of_local_experts"), py::arg("hidden_size"), py::arg("num_permuted_token"),
          py::arg("num_of_blocks"), py::arg("use_fp8") = false);
    m.def("unpermute_launcher", &jit_unpermute_launcher, py::arg("permuted_tokens"),
          py::arg("output_tokens"), py::arg("row_id_map"),
          py::arg("num_dispatched_tokens_tensor"), py::arg("num_of_local_experts"),
          py::arg("hidden_size"), py::arg("num_of_blocks"));
}
"""


def _load_jit_extension():
    """JIT-build the permute_v1 + v2 + v3 + v4 + v5 kernels into a single extension."""
    from torch.utils.cpp_extension import load

    JIT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    bind_src = JIT_BUILD_DIR / "permute_v1_v2_v3_v4_v5_jit_bind.cu"
    bind_src.write_text(_JIT_BIND_CPP)

    print(
        f"[bench] JIT-building permute_v1 + v2 + v3 + v4 + v5 (this happens once; "
        f"output at {JIT_BUILD_DIR})..."
    )
    ext = load(
        name="primus_turbo_permute_v1_v2_v3_v4_v5_jit",
        sources=[
            str(REPO_ROOT / "csrc" / "kernels" / "permute" / "permute_v1.cu"),
            str(REPO_ROOT / "csrc" / "kernels" / "permute" / "permute_v2.cu"),
            str(REPO_ROOT / "csrc" / "kernels" / "permute" / "permute_v3.cu"),
            str(REPO_ROOT / "csrc" / "kernels" / "permute" / "permute_v4.cu"),
            str(REPO_ROOT / "csrc" / "kernels" / "permute" / "permute_v5.cu"),
            str(bind_src),
        ],
        extra_include_paths=[
            str(REPO_ROOT / "csrc"),
            str(REPO_ROOT / "csrc" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-fno-offload-uniform-block",
            "-DPRIMUS_TURBO_GFX942",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT16_OPERATORS__",
            "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT162_OPERATORS__",
            "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        ],
        with_cuda=True,
        verbose=False,
        build_directory=str(JIT_BUILD_DIR),
    )
    return ext


SUPPORTED_VERSIONS = ("v1", "v2", "v3", "v4", "v5")


@dataclass
class PermuteAPI:
    """Three callables the benchmark needs, regardless of where they came from."""

    name: str                   # "v1", "v2" or "v3"
    preprocess: callable        # routing_map, ndt, max_T, E, pad, npt -> (rim, tpe, of)
    permute: callable           # tokens, out, rim, ndt, *, kwargs -> None
    unpermute: callable         # ptokens, out, rim, ndt, *, kwargs -> None
    source: str


def _try_package_api_v1() -> Optional[PermuteAPI]:
    """Use the installed primus_turbo._C v1 ops if available (and not forced off)."""
    if os.environ.get("PRIMUS_TURBO_PERMUTE_BENCH_FORCE_JIT", "0") == "1":
        return None
    try:
        from primus_turbo.pytorch.ops.moe.permute_v1 import (
            permute_preprocessing_v1,
            permute_v1,
            unpermute_v1,
        )
    except Exception as exc:
        print(f"[bench] primus_turbo permute v1 ops not available ({exc}); will JIT.")
        return None
    return PermuteAPI(
        name="v1",
        preprocess=lambda rm, ndt, maxT, E, pad, npt: permute_preprocessing_v1(
            rm, ndt, maxT, E, pad, npt
        ),
        permute=lambda tokens, out, rim, ndt, *, pad, E, H, npt, blocks, use_fp8: permute_v1(
            tokens,
            out,
            rim,
            ndt,
            pad_multiple=pad,
            num_of_local_experts=E,
            hidden_size=H,
            num_permuted_token=npt,
            num_of_blocks_permute=blocks,
            use_fp8=use_fp8,
        ),
        unpermute=lambda ptokens, out, rim, ndt, *, E, H, blocks: unpermute_v1(
            ptokens, out, rim, ndt, num_of_local_experts=E, hidden_size=H,
            num_of_blocks_unpermute=blocks,
        ),
        source="primus_turbo.pytorch.ops",
    )


def _build_jit_v1_api(ext) -> PermuteAPI:
    return PermuteAPI(
        name="v1",
        preprocess=lambda rm, ndt, maxT, E, pad, npt: ext.permute_preprocessing(
            rm, ndt, maxT, E, pad, npt
        ),
        permute=lambda tokens, out, rim, ndt, *, pad, E, H, npt, blocks, use_fp8: ext.permute_launcher(
            tokens, out, rim, ndt, pad, E, H, npt, blocks, use_fp8
        ),
        unpermute=lambda ptokens, out, rim, ndt, *, E, H, blocks: ext.unpermute_launcher(
            ptokens, out, rim, ndt, E, H, blocks
        ),
        source="JIT (torch.utils.cpp_extension.load)",
    )


def _build_jit_v2_api(ext) -> PermuteAPI:
    """v2 only changes preprocessing; permute / unpermute reuse v1 launchers."""
    return PermuteAPI(
        name="v2",
        preprocess=lambda rm, ndt, maxT, E, pad, npt: ext.permute_preprocessing_v2(
            rm, ndt, maxT, E, pad, npt
        ),
        permute=lambda tokens, out, rim, ndt, *, pad, E, H, npt, blocks, use_fp8: ext.permute_launcher(
            tokens, out, rim, ndt, pad, E, H, npt, blocks, use_fp8
        ),
        unpermute=lambda ptokens, out, rim, ndt, *, E, H, blocks: ext.unpermute_launcher(
            ptokens, out, rim, ndt, E, H, blocks
        ),
        source="JIT (torch.utils.cpp_extension.load)",
    )


def _build_jit_v3_api(ext) -> PermuteAPI:
    """v3 only changes preprocessing; permute / unpermute reuse v1 launchers."""
    return PermuteAPI(
        name="v3",
        preprocess=lambda rm, ndt, maxT, E, pad, npt: ext.permute_preprocessing_v3(
            rm, ndt, maxT, E, pad, npt
        ),
        permute=lambda tokens, out, rim, ndt, *, pad, E, H, npt, blocks, use_fp8: ext.permute_launcher(
            tokens, out, rim, ndt, pad, E, H, npt, blocks, use_fp8
        ),
        unpermute=lambda ptokens, out, rim, ndt, *, E, H, blocks: ext.unpermute_launcher(
            ptokens, out, rim, ndt, E, H, blocks
        ),
        source="JIT (torch.utils.cpp_extension.load)",
    )


def _build_jit_v4_api(ext) -> PermuteAPI:
    """v4 only changes preprocessing; permute / unpermute reuse v1 launchers."""
    return PermuteAPI(
        name="v4",
        preprocess=lambda rm, ndt, maxT, E, pad, npt: ext.permute_preprocessing_v4(
            rm, ndt, maxT, E, pad, npt
        ),
        permute=lambda tokens, out, rim, ndt, *, pad, E, H, npt, blocks, use_fp8: ext.permute_launcher(
            tokens, out, rim, ndt, pad, E, H, npt, blocks, use_fp8
        ),
        unpermute=lambda ptokens, out, rim, ndt, *, E, H, blocks: ext.unpermute_launcher(
            ptokens, out, rim, ndt, E, H, blocks
        ),
        source="JIT (torch.utils.cpp_extension.load)",
    )


def _build_jit_v5_api(ext) -> PermuteAPI:
    """v5 only changes preprocessing; permute / unpermute reuse v1 launchers."""
    return PermuteAPI(
        name="v5",
        preprocess=lambda rm, ndt, maxT, E, pad, npt: ext.permute_preprocessing_v5(
            rm, ndt, maxT, E, pad, npt
        ),
        permute=lambda tokens, out, rim, ndt, *, pad, E, H, npt, blocks, use_fp8: ext.permute_launcher(
            tokens, out, rim, ndt, pad, E, H, npt, blocks, use_fp8
        ),
        unpermute=lambda ptokens, out, rim, ndt, *, E, H, blocks: ext.unpermute_launcher(
            ptokens, out, rim, ndt, E, H, blocks
        ),
        source="JIT (torch.utils.cpp_extension.load)",
    )


def get_apis(versions: List[str]) -> "dict[str, PermuteAPI]":
    """Return ``{version: PermuteAPI}`` for the requested versions."""
    for v in versions:
        if v not in SUPPORTED_VERSIONS:
            raise ValueError(f"unknown version {v!r}; supported: {SUPPORTED_VERSIONS}")

    apis: "dict[str, PermuteAPI]" = {}
    ext = None

    if "v1" in versions:
        api_v1 = _try_package_api_v1()
        if api_v1 is None:
            ext = _load_jit_extension()
            api_v1 = _build_jit_v1_api(ext)
        apis["v1"] = api_v1

    if "v2" in versions:
        if ext is None:
            ext = _load_jit_extension()
        apis["v2"] = _build_jit_v2_api(ext)

    if "v3" in versions:
        if ext is None:
            ext = _load_jit_extension()
        apis["v3"] = _build_jit_v3_api(ext)

    if "v4" in versions:
        if ext is None:
            ext = _load_jit_extension()
        apis["v4"] = _build_jit_v4_api(ext)

    if "v5" in versions:
        if ext is None:
            ext = _load_jit_extension()
        apis["v5"] = _build_jit_v5_api(ext)

    return apis


# -----------------------------------------------------------------------------
# DeepEP-style dispatch simulator.
#
# 思路：
#   * 全局共有 R 个 rank，每个 rank 上有 N 个 source token。
#   * 每个 token 在 E 个 expert 上 *不放回均匀* 选择 K 个 (topk)。
#   * Local rank 持有 expert idx ∈ [0, E/R)。
#   * DeepEP 对 token 去重：同一 token 即使路由到本 rank 多个 local expert，
#     也只发送一次。所以一个 token 进入本 rank 的 ⇔ 至少有一个 expert 落在
#     [0, E/R) 内。
#
# 期望接收 token 数：
#   E[N_recv] = R * N * (1 - C(E - E/R, K) / C(E, K))
# -----------------------------------------------------------------------------


def expected_recv_num_tokens(num_tokens_per_rank: int, num_experts: int,
                             num_topk: int, num_ranks: int) -> float:
    """Closed-form expectation E[recv_num_tokens] for the simulator above."""
    if num_experts <= 0 or num_topk <= 0 or num_ranks <= 0:
        return 0.0
    num_local_experts = num_experts // num_ranks
    p_miss = 1.0
    for i in range(num_topk):
        num = (num_experts - num_local_experts) - i
        den = num_experts - i
        if num <= 0:
            p_miss = 0.0
            break
        p_miss *= num / den
    return num_ranks * num_tokens_per_rank * (1.0 - p_miss)


def simulate_dispatch_recv(
    num_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    *,
    device: torch.device,
    seed: int = 0,
):
    """模拟 DeepEP intranode dispatch 的输出。

    Returns
    -------
    recv_topk_idx : torch.Tensor
        ``int64 [recv_num_tokens, num_topk]``，元素为本 rank 的 local expert
        idx ∈ [0, num_local_experts)，无效槽位填 ``-1``。
    recv_num_tokens : int
    """
    assert num_experts % num_ranks == 0, \
        f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
    num_local_experts = num_experts // num_ranks
    total = num_ranks * num_tokens_per_rank

    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((total, num_experts), generator=g)
    _, topk_idx = scores.topk(num_topk, dim=1)  # [total, K] in [0, E)

    local_mask = topk_idx < num_local_experts                # [total, K] bool
    has_local = local_mask.any(dim=1)                        # [total] bool
    recv_num_tokens = int(has_local.sum().item())

    # 把非本地 expert 替换为大值 (sentinel)，sort 后大值在右；最后再换回 -1。
    sentinel = num_experts
    masked = torch.where(local_mask, topk_idx, torch.full_like(topk_idx, sentinel))
    sorted_idx, _ = torch.sort(masked, dim=1)
    sorted_idx = torch.where(
        sorted_idx == sentinel, torch.full_like(sorted_idx, -1), sorted_idx
    )

    recv_topk_idx = sorted_idx[has_local].to(dtype=torch.int64, device=device)
    return recv_topk_idx, recv_num_tokens


def routing_map_from_recv_topk_idx(recv_topk_idx: torch.Tensor,
                                   num_local_experts: int) -> torch.Tensor:
    """Build the boolean ``routing_map [recv_num_tokens, num_local_experts]``
    that the permute kernels expect from the simulated ``recv_topk_idx``.
    """
    T, _ = recv_topk_idx.shape
    device = recv_topk_idx.device
    routing = torch.zeros((T, num_local_experts), dtype=torch.bool, device=device)
    if T == 0:
        return routing
    valid = recv_topk_idx >= 0
    if valid.any():
        rows = torch.arange(T, device=device).unsqueeze(1).expand_as(recv_topk_idx)
        routing[rows[valid], recv_topk_idx[valid].long()] = True
    return routing


# -----------------------------------------------------------------------------
# Reference implementations (pure torch).
# -----------------------------------------------------------------------------

def reference_permute(tokens: torch.Tensor, row_id_map: torch.Tensor, num_permuted: int,
                      num_local_experts: int) -> torch.Tensor:
    T, H = tokens.shape
    permuted = torch.zeros((num_permuted, H), dtype=tokens.dtype, device=tokens.device)
    flat = row_id_map[:T].reshape(-1)              # (T*E,)
    src = torch.arange(T, device=tokens.device).repeat_interleave(num_local_experts)
    pos_mask = flat > 0
    dest = (flat[pos_mask] - 1).long()
    permuted[dest] = tokens[src[pos_mask]]
    return permuted


def reference_unpermute(permuted: torch.Tensor, row_id_map: torch.Tensor, num_dispatched: int,
                        num_local_experts: int) -> torch.Tensor:
    H = permuted.shape[-1]
    flat = row_id_map[:num_dispatched].reshape(-1)
    src_token = torch.arange(num_dispatched, device=permuted.device).repeat_interleave(
        num_local_experts)
    pos_mask = flat > 0
    src = (flat[pos_mask] - 1).long()
    dst = src_token[pos_mask]
    acc = torch.zeros((num_dispatched, H), dtype=torch.float32, device=permuted.device)
    acc.index_add_(0, dst, permuted[src].float())
    return acc.to(permuted.dtype)


# -----------------------------------------------------------------------------
# Test-case generator (driven by config.MoEModelConfigs).
# -----------------------------------------------------------------------------

@dataclass
class PermuteCase:
    label: str          # 例如 "DeepSeek-V3/MBS=2/EP=8"
    model: str
    mbs: int
    ep: int
    num_tokens_per_rank: int   # 单 rank 上的 source token 数
    hidden_size: int
    num_experts: int           # 全局 expert 数
    num_topk: int
    num_local_experts: int     # = num_experts // ep


# LDS budget for the v1/v2 preprocessing kernels on gfx942 (MI300X / MI325X).
#
# Both implementations use:
#   dyn_shmem = block_size * num_local_experts * sizeof(int) bytes
# plus a static `hipcub::BlockScan<int32_t, block_size>::TempStorage` that
# costs roughly 2 KiB at block_size=512. The HW per-block LDS cap on gfx942
# is 65536 bytes (64 KiB).
#
# Empirically, ``num_local_experts >= 32`` overflows this budget (dyn = 64 KiB
# alone) and triggers `HSA_STATUS_ERROR_INVALID_ALLOCATION` — a *fatal* ROCm
# error that aborts the process and corrupts subsequent launches. We therefore
# skip such cases up front and surface the reason in the report instead of
# letting them crash the whole run.
PREPROC_BLOCK_SIZE = 512
LDS_LIMIT_BYTES = 65536
LDS_BLOCKSCAN_STATIC_BYTES = 2 * 1024  # conservative upper bound


def _expected_preproc_lds_bytes(num_local_experts: int) -> int:
    return PREPROC_BLOCK_SIZE * num_local_experts * 4 + LDS_BLOCKSCAN_STATIC_BYTES


def _case_skip_reason(case: "PermuteCase") -> Optional[str]:
    """Return a non-empty reason string if the case is known to fatally crash."""
    lds = _expected_preproc_lds_bytes(case.num_local_experts)
    if lds > LDS_LIMIT_BYTES:
        return (
            f"preproc LDS estimate {lds} B > device cap {LDS_LIMIT_BYTES} B "
            f"(num_local_experts={case.num_local_experts}, block_size={PREPROC_BLOCK_SIZE}); "
            f"would trigger HSA_STATUS_ERROR_INVALID_ALLOCATION"
        )
    return None


def gen_permute_test_cases() -> List["PermuteCase"]:
    """对每个 MoE 模型 × MBS × EP 生成 permute case。

    每个 case 描述一次 dispatch 后 *本 rank* 上的 permute 工作负载。
    """
    cases: List["PermuteCase"] = []
    for model_name, cfg in MoEModelConfigs.items():
        seqlen = cfg["seqlen"]
        hidden = cfg["hidden_size"]
        num_experts = cfg["num_experts"]
        num_topk = cfg["num_topk"]
        for mbs in BATCH_SIZE_LIST:
            num_tokens_per_rank = seqlen * mbs
            for ep in GROUPED_GEMM_EP_SIZE_LIST:
                if num_experts % ep != 0:
                    continue
                num_local_experts = num_experts // ep
                if num_local_experts == 0:
                    continue
                cases.append(
                    PermuteCase(
                        label=f"{model_name}/MBS={mbs}/EP={ep}",
                        model=model_name,
                        mbs=mbs,
                        ep=ep,
                        num_tokens_per_rank=num_tokens_per_rank,
                        hidden_size=hidden,
                        num_experts=num_experts,
                        num_topk=num_topk,
                        num_local_experts=num_local_experts,
                    )
                )
    return cases


# -----------------------------------------------------------------------------
# Benchmark drivers.
# -----------------------------------------------------------------------------

def _check_close(name: str, ref: torch.Tensor, got: torch.Tensor) -> bool:
    if not torch.allclose(ref.float(), got.float(), atol=1e-2, rtol=1e-2):
        diff = (ref.float() - got.float()).abs()
        print(f"  [{name}] FAIL: max-abs-diff={diff.max().item():.4f}")
        return False
    return True


def _bench_callable(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    return t.timeit(iters).mean * 1e3  # ms


def _profile_kernel_breakdown(fn, kernel_substrs: "list[tuple[str, str]]", *,
                              num_warmups: int = 3, num_tests: int = 10
                              ) -> "dict[str, Optional[float]]":
    """跑 ``fn`` ``num_tests`` 次，返回每个匹配 kernel 的 device 平均时间 (µs)。

    ``kernel_substrs`` 形如 ``[(label, must-include-substring), ...]``。每个标签
    返回 ``None`` 表示在 profiler 输出里没有匹配上。
    """
    import contextlib

    from torch.profiler import ProfilerActivity, profile, schedule

    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    @contextlib.contextmanager
    def _silence():
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield

    sched = schedule(wait=1, warmup=0, active=1, repeat=1)
    with _silence():
        with profile(activities=[ProfilerActivity.CUDA], schedule=sched) as prof:
            for _ in range(2):
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    out: "dict[str, Optional[float]]" = {label: None for label, _ in kernel_substrs}
    for evt in prof.key_averages():
        key = evt.key
        for label, substr in kernel_substrs:
            if out[label] is None and substr in key:
                dev_us = getattr(evt, "device_time_total", 0) or 0
                if dev_us == 0:
                    dev_us = getattr(evt, "cuda_time_total", 0) or 0
                count = max(int(evt.count), 1)
                out[label] = float(dev_us) / count
                break
    return out


# 每个版本的 preproc kernel mangled-name 子串，用于在 torch.profiler 输出里
# 定位对应的 device 计时项。
#   * v1 — 单个 cooperative kernel
#   * v2 — 拆成 4 个 pass
#   * v3 — 单个非 cooperative kernel（lookback scan）
#
# 注意：v1 / v3 都把 preproc kernel 命名为 `permute_preprocessing_kernel`，但
# 它们位于不同 namespace（v1 在 `primus_turbo::`；v3 在 `primus_turbo::v3::`），
# 所以 demangled name 不一样。下面用 `::v3::permute_preprocessing_kernel` 区分。
# v1 这一项无需匹配 v3 的同名 kernel——`_profile_kernel_breakdown` 是按各自的
# `preproc_only` fn 单独 profile 的，v1 fn 调用只会产生 v1 kernel。
PREPROC_KERNEL_BREAKDOWN: "dict[str, list[tuple[str, str]]]" = {
    "v1": [
        ("permute_preprocessing", "permute_preprocessing_kernel"),
    ],
    "v2": [
        ("pass1",     "permute_pass1_kernel"),
        ("pass2",     "permute_pass2_kernel"),
        ("pass3_pad", "permute_pass3_pad_kernel"),
        ("finalize",  "permute_finalize_kernel"),
    ],
    "v3": [
        ("permute_preprocessing", "v3::permute_preprocessing_kernel"),
    ],
    "v4": [
        ("permute_preprocessing", "v4::permute_preprocessing_kernel"),
    ],
    "v5": [
        ("permute_preprocessing", "v5::permute_preprocessing_kernel"),
    ],
}


@dataclass
class HostTimingResult:
    correct: bool
    preproc_ms: float    # forward stage 1: build row_id_map / tokens_per_expert
    permute_ms: float    # forward stage 2: data movement
    fwd_ms: float        # = preproc_ms + permute_ms
    fwd_gbps: float
    bwd_ms: float        # backward = unpermute kernel only
    bwd_gbps: float
    # preproc 内部 kernel breakdown (device avg µs)；v1 只有 1 项，v2 有 4 项；
    # 失败 / 缺失时该项为 None。键与 PREPROC_KERNEL_BREAKDOWN[name] 对齐。
    preproc_breakdown_us: "dict[str, Optional[float]]" = None


def run_one(apis: "dict[str, PermuteAPI]", case: PermuteCase, *,
            num_blocks: int, warmup: int, iters: int, seed: int = 0
            ) -> "tuple[dict[str, HostTimingResult], int]":
    """Benchmark all selected APIs against one case independently.

    每个 api 自己 preproc + permute + unpermute，并各自和 torch reference 比对；
    一个 api 失败不会牵连其它 api（v1 的 cooperative launch 在 num_local_experts
    较大时容易超出 active block 上限，而 v2 的普通 launch 仍能跑通）。

    返回 ``num_permuted`` 取自首个成功的 api，仅供报表展示。
    """
    device = torch.device("cuda")

    recv_topk_idx, recv_num_tokens = simulate_dispatch_recv(
        case.num_tokens_per_rank,
        case.num_experts,
        case.num_topk,
        case.ep,
        device=device,
        seed=seed,
    )

    if recv_num_tokens == 0:
        return {name: HostTimingResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for name in apis}, 0

    routing = routing_map_from_recv_topk_idx(recv_topk_idx, case.num_local_experts)

    num_dispatched = recv_num_tokens
    num_dispatched_t = torch.tensor([num_dispatched], dtype=torch.int32, device=device)
    pad_multiple = 0

    tokens = torch.randn(
        (num_dispatched, case.hidden_size), dtype=torch.bfloat16, device=device)
    bytes_per_elem = 2  # bfloat16

    results: "dict[str, HostTimingResult]" = {}
    reported_num_permuted: Optional[int] = None
    for name, api in apis.items():
        try:
            row_id_map, tokens_per_expert, overflow = api.preprocess(
                routing, num_dispatched_t, num_dispatched, case.num_local_experts,
                pad_multiple, -1,
            )
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"  [{name}] preproc launch failed: {exc}")
            results[name] = HostTimingResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue

        if overflow.item() != 0:
            print(f"  [{name}] preproc overflow flag set unexpectedly")
            results[name] = HostTimingResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue

        num_permuted = int(tokens_per_expert.sum().item())
        if reported_num_permuted is None:
            reported_num_permuted = num_permuted

        # 每个 api 各自构造 reference (取决于 row_id_map 编号约定)。
        ref_perm = reference_permute(
            tokens, row_id_map, num_permuted, case.num_local_experts)
        ref_unperm = reference_unpermute(
            ref_perm, row_id_map, num_dispatched, case.num_local_experts)

        perm_bytes = (num_dispatched + num_permuted) * case.hidden_size * bytes_per_elem
        unperm_bytes = (num_permuted + num_dispatched) * case.hidden_size * bytes_per_elem

        permuted_tokens = torch.empty(
            (num_permuted, case.hidden_size), dtype=torch.bfloat16, device=device)
        recovered = torch.empty(
            (num_dispatched, case.hidden_size), dtype=torch.bfloat16, device=device)

        try:
            api.permute(
                tokens, permuted_tokens, row_id_map, num_dispatched_t,
                pad=pad_multiple, E=case.num_local_experts, H=case.hidden_size,
                npt=num_permuted, blocks=num_blocks, use_fp8=False,
            )
            torch.cuda.synchronize()
            ok_p = _check_close(f"{name}/permute", ref_perm, permuted_tokens)

            api.unpermute(
                permuted_tokens, recovered, row_id_map, num_dispatched_t,
                E=case.num_local_experts, H=case.hidden_size, blocks=num_blocks,
            )
            torch.cuda.synchronize()
            ok_u = _check_close(f"{name}/unpermute", ref_unperm, recovered)
            correct = ok_p and ok_u
        except Exception as exc:
            print(f"  [{name}] permute / unpermute launch failed: {exc}")
            results[name] = HostTimingResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue

        # Host-side wall-clock timing.
        # Forward 拆成两个独立计时阶段：
        #   preproc — 构建 row_id_map / tokens_per_expert
        #   permute — 真正的数据搬运
        # 训练侧 forward 总耗时 = preproc + permute；带宽用合计时间。
        def preproc_only(api=api):
            api.preprocess(
                routing, num_dispatched_t, num_dispatched, case.num_local_experts,
                pad_multiple, -1)

        def permute_only(api=api, row_id_map=row_id_map,
                         permuted_tokens=permuted_tokens, num_permuted=num_permuted):
            api.permute(
                tokens, permuted_tokens, row_id_map, num_dispatched_t,
                pad=pad_multiple, E=case.num_local_experts, H=case.hidden_size,
                npt=num_permuted, blocks=num_blocks, use_fp8=False,
            )

        def bwd(api=api, row_id_map=row_id_map, permuted_tokens=permuted_tokens,
                recovered=recovered):
            api.unpermute(
                permuted_tokens, recovered, row_id_map, num_dispatched_t,
                E=case.num_local_experts, H=case.hidden_size, blocks=num_blocks,
            )

        preproc_ms = _bench_callable(preproc_only, warmup=warmup, iters=iters)
        permute_ms = _bench_callable(permute_only, warmup=warmup, iters=iters)
        bwd_ms = _bench_callable(bwd, warmup=warmup, iters=iters)
        fwd_ms = preproc_ms + permute_ms
        fwd_gbps = perm_bytes / (fwd_ms * 1e-3) / 1e9
        bwd_gbps = unperm_bytes / (bwd_ms * 1e-3) / 1e9

        # Preproc kernel breakdown via torch.profiler. v1 = 1 cooperative kernel,
        # v2 = pass1 + pass2 + pass3_pad + finalize (4 kernels).
        breakdown_specs = PREPROC_KERNEL_BREAKDOWN.get(name, [])
        try:
            breakdown_us = (
                _profile_kernel_breakdown(preproc_only, breakdown_specs)
                if breakdown_specs else {}
            )
        except Exception as exc:
            print(f"  [{name}] kernel breakdown profile failed: {exc}")
            breakdown_us = {label: None for label, _ in breakdown_specs}

        results[name] = HostTimingResult(
            correct=correct,
            preproc_ms=preproc_ms,
            permute_ms=permute_ms,
            fwd_ms=fwd_ms,
            fwd_gbps=fwd_gbps,
            bwd_ms=bwd_ms,
            bwd_gbps=bwd_gbps,
            preproc_breakdown_us=breakdown_us,
        )

    return results, (reported_num_permuted if reported_num_permuted is not None else 0)


# -----------------------------------------------------------------------------
# Entry point.
# -----------------------------------------------------------------------------

def benchmark_permute(versions: List[str], *, num_blocks: int, warmup: int,
                      iters: int, output_csv: Optional[str], seed: int,
                      case_filter: Optional[str] = None, limit: Optional[int] = None,
                      include_unsafe: bool = False):
    platform, gpu_name = get_platform_info()
    apis = get_apis(versions)
    for name, api in apis.items():
        print(f"[bench] {name} kernel from: {api.source}")

    cases = gen_permute_test_cases()
    if case_filter:
        keep = [k.strip() for k in case_filter.split(",") if k.strip()]
        cases = [c for c in cases if any(k in c.label for k in keep)]
    if limit is not None and limit > 0:
        cases = cases[:limit]
    print(f"[bench] {len(cases)} cases to run "
          f"(filter={case_filter!r}, limit={limit})")

    rows: List[dict] = []
    test_id = 0
    for case in cases:
        test_id += 1
        exp_recv = expected_recv_num_tokens(
            case.num_tokens_per_rank, case.num_experts, case.num_topk, case.ep)
        print(f"\n{'='*60}")
        print(
            f"TestID: {test_id}, Case: {case.label}, "
            f"N/rank: {case.num_tokens_per_rank}, hidden: {case.hidden_size}, "
            f"E: {case.num_experts}, K: {case.num_topk}, "
            f"local_E: {case.num_local_experts}, "
            f"E[recv_tokens]: {exp_recv:.0f}"
        )
        print(f"{'='*60}")

        row: dict = {
            "TestID": test_id,
            "Platform": platform,
            "GPU": gpu_name,
            "Case": case.model,
            "MBS": case.mbs,
            "EP": case.ep,
            "num_tokens_per_rank": case.num_tokens_per_rank,
            "hidden": case.hidden_size,
            "num_experts": case.num_experts,
            "num_topk": case.num_topk,
            "num_local_experts": case.num_local_experts,
            "expected_recv_tokens": int(round(exp_recv)),
        }

        # Skip cases that would trigger ROCm fatal aborts so the rest of the
        # suite can still run. Use --include-unsafe to override.
        skip_reason = _case_skip_reason(case)
        if skip_reason and not include_unsafe:
            print(f"  SKIP: {skip_reason}")
            row["num_permuted"] = "SKIP"
            for v in versions:
                row[f"{v}/Check"] = "SKIP"
                row[f"{v}/Preproc Time (ms)"] = "SKIP"
                row[f"{v}/Permute Time (ms)"] = "SKIP"
                row[f"{v}/Forward Time (ms)"] = "SKIP"
                row[f"{v}/Forward Bandwidth (GB/s)"] = "0.00"
                row[f"{v}/Backward Time (ms)"] = "SKIP"
                row[f"{v}/Backward Bandwidth (GB/s)"] = "0.00"
                for label, _ in PREPROC_KERNEL_BREAKDOWN.get(v, []):
                    row[f"{v}/preproc_{label}_us"] = "SKIP"
            if len(versions) >= 2:
                row[f"Speedup Forward ({versions[-1]}/{versions[0]})"] = "-"
                row[f"Speedup Backward ({versions[-1]}/{versions[0]})"] = "-"
            row["SkipReason"] = skip_reason
            rows.append(row)
            continue

        try:
            results, num_permuted = run_one(
                apis, case,
                num_blocks=num_blocks, warmup=warmup, iters=iters, seed=seed,
            )
            # num_permuted = 真实路由次数（含每 token 的多个 local-expert）。
            row["num_permuted"] = num_permuted
            for v in versions:
                r = results[v]
                row[f"{v}/Check"] = "PASS" if r.correct else "FAIL"
                row[f"{v}/Preproc Time (ms)"] = f"{r.preproc_ms:.3f}"
                row[f"{v}/Permute Time (ms)"] = f"{r.permute_ms:.3f}"
                row[f"{v}/Forward Time (ms)"] = f"{r.fwd_ms:.3f}"
                row[f"{v}/Forward Bandwidth (GB/s)"] = f"{r.fwd_gbps:.2f}"
                row[f"{v}/Backward Time (ms)"] = f"{r.bwd_ms:.3f}"
                row[f"{v}/Backward Bandwidth (GB/s)"] = f"{r.bwd_gbps:.2f}"
                # 把 preproc kernel breakdown (device µs) 加进 CSV。
                breakdown = r.preproc_breakdown_us or {}
                for label, _ in PREPROC_KERNEL_BREAKDOWN.get(v, []):
                    val = breakdown.get(label)
                    row[f"{v}/preproc_{label}_us"] = (
                        f"{val:.2f}" if val is not None else "-"
                    )

                print(
                    f"  [{v}] Forward {r.preproc_ms:.3f} + {r.permute_ms:.3f} ms "
                    f"({r.fwd_gbps:.1f} GB/s) | "
                    f"Backward {r.bwd_ms:.3f} ms ({r.bwd_gbps:.1f} GB/s) | "
                    f"{'PASS' if r.correct else 'FAIL'}"
                )
                if breakdown:
                    parts = []
                    total = 0.0
                    for label, _ in PREPROC_KERNEL_BREAKDOWN.get(v, []):
                        val = breakdown.get(label)
                        parts.append(
                            f"{label}={val:.1f}" if val is not None else f"{label}=?"
                        )
                        if val is not None:
                            total += val
                    print(
                        f"       {v} preproc kernels (device µs): "
                        f"{'  '.join(parts)}  →  total={total:.1f}"
                    )

            if len(versions) >= 2:
                base, tgt = results[versions[0]], results[versions[-1]]
                fwd_sp = (base.fwd_ms / tgt.fwd_ms) if (base.fwd_ms > 0 and tgt.fwd_ms > 0) else 0.0
                bwd_sp = (base.bwd_ms / tgt.bwd_ms) if (base.bwd_ms > 0 and tgt.bwd_ms > 0) else 0.0
                row[f"Speedup Forward ({versions[-1]}/{versions[0]})"] = f"{fwd_sp:.2f}"
                row[f"Speedup Backward ({versions[-1]}/{versions[0]})"] = f"{bwd_sp:.2f}"

        except Exception as exc:
            print(f"Failed: {exc}")
            row["num_permuted"] = "ERROR"
            for v in versions:
                row[f"{v}/Check"] = "ERROR"
                row[f"{v}/Preproc Time (ms)"] = "ERROR"
                row[f"{v}/Permute Time (ms)"] = "ERROR"
                row[f"{v}/Forward Time (ms)"] = "ERROR"
                row[f"{v}/Forward Bandwidth (GB/s)"] = "0.00"
                row[f"{v}/Backward Time (ms)"] = "ERROR"
                row[f"{v}/Backward Bandwidth (GB/s)"] = "0.00"
                for label, _ in PREPROC_KERNEL_BREAKDOWN.get(v, []):
                    row[f"{v}/preproc_{label}_us"] = "ERROR"
            if len(versions) >= 2:
                row[f"Speedup Forward ({versions[-1]}/{versions[0]})"] = "-"
                row[f"Speedup Backward ({versions[-1]}/{versions[0]})"] = "-"

        rows.append(row)

    # 用所有行的 key 并集作为 DataFrame 列，保证缺列 case 也能写出。
    all_keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    results_df = pd.DataFrame(rows, columns=all_keys)
    print("\nFinal Results:")
    print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))

    for v in versions:
        col = f"{v}/Forward Bandwidth (GB/s)"
        if col in results_df.columns:
            avg = pd.to_numeric(results_df[col], errors="coerce").mean()
            print(f"Average {v} Forward Bandwidth (GB/s): {avg:.2f}")
        col = f"{v}/Backward Bandwidth (GB/s)"
        if col in results_df.columns:
            avg = pd.to_numeric(results_df[col], errors="coerce").mean()
            print(f"Average {v} Backward Bandwidth (GB/s): {avg:.2f}")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        tag = "_".join(versions)
        filename = f"permute_{tag}_{timestamp}_{gpu_name}.csv"
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--versions", type=str, default="v1,v2,v3,v5",
        help=f"comma-separated permute versions (supported: {','.join(SUPPORTED_VERSIONS)}; "
             f"default: v1,v2,v3,v5). Speedup column compares last vs first.")
    parser.add_argument("--num-blocks", type=int, default=304,
                        help="grid size passed to the kernel (0 = use device CU count)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="optional CSV output path")
    parser.add_argument("--case-filter", type=str, default=None,
                        help="comma-separated substrings to filter case labels "
                             "(e.g. 'Mixtral-8x7B,DeepSeek-V2-Lite/MBS=1')")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of cases (after filtering) to run")
    parser.add_argument("--include-unsafe", action="store_true",
                        help="run cases that are predicted to trigger fatal "
                             "HSA_STATUS_ERROR_INVALID_ALLOCATION (LDS overflow). "
                             "WARNING: a fatal abort will terminate the entire process.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm not available; aborting.")
        sys.exit(1)

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    if not versions:
        print("error: --versions must list at least one version")
        sys.exit(1)

    benchmark_permute(
        versions,
        num_blocks=args.num_blocks,
        warmup=args.warmup,
        iters=args.iters,
        output_csv=args.output,
        seed=args.seed,
        case_filter=args.case_filter,
        limit=args.limit,
        include_unsafe=args.include_unsafe,
    )


if __name__ == "__main__":
    main()
