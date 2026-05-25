###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Numerical-verification test for the gfx950 mega-MoE kernel,
JIT-loaded from ``tests/pytorch/ops/jit/mega_moe_jit_*``.

Three gates (all hard pass/fail — process exits non-zero on any fail):

  1. Layout parity (C++ <-> Python buffer offsets agree across 4 shapes)
     plus a ``num_tokens == 0`` early-exit smoke kernel launch.

  2. Non-zero token kernel launch that actually exercises the dispatch
     path, followed by **byte-exact numerical verification of the L1
     pool contents** (token bytes + UE8M0 SF + topk weights) against a
     Python reference built from the inputs.

  3. **Kernel y precision standard.**  End-to-end fused-MoE numerics
     comparison of the kernel's ``y`` output against a Python reference
     that mirrors the kernel pipeline (FP8 GEMM → SwiGLU → FP8 quantise
     → FP8 GEMM → BF16 topk-sum).  PASS requires both
     ``cos_sim >= 0.99`` AND ``rel_rmse <= 0.05`` (production-grade FP8
     tolerance — see ``_GATE_3_*`` constants).  Gate 3 currently FAILS
     because the kernel's GEMM body has at least one structural bug
     beyond the documented L2 SF placeholder (cos_sim ≈ 0); see
     ``project_megamoe_y_numerics_uncorrelated.md`` for the suspect
     ranking and bisection plan.

Usage:

    # gate 1: parity + early-exit smoke (num_tokens=0 path)
    python tests/pytorch/ops/jit/test_mega_moe_jit_perf.py --parity-only

    # gate 2: kernel launch + dispatch numerical verification
    python tests/pytorch/ops/jit/test_mega_moe_jit_perf.py \\
        --num-tokens 64 --num-max-tokens-per-rank 384 --hidden 256 \\
        --intermediate-hidden 128 --num-experts 8 --num-topk 2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from dataclasses import dataclass

# Force the JIT compiler to only target gfx950 — PyTorch's HIP backend
# otherwise pre-pends every visible arch (incl. gfx942) and that fails
# on ``primus_turbo/float8.h`` 'half <-> float' conversions which are
# gated off via ``__HIP_NO_HALF_CONVERSIONS__`` on older targets.
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")


# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
CSRC_INCLUDE = os.path.join(REPO_ROOT, "csrc", "include")
CSRC_ROOT = os.path.join(REPO_ROOT, "csrc")


# ---------------------------------------------------------------------------
# Shape config (kept for the layout-parity tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeCfg:
    num_max_tokens_per_rank: int
    hidden: int
    intermediate_hidden: int
    num_experts: int
    num_topk: int
    num_ranks: int

    @property
    def num_experts_per_rank(self) -> int:
        assert self.num_experts % self.num_ranks == 0
        return self.num_experts // self.num_ranks


# ---------------------------------------------------------------------------
# Python reference for the DG-aligned layout helpers.
# ---------------------------------------------------------------------------

# Mirrors ``primus_turbo::mega_moe::kCandidateBlockM``.
_K_CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192)
_K_MAX_CANDIDATE_BLOCK_M = max(_K_CANDIDATE_BLOCK_M)
_K_MIN_CANDIDATE_BLOCK_M = min(_K_CANDIDATE_BLOCK_M)
_K_TOKEN_ALIGNMENT = 384  # == LCM(_K_CANDIDATE_BLOCK_M); kLCMCandidateBlockM in DG.
_K_SCALE_GROUP_K = 32
_K_SCALE_BLOCK_MN = 128

# Compile-time tile geometry the JIT smoke template instantiates.  Must
# stay in sync with the MEGA_MOE_JIT_KBLOCK_M / MEGA_MOE_JIT_KSF_BLOCK_M
# defaults in ``mega_moe_jit_launch.cu``.
_K_LOADER_BLOCK_M = 128
_K_LOADER_SF_BLOCK_M = 128


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def py_get_num_max_pool_tokens(
    num_ranks: int, num_max_tokens_per_rank: int, num_topk: int, num_experts_per_rank: int
) -> int:
    """Python port of ``layout::get_num_max_pool_tokens``."""
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return _align_up(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (_K_MAX_CANDIDATE_BLOCK_M - 1),
        _K_TOKEN_ALIGNMENT,
    )


def py_get_num_padded_sf_pool_tokens(num_max_pool_tokens: int, block_m: int) -> int:
    """Python port of ``layout::get_num_padded_sf_pool_tokens``."""
    return (num_max_pool_tokens // block_m) * _align_up(block_m, _K_SCALE_BLOCK_MN)


def py_workspace_bytes(num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, num_topk: int) -> int:
    """Python port of ``layout::Workspace::get_num_bytes``."""
    num_experts_per_rank = num_experts // num_ranks
    num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank
    num_max_pool_tokens = py_get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank
    )
    num_max_pool_blocks = num_max_pool_tokens // _K_MIN_CANDIDATE_BLOCK_M

    num_bytes = 32
    num_bytes += num_experts * 8 * 2
    num_bytes += num_experts_per_rank * 8
    num_bytes += _align_up(num_max_pool_blocks, 2) * 4
    num_bytes += num_max_pool_blocks * 8
    num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * 4
    num_bytes += num_max_pool_tokens * 12
    num_bytes = _align_up(num_bytes, 16)
    return num_bytes


@dataclass(frozen=True)
class _Layout:
    workspace_offset: int
    input_x_offset: int
    input_x_sf_offset: int
    input_topk_idx_offset: int
    input_topk_weights_offset: int
    l1_pool_x_offset: int
    l1_pool_x_sf_offset: int
    l1_pool_weights_offset: int
    l2_pool_x_offset: int
    l2_pool_x_sf_offset: int
    combine_buffer_offset: int
    total_bytes: int
    num_max_pool_tokens: int
    num_padded_sf_pool_tokens: int


def python_symm_buffer_layout(cfg: ShapeCfg) -> _Layout:
    """Pure-Python port of
    ``csrc/kernels/mega_moe/mega_moe.cu::get_symm_buffer_size_for_mega_moe``.
    """
    assert cfg.num_experts % cfg.num_ranks == 0
    assert cfg.hidden % 128 == 0
    assert cfg.intermediate_hidden % 128 == 0

    num_experts_per_rank = cfg.num_experts // cfg.num_ranks
    num_max_pool_tokens = py_get_num_max_pool_tokens(
        cfg.num_ranks, cfg.num_max_tokens_per_rank, cfg.num_topk, num_experts_per_rank
    )
    num_padded_sf_pool_tokens = max(
        py_get_num_padded_sf_pool_tokens(num_max_pool_tokens, bm) for bm in _K_CANDIDATE_BLOCK_M
    )
    workspace_bytes = py_workspace_bytes(
        cfg.num_ranks, cfg.num_experts, cfg.num_max_tokens_per_rank, cfg.num_topk
    )

    fp8_token_bytes = cfg.hidden
    bf16_token_bytes = cfg.hidden * 2
    fp8_inter_bytes = cfg.intermediate_hidden
    fp8_sf_bytes = cfg.hidden // _K_SCALE_GROUP_K
    fp8_inter_sf_bytes = cfg.intermediate_hidden // _K_SCALE_GROUP_K
    topk_idx_bytes = cfg.num_topk * 8  # int64
    topk_weights_bytes = cfg.num_topk * 4  # float32

    cursor = 0

    def bump(n: int) -> int:
        nonlocal cursor
        aligned = _align_up(cursor, 256)
        cursor = aligned + n
        return aligned

    workspace_offset = bump(workspace_bytes)
    input_x_offset = bump(cfg.num_max_tokens_per_rank * fp8_token_bytes)
    input_x_sf_offset = bump(cfg.num_max_tokens_per_rank * fp8_sf_bytes)
    input_topk_idx_offset = bump(cfg.num_max_tokens_per_rank * topk_idx_bytes)
    input_topk_weights_offset = bump(cfg.num_max_tokens_per_rank * topk_weights_bytes)
    l1_pool_x_offset = bump(num_max_pool_tokens * fp8_token_bytes)
    l1_pool_x_sf_offset = bump(num_padded_sf_pool_tokens * fp8_sf_bytes)
    l1_pool_weights_offset = bump(num_max_pool_tokens * 4)
    l2_pool_x_offset = bump(num_max_pool_tokens * fp8_inter_bytes)
    l2_pool_x_sf_offset = bump(num_padded_sf_pool_tokens * fp8_inter_sf_bytes)
    combine_buffer_offset = bump(cfg.num_topk * cfg.num_max_tokens_per_rank * bf16_token_bytes)

    return _Layout(
        workspace_offset=workspace_offset,
        input_x_offset=input_x_offset,
        input_x_sf_offset=input_x_sf_offset,
        input_topk_idx_offset=input_topk_idx_offset,
        input_topk_weights_offset=input_topk_weights_offset,
        l1_pool_x_offset=l1_pool_x_offset,
        l1_pool_x_sf_offset=l1_pool_x_sf_offset,
        l1_pool_weights_offset=l1_pool_weights_offset,
        l2_pool_x_offset=l2_pool_x_offset,
        l2_pool_x_sf_offset=l2_pool_x_sf_offset,
        combine_buffer_offset=combine_buffer_offset,
        total_bytes=cursor,
        num_max_pool_tokens=num_max_pool_tokens,
        num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
    )


# ---------------------------------------------------------------------------
# Workspace inner-offset helpers (Python port of layout::Workspace
# pointer arithmetic).  Used by the dispatch-numerical-verifier to read
# back ``src_token_topk_idx`` and ``token_src_metadata`` after the
# kernel finishes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkspaceInner:
    src_token_topk_idx_offset: int  # relative to workspace base
    token_src_metadata_offset: int


def python_workspace_inner_layout(
    num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, num_topk: int
) -> _WorkspaceInner:
    num_experts_per_rank = num_experts // num_ranks
    num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank
    num_max_pool_tokens = py_get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank
    )
    num_max_pool_blocks = num_max_pool_tokens // _K_MIN_CANDIDATE_BLOCK_M

    off = 32  # barrier signal pad
    off += num_experts * 8 * 2  # send + recv counts (uint64 each)
    off += num_experts_per_rank * 8  # recv_count_sum
    off += _align_up(num_max_pool_blocks, 2) * 4  # l1_arrival_count
    off += num_max_pool_blocks * 8  # l2_arrival_mask
    src_off = off
    off += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * 4
    md_off = off
    return _WorkspaceInner(src_token_topk_idx_offset=src_off, token_src_metadata_offset=md_off)


# ---------------------------------------------------------------------------
# JIT loader (cached across all ``ShapeCfg`` invocations).
# ---------------------------------------------------------------------------


_JIT_MODULE = None
_JIT_MODULE_KEY = None


_JIT_SHAPE_MACROS = {
    "num_max_tokens_per_rank": "MEGA_MOE_JIT_KNUMMAXTOKENSPERRANK",
    "hidden": "MEGA_MOE_JIT_KHIDDEN",
    "intermediate_hidden": "MEGA_MOE_JIT_KINTERMEDIATEHIDDEN",
    "num_experts": "MEGA_MOE_JIT_KNUMEXPERTS",
    "num_topk": "MEGA_MOE_JIT_KNUMTOPK",
    "num_ranks": "MEGA_MOE_JIT_KNUMRANKS",
}


def load_mega_moe_jit(shape: "dict | None" = None):
    """JIT-loads the layout-helper TU."""
    global _JIT_MODULE, _JIT_MODULE_KEY
    if shape is None:
        if _JIT_MODULE is not None:
            return _JIT_MODULE
        norm_shape = {}
    else:
        norm_shape = {k: int(v) for k, v in shape.items() if k in _JIT_SHAPE_MACROS}
    cache_key = tuple(sorted(norm_shape.items()))
    if _JIT_MODULE is not None:
        if _JIT_MODULE_KEY == cache_key:
            return _JIT_MODULE
        raise RuntimeError(
            f"load_mega_moe_jit called with conflicting shapes: cached={_JIT_MODULE_KEY}, "
            f"new={cache_key}. Pass the same shape to every call within a process."
        )

    from torch.utils.cpp_extension import load as cpp_load

    sources = [
        os.path.join(THIS_DIR, "mega_moe_jit_binding.cpp"),
        os.path.join(THIS_DIR, "mega_moe_jit_launch.cu"),
    ]
    half_undefs = [
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
    ]
    shape_defines = [f"-D{_JIT_SHAPE_MACROS[k]}={v}u" for k, v in sorted(norm_shape.items())]

    if shape_defines:
        ext_suffix = hashlib.sha1(
            ";".join(f"{k}={v}" for k, v in sorted(norm_shape.items())).encode()
        ).hexdigest()[:12]
        ext_name = f"mega_moe_jit_layout_probe_{ext_suffix}"
    else:
        ext_name = "mega_moe_jit_layout_probe"

    _JIT_MODULE = cpp_load(
        name=ext_name,
        sources=sources,
        extra_include_paths=[CSRC_INCLUDE, CSRC_ROOT],
        extra_cflags=["-O3", "-std=c++20"] + half_undefs + shape_defines,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--offload-arch=gfx950",
            "-fno-gpu-rdc",
        ]
        + half_undefs
        + shape_defines,
        with_cuda=True,
        verbose=True,
    )
    _JIT_MODULE_KEY = cache_key
    return _JIT_MODULE


# ---------------------------------------------------------------------------
# Layout parity checks
# ---------------------------------------------------------------------------


def _check_pool_token_helpers(cfg: ShapeCfg, jit) -> None:
    py_pool = py_get_num_max_pool_tokens(
        cfg.num_ranks, cfg.num_max_tokens_per_rank, cfg.num_topk, cfg.num_experts_per_rank
    )
    cpp_pool = jit.num_max_pool_tokens(
        cfg.num_ranks, cfg.num_max_tokens_per_rank, cfg.num_topk, cfg.num_experts_per_rank
    )
    assert py_pool == cpp_pool, f"get_num_max_pool_tokens mismatch for {cfg}: py={py_pool}, cpp={cpp_pool}"
    for bm in _K_CANDIDATE_BLOCK_M:
        py_sf = py_get_num_padded_sf_pool_tokens(py_pool, bm)
        cpp_sf = jit.num_padded_sf_pool_tokens(py_pool, bm)
        assert (
            py_sf == cpp_sf
        ), f"get_num_padded_sf_pool_tokens mismatch for {cfg}, block_m={bm}: py={py_sf}, cpp={cpp_sf}"


def _check_workspace_bytes(cfg: ShapeCfg, jit) -> None:
    py_ws = py_workspace_bytes(cfg.num_ranks, cfg.num_experts, cfg.num_max_tokens_per_rank, cfg.num_topk)
    cpp_ws_bytes, cpp_pool_tokens, cpp_pool_blocks = jit.workspace_probe(
        cfg.num_ranks, cfg.num_experts, cfg.num_max_tokens_per_rank, cfg.num_topk
    )
    assert (
        py_ws == cpp_ws_bytes
    ), f"Workspace::get_num_bytes mismatch for {cfg}: py={py_ws}, cpp={cpp_ws_bytes}"
    py_pool_tokens = py_get_num_max_pool_tokens(
        cfg.num_ranks, cfg.num_max_tokens_per_rank, cfg.num_topk, cfg.num_experts_per_rank
    )
    assert py_pool_tokens == cpp_pool_tokens
    assert py_pool_tokens // _K_MIN_CANDIDATE_BLOCK_M == cpp_pool_blocks


def _check_full_layout(cfg: ShapeCfg, jit) -> None:
    py_layout = python_symm_buffer_layout(cfg)
    cpp_offsets, cpp_total, cpp_pool, cpp_sf_pool = jit.compute_layout(
        cfg.num_ranks,
        cfg.num_experts,
        cfg.num_max_tokens_per_rank,
        cfg.num_topk,
        cfg.hidden,
        cfg.intermediate_hidden,
    )
    py_offsets = [
        py_layout.workspace_offset,
        py_layout.input_x_offset,
        py_layout.input_x_sf_offset,
        py_layout.input_topk_idx_offset,
        py_layout.input_topk_weights_offset,
        py_layout.l1_pool_x_offset,
        py_layout.l1_pool_x_sf_offset,
        py_layout.l1_pool_weights_offset,
        py_layout.l2_pool_x_offset,
        py_layout.l2_pool_x_sf_offset,
        py_layout.combine_buffer_offset,
    ]
    assert py_offsets == list(
        cpp_offsets
    ), f"Layout offsets mismatch for {cfg}:\n  py = {py_offsets}\n  cpp = {list(cpp_offsets)}"
    assert (
        py_layout.total_bytes == cpp_total
    ), f"total_bytes mismatch for {cfg}: py={py_layout.total_bytes}, cpp={cpp_total}"
    assert py_layout.num_max_pool_tokens == cpp_pool
    assert py_layout.num_padded_sf_pool_tokens == cpp_sf_pool


def _representative_shapes() -> "list[ShapeCfg]":
    return [
        ShapeCfg(384, 256, 128, 8, 2, 1),
        ShapeCfg(768, 1024, 1024, 8, 2, 1),
        ShapeCfg(768, 2048, 2048, 16, 4, 2),
        ShapeCfg(1536, 4096, 1024, 32, 4, 4),
    ]


def run_parity_checks() -> int:
    jit = load_mega_moe_jit()
    shapes = _representative_shapes()
    print(f"[mega_moe-jit] running layout parity over {len(shapes)} shapes", flush=True)
    failures = 0
    for cfg in shapes:
        try:
            _check_pool_token_helpers(cfg, jit)
            _check_workspace_bytes(cfg, jit)
            _check_full_layout(cfg, jit)
            print(f"  PASS  {cfg}", flush=True)
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {cfg}\n    {exc}", flush=True)

    run_status = jit.run_stub()
    if run_status == 0:
        msg = "OK: kernel launched + synchronized successfully"
    else:
        msg = f"FAIL: hipError_t = {run_status}"
        failures += 1
    print(f"[mega_moe-jit] device kernel status = {run_status} ({msg})", flush=True)
    return failures


# ---------------------------------------------------------------------------
# DG-aligned FP8 cast (kept for input quantization)
# ---------------------------------------------------------------------------


def _ceil_to_ue8m0(x):
    import torch

    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _pack_ue8m0_to_int(x):
    import torch

    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(x, use_ue8m0: bool, gran_k: int = 128, use_packed_ue8m0: bool = False):
    """Port of DG's ``per_token_cast_to_fp8``."""
    import torch

    assert x.dim() == 2
    m, n = x.shape
    padded_n = _align_up(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, _pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


# ---------------------------------------------------------------------------
# Single-rank init helper (the kernel template static_asserts kNumRanks==1)
# ---------------------------------------------------------------------------


def init_single_rank(local_rank: int = 0):
    import torch

    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)


# ---------------------------------------------------------------------------
# Symmetric buffer (single-rank only, since the JIT template only
# instantiates kNumRanks==1 today).
# ---------------------------------------------------------------------------


class MegaMoEBuffer:
    """Single-rank symmetric buffer that mirrors the DG-aligned layout."""

    def __init__(
        self,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_ranks: int = 1,
        rank_idx: int = 0,
    ):
        import torch

        assert num_ranks == 1, "numerical verifier only supports kNumRanks==1"
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = _align_up(num_max_tokens_per_rank, _K_TOKEN_ALIGNMENT)
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ranks = num_ranks
        self.rank_idx = rank_idx
        self.num_experts_per_rank = num_experts // num_ranks

        jit = load_mega_moe_jit()
        offsets, total_bytes, num_max_pool_tokens, num_padded_sf_pool_tokens = jit.compute_layout(
            num_ranks, num_experts, self.num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
        )
        self.offsets = list(offsets)
        self.total_bytes = int(total_bytes)
        self.num_max_pool_tokens = int(num_max_pool_tokens)
        self.num_padded_sf_pool_tokens = int(num_padded_sf_pool_tokens)

        self.buffer = torch.empty(self.total_bytes, dtype=torch.int8, device="cuda")
        self.buffer.zero_()
        torch.cuda.synchronize()
        self.sym_buffer_ptrs = [int(self.buffer.data_ptr())]

        (
            workspace_off,
            input_x_off,
            input_x_sf_off,
            input_topk_idx_off,
            input_topk_weights_off,
            l1_pool_x_off,
            l1_pool_x_sf_off,
            l1_pool_weights_off,
            l2_pool_x_off,
            l2_pool_x_sf_off,
            combine_buffer_off,
        ) = self.offsets

        M = self.num_max_tokens_per_rank
        fp8_sf_n = hidden // _K_SCALE_GROUP_K
        # Input views (host writes these before kernel launch).
        self.x = self._slice_view(input_x_off, (M, hidden), torch.float8_e4m3fn)
        self.x_sf = self._slice_view(input_x_sf_off, (M, fp8_sf_n // 4), torch.int32)
        self.topk_idx = self._slice_view(input_topk_idx_off, (M, num_topk), torch.int64)
        self.topk_weights = self._slice_view(input_topk_weights_off, (M, num_topk), torch.float32)

        # L1 pool views (kernel writes these; host reads after sync).
        self.l1_pool_x = self._slice_view(l1_pool_x_off, (self.num_max_pool_tokens, hidden), torch.uint8)
        # SF is transposed: kernel writes uint32 stride such that
        # l1_pool_sf_u32[j, sf_pool_token_idx] = input_x_sf_u32[src, j],
        # with j in [0, hidden/128).
        self.num_sf_uint32_per_token = hidden // 128
        self.l1_pool_sf_u32 = self._slice_view(
            l1_pool_x_sf_off,
            (self.num_sf_uint32_per_token, self.num_padded_sf_pool_tokens),
            torch.int32,
        )
        self.l1_pool_weights = self._slice_view(
            l1_pool_weights_off, (self.num_max_pool_tokens,), torch.float32
        )

        # L2 pool SF view + host-side pre-fill (TODO §B.2 workaround).
        #
        # The kernel's Linear1 writeback in
        # ``gfx950_fp8_fp4_mega_moe.cuh`` writes the per-element FP8 byte
        # to the L2 pool but does NOT write the paired UE8M0 scale into
        # ``l2_sf_buffer`` (see the "Real UE8M0 per-block scale (vs the
        # fixed 1.0 placeholder)" comment in the loader-role writeback).
        # Without this write the L2 SFA dword read by Linear2 stays at
        # 0x00000000 (post-zero-init), which the MFMA expands as
        # ``0x7f7f7f00`` (low byte == 0) → 2^(-127) ≈ subnormal flush on
        # half the K lanes and 1.0 on the other half, halving Linear2.
        # Pre-filling the L2 SF region with 0x7F bytes makes every read
        # return UE8M0 1.0 regardless of what (if anything) the kernel
        # writes there, matching the Python reference's SF=1.0
        # assumption.
        self.num_sf_uint32_per_l2_pool_token = max(intermediate_hidden // 128, 1)
        l2_sf_nbytes = self.num_sf_uint32_per_l2_pool_token * self.num_padded_sf_pool_tokens * 4
        self.l2_pool_sf = self.buffer.narrow(0, l2_pool_x_sf_off, l2_sf_nbytes).view(torch.uint8)
        self.l2_pool_sf.fill_(0x7F)
        torch.cuda.synchronize()

        # Workspace inner views.
        inner = python_workspace_inner_layout(num_ranks, num_experts, self.num_max_tokens_per_rank, num_topk)
        num_max_recv = num_ranks * self.num_max_tokens_per_rank
        self.src_token_topk_idx = self._slice_view(
            workspace_off + inner.src_token_topk_idx_offset,
            (self.num_experts_per_rank, num_ranks, num_max_recv),
            torch.int32,
        )
        # token_src_metadata: 3 × uint32 per pool token (rank, token, topk).
        self.token_src_metadata = self._slice_view(
            workspace_off + inner.token_src_metadata_offset,
            (self.num_max_pool_tokens, 3),
            torch.int32,
        )

    def _slice_view(self, offset: int, shape, dtype):
        import torch

        elem_size = torch.empty((), dtype=dtype).element_size()
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = nelem * elem_size
        byte_view = self.buffer.narrow(0, offset, nbytes)
        return byte_view.view(dtype).view(*shape)

    def destroy(self):
        self.buffer = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_pool_x = None
        self.l1_pool_sf_u32 = None
        self.l1_pool_weights = None
        self.l2_pool_sf = None
        self.src_token_topk_idx = None
        self.token_src_metadata = None
        self.sym_buffer_ptrs = None


# ---------------------------------------------------------------------------
# Dispatch reference + verifier
# ---------------------------------------------------------------------------


def _transform_sf_token_idx(
    t: int, block_m: int = _K_LOADER_BLOCK_M, sf_block_m: int = _K_LOADER_SF_BLOCK_M
) -> int:
    """Python port of the lambda in ``gfx950_fp8_fp4_mega_moe.cuh``."""
    idx = t % block_m
    return (t // block_m) * sf_block_m + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)


def verify_dispatch(
    buffer: MegaMoEBuffer,
    x_fp8,
    x_sf_u32,
    topk_idx,
    topk_weights,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_ranks: int,
    rank_idx: int,
    block_m: int = _K_LOADER_BLOCK_M,
) -> None:
    """Byte-exact verification of the dispatch path's L1 pool writes.

    For each local expert ``e``, the kernel:
      * Populated ``src_token_topk_idx[e, rank, :count_e]`` with the
        flat ``(token, topk)`` indices that routed to ``e`` from ``rank``.
        Slot order is non-deterministic (Pass 3 uses ``atomic_add_block``),
        so we compare as a *set*.
      * Filled pool slots ``[pool_block_offset[e]*BLOCK_M,
        pool_block_offset[e]*BLOCK_M + count_e)`` with copies of the
        source tokens' FP8 bytes, packed UE8M0 SF (transposed), and
        topk weights.  We use ``token_src_metadata`` (also written by
        Pass 4) to recover the per-slot ``(src_token, src_topk)`` and
        check each piece byte-for-byte.
    """
    import torch

    # Local copies on CPU for set/dict operations.
    topk_idx_cpu = topk_idx.cpu().numpy()  # (num_tokens, num_topk)
    topk_weights_cpu = topk_weights.cpu().numpy()  # (num_tokens, num_topk)
    x_fp8_bytes_cpu = x_fp8.view(torch.uint8).cpu().numpy()  # (num_tokens, hidden)
    x_sf_u32_cpu = x_sf_u32.cpu().numpy()  # (num_tokens, num_sf_uint32)

    # ---- Per-expert: expected set of (token, topk) and count.
    expected_per_expert: "list[list[tuple[int, int]]]" = [[] for _ in range(num_experts_per_rank)]
    rank_offset = rank_idx * num_experts_per_rank
    for t in range(num_tokens):
        for k in range(num_topk):
            e = int(topk_idx_cpu[t, k])
            if e < 0:
                continue
            if rank_offset <= e < rank_offset + num_experts_per_rank:
                expected_per_expert[e - rank_offset].append((t, k))

    # ---- Pool block offsets per expert (loader's running prefix sum).
    pool_block_offsets = [0]
    for e in range(num_experts_per_rank):
        cnt = len(expected_per_expert[e])
        pool_block_offsets.append(pool_block_offsets[-1] + (cnt + block_m - 1) // block_m)

    # ---- 1) src_token_topk_idx set check.
    src_idx_cpu = buffer.src_token_topk_idx.cpu().numpy()  # (E, R, num_max_recv)
    for e in range(num_experts_per_rank):
        expected_flat = {t * num_topk + k for (t, k) in expected_per_expert[e]}
        actual_flat = set()
        for r in range(num_ranks):
            cnt_r = (
                sum(1 for (t, k) in expected_per_expert[e])  # single-rank: all from rank 0
                if r == rank_idx
                else 0
            )
            # For kNumRanks==1, all routed tokens land in src_idx_cpu[e, 0, :count].
            # For multi-rank we'd compare per (e, r); kNumRanks==1 here.
            for s in range(cnt_r):
                actual_flat.add(int(src_idx_cpu[e, r, s]))
        assert expected_flat == actual_flat, (
            f"src_token_topk_idx mismatch for expert {e}:\n"
            f"  expected={sorted(expected_flat)}\n  actual  ={sorted(actual_flat)}"
        )

    # ---- 2) Per-slot byte-exact check via token_src_metadata.
    md_cpu = buffer.token_src_metadata.cpu().numpy()  # (num_max_pool_tokens, 3)
    l1_pool_x_cpu = buffer.l1_pool_x.cpu().numpy()  # (num_max_pool_tokens, hidden)
    l1_pool_sf_u32_cpu = buffer.l1_pool_sf_u32.cpu().numpy()  # (num_sf_u32, num_padded_sf_pool_tokens)
    l1_pool_w_cpu = buffer.l1_pool_weights.cpu().numpy()  # (num_max_pool_tokens,)

    for e in range(num_experts_per_rank):
        base_pool = pool_block_offsets[e] * block_m
        cnt = len(expected_per_expert[e])
        if cnt == 0:
            continue
        for s in range(cnt):
            pool_idx = base_pool + s
            md_rank = int(md_cpu[pool_idx, 0])
            md_token = int(md_cpu[pool_idx, 1])
            md_topk = int(md_cpu[pool_idx, 2])

            # Validate the metadata is consistent.
            assert md_rank == rank_idx, (
                f"expert {e} slot {s} (pool_idx={pool_idx}): "
                f"metadata rank={md_rank} != rank_idx={rank_idx}"
            )
            assert 0 <= md_token < num_tokens, (
                f"expert {e} slot {s} (pool_idx={pool_idx}): "
                f"metadata token={md_token} out of range [0,{num_tokens})"
            )
            assert 0 <= md_topk < num_topk, (
                f"expert {e} slot {s} (pool_idx={pool_idx}): "
                f"metadata topk={md_topk} out of range [0,{num_topk})"
            )
            routed_expert = int(topk_idx_cpu[md_token, md_topk])
            assert routed_expert == e + rank_offset, (
                f"expert {e} slot {s} (pool_idx={pool_idx}): metadata "
                f"({md_token},{md_topk}) routes to expert {routed_expert}, expected {e + rank_offset}"
            )

            # L1 pool x bytes == input x bytes for src_token.
            expected_bytes = x_fp8_bytes_cpu[md_token]
            actual_bytes = l1_pool_x_cpu[pool_idx]
            if not (actual_bytes == expected_bytes).all():
                ne_idx = (actual_bytes != expected_bytes).nonzero()[0][:10]
                raise AssertionError(
                    f"L1 pool x byte mismatch for expert {e} slot {s} (pool_idx={pool_idx}, "
                    f"src_token={md_token}): first diff indices={ne_idx.tolist()}, "
                    f"expected={expected_bytes[ne_idx].tolist()}, actual={actual_bytes[ne_idx].tolist()}"
                )

            # L1 pool SF (transposed, one uint32 per 128 hidden elems).
            sf_pool_idx = pool_block_offsets[e] * _K_LOADER_SF_BLOCK_M + _transform_sf_token_idx(s)
            for j in range(buffer.num_sf_uint32_per_token):
                exp_sf = int(x_sf_u32_cpu[md_token, j])
                act_sf = int(l1_pool_sf_u32_cpu[j, sf_pool_idx])
                assert exp_sf == act_sf, (
                    f"L1 pool SF mismatch for expert {e} slot {s} (pool_idx={pool_idx}, "
                    f"sf_pool_idx={sf_pool_idx}, j={j}, src_token={md_token}): "
                    f"expected=0x{exp_sf & 0xffffffff:08x}, actual=0x{act_sf & 0xffffffff:08x}"
                )

            # L1 pool topk weight.
            exp_w = float(topk_weights_cpu[md_token, md_topk])
            act_w = float(l1_pool_w_cpu[pool_idx])
            assert exp_w == act_w, (
                f"L1 pool weight mismatch for expert {e} slot {s} (pool_idx={pool_idx}, "
                f"src=({md_token},{md_topk})): expected={exp_w}, actual={act_w}"
            )

    total_routed = sum(len(g) for g in expected_per_expert)
    print(
        f"[mega_moe-jit] dispatch verification: PASS "
        f"(experts={num_experts_per_rank}, routed_pairs={total_routed})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# End-to-end fused-MoE reference + verifier (TODO §B.2 numerics gate)
# ---------------------------------------------------------------------------


def _unpack_ue8m0_from_int32(packed_u32, gran_k: int = _K_SCALE_GROUP_K):
    """Unpack the int32-packed UE8M0 scales back to per-(M, K/gran_k) FP32 multipliers.

    ``per_token_cast_to_fp8(..., use_packed_ue8m0=True)`` packs 4 consecutive
    UE8M0 bytes per int32 (one byte = ``2 ** (sf - 127)``).  Returns a float32
    tensor of shape ``(M, K/gran_k)``.
    """
    import torch

    assert packed_u32.dtype == torch.int32
    m, k_div4 = packed_u32.shape
    bytes_u8 = packed_u32.view(torch.uint8).reshape(m, k_div4 * 4)
    exponent = bytes_u8.to(torch.int32) - 127
    return torch.pow(torch.tensor(2.0, device=packed_u32.device), exponent.float())


def _dequant_fp8_with_ue8m0(fp8_u8, sf_factors, gran_k: int = _K_SCALE_GROUP_K):
    """``fp8_u8`` viewed as e4m3fn × per-(K_group) UE8M0 factor → fp32.

    ``fp8_u8``     shape: ``(M, K)`` uint8 (bytes are valid e4m3fn encodings).
    ``sf_factors`` shape: ``(M, K/gran_k)`` float32 multipliers.
    """
    import torch

    fp8_view = fp8_u8.view(torch.float8_e4m3fn)
    fp32 = fp8_view.float()
    m, k = fp32.shape
    sf_broadcast = sf_factors.unsqueeze(-1).expand(m, k // gran_k, gran_k).reshape(m, k)
    return fp32 * sf_broadcast


def reference_y(
    *,
    x_fp8,
    x_sf_packed_u32,
    topk_idx,
    topk_weights,
    l1_w_fp8,
    l2_w_fp8,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts_per_rank: int,
    rank_idx: int,
    gran_k: int = _K_SCALE_GROUP_K,
):
    """End-to-end fused-MoE reference matching the gfx950 kernel's pipeline.

    Per ``(token, k in [0, num_topk))`` the kernel:
      1. Dequantises x with the per-token UE8M0 SF.
      2. Runs Linear1 ``h1 = x @ W1[expert].T``   (W1 SF is fixed 1.0).
      3. SwiGLU: ``h = silu(h1[:I]) * h1[I:2I]``.
      4. Quantises ``h`` to FP8 e4m3 with implicit SF=1.0 (Linear2 SFA).
      5. Runs Linear2 ``y_partial = h_fp8.float() @ W2[expert].T`` (W2 SF=1.0).
      6. Casts ``y_partial`` to BF16, pushes into the combine slot.
      7. Epilogue sums BF16 partials across topk experts (no topk_weight
         multiplication — the kernel mirrors DG and applies the weight
         outside).

    Returns ``y`` of shape ``(num_tokens, hidden)`` in BF16 to match the
    kernel's output tensor dtype.
    """
    import torch

    x_sf_factors = _unpack_ue8m0_from_int32(x_sf_packed_u32, gran_k=gran_k)[:num_tokens, : hidden // gran_k]
    x_deq = _dequant_fp8_with_ue8m0(
        x_fp8.view(torch.uint8)[:num_tokens, :hidden], x_sf_factors, gran_k=gran_k
    )  # (num_tokens, hidden)

    # Weight SFs are fixed 1.0 so we just float-cast the e4m3 bytes.
    l1_w_f32 = l1_w_fp8.view(torch.float8_e4m3fn).float()  # (E, 2I, H)
    l2_w_f32 = l2_w_fp8.view(torch.float8_e4m3fn).float()  # (E, H, I)

    topk_idx_cpu = topk_idx.cpu()
    rank_offset = rank_idx * num_experts_per_rank
    y = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x_fp8.device)

    for t in range(num_tokens):
        x_row = x_deq[t]  # (H,)
        for k in range(num_topk):
            e_global = int(topk_idx_cpu[t, k])
            if e_global < 0:
                continue
            if not (rank_offset <= e_global < rank_offset + num_experts_per_rank):
                continue
            e_local = e_global - rank_offset
            W1 = l1_w_f32[e_local]  # (2I, H)
            h1 = torch.matmul(W1, x_row)  # (2I,)
            gate = h1[:intermediate_hidden]
            up = h1[intermediate_hidden:]
            silu_gate = gate / (1.0 + torch.exp(-gate))
            h = silu_gate * up  # (I,)

            # Kernel quantises ``h`` to e4m3 (single-byte, scale-1.0) for
            # the L2 input; mirror that rounding here.
            h_fp8 = h.to(torch.float8_e4m3fn).float()

            W2 = l2_w_f32[e_local]  # (H, I)
            y_partial = torch.matmul(W2, h_fp8)  # (H,)

            # Cast each topk partial to BF16 before summing (mirrors the
            # combine buffer's BF16 storage); then accumulate in FP32 like
            # the epilogue's ``reduced[l].x += fp32.x``.
            y[t] += y_partial.bfloat16().float()

    return y.bfloat16()


# Gate 3 thresholds — FP8 GEMM numerics standard (see CLAUDE.md gate 3).
# Production-grade FP8 fused-MoE precision: tight enough to catch structural
# bugs in the loader / SwiGLU / writeback / combine paths, loose enough to
# tolerate FP8 quantisation + BF16 round-off noise in the topk reduction.
_GATE_3_COS_SIM_MIN = 0.99
_GATE_3_REL_RMSE_MAX = 0.05


def verify_y(
    y_kernel,
    y_ref,
    *,
    cos_sim_min: float = _GATE_3_COS_SIM_MIN,
    rel_rmse_max: float = _GATE_3_REL_RMSE_MAX,
    name: str = "y",
) -> bool:
    """Gate 3 — kernel-vs-reference numerics check (hard pass/fail).

    Returns ``True`` if both ``cos_sim >= cos_sim_min`` and
    ``rel_rmse <= rel_rmse_max``; otherwise prints a FAIL line and returns
    ``False`` (the caller is expected to ``raise SystemExit(1)`` so the
    process exit code propagates the gate failure to CI / shells).

    Always reports max|diff|, RMSE, rel_RMSE, and cosine similarity so
    that future bisection work can watch each statistic move
    independently as kernel fixes land.
    """
    import torch

    y_k = y_kernel.detach().float().contiguous()
    y_r = y_ref.detach().float().contiguous()
    diff = (y_k - y_r).abs()
    max_abs = diff.max().item()
    rmse = diff.pow(2).mean().sqrt().item()
    denom = max(y_r.abs().mean().item(), 1e-8)
    rel_rmse = rmse / denom
    cos = torch.nn.functional.cosine_similarity(y_k.reshape(1, -1), y_r.reshape(1, -1)).item()
    print(
        f"[mega_moe-jit] {name} numerics report: "
        f"max|diff|={max_abs:.4f}  rmse={rmse:.4f}  rel_rmse={rel_rmse:.4f}  "
        f"cos_sim={cos:.4f}  (ref|y|_mean={denom:.4f})",
        flush=True,
    )
    passed = (cos >= cos_sim_min) and (rel_rmse <= rel_rmse_max)
    if passed:
        print(
            f"[mega_moe-jit] gate 3 ({name}) PASS "
            f"(cos_sim>={cos_sim_min:.4f} & rel_rmse<={rel_rmse_max:.4f})",
            flush=True,
        )
    else:
        print(
            f"[mega_moe-jit] gate 3 ({name}) FAIL "
            f"(need cos_sim>={cos_sim_min:.4f} & rel_rmse<={rel_rmse_max:.4f}; "
            f"got cos_sim={cos:.4f}, rel_rmse={rel_rmse:.4f}). "
            f"See TODO §B.2 + memory project_megamoe_y_numerics_uncorrelated.md "
            f"for the open structural GEMM-body bug.",
            flush=True,
        )
    return passed


# ---------------------------------------------------------------------------
# Test entry
# ---------------------------------------------------------------------------


def test(args: argparse.Namespace) -> None:
    import torch

    init_single_rank(0)
    rank_idx = 0
    num_ranks = 1
    torch.manual_seed(0)
    random.seed(0)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    if args.num_tokens is not None:
        num_tokens = args.num_tokens
    else:
        num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens))
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    jit_shape = {
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "hidden": hidden,
        "intermediate_hidden": intermediate_hidden,
        "num_experts": num_experts,
        "num_topk": num_topk,
        "num_ranks": num_ranks,
    }
    load_mega_moe_jit(jit_shape)

    if not args.skip_parity:
        if run_parity_checks() != 0:
            raise SystemExit(1)

    buffer = MegaMoEBuffer(
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_ranks=num_ranks,
        rank_idx=rank_idx,
    )

    print("Config:", flush=True)
    print(f" > Tokens: {num_tokens}/{num_max_tokens_per_rank}", flush=True)
    print(f" > Hidden: {hidden}", flush=True)
    print(f" > Intermediate: {intermediate_hidden}", flush=True)
    print(f" > Experts: {num_topk}/{num_experts} ({num_experts_per_rank}/rank)", flush=True)
    print(f" > Buffer: {buffer.total_bytes / 2 ** 30:.4f} GiB", flush=True)
    print(f" > Ranks: {rank_idx}/{num_ranks}", flush=True)
    print(flush=True)

    jit = load_mega_moe_jit()

    # Probe: zero-token kernel call to confirm the shape matches the
    # JIT-instantiated smoke template.
    probe_y = torch.empty((1, hidden), dtype=torch.bfloat16, device="cuda")
    probe_aux = torch.zeros(1, dtype=torch.uint8, device="cuda")
    probe_status = jit.run_mega_moe(
        buffer.sym_buffer_ptrs,
        rank_idx,
        0,
        buffer.num_max_tokens_per_rank,
        hidden,
        intermediate_hidden,
        num_experts,
        num_topk,
        num_ranks,
        probe_y,
        probe_aux,
        probe_aux,
        probe_aux,
        probe_aux,
        None,
        args.activation_clamp,
        bool(args.fast_math),
    )
    del probe_y, probe_aux
    if probe_status != 0:
        print(
            f" > Launch status: FAIL hipError_t={probe_status} (probe at num_tokens=0; "
            f"shape may not match the JIT-instantiated smoke template)",
            flush=True,
        )
        buffer.destroy()
        raise SystemExit(1)

    # ---- Build inputs.
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_fp8, x_sf_u32 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # Weight + SF buffers in *kernel-readable* layouts.  Shapes/dtypes
    # mirror the kernel's reads:
    #   L1 weights:  (E_per_rank, 2*intermediate, hidden)        FP8 bytes (uint8 view of e4m3fn)
    #   L1 SF:       (E_per_rank, 2*intermediate, hidden / 32)   uint8 (E8M0)
    #   L2 weights:  (E_per_rank, hidden, intermediate)          FP8 bytes (uint8 view of e4m3fn)
    #   L2 SF:       (E_per_rank, hidden, intermediate / 32)     uint8 (E8M0)
    #
    # All weight SFs are set to 0x7F (UE8M0 1.0) so the Python reference
    # only has to dequant the FP8 byte itself; the kernel's per-tile
    # ``sfb_byte`` reads land on 0x7F → factor 1.0.  Weights are small
    # bf16 randn values quantised to e4m3fn so every byte is a valid
    # finite e4m3 (avoids NaN encodings 0x7F / 0xFF in the raw uint8
    # path).
    def _rand_fp8_weights(shape):
        bf = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.1
        return bf.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()

    l1_w_fp8 = _rand_fp8_weights((num_experts_per_rank, 2 * intermediate_hidden, hidden))
    l1_w_sf = torch.full(
        (num_experts_per_rank, 2 * intermediate_hidden, hidden // _K_SCALE_GROUP_K),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    )
    l2_w_fp8 = _rand_fp8_weights((num_experts_per_rank, hidden, intermediate_hidden))
    l2_w_sf = torch.full(
        (num_experts_per_rank, hidden, intermediate_hidden // _K_SCALE_GROUP_K),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    )
    recv_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")

    # ---- Copy inputs into the symmetric buffer.
    if num_tokens > 0:
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens, : x_sf_u32.shape[1]].copy_(x_sf_u32)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
    torch.cuda.synchronize()

    # ---- Launch.
    print("Running fused kernel:", flush=True)
    y = torch.empty((max(num_tokens, 1), hidden), dtype=torch.bfloat16, device="cuda")

    def _launch():
        return jit.run_mega_moe(
            buffer.sym_buffer_ptrs,
            rank_idx,
            num_tokens,
            buffer.num_max_tokens_per_rank,
            hidden,
            intermediate_hidden,
            num_experts,
            num_topk,
            num_ranks,
            y,
            l1_w_fp8,
            l1_w_sf,
            l2_w_fp8,
            l2_w_sf,
            recv_stats,
            args.activation_clamp,
            bool(args.fast_math),
        )

    status = _launch()
    torch.cuda.synchronize()
    if status == 0:
        print(" > Launch status: OK", flush=True)
    else:
        print(f" > Launch status: FAIL hipError_t={status}", flush=True)
        buffer.destroy()
        raise SystemExit(1)

    # ---- Timing: warmup + N timed iterations via HIP events.
    num_warmup = max(int(getattr(args, "num_warmup", 5)), 0)
    num_iters = max(int(getattr(args, "num_iters", 20)), 1)
    for _ in range(num_warmup):
        _launch()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(num_iters):
        _launch()
    end_evt.record()
    torch.cuda.synchronize()
    per_launch_ms = start_evt.elapsed_time(end_evt) / num_iters
    per_launch_us = per_launch_ms * 1000.0
    # Fused-MoE FLOPs (proxy): 2 GEMMs per (token,topk) → L1 + L2.
    #   L1: y_l1 = x @ W1.T          [N=intermediate_hidden, K=hidden]
    #   L2: y    = act(y_l1) @ W2.T  [N=hidden,              K=intermediate_hidden]
    # FLOPs per (token,topk) pair = 2 * (N1*K1 + N2*K2) = 4 * hidden * intermediate_hidden.
    flops = 4.0 * num_tokens * num_topk * hidden * intermediate_hidden
    tflops = flops / (per_launch_ms * 1e-3) / 1e12 if per_launch_ms > 0 else 0.0

    # ---- Dispatch numerical verification (only meaningful for num_tokens > 0).
    if num_tokens > 0:
        verify_dispatch(
            buffer=buffer,
            x_fp8=x_fp8,
            x_sf_u32=x_sf_u32,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
            num_topk=num_topk,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_ranks=num_ranks,
            rank_idx=rank_idx,
        )

        # Gate 3 — end-to-end fused MoE numerics (kernel y vs Python ref).
        # Hard gate: cos_sim >= 0.99 AND rel_rmse <= 0.05.  See CLAUDE.md
        # gate 3 + ``verify_y`` thresholds (``_GATE_3_*``).  A FAIL here is
        # NOT bypassed by ``--skip-parity`` or any other flag — the test
        # exits non-zero so CI / shells catch a regression in the GEMM /
        # SwiGLU / writeback / combine paths.
        y_ref = reference_y(
            x_fp8=x_fp8,
            x_sf_packed_u32=x_sf_u32,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            l1_w_fp8=l1_w_fp8,
            l2_w_fp8=l2_w_fp8,
            num_tokens=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts_per_rank=num_experts_per_rank,
            rank_idx=rank_idx,
        )
        _GATE_3_PASSED = verify_y(y[:num_tokens], y_ref, name="y")
    else:
        print(
            "[mega_moe-jit] dispatch verification: SKIPPED (num_tokens=0 early-exit path)",
            flush=True,
        )
        _GATE_3_PASSED = True  # Gate 3 is N/A on the early-exit path.

    print("Performance:", flush=True)
    print(
        f" > EP: {rank_idx:2}/{num_ranks} | num_tokens={num_tokens} | "
        f"avg latency = {per_launch_us:.2f} us over {num_iters} iters "
        f"(warmup {num_warmup}) | FLOPs proxy = {tflops:.2f} TFLOPS "
        f"(GEMM body still scaffolding, TODO §B.2)",
        flush=True,
    )
    print(
        " > dispatch verified byte-exactly; gate 3 (kernel y numerics) " "result printed above",
        flush=True,
    )

    buffer.destroy()
    if not _GATE_3_PASSED:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Numerical-verification test for the gfx950 mega-MoE JIT kernel"
    )

    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="Only run layout-parity + zero-token kernel smoke",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip layout-parity sanity check before the kernel test",
    )

    # Shape knobs.  Defaults match the JIT smoke template so the
    # ``--parity-only`` and default invocations both exercise the
    # pre-instantiated kernel without re-JIT.
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=384)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=128,
        help="Tokens per rank (0 = early-exit smoke path; default 128)",
    )
    parser.add_argument("--num-max-removed-tokens", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate-hidden", type=int, default=128)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-topk", type=int, default=2)
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--fast-math", type=int, default=1)

    # Compatibility no-ops (kept so existing CI invocations don't break).
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--ep-ranks", type=int, default=None)
    parser.add_argument("--num-warmup-iters", type=int, default=0)
    parser.add_argument("--num-perf-iters", type=int, default=0)
    parser.add_argument("--ncu-profile-only", action="store_true")
    parser.add_argument("--dump-profile-traces", type=str, default="")
    parser.add_argument("--local-rank-idx", type=int, default=None)
    return parser


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.num_processes > 1:
        print(
            "[mega_moe-jit] --num-processes > 1 is not supported by the numerical verifier; "
            "the JIT smoke template instantiates kNumRanks==1 only.",
            flush=True,
        )
        return 1

    if args.parity_only:
        # Parity-only path keeps the same JIT cache key as the default
        # test() path: build with the requested shape so the run_stub
        # call inside run_parity_checks hits the same instantiation.
        jit_shape = {
            "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
            "hidden": args.hidden,
            "intermediate_hidden": args.intermediate_hidden,
            "num_experts": args.num_experts,
            "num_topk": args.num_topk,
            "num_ranks": 1,
        }
        load_mega_moe_jit(jit_shape)
        return 1 if run_parity_checks() != 0 else 0

    test(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
