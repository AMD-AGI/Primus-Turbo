###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Layout-parity sanity test for the DG-aligned mega-MoE kernel-level
C++ headers.

The previous version of this test JIT-compiled the Round 0 chain
(dispatch -> L1 GEMM -> SwiGLU -> L2 GEMM -> combine), and the Round 3
fused / Round 4 per-wave persistent kernels.  Those device kernels have
been removed in favor of a single DG-aligned launcher whose body is
not yet implemented (the alignment pass only brings the kernel-level
C++ API surface across).  As a result, this test now exercises the
**layout / pool-token math** instead of the device kernel itself:

  * Loads the JIT TU under ``mega_moe_jit_launch.cu`` /
    ``mega_moe_jit_binding.cpp``.  These TUs include the new headers
      kernels/mega_moe/layout/mega_moe.cuh
      kernels/mega_moe/layout/sym_buffer.cuh
      kernels/mega_moe/scheduler/mega_moe.cuh
      kernels/mega_moe/impls/gfx950_fp8_fp4_mega_moe.cuh
    so loading the module confirms the new C++ API surface compiles
    cleanly under hipcc with the gfx950 offload arch.
  * Cross-checks ``layout::get_num_max_pool_tokens`` and
    ``layout::get_num_padded_sf_pool_tokens`` against pure-Python
    references for a representative set of shapes.
  * Cross-checks the full 11-region symmetric-buffer layout offsets
    against the Python port of
    ``get_symm_buffer_size_for_mega_moe`` byte-for-byte.

When the DG-aligned device kernel lands, this test will be extended to
cover correctness and performance on top of the layout parity check.

Usage:
    python tests/pytorch/ops/jit/test_mega_moe_jit_perf.py
"""
from __future__ import annotations

import argparse
import os
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
# Shape config
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
# Python reference for the DG-aligned layout helpers.  Naming + formulas
# mirror ``deep_gemm::layout`` / ``primus_turbo::mega_moe::layout`` so the
# parity check exercises the same code path the device kernel will see.
# ---------------------------------------------------------------------------

# Mirrors ``primus_turbo::mega_moe::kCandidateBlockM``.
_K_CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192)
_K_MAX_CANDIDATE_BLOCK_M = max(_K_CANDIDATE_BLOCK_M)
_K_MIN_CANDIDATE_BLOCK_M = min(_K_CANDIDATE_BLOCK_M)
_K_TOKEN_ALIGNMENT = 384  # == LCM(_K_CANDIDATE_BLOCK_M); kLCMCandidateBlockM in DG.
_K_SCALE_GROUP_K = 32
_K_SCALE_BLOCK_MN = 128


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

    # kNumBarrierSignalBytes
    num_bytes = 32
    # Expert send/recv counts: ``num_experts`` × 2 × sizeof(uint64_t)
    num_bytes += num_experts * 8 * 2
    # Per-local-expert recv count sum.
    num_bytes += num_experts_per_rank * 8
    # L1 arrival count (padded to even entries for ``uint64_t`` alignment).
    num_bytes += _align_up(num_max_pool_blocks, 2) * 4
    # L2 block arrival mask.
    num_bytes += num_max_pool_blocks * 8
    # Dispatch pulling source ``(token, topk)`` indices.
    num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * 4
    # Combine push source indices (TokenSrcMetadata: 3 × uint32_t = 12 B).
    num_bytes += num_max_pool_tokens * 12
    # Round up to TMA descriptor alignment.
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
# JIT loader (cached across all ``ShapeCfg`` invocations — the TU is
# shape-agnostic and parameterizes everything at runtime via the
# extern "C" entry points).
# ---------------------------------------------------------------------------


_JIT_MODULE = None


def load_mega_moe_jit():
    """JIT-loads the layout-helper TU.  Returns the loaded pybind module."""
    global _JIT_MODULE
    if _JIT_MODULE is not None:
        return _JIT_MODULE

    # Importing torch inside the loader keeps a pure ``python -c`` import
    # of this file (e.g. for a quick lint pass) lightweight.
    from torch.utils.cpp_extension import load as cpp_load

    sources = [
        os.path.join(THIS_DIR, "mega_moe_jit_binding.cpp"),
        os.path.join(THIS_DIR, "mega_moe_jit_launch.cu"),
    ]
    # ``primus_turbo/float8.h`` relies on ``half<->float`` implicit
    # conversions; PyTorch's cpp_extension defines
    # ``__HIP_NO_HALF_CONVERSIONS__`` / ``__HIP_NO_HALF_OPERATORS__`` by
    # default which break those.  Undefine them in the JIT TU.
    half_undefs = [
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
    ]
    _JIT_MODULE = cpp_load(
        name="mega_moe_jit_layout_probe",
        sources=sources,
        # Include both ``csrc/include`` (for ``primus_turbo/mega_moe.h``)
        # and ``csrc`` (so headers can use ``kernels/mega_moe/layout/...``
        # relative paths that mirror the on-disk DG-aligned structure).
        extra_include_paths=[CSRC_INCLUDE, CSRC_ROOT],
        extra_cflags=["-O3", "-std=c++20"] + half_undefs,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--offload-arch=gfx950",
            "-fno-gpu-rdc",
        ]
        + half_undefs,
        with_cuda=True,
        verbose=True,
    )
    return _JIT_MODULE


# ---------------------------------------------------------------------------
# Parity checks
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
        assert py_sf == cpp_sf, (
            f"get_num_padded_sf_pool_tokens mismatch for {cfg}, block_m={bm}: " f"py={py_sf}, cpp={cpp_sf}"
        )


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


def _representative_shapes() -> list[ShapeCfg]:
    return [
        ShapeCfg(
            num_max_tokens_per_rank=384,
            hidden=256,
            intermediate_hidden=128,
            num_experts=8,
            num_topk=2,
            num_ranks=1,
        ),
        ShapeCfg(
            num_max_tokens_per_rank=768,
            hidden=1024,
            intermediate_hidden=1024,
            num_experts=8,
            num_topk=2,
            num_ranks=1,
        ),
        ShapeCfg(
            num_max_tokens_per_rank=768,
            hidden=2048,
            intermediate_hidden=2048,
            num_experts=16,
            num_topk=4,
            num_ranks=2,
        ),
        ShapeCfg(
            num_max_tokens_per_rank=1536,
            hidden=4096,
            intermediate_hidden=1024,
            num_experts=32,
            num_topk=4,
            num_ranks=4,
        ),
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

    # Sanity: confirm the host launcher template parses under hipcc.  The
    # stub returns 1 to signal "device kernel body not yet implemented".
    run_status = jit.run_stub()
    print(
        f"[mega_moe-jit] device kernel status = {run_status} "
        f"({'TBD: not implemented yet' if run_status == 1 else 'unexpected'})",
        flush=True,
    )
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    failures = run_parity_checks()
    if failures:
        print(f"[summary] {failures} shape(s) failed parity check", flush=True)
        return 1
    print("[summary] all shapes PASS layout parity", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
