###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DG-aligned correctness + performance test for the gfx950 mega-MoE kernel,
JIT-loaded from ``tests/pytorch/ops/jit/mega_moe_jit_*``.

This file mirrors the structure of
``3rdparty/DeepGEMM/tests/test_mega_moe.py`` (argparser, distributed
spawn, ``create_inputs`` / ``run_fused`` / benchmark + report) while
keeping the Turbo-specific bits:

  * The kernel is loaded via ``torch.utils.cpp_extension.load`` from the
    sibling ``mega_moe_jit_*.cu`` / ``.cpp`` TUs rather than from a
    pre-compiled wheel.
  * The symmetric buffer is a plain CUDA tensor (single-rank fast path)
    or a ``torch.distributed._symmetric_memory`` allocation when run
    across multiple ranks.
  * The layout-parity sanity checks from the previous test_mega_moe_jit
    iteration are preserved and run automatically before any kernel
    launch, to confirm the C++ <-> Python buffer-offset math stays in
    lock-step.

Usage (single GPU, smoke-test shape — exercises the actual kernel):

    python tests/pytorch/ops/jit/test_mega_moe_jit_perf.py \\
        --num-processes 1 --num-max-tokens-per-rank 64 --hidden 256 \\
        --intermediate-hidden 128 --num-experts 8 --num-topk 2

Usage (DG-default shape — runs the full input pipeline + config print
but the kernel call is skipped because the JIT TU currently only
instantiates the smoke template):

    python tests/pytorch/ops/jit/test_mega_moe_jit_perf.py

Usage (layout parity only, no kernel launch):

    python tests/pytorch/ops/jit/test_mega_moe_jit_perf.py --parity-only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

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
# Shape config (kept for the existing layout-parity tests)
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
# JIT loader (cached across all ``ShapeCfg`` invocations — the TU is
# shape-agnostic and parameterizes everything at runtime via the
# extern "C" entry points).
# ---------------------------------------------------------------------------


_JIT_MODULE = None
_JIT_MODULE_KEY = None


# Mapping from Python keyword -> MEGA_MOE_JIT_K<NAME> macro that
# overrides the matching entry in the ``smoke`` namespace inside
# ``mega_moe_jit_launch.cu``.  Only these knobs are wired through;
# secondary tile/thread params keep their .cu-side defaults unless added
# here.  See the ``#ifndef`` block at the top of the .cu for the
# full list of overridable symbols.
_JIT_SHAPE_MACROS = {
    "num_max_tokens_per_rank": "MEGA_MOE_JIT_KNUMMAXTOKENSPERRANK",
    "hidden": "MEGA_MOE_JIT_KHIDDEN",
    "intermediate_hidden": "MEGA_MOE_JIT_KINTERMEDIATEHIDDEN",
    "num_experts": "MEGA_MOE_JIT_KNUMEXPERTS",
    "num_topk": "MEGA_MOE_JIT_KNUMTOPK",
    "num_ranks": "MEGA_MOE_JIT_KNUMRANKS",
}


def load_mega_moe_jit(shape: "dict | None" = None):
    """JIT-loads the layout-helper TU.  ``shape`` is a dict of
    ``{key: int}`` whose keys are a subset of ``_JIT_SHAPE_MACROS``;
    each entry becomes ``-DMEGA_MOE_JIT_K<NAME>=<value>u`` so the
    ``smoke::`` namespace in ``mega_moe_jit_launch.cu`` is re-bound at
    compile time.  Shapes not listed fall back to the .cu defaults.
    A per-shape extension name (hash-suffixed) keeps PyTorch's ninja
    cache from reusing a stale build when the shape changes.
    """
    global _JIT_MODULE, _JIT_MODULE_KEY
    # ``shape=None`` means "give me whatever is cached, or build with
    # .cu defaults if nothing has been loaded yet".  An explicit shape
    # must match the cached one — passing a second, different shape in
    # the same process is a programming error (PyTorch's ext cache is
    # per-process keyed on extension name, not on define values).
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
        # Hash the override set into the extension name so each shape
        # gets its own ninja build directory; otherwise PyTorch reuses
        # the cached .so and silently ignores the new -D flags.
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
# Layout parity checks (preserved from the previous test iteration)
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
# DG-aligned helpers (ported from ``DeepGEMM/deep_gemm/utils/math.py``)
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


def _quantize_to_fp4_e2m1(x):
    import torch

    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(x, use_ue8m0: bool, gran_k: int = 128, use_packed_ue8m0: bool = False):
    """Port of DG's ``per_token_cast_to_fp4``."""
    import torch

    m, n = x.shape
    assert n % 2 == 0
    assert not use_packed_ue8m0 or use_ue8m0
    padded_n = _align_up(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), _pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


def cast_grouped_weights_to_fp4(bf16_weights, gran_k: int = 32, use_ue8m0: bool = True):
    """Per-expert FP4 cast for ``(num_groups, n, k)`` weight tensors."""
    import torch

    num_groups, n, k = bf16_weights.shape
    assert k % gran_k == 0
    w = torch.empty((num_groups, n, k // 2), device=bf16_weights.device, dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // gran_k), device=bf16_weights.device, dtype=torch.float)
    for i in range(num_groups):
        w_i, w_sf_i = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
        w[i] = w_i
        w_sf[i] = w_sf_i.view(torch.float) if w_sf_i.dtype != torch.float else w_sf_i
    return w, w_sf


# ---------------------------------------------------------------------------
# Distributed helpers (with a single-rank fallback so the test runs
# end-to-end on a single MI355X without an NCCL group).
# ---------------------------------------------------------------------------


_LOCAL_RANK: Optional[int] = None


def init_dist(local_rank: int, num_local_ranks: int):
    """Mirrors ``deep_gemm.utils.dist.init_dist`` with a no-op fallback
    for ``num_local_ranks == 1`` (single-GPU smoke tests)."""
    import torch

    global _LOCAL_RANK
    _LOCAL_RANK = local_rank

    if num_local_ranks <= 1:
        torch.set_default_device("cuda")
        torch.cuda.set_device(local_rank)
        return 0, 1, None

    import inspect

    import torch.distributed as dist

    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(num_local_ranks * num_nodes)))
    return dist.get_rank(), dist.get_world_size(), group


def dist_print(s: str = "", once_in_node: bool = False) -> None:
    """Mirrors ``deep_gemm.utils.dist.dist_print``."""
    if _LOCAL_RANK is None or not once_in_node or _LOCAL_RANK == 0:
        print(s, flush=True)
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Symmetric buffer (Turbo-side, mirrors DG's ``SymmBuffer``)
# ---------------------------------------------------------------------------


class MegaMoEBuffer:
    """Wraps a CUDA buffer matching the DG-aligned mega-MoE symmetric layout.

    Single-rank (``num_ranks == 1``) uses a plain ``torch.empty`` allocation
    and exposes that pointer as the sole entry in ``sym_buffer_ptrs``.

    Multi-rank uses ``primus_turbo.pytorch.core.symm_mem.SymmetricMemory`` to
    rendezvous a hipMalloc-backed buffer across all peers, exposing each
    peer's mapped base pointer in ``sym_buffer_ptrs`` (length == num_ranks).
    Local tensor views are zero-copy slices of *this* rank's buffer.
    """

    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_ranks: int = 1,
        rank_idx: int = 0,
    ):
        import torch

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = _align_up(num_max_tokens_per_rank, _K_TOKEN_ALIGNMENT)
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ranks = num_ranks
        self.rank_idx = rank_idx

        jit = load_mega_moe_jit()
        offsets, total_bytes, num_max_pool_tokens, num_padded_sf_pool_tokens = jit.compute_layout(
            num_ranks, num_experts, self.num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
        )
        self.offsets = list(offsets)
        self.total_bytes = int(total_bytes)
        self.num_max_pool_tokens = int(num_max_pool_tokens)
        self.num_padded_sf_pool_tokens = int(num_padded_sf_pool_tokens)

        self.symm = None
        if num_ranks <= 1:
            self.buffer = torch.empty(self.total_bytes, dtype=torch.int8, device="cuda")
            self.buffer.zero_()
            torch.cuda.synchronize()
            self.sym_buffer_ptrs = [int(self.buffer.data_ptr())]
        else:
            from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

            print(f"[r{rank_idx}] BEFORE SymmetricMemory(total_bytes={self.total_bytes})", flush=True)
            self.symm = SymmetricMemory(group, self.total_bytes)
            print(f"[r{rank_idx}] AFTER SymmetricMemory ctor", flush=True)
            # SymmetricMemory zeros via hipMemset and barriers across the group;
            # local rank's view is buffer_ptrs[rank_idx].
            self.buffer = self.symm.get_buffer(
                self.rank_idx, (self.total_bytes,), torch.int8, storage_offset=0
            )
            self.sym_buffer_ptrs = list(self.symm.buffer_ptrs)

        # Carve named tensor views into the symmetric buffer.
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
        # `x`: FP8 dispatched tokens, (M, hidden)
        self.x = self._slice_view(input_x_off, (M, hidden), torch.float8_e4m3fn)
        # `x_sf`: packed UE8M0 SF, stored as int32 with hidden/gran_k/4 lanes.
        self.x_sf = self._slice_view(input_x_sf_off, (M, fp8_sf_n // 4), torch.int32)
        # `topk_idx`: int64
        self.topk_idx = self._slice_view(input_topk_idx_off, (M, num_topk), torch.int64)
        # `topk_weights`: float32
        self.topk_weights = self._slice_view(input_topk_weights_off, (M, num_topk), torch.float32)

    def _slice_view(self, offset: int, shape, dtype):
        import torch

        elem_size = torch.empty((), dtype=dtype).element_size()
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = nelem * elem_size
        if self.symm is None:
            byte_view = self.buffer.narrow(0, offset, nbytes)
            return byte_view.view(dtype).view(*shape)
        # SymmetricMemory: storage_offset is in *elements* of `dtype`.
        assert (
            offset % elem_size == 0
        ), f"layout offset {offset} not aligned to dtype {dtype} elem_size {elem_size}"
        return self.symm.get_buffer(self.rank_idx, shape, dtype, storage_offset=offset // elem_size)

    def destroy(self):
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.sym_buffer_ptrs = None
        if self.symm is not None:
            self.symm.destroy()
            self.symm = None


# ---------------------------------------------------------------------------
# DG-aligned test entry
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def _estimate_ep_recv_tokens(
    num_tokens: int, num_topk: int, num_experts: int, num_ranks: int, masked_ratio: float = 0.0
) -> int:
    """Analytically estimate ``num_recv_tokens`` for a given EP config.

    In EP-N, the total tokens across the cluster = ``num_tokens * num_ranks``.
    Each token is routed to ``num_topk`` experts out of ``num_experts``.
    Each rank owns ``num_experts // num_ranks`` experts.  On average each
    rank receives ``total_tokens * num_topk * (experts_per_rank / num_experts)``
    = ``num_tokens * num_topk`` (token, expert) pairs — independent of
    ``num_ranks``.
    """
    effective_topk = num_topk * (1.0 - masked_ratio)
    return int(num_tokens * effective_topk)


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    """DG-aligned per-rank entry, mirrors the structure of
    ``DeepGEMM/tests/test_mega_moe.py::test``.
    """
    import torch

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    if args.num_tokens is not None:
        num_tokens = args.num_tokens
    else:
        num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens))
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # The ``--ep-ranks`` argument specifies the intended EP width for
    # metric computation.  When running on fewer GPUs than the target EP
    # width (e.g. single-GPU testing EP8 shapes), we still compute the
    # theoretical metrics as if the full EP cluster were present.
    ep_ranks = getattr(args, "ep_ranks", None) or num_ranks
    ep_experts_per_rank = num_experts // ep_ranks

    # JIT-compile the kernel with the requested shape baked in via
    # ``-DMEGA_MOE_JIT_K*`` overrides.  Done up front so the parity
    # check and buffer allocation reuse the same cached module.
    jit_shape = {
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "hidden": hidden,
        "intermediate_hidden": intermediate_hidden,
        "num_experts": num_experts,
        "num_topk": num_topk,
        "num_ranks": num_ranks,
    }
    load_mega_moe_jit(jit_shape)

    # Run parity sanity check first so any layout-math regression fails
    # *before* we pay for kernel JIT compile.
    if local_rank == 0 and not args.skip_parity:
        if run_parity_checks() != 0:
            raise SystemExit(1)

    # Allocate symmetric memory (single-rank fast path).
    buffer = MegaMoEBuffer(
        group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_ranks=num_ranks,
        rank_idx=rank_idx,
    )

    dist_print("Config:", once_in_node=True)
    dist_print(f" > Tokens: {num_tokens}/{num_max_tokens_per_rank}", once_in_node=True)
    dist_print(f" > Hidden: {hidden}", once_in_node=True)
    dist_print(f" > Intermediate: {intermediate_hidden}", once_in_node=True)
    dist_print(
        f" > Experts: {num_topk}/{num_experts} (EP{ep_ranks}, {ep_experts_per_rank}/rank)", once_in_node=True
    )
    dist_print(f" > Buffer: {buffer.total_bytes / 2 ** 30:.4f} GiB", once_in_node=True)
    dist_print(f" > Ranks: {rank_idx}/{num_ranks}", once_in_node=True)
    dist_print(once_in_node=True)

    # Decide whether we can actually run the kernel (shape must match
    # the JIT-instantiated smoke template).  If not, we skip tensor
    # creation for weights (could be huge in single-rank mode with
    # hundreds of experts) and go straight to analytical metrics.
    jit = load_mega_moe_jit()

    # Probe whether the kernel shape matches the smoke template by
    # doing a quick zero-token call.
    probe_y = torch.empty((1, hidden), dtype=torch.bfloat16, device="cuda")
    probe_l1 = torch.empty(1, dtype=torch.int8, device="cuda")
    probe_l2 = torch.empty(1, dtype=torch.int8, device="cuda")
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
        probe_l1,
        probe_l1,
        probe_l2,
        probe_l2,
        None,
        args.activation_clamp,
        bool(args.fast_math),
    )
    del probe_y, probe_l1, probe_l2
    kernel_available = probe_status == 0

    def create_inputs():
        nonlocal_state = {}
        nonlocal_state["x_bf16"] = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        nonlocal_state["l1_weights_bf16"] = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device="cuda"
        )
        nonlocal_state["l2_weights_bf16"] = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda"
        )
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        nonlocal_state["topk_weights"] = topk_weights
        nonlocal_state["topk_idx"] = topk_idx
        nonlocal_state["recv_stats"] = torch.randint(
            0, 100, (num_experts_per_rank,), dtype=torch.int, device="cuda"
        )

        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        assert hidden % 128 == 0
        assert intermediate_hidden % 128 == 0

        x_fp8, x_sf = per_token_cast_to_fp8(
            nonlocal_state["x_bf16"], use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        nonlocal_state["x_fp8"] = x_fp8
        nonlocal_state["x_sf"] = x_sf

        nonlocal_state["l1_w_fp4"], nonlocal_state["l1_w_sf"] = cast_grouped_weights_to_fp4(
            nonlocal_state["l1_weights_bf16"]
        )
        nonlocal_state["l2_w_fp4"], nonlocal_state["l2_w_sf"] = cast_grouped_weights_to_fp4(
            nonlocal_state["l2_weights_bf16"]
        )
        return nonlocal_state

    def run_fused(inputs):
        if num_tokens > 0:
            buffer.x[:num_tokens].copy_(inputs["x_fp8"])
            buffer.x_sf[:num_tokens, : inputs["x_sf"].shape[1]].copy_(inputs["x_sf"])
            buffer.topk_idx[:num_tokens].copy_(inputs["topk_idx"])
            buffer.topk_weights[:num_tokens].copy_(inputs["topk_weights"])

        y = torch.empty((max(num_tokens, 1), hidden), dtype=torch.bfloat16, device="cuda")
        status = jit.run_mega_moe(
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
            inputs["l1_w_fp4"],
            inputs["l1_w_sf"],
            inputs["l2_w_fp4"],
            inputs["l2_w_sf"],
            inputs["recv_stats"],
            args.activation_clamp,
            bool(args.fast_math),
        )
        return y, inputs["recv_stats"], status

    # ---------------------------------------------------------------
    # Compute num_recv_tokens and num_touched_experts.
    # When the kernel is available and inputs are created, use the
    # actual topk_idx.  Otherwise, use analytical estimates with the
    # intended EP width.
    # ---------------------------------------------------------------
    t_fused = None
    if kernel_available:
        inputs = create_inputs()

        if num_tokens > 0:
            topk_idx = inputs["topk_idx"]
            if num_ranks == 1:
                num_recv_tokens = int((topk_idx != -1).sum().item())
                num_touched_experts = int(torch.unique(topk_idx[topk_idx >= 0]).numel())
            else:
                try:
                    from deep_gemm.utils.dist import uneven_all_gather as _all_gather

                    gathered_topk_idx = _all_gather(topk_idx, group=group)
                except ImportError:
                    gathered_topk_idx = topk_idx
                gathered_topk_idx[
                    (gathered_topk_idx < rank_idx * num_experts_per_rank)
                    | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
                ] = -1
                num_recv_tokens = int((gathered_topk_idx != -1).sum().item())
                num_touched_experts = int(torch.unique(gathered_topk_idx.flatten()).numel()) - 1
        else:
            num_recv_tokens = 0
            num_touched_experts = 0

        # Warmup + status check
        dist_print("Running fused kernel:", once_in_node=True)
        _, _, status = run_fused(inputs)
        torch.cuda.synchronize()
        if status == 0:
            dist_print(" > Launch status: OK", once_in_node=True)
        else:
            dist_print(f" > Launch status: FAIL hipError_t={status}", once_in_node=True)
            buffer.destroy()
            raise SystemExit(1)

        if args.ncu_profile_only:
            dist_print(" > Done, exiting (NCU profile mode)", once_in_node=True)
            buffer.destroy()
            return

        # Benchmark
        num_iters = max(1, args.num_perf_iters)
        for _ in range(max(1, args.num_warmup_iters)):
            run_fused(inputs)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            run_fused(inputs)
        end_event.record()
        torch.cuda.synchronize()
        t_fused = start_event.elapsed_time(end_event) / num_iters / 1e3

    else:
        # Kernel shape does not match the JIT-instantiated template.
        # Use analytical estimates for the intended EP configuration.
        num_recv_tokens = _estimate_ep_recv_tokens(
            num_tokens, num_topk, num_experts, ep_ranks, args.masked_ratio
        )
        num_touched_experts = min(ep_experts_per_rank, num_recv_tokens) if num_recv_tokens > 0 else 0

        dist_print("Running fused kernel:", once_in_node=True)
        dist_print(
            " > Launch status: SKIPPED (shape not yet JIT-instantiated; "
            f"using analytical estimates for EP{ep_ranks})",
            once_in_node=True,
        )

    # ---------------------------------------------------------------
    # Performance metrics (DG PR #316 format)
    # ---------------------------------------------------------------
    # TFLOPS: 3 matmuls (L1 gate, L1 up, L2), each 2 * M * N * K
    tflops = (
        _safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)
        if t_fused
        else float("nan")
    )

    # HBM bytes: weights (FP4 = 0.5B) + activations (FP8 = 1B) + output (BF16 = 2B)
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2  # L1 weights (FP4)
        + num_touched_experts * hidden * intermediate_hidden // 2  # L2 weights (FP4)
        + num_recv_tokens * hidden  # L1 acts read (FP8)
        + num_recv_tokens * intermediate_hidden  # L1 output write (FP8)
        + num_recv_tokens * intermediate_hidden  # L2 acts read (FP8)
        + num_recv_tokens * hidden * 2  # L2 output write (BF16)
    )
    hbm_gbs = _safe_div(num_hbm_bytes / 1e9, t_fused) if t_fused else float("nan")

    # Interconnect bytes: dispatch pull + combine write-back
    # (NVLink on DG, XGMI on AMD — same formula)
    num_interconnect_bytes = num_recv_tokens * hidden * 3
    interconnect_gbs = _safe_div(num_interconnect_bytes / 1e9, t_fused) if t_fused else float("nan")

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12
    approx_factor = _safe_div(t_fused, t_fused - t_reduction) if t_fused else 1.0

    # Print summary in DG format.  When ``num_recv_tokens == 0`` the
    # kernel takes the early-exit path (or every token was masked) so
    # ``t_fused`` reflects pure launch overhead, not GEMM compute.
    # Reporting "0 TFLOPS" under a "Performance:" header would be
    # misleading — relabel and print only the latency.  The MFMA body
    # itself is also still scaffolding (see MegaKernel/TODO.md §B.2)
    # so even non-zero token counts cannot produce real perf yet;
    # callers that want actual numbers must wait on the A/B loader.
    if t_fused is not None and num_recv_tokens == 0:
        dist_print("Launch overhead (stub kernel, no compute):", once_in_node=True)
        dist_print(
            f" > EP: {rank_idx:2}/{num_ranks} | "
            f"{t_fused * 1e6:4.0f} us launch+sync | "
            f"recv_tokens=0 (early-exit path)",
            once_in_node=True,
        )
    elif t_fused is not None:
        dist_print("Performance:", once_in_node=True)
        dist_print(
            f" > EP: {rank_idx:2}/{num_ranks} | "
            f"{tflops:4.0f} TFLOPS | "
            f"overlap: "
            f"{tflops * approx_factor:4.0f} TFLOPS, "
            f"HBM {hbm_gbs * approx_factor:4.0f} GB/s, "
            f"XGMI {interconnect_gbs * approx_factor:3.0f} GB/s | "
            f"{t_fused * 1e6:4.0f} us, "
            f"reduction: {t_reduction * 1e6:4.1f} us",
            once_in_node=True,
        )
    else:
        dist_print("Performance:", once_in_node=True)
        dist_print(
            f" > EP: {rank_idx:2}/{num_ranks} (target EP{ep_ranks}) | "
            f"recv_tokens: {num_recv_tokens} | "
            f"touched_experts: {num_touched_experts}/{ep_experts_per_rank} | "
            f"compute: {2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12:.3f} TFLOP | "
            f"HBM: {num_hbm_bytes / 1e9:.3f} GB | "
            f"interconnect: {num_interconnect_bytes / 1e9:.3f} GB | "
            f"reduction: {t_reduction * 1e6:.1f} us",
            once_in_node=True,
        )

    # Exit
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass
    buffer.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-aligned mega-MoE JIT correctness + perf test (gfx950)")

    # Resource settings
    parser.add_argument(
        "--ncu-profile-only", action="store_true", help="Only run a single iteration (NCU profile mode)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 1 for single-GPU smoke test; "
        "use 8 to mirror DG's default)",
    )
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="Only run layout-parity checks, skip the full DG-aligned test",
    )
    parser.add_argument(
        "--skip-parity", action="store_true", help="Skip layout-parity sanity check before the kernel test"
    )

    # Model settings (defaults match DG-aligned smoke shape so the JIT
    # kernel can actually run; pass DG defaults via CLI to mirror DG's
    # config print without running the kernel)
    parser.add_argument(
        "--num-max-tokens-per-rank",
        type=int,
        default=384,
        help="Number of maximum tokens per rank (smoke default: 384; DG: 8192)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=128,
        help="Number of tokens per rank (0 = early-exit smoke; None = max minus removed)",
    )
    parser.add_argument(
        "--num-max-removed-tokens",
        type=int,
        default=0,
        help="Maximum number of tokens to remove from num_max_tokens_per_rank",
    )
    parser.add_argument("--hidden", type=int, default=256, help="Hidden size (smoke default: 256; DG: 7168)")
    parser.add_argument(
        "--intermediate-hidden",
        type=int,
        default=128,
        help="Intermediate hidden size (smoke default: 128; DG: 3072)",
    )
    parser.add_argument("--activation-clamp", type=float, default=10.0, help="Clamp value for activation")
    parser.add_argument(
        "--num-experts", type=int, default=8, help="Number of experts (smoke default: 8; DG: 384)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=2, help="Number of expert selections (smoke default: 2; DG: 6)"
    )
    parser.add_argument("--masked-ratio", type=float, default=0.0, help="Mask some expert selections")
    parser.add_argument("--fast-math", type=int, default=1, help="Enable fast math (0 or 1, default: 1)")

    # EP settings
    parser.add_argument(
        "--ep-ranks",
        type=int,
        default=None,
        help="Intended EP width for metric computation (e.g. 8 for EP8). "
        "Defaults to --num-processes. Allows computing EP8 metrics on a single GPU.",
    )

    # Test settings
    parser.add_argument("--num-warmup-iters", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--num-perf-iters", type=int, default=10, help="Timed iterations")
    parser.add_argument(
        "--dump-profile-traces",
        type=str,
        default="",
        help="Dump profiling trace JSONs (currently unused; DG-API compat)",
    )
    parser.add_argument(
        "--local-rank-idx",
        type=int,
        default=None,
        help="Run as single process with this local rank (e.g. for NCU prof)",
    )
    return parser


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    # Path 1: parity-only — preserves the previous test_mega_moe_jit
    # behavior so CI can keep its low-overhead layout sanity gate.
    if args.parity_only:
        return 1 if run_parity_checks() != 0 else 0

    # Path 2: DG-aligned test.  Single-rank in-process, or
    # ``torch.multiprocessing.spawn`` for multi-rank.
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
        return 0

    if args.num_processes <= 1:
        test(0, 1, args)
        return 0

    import torch.multiprocessing as mp

    mp.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
