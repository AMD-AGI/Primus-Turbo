###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP6 (E2M3) quantize + pack, in both contraction directions.

Unlike MXFP8/MXFP4, an MXFP6 operand is not a strided tensor plus a scale tensor. The
A6W6 assembly consumes an opaque blob in AITER's ``mxfp6_c0c1_256_padk2`` layout: 6-bit
codes re-tiled into 256-row / 128-K tiles split across two planes, with the mandatory
32-point Hadamard rotation already applied along the contraction axis, plus a separate
packed E8M0 scale blob. So "quantizing" here means producing that pair of blobs, and the
logical shape has to be carried alongside because the blob does not encode it.

Training needs each tensor packed along *two* different axes -- see
``quantize_mxfp6_dual`` -- because the three GEMM directions contract over different
dimensions:

    fprop  Y  = X @ W.T    contract K  ->  row(X),  row(W)
    dgrad  dX = dY @ W     contract N  ->  row(dY), col(W)
    wgrad  dW = dY.T @ X   contract M  ->  col(dY), col(X)

Implementation
--------------
These route through Primus-Turbo's fused packer, which reads the input once and emits
both directions from a single staged tile. AITER only ships a row-direction packer, so
the column direction previously came from ``pack(x.t().contiguous())``; profiling a Flux
12B step attributed 82% of the MXFP6-vs-MXFP4 step-time gap to that materialised
transpose alone, and removing it makes the dual pack ~2.5x faster.

The fused kernel is bit-exact with AITER's packer in the row direction, which is the
property that lets it be swapped in freely: AITER's packer is the oracle the A6W6
assembly was validated against. ``PRIMUS_TURBO_MXFP6_PACKER=aiter`` restores the old path
for A/B comparison, and is also the automatic fallback on a build whose extension
predates the fused op.
"""

import os
from typing import Optional, Tuple

import torch

from primus_turbo.common.aiter_utils import get_aiter
from primus_turbo.pytorch.core.low_precision import (
    MXFP6_BLOCK_SIZE,
    MXFP6_COL_SUM_TILE_M,
    MXFP6_GUARD_K_TILES,
    MXFP6_K_TILE_SIZE,
    MXFP6_PACKED_TILE_BYTES,
    MXFP6_PROLOGUE_BIAS_GELU,
    MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
    MXFP6_PROLOGUE_IDENTITY,
    MXFP6_SCALE_TILE_BYTES,
    MXFP6_TILE_SIZE,
)

__all__ = [
    "check_mxfp6_support",
    "mxfp6_apply_prologue",
    "mxfp6_col_sum_rows",
    "mxfp6_data_region",
    "mxfp6_pack_sizes",
    "quantize_mxfp6_col",
    "quantize_mxfp6_dual",
    "quantize_mxfp6_fused_dual",
    "quantize_mxfp6_row",
]

_A6W6_REQUIRED_ATTRS = ("quant_mxfp6_gemm", "gemm_a6w6", "mxfp6_gemm_pack_size")

_MISSING_A6W6_HINT = (
    "The installed aiter has no MXFP6 (A6W6) support. It arrived in "
    "https://github.com/ROCm/aiter/pull/4859, which is not in the pinned release, so "
    "an aiter built from a branch containing that PR is required."
)


def _ceil(x: int, m: int) -> int:
    return -(-x // m) * m


def check_mxfp6_support() -> Tuple[bool, str]:
    """Return whether MXFP6 can run here, and why not if it cannot."""
    from primus_turbo.pytorch.core.utils import is_gfx950

    if not is_gfx950():
        return False, "MXFP6 requires gfx950 (MI350/MI355): the A6W6 kernels are gfx950 asm."
    aiter = get_aiter()
    missing = [a for a in _A6W6_REQUIRED_ATTRS if not hasattr(aiter, a)]
    if missing:
        return False, f"{_MISSING_A6W6_HINT} (missing: {', '.join(missing)})"
    return True, ""


def _assert_supported() -> None:
    ok, reason = check_mxfp6_support()
    assert ok, reason


def mxfp6_pack_sizes(rows: int, k: int) -> Tuple[int, int]:
    """Byte sizes of the (operand, scale) blobs for a ``[rows, k]`` operand.

    Both include the ``MXFP6_GUARD_K_TILES`` trailing tiles. Their contents are never
    read by the kernel, but the space is mandatory: the assembly derives its row-tile
    stride from ``k/128 + 2``, so a blob sized without them makes every stride wrong.
    """
    n_row_tiles = _ceil(rows, MXFP6_TILE_SIZE) // MXFP6_TILE_SIZE
    n_k_tiles = _ceil(k, MXFP6_K_TILE_SIZE) // MXFP6_K_TILE_SIZE + MXFP6_GUARD_K_TILES
    return (
        n_row_tiles * n_k_tiles * MXFP6_PACKED_TILE_BYTES,
        n_row_tiles * n_k_tiles * MXFP6_SCALE_TILE_BYTES,
    )


def mxfp6_data_region(blob: torch.Tensor, rows: int, k: int, *, is_scale: bool = False) -> torch.Tensor:
    """View of the meaningful bytes of a packed blob, with guard tiles dropped.

    Any bit-exactness assertion against a packed blob has to go through this. The guard
    tiles are never read by the kernel and the packers do not initialise them, so their
    contents differ between two calls on identical input -- comparing whole blobs
    reports spurious mismatches.
    """
    tile_bytes = MXFP6_SCALE_TILE_BYTES if is_scale else MXFP6_PACKED_TILE_BYTES
    n_row_tiles = _ceil(rows, MXFP6_TILE_SIZE) // MXFP6_TILE_SIZE
    n_k_tiles = _ceil(k, MXFP6_K_TILE_SIZE) // MXFP6_K_TILE_SIZE
    view = blob.view(n_row_tiles, n_k_tiles + MXFP6_GUARD_K_TILES, tile_bytes)
    return view[:, :n_k_tiles, :]


def _check_input(x: torch.Tensor, block_size: int) -> None:
    assert x.ndim == 2, f"MXFP6 quantization expects a 2D tensor, got {x.ndim}D"
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"MXFP6 quantization expects bf16/fp16/fp32 input, got {x.dtype}"
    )
    assert block_size == MXFP6_BLOCK_SIZE, (
        f"MXFP6 scaling is strictly per-1x{MXFP6_BLOCK_SIZE} along the contraction "
        f"axis, so block_size must be {MXFP6_BLOCK_SIZE}, got {block_size}"
    )


def _use_fused_packer() -> bool:
    """Whether to use Primus-Turbo's fused packer rather than AITER's row packer.

    Falls back to AITER when the extension predates the fused op, so an older build keeps
    working instead of failing at the op lookup.
    """
    choice = os.environ.get("PRIMUS_TURBO_MXFP6_PACKER", "").strip().lower() or "auto"
    assert choice in ("auto", "fused", "aiter"), (
        f"PRIMUS_TURBO_MXFP6_PACKER must be auto, fused or aiter, got {choice!r}"
    )
    if choice == "aiter":
        return False
    available = hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_mxfp6_dual")
    assert available or choice != "fused", (
        "PRIMUS_TURBO_MXFP6_PACKER=fused, but this build has no quantize_mxfp6_dual. "
        "The kernel is gfx950-only and is compiled in only when gfx950 is an offload arch."
    )
    return available


def quantize_mxfp6_row(
    x: torch.Tensor, block_size: int = MXFP6_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, C]`` contracting along ``C`` (the last axis).

    Returns ``(operand_blob, scale_blob)``, both 1-D uint8. Rows are padded to 256 and
    C to 128 inside the blob; the logical shape is the caller's to remember.
    """
    _assert_supported()
    _check_input(x, block_size)
    x = x.contiguous()
    if _use_fused_packer():
        packed, scale = torch.ops.primus_turbo_cpp_extension.quantize_mxfp6(x, 1)
        return packed, scale
    return get_aiter().quant_mxfp6_gemm(x)


def quantize_mxfp6_col(
    x: torch.Tensor, block_size: int = MXFP6_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, C]`` contracting along ``R`` (the first axis).

    Equivalent to ``quantize_mxfp6_row(x.T)``, but the fused packer reads ``x`` in place
    rather than materialising the transpose.
    """
    _assert_supported()
    _check_input(x, block_size)
    x = x.contiguous()
    if _use_fused_packer():
        packed, scale = torch.ops.primus_turbo_cpp_extension.quantize_mxfp6(x, 0)
        return packed, scale
    return get_aiter().quant_mxfp6_gemm(x.t().contiguous())


def quantize_mxfp6_dual(
    x: torch.Tensor, block_size: int = MXFP6_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``x`` in both directions at once.

    Returns ``(row_operand, row_scale, col_operand, col_scale)``, mirroring the
    ``(out, scale, out_t, scale_t)`` shape that ``quantize_mxfp4_impl(with_trans=True)``
    returns so the autograd functions look the same.

    One pass over ``x``: the fused kernel stages each tile once and packs it along both
    axes, which is the whole reason this beats calling the row packer twice.
    """
    _assert_supported()
    _check_input(x, block_size)
    x = x.contiguous()
    if _use_fused_packer():
        row_p, row_s, col_p, col_s = torch.ops.primus_turbo_cpp_extension.quantize_mxfp6_dual(x)
        return row_p, row_s, col_p, col_s
    aiter = get_aiter()
    row_p, row_s = aiter.quant_mxfp6_gemm(x)
    col_p, col_s = aiter.quant_mxfp6_gemm(x.t().contiguous())
    return row_p, row_s, col_p, col_s


def mxfp6_col_sum_rows(m: int) -> int:
    """Rows of the packer's bias-gradient partial buffer for an ``[m, n]`` input.

    One per M-tile of the launch grid, which covers ``m`` padded to whole 256-row tiles.
    Must agree with ``mxfp6_col_sum_rows`` in ``quantization.h``.
    """
    return _ceil(_ceil(m, MXFP6_TILE_SIZE), MXFP6_COL_SUM_TILE_M) // MXFP6_COL_SUM_TILE_M


def mxfp6_apply_prologue(
    x: torch.Tensor,
    aux: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    mode: int,
) -> torch.Tensor:
    """Materialise the epilogue that ``quantize_mxfp6_fused_dual`` folds into the pack.

    This is the reference the fused kernel is checked against, and the fallback when the
    extension has no fused op. Kept in one place so the two cannot drift.

    Deliberately ATen's own GELU rather than a transliteration of the kernel's: as the
    fallback it should be the operation the model would have run anyway, and as the reference
    it is only useful if it was written independently of what it is checking.
    """
    if mode == MXFP6_PROLOGUE_IDENTITY:
        return x
    pre = x if bias is None else x + bias
    if mode == MXFP6_PROLOGUE_BIAS_GELU:
        return torch.nn.functional.gelu(pre, approximate="tanh")
    if mode == MXFP6_PROLOGUE_BIAS_GELU_BACKWARD:
        assert aux is not None, "the backward prologue needs the incoming gradient"
        return torch.ops.aten.gelu_backward(aux, pre, approximate="tanh")
    raise ValueError(f"unknown MXFP6 prologue mode {mode}")


def quantize_mxfp6_fused_dual(
    x: torch.Tensor,
    aux: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    mode: int = MXFP6_PROLOGUE_IDENTITY,
    want_col_sum: bool = False,
    block_size: int = MXFP6_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual pack with an elementwise epilogue folded into the staging read.

    The tensor the epilogue produces is never written to HBM: the packer computes it while
    staging its tile into LDS, which removes both that write and the packer's read of it.
    In a Flux 12B MXFP6 step those round-trips are the single largest remaining block of
    non-GEMM traffic.

    ``mode`` is one of the ``MXFP6_PROLOGUE_*`` constants. ``bias`` is broadcast along the
    last axis and may be None. ``aux`` is the incoming gradient, required by
    ``MXFP6_PROLOGUE_BIAS_GELU_BACKWARD`` and unused otherwise.

    The four blobs reproduce ``quantize_mxfp6_dual(mxfp6_apply_prologue(...))``: the epilogue
    is evaluated in fp32 and rounded back to ``x.dtype`` before staging, exactly as a
    separate kernel writing bf16 to HBM would have. Every part of it is bit-identical except
    the tanh, which the kernel evaluates in closed form from one hardware exp2 rather than
    calling a libm tanh -- a 54-instruction-per-element difference that decides whether
    fusing the epilogue is faster than running it separately at all. That is a different
    rounding of the activation, not a less accurate one, and it leaves ~0.0003% of the packed
    codes differing by one and the E8M0 scales untouched. ``MXFP6_PROLOGUE_IDENTITY`` has no
    tanh and stays exactly equal.

    ``want_col_sum`` additionally returns per-column sums of the staged values as a
    ``[mxfp6_col_sum_rows(M), N]`` fp32 partial buffer, to be finished with ``.sum(0)``.
    That is how a bias gradient survives the fusion: the tensor it would be reduced from no
    longer exists in HBM. Unlike the blobs this is *not* bit-exact with the eager reduction
    -- it is a different (tree-ordered, fp32-accumulated) summation of the same values.
    The fifth return is a degenerate empty tensor when not requested.
    """
    _assert_supported()
    _check_input(x, block_size)

    x = x.contiguous()
    if aux is not None:
        aux = aux.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    if _use_fused_packer() and hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_mxfp6_fused_dual"):
        blobs = torch.ops.primus_turbo_cpp_extension.quantize_mxfp6_fused_dual(
            x, aux, bias, mode, want_col_sum
        )
        return tuple(blobs)

    # Fallback: materialise the epilogue and pack that. Same values, more traffic.
    staged = mxfp6_apply_prologue(x, aux, bias, mode)
    row_p, row_s, col_p, col_s = quantize_mxfp6_dual(staged, block_size)
    if not want_col_sum:
        empty = torch.empty((0, 0), dtype=torch.float32, device=x.device)
        return row_p, row_s, col_p, col_s, empty
    # One row, so the caller's .sum(0) is a no-op reshape rather than a special case.
    return row_p, row_s, col_p, col_s, staged.float().sum(0, keepdim=True)
