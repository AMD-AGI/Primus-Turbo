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

Implementation status
---------------------
Both directions currently route through AITER's row-direction packer, with the column
direction obtained by materialising a transpose first. That is correct by construction
(it reuses the packer that upstream tests and our parity run cover) and it is what makes
the rest of the stack testable, but it costs an extra full-tensor read/write per column
pack and does not fuse the two directions into a single pass the way MXFP4's
``quantize_mxfp4_dual`` HIP kernel does.

Replacing this with a fused kernel is a performance change, not a correctness one, and
these functions are the oracle it must reproduce. Until then, do not read step-time
comparisons against MXFP4 as MXFP6-vs-MXFP4: they also include transpose-vs-fused.
"""

from typing import Tuple

import torch

from primus_turbo.common.aiter_utils import get_aiter
from primus_turbo.pytorch.core.low_precision import (
    MXFP6_BLOCK_SIZE,
    MXFP6_GUARD_K_TILES,
    MXFP6_K_TILE_SIZE,
    MXFP6_PACKED_TILE_BYTES,
    MXFP6_SCALE_TILE_BYTES,
    MXFP6_TILE_SIZE,
)

__all__ = [
    "check_mxfp6_support",
    "mxfp6_data_region",
    "mxfp6_pack_sizes",
    "quantize_mxfp6_col",
    "quantize_mxfp6_dual",
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


def quantize_mxfp6_row(
    x: torch.Tensor, block_size: int = MXFP6_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, C]`` contracting along ``C`` (the last axis).

    Returns ``(operand_blob, scale_blob)``, both 1-D uint8. Rows are padded to 256 and
    C to 128 inside the blob; the logical shape is the caller's to remember.
    """
    _assert_supported()
    _check_input(x, block_size)
    return get_aiter().quant_mxfp6_gemm(x.contiguous())


def quantize_mxfp6_col(
    x: torch.Tensor, block_size: int = MXFP6_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, C]`` contracting along ``R`` (the first axis).

    Equivalent to ``quantize_mxfp6_row(x.T)``, and implemented that way: the transpose
    is materialised so the verified row packer can be reused. A fused kernel would read
    ``x`` once instead.
    """
    _assert_supported()
    _check_input(x, block_size)
    return get_aiter().quant_mxfp6_gemm(x.t().contiguous())


def quantize_mxfp6_dual(
    x: torch.Tensor, block_size: int = MXFP6_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``x`` in both directions at once.

    Returns ``(row_operand, row_scale, col_operand, col_scale)``, mirroring the
    ``(out, scale, out_t, scale_t)`` shape that ``quantize_mxfp4_impl(with_trans=True)``
    returns so the autograd functions look the same.

    This is two independent passes today. The MXFP4 analogue is a single fused HIP
    kernel, and matching that is the main remaining performance item for MXFP6.
    """
    _assert_supported()
    _check_input(x, block_size)
    x = x.contiguous()
    aiter = get_aiter()
    row_p, row_s = aiter.quant_mxfp6_gemm(x)
    col_p, col_s = aiter.quant_mxfp6_gemm(x.t().contiguous())
    return row_p, row_s, col_p, col_s
