###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP4 (E2M1) quantize + pack into AITER's A6W4 weight-operand blob layout.

This is the *weight* half of A6W4 (MXFP6 activations x MXFP4 weights). The activation
half needs nothing new: ``gemm_a6w4`` consumes exactly the blob
:mod:`~primus_turbo.pytorch.kernels.quantization.mxfp6_pack` already produces, because
upstream built A6W4 on the same H32-rotated packing contract.

Deliberately narrower than ``mxfp6_pack``:

* **No prologues.** Every MXFP6 prologue (bias+GELU, QK-norm+RoPE, LN-modulate) fuses an
  elementwise op into the packer's staging read of an *activation*. Weights arrive as
  plain parameters with nothing to fuse, so there is no ``MXFP4Prologue``.
* **No column sums.** That buffer exists to give the bias gradient for free out of a
  packed activation. A weight has no bias gradient.
* **No wgrad direction.** ``grad_w = g_col @ x_col`` contracts M and never reads the
  weight, so wgrad has no MXFP4 operand and stays A6W6. Only row (forward) and column
  (dgrad) exist here.

The blob geometry is shared with MXFP6 -- 256-row tiles, 128-K tiles, two guard K tiles,
the same E8M0 scale plane -- so the constants below mirror it. The one that differs is
``MXFP4_PACKED_TILE_BYTES``: 16384 against MXFP6's 24576, because E2M1 spends 4 bits per
value in one compact plane where E2M3 spends 6 across a C0 and a C1 plane.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP4_GEMM_PACKED_TILE_BYTES,
    MXFP4_GEMM_SCALE_TILE_BYTES,
    MXFP6_GUARD_K_TILES,
    MXFP6_K_TILE_SIZE,
    MXFP6_TILE_SIZE,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import check_mxfp6_support

__all__ = [
    "mxfp4_data_region",
    "mxfp4_gemm_pack_sizes",
    "quantize_mxfp4_gemm_col",
    "quantize_mxfp4_gemm_dual",
    "quantize_mxfp4_gemm_row",
]


def _ceil(x: int, m: int) -> int:
    # Rounds up to a multiple of m, NOT to a count of them -- same contract as
    # mxfp6_pack._ceil, so the `_ceil(...) // tile` idiom below reads the same in both.
    return -(-x // m) * m


def _require_supported(device: Optional[torch.device] = None) -> None:
    # The gate is identical to MXFP6's -- gfx950 plus an aiter carrying the asm GEMM --
    # so it is reused rather than restated. A6W4 is never reachable without A6W6 anyway:
    # its activation operand comes from the MXFP6 packer.
    supported, reason = check_mxfp6_support(device)
    if not supported:
        raise RuntimeError(reason)


def mxfp4_gemm_pack_sizes(rows: int, k: int) -> Tuple[int, int]:
    """Byte counts of the ``(operand, scale)`` blobs for a ``[rows, k]`` operand.

    Both include the two guard K-tiles. Their contents are never read, but the space is
    mandatory: the A6W4 assembly derives its row-tile stride from ``k/128 + 2``. Must
    agree with ``aiter.ops.gemm_op_a6w4.mxfp4_gemm_pack_size``.
    """
    n_row_tiles = _ceil(rows, MXFP6_TILE_SIZE) // MXFP6_TILE_SIZE
    n_k_tiles = _ceil(k, MXFP6_K_TILE_SIZE) // MXFP6_K_TILE_SIZE + MXFP6_GUARD_K_TILES
    return (
        n_row_tiles * n_k_tiles * MXFP4_GEMM_PACKED_TILE_BYTES,
        n_row_tiles * n_k_tiles * MXFP4_GEMM_SCALE_TILE_BYTES,
    )


def mxfp4_data_region(
    blob: torch.Tensor, rows: int, k: int, *, is_scale: bool = False
) -> torch.Tensor:
    """View of the meaningful bytes of a packed blob, with guard tiles dropped.

    Any bit-exactness assertion against a packed blob has to go through this, for the same
    reason as ``mxfp6_data_region``: the packers do not initialise the guard tiles, so
    comparing whole blobs reports mismatches that mean nothing.
    """
    tile_bytes = MXFP4_GEMM_SCALE_TILE_BYTES if is_scale else MXFP4_GEMM_PACKED_TILE_BYTES
    n_row_tiles = _ceil(rows, MXFP6_TILE_SIZE) // MXFP6_TILE_SIZE
    n_k_tiles = _ceil(k, MXFP6_K_TILE_SIZE) // MXFP6_K_TILE_SIZE
    view = blob.view(n_row_tiles, n_k_tiles + MXFP6_GUARD_K_TILES, tile_bytes)
    return view[:, :n_k_tiles, :]


def _check_input(x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"MXFP4 GEMM packing expects a 2D tensor, got {x.ndim}D")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"MXFP4 GEMM packing expects bf16 or fp16 input, got {x.dtype}")
    if x.shape[0] % MXFP4_BLOCK_SIZE or x.shape[1] % MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 scales strictly per 1x{MXFP4_BLOCK_SIZE} along whichever axis is "
            f"contracted, and this packs both, so both dimensions must be multiples of "
            f"{MXFP4_BLOCK_SIZE}. Got {tuple(x.shape)}"
        )


def quantize_mxfp4_gemm_row(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, C]`` contracting along ``C`` -- the forward's weight operand.

    Returns ``(operand_blob, scale_blob)``, both 1-D uint8. Rows are padded to 256 and C
    to 128 inside the blob; the logical shape is the caller's to remember.
    """
    _require_supported(x.device)
    _check_input(x)
    return tuple(torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_gemm(x.contiguous(), 1))


def quantize_mxfp4_gemm_col(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[R, C]`` contracting along ``R`` -- dgrad's weight operand.

    Equivalent to ``quantize_mxfp4_gemm_row(x.T)``, but the packer reads ``x`` in place
    rather than materialising the transpose. That is the whole reason this exists:
    AITER's ``quant_mxfp4_gemm`` packs the row direction only, so the column direction
    there costs a real transpose of every weight, every microbatch.
    """
    _require_supported(x.device)
    _check_input(x)
    return tuple(torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_gemm(x.contiguous(), 0))


def quantize_mxfp4_gemm_dual(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``x`` in both directions at once.

    Returns ``(row_operand, row_scale, col_operand, col_scale)``, mirroring
    ``quantize_mxfp6_dual`` so the autograd functions look the same on both operands.

    One pass over ``x``: the kernel stages each tile once and packs it along both axes,
    which is what makes this cheaper than calling the two single-direction packers.
    """
    _require_supported(x.device)
    _check_input(x)
    return tuple(torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_gemm_dual(x.contiguous()))
