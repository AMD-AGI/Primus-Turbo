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

import functools
import inspect
from typing import Optional, Tuple

import torch

from primus_turbo.common.aiter_utils import get_aiter

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
    "aiter_has_a6w4_bias_epilogue",
    "quantize_mxfp6_row_mxfp4_col_dual",
    "check_a6w4_support",
    "mxfp4_data_region",
    "mxfp4_gemm_pack_sizes",
    "quantize_mxfp4_gemm_col",
    "quantize_mxfp4_gemm_dual",
    "quantize_mxfp4_gemm_row",
]

# What an aiter must expose for A6W4 to be usable. `gemm_a6w4` and its packer both
# landed in ROCm/aiter#5587; an aiter predating it runs MXFP6 perfectly well and simply
# cannot do A6W4, which is a configuration error rather than a crash.
_A6W4_REQUIRED_ATTRS = ("gemm_a6w4", "quant_mxfp4_gemm")

_MISSING_A6W4_HINT = (
    "A6W4 (MXFP6 activations x MXFP4 weights) needs an aiter carrying the A6W4 asm GEMM "
    "from ROCm/aiter#5587. Set mxfp6_weight_format='mxfp6' to stay on A6W6"
)


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


@functools.lru_cache(maxsize=1)
def check_a6w4_support(device: Optional[torch.device] = None) -> Tuple[bool, str]:
    """Whether A6W4 is usable here: everything MXFP6 needs, plus the A6W4 entry points.

    Probed rather than version-checked, like ``check_mxfp6_support``: no version string
    can express "contains commit X". A6W4 is a strict superset of A6W6's requirements,
    because its activation operand comes from the MXFP6 packer, so the MXFP6 gate runs
    first and its message is returned unchanged when it is the one that fails.
    """
    supported, reason = check_mxfp6_support(device)
    if not supported:
        return False, reason
    try:
        aiter = get_aiter()
    except ImportError as exc:
        return False, f"{_MISSING_A6W4_HINT} ({exc})"
    missing = [a for a in _A6W4_REQUIRED_ATTRS if not hasattr(aiter, a)]
    if missing:
        return False, f"{_MISSING_A6W4_HINT} (missing: {', '.join(missing)})"
    return True, ""


@functools.lru_cache(maxsize=1)
def aiter_has_a6w4_bias_epilogue() -> bool:
    """Whether ``aiter.gemm_a6w4`` can fold a bias into its store epilogue.

    Probed by parameter rather than by symbol, and for the same reason as
    ``aiter_has_bias_epilogue``: the entry point is the same one either way and only its
    signature changed, so ``hasattr`` cannot tell the two apart. Upstream's A6W4 has no
    epilogue; ours does, and an aiter without it gets the separate elementwise pass.
    """
    try:
        aiter = get_aiter()
    except ImportError:
        return False
    fn = getattr(aiter, "gemm_a6w4", None)
    if fn is None:
        return False
    try:
        return "bias" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # No introspectable signature. Answering "no" costs one pass over the output;
        # answering "yes" wrongly is a TypeError at the first GEMM.
        return False


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


def quantize_mxfp6_row_mxfp4_col_dual(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``x`` as MXFP6 along the last axis and MXFP4 along the first, in one pass.

    Returns ``(row_operand, row_scale, col_operand, col_scale)``. The row blob is what a
    forward or dgrad GEMM consumes as its MXFP6 operand; the column blob is what wgrad
    consumes as its MXFP4 one.

    This is what makes wgrad eligible for a mixed-format GEMM. ``grad_w = g_col @ x_col``
    contracts the token dimension, so neither operand is the weight and A6W4 cannot reach
    it -- a third of GEMM time. Narrowing one of the two fixes that, and whichever tensor
    is narrowed needs exactly this shape of pack: fp6 in the direction the forward or dgrad
    reads, fp4 in the direction wgrad reads.

    It costs nothing extra. The column half writes two thirds of MXFP6's bytes, and both
    halves still come from a single staged read of ``x``.
    """
    _require_supported(x.device)
    _check_input(x)
    return tuple(
        torch.ops.primus_turbo_cpp_extension.quantize_mxfp6_row_mxfp4_col_dual(x.contiguous())
    )
