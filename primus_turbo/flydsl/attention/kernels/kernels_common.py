###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Small FlyDSL helpers vendored for the ported DeepSeek-V4 attention kernels.

The reference kernels (ported from the FlyDSL-amd source tree) import a single
symbol, ``dtype_to_elem_type``, from ``kernels.kernels_common``. That package is
part of the FlyDSL-amd *source* layout and is not present in the pip-installed
``flydsl`` runtime, so the one helper the kernels need is reconstructed here.

``dtype_to_elem_type`` maps a dtype string ("bf16" / "f16" / ...) to the FlyDSL
MLIR element type (``flydsl.expr.typing.T.bf16`` etc.). The ``T.*`` properties
build their MLIR type lazily, so this must be called inside an active MLIR
``Context`` (i.e. during kernel build), exactly as in the reference.
"""

from __future__ import annotations

from flydsl.expr.typing import T

__all__ = [
    "dtype_to_elem_type",
    "mfma_f32_16x16x32",
]

# Accepted spellings per logical dtype (lower-cased before lookup).
_BF16 = {"bf16", "bfloat16"}
_F16 = {"f16", "fp16", "float16", "half"}
_F32 = {"f32", "fp32", "float32", "float"}
_F8E4M3 = {"f8", "fp8", "f8e4m3", "float8_e4m3", "e4m3"}
_F8E5M2 = {"f8e5m2", "float8_e5m2", "e5m2"}


def dtype_to_elem_type(dtype_str):
    """Return the FlyDSL element type for ``dtype_str`` (call inside a Context)."""
    s = str(dtype_str).lower()
    if s in _BF16:
        return T.bf16
    if s in _F16:
        return T.f16
    if s in _F32:
        return T.f32
    if s in _F8E5M2:
        return T.f8
    if s in _F8E4M3:
        return T.f8
    raise ValueError(f"unsupported dtype_str for FlyDSL elem type: {dtype_str!r}")


def mfma_f32_16x16x32(a_pack, b_pack, c_acc, dtype_str="bf16"):
    """Single CDNA4 (gfx950) ``v_mfma_f32_16x16x32_{bf16,f16}``.

    Computes one matrix-core tile ``D[16, 16] += A[16, 32] @ B[32, 16]`` with
    f32 accumulation. ``a_pack`` / ``b_pack`` are the MFMA operand packs
    (``vec(8, elem)`` -- 4 VGPRs, 8 packed 16-bit items per lane, in the
    standard A/B lane layout: lane ``L`` holds row/col ``L % 16`` and K-subgroup
    ``(L // 16) * 8 + 0..7``). ``c_acc`` is the ``vec(4, f32)`` accumulator
    (column-major: lane ``L`` holds ``C[(L // 16) * 4 + k, L % 16]`` for
    ``k = 0..3``). Returns the updated ``vec(4, f32)`` accumulator.

    ``cbsz / abid / blgp`` are left at 0; operand sharing across the 16 M-rows
    (the single-query broadcast) is realised by a per-lane redundant load of the
    broadcast operand, which is functionally identical and avoids illegal
    broadcast encodings. Must be called inside an active MLIR Context.
    """
    from flydsl.expr import rocdl

    rty = T.vec(4, T.f32)
    s = str(dtype_str).lower()
    if s in _F16:
        return rocdl.mfma_f32_16x16x32_f16(rty, [a_pack, b_pack, c_acc])
    return rocdl.mfma_f32_16x16x32_bf16(rty, [a_pack, b_pack, c_acc])
