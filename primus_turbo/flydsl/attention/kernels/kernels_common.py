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
from flydsl.expr import rocdl, vector
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm

__all__ = ["dtype_to_elem_type", "ds_read_tr16_b_pack"]

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


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def ds_read_tr16_b_pack(lane_base_ptr, base_elem, lds_stride,
                        mfma_pack_type, v4_type):
    """Build one MFMA 16x16x32 B-fragment (8 bf16/lane) from LDS using two
    hardware transposing reads (``rocdl.ds_read_tr16_b64``).

    This is the EXACT read schedule already used by the sibling backward
    kernels (``sla_bwd_dkv`` GEMM2/GEMM4, ``hca_bwd_dq_pool``): it replaces
    the per-element scalar LDS gather (``MFMA_LANE_K`` separate
    ``memref.load`` + ``vector.from_elements``) with two wide transpose
    reads, collapsing the on-chip LDS-issue serialization that sits ahead
    of the matrix core.

    Args:
        lane_base_ptr: per-lane LDS byte base pointer (``!llvm.ptr<3>``),
            already folded with the lane decomposition and the buffer's
            absolute byte offset by the caller.
        base_elem: per-call constexpr ELEMENT offset
            (``m_step*K_STEP*LDS_STRIDE + dc*D_CHUNK``); folds into the LDS
            instruction ``offset:`` immediate.
        lds_stride: row stride of the LDS buffer in elements.
        mfma_pack_type: target B-frag vector type (``vec(8, elem)``).
        v4_type: per-read result vector type (``vec(4, elem)``).

    The collapsed wide transpose reads feed a byte-identical MFMA B-pack
    (algebraically identical math); only instruction count / issue
    pressure changes.
    """
    byte_imm_0 = base_elem * 2
    byte_imm_1 = byte_imm_0 + 4 * lds_stride * 2
    gep0 = _llvm.GEPOp(_llvm_lds_ptr_ty(), lane_base_ptr, [],
                       rawConstantIndices=[byte_imm_0],
                       elem_type=T.i8, noWrapFlags=0)
    v_lo = rocdl.ds_read_tr16_b64(v4_type, gep0.result).result
    gep1 = _llvm.GEPOp(_llvm_lds_ptr_ty(), lane_base_ptr, [],
                       rawConstantIndices=[byte_imm_1],
                       elem_type=T.i8, noWrapFlags=0)
    v_hi = rocdl.ds_read_tr16_b64(v4_type, gep1.result).result
    v_full = vector.shuffle(v_lo, v_hi, [0, 1, 2, 3, 4, 5, 6, 7])
    return vector.bitcast(mfma_pack_type, v_full)
