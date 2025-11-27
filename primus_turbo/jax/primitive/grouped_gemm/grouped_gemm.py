###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import xla

from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
grouped_gemm_p = Primitive("grouped_gemm")
grouped_gemm_p.multiple_results = False

grouped_gemm_variable_k_p = Primitive("grouped_gemm_variable_k")
grouped_gemm_variable_k_p.multiple_results = False

compute_group_offs_p = Primitive("compute_group_offs")
compute_group_offs_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[grouped_gemm_p] = partial(xla.apply_primitive, grouped_gemm_p)
IMPL_TABLE[grouped_gemm_variable_k_p] = partial(xla.apply_primitive, grouped_gemm_variable_k_p)
IMPL_TABLE[compute_group_offs_p] = partial(xla.apply_primitive, compute_group_offs_p)


# ----------------------------------------
# Helper function to compute workspace size
# ----------------------------------------
def get_ck_grouped_gemm_args_sizes(group_num):
    """Get the workspace size needed for grouped_gemm CK kernel.

    Args:
        group_num: Number of groups (batch size)

    Returns:
        Required workspace size in bytes
    """
    # This matches the C++ implementation:
    # group_num * sizeof(ck_tile::GemmTransKernelArg<>)
    #
    # GemmTransKernelArg contains:
    # - UniversalGemmKernelArgs (contains pointers and dimensions)
    # - 2 index_t fields (block_start, block_end)
    #
    # Conservative estimate based on CK structure:
    # ~200 bytes per group for typical configuration
    return group_num * 256  # Use 256 bytes to be safe


def get_ck_grouped_gemm_fp8_args_sizes(group_num):
    """Get the workspace size needed for grouped_gemm_fp8 CK kernel.

    Args:
        group_num: Number of groups (batch size)

    Returns:
        Required workspace size in bytes
    """
    # FP8 kernels use similar argument structures as regular grouped_gemm
    return group_num * 256


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_abstract_eval(a, b, group_lens, group_offs, workspace, transA, transB, num_cu):
    """Abstract evaluation for grouped_gemm.

    Args:
        a: Input tensor A with shape [m, k] or [k, m] if transA
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if transB
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        workspace: Workspace buffer for CK kernel
        transA: Whether A is transposed
        transB: Whether B is transposed
        num_cu: Number of compute units

    Returns:
        Output tensor with shape [m, n]
    """
    assert a.dtype == b.dtype, "dtype mismatch between a and b"

    # Calculate output shape based on transpose flags
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    return ShapedArray((m, n), a.dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_p] = _grouped_gemm_abstract_eval


def _compute_group_offs_abstract_eval(group_lens):
    """Abstract evaluation for compute_group_offs.

    Args:
        group_lens: Group lengths tensor [bs]

    Returns:
        Group offsets tensor [bs + 1]
    """
    bs = group_lens.shape[0]
    return ShapedArray((bs + 1,), group_lens.dtype)


ABSTRACT_EVAL_TABLE[compute_group_offs_p] = _compute_group_offs_abstract_eval


def _grouped_gemm_variable_k_abstract_eval(a, b, group_lens, group_offs, workspace, transA, transB, num_cu):
    """Abstract evaluation for grouped_gemm_variable_k.

    Note: Only supports transA=True, transB=False

    Args:
        a: Input tensor A with shape [k, m] (will be transposed)
        b: Input tensor B with shape [k, n]
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        workspace: Workspace buffer for CK kernel
        transA: Must be True
        transB: Must be False
        num_cu: Number of compute units

    Returns:
        Output tensor with shape [bs, m, n]
    """
    assert a.dtype == b.dtype, "dtype mismatch between a and b"
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    # For transA=True, transB=False:
    # a: [k, m] (will be transposed to [m, k]), b: [k, n]
    # a.T @ b = [m, k] @ [k, n] = [m, n] for each group
    # output: [bs, m, n]
    bs = group_lens.shape[0]
    m = a.shape[1]
    n = b.shape[1]

    return ShapedArray((bs, m, n), a.dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_variable_k_p] = _grouped_gemm_variable_k_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[grouped_gemm_p] = jax.ffi.ffi_lowering("grouped_gemm")

LOWERING_TABLE[grouped_gemm_variable_k_p] = jax.ffi.ffi_lowering("grouped_gemm_variable_k")

LOWERING_TABLE[compute_group_offs_p] = jax.ffi.ffi_lowering("compute_group_offs")


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "grouped_gemm_p",
    "grouped_gemm_variable_k_p",
    "compute_group_offs_p",
    "get_ck_grouped_gemm_args_sizes",
]
