###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl
from triton.language.standard import _log2


@triton.jit
def _to_int_bitcast(val, signed: tl.constexpr = True):
    """Bitcast a value to signed/unsigned integer type matching its primitive bitwidth."""
    if signed:
        if val.dtype.primitive_bitwidth == 8:
            return val.to(tl.int8, bitcast=True)
        elif val.dtype.primitive_bitwidth == 16:
            return val.to(tl.int16, bitcast=True)
        elif val.dtype.primitive_bitwidth == 32:
            return val.to(tl.int32, bitcast=True)
        else:
            return val.to(tl.int64, bitcast=True)
    else:
        if val.dtype.primitive_bitwidth == 8:
            return val.to(tl.uint8, bitcast=True)
        elif val.dtype.primitive_bitwidth == 16:
            return val.to(tl.uint16, bitcast=True)
        elif val.dtype.primitive_bitwidth == 32:
            return val.to(tl.uint32, bitcast=True)
        else:
            return val.to(tl.uint64, bitcast=True)


@triton.jit
def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    z = tl.reshape(indices, shape)
    mask = tl.arange(0, 2)[None, :, None]
    l_value = tl.reshape(tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape).to(x.dtype)
    r_value = tl.reshape(tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape).to(x.dtype)
    l_indice = tl.reshape(tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape)
    r_indice = tl.reshape(tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape)
    # Bitcast to int based on value's primitive bitwidth
    il_value = _to_int_bitcast(l_value, signed=True)
    ir_value = _to_int_bitcast(r_value, signed=True)
    ix = _to_int_bitcast(x, signed=True)
    flag1 = tl.where(((l_value > r_value) ^ flip) != 0, il_value ^ ir_value, tl.zeros_like(ix))
    ret = ix ^ flag1
    flag2 = tl.where(((l_value > r_value) ^ flip) != 0, l_indice ^ r_indice, tl.zeros_like(ix))
    ind = indices ^ flag2
    return ret.to(x.dtype, bitcast=True), ind


@triton.jit
def _bitonic_merge(x, indices, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    if order == 2:
        shape: tl.constexpr = [n_outer * (2 ** (n_dims - 1 - stage)), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = tl.full(x.shape, value=order, dtype=tl.int32)
    for i in tl.static_range(stage):
        x, indices = _compare_and_swap(x, indices, flip, i + (n_dims - stage), n_dims)
    return x, indices


@triton.jit
def argsort(x, ids, dim: tl.constexpr = None, descending: tl.constexpr = 0):
    """
    Perform argsort on the input tensor using bitonic sort.

    Parameters
    ----------
    x: tensor
        The input tensor to sort.
    ids: tensor
        The indices tensor to be permuted along with x.
    dim: tl.constexpr
        The dimension to sort along. Only the last dimension is currently supported.
        If None, defaults to the last dimension.
    descending: tl.constexpr
        If 0 (default), sort in ascending order. If 1 (or True), sort in descending order.

    Returns
    -------
    sorted_x: tensor
        The sorted tensor.
    sorted_ids: tensor
        The permuted indices tensor.
    """
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])
    # Convert descending to int (handle both bool True and int 1)
    _descending: tl.constexpr = 1 if descending else 0

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else _descending, n_dims)
    return x, ids
