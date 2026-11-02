###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Device helpers shared by the FlyDSL DeepEP intranode kernels.

The helpers whose code differs on gfx1250 (``read_clock``, ``wait_mem_all``, ``copy_warp``) are
DeepEP's own, picked by ``TRAITS``, so ``primus_turbo.flydsl.utils`` stays as the gfx9 kernels use it.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.primitive import get_dyn_shared
from flydsl.expr.primitive import ptrtoint as _fly_ptrtoint

from primus_turbo.flydsl.deep_ep.arch import TRAITS
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    extract_base_index,
)
from primus_turbo.flydsl.utils.prims import cast, ld, st

IS_GFX1250 = TRAITS.name == "gfx1250"
WARP_SIZE = TRAITS.wave_size
SYS_LOAD = TRAITS.cpol_sys_load
SYS_STORE = TRAITS.cpol_sys_store

_COPY_VEC_I32 = 4  # 4 i32 = 16 B (b128) per lane per step
_I32_BYTES = 4


def _asm(text):
    llvm.inline_asm(res=None, operands_=[], asm_string=text, constraints="~{memory}", has_side_effects=True)


def read_clock():
    """The realtime counter (100 MHz) for the spin-wait watchdogs; unsigned so deltas compare right."""
    # gfx12 dropped s_memrealtime and reads the counter through a returning sendmsg
    if IS_GFX1250:
        asm = "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0"
    else:
        asm = "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)"
    op = llvm.inline_asm(fx.T.i64, [], asm, "=s", has_side_effects=True)
    return fx.arith.ArithValue(op, signed=False)


def timed_out(spin_start):
    """The spin-wait watchdog's predicate, after ``ArchTraits.spin_timeout_cycles``."""
    return (read_clock() - spin_start) > fx.Int64(TRAITS.spin_timeout_cycles)


def wait_mem_all():
    """Retire every outstanding memory op of this wave, stores included (e.g. before a flag publish)."""
    # gfx12 split the unified counter: stores retire on storecnt, which vmcnt no longer covers
    if IS_GFX1250:
        _asm("s_wait_dscnt 0\n\ts_wait_kmcnt 0\n\ts_wait_loadcnt 0\n\ts_wait_storecnt 0")
    else:
        rocdl.s_waitcnt(0)


def copy_warp(dst, src, nbytes, dst_off=0, src_off=0, load_cache_modifier=0, store_cache_modifier=0):
    """``prims.copy_warp`` over the lanes of one hardware wave (``WARP_SIZE``) instead of 64: all loads
    of the copy, then all stores. ``dst`` / ``src`` are buffer resources or i64 addresses, the offsets
    i32 words; ``nbytes`` is a multiple of 16."""

    def _addr_i64(addr):
        if isinstance(addr, int):
            return fx.Int64(addr)
        v = fx.as_ir_value(addr)
        if isinstance(v.type, ir.IndexType):
            v = std_arith.IndexCastOp(fx.T.i64, v).result
        elif isinstance(v.type, ir.IntegerType) and v.type.width < 64:
            v = std_arith.ExtSIOp(fx.T.i64, v).result
        return fx.arith.ArithValue(v, signed=True)

    def _copy_operand(operand, word_off, nbytes):
        if "ptr" in str(fx.as_ir_value(operand).type):
            return operand, fx.Int32(word_off) if isinstance(word_off, int) else word_off
        base = _addr_i64(operand) + _addr_i64(word_off) * fx.Int64(_I32_BYTES)
        return create_buffer_resource_from_addr(base, num_records_bytes=nbytes), fx.Int32(0)

    assert nbytes % (_COPY_VEC_I32 * _I32_BYTES) == 0, "copy_warp nbytes must be a multiple of 16"
    cols = WARP_SIZE * _COPY_VEC_I32
    num_full_chunks = nbytes // _I32_BYTES // cols
    tail_words = nbytes // _I32_BYTES - num_full_chunks * cols
    # A ragged tail parks its idle lanes' stores one row past the end, which only the
    # num_records of an address-built dst resource turns into a dropped store.
    assert not tail_words or "ptr" not in str(fx.as_ir_value(dst).type), (
        "a ragged copy needs dst as an address"
    )
    src, src_off = _copy_operand(src, src_off, nbytes)
    dst, dst_off = _copy_operand(dst, dst_off, nbytes)
    lane_off = (fx.thread_idx.x % fx.Int32(WARP_SIZE)) * fx.Int32(_COPY_VEC_I32)
    offs = [fx.Int32(c * cols) + lane_off for c in range_constexpr(num_full_chunks)]
    load_offs = list(offs)
    if tail_words:
        in_tail = lane_off < fx.Int32(tail_words)
        tail_off = fx.Int32(num_full_chunks * cols) + lane_off
        load_offs.append(arith.select(in_tail, tail_off, fx.Int32(0)))
        offs.append(arith.select(in_tail, tail_off, fx.Int32(nbytes // _I32_BYTES)))
    vals = [
        buffer_load(
            src, src_off + o, vec_width=_COPY_VEC_I32, dtype=fx.T.i32, cache_modifier=load_cache_modifier
        )
        for o in load_offs
    ]
    for o, v in zip(offs, vals):
        buffer_store(v, dst, dst_off + o, cache_modifier=store_cache_modifier)


def timeout_hit():
    """After a watchdog's report: abort the queue where ``ArchTraits.trap_on_timeout`` says so."""
    if TRAITS.trap_on_timeout:
        _asm("s_trap 2")


def load_index(res, elem_idx, element_type):
    """Load one i32 / i64 index element as i32."""
    value = buffer_load(res, elem_idx, vec_width=1, dtype=element_type)
    if element_type.width == 64:
        value = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32, fx.as_ir_value(value)), signed=True)
    return value


def load_i32(res, idx):
    return buffer_load(res, idx, vec_width=1, dtype=fx.T.i32)


def tensor_addr(tensor):
    """Base address of a kernel tensor argument as i64."""
    return cast(fx.arith.ArithValue(extract_base_index(tensor, address_space=1)), fx.T.i64)


def lds_base():
    """Base address of the dynamic LDS allocation."""
    return fx.as_ir_value(_fly_ptrtoint(get_dyn_shared()))


def ld_lds(base, idx):
    return ld(base, fx.Int32(idx) if isinstance(idx, int) else idx, scope="workgroup", space=3)


def st_lds(base, idx, value):
    st(base, fx.Int32(idx) if isinstance(idx, int) else idx, value, scope="workgroup", space=3)
