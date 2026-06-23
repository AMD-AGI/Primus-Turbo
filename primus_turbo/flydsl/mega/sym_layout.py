###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``SymLayout`` -- FlyDSL port of ``deep_gemm::layout::Workspace`` + ``SymBuffer::map``.

A single symmetric workspace per rank (same layout, different base). The struct
carries the cross-rank delta table (``SymBuffer``) AND the sub-buffer layout
(``Workspace``), so one object both:

  * ``map(ptr, dst_rank)`` -> translate any local workspace ptr to peer ``dst_rank``
  * ``get_*_ptr(...)``      -> locate a specific sub-buffer (grid-sync counters,
                              NVLink barrier, expert counts, arrival flags, the
                              dispatch src-token-topk table, combine metadata)

FlyDSL specifics (see [[reference_flydsl_struct_kernel_arg]]):
  * the struct is passed to a kernel BY VALUE (fields flatten to scalar params).
  * shape members are ``Constexpr[int]`` -> folded into the type, readable as plain
    Python ints at trace time so the byte-offset math constant-folds.
  * ``offsets`` cannot be an inline ``Array`` by value, so it is carried as an i64
    address (``offsets_ptr``); ``map`` loads ``offsets_ptr[dst]`` (one i64 load).

Byte layout (mirrors mega_moe.cuh ``Workspace::get_num_bytes`` exactly):
  [ 32B barrier ][ send u64[E] ][ recv u64[E] ][ recv_sum u64[Epr] ]
  [ l1 u32[align(NPB,2)] ][ l2 u64[NPB] ][ src_topk i32[Epr*R*Trecv] ]
  [ token_src_meta (12B)[NMPT] ]  -> aligned to 16B
"""

import flydsl.expr as fx
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.prims import addr_buffer_resource

# ---- constants (from layout/mega_moe.cuh) ----
_MAX_BLOCK_M = 192
_MIN_BLOCK_M = 8
_LCM_BLOCK_M = 384
_BARRIER_BYTES = 32
_MAX_GRID_SYNC = 4
_SZ_U32, _SZ_U64, _SZ_META = 4, 8, 12  # TokenSrcMetadata = 3 x u32


def _align(x, a):
    return (x + a - 1) // a * a


def get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    """Worst-case shared pool capacity (mega_moe.cuh ``get_num_max_pool_tokens``)."""
    recv = num_ranks * num_max_tokens_per_rank
    m = min(num_topk, num_experts_per_rank)
    return _align(recv * m + num_experts_per_rank * (_MAX_BLOCK_M - 1), _LCM_BLOCK_M)


def compute_dims(num_ranks, num_experts, num_max_tokens_per_rank, num_topk):
    """Derive every SymLayout shape member from the four inputs (host)."""
    epr = num_experts // num_ranks
    nmpt = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, epr)
    npb = nmpt // _MIN_BLOCK_M
    return dict(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_experts_per_rank=epr,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_max_pool_tokens=nmpt,
        num_max_pool_blocks=npb,
    )


def _trecv(d):  # num_max_recv_tokens_per_expert
    return d["num_ranks"] * d["num_max_tokens_per_rank"]


# ---- compile-time sub-buffer byte offsets (pure functions of dims) ----
def _off_send(d):
    return _BARRIER_BYTES


def _off_recv(d):
    return _BARRIER_BYTES + d["num_experts"] * _SZ_U64


def _off_recv_sum(d):
    return _BARRIER_BYTES + d["num_experts"] * 2 * _SZ_U64


def _off_l1(d):
    return _BARRIER_BYTES + (d["num_experts"] * 2 + d["num_experts_per_rank"]) * _SZ_U64


def _off_l2(d):
    return _off_l1(d) + _align(d["num_max_pool_blocks"], 2) * _SZ_U32


def _off_src(d):
    return _off_l2(d) + d["num_max_pool_blocks"] * _SZ_U64


def _off_meta(d):
    return _off_src(d) + d["num_experts_per_rank"] * d["num_ranks"] * _trecv(d) * _SZ_U32


def get_num_bytes(d):
    """Total workspace bytes for one rank (mega_moe.cuh ``Workspace::get_num_bytes``)."""
    n = _off_meta(d) + d["num_max_pool_tokens"] * _SZ_META
    return _align(n, 16)


# ---------------------------------------------------------------------------
# The struct (passed to kernels by value)
# ---------------------------------------------------------------------------
@struct
class SymLayout:
    base: Int64  # this rank's workspace base address
    offsets_ptr: Int64  # i64* to per-peer delta table (peer_base[i] - base)
    rank_idx: Int32
    num_ranks: Constexpr[int]
    num_experts: Constexpr[int]
    num_experts_per_rank: Constexpr[int]
    num_max_tokens_per_rank: Constexpr[int]
    num_max_pool_tokens: Constexpr[int]
    num_max_pool_blocks: Constexpr[int]


def _dims(sl):
    """Read the Constexpr shape members off a SymLayout instance as Python ints."""
    return dict(
        num_ranks=int(sl.num_ranks),
        num_experts=int(sl.num_experts),
        num_experts_per_rank=int(sl.num_experts_per_rank),
        num_max_tokens_per_rank=int(sl.num_max_tokens_per_rank),
        num_max_pool_tokens=int(sl.num_max_pool_tokens),
        num_max_pool_blocks=int(sl.num_max_pool_blocks),
    )


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def _i64(x):
    """Sign-extend an fx i32 (or fold a python int) to an i64 ArithValue."""
    if isinstance(x, int):
        return fx.Int64(x)
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), x.ir_value()), signed=True)


def _addr(sl, const_bytes, index=0, stride=0):
    """``sl.base + const_bytes + index*stride`` as an i64 address."""
    addr = sl.base + fx.Int64(const_bytes)
    if stride and not (isinstance(index, int) and index == 0):
        addr = addr + _i64(index) * fx.Int64(stride)
    return addr


def map(sl, ptr, dst_rank):
    """Translate local workspace ptr ``ptr`` (i64) to peer ``dst_rank`` (SymBuffer::map).

    ``ptr + offsets[dst_rank]``; self -> identity (delta 0)."""
    res = addr_buffer_resource(sl.offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())


# barrier region (first 32B): 4 grid-sync counters, 1 NVLink counter, 2 NVLink signals
def get_grid_sync_count_ptr(sl, index=0):
    return _addr(sl, 0, index, _SZ_U32)


def get_nvl_barrier_counter_ptr(sl):
    return _addr(sl, _MAX_GRID_SYNC * _SZ_U32)


def get_nvl_barrier_signal_ptr(sl, phase):
    return _addr(sl, (_MAX_GRID_SYNC + 1) * _SZ_U32, phase, _SZ_U32)


# expert count tables (u64)
def get_expert_send_count_ptr(sl, expert=0):
    return _addr(sl, _off_send(_dims(sl)), expert, _SZ_U64)


def get_expert_recv_count_ptr(sl, rank=0, expert=0):
    d = _dims(sl)
    idx = _i64(rank) * fx.Int64(d["num_experts_per_rank"]) + _i64(expert)
    return sl.base + fx.Int64(_off_recv(d)) + idx * fx.Int64(_SZ_U64)


def get_expert_recv_count_sum_ptr(sl, expert=0):
    return _addr(sl, _off_recv_sum(_dims(sl)), expert, _SZ_U64)


# arrival flags
def get_l1_arrival_count_ptr(sl, pool_block_idx=0):
    return _addr(sl, _off_l1(_dims(sl)), pool_block_idx, _SZ_U32)


def get_l2_arrival_mask_ptr(sl, pool_block_idx=0):
    return _addr(sl, _off_l2(_dims(sl)), pool_block_idx, _SZ_U64)


# dispatch-pull src token-topk table: idx = e*(R*Trecv) + rank*Trecv + token
def get_src_token_topk_idx_ptr(sl, expert=0, rank=0, token=0):
    d = _dims(sl)
    trecv = _trecv(d)
    idx = _i64(expert) * fx.Int64(d["num_ranks"] * trecv) + _i64(rank) * fx.Int64(trecv) + _i64(token)
    return sl.base + fx.Int64(_off_src(d)) + idx * fx.Int64(_SZ_U32)


# combine write-back metadata (TokenSrcMetadata, 12B each)
def get_token_src_metadata_ptr(sl, pool_token_idx=0):
    return _addr(sl, _off_meta(_dims(sl)), pool_token_idx, _SZ_META)
