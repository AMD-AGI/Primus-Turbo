###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-heap symmetric-workspace descriptor for the mega-MoE FlyDSL kernels.

A single ``SymLayout`` struct (passed to kernels by value) names every sub-buffer
the mega-MoE pipeline touches across TWO symmetric heaps:

  * the cached MAIN heap (``base`` / ``offsets_ptr``): pool, c_buffer, the cross-rank
    origin tables, weights, device scalars, GEMM intermediates;
  * the uncached SIGNAL heap (``signal_base`` / ``signal_offsets_ptr``): spin-wait
    flags + the combine buffer (must bypass the cache, see mega_moe_fused).

Region byte offsets are recomputed from the struct's Constexpr dims and MUST match
the host ``SymmBuffer`` allocation (same region order + 256B alignment). Each
``offsets_ptr`` is an ``i64[num_ranks]`` table of per-peer base DELTAS
(``peer_base[i] - my_base``); ``get_*_ptr(sl, ..., dst_rank=)`` adds the delta to
translate a local sub-buffer address into peer ``dst_rank`` (SymBuffer::map).
"""

import flydsl.expr as fx
from flydsl.expr import Int32, Int64, struct
from flydsl.expr.buffer_ops import buffer_load
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.mega.fp8.prims import addr_buffer_resource

# ---- constants (from layout/mega_moe.cuh) ----
_MAX_BLOCK_M = 192
_MIN_BLOCK_M = 8
_LCM_BLOCK_M = 384
_BASE_ALIGNMENT = 256
_BF16, _I32, _F32, _I64 = 2, 4, 4, 8  # element byte sizes


def _align(x, a=_BASE_ALIGNMENT):
    return (x + a - 1) // a * a


def get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    """Worst-case shared pool capacity (mega_moe.cuh ``get_num_max_pool_tokens``)."""
    recv = num_ranks * num_max_tokens_per_rank
    m = min(num_topk, num_experts_per_rank)
    return _align(recv * m + num_experts_per_rank * (_MAX_BLOCK_M - 1), _LCM_BLOCK_M)


# ---------------------------------------------------------------------------
# The struct (passed to kernels by value) -- the single source of truth.
# ---------------------------------------------------------------------------
@struct
class SymLayout:
    base: Int64  # this rank's MAIN (cached) heap base address
    offsets_ptr: Int64  # i64[num_ranks] MAIN per-peer delta table (peer_base - base)
    signal_base: Int64  # this rank's SIGNAL (uncached) heap base address
    signal_offsets_ptr: Int64  # i64[num_ranks] SIGNAL per-peer delta table
    rank_idx: Int32
    num_ranks: Constexpr[int]
    num_experts: Constexpr[int]
    num_experts_per_rank: Constexpr[int]
    num_max_tokens_per_rank: Constexpr[int]
    num_topk: Constexpr[int]
    hidden: Constexpr[int]
    intermediate_hidden: Constexpr[int]
    num_max_pool_tokens: Constexpr[int]  # pool capacity (rows)
    num_max_pool_blocks: Constexpr[int]  # num_max_pool_tokens // block_m
    combine_slots: Constexpr[int]  # num_topk * num_max_tokens_per_rank
    # mxfp8 forward: append fp8 pool/act data + raw E8M0 block-scale regions (1 = on).
    # Appended AFTER every bf16 region so the bf16 (use_mxfp8=0) byte layout is unchanged.
    use_mxfp8: Constexpr[int]


# ---------------------------------------------------------------------------
# Memory layout: two heaps, each a 256B-aligned region packer. The region order
# MUST mirror ``mega_moe_fused`` (``main`` list and ``_signal_regions``).
# ---------------------------------------------------------------------------
def _main_regions(sl):
    R, E = int(sl.num_ranks), int(sl.num_experts)
    P, H, I = int(sl.num_max_pool_tokens), int(sl.hidden), int(sl.intermediate_hidden)
    CS = int(sl.combine_slots)
    EPR, T = int(sl.num_experts_per_rank), int(sl.num_max_tokens_per_rank)
    regions = [
        ("pool", _BF16, P * H),
        ("c_buffer", _I32, R * E),
        ("signal", _I32, R),
        ("origin_rank", _I32, P),
        ("origin_slot", _I32, P),
        ("weight_recv_buf", _F32, P),
        # token-dedup map2 (dense_to_expert): secondary dest slot -> primary slot to
        # copy from (-1 = primary). Source rank writes it cross-rank -> symmetric.
        ("dedup_src_row", _I32, P),
        ("combine_gate", _F32, CS),
        ("meta_scalars", _I32, 8),
        ("grid_sync_count", _I32, 2),
        ("profile", _I64, 8),
        ("act", _BF16, P * I),
        ("l2_token_buffer", _BF16, P * H),
        # DG dispatch index (dest-side): src_token_topk_idx[le, src_rank, slot] = token*K+k.
        # Source rank scatters cross-rank into the dest -> symmetric. Appended LAST so
        # every preceding region keeps its byte offset.
        ("src_token_topk_idx", _I32, EPR * R * R * T),
    ]
    # mxfp8 forward-only regions (fp8 = 1B/elem, E8M0 scale = 1B / 32 K-elems). Appended
    # last (offset-stable). Pushed cross-rank by dispatch (pool_*) / written by SwiGLU
    # (act_*); read as A operands by the grouped mxfp8 GEMM (which preshuffles internally).
    if int(getattr(sl, "use_mxfp8", 0)):
        # pool_scale_ps: fused quant-in-push writes the pool E8M0 scale directly in the
        # ScaleS2R broadcast layout-1 (int32, ceildiv(P,64)*(H//128)*256), so the fused L1
        # GEMM reads it with ScaleS2R (no preshuffle pass). The raw ``pool_scale`` is kept
        # for the decoupled fp8 path (push raw -> grouped GEMM preshuffles internally).
        ps_i32 = ((P + 63) // 64) * (H // 128) * 256
        regions += [
            ("pool_fp8", 1, P * H),
            ("pool_scale", 1, P * (H // 32)),
            ("act_fp8", 1, P * I),
            ("act_scale", 1, P * (I // 32)),
            ("pool_scale_ps", _I32, ps_i32),
        ]
    return regions


def _signal_regions(sl):
    R, NPB = int(sl.num_ranks), int(sl.num_max_pool_blocks)
    CS, H = int(sl.combine_slots), int(sl.hidden)
    # All epoch flags (bf16-style self-reset): 2 banks (parity) x length, i64. Never host-reset;
    # each spins on a cumulative per-bank expected -> no consuming store, no cross-call reset race.
    return [
        ("_ipc_barrier", _I32, R),
        ("dispatch_flag", _I64, 2 * NPB),    # cross-rank comm->preshuffle gate (per-expert, atomic_add)
        ("preshuffle_flag", _I64, 2 * NPB),  # local preshuffle->gemm gate (per-block, st=expected)
        ("combine_flag", _I64, 2 * NPB),     # L2 GEMM->combine gate (per-block, atomic_add)
        ("comb", _BF16, CS * H),
        ("reduce_flag", _I64, 2 * CS),       # L2 combine->reduce gate (per-slot, st=expected)
    ]


def _pack(regions):
    """256B-aligned packer -> ({name: (offset, itemsize, numel)}, total_bytes)."""
    offsets, cursor = {}, 0
    for name, item, numel in regions:
        cursor = _align(cursor)
        offsets[name] = (cursor, item, numel)
        cursor += numel * item
    return offsets, _align(cursor)


def layout(sl):
    """Both heaps' offset maps + totals: (main_off, sig_off, main_bytes, sig_bytes).

    Each offset map is ``{name: (byte_offset, itemsize, numel)}`` -- the single source
    of truth for region order / sizes (the host ``SymmBuffer`` builds its views from it)."""
    main_off, main_total = _pack(_main_regions(sl))
    sig_off, sig_total = _pack(_signal_regions(sl))
    return main_off, sig_off, main_total, sig_total


# host-facing layout views (used by tests / sizing)
def num_bytes(sl):
    """(main_bytes, signal_bytes) for one rank's two heaps."""
    _, _, mb, sb = layout(sl)
    return mb, sb


def region_offset(sl, name):
    """(byte offset, itemsize, heap) of a named region."""
    main_off, sig_off, _, _ = layout(sl)
    if name in main_off:
        off, item, _ = main_off[name]
        return off, item, "main"
    off, item, _ = sig_off[name]
    return off, item, "signal"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def _as_i64(x):
    """Sign-extend an fx i32 (or fold a python int) to an i64 ArithValue."""
    if isinstance(x, int):
        return fx.Int64(x)
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), x.ir_value()), signed=True)


def _region_ptr(sl, name, index=0, dst_rank=None):
    """Address (i64) of ``region[index]``; ``dst_rank`` translates into that peer."""
    main_off, sig_off, _, _ = layout(sl)
    if name in main_off:
        off, item, _ = main_off[name]
        base, offsets_ptr = sl.base, sl.offsets_ptr
    else:
        off, item, _ = sig_off[name]
        base, offsets_ptr = sl.signal_base, sl.signal_offsets_ptr
    addr = base + fx.Int64(off)
    if not (isinstance(index, int) and index == 0):
        addr = addr + _as_i64(index) * fx.Int64(item)
    if dst_rank is not None:
        res = addr_buffer_resource(offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
        addr = addr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())
    return addr


def sym_map(sl, ptr, dst_rank):
    """Translate a local MAIN-heap ptr ``ptr`` (i64) into peer ``dst_rank``."""
    res = addr_buffer_resource(sl.offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())


def map_signal(sl, ptr, dst_rank):
    """Translate a local SIGNAL-heap ptr ``ptr`` (i64) into peer ``dst_rank``."""
    res = addr_buffer_resource(sl.signal_offsets_ptr, num_records_bytes=int(sl.num_ranks) * 8)
    return ptr + buffer_load(res, dst_rank, vec_width=1, dtype=fx.T.i64())


# ---------------------------------------------------------------------------
# Host builder
# ---------------------------------------------------------------------------
def build_sym_layout(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    num_max_pool_tokens,
    num_max_pool_blocks,
    combine_slots,
    *,
    use_mxfp8=0,
    base=0,
    offsets_ptr=0,
    signal_base=0,
    signal_offsets_ptr=0,
    rank_idx=0,
):
    """Build a ``SymLayout`` from concrete dims + the two heaps' base/delta-table ptrs.

    Pool sizes (``num_max_pool_tokens`` / ``num_max_pool_blocks`` / ``combine_slots``)
    are passed explicitly so the layout matches whatever the host ``SymmBuffer``
    actually allocated (which derives them from ``block_m`` / ``pool_mult``)."""
    return SymLayout(
        base=Int64(base),
        offsets_ptr=Int64(offsets_ptr),
        signal_base=Int64(signal_base),
        signal_offsets_ptr=Int64(signal_offsets_ptr),
        rank_idx=Int32(rank_idx),
        num_ranks=int(num_ranks),
        num_experts=int(num_experts),
        num_experts_per_rank=int(num_experts) // int(num_ranks),
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        num_topk=int(num_topk),
        hidden=int(hidden),
        intermediate_hidden=int(intermediate_hidden),
        num_max_pool_tokens=int(num_max_pool_tokens),
        num_max_pool_blocks=int(num_max_pool_blocks),
        combine_slots=int(combine_slots),
        use_mxfp8=int(use_mxfp8),
    )


# ---------------------------------------------------------------------------
# Convenience accessors: ``sym_layout.<region>_ptr`` -> this rank's i64 base ptr of that
# region (a read-only property, so it is safe to read inside scf if/while regions -- a
# struct method call would otherwise be treated as a state variable). Peer translation
# uses ``sym_map(sym_layout, sym_layout.<region>_ptr, dst_rank)`` (``map_signal`` for the
# signal heap). (_specialize_type subclasses SymLayout, so these properties are inherited
# by the per-shape specialized instances.) Names must match _main_regions / _signal_regions.
# ---------------------------------------------------------------------------
_REGION_ACCESSORS = (
    "pool",
    "c_buffer",
    "signal",
    "origin_rank",
    "origin_slot",
    "weight_recv_buf",
    "dedup_src_row",
    "combine_gate",
    "meta_scalars",
    "grid_sync_count",
    "profile",
    "act",
    "l2_token_buffer",
    "src_token_topk_idx",
    # mxfp8-only (the property raises if read when use_mxfp8=0, i.e. the region is absent)
    "pool_fp8",
    "pool_scale",
    "act_fp8",
    "act_scale",
    "pool_scale_ps",
    "dispatch_flag",
    "preshuffle_flag",
    "combine_flag",
    "comb",
    "reduce_flag",
)


def _make_region_accessor(region_name):
    def _get(self):
        return _region_ptr(self, region_name)

    _get.__name__ = region_name
    _get.__doc__ = f"This rank's i64 base ptr of the '{region_name}' region."
    return property(_get)


for _region_name in _REGION_ACCESSORS:
    setattr(SymLayout, f"{_region_name}_ptr", _make_region_accessor(_region_name))
