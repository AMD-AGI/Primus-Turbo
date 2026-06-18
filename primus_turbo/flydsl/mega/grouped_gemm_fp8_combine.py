###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused grouped FP8 GEMM + cross-rank combine PUSH (the K2 mirror of the K1
dispatch fusion), FlyDSL.

Role-specialized single kernel, opposite producer/consumer to the dispatch one:
the ``block_index >= comb_blocks`` blocks each compute one output tile of the
grouped FP8 GEMM (``A=act[M,K]`` fp8, per-expert ``B=weight`` fp8 -> ``L2Y[M,N]``
bf16) and signal a per-pool-block local scoreboard; the ``block_index <
comb_blocks`` blocks spin on the scoreboard and, once a pool block's N-tiles are
all done, push its finished L2Y rows back to their origin ranks' combine buffers
(indexed by per-row origin_rank / origin_slot). The combine latency is hidden
under the MFMA-bound GEMM.

This is packaged as ``CombineGroupFP8TileSpec(GroupFp8TileSpec)`` (the K2 mirror of
``DispatchGroupFP8TileSpec``): the fused launcher in ``_compile`` emits BOTH roles --
the GEMM tile (the GEMM role reuses ``spec.emit(...)`` unchanged, then signals the
local L2 scoreboard) and the combine push (the ``combine_tile`` closure, the ONLY
kernel-specific code). The base ``GroupFp8TileSpec`` supplies the grouped seam
(LINEAR no-sync scheduler + per-expert ``group_base`` slab) over ``common.tile_spec``.

``combine_tile`` stays a closure inside the fused kernel body in ``_compile``
(not a spec method): its dynamic ``while``-spin / ``for`` / ``if`` only lower under
the kernel AST rewriter. The push moves bf16 L2Y rows (16-byte vec, legal), so no
i32 repack is needed."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.mega_group_tile_spec import (
    _LAYOUT_AGPR,
    _atomic_add,
    _fence_acquire,
    _fence_release,
    _ld_relaxed,
    BLOCK_K,
    GroupFp8TileSpec,
    as_i8_flat as _as_i8_flat,
    scalar_f32 as _scalar,
    weight_layout as _weight_layout,
)
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs

# Scoreboard prims are shared from group_tile_spec (imported above); only
# combine_tile (the comm push) stays kernel-specific.

_VEC = 8                # bf16 elems per lane per push step (16 bytes, legal)
_PUSH_LANES = 256       # combine push lanes per row
_BLOCK_THREADS = 512    # 8 waves — the tile-spec block size


# ──────────────────────────────────────────────────────────────────────
# Fused scatter epilogue (Phase 2): the GEMM tile's accumulator is scattered
# straight into the per-row peer combine buffer, removing the L2Y round-trip.
# A row's destination = comb_addrs[origin_rank[row]] at slot origin_slot[row].
# ──────────────────────────────────────────────────────────────────────
class ScatterStoreC:
    """Scatter the scaled output tile into the per-row peer combine buffer instead
    of a local C store. Mirrors ``StoreCPerTensor.store``'s (ti,tj,i) lane->row map
    (row = base_row + ti*16 + (lane//16)*4 + i); the destination is resolved per row
    via the origin/slot/peer-addr buffers. No control flow: an OOB index drops the
    no-origin / column-tail elements (hardware num-records clamp)."""

    def __init__(self, *, A_scale, B_scale, c_idx_fn, n_tiles_a, n_tiles_b, out_ty,
                 out_features, comb_records, n_slots,
                 origin_rank_res, origin_slot_res, comb_addr_res, elem_fn=None):
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        self.out_features = out_features
        self.comb_records = comb_records
        self.oob = n_slots * out_features        # one past the buffer -> dropped
        self.origin_rank_res = origin_rank_res
        self.origin_slot_res = origin_slot_res
        self.comb_addr_res = comb_addr_res
        self.elem_fn = elem_fn
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def store(self, c_frag, base_row, base_col):
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div)
        out_features = fx.Int32(self.out_features)
        oob = fx.Int32(self.oob)
        for ti in range_constexpr(self.n_tiles_a):
            row_base = base_row + ti * 16 + (self.lane_id // 16) * 4
            for i in range_constexpr(4):
                row = row_base + i
                origin = buffer_load(self.origin_rank_res, row, vec_width=1, dtype=fx.T.i32())
                slot = buffer_load(self.origin_slot_res, row, vec_width=1, dtype=fx.T.i32())
                origin_ok = origin >= fx.Int32(0)
                addr_idx = arith.select(origin_ok, origin, fx.Int32(0))
                addr = buffer_load(self.comb_addr_res, addr_idx, vec_width=1, dtype=fx.T.i64())
                peer = create_buffer_resource_from_addr(addr, num_records_bytes=self.comb_records)
                slot_base = slot * out_features
                for tj in range_constexpr(self.n_tiles_b):
                    col = base_col + tj * 16 + self.lane_id % 16
                    val = Vec(c_frag[self.c_idx_fn(ti, tj)])[i] * scale
                    if self.elem_fn is not None:
                        val = self.elem_fn(val)
                    scaled = val.to(self.out_ty)
                    idx_ok = arith.select(col < out_features, slot_base + col, oob)
                    idx = arith.select(origin_ok, idx_ok, oob)
                    buffer_store(scaled, peer, idx)


class ScatterCombineEpilogue:
    """``EpilogueSpec`` that scatters the tile to peer combine buffers (no local C /
    L2Y). ``epilogue_ctx`` carries the runtime scatter buffers (built in the kernel
    body). ``consume`` reuses the per-quadrant base-offset map of PerTensorEpilogue."""

    cache_key = ("scatter_combine",)

    def __init__(self, *, out_fp16=False):
        self.out_ty = fx.Float16 if out_fp16 else fx.BFloat16

    def build(self, geom, *, A_scale, B_scale, C, c_m, c_n, mfma, epilogue_ctx=None):
        ctx = epilogue_ctx
        return ScatterStoreC(
            A_scale=A_scale, B_scale=B_scale, c_idx_fn=mfma.idx,
            n_tiles_a=geom.N_TILES_A, n_tiles_b=geom.N_TILES_B, out_ty=self.out_ty,
            out_features=ctx["out_features"], comb_records=ctx["comb_records"],
            n_slots=ctx["n_slots"], origin_rank_res=ctx["origin_rank_res"],
            origin_slot_res=ctx["origin_slot_res"], comb_addr_res=ctx["comb_addr_res"],
        )

    def consume(self, geom, *, store_c, accum, epilogue_ctx=None):
        N_TILES_A, N_TILES_B = geom.N_TILES_A, geom.N_TILES_B
        LDS_BLOCK_M, LDS_BLOCK_N = geom.LDS_BLOCK_M, geom.LDS_BLOCK_N
        wave_n_offset = accum.wave_n * (N_TILES_B * 16)
        wave_m_offset = accum.wave_m * (N_TILES_A * 16)
        base_row = accum.block_m * geom.BLOCK_M + wave_m_offset
        base_col = accum.block_n * geom.BLOCK_N + wave_n_offset
        store_c.store(accum.c00, base_row + 0, base_col + 0)
        store_c.store(accum.c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(accum.c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(accum.c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


class CombineGroupFP8TileSpec(GroupFp8TileSpec):
    """Fused grouped FP8 GEMM + combine PUSH, expressed as a ``GroupFp8TileSpec``
    subclass. The GEMM tile reuses the parent ``emit``/``schedule``/``base_offsets``
    UNCHANGED; the fused launcher (in ``_compile``) emits BOTH roles in one kernel --
    the GEMM tile (via ``spec.emit``, which signals the local L2 scoreboard when done)
    and the combine push (``combine_tile``, which spins then pushes finished rows
    cross-rank). Only ``combine_tile`` is kernel-specific.

    Extra construction config (vs the base): ``out_features`` (-> n_blocks / push
    geometry / grid), ``pool_capacity`` (-> the no-sync over-launch bound),
    ``combine_slots`` (-> the peer combine-buffer record bound)."""

    def __init__(self, *, out_features, pool_capacity, combine_slots, **kw):
        super().__init__(**kw)
        self.out_features = out_features
        self.pool_capacity = pool_capacity
        self.combine_slots = combine_slots
        self.kernel_name = "combine_grouped_" + self.layout
        # Swap the local-C epilogue for the fused scatter-to-peer epilogue (the GEMM
        # tile writes straight into the peers' combine buffers -> no L2Y round-trip).
        self.epilogue_spec = ScatterCombineEpilogue(out_fp16=False)
        # rebuild cache_tag (scatter epilogue's cache_key folds in) + combine config
        self.cache_tag = self._assemble_cache_tag() + (out_features, pool_capacity, combine_slots)


@functools.lru_cache(maxsize=256)
def make_combine_tile_spec(*, layout, K, out_features, pool_capacity, combine_slots,
                           BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3, num_comm_blocks=0,
                           vmcnt_hint=2):
    """Cached ``CombineGroupFP8TileSpec`` factory (mirrors ``make_group_tile_spec``
    + the combine config). TN uses a deeper tr8 drain hint."""
    vh = 3 if layout == "tn" else vmcnt_hint
    return CombineGroupFP8TileSpec(
        out_features=out_features, pool_capacity=pool_capacity, combine_slots=combine_slots,
        num_comm_blocks=num_comm_blocks, layout=layout, K=K,
        block_tile=(BLOCK_M, BLOCK_N, BLOCK_K),
        warp_tile=(BLOCK_M // 4, BLOCK_N // 8, BLOCK_K),
        GROUP_M=1, num_xcd=1, group_n=0, nt_vmcnt=nt_vmcnt, vmcnt_hint=vh,
        b_inline_asm_load=False, cbsz=0, blgp=0, out_fp16=False)


@functools.lru_cache(maxsize=256)
def _compile(layout, out_features, hidden_size, pool_capacity, BLOCK_M, BLOCK_N, comb_blocks,
             combine_slots, nt_vmcnt=3, waves_per_eu=2, agpr_alloc=0):
    # num_comm_blocks=0: the fused scatter epilogue has NO separate COMBINE role, so
    # the LINEAR scheduler must not offset past any comm blocks (block_m = pid//n_blocks).
    spec = make_combine_tile_spec(layout=layout, K=hidden_size, out_features=out_features,
                                  pool_capacity=pool_capacity, combine_slots=combine_slots,
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, nt_vmcnt=nt_vmcnt,
                                  num_comm_blocks=0)
    layout = spec.layout
    BLOCK_M, BLOCK_N, _ = spec.block_tile
    out_features = spec.out_features
    pool_capacity = spec.pool_capacity
    combine_slots = spec.combine_slots
    if agpr_alloc is None:
        agpr_alloc = _LAYOUT_AGPR[layout]
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = pool_capacity // BLOCK_M
    comb_records = combine_slots * out_features * 2   # bf16 peer combine-buffer bound (bytes)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_grouped(ACT: fx.Tensor, WEIGHTS: fx.Tensor, L2Y: fx.Tensor, A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
           TILE_TO_GROUP: fx.Tensor, SB_L2: fx.Tensor, ORIGIN_RANK: fx.Tensor, ORIGIN_SLOT: fx.Tensor,
           COMB_ADDRS: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor, c_n: fx.Int32):
        # Phase 2: each GEMM tile scatters its result straight into the peers' combine
        # buffers via ScatterCombineEpilogue -- no L2Y round-trip, no separate COMBINE
        # role, no SB_L2 scoreboard. L2Y / SB_L2 args kept for API compat (unused).
        _ = spec.cache_tag  # JIT cache-key discriminator; emits no IR
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        # NT/TN: LDS at the top; NN: lds=None -> emit allocs it inside the guard.
        lds = fx.SharedAllocator().allocate(spec.shared_storage).peek() if layout != "nn" else None

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        origin_rank_resource = create_buffer_resource(ORIGIN_RANK, max_size=True)
        origin_slot_resource = create_buffer_resource(ORIGIN_SLOT, max_size=True)
        combine_address_resource = create_buffer_resource(COMB_ADDRS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        # runtime scatter buffers handed to the fused epilogue (ScatterCombineEpilogue).
        epi_ctx = {
            "out_features": out_features, "comb_records": comb_records, "n_slots": combine_slots,
            "origin_rank_res": origin_rank_resource, "origin_slot_res": origin_slot_resource,
            "comb_addr_res": combine_address_resource,
        }

        block_m = block_index // fx.Int32(n_blocks)   # LinearNoSync map (num_comm_blocks==0)
        if block_m < real_tiles:
            c_m_real = real_tiles * fx.Int32(BLOCK_M)
            gbase = spec.group_base(group_resource, block_m, spec.K, c_n)
            spec.emit(A=ACT, B=WEIGHTS, C=L2Y, A_scale=A_SCALE, B_scale=B_SCALE,
                      c_m=c_m_real, c_n=c_n, lds=lds, group_base=gbase, epilogue_ctx=epi_ctx)
            fx.rocdl.s_waitcnt(0)
            # _fence_release()   # multi-rank: publish the scattered rows to peers

    @flyc.jit
    def launch(ACT, WEIGHTS, L2Y, A_SCALE, B_SCALE, TILE_TO_GROUP, SB_L2, ORIGIN_RANK, ORIGIN_SLOT,
               COMB_ADDRS, NUM_TILE_BLOCKS, c_n: int, stream: fx.Stream = fx.Stream(None)):
        grid_size = worst_case_tiles * n_blocks
        combine_grouped(ACT, WEIGHTS, L2Y, A_SCALE, B_SCALE, TILE_TO_GROUP, SB_L2, ORIGIN_RANK, ORIGIN_SLOT,
           COMB_ADDRS, NUM_TILE_BLOCKS, c_n,
           value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
            grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def grouped_gemm_fp8_combine(act_fp8, weight_fp8, l2y, tile_to_group, sb_l2, origin_rank,
                             origin_slot, comb_addrs, combine_slots, mblk_dev, *,
                             a_scale, b_scale, layout="nt", BM=256, BN=256, comb_blocks=64,
                             nt_vmcnt=3):
    """Fused grouped FP8 GEMM + combine PUSH.

    ``layout`` selects the GEMM (``nt``: weight [G,N,K], A=act[M,K]; ``nn``/``tn``:
    weight [G,K,N], A=act[M,K] (nn) or [K,M] (tn)). ``l2y`` [M,N] bf16 (local GEMM
    output); per-row ``origin_rank`` / ``origin_slot`` route finished rows into the
    origin rank's combine buffer (``comb_addrs[origin]``); per-tensor scales.
    ``sb_l2`` (the L2 scoreboard) must be zeroed before the call."""
    G, K_contract, out_features, weight_bytes = _weight_layout(layout, weight_fp8)
    if layout == "tn":
        hidden_size = act_fp8.size(0)         # A=[K,M]
        pool_capacity = act_fp8.size(1)       # output rows = M
        agpr_alloc = _LAYOUT_AGPR["tn"]       # inplace MFMA needs 128 AGPRs
    else:
        hidden_size = act_fp8.size(1)         # A=[M,K]
        pool_capacity = act_fp8.size(0)
        agpr_alloc = 0
    assert K_contract == hidden_size, f"weight K={K_contract} != activation K={hidden_size}"
    c_n = out_features
    device = act_fp8.device

    sa = _scalar(a_scale, device)
    sb = _scalar(b_scale, device)
    act_bytes = _as_i8_flat(act_fp8)
    l2y_flat = l2y.contiguous().view(-1)

    launch = _compile(layout, out_features, hidden_size, pool_capacity, BM, BN, int(comb_blocks),
                      int(combine_slots), int(nt_vmcnt), 2, int(agpr_alloc))
    launch(act_bytes, weight_bytes, l2y_flat, sa, sb, tile_to_group, sb_l2, origin_rank,
           origin_slot, comb_addrs, mblk_dev, c_n, stream=torch.cuda.current_stream())
    return l2y
