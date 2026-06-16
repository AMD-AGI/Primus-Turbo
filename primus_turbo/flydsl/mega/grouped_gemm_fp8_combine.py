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
``DispatchGroupFP8TileSpec``): its ``build_launch`` emits BOTH roles -- the GEMM
tile (the GEMM role reuses the parent ``spec.emit(...)`` unchanged, then signals the
local L2 scoreboard) and the combine push (the ``combine_tile`` closure, the ONLY
kernel-specific code). The base ``GroupFp8TileSpec`` supplies the grouped seam
(``schedule`` tile-id map + per-expert ``base_offsets`` slab) over ``gemm_tile_spec``.

``combine_tile`` stays a closure inside ``build_launch``'s ``@flyc.kernel`` body
(not a spec method): its dynamic ``while``-spin / ``for`` / ``if`` only lower under
the kernel AST rewriter. The push moves bf16 L2Y rows (16-byte vec, legal), so no
i32 repack is needed."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.group_tile_spec import (
    _LAYOUT_AGPR,
    _atomic_add,
    _fence_acquire,
    _fence_release,
    _ld_relaxed,
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


class CombineGroupFP8TileSpec(GroupFp8TileSpec):
    """Fused grouped FP8 GEMM + combine PUSH, expressed as a ``GroupFp8TileSpec``
    subclass. The GEMM tile reuses the parent ``emit``/``schedule``/``base_offsets``
    UNCHANGED; ``build_launch`` emits BOTH roles in one kernel -- the GEMM tile
    (gemm_tile, which signals the local L2 scoreboard when done) and the combine
    push (``combine_tile``, which spins then pushes finished rows cross-rank). Only
    ``combine_tile`` is kernel-specific.

    Extra construction config (vs the base): ``out_features`` (-> n_blocks / push
    geometry / grid), ``pool_capacity`` (-> the no-sync over-launch bound),
    ``combine_slots`` (-> the peer combine-buffer record bound)."""

    def __init__(self, *, out_features, pool_capacity, combine_slots, **kw):
        super().__init__(**kw)
        self.out_features = out_features
        self.pool_capacity = pool_capacity
        self.combine_slots = combine_slots
        self.kernel_name = "combine_grouped_" + self.layout
        self.cache_tag = self.cache_tag + (out_features, pool_capacity, combine_slots)

    def build_launch(self, *, waves_per_eu=2, agpr_alloc=None):
        """Build the fused ``@flyc.kernel`` + ``@flyc.jit`` launcher: the front
        ``num_comm_blocks`` blocks run ``combine_tile`` (spin + cross-rank push), the
        rest each compute one GEMM tile via the stock ``self.emit`` then signal the
        local L2 scoreboard. Decorator form (NOT the base's functional form): the
        combine push has dynamic while/if that need the @flyc.kernel AST rewriter."""
        spec = self
        layout = self.layout
        BLOCK_M, BLOCK_N = self.BLOCK_M, self.BLOCK_N
        out_features = self.out_features
        pool_capacity = self.pool_capacity
        combine_slots = self.combine_slots
        num_comb_blocks = self.num_comm_blocks
        if agpr_alloc is None:
            agpr_alloc = _LAYOUT_AGPR[layout]
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
        n_blocks = out_features // BLOCK_N
        worst_case_tiles = pool_capacity // BLOCK_M

        # combine push geometry (bf16 L2Y rows): n_wg rows in flight per block
        push_lanes = min(_PUSH_LANES, _BLOCK_THREADS)
        n_wg = _BLOCK_THREADS // push_lanes
        cols_per_step = push_lanes * _VEC
        n_full = out_features // cols_per_step
        n_tail = out_features % cols_per_step
        comb_records = combine_slots * out_features * 2   # bf16 combine buffer bound (bytes)

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def combine_grouped(ACT: fx.Tensor, WEIGHTS: fx.Tensor, L2Y: fx.Tensor, A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
               TILE_TO_GROUP: fx.Tensor, SB_L2: fx.Tensor, ORIGIN_RANK: fx.Tensor, ORIGIN_SLOT: fx.Tensor,
               COMB_ADDRS: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor, c_n: fx.Int32):
            _ = spec.cache_tag  # JIT cache-key discriminator; emits no IR
            thread_index = fx.thread_idx.x
            block_index, _b, _c = fx.block_idx
            comb_block_count = fx.Int32(num_comb_blocks)
            # NT/TN: LDS at the top (unconditional); NN: lds=None -> emit allocs it
            # inside the guard. Ternary (not an if-statement). Opposite codegen
            # sensitivity per layout; see dispatch_grouped_gemm_fp8.
            lds = fx.SharedAllocator().allocate(spec.shared_storage).peek() if layout != "nn" else None

            group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
            num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
            l2y_resource = create_buffer_resource(L2Y, max_size=True)
            origin_rank_resource = create_buffer_resource(ORIGIN_RANK, max_size=True)
            origin_slot_resource = create_buffer_resource(ORIGIN_SLOT, max_size=True)
            combine_address_resource = create_buffer_resource(COMB_ADDRS, max_size=True)
            real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())

            def combine_tile(m):
                # CONSUMER: spin on the GEMM signal, then push finished rows cross-rank
                if thread_index == fx.Int32(0):
                    signal = _ld_relaxed(SB_L2, m)
                    while signal < fx.Int32(n_blocks):
                        fx.rocdl.s_sleep(fx.Int32(2))
                        signal = _ld_relaxed(SB_L2, m)
                fx.gpu.barrier()
                # _fence_acquire()   # invalidate L1 so the push reads the freshly-written L2Y
                base_row = m * fx.Int32(BLOCK_M)
                lane = thread_index % fx.Int32(push_lanes)
                workgroup = thread_index // fx.Int32(push_lanes)
                if thread_index < fx.Int32(push_lanes * n_wg):
                    for row_offset in range(0, BLOCK_M, n_wg):
                        row = base_row + fx.Int32(row_offset) + workgroup
                        origin = buffer_load(origin_rank_resource, row, vec_width=1, dtype=fx.T.i32())
                        if origin >= fx.Int32(0):
                            slot = buffer_load(origin_slot_resource, row, vec_width=1, dtype=fx.T.i32())
                            combine_address = buffer_load(combine_address_resource, origin, vec_width=1, dtype=fx.T.i64())
                            peer_combine = create_buffer_resource_from_addr(combine_address, num_records_bytes=comb_records)
                            for chunk_index in fx.range_constexpr(n_full):
                                col = fx.Int32(chunk_index * cols_per_step) + lane * fx.Int32(_VEC)
                                value = buffer_load(l2y_resource, row * fx.Int32(out_features) + col, vec_width=_VEC, dtype=fx.T.bf16())
                                buffer_store(value, peer_combine, slot * fx.Int32(out_features) + col)
                            if n_tail:
                                col = fx.Int32(n_full * cols_per_step) + lane * fx.Int32(_VEC)
                                if col < fx.Int32(out_features):
                                    value = buffer_load(l2y_resource, row * fx.Int32(out_features) + col, vec_width=_VEC, dtype=fx.T.bf16())
                                    buffer_store(value, peer_combine, slot * fx.Int32(out_features) + col)

            if block_index < comb_block_count:
                # COMBINE role: grid-stride over this block's pool blocks
                local_count = (real_tiles - block_index + comb_block_count - fx.Int32(1)) // comb_block_count
                for iteration in range(local_count):
                    combine_tile(block_index + iteration * comb_block_count)
            else:
                # GEMM role: custom LINEAR tile-id map + stock spec.emit, then signal
                # the local L2 scoreboard so the combine role above can push the tile.
                tile_index = block_index - comb_block_count
                block_m = tile_index // fx.Int32(n_blocks)
                if block_m < real_tiles:
                    c_m_real = real_tiles * fx.Int32(BLOCK_M)
                    # per-expert B slab via the stock emit scalar seam
                    gbase = spec.group_base(group_resource, block_m, spec.K, c_n)
                    spec.emit(A=ACT, B=WEIGHTS, C=L2Y, A_scale=A_SCALE, B_scale=B_SCALE,
                              c_m=c_m_real, c_n=c_n, lds=lds, group_base=gbase)
                    fx.rocdl.s_waitcnt(0)
                    # _fence_release()                 # ALL waves flush their L2Y stores to L2
                    fx.gpu.barrier()                 # all stores landed before signal
                    if thread_index == fx.Int32(0):
                        _atomic_add(SB_L2, block_m, fx.Int32(1))

        @flyc.jit
        def launch(ACT, WEIGHTS, L2Y, A_SCALE, B_SCALE, TILE_TO_GROUP, SB_L2, ORIGIN_RANK, ORIGIN_SLOT,
                   COMB_ADDRS, NUM_TILE_BLOCKS, c_n: int, stream: fx.Stream = fx.Stream(None)):
            grid_size = num_comb_blocks + worst_case_tiles * n_blocks
            combine_grouped(ACT, WEIGHTS, L2Y, A_SCALE, B_SCALE, TILE_TO_GROUP, SB_L2, ORIGIN_RANK, ORIGIN_SLOT,
               COMB_ADDRS, NUM_TILE_BLOCKS, c_n,
               value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
                grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

        return launch


@functools.lru_cache(maxsize=256)
def make_combine_tile_spec(*, layout, K, out_features, pool_capacity, combine_slots,
                           BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3, num_comm_blocks=0,
                           vmcnt_hint=2):
    """Cached ``CombineGroupFP8TileSpec`` factory (mirrors ``make_group_tile_spec``
    + the combine config). TN uses a deeper tr8 drain hint."""
    vh = 3 if layout == "tn" else vmcnt_hint
    return CombineGroupFP8TileSpec(
        out_features=out_features, pool_capacity=pool_capacity, combine_slots=combine_slots,
        num_comm_blocks=num_comm_blocks, layout=layout, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        GROUP_M=1, num_xcd=1, group_n=0, nt_vmcnt=nt_vmcnt, vmcnt_hint=vh,
        b_inline_asm_load=False, cbsz=0, blgp=0, out_fp16=False)


@functools.lru_cache(maxsize=256)
def _compile(layout, out_features, hidden_size, pool_capacity, BLOCK_M, BLOCK_N, comb_blocks,
             combine_slots, nt_vmcnt=3, waves_per_eu=2, agpr_alloc=0):
    spec = make_combine_tile_spec(layout=layout, K=hidden_size, out_features=out_features,
                                  pool_capacity=pool_capacity, combine_slots=combine_slots,
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, nt_vmcnt=nt_vmcnt,
                                  num_comm_blocks=int(comb_blocks))
    return spec.build_launch(waves_per_eu=waves_per_eu, agpr_alloc=agpr_alloc)


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
