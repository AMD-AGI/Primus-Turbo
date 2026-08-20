###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega.ep_intranode import (
    combine_dedup_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.prims import (
    atomic_add,
    cast,
    ld,
)
from primus_turbo.flydsl.mega.symm_buffer import (
    COMBINE_FLAG_STRIDE,
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
)

_WARP = 64
_BLOCK_THREADS = 512


_PVEC = 8
_NUM_WARPS = _BLOCK_THREADS // _WARP

_LAYOUTS = ("nt", "nn", "tn")
_LAYOUT_CODES = {name: code for code, name in enumerate(_LAYOUTS)}

_COMBINE_DEDUP_NPASS = 2

_NUM_REDUCE_BLOCKS = 2048

_COMBINE_FLAG_SCOPE = "sys"
_COMBINE_GATE_SLEEP = 32

# Lead GEMM blocks placed BEFORE the combine region in the grid. Workgroups are
# dispatched in block order and 128 KB of LDS pins occupancy at 1 WG/CU, so a
# combine region at ordinal 0 parks CUs in the gate spin before any tile exists.
# Putting one full machine's worth of GEMM tiles first lets the GEMM open at the
# full 256 CUs; the combine blocks land as those retire, with data already there.
# Must be a multiple of 8 (the XCD residue the rect tile map is affine to), and it
# is pinned per layout alongside the block count below.

# Sub-segment tickets. A block cannot push a segment before the GEMM has produced
# every tile it spans, so a coarse atom drawn just after a tile release is dead
# wall time. Slicing each segment into _COMBINE_SEG_PARTS pieces, round-robined
# across blocks, spreads each release burst over more CUs. 4 is the optimum: 8 and
# a 16-way split of just the tail segments both lose to the per-task floor.
_COMBINE_SEG_PARTS = 4

# Traversal order inside the rectangle tile map: 2 rows at a time, column-minor.
_RECT_P = 2

# TEMPORARY: autotune replaced by a per-layout pin. nt/nn from the 2026-08-17
# corrected both-directions sweep (nt plateau 56-64, cliff at 72; nn sharp at 32).
# combine only ever runs nt (fwd) and nn (bwd dgrad) -- tn mirrors nn, unswept.
# These counts are NOT "enough CUs to fill the link" -- the dedup push tops out near
# 13 GB/s per CU (gather-reduce over scattered member rows into slot-scattered
# destinations), vs ~45 GB/s per CU for dispatch's contiguous copy. Standalone push
# scaling: 16 CU 211 GB/s, 32 CU 315, 64 CU 360, 128 CU 367. So 32 is already on the
# steep part of the curve, traded against leaving the GEMM its CUs; halving it to 16
# costs far more than the 16 CUs are worth (nn 3.68 -> 4.74 ms).
_PINNED_CONFIG = {
    "nt": {"num_combine_blocks": 64, "lead_gemm_blocks": 3072},
    "nn": {"num_combine_blocks": 32, "lead_gemm_blocks": 2048},
    "tn": {"num_combine_blocks": 32, "lead_gemm_blocks": 2048},
}
# Reserved combine region: kept at the old sweep's widest split so pinning does
# not move the role boundary or the grid shape.
_WIDEST_COMBINE_BLOCKS = 128


@functools.lru_cache(maxsize=8)
def _get_dummy_tensor(device):
    return torch.empty(1, dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=256)
def _make_grouped_gemm_combine(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_combine_slots,
    topk,
    num_experts,
    rank,
    num_ranks=0,
    num_max_tokens_per_rank=0,
    nt_vmcnt=3,
    out_fp16=False,
    layout="nt",
    apply_weights=False,
    with_gate=False,
    dedup_npass=2,
    seg_parts=1,
    lead_gemm_blocks=0,
):
    K = hidden_size
    gemm_tile = functools.partial(gemm_bf16_tile, layout)
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    comb_records = num_combine_slots * out_features * 2
    gate_records = num_combine_slots * 4
    # Blocks [0, num_max_combine_blocks) are the combine role, the rest are GEMM tiles.
    # Reserving the widest split the sweep can pick lets every config share one compiled
    # kernel; the unused combine blocks exit immediately. The boundary is rounded up to a
    # multiple of the XCD count so that gemm_tile_index = block_index - boundary keeps the
    # block_index % 8 residue the rectangle tile map is affine to.
    num_max_combine_blocks = (_WIDEST_COMBINE_BLOCKS + 7) // 8 * 8
    # Rectangular XCD-affine tile map; needs n_blocks divisible by 4.
    use_rect = bool(n_blocks) and n_blocks % 4 == 0
    num_tasks = num_experts * seg_parts
    # Never lead with more tiles than exist, and keep the XCD residue intact.
    lead_blocks = min(lead_gemm_blocks, worst_case_tiles * n_blocks) // 8 * 8
    # The dispatch prologue initialises every pool row's slot id to this value and only
    # real rows overwrite it, so slot_id >= sentinel <=> that pool row is block padding.
    pad_slot_sentinel = num_ranks * num_max_tokens_per_rank
    # tn's A rows are not pool rows, so the probe must never fire there.
    probe_sentinel = 0x7FFFFFFF if layout == "tn" else pad_slot_sentinel

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def grouped_gemm_combine_kernel(
        ACT: fx.Tensor,
        WEIGHTS: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        RECV_DST_RANK: fx.Tensor,
        RECV_START_ROW: fx.Tensor,
        RECV_COUNT: fx.Tensor,
        POOL_SRC_SLOT: fx.Tensor,
        OUTPUT: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        D_TOPK_W: fx.Tensor,
        SORTED_SLOT_IDS: fx.Tensor,
        DEDUP_KEY_ROW: fx.Tensor,
        SOURCE_SLOT_KIND: fx.Tensor,
        sym_buffer: SymBuffer,
        c_n: fx.Int32,
        num_combine_blocks: fx.Int32,
        COMBINE_PARITY: fx.Tensor,
        COMBINE_EXPECTED: fx.Tensor,
        REDUCE_EXPECTED: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        # Rotate the dispatch order back into the old [combine | gemm] ordinal space:
        # ordinals 0..lead_blocks-1 of the grid are GEMM tiles, the combine region
        # follows them, then the GEMM resumes. lead_blocks and num_max_combine_blocks
        # are both multiples of 8, so gemm_tile_index keeps its block_index % 8 residue.
        if lead_blocks:
            block_index = fx.arith.select(
                block_index < fx.Int32(lead_blocks),
                block_index + fx.Int32(num_max_combine_blocks),
                fx.arith.select(
                    block_index < fx.Int32(lead_blocks + num_max_combine_blocks),
                    block_index - fx.Int32(lead_blocks),
                    block_index,
                ),
            )
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # Only the reserved combine blocks exist; a wider split would corrupt the role
        # boundary. Clamped here (not host-side) because the split is a runtime value.
        num_combine_blocks = fx.arith.select(
            num_combine_blocks < fx.Int32(num_max_combine_blocks),
            num_combine_blocks,
            fx.Int32(num_max_combine_blocks),
        )
        # Early-out for the worst-case padding blocks, before any prologue work.
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        blocks_live = (
            fx.Int32(num_max_combine_blocks) + real_tiles * fx.Int32(n_blocks) + fx.Int32(_NUM_REDUCE_BLOCKS)
        )
        if block_index < blocks_live:
            workspace = Workspace(
                sym_buffer.get_base_ptr(),
                num_ranks,
                num_experts,
                num_max_tokens_per_rank,
                topk,
                out_features,
                token_dtype=TOKEN_DTYPE,
            )
            # Epoch parity is the only prologue value all three roles share.
            combine_parity_res = create_buffer_resource(COMBINE_PARITY, max_size=True)
            combine_parity = cast(
                buffer_load(combine_parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
            )
            # Fold the per-cache-line flag stride into the bank.
            combine_bank = combine_parity * fx.Int32(worst_case_tiles * COMBINE_FLAG_STRIDE)
            # Must be hoisted: the rewriter rejects workspace.<method>() inside a traced branch.
            combine_flag_base = workspace.get_combine_flag_ptr()
            l2_token_buffer_base = workspace.get_l2_token_buffer_ptr()
            comb_base = workspace.get_combine_token_buffer_ptr()
            reduce_flag_base = workspace.get_reduce_flag_ptr()
            gate_base = workspace.get_combine_gate_ptr() if with_gate else None

            # Region test uses the reserved width; blocks past the live split exit.
            if block_index < fx.Int32(num_max_combine_blocks):
                if block_index < num_combine_blocks:
                    # Task-based combine: one warp per recv-segment, gated on its spanned GEMM tiles.
                    reduce_bank = combine_parity * fx.Int32(num_combine_slots)
                    combine_expected_res = create_buffer_resource(COMBINE_EXPECTED, max_size=True)
                    reduce_expected_res = create_buffer_resource(REDUCE_EXPECTED, max_size=True)
                    expected_combine_i64 = buffer_load(
                        combine_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64()
                    )
                    expected_reduce_i64 = buffer_load(
                        reduce_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64()
                    )
                    # These tables ride the handle, not shared symm, else bwd reads stale.
                    recv_dst_rank_res = create_buffer_resource(RECV_DST_RANK, max_size=True)
                    recv_start_row_res = create_buffer_resource(RECV_START_ROW, max_size=True)
                    recv_count_res = create_buffer_resource(RECV_COUNT, max_size=True)
                    origin_slot_res = create_buffer_resource(POOL_SRC_SLOT, max_size=True)
                    sorted_slot_res = create_buffer_resource(SORTED_SLOT_IDS, max_size=True)
                    key_row_res = create_buffer_resource(DEDUP_KEY_ROW, max_size=True)
                    grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
                    seg_local = (
                        fx.Int32(num_tasks) - block_index + num_combine_blocks - fx.Int32(1)
                    ) // num_combine_blocks
                    # Cursor rides seg_iter so each tile is polled once.
                    combine_cursor = fx.Int32(0)
                    for seg_iter in range(seg_local):
                        task_index = block_index + seg_iter * num_combine_blocks
                        seg_index = task_index // fx.Int32(seg_parts)
                        seg_part = task_index % fx.Int32(seg_parts)
                        full_start = buffer_load(recv_start_row_res, seg_index, vec_width=1, dtype=fx.T.i32())
                        full_count = buffer_load(recv_count_res, seg_index, vec_width=1, dtype=fx.T.i32())
                        # Even row split; the last part absorbs the shorter remainder.
                        part_rows = (full_count + fx.Int32(seg_parts - 1)) // fx.Int32(seg_parts)
                        part_off = seg_part * part_rows
                        seg_start = full_start + part_off
                        part_rest = full_count - part_off
                        seg_count = fx.arith.select(part_rest < part_rows, part_rest, part_rows)
                        if seg_count > fx.Int32(0):
                            t1 = (seg_start + seg_count - fx.Int32(1)) // fx.Int32(BLOCK_M)
                            # Clamp the poll cursor to this segment's own first GEMM tile.
                            # combine_cursor rides seg_iter to avoid re-polling confirmed
                            # tiles, but its init of 0 forced the first segment to spin on
                            # every tile below its own start (one dependent sys-scope flag
                            # load each) -- tiles that hold no row this block consumes.
                            # Bit-exact: each of tiles [t0, t1] is still gated; only tiles
                            # strictly below t0 (not read here) are skipped. Monotonic seg
                            # rows keep the cursor non-decreasing across segments.
                            t0 = seg_start // fx.Int32(BLOCK_M)
                            tile_cursor = fx.arith.select(combine_cursor > t0, combine_cursor, t0)
                            if thread_index == fx.Int32(0):
                                while tile_cursor <= t1:
                                    fx.rocdl.s_waitcnt(0)
                                    signal_count = ld(
                                        combine_flag_base,
                                        combine_bank + tile_cursor * fx.Int32(COMBINE_FLAG_STRIDE),
                                        scope=_COMBINE_FLAG_SCOPE,
                                        dtype=fx.T.i64(),
                                    )
                                    while signal_count != expected_combine_i64:
                                        fx.rocdl.s_sleep(fx.Int32(_COMBINE_GATE_SLEEP))
                                        fx.rocdl.s_waitcnt(0)
                                        signal_count = ld(
                                            combine_flag_base,
                                            combine_bank + tile_cursor * fx.Int32(COMBINE_FLAG_STRIDE),
                                            order="relaxed",
                                            scope=_COMBINE_FLAG_SCOPE,
                                            dtype=fx.T.i64(),
                                        )
                                    tile_cursor = tile_cursor + fx.Int32(1)
                            combine_cursor = tile_cursor
                            fx.rocdl.s_waitcnt(0)
                            fx.gpu.barrier()
                            combine_dedup_bf16_tile(
                                sym_buffer,
                                workspace,
                                thread_index=thread_index,
                                task_index=seg_index,
                                row_start=seg_start,
                                row_count=seg_count,
                                recv_dst_rank_res=recv_dst_rank_res,
                                recv_start_row_res=recv_start_row_res,
                                recv_count_res=recv_count_res,
                                origin_slot_res=origin_slot_res,
                                sorted_slot_res=sorted_slot_res,
                                key_row_res=key_row_res,
                                grad_gate_res=grad_gate_res,
                                topk=topk,
                                apply_weights=apply_weights,
                                signal=True,
                                epoch=expected_reduce_i64,
                                bank_offset=reduce_bank,
                                with_gate=with_gate,
                                npass=dedup_npass,
                            )

            else:
                gemm_tile_index = block_index - fx.Int32(num_max_combine_blocks)
                block_m = gemm_tile_index // fx.Int32(n_blocks)
                block_n = gemm_tile_index % fx.Int32(n_blocks)
                if use_rect:
                    # Give each XCD a 4(block_m) x cols_per_xcd(block_n) rectangle of the
                    # band instead of a 1x8 sliver, so its A/B panels stay L2-resident. A
                    # band is 8*n_blocks tiles, so the XCD residue (t % 8) survives the
                    # remap. Applied only on the whole-band prefix; the row tail and the
                    # empty/reduce ordinal range keep the identity numbering the combine
                    # gate's prefix accounting depends on.
                    cols_per_xcd = n_blocks // 4
                    pass_size = _RECT_P * cols_per_xcd
                    band_tiles = fx.Int32(8 * n_blocks)
                    whole_band_tiles = (real_tiles // fx.Int32(8)) * band_tiles
                    band_idx = gemm_tile_index // band_tiles
                    tile_in_band = gemm_tile_index % band_tiles
                    xcd = tile_in_band % fx.Int32(8)
                    slot_in_xcd = tile_in_band // fx.Int32(8)
                    pass_group = slot_in_xcd // fx.Int32(pass_size)
                    pos_in_pass = slot_in_xcd % fx.Int32(pass_size)
                    row_local = pass_group * fx.Int32(_RECT_P) + (pos_in_pass % fx.Int32(_RECT_P))
                    col_local = pos_in_pass // fx.Int32(_RECT_P)
                    rect_n = (xcd // fx.Int32(2)) * fx.Int32(cols_per_xcd) + col_local
                    rect_m = band_idx * fx.Int32(8) + (xcd % fx.Int32(2)) * fx.Int32(4) + row_local
                    in_rect = gemm_tile_index < whole_band_tiles
                    block_m = fx.arith.select(in_rect, rect_m, block_m)
                    block_n = fx.arith.select(in_rect, rect_n, block_n)
                if block_m < real_tiles:
                    # GEMM role: one real tile (block_m, block_n) per block.
                    group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
                    group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    group_base = group_index * fx.Int32(K) * c_n
                    act_base = fx.arith.ArithValue(
                        fx.arith.index_cast(fx.T.i64(), extract_base_index(ACT)), signed=True
                    )
                    # Fold per-tile base in int64 (pool >4GB); voffset stays int32.
                    a_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * K * 2)
                    c_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * 2) * cast(c_n, fx.T.i64())
                    A_tile = make_bf16_fp16_tile_tensor(act_base, a_off, BLOCK_M * K)
                    C_tile = make_bf16_fp16_tile_tensor(l2_token_buffer_base, c_off, 0x40000000)
                    # Padding-skip probe -- see the same construct in the dispatch GEMM.
                    # Each expert's pool region is padded up to a BLOCK_M multiple; those
                    # rows hold the zeroed pad slot all the way through dispatch+act, and
                    # nothing ever reads their C rows back (the combine role walks only
                    # real recv segments). If the first row a wave's c10/c11 quadrants
                    # cover is already padding, all 64 are, so those MFMA groups are
                    # no-ops. Loads/barriers/stores are untouched, so waves may disagree.
                    slot_probe_resource = create_buffer_resource(SORTED_SLOT_IDS, max_size=True)
                    wave_m_probe = (thread_index // fx.Int32(64)) // fx.Int32(4)
                    probe_row = (
                        block_m * fx.Int32(BLOCK_M)
                        + fx.Int32(BLOCK_M // 2)
                        + wave_m_probe * fx.Int32((BLOCK_M // 128) * 32)
                    )
                    pad_hi = buffer_load(
                        slot_probe_resource, probe_row, vec_width=1, dtype=fx.T.i32()
                    ) >= fx.Int32(probe_sentinel)
                    # Wave-independent probe: when the whole workgroup agrees the lower A
                    # half is padding, the cooperative A1 global->LDS copies are dead too.
                    pad_uniform = buffer_load(
                        slot_probe_resource,
                        block_m * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M // 2),
                        vec_width=1,
                        dtype=fx.T.i32(),
                    ) >= fx.Int32(probe_sentinel)
                    if pad_uniform:
                        gemm_tile(
                            A_tile,
                            WEIGHTS,
                            C_tile,
                            fx.Int32(BLOCK_M),
                            c_n,
                            lds,
                            fx.Int32(0),
                            block_n,
                            K=K,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            out_fp16=out_fp16,
                            nt_vmcnt=nt_vmcnt,
                            b_group_base=group_base,
                            c_cache_modifier=16,
                            n_exact=True,
                            SKIP_A1=True,
                            SKIP_A1_STORES=True,
                        )
                    elif pad_hi:
                        gemm_tile(
                            A_tile,
                            WEIGHTS,
                            C_tile,
                            fx.Int32(BLOCK_M),
                            c_n,
                            lds,
                            fx.Int32(0),
                            block_n,
                            K=K,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            out_fp16=out_fp16,
                            nt_vmcnt=nt_vmcnt,
                            b_group_base=group_base,
                            c_cache_modifier=16,
                            n_exact=True,
                            SKIP_A1=True,
                        )
                    else:
                        gemm_tile(
                            A_tile,
                            WEIGHTS,
                            C_tile,
                            fx.Int32(BLOCK_M),
                            c_n,
                            lds,
                            fx.Int32(0),
                            block_n,
                            K=K,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            out_fp16=out_fp16,
                            nt_vmcnt=nt_vmcnt,
                            b_group_base=group_base,
                            # sc1 only (16), NOT sc1|nt: sc1 is load-bearing for cross-XCD
                            # visibility, and letting C retire into L2 shortens the drain.
                            c_cache_modifier=16,
                            n_exact=True,
                            SKIP_A1=False,
                        )
                    # Whole-workgroup release rendezvous; two are required, and LLVM
                    # folds adjacent barriers, so keep the separator.
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()
                    if thread_index == fx.Int32(0):
                        atomic_add(
                            combine_flag_base,
                            combine_bank + block_m * fx.Int32(COMBINE_FLAG_STRIDE),
                            fx.Int64(1),
                            scope=_COMBINE_FLAG_SCOPE,
                        )
                else:
                    # Empty region: the first _NUM_REDUCE_BLOCKS blocks do topk reduce, rest exit.
                    empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                    if empty_ordinal < fx.Int32(_NUM_REDUCE_BLOCKS):
                        reduce_bank = combine_parity * fx.Int32(num_combine_slots)
                        reduce_expected_res = create_buffer_resource(REDUCE_EXPECTED, max_size=True)
                        expected_reduce_i64 = buffer_load(
                            reduce_expected_res, combine_parity, vec_width=1, dtype=fx.T.i64()
                        )
                        comb_local_res = create_buffer_resource_from_addr(
                            comb_base, num_records_bytes=comb_records
                        )
                        output_res = create_buffer_resource(OUTPUT, max_size=True)
                        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
                        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
                        gate_local_res = (
                            create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records)
                            if with_gate
                            else None
                        )
                        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True) if with_gate else None
                        kind_res = create_buffer_resource(SOURCE_SLOT_KIND, max_size=True)
                        # Never-reset alignment: bump empty block_m flags to cumulative expected.
                        n_empty = fx.Int32(worst_case_tiles) - real_tiles
                        reduce_stride = fx.Int32(_NUM_REDUCE_BLOCKS)
                        align_count = (n_empty - empty_ordinal + reduce_stride - fx.Int32(1)) // reduce_stride
                        for align_iter in range(align_count):
                            empty_block_m = real_tiles + empty_ordinal + align_iter * reduce_stride
                            if thread_index == fx.Int32(0):
                                atomic_add(
                                    combine_flag_base,
                                    combine_bank + empty_block_m * fx.Int32(COMBINE_FLAG_STRIDE),
                                    fx.Int64(n_blocks),
                                    scope=_COMBINE_FLAG_SCOPE,
                                )

                        n_reduce_tiles = n_empty * fx.Int32(n_blocks)
                        active_reduce_blocks = fx.arith.select(
                            n_reduce_tiles < fx.Int32(_NUM_REDUCE_BLOCKS),
                            n_reduce_tiles,
                            fx.Int32(_NUM_REDUCE_BLOCKS),
                        )
                        topk_reduce_bf16_tile(
                            True,
                            False,  # dedup already applied the routing weight on the sender
                            with_gate,
                            thread_index,
                            empty_ordinal,
                            active_reduce_blocks * fx.Int32(_NUM_WARPS),
                            topk,
                            out_features,
                            num_experts,
                            rank,
                            comb_local_res,
                            output_res,
                            topk_indices_res,
                            num_tokens_res,
                            reduce_flag_base,
                            reduce_bank,
                            None,  # weights already folded in by the sender-side dedup
                            gate_local_res,
                            d_topk_w_res,
                            expected_reduce_i64,
                            dedup=True,
                            kind_res=kind_res,
                            num_combine_slots=num_combine_slots,
                        )

    return grouped_gemm_combine_kernel, num_max_combine_blocks


@functools.lru_cache(maxsize=4)
def _make_epoch_bump(add_combine, add_reduce):
    """Single-block kernel: flip parity, bump combine and reduce expected."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def epoch_bump_kernel(PARITY: fx.Tensor, COMBINE_EXP: fx.Tensor, REDUCE_EXP: fx.Tensor):
        if fx.thread_idx.x == fx.Int32(0):
            parity_res = create_buffer_resource(PARITY, max_size=True)
            combine_res = create_buffer_resource(COMBINE_EXP, max_size=True)
            reduce_res = create_buffer_resource(REDUCE_EXP, max_size=True)
            new_parity = buffer_load(parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()) ^ fx.Int64(1)
            buffer_store(new_parity, parity_res, fx.Int32(0))
            idx = cast(new_parity, fx.T.i32())
            new_combine = buffer_load(combine_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_combine)
            buffer_store(new_combine, combine_res, idx)
            new_reduce = buffer_load(reduce_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_reduce)
            buffer_store(new_reduce, reduce_res, idx)

    return epoch_bump_kernel


@flyc.jit
def _compiled_grouped_gemm_combine(
    ACT,
    WEIGHTS,
    TILE_TO_GROUP,
    NUM_TILE_BLOCKS,
    RECV_DST_RANK,
    RECV_START_ROW,
    RECV_COUNT,
    POOL_SRC_SLOT,
    OUTPUT,
    TOPK_INDICES,
    NUM_TOKENS_PER_RANK,
    GRAD_GATE,
    D_TOPK_W,
    SORTED_SLOT_IDS,
    DEDUP_KEY_ROW,
    SOURCE_SLOT_KIND,
    sym_buffer,
    c_n: fx.Int32,
    num_combine_blocks: fx.Int32,
    COMBINE_PARITY,
    COMBINE_EXPECTED,
    REDUCE_EXPECTED,
    out_features: fx.Constexpr[int],
    hidden_size: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int],
    BLOCK_N: fx.Constexpr[int],
    num_combine_slots: fx.Constexpr[int],
    topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    num_max_tokens_per_rank: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    apply_weights: fx.Constexpr[bool],
    with_gate: fx.Constexpr[bool],
    out_fp16: fx.Constexpr[bool],
    dedup_npass: fx.Constexpr[int],
    seg_parts: fx.Constexpr[int],
    lead_gemm_blocks: fx.Constexpr[int],
    stream: fx.Stream,
    nt_vmcnt: fx.Constexpr[int] = 3,
    agpr_alloc: fx.Constexpr[int] = 0,
    waves: fx.Constexpr[int] = 2,
):
    kernel, num_max_combine_blocks = _make_grouped_gemm_combine(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        num_combine_slots,
        topk,
        num_experts,
        rank,
        num_ranks,
        num_max_tokens_per_rank,
        nt_vmcnt,
        out_fp16,
        _LAYOUTS[layout_code],
        apply_weights,
        with_gate,
        dedup_npass,
        seg_parts,
        lead_gemm_blocks,
    )
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    # Sized for the worst case so the launch shape stays static under graph capture;
    # the null tiles are dispatch-limited and measured free.
    grid_size = num_max_combine_blocks + worst_case_tiles * n_blocks
    # Bump the epoch on device before the GEMM; same-stream makes it visible.
    _make_epoch_bump(int(n_blocks), 1)(COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )
    kernel(
        ACT,
        WEIGHTS,
        TILE_TO_GROUP,
        NUM_TILE_BLOCKS,
        RECV_DST_RANK,
        RECV_START_ROW,
        RECV_COUNT,
        POOL_SRC_SLOT,
        OUTPUT,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        GRAD_GATE,
        D_TOPK_W,
        SORTED_SLOT_IDS,
        DEDUP_KEY_ROW,
        SOURCE_SLOT_KIND,
        sym_buffer,
        c_n,
        num_combine_blocks,
        COMBINE_PARITY,
        COMBINE_EXPECTED,
        REDUCE_EXPECTED,
        value_attrs=make_value_attrs(waves, agpr_alloc, "512,512"),
    ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def grouped_gemm_combine_bf16_flydsl_kernel(
    x,
    l2_weights,
    handle,
    *,
    topk_indices,
    topk_weights=None,
    grad_gate=None,
    layout="nt",
    BM=256,
    BN=256,
):
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    assert x.dtype == torch.bfloat16 and l2_weights.dtype == torch.bfloat16
    assert topk_indices is not None, "topk reduce needs topk_indices"
    (
        num_tile_blocks,
        sorted_slot_ids,
        tile_to_expert,
        source_slot_kind,
        recv_dst_rank,
        recv_start_row,
        recv_count,
        pool_src_slot,
        dedup_key_row,
        *_dispatch_only,
    ) = handle
    symm = get_symm_buffer_for_mega_moe()
    sym_buffer = symm.get_sym_buffer()
    if layout == "tn":
        hidden_size, num_max_pool_tokens = x.shape
    else:
        num_max_pool_tokens, hidden_size = x.shape
    if layout == "nt":
        G, N, K = l2_weights.shape
    else:
        G, K, N = l2_weights.shape
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    c_n = out_features
    assert out_features == int(symm.hidden), (
        f"out_features {out_features} != SymmBuffer hidden {int(symm.hidden)}"
    )
    assert num_max_pool_tokens == int(symm.num_max_pool_tokens), "x rows must match SymmBuffer pool capacity"

    device = x.device
    num_combine_slots = int(symm.num_combine_slots)
    rank = int(symm.rank)
    topk = int(symm.num_topk)
    num_experts = int(symm.num_experts)
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"

    apply_weights = topk_weights is not None
    with_gate = grad_gate is not None

    # Pass 2D: kernel advances ACT base per-tile in int64 (flat MxK overflows int32 ABI).
    act_2d = x.contiguous()
    if layout == "nt":
        weight_flat = l2_weights.reshape(G * N, K).contiguous().view(-1)
    else:
        weight_flat = l2_weights.reshape(G * K, N).contiguous().view(-1)
    num_tokens = int(symm.num_tokens)
    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=device)
    output_d = output.view(-1)
    topk_indices_d = topk_indices.contiguous().view(-1)
    num_tokens_d = symm.num_tokens_per_rank
    # Gate tensors are traced out when with_gate is False; the kernel still needs an arg.
    dummy = _get_dummy_tensor(device)
    grad_gate_d = grad_gate.contiguous().view(-1) if with_gate else dummy
    d_topk_w = torch.empty(num_combine_slots, dtype=torch.float32, device=device) if with_gate else None
    d_topk_w_d = d_topk_w if with_gate else dummy

    # Sender-side dedup is the only combine path.
    assert dedup_key_row.numel() > 1, "combine needs the dispatch dedup tables; run dispatch with dedup=True"

    _compiled_grouped_gemm_combine(
        act_2d,
        weight_flat,
        tile_to_expert,
        num_tile_blocks,
        recv_dst_rank,
        recv_start_row,
        recv_count,
        pool_src_slot,
        output_d,
        topk_indices_d,
        num_tokens_d,
        grad_gate_d,
        d_topk_w_d,
        sorted_slot_ids,
        dedup_key_row,
        source_slot_kind,
        sym_buffer,
        c_n,
        num_combine_blocks=int(_PINNED_CONFIG[layout]["num_combine_blocks"]),
        COMBINE_PARITY=symm._combine_parity,
        COMBINE_EXPECTED=symm._combine_expected,
        REDUCE_EXPECTED=symm._reduce_expected,
        out_features=out_features,
        hidden_size=hidden_size,
        num_max_pool_tokens=num_max_pool_tokens,
        BLOCK_M=BM,
        BLOCK_N=BN,
        num_combine_slots=int(num_combine_slots),
        topk=int(topk),
        num_experts=int(num_experts),
        rank=int(rank),
        num_ranks=int(symm.world),
        num_max_tokens_per_rank=int(symm.num_max_tokens_per_rank),
        layout_code=_LAYOUT_CODES[layout],
        apply_weights=bool(apply_weights),
        with_gate=bool(with_gate),
        out_fp16=False,
        dedup_npass=int(_COMBINE_DEDUP_NPASS),
        seg_parts=int(_COMBINE_SEG_PARTS),
        lead_gemm_blocks=int(_PINNED_CONFIG[layout]["lead_gemm_blocks"]),
        stream=torch.cuda.current_stream(),
    )
    return output, d_topk_w
