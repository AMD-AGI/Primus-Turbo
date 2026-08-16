###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import os

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
    read_clock,
    spin_timed_out,
)
from primus_turbo.flydsl.mega.symm_buffer import (
    COMBINE_FLAG_STRIDE,
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.mega.tune_utils import (
    Config,
    autotune,
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

# accumulator chunks live per pass; 2 keeps the gather-reduce under the GEMM VGPR budget
_COMBINE_DEDUP_NPASS = int(os.environ.get("TURBO_COMBINE_DEDUP_NPASS", "2"))

_H_SOURCE_SLOT_KIND = 13
_H_SORTED_DISPATCH_SLOT_IDS = 19
_H_DEDUP_KEY_ROW = 20

# Grid role split, pinned in source per layout (see _compiled_grouped_gemm_combine).
# TURBO_COMBINE_CCRC="nt:48/768,nn:32/768" overrides for a sweep; unset in production.
# Measured (rc=768, both cases per run, ms):
#   nt  cc=64 2.479/2.489 | 56 2.440/2.448/2.448 | 48 2.441/2.426/2.431/2.429 | 40 2.446
#   nn  cc=48 4.326       | 40 4.271            | 32 4.064/4.094/4.065       | 24 4.199/4.217 | 16 4.776 | 8 7.145
# nt sits on a flat 40-56 plateau ~2.43 and falls off a cliff at 64; nn has a sharp
# optimum at 32 and degrades hard in both directions -- its longer window means the
# pushes it must keep ahead of are denser per unit time, so it is push-bound below 32
# while every CU past 32 is stolen from a GEMM that is already the critical path.
_CC_PINNED = {"nt": 48, "nn": 32, "tn": 64}
_RC_PINNED = {"nt": 768, "nn": 768, "tn": 768}

# combine_flag is a purely intra-device handoff (never sym.map()'d to a peer), so agent
# scope is *semantically* sufficient -- but it is measurably WORSE: nt 2.421 -> 2.446,
# nn 4.062 -> 4.085. The 8 XCDs have private, mutually incoherent L2s, so an agent-scope
# atomic/poll on a line shared by GEMM producers and combine consumers sitting on different
# XCDs still has to resolve past L2 and now pays line migration on top. sys scope keeps it a
# straight far-atomic at the fabric point. Keep "sys"; the knob exists only to re-run the A/B.
_COMBINE_FLAG_SCOPE = os.environ.get("TURBO_COMBINE_FLAG_SCOPE", "sys")

# Spin backoff for the two producer->consumer gates, in s_sleep units (~64 clocks each).
# Both gates used to spin at s_sleep(1), i.e. re-read every ~64 clocks. That poll is a
# scope="sys" load, so it bypasses L2 and costs a fabric/memory round trip EVERY time, and
# each iteration additionally issues s_memrealtime (+ s_waitcnt lgkmcnt(0)) for the watchdog.
# The combine gate has num_combine_cu pollers running for the whole kernel; the reduce gate
# has up to 8 warps x ~200 resident reduce blocks polling through the tail. At s_sleep(1)
# that is a continuous uncached read storm aimed at exactly the lines the release atomics
# need, on a kernel that is already co-critical on memory. Backing off costs at most a
# fraction of a microsecond of wake latency per gate crossing.
#
# 8 -> 32 measured over 22 counterbalanced canonical runs: nt -1.00%, nn -0.41%
# (t = -3.10 / -2.97). Fitting the observed arms to Dt = a/(64s + c) + b*s recovers an
# interior optimum at s* ~= 32.3 with a very flat basin -- 24 and 48 sit within 0.1% of
# it, i.e. far under the ~0.5% single-run sd, so there is nothing left to sweep here.
# Move this gate and _REDUCE_GATE_SLEEP together: both asymmetric arms measured worse
# than either symmetric one. (S_SLEEP takes SIMM16[6:0], so N <= 127 regardless.)
#
# CAVEAT, measured after COMBINE_FLAG_STRIDE landed: that -1.00%/-0.41% was fitted on the
# DENSE flag array, where the basin was steep because a slow poll rate was stealing a
# contended line. On the padded build 8 and 32 are a dead heat (nt 2.387/nn 4.036 at 8 vs
# nt 2.392,2.382/nn 4.022,4.048 at 32), i.e. padding absorbed most of what backoff was
# buying. 32 is kept as the better-evidenced constant, but treat this knob as spent --
# the remaining win is in the flag traffic itself, not in how often it is sampled.
_COMBINE_GATE_SLEEP = int(os.environ.get("TURBO_COMBINE_GATE_SLEEP", "32"))


def _combine_role_split(layout):
    override = os.environ.get("TURBO_COMBINE_CCRC", "")
    if override:
        table = dict(part.split(":", 1) for part in override.split(",") if part)
        if layout in table:
            cc, _, rc = table[layout].partition("/")
            return int(cc), int(rc)
    return _CC_PINNED[layout], _RC_PINNED[layout]


def _make_grouped_gemm_combine(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_combine_cu,
    num_reduce_cu,
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
    gemm_base = num_combine_cu
    # Per-tile-row rotation of block_n that cancels the XCD drift of the row-major tile
    # map (see the use site). Only enabled for the nt layout: measured a consistent win
    # there and a consistent loss for nn, whose B operand is walked column-wise so the
    # rotation buys no extra panel residency and only perturbs the A-tile sharing.
    _N_ROT = ((n_blocks - (n_blocks % 8)) % n_blocks) if (layout == "nt" and n_blocks) else 0

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
        TOPK_WEIGHTS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        D_TOPK_W: fx.Tensor,
        SORTED_SLOT_IDS: fx.Tensor,
        DEDUP_KEY_ROW: fx.Tensor,
        SOURCE_SLOT_KIND: fx.Tensor,
        sym_buffer: SymBuffer,
        c_n: fx.Int32,
        COMBINE_PARITY: fx.Tensor,
        COMBINE_EXPECTED: fx.Tensor,
        REDUCE_EXPECTED: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # build the layout from explicit dims (bf16 path -> TOKEN_DTYPE); the token pools are
        # out_features-wide (the model hidden), not the down-proj K (hidden_size)
        # Early-out gate for the worst-case padding blocks. The grid is sized for
        # worst_case_tiles*n_blocks so the launch shape stays static under HIP-graph
        # capture, but at run time only gemm_base + real_tiles*n_blocks + num_reduce_cu
        # blocks have anything to do -- the other ~87% of workgroups used to execute the
        # whole prologue (epoch parity load, expected-count loads, ~20 buffer descriptors)
        # before discovering they were empty. Each holds a full 128 KB LDS slot (one WG per
        # CU) for the duration, so that prologue latency serializes across the padding
        # rounds. Gating on real_tiles alone leaves a single dependent scalar load.
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        blocks_live = (
            fx.Int32(gemm_base) + real_tiles * fx.Int32(n_blocks) + fx.Int32(num_reduce_cu)
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
            # read epoch (already bumped by the bump kernel): parity -> bank, expected -> spin target.
            # Only the parity load itself is shared by all three roles, so it is the only
            # thing that stays here. Everything downstream of it -- in particular the two
            # expected-count loads, which are a SECOND dependent global load hanging off
            # `combine_parity` -- is sunk into the branch that actually consumes it. The
            # GEMM role is ~7.4k of the ~7.5k live workgroups and needs none of it until
            # its epilogue, so the old prologue put a 2-deep dependent load chain plus ~20
            # buffer descriptors in front of every tile's first B fetch.
            combine_parity_res = create_buffer_resource(COMBINE_PARITY, max_size=True)
            combine_parity = cast(
                buffer_load(combine_parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
            )
            # combine_flag counters are strided one per cache line (COMBINE_FLAG_STRIDE
            # i64 = 128B) so that the ~7 block_m in flight at any instant, each drawing
            # n_blocks release atomics against a continuous sys-scope poll, stop sharing
            # one line. Fold the stride into the bank so the per-index cost is a shift.
            combine_bank = combine_parity * fx.Int32(worst_case_tiles * COMBINE_FLAG_STRIDE)
            # These are constant-offset i64 adds on the symmetric base -- no loads, a few
            # SALU each -- and they must be evaluated here regardless: the AST rewriter
            # treats `workspace.<method>()` inside a traced branch as if/else state and
            # rejects it, since a Workspace is not an MLIR value.
            combine_flag_base = workspace.get_combine_flag_ptr()
            l2_token_buffer_base = workspace.get_l2_token_buffer_ptr()
            comb_base = workspace.get_combine_token_buffer_ptr()
            reduce_flag_base = workspace.get_reduce_flag_ptr()
            gate_base = workspace.get_combine_gate_ptr() if with_gate else None

            if block_index < combine_cu:
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
                # recv-segment table + origin slots ride the handle (per-forward), NOT shared symm -> else bwd reads stale.
                recv_dst_rank_res = create_buffer_resource(RECV_DST_RANK, max_size=True)
                recv_start_row_res = create_buffer_resource(RECV_START_ROW, max_size=True)
                recv_count_res = create_buffer_resource(RECV_COUNT, max_size=True)
                origin_slot_res = create_buffer_resource(POOL_SRC_SLOT, max_size=True)
                sorted_slot_res = create_buffer_resource(SORTED_SLOT_IDS, max_size=True)
                key_row_res = create_buffer_resource(DEDUP_KEY_ROW, max_size=True)
                grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
                seg_local = (fx.Int32(num_experts) - block_index + combine_cu - fx.Int32(1)) // combine_cu
                # Dedup gathers rows below the segment, so it gates on the whole tile prefix.
                # The cursor rides seg_iter (task_index is row-ordered), so each tile polls once.
                combine_cursor = fx.Int32(0)
                for seg_iter in range(seg_local):
                    task_index = block_index + seg_iter * combine_cu
                    seg_start = buffer_load(recv_start_row_res, task_index, vec_width=1, dtype=fx.T.i32())
                    seg_count = buffer_load(recv_count_res, task_index, vec_width=1, dtype=fx.T.i32())
                    if seg_count > fx.Int32(0):
                        t1 = (seg_start + seg_count - fx.Int32(1)) // fx.Int32(BLOCK_M)
                        tile_cursor = combine_cursor
                        if thread_index == fx.Int32(0):
                            while tile_cursor <= t1:
                                spin_start = read_clock()
                                fx.rocdl.s_waitcnt(0)
                                signal_count = ld(
                                    combine_flag_base,
                                    combine_bank + tile_cursor * fx.Int32(COMBINE_FLAG_STRIDE),
                                    scope=_COMBINE_FLAG_SCOPE,
                                    dtype=fx.T.i64(),
                                )
                                while signal_count != expected_combine_i64:
                                    fx.rocdl.s_sleep(fx.Int32(_COMBINE_GATE_SLEEP))
                                    if spin_timed_out(spin_start):
                                        fx.printf(
                                            "MEGA combine(task) gate timeout: tile={} signal={} thr={}\n",
                                            tile_cursor,
                                            signal_count,
                                            expected_combine_i64,
                                        )
                                        spin_start = read_clock()
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
                            task_index=task_index,
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
                gemm_tile_index = block_index - fx.Int32(gemm_base)
                block_m = gemm_tile_index // fx.Int32(n_blocks)
                block_n = gemm_tile_index % fx.Int32(n_blocks)
                if _N_ROT != 0:
                    # XCD-affine N rotation. Workgroups round-robin over the 8 XCDs by
                    # index, so under the plain row-major tile map the n-panel a given XCD
                    # works on drifts by (n_blocks % 8) every tile row; over the kernel each
                    # XCD ends up touching about twice as many distinct B panels as its L2
                    # slice can hold. Rotating block_n by that same drift per row cancels it,
                    # pinning each XCD to a fixed residue class of block_n whose panels then
                    # stay resident. It is a per-row constant rotation, so still a bijection
                    # over the tile set, and block_m is untouched -- the combine gate's
                    # tile-prefix ordering is unchanged.
                    block_n = (block_n + block_m * fx.Int32(_N_ROT)) % fx.Int32(n_blocks)
                if block_m < real_tiles:
                    # GEMM role: one real tile (block_m, block_n) per block (unchanged).
                    group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
                    group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    group_base = group_index * fx.Int32(K) * c_n
                    # A base = ACT tensor; C base = l2_token_buffer (int64 symm addr).
                    act_base = fx.arith.ArithValue(
                        fx.arith.index_cast(fx.T.i64(), extract_base_index(ACT)), signed=True
                    )
                    # Fold per-tile base in int64 (pool >4GB), voffset stays int32. A: precise bound; C: HW num_records via 0x40000000.
                    a_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * K * 2)
                    c_off = cast(block_m, fx.T.i64()) * fx.Int64(BLOCK_M * 2) * cast(c_n, fx.T.i64())
                    A_tile = make_bf16_fp16_tile_tensor(act_base, a_off, BLOCK_M * K)
                    C_tile = make_bf16_fp16_tile_tensor(l2_token_buffer_base, c_off, 0x40000000)
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
                        c_cache_modifier=18,  # sc1|nt: agent-visible non-temporal local stage.
                        # out_features % BLOCK_N == 0 is asserted above and the grid only
                        # ever produces block_n < out_features // BLOCK_N, so every column
                        # this epilogue writes is in range: the store's per-element column
                        # mask is statically true and can be dropped.
                        n_exact=True,
                    )
                    # Release rendezvous. Per-wave release (each wave drains its own stores
                    # and signals with its own lane-0 atomic, consumer counting _NUM_WARPS
                    # signals per tile) was measured and is a large LOSS: nt 2.420 -> 2.690,
                    # nn 4.084 -> 4.308. The 8x sys-scope atomic traffic lands on the one
                    # 64-bit combine_flag word that the combine warps are simultaneously
                    # polling with sys-scope loads, so every extra atomic steals the line
                    # from the poller. The whole-workgroup rendezvous is cheaper than the
                    # coherence ping-pong it avoids -- do not re-try this.
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()
                    # Keep a separator: LLVM folds adjacent barriers, but two rendezvous are required.
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
                    # Empty region: first num_reduce_cu blocks do topk reduce, rest early-exit.
                    empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                    if empty_ordinal < fx.Int32(num_reduce_cu):
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
                        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
                        gate_local_res = (
                            create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records)
                            if with_gate
                            else None
                        )
                        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True) if with_gate else None
                        kind_res = create_buffer_resource(SOURCE_SLOT_KIND, max_size=True)
                        # Never-reset alignment: reduce blocks bump empty block_m's combine_flag to cumulative expected.
                        n_empty = fx.Int32(worst_case_tiles) - real_tiles
                        reduce_stride = fx.Int32(num_reduce_cu)
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
                            n_reduce_tiles < fx.Int32(num_reduce_cu), n_reduce_tiles, fx.Int32(num_reduce_cu)
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
                            topk_weights_res,
                            gate_local_res,
                            d_topk_w_res,
                            expected_reduce_i64,
                            dedup=True,
                            kind_res=kind_res,
                            num_combine_slots=num_combine_slots,
                        )

    return grouped_gemm_combine_kernel


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


@autotune(
    # 96/128 are headroom for wider folds; DSv3 still tunes to 64 (nt) / 32 (nn).
    configs=[Config(num_combine_cu=cc, num_reduce_cu=rc) for cc in (16, 32, 64, 96, 128) for rc in (256,)],
    # layout_code MUST be a key: nt/nn have OPPOSITE combine_cu optima (see wrapper note).
    key=[
        "out_features",
        "hidden_size",
        "num_max_pool_tokens",
        "BLOCK_M",
        "BLOCK_N",
        "num_combine_slots",
        "topk",
        "num_experts",
        "rank",
        "layout_code",
        "apply_weights",
        "with_gate",
        "out_fp16",
    ],
    rep=5,
)
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
    TOPK_WEIGHTS,
    GRAD_GATE,
    D_TOPK_W,
    SORTED_SLOT_IDS,
    DEDUP_KEY_ROW,
    SOURCE_SLOT_KIND,
    sym_buffer,
    c_n,
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
    stream: fx.Stream,
    num_combine_cu: fx.Constexpr[int] = 64,
    num_reduce_cu: fx.Constexpr[int] = 256,
    nt_vmcnt: fx.Constexpr[int] = 3,
    agpr_alloc: fx.Constexpr[int] = 0,
    waves: fx.Constexpr[int] = 2,
):
    # Grid role split, pinned here rather than swept: the autotune cache is keyed on
    # shapes+layout only and carries no code version, so the values it serves are
    # whatever a parent-era sweep happened to pick, and widening the swept config list
    # silently does nothing for shapes that have been tuned once. Pinning both makes
    # the launch geometry a property of the source, not of ~/.flydsl.
    #
    # num_combine_cu trades push CUs against math CUs: the GEMM makespan scales as
    # 1/(256 - cc) at ~39 tile rounds, so every 8 CUs handed to combine costs ~3.5% of
    # the GEMM window, and combine only pays that back while pushes are the binding
    # constraint. nt (forward L2) and nn (backward L1 dgrad) push the same ~0.94 GB but
    # nn's window is 1.65x longer, so nn needs fewer push CUs to stay ahead -- hence the
    # asymmetric pin.
    #
    # Reduce-region width:
    # 256 reduce blocks cannot all be resident -- one workgroup per CU (128 KB LDS) and
    # the combine region already owns 32-48 of the 256 CUs -- so the reduce tail ran as
    # two ragged rounds, the second only ~a quarter full. Splitting the same token work
    # over 768 blocks makes each slice ~3x shorter, so the tail packs into whatever CUs
    # free up as GEMM tiles retire and the final ragged round costs a third as much.
    # The extra blocks are free: they are carved out of the ~50k padding blocks the
    # worst-case grid already launches. Measured nt 2.486 -> 2.465, nn 4.106 -> 4.064 ms.
    num_combine_cu, num_reduce_cu = _combine_role_split(_LAYOUTS[layout_code])
    kernel = _make_grouped_gemm_combine(
        out_features,
        hidden_size,
        num_max_pool_tokens,
        BLOCK_M,
        BLOCK_N,
        num_combine_cu,
        num_reduce_cu,
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
    )
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    grid_size = num_combine_cu + worst_case_tiles * n_blocks
    # bump epoch on device (combine += n_blocks, reduce += 1) before the GEMM; same-stream visible
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
        TOPK_WEIGHTS,
        GRAD_GATE,
        D_TOPK_W,
        SORTED_SLOT_IDS,
        DEDUP_KEY_ROW,
        SOURCE_SLOT_KIND,
        sym_buffer,
        c_n,
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
    tile_to_expert = handle[5]
    num_tile_blocks = handle[8]
    recv_dst_rank = handle[9]
    recv_start_row = handle[10]
    recv_count = handle[11]
    pool_src_slot = handle[12]
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

    dummy = num_tile_blocks

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
    topk_weights_d = topk_weights.contiguous().view(-1) if apply_weights else dummy
    grad_gate_d = grad_gate.contiguous().view(-1) if with_gate else dummy
    d_topk_w = torch.empty(num_combine_slots, dtype=torch.float32, device=device) if with_gate else None
    d_topk_w_d = d_topk_w if with_gate else dummy

    # Sender-side dedup is the only combine path: it folds a token's local routes into
    # one push, so combine moves exactly the bytes dispatch did.
    assert len(handle) > _H_DEDUP_KEY_ROW and handle[_H_DEDUP_KEY_ROW].numel() > 1, (
        "combine needs the dispatch dedup tables; run dispatch with dedup=True"
    )
    sorted_slot_ids = handle[_H_SORTED_DISPATCH_SLOT_IDS]
    dedup_key_row = handle[_H_DEDUP_KEY_ROW]
    source_slot_kind = handle[_H_SOURCE_SLOT_KIND]

    # epoch advance moved inside _compiled_grouped_gemm_combine (autotune-safe, no rewind)
    # num_combine_cu / num_reduce_cu are tunable per shape+layout (nt/nn optima differ).
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
        topk_weights_d,
        grad_gate_d,
        d_topk_w_d,
        sorted_slot_ids,
        dedup_key_row,
        source_slot_kind,
        sym_buffer,
        c_n,
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
        stream=torch.cuda.current_stream(),
    )
    return output, d_topk_w
