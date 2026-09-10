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

"""Dual-wave, software-pipelined flash-attention kernel for gfx950 (D=64/128, bf16/fp16).

Dispatched when gpu_arch >= gfx950, head_dim in (64, 128) and seq_len >= 384.
seq_len need not be a multiple of 256/64: partial q-blocks and odd kv-tile counts
are covered by num_records bounds, an even-rounded tile count and a kv pad mask.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from primus_turbo.flydsl.utils.attn_helper import (
    DualwaveKernelContext,
    _anchor_v_o,
    _anchor_v_p,
    _dualwave_sync_barrier,
    _make_dualwave_swp_traits,
    _s_barrier,
    _s_nop,
    _s_setprio,
    _s_waitcnt,
    _sched_barrier,
    _sched_barrier_pairs,
    _waitcnt_vm_n,
    dtype_to_elem_type,
)


def build_flash_attn_dualwave_swp_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_fixed_max=None,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    varlen=False,
    cross_seqlen=False,
    emit_lse=False,
    window_left=-1,
    block_m=None,
    gqa_merge=None,
    sbhd=False,
    has_sink=False,
):
    """Build a DUALWAVE_SWP flash_attn launcher for D=64/128 bf16/f16 on gfx950.

    Supports dense (SBHD) and varlen packed QKV (THD) layouts. has_sink folds a learned
    per-q-head attention sink (SINK[Hq] fp32) into the online-softmax denominator.
    """
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(
            f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}"
        )
    if head_dim not in (64, 128):
        raise RuntimeError(f"flash_attn_dualwave_swp supports D=64 or D=128 only, got head_dim={head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise RuntimeError(f"flash_attn_dualwave_swp supports bf16/f16 only, got dtype={dtype_str}")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    traits = _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_fixed_max=dualwave_swp_fixed_max,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        emit_lse=emit_lse,
        window_left=window_left,
        block_m=block_m,
        gqa_merge=gqa_merge,
        sbhd=sbhd,
        has_sink=has_sink,
    )
    _dualwave_swp_cache_tag = traits.cache_tag

    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_dualwave_swp_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        SINK: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
    ):
        ctx = DualwaveKernelContext(
            traits,
            Q,
            K,
            V,
            O,
            DebugCounts,
            CuSeqQ,
            CuSeqKv,
            BlockTable,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            block_table_stride,
            SINK=SINK,
        )
        ctx._setup(SharedStorage)

        active = ctx.active
        elem_dtype = ctx.elem_dtype
        # A tile's LDS buffer is not overwritten until two barriers later, so its drain can sink
        # into the consuming compute cluster; stagger spends that same slack, hence the exclusion.
        sink_drains = not traits.DUALWAVE_SWP_ENABLE_STAGGER
        # Keeping only part of each K/V tile in registers (the rest read from LDS in the consuming
        # compute cluster) is what fits the kernel under the 4-waves-per-SIMD register budget.
        split_kv_reads = sink_drains
        # Issuing the overwrite DMA in the compute cluster leaves the memory cluster holding LDS
        # reads only, which is what makes fold_mem_barriers legal.
        late_dma = sink_drains
        # QK k-steps whose K packs are read before the first MFMA group; the rest follow it.
        K_HEAD = traits.K_STEPS_QK // 2
        # P*V k-substeps of V read up front; substep k + V_HEAD is issued after step k.
        V_HEAD = 1
        split_k_reads = split_kv_reads and K_HEAD < traits.K_STEPS_QK
        # The write-after-read edge is already covered by the compute-cluster barrier preceding each
        # overwrite DMA, so the memory-cluster rendezvous protects nothing (and costs 8 waves merged).
        fold_mem_barriers = sink_drains and traits.Q_HEADS_PER_WG > 1
        fold_pv_barriers = fold_mem_barriers and late_dma
        vm_drain_sync = 0 if fold_pv_barriers else ctx.VM_DRAIN_KV
        # Drain length. The pipeline's legal trip counts are set by
        # the drain: three named tiles means N even, two means N odd. The windowed body
        # needs exactly three KV tiles, so it takes the two-tile drain and attn_helper.py
        # keeps its span odd. Every other configuration keeps the three-tile drain.
        short_drain = traits.CAUSAL and traits.WINDOW_LEFT >= 0

        def _mem_cluster_sync():
            if const_expr(fold_mem_barriers):
                _sched_barrier(0)
            else:
                _dualwave_sync_barrier()

        def _qk_cluster_sync(sched_region=False):
            if const_expr(sink_drains):
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(vm_drain_sync)
            # ⚠ The `const_expr` branch below is NOT dead code, however much it looks like it.
            #
            # `sched_region` is False at every call site, so the condition never holds and the
            # emitted instruction sequence is the same either way -- the `else` arm is always
            # what runs. Collapsing the branch to a bare `_dualwave_sync_barrier()` nevertheless
            # costs **5%** at head_dim 64 dense (1261 -> 1199 TFLOP/s, s=8192, measured
            # palindromically against the branch-ful build in one session, both arms within
            # 0.2%). Bisected to this function: the same cleanup applied to `attn_helper.py`
            # alone is neutral, and restoring only this branch recovers the whole 5%.
            #
            # The tracer evidently makes a region of the `const_expr` arms, and that region
            # boundary reaches the scheduler. This kernel is unusually sensitive to exactly
            # that: it is power-bound with the vector pipe as the secondary constraint, and
            # sweeping the `sched_group_barrier` anchor count alone moves it 4%. So the branch
            # is a scheduling fence that happens to be spelled as a condition.
            #
            # This is understood only that far. It is kept because removing it is measurably
            # worse, not because the mechanism is settled -- if you find the real one, replace
            # this with the fence it deserves and delete the note.
            if const_expr(sched_region):
                _sched_barrier(0)
                _sched_barrier(0)
            else:
                _dualwave_sync_barrier()

        def _pv_cluster_sync():
            if const_expr(fold_pv_barriers):
                _sched_barrier(0)
            else:
                _qk_cluster_sync()

        if const_expr(traits.DUALWAVE_SWP_MFMA_ROWSUM):
            l_row_init = ctx.c_zero_v4f32
        else:
            l_row_init = ctx.c_zero_f
        split_t_end = ctx.split_t_end
        v_o_zero = ctx.c_zero_v16f32

        def _main_body():
            ctx.load_k_split(0, 0)
            _s_waitcnt(0)
            _sched_barrier(0)
            _s_barrier()

            q_all_bf16 = ctx.load_all()
            q_all_scaled_bf16 = ctx.scale_all(q_all_bf16)

            def _load_k_head(buf_id):
                if const_expr(split_k_reads):
                    return ctx.lds_load_k(buf_id, ks_range=(0, K_HEAD))
                return ctx.lds_load_k(buf_id)

            def _qk(v_k, buf_id, tile_idx=None, live=None):
                """QK for one KV tile, issuing the tail K packs between the two MFMA
                groups so only the head half is resident across the softmax cluster.

                `tile_idx` is passed only on the two band-edge
                tiles of the windowed body, where one of the tile's two 32-column MFMA
                chains is wholly masked for each wave row group. See
                `attn_helper.chunk_live`."""
                if const_expr(tile_idx is None and live is None):
                    if const_expr(not split_k_reads):
                        return ctx.qk(v_k, q_all_scaled_bf16)
                    ks_tail = (K_HEAD, traits.K_STEPS_QK)
                    v_s = ctx.qk(v_k, q_all_scaled_bf16, ks_range=(0, K_HEAD))
                    v_k = ctx.lds_load_k(buf_id, ks_range=ks_tail, k_regs=v_k)
                    return ctx.qk(v_k, q_all_scaled_bf16, v_s=v_s, ks_range=ks_tail)
                # `live` lets a call site supply the pair directly and
                # pass None for a half it has proved is always live, so no branch is emitted
                # for it. `chunk_live` alone would hand back a predicate that is always true
                # there, which costs a scalar compare and an scf.if for nothing.
                if const_expr(live is None):
                    live_lo, live_hi = ctx.chunk_live(tile_idx)
                else:
                    live_lo, live_hi = live
                if const_expr(not split_k_reads):
                    return ctx.qk_live(v_k, q_all_scaled_bf16, live_lo, live_hi)
                ks_tail = (K_HEAD, traits.K_STEPS_QK)
                v_s = ctx.qk_live(v_k, q_all_scaled_bf16, live_lo, live_hi, ks_range=(0, K_HEAD))
                v_k = ctx.lds_load_k(buf_id, ks_range=ks_tail, k_regs=v_k)
                return ctx.qk_live(v_k, q_all_scaled_bf16, live_lo, live_hi, v_s=v_s, ks_range=ks_tail)

            def _load_v_head(buf_id):
                if const_expr(split_kv_reads):
                    return ctx.lds_load_v(buf_id, substeps=(0, V_HEAD))
                return ctx.lds_load_v(buf_id)

            def _pv_step(step, v_p, v_v, v_o, buf_id):
                """One P*V k-substep, trailed by the read of the substep V_HEAD ahead so
                that read is covered by this MFMA and only V_HEAD + 1 substeps are live."""
                v_o = ctx.pv_step_k(step, v_p, v_v, v_o)
                nxt = step + V_HEAD
                if const_expr(split_kv_reads and nxt < 4):
                    ctx.lds_load_v(buf_id, substeps=(nxt, nxt + 1), packs=v_v)
                return v_o

            def _pv(v_p, v_v, v_o, buf_id):
                if const_expr(not split_kv_reads):
                    return ctx.pv(v_p, v_v, v_o)
                for step in range_constexpr(4):
                    v_o = _pv_step(step, v_p, v_v, v_o, buf_id)
                return v_o

            # DEAD HALF-TILE P*V AND V LDS ELISION.
            # `chunk_live` already skips the QK MFMA chain of the 32-column half-tile that
            # is provably wholly masked for this wave (see `attn_helper.chunk_live`). Those
            # scores are then -inf, so P is exactly 0 across that half and its two P*V steps
            # add exactly 0*V to the accumulator -- the elision is bit-exact for the same
            # reason g28 was. What g28 did NOT remove, and what these two wrappers do, is
            # the feeding `ds_read_b64_tr_b16` stream: one V substep is 2 * D_CHUNKS reads,
            # so skipping two substeps removes 8 LDS instructions per wave at hd64, on top
            # of 2 * D_CHUNKS = 4 MFMAs. Both wrappers are inside `short_drain`, so no dense
            # shape sees them. The predicate is wave-uniform (`q_start_pos_i32` comes from
            # readfirstlane), hence a scalar branch, not divergence.

            def _pv_branch(v_o, live, body):
                """Run `body` on the accumulator list under a wave-uniform scalar branch.
                `flyc.jit` cannot carry a Python list across an `scf.if` ("state variable
                is list, not an MLIR Value"), so the D_CHUNKS accumulators are passed and
                yielded positionally. D_CHUNKS is 2 at hd64 and 4 at hd128."""
                if const_expr(traits.D_CHUNKS == 2):

                    @flyc.jit
                    def _branch2(o0, o1, live):
                        if live:
                            _r = body([o0, o1])
                            o0 = _r[0]
                            o1 = _r[1]
                        return o0, o1

                    return list(_branch2(v_o[0], v_o[1], live))

                @flyc.jit
                def _branch4(o0, o1, o2, o3, live):
                    if live:
                        _r = body([o0, o1, o2, o3])
                        o0 = _r[0]
                        o1 = _r[1]
                        o2 = _r[2]
                        o3 = _r[3]
                    return o0, o1, o2, o3

                return list(_branch4(v_o[0], v_o[1], v_o[2], v_o[3], live))

            def _pv_hi_live(v_p, v_v, v_o, buf_id, live_hi):
                """P*V for a tile whose HI 32 columns may be wholly masked -- the band's
                LAST tile, for the low q-row group. Steps 0/1 always run; steps 2/3 and the
                substep-2/3 V reads sit under the branch, so this site removes 2 * D_CHUNKS
                MFMAs and 2 * 2 * D_CHUNKS `ds_read_b64_tr_b16` (8 at hd64)."""
                v_o = ctx.pv_step_k(0, v_p, v_v, v_o)
                if const_expr(split_kv_reads):
                    ctx.lds_load_v(buf_id, substeps=(1, 2), packs=v_v)
                v_o = ctx.pv_step_k(1, v_p, v_v, v_o)

                def _tail(v_o):
                    if const_expr(split_kv_reads):
                        ctx.lds_load_v(buf_id, substeps=(2, 4), packs=v_v)
                    v_o = ctx.pv_step_k(2, v_p, v_v, v_o)
                    return ctx.pv_step_k(3, v_p, v_v, v_o)

                return _pv_branch(v_o, live_hi, _tail)

            def _pv_lo_live(v_p, v_v, v_o, buf_id, live_lo):
                """P*V for a tile whose LO 32 columns may be wholly masked -- the band's
                FIRST tile, for the high q-row group. Substep 2 is hoisted above the branch
                because step 2 needs it on both paths, and substep 0 was already read by the
                preceding memory cluster, so this site removes 2 * D_CHUNKS MFMAs but only
                2 * D_CHUNKS reads (4 at hd64) rather than 8."""
                if const_expr(split_kv_reads):
                    ctx.lds_load_v(buf_id, substeps=(2, 3), packs=v_v)

                def _head(v_o):
                    v_o = ctx.pv_step_k(0, v_p, v_v, v_o)
                    if const_expr(split_kv_reads):
                        ctx.lds_load_v(buf_id, substeps=(1, 2), packs=v_v)
                    return ctx.pv_step_k(1, v_p, v_v, v_o)

                v_o = _pv_branch(v_o, live_lo, _head)
                if const_expr(split_kv_reads):
                    ctx.lds_load_v(buf_id, substeps=(3, 4), packs=v_v)
                v_o = ctx.pv_step_k(2, v_p, v_v, v_o)
                return ctx.pv_step_k(3, v_p, v_v, v_o)

            def _pv_both_live(v_p, v_v, v_o, buf_id, live_lo, live_hi):
                """P*V for the band's LAST tile, where BOTH 32-column
                halves can be wholly masked for a wave. Composition of `_pv_lo_live` and
                `_pv_hi_live` with nothing hoisted between them: steps 0/1 and the substep-1
                read sit under `live_lo`, steps 2/3 and the substep-2/3 reads under
                `live_hi`. Substep 0 was already read by the preceding memory cluster. For
                the two low q-row groups both predicates are false and the tile then
                contributes no MFMA and no `ds_read_b64_tr_b16` at all, which is what
                `_pv_hi_live` alone could not do -- it runs steps 0/1 unconditionally.
                Bit-exact for the same reason as the QK and P*V elisions above: the skipped
                columns are exactly the ones the causal mask writes -inf over, so their P is
                exactly 0 and their P*V contribution is exactly 0*V."""

                def _head(v_o):
                    v_o = ctx.pv_step_k(0, v_p, v_v, v_o)
                    if const_expr(split_kv_reads):
                        ctx.lds_load_v(buf_id, substeps=(1, 2), packs=v_v)
                    return ctx.pv_step_k(1, v_p, v_v, v_o)

                v_o = _pv_branch(v_o, live_lo, _head)

                def _tail(v_o):
                    if const_expr(split_kv_reads):
                        ctx.lds_load_v(buf_id, substeps=(2, 4), packs=v_v)
                    v_o = ctx.pv_step_k(2, v_p, v_v, v_o)
                    return ctx.pv_step_k(3, v_p, v_v, v_o)

                return _pv_branch(v_o, live_hi, _tail)

            ctx.load_k_split(1, 1)
            ctx.load_v_split(0, 0)
            v_k = _load_k_head(0)
            _sched_barrier(0)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.VM_DRAIN_V)

            _sched_barrier(0)
            _dualwave_sync_barrier()

            if const_expr(short_drain):
                v_s_0 = _qk(v_k, 0, tile_idx=ctx.split_tile(0))
            else:
                v_s_0 = _qk(v_k, 0)
            _sched_barrier(0)

            if const_expr(traits.CAUSAL):
                # split_tile(0) = split_t0 (0 dense, swa_lo for SWA) -- needed so the
                # window/causal mask on the prologue tile uses the correct tile index.
                v_s_0 = ctx.causal_mask_split_prologue_if_needed(v_s_0)
            else:
                # Non-causal tiny seq_len needs tile-0 padding masked before the full-tile no-op gate.
                v_s_0 = ctx.seq_pad_mask_if_needed(v_s_0, ctx.split_tile(0))
            if const_expr(traits.DUALWAVE_SWP_FIXED_MAX):
                # Softmax is shift-invariant, so a zero reference max is valid; masked scores
                # already exp2 to 0, so fully-masked rows need no finite floor.
                m_row_pro = ctx.zero_row_max()
            else:
                m_row_pro = ctx.reduce_max(v_s_0)
                if const_expr(traits.CAUSAL):
                    # Floor fully-masked rows (-inf) to finite so exp2 yields 0, not NaN.
                    m_row_pro = ctx.floor_masked_max(m_row_pro)
            v_s_0 = ctx.shift_scores(v_s_0, m_row_pro)
            v_p_0 = ctx.exp2(v_s_0, 0, 16)
            _dualwave_sync_barrier()

            # Inner-loop tile indices are split_tile-relative: split_t0 = 0 dense/causal,
            # swa_lo for SWA, chunk start for split-K.
            loop_lb = ctx.split_tile(3)

            ctx.load_k_split(2, 0)

            # m_row IS NOT CARRIED. See the note at the unpack below.
            init_args = [l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(v_o_zero)
            init_args.append(v_p_0[0])
            init_args.append(v_p_0[1])
            loop_results = init_args
            for j, loop_args in range(
                loop_lb,
                split_t_end - fx.Index(1),
                fx.Index(2),
                init=init_args,
            ):
                m_row = m_row_pro
                l_row = loop_args[0]
                v_o = [loop_args[1 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_p_0 = (loop_args[1 + traits.D_CHUNKS], loop_args[2 + traits.D_CHUNKS])
                j_idx = j

                # Cluster 0: prefetch V buf1, read resident K for MMA0, and use carried page ids.
                _sched_barrier(0)
                if const_expr(not late_dma):
                    ctx.load_v_tile(j_idx - 2, 1)
                v_k = _load_k_head(1)
                if const_expr(not sink_drains):
                    _s_waitcnt(traits.LGKMCNT_0_ONLY)
                    _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _mem_cluster_sync()

                # Cluster 1 finishes v_p_0 softmax, updates l_row, casts P, then computes MMA0.
                if const_expr(late_dma and not fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 2, 1)
                # MMA0 issues after cast_p so its 32 fresh score regs never coexist with the
                # 32 f32 of the carried P; sched_group_barrier still interleaves the two.
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                _sched_barrier(0)
                v_s_1 = _qk(v_k, 1)
                _sched_barrier_pairs(traits, 6, 3, 1, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 1)
                _qk_cluster_sync()

                # Cluster 2 prefetches next K, reads this tile's V for P*V, then waits and syncs.
                _sched_barrier(0)
                if const_expr(not late_dma):
                    ctx.load_k_tile(j_idx, 1)
                v_v = _load_v_head(0)
                if const_expr(not sink_drains):
                    _s_waitcnt(traits.LGKMCNT_0_ONLY)
                    _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _mem_cluster_sync()

                # Cluster 3 computes P*V, row max, rescale, sub row, and first-half exp2.
                if const_expr(late_dma):
                    ctx.load_k_tile(j_idx, 1)
                if const_expr(fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 2, 1)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv_step(0, v_p_0, v_v, v_o, 0)
                # Cross-seqlen can put a diagonal tile in v_s_1; so can SWA's lower window edge.
                if const_expr(traits.CAUSAL and (traits.CROSS_SEQLEN or traits.WINDOW_LEFT >= 0)):
                    v_s_1 = ctx.causal_mask_prologue_if_needed(
                        v_s_1,
                        j_idx - 2,
                        kv_end_tile=j_idx - 1,
                    )
                else:
                    v_s_1 = ctx.scores_for_softmax(v_s_1)
                v_o, m_row, l_row, v_p_0 = ctx.tile_rescale_o(v_o, m_row, l_row, v_s_1, v_p_0, 2)
                for pvs in range_constexpr(1, 4):
                    v_o = _pv_step(pvs, v_p_0, v_v, v_o, 0)
                v_s_1 = ctx.shift_scores(v_s_1, m_row)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)

                _sched_barrier_pairs(traits, 6, 6, 2)
                # IGroupLP group 2 keeps softmax exp2 near its MFMA window.
                _sched_barrier_pairs(traits, 6, 3, 2, traits.SCHED_EXP_MASK)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _pv_cluster_sync()

                # Cluster 4 mirrors C0: prefetch V, read K into v_k, wait, and sync.
                _sched_barrier(0)
                if const_expr(not late_dma):
                    ctx.load_v_tile(j_idx - 1, 0)
                v_k = _load_k_head(0)
                if const_expr(not sink_drains):
                    _s_waitcnt(traits.LGKMCNT_0_ONLY)
                    _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _mem_cluster_sync()

                # Cluster 5 mirrors C1: finish v_p_1 softmax, update l_row, cast P, then MMA0.
                if const_expr(late_dma and not fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 1, 0)
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
                _sched_barrier(0)
                v_s_0 = _qk(v_k, 0)
                _sched_barrier_pairs(traits, 6, 3, 3, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 3)
                _qk_cluster_sync()

                # Cluster 6 prefetches next K, reads V packs, optionally masks v_s_0, waits, and syncs.
                _sched_barrier(0)
                if const_expr(not late_dma):
                    ctx.load_k_tile(j_idx + 1, 0)
                v_v = _load_v_head(1)
                # CLUSTER 6'S MASK IS DEAD ON THE PLAIN DENSE PATH.
                # Cluster 3 forty lines up already carries this exact predicate; Cluster 6 did
                # not, and called the mask on `CAUSAL` alone. It can never fire here: dense has
                # split_t0 = 0 and split_t_end = max_num_tiles = 2b+2 (even, no rounding) for
                # q_block b, the loop is `range(3, split_t_end - 1, 2)` so j <= split_t_end - 3,
                # and this call masks tile j-1 with kv_end_tile = j, i.e.
                # kv_end_pos = 64j <= 64(split_t_end-3) = 128b - 64. The guard inside is
                # `q_start_pos + delta < kv_end_pos` with q_start_pos >= 128b and delta = 0, so
                # it is false on every iteration for every wave row group. Masking is only ever
                # needed on tiles split_t_end-2 and split_t_end-1 -- the 128-row diagonal band is
                # exactly two 64-column tiles -- and both belong to the three-tile epilogue drain.
                # What the dead call still emits is a scalar compare, a branch, and an scf.if
                # whose merge carries 32 f32 of scores, inside a memory cluster that also carries
                # hand-written _waitcnt_vm_n. Bit-exact: the branch is proven not taken and the
                # else arm is `scores_for_softmax`, the identity on this configuration.
                if const_expr(traits.CAUSAL and (traits.CROSS_SEQLEN or traits.WINDOW_LEFT >= 0)):
                    v_s_0 = ctx.causal_mask_prologue_if_needed(
                        v_s_0,
                        j_idx - 1,
                        kv_end_tile=j_idx,
                    )
                else:
                    v_s_0 = ctx.scores_for_softmax(v_s_0)
                if const_expr(not sink_drains):
                    _s_waitcnt(traits.LGKMCNT_0_ONLY)
                    _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _mem_cluster_sync()

                # Cluster 7 mirrors C3 and carries m_row, l_row, v_o, and packed v_p_0.
                if const_expr(late_dma):
                    ctx.load_k_tile(j_idx + 1, 0)
                if const_expr(fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 1, 0)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv_step(0, v_p_1, v_v, v_o, 1)
                v_o, m_row, l_row, v_p_1 = ctx.tile_rescale_o(v_o, m_row, l_row, v_s_0, v_p_1, 4)
                for pvs in range_constexpr(1, 4):
                    v_o = _pv_step(pvs, v_p_1, v_v, v_o, 1)
                v_s_0 = ctx.shift_scores(v_s_0, m_row)
                v_p_0 = ctx.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 6, 5, 4)
                _sched_barrier_pairs(traits, 6, 3, 4, traits.SCHED_EXP_MASK)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _pv_cluster_sync()

                yield_args = [l_row] + v_o + [v_p_0[0], v_p_0[1]]
                loop_results = yield yield_args

            # Epilogue drains the final in-flight tiles without further prefetch-ahead.
            # m_row IS LOOP-INVARIANT, SO IT DOES NOT CROSS THE BACK EDGE.
            # In this tree the max-rebase machinery is compiled out unconditionally rather than
            # under a const_expr: attn_helper.py `tile_rescale_o` returns its (v_o, m_row, l_row,
            # v_p) verbatim, `tile_row_max` returns (m_row, None), and `shift_scores` IGNORES its
            # row_max argument. There is exactly one definition of each in the whole _vendor tree.
            # So inside the loop m_row was read out of the iter args, handed to two identities and
            # two functions that ignore it, and yielded back bit-identical to the block argument it
            # arrived as -- one per-lane f32 held live across the entire KV loop for a consumer in
            # the epilogue (`finalize_o_scale`). That is the same shape as the deferred sink
            # load, and worth +13.2% here. Substituting m_row_pro is bit-exact by construction: it is the same SSA
            # value the loop was returning.
            m_row = m_row_pro
            l_row = loop_results[0]
            v_o = [loop_results[1 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = (loop_results[1 + traits.D_CHUNKS], loop_results[2 + traits.D_CHUNKS])

            max_m3 = split_t_end - 3
            max_m2 = split_t_end - 2
            max_m1 = split_t_end - 1

            if const_expr(fold_pv_barriers):
                _dualwave_sync_barrier()

            if const_expr(short_drain):
                # TWO-TILE DRAIN.
                # The three-tile drain below with its middle stage removed. The last two
                # tiles are max_m2 and max_m1 and both of their K tiles are already
                # resident -- buf1 from the loop's C2 (or the prologue's load_k_split(1, 1)
                # when the loop does not run) and buf0 from the loop's C6 (or the
                # prologue's load_k_split(2, 0)) -- so this drain issues no K DMA at all.
                # Legal trip counts are N in {3, 5, 7, ...}.

                # S0 (memory): prefetch V max_m2 into buf1, read the resident K max_m2.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_v_tile(max_m2, 1)
                v_k = _load_k_head(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _dualwave_sync_barrier()

                # S1 (compute): finish the carried P, then MMA0 -> scores of max_m2.
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                v_s_1 = _qk(v_k, 1)
                _sched_barrier_pairs(traits, 6, 3, 5, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 5)
                _dualwave_sync_barrier()

                # S2 (memory): read V of the tile the carried P belongs to, mask max_m2.
                _s_nop(3)
                _sched_barrier(0)
                v_packs_s2 = _load_v_head(0)
                v_s_1 = ctx.causal_mask_prologue_if_needed(
                    v_s_1,
                    max_m2,
                    kv_end_tile=max_m1,
                )
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _dualwave_sync_barrier()

                # S3 (compute): full P*V + unconditional rescale.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                # v_p_0 here belongs to tile max_m3, the band's first tile.
                _live_lo_m3, _ = ctx.chunk_live(max_m3)
                v_o = _pv_lo_live(v_p_0, v_packs_s2, v_o, 0, _live_lo_m3)
                m_row, rescale_s3 = ctx.tile_row_max(m_row, v_s_1)
                v_s_1 = ctx.shift_scores(v_s_1, m_row)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 10, 5, 6)
                _sched_barrier_pairs(traits, 6, 3, 6, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                ctx.scale_o_by(v_o, rescale_s3)
                v_o = _anchor_v_o(traits, v_o)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _dualwave_sync_barrier()

                # S4 (memory): prefetch V max_m1 into buf0, read the resident K max_m1.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_v_tile(max_m1, 0)
                v_k = _load_k_head(0)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _dualwave_sync_barrier()

                # S5 (compute): fold rescale, finish P of max_m2, MMA0 -> scores of max_m1.
                l_row = ctx.scale_l_by(l_row, rescale_s3)
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
                v_s_0 = _qk(v_k, 0, tile_idx=max_m1)
                _sched_barrier_pairs(traits, 6, 3, 7, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 7)
                _dualwave_sync_barrier()

                # S6 (memory): read V max_m2 (buf1), mask the final tile.
                v_packs_s6 = _load_v_head(1)
                v_s_0 = ctx.causal_mask_prologue_if_needed(
                    v_s_0,
                    max_m1,
                    kv_end_tile=split_t_end,
                )
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                # S6 ISSUES NO DMA, so it must not drain any.
                # In the two-tile drain both K tiles are already resident, so S2
                # and S6 issue nothing: the only vm traffic S6 can see is S4's V prefetch for
                # max_m1, exactly NUM_DMA_V * dma_wave_reps deep. The inherited VM_DRAIN_V is
                # half that, so this wait used to retire half of S4's prefetch a whole compute
                # cluster (S7 -- the drain's largest, a P*V plus a complete softmax) before S8's
                # vmcnt(0) would have drained it anyway. S6's own ds_read is of buf1, covered by
                # S4's wait plus its barrier; S8 still drains its own with vmcnt(0). Bit-exact.
                _waitcnt_vm_n(ctx.NUM_DMA_V * (traits.SMEM_N_RPT // traits.NUM_WAVES))
                _dualwave_sync_barrier()

                # S7 (compute): P*V of max_m2, rescale, and the final tile's whole softmax.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv(v_p_1, v_packs_s6, v_o, 1)
                m_row, rescale_s7 = ctx.tile_row_max(m_row, v_s_0)
                v_s_0 = ctx.shift_scores(v_s_0, m_row)
                v_p_0 = ctx.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 9, 6, 8)
                _sched_barrier_pairs(traits, 7, 3, 8, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                l_row = ctx.scale_l_by(l_row, rescale_s7)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                _sched_barrier(0)
                ctx.scale_o_by(v_o, rescale_s7)
                v_o = _anchor_v_o(traits, v_o)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _s_barrier()
                _sched_barrier(0)

                # S8 (memory): read the final V packs (buf0, prefetched at S4).
                # The drain is BEFORE the read here, unlike every other memory cluster.
                # A cluster's `_waitcnt_vm_n(VM_DRAIN_KV / VM_DRAIN_V)` is what retires the
                # PREVIOUS cluster's tile -- the drains are one and a half tiles wide and
                # each memory cluster issues one tile -- so a read is covered by the wait
                # two clusters back. This drain removes the stage that would have issued
                # that DMA, leaving half of the S4 prefetch still in flight, so it has to
                # drain its own: vmcnt(0), then the barrier, because each wave DMAs the
                # lines the other waves read.
                _waitcnt_vm_n(0)
                _dualwave_sync_barrier()
                v_packs_s8 = _load_v_head(0)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _dualwave_sync_barrier()

                # S9 (compute): final P*V -> v_o holds the unnormalized output.
                # v_p_0 here belongs to tile max_m1, the band's last tile.
                _, _live_hi_m1 = ctx.chunk_live(max_m1)
                v_o = _pv_hi_live(v_p_0, v_packs_s8, v_o, 0, _live_hi_m1)
            else:
                # WAVE-UNIFORM ELISION ON THE DENSE CAUSAL BAND.
                # The mechanism (`chunk_live` + `qk_live` + `_pv_branch`) has been in this
                # kernel since the QK and P*V elisions but sat entirely inside `short_drain`,
                # i.e. SWA only -- `flash_attn_fwd.py` said so in as many words and no dense
                # shape ever reached it. The three-tile dense drain has the same structure:
                # with split_t_end = 2b+2 for q block b, tile max_m3 = 2b-1 starts at
                # 128b-64 and is wholly live for every wave, while the two band tiles are not.
                # A wave owns ROWS_PER_WAVE = 32 rows, so q_hi = 128b + wave_off + 31:
                #   max_m2 (kv0 = 128b):    LO live for all four row groups; HI dead for
                #                           row group 0            -> 1 x 32x32
                #   max_m1 (kv0 = 128b+64): LO dead for row groups 0 and 32, HI dead for
                #                           0, 32 and 64           -> 2 x 32x32 + 3 x 32x32
                # 6144 of the 8128 dead score elements per q block are reachable this way;
                # the remaining 1984 are the per-lane triangle and are not wave-uniform.
                # max_m2's LO predicate is omitted rather than computed: kv0 = 128b <= q_hi
                # holds for every row group, so it would be an always-taken branch.
                _live_hi_m2 = ctx.chunk_live(max_m2)[1]
                _live_lo_m1, _live_hi_m1 = ctx.chunk_live(max_m1)

                # Epilogue C0 prefetches V and reads K.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_v_tile(max_m3, 1)
                v_k = _load_k_head(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _dualwave_sync_barrier()

                # Epilogue C1 (compute): finish v_p_0 softmax, then MMA0 -> v_s_1 (like C1).
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                v_s_1 = _qk(v_k, 1)
                _sched_barrier_pairs(traits, 6, 3, 5, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 5)
                _dualwave_sync_barrier()

                # Epilogue C2 (memory): prefetch K max_m1, read V packs (buf0), causal mask v_s_1, sync.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_k_tile(max_m1, 1)
                v_packs_e3 = _load_v_head(0)
                if const_expr(traits.CAUSAL):
                    v_s_1 = ctx.causal_mask_prologue_if_needed(
                        v_s_1,
                        max_m3,
                        kv_end_tile=max_m2,
                    )
                else:
                    v_s_1 = ctx.seq_pad_mask_if_needed(v_s_1, max_m3)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _dualwave_sync_barrier()

                # Epilogue C3 (compute): full P*V + unconditional rescale
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv(v_p_0, v_packs_e3, v_o, 0)
                m_row, rescale_e3 = ctx.tile_row_max(m_row, v_s_1)
                v_s_1 = ctx.shift_scores(v_s_1, m_row)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 10, 5, 6)
                _sched_barrier_pairs(traits, 6, 3, 6, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                ctx.scale_o_by(v_o, rescale_e3)
                v_o = _anchor_v_o(traits, v_o)

                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _dualwave_sync_barrier()

                # Epilogue C4 (memory): prefetch V max_m2 (buf0), read K from buf0, sync.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_v_tile(max_m2, 0)
                v_k = _load_k_head(0)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _dualwave_sync_barrier()

                # Epilogue C5 folds rescale_e3 into l_row, finishes v_p_1 softmax, then computes MMA0.
                l_row = ctx.scale_l_by(l_row, rescale_e3)
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
                v_s_0 = _qk(v_k, 0, live=(None, _live_hi_m2))
                _sched_barrier_pairs(traits, 6, 3, 7, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 7)
                _dualwave_sync_barrier()

                # Epilogue C6 (memory): read V packs (buf1), causal mask v_s_0, sync.
                v_packs_e7 = _load_v_head(1)
                if const_expr(traits.CAUSAL):
                    v_s_0 = ctx.causal_mask_prologue_if_needed(
                        v_s_0,
                        max_m2,
                        kv_end_tile=max_m1,
                    )
                else:
                    v_s_0 = ctx.seq_pad_mask_if_needed(v_s_0, max_m2)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_V)
                _dualwave_sync_barrier()

                # Epilogue C7 (compute, mirror of C3): full P*V + unconditional rescale.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv(v_p_1, v_packs_e7, v_o, 1)
                m_row, rescale_e7 = ctx.tile_row_max(m_row, v_s_0)
                v_s_0 = ctx.shift_scores(v_s_0, m_row)
                v_p_0 = ctx.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 10, 5, 8)
                _sched_barrier_pairs(traits, 6, 3, 8, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                ctx.scale_o_by(v_o, rescale_e7)
                v_o = _anchor_v_o(traits, v_o)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _dualwave_sync_barrier()

                # Epilogue C8 (memory): prefetch V max_m1 (buf1), read K from buf1, sync.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_v_tile(max_m1, 1)
                v_k = _load_k_head(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.VM_DRAIN_V)
                _dualwave_sync_barrier()

                # Epilogue C9 folds rescale_e7 into l_row, finishes v_p_0, then computes last-tile MMA0.
                l_row = ctx.scale_l_by(l_row, rescale_e7)
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                v_s_1 = _qk(v_k, 1, live=(_live_lo_m1, _live_hi_m1))
                _sched_barrier_pairs(traits, 6, 3, 9, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 9)
                _dualwave_sync_barrier()

                # Epilogue C10 reads final V packs, masks v_s_1, drains DMAs, and syncs.
                v_packs_e11 = _load_v_head(0)
                if const_expr(traits.CAUSAL):
                    v_s_1 = ctx.causal_mask_prologue_if_needed(
                        v_s_1,
                        max_m1,
                        kv_end_tile=split_t_end,
                    )
                else:
                    v_s_1 = ctx.seq_pad_mask_if_needed(v_s_1, max_m1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(0)
                _dualwave_sync_barrier()

                # Epilogue C11: final rescale and complete the last tile's softmax in-place.
                # v_p_0 belongs to max_m2, whose HI half is dead for row group 0.
                v_o = _pv_hi_live(v_p_0, v_packs_e11, v_o, 0, _live_hi_m2)
                m_row, rescale_e11 = ctx.tile_row_max(m_row, v_s_1)
                v_s_1 = ctx.shift_scores(v_s_1, m_row)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 9, 6, 10)
                _sched_barrier_pairs(traits, 7, 3, 10, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                l_row = ctx.scale_l_by(l_row, rescale_e11)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
                _sched_barrier(0)
                ctx.scale_o_by(v_o, rescale_e11)
                v_o = _anchor_v_o(traits, v_o)
                _s_barrier()
                _sched_barrier(0)

                # Epilogue C12 (memory): read the final V packs for the closing P*V.
                v_packs_e13 = _load_v_head(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _dualwave_sync_barrier()

                # Epilogue C13 (compute): final P*V -> v_o holds the unnormalized output.
                # v_p_1 belongs to max_m1, the band's last tile -- both halves are
                # dead for row groups 0 and 32, so those waves issue no P*V for it at all.
                v_o = _pv_both_live(v_p_1, v_packs_e13, v_o, 1, _live_lo_m1, _live_hi_m1)

            # Normalize O; split-K stores normalized partials for later w_s * l_s reweighting.
            # finalize_o_scale folds the learned sink into the denominator when HAS_SINK
            # (else it is the plain 1/l path, byte-identical) and returns the (max, denom)
            # to record as LSE.
            l_row = ctx.finish_row_sum(l_row)
            o_scale, m_lse, l_lse = ctx.finalize_o_scale(m_row, l_row)
            ctx.scale_o(v_o, o_scale)

            _s_barrier()

            # The LSE store goes BEFORE the O store, not after. With HAS_SINK,
            # finalize_o_scale derives two fresh per-lane fp32 (mf, l_out) that only store_lse
            # consumes; leaving the O store between them and their use holds both live across
            # the kernel's register peak (permlane32_swap re-lane + packed dwordx4 stores).
            # That is the one configuration that pays 112 B of scratch -- sink AND emit_lse,
            # i.e. exactly the scored dense shapes; the sink-free build pays 20 B because its
            # m_lse is the fixed zero reference and its l_lse is already live for the
            # reciprocal. Consuming them first is bit-exact: different output tensors, no
            # aliasing, both already after the same barrier.
            if const_expr(traits.EMIT_LSE):
                ctx.store_lse(m_lse, l_lse, ctx.q_row)
            ctx.store_final_o(v_o, ctx.q_row)

        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            ctx.zero_o_block_if_needed()

        if active is None:
            _main_body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _main_body()

            _run_body_if_active()

    @flyc.jit
    def launch_flash_attn_dualwave_swp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        SINK: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _dualwave_swp_cache_tag
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        grid_z = bs_idx

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(traits.DAZ)
            else None
        )
        flash_attn_dualwave_swp_gfx950_kernel(
            Q,
            K,
            V,
            O,
            DebugCounts,
            CuSeqQ,
            CuSeqKv,
            BlockTable,
            SINK,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            block_table_stride,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(traits.NUM_HEADS_Q // traits.Q_HEADS_PER_WG, num_q_blocks, grid_z),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # K/V LDS reads are spread across compute clusters, so this module compiles with a
    # memory-clause pre-RA schedule plus the post-RA waitcnt cleanup.
    _dualwave_swp_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-sched-strategy": "max-memory-clause",
            "enable-post-misched": True,
            "lsr-drop-solution": True,
        },
    }

    _compiled: dict = {}
    _COMPILED_MAX = 64

    def _fill_defaults(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        stride_kv_n,
        stride_q_n,  # noqa: E741
        head_dim_runtime,
        debug_counts,
        seq_len_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        block_table,
        block_table_stride,
        sink,
    ):
        # cu_seqlens_*/block_table/sink are unused kernel-signature placeholders for dense
        # launches / has_sink=False; the kernel only reads them under const_expr(traits.VARLEN
        # / HAS_SINK). O fills those slots. Returns the ordered 16-tuple the JIT entry expects.
        return (
            Q,
            K,
            V,
            O,
            O if debug_counts is None else debug_counts,
            O if cu_seqlens_q is None else cu_seqlens_q,
            O if cu_seqlens_kv is None else cu_seqlens_kv,
            O if block_table is None else block_table,
            O if sink is None else sink,
            batch_size,
            seq_len,
            seq_len if seq_len_kv is None else seq_len_kv,
            traits.DEFAULT_STRIDE_Q_N if stride_q_n is None else stride_q_n,
            traits.DEFAULT_STRIDE_KV_N if stride_kv_n is None else stride_kv_n,
            traits.HEAD_DIM if head_dim_runtime is None else head_dim_runtime,
            0 if block_table_stride is None else block_table_stride,
        )

    def _launch(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,  # noqa: E741
        head_dim_runtime=None,
        debug_counts=None,
        *,
        seq_len_kv=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=None,
        sink=None,
        stream=None,
    ):
        args = _fill_defaults(
            Q,
            K,
            V,
            O,
            batch_size,
            seq_len,
            stride_kv_n,
            stride_q_n,
            head_dim_runtime,
            debug_counts,
            seq_len_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            block_table,
            block_table_stride,
            sink,
        )
        # SINK now sits at index 8; the scalar shape/mode args (JIT cache key) start at 9.
        # has_sink is baked into the module (separate build), so the SINK tensor stays out
        # of the key.
        key = args[9:] + (stream is None,)
        fn = _compiled.get(key)
        if fn is None:
            if len(_compiled) >= _COMPILED_MAX:
                _compiled.clear()
            with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
                fn = flyc.compile(launch_flash_attn_dualwave_swp, *args, stream)
            _compiled[key] = fn
        return fn(*args, stream)

    def _compile(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,  # noqa: E741
        head_dim_runtime=None,
        debug_counts=None,
        *,
        seq_len_kv=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=None,
        sink=None,
        stream=None,
    ):
        args = _fill_defaults(
            Q,
            K,
            V,
            O,
            batch_size,
            seq_len,
            stride_kv_n,
            stride_q_n,
            head_dim_runtime,
            debug_counts,
            seq_len_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            block_table,
            block_table_stride,
            sink,
        )
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return flyc.compile(launch_flash_attn_dualwave_swp, *args, fx.Stream(stream))

    _launch.compile = _compile

    return _launch
