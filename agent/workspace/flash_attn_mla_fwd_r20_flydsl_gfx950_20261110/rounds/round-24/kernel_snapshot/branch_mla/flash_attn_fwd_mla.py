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

MegaAttn prototype copy: adds D_QK != D_V (192/128, DeepSeek-V3 MLA). Q/K carry
``head_dim`` columns, V/O carry ``head_dim_v``; strides must be those of contiguous
[S, B, H, D] / [T, H, D] tensors.

Dispatched when gpu_arch >= gfx950, head_dim in (64, 128) and seq_len >= 384.
seq_len need not be a multiple of 256/64: partial q-blocks are covered by num_records
bounds and a kv pad mask. On the dense causal path, odd KV-tile spans are even-rounded;
the left-windowed path keeps an odd span and pairs it with a two-tile epilogue drain.
"""

import heapq

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from primus_turbo.flydsl.utils import buffer_ops

from .attn_helper_mla import (
    NUM_XCD,
    DualwaveKernelContext,
    _anchor_v_o,
    _dualwave_sync_barrier,
    _interleave,
    _make_dualwave_swp_traits,
    _numel,
    _qk16_slot,
    _s_barrier,
    _s_nop,
    _s_setprio,
    _s_waitcnt,
    _sched_barrier,
    _sched_barrier_pairs,
    _waitcnt_vm_n,
    dtype_to_elem_type,
    host_job_order,
)

# Persistent schedule entry: head | q block << 9 | batch << 24.
JOB_HEAD_MASK = (1 << 9) - 1
JOB_QBLOCK_SHIFT = 9
JOB_QBLOCK_MASK = (1 << 15) - 1
JOB_BATCH_SHIFT = 24
JOB_BATCH_LIMIT = 1 << 7
# Per-job cost outside the KV tiles (prologue, drain, store), in KV tiles, for the host's
# list-schedule replay.
JOB_FIXED_TILES = 8


def build_flash_attn_dualwave_swp_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
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
    head_dim_v=None,
    k_head=None,
    qk_mfma16=None,
    pv_mfma16=None,
    rows_per_wave=32,
    fold_barriers=None,
    persistent=0,
):
    """Build a DUALWAVE_SWP flash_attn launcher for (D_QK, D_V) in {64/64, 128/128, 192/128}.

    Supports dense (SBHD) and varlen packed QKV (THD) layouts. has_sink folds a learned
    per-q-head attention sink (SINK[Hq] fp32) into the online-softmax denominator.
    ``k_head`` is the number of QK k-steps whose K packs are read before the first MFMA
    group (default K_STEPS_QK - 2). ``qk_mfma16`` runs QK on 16x16x32 MFMAs (default: on
    for D_QK 192 without a causal window); ``pv_mfma16`` runs P*V on them too (default: with
    ``qk_mfma16``). ``rows_per_wave`` 64 halves the waves per q block (one per SIMD at
    block_m 256). ``fold_barriers`` drops the memory-cluster rendezvous (default: above 4 waves).
    ``persistent`` > 0 launches at most that many CTAs (one per CU), each running the jobs the
    host schedule (``launcher.schedule``) gives it; dense layouts without a window only.
    """
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(
            f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}"
        )
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    if (head_dim, head_dim_v) not in ((64, 64), (128, 128), (192, 128)):
        raise RuntimeError(
            f"flash_attn_dualwave_swp supports D_QK/D_V 64/64, 128/128, 192/128, got {head_dim}/{head_dim_v}"
        )
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
        head_dim_v=head_dim_v,
        qk_mfma16=qk_mfma16,
        pv_mfma16=pv_mfma16,
        rows_per_wave=rows_per_wave,
    )
    traits.FOLD_BARRIERS = traits.NUM_WAVES > 4 if fold_barriers is None else bool(fold_barriers)
    traits.K_HEAD = max(1, traits.K_STEPS_QK - 2) if k_head is None else int(k_head)
    assert 1 <= traits.K_HEAD <= traits.K_STEPS_QK
    traits.PERSISTENT = int(persistent)
    assert not traits.PERSISTENT or not (varlen or cross_seqlen or window_left >= 0)
    # Persistent drains DMA part of the next job's Q through the O stage tiles (stage_q_dma);
    # its geometry is the D_QK 192 paired-step split.
    traits.Q_LDS_STAGE = bool(
        traits.PERSISTENT and traits.O_LDS_STAGE and traits.QK_MFMA16 and traits.HEAD_DIM == 192
    )
    # Persistent dense causal with four q tiles a wave: q tile qt of wave w takes block rows
    # 64 qt + 16 w, not 64 w + 16 qt. Every wave then holds the same share of the diagonal, and
    # the drain's dead q tiles are the same static ones on every wave (`_qk_diag`).
    traits.ROW_INTERLEAVE = bool(
        traits.PERSISTENT
        and traits.CAUSAL
        and traits.Q_TILES == 4
        and traits.QK_MFMA16
        and traits.PV_MFMA16
        and traits.Q_HEADS_PER_WG == 1
        and traits.NUM_WAVES * 16 == traits.BLOCK_M // traits.Q_TILES == traits.BLOCK_N
    )
    if traits.ROW_INTERLEAVE:
        traits.QT_ROW_STRIDE = traits.BLOCK_M // traits.Q_TILES
        traits.WAVE_ROW_STRIDE = 16
    _dualwave_swp_cache_tag = traits.cache_tag

    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
        if traits.O_LDS_STAGE:
            o: fx.Array[_lds_elem_dtype, traits.LDS_O_STAGE_SIZE, 16]

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
        ScoreBound: fx.Tensor,
        Sched: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
        sched_stride: fx.Int32,
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
            ScoreBound=ScoreBound,
        )
        ctx._setup(SharedStorage, bind_block=not traits.PERSISTENT)

        active = None if traits.PERSISTENT else ctx.active
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
        K_HEAD = traits.K_HEAD
        # P*V k-substeps of V read up front; substep k + V_HEAD is issued after step k.
        V_HEAD = 1
        split_k_reads = split_kv_reads and K_HEAD < traits.K_STEPS_QK
        # The write-after-read edge is already covered by the compute-cluster barrier preceding each
        # overwrite DMA, so the memory-cluster rendezvous protects nothing (and costs at 8 waves,
        # whether two merged q-heads or one block_m 256 head).
        fold_mem_barriers = sink_drains and traits.FOLD_BARRIERS
        fold_pv_barriers = fold_mem_barriers and late_dma
        vm_drain_sync = 0 if fold_pv_barriers else ctx.VM_DRAIN_KV
        # Two-tile drain admits odd KV spans (windowed); three-tile drain needs even N.
        short_drain = traits.CAUSAL and traits.WINDOW_LEFT >= 0
        # The three-tile drain masks only the last three KV tiles, so a diagonal band wider
        # than that (BLOCK_M > 3 * BLOCK_N) reaches into the loop and needs the loop masks.
        loop_masks = traits.CAUSAL and (
            traits.CROSS_SEQLEN or traits.WINDOW_LEFT >= 0 or traits.BLOCK_M > 3 * traits.BLOCK_N
        )

        def _mem_cluster_sync():
            if const_expr(fold_mem_barriers):
                _sched_barrier(0)
            else:
                _dualwave_sync_barrier()

        def _qk_cluster_sync(sched_region=False):
            if const_expr(sink_drains):
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(vm_drain_sync)
            # Not dead: `sched_region` is always False, but collapsing this to a bare
            # `_dualwave_sync_barrier()` costs ~5% at head_dim 64. Keep the `const_expr`.
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

        if const_expr(traits.PV_MFMA16):
            l_row_init = ctx.c_zero_l16
        elif const_expr(traits.DUALWAVE_SWP_MFMA_ROWSUM):
            l_row_init = ctx.c_zero_v4f32
        else:
            l_row_init = ctx.c_zero_f
        v_o_zero = ctx.c_zero_o if traits.PV_MFMA16 else ctx.c_zero_v16f32

        def _prefetch_k(job):
            """First two K tiles of a packed persistent job (split_t0 is 0 there) into buf0 / buf1."""
            h_idx, batch_idx = fx.Index(job & JOB_HEAD_MASK), fx.Index(job >> JOB_BATCH_SHIFT)
            for t in range_constexpr(2):
                ctx.prefetch_k_tile(h_idx, batch_idx, fx.Index(t), t)

        def _job_coords(job):
            """(head, q block, batch) of a packed persistent job."""
            return (
                fx.Index(job & JOB_HEAD_MASK),
                fx.Index((job >> JOB_QBLOCK_SHIFT) & JOB_QBLOCK_MASK),
                fx.Index(job >> JOB_BATCH_SHIFT),
            )

        def _prefetch_q(job, staged=False):
            """Raw Q packs of a packed persistent job, for its `_main_body(q_raw=...)`; `staged`
            leaves the O-stage ones None for `read_q_stage`."""
            if const_expr(staged):
                return ctx.prefetch_q_staged(*_job_coords(job))
            return ctx.prefetch_q_raw(*_job_coords(job))

        def _main_body(next_job=None, q_raw=None):
            """With `next_job` (persistent), this job's first two K tiles and its Q (`q_raw`) were
            prefetched by the previous job's drain, and this drain prefetches the next job's, which
            it returns."""
            split_t_end = ctx.split_t_end
            q_next = None
            if const_expr(next_job is None):
                ctx.load_k_split(0, 0)
                _s_waitcnt(0)
            else:
                # The previous job issued its O stores after this job's K and Q prefetch, and vmcnt
                # retires in order: wait for the prefetch only, so the stores drain under the prologue.
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.o_stores_per_lane)
            _sched_barrier(0)
            _s_barrier()

            if const_expr(q_raw is None):
                q_all_bf16 = ctx.load_all()
            else:
                ctx.bind_q_rows()
                q_all_bf16 = ctx.assemble_q(q_raw)
            q_all_scaled_bf16 = ctx.scale_all(q_all_bf16)

            def _load_k_head(buf_id):
                if const_expr(split_k_reads):
                    return ctx.lds_load_k(buf_id, ks_range=(0, K_HEAD))
                return ctx.lds_load_k(buf_id)

            def _qk(v_k, buf_id, tile_idx=None, live=None, qts=None):
                """QK for one KV tile. `tile_idx`/`live` skip a wave-uniform dead 32-col half,
                `qts` (16x16x32) computes those q tiles only."""
                if const_expr(tile_idx is None and live is None):
                    if const_expr(not split_k_reads):
                        return ctx.qk(v_k, q_all_scaled_bf16, qts=qts)
                    ks_tail = (K_HEAD, traits.K_STEPS_QK)
                    v_s = ctx.qk(v_k, q_all_scaled_bf16, ks_range=(0, K_HEAD), qts=qts)
                    v_k = ctx.lds_load_k(buf_id, ks_range=ks_tail, k_regs=v_k)
                    return ctx.qk(v_k, q_all_scaled_bf16, v_s=v_s, ks_range=ks_tail, qts=qts)
                assert qts is None
                # Pass live=(None, pred) to skip the always-true half (avoids a dead scf.if).
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

            def _pv_step(step, v_p, v_v, v_o, buf_id, qts=None):
                """One P*V k-substep, trailed by the read of the substep V_HEAD ahead so
                that read is covered by this MFMA and only V_HEAD + 1 substeps are live."""
                v_o = ctx.pv_step_k(step, v_p, v_v, v_o, qts=qts)
                nxt = step + V_HEAD
                if const_expr(split_kv_reads and nxt < 4):
                    ctx.lds_load_v(buf_id, substeps=(nxt, nxt + 1), packs=v_v)
                return v_o

            def _pv(v_p, v_v, v_o, buf_id, qts=None):
                if const_expr(not split_kv_reads and qts is None):
                    return ctx.pv(v_p, v_v, v_o)
                if const_expr(not split_kv_reads):
                    for step in range_constexpr(4):
                        v_o = ctx.pv_step_k(step, v_p, v_v, v_o, qts=qts)
                    return v_o
                for step in range_constexpr(4):
                    v_o = _pv_step(step, v_p, v_v, v_o, buf_id, qts)
                return v_o

            # Skip P*V + V LDS reads on a wave-uniform dead 32-col half (P is 0 after the mask).

            def _pv_branch(v_o, live, body):
                """Scalar-branch `body` over D_CHUNKS accs; flyc.jit cannot carry a Python list."""
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
                """P*V with HI 32-col (steps 2/3) under `live_hi`."""
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
                """P*V with LO 32-col (steps 0/1) under `live_lo`. Substep 2 is live on both paths."""
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
                """P*V with both 32-col halves predicated; skipped accs are overwritten by the mask."""

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

            if const_expr(next_job is None):
                ctx.load_k_split(1, 1)
            ctx.load_v_split(0, 0)
            v_k = _load_k_head(0)
            _sched_barrier(0)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            # This drain is for K1. A prefetched K1 is already resident, and V0 is first read after
            # the next cluster's drain, so it stays in flight under QK0.
            if const_expr(next_job is None):
                _waitcnt_vm_n(ctx.VM_DRAIN_V)

            _sched_barrier(0)
            _dualwave_sync_barrier()

            if const_expr(short_drain):
                v_s_0 = _qk(v_k, 0, tile_idx=ctx.split_tile(0))
            else:
                v_s_0 = _qk(v_k, 0)
            _sched_barrier(0)
            v_s_0 = ctx.qk_scores(v_s_0)

            if const_expr(traits.CAUSAL):
                # split_tile(0) = split_t0 (0 dense, swa_lo for SWA) -- needed so the
                # window/causal mask on the prologue tile uses the correct tile index.
                v_s_0 = ctx.causal_mask_split_prologue_if_needed(v_s_0)
            else:
                # Non-causal tiny seq_len needs tile-0 padding masked before the full-tile no-op gate.
                v_s_0 = ctx.seq_pad_mask_if_needed(v_s_0, ctx.split_tile(0))
            if const_expr(ctx.use_score_bound):
                score_bound = ctx.score_bound()
            m_row_pro = ctx.reduce_max(v_s_0)
            if const_expr(traits.CAUSAL):
                # Floor fully-masked rows (-inf) to finite so exp2 yields 0, not NaN.
                m_row_pro = ctx.floor_masked_max(m_row_pro)
            if const_expr(ctx.use_score_bound):
                # -bound is itself a valid row maximum, so flooring there keeps the
                m_row_pro = ctx.floor_at_score_bound(m_row_pro, score_bound)
            m_row_pro = ctx.lift_ref(m_row_pro)
            if const_expr(ctx.use_score_bound):
                ctx.arm_score_bound(score_bound, m_row_pro)
            v_s_0 = ctx.shift_scores(v_s_0, m_row_pro)
            ctx.set_ref(m_row_pro)
            v_p_0 = ctx.exp2(v_s_0, 0, 16)
            _dualwave_sync_barrier()

            # Inner-loop tile indices are split_tile-relative: split_t0 = 0 dense/causal,
            # swa_lo for SWA, chunk start for split-K.
            loop_lb = ctx.split_tile(3)

            ctx.load_k_split(2, 0)

            # The running maximum is what every rescale is measured against, so it rides the
            # back edge; restarting it each iteration silently loses row mass.
            init_args = [m_row_pro, l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(v_o_zero)
            init_args.append(v_p_0[0])
            init_args.append(v_p_0[1])
            init_args.append(ctx.score_acc_seed())

            split_rebase = ctx.use_score_bound
            spec_exp = traits.SOLO_WAVE and not split_rebase
            if const_expr(split_rebase):
                init_args.append(ctx.no_rescale())

            # The compute-cluster rendezvous sits this many k-slots before the QK's end: the K
            # reads are all issued by then, and the V head read after it lands under the rest.
            QK_SYNC_TAIL = 2

            def _qk_finish_p(v_k, buf_id, v_p, l_row, group, v_buf=None):
                """QK of the next tile next to the carried P's second exp2 half, cast and row sum.

                SOLO_WAVE takes the cluster sync and the next P*V's V head read (buffer v_buf)
                into the QK, and returns that read too."""
                if const_expr(not traits.SOLO_WAVE):
                    # MMA0 issues after cast_p so its 32 fresh score regs never coexist with the
                    # 32 f32 of the carried P; sched_group_barrier still interleaves the two.
                    v_p = ctx.exp2(v_p, 16, 16)
                    v_p, l_row = ctx.cast_p_and_sum(l_row, v_p)
                    _sched_barrier(0)
                    v_s = _qk(v_k, buf_id)
                    _sched_barrier_pairs(traits, 6, 3, group, traits.SCHED_EXP_MASK)
                    _sched_barrier_pairs(traits, 10, 5, group)
                    return v_s, v_p, l_row
                # Alone on its SIMD the wave must hide both its LDS latency and this VALU under
                # the QK itself. Left to the scheduler, the K tail is read just in time and the
                # VALU bunches; fencing each k-slot keeps every read K_HEAD slots ahead and one
                # piece of the P work in each slot.
                pieces, finish_p = ctx.finish_p_pieces(l_row, v_p)
                v_s = None
                v_v = None
                for ks in range_constexpr(traits.K_STEPS_QK):
                    if const_expr(ks == traits.K_STEPS_QK - QK_SYNC_TAIL):
                        _qk_cluster_sync()
                        _sched_barrier(0)
                        v_v = _load_v_head(v_buf)
                    nxt = ks + K_HEAD
                    if const_expr(nxt < traits.K_STEPS_QK):
                        v_k = ctx.lds_load_k(buf_id, ks_range=(nxt, nxt + 1), k_regs=v_k)
                    v_s = ctx.qk(v_k, q_all_scaled_bf16, v_s=v_s, ks_range=(ks, ks + 1))
                    if const_expr(ks < len(pieces) - 1 and traits.QK_MFMA16):
                        pieces[ks]()
                        _qk16_slot(traits, 2 if nxt < traits.K_STEPS_QK else 0, 4, group)
                    elif const_expr(ks < len(pieces) - 1):
                        pieces[ks]()
                        _interleave(traits, 4, 1, group, traits.SCHED_EXP_MASK)
                    elif const_expr(ks == len(pieces) - 1):
                        pieces[ks]()
                        _interleave(traits, 16, 2, group)
                    _sched_barrier(0)
                v_p, l_row = finish_p()
                return v_s, v_p, l_row, v_v

            # One slot follows each asm P*V MFMA (16 cycles of pipe, 4 to issue): a VALU piece of
            # ~12 cycles, or one of the cluster's DMAs, spaced so their issue stalls drain.
            PV_DMA_EVERY = 6

            # P*V slot that reads the next QK's K head, early enough to land by that QK.
            PV_K_HEAD_SLOT = 56

            def _pv_spec_site(v_p, v_v, v_o, buf_id, v_s, m_row, l_row, dma, k_head_buf=None):
                """The carried P's P*V with the new tile's max tree, first exp2 half and the
                cluster's DMAs slotted between its asm MFMAs by hand, then the rebase branch.

                Each substep reads the next one's V one d chunk per slot (slots 3, 5, 7, 9 of
                the substep), early enough to land before that chunk's MFMAs. A slot holds
                about one 12-cycle MFMA shadow of work, so V-read slots take a max-tree piece
                (two max3) and never an exp2; the exp2 pieces go first everywhere else.

                ``k_head_buf``: also read the next QK's K head from that (already visible, not
                being refilled) buffer and return it."""
                pieces, spec = ctx.spec_softmax(v_s, max_chunk=4, exps_per_piece=1)
                n_exp = _numel(v_s[0])
                folds, exps = pieces[: len(pieces) - n_exp], pieces[len(pieces) - n_exp :]
                v_slots = {}
                if const_expr(split_kv_reads):
                    for pvs in range_constexpr(4 - V_HEAD):
                        for dc in range_constexpr(traits.D_CHUNKS):
                            v_slots[16 * pvs + 3 + 2 * dc] = (pvs + V_HEAD, dc)
                slots = []
                for s in range_constexpr(16 * 4):
                    work = []
                    if const_expr(s in v_slots):
                        nxt, dc = v_slots[s]
                        work.append(
                            lambda nxt=nxt, dc=dc: ctx.lds_load_v(
                                buf_id, substeps=(nxt, nxt + 1), packs=v_v, dcs=(dc, dc + 1)
                            )
                        )
                        if const_expr(len(folds) > 0):
                            work.append(folds.pop(0))
                    elif const_expr(s % PV_DMA_EVERY == 0 and len(dma) > 0):
                        work.append(dma.pop(0))
                    elif const_expr(len(exps) > 0):
                        work.append(exps.pop(0))
                    elif const_expr(len(folds) > 0):
                        work.append(folds.pop(0))
                    slots.append(work)
                rest = dma + folds + exps
                got = {}
                if const_expr(k_head_buf is not None):
                    slots[PV_K_HEAD_SLOT].append(lambda: got.update(v_k=_load_k_head(k_head_buf)))
                for pvs in range_constexpr(4):

                    def _after(i, pvs=pvs):
                        # Fenced on both sides: otherwise the scheduler hoists a slot's VALU
                        # above its own MFMA, into the previous (V-read) slot's shadow.
                        work = slots[16 * pvs + i]
                        if const_expr(len(work) > 0):
                            _sched_barrier(0)
                        for w in work:
                            w()
                        if const_expr(len(work) > 0):
                            _sched_barrier(0)

                    v_o = ctx.pv_step_k(pvs, v_p, v_v, v_o, after_mfma=_after)
                for w in rest:
                    w()
                m_row, l_row, v_p_new, v_o = ctx.rebase_if_needed(m_row, l_row, v_s, v_o, spec=spec())
                if const_expr(k_head_buf is not None):
                    return v_o, m_row, l_row, v_p_new, got["v_k"]
                return v_o, m_row, l_row, v_p_new

            for j, loop_args in range(
                loop_lb,
                split_t_end - fx.Index(1),
                fx.Index(2),
                init=init_args,
            ):
                m_row = loop_args[0]
                ctx.set_ref(m_row)
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_p_0 = (loop_args[2 + traits.D_CHUNKS], loop_args[3 + traits.D_CHUNKS])
                ctx.set_seed_vec(loop_args[4 + traits.D_CHUNKS])
                if const_expr(split_rebase):
                    pending = (ctx.bound_over_ref, loop_args[5 + traits.D_CHUNKS])
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
                if const_expr(traits.SOLO_WAVE):
                    v_s_1, v_p_0, l_row, v_v = _qk_finish_p(v_k, 1, v_p_0, l_row, 1, v_buf=0)
                else:
                    v_s_1, v_p_0, l_row = _qk_finish_p(v_k, 1, v_p_0, l_row, 1)
                    _qk_cluster_sync()

                # Cluster 2 prefetches next K, reads this tile's V for P*V, then waits and syncs.
                _sched_barrier(0)
                if const_expr(not late_dma):
                    ctx.load_k_tile(j_idx, 1)
                if const_expr(not traits.SOLO_WAVE):
                    v_v = _load_v_head(0)
                v_s_1 = ctx.qk_scores(v_s_1)
                if const_expr(not sink_drains):
                    _s_waitcnt(traits.LGKMCNT_0_ONLY)
                    _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _mem_cluster_sync()

                # Cluster 3 computes P*V, row max, rescale, sub row, and first-half exp2.
                dma = [] if spec_exp else None
                if const_expr(late_dma):
                    ctx.load_k_tile(j_idx, 1, pieces=dma)
                if const_expr(fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 2, 1, pieces=dma)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                if const_expr(not split_rebase and not spec_exp):
                    v_o = _pv_step(0, v_p_0, v_v, v_o, 0)
                # Cross-seqlen can put a diagonal tile in v_s_1; so can SWA's lower window edge.
                if const_expr(loop_masks):
                    v_s_1 = ctx.causal_mask_prologue_if_needed(
                        v_s_1,
                        j_idx - 2,
                        kv_end_tile=j_idx - 1,
                    )
                else:
                    v_s_1 = ctx.scores_for_softmax(v_s_1)
                v_k = None
                if const_expr(spec_exp):
                    # K buffer 0 is visible since C1's sync and this cluster's DMAs fill buffer 1.
                    v_o, m_row, l_row, v_p_1, v_k = _pv_spec_site(
                        v_p_0, v_v, v_o, 0, v_s_1, m_row, l_row, dma, k_head_buf=0
                    )
                else:
                    if const_expr(split_rebase):
                        m_row, l_row, v_s_1, v_o, pending = ctx.rebase_if_needed(
                            m_row, l_row, v_s_1, v_o, pending
                        )
                        v_o = _pv_step(0, v_p_0, v_v, v_o, 0)
                    for pvs in range_constexpr(1, 4):
                        v_o = _pv_step(pvs, v_p_0, v_v, v_o, 0)
                    if const_expr(not split_rebase):
                        m_row, l_row, v_s_1, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_1, v_o)
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
                if const_expr(v_k is None):
                    v_k = _load_k_head(0)
                if const_expr(not sink_drains):
                    _s_waitcnt(traits.LGKMCNT_0_ONLY)
                    _waitcnt_vm_n(ctx.VM_DRAIN_KV)
                _mem_cluster_sync()

                # Cluster 5 mirrors C1: finish v_p_1 softmax, update l_row, cast P, then MMA0.
                if const_expr(late_dma and not fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 1, 0)
                if const_expr(traits.SOLO_WAVE):
                    v_s_0, v_p_1, l_row, v_v = _qk_finish_p(v_k, 0, v_p_1, l_row, 3, v_buf=1)
                else:
                    v_s_0, v_p_1, l_row = _qk_finish_p(v_k, 0, v_p_1, l_row, 3)
                    _qk_cluster_sync()

                # Cluster 6 prefetches next K, reads V packs, optionally masks v_s_0, waits, and syncs.
                _sched_barrier(0)
                if const_expr(not late_dma):
                    ctx.load_k_tile(j_idx + 1, 0)
                if const_expr(not traits.SOLO_WAVE):
                    v_v = _load_v_head(1)
                v_s_0 = ctx.qk_scores(v_s_0)
                # Dense causal: Cluster 6 is before the diagonal band; mask only on SWA / cross-seqlen.
                if const_expr(loop_masks):
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
                dma = [] if spec_exp else None
                if const_expr(late_dma):
                    ctx.load_k_tile(j_idx + 1, 0, pieces=dma)
                if const_expr(fold_pv_barriers):
                    ctx.load_v_tile(j_idx - 1, 0, pieces=dma)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                if const_expr(spec_exp):
                    v_o, m_row, l_row, v_p_0 = _pv_spec_site(v_p_1, v_v, v_o, 1, v_s_0, m_row, l_row, dma)
                else:
                    if const_expr(split_rebase):
                        m_row, l_row, v_s_0, v_o, pending = ctx.rebase_if_needed(
                            m_row, l_row, v_s_0, v_o, pending
                        )
                    v_o = _pv_step(0, v_p_1, v_v, v_o, 1)
                    for pvs in range_constexpr(1, 4):
                        v_o = _pv_step(pvs, v_p_1, v_v, v_o, 1)
                    if const_expr(not split_rebase):
                        m_row, l_row, v_s_0, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_0, v_o)
                    v_p_0 = ctx.exp2(v_s_0, 0, 16)
                    _sched_barrier_pairs(traits, 6, 5, 4)
                    _sched_barrier_pairs(traits, 6, 3, 4, traits.SCHED_EXP_MASK)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _pv_cluster_sync()

                yield_args = [m_row, l_row] + v_o + [v_p_0[0], v_p_0[1], ctx.score_acc_seed()]
                if const_expr(split_rebase):
                    yield_args.append(pending[1])
                loop_results = yield yield_args

            # Epilogue drains the final in-flight tiles, continuing from the loop's maximum.
            m_row = loop_results[0]
            # the epilogue seeds against the loop's result, not the last body value
            ctx.set_ref(m_row)
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = (loop_results[2 + traits.D_CHUNKS], loop_results[3 + traits.D_CHUNKS])
            ctx.set_seed_vec(loop_results[4 + traits.D_CHUNKS])
            if const_expr(split_rebase):
                v_o = ctx.rebase_acc(v_o, (ctx.bound_over_ref, loop_results[5 + traits.D_CHUNKS]))
            max_m3 = split_t_end - 3
            max_m2 = split_t_end - 2
            max_m1 = split_t_end - 1

            if const_expr(fold_pv_barriers):
                _dualwave_sync_barrier()

            if const_expr(short_drain):
                # Two-tile drain: last two K tiles already resident, no K DMA. Odd trip counts only.

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
                v_s_1 = _qk(v_k, 1)
                _sched_barrier_pairs(traits, 6, 3, 5, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 5)
                _dualwave_sync_barrier()

                # S2 (memory): read V of the tile the carried P belongs to, mask max_m2.
                _s_nop(3)
                _sched_barrier(0)
                v_packs_s2 = _load_v_head(0)
                v_s_1 = ctx.qk_scores(v_s_1)
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
                m_row, l_row, v_s_1, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_1, v_o)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 10, 5, 6)
                _sched_barrier_pairs(traits, 6, 3, 6, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
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
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                v_s_0 = _qk(v_k, 0, tile_idx=max_m1)
                _sched_barrier_pairs(traits, 6, 3, 7, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 7)
                _dualwave_sync_barrier()

                # S6 (memory): read V max_m2 (buf1), mask the final tile.
                v_packs_s6 = _load_v_head(1)
                v_s_0 = ctx.qk_scores(v_s_0)
                v_s_0 = ctx.causal_mask_prologue_if_needed(
                    v_s_0,
                    max_m1,
                    kv_end_tile=split_t_end,
                )
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                # S6 issues no DMA; wait the full S4 V prefetch, not VM_DRAIN_V (half a tile).
                _waitcnt_vm_n(ctx.NUM_DMA_V * (traits.SMEM_N_RPT // traits.NUM_WAVES))
                _dualwave_sync_barrier()

                # S7 (compute): P*V of max_m2, rescale, and the final tile's whole softmax.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv(v_p_1, v_packs_s6, v_o, 1)
                m_row, l_row, v_s_0, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_0, v_o)
                v_p_0 = ctx.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 9, 6, 8)
                _sched_barrier_pairs(traits, 7, 3, 8, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                _sched_barrier(0)
                v_o = _anchor_v_o(traits, v_o)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                _s_barrier()
                _sched_barrier(0)

                # S8 (memory): read the final V packs (buf0, prefetched at S4).
                # Drain this stage itself: the missing middle tile left S4's prefetch in flight.
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
                # Dense causal drain: skip wave-uniform dead halves on the diagonal KV tiles.
                def _diag_live(k):
                    """chunk_live of the k-th last KV tile; None for a half no wave can find dead
                    (its start never passes wave 0's last row), which saves an always-true branch."""
                    lo_start = traits.BLOCK_M - k * traits.BLOCK_N
                    if const_expr(
                        traits.ROW_INTERLEAVE or lo_start + traits.BLOCK_N // 2 < traits.ROWS_PER_WAVE
                    ):
                        return (None, None)
                    live_lo, live_hi = ctx.chunk_live(split_t_end - k)
                    return (live_lo if lo_start >= traits.ROWS_PER_WAVE else None, live_hi)

                # ROW_INTERLEAVE: the k-th last KV tile is the block's diagonal tile 4 - k (every
                # band is full, `block_row0`), so the q tiles below it are dead on every wave.
                def _qk_diag(v_k, buf_id, live, k):
                    if const_expr(traits.ROW_INTERLEAVE):
                        return _qk(v_k, buf_id, qts=range(traits.Q_TILES - k, traits.Q_TILES))
                    if const_expr(live[1] is None):
                        return _qk(v_k, buf_id)
                    return _qk(v_k, buf_id, live=live)

                def _pv_diag(v_p, v_v, v_o, buf_id, live, k):
                    if const_expr(traits.ROW_INTERLEAVE):
                        return _pv(v_p, v_v, v_o, buf_id, qts=range(traits.Q_TILES - k, traits.Q_TILES))
                    if const_expr(live[1] is None):
                        return _pv(v_p, v_v, v_o, buf_id)
                    if const_expr(live[0] is None):
                        return _pv_hi_live(v_p, v_v, v_o, buf_id, live[1])
                    return _pv_both_live(v_p, v_v, v_o, buf_id, live[0], live[1])

                _live_m4 = _diag_live(4)
                _live_m3 = _diag_live(3)
                _live_m2 = _diag_live(2)
                _live_m1 = _diag_live(1)

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
                v_s_1 = _qk_diag(v_k, 1, _live_m3, 3)
                _sched_barrier_pairs(traits, 6, 3, 5, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 5)
                _dualwave_sync_barrier()

                # Epilogue C2 (memory): prefetch K max_m1, read V packs (buf0), causal mask v_s_1, sync.
                _s_nop(3)
                _sched_barrier(0)
                ctx.load_k_tile(max_m1, 1)
                v_packs_e3 = _load_v_head(0)
                v_s_1 = ctx.qk_scores(v_s_1)
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

                # Epilogue C3 (compute): P*V of max_m4 + unconditional rescale
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv_diag(v_p_0, v_packs_e3, v_o, 0, _live_m4, 4)
                m_row, l_row, v_s_1, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_1, v_o)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 10, 5, 6)
                _sched_barrier_pairs(traits, 6, 3, 6, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
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

                # Epilogue C5 finishes the v_p_1 softmax, then computes MMA0.
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                v_s_0 = _qk_diag(v_k, 0, _live_m2, 2)
                _sched_barrier_pairs(traits, 6, 3, 7, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 7)
                _dualwave_sync_barrier()

                # Epilogue C6 (memory): read V packs (buf1), causal mask v_s_0, sync.
                v_packs_e7 = _load_v_head(1)
                v_s_0 = ctx.qk_scores(v_s_0)
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

                # Epilogue C7 (compute, mirror of C3): P*V of max_m3 + unconditional rescale.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = _pv_diag(v_p_1, v_packs_e7, v_o, 1, _live_m3, 3)
                m_row, l_row, v_s_0, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_0, v_o)
                v_p_0 = ctx.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 10, 5, 8)
                _sched_barrier_pairs(traits, 6, 3, 8, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
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

                # Epilogue C9 finishes v_p_0, then computes the last-tile MMA0.
                if const_expr(next_job is not None and traits.Q_LDS_STAGE):
                    # Only the last wave's QK half is live in this cluster; the other waves fill
                    # the next job's Q stage meanwhile. C10's vmcnt(0) and sync publish it.
                    ctx.stage_q_dma(*_job_coords(next_job))
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                v_s_1 = _qk_diag(v_k, 1, _live_m1, 1)
                _sched_barrier_pairs(traits, 6, 3, 9, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 9)
                _dualwave_sync_barrier()

                # Epilogue C10 reads final V packs, masks v_s_1, drains DMAs, and syncs.
                v_packs_e11 = _load_v_head(0)
                v_s_1 = ctx.qk_scores(v_s_1)
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
                if const_expr(next_job is not None):
                    # K buf0 was last read in C5, buf1 in C9; nothing waits on these until the
                    # next job's prologue, so they land under C11-C13 and the O store. C9 was also
                    # the last QK, so this job's Q is dead and the next job's can load.
                    _prefetch_k(next_job)
                    q_next = _prefetch_q(next_job, staged=traits.Q_LDS_STAGE)
                _dualwave_sync_barrier()

                # Epilogue C11: final rescale and complete the last tile's softmax in-place.
                # v_p_0 belongs to max_m2, whose HI half is dead for row group 0.
                v_o = _pv_diag(v_p_0, v_packs_e11, v_o, 0, _live_m2, 2)
                m_row, l_row, v_s_1, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_1, v_o)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 9, 6, 10)
                _sched_barrier_pairs(traits, 7, 3, 10, traits.SCHED_EXP_MASK)
                _sched_barrier(0)
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                _sched_barrier(0)
                v_o = _anchor_v_o(traits, v_o)
                _s_barrier()
                _sched_barrier(0)

                # Epilogue C12 (memory): read the final V packs for the closing P*V.
                v_packs_e13 = _load_v_head(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _dualwave_sync_barrier()

                # Epilogue C13 (compute): final P*V -> v_o holds the unnormalized output.
                # v_p_1 belongs to max_m1; both halves are dead for row groups 0 and 32.
                v_o = _pv_diag(v_p_1, v_packs_e13, v_o, 1, _live_m1, 1)

            # Normalize O; split-K stores normalized partials for later w_s * l_s reweighting.
            # finalize_o_scale folds the learned sink into the denominator when HAS_SINK
            # (else it is the plain 1/l path, byte-identical) and returns the (max, denom)
            # to record as LSE.
            l_row = ctx.finish_row_sum(l_row)
            o_scale, m_lse, l_lse = ctx.finalize_o_scale(m_row, l_row)
            ctx.scale_o(v_o, o_scale)

            _s_barrier()

            if const_expr(q_next is not None and traits.Q_LDS_STAGE):
                # The O store below reuses the stage, so read it back first.
                q_next = ctx.read_q_stage(q_next)

            # Store LSE before O so sink+LSE temps are not live across the O-store register peak.
            if const_expr(traits.EMIT_LSE):
                ctx.store_lse(m_lse, l_lse, ctx.q_row)
            ctx.store_final_o(v_o, ctx.q_row)
            return q_next

        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            ctx.zero_o_block_if_needed()

        if const_expr(traits.PERSISTENT):
            # Row block_idx.x of the schedule: job count, then the packed jobs; each next job is
            # read one job ahead (the read past the row's last job is never used).
            sched_base = as_mlir_value(fx.Int64(fx.ptrtoint(fx.get_iter(Sched))) + fx.Int64(fx.Int64(0)))
            sched_rsrc = buffer_ops.create_buffer_resource_from_addr(sched_base)
            sched_row = fx.Int32(gpu.block_idx.x) * sched_stride

            def _sched_load(idx):
                return fx.Int32(buffer_ops.buffer_load(sched_rsrc, idx, vec_width=1, is_scalar=True))

            n_jobs = _sched_load(sched_row)
            job_first = _sched_load(sched_row + 1)
            _prefetch_k(job_first)
            q_first = _prefetch_q(job_first)
            # No O stores follow the first job's prefetch for the job-top wait to count past.
            _s_waitcnt(0)
            n_q = len(q_first)
            for k, job_args in range(fx.Index(0), fx.Index(n_jobs), fx.Index(1), init=[job_first] + q_first):
                job = fx.Int32(job_args[0])
                q_cur = [job_args[1 + i] for i in range_constexpr(n_q)]
                job_next = _sched_load(sched_row + fx.Int32(k) + 2)
                # The last job's drain re-fetches its own tiles rather than whatever follows the row.
                job_prefetch = fx.Int32(ArithValue(fx.Int32(k) + 1 < n_jobs).select(job_next, job))
                ctx.bind_item(
                    fx.Index(job & JOB_HEAD_MASK),
                    fx.Index((job >> JOB_QBLOCK_SHIFT) & JOB_QBLOCK_MASK),
                    fx.Index(job >> JOB_BATCH_SHIFT),
                )
                q_next = _main_body(next_job=job_prefetch, q_raw=q_cur)
                _ = yield [job_next] + q_next
            _s_waitcnt(0)
        elif active is None:
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
        ScoreBound: fx.Tensor,
        Sched: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
        sched_ctas: fx.Int32,
        sched_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _dualwave_swp_cache_tag
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        if const_expr(traits.PERSISTENT):
            grid = (fx.Index(sched_ctas), 1, 1)
        else:
            grid = (traits.NUM_HEADS_Q // traits.Q_HEADS_PER_WG, num_q_blocks, bs_idx)

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
            ScoreBound,
            Sched,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            block_table_stride,
            sched_stride,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "llvm.passthrough": passthrough_entries,
            },
        ).launch(
            grid=grid,
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
        score_bound,
        sched,
    ):
        # cu_seqlens_*/block_table/sink/sched are unused kernel-signature placeholders for dense
        # launches / has_sink=False / one job per CTA; the kernel only reads them under
        # const_expr(traits.VARLEN / HAS_SINK / PERSISTENT). O fills those slots. Returns the
        # ordered 20-tuple the JIT entry expects.
        assert (sched is not None) == bool(traits.PERSISTENT), "persistent launches need `schedule`"
        sched_table, sched_ctas, sched_stride = (O, 0, 0) if sched is None else sched
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
            O if score_bound is None else score_bound,
            sched_table,
            batch_size,
            seq_len,
            seq_len if seq_len_kv is None else seq_len_kv,
            traits.DEFAULT_STRIDE_Q_N if stride_q_n is None else stride_q_n,
            traits.DEFAULT_STRIDE_KV_N if stride_kv_n is None else stride_kv_n,
            traits.HEAD_DIM if head_dim_runtime is None else head_dim_runtime,
            0 if block_table_stride is None else block_table_stride,
            sched_ctas,
            sched_stride,
        )

    def _schedule(batch_size, seq_len, seq_len_kv=None):
        """Persistent schedule: (flat int32 rows, number of CTAs, row stride).

        Replays the one-job-per-CTA dispatch: each XCD's share of the job order (same L2
        grouping, longest first) goes job by job to whichever of its CTAs the cost model frees
        first. Row c holds CTA c's job count, then its packed jobs in run order."""
        assert traits.PERSISTENT
        seq_len_kv = seq_len if seq_len_kv is None else seq_len_kv
        nh = traits.NUM_HEADS_Q // traits.Q_HEADS_PER_WG
        nq = -(-seq_len // traits.BLOCK_M)
        assert nh <= JOB_HEAD_MASK + 1 and nq <= JOB_QBLOCK_MASK + 1 and batch_size <= JOB_BATCH_LIMIT
        num_kv_tiles = -(-seq_len_kv // traits.BLOCK_N)

        def _tiles(q_block):
            n = num_kv_tiles
            if traits.CAUSAL:
                causal_end = max(0, (q_block + 1) * traits.BLOCK_M + seq_len_kv - seq_len)
                n = min(-(-causal_end // traits.BLOCK_N), n)
            return max(4, (n + 1) // 2 * 2)

        order = host_job_order(nh, nq, traits.CAUSAL)
        jobs = [(b, h, q) for b in range(batch_size) for h, q in order]
        ctas = min(traits.PERSISTENT, len(jobs))
        queues = NUM_XCD if ctas % NUM_XCD == 0 else 1
        rows = [[] for _ in range(ctas)]
        for x in range(queues):
            free = [(0, c) for c in range(x, ctas, queues)]
            for b, h, q in jobs[x::queues]:
                t, c = heapq.heappop(free)
                rows[c].append(h | q << JOB_QBLOCK_SHIFT | b << JOB_BATCH_SHIFT)
                heapq.heappush(free, (t + JOB_FIXED_TILES + _tiles(q), c))
        stride = 1 + max(len(r) for r in rows)
        flat = []
        for r in rows:
            flat += [len(r)] + r + [0] * (stride - 1 - len(r))
        return flat, ctas, stride

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
        score_bound=None,
        sched=None,
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
            score_bound,
            sched,
        )
        # The scalar shape/mode args (JIT cache key) follow the 11 tensors. has_sink is baked
        # into the module (separate build), so the SINK tensor stays out of the key.
        key = args[11:] + (stream is None,)
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
        score_bound=None,
        sched=None,
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
            score_bound,
            sched,
        )
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return flyc.compile(launch_flash_attn_dualwave_swp, *args, fx.Stream(stream))

    _launch.compile = _compile
    _launch.schedule = _schedule

    return _launch
