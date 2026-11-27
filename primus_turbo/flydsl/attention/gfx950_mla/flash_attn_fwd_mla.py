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

from primus_turbo.flydsl.attention.flash_attn_fwd_stock import (
    JOB_BATCH_LIMIT,
    JOB_BATCH_SHIFT,
    JOB_FIXED_TILES,
    JOB_HEAD_MASK,
    JOB_QBLOCK_MASK,
    JOB_QBLOCK_SHIFT,
    emit_dualwave_swp_body,
    prefetch_k,
    prefetch_q,
)
from primus_turbo.flydsl.utils import buffer_ops
from primus_turbo.flydsl.utils.attn_helper_stock import (
    NUM_XCD,
    DualwaveKernelContext,
    _make_dualwave_swp_traits,
    _s_waitcnt,
    dtype_to_elem_type,
    host_job_order,
)


def make_mla_traits(
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
    sm_scale=None,
):
    """The traits ``build_flash_attn_dualwave_swp_module`` compiles (same arguments), for callers
    that embed its job body (``emit_dualwave_swp_body``)."""
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
        xcd_q_order=True,
    )
    traits.BIND_JOB_LAST = True
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
    traits.SM_SCALE = None if sm_scale is None else float(sm_scale)
    return traits


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
    sm_scale=None,
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
    traits = make_mla_traits(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
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
        k_head=k_head,
        qk_mfma16=qk_mfma16,
        pv_mfma16=pv_mfma16,
        rows_per_wave=rows_per_wave,
        fold_barriers=fold_barriers,
        persistent=persistent,
        sm_scale=sm_scale,
    )
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
            prefetch_k(ctx, job_first)
            q_first = prefetch_q(ctx, job_first)
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
                q_next = emit_dualwave_swp_body(ctx, traits, next_job=job_prefetch, q_raw=q_cur)
                _ = yield [job_next] + q_next
            _s_waitcnt(0)
        elif active is None:
            emit_dualwave_swp_body(ctx, traits)
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    emit_dualwave_swp_body(ctx, traits)

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
        # The kernel reads one job past each row's last; a longest last row would read past the table.
        flat.append(0)
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
