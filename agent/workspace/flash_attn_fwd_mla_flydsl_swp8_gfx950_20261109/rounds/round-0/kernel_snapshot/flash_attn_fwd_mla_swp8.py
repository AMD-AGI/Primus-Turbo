"""Eight-stage, wave-group-staggered flash-attention forward for gfx950 MLA (D_QK 192, D_V 128).

Port of the aiter opus d192 schedule onto the FlyDSL dualwave helpers (dense causal, SBHD,
bf16, block_m 256 = 8 waves). Every KV tile runs eight barrier-delimited stages that
alternate LDS traffic and MFMA work, and waves 4-7 trail waves 0-3 by one stage, so on each
SIMD one wave's MFMA runs while its partner wave reads LDS:

  stage0 mem : DMA V(t) into V[t&1] ; read K(t) KV-half 0
  stage1 mma : QK half 0 of S(t)    ; exp2 of S(t-1)
  stage2 mem : read K(t) KV-half 1  ; vmcnt -> V(t-1) landed
  stage3 mma : QK half 1 of S(t)    ; cast P(t-1), MFMA row sum
  stage4 mem : DMA K(t+2) into K[t&1] ; read V(t-1) D 0..63 ; causal mask S(t)
  stage5 mma : P*V(t-1) D 0..63     ; S(t) max trigger
  stage6 mem : read V(t-1) D 64..127 ; vmcnt -> K(t+1) landed
  stage7 mma : P*V(t-1) D 64..127   ; rebase if a score of S(t) outran the reference

A K/V buffer is read two stages before it is overwritten, which is the slack the one-stage
wave-group offset consumes. Softmax arithmetic is that of flash_attn_fwd_mla (QK seeded
with -reference, lifted reference, exact per-tile trigger), so outputs match it bitwise.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.utils.arith import ArithValue
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from .attn_helper_mla import (
    _fmax,
    _lane_pair_reduce,
    _make_dualwave_swp_traits,
    _s_barrier,
    _s_waitcnt,
    _sched_barrier,
    _sched_barrier_pairs,
    _waitcnt_vm_n,
    dtype_to_elem_type,
)
from .attn_helper_mla_swp8 import Swp8Context

SWP8_REV = 2


def build_flash_attn_swp8_module(
    num_heads,
    head_dim=192,
    head_dim_v=128,
    num_kv_heads=None,
    emit_lse=False,
    waves_per_eu=2,
    stagger=True,
    sched_hints=True,
):
    gpu_arch = get_hip_arch()
    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"flash_attn_swp8 requires gfx950, got {gpu_arch}")
    if (head_dim, head_dim_v) != (192, 128):
        raise RuntimeError(f"flash_attn_swp8 covers D_QK/D_V 192/128, got {head_dim}/{head_dim_v}")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    traits = _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        dtype_str="bf16",
        waves_per_eu=waves_per_eu,
        daz=True,
        dualwave_swp_setprio=False,
        dualwave_swp_enable_stagger=False,
        varlen=False,
        cross_seqlen=False,
        emit_lse=emit_lse,
        window_left=-1,
        block_m=256,
        gqa_merge=False,
        sbhd=True,
        has_sink=False,
        head_dim_v=head_dim_v,
    )
    assert traits.NUM_WAVES == 8 and traits.SMEM_N_RPT == traits.NUM_WAVES
    _cache_tag = (SWP8_REV, traits.cache_tag, bool(stagger), bool(sched_hints))
    # One K tile plus one V tile of DMA stays in flight across every rolling wait.
    KEEP_VM = traits.SMEM_D_RPT_K + traits.SMEM_D_RPT_V
    LGKM0 = traits.LGKMCNT_0_ONLY
    EXP = traits.SCHED_EXP_MASK

    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_swp8_gfx950_kernel(
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
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
    ):
        ctx = Swp8Context(
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
        ctx._setup(SharedStorage)
        t_end = ctx.split_t_end
        wave_i32 = fx.Int32(ctx.wave_id_uni)

        def stage_end():
            _sched_barrier(0)
            _s_barrier()
            _sched_barrier(0)

        @flyc.jit
        def _trailing_group_barrier(trailing):
            if trailing:
                _s_barrier()

        @flyc.jit
        def _leading_group_barrier(leading):
            if leading:
                _s_barrier()

        def _phase(t, cur, prev, ref, l_row, v_o, v_s_prev, q, dma_k):
            """Tile t into S buffers while S(t-1) is finished and folded into O."""
            ctx.set_ref(ref)

            # stage0 (mem)
            ctx.load_v_tile(t, cur)
            k_pk = ctx.lds_load_k_half(cur, 0)
            _s_waitcnt(LGKM0)
            stage_end()

            # stage1 (mma)
            s_lo = ctx.qk_half(k_pk, q, ctx.score_acc_seed())
            v_p = ctx.exp2(ctx.exp2(v_s_prev, 0, 16), 16, 16)
            if const_expr(sched_hints):
                _sched_barrier_pairs(traits, 11, 3, 1, EXP)
            stage_end()

            # stage2 (mem)
            k_pk = ctx.lds_load_k_half(cur, 1)
            _s_waitcnt(LGKM0)
            _waitcnt_vm_n(KEEP_VM)
            stage_end()

            # stage3 (mma)
            s_hi = ctx.qk_half(k_pk, q, ctx.fresh_seed())
            p_packs = ctx.cast_p(v_p)
            if const_expr(sched_hints):
                _sched_barrier_pairs(traits, 8, 2, 2)
            l_row = ctx.rowsum(l_row, p_packs)
            stage_end()

            # stage4 (mem)
            if const_expr(dma_k):
                nxt = t + fx.Index(2)
                last = t_end - fx.Index(1)
                ctx.load_k_tile(fx.Index(ArithValue(nxt < last).select(nxt, last)), cur)
            v_pk = ctx.lds_load_v_pair(prev, 0)
            v_s = ctx.causal_mask_prologue_if_needed((s_lo, s_hi), t, kv_end_tile=t + fx.Index(1))
            _s_waitcnt(LGKM0)
            stage_end()

            # stage5 (mma)
            v_o = ctx.pv_pair(p_packs, v_pk, v_o, 0)
            m_lane = ctx.tile_max_lane(v_s)
            fired = ctx.outran(m_lane)
            if const_expr(sched_hints):
                _sched_barrier_pairs(traits, 8, 2, 3)
            stage_end()

            # stage6 (mem)
            v_pk = ctx.lds_load_v_pair(prev, 1)
            _s_waitcnt(LGKM0)
            if const_expr(dma_k):
                _waitcnt_vm_n(KEEP_VM)
            stage_end()

            # stage7 (mma)
            v_o = ctx.pv_pair(p_packs, v_pk, v_o, 1)
            ref, l_row, v_s, v_o = ctx.rebase(fired, m_lane, ref, l_row, v_s, v_o)
            stage_end()
            return ref, l_row, v_o, v_s

        def _main_body():
            # Prologue: K0 -> K[0], Q -> registers, K1 -> K[1], V0 -> V[0].
            ctx.load_k_tile(fx.Index(0), 0)
            q_all_bf16 = ctx.load_all()
            ctx.load_k_tile(fx.Index(1), 1)
            ctx.load_v_tile(fx.Index(0), 0)
            q = ctx.scale_all(q_all_bf16)
            _waitcnt_vm_n(KEEP_VM)
            stage_end()
            # The trailing group spends one extra barrier here and the leading group one at the end.
            if const_expr(stagger):
                _trailing_group_barrier(wave_i32 >= fx.Int32(4))

            k_pk = ctx.lds_load_k_half(0, 0)
            _s_waitcnt(LGKM0)
            stage_end()
            s_lo = ctx.qk_half(k_pk, q, ctx.c_zero_v16f32)
            stage_end()
            k_pk = ctx.lds_load_k_half(0, 1)
            _s_waitcnt(LGKM0)
            stage_end()
            s_hi = ctx.qk_half(k_pk, q, ctx.c_zero_v16f32)
            v_s = ctx.causal_mask_split_prologue_if_needed((s_lo, s_hi))
            m_pro = _lane_pair_reduce(ctx.tile_max_lane(v_s), _fmax, ctx.fm_fast)
            m_pro = ctx.lift_ref(ctx.floor_masked_max(m_pro))
            v_s = ctx.shift_scores(v_s, m_pro)
            ctx.set_ref(m_pro)
            stage_end()
            # K0 is fully read, so K2 reuses its buffer; the wait lands K1.
            ctx.load_k_tile(fx.Index(2), 0)
            _waitcnt_vm_n(KEEP_VM)
            stage_end()

            init_args = [m_pro, ctx.c_zero_v4f32]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(ctx.c_zero_v16f32)
            init_args += [v_s[0], v_s[1]]

            # Phases 1 .. t_end-2 in ping/pong pairs (t_end is even), then the last phase peeled.
            for j, loop_args in range(fx.Index(1), t_end - fx.Index(1), fx.Index(2), init=init_args):
                ref = loop_args[0]
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_s = (loop_args[2 + traits.D_CHUNKS], loop_args[3 + traits.D_CHUNKS])
                ref, l_row, v_o, v_s = _phase(j, 1, 0, ref, l_row, v_o, v_s, q, True)
                ref, l_row, v_o, v_s = _phase(j + fx.Index(1), 0, 1, ref, l_row, v_o, v_s, q, True)
                loop_results = yield [ref, l_row] + v_o + [v_s[0], v_s[1]]

            ref = loop_results[0]
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_s = (loop_results[2 + traits.D_CHUNKS], loop_results[3 + traits.D_CHUNKS])
            ref, l_row, v_o, v_s = _phase(t_end - fx.Index(1), 1, 0, ref, l_row, v_o, v_s, q, False)

            # Epilogue: finish the last tile (V[1]).
            _waitcnt_vm_n(0)
            stage_end()
            ctx.set_ref(ref)
            v_p = ctx.exp2(ctx.exp2(v_s, 0, 16), 16, 16)
            p_packs = ctx.cast_p(v_p)
            l_row = ctx.rowsum(l_row, p_packs)
            stage_end()
            v_pk = ctx.lds_load_v_pair(1, 0)
            _s_waitcnt(LGKM0)
            stage_end()
            v_o = ctx.pv_pair(p_packs, v_pk, v_o, 0)
            stage_end()
            v_pk = ctx.lds_load_v_pair(1, 1)
            _s_waitcnt(LGKM0)
            stage_end()
            v_o = ctx.pv_pair(p_packs, v_pk, v_o, 1)
            _sched_barrier(0)
            if const_expr(stagger):
                _leading_group_barrier(wave_i32 < fx.Int32(4))

            l_fin = ctx.finish_row_sum(l_row)
            o_scale, m_lse, l_lse = ctx.finalize_o_scale(ref, l_fin)
            ctx.scale_o(v_o, o_scale)
            if const_expr(traits.EMIT_LSE):
                ctx.store_lse(m_lse, l_lse, ctx.q_row)
            ctx.store_final_o(v_o, ctx.q_row)

        _main_body()

    @flyc.jit
    def launch_flash_attn_swp8(
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
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _cache_tag
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        flash_attn_swp8_gfx950_kernel(
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
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            block_table_stride,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "llvm.passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(traits.NUM_HEADS_Q, num_q_blocks, bs_idx),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-sched-strategy": "max-memory-clause",
            "enable-post-misched": True,
            "lsr-drop-solution": True,
        },
    }

    _compiled: dict = {}

    def _launch(Q, K, V, O, batch_size, seq_len, *, stride_q_n, stride_kv_n, seq_len_kv=None, lse=None, stream=None):  # noqa: E741
        args = (
            Q,
            K,
            V,
            O,
            O if lse is None else lse,
            O,
            O,
            O,
            O,
            O,
            batch_size,
            seq_len,
            seq_len if seq_len_kv is None else seq_len_kv,
            stride_q_n,
            stride_kv_n,
            traits.HEAD_DIM,
            0,
        )
        key = args[10:] + (stream is None,)
        fn = _compiled.get(key)
        if fn is None:
            with CompilationContext.compile_hints(_compile_hints):
                fn = flyc.compile(launch_flash_attn_swp8, *args, stream)
            _compiled[key] = fn
        return fn(*args, stream)

    return _launch
