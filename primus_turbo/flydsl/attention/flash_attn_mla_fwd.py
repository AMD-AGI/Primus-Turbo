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

"""Dual-wave, software-pipelined MLA flash-attention forward for gfx950 (qk 192 / v 128, bf16).

The DeepSeek-V3 / Kimi training head dims: Q and K carry head_dim 192, V and O carry 128.
SBHD dense, bottom-right causal with Sq == Skv, no window, no sink. It reuses the D=128
dual-wave pipeline of flash_attn_fwd.py for the P*V / softmax side and widens only the QK side:
twelve 16-wide k-steps over three 64-column K LDS strips, read one strip at a time so the K
tile never sits in registers whole.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from primus_turbo.flydsl.utils.attn_helper import (
    DualwaveKernelContext,
    DualwaveSwpTraits,
    _anchor_v_o,
    _concat_vectors,
    _dualwave_sync_barrier,
    _k_buf_base,
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

MLA_HEAD_DIM_QK = 192
MLA_HEAD_DIM_V = 128
# One K LDS strip is 64 columns = 4 k-steps of the 32x32x16 MFMA.
_KS_PER_STRIP = 4


class MlaDualwaveSwpTraits(DualwaveSwpTraits):
    @property
    def cache_tag(self):
        # The on-disk cache does not hash MlaDualwaveKernelContext; name its codegen-relevant revision.
        return super().cache_tag + ("mla", self.HEAD_DIM_QK, self.K_HEAD_STEPS, "qpin")


def _make_mla_fwd_traits(num_heads, num_kv_heads, block_m=256, waves_per_eu=2, emit_lse=True, k_head_steps=4):
    """D=128 dual-wave traits for the V/O side, with the QK side widened to head_dim 192."""
    base = _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        MLA_HEAD_DIM_V,
        causal=True,
        dtype_str="bf16",
        waves_per_eu=waves_per_eu,
        daz=True,
        dualwave_swp_setprio=True,
        dualwave_swp_enable_stagger=False,
        varlen=False,
        cross_seqlen=False,
        emit_lse=emit_lse,
        window_left=-1,
        block_m=block_m,
        gqa_merge=False,
        sbhd=True,
        has_sink=False,
    )
    t = MlaDualwaveSwpTraits(**base.__dict__)
    k_d_rpt = MLA_HEAD_DIM_QK // t.D_128B_SIZE
    k_tile = t.SMEM_N_RPT * k_d_rpt * t.SMEM_K_LINE_STRIDE
    v_tile = t.SMEM_N_RPT * t.SMEM_D_RPT * t.SMEM_V_LINE_STRIDE
    per_buf = k_tile + v_tile
    t.__dict__.update(
        HEAD_DIM_QK=MLA_HEAD_DIM_QK,
        HEAD_DIM_V=MLA_HEAD_DIM_V,
        K_STEPS_QK=MLA_HEAD_DIM_QK // t.K_STEP_QK,
        K_HEAD_STEPS=int(k_head_steps),
        SMEM_K_D_RPT=k_d_rpt,
        SMEM_K_TILE_ELEMS=k_tile,
        DUALWAVE_SWP_KV_PER_BUFFER=per_buf,
        LDS_KV_TOTAL_SIZE=2 * per_buf,
        DUALWAVE_SWP_K_BUF_BASE=(0, per_buf),
        DUALWAVE_SWP_V_BUF_BASE=(k_tile, k_tile + per_buf),
    )
    assert t.K_STEPS_QK % _KS_PER_STRIP == 0
    assert 0 < t.K_HEAD_STEPS <= t.K_STEPS_QK and t.K_HEAD_STEPS % _KS_PER_STRIP == 0
    return t


class MlaDualwaveKernelContext(DualwaveKernelContext):
    """DualwaveKernelContext with distinct QK (192) and V/O (128) head dims and row strides."""

    def __init__(self, traits, Q, K, V, O, Lse, seq_len, stride_q_n, stride_k_n, stride_v_n, stride_o_n, head_dim_qk):  # noqa: E741
        super().__init__(
            traits,
            Q,
            K,
            V,
            O,
            DebugCounts=Lse,
            seq_len=seq_len,
            seq_len_kv=seq_len,
            stride_q_n=stride_q_n,
            stride_kv_n=stride_k_n,
            head_dim_runtime=head_dim_qk,
        )
        self.stride_v_n = stride_v_n
        self.stride_o_n = stride_o_n

    def _setup(self, shared_storage):
        super()._setup(shared_storage)
        traits = self.traits
        self.NUM_DMA_K = traits.SMEM_K_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT
        reps = self.dma_wave_reps
        # A drain waits for every DMA group but the newest; the smaller group bounds that from below.
        self.VM_DRAIN_KV = min(self.NUM_DMA_K, self.NUM_DMA_V) * reps
        self.VM_DRAIN_V = self.NUM_DMA_V * reps // 2
        self.k_dma_m0 = self.dma_m0_table(
            lambda b, w, d: self._dma_base(b, w, d, _k_buf_base, traits.SMEM_K_LINE_STRIDE), self.NUM_DMA_K
        )

        self.stride_v_n_v = fx.Index(self.stride_v_n)
        self.stride_o_n_v = fx.Index(self.stride_o_n)
        self.q_gmem_elem_offset = self.q_start * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM_QK
        self.k_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM_QK
        self.v_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM_V

        def _view(tensor, heads, head_dim, stride):
            # SBHD [S, B, H, D]: the batch base is b*H*D, the seq step lives in the row stride.
            nrec = self.seq_len_v * stride * fx.Index(traits.BF16_BYTES)
            off = self.batch_idx * fx.Index(heads * head_dim) * fx.Index(traits.BF16_BYTES)
            base_iter = fx.get_iter(tensor)
            base_i64 = fx.Int64(fx.ptrtoint(base_iter))
            shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(off))
            buf_ptr = fx.rocdl.make_buffer_ptr(shifted, num_records_bytes=fx.Int64(nrec))
            layout = fx.make_layout(fx.Int32(self.seq_len_v * stride), fx.Int32(1))
            return fx.logical_divide(fx.make_view(buf_ptr, layout), fx.make_layout(1, 1))

        self.q_div = _view(self.Q, traits.NUM_HEADS_Q, traits.HEAD_DIM_QK, self.stride_q_n_v)
        self.k_div = _view(self.K, traits.NUM_HEADS_KV, traits.HEAD_DIM_QK, self.stride_kv_n_v)
        self.v_div = _view(self.V, traits.NUM_HEADS_KV, traits.HEAD_DIM_V, self.stride_v_n_v)
        self.o_div = _view(self.O, traits.NUM_HEADS_Q, traits.HEAD_DIM_V, self.stride_o_n_v)

    def load_all(self):
        traits = self.traits
        self.q_row_in_block = self.wave_q_offset + self.lane_mod_32
        self.q_start_pos_i32 = fx.Int32(self.q_start + self.wave_row_group_uni * traits.ROWS_PER_WAVE)
        self.q_row = self.q_start + self.q_row_in_block
        self.q_row_i32 = fx.Int32(self.q_row)

        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            elem_index = self.q_gmem_elem_offset + (
                self.q_row_in_block * self.stride_q_n_v + (ks * traits.K_STEP_QK + self.lane_div_32 * traits.MFMA_LANE_K)
            )
            q_i32_pack = fly.copy_atom_call_ssa(
                [self.q_load_i32x4_type],
                self.load_atom_128,
                fx.slice(self.q_div, (None, fx.Int32(elem_index))),
            )
            packs.append(Vec(q_i32_pack, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value())
        while len(packs) > 1:
            nxt = [_concat_vectors(packs[i], packs[i + 1]) for i in range(0, len(packs) - 1, 2)]
            if len(packs) % 2:
                nxt[-1] = _concat_vectors(nxt[-1], packs[-1])
            packs = nxt
        return Vec(packs[0], (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)

    def scale_all(self, q_all_bf16):
        """Scaled Q, pinned packed: each k-step's 4 dwords pass an empty asm identity. Otherwise the
        backend sinks the bf16 repack into every QK and keeps the 96-dword f32 product live across
        the loop and the drain instead of the 48-dword packed one, which spills to scratch."""
        traits = self.traits
        q = Vec(super().scale_all(q_all_bf16))
        lane_k = traits.MFMA_LANE_K
        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            pack = q.shuffle(q, [ks * lane_k + i for i in range(lane_k)]).bitcast(fx.Int32)
            pinned = llvm.inline_asm(self.q_load_i32x4_type, [pack.ir_value()], "", "=v,0", has_side_effects=False)
            packs.append(Vec(pinned, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value())
        while len(packs) > 1:
            nxt = [_concat_vectors(packs[i], packs[i + 1]) for i in range(0, len(packs) - 1, 2)]
            if len(packs) % 2:
                nxt[-1] = _concat_vectors(nxt[-1], packs[-1])
            packs = nxt
        return Vec(packs[0], (traits.K_STEPS_QK * lane_k,), self.elem_dtype)

    def _load_kv_x(self, tile_start, buf_id, dma_m0, div, num_dma, src_base, stride_n):
        soffset = tile_start * stride_n
        for wr in range_constexpr(self.dma_wave_reps):
            eff_wave = self.wave_id + fx.Index(wr * self.traits.NUM_WAVES)
            for d in range_constexpr(num_dma):
                n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + eff_wave
                global_d = self.d_bucket * self.traits.VEC_KV + d * self.traits.D_128B_SIZE
                src_elem = src_base + n_in_tile * stride_n + global_d
                lds_ptr = fx.inttoptr(self.lds_ptr_ty, fx.Int32(dma_m0[buf_id][wr][d]))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(div, (None, fx.Int32(src_elem)))
                fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(soffset))

    def load_k_tile(self, tile_idx, buf_id):
        self._load_kv_x(
            self.tile_start(tile_idx),
            buf_id,
            self.k_dma_m0,
            self.k_div,
            self.NUM_DMA_K,
            self.k_gmem_elem_offset,
            self.stride_kv_n_v,
        )

    def load_v_tile(self, tile_idx, buf_id):
        self._load_kv_x(
            self.tile_start(tile_idx),
            buf_id,
            self.v_dma_m0,
            self.v_div,
            self.NUM_DMA_V,
            self.v_gmem_elem_offset,
            self.stride_v_n_v,
        )

    def _final_o_base(self, q_row):
        return q_row * self.stride_o_n_v + self.q_head_idx * self.traits.HEAD_DIM_V + self.lane_div_32 * 8


def build_flash_attn_mla_fwd_module(num_heads, num_kv_heads=None, emit_lse=True, block_m=256, waves_per_eu=2, k_head_steps=4):
    """Build the gfx950 MLA (qk 192 / v 128) SBHD causal forward launcher."""
    gpu_arch = get_hip_arch()
    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"flash_attn_mla_fwd requires gfx950 (uses ds_read_tr16_b64), got {gpu_arch}")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    traits = _make_mla_fwd_traits(
        num_heads, num_kv_heads, block_m=block_m, waves_per_eu=waves_per_eu, emit_lse=emit_lse, k_head_steps=k_head_steps
    )
    _mla_cache_tag = traits.cache_tag
    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(name="flash_attn_mla_fwd_gfx950_kernel", known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_mla_fwd_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Lse: fx.Tensor,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_k_n: fx.Int32,
        stride_v_n: fx.Int32,
        stride_o_n: fx.Int32,
        head_dim_qk: fx.Int32,
    ):
        ctx = MlaDualwaveKernelContext(
            traits, Q, K, V, O, Lse, seq_len, stride_q_n, stride_k_n, stride_v_n, stride_o_n, head_dim_qk
        )
        ctx._setup(SharedStorage)

        K_HEAD = traits.K_HEAD_STEPS
        V_HEAD = 1
        vm_drain_sync = ctx.VM_DRAIN_KV

        def _qk_cluster_sync():
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(vm_drain_sync)
            _dualwave_sync_barrier()

        l_row_init = ctx.c_zero_v4f32
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
                return ctx.lds_load_k(buf_id, ks_range=(0, K_HEAD))

            def _qk(v_k, buf_id, live=None):
                """QK over all 12 k-steps; each later 64-col strip is read after the previous strip's MFMAs."""
                if const_expr(live is None):
                    v_s = ctx.qk(v_k, q_all_scaled_bf16, ks_range=(0, K_HEAD))
                    for ks0 in range_constexpr(K_HEAD, traits.K_STEPS_QK, _KS_PER_STRIP):
                        rng = (ks0, ks0 + _KS_PER_STRIP)
                        v_k = ctx.lds_load_k(buf_id, ks_range=rng, k_regs=v_k)
                        v_s = ctx.qk(v_k, q_all_scaled_bf16, v_s=v_s, ks_range=rng)
                    return v_s
                live_lo, live_hi = live
                v_s = ctx.qk_live(v_k, q_all_scaled_bf16, live_lo, live_hi, ks_range=(0, K_HEAD))
                for ks0 in range_constexpr(K_HEAD, traits.K_STEPS_QK, _KS_PER_STRIP):
                    rng = (ks0, ks0 + _KS_PER_STRIP)
                    v_k = ctx.lds_load_k(buf_id, ks_range=rng, k_regs=v_k)
                    v_s = ctx.qk_live(v_k, q_all_scaled_bf16, live_lo, live_hi, v_s=v_s, ks_range=rng)
                return v_s

            def _load_v_head(buf_id):
                return ctx.lds_load_v(buf_id, substeps=(0, V_HEAD))

            def _pv_step(step, v_p, v_v, v_o, buf_id):
                v_o = ctx.pv_step_k(step, v_p, v_v, v_o)
                nxt = step + V_HEAD
                if const_expr(nxt < 4):
                    ctx.lds_load_v(buf_id, substeps=(nxt, nxt + 1), packs=v_v)
                return v_o

            def _pv(v_p, v_v, v_o, buf_id):
                for step in range_constexpr(4):
                    v_o = _pv_step(step, v_p, v_v, v_o, buf_id)
                return v_o

            def _pv_branch(v_o, live, body):
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
                v_o = ctx.pv_step_k(0, v_p, v_v, v_o)
                ctx.lds_load_v(buf_id, substeps=(1, 2), packs=v_v)
                v_o = ctx.pv_step_k(1, v_p, v_v, v_o)

                def _tail(v_o):
                    ctx.lds_load_v(buf_id, substeps=(2, 4), packs=v_v)
                    v_o = ctx.pv_step_k(2, v_p, v_v, v_o)
                    return ctx.pv_step_k(3, v_p, v_v, v_o)

                return _pv_branch(v_o, live_hi, _tail)

            def _pv_both_live(v_p, v_v, v_o, buf_id, live_lo, live_hi):
                def _head(v_o):
                    v_o = ctx.pv_step_k(0, v_p, v_v, v_o)
                    ctx.lds_load_v(buf_id, substeps=(1, 2), packs=v_v)
                    return ctx.pv_step_k(1, v_p, v_v, v_o)

                v_o = _pv_branch(v_o, live_lo, _head)

                def _tail(v_o):
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

            v_s_0 = _qk(v_k, 0)
            _sched_barrier(0)
            v_s_0 = ctx.causal_mask_split_prologue_if_needed(v_s_0)
            m_row_pro = ctx.reduce_max(v_s_0)
            # Floor fully-masked rows (-inf) to finite so exp2 yields 0, not NaN.
            m_row_pro = ctx.floor_masked_max(m_row_pro)
            m_row_pro = ctx.lift_ref(m_row_pro)
            v_s_0 = ctx.shift_scores(v_s_0, m_row_pro)
            ctx.set_ref(m_row_pro)
            v_p_0 = ctx.exp2(v_s_0, 0, 16)
            _dualwave_sync_barrier()

            loop_lb = ctx.split_tile(3)
            ctx.load_k_split(2, 0)

            init_args = [m_row_pro, l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(v_o_zero)
            init_args.append(v_p_0[0])
            init_args.append(v_p_0[1])
            init_args.append(ctx.score_acc_seed())

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
                j_idx = j

                # Cluster 0: read resident K head for MMA0.
                _sched_barrier(0)
                v_k = _load_k_head(1)
                _dualwave_sync_barrier()

                # Cluster 1: V DMA, finish v_p_0 softmax, update l_row, cast P, QK of tile j-1.
                ctx.load_v_tile(j_idx - 2, 1)
                v_p_0 = ctx.exp2(v_p_0, 16, 16)
                v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
                _sched_barrier(0)
                v_s_1 = _qk(v_k, 1)
                _sched_barrier_pairs(traits, 6, 3, 1, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 1)
                _qk_cluster_sync()

                # Cluster 2: read this tile's V head for P*V.
                _sched_barrier(0)
                v_v = _load_v_head(0)
                _dualwave_sync_barrier()

                # Cluster 3: K DMA, P*V, rescale, first-half exp2.
                ctx.load_k_tile(j_idx, 1)
                _s_setprio(1)
                v_o = _pv_step(0, v_p_0, v_v, v_o, 0)
                # BLOCK_M=256 spans four diagonal KV tiles and the drain masks only three, so the
                # loop can hold the first one.
                v_s_1 = ctx.causal_mask_prologue_if_needed(v_s_1, j_idx - 2, kv_end_tile=j_idx - 1)
                for pvs in range_constexpr(1, 4):
                    v_o = _pv_step(pvs, v_p_0, v_v, v_o, 0)
                m_row, l_row, v_s_1, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_1, v_o)
                v_p_1 = ctx.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 6, 6, 2)
                _sched_barrier_pairs(traits, 6, 3, 2, traits.SCHED_EXP_MASK)
                _s_setprio(0)
                _qk_cluster_sync()

                # Cluster 4 mirrors C0.
                _sched_barrier(0)
                v_k = _load_k_head(0)
                _dualwave_sync_barrier()

                # Cluster 5 mirrors C1.
                ctx.load_v_tile(j_idx - 1, 0)
                v_p_1 = ctx.exp2(v_p_1, 16, 16)
                v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
                _sched_barrier(0)
                v_s_0 = _qk(v_k, 0)
                _sched_barrier_pairs(traits, 6, 3, 3, traits.SCHED_EXP_MASK)
                _sched_barrier_pairs(traits, 10, 5, 3)
                _qk_cluster_sync()

                # Cluster 6 mirrors C2.
                _sched_barrier(0)
                v_v = _load_v_head(1)
                v_s_0 = ctx.causal_mask_prologue_if_needed(v_s_0, j_idx - 1, kv_end_tile=j_idx)
                _dualwave_sync_barrier()

                # Cluster 7 mirrors C3 and carries m_row, l_row, v_o, and packed v_p_0.
                ctx.load_k_tile(j_idx + 1, 0)
                _s_setprio(1)
                v_o = _pv_step(0, v_p_1, v_v, v_o, 1)
                for pvs in range_constexpr(1, 4):
                    v_o = _pv_step(pvs, v_p_1, v_v, v_o, 1)
                m_row, l_row, v_s_0, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_0, v_o)
                v_p_0 = ctx.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 6, 5, 4)
                _sched_barrier_pairs(traits, 6, 3, 4, traits.SCHED_EXP_MASK)
                _s_setprio(0)
                _qk_cluster_sync()

                yield_args = [m_row, l_row] + v_o + [v_p_0[0], v_p_0[1], ctx.score_acc_seed()]
                loop_results = yield yield_args

            m_row = loop_results[0]
            ctx.set_ref(m_row)
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = (loop_results[2 + traits.D_CHUNKS], loop_results[3 + traits.D_CHUNKS])
            ctx.set_seed_vec(loop_results[4 + traits.D_CHUNKS])

            max_m3 = split_t_end - 3
            max_m2 = split_t_end - 2
            max_m1 = split_t_end - 1

            # Dense causal drain: skip wave-uniform dead halves on the last two KV tiles.
            _live_hi_m2 = ctx.chunk_live(max_m2)[1]
            _live_lo_m1, _live_hi_m1 = ctx.chunk_live(max_m1)

            # E0: prefetch V max_m3, read K max_m3.
            _s_nop(3)
            _sched_barrier(0)
            ctx.load_v_tile(max_m3, 1)
            v_k = _load_k_head(1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.VM_DRAIN_KV)
            _dualwave_sync_barrier()

            # E1: finish v_p_0, QK of max_m3.
            v_p_0 = ctx.exp2(v_p_0, 16, 16)
            v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
            v_s_1 = _qk(v_k, 1)
            _sched_barrier_pairs(traits, 6, 3, 5, traits.SCHED_EXP_MASK)
            _sched_barrier_pairs(traits, 10, 5, 5)
            _dualwave_sync_barrier()

            # E2: prefetch K max_m1, read V (buf0), mask v_s_1.
            _s_nop(3)
            _sched_barrier(0)
            ctx.load_k_tile(max_m1, 1)
            v_packs_e3 = _load_v_head(0)
            v_s_1 = ctx.causal_mask_prologue_if_needed(v_s_1, max_m3, kv_end_tile=max_m2)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.VM_DRAIN_KV)
            _dualwave_sync_barrier()

            # E3: full P*V + rescale.
            _s_setprio(1)
            v_o = _pv(v_p_0, v_packs_e3, v_o, 0)
            m_row, l_row, v_s_1, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_1, v_o)
            v_p_1 = ctx.exp2(v_s_1, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, 6)
            _sched_barrier_pairs(traits, 6, 3, 6, traits.SCHED_EXP_MASK)
            _sched_barrier(0)
            v_o = _anchor_v_o(traits, v_o)
            _s_setprio(0)
            _dualwave_sync_barrier()

            # E4: prefetch V max_m2 (buf0), read K buf0.
            _s_nop(3)
            _sched_barrier(0)
            ctx.load_v_tile(max_m2, 0)
            v_k = _load_k_head(0)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.VM_DRAIN_KV)
            _dualwave_sync_barrier()

            # E5: finish v_p_1, QK of max_m2.
            v_p_1 = ctx.exp2(v_p_1, 16, 16)
            v_p_1, l_row = ctx.cast_p_and_sum(l_row, v_p_1)
            v_s_0 = _qk(v_k, 0, live=(None, _live_hi_m2))
            _sched_barrier_pairs(traits, 6, 3, 7, traits.SCHED_EXP_MASK)
            _sched_barrier_pairs(traits, 10, 5, 7)
            _dualwave_sync_barrier()

            # E6: read V (buf1), mask v_s_0.
            v_packs_e7 = _load_v_head(1)
            v_s_0 = ctx.causal_mask_prologue_if_needed(v_s_0, max_m2, kv_end_tile=max_m1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.VM_DRAIN_V)
            _dualwave_sync_barrier()

            # E7: full P*V + rescale.
            _s_setprio(1)
            v_o = _pv(v_p_1, v_packs_e7, v_o, 1)
            m_row, l_row, v_s_0, v_o = ctx.rebase_if_needed(m_row, l_row, v_s_0, v_o)
            v_p_0 = ctx.exp2(v_s_0, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, 8)
            _sched_barrier_pairs(traits, 6, 3, 8, traits.SCHED_EXP_MASK)
            _sched_barrier(0)
            v_o = _anchor_v_o(traits, v_o)
            _s_setprio(0)
            _dualwave_sync_barrier()

            # E8: prefetch V max_m1 (buf1), read K buf1.
            _s_nop(3)
            _sched_barrier(0)
            ctx.load_v_tile(max_m1, 1)
            v_k = _load_k_head(1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.VM_DRAIN_V)
            _dualwave_sync_barrier()

            # E9: finish v_p_0, QK of max_m1.
            v_p_0 = ctx.exp2(v_p_0, 16, 16)
            v_p_0, l_row = ctx.cast_p_and_sum(l_row, v_p_0)
            v_s_1 = _qk(v_k, 1, live=(_live_lo_m1, _live_hi_m1))
            _sched_barrier_pairs(traits, 6, 3, 9, traits.SCHED_EXP_MASK)
            _sched_barrier_pairs(traits, 10, 5, 9)
            _dualwave_sync_barrier()

            # E10: read final V packs, mask v_s_1, drain DMAs.
            v_packs_e11 = _load_v_head(0)
            v_s_1 = ctx.causal_mask_prologue_if_needed(v_s_1, max_m1, kv_end_tile=split_t_end)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(0)
            _dualwave_sync_barrier()

            # E11: P*V of max_m2, rescale, last tile softmax.
            v_o = _pv_hi_live(v_p_0, v_packs_e11, v_o, 0, _live_hi_m2)
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

            # E12: read the final V packs.
            v_packs_e13 = _load_v_head(1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _dualwave_sync_barrier()

            # E13: final P*V of max_m1.
            v_o = _pv_both_live(v_p_1, v_packs_e13, v_o, 1, _live_lo_m1, _live_hi_m1)

            l_row = ctx.finish_row_sum(l_row)
            o_scale, m_lse, l_lse = ctx.finalize_o_scale(m_row, l_row)
            ctx.scale_o(v_o, o_scale)
            _s_barrier()
            if const_expr(traits.EMIT_LSE):
                ctx.store_lse(m_lse, l_lse, ctx.q_row)
            ctx.store_final_o(v_o, ctx.q_row)

        _main_body()

    @flyc.jit
    def launch_flash_attn_mla_fwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Lse: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_k_n: fx.Int32,
        stride_v_n: fx.Int32,
        stride_o_n: fx.Int32,
        head_dim_qk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _mla_cache_tag
        num_q_blocks = (fx.Index(seq_len) + traits.BLOCK_M - 1) // traits.BLOCK_M
        flash_attn_mla_fwd_gfx950_kernel(
            Q,
            K,
            V,
            O,
            Lse,
            seq_len,
            stride_q_n,
            stride_k_n,
            stride_v_n,
            stride_o_n,
            head_dim_qk,
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
            grid=(traits.NUM_HEADS_Q, num_q_blocks, fx.Index(batch_size)),
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
    _COMPILED_MAX = 64

    def _launch(Q, K, V, O, Lse, batch_size, seq_len, stream=None):  # noqa: E741
        """q [S,B,Hq,192], k [S,B,Hkv,192], v [S,B,Hkv,128], o [S,B,Hq,128], all contiguous SBHD."""
        args = (
            Q,
            K,
            V,
            O,
            O if Lse is None else Lse,
            batch_size,
            seq_len,
            batch_size * traits.NUM_HEADS_Q * traits.HEAD_DIM_QK,
            batch_size * traits.NUM_HEADS_KV * traits.HEAD_DIM_QK,
            batch_size * traits.NUM_HEADS_KV * traits.HEAD_DIM_V,
            batch_size * traits.NUM_HEADS_Q * traits.HEAD_DIM_V,
            traits.HEAD_DIM_QK,
        )
        key = args[5:] + (stream is None,)
        fn = _compiled.get(key)
        if fn is None:
            if len(_compiled) >= _COMPILED_MAX:
                _compiled.clear()
            with CompilationContext.compile_hints(_compile_hints):
                fn = flyc.compile(launch_flash_attn_mla_fwd, *args, stream)
            _compiled[key] = fn
        return fn(*args, stream)

    _launch.traits = traits
    return _launch
