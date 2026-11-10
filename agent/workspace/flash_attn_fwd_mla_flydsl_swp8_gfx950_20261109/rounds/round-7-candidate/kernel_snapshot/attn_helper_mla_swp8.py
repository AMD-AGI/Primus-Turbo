"""Super-unit LDS reads and MFMA chains for the eight-stage MLA forward pipeline.

A super unit is one 32-column KV half of a tile for QK (all 12 k-steps, 48 VGPRs of K) and
one 64-wide D_V pair of chunks for P*V (all 4 k-substeps, 32 VGPRs of V), so only one K or
V fragment is resident at a time.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from primus_turbo.flydsl.utils import buffer_ops

from .attn_helper_mla import (
    REF_MARGIN,
    DualwaveKernelContext,
    _ds_read_tr16_b64,
    _fadd,
    _fmax,
    _fsub,
    _if_wave,
    _k_buf_base,
    _lane_pair_reduce,
    _load_k_pack_aligned,
    _mfma_acc,
    _reduce_score_tree,
    _sched_barrier,
    _v_buf_base,
    _wave_any,
)

DS_IMM_MAX = 65535


class Swp8Context(DualwaveKernelContext):
    v_lane_bytes_opaque = (None, None)

    def init_v_read_bases(self):
        """Per-lane V read bytes for each buffer whose reads would exceed the 16-bit ds offset.
        Past it LLVM folds base + imm into one constant per read and materializes a VGPR for
        each; an opaque base keeps the per-read offsets immediate. It is built once before the
        loop and stays live: a per-stage copy got reallocated right after its ds_reads issued,
        and that version raced (wrong V for some masked tiles)."""
        traits = self.traits
        pair_off = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
        max_imm = (
            3 * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE
            + traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE
            + traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE
        ) * traits.BF16_BYTES + pair_off
        bases = []
        for buf_id in range_constexpr(2):
            v_base = _v_buf_base(traits, buf_id)
            if const_expr(v_base * traits.BF16_BYTES + max_imm > DS_IMM_MAX):
                lane_bytes = as_mlir_value(fx.Int32((v_base + self.v_lds_read_base_per_lane) * traits.BF16_BYTES))
                bases.append(llvm.inline_asm(T.i32, [lane_bytes], "", "=v,0", has_side_effects=True))
            else:
                bases.append(None)
        self.v_lane_bytes_opaque = tuple(bases)

    def load_k_tile_part(self, tile_idx, buf_id, d_lo, d_hi):
        """DMA instructions [d_lo, d_hi) of one K tile (``load_k_tile`` issues all NUM_DMA_K)."""
        traits = self.traits
        soffset = self.tile_start(tile_idx) * self.stride_kv_n_v
        for wr in range_constexpr(self.dma_wave_reps):
            eff_wave = self.wave_id + fx.Index(wr * traits.NUM_WAVES)
            for d in range_constexpr(d_lo, d_hi):
                n_in_tile = self.n_in_warp * traits.SMEM_N_RPT + eff_wave
                global_d = self.d_bucket * traits.VEC_KV + d * traits.D_128B_SIZE
                src_elem = self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
                lds_ptr = fx.inttoptr(self.lds_ptr_ty, fx.Int32(self.k_dma_m0[buf_id][wr][d]))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(self.k_div, (None, fx.Int32(src_elem)))
                fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(soffset))

    def lds_load_k_half(self, buf_id, half):
        """K packs of one 32-column KV half, k-steps 0..K_STEPS_QK-1."""
        traits = self.traits
        k_base = _k_buf_base(traits, buf_id)
        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            ks_off = (ks // 4) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
                ks % 4
            ) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE
            idx = k_base + self.k_lds_read_base_per_lane + ks_off + half * traits.K_LDS_TO_REG_N_STRIP_STRIDE
            packs.append(
                _load_k_pack_aligned(traits, self.lds_kv_base_ptr, idx, buf_id, self.kv_mfma_pack_type)
            )
        return packs

    def fresh_seed(self):
        """-reference splat built from an opaque copy: CSE would otherwise merge the two halves'
        seeds and keep one 16-VGPR splat live from the first QK half to the second."""
        ref_neg = llvm.inline_asm(T.f32, [as_mlir_value(self.ref_neg)], "", "=v,0", has_side_effects=True)
        return Vec.from_elements([ref_neg], fx.Float32).broadcast_to(16)

    def qk_half(self, k_packs, q_all_scaled_bf16, acc):
        q_vec = Vec(q_all_scaled_bf16)
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            base = ks * self.traits.MFMA_LANE_K
            q_pack = q_vec.shuffle(q_vec, [base + i for i in range(self.traits.MFMA_LANE_K)]).ir_value()
            acc = _mfma_acc(k_packs[ks], q_pack, acc, self.mma_atom, self.mfma_acc_vec_type)
        return acc

    def lds_load_v_pair(self, buf_id, pair, opaque=True):
        """V packs of D chunks (2 * pair, 2 * pair + 1), indexed [k_substep][chunk_in_pair]."""
        traits = self.traits
        v_base = _v_buf_base(traits, buf_id)
        pair_off = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
        if const_expr(opaque and self.v_lane_bytes_opaque[buf_id] is not None):
            lane_bytes = self.v_lane_bytes_opaque[buf_id]
        else:
            lane_bytes = as_mlir_value(fx.Int32((v_base + self.v_lds_read_base_per_lane) * traits.BF16_BYTES))
        lane_ptr = buffer_ops.get_element_ptr(self.lds_kv_base_ptr, byte_offset=lane_bytes, elem_type=T.i8)
        packs = [[None, None] for _ in range(4)]
        for k_substep in range_constexpr(4):
            for j in range_constexpr(2):
                dc_off = pair * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + j * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE
                imm = (k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + dc_off) * traits.BF16_BYTES
                a = _ds_read_tr16_b64(traits, self.v_lds_read_vec4_type, lane_ptr, imm, buf_id)
                b = _ds_read_tr16_b64(traits, self.v_lds_read_vec4_type, lane_ptr, imm + pair_off, buf_id)
                packs[k_substep][j] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def pv_pair(self, p_packs, v_packs, v_o, pair):
        p_lo, p_hi = p_packs
        for step in range_constexpr(4):
            p_pk = p_lo[step] if step < 2 else p_hi[step - 2]
            for j in range_constexpr(2):
                dc = pair * 2 + j
                v_o[dc] = _mfma_acc(v_packs[step][j], p_pk, v_o[dc], self.mma_atom, self.mfma_acc_vec_type)
        return v_o

    def pin_p(self, p_packs):
        """Opaque copies of the P packs. Without a use before stage 5, the exp + cast chain is sunk
        past stage 4's causal-mask branch into a memory stage. The consuming MFMAs are a stage
        later, so the wait states the hidden VALU def needs are long satisfied."""
        return tuple(
            [llvm.inline_asm(pk.type, [pk], "", "=v,0", has_side_effects=True) for pk in packs] for packs in p_packs
        )

    def rowsum(self, l_row, p_packs):
        """Ones-matrix MFMA row sum of the bf16 P packs (see cast_p_and_sum)."""
        p_lo, p_hi = p_packs
        _sched_barrier(0)
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            for pack in (p_lo[pks], p_hi[pks]):
                l_row = _mfma_acc(self.rowsum_ones_a, pack, l_row, self.rowsum_mma_atom, self.rowsum_acc_vec_type)
        return l_row

    def tile_max_lane(self, v_s):
        return _reduce_score_tree([Vec(h) for h in v_s], _fmax, self.fm_fast)

    def outran(self, m_lane):
        return _wave_any(ArithValue(fx.Float32(m_lane) > self.c_zero_f))

    def rebase(self, fired, m_lane, ref, l_row, v_s, v_o):
        """rebase_if_needed with the trigger computed earlier (dense causal, no score bound)."""
        fm = self.fm_fast

        def _then():
            m_tile = _lane_pair_reduce(m_lane, _fmax, fm)
            m_tile = self.floor_masked_max(m_tile)
            delta = _fmax(_fadd(m_tile, fx.Float32(REF_MARGIN), fm), self.c_zero_f, fm)
            r = fx.Float32(rocdl.exp2(T.f32, _fsub(self.c_zero_f, delta, fm)))
            out = [_fadd(ref, delta, fm), self.scale_l_by(l_row, r)]
            for half in v_s:
                out.append(
                    Vec.from_elements(
                        [as_mlir_value(_fsub(Vec(half)[i], delta, fm)) for i in range_constexpr(16)],
                        fx.Float32,
                    ).ir_value()
                )
            out += self._scale_acc(v_o, r)
            return out

        nd = self.traits.D_CHUNKS
        res = _if_wave(fired, [ref, l_row] + list(v_s) + list(v_o), _then)
        return res[0], res[1], (res[2], res[3]), list(res[4 : 4 + nd])
