###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 dense / SWA attention BACKWARD kernel (FlyDSL, design §5).

MFMA INITIAL VERSION (design §5.2, backward campaign round-2): all five backward
GEMMs run on the ``mfma_f32_16x16x32`` atom (bf16 / fp16 in, fp32 accumulate),
replacing the round-1 scalar wave-per-row dot + ``shuffle_xor`` dataflow. The
softmax probabilities are re-materialised from the saved fp32 ``LSE`` (no
``[Sq, Sk]`` matrix), following standard FlashAttention-2 backward (design §5.1):

    P    = exp(scale * (Q·K) - LSE)          (re-materialised, masked)
    Dv   = rowsum(dO * O)                     (per-query scalar; torch pre-pass)
    dP   = dO · V
    dS   = P * (dP - Dv)
    dQ  += scale * dS · K
    dK  += scale * dS · Q
    dV  += P  · dO
    dsink_h = -sum_t (exp(sink_h - LSE_t) * Dv_t)   (torch reduction)

Three single-wave (64-lane) kernels, atomic-free (mirrors the Triton split BWD).
Atom is ``A[M,K] x B[N,K] -> C[M,N]`` contracting ``K``; every contraction is a
multiple of 32 (MFMA-atom requirement, fwd tips). ``BM = 16`` is the WG-owned row
count, ``BN = 32`` the inner loop block and the accumulation contraction.

* ``kernel_dsk_attn_bwd_dq``  — grid ``(Sq/16, B*HQ)``, owns ``dQ[16,512]``.
  Loops key-blocks of 32; ``S = Q·Kᵀ`` and ``dP = dO·Vᵀ`` contract ``D = 512``,
  the softmax / ``dS`` runs per-row (``row = lane%16``) on the LDS-staged tiles,
  then ``dQ += dS·Kt`` contracts ``BN = 32`` (``Kt`` = host-transposed K-block).
* ``kernel_dsk_attn_bwd_dk`` / ``..._dv`` — grid ``(Sk/16, B*HQ)``, own
  ``dK[16,512]`` / ``dV[16,512]``. Loop query-blocks of 32; ``sᵀ = K·Qᵀ`` (and
  ``dpᵀ = V·dOᵀ`` for dK) give the transposed score tile, the softmax indexes
  ``lse / Dv`` by column ``m``, then ``dK += dsᵀ·Qt`` / ``dV += pᵀ·dOt`` (``Qt`` /
  ``dOt`` host-transposed). dK and dV stay separate kernels so each accumulator
  is ``[16,512] = 128 f32/lane`` (no spill); merging them via a D-split across
  waves (fwd round-4 pattern) is the next round.

Each WG writes a *per-query-head* ``dK/dV`` slice into ``[B, HQ, Sk, D]``; the
launcher sums over the head axis for MQA. Host-side ``Kt / Qt / dOt`` transposes
are cheap contiguous copies. Scope (``can_handle`` in the dispatcher): dense /
SWA / causal, bf16/fp16, ``D = 512``, MQA / MHA, optional sink, ``additive_mask
is None`` and ``hca_local_seqlen == 0``. HCA split-mask and CSA backward stay on
Triton (design §5.4 defers them to a later round).
"""

from __future__ import annotations

import functools

import torch

# isort: off
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl  # noqa: F401
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.attention.deepseek_attn_fwd_kernel import _get_compiled
from primus_turbo.flydsl.utils.attn_helper import LOG2E, NEG_INF, make_value_attrs

# isort: on

# A score at or below this is a NEG_INF sentinel, so its softmax weight is 0.
_MASK_THRESH = -1.0e29

BM = 16  # WG-owned rows (queries for dQ, keys for dK/dV)
BN = 32  # inner loop block + MFMA contraction for the accumulation GEMM


def _exp2s(x):
    return (x * fx.Float32(LOG2E)).exp2()


def _ceil(a, b):
    return ((a + b - 1) // b) * b


# ─────────────────────────────────────────────────────────────────────────────
# dQ kernel: WG owns BM=16 queries, loops key-blocks of BN=32
# ─────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dsk_attn_bwd_dq(
    D: int,
    scale: float,
    SWA_WINDOW: int = 0,
    USE_CAUSAL: bool = True,
    is_fp16: bool = False,
    waves_per_eu: int = 2,
):
    assert D == 512, "MFMA backward specialises the V4-Flash head_dim D = 512"
    elem_ty = fx.Float16 if is_fp16 else fx.BFloat16
    SCALE = float(scale)

    S_BYTES = BM * BN * 4
    DS_BYTES = BM * BN * 2
    OFF_DP = S_BYTES
    OFF_DS = OFF_DP + S_BYTES
    TOTAL = OFF_DS + DS_BYTES

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel_dsk_attn_bwd_dq(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        KT: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DVEC: fx.Tensor,
        DQ: fx.Tensor,
        seqlen_q_pad: fx.Int32,
        seqlen_k: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        nkv: fx.Int32,
    ):
        pid_m = fx.block_idx.x
        pid_bh = fx.block_idx.y
        lane = fx.thread_idx.x % fx.Int32(64)
        row = lane % fx.Int32(BM)

        bid = pid_bh // head_q
        qhid = pid_bh % head_q
        khid = arith.select(head_k == head_q, qhid, fx.Int32(0))
        kv_bh = bid * head_k + khid

        q_mtiles = seqlen_q_pad // fx.Int32(BM)
        k_ntiles = seqlen_k_pad // fx.Int32(BN)
        qo_tile = pid_bh * q_mtiles + pid_m
        m_glob = pid_m * fx.Int32(BM) + row

        gQ = fx.rocdl.make_buffer_tensor(Q)
        gK = fx.rocdl.make_buffer_tensor(K)
        gV = fx.rocdl.make_buffer_tensor(V)
        gKT = fx.rocdl.make_buffer_tensor(KT)
        gDO = fx.rocdl.make_buffer_tensor(DOUT)
        gL = fx.rocdl.make_buffer_tensor(LSE)
        gD = fx.rocdl.make_buffer_tensor(DVEC)
        gDQ = fx.rocdl.make_buffer_tensor(DQ)

        alloc = fx.SharedAllocator(static=False)
        alloc.allocate(TOTAL)
        base = fx.recast_iter(fx.Uint8, fx.get_dyn_shared())
        s_lds = fx.make_view(fx.recast_iter(fx.Float32, base), fx.make_layout((BM, BN), (BN, 1)))
        dp_lds = fx.make_view(
            fx.recast_iter(fx.Float32, fx.add_offset(base, fx.make_int_tuple(OFF_DP))),
            fx.make_layout((BM, BN), (BN, 1)),
        )
        ds_lds = fx.make_view(
            fx.recast_iter(elem_ty, fx.add_offset(base, fx.make_int_tuple(OFF_DS))),
            fx.make_layout((BM, BN), (BN, 1)),
        )

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, elem_ty))
        tmma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr = tmma.thr_slice(lane)

        cpA = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        cpB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        cpCf = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        ucA = fx.make_copy_atom(fx.UniversalCopy(128), elem_ty)
        tcA = fx.make_tiled_copy_A(cpA, tmma)
        tcB = fx.make_tiled_copy_B(cpB, tmma)
        tcCf = fx.make_tiled_copy_C(cpCf, tmma)
        tcPA = fx.make_tiled_copy_A(ucA, tmma)

        atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        regf = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

        def load_scalar(div, idx):
            fx.copy(atom_f32, fx.slice(div, (None, idx)), regf)
            return Vec(fx.memref_load_vec(regf))[0]

        ldiv = fx.logical_divide(gL, fx.make_layout(1, 1))
        ddiv = fx.logical_divide(gD, fx.make_layout(1, 1))
        idx_l = pid_bh * seqlen_q_pad + m_glob
        lse_m = load_scalar(ldiv, idx_l)
        d_m = load_scalar(ddiv, idx_l)

        bQ = fx.slice(fx.zipped_divide(gQ, fx.make_tile(BM, D)), (None, qo_tile))
        bDO = fx.slice(fx.zipped_divide(gDO, fx.make_tile(BM, D)), (None, qo_tile))
        fragQ = thr.make_fragment_A(bQ)
        fragDO = thr.make_fragment_A(bDO)
        fx.copy(cpA, tcA.get_slice(lane).partition_S(bQ), tcA.get_slice(lane).retile(fragQ))
        fx.copy(cpA, tcA.get_slice(lane).partition_S(bDO), tcA.get_slice(lane).retile(fragDO))

        bDQ = fx.slice(fx.zipped_divide(gDQ, fx.make_tile(BM, D)), (None, qo_tile))
        fragdQ = thr.make_fragment_C(bDQ)
        fragdQ.fill(0.0)

        sp_tile = fx.slice(fx.zipped_divide(s_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))
        dpp_tile = fx.slice(fx.zipped_divide(dp_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))
        ds_tile = fx.slice(fx.zipped_divide(ds_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))

        def process_block(n_blk):
            n0 = n_blk * fx.Int32(BN)
            kt = kv_bh * k_ntiles + n_blk

            bK = fx.slice(fx.zipped_divide(gK, fx.make_tile(BN, D)), (None, kt))
            fragK = thr.make_fragment_B(bK)
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bK), tcB.get_slice(lane).retile(fragK))
            fragS = thr.make_fragment_C(sp_tile)
            fragS.fill(0.0)
            fx.gemm(mma, fragS, fragQ, fragK, fragS)
            fx.copy(cpCf, tcCf.get_slice(lane).retile(fragS), tcCf.get_slice(lane).partition_D(sp_tile))

            bV = fx.slice(fx.zipped_divide(gV, fx.make_tile(BN, D)), (None, kt))
            fragV = thr.make_fragment_B(bV)
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bV), tcB.get_slice(lane).retile(fragV))
            fragdP = thr.make_fragment_C(dpp_tile)
            fragdP.fill(0.0)
            fx.gemm(mma, fragdP, fragDO, fragV, fragdP)
            fx.copy(cpCf, tcCf.get_slice(lane).retile(fragdP), tcCf.get_slice(lane).partition_D(dpp_tile))
            fx.gpu.barrier()

            for c in range_constexpr(BN):
                ki = n0 + fx.Int32(c)
                sv = ArithValue(fx.memref_load(s_lds, [row, fx.Int32(c)])) * fx.Float32(SCALE)
                if const_expr(SWA_WINDOW > 0):
                    lo = m_glob - fx.Int32(SWA_WINDOW - 1)
                    vis = (ki >= lo) & (ki <= m_glob) & (ki < seqlen_k)
                elif const_expr(USE_CAUSAL):
                    vis = (ki <= m_glob) & (ki < seqlen_k)
                else:
                    vis = ki < seqlen_k
                sv = arith.select(vis, sv, fx.Float32(NEG_INF))
                p = _exp2s(sv - lse_m)
                p = arith.select(sv > fx.Float32(_MASK_THRESH), p, fx.Float32(0.0))
                dpv = ArithValue(fx.memref_load(dp_lds, [row, fx.Int32(c)]))
                ds = p * (dpv - d_m)
                if const_expr(is_fp16):
                    fx.memref_store(fx.Float16(ds), ds_lds, [row, fx.Int32(c)])
                else:
                    fx.memref_store(fx.BFloat16(ds), ds_lds, [row, fx.Int32(c)])
            fx.gpu.barrier()

            kt_seg = kv_bh + n_blk * nkv
            bKt = fx.slice(fx.zipped_divide(gKT, fx.make_tile(D, BN)), (None, kt_seg))
            fragDS = thr.make_fragment_A(ds_tile)
            fragKt = thr.make_fragment_B(bKt)
            fx.copy(ucA, tcPA.get_slice(lane).partition_S(ds_tile), tcPA.get_slice(lane).retile(fragDS))
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bKt), tcB.get_slice(lane).retile(fragKt))
            fx.gemm(mma, fragdQ, fragDS, fragKt, fragdQ)
            fx.gpu.barrier()

        init = [fx.Float32(0.0)]
        for nb, st in range(fx.Index(0), fx.Index(k_ntiles), fx.Index(1), init=init):
            process_block(fx.Int32(nb))
            st = (yield init)

        vO = Vec(fx.memref_load_vec(fragdQ))
        fx.memref_store_vec(vO * fx.Float32(SCALE), fragdQ)
        cpO = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcO = fx.make_tiled_copy_C(cpO, tmma)
        fx.copy(cpO, tcO.get_slice(lane).retile(fragdQ), tcO.get_slice(lane).partition_D(bDQ))

    @flyc.jit
    def launch_dq(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        KT: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DVEC: fx.Tensor,
        DQ: fx.Tensor,
        seqlen_q_pad: fx.Int32,
        seqlen_k: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        nkv: fx.Int32,
        grid_m: fx.Int32,
        grid_bh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_dsk_attn_bwd_dq(
            Q, K, V, KT, DOUT, LSE, DVEC, DQ,
            seqlen_q_pad, seqlen_k, seqlen_k_pad, head_q, head_k, nkv,
            value_attrs=make_value_attrs(waves_per_eu, 0, "64,64"),
        ).launch(grid=(grid_m, grid_bh, 1), block=(64, 1, 1), stream=stream)

    return launch_dq


# ─────────────────────────────────────────────────────────────────────────────
# dK kernel: WG owns BM=16 keys, loops query-blocks of BN=32 (transposed softmax)
# ─────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dsk_attn_bwd_dk(
    D: int,
    scale: float,
    SWA_WINDOW: int = 0,
    USE_CAUSAL: bool = True,
    is_fp16: bool = False,
    waves_per_eu: int = 2,
):
    assert D == 512, "MFMA backward specialises the V4-Flash head_dim D = 512"
    elem_ty = fx.Float16 if is_fp16 else fx.BFloat16
    SCALE = float(scale)

    S_BYTES = BM * BN * 4
    DS_BYTES = BM * BN * 2
    OFF_DP = S_BYTES
    OFF_DS = OFF_DP + S_BYTES
    TOTAL = OFF_DS + DS_BYTES

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel_dsk_attn_bwd_dk(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        QT: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DVEC: fx.Tensor,
        DK: fx.Tensor,
        seqlen_q: fx.Int32,
        seqlen_q_pad: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        nq: fx.Int32,
    ):
        pid_n = fx.block_idx.x
        pid_bh = fx.block_idx.y
        lane = fx.thread_idx.x % fx.Int32(64)
        row = lane % fx.Int32(BM)

        bid = pid_bh // head_q
        qhid = pid_bh % head_q
        khid = arith.select(head_k == head_q, qhid, fx.Int32(0))
        kv_bh = bid * head_k + khid

        k_mtiles = seqlen_k_pad // fx.Int32(BM)
        q_ntiles = seqlen_q_pad // fx.Int32(BN)
        kv_tile = kv_bh * k_mtiles + pid_n     # read shared K/V (MQA broadcast)
        out_tile = pid_bh * k_mtiles + pid_n   # write per-query-head dK slice
        n_glob = pid_n * fx.Int32(BM) + row

        gQ = fx.rocdl.make_buffer_tensor(Q)
        gK = fx.rocdl.make_buffer_tensor(K)
        gV = fx.rocdl.make_buffer_tensor(V)
        gQT = fx.rocdl.make_buffer_tensor(QT)
        gDO = fx.rocdl.make_buffer_tensor(DOUT)
        gL = fx.rocdl.make_buffer_tensor(LSE)
        gD = fx.rocdl.make_buffer_tensor(DVEC)
        gDK = fx.rocdl.make_buffer_tensor(DK)

        alloc = fx.SharedAllocator(static=False)
        alloc.allocate(TOTAL)
        base = fx.recast_iter(fx.Uint8, fx.get_dyn_shared())
        s_lds = fx.make_view(fx.recast_iter(fx.Float32, base), fx.make_layout((BM, BN), (BN, 1)))
        dp_lds = fx.make_view(
            fx.recast_iter(fx.Float32, fx.add_offset(base, fx.make_int_tuple(OFF_DP))),
            fx.make_layout((BM, BN), (BN, 1)),
        )
        ds_lds = fx.make_view(
            fx.recast_iter(elem_ty, fx.add_offset(base, fx.make_int_tuple(OFF_DS))),
            fx.make_layout((BM, BN), (BN, 1)),
        )

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, elem_ty))
        tmma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr = tmma.thr_slice(lane)

        cpA = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        cpB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        cpCf = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        ucA = fx.make_copy_atom(fx.UniversalCopy(128), elem_ty)
        tcA = fx.make_tiled_copy_A(cpA, tmma)
        tcB = fx.make_tiled_copy_B(cpB, tmma)
        tcCf = fx.make_tiled_copy_C(cpCf, tmma)
        tcPA = fx.make_tiled_copy_A(ucA, tmma)

        atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        regf = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

        def load_scalar(div, idx):
            fx.copy(atom_f32, fx.slice(div, (None, idx)), regf)
            return Vec(fx.memref_load_vec(regf))[0]

        ldiv = fx.logical_divide(gL, fx.make_layout(1, 1))
        ddiv = fx.logical_divide(gD, fx.make_layout(1, 1))

        bK = fx.slice(fx.zipped_divide(gK, fx.make_tile(BM, D)), (None, kv_tile))
        bV = fx.slice(fx.zipped_divide(gV, fx.make_tile(BM, D)), (None, kv_tile))
        fragK = thr.make_fragment_A(bK)
        fragV = thr.make_fragment_A(bV)
        fx.copy(cpA, tcA.get_slice(lane).partition_S(bK), tcA.get_slice(lane).retile(fragK))
        fx.copy(cpA, tcA.get_slice(lane).partition_S(bV), tcA.get_slice(lane).retile(fragV))

        bDK = fx.slice(fx.zipped_divide(gDK, fx.make_tile(BM, D)), (None, out_tile))
        fragdK = thr.make_fragment_C(bDK)
        fragdK.fill(0.0)

        sp_tile = fx.slice(fx.zipped_divide(s_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))
        dpp_tile = fx.slice(fx.zipped_divide(dp_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))
        ds_tile = fx.slice(fx.zipped_divide(ds_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))

        def process_block(m_blk):
            m0 = m_blk * fx.Int32(BN)
            qt = pid_bh * q_ntiles + m_blk

            bQ = fx.slice(fx.zipped_divide(gQ, fx.make_tile(BN, D)), (None, qt))
            fragQ = thr.make_fragment_B(bQ)
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bQ), tcB.get_slice(lane).retile(fragQ))
            fragS = thr.make_fragment_C(sp_tile)
            fragS.fill(0.0)
            fx.gemm(mma, fragS, fragK, fragQ, fragS)
            fx.copy(cpCf, tcCf.get_slice(lane).retile(fragS), tcCf.get_slice(lane).partition_D(sp_tile))

            bDO = fx.slice(fx.zipped_divide(gDO, fx.make_tile(BN, D)), (None, qt))
            fragDO = thr.make_fragment_B(bDO)
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bDO), tcB.get_slice(lane).retile(fragDO))
            fragdP = thr.make_fragment_C(dpp_tile)
            fragdP.fill(0.0)
            fx.gemm(mma, fragdP, fragV, fragDO, fragdP)
            fx.copy(cpCf, tcCf.get_slice(lane).retile(fragdP), tcCf.get_slice(lane).partition_D(dpp_tile))
            fx.gpu.barrier()

            for c in range_constexpr(BN):
                mm = m0 + fx.Int32(c)
                idx_l = pid_bh * seqlen_q_pad + mm
                lse_m = load_scalar(ldiv, idx_l)
                d_m = load_scalar(ddiv, idx_l)
                sv = ArithValue(fx.memref_load(s_lds, [row, fx.Int32(c)])) * fx.Float32(SCALE)
                if const_expr(SWA_WINDOW > 0):
                    lo = mm - fx.Int32(SWA_WINDOW - 1)
                    vis = (n_glob >= lo) & (n_glob <= mm) & (mm < seqlen_q)
                elif const_expr(USE_CAUSAL):
                    vis = (n_glob <= mm) & (mm < seqlen_q)
                else:
                    vis = mm < seqlen_q
                sv = arith.select(vis, sv, fx.Float32(NEG_INF))
                p = _exp2s(sv - lse_m)
                p = arith.select(sv > fx.Float32(_MASK_THRESH), p, fx.Float32(0.0))
                dpv = ArithValue(fx.memref_load(dp_lds, [row, fx.Int32(c)]))
                ds = p * (dpv - d_m)
                if const_expr(is_fp16):
                    fx.memref_store(fx.Float16(ds), ds_lds, [row, fx.Int32(c)])
                else:
                    fx.memref_store(fx.BFloat16(ds), ds_lds, [row, fx.Int32(c)])
            fx.gpu.barrier()

            qt_seg = pid_bh + m_blk * nq
            bQt = fx.slice(fx.zipped_divide(gQT, fx.make_tile(D, BN)), (None, qt_seg))
            fragDS = thr.make_fragment_A(ds_tile)
            fragQt = thr.make_fragment_B(bQt)
            fx.copy(ucA, tcPA.get_slice(lane).partition_S(ds_tile), tcPA.get_slice(lane).retile(fragDS))
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bQt), tcB.get_slice(lane).retile(fragQt))
            fx.gemm(mma, fragdK, fragDS, fragQt, fragdK)
            fx.gpu.barrier()

        init = [fx.Float32(0.0)]
        for mb, st in range(fx.Index(0), fx.Index(q_ntiles), fx.Index(1), init=init):
            process_block(fx.Int32(mb))
            st = (yield init)

        vO = Vec(fx.memref_load_vec(fragdK))
        fx.memref_store_vec(vO * fx.Float32(SCALE), fragdK)
        cpO = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcO = fx.make_tiled_copy_C(cpO, tmma)
        fx.copy(cpO, tcO.get_slice(lane).retile(fragdK), tcO.get_slice(lane).partition_D(bDK))

    @flyc.jit
    def launch_dk(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        QT: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DVEC: fx.Tensor,
        DK: fx.Tensor,
        seqlen_q: fx.Int32,
        seqlen_q_pad: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        nq: fx.Int32,
        grid_n: fx.Int32,
        grid_bh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_dsk_attn_bwd_dk(
            Q, K, V, QT, DOUT, LSE, DVEC, DK,
            seqlen_q, seqlen_q_pad, seqlen_k_pad, head_q, head_k, nq,
            value_attrs=make_value_attrs(waves_per_eu, 0, "64,64"),
        ).launch(grid=(grid_n, grid_bh, 1), block=(64, 1, 1), stream=stream)

    return launch_dk


# ─────────────────────────────────────────────────────────────────────────────
# dV kernel: WG owns BM=16 keys, loops query-blocks of BN=32 (transposed softmax)
# ─────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dsk_attn_bwd_dv(
    D: int,
    scale: float,
    SWA_WINDOW: int = 0,
    USE_CAUSAL: bool = True,
    is_fp16: bool = False,
    waves_per_eu: int = 2,
):
    assert D == 512, "MFMA backward specialises the V4-Flash head_dim D = 512"
    elem_ty = fx.Float16 if is_fp16 else fx.BFloat16
    SCALE = float(scale)

    S_BYTES = BM * BN * 4
    P_BYTES = BM * BN * 2
    OFF_P = S_BYTES
    TOTAL = OFF_P + P_BYTES

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel_dsk_attn_bwd_dv(
        Q: fx.Tensor,
        K: fx.Tensor,
        DOUTT: fx.Tensor,
        LSE: fx.Tensor,
        DV: fx.Tensor,
        seqlen_q: fx.Int32,
        seqlen_q_pad: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        nq: fx.Int32,
    ):
        pid_n = fx.block_idx.x
        pid_bh = fx.block_idx.y
        lane = fx.thread_idx.x % fx.Int32(64)
        row = lane % fx.Int32(BM)

        bid = pid_bh // head_q
        qhid = pid_bh % head_q
        khid = arith.select(head_k == head_q, qhid, fx.Int32(0))
        kv_bh = bid * head_k + khid

        k_mtiles = seqlen_k_pad // fx.Int32(BM)
        q_ntiles = seqlen_q_pad // fx.Int32(BN)
        kv_tile = kv_bh * k_mtiles + pid_n
        out_tile = pid_bh * k_mtiles + pid_n
        n_glob = pid_n * fx.Int32(BM) + row

        gQ = fx.rocdl.make_buffer_tensor(Q)
        gK = fx.rocdl.make_buffer_tensor(K)
        gDOT = fx.rocdl.make_buffer_tensor(DOUTT)
        gL = fx.rocdl.make_buffer_tensor(LSE)
        gDV = fx.rocdl.make_buffer_tensor(DV)

        alloc = fx.SharedAllocator(static=False)
        alloc.allocate(TOTAL)
        base = fx.recast_iter(fx.Uint8, fx.get_dyn_shared())
        s_lds = fx.make_view(fx.recast_iter(fx.Float32, base), fx.make_layout((BM, BN), (BN, 1)))
        p_lds = fx.make_view(
            fx.recast_iter(elem_ty, fx.add_offset(base, fx.make_int_tuple(OFF_P))),
            fx.make_layout((BM, BN), (BN, 1)),
        )

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, elem_ty))
        tmma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr = tmma.thr_slice(lane)

        cpA = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        cpB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        cpCf = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        ucA = fx.make_copy_atom(fx.UniversalCopy(128), elem_ty)
        tcA = fx.make_tiled_copy_A(cpA, tmma)
        tcB = fx.make_tiled_copy_B(cpB, tmma)
        tcCf = fx.make_tiled_copy_C(cpCf, tmma)
        tcPA = fx.make_tiled_copy_A(ucA, tmma)

        atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        regf = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

        def load_scalar(div, idx):
            fx.copy(atom_f32, fx.slice(div, (None, idx)), regf)
            return Vec(fx.memref_load_vec(regf))[0]

        ldiv = fx.logical_divide(gL, fx.make_layout(1, 1))

        bK = fx.slice(fx.zipped_divide(gK, fx.make_tile(BM, D)), (None, kv_tile))
        fragK = thr.make_fragment_A(bK)
        fx.copy(cpA, tcA.get_slice(lane).partition_S(bK), tcA.get_slice(lane).retile(fragK))

        bDV = fx.slice(fx.zipped_divide(gDV, fx.make_tile(BM, D)), (None, out_tile))
        fragdV = thr.make_fragment_C(bDV)
        fragdV.fill(0.0)

        sp_tile = fx.slice(fx.zipped_divide(s_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))
        p_tile = fx.slice(fx.zipped_divide(p_lds, fx.make_tile(BM, BN)), (None, fx.Int32(0)))

        def process_block(m_blk):
            m0 = m_blk * fx.Int32(BN)
            qt = pid_bh * q_ntiles + m_blk

            bQ = fx.slice(fx.zipped_divide(gQ, fx.make_tile(BN, D)), (None, qt))
            fragQ = thr.make_fragment_B(bQ)
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bQ), tcB.get_slice(lane).retile(fragQ))
            fragS = thr.make_fragment_C(sp_tile)
            fragS.fill(0.0)
            fx.gemm(mma, fragS, fragK, fragQ, fragS)
            fx.copy(cpCf, tcCf.get_slice(lane).retile(fragS), tcCf.get_slice(lane).partition_D(sp_tile))
            fx.gpu.barrier()

            for c in range_constexpr(BN):
                mm = m0 + fx.Int32(c)
                idx_l = pid_bh * seqlen_q_pad + mm
                lse_m = load_scalar(ldiv, idx_l)
                sv = ArithValue(fx.memref_load(s_lds, [row, fx.Int32(c)])) * fx.Float32(SCALE)
                if const_expr(SWA_WINDOW > 0):
                    lo = mm - fx.Int32(SWA_WINDOW - 1)
                    vis = (n_glob >= lo) & (n_glob <= mm) & (mm < seqlen_q)
                elif const_expr(USE_CAUSAL):
                    vis = (n_glob <= mm) & (mm < seqlen_q)
                else:
                    vis = mm < seqlen_q
                sv = arith.select(vis, sv, fx.Float32(NEG_INF))
                p = _exp2s(sv - lse_m)
                p = arith.select(sv > fx.Float32(_MASK_THRESH), p, fx.Float32(0.0))
                if const_expr(is_fp16):
                    fx.memref_store(fx.Float16(p), p_lds, [row, fx.Int32(c)])
                else:
                    fx.memref_store(fx.BFloat16(p), p_lds, [row, fx.Int32(c)])
            fx.gpu.barrier()

            qt_seg = pid_bh + m_blk * nq
            bDOt = fx.slice(fx.zipped_divide(gDOT, fx.make_tile(D, BN)), (None, qt_seg))
            fragP = thr.make_fragment_A(p_tile)
            fragDOt = thr.make_fragment_B(bDOt)
            fx.copy(ucA, tcPA.get_slice(lane).partition_S(p_tile), tcPA.get_slice(lane).retile(fragP))
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bDOt), tcB.get_slice(lane).retile(fragDOt))
            fx.gemm(mma, fragdV, fragP, fragDOt, fragdV)
            fx.gpu.barrier()

        init = [fx.Float32(0.0)]
        for mb, st in range(fx.Index(0), fx.Index(q_ntiles), fx.Index(1), init=init):
            process_block(fx.Int32(mb))
            st = (yield init)

        cpO = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcO = fx.make_tiled_copy_C(cpO, tmma)
        fx.copy(cpO, tcO.get_slice(lane).retile(fragdV), tcO.get_slice(lane).partition_D(bDV))

    @flyc.jit
    def launch_dv(
        Q: fx.Tensor,
        K: fx.Tensor,
        DOUTT: fx.Tensor,
        LSE: fx.Tensor,
        DV: fx.Tensor,
        seqlen_q: fx.Int32,
        seqlen_q_pad: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        nq: fx.Int32,
        grid_n: fx.Int32,
        grid_bh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_dsk_attn_bwd_dv(
            Q, K, DOUTT, LSE, DV,
            seqlen_q, seqlen_q_pad, seqlen_k_pad, head_q, head_k, nq,
            value_attrs=make_value_attrs(waves_per_eu, 0, "64,64"),
        ).launch(grid=(grid_n, grid_bh, 1), block=(64, 1, 1), stream=stream)

    return launch_dv


def hca_attention_bwd_flydsl_kernel(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]
    v: torch.Tensor,  # [B, K_H, Sk, D]
    out: torch.Tensor,  # [B, H, Sq, D] (FWD output)
    dout: torch.Tensor,  # [B, H, Sq, D]
    lse: torch.Tensor,  # [B, H, Sq] fp32
    *,
    sink=None,  # [H] or None
    swa_window: int,
    additive_mask=None,
    scale: float,
    hca_local_seqlen: int = 0,
):
    """FlyDSL dense / SWA attention backward; returns ``(dq, dk, dv, dsink)``.

    Mirrors the Triton ``_launch_hca_attention_bwd`` contract. Scope (gated by
    the dispatcher ``can_handle``): dense / SWA / causal, ``D = 512``, MQA / MHA,
    optional sink, ``additive_mask is None`` and ``hca_local_seqlen == 0`` (HCA
    split-mask and CSA backward stay on Triton).
    """
    if additive_mask is not None or int(hca_local_seqlen) != 0:
        raise ValueError(
            "FlyDSL attention backward handles only the dense / SWA path; HCA "
            "split-mask and additive bias stay on Triton."
        )
    B, HQ, Sq, D = q.shape
    HK, Sk = k.shape[1], k.shape[2]
    if D != 512:
        raise ValueError(f"FlyDSL attention backward specialises D = 512; got {D}.")
    if q.dtype not in (torch.bfloat16, torch.float16) or not (q.dtype == k.dtype == v.dtype):
        raise ValueError("FlyDSL attention backward requires matching bf16/fp16 q/k/v.")

    is_fp16 = q.dtype == torch.float16
    use_causal = swa_window <= 0
    swa_cx = int(swa_window) if swa_window > 0 else 0

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()
    lse = lse.contiguous()

    Sq_pad = _ceil(Sq, BN)
    Sk_pad = _ceil(Sk, BN)

    def pad_seq(t, target):
        pad = target - t.shape[2]
        return torch.nn.functional.pad(t, (0, 0, 0, pad)) if pad else t

    qp = pad_seq(q, Sq_pad)
    dop = pad_seq(dout, Sq_pad)
    kp = pad_seq(k, Sk_pad)
    vp = pad_seq(v, Sk_pad)

    # Host-side transposes feeding the accumulation GEMMs (B operands).
    ktp = kp.transpose(-1, -2).contiguous()   # [B, HK, D, Sk_pad]   (dQ)
    qtp = qp.transpose(-1, -2).contiguous()   # [B, HQ, D, Sq_pad]   (dK)
    dotp = dop.transpose(-1, -2).contiguous()  # [B, HQ, D, Sq_pad]  (dV)

    # Dv_i = rowsum(dO * O) and dsink are cheap per-row reductions (torch).
    d_vec = (dout.to(torch.float32) * out.to(torch.float32)).sum(-1)  # [B, HQ, Sq]
    lse_p = torch.nn.functional.pad(lse, (0, Sq_pad - Sq)) if Sq_pad != Sq else lse
    dvec_p = torch.nn.functional.pad(d_vec, (0, Sq_pad - Sq)) if Sq_pad != Sq else d_vec

    dq_f32 = torch.zeros((B, HQ, Sq_pad, D), device=q.device, dtype=torch.float32)
    dk_head = torch.zeros((B, HQ, Sk_pad, D), device=q.device, dtype=torch.float32)
    dv_head = torch.zeros((B, HQ, Sk_pad, D), device=q.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    launch_dq = _compile_dsk_attn_bwd_dq(
        D=D, scale=float(scale), SWA_WINDOW=swa_cx, USE_CAUSAL=use_causal, is_fp16=is_fp16
    )
    launch_dk = _compile_dsk_attn_bwd_dk(
        D=D, scale=float(scale), SWA_WINDOW=swa_cx, USE_CAUSAL=use_causal, is_fp16=is_fp16
    )
    launch_dv = _compile_dsk_attn_bwd_dv(
        D=D, scale=float(scale), SWA_WINDOW=swa_cx, USE_CAUSAL=use_causal, is_fp16=is_fp16
    )

    dq_args = (
        qp.reshape(-1, D), kp.reshape(-1, D), vp.reshape(-1, D), ktp.reshape(-1, Sk_pad),
        dop.reshape(-1, D), lse_p.reshape(-1), dvec_p.reshape(-1), dq_f32.reshape(-1, D),
        Sq_pad, Sk, Sk_pad, HQ, HK, B * HK,
        Sq_pad // BM, B * HQ, stream,
    )
    _get_compiled(launch_dq, dq_args)(*dq_args)

    dk_args = (
        qp.reshape(-1, D), kp.reshape(-1, D), vp.reshape(-1, D), qtp.reshape(-1, Sq_pad),
        dop.reshape(-1, D), lse_p.reshape(-1), dvec_p.reshape(-1), dk_head.reshape(-1, D),
        Sq, Sq_pad, Sk_pad, HQ, HK, B * HQ,
        Sk_pad // BM, B * HQ, stream,
    )
    _get_compiled(launch_dk, dk_args)(*dk_args)

    dv_args = (
        qp.reshape(-1, D), kp.reshape(-1, D), dotp.reshape(-1, Sq_pad),
        lse_p.reshape(-1), dv_head.reshape(-1, D),
        Sq, Sq_pad, Sk_pad, HQ, HK, B * HQ,
        Sk_pad // BM, B * HQ, stream,
    )
    _get_compiled(launch_dv, dv_args)(*dv_args)

    dq = dq_f32[:, :, :Sq, :].to(q.dtype)
    dk_head = dk_head[:, :, :Sk, :]
    dv_head = dv_head[:, :, :Sk, :]
    if HK == 1:
        dk = dk_head.sum(dim=1, keepdim=True).to(k.dtype)
        dv = dv_head.sum(dim=1, keepdim=True).to(v.dtype)
    else:
        dk = dk_head.to(k.dtype)
        dv = dv_head.to(v.dtype)

    if sink is not None:
        sink_f = sink.to(torch.float32).reshape(1, HQ, 1)
        p_sink = torch.exp(sink_f - lse)  # [B, HQ, Sq]
        dsink = -(p_sink * d_vec).sum(dim=(0, 2)).to(sink.dtype)
    else:
        dsink = None

    return dq, dk, dv, dsink


__all__ = ["hca_attention_bwd_flydsl_kernel"]
