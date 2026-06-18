###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 dense / HCA attention FORWARD kernel (FlyDSL, design §4).

FlashAttention-style online softmax for the V4-Flash envelope (``head_dim
D = 512``, ``SWA = 128``, MQA / MHA, optional per-head sink, HCA split-mask pool
branch). Returns ``(out, lse)`` matching the Triton ``_launch_hca_attention_fwd``
contract so the dispatcher can A/B the two backends on identical inputs.

MFMA block kernel (design §7.3 rounds 1–4):

* grid = ``(cdiv(Sq, BLOCK_M=16), B * HQ)``; a WG = ``WAVE_N=4`` waves (block 256).
* QK is an ``mfma_f32_16x16x32`` (bf16 / fp16) matmul over ``D``; online softmax
  (SWA / causal + boundary mask, finite ``NEG_INF`` sentinel, ``exp2`` / ``log2e``
  prescale); PV is an ``mfma`` of ``P[16, BLOCK_N]`` × ``Vt`` accumulated into the
  running ``O_acc`` C-fragment.
* HCA split-mask pool branch (additive pool bias), per-head sink virtual column,
  ``O = O_acc * rcp(l_i)`` normalise + ``LSE = m_i + ln(l_i)``.

ROUND-3 moved ``O_acc`` into an in-register f32 C-fragment (``fragO``), rescaled
per block by a column-stride-0 broadcast factor fragment (LDS ~64 KB → ~3 KB).

ROUND-4 (design §4.9.3, D-split-for-PV) splits ``D=512`` across ``WAVE_N=4``
waves so each wave owns only the ``[16, D/WAVE_N=128]`` segment of ``O_acc`` (32
AGPR/lane instead of 128 — the round-3 AGPR limiter that pinned occupancy at ~2
waves/SIMD). The QK contraction dim is ``D``, so each wave produces a *partial*
``S[16,32]`` over its 128-D slice; the ``WAVE_N`` partials are summed across waves
through a ``[WAVE_N,16,32]`` LDS buffer (the softmax then runs identically on each
wave). PV is per-wave (``P[16,32]`` × ``Vt_seg[128,32]`` → ``O_seg[16,128]``);
each wave normalises and stores its own 128 output columns, LSE (row scalar,
identical on every wave) is written idempotently. LDS double-buffer,
``ds_read_*_tr`` in-kernel transpose and per-shape tile sweep are later rounds;
V is still pre-transposed on host (``Vt[..., D, Sk]``).
"""

import functools

import torch

# isort: off
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl  # noqa: F401
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.attn_helper import (
    LOG2E,
    NEG_INF,
    make_value_attrs,
)

# isort: on

# Masked-key threshold: a score at or below this is a NEG_INF sentinel, so its
# softmax weight must be forced to 0 (design §6 NaN/0 guard).
_MASK_THRESH = -1.0e29

BLOCK_M = 16
BLOCK_N = 32
WAVE_N = 4  # D-split-for-PV wave count (design §4.9.3)


def _exp2s(x):
    return (x * fx.Float32(LOG2E)).exp2()


def _log2(x):
    from flydsl._mlir.dialects import math as _math

    return _math.log2(x)


def _wave_sum(val_f32):
    """Sum ``val_f32`` across all 64 lanes of the wave (butterfly XOR shuffle);
    every lane returns the full reduction (design §4.4 peer reduction). Retained
    for the FlyDSL backward kernel's lane-wise D-split QK / dP reductions."""
    s = val_f32
    for off in (1, 2, 4, 8, 16, 32):
        s = s + ArithValue(s.shuffle_xor(off, 64))
    return s


@functools.lru_cache(maxsize=256)
def _compile_dsk_attn_fwd(
    D: int,
    scale: float,
    SWA_WINDOW: int = 0,
    HAS_SINK: bool = False,
    HAS_ADD_MASK: bool = False,
    HCA_LOCAL_SEQLEN: int = 0,
    USE_CAUSAL: bool = True,
    is_fp16: bool = False,
    waves_per_eu: int = 0,
):
    """Build & cache the constexpr-specialised forward launcher.

    ``scale`` is folded in as a constexpr (the V4 head_dim scale is fixed per
    shape). BLOCK_M=16 query rows; D split across WAVE_N waves; QK / PV use the
    16x16x32 MFMA atom.
    """
    assert D == 512, "specialises the V4-Flash head_dim D = 512"
    elem_ty = fx.Float16 if is_fp16 else fx.BFloat16
    SCALE = float(scale)
    NK = BLOCK_N
    DPW = D // WAVE_N  # output columns owned by each wave

    SP_BYTES = BLOCK_M * NK * 4
    P_BYTES = BLOCK_M * NK * 2
    CORR_BYTES = BLOCK_M * 4
    OFF_P = WAVE_N * SP_BYTES
    OFF_CORR = OFF_P + P_BYTES
    TOTAL = OFF_CORR + CORR_BYTES

    @flyc.kernel(known_block_size=[WAVE_N * 64, 1, 1])
    def kernel_dsk_attn_fwd(
        Q: fx.Tensor,    # [B*HQ*Sq, D]
        K: fx.Tensor,    # [B*HK*Sk_pad, D]
        Vt: fx.Tensor,   # [B*HK*D, Sk_pad]
        OUT: fx.Tensor,  # [B*HQ*Sq, D]
        LSE: fx.Tensor,  # flat [B*HQ*Sq]
        SINK: fx.Tensor,  # [HQ]
        ADD_MASK: fx.Tensor,  # flat [Sq*P]
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,      # real Sk (for masking)
        seqlen_k_pad: fx.Int32,  # padded Sk (for tiling)
        head_q: fx.Int32,
        head_k: fx.Int32,
        pool_size: fx.Int32,     # P (additive-mask width)
        nkv: fx.Int32,           # B*HK
        qm_tot: fx.Int32,        # B*HQ*(Sq/16): col-major D-tile stride for Q/OUT
        km_tot: fx.Int32,        # B*HK*(Sk_pad/32): col-major D-tile stride for K
    ):
        pid_m = fx.block_idx.x
        pid_bh = fx.block_idx.y
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(64)
        wave_n = tid // fx.Int32(64)  # this wave's D-segment id (§4.9.3 D-split)
        row = lane % fx.Int32(BLOCK_M)

        bid = pid_bh // head_q
        qhid = pid_bh % head_q
        khid = arith.select(head_k == head_q, qhid, fx.Int32(0))  # MQA broadcast
        kv_bh = bid * head_k + khid

        q_mtiles = seqlen_q // fx.Int32(BLOCK_M)
        k_ntiles = seqlen_k_pad // fx.Int32(NK)
        qo_tile = pid_bh * q_mtiles + pid_m
        qi = pid_m * fx.Int32(BLOCK_M) + row  # this lane's query row

        # D-segment tile indices (zipped_divide flattens the tile grid col-major).
        q_seg = qo_tile + wave_n * qm_tot
        vt_row_tot = nkv * fx.Int32(WAVE_N)

        gQ = fx.rocdl.make_buffer_tensor(Q)
        gK = fx.rocdl.make_buffer_tensor(K)
        gVt = fx.rocdl.make_buffer_tensor(Vt)
        gO = fx.rocdl.make_buffer_tensor(OUT)
        gL = fx.rocdl.make_buffer_tensor(LSE)

        # ── LDS: WAVE_N partial-S slots [16,32] f32 | P[16,32] elem | corr[16] ──
        alloc = fx.SharedAllocator(static=False)
        alloc.allocate(TOTAL)
        base = fx.recast_iter(fx.Uint8, fx.get_dyn_shared())
        sp_all = fx.make_view(fx.recast_iter(fx.Float32, base),
                              fx.make_layout((WAVE_N, BLOCK_M, NK), (BLOCK_M * NK, NK, 1)))
        sp_me = fx.slice(sp_all, (wave_n, None, None))  # this wave's partial-S slot
        p_lds = fx.make_view(
            fx.recast_iter(elem_ty, fx.add_offset(base, fx.make_int_tuple(OFF_P))),
            fx.make_layout((BLOCK_M, NK), (NK, 1)))
        corr_ptr = fx.recast_iter(fx.Float32, fx.add_offset(base, fx.make_int_tuple(OFF_CORR)))
        corr_lds = fx.make_view(corr_ptr, fx.make_layout((BLOCK_M,), (1,)))
        corr_bc = fx.make_view(corr_ptr, fx.make_layout((BLOCK_M, DPW), (1, 0)))  # col stride 0

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, elem_ty))
        tmma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr = tmma.thr_slice(lane)

        bO = fx.slice(fx.zipped_divide(gO, fx.make_tile(BLOCK_M, DPW)), (None, q_seg))
        bC = fx.slice(fx.zipped_divide(corr_bc, fx.make_tile(BLOCK_M, DPW)), (None, fx.Int32(0)))
        cpCf = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        tcCf = fx.make_tiled_copy_C(cpCf, tmma)

        # ── preload this wave's Q D-segment [16,128] -> fragQ ──
        bQ = fx.slice(fx.zipped_divide(gQ, fx.make_tile(BLOCK_M, DPW)), (None, q_seg))
        fragQ = thr.make_fragment_A(bQ)
        cpA = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        tcA = fx.make_tiled_copy_A(cpA, tmma)
        fx.copy(cpA, tcA.get_slice(lane).partition_S(bQ), tcA.get_slice(lane).retile(fragQ))

        # ── O_acc for this wave's D-segment (f32 acc, 32 AGPR/lane) ──
        fragO = thr.make_fragment_C(bO)
        fragO.fill(0.0)

        gM = fx.rocdl.make_buffer_tensor(ADD_MASK)
        mdiv = fx.logical_divide(gM, fx.make_layout(1, 1))
        atom_m = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        regm = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

        def full_score(jc, n0, is_pool):
            # cross-wave reduce: sum the WAVE_N partial-S contributions, then mask
            ki = n0 + fx.Int32(jc)
            acc = fx.Float32(0.0)
            for w in range_constexpr(WAVE_N):
                acc = acc + ArithValue(fx.memref_load(sp_all, [fx.Int32(w), row, fx.Int32(jc)]))
            sv = acc * fx.Float32(SCALE)
            if const_expr(is_pool):
                if const_expr(HAS_ADD_MASK):
                    jp = ki - fx.Int32(HCA_LOCAL_SEQLEN)
                    fx.copy(atom_m, fx.slice(mdiv, (None, qi * pool_size + jp)), regm)
                    sv = sv + ArithValue(Vec(fx.memref_load_vec(regm))[0])
                vis = ki < seqlen_k
            else:
                if const_expr(SWA_WINDOW > 0):
                    lo = qi - fx.Int32(SWA_WINDOW - 1)
                    vis = (ki >= lo) & (ki <= qi) & (ki < seqlen_k)
                elif const_expr(USE_CAUSAL):
                    vis = (ki <= qi) & (ki < seqlen_k)
                else:
                    vis = ki < seqlen_k
            return arith.select(vis, sv, fx.Float32(NEG_INF))

        def rescale_fragO(factor):
            # broadcast per-row factor through corr_lds + stride-0 view -> factor
            # fragment, then elementwise-multiply the in-register O_seg.
            fx.memref_store(fx.Float32(factor), corr_lds, [row])
            fx.gpu.barrier()
            fragF = thr.make_fragment_C(bO)
            fx.copy(cpCf, tcCf.get_slice(lane).partition_S(bC), tcCf.get_slice(lane).retile(fragF))
            vO = Vec(fx.memref_load_vec(fragO))
            vF = Vec(fx.memref_load_vec(fragF))
            fx.memref_store_vec(vO * vF, fragO)
            fx.gpu.barrier()

        def process_block(n_blk, m_i, l_i, is_pool):
            n0 = n_blk * fx.Int32(NK)

            # ── QK partial over this wave's D-segment -> sp_me ──
            kt = kv_bh * k_ntiles + n_blk
            k_seg = kt + wave_n * km_tot
            bK = fx.slice(fx.zipped_divide(gK, fx.make_tile(NK, DPW)), (None, k_seg))
            fragK = thr.make_fragment_B(bK)
            fragS = thr.make_fragment_C(sp_me)
            fragS.fill(0.0)
            cpB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
            tcB = fx.make_tiled_copy_B(cpB, tmma)
            fx.copy(cpB, tcB.get_slice(lane).partition_S(bK), tcB.get_slice(lane).retile(fragK))
            fx.gemm(mma, fragS, fragQ, fragK, fragS)
            bSP = fx.slice(fx.zipped_divide(sp_me, fx.make_tile(BLOCK_M, NK)), (None, fx.Int32(0)))
            cpC = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
            tcC = fx.make_tiled_copy_C(cpC, tmma)
            fx.copy(cpC, tcC.get_slice(lane).retile(fragS), tcC.get_slice(lane).partition_D(bSP))
            fx.gpu.barrier()

            # ── online softmax on the cross-wave-reduced full S (row = lane%16) ──
            block_max = fx.Float32(NEG_INF)
            for j in range_constexpr(NK):
                sv = full_score(j, n0, is_pool)
                block_max = arith.select(sv > block_max, sv, block_max)
            m_new = m_i.maximumf(block_max)
            corr = _exp2s(m_i - m_new)
            block_l = fx.Float32(0.0)
            for j in range_constexpr(NK):
                sv = full_score(j, n0, is_pool)
                p = _exp2s(sv - m_new)
                p = arith.select(sv > fx.Float32(_MASK_THRESH), p, fx.Float32(0.0))
                block_l = block_l + p
                if const_expr(is_fp16):
                    fx.memref_store(fx.Float16(p), p_lds, [row, fx.Int32(j)])
                else:
                    fx.memref_store(fx.BFloat16(p), p_lds, [row, fx.Int32(j)])
            l_i = corr * l_i + block_l
            m_i = m_new
            fx.gpu.barrier()

            # ── rescale O_seg by corr (in registers), then accumulate P@V ──
            rescale_fragO(corr)

            # ── PV MFMA: P[16,NK] x Vt_seg[DPW,NK] accumulated into fragO ──
            vt_seg = (kv_bh * fx.Int32(WAVE_N) + wave_n) + n_blk * vt_row_tot
            bVt = fx.slice(fx.zipped_divide(gVt, fx.make_tile(DPW, NK)), (None, vt_seg))
            bP = fx.slice(fx.zipped_divide(p_lds, fx.make_tile(BLOCK_M, NK)), (None, fx.Int32(0)))
            fragP = thr.make_fragment_A(bP)
            fragV = thr.make_fragment_B(bVt)
            ucA = fx.make_copy_atom(fx.UniversalCopy(128), elem_ty)
            tcPA = fx.make_tiled_copy_A(ucA, tmma)
            fx.copy(ucA, tcPA.get_slice(lane).partition_S(bP), tcPA.get_slice(lane).retile(fragP))
            cpVB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
            tcVB = fx.make_tiled_copy_B(cpVB, tmma)
            fx.copy(cpVB, tcVB.get_slice(lane).partition_S(bVt), tcVB.get_slice(lane).retile(fragV))
            fx.gemm(mma, fragO, fragP, fragV, fragO)
            fx.gpu.barrier()
            return m_i, l_i

        # ── local KV-block range for this query tile (design §4.5) ──────────
        if const_expr(HCA_LOCAL_SEQLEN > 0):
            local_end = fx.Int32(HCA_LOCAL_SEQLEN)
        else:
            local_end = seqlen_k
        qi_max = pid_m * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)
        up = arith.select(qi_max + fx.Int32(1) < local_end, qi_max + fx.Int32(1), local_end)
        blk_end = (up + fx.Int32(NK - 1)) // fx.Int32(NK)
        if const_expr(SWA_WINDOW > 0):
            qi_min = pid_m * fx.Int32(BLOCK_M)
            lo0 = qi_min - fx.Int32(SWA_WINDOW - 1)
            lo0 = arith.select(lo0 > fx.Int32(0), lo0, fx.Int32(0))
            blk_start = lo0 // fx.Int32(NK)
        else:
            blk_start = fx.Int32(0)

        m_i = fx.Float32(NEG_INF)
        l_i = fx.Float32(0.0)
        init = [m_i, l_i]
        for nb, st in range(fx.Index(blk_start), fx.Index(blk_end), fx.Index(1), init=init):
            m_i, l_i = process_block(fx.Int32(nb), st[0], st[1], False)
            st = (yield [m_i, l_i])
        m_i, l_i = st[0], st[1]

        # ── HCA split-mask pool branch (design §4.6) ────────────────────────
        if const_expr(HCA_LOCAL_SEQLEN > 0):
            pool_blk_start = fx.Int32(HCA_LOCAL_SEQLEN) // fx.Int32(NK)
            init2 = [m_i, l_i]
            for nb, st in range(fx.Index(pool_blk_start), fx.Index(k_ntiles), fx.Index(1), init=init2):
                m_i, l_i = process_block(fx.Int32(nb), st[0], st[1], True)
                st = (yield [m_i, l_i])
            m_i, l_i = st[0], st[1]

        # ── sink virtual column (design §4.8) ───────────────────────────────
        if const_expr(HAS_SINK):
            gS = fx.rocdl.make_buffer_tensor(SINK)
            sdiv = fx.logical_divide(gS, fx.make_layout(1, 1))
            atom_s = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
            regs = fx.make_rmem_tensor(fx.make_layout(1, 1), elem_ty)
            fx.copy(atom_s, fx.slice(sdiv, (None, qhid)), regs)
            sink_h = ArithValue(Vec(fx.memref_load_vec(regs))[0].to(fx.Float32))
            m_new = m_i.maximumf(sink_h)
            corr = _exp2s(m_i - m_new)
            w = _exp2s(sink_h - m_new)
            l_i = corr * l_i + w
            rescale_fragO(corr)
            m_i = m_new

        # ── normalise (O_seg *= 1/l) + cast f32 acc -> elem + store O tile ──
        inv = arith.select(l_i > fx.Float32(0.0), fx.Float32(1.0) / l_i, fx.Float32(0.0))
        rescale_fragO(inv)
        fragOut = fx.make_fragment_like(fragO, dtype=elem_ty)
        fx.memref_store_vec(Vec(fx.memref_load_vec(fragO)).to(elem_ty), fragOut)
        cpO = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
        tcO = fx.make_tiled_copy_C(cpO, tmma)
        fx.copy(cpO, tcO.get_slice(lane).retile(fragOut), tcO.get_slice(lane).partition_D(bO))

        # LSE = m_i + ln(l_i); identical on every wave (same softmax), written
        # idempotently by all waves.
        ln_l = arith.select(l_i > fx.Float32(0.0),
                            _log2(l_i) * fx.Float32(1.0 / LOG2E), fx.Float32(0.0))
        lse_val = fx.Float32(m_i + ln_l)
        ldiv = fx.logical_divide(gL, fx.make_layout(1, 1))
        atom_l = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        regl = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        fx.memref_store_vec(Vec.filled(1, lse_val, fx.Float32), regl)
        fx.copy(atom_l, regl, fx.slice(ldiv, (None, pid_bh * seqlen_q + qi)))

    @flyc.jit
    def launch_dsk_attn_fwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        Vt: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        SINK: fx.Tensor,
        ADD_MASK: fx.Tensor,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        seqlen_k_pad: fx.Int32,
        head_q: fx.Int32,
        head_k: fx.Int32,
        pool_size: fx.Int32,
        nkv: fx.Int32,
        qm_tot: fx.Int32,
        km_tot: fx.Int32,
        grid_m: fx.Int32,
        grid_bh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_dsk_attn_fwd(
            Q, K, Vt, OUT, LSE, SINK, ADD_MASK,
            seqlen_q, seqlen_k, seqlen_k_pad, head_q, head_k, pool_size, nkv,
            qm_tot, km_tot,
            value_attrs=make_value_attrs(waves_per_eu, 0, f"{WAVE_N * 64},{WAVE_N * 64}"),
        ).launch(grid=(grid_m, grid_bh, 1), block=(WAVE_N * 64, 1, 1), stream=stream)

    return launch_dsk_attn_fwd


_COMPILED_ATTN_CACHE: dict = {}


def _get_compiled(launch, args):
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_ATTN_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_ATTN_CACHE[key] = cached
    return cached


def hca_attention_fwd_flydsl_kernel(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]   K_H in {1, H}
    v: torch.Tensor,  # [B, K_H, Sk, D]
    *,
    sink=None,  # [H] or None
    swa_window: int,
    additive_mask=None,  # [Sq, P] pool-only bias when HCA split, else None
    scale: float,
    hca_local_seqlen: int = 0,
):
    """FlyDSL dense / HCA attention forward; returns ``(out, lse)``.

    Mirrors the Triton ``_launch_hca_attention_fwd`` contract: ``out`` matches
    ``q.dtype``, ``lse`` is fp32 ``[B, H, Sq]``.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"hca_attention FlyDSL forward expects rank-4 q/k/v (got "
            f"{q.dim()}/{k.dim()}/{v.dim()})"
        )
    B, HQ, Sq, D = q.shape
    Bk, HK, Sk, Dk = k.shape
    if (Bk, Sk, Dk) != (B, Sk, D) or tuple(v.shape) != (Bk, HK, Sk, D):
        raise ValueError(
            f"hca_attention FlyDSL shape mismatch: q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if HK != 1 and HK != HQ:
        raise ValueError(f"hca_attention FlyDSL requires K_H in {{1, {HQ}}}; got {HK}.")
    if D != 512:
        raise ValueError(f"hca_attention FlyDSL specialises D = 512; got {D}.")
    if Sq % BLOCK_M != 0:
        raise ValueError(f"hca_attention FlyDSL requires Sq % {BLOCK_M} == 0; got {Sq}.")
    if q.dtype not in (torch.bfloat16, torch.float16) or not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            "hca_attention FlyDSL requires bf16/fp16 with q.dtype == k.dtype == v.dtype "
            f"(got {q.dtype}/{k.dtype}/{v.dtype})."
        )

    has_sink = sink is not None
    has_add_mask = additive_mask is not None
    hca_local_seqlen = int(hca_local_seqlen)
    use_causal = (not has_add_mask) and (swa_window <= 0)
    swa_window_cx = int(swa_window) if swa_window > 0 else 0
    is_fp16 = q.dtype == torch.float16

    Sk_pad = ((Sk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if Sk_pad != Sk:
        k = torch.nn.functional.pad(k, (0, 0, 0, Sk_pad - Sk))
        v = torch.nn.functional.pad(v, (0, 0, 0, Sk_pad - Sk))
    vt = v.transpose(-1, -2).contiguous()  # [B, HK, D, Sk_pad]
    out = torch.empty_like(q)
    lse = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)

    P = additive_mask.shape[1] if has_add_mask else 0
    sink_t = sink.contiguous().reshape(-1).to(q.dtype) if has_sink else q.reshape(-1)
    mask_t = (
        additive_mask.contiguous().reshape(-1).to(torch.float32)
        if has_add_mask
        else lse.reshape(-1)
    )

    qm_tot = B * HQ * (Sq // BLOCK_M)
    km_tot = B * HK * (Sk_pad // BLOCK_N)

    launch = _compile_dsk_attn_fwd(
        D=D,
        scale=float(scale),
        SWA_WINDOW=swa_window_cx,
        HAS_SINK=has_sink,
        HAS_ADD_MASK=has_add_mask,
        HCA_LOCAL_SEQLEN=hca_local_seqlen,
        USE_CAUSAL=use_causal,
        is_fp16=is_fp16,
    )

    args = (
        q.reshape(-1, D),
        k.reshape(-1, D),
        vt.reshape(-1, Sk_pad),
        out.reshape(-1, D),
        lse.reshape(-1),
        sink_t,
        mask_t,
        Sq,
        Sk,
        Sk_pad,
        HQ,
        HK,
        P,
        B * HK,
        qm_tot,
        km_tot,
        Sq // BLOCK_M,
        B * HQ,
        torch.cuda.current_stream(),
    )
    _get_compiled(launch, args)(*args)
    return out, lse


__all__ = [
    "hca_attention_fwd_flydsl_kernel",
    "_compile_dsk_attn_fwd",
]
