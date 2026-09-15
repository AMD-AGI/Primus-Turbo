###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""MXFP8 grouped GEMM followed by GLU / dGLU dual quantization."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_glu_kernel import GradProbsPartialSpec
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import MB
from primus_turbo.flydsl.quantization.mxfp8_quant_flydsl import (
    _CM,
    _GQD_BK,
    _GQD_BM,
    _GQD_NTH,
    _OOB,
    _col_store_res,
    _decode_pid,
    _e8_or_one,
    _ep,
    _load_i32_at,
    _sat,
    fp8_params,
    grouped_qdual_grid,
)
from primus_turbo.flydsl.utils.gemm_epilogue_helper import (
    _check_activation,
    _check_clamp_limit,
    _glu_act,
    _glu_act_grad,
    _glu_clamp,
    _glu_kept,
)
from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource
from primus_turbo.flydsl.utils.prims import _row16_sum_f32, ceildiv

# Both epilogues are 2-phase: a tuned BLOCK_M=256 GEMM, then the quant kernel. Folding the
# epilogue into the GEMM costs more than the round trip it saves, because the band it has
# to keep live pushes the tile past 128 VGPRs and into scratch.
_PHASE2_QDUAL_CACHE: dict = {}
_PHASE2_DGLU_QDUAL_CACHE: dict = {}
_DGLU_DACT_WS: dict = {}


def _compile_grouped_glu_qdual(
    N,
    G,
    out_fp8="e4m3",
    pad_extra=4,
    col_data_band=True,
    col_scale_band=True,
    activation="silu",
    clamp_limit=None,
):
    """Compile grouped GLU dual quantization."""
    elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    N_pad = ((N + 127) // 128) * 128
    assert N % 32 == 0
    BMv, BKv, NTHv = _GQD_BM, _GQD_BK, _GQD_NTH
    assert BMv * BKv == 16 * NTHv and 128 % BMv == 0 and BMv % 32 == 0 and BKv % 32 == 0
    HALF = NTHv // 2
    PAD_L = BKv + pad_extra
    LMASK = N_pad != N
    NBK = N_pad // BKv
    NKB = BKv // 32
    NMB = BMv // 32
    DWPC = BMv // 4
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv
    assert DWPC % 4 == 0
    SCALEN_ROW = N_pad // 32
    COL_DATA_BAND = col_data_band
    COL_SCALE_BAND = col_scale_band
    XN = 2 * N

    @flyc.kernel(known_block_size=[NTHv, 1, 1])
    def kern(
        X: fx.Tensor,
        P: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        GO: fx.Tensor,
        m_pad_row: fx.Int32,
        m_pad_col: fx.Int32,
        scalen_col: fx.Int32,
    ):
        I32 = fx.Int32
        BF = elt.ir_type
        F32 = fx.Float32
        z = I32(0)
        IRI = fx.Int32.ir_type

        @fx.struct
        class Smem:
            tile: fx.Array[elt, BMv * PAD_L, 16]
            ldsc: fx.Array[fx.Int32, LDSC_DW, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        tile = sm.tile
        ldsc = sm.ldsc
        t = fx.thread_idx.x
        pid = fx.block_idx.x
        br, bkc = _decode_pid(pid, I32(NBK), None, False)
        base_m = br * I32(BMv)

        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        go_vals = [_load_i32_at(go_div, 2 * g) for g in range_constexpr(G + 1)]
        found = z
        go_orig_g = z
        go_orig_g1 = z
        go_row_g = z
        go_col_g = z
        acc_row = z
        acc_col = z
        for g in range_constexpr(G):
            prev = go_vals[g]
            nxt = go_vals[g + 1]
            ln = nxt - prev
            lrow = ((ln + I32(63)) // I32(64)) * I32(64)
            lcol = ((ln + I32(127)) // I32(128)) * I32(128)
            inq = (base_m >= acc_col) & (base_m < acc_col + lcol)
            go_col_g = arith.select(inq, acc_col, go_col_g)
            go_orig_g = arith.select(inq, prev, go_orig_g)
            go_orig_g1 = arith.select(inq, nxt, go_orig_g1)
            go_row_g = arith.select(inq, acc_row, go_row_g)
            found = arith.select(inq, I32(1), found)
            acc_row = acc_row + lrow
            acc_col = acc_col + lcol
        isreal = found == I32(1)
        mrel = base_m - go_col_g
        in_rebase = arith.select(isreal, go_orig_g + mrel, z)
        rowbase_out = arith.select(isreal, go_row_g + mrel, z)
        real_end = arith.select(isreal, go_col_g + (go_orig_g1 - go_orig_g), base_m)
        in_end = arith.select(isreal, go_orig_g1, z)

        rx = make_row_band_resource(bo.extract_base_index(X), in_rebase, in_end, I32(XN), 2)
        rp = bo.create_buffer_resource(P, max_size=False, num_records_bytes=go_vals[G] * I32(4))
        VPR8 = BKv // 8
        ITERS8 = BMv * BKv // 8 // NTHv
        for ls in range_constexpr(ITERS8):
            lin = t + I32(ls * NTHv)
            lr = lin // I32(VPR8)
            cv = (lin - lr * I32(VPR8)) * I32(8)
            pbase = lr * I32(PAD_L) + cv
            fcol = bkc * I32(BKv) + cv
            ioff = lr * I32(XN) + fcol
            uoff = ioff + I32(N)
            if LMASK:
                valid_col = fcol < I32(N)
                ioff = valid_col.select(ioff, I32(_OOB))
                uoff = valid_col.select(uoff, I32(_OOB))
            v = bo.buffer_load(rx, ioff, vec_width=8, dtype=BF)
            uv = bo.buffer_load(rx, uoff, vec_width=8, dtype=BF)
            vf, uf = Vec(v).to(F32), Vec(uv).to(F32)
            prob = F32(bo.buffer_load(rp, in_rebase + lr, vec_width=1, dtype=_T.f32))
            vals = []
            for i in range_constexpr(8):
                gc, uc, _, _ = _glu_clamp(vf[i], uf[i], clamp_limit)
                if const_expr(activation == "silu"):
                    neg_log2e = gc * F32(-1.4426950408889634)
                    ex = F32(rocdl.exp2(_T.f32, _raw(neg_log2e)))
                    denom = F32(1.0) + ex
                    gate = F32(arith.divf(_raw(gc), _raw(denom)))
                else:
                    gate = _glu_act(gc, activation)
                val = F32(arith.mulf(arith.mulf(_raw(gate), _raw(uc)), _raw(prob)))
                vals.append(val.to(fx.BFloat16))
            v = Vec.from_elements(vals, fx.BFloat16).ir_value()
            p = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase))
            fx.make_view(p, fx.make_layout(8, 1)).store(Vec(v))
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )
        rocdl.s_barrier()
        half = t // I32(HALF)
        lt = t - half * I32(HALF)
        if half == z:
            row = lt // I32(NKB)
            kb = lt - row * I32(NKB)
            loff = row * I32(PAD_L) + kb * I32(32)
            chunks = []
            for i in range_constexpr(8):
                p = fx.add_offset(tile.ptr, fx.make_int_tuple(loff + I32(4 * i)))
                chunks.append(Vec(fx.make_view(p, fx.make_layout(4, 1)).load()).to(F32))
            amax = F32(0.0)
            for i in range_constexpr(8):
                a = fm.absf(chunks[i]).reduce("max")
                amax = (amax > a).select(amax, a)
            grow = base_m + row
            row_ok = grow < real_end
            gcol0 = bkc * I32(BKv) + kb * I32(32)
            ep = _ep(amax, va, ep_sub)
            inv = F32(1.0) / fm.exp2(ep.to(F32))
            rqr = make_row_band_resource(bo.extract_base_index(Qr), rowbase_out, m_pad_row, I32(N_pad), 1)
            words = []
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                words.append(word)
            row_byte0 = row * I32(N_pad) + gcol0
            for v in range_constexpr(2):
                off = row_ok.select(row_byte0 + I32(16 * v), I32(_OOB))
                v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                bo.buffer_store(v4.ir_value(), rqr, off, cache_modifier=_CM, offset_is_bytes=True)
            kcol = bkc * I32(BKv // 32) + kb
            rasp = make_row_band_resource(
                bo.extract_base_index(ASp), rowbase_out, m_pad_row, I32(SCALEN_ROW), 1
            )
            e8b = _e8_or_one(amax, ep)
            bo.buffer_store(
                ArithValue(e8b & I32(255)).trunci(_T.i8),
                rasp,
                row * I32(SCALEN_ROW) + kcol,
                mask=row_ok,
                offset_is_bytes=True,
                cache_modifier=_CM,
            )
        else:
            c = lt // I32(NMB)
            mblk = lt - c * I32(NMB)
            base = fx.add_offset(tile.ptr, fx.make_int_tuple(c + I32(mblk * 32) * I32(PAD_L)))
            cv2 = Vec(fx.make_view(base, fx.make_layout(32, PAD_L)).load()).to(F32)
            ca = fm.absf(cv2).reduce("max")
            camax = (F32(0.0) > ca).select(F32(0.0), ca)
            cep = _ep(camax, va, ep_sub)
            cinv = F32(1.0) / fm.exp2(cep.to(F32))
            cq = cv2 * cinv
            cwords = []
            for wi in range_constexpr(8):
                word = I32(cvt(IRI, _sat(cq[4 * wi + 0], sat_bnd), _sat(cq[4 * wi + 1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(cq[4 * wi + 2], sat_bnd), _sat(cq[4 * wi + 3], sat_bnd), word, 1))
                cwords.append(word)
            csbase = c * I32(DWPC) + mblk * I32(8)
            for v in range_constexpr(2):
                sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(csbase + I32(4 * v)))
                fx.make_view(sp, fx.make_layout(4, 1)).store(
                    Vec.from_elements(cwords[4 * v : 4 * v + 4], fx.Int32)
                )
            gc = bkc * I32(BKv) + c
            mcol = br * I32(BMv // 32) + mblk
            ce8b = _e8_or_one(camax, cep)
            catsp, csoff = _col_store_res(
                COL_SCALE_BAND,
                bo.extract_base_index(AtSp),
                bkc * I32(BKv),
                gc,
                c,
                I32(N),
                scalen_col,
                mcol,
            )
            bo.buffer_store(
                ArithValue(ce8b & I32(255)).trunci(_T.i8),
                catsp,
                csoff,
                offset_is_bytes=True,
                cache_modifier=_CM,
            )
        _llvm.inline_asm(
            res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
        )
        rocdl.s_barrier()
        for it in range_constexpr(CW_ITERS_V):
            lo = (t + I32(it * NTHv)) * I32(4)
            cc = lo // I32(DWPC)
            dwi0 = lo - cc * I32(DWPC)
            rp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(lo))
            v4 = Vec(fx.make_view(rp, fx.make_layout(4, 1)).load())
            gc = bkc * I32(BKv) + cc
            raqd, off = _col_store_res(
                COL_DATA_BAND,
                bo.extract_base_index(AtQd),
                bkc * I32(BKv),
                gc,
                cc,
                I32(N),
                m_pad_col,
                base_m + dwi0 * I32(4),
            )
            bo.buffer_store(
                v4.ir_value(),
                raqd,
                off,
                cache_modifier=_CM,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        X: fx.Tensor,
        P: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        GO: fx.Tensor,
        m_pad_row: fx.Int32,
        m_pad_col: fx.Int32,
        scalen_col: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, P, Qr, ASp, AtQd, AtSp, GO, m_pad_row, m_pad_col, scalen_col).launch(
            grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream
        )

    return launch


def epi_glu_quant_flydsl_kernel(
    l1,
    probs,
    group_offs,
    row_out,
    row_sc,
    col_out,
    col_sc,
    *,
    activation="silu",
    clamp_limit=None,
):
    """Apply GLU and write row-wise and column-wise MXFP8 outputs."""
    assert l1.ndim == 2 and l1.is_contiguous() and l1.dtype == torch.bfloat16
    M, N2 = l1.shape
    assert N2 % 2 == 0
    N = N2 // 2
    G = int(group_offs.numel()) - 1
    n_pad = (N + 127) // 128 * 128
    assert probs.shape == (M,) and probs.dtype == torch.float32
    assert row_out.shape[1] == n_pad
    assert col_out.shape[0] == N
    m_pad_row, m_pad_col = int(row_out.shape[0]), int(col_out.shape[1])
    assert row_sc.shape == (m_pad_row, n_pad // 32)
    assert col_sc.shape == (N, m_pad_col // 32)

    grid = grouped_qdual_grid(m_pad_col, N)
    scalen_col = m_pad_col // 32
    out_fp8 = "e5m2" if row_out.dtype == torch.float8_e5m2 else "e4m3"
    col_data_band = _GQD_BK * m_pad_col < (1 << 31)
    col_scale_band = _GQD_BK * scalen_col < (1 << 31)
    key = (N, G, out_fp8, activation, clamp_limit, col_data_band, col_scale_band)
    stream = torch.cuda.current_stream()
    go = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)
    qr = row_out.view(torch.uint8)
    qrs = row_sc.view(torch.uint8)
    qc = col_out.view(torch.uint8)
    qcs = col_sc.view(torch.uint8)
    args = (
        l1,
        probs,
        qr,
        qrs,
        qc,
        qcs,
        go,
        m_pad_row,
        m_pad_col,
        scalen_col,
        grid,
        stream,
    )
    comp = _PHASE2_QDUAL_CACHE.get(key)
    if comp is None:
        launch = _compile_grouped_glu_qdual(
            N,
            G,
            out_fp8=out_fp8,
            col_data_band=col_data_band,
            col_scale_band=col_scale_band,
            activation=activation,
            clamp_limit=clamp_limit,
        )
        comp = flyc.compile(launch, *args)
        _PHASE2_QDUAL_CACHE[key] = comp
    comp(*args)
    return row_out, row_sc, col_out, col_sc


def glu_epi_quant_supported(K: int, I: int, out_dtype=torch.bfloat16) -> bool:
    """Return whether the kernel supports the given padded shape."""
    if out_dtype != torch.bfloat16:
        return False
    if K % 128 or K < 256:
        return False
    return I > 0 and I % 64 == 0 and I % MB == 0


def _act_operand_geometry(I: int, M_pad_row: int, M_pad_col: int):
    """The row/col operand shapes ``grouped_quant_mxfp8_raw`` would have produced."""
    return (M_pad_row, ceildiv(I, 128) * 128), (I, M_pad_col)


def grouped_gemm_mxfp8_epi_glu_quant_flydsl_kernel(
    a: "torch.Tensor",
    a_scale: "torch.Tensor",
    b: "torch.Tensor",
    b_scale: "torch.Tensor",
    probs: "torch.Tensor",
    group_offs: "torch.Tensor",  # padded (row-64) read offsets [G+1]
    intermediate_out: "torch.Tensor",
    row_out: "torch.Tensor",
    row_sc: "torch.Tensor",
    col_out: "torch.Tensor",
    col_sc: "torch.Tensor",
    N: int,
    K: int,
    *,
    group_offs_out: "torch.Tensor",  # tight write offsets [G+1]
    activation: str = "silu",
    clamp_limit: "float | None" = None,
    out_dtype=torch.bfloat16,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Grouped GEMM, then GLU and dual quantization epilogue."""
    _check_activation(activation)
    _check_clamp_limit(activation, clamp_limit)
    assert a.ndim == 2 and b.ndim == 3
    assert N % 2 == 0, f"fc1 width must be even (gate||up), got {N}"
    I = N // 2
    assert glu_epi_quant_supported(K, I, out_dtype), (
        f"the fused GLU quant epilogue does not cover I={I} K={K} out_dtype={out_dtype}"
    )
    G = int(b.shape[0])
    assert a.shape[1] == K and b.shape[1] == N and b.shape[2] == K, (
        f"want a [*, {K}] and b [{G}, {N}, {K}], got {tuple(a.shape)} {tuple(b.shape)}"
    )
    l1 = intermediate_out
    M_total = int(l1.shape[0])
    assert l1.shape == (M_total, N) and l1.dtype == out_dtype
    assert probs.ndim == 1 and probs.shape[0] == M_total and probs.dtype == torch.float32
    row_shape, col_shape = _act_operand_geometry(I, int(row_out.shape[0]), int(col_out.shape[1]))
    assert tuple(row_out.shape) == row_shape, f"row_out must be {row_shape}, got {tuple(row_out.shape)}"
    assert tuple(col_out.shape) == col_shape, f"col_out must be {col_shape}, got {tuple(col_out.shape)}"
    assert tuple(row_sc.shape) == (row_shape[0], row_shape[1] // MB)
    assert tuple(col_sc.shape) == (col_shape[0], col_shape[1] // MB)

    # stage 1: grouped GEMM
    grouped_gemm_mxfp8_flydsl_kernel(
        a,
        a_scale,
        b,
        b_scale,
        group_offs,
        N,
        K,
        group_offs_out=group_offs_out,
        out_dtype=out_dtype,
        out=l1,
    )
    # stage 2: GLU and dual quantization epilogue
    epi_glu_quant_flydsl_kernel(
        l1,
        probs,
        group_offs_out,
        row_out,
        row_sc,
        col_out,
        col_sc,
        activation=activation,
        clamp_limit=clamp_limit,
    )
    return l1, row_out, row_sc, col_out, col_sc


def _compile_grouped_dglu_qdual(
    I,
    G,
    out_fp8="e4m3",
    pad_extra=4,
    col_data_band=True,
    col_scale_band=True,
    activation="silu",
    clamp_limit=None,
):
    """Compile grouped dGLU dual quantization over the I-wide dact tile."""
    elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    I_pad = ((I + 127) // 128) * 128
    I2 = 2 * I
    I2_pad = ((I2 + 127) // 128) * 128
    assert I % 32 == 0
    BMv, BKv, NTHv = _GQD_BM, _GQD_BK, _GQD_NTH
    assert BMv * BKv == 16 * NTHv and 128 % BMv == 0 and BMv % 32 == 0 and BKv % 32 == 0
    HALF = NTHv // 2
    PAD_L = BKv + pad_extra
    LMASK = I_pad != I
    NBK = I_pad // BKv
    NKB = BKv // 32
    NMB = BMv // 32
    DWPC = BMv // 4
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv
    assert DWPC % 4 == 0
    SCALEN_ROW = I2_pad // 32
    COL_DATA_BAND = col_data_band
    COL_SCALE_BAND = col_scale_band
    XN = I2

    @flyc.kernel(known_block_size=[NTHv, 1, 1])
    def kern(
        DACT: fx.Tensor,
        L1: fx.Tensor,
        P: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        GO: fx.Tensor,
        GP: fx.Tensor,
        m_pad_row: fx.Int32,
        m_pad_col: fx.Int32,
        scalen_col: fx.Int32,
        gp_stride: fx.Int32,
    ):
        I32 = fx.Int32
        BF = elt.ir_type
        F32 = fx.Float32
        z = I32(0)
        IRI = fx.Int32.ir_type

        @fx.struct
        class Smem:
            tile_g: fx.Array[elt, BMv * PAD_L, 16]
            tile_u: fx.Array[elt, BMv * PAD_L, 16]
            ldsc: fx.Array[fx.Int32, LDSC_DW, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        tile_g = sm.tile_g
        tile_u = sm.tile_u
        ldsc = sm.ldsc
        t = fx.thread_idx.x
        pid = fx.block_idx.x
        br, bkc = _decode_pid(pid, I32(NBK), None, False)
        base_m = br * I32(BMv)

        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        go_vals = [_load_i32_at(go_div, 2 * g) for g in range_constexpr(G + 1)]
        found = z
        go_orig_g = z
        go_orig_g1 = z
        go_row_g = z
        go_col_g = z
        acc_row = z
        acc_col = z
        for g in range_constexpr(G):
            prev = go_vals[g]
            nxt = go_vals[g + 1]
            ln = nxt - prev
            lrow = ((ln + I32(63)) // I32(64)) * I32(64)
            lcol = ((ln + I32(127)) // I32(128)) * I32(128)
            inq = (base_m >= acc_col) & (base_m < acc_col + lcol)
            go_col_g = arith.select(inq, acc_col, go_col_g)
            go_orig_g = arith.select(inq, prev, go_orig_g)
            go_orig_g1 = arith.select(inq, nxt, go_orig_g1)
            go_row_g = arith.select(inq, acc_row, go_row_g)
            found = arith.select(inq, I32(1), found)
            acc_row = acc_row + lrow
            acc_col = acc_col + lcol
        isreal = found == I32(1)
        mrel = base_m - go_col_g
        in_rebase = arith.select(isreal, go_orig_g + mrel, z)
        rowbase_out = arith.select(isreal, go_row_g + mrel, z)
        real_end = arith.select(isreal, go_col_g + (go_orig_g1 - go_orig_g), base_m)
        in_end = arith.select(isreal, go_orig_g1, z)

        rd = make_row_band_resource(bo.extract_base_index(DACT), in_rebase, in_end, I32(I), 2)
        rx = make_row_band_resource(bo.extract_base_index(L1), in_rebase, in_end, I32(XN), 2)
        rp = bo.create_buffer_resource(P, max_size=False, num_records_bytes=go_vals[G] * I32(4))
        rgp = bo.create_buffer_resource(GP, max_size=False, num_records_bytes=I32(NBK) * gp_stride * I32(4))
        VPR8 = BKv // 8
        ITERS8 = BMv * BKv // 8 // NTHv
        for ls in range_constexpr(ITERS8):
            lin = t + I32(ls * NTHv)
            lr = lin // I32(VPR8)
            cv = (lin - lr * I32(VPR8)) * I32(8)
            pbase = lr * I32(PAD_L) + cv
            fcol = bkc * I32(BKv) + cv
            ioff = lr * I32(XN) + fcol
            uoff = ioff + I32(I)
            doff = lr * I32(I) + fcol
            if LMASK:
                valid_col = fcol < I32(I)
                ioff = valid_col.select(ioff, I32(_OOB))
                uoff = valid_col.select(uoff, I32(_OOB))
                doff = valid_col.select(doff, I32(_OOB))
            gv = bo.buffer_load(rx, ioff, vec_width=8, dtype=BF)
            uv = bo.buffer_load(rx, uoff, vec_width=8, dtype=BF)
            dv = bo.buffer_load(rd, doff, vec_width=8, dtype=BF)
            gf, uf, df = Vec(gv).to(F32), Vec(uv).to(F32), Vec(dv).to(F32)
            prob = F32(bo.buffer_load(rp, in_rebase + lr, vec_width=1, dtype=_T.f32))
            dgs, dus = [], []
            gp = F32(0.0)
            for i in range_constexpr(8):
                gc, uc, kept_g, kept_u = _glu_clamp(gf[i], uf[i], clamp_limit)
                act, dact_dg = _glu_act_grad(gc, activation)
                d_raw = df[i]
                gp = gp + d_raw * act * uc
                d = d_raw * prob
                dgs.append(_glu_kept(d * uc * dact_dg, kept_g).to(fx.BFloat16))
                dus.append(_glu_kept(d * act, kept_u).to(fx.BFloat16))
            pg = fx.add_offset(tile_g.ptr, fx.make_int_tuple(pbase))
            pu = fx.add_offset(tile_u.ptr, fx.make_int_tuple(pbase))
            fx.make_view(pg, fx.make_layout(8, 1)).store(Vec.from_elements(dgs, fx.BFloat16))
            fx.make_view(pu, fx.make_layout(8, 1)).store(Vec.from_elements(dus, fx.BFloat16))
            chunk = lin - lr * I32(VPR8)
            tok = in_rebase + lr
            gp = _row16_sum_f32(gp)
            bo.buffer_store(
                gp,
                rgp,
                (bkc * gp_stride + tok) * I32(4),
                mask=isreal & (tok < in_end) & (chunk == I32(15)),
                offset_is_bytes=True,
            )
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )
        rocdl.s_barrier()

        for hi in range_constexpr(2):
            tile = tile_g if hi == 0 else tile_u
            feat_off_py = 0 if hi == 0 else I
            feat_off = I32(feat_off_py)
            half = t // I32(HALF)
            lt = t - half * I32(HALF)
            if half == z:
                row = lt // I32(NKB)
                kb = lt - row * I32(NKB)
                loff = row * I32(PAD_L) + kb * I32(32)
                chunks = []
                for i in range_constexpr(8):
                    p = fx.add_offset(tile.ptr, fx.make_int_tuple(loff + I32(4 * i)))
                    chunks.append(Vec(fx.make_view(p, fx.make_layout(4, 1)).load()).to(F32))
                amax = F32(0.0)
                for i in range_constexpr(8):
                    a = fm.absf(chunks[i]).reduce("max")
                    amax = (amax > a).select(amax, a)
                grow = base_m + row
                gcol0 = bkc * I32(BKv) + kb * I32(32)
                in_i = gcol0 < I32(I)
                row_ok = (grow < real_end) & in_i
                ep = _ep(amax, va, ep_sub)
                inv = F32(1.0) / fm.exp2(ep.to(F32))
                rqr = make_row_band_resource(
                    bo.extract_base_index(Qr), rowbase_out, m_pad_row, I32(I2_pad), 1
                )
                words = []
                for wi in range_constexpr(8):
                    qf = chunks[wi] * inv
                    word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                    word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                    words.append(word)
                row_byte0 = row * I32(I2_pad) + feat_off + gcol0
                for v in range_constexpr(2):
                    off = row_ok.select(row_byte0 + I32(16 * v), I32(_OOB))
                    v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                    bo.buffer_store(v4.ir_value(), rqr, off, cache_modifier=_CM, offset_is_bytes=True)
                kcol = I32(feat_off_py // 32) + bkc * I32(BKv // 32) + kb
                rasp = make_row_band_resource(
                    bo.extract_base_index(ASp), rowbase_out, m_pad_row, I32(SCALEN_ROW), 1
                )
                e8b = _e8_or_one(amax, ep)
                bo.buffer_store(
                    ArithValue(e8b & I32(255)).trunci(_T.i8),
                    rasp,
                    row * I32(SCALEN_ROW) + kcol,
                    mask=row_ok,
                    offset_is_bytes=True,
                    cache_modifier=_CM,
                )
            else:
                c = lt // I32(NMB)
                mblk = lt - c * I32(NMB)
                base = fx.add_offset(tile.ptr, fx.make_int_tuple(c + I32(mblk * 32) * I32(PAD_L)))
                cv2 = Vec(fx.make_view(base, fx.make_layout(32, PAD_L)).load()).to(F32)
                ca = fm.absf(cv2).reduce("max")
                camax = (F32(0.0) > ca).select(F32(0.0), ca)
                cep = _ep(camax, va, ep_sub)
                cinv = F32(1.0) / fm.exp2(cep.to(F32))
                cq = cv2 * cinv
                cwords = []
                for wi in range_constexpr(8):
                    word = I32(cvt(IRI, _sat(cq[4 * wi + 0], sat_bnd), _sat(cq[4 * wi + 1], sat_bnd), z, 0))
                    word = I32(
                        cvt(IRI, _sat(cq[4 * wi + 2], sat_bnd), _sat(cq[4 * wi + 3], sat_bnd), word, 1)
                    )
                    cwords.append(word)
                csbase = c * I32(DWPC) + mblk * I32(8)
                for v in range_constexpr(2):
                    sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(csbase + I32(4 * v)))
                    fx.make_view(sp, fx.make_layout(4, 1)).store(
                        Vec.from_elements(cwords[4 * v : 4 * v + 4], fx.Int32)
                    )
                gc = feat_off + bkc * I32(BKv) + c
                in_i = (bkc * I32(BKv) + c) < I32(I)
                mcol = br * I32(BMv // 32) + mblk
                ce8b = _e8_or_one(camax, cep)
                catsp, csoff = _col_store_res(
                    COL_SCALE_BAND,
                    bo.extract_base_index(AtSp),
                    feat_off + bkc * I32(BKv),
                    gc,
                    c,
                    I32(I2),
                    scalen_col,
                    mcol,
                )
                bo.buffer_store(
                    ArithValue(ce8b & I32(255)).trunci(_T.i8),
                    catsp,
                    in_i.select(csoff, I32(_OOB)),
                    offset_is_bytes=True,
                    cache_modifier=_CM,
                )
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
            rocdl.s_barrier()
            for it in range_constexpr(CW_ITERS_V):
                lo = (t + I32(it * NTHv)) * I32(4)
                cc = lo // I32(DWPC)
                dwi0 = lo - cc * I32(DWPC)
                rp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(lo))
                v4 = Vec(fx.make_view(rp, fx.make_layout(4, 1)).load())
                gc = feat_off + bkc * I32(BKv) + cc
                in_i = (bkc * I32(BKv) + cc) < I32(I)
                raqd, off = _col_store_res(
                    COL_DATA_BAND,
                    bo.extract_base_index(AtQd),
                    feat_off + bkc * I32(BKv),
                    gc,
                    cc,
                    I32(I2),
                    m_pad_col,
                    base_m + dwi0 * I32(4),
                )
                bo.buffer_store(
                    v4.ir_value(),
                    raqd,
                    in_i.select(off, I32(_OOB)),
                    cache_modifier=_CM,
                    offset_is_bytes=True,
                )
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
            rocdl.s_barrier()

    @flyc.jit
    def launch(
        DACT: fx.Tensor,
        L1: fx.Tensor,
        P: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        GO: fx.Tensor,
        GP: fx.Tensor,
        m_pad_row: fx.Int32,
        m_pad_col: fx.Int32,
        scalen_col: fx.Int32,
        gp_stride: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kern(DACT, L1, P, Qr, ASp, AtQd, AtSp, GO, GP, m_pad_row, m_pad_col, scalen_col, gp_stride).launch(
            grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream
        )

    return launch


def epi_dglu_quant_flydsl_kernel(
    dact,
    l1,
    probs,
    group_offs,
    grad_probs_partial,
    row_out,
    row_sc,
    col_out,
    col_sc,
    *,
    activation="silu",
    clamp_limit=None,
):
    """Apply dGLU and write row-wise and column-wise MXFP8 ``grad_l1`` plus grad_probs partials."""
    assert dact.ndim == 2 and l1.ndim == 2 and dact.is_contiguous() and l1.is_contiguous()
    assert dact.dtype == torch.bfloat16 and l1.dtype == torch.bfloat16
    M, I = dact.shape
    assert l1.shape == (M, 2 * I)
    G = int(group_offs.numel()) - 1
    n_pad = (2 * I + 127) // 128 * 128
    assert probs.shape == (M,) and probs.dtype == torch.float32
    assert row_out.shape[1] == n_pad
    assert col_out.shape[0] == 2 * I
    m_pad_row, m_pad_col = int(row_out.shape[0]), int(col_out.shape[1])
    assert row_sc.shape == (m_pad_row, n_pad // 32)
    assert col_sc.shape == (2 * I, m_pad_col // 32)
    n_bk = (I + 127) // 128
    assert tuple(grad_probs_partial.shape) == (n_bk, int(grad_probs_partial.shape[1]))
    assert grad_probs_partial.dtype == torch.float32
    gp_stride = int(grad_probs_partial.shape[1])

    grid = grouped_qdual_grid(m_pad_col, I)
    scalen_col = m_pad_col // 32
    out_fp8 = "e5m2" if row_out.dtype == torch.float8_e5m2 else "e4m3"
    col_data_band = _GQD_BK * m_pad_col < (1 << 31)
    col_scale_band = _GQD_BK * scalen_col < (1 << 31)
    key = (I, G, out_fp8, activation, clamp_limit, col_data_band, col_scale_band)
    stream = torch.cuda.current_stream()
    go = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)
    qr = row_out.view(torch.uint8)
    qrs = row_sc.view(torch.uint8)
    qc = col_out.view(torch.uint8)
    qcs = col_sc.view(torch.uint8)
    args = (
        dact,
        l1,
        probs,
        qr,
        qrs,
        qc,
        qcs,
        go,
        grad_probs_partial,
        m_pad_row,
        m_pad_col,
        scalen_col,
        gp_stride,
        grid,
        stream,
    )
    comp = _PHASE2_DGLU_QDUAL_CACHE.get(key)
    if comp is None:
        launch = _compile_grouped_dglu_qdual(
            I,
            G,
            out_fp8=out_fp8,
            col_data_band=col_data_band,
            col_scale_band=col_scale_band,
            activation=activation,
            clamp_limit=clamp_limit,
        )
        comp = flyc.compile(launch, *args)
        _PHASE2_DGLU_QDUAL_CACHE[key] = comp
    comp(*args)
    return row_out, row_sc, col_out, col_sc


def dglu_epi_quant_supported(K: int, I: int, out_dtype=torch.bfloat16) -> bool:
    """Return whether the kernel supports the given padded shape."""
    if out_dtype != torch.bfloat16:
        return False
    if K % 128 or K < 256:
        return False
    return I > 0 and I % 64 == 0 and I % MB == 0


def grouped_gemm_mxfp8_dglu_grad_probs_partial_spec(
    a: "torch.Tensor", b: "torch.Tensor"
) -> GradProbsPartialSpec:
    """The buffer :func:`grouped_gemm_mxfp8_epi_dglu_quant_flydsl_kernel` expects.

    ``grad_probs_partial.sum(0)`` is the gradient wrt ``probs``. One slice per
    128-column I-tile; ``needs_zero`` because a group's last M-tile is clamped
    at the group end, leaving the rows past it unwritten.
    """
    M_total = int(a.shape[0])
    I = int(b.shape[1])
    return GradProbsPartialSpec(shape=(ceildiv(I, _GQD_BK), M_total), needs_zero=True)


def _get_dact_ws(M, I, dtype, device):
    key = (M, I, dtype, device)
    t = _DGLU_DACT_WS.get(key)
    if t is None:
        t = torch.empty((M, I), dtype=dtype, device=device)
        _DGLU_DACT_WS[key] = t
    return t


def grouped_gemm_mxfp8_epi_dglu_quant_flydsl_kernel(
    a: "torch.Tensor",
    a_scale: "torch.Tensor",
    b: "torch.Tensor",
    b_scale: "torch.Tensor",
    intermediate: "torch.Tensor",
    group_offs: "torch.Tensor",
    probs: "torch.Tensor",
    grad_probs_partial: "torch.Tensor",
    row_out: "torch.Tensor",
    row_sc: "torch.Tensor",
    col_out: "torch.Tensor",
    col_sc: "torch.Tensor",
    N: int,
    K: int,
    *,
    group_offs_out: "torch.Tensor",
    activation: str = "silu",
    clamp_limit: "float | None" = None,
    out_dtype=torch.bfloat16,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Grouped GEMM, then dGLU and dual quantization epilogue."""
    _check_activation(activation)
    _check_clamp_limit(activation, clamp_limit)
    assert a.ndim == 2 and b.ndim == 3 and intermediate.ndim == 2
    I = N
    assert dglu_epi_quant_supported(K, I, out_dtype), (
        f"the dGLU quant epilogue does not cover I={I} K={K} out_dtype={out_dtype}"
    )
    M_total, G = int(intermediate.shape[0]), int(b.shape[0])
    assert int(b.shape[1]) == I and int(b.shape[2]) == K, f"b must be [{G}, {I}, {K}], got {tuple(b.shape)}"
    assert a.shape[1] == K, f"a must be [*, {K}], got {tuple(a.shape)}"
    assert intermediate.shape == (M_total, 2 * I), (
        f"intermediate must be [{M_total}, {2 * I}], got {tuple(intermediate.shape)}"
    )
    assert probs.ndim == 1 and probs.shape[0] == M_total and probs.dtype == torch.float32
    want = grouped_gemm_mxfp8_dglu_grad_probs_partial_spec(a, b).shape
    assert tuple(grad_probs_partial.shape) == want and grad_probs_partial.dtype == torch.float32, (
        f"grad_probs_partial must be {list(want)} float32, got "
        f"{list(grad_probs_partial.shape)} {grad_probs_partial.dtype}; "
        "size it with grouped_gemm_mxfp8_dglu_grad_probs_partial_spec"
    )
    row_shape, col_shape = _act_operand_geometry(2 * I, int(row_out.shape[0]), int(col_out.shape[1]))
    assert tuple(row_out.shape) == row_shape, f"row_out must be {row_shape}, got {tuple(row_out.shape)}"
    assert tuple(col_out.shape) == col_shape, f"col_out must be {col_shape}, got {tuple(col_out.shape)}"
    assert tuple(row_sc.shape) == (row_shape[0], row_shape[1] // MB)
    assert tuple(col_sc.shape) == (col_shape[0], col_shape[1] // MB)

    # stage 1: grouped GEMM, writing dact at the tight rows stage 2 reads
    dact = _get_dact_ws(M_total, I, out_dtype, a.device)
    grouped_gemm_mxfp8_flydsl_kernel(
        a,
        a_scale,
        b,
        b_scale,
        group_offs,
        I,
        K,
        group_offs_out=group_offs_out,
        out_dtype=out_dtype,
        out=dact,
    )
    # stage 2: dGLU and dual quantization epilogue
    epi_dglu_quant_flydsl_kernel(
        dact,
        intermediate,
        probs,
        group_offs_out,
        grad_probs_partial,
        row_out,
        row_sc,
        col_out,
        col_sc,
        activation=activation,
        clamp_limit=clamp_limit,
    )
    return row_out, row_sc, col_out, col_sc
