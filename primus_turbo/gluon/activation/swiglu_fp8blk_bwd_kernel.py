###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE SwiGLU backward + dual FP8 quant (d_gate | d_up) as a persistent Gluon kernel for gfx1250,
byte-identical to quantization_fp8blk_dual_kernel._swb_q2_kernel: q [P, 2I] + x4 s,
qT [2I, P] + compact sT, and the per-(row, 128-col block) d w partials gw [P, I/128] (fp32, same summation tree).

Same structure as swiglu_fp8blk_fwd_kernel: gate / up / d_act tiles TDM-streamed, the SwiGLU backward computed ONCE in
the 8 x 8-per-lane layout, both outputs quantized from registers (row amax in-warp, column amax via one LDS exchange,
qT through an LDS fp8 transpose), the two tiles' LDS exchanges interleaved so they share barriers.
gw: the reference sums d * silu * up in a [4 rows x 16 cols]-per-thread layout (8 warps across the 128 cols): a
pairwise tree over each thread's 16 columns (even and odd columns separately, then even + odd), then an xor
butterfly over the 8 warps' partials (warp bit 2, then 1, then 0). Here the same additions in the same order: the
in-lane part of the tree on registers, the cross-lane parts with DPP row_xmask adds.
sigmoid: v_exp_f32 + v_rcp_f32 + one Newton step, bit-equal to tl.sigmoid for every bf16 gate value under the
default activation (exhaustive check)."""

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna5 import CachePolicy, tdm
from triton.experimental.gluon.language.amd.cdna5 import buffer as bfr

from primus_turbo.gluon.activation.swiglu_fp8blk_fwd_kernel import (
    _e_fly,
    _p2neg_fly,
    _vexp,
    _vrcp,
)


@gluon.jit
def _sig(g, ALPHA: gl.constexpr):
    """tl.sigmoid(ALPHA g) bit for bit (VERIFIED_ACT): rcp(1 + exp2(-(ALPHA g) log2e)) + one Newton step."""
    d = _vexp((ALPHA * g) * -1.4426950408889634) + 1.0
    r = _vrcp(d)
    return gl.fma(gl.fma(-d, r, 1.0), r, r)


@gluon.jit
def _dpp_add(x, MASK: gl.constexpr):
    """x + x[lane ^ MASK] (within 16-lane rows); fp32 add is commutative, every lane gets the same pair sum."""
    if MASK == 1:
        return gl.inline_asm_elementwise(
            "v_add_f32_dpp $0, $1, $1 row_xmask:1 row_mask:0xf bank_mask:0xf",
            "=v,v",
            [x],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    elif MASK == 2:
        return gl.inline_asm_elementwise(
            "v_add_f32_dpp $0, $1, $1 row_xmask:2 row_mask:0xf bank_mask:0xf",
            "=v,v",
            [x],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    elif MASK == 4:
        return gl.inline_asm_elementwise(
            "v_add_f32_dpp $0, $1, $1 row_xmask:4 row_mask:0xf bank_mask:0xf",
            "=v,v",
            [x],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        return gl.inline_asm_elementwise(
            "v_add_f32_dpp $0, $1, $1 row_xmask:8 row_mask:0xf bank_mask:0xf",
            "=v,v",
            [x],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )


@gluon.jit
def _gw_sum(x, GWO: gl.constexpr):
    """row sums of x [128, 128] (layout L) in the reference's addition order -> [128, 16] (equal over dim 1)."""
    x5 = gl.reshape(x, [128, 16, 2, 2, 2])  # (row, 8-col group = lane, c bit 2, c bit 1, c bit 0)
    xe, xo = gl.split(x5)  # even / odd columns
    e0, e1 = gl.split(xe)
    ea = e0 + e1  # (x0 + x2), (x4 + x6)
    ea0, ea1 = gl.split(ea)
    ae = ea0 + ea1  # (x0 + x2) + (x4 + x6)
    o0, o1 = gl.split(xo)
    oa = o0 + o1
    oa0, oa1 = gl.split(oa)
    ao = oa0 + oa1
    ev = _dpp_add(ae, 1)  # + the other 8 columns of the 16-col group (lane ^ 1)
    od = _dpp_add(ao, 1)
    pw = ev + od  # one warp's 16-column partial of the reference
    if GWO == 0:  # butterfly over the 8 partials: warp bit 2 (lane ^ 8), bit 1 (lane ^ 4), bit 0 (lane ^ 2)
        pw = _dpp_add(pw, 8)
        pw = _dpp_add(pw, 4)
        pw = _dpp_add(pw, 2)
    else:
        pw = _dpp_add(pw, 2)
        pw = _dpp_add(pw, 4)
        pw = _dpp_add(pw, 8)
    return pw


@gluon.jit
def _rowq(
    v,
    p,
    pt,
    kb,
    cq0,
    qdesc,
    S,
    NKB: gl.constexpr,
    CQ,
    L: gl.constexpr,
    LS1: gl.constexpr,
    MAX_EXP: gl.constexpr,
    dty,
):
    """row (1x128) quant of v [128, 128] (L): q columns cq0 + j of the [*, CQ] row-major copy, x4 scale block kb."""
    er = _e_fly(gl.max(gl.abs(v), axis=1), MAX_EXP)
    q = (v * _p2neg_fly(er)[:, None]).to(dty)
    c = cq0 + gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    bfr.store(q, qdesc, p[:, None] * CQ + c[None, :])
    ws = gl.convert_layout(((er + 127) & 0xFF) * 0x01010101, LS1)
    ps = pt * 128 + gl.arange(0, 128, layout=LS1)
    gl.store(S.to(gl.pointer_type(gl.int32)) + ps.to(gl.int64) * NKB + kb, ws)


@gluon.jit
def _swb_gl_kernel(
    DA,
    L1,
    W,
    Q,
    S,
    QT,
    ST,
    GW,
    P,
    I,
    PREAL,
    ALPHA: gl.constexpr,
    OFF: gl.constexpr,
    GLO: gl.constexpr,
    GHI: gl.constexpr,
    UCL: gl.constexpr,
    NPROG: gl.constexpr,
    NS: gl.constexpr,
    MAX_EXP: gl.constexpr,
    GWO: gl.constexpr = 0,
    QTNT: gl.constexpr = 1,
):
    gl.static_assert(gl.num_warps() == 8)
    L: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [8, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[128, 128],
    )
    L2: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [8, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[128, 128],
    )
    LSC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0]],
        warp_bases=[[4, 0], [8, 0], [16, 0]],
        block_bases=[],
        shape=[128, 128],
    )
    LS1: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1], [2], [4], [0], [8]],
        warp_bases=[[16], [32], [64]],
        block_bases=[],
        shape=[128],
    )
    LRD: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0]],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        warp_bases=[[0, 32], [0, 64], [0, 0]],
        block_bases=[],
        shape=[8, 128],
    )
    SH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [128, 128], [1, 0])
    RS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    FS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    pid = gl.program_id(0)
    nct = I // 128
    ntiles = (gl.load(PREAL) // 128) * nct
    gbuf = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    ubuf = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    dbuf = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    tb_g = gl.allocate_shared_memory(QT.dtype.element_ty, [128, 128], SH)
    tb_u = gl.allocate_shared_memory(QT.dtype.element_ty, [128, 128], SH)
    rd_g = gl.allocate_shared_memory(gl.float32, [8, 128], RS)
    rd_u = gl.allocate_shared_memory(gl.float32, [8, 128], RS)
    fb_g = gl.allocate_shared_memory(gl.float32, [128], FS)
    fb_u = gl.allocate_shared_memory(gl.float32, [128], FS)
    l1d = tdm.make_tensor_descriptor(
        base=L1, shape=(P, 2 * I), strides=(2 * I, 1), block_shape=(128, 128), layout=SH
    )
    dad = tdm.make_tensor_descriptor(base=DA, shape=(P, I), strides=(I, 1), block_shape=(128, 128), layout=SH)
    qdesc = bfr.make_buffer_descriptor(Q, P * 2 * I)
    qtdesc = bfr.make_buffer_descriptor(QT, P * 2 * I)
    n_my = (ntiles - pid + NPROG - 1) // NPROG
    for s in gl.static_range(NS - 1):
        t0 = gl.maximum(gl.minimum(pid + s * NPROG, ntiles - 1), 0)
        tdm.async_load(l1d, [(t0 // nct) * 128, (t0 % nct) * 128], gbuf.index(s))
        tdm.async_load(l1d, [(t0 // nct) * 128, I + (t0 % nct) * 128], ubuf.index(s))
        tdm.async_load(dad, [(t0 // nct) * 128, (t0 % nct) * 128], dbuf.index(s))
    for it in range(0, n_my):
        t = pid + it * NPROG
        pt = t // nct
        pc = t % nct
        tn = gl.minimum(t + (NS - 1) * NPROG, ntiles - 1)  # tail iterations re-load a valid tile (never read)
        tdm.async_wait((NS - 2) * 3)
        sl = (it + NS - 1) % NS
        tdm.async_load(l1d, [(tn // nct) * 128, (tn % nct) * 128], gbuf.index(sl))
        tdm.async_load(l1d, [(tn // nct) * 128, I + (tn % nct) * 128], ubuf.index(sl))
        tdm.async_load(dad, [(tn // nct) * 128, (tn % nct) * 128], dbuf.index(sl))
        g0 = gbuf.index(it % NS).load(L).to(gl.float32)
        u0 = ubuf.index(it % NS).load(L).to(gl.float32)
        d = dbuf.index(it % NS).load(L).to(gl.float32)
        p = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, L))
        w = gl.load(W + p)
        # quant_blk_gluon_gfx1250._swb_tdm_kernel's math, same expressions and roundings
        g = gl.minimum(gl.maximum(g0, GLO), GHI)
        u = gl.minimum(gl.maximum(u0, -UCL), UCL)
        sg = _sig(g, ALPHA)
        silu = g * sg
        up = u + OFF
        gws = _gw_sum(d * silu * up, GWO)
        cg = gl.arange(0, 16, layout=gl.SliceLayout(0, gws.type.layout))
        pg = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, gws.type.layout))
        gl.store(GW + pg[:, None].to(gl.int64) * nct + pc + cg[None, :] * 0, gws, mask=(cg == 0)[None, :])
        dw = d * w[:, None]
        gin = (g0 >= GLO) & (g0 <= GHI)
        uin = (u0 >= -UCL) & (u0 <= UCL)
        d_up = gl.where(uin, dw * silu, 0.0)
        d_gate = gl.where(gin, dw * up * (sg + ALPHA * g * sg * (1.0 - sg)), 0.0)
        # column partials of both tiles first; the row quants run while the other warps reach the exchange
        vg2 = gl.convert_layout(d_gate, L2)
        vu2 = gl.convert_layout(d_up, L2)
        rd_g.store(gl.max(gl.reshape(gl.abs(vg2), [8, 16, 128]), axis=1))
        rd_u.store(gl.max(gl.reshape(gl.abs(vu2), [8, 16, 128]), axis=1))
        _rowq(d_gate, p, pt, pc, pc * 128, qdesc, S, 2 * nct, 2 * I, L, LS1, MAX_EXP, Q.dtype.element_ty)
        _rowq(
            d_up, p, pt, nct + pc, I + pc * 128, qdesc, S, 2 * nct, 2 * I, L, LS1, MAX_EXP, Q.dtype.element_ty
        )
        cc = pc * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LRD))
        ecg = _e_fly(gl.max(rd_g.load(LRD), axis=0), MAX_EXP)
        ecu = _e_fly(gl.max(rd_u.load(LRD), axis=0), MAX_EXP)
        gl.store(ST + cc.to(gl.int64) * (P // 128) + pt, ((ecg + 127) & 0xFF).to(gl.uint8))
        gl.store(ST + (I + cc).to(gl.int64) * (P // 128) + pt, ((ecu + 127) & 0xFF).to(gl.uint8))
        fb_g.store(_p2neg_fly(ecg))
        fb_u.store(_p2neg_fly(ecu))
        qcg = (vg2 * fb_g.load(gl.SliceLayout(0, L2))[None, :]).to(QT.dtype.element_ty)
        qcu = (vu2 * fb_u.load(gl.SliceLayout(0, L2))[None, :]).to(QT.dtype.element_ty)
        tb_g.store(gl.permute(qcg, [1, 0]))
        tb_u.store(gl.permute(qcu, [1, 0]))
        qsg = tb_g.load(LSC)
        qsu = tb_u.load(LSC)
        cs = pc * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LSC))
        ps2 = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LSC))
        if QTNT:
            bfr.store(qsg, qtdesc, cs[:, None] * P + ps2[None, :], cache_policy=CachePolicy("none"))
            bfr.store(qsu, qtdesc, (I + cs)[:, None] * P + ps2[None, :], cache_policy=CachePolicy("none"))
        else:
            bfr.store(qsg, qtdesc, cs[:, None] * P + ps2[None, :])
            bfr.store(qsu, qtdesc, (I + cs)[:, None] * P + ps2[None, :])
    tdm.async_wait(0)


_NCU = {}


def _ncu(dev):
    n = _NCU.get(dev.index)
    if n is None:
        n = _NCU[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


def swb_gl(d_act, l1, w, q, s, qT, sT, gw, preal, alpha, off, glo, ghi, ucl, ns=2, wpc=1, **kn):
    """quantization_fp8blk_dual_kernel.swb_q2's contract (outputs allocated by swiglu_bwd_dual_blk)."""
    P, I = d_act.shape
    assert l1.shape == (P, 2 * I) and P % 128 == 0 and I % 128 == 0
    for t in (d_act, l1, w, q, s, qT, sT, gw, preal):
        assert t.is_cuda and t.is_contiguous() and t.data_ptr() != 0
    nprog = _ncu(d_act.device) * wpc
    _swb_gl_kernel[(nprog,)](
        d_act,
        l1,
        w,
        q,
        s,
        qT,
        sT,
        gw,
        P,
        I,
        preal,
        ALPHA=alpha,
        OFF=off,
        GLO=glo,
        GHI=ghi,
        UCL=ucl,
        NPROG=nprog,
        NS=ns,
        MAX_EXP=8,
        **{k.upper(): v for k, v in kn.items()},
        num_warps=8,
    )
