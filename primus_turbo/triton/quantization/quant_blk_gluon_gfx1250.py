###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""fuse-moe: Gluon (Triton main, gluon cdna5) TDM-streamed MegaMoE SwiGLU backward + dual quant
(PRIMUS_FUSE_MOE_SWB_TDM=1, default off; dispatched from quant_blk_gfx1250.swiglu_bwd_dual_blk).

Same per-element math, amax, E8M0 exponents, FP8 bytes, scale bytes and d w partials as
quant_blk_gfx1250._swiglu_bwd_dual_kernel (CT=1, x4 row scales, canonical column order): byte-identical outputs
(checked at the proxy shape, bench_gl.py / test_moeq2.py).

Differences: a persistent grid (one program per CU) walks the live 128x128 tiles; the next tile's gate / up / d
tiles are TDM-loaded into the other LDS stage while the current one is computed; registers hold 4 pool rows x 16
channels per thread with the 32 lanes down the rows and the warps across the channels, so the column (128x1) amax
stays inside a warp and the transposed FP8 store is the same registers re-viewed (order [0, 1]).
Measured (graph replay, cold, proxy shape): 131-132 us -> 118.7-119.2 us.
"""
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm


@gluon.jit
def _e_fly(amax, MAX_EXP: gl.constexpr):
    """quant_epi_gfx1250.ue8m0_exp_fly"""
    bits = amax.to(gl.int32, bitcast=True)
    fld = (bits >> 23) & 0xFF
    e = fld - (127 + MAX_EXP) + ((bits & 0x7FFFFF) > 0x600000).to(gl.int32)
    e = gl.where(fld == 0xFF, 126, e)
    e = gl.where(amax > 0, e, 0)
    return gl.minimum(gl.maximum(e, -126), 126)


@gluon.jit
def _p2neg(e):
    return ((127 - e) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _sigmoid(x):
    return 1 / (1 + gl.exp(-x))  # tl.sigmoid's expression


@gluon.jit
def _quant_both(v, p, cq, kbq, cT, pt, P, CQ, Q, S, QT, ST, L: gl.constexpr, LT: gl.constexpr):
    """v [128 pool rows, 128 cols] fp32 in layout L: 1x128 row quant -> Q[p, cq] + S x4 (block kbq of CQ columns);
    128x1 column quant -> QT[cT, p] + compact ST[cT, p / 128]."""
    av = gl.abs(v)
    er = _e_fly(gl.max(av, axis=1), 8)
    gl.store(Q + p[:, None].to(gl.int64) * CQ + cq[None, :], (v * _p2neg(er)[:, None]).to(Q.dtype.element_ty))
    gl.store(S.to(gl.pointer_type(gl.int32)) + p.to(gl.int64) * (CQ // 128) + kbq, ((er + 127) & 0xFF) * 0x01010101)
    ec = _e_fly(gl.max(av, axis=0), 8)
    qc = gl.convert_layout((v * _p2neg(ec)[None, :]).to(QT.dtype.element_ty), LT)
    pT = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LT))
    cTT = gl.convert_layout(cT, gl.SliceLayout(0, LT))
    gl.store(QT + cTT[None, :].to(gl.int64) * P + pT[:, None], qc)
    gl.store(ST + cT.to(gl.int64) * (P // 128) + pt, ((ec + 127) & 0xFF).to(gl.uint8))


@gluon.jit
def _swb_tdm_kernel(DA, L1, W, Q, S, QT, ST, GW, P, I, PREAL, ALPHA: gl.constexpr, OFF: gl.constexpr,
                    GLO: gl.constexpr, GHI: gl.constexpr, UCL: gl.constexpr, NPROG: gl.constexpr, NW: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout(size_per_thread=[4, 16], threads_per_warp=[32, 1], warps_per_cta=[1, NW],
                                       order=[1, 0])
    LT: gl.constexpr = gl.BlockedLayout(size_per_thread=[4, 16], threads_per_warp=[32, 1], warps_per_cta=[1, NW],
                                        order=[0, 1])
    SH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [128, 128], [1, 0])
    pid = gl.program_id(0)
    nct = I // 128
    ntiles = (gl.load(PREAL) // 128) * nct
    gbuf = gl.allocate_shared_memory(gl.bfloat16, [2, 128, 128], SH)
    ubuf = gl.allocate_shared_memory(gl.bfloat16, [2, 128, 128], SH)
    dbuf = gl.allocate_shared_memory(gl.bfloat16, [2, 128, 128], SH)
    l1d = tdm.make_tensor_descriptor(base=L1, shape=(P, 2 * I), strides=(2 * I, 1), block_shape=(128, 128), layout=SH)
    dad = tdm.make_tensor_descriptor(base=DA, shape=(P, I), strides=(I, 1), block_shape=(128, 128), layout=SH)
    n_my = (ntiles - pid + NPROG - 1) // NPROG
    t0 = gl.minimum(pid, ntiles - 1)
    tdm.async_load(l1d, [(t0 // nct) * 128, (t0 % nct) * 128], gbuf.index(0))
    tdm.async_load(l1d, [(t0 // nct) * 128, I + (t0 % nct) * 128], ubuf.index(0))
    tdm.async_load(dad, [(t0 // nct) * 128, (t0 % nct) * 128], dbuf.index(0))
    for it in range(0, n_my):
        slot = it % 2
        t = pid + it * NPROG
        tn = gl.minimum(t + NPROG, ntiles - 1)  # the last iteration re-loads a valid tile (never read)
        tdm.async_wait(0)  # this tile's loads (issued one iteration earlier) are complete for this wave ...
        gl.barrier()  # ... and for all waves; the other stage was released by the barrier ending the last iteration
        tdm.async_load(l1d, [(tn // nct) * 128, (tn % nct) * 128], gbuf.index(1 - slot))
        tdm.async_load(l1d, [(tn // nct) * 128, I + (tn % nct) * 128], ubuf.index(1 - slot))
        tdm.async_load(dad, [(tn // nct) * 128, (tn % nct) * 128], dbuf.index(1 - slot))
        pt = t // nct
        pc = t % nct
        p = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, L))
        c = pc * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, L))
        g0 = gbuf.index(slot).load(L).to(gl.float32)
        u0 = ubuf.index(slot).load(L).to(gl.float32)
        d = dbuf.index(slot).load(L).to(gl.float32)
        w = gl.load(W + p)
        # _swiglu_bwd_dual_kernel's expressions, in its order
        g = gl.minimum(gl.maximum(g0, GLO), GHI)
        u = gl.minimum(gl.maximum(u0, -UCL), UCL)
        sg = _sigmoid(ALPHA * g)
        silu = g * sg
        up = u + OFF
        gl.store(GW + p.to(gl.int64) * (I // 128) + pc, gl.sum(d * silu * up, axis=1))
        dw = d * w[:, None]
        gin = (g0 >= GLO) & (g0 <= GHI)
        uin = (u0 >= -UCL) & (u0 <= UCL)
        d_up = gl.where(uin, dw * silu, 0.0)
        d_gate = gl.where(gin, dw * up * (sg + ALPHA * g * sg * (1.0 - sg)), 0.0)
        _quant_both(d_gate, p, c, pc, c, pt, P, 2 * I, Q, S, QT, ST, L, LT)
        _quant_both(d_up, p, I + c, pc + I // 128, I + c, pt, P, 2 * I, Q, S, QT, ST, L, LT)
        gl.barrier()  # every wave is done with this stage before the next iteration's loads overwrite it
    tdm.async_wait(0)


_NPROG = {}


def swiglu_bwd_dual_tdm(d_act, l1, w, q, s, qT, sT, gw, preal, alpha, off, glo, ghi, ucl):
    """Launch on the tensors swiglu_bwd_dual_blk allocated (compact column scales, x4 row scales, canonical order)."""
    P, I = d_act.shape
    assert l1.shape == (P, 2 * I) and P % 128 == 0 and I % 128 == 0
    for t in (d_act, l1, w, q, s, qT, sT, gw, preal):
        assert t.is_cuda and t.is_contiguous() and t.data_ptr() != 0
    dev = d_act.device.index if d_act.device.index is not None else 0
    nprog = _NPROG.get(dev)
    if nprog is None:
        import torch

        nprog = _NPROG[dev] = torch.cuda.get_device_properties(d_act.device).multi_processor_count
    _swb_tdm_kernel[(nprog,)](d_act, l1, w, q, s, qT, sT, gw, P, I, preal, ALPHA=alpha, OFF=off, GLO=glo, GHI=ghi,
                              UCL=ucl, NPROG=nprog, NW=8, num_warps=8)


__all__ = ["swiglu_bwd_dual_tdm"]
