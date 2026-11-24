###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE SwiGLU forward + dual FP8 quant as a persistent Gluon kernel for gfx1250 (needs the Gluon cdna5
dialect, see ``primus_turbo.gluon.gate``). Byte-identical to quantization_fp8blk_moe_kernel._swiglu_fwd_dual_kernel with
CT=1, SL=0 (x4 row scales), ILV=0 and the default activation:

  act = w * g * sigmoid(g) * (u + 0), g / u = the +-10 clamped gate / up halves of l1 [P, 2I] (bf16),
  q [P, I] + s (x4 int32 words, 1x128 along I), qT [I, P] + compact sT [I, P/128] (128x1 along the pool rows).

Per 128 x 128 tile (persistent programs, 2 per CU, NS-stage TDM ring for the gate / up tiles):
  * the SwiGLU is computed ONCE, in a register layout where every lane owns an 8 x 8 block (8 rows x 8 channels,
    16 lanes across the 128 channels, lane bit 4 = row half, warps down the rows);
  * row amax in-lane + 4 DPP steps; q stored straight from registers (8 B per lane, 16 lanes per 128-B row);
  * column amax in-lane + one cross-half shuffle, then one LDS exchange of the 8 warps' partials (one lane per
    column reduces them, stores the sT byte and the 2^-e factor);
  * the column quant packs 8 pool rows per cvt (8 qT bytes), goes through an LDS transpose buffer and is stored
    coalesced (16 B per lane, 8 lanes per 128-B qT row).
  * g * sigmoid(g) = g * r with r = rcp(1 + exp2(-g log2e)) refined by one Newton step (v_exp_f32, v_rcp_f32, 2 FMA).
    For every bf16 gate value in the default activation's domain this equals tl.sigmoid's correctly rounded divide
    bit for bit (exhaustive check over all 65536 bf16 inputs; the llvm.exp2 range fix-up never fires there). Other
    activations are not verified and take the Triton kernel.
"""

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna5 import CachePolicy, tdm
from triton.experimental.gluon.language.amd.cdna5 import buffer as bfr

# activation_constexpr() tuples whose fast g * sigmoid(g) is verified bit-exact against tl.sigmoid for every bf16 g
VERIFIED_ACT = {(1.0, 0.0, -10.0, 10.0, 10.0)}


@gluon.jit
def _vexp(x):
    return gl.inline_asm_elementwise("v_exp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _vrcp(x):
    return gl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _gsig(g, ALPHA: gl.constexpr):
    """g * sigmoid(ALPHA g) with tl.sigmoid's roundings: x = fl(-(ALPHA g) log2e), e = exp2(x) (= v_exp_f32 for
    x >= -126), d = fl(1 + e), 1 / d correctly rounded (rcp + one Newton step: exhaustively verified, VERIFIED_ACT)."""
    d = _vexp((ALPHA * g) * -1.4426950408889634) + 1.0
    r = _vrcp(d)
    r = gl.fma(gl.fma(-d, r, 1.0), r, r)
    return g * r


@gluon.jit
def _e_fly(amax, MAX_EXP: gl.constexpr):
    """quantization_fp8blk_utils.ue8m0_exp_fly"""
    bits = amax.to(gl.int32, bitcast=True)
    fld = (bits >> 23) & 0xFF
    e = fld - (127 + MAX_EXP) + ((bits & 0x7FFFFF) > 0x600000).to(gl.int32)
    e = gl.where(fld == 0xFF, 126, e)
    e = gl.where(amax > 0, e, 0)
    return gl.minimum(gl.maximum(e, -126), 126)


@gluon.jit
def _p2neg_fly(e):
    return ((127 - e) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _swf_gl_kernel(
    L1,
    W,
    Q,
    S,
    QT,
    ST,
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
    EARLY: gl.constexpr,
    MAX_EXP: gl.constexpr,
    ORD: gl.constexpr = 1,
    BST: gl.constexpr = 1,
    TBL: gl.constexpr = 0,
    QTNT: gl.constexpr = 1,
):
    gl.static_assert(gl.num_warps() == 8)
    # math layout [128 pool rows, 128 channels]: lane = 8 rows x 8 channels (channels fastest in registers)
    L: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [8, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[128, 128],
    )
    # same lane / warp map, rows fastest in registers (the column quant packs 8 rows -> 8 qT bytes per cvt)
    L2: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [8, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[128, 128],
    )
    # coalesced qT store layout [128 channels, 128 pool rows]: 16 B per lane, 8 lanes per 128-B qT row
    LSC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0]],
        warp_bases=[[4, 0], [8, 0], [16, 0]],
        block_bases=[],
        shape=[128, 128],
    )
    # one lane per row (row scale words) / one lane per channel with the 8 warps' partials in-lane (column amax)
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
    pid = gl.program_id(0)
    nct = I // 128
    ntiles = (gl.load(PREAL) // 128) * nct
    gbuf = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    ubuf = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    if TBL == 0:
        TSH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [128, 128], [1, 0])
    else:
        TSH: gl.constexpr = gl.SwizzledSharedLayout(16, 8, 8, [1, 0])
    tbuf = gl.allocate_shared_memory(QT.dtype.element_ty, [128, 128], TSH)
    qdesc = bfr.make_buffer_descriptor(Q, P * I)
    qtdesc = bfr.make_buffer_descriptor(QT, P * I)
    red = gl.allocate_shared_memory(gl.float32, [8, 128], gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    fbuf = gl.allocate_shared_memory(gl.float32, [128], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    l1d = tdm.make_tensor_descriptor(
        base=L1, shape=(P, 2 * I), strides=(2 * I, 1), block_shape=(128, 128), layout=SH
    )
    n_my = (ntiles - pid + NPROG - 1) // NPROG
    for s in gl.static_range(NS if EARLY else NS - 1):
        t0 = gl.maximum(gl.minimum(pid + s * NPROG, ntiles - 1), 0)
        tdm.async_load(l1d, [(t0 // nct) * 128, (t0 % nct) * 128], gbuf.index(s))
        tdm.async_load(l1d, [(t0 // nct) * 128, I + (t0 % nct) * 128], ubuf.index(s))
    for it in range(0, n_my):
        t = pid + it * NPROG
        pt = t // nct
        pc = t % nct
        if EARLY:
            # every slot holds a future tile; it is re-filled as soon as its tile is in registers
            tn = gl.minimum(t + NS * NPROG, ntiles - 1)  # tail iterations re-load a valid tile (never read)
            tdm.async_wait((NS - 1) * 2)
            g = gbuf.index(it % NS).load(L)
            u = ubuf.index(it % NS).load(L)
            tdm.async_load(l1d, [(tn // nct) * 128, (tn % nct) * 128], gbuf.index(it % NS))
            tdm.async_load(l1d, [(tn // nct) * 128, I + (tn % nct) * 128], ubuf.index(it % NS))
            g = g.to(gl.float32)
            u = u.to(gl.float32)
        else:
            tn = gl.minimum(t + (NS - 1) * NPROG, ntiles - 1)
            tdm.async_wait((NS - 2) * 2)
            sl = (it + NS - 1) % NS
            tdm.async_load(l1d, [(tn // nct) * 128, (tn % nct) * 128], gbuf.index(sl))
            tdm.async_load(l1d, [(tn // nct) * 128, I + (tn % nct) * 128], ubuf.index(sl))
            g = gbuf.index(it % NS).load(L).to(gl.float32)
            u = ubuf.index(it % NS).load(L).to(gl.float32)
        p = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, L))
        w = gl.load(W + p)
        # quantization_fp8blk_moe_kernel.swiglu_act_w, same roundings
        g = gl.minimum(gl.maximum(g, GLO), GHI)
        u = gl.minimum(gl.maximum(u, -UCL), UCL)
        v = _gsig(g, ALPHA) * (u + OFF) * w[:, None]
        v2 = gl.convert_layout(v, L2)  # register renaming only
        if ORD:
            # column partials first: the row quant below runs while the other warps reach the exchange
            red.store(gl.max(gl.reshape(gl.abs(v2), [8, 16, 128]), axis=1))  # in-warp partials of the 8 warps
        # ---- row quant (1x128 along the channels) -> Q, x4 S
        er = _e_fly(gl.max(gl.abs(v), axis=1), MAX_EXP)
        q = (v * _p2neg_fly(er)[:, None]).to(Q.dtype.element_ty)
        c = pc * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, L))
        if BST:
            bfr.store(q, qdesc, p[:, None] * I + c[None, :])
        else:
            gl.store(Q + p[:, None].to(gl.int64) * I + c[None, :], q)
        ws = gl.convert_layout(((er + 127) & 0xFF) * 0x01010101, LS1)  # intra-warp regroup: one lane per row
        ps = pt * 128 + gl.arange(0, 128, layout=LS1)
        gl.store(S.to(gl.pointer_type(gl.int32)) + ps.to(gl.int64) * nct + pc, ws)
        # ---- column quant (128x1 along the pool rows) -> QT, compact ST
        if not ORD:
            red.store(gl.max(gl.reshape(gl.abs(v2), [8, 16, 128]), axis=1))
        ec = _e_fly(gl.max(red.load(LRD), axis=0), MAX_EXP)
        cc = pc * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LRD))
        gl.store(ST + cc.to(gl.int64) * (P // 128) + pt, ((ec + 127) & 0xFF).to(gl.uint8))
        fbuf.store(_p2neg_fly(ec))
        qc = (v2 * fbuf.load(gl.SliceLayout(0, L2))[None, :]).to(QT.dtype.element_ty)
        tbuf.store(gl.permute(qc, [1, 0]))
        qs = tbuf.load(LSC)
        cs = pc * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LSC))
        ps2 = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LSC))
        if BST:
            if QTNT:
                bfr.store(qs, qtdesc, cs[:, None] * P + ps2[None, :], cache_policy=CachePolicy("none"))
            else:
                bfr.store(qs, qtdesc, cs[:, None] * P + ps2[None, :])
        else:
            gl.store(QT + cs[:, None].to(gl.int64) * P + ps2[None, :], qs)
    tdm.async_wait(0)


_NCU = {}


def _ncu(dev):
    n = _NCU.get(dev.index)
    if n is None:
        n = _NCU[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


def swf_gl_ok(l1, w, preal, act, P, I, compact_t, sl, dtype) -> bool:
    """the shapes / layouts / activation this kernel reproduces bit for bit (else the caller keeps the Triton kernel)"""
    return (
        compact_t
        and sl == 0
        and dtype == torch.float8_e4m3fn
        and tuple(act) in VERIFIED_ACT
        and l1.dtype == torch.bfloat16
        and w.dtype == torch.float32
        and l1.is_contiguous()
        and w.is_contiguous()
        and l1.data_ptr() % 16 == 0
        and P % 128 == 0
        and I % 128 == 0
        and w.numel() >= P
        and (preal is None or (preal.dtype == torch.int32 and preal.is_contiguous()))
    )


def swf_gl(l1, w, preal, act, q, s, qT, sT, ns=2, early=0, wpc=2, **kn):
    """quantization_fp8blk_moe_kernel.swiglu_fwd_dual_blk(l1, w, act, preal=preal, compact_t=True) into preallocated outputs."""
    P, I2 = l1.shape
    I = I2 // 2
    alpha, off, glo, ghi, ucl = act
    if preal is None:
        preal = torch.full((1,), P, dtype=torch.int32, device=l1.device)
    for t in (l1, w, preal, q, s, qT, sT):
        assert t.is_cuda and t.data_ptr() != 0
    nprog = _ncu(l1.device) * wpc
    _swf_gl_kernel[(nprog,)](
        l1,
        w,
        q,
        s,
        qT,
        sT,
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
        EARLY=early,
        MAX_EXP=8,
        **{k.upper(): v for k, v in kn.items()},
        num_warps=8,
    )
    return q, s, qT, sT
