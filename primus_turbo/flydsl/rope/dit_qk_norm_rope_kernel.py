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

"""gfx950 backward of ``dit_qk_norm_rope`` (q / k RMSNorm + interleaved RoPE), FlyDSL.

Same arithmetic and roundings as the Triton backward (``primus_turbo/triton/rope/dit_qk_norm_rope_kernel.py``),
same addressing (token stride ``TS``, head stride ``HS``, q / k / v column offsets,
token order ``SB``), so it reads both the diffusers ``[B, S, 3*H*D]`` and the Megatron ``[S, B, H, 3D]`` QKV
GEMM output in place. The op that dispatches to it is in ``primus_turbo/pytorch/kernels/rope/
dit_qk_norm_rope_impl.py``.

Structure. One launch for one or two streams: 256 blocks of ``H`` 16-lane groups (one group per head, 8 values
per lane, D = 128); block ``bx`` walks tokens ``bx, bx + 256, ...`` of each segment in turn. Per token:

- the seven loads are issued in first-use order (cos, sin, x_q, dq, x_k, dk, dv), followed by a scheduling
  barrier, so they are in flight together and the compiler cannot meet the register cap by serialising them;
- the five large loads and the three d(qkv) stores are non-temporal (``nt``): every byte is touched once;
- the four 16-lane row sums (rstd and the dot product, for q and for k) are DPP butterflies
  (``flydsl.extension.coop.warp_reduce``), not LDS-pipe swizzles.

With ``add_bias`` (the op adds the QKV bias in its forward) each lane loads its q and k bias columns once per
segment, before the token loop, and adds them to x where rstd is recomputed, rounded as the forward rounds.
The bias and w then stay bf16 across the loop and are widened to fp32 where used, to fit the 128-VGPR cap.

At the end of a segment a block sums its heads' dw in LDS, in head order, and writes one fp32 row
``[2 * D]``; with ``bias_grad`` it also writes its fp32 column sums of d(qkv) ``[3 * H * D]``. The block rows of
every segment (per segment ``[2D dw | 3HD db]``) form one ``[256, ROWP]`` buffer that a single column-sum launch reduces to dw and db of every
stream. The order of dw's fp32 sum is: each block's tokens, then its heads, then the 256 blocks.

Built with ``waves_per_eu=4`` (at most 128 VGPRs: four waves per SIMD, so a 6-wave block still fits on a CU
next to a collective's workgroup). ``tests/pytorch/ops/test_dit_qk_norm_rope.py`` checks the built ISA.
"""

import functools

D = 128
VEC = 8  # values per lane
LANES = D // VEC  # lanes per head
GRID_X = 256  # blocks (one per CU on MI355X)
WAVES_PER_EU = 4
_NT = 2  # gfx950 buffer cache policy: nt


@functools.lru_cache(maxsize=None)
def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import flydsl.expr.math as fmath
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _marith
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl._mlir.dialects import scf as _scf
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.expr import rocdl as _xrocdl
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import ReductionOp as _RedOp
    from flydsl.expr.typing import Vector as Vec
    from flydsl.extension.coop import warp_reduce as _warp_reduce

    from primus_turbo.flydsl.utils.buffer_ops import (
        buffer_load,
        buffer_store,
        create_buffer_resource,
    )

    def make(H, NSEG, segs, s_out, eps, add_bias, bias_grad, TS, HS, QO, KO, VO, NB, SB, CSS, CSB, ROUND):
        # One fp32 row per block: per segment [2D dw | 3HD bias] (ROWP floats).
        SEGW = 2 * D + (3 * H * D if bias_grad else 0)
        ROWP = NSEG * SEGW
        threads = H * LANES
        hd = H * D

        # Plain-Python helpers, outside the kernel body: the kernel's AST pass would turn their Python ifs and for
        # loops into control flow.
        def row_sum(v, f32v):
            s = ArithValue(_vector.reduction(ir.F32Type.get(), _vector.CombiningKind.ADD, v))
            s = ArithValue(_warp_reduce(fx.Float32(s), _RedOp.ADD, width=LANES).ir_value())
            return _vector.broadcast(f32v, s)

        def sfor(n, fn, carried=()):
            """scf.for over [0, n) with fn(i32 induction value, *carried) as the body; fn returns the next
            iteration's carried values."""
            idx = ir.IndexType.get()
            op = _scf.ForOp(
                _marith.ConstantOp(idx, 0).result,
                _marith.ConstantOp(idx, n).result,
                _marith.ConstantOp(idx, 1).result,
                list(carried),
            )
            with ir.InsertionPoint(op.body):
                nxt = fn(
                    ArithValue(
                        _marith.IndexCastOp(ir.IntegerType.get_signless(32), op.induction_variable).result
                    ),
                    *op.inner_iter_args,
                )
                _scf.YieldOp(list(nxt) if carried else [])

        def emit(SRC0, SRC1, DSRC0, DSRC1, DQ, DK, DV, WQ0, WK0, WQ1, WK1, COS, SIN, B0, B1, PART, red):
            f32v = ir.VectorType.get([VEC], ir.F32Type.get())
            f32v4 = ir.VectorType.get([4], ir.F32Type.get())
            bf16v = ir.VectorType.get([VEC], ir.BF16Type.get())
            bf16 = ir.BF16Type.get()
            t = fx.thread_idx.x
            bx, _, _ = fx.block_idx
            lane8 = (t % fx.Int32(LANES)) * fx.Int32(VEC)
            col = t * fx.Int32(VEC)  # output column (h * D + lane8)
            scol = (t // fx.Int32(LANES)) * fx.Int32(HS) + fx.Int32(QO) + lane8  # q column in a source row
            dq_r, dk_r, dv_r = (create_buffer_resource(x, max_size=True) for x in (DQ, DK, DV))
            cos_r, sin_r = (create_buffer_resource(x, max_size=True) for x in (COS, SIN))
            part_r = create_buffer_resource(PART, max_size=True)
            inv_d = fx.arith.constant_vector(1.0 / D, f32v)
            eps_v = fx.arith.constant_vector(eps, f32v)
            zero = fx.arith.constant_vector(0.0, f32v)
            neg1 = fx.arith.constant_vector(-1.0, f32v4)
            accs = [
                fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(5 if bias_grad else 2)
            ]

            i32v4 = ir.VectorType.get([4], ir.IntegerType.get_signless(32))

            def opaque(v):
                """v, which the compiler can no longer prove loop-invariant (an empty volatile asm)."""
                return _llvm.inline_asm(i32v4, [v], "", "=v,0", has_side_effects=True)

            def widen(p):
                return fx.arith.extf(f32v, _vector.bitcast(bf16v, p))

            def norm_rope_bwd(x_bf, go_bf, w, c, s_signed, b):
                x = fx.arith.extf(f32v, x_bf)
                if add_bias:  # the forward's input, as it formed it
                    x = fx.arith.addf(x, widen(b))
                    if ROUND:
                        x = fx.arith.extf(f32v, fx.arith.trunc_f(bf16v, x))
                go = fx.arith.extf(f32v, go_bf)
                rstd = fmath.rsqrt(
                    fx.arith.addf(fx.arith.mulf(row_sum(fx.arith.mulf(x, x), f32v), inv_d), eps_v)
                )
                xhat = fx.arith.mulf(x, rstd)
                ge, g_o = _vector.deinterleave(go)
                dy = fx.arith.addf(fx.arith.mulf(go, c), fx.arith.mulf(_vector.interleave(g_o, ge), s_signed))
                if ROUND:  # the gradient of the rounded norm output, rounded as autograd rounds it
                    dy = fx.arith.extf(f32v, fx.arith.trunc_f(bf16v, dy))
                g = fx.arith.mulf(dy, widen(w) if add_bias else w)
                dot = fx.arith.mulf(row_sum(fx.arith.mulf(g, xhat), f32v), inv_d)
                dx = fx.arith.trunc_f(bf16v, fx.arith.mulf(rstd, fx.arith.subf(g, fx.arith.mulf(xhat, dot))))
                return dx, fx.arith.mulf(dy, xhat)

            def accumulate(acc, v):
                fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(acc), v), acc)

            def store8(v, rsrc, off):
                for hf in (0, 1):
                    buffer_store(
                        _vector.extract_strided_slice(f32v4, v, [hf * 4], [4], [1]),
                        rsrc,
                        off + fx.Int32(hf * 4),
                    )

            for si, (n_tok, s_src, seq_off) in enumerate(segs):
                SRC, DSRC, WQ, WK, BIAS = (
                    (SRC0, DSRC0, WQ0, WK0, B0) if si == 0 else (SRC1, DSRC1, WQ1, WK1, B1)
                )
                src_r, dsrc_r = (create_buffer_resource(x, max_size=True) for x in (SRC, DSRC))
                wq, wk = (
                    buffer_load(create_buffer_resource(w, max_size=True), lane8, vec_width=VEC, dtype=bf16)
                    for w in (WQ, WK)
                )
                carried = ()
                if not add_bias:
                    wq, wk = fx.arith.extf(f32v, wq), fx.arith.extf(f32v, wk)
                else:
                    # w and the bias stay bf16 (4 VGPRs per vector), carried through the loop as opaque values
                    # and widened where used: hoisted out of the loop, their fp32 copies (16 VGPRs more) spill
                    # under the 128-VGPR cap next to the bias gradient's accumulators.
                    bias_r = create_buffer_resource(BIAS, max_size=True)
                    bq = buffer_load(bias_r, scol, vec_width=VEC, dtype=bf16)
                    bk = buffer_load(bias_r, scol + fx.Int32(KO - QO), vec_width=VEC, dtype=bf16)
                    carried = [_vector.bitcast(i32v4, v) for v in (wq, wk, bq, bk)]
                for a in accs:
                    fx.memref_store_vec(zero, a)

                def offs(it, s_src=s_src, seq_off=seq_off):
                    tok = bx + it * fx.Int32(GRID_X)
                    b = tok % fx.Int32(NB) if SB else tok // fx.Int32(s_src)
                    pos = (tok // fx.Int32(NB) if SB else tok % fx.Int32(s_src)) + fx.Int32(seq_off)
                    orow = pos * fx.Int32(NB) + b if SB else b * fx.Int32(s_out) + pos
                    cs_row = pos * fx.Int32(NB) + b if CSB else pos
                    return (
                        orow * fx.Int32(hd) + col,
                        tok * fx.Int32(TS) + scol,
                        cs_row * fx.Int32(CSS) + lane8,
                    )

                def trip(it, *carried, src_r=src_r, dsrc_r=dsrc_r, wq=wq, wk=wk, offs=offs):
                    go_off, src_off, cs_off = offs(it)

                    def ld(r, off):
                        return buffer_load(r, off, vec_width=VEC, dtype=bf16, cache_modifier=_NT)

                    # Issue order = first use: cos, sin, the q pair, the k pair, dv.
                    cb, sb = (buffer_load(r, cs_off, vec_width=VEC, dtype=bf16) for r in (cos_r, sin_r))
                    xq, gq = ld(src_r, src_off), ld(dq_r, go_off)
                    xk, gk = ld(src_r, src_off + fx.Int32(KO - QO)), ld(dk_r, go_off)
                    dv = ld(dv_r, go_off)
                    _xrocdl.sched_barrier(0)  # keep the seven loads together, ahead of the math
                    _, src_off, _ = offs(it)
                    bq = bk = None
                    if add_bias:
                        carried = [opaque(v) for v in carried]
                        wq, wk, bq, bk = carried
                    c = fx.arith.extf(f32v, cb)
                    s_e, s_o = _vector.deinterleave(fx.arith.extf(f32v, sb))
                    s_signed = _vector.interleave(s_e, fx.arith.mulf(s_o, neg1))
                    dxq, dwq = norm_rope_bwd(xq, gq, wq, c, s_signed, bq)
                    dxk, dwk = norm_rope_bwd(xk, gk, wk, c, s_signed, bk)
                    buffer_store(dxq, dsrc_r, src_off, cache_modifier=_NT)
                    buffer_store(dxk, dsrc_r, src_off + fx.Int32(KO - QO), cache_modifier=_NT)
                    buffer_store(dv, dsrc_r, src_off + fx.Int32(VO - QO), cache_modifier=_NT)
                    accumulate(accs[0], dwq)
                    accumulate(accs[1], dwk)
                    if bias_grad:
                        accumulate(accs[2], fx.arith.extf(f32v, dxq))
                        accumulate(accs[3], fx.arith.extf(f32v, dxk))
                        accumulate(accs[4], fx.arith.extf(f32v, dv))
                    return carried

                sfor(n_tok // GRID_X, trip, carried)

                if bias_grad:
                    bbase = bx * fx.Int32(ROWP) + fx.Int32(si * SEGW + 2 * D) + scol
                    for o, a in zip((0, KO - QO, VO - QO), accs[2:]):
                        store8(fx.memref_load_vec(a), part_r, bbase + fx.Int32(o))
                # dw: sum the block's heads in LDS (h0 + h1 + ... in order); one [2, D] row per block.
                fx.barrier()  # the previous segment's readers are done with red
                for k_, a in ((0, accs[0]), (1, accs[1])):
                    p = fx.add_offset(
                        red.ptr,
                        fx.make_int_tuple(
                            ((t // fx.Int32(LANES)) * fx.Int32(2) + fx.Int32(k_)) * fx.Int32(D) + lane8
                        ),
                    )
                    fx.make_view(p, fx.make_layout(VEC, 1)).store(Vec(fx.memref_load_vec(a)))
                fx.barrier()
                cidx = t % fx.Int32(2 * D)  # column of [q | k]; threads >= 2D repeat a column (same value)
                tot = None
                for h in range(H):
                    p = fx.add_offset(red.ptr, fx.make_int_tuple(fx.Int32(h * 2 * D) + cidx))
                    v = ArithValue(Vec(fx.make_view(p, fx.make_layout(1, 1)).load())[0])
                    tot = v if tot is None else tot.addf(v)
                # Threads >= 2D store the same bits to the same address as thread t - 2D. No mask: a masked buffer
                # store relies on the descriptor's range check, which max_size resources do not have.
                buffer_store(tot, part_r, bx * fx.Int32(ROWP) + fx.Int32(si * SEGW) + cidx)

        @flyc.kernel(name="dit_qk_norm_rope_bwd", known_block_size=[threads, 1, 1])
        def dit_qk_norm_rope_bwd(
            SRC0: fx.Tensor,
            SRC1: fx.Tensor,
            DSRC0: fx.Tensor,
            DSRC1: fx.Tensor,
            DQ: fx.Tensor,
            DK: fx.Tensor,
            DV: fx.Tensor,
            WQ0: fx.Tensor,
            WK0: fx.Tensor,
            WQ1: fx.Tensor,
            WK1: fx.Tensor,
            COS: fx.Tensor,
            SIN: fx.Tensor,
            B0: fx.Tensor,
            B1: fx.Tensor,
            PART: fx.Tensor,
        ):
            @fx.struct
            class Smem:
                red: fx.Array[fx.Float32, 2 * H * D, 16]

            red = fx.SharedAllocator().allocate(Smem).peek().red
            emit(SRC0, SRC1, DSRC0, DSRC1, DQ, DK, DV, WQ0, WK0, WQ1, WK1, COS, SIN, B0, B1, PART, red)

        return dit_qk_norm_rope_bwd

    @flyc.jit
    def launch(
        SRC0,
        SRC1,
        DSRC0,
        DSRC1,
        DQ,
        DK,
        DV,
        WQ0,
        WK0,
        WQ1,
        WK1,
        COS,
        SIN,
        B0,
        B1,
        PART,
        H: fx.Constexpr[int],
        NSEG: fx.Constexpr[int],
        n_tok0: fx.Constexpr[int],
        s_src0: fx.Constexpr[int],
        n_tok1: fx.Constexpr[int],
        s_src1: fx.Constexpr[int],
        s_out: fx.Constexpr[int],
        eps: fx.Constexpr[float],
        add_bias: fx.Constexpr[int],
        bias_grad: fx.Constexpr[int],
        TS: fx.Constexpr[int],
        HS: fx.Constexpr[int],
        QO: fx.Constexpr[int],
        KO: fx.Constexpr[int],
        VO: fx.Constexpr[int],
        NB: fx.Constexpr[int],
        SB: fx.Constexpr[int],
        CSS: fx.Constexpr[int],
        CSB: fx.Constexpr[int],
        ROUND: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        segs = [(n_tok0, s_src0, 0), (n_tok1, s_src1, s_src0)][:NSEG]
        kernel = make(
            H,
            NSEG,
            segs,
            s_out,
            eps,
            bool(add_bias),
            bool(bias_grad),
            TS,
            HS,
            QO,
            KO,
            VO,
            NB,
            bool(SB),
            CSS,
            bool(CSB),
            bool(ROUND),
        )
        kernel(SRC0, SRC1, DSRC0, DSRC1, DQ, DK, DV, WQ0, WK0, WQ1, WK1, COS, SIN, B0, B1, PART).launch(
            grid=(GRID_X, 1, 1), block=(H * LANES, 1, 1), stream=stream
        )

    launch.compile_hints = {**launch.compile_hints, "waves_per_eu": WAVES_PER_EU}
    return launch


def partial_row_width(H: int, nseg: int, bias_grad: bool) -> int:
    """Floats per block row of the partial buffer: per segment [2D dw | 3HD bias]."""
    return nseg * (2 * D + (3 * H * D if bias_grad else 0))


def dit_qk_norm_rope_bwd_launch(
    srcs,
    dsrcs,
    dq,
    dk,
    dv,
    ws,
    cos,
    sin,
    part,
    *,
    H,
    s_out,
    eps,
    biases,
    bias_grad,
    layout,
    cs_row_stride,
    cs_per_batch,
    round_normed,
    stream,
):
    """Launch the backward over one or two segments (text first). ``layout`` holds the addressing of the
    sources (``TS``, ``HS``, ``QO``, ``KO``, ``VO``, ``SB``, ``NB``) and per segment ``(n_tok, s_src)``; ``biases``
    is None or the bias of every segment (bf16, ``[3*H*D]`` in the source's column order), added where x is
    recomputed; ``part`` is fp32 ``[GRID_X, partial_row_width(H, len(srcs), bias_grad)]``."""
    nseg = len(srcs)
    (wq0, wk0), (wq1, wk1) = ws[0], ws[-1]
    (n_tok0, s_src0), (n_tok1, s_src1) = layout["segs"][0], layout["segs"][-1]
    b0, b1 = (biases[0], biases[-1]) if biases is not None else (wq0, wq0)  # unread without the bias
    _build()(
        srcs[0],
        srcs[-1],
        dsrcs[0],
        dsrcs[-1],
        dq,
        dk,
        dv,
        wq0,
        wk0,
        wq1,
        wk1,
        cos,
        sin,
        b0,
        b1,
        part,
        H=H,
        NSEG=nseg,
        n_tok0=n_tok0,
        s_src0=s_src0,
        n_tok1=n_tok1 if nseg == 2 else 0,
        s_src1=s_src1 if nseg == 2 else 0,
        s_out=s_out,
        eps=float(eps),
        add_bias=int(biases is not None),
        bias_grad=int(bias_grad),
        TS=layout["TS"],
        HS=layout["HS"],
        QO=layout["QO"],
        KO=layout["KO"],
        VO=layout["VO"],
        NB=layout["NB"],
        SB=layout["SB"],
        CSS=cs_row_stride,
        CSB=int(cs_per_batch),
        ROUND=int(round_normed),
        stream=stream,
    )
