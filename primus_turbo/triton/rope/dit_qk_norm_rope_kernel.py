###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton kernels of ``dit_qk_norm_rope``: q / k RMSNorm + RoPE (+ the v copy) between a DiT's QKV GEMM and
attention, forward and backward, for both token orders and one or two streams.

The ops and the layout rules are in ``primus_turbo/pytorch/kernels/rope/dit_qk_norm_rope_impl.py``.

Addressing. A token is one row of the QKV GEMM output. ``TS`` is the token stride, ``HS`` the head stride and
``QO`` / ``KO`` / ``VO`` the q / k / v column offsets inside a token row, so the same kernel reads

    diffusers  [B, S, 3*H*D]  (TS = 3*H*D, HS = D,   QO, KO, VO = 0, H*D, 2*H*D), tokens in (b, s) order, and
    Megatron   [S, B, H, 3D]  (TS = 3*H*D, HS = 3*D, QO, KO, VO = 0, D, 2*D),     tokens in (s, b) order,

in place. The outputs q / k / v are [B, S_out, H, D] or [S_out, B, H, D] (the source's token order). With two
streams (text first) both are written into one joint output in one launch: programs [0, n_prog0) take the text
segment, the rest the image segment at sequence offset ``s_src0``.

Arithmetic. Per (token, head): rstd over D in fp32, ``y = x * rstd * w``, RoPE in fp32, one rounding of the
result. ``ROUND`` also rounds ``y`` to the storage dtype before RoPE, and the inverse-RoPE gradient before the
RMSNorm backward, which is what an unfused RMSNorm module followed by a RoPE function computes (diffusers'
Flux, #556). ``INTERLEAVED`` pairs (2i, 2i+1), as Megatron's ``rotary_interleaved`` and diffusers' Flux do;
otherwise (i, i + D/2). The cos / sin rows hold each pair's angle twice, as both libraries build them; the
kernels read the first copy.

``ADD_BIAS``: the QKV GEMM ran without its bias and the kernels add it, in the source's column order (``bias[QO +
h * HS + d]`` for q, likewise k and v), in fp32 to every q / k / v value they read; ``ROUND`` rounds the sum to
the storage dtype first, as a separate bias add does. The v copy is ``v + bias`` rounded once. The backward
adds the bias again where it recomputes rstd.

The backward writes d(qkv) in the source layout (dv into the v columns), and fp32 partials of the norm-weight
gradients and, with ``BIAS_GRAD``, of the QKV bias gradient (the column sums of d(qkv)). ``reduce_partials_kernel``
turns the partials of every stream into dw and db in one launch, in a fixed order, so the result is
deterministic.
"""

import triton
import triton.language as tl


@triton.jit
def _rope(x, c, s, BLOCK_H: tl.constexpr, D: tl.constexpr, INTERLEAVED: tl.constexpr, INVERSE: tl.constexpr):
    """Rotate each pair of x [BLOCK_H, D] (fp32) by its angle; ``INVERSE`` rotates by minus the angle (the
    gradient of the rotation). c / s are [D // 2]."""
    if INTERLEAVED:
        a, b = tl.split(tl.reshape(x, (BLOCK_H, D // 2, 2)))
    else:
        a, b = tl.split(tl.permute(tl.reshape(x, (BLOCK_H, 2, D // 2)), (0, 2, 1)))
    if INVERSE:
        ya = a * c[None, :] + b * s[None, :]
        yb = b * c[None, :] - a * s[None, :]
    else:
        ya = a * c[None, :] - b * s[None, :]
        yb = b * c[None, :] + a * s[None, :]
    if INTERLEAVED:
        return tl.reshape(tl.join(ya, yb), (BLOCK_H, D))
    return tl.reshape(tl.permute(tl.join(ya, yb), (0, 2, 1)), (BLOCK_H, D))


@triton.jit
def _token(
    tok,
    s_src,
    seq_off,
    s_out,
    NB,
    H: tl.constexpr,
    D: tl.constexpr,
    SB: tl.constexpr,
    CS_PER_BATCH: tl.constexpr,
):
    """(output offset of the token's row, cos / sin row) for token ``tok`` of a segment."""
    if SB:  # Megatron: tok = s * B + b
        pos = tok // NB + seq_off
        b = tok % NB
        dst = (pos.to(tl.int64) * NB + b) * (H * D)
    else:  # diffusers: tok = b * S + s
        b = tok // s_src
        pos = tok % s_src + seq_off
        dst = (b.to(tl.int64) * s_out + pos) * (H * D)
    if CS_PER_BATCH:
        cs_row = pos * NB + b
    else:
        cs_row = pos
    return dst, cs_row


@triton.jit
def _norm_rope(
    x_ptrs,
    out_ptrs,
    w,
    b,
    c,
    s,
    eps,
    hmask,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROUND: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    CM: tl.constexpr,
):
    x = tl.load(x_ptrs, mask=hmask, other=0.0, cache_modifier=CM).to(tl.float32)
    if ADD_BIAS:
        x = x + b
        if ROUND:
            x = x.to(out_ptrs.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = x * rstd[:, None] * w[None, :]
    if ROUND:
        y = y.to(out_ptrs.dtype.element_ty).to(tl.float32)
    out = _rope(y, c, s, BLOCK_H, D, INTERLEAVED, False)
    tl.store(out_ptrs, out.to(out_ptrs.dtype.element_ty), mask=hmask, cache_modifier=CM)


@triton.jit
def _fwd_segment(
    src_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    bias_ptr,
    prog,
    n_tok,
    s_src,
    seq_off,
    s_out,
    eps,
    TS,
    HS,
    QO,
    KO,
    VO,
    NB,
    CSS,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOKENS: tl.constexpr,
    SB: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROUND: tl.constexpr,
    V_COPY: tl.constexpr,
    CS_PER_BATCH: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    CM: tl.constexpr,
):
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, D)
    hmask = (h < H)[:, None]
    hs = h[:, None] * HS + d[None, :]
    hd = h[:, None] * D + d[None, :]
    if INTERLEAVED:
        cs_cols = tl.arange(0, D // 2) * 2
    else:
        cs_cols = tl.arange(0, D // 2)
    wq = tl.load(wq_ptr + d).to(tl.float32)
    wk = tl.load(wk_ptr + d).to(tl.float32)
    bq = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    bk = bq
    bv = bq
    if ADD_BIAS:
        bq = tl.load(bias_ptr + QO + hs, mask=hmask, other=0.0).to(tl.float32)
        bk = tl.load(bias_ptr + KO + hs, mask=hmask, other=0.0).to(tl.float32)
        bv = tl.load(bias_ptr + VO + hs, mask=hmask, other=0.0).to(tl.float32)
    for i in tl.static_range(TOKENS):
        tok = prog * TOKENS + i
        if tok < n_tok:
            dst, cs_row = _token(tok, s_src, seq_off, s_out, NB, H, D, SB, CS_PER_BATCH)
            src = src_ptr + tok.to(tl.int64) * TS
            c = tl.load(cos_ptr + cs_row * CSS + cs_cols).to(tl.float32)
            s = tl.load(sin_ptr + cs_row * CSS + cs_cols).to(tl.float32)
            _norm_rope(
                src + QO + hs, q_ptr + dst + hd, wq, bq, c, s, eps, hmask, BLOCK_H, D, INTERLEAVED, ROUND,
                ADD_BIAS, CM,
            )  # fmt: skip
            _norm_rope(
                src + KO + hs, k_ptr + dst + hd, wk, bk, c, s, eps, hmask, BLOCK_H, D, INTERLEAVED, ROUND,
                ADD_BIAS, CM,
            )  # fmt: skip
            if V_COPY:
                vv = tl.load(src + VO + hs, mask=hmask, cache_modifier=CM)
                if ADD_BIAS:
                    vv = (vv.to(tl.float32) + bv).to(v_ptr.dtype.element_ty)
                tl.store(v_ptr + dst + hd, vv, mask=hmask, cache_modifier=CM)


@triton.jit
def dit_qk_norm_rope_fwd_kernel(
    src0_ptr,
    src1_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    wq0_ptr,
    wk0_ptr,
    wq1_ptr,
    wk1_ptr,
    cos_ptr,
    sin_ptr,
    b0_ptr,
    b1_ptr,
    n_tok0,
    s_src0,
    n_tok1,
    s_src1,
    n_prog0,
    s_out,
    eps,
    TS,
    HS,
    QO,
    KO,
    VO,
    NB,
    CSS,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOKENS: tl.constexpr,
    SB: tl.constexpr,
    NSEG: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROUND: tl.constexpr,
    V_COPY: tl.constexpr,
    CS_PER_BATCH: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    CM: tl.constexpr,
):
    """Grid (n_prog0 [+ n_prog1], cdiv(H, BLOCK_H)); a program normalises and rotates TOKENS tokens' q and k
    for BLOCK_H heads and copies their v."""
    pid = tl.program_id(0)
    if NSEG == 2:
        if pid < n_prog0:
            _fwd_segment(
                src0_ptr,
                q_ptr,
                k_ptr,
                v_ptr,
                wq0_ptr,
                wk0_ptr,
                cos_ptr,
                sin_ptr,
                b0_ptr,
                pid,
                n_tok0,
                s_src0,
                0,
                s_out,
                eps,
                TS,
                HS,
                QO,
                KO,
                VO,
                NB,
                CSS,
                H,
                D,
                BLOCK_H,
                TOKENS,
                SB,
                INTERLEAVED,
                ROUND,
                V_COPY,
                CS_PER_BATCH,
                ADD_BIAS,
                CM,
            )
        else:
            _fwd_segment(
                src1_ptr,
                q_ptr,
                k_ptr,
                v_ptr,
                wq1_ptr,
                wk1_ptr,
                cos_ptr,
                sin_ptr,
                b1_ptr,
                pid - n_prog0,
                n_tok1,
                s_src1,
                s_src0,
                s_out,
                eps,
                TS,
                HS,
                QO,
                KO,
                VO,
                NB,
                CSS,
                H,
                D,
                BLOCK_H,
                TOKENS,
                SB,
                INTERLEAVED,
                ROUND,
                V_COPY,
                CS_PER_BATCH,
                ADD_BIAS,
                CM,
            )
    else:
        _fwd_segment(
            src0_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            wq0_ptr,
            wk0_ptr,
            cos_ptr,
            sin_ptr,
            b0_ptr,
            pid,
            n_tok0,
            s_src0,
            0,
            s_out,
            eps,
            TS,
            HS,
            QO,
            KO,
            VO,
            NB,
            CSS,
            H,
            D,
            BLOCK_H,
            TOKENS,
            SB,
            INTERLEAVED,
            ROUND,
            V_COPY,
            CS_PER_BATCH,
            ADD_BIAS,
            CM,
        )


@triton.jit
def _norm_rope_bwd(
    x_ptrs,
    go_ptrs,
    dx_ptrs,
    w,
    b,
    c,
    s,
    eps,
    hmask,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROUND: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    x = tl.load(x_ptrs, mask=hmask, other=0.0).to(tl.float32)
    if ADD_BIAS:  # the forward's input, as it formed it
        x = x + b
        if ROUND:
            x = x.to(dx_ptrs.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    xhat = x * rstd[:, None]
    go = tl.load(go_ptrs, mask=hmask, other=0.0).to(tl.float32)
    dy = _rope(go, c, s, BLOCK_H, D, INTERLEAVED, True)
    if ROUND:  # the gradient of the rounded norm output, rounded as autograd rounds it
        dy = dy.to(dx_ptrs.dtype.element_ty).to(tl.float32)
    g = dy * w[None, :]
    dx = (rstd[:, None] * (g - xhat * (tl.sum(g * xhat, axis=1) / D)[:, None])).to(dx_ptrs.dtype.element_ty)
    tl.store(dx_ptrs, dx, mask=hmask)
    return tl.sum(dy * xhat, axis=0), dx.to(tl.float32)


@triton.jit
def _bwd_segment(
    src_ptr,
    dsrc_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    bias_ptr,
    part_ptr,
    bpart_ptr,
    prog,
    row0,
    n_tok,
    s_src,
    seq_off,
    s_out,
    eps,
    tok_per_prog,
    TS,
    HS,
    QO,
    KO,
    VO,
    NB,
    CSS,
    ROW_W,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    BIAS_GRAD: tl.constexpr,
    SB: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROUND: tl.constexpr,
    CS_PER_BATCH: tl.constexpr,
):
    pid_h = tl.program_id(1)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, D)
    hmask = (h < H)[:, None]
    hs = h[:, None] * HS + d[None, :]
    hd = h[:, None] * D + d[None, :]
    if INTERLEAVED:
        cs_cols = tl.arange(0, D // 2) * 2
    else:
        cs_cols = tl.arange(0, D // 2)
    wq = tl.load(wq_ptr + d).to(tl.float32)
    wk = tl.load(wk_ptr + d).to(tl.float32)
    bq = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    bk = bq
    if ADD_BIAS:
        bq = tl.load(bias_ptr + QO + hs, mask=hmask, other=0.0).to(tl.float32)
        bk = tl.load(bias_ptr + KO + hs, mask=hmask, other=0.0).to(tl.float32)
    acc_q = tl.zeros([D], dtype=tl.float32)
    acc_k = tl.zeros([D], dtype=tl.float32)
    # Column sums of d(qkv) as written (the QKV bias gradient).
    acc_bq = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    acc_bk = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    acc_bv = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    t0 = prog * tok_per_prog
    t1 = tl.minimum(t0 + tok_per_prog, n_tok)
    for tok in range(t0, t1):
        dst, cs_row = _token(tok, s_src, seq_off, s_out, NB, H, D, SB, CS_PER_BATCH)
        row = tok.to(tl.int64) * TS
        c = tl.load(cos_ptr + cs_row * CSS + cs_cols).to(tl.float32)
        s = tl.load(sin_ptr + cs_row * CSS + cs_cols).to(tl.float32)
        dw_q, dx_q = _norm_rope_bwd(
            src_ptr + row + QO + hs,
            dq_ptr + dst + hd,
            dsrc_ptr + row + QO + hs,
            wq,
            bq,
            c,
            s,
            eps,
            hmask,
            BLOCK_H,
            D,
            INTERLEAVED,
            ROUND,
            ADD_BIAS,
        )
        dw_k, dx_k = _norm_rope_bwd(
            src_ptr + row + KO + hs,
            dk_ptr + dst + hd,
            dsrc_ptr + row + KO + hs,
            wk,
            bk,
            c,
            s,
            eps,
            hmask,
            BLOCK_H,
            D,
            INTERLEAVED,
            ROUND,
            ADD_BIAS,
        )
        acc_q += dw_q
        acc_k += dw_k
        dv = tl.load(dv_ptr + dst + hd, mask=hmask)
        tl.store(dsrc_ptr + row + VO + hs, dv, mask=hmask)
        if BIAS_GRAD:
            acc_bq += dx_q
            acc_bk += dx_k
            acc_bv += dv.to(tl.float32)
    prow = row0 + prog
    base = part_ptr + (prow * tl.num_programs(1) + pid_h) * (2 * D)
    tl.store(base + d, acc_q)
    tl.store(base + D + d, acc_k)
    if BIAS_GRAD:
        bbase = bpart_ptr + prow.to(tl.int64) * ROW_W
        tl.store(bbase + QO + hs, acc_bq, mask=hmask)
        tl.store(bbase + KO + hs, acc_bk, mask=hmask)
        tl.store(bbase + VO + hs, acc_bv, mask=hmask)


@triton.jit
def dit_qk_norm_rope_bwd_kernel(
    src0_ptr,
    src1_ptr,
    dsrc0_ptr,
    dsrc1_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    wq0_ptr,
    wk0_ptr,
    wq1_ptr,
    wk1_ptr,
    cos_ptr,
    sin_ptr,
    b0_ptr,
    b1_ptr,
    part_ptr,
    bpart_ptr,
    n_tok0,
    s_src0,
    n_tok1,
    s_src1,
    n_prog0,
    s_out,
    eps,
    per0,
    per1,
    TS,
    HS,
    QO,
    KO,
    VO,
    NB,
    CSS,
    ROW_W,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    BIAS_GRAD: tl.constexpr,
    SB: tl.constexpr,
    NSEG: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROUND: tl.constexpr,
    CS_PER_BATCH: tl.constexpr,
):
    """Grid (n_prog0 [+ n_prog1], cdiv(H, BLOCK_H)); a program walks ``per`` consecutive tokens of its segment
    and writes one partial row: dw [prow, pid_h, 2, D] and, with BIAS_GRAD, db [prow, 3*H*D] (source column
    order)."""
    pid = tl.program_id(0)
    if NSEG == 2:
        if pid < n_prog0:
            _bwd_segment(
                src0_ptr,
                dsrc0_ptr,
                dq_ptr,
                dk_ptr,
                dv_ptr,
                wq0_ptr,
                wk0_ptr,
                cos_ptr,
                sin_ptr,
                b0_ptr,
                part_ptr,
                bpart_ptr,
                pid,
                0,
                n_tok0,
                s_src0,
                0,
                s_out,
                eps,
                per0,
                TS,
                HS,
                QO,
                KO,
                VO,
                NB,
                CSS,
                ROW_W,
                H,
                D,
                BLOCK_H,
                ADD_BIAS,
                BIAS_GRAD,
                SB,
                INTERLEAVED,
                ROUND,
                CS_PER_BATCH,
            )
        else:
            _bwd_segment(
                src1_ptr,
                dsrc1_ptr,
                dq_ptr,
                dk_ptr,
                dv_ptr,
                wq1_ptr,
                wk1_ptr,
                cos_ptr,
                sin_ptr,
                b1_ptr,
                part_ptr,
                bpart_ptr,
                pid - n_prog0,
                n_prog0,
                n_tok1,
                s_src1,
                s_src0,
                s_out,
                eps,
                per1,
                TS,
                HS,
                QO,
                KO,
                VO,
                NB,
                CSS,
                ROW_W,
                H,
                D,
                BLOCK_H,
                ADD_BIAS,
                BIAS_GRAD,
                SB,
                INTERLEAVED,
                ROUND,
                CS_PER_BATCH,
            )
    else:
        _bwd_segment(
            src0_ptr,
            dsrc0_ptr,
            dq_ptr,
            dk_ptr,
            dv_ptr,
            wq0_ptr,
            wk0_ptr,
            cos_ptr,
            sin_ptr,
            b0_ptr,
            part_ptr,
            bpart_ptr,
            pid,
            0,
            n_tok0,
            s_src0,
            0,
            s_out,
            eps,
            per0,
            TS,
            HS,
            QO,
            KO,
            VO,
            NB,
            CSS,
            ROW_W,
            H,
            D,
            BLOCK_H,
            ADD_BIAS,
            BIAS_GRAD,
            SB,
            INTERLEAVED,
            ROUND,
            CS_PER_BATCH,
        )


@triton.jit
def reduce_partials_kernel(
    part_ptr,
    bpart_ptr,
    out_ptr,
    n_prog0,
    n_prog1,
    n_hblk,
    OUT_W,
    ROW_W,
    D2: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """dw and db of every stream from the Triton backward's partials, in one launch.

    Grid (cdiv(OUT_W, BLOCK_C), NSEG); ``out`` is [NSEG, OUT_W] with OUT_W = 2*D (+ 3*H*D with the bias). A
    column block lies wholly in the dw columns (D2 = 2*D, a multiple of BLOCK_C) or wholly in the bias columns.
    Each column is summed in a fixed order: BLOCK_R rows at a time (and, for dw, the head blocks of a row in
    order), then across the BLOCK_R lanes."""
    pid_c, seg = tl.program_id(0), tl.program_id(1)
    r_start = tl.where(seg == 0, 0, n_prog0)
    n_rows = tl.where(seg == 0, n_prog0, n_prog1)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    if pid_c * BLOCK_C < D2:
        for r0 in range(0, n_rows, BLOCK_R):
            rows = (r_start + r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
            rmask = (r0 + tl.arange(0, BLOCK_R)) < n_rows
            for hb in range(n_hblk):
                acc += tl.load(
                    part_ptr + (rows[:, None] * n_hblk + hb) * D2 + cols[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
    else:
        bcols = cols - D2
        cmask = bcols < ROW_W
        for r0 in range(0, n_rows, BLOCK_R):
            rows = (r_start + r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
            rmask = (r0 + tl.arange(0, BLOCK_R)) < n_rows
            acc += tl.load(
                bpart_ptr + rows[:, None] * ROW_W + bcols[None, :],
                mask=rmask[:, None] & cmask[None, :],
                other=0.0,
            )
    tl.store(
        out_ptr + seg * OUT_W + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=cols < OUT_W
    )


@triton.jit
def colsum_kernel(src_ptr, out_ptr, n_rows, n_cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """out[c] = sum_r src[r, c] over fp32 block rows, BLOCK_R rows at a time (the FlyDSL backward's dw / db)."""
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    for r0 in range(0, n_rows, BLOCK_R):
        rows = (r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        acc += tl.load(
            src_ptr + rows[:, None] * n_cols + cols[None, :],
            mask=(rows < n_rows)[:, None] & col_mask[None, :],
            other=0.0,
        )
    tl.store(out_ptr + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=col_mask)
