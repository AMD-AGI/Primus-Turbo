###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus_turbo::dit_qk_norm_rope`` / ``dit_qk_norm_rope_backward``: operand checks, layout, dispatch.

Kernels: Triton forward and backward (every arch) in ``primus_turbo/triton/rope/dit_qk_norm_rope_kernel.py``;
FlyDSL backward (gfx950) in ``primus_turbo/flydsl/rope/dit_qk_norm_rope_kernel.py``. The public function is
``primus_turbo.pytorch.ops.rope.dit_qk_norm_rope``.

Source forms. ``layout`` names the token order of the QKV GEMM output and of q / k / v:

- ``"sbhd"`` (Megatron): ``[S, B, H, 3*D]`` (per head q | k | v) or ``[S, B, 3*H*D]`` -> q, k, v ``[S, B, H, D]``;
- ``"bshd"`` (diffusers): ``[B, S, 3*H*D]`` (q heads | k heads | v heads) or ``[B, S, H, 3*D]`` -> ``[B, S, H, D]``.

The last dimension needs a unit stride and the two token dimensions one uniform token stride; anything else is
read in place at its strides (a slice of a wider GEMM output, for example). With a text stream, its tokens come
first in the joint sequence of the outputs, and it must have the image stream's form and strides.
"""

from typing import List, Optional, Tuple

import torch
import triton

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.rope.dit_qk_norm_rope_kernel import (
    colsum_kernel,
    dit_qk_norm_rope_bwd_kernel,
    dit_qk_norm_rope_fwd_kernel,
    reduce_partials_kernel,
)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_INDEX = 2**31  # the FlyDSL kernel and the cos / sin rows use 32-bit offsets
_LAYOUTS = ("sbhd", "bshd")
_ROPES = ("interleaved", "half")

# Triton launch shape: BLOCK_H heads per program; forward TOKENS tokens per program;
# backward programs per stream (fewer with the bias: each writes a 3*H*D-wide fp32 row of partial sums).
_BLOCK_H = 8
_FWD_TOKENS = 4
_BWD_PROGRAMS = 2048
_BWD_PROGRAMS_BIAS = 512
_RED_BLOCK_R = 32
_RED_BLOCK_C = 64


def _err(msg: str) -> ValueError:
    return ValueError(f"dit_qk_norm_rope: {msg}")


def _span(t: torch.Tensor) -> int:
    """Elements from t's first to one past its last addressed element."""
    return 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()) if n > 0) if t.numel() else 0


def source_layout(x: torch.Tensor, D: int, layout: str) -> dict:
    """Addressing of one stream's QKV GEMM output: token stride TS, head stride HS, q / k / v column offsets,
    token order SB (1: tok = s * B + b), batch NB, sequence S, heads H, tokens n_tok, row width ROW_W."""
    if layout not in _LAYOUTS:
        raise _err(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    if x.dim() == 4:  # [T0, T1, H, 3D]: q | k | v per head
        T0, T1, H, W = x.shape
        if W != 3 * D or x.stride(3) != 1:
            raise _err(f"a 4-D qkv must be [.., .., H, 3*{D}] with unit last stride, got {list(x.shape)}")
        HS = x.stride(2) if H > 1 else 3 * D
        if HS < 3 * D:
            raise _err(f"qkv heads overlap (head stride {HS} < {3 * D})")
        QO, KO, VO, row = 0, D, 2 * D, (H - 1) * HS + 3 * D
    elif x.dim() == 3:  # [T0, T1, 3HD]: q heads | k heads | v heads
        T0, T1, W = x.shape
        if W % (3 * D) or x.stride(2) != 1:
            raise _err(f"a 3-D qkv must be [.., .., 3*H*{D}] with unit last stride, got {list(x.shape)}")
        H = W // (3 * D)
        HS, QO, KO, VO, row = D, 0, H * D, 2 * H * D, W
    else:
        raise _err(f"qkv must be 3-D or 4-D, got {list(x.shape)}")
    TS = x.stride(1) if T1 > 1 else x.stride(0)
    if T0 > 1 and T1 > 1 and x.stride(0) != T1 * TS:
        raise _err(
            f"the two token dimensions of qkv must have one uniform token stride, got {list(x.stride())}"
        )
    if T0 * T1 > 1 and TS < row:
        raise _err(f"qkv tokens overlap (token stride {TS} < row width {row})")
    S, B = (T0, T1) if layout == "sbhd" else (T1, T0)
    return dict(
        TS=TS,
        HS=HS,
        QO=QO,
        KO=KO,
        VO=VO,
        SB=int(layout == "sbhd"),
        NB=B,
        S=S,
        H=H,
        n_tok=S * B,
        ROW_W=3 * H * D,
        dense_row=HS == 3 * D or x.dim() == 3,
    )


def check_operands(qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, layout, rope, bias_grad=False):
    """Validate every operand on the host (no device sync); returns (layouts per stream, text first; D; S_out;
    cos / sin per (position, batch))."""
    if rope not in _ROPES:
        raise _err(f"rope must be one of {_ROPES}, got {rope!r}")
    if qkv.dtype not in _DTYPES:
        raise _err(f"qkv dtype {qkv.dtype} is not supported (fp16, bf16 or fp32)")
    if w_q.dim() != 1:
        raise _err(f"w_q must be [D], got {list(w_q.shape)}")
    D = w_q.shape[0]
    if D < 16 or D > 256 or D & (D - 1):
        raise _err(f"head_dim must be a power of two in [16, 256], got {D}")
    streams = [(qkv, w_q, w_k, "")]
    if txt_qkv is not None:
        if w_txt_q is None or w_txt_k is None:
            raise _err("a text stream needs w_txt_q and w_txt_k")
        streams.insert(0, (txt_qkv, w_txt_q, w_txt_k, "txt_"))
    elif w_txt_q is not None or w_txt_k is not None:
        raise _err("w_txt_q / w_txt_k given without txt_qkv")
    metas = []
    for x, wq, wk, pfx in streams:
        if x.dtype != qkv.dtype or x.device != qkv.device:
            raise _err(f"{pfx}qkv must have qkv's dtype and device")
        for name, w in ((f"w_{pfx}q", wq), (f"w_{pfx}k", wk)):
            if w.shape != (D,) or w.stride(0) != 1 or w.dtype != qkv.dtype or w.device != qkv.device:
                raise _err(
                    f"{name} must be [{D}], unit stride, qkv's dtype {qkv.dtype} and device; got "
                    f"{list(w.shape)} {w.dtype} {w.device}"
                )
        m = source_layout(x, D, layout)
        if bias_grad and not m["dense_row"]:
            raise _err("bias_grad needs each head's q | k | v columns adjacent (head stride 3*D)")
        metas.append(m)
    if len(metas) == 2:
        a, b = metas
        if any(a[k] != b[k] for k in ("TS", "HS", "QO", "KO", "VO", "NB", "H")) or txt_qkv.dim() != qkv.dim():
            raise _err("txt_qkv and qkv must have the same form, batch, heads and strides")
    S_out = sum(m["S"] for m in metas)
    B = metas[0]["NB"]
    for name, t in (("cos", cos), ("sin", sin)):
        if (
            t.dim() != 2
            or t.shape[1] != D
            or t.stride(1) != 1
            or t.dtype not in _DTYPES
            or t.device != qkv.device
        ):
            raise _err(
                f"{name} must be [rows, {D}] with unit last stride on qkv's device, got {list(t.shape)} "
                f"strides {list(t.stride())}"
            )
    if cos.shape != sin.shape or cos.stride() != sin.stride() or cos.dtype != sin.dtype:
        raise _err("cos and sin must have the same shape, strides and dtype")
    if cos.shape[0] not in (S_out, S_out * B):
        raise _err(
            f"cos / sin need {S_out} rows (per position) or {S_out * B} (per position and batch element, "
            f"row = s * B + b), got {cos.shape[0]}"
        )
    if _span(cos) >= _MAX_INDEX:
        raise _err("cos / sin span 2^31 elements or more")
    return metas, D, S_out, cos.shape[0] == S_out * B and B > 1


def _out_shape(layout, S_out, B, H, D):
    return (S_out, B, H, D) if layout == "sbhd" else (B, S_out, H, D)


def _cs_row_stride(cos):
    return cos.stride(0) if cos.shape[0] > 1 else cos.shape[1]


# --------------------------------------------------------------------------------------------------- forward
def dit_qk_norm_rope_fwd_impl(
    qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, eps, layout, rope, round_normed, v_copy
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    metas, D, S_out, cs_per_batch = check_operands(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, layout, rope
    )
    m0 = metas[0]
    H, B = m0["H"], m0["NB"]
    shape = _out_shape(layout, S_out, B, H, D)
    q = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    v = torch.empty(shape if v_copy else (0,), device=qkv.device, dtype=qkv.dtype)
    if q.numel() == 0:
        return q, k, v
    srcs = [qkv] if txt_qkv is None else [txt_qkv, qkv]
    ws = [(w_q, w_k)] if txt_qkv is None else [(w_txt_q, w_txt_k), (w_q, w_k)]
    m1 = metas[-1]
    p0, p1 = triton.cdiv(m0["n_tok"], _FWD_TOKENS), triton.cdiv(m1["n_tok"], _FWD_TOKENS)
    nseg = len(srcs)
    # gfx950: streaming (non-temporal) loads and stores; every byte is touched once.
    cm = ".cs" if qkv.is_cuda and is_gfx950() else ""
    dit_qk_norm_rope_fwd_kernel[(p0 + (p1 if nseg == 2 else 0), triton.cdiv(H, _BLOCK_H))](
        srcs[0], srcs[-1], q, k, v, ws[0][0], ws[0][1], ws[-1][0], ws[-1][1], cos, sin,
        m0["n_tok"], m0["S"], m1["n_tok"], m1["S"], p0, S_out, eps,
        m0["TS"], m0["HS"], m0["QO"], m0["KO"], m0["VO"], B, _cs_row_stride(cos),
        H=H, D=D, BLOCK_H=_BLOCK_H, TOKENS=_FWD_TOKENS, SB=m0["SB"], NSEG=nseg,
        INTERLEAVED=rope == "interleaved", ROUND=bool(round_normed), V_COPY=bool(v_copy),
        CS_PER_BATCH=bool(cs_per_batch), CM=cm, num_warps=4,
    )  # fmt: skip
    return q, k, v


# --------------------------------------------------------------------------------------------------- backward
def _empty_like_strided(x):
    if x.is_contiguous():
        return torch.empty_like(x)
    return torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)


def flydsl_backward_supported(qkv, cos, metas, D, rope, S_out) -> bool:
    """Whether the gfx950 FlyDSL backward covers this call (otherwise the Triton backward runs)."""
    from primus_turbo.flydsl.rope.dit_qk_norm_rope_kernel import GRID_X
    from primus_turbo.flydsl.rope.dit_qk_norm_rope_kernel import D as FD

    m0 = metas[0]
    if not (qkv.is_cuda and is_gfx950()):
        return False
    if qkv.dtype != torch.bfloat16 or cos.dtype != torch.bfloat16 or D != FD or rope != "interleaved":
        return False
    if not 16 <= m0["H"] <= 64:  # 16 lanes per head, 2*D <= threads <= 1024
        return False
    if any(m["n_tok"] % GRID_X or m["n_tok"] == 0 for m in metas):
        return False
    n_out = S_out * m0["NB"] * m0["H"] * D
    if n_out >= _MAX_INDEX or any(m["n_tok"] * m["TS"] >= _MAX_INDEX for m in metas):
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def dit_qk_norm_rope_bwd_impl(
    dq,
    dk,
    dv,
    qkv,
    w_q,
    w_k,
    cos,
    sin,
    txt_qkv,
    w_txt_q,
    w_txt_k,
    eps,
    layout,
    rope,
    round_normed,
    bias_grad,
    backend: Optional[str] = None,
) -> List[torch.Tensor]:
    """[d_qkv, (d_txt_qkv,) red]: d(qkv) per stream in the source layout and ``red`` [NSEG, 2*D (+ 3*H*D)]: per
    stream (text first) dw_q | dw_k (| db in the source's column order). ``backend`` (tests): None = by arch and
    shape, "triton" or "flydsl"."""
    metas, D, S_out, cs_per_batch = check_operands(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, layout, rope, bias_grad
    )
    m0, m1 = metas[0], metas[-1]
    H, B = m0["H"], m0["NB"]
    shape = _out_shape(layout, S_out, B, H, D)
    for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
        if g.shape != shape or g.dtype != qkv.dtype or g.device != qkv.device:
            raise _err(f"{name} must be {list(shape)} {qkv.dtype}, got {list(g.shape)} {g.dtype}")
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
    srcs = [qkv] if txt_qkv is None else [txt_qkv, qkv]
    ws = [(w_q, w_k)] if txt_qkv is None else [(w_txt_q, w_txt_k), (w_q, w_k)]
    nseg = len(srcs)
    dsrcs = [_empty_like_strided(x) for x in srcs]
    row_w = m0["ROW_W"] if bias_grad else 0
    out_w = 2 * D + row_w
    red = torch.empty((nseg, out_w), device=qkv.device, dtype=qkv.dtype)
    if backend is None:
        backend = "flydsl" if flydsl_backward_supported(qkv, cos, metas, D, rope, S_out) else "triton"
    if backend == "flydsl":
        from primus_turbo.flydsl.rope.dit_qk_norm_rope_kernel import (
            GRID_X,
            dit_qk_norm_rope_bwd_launch,
            partial_row_width,
        )

        rowp = partial_row_width(H, nseg, bias_grad)
        part = torch.empty((GRID_X, rowp), device=qkv.device, dtype=torch.float32)
        lay = dict(m0, segs=[(m["n_tok"], m["S"]) for m in metas])
        dit_qk_norm_rope_bwd_launch(
            srcs, dsrcs, dq, dk, dv, ws, cos, sin, part, H=H, s_out=S_out, eps=eps, bias=bias_grad, layout=lay,
            cs_row_stride=_cs_row_stride(cos), cs_per_batch=cs_per_batch, round_normed=round_normed,
            stream=torch.cuda.current_stream(qkv.device),
        )  # fmt: skip
        # One launch: dw (and db) of every stream, from the block rows; per column a fixed order.
        colsum_kernel[(triton.cdiv(rowp, _RED_BLOCK_C),)](
            part, red, GRID_X, rowp, BLOCK_R=_RED_BLOCK_R, BLOCK_C=_RED_BLOCK_C, num_warps=4
        )
        return dsrcs + [red]
    if backend != "triton":
        raise _err(f"unknown backend {backend!r}")
    n_hblk = triton.cdiv(H, _BLOCK_H)
    progs, pers = [], []
    for m in metas:
        n = max(m["n_tok"], 1)
        per = triton.cdiv(n, min(_BWD_PROGRAMS_BIAS if bias_grad else _BWD_PROGRAMS, n))
        progs.append(triton.cdiv(n, per))
        pers.append(per)
    n_prog = sum(progs)
    part = torch.empty((n_prog, n_hblk, 2, D), device=qkv.device, dtype=torch.float32)
    bpart = torch.empty((n_prog, row_w) if bias_grad else (1,), device=qkv.device, dtype=torch.float32)
    dit_qk_norm_rope_bwd_kernel[(n_prog, n_hblk)](
        srcs[0], srcs[-1], dsrcs[0], dsrcs[-1], dq, dk, dv, ws[0][0], ws[0][1], ws[-1][0], ws[-1][1], cos, sin,
        part, bpart, m0["n_tok"], m0["S"], m1["n_tok"], m1["S"], progs[0], S_out, eps, pers[0], pers[-1],
        m0["TS"], m0["HS"], m0["QO"], m0["KO"], m0["VO"], B, _cs_row_stride(cos), max(row_w, 1),
        H=H, D=D, BLOCK_H=_BLOCK_H, BIAS=bool(bias_grad), SB=m0["SB"], NSEG=nseg,
        INTERLEAVED=rope == "interleaved", ROUND=bool(round_normed), CS_PER_BATCH=bool(cs_per_batch), num_warps=4,
    )  # fmt: skip
    block_c = min(_RED_BLOCK_C, 2 * D)  # a column block lies wholly in the dw or in the bias columns
    reduce_partials_kernel[(triton.cdiv(out_w, block_c), nseg)](
        part, bpart, red, progs[0], progs[-1], n_hblk, out_w, max(row_w, 1),
        D2=2 * D, BLOCK_R=_RED_BLOCK_R, BLOCK_C=block_c, num_warps=4,
    )  # fmt: skip
    return dsrcs + [red]


# --------------------------------------------------------------------------------------------------- ops
@torch.library.custom_op("primus_turbo::dit_qk_norm_rope", mutates_args=(), device_types="cuda")
def dit_qk_norm_rope_op(
    qkv: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    txt_qkv: Optional[torch.Tensor],
    w_txt_q: Optional[torch.Tensor],
    w_txt_k: Optional[torch.Tensor],
    eps: float,
    layout: str,
    rope: str,
    round_normed: bool,
    v_copy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v (joint, text first); v is empty when not ``v_copy``."""
    return dit_qk_norm_rope_fwd_impl(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, eps, layout, rope, round_normed, v_copy
    )


@dit_qk_norm_rope_op.register_fake
def _(qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, eps, layout, rope, round_normed, v_copy):
    metas, D, S_out, _ = check_operands(qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, layout, rope)
    shape = _out_shape(layout, S_out, metas[0]["NB"], metas[0]["H"], D)
    return qkv.new_empty(shape), qkv.new_empty(shape), qkv.new_empty(shape if v_copy else (0,))


@torch.library.custom_op("primus_turbo::dit_qk_norm_rope_backward", mutates_args=(), device_types="cuda")
def dit_qk_norm_rope_backward_op(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    qkv: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    txt_qkv: Optional[torch.Tensor],
    w_txt_q: Optional[torch.Tensor],
    w_txt_k: Optional[torch.Tensor],
    eps: float,
    layout: str,
    rope: str,
    round_normed: bool,
    bias_grad: bool,
) -> List[torch.Tensor]:
    """[d_qkv, (d_txt_qkv,) red]; ``red`` is [NSEG, 2*D (+ 3*H*D)]: per stream, text first, dw_q | dw_k (| db)."""
    return dit_qk_norm_rope_bwd_impl(
        dq,
        dk,
        dv,
        qkv,
        w_q,
        w_k,
        cos,
        sin,
        txt_qkv,
        w_txt_q,
        w_txt_k,
        eps,
        layout,
        rope,
        round_normed,
        bias_grad,
    )


@dit_qk_norm_rope_backward_op.register_fake
def _(
    dq, dk, dv, qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, eps, layout, rope, round_normed, bias_grad
):
    metas, D, _, _ = check_operands(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, layout, rope, bias_grad
    )
    srcs = [qkv] if txt_qkv is None else [txt_qkv, qkv]
    out_w = 2 * D + (metas[0]["ROW_W"] if bias_grad else 0)
    return [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device) for x in srcs] + [
        qkv.new_empty((len(srcs), out_w))
    ]
