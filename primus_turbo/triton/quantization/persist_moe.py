###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""persist track (PRIMUS_PERSIST_MOE_COMBINE_DW=1): the MegaMoE fp8_block backward's combine (dx of the routed experts,
_combine_kernel) and d topk-weights (_route_dw_kernel) are independent; run them as ONE launch (horizontal merge):
programs [0, NDW) = _route_dw_kernel's programs, [NDW, NDW + T * H / BH) = _combine_kernel's (t, h-block) programs.
Same code, tiles and warps -> the same bits; no cross-workgroup communication at all."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from primus_turbo.triton.quantization.quant_blk_gfx1250 import _gw4


def combine_dw_enabled() -> bool:
    return os.environ.get("PRIMUS_PERSIST_MOE_COMBINE_DW", "0") == "1"


@triton.jit
def _combine_route_dw_kernel(Y, INV, OUT, H, GW, DW, R, NDW, ND,
                             K: tl.constexpr, BH: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NDW:
        # == _route_dw_kernel, program pid
        r = pid * BLOCK + tl.arange(0, BLOCK)
        m = r < R
        p = tl.load(INV + r, mask=m, other=0).to(tl.int64)
        if NC == 16:
            acc = (_gw4(GW, p, m, NC, 0) + _gw4(GW, p, m, NC, 1)) + (_gw4(GW, p, m, NC, 2) + _gw4(GW, p, m, NC, 3))
        elif NC == 8:
            acc = _gw4(GW, p, m, NC, 0) + _gw4(GW, p, m, NC, 1)
        else:
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for c in tl.static_range(NC):
                acc += tl.load(GW + p * NC + c, mask=m, other=0.0)
        tl.store(DW + r, acc.to(DW.dtype.element_ty), mask=m)
    else:
        # == _combine_kernel (HAS_ADD = 0), program (t, hb)
        q = pid - NDW
        t = q // ND
        h = (q % ND) * BH + tl.arange(0, BH)
        cacc = tl.zeros((BH,), dtype=tl.float32)
        for k in tl.static_range(K):
            rk = tl.load(INV + t * K + k).to(tl.int64)
            cacc += tl.load(Y + rk * H + h).to(tl.float32)
        o = cacc.to(OUT.dtype.element_ty)
        tl.store(OUT + t.to(tl.int64) * H + h, o)


def combine_rows_route_dw(y: torch.Tensor, inv: torch.Tensor, T: int, K: int, gw: torch.Tensor, out_dtype,
                          bh: int = 1024):
    """= (combine_rows(y, inv, T, K, bh), route_dw_partials(gw, inv, out_dtype)) in one launch."""
    H = y.shape[1]
    assert H % bh == 0 and y.is_contiguous() and inv.is_contiguous() and gw.is_contiguous()
    out = torch.empty((T, H), device=y.device, dtype=y.dtype)
    R = inv.numel()
    NC = gw.shape[1]
    dw = torch.empty(R, device=gw.device, dtype=out_dtype)
    ndw = triton.cdiv(R, 1024)
    nd = H // bh
    _combine_route_dw_kernel[(ndw + T * nd,)](y, inv, out, H, gw, dw, R, ndw, nd, K=K, BH=bh, NC=NC, BLOCK=1024,
                                              num_warps=4)
    return out, dw


__all__ = ["combine_dw_enabled", "combine_rows_route_dw"]
