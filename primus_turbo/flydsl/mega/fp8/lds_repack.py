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

"""The fused MXFP8 mega MoE's own copy of the LDS scale-repack body.

Not imported from ``primus_turbo.flydsl.utils.gemm_helper``: that helper serves the host-side
grouped-GEMM preshuffle kernels, where producer and consumer live in one kernel and a barrier is
enough, so its ``rd_cm``/``st_cm`` cache modifiers look like dead defaults and have now been
dropped by two separate rewrites of that file (#436, then #483 after #478 restored them). The
fused mega MoE is the only caller that needs them, and each time they went the fused MXFP8 path
failed at trace time with ``TypeError: unexpected keyword argument 'rd_cm'``. Owning the body here
makes that impossible: a refactor of the shared helper can no longer reach this path.

Kept as a verbatim copy of the working version rather than specialised to this call site, so the
emitted code is provably the same. ``is_a`` is a Python bool at trace time, so the branch this
caller does not take is never traced.
"""

import flydsl.expr as fx
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.prims import _lds_barrier, ceildiv


def _emit_lds_repack(
    is_a,
    grp,
    k0,
    tile,
    rin,
    rout,
    dim,
    K128,
    KT,
    tid,
    BLK,
    rd_base=0,
    wr_base=0,
    pack=1,
    kbound=None,
    k128p=None,
    comb=None,
    # rd_cm/st_cm (default 0 = cached): CPol bits on the raw load / broadcast store. On gfx950
    # 1 emits sc0, 16 emits sc1, 2 emits nt. Host-side preshuffle callers
    # (build_preshuffle_ab_kernel, the grouped mxfp8 kernels) keep the defaults: producer and
    # consumer are in one kernel, so a barrier is enough. A fused-kernel preshuffle role, whose
    # producer is a peer rank and whose consumer may sit on another XCD, passes rd_cm=1 (sc0
    # acquire) + st_cm=16 (sc1 write-through release) so the transpose itself carries the fence --
    # no whole-L2 buffer_inv before it and no device-wide buffer_wbl2 after it.
    rd_cm=0,
    st_cm=0,
):
    # LDS-tiled transpose body (one workgroup, one (grp,k-chunk)). rd_base/wr_base
    # (default 0) shift the flat read/write offset to a group's slab (0 = dense).
    # kbound (default K128) bounds this chunk's k index; k128p (default ceildiv(K128,pack)) is the output k-stride, so the variable-K wgrad packs each group from its own k0.
    # comb (B only, default (256,128)) is the (tile-block, second-half) row stride pair: a
    # group already interleaves the block's two 128-column halves, which is what lets the
    # reader split one ScaleBComb load in two. Widening the second stride past the block
    # makes those halves two distant bands -- the fused GLU's gate and up.
    NT = 4
    CB_BLK, CB_HALF = comb if comb is not None else (256, 128)
    TILE = 64 * KT
    assert KT % pack == 0 and TILE % BLK == 0 and ((KT // pack) * 64) % BLK == 0
    KBND = K128 if kbound is None else kbound
    for i in range_constexpr(TILE // BLK):
        idx = tid + i * BLK
        rr = idx // KT
        kk = idx % KT
        gk = k0 + kk
        if is_a:
            grow = grp * 64 + rr  # A: rows grp*64 + (s*16+r)
        else:
            s = rr // 16  # B-comb: row = nblk*CB_BLK + wn*32 + OFF[s] + rinner
            off = (s % 2) * fx.Int32(16) + (s // 2) * fx.Int32(CB_HALF)
            grow = (grp // 4) * CB_BLK + (grp % 4) * 32 + off + (rr % 16)
        dw = _buffer_ops.buffer_load(
            rin,
            grow * K128 + gk + rd_base,
            vec_width=1,
            dtype=T.i32,
            mask=(gk < KBND) & (grow < dim),
            cache_modifier=rd_cm,
        )
        fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)).store(
            Vec.from_elements([fx.Int32(dw)], fx.Int32)
        )
    _lds_barrier()
    # Packed store: pack PACK consecutive K-iters into one output dword per lane (the
    # reader mirrors this via kk=k//PACK + MFMA op_sel). PACK=1 = unpacked.
    PACK = pack
    K128p = ceildiv(K128, PACK) if k128p is None else k128p
    NGP = KT // PACK  # packed groups produced per KT-chunk
    NOUTp = NGP * 64
    for j in range_constexpr(NOUTp // BLK):
        ol = tid + j * BLK
        kkp = ol // 64
        lane = ol % 64
        r = lane % 16
        sh = (lane // 16) * fx.Int32(8)
        gkp = (k0 // PACK) + kkp
        elems = []
        for s in range_constexpr(NT):
            packed = fx.Int32(0)
            for bb in range_constexpr(PACK):
                so = (s * 16 + r) * KT + (kkp * PACK + bb)
                val = Vec(
                    fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(so)), fx.make_layout(1, 1)).load()
                )
                b = (fx.Int32(val[0]) >> sh) & fx.Int32(0xFF)
                packed = packed | (b << fx.Int32(bb * 8))
            elems.append(packed)
        vec = Vec.from_elements(elems, fx.Int32)
        _buffer_ops.buffer_store(
            vec.ir_value(),
            rout,
            ((grp * K128p + gkp) * 64 + lane) * 4 + wr_base,
            mask=(k0 + kkp * PACK) < KBND,
            cache_modifier=st_cm,
        )
