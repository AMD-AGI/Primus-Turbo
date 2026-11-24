###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Dispatch of the 128-block (UE8M0) FP8 dense GEMMs to the dsv4up Gluon kernel (gfx1250).

Enabled by PRIMUS_TURBO_FP8_BLOCK_KERNEL=dsv4up inside FP8GemmBlockFunction; rides on the g90 path (same
quantization, same NT formulation of fwd / dgrad / wgrad, same wgrad main_grad beta / first-write policy) and
replaces the GEMM call per pass when the shape is supported:
  fwd   Y[M,N]   = X[M,K]   . W[N,K]^T        A 1x128, B 128x128 scales
  dgrad dX[M,K]  = dY[M,N]  . (W^T)[K,N]^T    A 1x128, B 128x128 (transposed block grid)
  wgrad dW[N,K]  = dY^T[N,M] . X^T[K,M]^T     both 1x128 along tokens; beta 1 epilogue into main_grad
Unsupported shapes (M or N not a multiple of 256, K not a multiple of 512 or K < 1024) stay on g90; the fallback
is logged once per (pass, shape).  When the Gluon gate is closed (``primus_turbo.gluon.gate.gluon_ok``: gfx1250, Triton
>= 3.9 with gluon cdna5, PRIMUS_GLUON != 0) or the kernel module cannot be imported, everything stays on g90 and the
reason is logged once.
"""

import logging

import primus_turbo.common.envs as envs

_log = logging.getLogger("primus_turbo.dsv4up")
_LOGGED = set()
_AVAILABLE = None


def enabled() -> bool:
    """dsv4up is the selected block-FP8 kernel and can run (Gluon gate open, kernel module importable)."""
    return envs.PRIMUS_TURBO_FP8_BLOCK_KERNEL == "dsv4up" and available()


def _warn_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.warning(msg)


def available() -> bool:
    """True when the Gluon gate is open and gemm_fp8blk_kernel imports; otherwise False, with the reason logged once."""
    global _AVAILABLE
    if _AVAILABLE is None:
        from primus_turbo.gluon.gate import gluon_ok, why_not

        if not gluon_ok():
            _AVAILABLE = False
            _warn_once("gate", f"dsv4up is off: {why_not()}; dense block-FP8 GEMMs use g90")
        else:
            try:
                from primus_turbo.gluon.gemm import gemm_fp8blk_kernel  # noqa: F401

                _AVAILABLE = True
            except Exception as e:  # a Triton whose gluon cdna5 API differs from the one the kernels target
                _AVAILABLE = False
                _warn_once(
                    "import",
                    f"dsv4up is unavailable ({type(e).__name__}: {e}); dense block-FP8 GEMMs use g90",
                )
    return _AVAILABLE


def shape_supported(M, N, K) -> bool:
    """GEMM dims the dsv4up 256x256 tile handles."""
    return M > 0 and N > 0 and M % 256 == 0 and N % 256 == 0 and K >= 1024 and K % 512 == 0


def use(M, N, K, tag) -> bool:
    """True: run this pass on dsv4up.  False: g90 (logged once per (tag, shape))."""
    if not shape_supported(M, N, K):
        _warn_once(
            (tag, M, N, K),
            f"{tag} GEMM {M}x{N}x{K} not supported by dsv4up (needs M,N%256, K%512, K>=1024) -> g90 fallback",
        )
        return False
    return True


def pack_scales(jobs):
    """Pack E8M0 scale bytes into the dsv4up layout (see gemm_fp8blk_utils.pack_scales)."""
    from primus_turbo.gluon.gemm.gemm_fp8blk_utils import pack_scales as _pack

    return _pack(jobs)


_SMALL = {}


def small_tile_of(M, N, K, accumulate) -> str:
    """PRIMUS_TURBO_DSV4UP_SMALLTILE="MxNxK[a]:tile,..." ("" = off): run those dsv4up GEMMs (GEMM dims of the
    call; the "a" suffix selects the beta-1 accumulate call, e.g. a wgrad with the same dims as a dgrad) on the dsv4s
    smaller-output-tile kernel (tiles in gemm_fp8blk_smalltile_kernel.TILES, e.g. 256x128s3). Bit-identical to dsv4up: same per-output
    K order and the same WMMA instruction per 128-wide K chunk; only the output tiling changes (more CUs busy)."""
    spec = envs.PRIMUS_TURBO_DSV4UP_SMALLTILE
    if not spec:
        return ""
    if spec not in _SMALL:
        from primus_turbo.gluon.gemm.gemm_fp8blk_smalltile_kernel import TILES

        m = {}
        for item in filter(None, (t.strip() for t in spec.split(","))):
            shp, _, tile = item.partition(":")
            acc = shp.endswith("a")
            dims = tuple(map(int, shp.rstrip("a").split("x")))
            if tile not in TILES:
                _warn_once(
                    ("smalltile", item), f"PRIMUS_TURBO_DSV4UP_SMALLTILE: unknown tile {tile!r} in {item!r}"
                )
                continue
            m[dims + (acc,)] = tile
        _SMALL[spec] = m
    tile = _SMALL[spec].get((M, N, K, bool(accumulate)), "")
    if tile:
        from primus_turbo.gluon.gemm.gemm_fp8blk_smalltile_kernel import shape_supported as _ss

        if not _ss(M, N, K, tile):
            _warn_once(
                ("smalltile", M, N, K, tile), f"dsv4s tile {tile} does not support {M}x{N}x{K} -> dsv4up"
            )
            return ""
    return tile


def gemm_nt(a, b, a_packed, b_packed, out_dtype=None, out=None, accumulate=False):
    """C[M,N] = A[M,K] . B[N,K]^T on dsv4up, or on the dsv4s tile that PRIMUS_TURBO_DSV4UP_SMALLTILE assigns to these
    dims; a / b fp8 e4m3, *_packed from pack_scales. accumulate: beta 1 into ``out`` (2: bf16 product + bf16 old C)."""
    from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm

    tile = small_tile_of(a.shape[0], b.shape[0], a.shape[1], accumulate)
    if tile:
        from primus_turbo.gluon.gemm.gemm_fp8blk_smalltile_kernel import dsv4s_gemm

        return dsv4s_gemm(
            a, b, a_packed, b_packed, out=out, out_dtype=out_dtype, accumulate=accumulate, tile=tile
        )
    return dsv4up_gemm(a, b, a_packed, b_packed, out=out, out_dtype=out_dtype, accumulate=accumulate)


def gemm_nt_split(a, b, a_packed, b_packed, out0, out1, axis, accumulate=False):
    """See gemm_fp8blk_kernel.dsv4up_gemm_split. PRIMUS_TURBO_DSV4UP_SMALLTILE entries also apply (dsv4s split epilogue) when
    the split point is a multiple of the tile edge."""
    from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm_split

    tile = small_tile_of(a.shape[0], b.shape[0], a.shape[1], accumulate)
    if tile:
        from primus_turbo.gluon.gemm.gemm_fp8blk_smalltile_kernel import TILES, dsv4s_gemm_split

        bm, bn, _, _ = TILES[tile]
        s0 = out0.shape[1] if axis == "n" else out0.shape[0]
        if s0 % (bn if axis == "n" else bm) == 0:
            return dsv4s_gemm_split(
                a, b, a_packed, b_packed, out0, out1, axis, accumulate=accumulate, tile=tile
            )

    return dsv4up_gemm_split(a, b, a_packed, b_packed, out0, out1, axis, accumulate=accumulate)
