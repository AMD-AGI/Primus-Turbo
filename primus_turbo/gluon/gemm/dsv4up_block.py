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
is logged once per (pass, shape).  If the Gluon cdna5 API is missing (Triton release), everything stays on g90.
"""

import logging
import os

_log = logging.getLogger("primus_turbo.dsv4up")
_LOGGED = set()
_AVAILABLE = None


def enabled() -> bool:
    return os.environ.get("PRIMUS_TURBO_FP8_BLOCK_KERNEL", "") == "dsv4up" and available()


def _warn_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.warning(msg)
        if os.environ.get("PRIMUS_TURBO_DSV4UP_VERBOSE", "1") == "1":
            print(f"[primus_turbo dsv4up] {msg}", flush=True)


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            from primus_turbo.gluon.gemm import dsv4up_gemm  # noqa: F401

            _AVAILABLE = True
        except Exception as e:  # Triton without gluon cdna5/tdm
            _AVAILABLE = False
            _warn_once("import", f"dsv4up unavailable ({type(e).__name__}: {e}); using g90")
    return _AVAILABLE


def shape_supported(M, N, K) -> bool:
    return M > 0 and N > 0 and M % 256 == 0 and N % 256 == 0 and K >= 1024 and K % 512 == 0


# Passes measured faster on g90 inside the Turbo fwd+bwd step (MI455X, T=4096, wgrad beta 1 into fp32 main_grad,
# rocprofv3 kernel trace, CUDA-graph replay; dsv4up_turbo/REPORT.md): small grids (64-128 CTAs of 256x256) where the
# 128x128 g90 tiling fills the GPU better.  Format "pass:MxNxK,..." (GEMM dims of that pass); "" disables.
_PREFER_G90_DEFAULT = "fwd:4096x1024x4096,wgrad:1024x4096x4096,wgrad:4096x2048x4096"
_PREFER = {}


def _prefer_g90():
    s = os.environ.get("PRIMUS_TURBO_DSV4UP_PREFER_G90", _PREFER_G90_DEFAULT)
    if s not in _PREFER:
        keys = set()
        for item in filter(None, (t.strip() for t in s.split(","))):
            tag, _, shp = item.partition(":")
            keys.add((tag, *map(int, shp.split("x"))))
        _PREFER[s] = keys
    return _PREFER[s]


def use(M, N, K, tag) -> bool:
    """True: run this pass on dsv4up.  False: g90 (logged once per (tag, shape))."""
    if not shape_supported(M, N, K):
        _warn_once((tag, M, N, K), f"{tag} GEMM {M}x{N}x{K} not supported by dsv4up (needs M,N%256, K%512, "
                   f"K>=1024) -> g90 fallback")
        return False
    if (tag, M, N, K) in _prefer_g90():
        _warn_once((tag, M, N, K), f"{tag} GEMM {M}x{N}x{K} measured faster on g90 (PRIMUS_TURBO_DSV4UP_PREFER_G90)"
                   f" -> g90")
        return False
    return True


_SPLITK = {}


def split_k_of(M, N, K) -> int:
    """PRIMUS_TURBO_DSV4UP_SPLITK="MxNxK:s,..." (GEMM dims of the call, default ""): split those dsv4up GEMMs into s
    K slices (fp32 partials, fixed-order sum; numerics differ from s = 1 by fp32 summation order only)."""
    spec = os.environ.get("PRIMUS_TURBO_DSV4UP_SPLITK", "")
    if spec not in _SPLITK:
        m = {}
        for item in filter(None, (t.strip() for t in spec.split(","))):
            shp, _, sk = item.partition(":")
            m[tuple(map(int, shp.split("x")))] = int(sk or "1")
        _SPLITK[spec] = m
    sk = _SPLITK[spec].get((M, N, K), 1)
    if sk > 1 and (K % sk or not shape_supported(M, N, K // sk)):
        _warn_once(("splitk", M, N, K, sk), f"split_k {sk} not valid for {M}x{N}x{K} (K/s must be >= 1024, % 512)")
        return 1
    return sk


def pack_scales(jobs):
    from primus_turbo.gluon.gemm.dsv4up_pack import pack_scales as _pack

    return _pack(jobs)


def gemm_nt(a, b, a_packed, b_packed, out_dtype=None, out=None, accumulate=False):
    from primus_turbo.gluon.gemm.dsv4up_gemm import dsv4up_gemm

    sk = split_k_of(a.shape[0], b.shape[0], a.shape[1])
    return dsv4up_gemm(a, b, a_packed, b_packed, out=out, out_dtype=out_dtype, accumulate=accumulate, split_k=sk)


def dual_enabled() -> bool:
    """PRIMUS_TURBO_DSV4UP_DUAL=1 (default off): a Linear's dgrad and wgrad run as one dsv4up launch (bit-identical
    outputs; the two GEMMs fill the GPU together)."""
    return os.environ.get("PRIMUS_TURBO_DSV4UP_DUAL", "0") == "1"


def gemm_nt_dual(job0, job1):
    """job = (a, b, a_packed, b_packed, out, accumulate); see dsv4up_gemm.dsv4up_dual."""
    from primus_turbo.gluon.gemm.dsv4up_gemm import dsv4up_dual

    return dsv4up_dual(job0, job1)
