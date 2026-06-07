###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tiled bf16 WMMA GEMM for gfx1250 (RDNA4), CuTe-style.

Computes ``C[M, N] = A[M, K] @ B[N, K].T`` (NT layout) in bf16 with f32
accumulation, using the gfx1250 ``wmma_f32_16x16x32_bf16`` atom through
FlyDSL's CuTe-style atom / fragment / tiled-copy abstraction (so the WMMA
register-fragment layout is handled by the compiler, not hand-coded).

This is the foundational matmul primitive for the FlyDSL attention backward
(every backward product is expressed as an NT GEMM with transposed operands).
One warp (32 lanes, wave32) computes one 16x16 output tile, accumulating over
K in 32-wide chunks. Correctness-first (not software-pipelined); the per-shape
launch is cached on (M, N, K).

FlyDSL is a hard dependency (see requirements.txt); a missing install raises
ImportError on import rather than degrading silently.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32


@functools.lru_cache(maxsize=256)
def _build_gemm_nt(M: int, N: int, K: int):
    """Build & cache the (M, N, K)-specialised NT bf16 WMMA launch."""
    assert M % WMMA_M == 0 and N % WMMA_N == 0 and K % WMMA_K == 0, (
        f"gemm_nt requires M%16==N%16==K%32==0, got M={M} N={N} K={K}"
    )
    NK = K // WMMA_K

    @flyc.kernel
    def kernel_gemm_nt(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.x
        bn = fx.block_idx.y
        gA = fx.rocdl.make_buffer_tensor(A, max_size=False)
        gB = fx.rocdl.make_buffer_tensor(B, max_size=False)
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False)

        aA = fx.flat_divide(gA, (WMMA_M, WMMA_K))[None, None, bm, None]  # (WM, WK, NK)
        aB = fx.flat_divide(gB, (WMMA_N, WMMA_K))[None, None, bn, None]  # (WN, WK, NK)
        aC = fx.flat_divide(gC, (WMMA_M, WMMA_N))[None, None, bm, bn]  # (WM, WN)

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        thr_A = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
        thr_B = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)
        thr_C = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)

        src_A = thr_A.partition_S(aA)
        src_B = thr_B.partition_S(aB)
        dst_C = thr_C.partition_S(aC)

        frag_A = thr_mma.make_fragment_A(aA[None, None, 0])
        frag_B = thr_mma.make_fragment_B(aB[None, None, 0])
        frag_C = thr_mma.make_fragment_C(aC)
        frag_C.fill(0)

        for ki in fx.range_constexpr(NK):
            fx.copy(cp_ab, src_A[None, None, None, ki], thr_A.retile(frag_A), pred=None)
            fx.copy(cp_ab, src_B[None, None, None, ki], thr_B.retile(frag_B), pred=None)
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        fx.copy(cp_c, thr_C.retile(frag_C), dst_C, pred=None)

    @flyc.jit
    def launch_gemm_nt(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        kernel_gemm_nt(A, B, C).launch(
            grid=(M // WMMA_M, N // WMMA_N, 1), block=(32, 1, 1), stream=stream
        )

    return launch_gemm_nt


_COMPILED: dict = {}


def _get_compiled(launch, args):
    key = (id(launch), tuple(a.shape for a in args if isinstance(a, torch.Tensor)))
    c = _COMPILED.get(key)
    if c is None:
        c = flyc.compile(launch, *args)
        _COMPILED[key] = c
    return c


def _ceil(x, m):
    return (x + m - 1) // m * m


def gemm_nt_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C[M, N] = a[M, K] @ b[N, K].T, bf16 inputs, f32 output.

    a and b must be 2-D bf16 CUDA tensors sharing the contraction dim K.
    M, N, K are zero-padded up to the WMMA tile granularity (16 / 16 / 32);
    the padding contributes 0 to the dot product and the extra output rows /
    cols are sliced off, so any M, N, K is accepted.
    """
    assert a.dim() == 2 and b.dim() == 2, "gemm_nt_bf16 expects 2-D operands"
    assert a.shape[1] == b.shape[1], f"K mismatch: a {a.shape}, b {b.shape}"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    M, K = a.shape
    N = b.shape[0]
    Mp, Np, Kp = _ceil(M, WMMA_M), _ceil(N, WMMA_N), _ceil(K, WMMA_K)

    ap = a
    bp = b
    if (Mp, Kp) != (M, K):
        ap = a.new_zeros((Mp, Kp))
        ap[:M, :K] = a
    if (Np, Kp) != (N, K):
        bp = b.new_zeros((Np, Kp))
        bp[:N, :K] = b
    ap = ap.contiguous()
    bp = bp.contiguous()
    out = torch.empty((Mp, Np), dtype=torch.float32, device=a.device)

    launch = _build_gemm_nt(Mp, Np, Kp)
    args = (ap, bp, out, torch.cuda.current_stream())
    _get_compiled(launch, args)(*args)
    return out[:M, :N]
