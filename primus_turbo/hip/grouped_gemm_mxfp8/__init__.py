###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# HIP MX-FP8 grouped GEMM for MoE — Phase A Python dispatcher.
# Reuses the production HIP single-GEMM tile (csrc/kernels/gemm/turbo/
# turbo_gemm_mxfp8_kernel.h) with per-expert flat-grid dispatch.
###############################################################################

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
# Primus-Turbo repo root (2 levels up from primus_turbo/hip/grouped_gemm_mxfp8/)
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
_INCLUDE_DIRS = [
    os.path.join(_REPO_ROOT, "csrc", "include"),
    os.path.join(_REPO_ROOT, "csrc", "kernels", "gemm"),
]
_MODULE = None


def _build():
    global _MODULE
    if _MODULE is None:
        sources = [os.path.join(_HERE, "turbo_grouped_gemm_mxfp8.cu")]
        build_dir = os.path.join(_HERE, "build")
        os.makedirs(build_dir, exist_ok=True)
        _MODULE = load(
            name="primus_turbo_grouped_gemm_mxfp8_hip",
            sources=sources,
            extra_include_paths=_INCLUDE_DIRS,
            extra_cuda_cflags=[
                "-O3",
                "--offload-arch=gfx950",
                "-std=c++17",
                # Match main Primus-Turbo build: need half<->float conversions
                # in float8.h; PyTorch cpp_extension disables these by default.
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-Wno-unused-function",
                "-Wno-unused-variable",
                "-Wno-unused-parameter",
                # Emit AMDGCN dump on demand (set env PRIMUS_DUMP_ASM=1):
                *(["-save-temps"] if os.environ.get("PRIMUS_DUMP_ASM") else []),
            ],
            build_directory=build_dir,
            verbose=bool(os.environ.get("PRIMUS_BUILD_VERBOSE")),
        )
    return _MODULE


def _compute_block_to_expert(group_offs: torch.Tensor, tiles_n: int,
                             block_m: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side flat-grid dispatch helper.

    Returns:
        block_to_expert: [total_tiles] int32 — expert index for each flat block
        tile_offs:       [G+1]        int32 — prefix sum of tiles per expert
    """
    group_offs_cpu = group_offs.to("cpu", dtype=torch.int64)
    g = group_offs_cpu.numel() - 1
    m_g = (group_offs_cpu[1:] - group_offs_cpu[:-1]).to(torch.int32)
    tiles_m = (m_g + block_m - 1) // block_m
    tiles_per_expert = (tiles_m * int(tiles_n)).to(torch.int32)
    tile_offs_cpu = torch.zeros(g + 1, dtype=torch.int32)
    tile_offs_cpu[1:] = torch.cumsum(tiles_per_expert, dim=0, dtype=torch.int32)
    total_tiles = int(tile_offs_cpu[-1].item())
    block_to_expert_cpu = torch.repeat_interleave(
        torch.arange(g, dtype=torch.int32), tiles_per_expert
    )
    assert block_to_expert_cpu.numel() == total_tiles
    device = group_offs.device
    return block_to_expert_cpu.to(device), tile_offs_cpu.to(device)


def grouped_gemm_mxfp8_hip_wgrad(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """HIP MX-FP8 grouped GEMM wgrad — v0 correctness path.

    Computes, for each expert g:
        dB[g, n, k] = Σ_{m ∈ [offs[g], offs[g+1])} grad_out[m, n] * a[m, k]

    v0 path (current): ONE global transpose+quant of grad_out and a, then a
    **per-expert fwd-kernel call** with operand role-swap. Correct for any
    balanced MoE where M_g ≥ 384 and M_g % 128 == 0. Unbalanced MoE (small
    experts < 384) is not supported on this path.

    v1 (future): single-launch variable-K kernel with jagged scales + LDS
    transpose, per `WGRAD_DESIGN.md`. Projected perf: parity with Triton
    (memory-bound ceiling). v0 is intentionally conservative on perf to
    ship correctness.

    Args:
        grad_out:   [M_total, N]  fp8 e4m3  contiguous  (row-quanted along K-axis if used)
        a:          [M_total, K]  fp8 e4m3  contiguous
        group_offs: [G+1]         int64
        out_dtype:  torch.bfloat16 | torch.float16

    Returns:
        dB: [G, N, K]  out_dtype
    """
    from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
        quant_mxfp8_rowwise,
    )

    assert grad_out.ndim == 2 and a.ndim == 2
    assert grad_out.dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
    assert a.dtype == grad_out.dtype
    M_total, N = grad_out.shape
    _, K = a.shape
    assert a.shape[0] == M_total
    g = group_offs.numel() - 1

    # v1 requires balanced MoE (uniform M_g across experts) so a single
    # permute-contiguous puts M inner-contig per expert without per-expert
    # copies, enabling a single fwd-kernel call across all G experts.
    M_g = M_total // g
    assert M_g >= 384 and M_g % 128 == 0, (
        f"HIP wgrad (v1) requires balanced MoE with M_g >= 384 and M_g % 128 == 0; "
        f"got M_total={M_total}, G={g}, M_g={M_g}"
    )
    expected_offs = torch.arange(0, M_total + 1, M_g, dtype=torch.int64, device=group_offs.device)
    assert torch.equal(group_offs.to(torch.int64), expected_offs), (
        "HIP wgrad (v1) requires balanced group_offs (uniform M_g per expert)."
    )

    if grad_out.dtype != torch.bfloat16:
        go_bf = grad_out.to(torch.bfloat16)
        a_bf = a.to(torch.bfloat16)
    else:
        go_bf = grad_out
        a_bf = a

    # ── ONE global bf16 permute-contiguous per operand ──
    # go_bf [M_total=G*M_g, N] → view [G, M_g, N] → permute [G, N, M_g] → contiguous.
    # After: go_T[gi] is a contiguous [N, M_g] slice (zero-copy, M inner).
    go_T = go_bf.view(g, M_g, N).permute(0, 2, 1).contiguous()   # [G, N, M_g]
    a_T  = a_bf.view(g, M_g, K).permute(0, 2, 1).contiguous()    # [G, K, M_g]

    # ── ONE batched rowwise quant each (flatten leading 2 dims) ──
    go_flat_fp8, go_flat_scale = quant_mxfp8_rowwise(go_T.view(g * N, M_g))   # [G*N, M_g], [G*N, M_g/32]
    a_flat_fp8,  a_flat_scale  = quant_mxfp8_rowwise(a_T.view(g * K, M_g))    # [G*K, M_g], [G*K, M_g/32]
    a_fp8_3d   = a_flat_fp8.view(g, K, M_g)             # [G, K, M_g]  (fwd B operand)
    a_scale_3d = a_flat_scale.view(g, K, M_g // 32)     # [G, K, M_g/32]

    # ── Single fwd-kernel call (balanced MoE → uniform K_kern = M_g) ──
    # The fwd grouped kernel computes C[m, n] = Σ_k A[m, k] * B[g(m), n, k].
    # We stack G experts along M by treating each block of N rows as one
    # "group" → group_offs_new = [0, N, 2N, ..., G*N]. For expert g:
    #   rows [g*N, (g+1)*N) reduce against B[g] = a[g] shape [K, M_g]
    # Uniform K_kern = M_g across all experts (balanced assumption).
    group_offs_stacked = torch.arange(
        0, g * N + 1, N, dtype=torch.int64, device=grad_out.device
    )  # [G+1]
    dB_flat = grouped_gemm_mxfp8_hip_fwd(
        go_flat_fp8,                                    # [G*N, M_g]
        a_fp8_3d,                                       # [G, K, M_g]
        go_flat_scale,                                  # [G*N, M_g/32]
        a_scale_3d,                                     # [G, K, M_g/32]
        group_offs_stacked,
        out_dtype=out_dtype,
    )  # [G*N, K]
    return dB_flat.view(g, N, K).contiguous()


def grouped_gemm_mxfp8_hip_variable_k_padded(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_offs: torch.Tensor,
    scale_offs: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """HIP variable-K wgrad for UNBALANCED MoE via per-expert padded calls.

    Handles arbitrary per-expert M_g (including M_g % 16 != 0 and M_g < 384)
    by padding each expert's slice to the next multiple of 128 (min 384) and
    calling ``grouped_gemm_mxfp8_hip_fwd`` once per expert. Zero-token experts
    are skipped (their output rows zeroed).

    Trade-off: many per-expert kernel launches (~50 us each). For G=32 this is
    ~1.6 ms launch overhead baseline. Competitive vs Triton only when per-expert
    shapes are small enough that Triton's persistent-kernel launch amortizes
    poorly (rare).

    Args mirror ``grouped_gemm_mxfp8_hip_variable_k``. Output layout matches
    the `trans_c` convention.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.dtype == torch.float8_e4m3fn and rhs.dtype == torch.float8_e4m3fn
    M_total, L_out = lhs.shape
    _, R_out = rhs.shape
    g = group_offs.numel() - 1
    sc_g_group_size = 32   # MX group size

    # Reinterpret scales as float8_e8m0fnu for the kernel binding.
    if lhs_scale.dtype == torch.uint8:
        lhs_scale = lhs_scale.view(torch.float8_e8m0fnu)
    if rhs_scale.dtype == torch.uint8:
        rhs_scale = rhs_scale.view(torch.float8_e8m0fnu)

    # Output shape
    if trans_c:
        out = torch.empty((g, R_out, L_out), dtype=out_dtype, device=lhs.device)
    else:
        out = torch.empty((g, L_out, R_out), dtype=out_dtype, device=lhs.device)

    # Host-side offsets (CPU sync at start — acceptable since we're already
    # in the slow per-expert path).
    go_cpu = group_offs.to("cpu", dtype=torch.int64).tolist()
    if scale_offs is not None:
        so_cpu = scale_offs.to("cpu", dtype=torch.int64).tolist()

    for gi in range(g):
        m_start = go_cpu[gi]
        m_end = go_cpu[gi + 1]
        M_g = m_end - m_start
        if M_g <= 0:
            out[gi].zero_()
            continue

        # Round up to next multiple of 128, min 384 (kernel prologue minimum).
        M_g_pad = ((max(M_g, 384) + 127) // 128) * 128

        # Allocate padded per-expert buffers: [M_g_pad, L_out] fp8 + scales.
        lhs_pad = torch.zeros((M_g_pad, L_out), dtype=lhs.dtype, device=lhs.device)
        rhs_pad = torch.zeros((M_g_pad, R_out), dtype=rhs.dtype, device=rhs.device)
        lhs_pad[:M_g] = lhs[m_start:m_end]
        rhs_pad[:M_g] = rhs[m_start:m_end]

        # Scales: lhs_scale [L, total_sc] jagged per expert. Extract this expert's
        # scale columns [L, M_g/32] and pad to [L, M_g_pad/32] with zeros
        # (e8m0 0x00 ≈ 0 — nullifies the contribution of padded rows).
        if scale_offs is None:
            sc_start = (m_start + 31) // 32
            sc_end = m_end // 32
        else:
            sc_start = so_cpu[gi]
            sc_end = so_cpu[gi + 1]
        sc_per = sc_end - sc_start
        sc_per_pad = M_g_pad // 32
        lhs_scale_pad = torch.zeros((L_out, sc_per_pad), dtype=torch.uint8, device=lhs.device)
        rhs_scale_pad = torch.zeros((R_out, sc_per_pad), dtype=torch.uint8, device=rhs.device)
        lhs_scale_pad[:, :sc_per] = lhs_scale.view(torch.uint8)[:, sc_start:sc_end]
        rhs_scale_pad[:, :sc_per] = rhs_scale.view(torch.uint8)[:, sc_start:sc_end]

        # Single-group call via the existing balanced HIP variable_k.
        single_offs = torch.tensor([0, M_g_pad], dtype=torch.int64, device=lhs.device)
        dB_g = grouped_gemm_mxfp8_hip_variable_k(
            lhs_pad, rhs_pad, lhs_scale_pad, rhs_scale_pad,
            single_offs, None,
            out_dtype=out_dtype, trans_c=trans_c,
        )  # shape [1, L_out, R_out] or [1, R_out, L_out]
        out[gi] = dB_g[0]

    return out


def grouped_gemm_mxfp8_hip_variable_k(
    lhs: torch.Tensor,         # [M_total, L_out]     fp8 e4m3 col-quanted (scales along M)
    rhs: torch.Tensor,         # [M_total, R_out]     fp8 e4m3 col-quanted
    lhs_scale: torch.Tensor,   # [L_out, total_sc]    uint8 e8m0 jagged along M-space
    rhs_scale: torch.Tensor,   # [R_out, total_sc]    uint8 e8m0 jagged
    group_offs: torch.Tensor,  # [G+1]                int64 M-space prefix sums
    scale_offs: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """HIP MX-FP8 variable-K grouped GEMM backend entry point.

    Computes, per expert g:
        dB[g][l_out, r_out] = Σ_{m ∈ [offs[g], offs[g+1])} lhs[m, l_out] * rhs[m, r_out]

    Matches the signature of ``grouped_gemm_mxfp8_variable_k_triton_kernel``
    and is drop-in for ``GroupedGEMMFP8VariableKHipblasltBackend``.

    Output layout:
      - trans_c=False:  [G, L_out, R_out]
      - trans_c=True:   [G, R_out, L_out]      (for autograd wgrad convention
                                                 where grad_b matches B layout)

    Constraints (v1 — balanced MoE):
      - Every M_g == M_total / G (uniform)
      - M_g >= 384 and M_g % 128 == 0

    Implementation path:
      1. fp8-permute lhs/rhs to [G, L_out, M_g] / [G, R_out, M_g]          (1 byte/elem)
      2. Scale-permute to [G, L_out, M_g/32] / [G, R_out, M_g/32]
      3. Single HIP fwd-kernel call with stacked M = G*L_out, K_red = M_g
    """
    assert lhs.ndim == 2 and rhs.ndim == 2, f"lhs {lhs.shape}, rhs {rhs.shape}"
    assert lhs.dtype == torch.float8_e4m3fn and rhs.dtype == torch.float8_e4m3fn
    M_total, L_out = lhs.shape
    _, R_out = rhs.shape
    assert rhs.shape[0] == M_total
    g = group_offs.numel() - 1

    # Balanced-MoE precondition (v1 constraint).
    M_g = M_total // g
    assert M_g >= 384 and M_g % 128 == 0, (
        f"HIP variable-K (v1) requires balanced MoE with M_g >= 384 and M_g % 128 == 0; "
        f"got M_total={M_total}, G={g}, M_g={M_g}"
    )
    # NOTE: we do NOT validate group_offs values here — `torch.equal` forces
    # a CPU-GPU sync on every call (~4 ms latency penalty per step). Caller
    # is responsible for passing balanced group_offs. The assert above on
    # M_total % g == 0 + M_g dims catches the most common misuse.

    # Dtype-reinterpret scales to match kernel binding.
    if lhs_scale.dtype == torch.uint8:
        lhs_scale = lhs_scale.view(torch.float8_e8m0fnu)
    if rhs_scale.dtype == torch.uint8:
        rhs_scale = rhs_scale.view(torch.float8_e8m0fnu)

    # ── fp8 permute: [M, L] → [G, L, M_g] contig ─────────────────────────────
    # PyTorch's generic .permute(0,2,1).contiguous() on fp8 runs at ~12% of HBM
    # peak (754us / 1534us for our 189/377 MB tensors). Use a custom Triton
    # LDS-tile transpose that hits ~85% of peak (83us / 168us) = 9× speedup.
    # See _permute_fp8.py for the kernel.
    from primus_turbo.hip.grouped_gemm_mxfp8._permute_fp8 import fp8_permute_M_to_GN
    lhs_T = fp8_permute_M_to_GN(lhs, g, M_g)   # [G, L_out, M_g]
    rhs_T = fp8_permute_M_to_GN(rhs, g, M_g)   # [G, R_out, M_g]

    # ── Scale permute: [L, total_sc = G*M_g/32] → [G, L, M_g/32] ────────────
    # Scales are 1/32 of data size — torch's .permute().contiguous() is already
    # ~10 us here (equivalent to Triton, so not worth the kernel jump).
    sc_g = M_g // 32
    lhs_scale_3d = lhs_scale.view(L_out, g, sc_g).permute(1, 0, 2).contiguous()  # [G, L_out, M_g/32]
    rhs_scale_3d = rhs_scale.view(R_out, g, sc_g).permute(1, 0, 2).contiguous()  # [G, R_out, M_g/32]

    # ── Single fwd-kernel call with stacked M = G*L_out, K_red = M_g ─────────
    # For trans_c=False: output [G*L_out, R_out] → reshape [G, L_out, R_out].
    # For trans_c=True:  swap A/B roles so output becomes [G, R_out, L_out].
    if trans_c:
        # A_k = rhs flat, B_k = lhs per-expert, output [G, R_out, L_out]
        a_flat_fp8   = rhs_T.view(g * R_out, M_g)                # [G*R_out, M_g]
        a_flat_scale = rhs_scale_3d.view(g * R_out, sc_g)        # [G*R_out, M_g/32]
        b_fp8_3d     = lhs_T                                     # [G, L_out, M_g]
        b_scale_3d   = lhs_scale_3d                              # [G, L_out, M_g/32]
        M_kern_stack = R_out
        N_kern       = L_out
    else:
        a_flat_fp8   = lhs_T.view(g * L_out, M_g)
        a_flat_scale = lhs_scale_3d.view(g * L_out, sc_g)
        b_fp8_3d     = rhs_T
        b_scale_3d   = rhs_scale_3d
        M_kern_stack = L_out
        N_kern       = R_out

    group_offs_stacked = torch.arange(
        0, g * M_kern_stack + 1, M_kern_stack,
        dtype=torch.int64, device=group_offs.device,
    )
    out_flat = grouped_gemm_mxfp8_hip_fwd(
        a_flat_fp8, b_fp8_3d, a_flat_scale, b_scale_3d,
        group_offs_stacked, out_dtype=out_dtype,
    )  # [G * M_kern_stack, N_kern]
    return out_flat.view(g, M_kern_stack, N_kern).contiguous()


def grouped_gemm_mxfp8_hip_dgrad(
    dc: torch.Tensor,
    b_dgrad: torch.Tensor,
    dc_scale: torch.Tensor,
    b_dgrad_scale: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """HIP MX-FP8 grouped GEMM dgrad: dA = dC @ B.

    Reuses the fwd NT kernel. The dgrad reduction is along the N-axis of the
    forward problem, so B must be in dgrad layout [G, K, N] fp8 + [G, K, N//32]
    e8m0 scale (produced by quant_mxfp8_weight_dgrad upstream). In kernel space
    this is equivalent to a fwd with M→M, K→K_fwd_N, N→K_fwd_K.

    Args:
        dc:              [M_total, N]        fp8 e4m3 (N = fwd output cols)
        b_dgrad:         [G, K, N]           fp8 e4m3 (transposed weight layout)
        dc_scale:        [M_total, N//32]    uint8 e8m0
        b_dgrad_scale:   [G, K, N//32]       uint8 e8m0
        group_offs:      [G+1] int64
        out_dtype:       torch.bfloat16 | torch.float16

    Returns:
        dA: [M_total, K] output tensor.
    """
    # In kernel-space: A_k=dc, B_k=b_dgrad, the kernel's "n" is K_fwd, "k" is N_fwd.
    return grouped_gemm_mxfp8_hip_fwd(
        dc, b_dgrad, dc_scale, b_dgrad_scale, group_offs,
        out_dtype=out_dtype,
    )


def grouped_gemm_mxfp8_hip_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    block_to_expert: Optional[torch.Tensor] = None,
    tile_offs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """HIP MX-FP8 grouped GEMM forward (Phase A: NT layout, balanced MoE).

    Args:
        a:          [M_total, K]  fp8 e4m3 contiguous
        b:          [G, N, K]     fp8 e4m3 contiguous (NT — N-first within each expert)
        a_scale:    [M_total, K//32]  uint8 e8m0
        b_scale:    [G, N, K//32]     uint8 e8m0
        group_offs: [G+1] int64, prefix sum of tokens per expert
        out_dtype:  torch.bfloat16 | torch.float16
        block_to_expert / tile_offs: Optional host-precomputed lookup tensors
            (will be computed on demand if not provided).

    Returns:
        [M_total, N] output tensor.

    Constraints (Phase A):
      - Each per-expert M_g must be a multiple of 16
      - N % 16 == 0, K % 128 == 0, K >= 384
      - All tensors on the same CUDA device
    """
    assert a.ndim == 2 and b.ndim == 3, f"a={a.shape}, b={b.shape}"
    g = b.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    tiles_n = (n + 255) // 256

    if block_to_expert is None or tile_offs is None:
        block_to_expert, tile_offs = _compute_block_to_expert(group_offs, tiles_n, block_m=256)

    # Triton's quant_mxfp8_* returns scales as torch.uint8; the HIP binding
    # wants torch.float8_e8m0fnu. Same underlying bytes — reinterpret is free.
    if a_scale.dtype == torch.uint8:
        a_scale = a_scale.view(torch.float8_e8m0fnu)
    if b_scale.dtype == torch.uint8:
        b_scale = b_scale.view(torch.float8_e8m0fnu)

    mod = _build()
    at_out_dtype = {torch.bfloat16: torch.bfloat16,
                    torch.float16: torch.float16}[out_dtype]
    return mod.grouped_gemm_mxfp8_hip_fwd(
        a, b, a_scale, b_scale, group_offs, block_to_expert, tile_offs, at_out_dtype
    )
