###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""bf16g2: batched BF16 GEMM for gfx1250 (Gluon, TDM + WMMA 16x16x32).

  out[g] = A[g] @ B[g]     (torch.bmm semantics; A [M,K] with K or M contiguous, B [K,N] with K or N contiguous,
                            out [M,N] row-major with any row stride; batch strides arbitrary; bf16 in / out)

Per output: one fp32 accumulator, v_wmma_f32_16x16x32_bf16 over 32-wide K chunks in ascending K, one bf16 rounding
at the end (meant to match hipBLASLt's non-split-K MI16x16x1 order; checked per shape with a bitdiff).
Main loop: dsv4s / upstream gemm_warp_pipeline_cdna5 style (warp-pipelined, BK=128 bf16 = 256 B rows, NS data slots,
TDM loads); epilogue through LDS + TDM store.
M / N edges are handled by the TDM descriptors (OOB loads read 0, OOB stores are dropped).
"""

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna5 import tdm

from primus_turbo.gluon.gate import require_gluon

BK = gl.constexpr(128)
TILES = {  # name: (BM, BN, NS)
    "256x256": (256, 256, 2),
    "256x128": (256, 128, 2),
    "256x128s3": (256, 128, 3),
    "128x256": (128, 256, 2),
}


@gluon.jit
def _chiplet(pid, num_workgroups, num_xcds: gl.constexpr):
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd


@gluon.jit
def _coords(t, NPM: gl.constexpr, NPN: gl.constexpr, GROUP_M: gl.constexpr):
    # t: tile index in [0, G*NPM*NPN); returns (g, pid_m, pid_n), GROUP_M swizzle inside each batch entry
    TPG: gl.constexpr = NPM * NPN
    g = t // TPG
    t = t % TPG
    num_pid_in_group: gl.constexpr = GROUP_M * NPN
    group_id = t // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(NPM - first_pid_m, GROUP_M)
    pid_m = first_pid_m + t % num_pid_in_group % group_size_m
    pid_n = t % num_pid_in_group // group_size_m
    return g, pid_m, pid_n


@gluon.jit
def _descs(
    a_ptr,
    b_ptr,
    g,
    pid_m,
    pid_n,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_am: gl.constexpr,
    stride_bn: gl.constexpr,
    stride_ag: gl.constexpr,
    stride_bg: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    # AT: A stored [K, M] (M contiguous; stride_am = row stride along K), else [M, K] (K contiguous)
    # BT: B stored [K, N] (N contiguous; stride_bn = row stride along K), else [N, K] (K contiguous)
    a_rows = M - pid_m * BLOCK_M
    b_rows = N - pid_n * BLOCK_N
    if AT:
        a_desc = tdm.make_tensor_descriptor(
            base=a_ptr + g * stride_ag + pid_m * BLOCK_M,
            shape=(K, a_rows),
            strides=(stride_am, 1),
            block_shape=(BK, BLOCK_M),
            layout=SHARED_A,
        )
    else:
        a_desc = tdm.make_tensor_descriptor(
            base=a_ptr + g * stride_ag + pid_m * BLOCK_M * stride_am,
            shape=(a_rows, K),
            strides=(stride_am, 1),
            block_shape=(BLOCK_M, BK),
            layout=SHARED_A,
        )
    if BT:
        b_desc = tdm.make_tensor_descriptor(
            base=b_ptr + g * stride_bg + pid_n * BLOCK_N,
            shape=(K, b_rows),
            strides=(stride_bn, 1),
            block_shape=(BK, BLOCK_N),
            layout=SHARED_B,
        )
    else:
        b_desc = tdm.make_tensor_descriptor(
            base=b_ptr + g * stride_bg + pid_n * BLOCK_N * stride_bn,
            shape=(b_rows, K),
            strides=(stride_bn, 1),
            block_shape=(BLOCK_N, BK),
            layout=SHARED_B,
        )
    return a_desc, b_desc


@gluon.jit
def _refill(a_desc, b_desc, a_buf, b_buf, slot: gl.constexpr, AT: gl.constexpr, BT: gl.constexpr):
    tdm.async_load(a_desc, [0, 0], a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, 0], b_buf.index(slot), warp_used_hint=0b00001111)
    if AT:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[BK, 0])
    else:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BK])
    if BT:
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[BK, 0])
    else:
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BK])
    return a_desc, b_desc


@gluon.jit
def _ld(
    a_buf,
    b_buf,
    slot: gl.constexpr,
    k0: gl.constexpr,
    n: gl.constexpr,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    if AT:
        a = a_buf.index(slot).slice(k0, n, 0).permute([1, 0]).load(layout=dot_a)
    else:
        a = a_buf.index(slot).slice(k0, n, 1).load(layout=dot_a)
    if BT:
        b = b_buf.index(slot).slice(k0, n, 0).load(layout=dot_b)
    else:
        b = b_buf.index(slot).slice(k0, n, 1).permute([1, 0]).load(layout=dot_b)
    return a, b


@gluon.jit
def _consume_plain(
    a_buf,
    b_buf,
    slot: gl.constexpr,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    # same K order as _consume, no warp-pipeline stages (for loop bodies that also hold a nested loop)
    HK: gl.constexpr = BK // 2
    a0, b0 = _ld(a_buf, b_buf, slot, 0, HK, dot_a, dot_b, AT, BT)
    acc = gl.amd.cdna5.wmma(a0, b0, acc)
    a1, b1 = _ld(a_buf, b_buf, slot, HK, HK, dot_a, dot_b, AT, BT)
    acc = gl.amd.cdna5.wmma(a1, b1, acc)
    return acc


@gluon.jit
def _consume(
    a_buf,
    b_buf,
    slot: gl.constexpr,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    HK: gl.constexpr = BK // 2
    with gl.amd.warp_pipeline_stage("s_stage0_load_k0", phase_gap=1):
        a0, b0 = _ld(a_buf, b_buf, slot, 0, HK, dot_a, dot_b, AT, BT)
    with gl.amd.warp_pipeline_stage("s_stage1_compute_k0"):
        acc = gl.amd.cdna5.wmma(a0, b0, acc)
    with gl.amd.warp_pipeline_stage("s_stage2_load_k1"):
        a1, b1 = _ld(a_buf, b_buf, slot, HK, HK, dot_a, dot_b, AT, BT)
    with gl.amd.warp_pipeline_stage("s_stage3_compute_k1"):
        acc = gl.amd.cdna5.wmma(a1, b1, acc)
    return acc


@gluon.jit
def _step(
    a_buf,
    b_buf,
    slot: gl.constexpr,
    a_desc,
    b_desc,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    NS: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    tdm.async_wait(2 * (NS - 1))
    acc = _consume(a_buf, b_buf, slot, acc, dot_a, dot_b, AT, BT)
    with gl.amd.warp_pipeline_stage("s_stage4_refill"):
        a_desc, b_desc = _refill(a_desc, b_desc, a_buf, b_buf, slot, AT, BT)
    return a_desc, b_desc, acc


@gluon.jit
def _step_plain(
    a_buf,
    b_buf,
    slot: gl.constexpr,
    a_desc,
    b_desc,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    NS: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    tdm.async_wait(2 * (NS - 1))
    acc = _consume_plain(a_buf, b_buf, slot, acc, dot_a, dot_b, AT, BT)
    gl.barrier()  # every warp is done reading the slot before it is refilled
    a_desc, b_desc = _refill(a_desc, b_desc, a_buf, b_buf, slot, AT, BT)
    return a_desc, b_desc, acc


@gluon.jit
def _mainloop(
    a_buf,
    b_buf,
    a_desc,
    b_desc,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    K: gl.constexpr,
    NS: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
):
    # NS slots already issued; runs the K steps that refill (all but the last NS)
    n_main: gl.constexpr = K // BK - NS
    U: gl.constexpr = 6 if NS == 3 else 4
    for _ in range(0, n_main // U):
        for inner in gl.static_range(U):
            a_desc, b_desc, acc = _step(
                a_buf, b_buf, inner % NS, a_desc, b_desc, acc, dot_a, dot_b, NS, AT, BT
            )
    for r in gl.static_range(n_main % U):
        a_desc, b_desc, acc = _step(a_buf, b_buf, r % NS, a_desc, b_desc, acc, dot_a, dot_b, NS, AT, BT)
    return a_desc, b_desc, acc


@gluon.jit
def bf16g2_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    G: gl.constexpr,
    stride_am: gl.constexpr,
    stride_bn: gl.constexpr,
    stride_cm: gl.constexpr,
    stride_ag: gl.constexpr,
    stride_bg: gl.constexpr,
    stride_cg: gl.constexpr,
    NTILES: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    WMMA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NS: gl.constexpr,
    GROUP_M: gl.constexpr,
    AT: gl.constexpr,
    BT: gl.constexpr,
    KW: gl.constexpr = 8,
):
    # one output tile per CTA
    NPM: gl.constexpr = (M + BLOCK_M - 1) // BLOCK_M
    NPN: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    pid = _chiplet(gl.program_id(axis=0), NTILES, 8)
    g, pid_m, pid_n = _coords(pid, NPM, NPN, GROUP_M)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, KW)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, KW)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [NS, BK, BLOCK_M] if AT else [NS, BLOCK_M, BK], SHARED_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [NS, BK, BLOCK_N] if BT else [NS, BLOCK_N, BK], SHARED_B
    )
    a_desc, b_desc = _descs(
        a_ptr,
        b_ptr,
        g,
        pid_m,
        pid_n,
        M,
        N,
        K,
        stride_am,
        stride_bn,
        stride_ag,
        stride_bg,
        SHARED_A,
        SHARED_B,
        BLOCK_M,
        BLOCK_N,
        AT,
        BT,
    )
    for s0 in gl.static_range(NS):
        a_desc, b_desc = _refill(a_desc, b_desc, a_buf, b_buf, s0, AT, BT)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    a_desc, b_desc, acc = _mainloop(a_buf, b_buf, a_desc, b_desc, acc, dot_a, dot_b, K, NS, AT, BT)
    n_main: gl.constexpr = K // BK - NS
    for t in gl.static_range(NS):
        tdm.async_wait(2 * (NS - 1 - t))
        acc = _consume(a_buf, b_buf, (n_main + t) % NS, acc, dot_a, dot_b, AT, BT)
    a_buf._keep_alive()
    b_buf._keep_alive()
    c_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0], WMMA.cga_layout
    )
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_layout)
    c_desc = tdm.make_tensor_descriptor(
        base=c_ptr + g * stride_cg,
        shape=(M, N),
        strides=(stride_cm, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=c_layout,
    )
    c_shared.store(acc.to(c_ptr.type.element_ty))
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


_LAYOUTS = {}
_PLANS = {}


def _pad(rows, cols, cga=None, pad=8):
    # [rows, cols] row-major smem tile, `pad` elements of padding per row
    if cga is None:
        return gl.PaddedSharedLayout.with_identity_for([[cols, pad]], [rows, cols], [1, 0])
    return gl.PaddedSharedLayout.with_identity_for([[cols, pad]], [rows, cols], [1, 0], cga)


def build_layouts(tile, at=False, bt=False, tpad=8):
    # tpad: row padding (elements) of the MN-major ([BK, BM] / [BK, BN]) operand tiles
    block_m, block_n, nslot = TILES[tile]
    bk = int(BK.value)
    sa = (bk, block_m) if at else (block_m, bk)
    sb = (bk, block_n) if bt else (block_n, bk)
    pa = tpad if at else 8
    pb = tpad if bt else 8
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
        block_m,
        block_n,
        _pad(*sa, pad=pa),
        _pad(*sb, pad=pb),
        8,
        [16, 16, 32],
        a_transposed=at,
        b_transposed=not bt,
        slice_m=block_m,
        slice_n=block_n,
    )
    load_wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases, local_wmma.instr_shape, []
    )
    dot_a = gl.DotOperandLayout(0, load_wmma, 8)
    dot_b = gl.DotOperandLayout(1, load_wmma, 8)

    def tr(c):
        return tuple((basis[1], basis[0]) for basis in c)

    cga_a = tr(dot_a.cga_layout) if at else dot_a.cga_layout
    cga_b = dot_b.cga_layout if bt else tr(dot_b.cga_layout)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(
        block_m,
        block_n,
        _pad(*sa, cga_a, pa),
        _pad(*sb, cga_b, pb),
        8,
        [16, 16, 32],
        a_transposed=at,
        b_transposed=not bt,
        slice_m=block_m,
        slice_n=block_n,
    )
    return dict(
        SHARED_A=shared_a,
        SHARED_B=shared_b,
        WMMA=load_wmma,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NS=nslot,
        AT=at,
        BT=bt,
    )


def triton_cdiv(x, y):
    return (x + y - 1) // y


def supported(a, b, out, tile="256x256"):
    """True if bf16g2_mm can take these operands (same rules as its asserts)."""
    try:
        if a.dim() != b.dim() or a.dim() not in (2, 3) or out.dim() != a.dim():
            return False
        if not (a.dtype == b.dtype == torch.bfloat16 and out.dtype in (torch.bfloat16, torch.float32)):
            return False
        K = a.shape[-1]
        _, _, ns = TILES[tile]
        return (
            (a.stride(-1) == 1 or a.stride(-2) == 1)
            and (b.stride(-2) == 1 or b.stride(-1) == 1)
            and out.stride(-1) == 1
            and K % int(BK.value) == 0
            and K // int(BK.value) >= ns + 1
        )
    except Exception:  # noqa: BLE001
        return False


def operand_strides(a, b):
    """(at, sam, bt, sbn) of 3-D operands a [G,M,K] (K or M contiguous) and b [G,K,N] (K or N contiguous): at / bt =
    the operand is stored MN-major, sam / sbn = its row stride."""
    if a.stride(2) == 1:
        at, sam = False, a.stride(1)
    else:
        assert a.stride(1) == 1, a.stride()
        at, sam = True, a.stride(2)
    if b.stride(1) == 1:
        bt, sbn = False, b.stride(2)
    else:
        assert b.stride(2) == 1, b.stride()
        bt, sbn = True, b.stride(1)
    return at, sam, bt, sbn


def bf16g2_mm(a, b, out, tile="256x256", group_m=4, kw=8, tpad=8):
    """out[g] = a[g] @ b[g] (torch.bmm / torch.mm semantics, 2-D or 3-D, bf16, fp32 accumulate, beta 0).
    a: [G,M,K] with K or M contiguous; b: [G,K,N] with K or N contiguous; out: [G,M,N] with N contiguous;
    batch strides arbitrary. out may be fp32 (bf16tune split-K partials: the fp32 accumulator stored unrounded).
    Returns out."""
    require_gluon("bf16g2_mm")
    if a.dim() == 2:
        a, b, out = a.unsqueeze(0), b.unsqueeze(0), out.unsqueeze(0)
    G, M, K = a.shape
    N = b.shape[2]
    assert b.shape == (G, K, N) and out.shape == (G, M, N), (a.shape, b.shape, out.shape)
    assert a.dtype == b.dtype == torch.bfloat16 and out.dtype in (
        torch.bfloat16,
        torch.float32,
    )  # fp32 out: split-K
    at, sam, bt, sbn = operand_strides(a, b)
    assert out.stride(2) == 1
    bm, bn, ns = TILES[tile]
    assert K % int(BK.value) == 0 and K // int(BK.value) >= ns + 1, (K, tile)
    for t in (a, b, out):
        assert t.is_cuda and t.data_ptr() != 0
    sag, sbg, scg = (a.stride(0), b.stride(0), out.stride(0)) if G > 1 else (0, 0, 0)
    key = (
        G,
        M,
        N,
        K,
        a.stride(),
        b.stride(),
        out.stride(),
        tile,
        group_m,
        kw,
        tpad,
    )
    p = _PLANS.get(key)
    if p is None:
        lk = (tile, at, bt, tpad)
        if lk not in _LAYOUTS:
            _LAYOUTS[lk] = build_layouts(tile, at, bt, tpad)
        ntiles = G * triton_cdiv(M, bm) * triton_cdiv(N, bn)
        kw = dict(
            M=M,
            N=N,
            K=K,
            G=G,
            stride_am=sam,
            stride_bn=sbn,
            stride_cm=out.stride(1),
            stride_ag=sag,
            stride_bg=sbg,
            stride_cg=scg,
            NTILES=ntiles,
            GROUP_M=group_m,
            num_warps=8,
            waves_per_eu=2,
            num_ctas=1,
            KW=kw,
            **_LAYOUTS[lk],
        )
        p = (bf16g2_kernel, (ntiles,), kw)
        _PLANS[key] = p
    kern, grid, kw = p
    kern[grid](a, b, out, **kw)
    return out
