###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""bf16grp: grouped (multi-problem) deterministic split-K BF16 GEMM for gfx1250 (Gluon TDM + WMMA 16x16x32).

  out_p = A_p @ W_p^T     for p in 0..P-1 (P <= 4), one GEMM launch + one reduce launch for all problems.

A_p [M_p, K_p] and W_p [N_p, K_p] are bf16 and K-contiguous (row strides arbitrary), each in its own allocation:
the GEMM kernel takes their ABSOLUTE int64 addresses as scalar arguments (int -> pointer cast, TDM descriptors), so
no pointer argument is ever offset into a different allocation (the >2 GB buffer-load trap of Triton on gfx1250).

Problem p's K is cut into S_p slices of L_p elements (the last one may be shorter; L_p % 64 == 0). Slice s of output
tile (m, n) is one program: one fp32 accumulator, v_wmma_f32_16x16x32_bf16 over 32-wide K chunks in ascending K
(operand kWidth 8, as bf16g2), a slice ending 64 elements into its last 128-wide tile skips that tile's second WMMA pair (the
TDM descriptor's inner-dimension bounds do not zero-fill on gfx1250, so the tail is never fed to the WMMA), the unrounded fp32 accumulator stored to a workspace. The reduce kernel sums the S_p partials of each element
in fixed order (((p0 + p1) + p2) + ...) in fp32 and rounds once to bf16 -- the order of hipBLASLt's GSU<S> +
PostGSU kernels (bit-identical where S_p / L_p match the hipBLASLt solution; e.g. 4096 -> 512: S=5, L=832; 4096 ->
192: S=8, L=512). No atomics: same bits run to run. All outputs are views of one bf16 buffer.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm

from primus_turbo.gluon.gemm.bf16g2_gemm import (BK, TILES, _chiplet, _consume, _consume_plain, _coords, _descs, _ld,
                                                 _refill, _step, _step_plain, build_layouts)

MAXP = 4


@gluon.jit
def _consume_tail(a_buf, b_buf, slot: gl.constexpr, acc, dot_a: gl.constexpr, dot_b: gl.constexpr, half):
    # last K step of a slice; half: the slice ends 64 elements into this 128-wide tile, so the second 64-wide WMMA
    # pair is skipped (its operands, loaded from the next slice or past K, are never used)
    HK: gl.constexpr = BK // 2
    a0, b0 = _ld(a_buf, b_buf, slot, 0, HK, dot_a, dot_b, False, False)
    acc = gl.amd.cdna5.wmma(a0, b0, acc)
    if not half:
        a1, b1 = _ld(a_buf, b_buf, slot, HK, HK, dot_a, dot_b, False, False)
        acc = gl.amd.cdna5.wmma(a1, b1, acc)
    return acc


@gluon.jit
def _pick(p, v0, v1, v2, v3):
    r = v0.to(gl.int64)
    if p == 1:
        r = v1.to(gl.int64)
    if p == 2:
        r = v2.to(gl.int64)
    if p == 3:
        r = v3.to(gl.int64)
    return r


_KARGS = ["A0", "A1", "A2", "A3", "W0", "W1", "W2", "W3", "SA0", "SA1", "SA2", "SA3", "SW0", "SW1", "SW2", "SW3",
          "M0", "M1", "M2", "M3", "N0", "N1", "N2", "N3", "K0", "K1", "K2", "K3", "L0", "L1", "L2", "L3",
          "O0", "O1", "O2", "O3", "T1", "T2", "T3"]


@gluon.jit(do_not_specialize=_KARGS)
def bf16grp_kernel(ws_ptr,
                   A0, A1, A2, A3, W0, W1, W2, W3,  # absolute int64 addresses (bf16)
                   SA0, SA1, SA2, SA3, SW0, SW1, SW2, SW3,  # row strides (elements)
                   M0, M1, M2, M3, N0, N1, N2, N3, K0, K1, K2, K3, L0, L1, L2, L3,
                   O0, O1, O2, O3,  # workspace offset (fp32 elements) of problem p
                   T1, T2, T3,  # first program of problems 1..3
                   NTOT: gl.constexpr, P: gl.constexpr, SHARED_A: gl.constexpr, SHARED_B: gl.constexpr,
                   WMMA: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, NS: gl.constexpr,
                   GROUP_M: gl.constexpr, KW: gl.constexpr = 8):
    gl.static_assert(NS == 2)
    pid = _chiplet(gl.program_id(axis=0), NTOT, 8)
    p = pid * 0
    t0 = pid * 0
    if P > 1:
        if pid >= T1:
            p = pid * 0 + 1
            t0 = T1
    if P > 2:
        if pid >= T2:
            p = pid * 0 + 2
            t0 = T2
    if P > 3:
        if pid >= T3:
            p = pid * 0 + 3
            t0 = T3
    a_addr = _pick(p, A0, A1, A2, A3)
    w_addr = _pick(p, W0, W1, W2, W3)
    sam = _pick(p, SA0, SA1, SA2, SA3)
    swn = _pick(p, SW0, SW1, SW2, SW3)
    M = _pick(p, M0, M1, M2, M3)
    N = _pick(p, N0, N1, N2, N3)
    K = _pick(p, K0, K1, K2, K3)
    L = _pick(p, L0, L1, L2, L3)
    wso = _pick(p, O0, O1, O2, O3)
    npm = (M + BLOCK_M - 1) // BLOCK_M
    npn = (N + BLOCK_N - 1) // BLOCK_N
    t = (pid - t0).to(gl.int64)
    s = t // (npm * npn)
    t = t % (npm * npn)
    # GROUP_M swizzle inside the slice
    num_pid_in_group = GROUP_M * npn
    group_id = t // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = gl.minimum(npm - first_pid_m, GROUP_M)
    pid_m = first_pid_m + t % num_pid_in_group % group_size_m
    pid_n = t % num_pid_in_group // group_size_m
    k0 = s * L
    klen = gl.minimum(L, K - k0)
    kext = (K - k0).to(gl.int32)  # descriptor K extent: the rest of the tensor (a half tail tile reads real data)
    half = (klen % BK) != 0
    a_ptr = (a_addr + (pid_m * BLOCK_M * sam + k0) * 2).to(gl.pointer_type(gl.bfloat16))
    w_ptr = (w_addr + (pid_n * BLOCK_N * swn + k0) * 2).to(gl.pointer_type(gl.bfloat16))
    a_desc = tdm.make_tensor_descriptor(base=a_ptr, shape=((M - pid_m * BLOCK_M).to(gl.int32), kext),
                                        strides=(sam, 1), block_shape=(BLOCK_M, BK), layout=SHARED_A)
    b_desc = tdm.make_tensor_descriptor(base=w_ptr, shape=((N - pid_n * BLOCK_N).to(gl.int32), kext),
                                        strides=(swn, 1), block_shape=(BLOCK_N, BK), layout=SHARED_B)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, KW)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, KW)
    a_buf = gl.allocate_shared_memory(gl.bfloat16, [NS, BLOCK_M, BK], SHARED_A)
    b_buf = gl.allocate_shared_memory(gl.bfloat16, [NS, BLOCK_N, BK], SHARED_B)
    for s0 in gl.static_range(NS):
        a_desc, b_desc = _refill(a_desc, b_desc, a_buf, b_buf, s0, False, False)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    nsteps = (klen + BK - 1) // BK  # >= NS (host check)
    n_main = nsteps - NS
    for _ in range(0, n_main // 2):
        a_desc, b_desc, acc = _step(a_buf, b_buf, 0, a_desc, b_desc, acc, dot_a, dot_b, NS, False, False)
        a_desc, b_desc, acc = _step(a_buf, b_buf, 1, a_desc, b_desc, acc, dot_a, dot_b, NS, False, False)
    if n_main % 2 == 1:
        a_desc, b_desc, acc = _step_plain(a_buf, b_buf, 0, a_desc, b_desc, acc, dot_a, dot_b, NS, False, False)
        tdm.async_wait(2)
        acc = _consume_plain(a_buf, b_buf, 1, acc, dot_a, dot_b, False, False)
        tdm.async_wait(0)
        acc = _consume_tail(a_buf, b_buf, 0, acc, dot_a, dot_b, half)
    else:
        tdm.async_wait(2)
        acc = _consume_plain(a_buf, b_buf, 0, acc, dot_a, dot_b, False, False)
        tdm.async_wait(0)
        acc = _consume_tail(a_buf, b_buf, 1, acc, dot_a, dot_b, half)
    a_buf._keep_alive()
    b_buf._keep_alive()
    c_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0],
                                                                     WMMA.cga_layout)
    c_shared = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], c_layout)
    c_desc = tdm.make_tensor_descriptor(base=ws_ptr + wso + s * M * N, shape=(M.to(gl.int32), N.to(gl.int32)),
                                        strides=(N, 1), block_shape=(BLOCK_M, BLOCK_N), layout=c_layout)
    c_shared.store(acc)
    tdm.async_store(c_desc, [(pid_m * BLOCK_M).to(gl.int32), (pid_n * BLOCK_N).to(gl.int32)], c_shared)
    tdm.async_wait(0)


_RARGS = ["M0", "M1", "M2", "M3", "N0", "N1", "N2", "N3", "S0", "S1", "S2", "S3", "O0", "O1", "O2", "O3",
          "Q0", "Q1", "Q2", "Q3", "R1", "R2", "R3"]


@triton.jit(do_not_specialize=_RARGS)
def _grp_reduce_kernel(ws_ptr, out_ptr, M0, M1, M2, M3, N0, N1, N2, N3, S0, S1, S2, S3, O0, O1, O2, O3,
                       Q0, Q1, Q2, Q3, R1, R2, R3, P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # ws: fp32 partials [S_p, M_p, N_p] at element offset O_p; out: bf16 [M_p, N_p] at element offset Q_p (one buffer)
    pid = tl.program_id(0)
    p = pid * 0
    r0 = pid * 0
    if P > 1:
        if pid >= R1:
            p = pid * 0 + 1
            r0 = R1
    if P > 2:
        if pid >= R2:
            p = pid * 0 + 2
            r0 = R2
    if P > 3:
        if pid >= R3:
            p = pid * 0 + 3
            r0 = R3
    M = tl.where(p == 0, M0, tl.where(p == 1, M1, tl.where(p == 2, M2, M3))).to(tl.int64)
    N = tl.where(p == 0, N0, tl.where(p == 1, N1, tl.where(p == 2, N2, N3))).to(tl.int64)
    S = tl.where(p == 0, S0, tl.where(p == 1, S1, tl.where(p == 2, S2, S3)))
    wo = tl.where(p == 0, O0, tl.where(p == 1, O1, tl.where(p == 2, O2, O3))).to(tl.int64)
    qo = tl.where(p == 0, Q0, tl.where(p == 1, Q1, tl.where(p == 2, Q2, Q3))).to(tl.int64)
    npn = (N + BLOCK_N - 1) // BLOCK_N
    t = pid - r0
    pid_m = t // npn
    pid_n = t % npn
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    off = rm[:, None] * N + rn[None, :]
    mn = M * N
    acc = tl.load(ws_ptr + wo + off, mask=mask, other=0.0)
    for s in range(1, S):
        acc += tl.load(ws_ptr + wo + s * mn + off, mask=mask, other=0.0)
    tl.store(out_ptr + qo + off, acc.to(out_ptr.dtype.element_ty), mask=mask)


_LAYOUTS = {}


def _cdiv(x, y):
    return (x + y - 1) // y


def grp_supported(problems, tile="128x256"):
    """problems: [(a [M,K], w [N,K], L)], a / w bf16 K-contiguous; True if bf16grp_mm can run them."""
    if not 1 <= len(problems) <= MAXP or tile not in TILES:
        return False
    for a, w, L in problems:
        if a.dim() != 2 or w.dim() != 2 or a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return False
        if a.stride(1) != 1 or w.stride(1) != 1 or a.shape[1] != w.shape[1] or not (a.is_cuda and w.is_cuda):
            return False
        if a.data_ptr() % 16 or w.data_ptr() % 16 or a.stride(0) % 8 or w.stride(0) % 8:
            return False
        K = a.shape[1]
        if L % 64 or L <= 0 or K % 64:
            return False
        for k0 in range(0, K, L):  # every slice needs >= NS (2) K steps of 128
            if _cdiv(min(L, K - k0), 128) < 2:
                return False
        if w.shape[0] % 4:  # 16-byte workspace rows for the TDM store
            return False
    return True


def bf16grp_mm(problems, tile="128x256", red=(32, 128), group_m=4, _debug=False):
    """[a_p @ w_p^T] for problems [(a_p [M,K] bf16, w_p [N,K] bf16, L_p)] -> list of bf16 [M_p, N_p] (views of one
    buffer). One grouped split-K GEMM launch (slice length L_p) + one fixed-order reduce launch."""
    P = len(problems)
    assert grp_supported(problems, tile), [(a.shape, a.stride(), w.shape, w.stride(), L) for a, w, L in problems]
    dev = problems[0][0].device
    bm, bn, ns = TILES[tile]
    rbm, rbn = red
    Ms, Ns, Ks, Ls, Ss, Os, Qs, Ts, Rs = [], [], [], [], [], [], [], [], []
    wo = qo = nt = nr = 0
    for a, w, L in problems:
        M, K = a.shape
        N = w.shape[0]
        S = _cdiv(K, L)
        Ms.append(M), Ns.append(N), Ks.append(K), Ls.append(L), Ss.append(S), Os.append(wo), Qs.append(qo)
        Ts.append(nt), Rs.append(nr)
        wo += S * M * N
        qo += M * N
        nt += S * _cdiv(M, bm) * _cdiv(N, bn)
        nr += _cdiv(M, rbm) * _cdiv(N, rbn)
    ws = torch.empty(wo, device=dev, dtype=torch.float32)  # per call: safe across streams / graph replays
    out = torch.empty(qo, device=dev, dtype=torch.bfloat16)

    def pad(v, fill=0):
        return list(v) + [fill] * (MAXP - len(v))

    A = pad([a.data_ptr() for a, _, _ in problems], problems[0][0].data_ptr())
    W = pad([w.data_ptr() for _, w, _ in problems], problems[0][1].data_ptr())
    SA = pad([a.stride(0) for a, _, _ in problems], 8)
    SW = pad([w.stride(0) for _, w, _ in problems], 8)
    lk = (tile, False, False, 8)
    if lk not in _LAYOUTS:
        lay = build_layouts(tile, False, False, 8)
        _LAYOUTS[lk] = {k: v for k, v in lay.items() if k not in ("AT", "BT")}
    T = pad(Ts[1:], nt)[: MAXP - 1]
    bf16grp_kernel[(nt,)](ws, *A, *W, *SA, *SW, *pad(Ms, 128), *pad(Ns, 128), *pad(Ks, 128), *pad(Ls, 64), *pad(Os), *T,
                          NTOT=nt, P=P, GROUP_M=group_m, num_warps=8, waves_per_eu=2, num_ctas=1, KW=8,
                          **_LAYOUTS[lk])
    R = pad(Rs[1:], nr)[: MAXP - 1]
    _grp_reduce_kernel[(nr,)](ws, out, *pad(Ms, 128), *pad(Ns, 128), *pad(Ss, 2), *pad(Os), *pad(Qs), *R, P=P,
                              BLOCK_M=rbm, BLOCK_N=rbn, num_warps=4)
    res = []
    for p in range(P):
        res.append(out[Qs[p]: Qs[p] + Ms[p] * Ns[p]].view(Ms[p], Ns[p]))
    if _debug:
        return res, ws, Os, Ss
    return res


# ---------------------------------------------------------------------------------------------------------------------
# bf16rot: bf16g2 (one fp32 accumulator, ascending 32-wide WMMA K chunks, kWidth 8) with hipBLASLt's StaggerU K start.
# The untuned hipBLASLt default solutions of the DSv4 step (Cijk_*_Bias_HA_S_SAV_*_MT*x*x128: CSA / HCA compressor
# fwd / dgrad / wgrad, o_a wgrad) sum K sequentially in 128-wide iterations but each workgroup starts at iteration
# (tn mod 32) mod (K / 128) and wraps around, tn = output column // 256 (torch row-major view). Reproducing that start
# makes bf16g2 bit-identical to them (bf16grp probes, ~/dsv41/work/bf16grp/ledger.md).
# ---------------------------------------------------------------------------------------------------------------------
@gluon.jit
def _refill_rot(a_desc, b_desc, a_buf, b_buf, slot: gl.constexpr, AT: gl.constexpr, BT: gl.constexpr, kpos,
                K: gl.constexpr):
    tdm.async_load(a_desc, [0, 0], a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, 0], b_buf.index(slot), warp_used_hint=0b00001111)
    nk = kpos + BK
    w = (nk >= K).to(gl.int32)
    delta = BK - K * w
    kpos = nk - K * w
    if AT:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[delta, 0])
    else:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, delta])
    if BT:
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[delta, 0])
    else:
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, delta])
    return a_desc, b_desc, kpos


@gluon.jit
def _step_rot(a_buf, b_buf, slot: gl.constexpr, a_desc, b_desc, acc, dot_a: gl.constexpr, dot_b: gl.constexpr,
              NS: gl.constexpr, AT: gl.constexpr, BT: gl.constexpr, kpos, K: gl.constexpr):
    tdm.async_wait(2 * (NS - 1))
    acc = _consume(a_buf, b_buf, slot, acc, dot_a, dot_b, AT, BT)
    with gl.amd.warp_pipeline_stage('s_stage4_refill'):
        a_desc, b_desc, kpos = _refill_rot(a_desc, b_desc, a_buf, b_buf, slot, AT, BT, kpos, K)
    return a_desc, b_desc, acc, kpos


@gluon.jit
def _rot_tile(a_ptr, b_ptr, c_ptr, pid, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
              stride_am: gl.constexpr, stride_bn: gl.constexpr, stride_cm: gl.constexpr, stride_ag: gl.constexpr,
              stride_bg: gl.constexpr, stride_cg: gl.constexpr, SHARED_A: gl.constexpr, SHARED_B: gl.constexpr,
              WMMA: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, NS: gl.constexpr,
              GROUP_M: gl.constexpr, AT: gl.constexpr, BT: gl.constexpr, SU_W: gl.constexpr, SU_N: gl.constexpr,
              KW: gl.constexpr):
    # one output tile of out[g] = A[g] @ B[g] in the StaggerU-rotated sequential K order, bf16 out through LDS + TDM
    NPM: gl.constexpr = (M + BLOCK_M - 1) // BLOCK_M
    NPN: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    NI: gl.constexpr = K // BK
    g, pid_m, pid_n = _coords(pid, NPM, NPN, GROUP_M)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, KW)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, KW)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [NS, BK, BLOCK_M] if AT else [NS, BLOCK_M, BK], SHARED_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [NS, BK, BLOCK_N] if BT else [NS, BLOCK_N, BK], SHARED_B)
    a_desc, b_desc = _descs(a_ptr, b_ptr, g, pid_m, pid_n, M, N, K, stride_am, stride_bn, stride_ag, stride_bg,
                            SHARED_A, SHARED_B, BLOCK_M, BLOCK_N, AT, BT)
    # StaggerU start: iteration (tn mod SU_N) mod NI, tn = first output column of the tile // SU_W
    kpos = (((pid_n * BLOCK_N) // SU_W) % SU_N) % NI * BK
    if AT:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[kpos, 0])
    else:
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, kpos])
    if BT:
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[kpos, 0])
    else:
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, kpos])
    for s0 in gl.static_range(NS):
        a_desc, b_desc, kpos = _refill_rot(a_desc, b_desc, a_buf, b_buf, s0, AT, BT, kpos, K)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    n_main: gl.constexpr = NI - NS
    U: gl.constexpr = 6 if NS == 3 else 4
    for _ in range(0, n_main // U):
        for inner in gl.static_range(U):
            a_desc, b_desc, acc, kpos = _step_rot(a_buf, b_buf, inner % NS, a_desc, b_desc, acc, dot_a, dot_b, NS,
                                                  AT, BT, kpos, K)
    for r in gl.static_range(n_main % U):
        a_desc, b_desc, acc, kpos = _step_rot(a_buf, b_buf, r % NS, a_desc, b_desc, acc, dot_a, dot_b, NS, AT, BT,
                                              kpos, K)
    for t in gl.static_range(NS):
        tdm.async_wait(2 * (NS - 1 - t))
        acc = _consume(a_buf, b_buf, (n_main + t) % NS, acc, dot_a, dot_b, AT, BT)
    a_buf._keep_alive()
    b_buf._keep_alive()
    c_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0],
                                                                     WMMA.cga_layout)
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_layout)
    c_desc = tdm.make_tensor_descriptor(base=c_ptr + g * stride_cg, shape=(M, N), strides=(stride_cm, 1),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def bf16rot_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, G: gl.constexpr,
                   stride_am: gl.constexpr, stride_bn: gl.constexpr, stride_cm: gl.constexpr, stride_ag: gl.constexpr,
                   stride_bg: gl.constexpr, stride_cg: gl.constexpr, NTILES: gl.constexpr, SHARED_A: gl.constexpr,
                   SHARED_B: gl.constexpr, WMMA: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   NS: gl.constexpr, GROUP_M: gl.constexpr, AT: gl.constexpr, BT: gl.constexpr,
                   SU_W: gl.constexpr, SU_N: gl.constexpr, KW: gl.constexpr = 8):
    pid = _chiplet(gl.program_id(axis=0), NTILES, 8)
    _rot_tile(a_ptr, b_ptr, c_ptr, pid, M, N, K, stride_am, stride_bn, stride_cm, stride_ag, stride_bg, stride_cg,
              SHARED_A, SHARED_B, WMMA, BLOCK_M, BLOCK_N, NS, GROUP_M, AT, BT, SU_W, SU_N, KW)


@gluon.jit
def bf16rot2_kernel(a0_ptr, b0_ptr, c0_ptr, a1_ptr, b1_ptr, c1_ptr,
                    M0: gl.constexpr, N0: gl.constexpr, K0: gl.constexpr, SAM0: gl.constexpr, SBN0: gl.constexpr,
                    SCM0: gl.constexpr, SAG0: gl.constexpr, SBG0: gl.constexpr, SCG0: gl.constexpr,
                    NT0: gl.constexpr, SHARED_A0: gl.constexpr, SHARED_B0: gl.constexpr, WMMA0: gl.constexpr,
                    BM0: gl.constexpr, BN0: gl.constexpr, NS0: gl.constexpr, AT0: gl.constexpr, BT0: gl.constexpr,
                    SUN0: gl.constexpr,
                    M1: gl.constexpr, N1: gl.constexpr, K1: gl.constexpr, SAM1: gl.constexpr, SBN1: gl.constexpr,
                    SCM1: gl.constexpr, SAG1: gl.constexpr, SBG1: gl.constexpr, SCG1: gl.constexpr,
                    NT1: gl.constexpr, SHARED_A1: gl.constexpr, SHARED_B1: gl.constexpr, WMMA1: gl.constexpr,
                    BM1: gl.constexpr, BN1: gl.constexpr, NS1: gl.constexpr, AT1: gl.constexpr, BT1: gl.constexpr,
                    SUN1: gl.constexpr,
                    GROUP_M: gl.constexpr, SU_W: gl.constexpr, KW: gl.constexpr = 8):
    # two independent (batched) problems, own layouts / tiles, in one launch: programs [0, NT0) -> problem 0, the rest
    # -> problem 1; each operand is its own pointer argument (no cross-allocation offsets). SUN_i = 32: hipBLASLt
    # untuned-SAV StaggerU K start; SUN_i = 1: plain ascending K (= bf16g2 / the tuned hipBLASLt solutions)
    pid = _chiplet(gl.program_id(axis=0), NT0 + NT1, 8)
    if pid < NT0:
        _rot_tile(a0_ptr, b0_ptr, c0_ptr, pid, M0, N0, K0, SAM0, SBN0, SCM0, SAG0, SBG0, SCG0, SHARED_A0, SHARED_B0,
                  WMMA0, BM0, BN0, NS0, GROUP_M, AT0, BT0, SU_W, SUN0, KW)
    else:
        _rot_tile(a1_ptr, b1_ptr, c1_ptr, pid - NT0, M1, N1, K1, SAM1, SBN1, SCM1, SAG1, SBG1, SCG1, SHARED_A1,
                  SHARED_B1, WMMA1, BM1, BN1, NS1, GROUP_M, AT1, BT1, SU_W, SUN1, KW)


_RPLANS = {}
_RLAYOUTS = {}


def rot_supported(a, b, out, tile="256x256"):
    """True if bf16rot_mm can take these operands (bf16g2 rules + StaggerU tile width 256 divisible by BLOCK_N)."""
    from primus_turbo.gluon.gemm.bf16g2_gemm import supported

    if tile not in TILES or 256 % TILES[tile][1]:
        return False
    return supported(a, b, out, tile) and out.dtype == torch.bfloat16


def bf16rot_mm(a, b, out, tile="256x256", group_m=4, tpad=8, su_w=256, su_n=32):
    """out[g] = a[g] @ b[g] (torch.bmm / torch.mm semantics, bf16) in hipBLASLt's untuned-SAV K order: sequential
    128-wide K iterations starting at ((col // su_w) mod su_n) mod (K / 128), wrapping. Returns out."""
    if a.dim() == 2:
        a, b, out = a.unsqueeze(0), b.unsqueeze(0), out.unsqueeze(0)
    G, M, K = a.shape
    N = b.shape[2]
    assert b.shape == (G, K, N) and out.shape == (G, M, N), (a.shape, b.shape, out.shape)
    assert a.dtype == b.dtype == out.dtype == torch.bfloat16
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
    assert out.stride(2) == 1
    bm, bn, ns = TILES[tile]
    assert su_w % bn == 0 and K % int(BK.value) == 0 and K // int(BK.value) >= ns + 1, (K, tile)
    for t in (a, b, out):
        assert t.is_cuda and t.data_ptr() != 0
    sag, sbg, scg = (a.stride(0), b.stride(0), out.stride(0)) if G > 1 else (0, 0, 0)
    key = (G, M, N, K, a.stride(), b.stride(), out.stride(), tile, group_m, tpad, su_w, su_n)
    p = _RPLANS.get(key)
    if p is None:
        lk = (tile, at, bt, tpad)
        if lk not in _RLAYOUTS:
            _RLAYOUTS[lk] = build_layouts(tile, at, bt, tpad)
        lay = _RLAYOUTS[lk]
        ntiles = G * _cdiv(M, bm) * _cdiv(N, bn)
        kw = dict(M=M, N=N, K=K, G=G, stride_am=sam, stride_bn=sbn, stride_cm=out.stride(1), stride_ag=sag,
                  stride_bg=sbg, stride_cg=scg, NTILES=ntiles, GROUP_M=group_m, num_warps=8, waves_per_eu=2,
                  num_ctas=1, KW=8, SU_W=su_w, SU_N=su_n, **lay)
        p = ((ntiles,), kw)
        _RPLANS[key] = p
    grid, kw = p
    bf16rot_kernel[grid](a, b, out, **kw)
    return out


def _rot_args(a, b, out, tile, tpad, sun):
    if a.dim() == 2:
        a, b, out = a.unsqueeze(0), b.unsqueeze(0), out.unsqueeze(0)
    G, M, K = a.shape
    N = b.shape[2]
    assert b.shape == (G, K, N) and out.shape == (G, M, N) and out.stride(2) == 1, (a.shape, b.shape, out.shape)
    assert a.dtype == b.dtype == out.dtype == torch.bfloat16
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
    bm, bn, ns = TILES[tile]
    assert 256 % bn == 0 and K % int(BK.value) == 0 and K // int(BK.value) >= ns + 1, (K, tile)
    for t in (a, b, out):
        assert t.is_cuda and t.data_ptr() != 0
    sag, sbg, scg = (a.stride(0), b.stride(0), out.stride(0)) if G > 1 else (0, 0, 0)
    lk = (tile, at, bt, tpad)
    if lk not in _RLAYOUTS:
        _RLAYOUTS[lk] = build_layouts(tile, at, bt, tpad)
    lay = _RLAYOUTS[lk]
    nt = G * _cdiv(M, bm) * _cdiv(N, bn)
    return dict(M=M, N=N, K=K, SAM=sam, SBN=sbn, SCM=out.stride(1), SAG=sag, SBG=sbg, SCG=scg, NT=nt,
                SHARED_A=lay["SHARED_A"], SHARED_B=lay["SHARED_B"], WMMA=lay["WMMA"], BM=bm, BN=bn, NS=ns, AT=at,
                BT=bt, SUN=sun)


def bf16rot2_mm(p0, p1, group_m=4, su_w=256):
    """Two independent GEMMs out_i[g] = a_i[g] @ b_i[g] (torch.mm / bmm semantics, bf16) in ONE launch.
    p_i = (a, b, out, tile, tpad[, su_n]); su_n 32 (default) = hipBLASLt untuned-SAV StaggerU K order (as bf16rot_mm),
    su_n 1 = plain ascending K (as bf16g2_mm). Problem 0's programs come first (put the longer-K problem first)."""
    ps = [tuple(p) + ((32,) if len(p) == 5 else ()) for p in (p0, p1)]
    key = tuple((a.shape, a.stride(), b.shape, b.stride(), o.shape, o.stride(), t, tp, su) for a, b, o, t, tp, su in ps)
    key = key + (group_m, su_w)
    kw = _RPLANS.get(key)
    if kw is None:
        kw = {}
        for i, (a, b, o, t, tp, su) in enumerate(ps):
            for k, v in _rot_args(a, b, o, t, tp, su).items():
                kw[f"{k}{i}"] = v
        kw.update(GROUP_M=group_m, SU_W=su_w, KW=8, num_warps=8, waves_per_eu=2, num_ctas=1)
        _RPLANS[key] = kw
    bf16rot2_kernel[(kw["NT0"] + kw["NT1"],)](ps[0][0], ps[0][1], ps[0][2], ps[1][0], ps[1][1], ps[1][2], **kw)
    return p0[2], p1[2]
