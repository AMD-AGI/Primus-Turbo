###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL) (kernels/gemm/).
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL BF16 GROUPED GEMM — M-grouped operator (forward NT / dgrad NN) + variable-K wgrad.

A is [M_total, K] (groups concatenated along M), B is [G, N, K] (trans_b, forward)
or [G, K, N] (dgrad), out is [M_total, N], and ``group_offs`` [G+1] int64 splits
M_total into G groups. Mirrors the fp8 grouped entry and the Triton one:

    out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g]

Design (CPU-sync-free, reuses the tuned dense bf16 tile body verbatim):
  * Grid is over-launched to the host upper bound ``(ceil(M_total/BLOCK_M) + G) *
    n_blocks`` (no device read of group lens); each WG computes the true
    ``total_tiles`` on-device via an O(G) scan and returns early when its pid is
    past the end.
  * The same O(G) scan maps pid -> (group_idx, local tile) -> (local_block_m,
    block_n). Per-group addressing rebases A/B/C per tile in int64 (so a >4GB
    expert pool keeps its in-tile buffer offsets int32) and then calls
    ``gemm_bf16_tile`` with block_m=0 — the dense tile API stays unchanged:
      - A view: base m_row*K, bounded by the rows left in the pool, so a partial
        M-tile's over-read is clamped to 0 by the HW SRD.
      - B view: base group_idx*N*K (NT) / group_idx*K*N (NN), one expert slab.
      - C view: base m_row*N, bounded by ``c_rows = min(m_end - m_row, BLOCK_M)``
        so a partial M-tile's extra rows (which belong to the next group) are
        clamped out — no spill across groups.

The variable-K wgrad (TN) entry below is a separate operator: the contraction dim
is the token axis and varies per group, so its grid is a fixed G * tiles-per-group
(the output shape is static) and ``masked_k`` bounds the K-loop to the valid rows.

See gemm_bf16_kernel.py for the K-loop / barrier / LDS rationale (identical here).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.buffer_ops import extract_base_index
from flydsl.expr.primitive import get_iter as _get_iter
from flydsl.expr.primitive import ptrtoint as _ptrtoint
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    _i64,
    _load_i64_as_i32,
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    Mfma16x16x32,
    S2RLoaderTr16x32Bf16,
    StoreCBf16,
    _readfirstlane_i32,
    ceildiv,
    compute_global_swizzle_nn_bf16,
    make_bf16_buffer_tensor_rebased,
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
    wait_barrier,
    xcd_remap_pid,
)

# 8 waves of 64 lanes; the dense bf16 tile body is built for exactly this shape.
_BLOCK_THREADS = 512
# xcd=1 keeps consecutive tiles (which share an expert's B slab) on one XCD; the
# grouped sweep measured +3% NT / +11% NN over xcd=8, where the remap scatters them.
_DEFAULT_NUM_XCD = 1
# 1-D super-block width over the within-group tile grid: group_m consecutive M
# blocks share a B N-stripe, keeping it L2-resident.
_DEFAULT_GROUP_M = 4


def _load_i32(div, idx):
    """Read one int32 scalar from an i32 buffer view at i32-element idx (per-lane,
    uniform across the WG since idx is uniform)."""
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return Vec(fx.memref_load_vec(reg))[0]


def _load_go(div, idx):
    """Read group_offs[idx] from an i32-view of the int64 [G+1] tensor. The dispatch
    passes group_offs.view(int32) (free reinterpret), so element idx's low 32 bits live
    at i32 index 2*idx; token offsets are < 2^31 so the high word is 0."""
    return _load_i32(div, idx * 2)


def _base_i64(t):
    """Tensor base address as an i64 ArithValue (for per-tile int64 rebasing)."""
    return fx.arith.ArithValue(arith.index_cast(fx.T.i64(), extract_base_index(t)), signed=True)


def _min_i32(a, b):
    return arith.select(a < b, a, b)


def _grouped_block_mn(local, m_start, m_end, n_blocks, block_m_size, group_m):
    """Map a within-group linear tile index ``local`` to (local_block_m, block_n)
    under a group_m 1-D super-block swizzle (falls back to row-major). The runtime
    ``use_gm`` guard degenerates to row-major when the group has fewer than group_m
    M blocks, so a tiny or skewed group can never be corrupted."""
    lm_r = local // n_blocks
    bn_r = local % n_blocks
    if const_expr(group_m <= 1):
        return lm_r, bn_r
    GM = fx.Int32(group_m)
    blocks_m = ceildiv(m_end - m_start, block_m_size)
    per_super = GM * n_blocks
    super_id = local // per_super
    first_m = super_id * GM
    rem_m = blocks_m - first_m
    size_m = _min_i32(rem_m, GM)
    in_super = local % per_super
    lm_g = first_m + (in_super % size_m)
    bn_g = in_super // size_m
    use_gm = blocks_m > GM
    return arith.select(use_gm, lm_g, lm_r), arith.select(use_gm, bn_g, bn_r)


@functools.lru_cache(maxsize=256)
def _compile_grouped_bf16(
    *,
    K: int,
    G: int,
    trans_b: bool,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    group_m: int = _DEFAULT_GROUP_M,
    num_xcd: int = _DEFAULT_NUM_XCD,
    nt_vmcnt: int = 3,  # gfx950 G2S LDS hazard: vmcnt>=4 races (nondeterministic); 3 is det
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
):
    """Compile (cached) the M-grouped BF16 GEMM launcher. trans_b=True -> NT forward
    (B [G,N,K]); trans_b=False -> NN dgrad (B [G,K,N])."""
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    # A partial trailing k slab is masked at the load, so K only needs a 16B-aligned
    # row stride; the pipeline still needs three k slabs to prefetch over.
    assert K % 8 == 0, f"bf16 grouped needs K % 8 == 0 (got K={K})"
    assert ceildiv(K, BLOCK_K) >= 3, f"bf16 grouped needs K > {2 * BLOCK_K} (got K={K})"
    assert G >= 1

    # ``layout`` must stay a plain str captured by the kernel closure: flydsl's
    # on-disk compile cache keys on the kernel source plus its *scalar* closure
    # values, so binding the tile fn here instead (e.g. functools.partial) makes
    # the NT and NN kernels hash identically and silently share one binary.
    layout = "nt" if trans_b else "nn"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel_grouped(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_offs: fx.Tensor,  # int32 view of int64 [G+1]; _load_go reads the low word
        c_n: fx.Int32,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)

        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))

        # total_tiles on-device (O(G) scan; no host read of the group lens).
        total_tiles = fx.Int32(0)
        prev_off = _load_go(go_div, 0)
        for g in range_constexpr(G):
            nxt_off = _load_go(go_div, g + 1)
            total_tiles = total_tiles + ceildiv(nxt_off - prev_off, BLOCK_M) * n_blocks
            prev_off = nxt_off
        total_tiles = _readfirstlane_i32(total_tiles)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x

        def _emit():
            tt = xcd_remap_pid(pid, total_tiles, num_xcd)
            # Re-scan (L1-cached) to map the tile id -> owning group + that group's
            # first tile; keeping the offsets live across the guard costs occupancy.
            cum = fx.Int32(0)
            group_idx = fx.Int32(0)
            tile_start = fx.Int32(0)
            p = _load_go(go_div, 0)
            for g in range_constexpr(G):
                nx = _load_go(go_div, g + 1)
                nc = cum + ceildiv(nx - p, BLOCK_M) * n_blocks
                in_group = (tt >= cum) & (tt < nc)
                group_idx = arith.select(in_group, fx.Int32(g), group_idx)
                tile_start = arith.select(in_group, cum, tile_start)
                cum = nc
                p = nx

            # Every value below is wave-uniform, but it is reached through a vector
            # buffer load of group_offs, so the compiler's divergence analysis marks
            # it divergent. Left unpinned it lands the A/B/C buffer descriptors in
            # VGPRs and the backend wraps *every* buffer_load/buffer_store in a
            # readfirstlane/saveexec waterfall loop (~13 extra instructions each,
            # and a basic-block split that blocks G2S/MFMA interleaving). Pinning the
            # scan outputs to SGPRs collapses the SRDs back to scalar regs.
            group_idx = _readfirstlane_i32(group_idx)
            tile_start = _readfirstlane_i32(tile_start)
            m_start = _readfirstlane_i32(_load_go(go_div, group_idx))
            m_end = _readfirstlane_i32(_load_go(go_div, group_idx + 1))
            m_total = _readfirstlane_i32(_load_go(go_div, G))
            local_block_m, block_n = _grouped_block_mn(
                tt - tile_start, m_start, m_end, n_blocks, BLOCK_M, group_m
            )
            local_block_m = _readfirstlane_i32(local_block_m)
            block_n = _readfirstlane_i32(block_n)
            m_row = _readfirstlane_i32(m_start + local_block_m * BLOCK_M)

            # Rows this tile may touch, clamped to the tile height: keeps every
            # num_records product inside int32 even for a multi-GB pool.
            a_rows = _min_i32(m_total - m_row, fx.Int32(BLOCK_M))
            c_rows = _min_i32(m_end - m_row, fx.Int32(BLOCK_M))

            a_off = _i64(m_row) * fx.Int64(K * 2)
            c_off = _i64(m_row) * _i64(c_n) * fx.Int64(2)
            # One expert slab: [N,K] (NT) or [K,N] (NN) -- both are c_n*K elements.
            b_off = _i64(group_idx) * _i64(c_n) * fx.Int64(K * 2)

            A_tile = make_bf16_fp16_tile_tensor(_base_i64(A), a_off, a_rows * fx.Int32(K))
            B_tile = make_bf16_fp16_tile_tensor(_base_i64(B), b_off, c_n * fx.Int32(K))
            C_tile = make_bf16_fp16_tile_tensor(_base_i64(C), c_off, c_rows * c_n)

            gemm_bf16_tile(
                layout,
                A_tile,
                B_tile,
                C_tile,
                c_rows,
                c_n,
                lds,
                fx.Int32(0),
                block_n,
                K=K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                nt_vmcnt=nt_vmcnt,
                # A's view spans a_rows (>= c_rows) rows of the pool; the tile
                # cannot infer that bound from c_m.
                a_oob_elems=a_rows * fx.Int32(K),
            )

        if pid < total_tiles:
            _emit()

    @flyc.jit
    def launch_grouped(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_offs: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        # Host upper bound: every group can waste at most one partial M block.
        grid_x = (ceildiv(c_m, BLOCK_M) + fx.Int32(G)) * ceildiv(c_n, BLOCK_N)
        kernel_grouped(
            A,
            B,
            C,
            group_offs,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_grouped


# (K, G, trans_b, BLOCK_M, BLOCK_N, group_m, num_xcd) -> flyc.compile'd object.
_GROUPED_COMPILED_CACHE: dict = {}


def grouped_gemm_bf16_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool = False,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    group_m: int = _DEFAULT_GROUP_M,
    num_xcd: int = _DEFAULT_NUM_XCD,
) -> torch.Tensor:
    """FlyDSL BF16 grouped GEMM (M-grouped), matching the Triton entry.

    out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g]
      trans_b=True  (forward): b [G, N, K] (b[g]^T); NT kernel.
      trans_b=False (dgrad)  : b [G, K, N];          NN kernel.
    a [M_total, K] bf16; group_offs [G+1] int (prefix sum of the group lengths).
    Output is bf16. Requires K % 8 == 0 and K > 128: the last K slab may be partial
    (masked), but the tile's pipeline needs three slabs of the LDS K-depth to run
    its prologue/loop/epilogue.
    """
    assert a.ndim == 2 and b.ndim == 3, f"a {tuple(a.shape)}, b {tuple(b.shape)}"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    M_total, K = a.shape
    G = b.shape[0]
    N = b.shape[1] if trans_b else b.shape[2]
    K_b = b.shape[2] if trans_b else b.shape[1]
    assert K == K_b, f"K mismatch a={K} b={K_b}"
    assert group_offs.numel() == G + 1, f"group_offs len {group_offs.numel()} != G+1 ({G + 1})"
    # One expert slab is addressed with an int32 in-buffer offset (the i64 rebase
    # only folds the slab base), so the slab itself must stay under 2^31 elements.
    assert N * K < 2**31, f"per-expert slab N*K={N * K} exceeds the int32 buffer offset range"

    out = torch.empty((M_total, N), device=a.device, dtype=torch.bfloat16)
    if M_total == 0:
        return out

    # The kernel reads group_offs as int64 low-words via a free int32 view.
    go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = go64.contiguous().view(torch.int32)

    launch = _compile_grouped_bf16(
        K=K,
        G=G,
        trans_b=trans_b,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        group_m=group_m,
        num_xcd=num_xcd,
    )
    # A/B/C stay 2-D/3-D: a flat view(-1) overflows the int32 shape ABI on a big
    # pool, and the kernel only ever reads their base pointer (rebased per tile).
    args = (
        a.contiguous(),
        b.contiguous(),
        out,
        go32,
        M_total,
        N,
        torch.cuda.current_stream(),
    )
    key = (K, G, trans_b, BLOCK_M, BLOCK_N, group_m, num_xcd)
    compiled = _GROUPED_COMPILED_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _GROUPED_COMPILED_CACHE[key] = compiled
    compiled(*args)
    return out


@ASTRewriter.transform
def gemm_bf16_variable_k_tile(
    A,
    B,
    C,
    group_idx,
    block_m,
    block_n,
    m_start,
    m_end,
    lds,
    out_m_rt,
    out_n_rt,
    *,
    G,
    OUT_M,
    OUT_N,
    BLOCK_M,
    BLOCK_N,
    out_fp16=False,
    c_cache_modifier=0,
):
    CHUNK = 4
    WGRAD_WAVES = 8  # fixed 8 waves per block
    assert BLOCK_M >= 128 and BLOCK_N >= 64 and BLOCK_M % 128 == 0 and BLOCK_N % 64 == 0
    N_TILES_A = BLOCK_M // 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = (BLOCK_M // 16) // WGRAD_WAVES
    N_LDS_STEPS_B = (BLOCK_N // 16) // WGRAD_WAVES
    N_WAVE_N = WGRAD_WAVES // 2

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // N_WAVE_N
    wave_n = wave_id % N_WAVE_N

    group_tokens = m_end - m_start
    bf16_ir = fx.BFloat16.ir_type
    # base offset and per-group span (group_tokens * OUT * 2 bytes) can both exceed
    # int32 for a worst-case pool; compute in int64 so the span does not wrap before
    # make_bf16_buffer_tensor_rebased clamps it to the 32-bit HW num_records field.
    a_base_off = _i64(m_start) * fx.Int64(OUT_M * 2)
    b_base_off = _i64(m_start) * fx.Int64(OUT_N * 2)
    a_span = _i64(group_tokens) * _i64(out_m_rt) * fx.Int64(2)
    b_span = _i64(group_tokens) * _i64(out_n_rt) * fx.Int64(2)
    gA = make_bf16_buffer_tensor_rebased(A, bf16_ir, a_base_off, a_span)
    gB = make_bf16_buffer_tensor_rebased(B, bf16_ir, b_base_off, b_span)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a, _ = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_M, N_LDS_STEPS_A)
    gl_off_b, _ = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_N, N_LDS_STEPS_B)

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + LDS_BLOCK_M
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + LDS_BLOCK_N
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    NTA16 = N_TILES_A * 2
    NTB16 = (BLOCK_N // 16) // (2 * N_WAVE_N)
    N_ACCUMS16 = NTA16 * NTB16
    mfma = Mfma16x16x32(NTA16, NTB16)
    a_s2r = S2RLoaderTr16x32Bf16(wave_m, NTA16)
    b_s2r = S2RLoaderTr16x32Bf16(wave_n, NTB16)
    ACC_VEC_N = 4
    N_ACCUMS_EFF = N_ACCUMS16
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, bf16_ir, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, bf16_ir, wave_id)
    out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, G * OUT_M, OUT_N, out_ty, cache_modifier=c_cache_modifier)

    acc00 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc01 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc10 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc11 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    for quad in (acc00, acc01, acc10, acc11):
        for reg in quad:
            fx.memref_store_vec(mfma.zero_value, reg)

    wait_barrier(0)
    b_g2s.load(lds.B_lds_cur_0, b0_off + 0 * b_k_step)
    a_g2s.load(lds.A_lds_cur_0, a0_off + 0 * a_k_step)
    b_g2s.load(lds.B_lds_cur_1, b1_off + 0 * b_k_step)
    a_g2s.load(lds.A_lds_cur_1, a1_off + 0 * a_k_step)
    if wave_m == 1:
        rocdl.s_barrier()
    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
    b_g2s.load(lds.B_lds_next_0, b0_off + 1 * b_k_step)
    a_g2s.load(lds.A_lds_next_0, a0_off + 1 * a_k_step)
    b_g2s.load(lds.B_lds_next_1, b1_off + 1 * b_k_step)
    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    # nested to isolate Python-level buffer rotation from the runtime chunk loop
    def _chunk(chunk_iv):
        chunk_idx = ArithValue(chunk_iv)
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
        for j in range_constexpr(CHUNK):
            k = chunk_idx * CHUNK + j
            # 4-buffer pipelined body: interleave s2r/g2s with the 4 mfma quadrants
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, a1_off + (k + 1) * a_k_step)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc00]
            c = mfma.call(a0, b0, c)
            for idx in range_constexpr(len(acc00)):
                fx.memref_store_vec(c[idx], acc00[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1 = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, b0_off + (k + 2) * b_k_step)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc01]
            c = mfma.call(a0, b1, c)
            for idx in range_constexpr(len(acc01)):
                fx.memref_store_vec(c[idx], acc01[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1 = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, a0_off + (k + 2) * a_k_step)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc10]
            c = mfma.call(a1, b0, c)
            for idx in range_constexpr(len(acc10)):
                fx.memref_store_vec(c[idx], acc10[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b_g2s.load(b_cur1, b1_off + (k + 2) * b_k_step)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc11]
            c = mfma.call(a1, b1, c)
            for idx in range_constexpr(len(acc11)):
                fx.memref_store_vec(c[idx], acc11[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

    for chunk_iv in range(n_chunks):
        _chunk(chunk_iv)

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    def _emit_q(cfrag, q_m, q_n):
        for i in range_constexpr(NTA16):
            for j in range_constexpr(NTB16):
                blk = [cfrag[i * NTB16 + j]]
                store_c.store16(blk, group_idx, q_m + i * 16, q_n + j * 16, OUT_M, OUT_N)

    # The store takes in-group coordinates and masks the m/n overhang, so a
    # partial tile on either output axis is safe.
    local_m = block_m * BLOCK_M + wave_m * (NTA16 * 16)
    local_n = block_n * BLOCK_N + wave_n * (NTB16 * 16)
    _emit_q(c00, local_m + 0, local_n + 0)
    _emit_q(c01, local_m + 0, local_n + LDS_BLOCK_N)
    _emit_q(c10, local_m + LDS_BLOCK_M, local_n + 0)
    _emit_q(c11, local_m + LDS_BLOCK_M, local_n + LDS_BLOCK_N)


@functools.lru_cache(maxsize=64)
def _compile_grouped_variable_k_bf16(
    OUT_M,
    OUT_N,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
):
    # Partial tiles on either output axis are masked at the store, so neither dim
    # has to tile exactly. The 128-bit global loads do need an 8-element (16B) row
    # stride on both operands.
    assert OUT_M % 8 == 0 and OUT_N % 8 == 0, "OUT_M/OUT_N must be multiples of 8 for 128-bit loads"
    N_BLOCKS_M = (OUT_M + BLOCK_M - 1) // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_variable_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_k_offsets: fx.Tensor,
        masked_k: fx.Tensor,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        go_base = fx.Int64(_ptrtoint(_get_iter(group_k_offsets)))
        gk_base = fx.Int64(_ptrtoint(_get_iter(masked_k)))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x

        def _do_tile(tile_idx):
            tile = xcd_remap_pid(tile_idx, TOTAL, num_xcd)
            group_idx = tile // TILES_PER_GROUP
            local_tile = tile % TILES_PER_GROUP
            block_m = local_tile // N_BLOCKS_N
            block_n = local_tile % N_BLOCKS_N
            m_start = _load_i64_as_i32(go_base, group_idx)
            # bound K to valid rows; padding tail never read
            m_end = m_start + _load_i64_as_i32(gk_base, group_idx)
            gemm_bf16_variable_k_tile(
                A,
                B,
                C,
                group_idx,
                block_m,
                block_n,
                m_start,
                m_end,
                lds,
                out_m_rt,
                out_n_rt,
                G=G,
                OUT_M=OUT_M,
                OUT_N=OUT_N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                out_fp16=out_fp16,
            )

        _do_tile(pid)

    @flyc.jit
    def launch_grouped_variable_k(
        A,
        B,
        C,
        group_k_offsets,
        masked_k,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int32(TOTAL)
        kernel_grouped_variable_k(
            A,
            B,
            C,
            group_k_offsets,
            masked_k,
            out_m_rt,
            out_n_rt,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_variable_k


_COMPILED_GROUPED_GEMM_CACHE = {}


def grouped_gemm_bf16_variable_k_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    num_xcd: int = 8,
    trans_c: bool = False,
) -> torch.Tensor:
    """FlyDSL BF16 variable-K grouped GEMM (TN wgrad), matching the fp8 variable-K entry.

    out[g] = a[offs[g]:offs[g]+masked_k[g]].T @ b[offs[g]:offs[g]+masked_k[g]]
    a [M_total, OUT_M] bf16, b [M_total, OUT_N] bf16, out [G, OUT_M, OUT_N]
    (or [G, OUT_N, OUT_M] when trans_c). The contraction dim (K) is the token
    axis and varies per group; ``masked_k`` defaults to the padded group span.
    """
    assert a.dim() == 2 and b.dim() == 2 and a.shape[0] == b.shape[0]
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    # (a^T @ b)^T == b^T @ a, so a transposed output is just the operands swapped --
    # which keeps the epilogue on the coalesced store. Transposing in the store
    # instead scatters it into 2-byte writes 16 rows apart and costs up to 2.2x on
    # wgrad shapes, where the output is ~14x larger per flop than in the forward.
    # Mirrors the fp8 variable-K entry.
    if trans_c:
        a, b = b, a
    OUT_M = a.shape[1]
    OUT_N = b.shape[1]
    G = group_k_offsets.numel() - 1
    out_fp16 = out_dtype == torch.float16
    out = torch.empty((G, OUT_M, OUT_N), device=a.device, dtype=out_dtype)
    # index tables loaded as i64 in-kernel
    offsets_i64 = group_k_offsets if group_k_offsets.dtype == torch.int64 else group_k_offsets.to(torch.int64)
    # per-expert valid K length; default = padded span
    if masked_k is None:
        masked_k_i64 = (offsets_i64[1:] - offsets_i64[:-1]).contiguous()
    else:
        assert masked_k.numel() == G, f"masked_k len {masked_k.numel()} != G {G}"
        masked_k_i64 = (masked_k if masked_k.dtype == torch.int64 else masked_k.to(torch.int64)).contiguous()
    launch = _compile_grouped_variable_k_bf16(
        OUT_M,
        OUT_N,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
    )
    # Pass operands as an int32 view: the kernel only reads their base pointer (rebased
    # by byte offsets in make_bf16_buffer_tensor_rebased), so dtype is irrelevant, while
    # halving the element count keeps the flyc CABI 32-bit numel field from overflowing
    # on production pools (>2^31 bf16 elems).
    args = (
        a.contiguous().view(torch.int32).view(-1),
        b.contiguous().view(torch.int32).view(-1),
        out.view(-1),
        offsets_i64,
        masked_k_i64,
        OUT_M,
        OUT_N,
        torch.cuda.current_stream(),
    )
    key = (OUT_M, OUT_N, G, BLOCK_M, BLOCK_N, out_fp16)
    compiled = _COMPILED_GROUPED_GEMM_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_GEMM_CACHE[key] = compiled
    compiled(*args)
    return out
