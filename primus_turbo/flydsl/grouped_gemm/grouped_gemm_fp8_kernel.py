###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL fp8 per-tensor (TENSORWISE) grouped GEMM, M-grouped operator.
Forward (NT) and dgrad (NN) for MoE GEMM; group_offs [G+1] splits M into G groups.
Grid is over-launched; each WG resolves its (group, tile) on-device (no CPU sync)."""

from collections import namedtuple

# Eager NT/NN pre-RA schedule: max-memory-clause interleaves ds_read incremental waits
# with MFMA, which is what the feed-latency regime wants.
_GG_SCHED_HINTS = {
    "llvm_options": {
        "amdgpu-sched-strategy": "max-memory-clause",
        "enable-post-misched": True,
        "lsr-drop-solution": True,
    }
}

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_epilogue_helper import (
    StoreCdSwiGLUCShuffle,
    StoreCSwiGLU,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderShear,
    S2RLoaderTr,
    StoreCPerTensor,
    StoreCPerTensorCShuffle,
    StoreCPerTensorRowN,
    _lane_tbl_count_le,
    _lane_tbl_get,
    _lane_tbl_load,
    _lane_tbl_scan,
    _robust_time,
    _sgpr_tbl_load,
    _sgpr_tbl_pick,
    _sgpr_tbl_scan,
    _store_quadrants,
    asm_mma_do,
    compile_with_scratch_out,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    compute_global_swizzle_shear,
    make_fp8_buffer_tensor_rebased,
    make_row_band_resource,
    make_value_attrs,
    mask_a_tail,
    resolve_accum_out,
    shear_mbias,
    spin_flag_eq,
    wait_barrier,
    xcd_remap_pid,
    xcd_remap_pid_u,
)
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    _readlane_i32,
    _sload_i32,
    _wave_count_le_i32,
    ceildiv,
    ceildiv_pow2,
    udiv,
    uindex,
    umod,
)

# Baked NT super-block swizzle width (0=row-major); autotune sweeps group_m per shape.
_GROUPED_NT_GROUPM = 0
# Group-table read path for the NT/NN tile decode: at/below this G the whole [G+1] offs
# table stays in SGPRs (tile decode gates on lgkmcnt, not a per-lane vmcnt gather); past it
# the lane-resident gather is cheaper in SGPR pressure and O(1) in G.
_SGPR_GO_MAX_G = 8
# Band-cyclic group interleave (skew load-balance, keeps B-stripe L2 reuse); always on.
_WG_INTERLEAVE = True

# Epilogue store schedules for the grouped NT/NN bodies (`nt_esplit`/`nn_esplit`, see
# _store_split): the tile's four accumulator quadrants (0=c00, 1=c01, 2=c10, 3=c11) split into
# barrier-separated store batches. 0 reproduces the single 128-store burst; 4 is the deployed one.
_NN_E_SCHED = {
    0: ((), (0, 2, 1, 3)),
    4: ((), (0,), (2,), (1,), (3,)),
}


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


def _build_mfma(N_TILES_A, N_TILES_B, cbsz, blgp, asm_mode=None):
    """Mfma16x16x128 with the e5m2/hybrid atom applied when cbsz|blgp, and (when asm_mode
    is given) an inline-asm _do_mma at that mode ("2"=AGPR in-place, "3"=VGPR in-place).
    asm_mode=None keeps the intrinsic MMA (VGPR accs)."""
    mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
    if cbsz or blgp:
        _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
        _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
        mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))
    if asm_mode is not None:
        mfma._do_mma = lambda _a, _b, _c: asm_mma_do(_a, _b, _c, mode=asm_mode, cbsz=cbsz, blgp=blgp)
    return mfma


def _nn_b_tr_issue(b_s2r, lds_src, vmcnt=None):
    """Issue every B tile's transpose reads of one LDS stage and leave them in flight.
    Like S2RLoaderTr.load minus its trailing drain, so the caller places the drain where
    the operand is consumed (_nn_b_tr_mfma) or lets a later scored wait cover it."""
    return [b_s2r._issue_one(lds_src, t, None, vmcnt) for t in range_constexpr(b_s2r.n_tiles)]


def _nn_b_tr_mfma(mfma, raw, a_frag, c):
    """Consume _nn_b_tr_issue's in-flight reads one B tile at a time (wait that tile, run
    its mfma column) so LDS latency hides behind mfma. Needed because the inline-asm reads
    are unscored. Returns accumulators plus assembled b fragments for a later reusing group."""
    n_rd = len(raw[0])  # ds_read per B tile
    b_frag = []
    for j in range_constexpr(mfma.n_tiles_b):
        # lgkm retires in issue order, so the tail still outstanding is the later tiles'.
        S2RLoaderTr._wait_lgkmcnt(n_rd * (mfma.n_tiles_b - 1 - j))
        rocdl.sched_barrier(0)  # nothing may cross a hand-written wait the backend cannot see
        b_frag.append(S2RLoaderTr._assemble(raw[j]))
        for i in range_constexpr(mfma.n_tiles_a):
            c[mfma.idx(i, j)] = mfma._do_mma(a_frag[i], b_frag[j], c[mfma.idx(i, j)])
    return c, b_frag


def _store_split(store_c, quad, base_row, base_col, esplit, full):
    """Emit a body's four accumulator quadrants as _NN_E_SCHED[esplit]'s barrier-separated
    store batches instead of one burst. ``quad`` is (frag, d_row, d_col) per quadrant in
    (c00, c01, c10, c11) order; esplit=0 reproduces the single burst."""
    sched = [tuple(q for q in b if full or q % 2 == 0) for b in _NN_E_SCHED[esplit]]
    for bi in range_constexpr(len(sched)):
        if const_expr(bi == 1 or (bi > 1 and len(sched[bi]) > 0)):
            rocdl.s_barrier()
        for qi in range_constexpr(len(sched[bi])):
            frag, d_row, d_col = quad[sched[bi][qi]]
            store_c.store(frag, base_row + d_row, base_col + d_col)


# PERSISTENT grouped NN dgrad: a fixed grid of num_sms WGs strides the tile space via scf.for.
_NUM_CUS_CACHE = None


def _num_cus():
    global _NUM_CUS_CACHE
    if _NUM_CUS_CACHE is None:
        _NUM_CUS_CACHE = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_CUS_CACHE


def _compile_grouped_nn(
    *,
    K: int,
    G: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    waves_per_eu: int = 2,
    nt_vmcnt: int = 3,
    num_xcd: int = 8,
    agpr_inplace: bool = True,
    acc_mode: str = "agpr",  # "agpr"=AGPR in-place (mma mode 2); "vgpr"=VGPR in-place (mode 3, avoids the accvgpr shuffle)
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
    group_m: int = 0,
    group_n: int = 0,  # >0 (with group_m): 2D band swizzle (N split into width-group_n bands) for big-N L2 reuse; sized off geometry, not a hardcoded N threshold
    store_cshuffle: bool = False,  # True = vectorized 128b CShuffle store_c (LDS-staged); False = scalar buffer_store_short
    sched_schedbar: bool = False,  # True = before-mfma inner s_barrier -> sched_barrier(0) (no runtime WG sync)
    persistent: bool = True,  # True = scf.for tile loop (fixed grid, cap_cu reserves CUs); False = one tile/WG + s_endpgm over-launch guard (full-device default)
    cap_cu: int = -1,  # >0: cap grid to this many WGs (reserve device CUs for comm-compute overlap). <=0: full device.
    i64_traverse: bool = False,  # B[K,N] traversal via per-load i64 SRD re-base (lifts G*K*n < 2^32 cap)
    nn_halfn: bool = True,  # skip the all-OOB b1 half (c01/c11 mfma+store) on the last N-block when c_n%BLOCK_N in (0, LDS_BLOCK_N]
    nn_halfn_noload: bool = False,  # (with nn_halfn) also drop the all-OOB b1 g2s loads + s2r; that half body then pays a full vmcnt(0) per K-iter since its halved g2s count makes the graded drain racy
    cstore_aux: int = 0,  # non-temporal aux for the dx C store (1=GLC bypass-L2): keeps write-once dx out of L2 so the reused weight B stays resident. 0 = default
    nn_loop_tr_vmcnt: int = -1,  # steady-state B transpose-read g2s drain hint. -1 = none: the per-K-iter rendezvous below already covers every main-loop LDS read, so an extra vmcnt only throttles g2s
    N: int = 0,  # compile-time output width (0 = unknown): lets _col_safe prove the epilogue's column OOB select dead. Part of the autotune cache key
    n_stride: int = 0,  # >0: padded N storage pitch for B (rows stored at n_stride, real width fed via N/c_n)
    nn_b0_dist2: bool = False,  # distance-2 prefetch for the always-load B0 half: 3 LDS buffers (cur/next/next2) so g2s writes the oldest consumed slot (no WAR stall) and a 3-iter window hides transpose-load latency. A/b1 stay distance-1 (+16KB LDS, still 1 WG/CU)
    nn_kshear: bool = True,  # K % 128 == 64: fetch A's line-aligned window per row instead of its raw K-block, so no row's 128B load splits a cache line (measured +25.7% TCP_TCC_READ_REQ on dY[M,2880]). Off = legacy split loads
    nn_elgk: bool = True,  # graded drain of the B transpose reads: the b0 stage rides the a0 fragment's compiler-scored wait, the b1 stage drains one tile at a time inside its mfma column. Off = one lgkmcnt(0) per stage with no mfma to cover it
    nn_esplit: int = 4,  # epilogue store schedule, see _NN_E_SCHED: how the tile's four accumulator quadrants are split into barrier-separated store batches. 0 = one 128-store burst after the trailing barrier
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
    dglu: bool = False,  # fuse the SwiGLU gradient into the epilogue: read l1, write dl1 [M,2I] and grad_probs partials, so dact never reaches HBM
    glu_i: int = 0,  # activation width I; the GEMM's N already equals it, so no geometry changes (unlike the fwd)
):
    """Persistent (CPU-sync-free) grouped NN dgrad: a fixed grid of WGs strides the tile
    space via scf.for, amortising per-WG fixed cost. ``group_m``/``group_n`` port the NT
    fwd L2-reuse swizzle (gated in-kernel, row-major fallback for small/skewed groups)."""
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert G >= 1
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K
    # Known N: scalar epilogue columns are always < N, so the per-element OOB compare is dead (mask kept when N % LDS_BLOCK_N != 0).
    _col_safe = N > 0 and (N % BLOCK_N == 0 or (nn_halfn and N % LDS_BLOCK_N == 0))
    _nb_c = ceildiv(N, BLOCK_N) if N > 0 else 0  # compile-time N-block count (0 = take it from c_n)
    # Pad-both: B carries a padded N pitch NS; only the real N columns are computed/stored.
    NS = n_stride if n_stride else N
    assert n_stride == 0 or NS >= N > 0, f"n_stride={n_stride} must be >= real N={N}"
    # Group-offs table form (see _SGPR_GO_MAX_G): SGPR/s_buffer_load vs lane-resident gather.
    _sgo = G <= _SGPR_GO_MAX_G
    _b32 = N > 0 and G * K * NS < 2**31
    _B_GRP = K * NS if _b32 else 1
    # A (dY) row pitch is K, so K%128!=0 straddles every other row's K-block across a cache
    # line (two L1->L2 requests). Fetching each row's enclosing line restores one request/row;
    # halves rotate over 3 LDS slots. Needs the CShuffle pool's LDS, so the two are exclusive.
    _kshear = nn_kshear and K % BLOCK_K == 64 and not store_cshuffle
    _cshuf_lds = store_cshuffle or not _kshear
    # Graded B drain only applies to the inline-asm transpose reads: the intrinsic path is
    # scored, so the backend already interleaves incremental waits with the mfma there.
    _elgk = nn_elgk and agpr_inplace and acc_mode == "agpr"
    # The split reorders the quadrants of the scalar epilogue; the LDS-staged CShuffle one
    # shares a single staging buffer across them and has to keep the emitted order.
    _esplit = 0 if store_cshuffle else nn_esplit
    if dglu:
        assert glu_i > 0 and not store_cshuffle and not beta_is_one
        assert N == glu_i, f"dglu needs the GEMM's N to be I, got N={N} I={glu_i}"
        # The half-N skip is fine for the pair store: the quadrant it drops is
        # entirely past I on the boundary block, so it contributes nothing to
        # grad_probs either, and the epilogue takes c_hi=None for it.
        _col_safe = N > 0 and (N % BLOCK_N == 0 or (nn_halfn and N % LDS_BLOCK_N == 0))

    _cshuf_ty = fx.Float16 if out_fp16 else fx.BFloat16
    _cshuf_n = 8 * 16 * (N_TILES_B * 16)
    if dglu:
        # The staging pool and the shear's two extra A slots cannot both fit in
        # the 160 KB that keeps this at one workgroup per CU, so the shear goes;
        # dropping it is also what turns _cshuf_lds on. f32 because it stages
        # dact, an input to the gradient: rounding to out_ty first would add a
        # second rounding ahead of the math. One staging region per wave_m group
        # (16 rows x the full 256 columns) = 32 KB, what the shear returned.
        _kshear = False
        _cshuf_lds = True
        _cshuf_ty = fx.Float32
        _cshuf_n *= 2

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        if _kshear:
            # 3rd slot per A half: holds the partner (k-1) window the splice reads.
            A_lds_prev_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_prev_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        if nn_b0_dist2:
            # 3rd B0 buffer (distance-2 rotate); allocated only when enabled -> default 0 LDS.
            B_lds_next2_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        if _cshuf_lds:
            C_lds_shuffle: fx.Array[_cshuf_ty, _cshuf_n, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_nn_persistent(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,  # int32 view of int64 [G+1]; _load_go reads low word at i32[2*idx]
        L1: fx.Tensor,  # dglu only: saved fc1 pre-activation [M,2I]; C is passed twice otherwise
        PROBS: fx.Tensor,  # dglu only: routing probs [M] fp32
        GRAD_PROBS_PARTIAL: fx.Tensor,  # dglu only: [n_blocks*2, M] fp32 grad_probs partials
        c_n: fx.Int32,
        grad_probs_stride: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)  # materialize before S2RLoaderTr (dense NN note)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        # Compile-time N (same autotune key supplies c_n) folds the tile-swizzle divisors to
        # constants, so the tile-id -> (block_m, block_n) chain that gates the first g2s costs
        # magic multiplies instead of v_rcp_iflag division sequences.
        n_blocks = fx.Int32(_nb_c) if _nb_c else ceildiv(c_n, BLOCK_N)

        # On-device group scan; entry g of the tile prefix _tcs is the first tile id of group g.
        go_rs = _buffer_ops.create_buffer_resource(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        # int32 view of the int64 [G+1] table: entry g at i32 element 2*g (offsets < 2^31).
        if const_expr(_sgo):
            # Small G: uniform s_buffer_load -> the scan/decode is pure SALU off lgkmcnt.
            _gt = _sgpr_tbl_load(go_rs, G + 1, stride=2)
            _g0, _g1 = _gt[:G], _gt[1:]
            _nt = [ceildiv_pow2(_g1[g] - _g0[g], BLOCK_M) * n_blocks for g in range_constexpr(G)]
            _tcs_end = _sgpr_tbl_scan(_nt)  # entry g = tiles owned by groups <= g
            _tcs = [_tcs_end[g] - _nt[g] for g in range_constexpr(G)]
            total_tiles = _tcs_end[-1]
            m_total = _gt[G]
        else:
            # Lane-parallel scan: lane g holds tile prefix _tcs[g] via one wave scan (no O(G) carry).
            lane_g = fx.thread_idx.x % 64
            _g0 = _lane_tbl_load(go_rs, lane_g, G + 1, stride=2)
            _g1 = _lane_tbl_load(go_rs, lane_g, G + 1, stride=2, first=1)
            _own = [lane_g + fx.Int32(64 * c) < fx.Int32(G) for c in range_constexpr(len(_g0))]
            _nt = [
                arith.select(_own[c], ceildiv_pow2(_g1[c] - _g0[c], BLOCK_M) * n_blocks, fx.Int32(0))
                for c in range_constexpr(len(_g0))
            ]
            _tcs_end = _lane_tbl_scan(_nt)  # entry g = tiles owned by groups <= g
            _tcs = [_tcs_end[c] - _nt[c] for c in range_constexpr(len(_nt))]
            total_tiles = _readlane_i32(_tcs_end[-1], 63)
            m_total = _lane_tbl_get(_g0, G)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x
        nsms = fx.grid_dim.x  # persistent stride = number of launched WGs

        if const_expr(not persistent):
            # one tile per WG: pin total_tiles to SGPR and s_endpgm the over-launched WGs.
            if const_expr(not _sgo):
                total_tiles = _readfirstlane_i32(total_tiles)
            _llvm.inline_asm(
                None,
                [pid.ir_value(), arith._to_raw(total_tiles)],
                "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
                "s,s,~{scc},~{memory}",
                has_side_effects=True,
            )

        def _find_group(tt):
            """tile id -> (group B base, its first tile id, its m range), all off the same G-1
            boundary compares (no group index materialised, no per-table select chain); with
            ``_b32`` the B base is a per-group literal so the chain never rebuilds the index."""
            if const_expr(_sgo):
                _gb = [fx.Int32(g * _B_GRP) for g in range_constexpr(G)]
                return _sgpr_tbl_pick(_tcs_end[:-1], tt, (_gb, _tcs, _g0, _g1))
            gi = _lane_tbl_count_le(_tcs_end, tt)
            gb = gi if const_expr(_B_GRP == 1) else gi * fx.Int32(_B_GRP)
            return gb, _lane_tbl_get(_tcs, gi), _lane_tbl_get(_g0, gi), _lane_tbl_get(_g1, gi)

        # Per-tile body as an inlined free function so the ast-rewriter handles `if wave_m==1`/range_constexpr without mis-collecting loaders as scf.for iter_args.
        def _do_tile(t):
            # XCD remap of the tile id (bijection; identity when num_xcd<=1) for per-XCD L2 reuse of B[g].
            tt = xcd_remap_pid_u(t, total_tiles, num_xcd)
            # tt -> owning group via the tile prefix table (no per-tile O(G) rescan of group_offs).
            group_b, tile_start, m_start, m_end = _find_group(tt)
            local = tt - tile_start
            # L2-reuse tile swizzle (group_n band -> group_m 1D -> row-major fallback for small groups).
            local_block_m, block_n = _grouped_block_mn(
                local, m_start, m_end, n_blocks, BLOCK_M, group_m, group_n, n_blocks_c=_nb_c, unsigned=True
            )

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            m_row = m_start + local_block_m * BLOCK_M
            # Fold each tile element base into the i64 SRD base; in-tile offsets stay int32, num_records clamps to the group.
            # All of these are non-negative, so index_castui: the signed cast sign-extends, which drags a
            # 64-bit multiply-and-carry chain behind every base and extent.
            cn_i = arith.index(NS) if const_expr(_b32 or n_stride) else arith.index_cast(T.index, c_n)
            # Sheared A: the line-aligned windows start up to (m_row*K)%128 bytes before the
            # tile's first row, so the SRD base drops back by that much and every A offset
            # adds it again (m_row*K == 0 => the bias is 0, so the base never goes negative).
            _mb = shear_mbias(m_row, K % BLOCK_K) if const_expr(_kshear) else None
            a_base = uindex(m_row) * arith.index(K)
            a_nrec = (uindex(m_total) - uindex(m_row)) * arith.index(K)
            if const_expr(_kshear):
                a_base = a_base - uindex(_mb)
                a_nrec = a_nrec + uindex(_mb)
            if const_expr(_b32):
                # Whole B fits in int32 -> base/extent are one int32 add and one int32 subtract.
                _b_off = group_b + block_n * fx.Int32(BLOCK_N)
                b_base = uindex(_b_off)
                b_nrec = arith.index(G * K * NS) - b_base
            else:
                b_base = arith.index_cast(T.index, group_b) * arith.index(K) * cn_i + arith.index_cast(
                    T.index, block_n * BLOCK_N
                )
                b_nrec = (arith.index(G) - arith.index_cast(T.index, group_b)) * arith.index(
                    K
                ) * cn_i - arith.index_cast(T.index, block_n * BLOCK_N)
            A0_gl_offset = 0
            A1_gl_offset = LDS_BLOCK_M * K
            B0_gl_offset = 0
            B1_gl_offset = LDS_BLOCK_N

            gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
            gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            _nnwz = True  # wave-swizzle dgrad B
            if const_expr(_kshear):
                # LDS_BLOCK_M is even, so the A1 half's rows have the same line offset as the
                # A0 half's: one offset list and one reader serve both.
                gl_off_a = compute_global_swizzle_shear(
                    lane_id, wave_id, K, N_LDS_ROUNDS, m_row, K % BLOCK_K, up=True
                )
            else:
                gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            _b_pitch = fx.Int32(NS) if const_expr(n_stride) else c_n
            gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, _b_pitch, N_LDS_ROUNDS, wswz=_nnwz)

            # AGPR in-place accum (mode 2) when agpr_inplace -> off the VGPR file (spill-free).
            mfma = _build_mfma(
                N_TILES_A,
                N_TILES_B,
                cbsz,
                blgp,
                asm_mode=("2" if acc_mode == "agpr" else "3") if agpr_inplace else None,
            )

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            # A0's window -1 sits one K-block before the SRD base, so it rides the per-lane
            # offset (a negative soffset would underflow the buffer range check). The rows it
            # takes out of range are the even ones, which never read the partner window.
            a_pv_g2s = (
                G2SLoader(a_div, [o - fx.Int32(BLOCK_K) for o in gl_off_a], N_LDS_STEPS_A, F8_IR_t, wave_id)
                if const_expr(_kshear)
                else a_g2s
            )
            # B[K,N] is the contraction-traversal operand: i64 mode re-bases its SRD per load.
            b_rebase = (B, F8_IR_t, b_base, b_nrec) if i64_traverse else None
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, rebase=b_rebase)
            a_s2r = (
                S2RLoaderShear(wave_m, N_TILES_A, m_row, K % BLOCK_K, up=True)
                if const_expr(_kshear)
                else S2RLoader(wave_m, N_TILES_A)
            )

            def _ld_a(cur, prev):
                """A fragment; the sheared reader splices the current and partner windows."""
                if const_expr(_kshear):
                    return a_s2r.load(cur, prev)
                return a_s2r.load(cur)

            def _rot_a(prev, cur, nxt):
                """Advance one A half by a phase: 3-slot rotate when sheared, otherwise the
                original 2-slot swap with ``prev`` kept aliased to the slot being read."""
                if const_expr(_kshear):
                    return cur, nxt, prev
                return nxt, nxt, cur

            # B transpose-load via inline-asm ds_read_b64_tr_b8 keeps load/mfma overlap (the intrinsic forces vmcnt(0)); needs agpr_alloc>0.
            b_s2r = S2RLoaderTr(
                wave_n, N_TILES_B, 32, inline_asm=(agpr_inplace and acc_mode == "agpr"), wswz=_nnwz
            )
            if const_expr(dglu):
                store_c = StoreCdSwiGLUCShuffle(
                    A_scale,
                    B_scale,
                    C,
                    L1,
                    PROBS,
                    GRAD_PROBS_PARTIAL,
                    # Every wave's rows are disjoint, so a partial slice only has
                    # to be distinct per (column block, 16-lane half).
                    block_n * fx.Int32(2),
                    grad_probs_stride,
                    m_end,
                    glu_i,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    lds.C_lds_shuffle,
                    wave_id,
                    col_safe=_col_safe,
                    store_aux=cstore_aux,
                )
            elif const_expr(store_cshuffle):
                store_c = StoreCPerTensorCShuffle(
                    A_scale,
                    B_scale,
                    C,
                    m_end,
                    c_n,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    lds.C_lds_shuffle,
                    wave_id,
                    beta_is_one=beta_is_one,
                )
            else:
                store_c = StoreCPerTensor(
                    A_scale,
                    B_scale,
                    C,
                    m_end,
                    c_n,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    store_aux=cstore_aux,
                    col_safe=_col_safe,
                    beta_is_one=beta_is_one,
                )

            # Before-mfma scheduling barrier; after-mfma barriers stay real (gfx950 mfma-src/ds-read VGPR-overlap race).
            def _ibar():
                if const_expr(sched_schedbar):
                    rocdl.sched_barrier(0)
                else:
                    rocdl.s_barrier()

            _nt_vmcnt = nt_vmcnt

            wave_n_offset = wave_n * (N_TILES_B * 16)
            wave_m_offset = wave_m * (N_TILES_A * 16)
            base_row = m_row + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset

            # Runtime half-N skip: on the boundary block the b1 column half is all-OOB; the nq==1 body drops only its mfma+stores, leaving g2s/barriers unchanged (race-free).
            def _do_body(nq):
                _full = nq == 2  # nq==2 full; nq in {0,1} half (skip c01/c11 mfma+store)
                _ld_b1 = nq != 0  # nq==0 half-noload: drop the all-OOB b1 g2s too
                # Half body (nq==1): b1 transpose reads are dead; its g2s is re-aimed at b0 so loads become L2 hits instead of wasted HBM past c_n.
                _b1_off = B1_gl_offset if _full else B0_gl_offset
                # Half-body drain: lowering allowed in-flight only over-drains; gfx950 retires vmcnt out of order so drop one issue group for slack.
                _wi = _w2 = _w1 = (0) if nq == 0 else (N_LDS_STEPS_A + N_LDS_STEPS_B)
                # Graded drain: attach vmcnt to phase-trailing barriers that already separate a fill from its reader; nq==0 keeps the fully-drained rendezvous.
                _nd = 2 * (N_LDS_STEPS_A + N_LDS_STEPS_B)

                def _dbar():
                    if nq == 0:
                        rocdl.s_barrier()
                    else:
                        wait_barrier(_nd)

                a_cur0 = lds.A_lds_cur_0
                a_cur1 = lds.A_lds_cur_1
                a_next0 = lds.A_lds_next_0
                a_next1 = lds.A_lds_next_1
                # Sheared: a third slot per half carries the partner (k-1) window. A0 rotates
                # (prev, cur, next) with its distance-2 fill landing in the just-freed prev;
                # A1 keeps its distance-1 fill but aims it at the free slot instead of prev.
                a_prev0 = lds.A_lds_prev_0 if const_expr(_kshear) else a_cur0
                a_prev1 = lds.A_lds_prev_1 if const_expr(_kshear) else a_cur1
                b_cur0 = lds.B_lds_cur_0
                b_cur1 = lds.B_lds_cur_1
                b_next0 = lds.B_lds_next_0
                b_next1 = lds.B_lds_next_1
                c00 = [mfma.zero_value] * N_ACCUMS
                c01 = [mfma.zero_value] * N_ACCUMS
                c10 = [mfma.zero_value] * N_ACCUMS
                c11 = [mfma.zero_value] * N_ACCUMS

                b_g2s.load(b_cur0, B0_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
                a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
                if const_expr(_ld_b1):
                    b_g2s.load(b_cur1, _b1_off + arith.index(0 * BLOCK_K) * cn_i)
                a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)
                if const_expr(_kshear):
                    # Window -1 carries k-block 0's low bytes for the rows the shear displaces.
                    a_pv_g2s.load(a_prev0, A0_gl_offset + 0 * BLOCK_K)
                    a_g2s.load(a_prev1, A1_gl_offset - 1 * BLOCK_K)
                # persistent: unconditional barrier for cross-tile phase-correctness; 8w: one tile/WG so the divergent `if wave_m==1` barrier is correct.
                if const_expr(persistent):
                    rocdl.s_barrier()
                else:
                    if wave_m == 1:
                        rocdl.s_barrier()
                wait_barrier(_w1)
                b_g2s.load(b_next0, B0_gl_offset + arith.index(1 * BLOCK_K) * cn_i)
                a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
                if const_expr(_ld_b1):
                    b_g2s.load(b_next1, _b1_off + arith.index(1 * BLOCK_K) * cn_i)
                wait_barrier(_w2)

                for k in range_constexpr(K_ITERS - 2):
                    # b0's reads stay in flight: a0 is read after them and lgkm retires in order,
                    # so the wait the backend must emit for a0 (it feeds the first mfma) already
                    # covers the asm prefix; draining here only exposes b0's latency ahead of a0.
                    b0_frag = b_s2r.load(b_cur0, vmcnt=nn_loop_tr_vmcnt, drain=not _elgk)
                    a0_frag = _ld_a(a_cur0, a_prev0)
                    a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                    _ibar()
                    rocdl.s_setprio(1)
                    c00 = mfma.call(a0_frag, b0_frag, c00)
                    rocdl.s_setprio(0)
                    _dbar()
                    if const_expr(_full):
                        # a0 is resident here, so nothing scored follows to carry b1's wait:
                        # issue now (keeping the g2s and the barrier as latency cover) and
                        # drain per tile inside the mfma column below.
                        if const_expr(_elgk):
                            b1_raw = _nn_b_tr_issue(b_s2r, b_cur1, nn_loop_tr_vmcnt)
                        else:
                            b1_frag = b_s2r.load(b_cur1, vmcnt=nn_loop_tr_vmcnt)
                    b_g2s.load(b_cur0, B0_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
                    _ibar()
                    rocdl.s_setprio(1)
                    if const_expr(_full):
                        if const_expr(_elgk):
                            c01, b1_frag = _nn_b_tr_mfma(mfma, b1_raw, a0_frag, c01)
                        else:
                            c01 = mfma.call(a0_frag, b1_frag, c01)
                    rocdl.s_setprio(0)
                    _dbar()
                    a1_frag = _ld_a(a_cur1, a_prev1)
                    # Sheared: a_prev0 held window k-1, consumed by the c00/c01 mfma above, and
                    # is the slot this distance-2 fill reuses. Unsheared it aliases a_cur0.
                    a_g2s.load(a_prev0, A0_gl_offset + (k + 2) * BLOCK_K)
                    _ibar()
                    rocdl.s_setprio(1)
                    c10 = mfma.call(a1_frag, b0_frag, c10)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    if const_expr(_ld_b1):
                        b_g2s.load(b_cur1, _b1_off + arith.index((k + 2) * BLOCK_K) * cn_i)
                    if nq == 0:
                        wait_barrier(_wi)
                    else:
                        _ibar()
                    rocdl.s_setprio(1)
                    if const_expr(_full):
                        c11 = mfma.call(a1_frag, b1_frag, c11)
                    rocdl.s_setprio(0)
                    _dbar()
                    if _nt_vmcnt >= 0:
                        _llvm.inline_asm(
                            res=None,
                            operands_=[],
                            asm_string=f"s_waitcnt vmcnt({_nt_vmcnt})",
                            constraints="",
                            has_side_effects=True,
                        )
                    a_prev0, a_cur0, a_next0 = _rot_a(a_prev0, a_cur0, a_next0)
                    a_prev1, a_cur1, a_next1 = _rot_a(a_prev1, a_cur1, a_next1)
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1

                k = K_ITERS - 2
                b0_frag = b_s2r.load(b_cur0, drain=not _elgk)
                a0_frag = _ld_a(a_cur0, a_prev0)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c00 = mfma.call(a0_frag, b0_frag, c00)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_full):
                    if const_expr(_elgk):
                        b1_raw = _nn_b_tr_issue(b_s2r, b_cur1)
                    else:
                        b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                if const_expr(_full):
                    if const_expr(_elgk):
                        c01, b1_frag = _nn_b_tr_mfma(mfma, b1_raw, a0_frag, c01)
                    else:
                        c01 = mfma.call(a0_frag, b1_frag, c01)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                a1_frag = _ld_a(a_cur1, a_prev1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c10 = mfma.call(a1_frag, b0_frag, c10)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                # This stage feeds the last phase's c00, past the c11 mfma below and the slot
                # rotate, so its reads ride the next a0 fragment's wait with a whole mfma group
                # of cover.
                b0_frag = b_s2r.load(b_next0, drain=not _elgk)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                if const_expr(_full):
                    c11 = mfma.call(a1_frag, b1_frag, c11)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                a_prev0, a_cur0, a_next0 = _rot_a(a_prev0, a_cur0, a_next0)
                a_prev1, a_cur1, a_next1 = _rot_a(a_prev1, a_cur1, a_next1)
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

                a0_frag = _ld_a(a_cur0, a_prev0)
                a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
                wait_barrier(0)
                rocdl.s_setprio(1)
                c00 = mfma.call(a0_frag, b0_frag, c00)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_full):
                    if const_expr(_elgk):
                        b1_raw = _nn_b_tr_issue(b_s2r, b_cur1)
                    else:
                        b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                if const_expr(_full):
                    if const_expr(_elgk):
                        c01, b1_frag = _nn_b_tr_mfma(mfma, b1_raw, a0_frag, c01)
                    else:
                        c01 = mfma.call(a0_frag, b1_frag, c01)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                a1_frag = _ld_a(a_cur1, a_prev1)
                a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c10 = mfma.call(a1_frag, b0_frag, c10)
                if const_expr(_full):
                    c11 = mfma.call(a1_frag, b1_frag, c11)
                rocdl.s_setprio(0)
                # Epilogue store schedule: batching each quadrant behind a barrier keeps all
                # waves inside one row/column band (contiguous columns) instead of one spread
                # burst. Separator must be the barrier; a vmcnt throttle drains but never aligns.
                if const_expr(dglu):
                    # Column pairs, not quadrants: grad_probs folds c00 with c01.
                    # The half body has no c01/c11 to fold; that quadrant is all
                    # past I, so it contributes nothing and the epilogue skips it.
                    _hi0 = c01 if const_expr(_full) else None
                    _hi1 = c11 if const_expr(_full) else None
                    store_c.store_pair(c00, _hi0, base_row, base_col, LDS_BLOCK_N)
                    store_c.store_pair(c10, _hi1, base_row + LDS_BLOCK_M, base_col, LDS_BLOCK_N)
                else:
                    _store_split(
                        store_c,
                        (
                            (c00, 0, 0),
                            (c01, 0, LDS_BLOCK_N),
                            (c10, LDS_BLOCK_M, 0),
                            (c11, LDS_BLOCK_M, LDS_BLOCK_N),
                        ),
                        base_row,
                        base_col,
                        _esplit,
                        _full,
                    )

            # Wave-uniform runtime half-N predicate (block_n is uniform per tile).
            _nb_last = n_blocks - fx.Int32(1)
            _lastw = c_n - _nb_last * fx.Int32(BLOCK_N)
            _is_half = arith.select(
                block_n == _nb_last,
                arith.select(_lastw <= fx.Int32(LDS_BLOCK_N), fx.Int32(1), fx.Int32(0)),
                fx.Int32(0),
            )
            # nn_halfn=False forces the full body; default True skips the all-OOB b1 half on the boundary block.
            _half_nq = 0 if const_expr(nn_halfn_noload) else 1
            if const_expr(nn_halfn):
                if _readfirstlane_i32(_is_half) == fx.Int32(1):
                    _do_body(_half_nq)
                else:
                    _do_body(2)
            else:
                _do_body(2)

        if const_expr(persistent):
            for t in range(pid, total_tiles, nsms):
                _do_tile(t)
        else:
            _do_tile(pid)

    @flyc.jit
    def launch_grouped_nn_persistent(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,
        m_total: int,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)
        upper = (ceildiv(m_total, BLOCK_M) + G) * n_blocks
        ncus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        _cap = ncus if cap_cu <= 0 else min(int(cap_cu), ncus)
        # persistent: cap to _cap WGs; non-persistent: full grid, over-launched WGs s_endpgm.
        grid_x = arith.select(upper < _cap, upper, fx.Int32(_cap)) if persistent else upper
        # agpr_alloc=128 when accumulating in AGPR (asm-inplace mode "2").
        attrs = make_value_attrs(waves_per_eu, 128 if (agpr_inplace and acc_mode == "agpr") else 0, "512,512")
        kernel_grouped_nn_persistent(
            A,
            B,
            C,
            A_scale,
            B_scale,
            group_offs,
            # C into the unused dglu slots so one kernel body serves both; the
            # const_expr branches keep the emitted code identical here.
            C,
            C,
            C,
            c_n,
            fx.Int32(0),
            value_attrs=attrs,
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    if const_expr(dglu):

        @flyc.jit
        def launch_grouped_nn_dglu_persistent(
            A: fx.Tensor,
            B: fx.Tensor,
            DL1: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            group_offs: fx.Tensor,
            L1: fx.Tensor,
            PROBS: fx.Tensor,
            GRAD_PROBS_PARTIAL: fx.Tensor,
            m_total: int,
            c_n: fx.Int32,
            grad_probs_stride: fx.Int32,
            stream: fx.Stream,
        ):
            n_blocks = ceildiv(c_n, BLOCK_N)
            upper = (ceildiv(m_total, BLOCK_M) + G) * n_blocks
            ncus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            _cap = ncus if cap_cu <= 0 else min(int(cap_cu), ncus)
            grid_x = arith.select(upper < _cap, upper, fx.Int32(_cap)) if persistent else upper
            attrs = make_value_attrs(
                waves_per_eu, 128 if (agpr_inplace and acc_mode == "agpr") else 0, "512,512"
            )
            kernel_grouped_nn_persistent(
                A,
                B,
                DL1,
                A_scale,
                B_scale,
                group_offs,
                L1,
                PROBS,
                GRAD_PROBS_PARTIAL,
                c_n,
                grad_probs_stride,
                value_attrs=attrs,
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_grouped_nn_dglu_persistent

    return launch_grouped_nn_persistent


def _compile_grouped_nt(
    *,
    K: int,
    k_stride: int = 0,  # row stride (elems) for A/B addressing; 0 => K. >K => padded aligned storage while compute stays K (tail masked): loads an unaligned real-K at a 128-aligned stride with no cache-line split. Caller must allocate A/B with this row stride.
    G: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    waves_per_eu: int = 2,
    nt_vmcnt: int = 3,
    num_xcd: int = 1,
    agpr_inplace: bool = True,
    acc_mode: str = "agpr",  # "agpr"=AGPR in-place (mma mode 2); "vgpr"=VGPR in-place (mode 3, avoids the accvgpr shuffle)
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
    group_m: int = 0,
    group_n: int = 0,  # >0 (with group_m): 2D band swizzle (N split into width-group_n bands) for big-N L2 reuse; sized off geometry, not a hardcoded N threshold
    store_cshuffle: bool = False,  # True = vectorized 128b CShuffle store_c (LDS-staged); False = scalar buffer_store_short
    sched_schedbar: bool = False,  # True = inner per-mfma s_barrier -> sched_barrier(0) (compile-time fence, no runtime WG sync)
    cs_pipe=None,  # depth-2 cshuffle softpipe; needs persistent+store_cshuffle
    cstore_aux=None,  # non-temporal aux immediate for the C store (0 = default)
    nt_dist2: bool = True,  # True = uniform distance-2 mainloop (A1@k+2 like mx, one wait_barrier/iter, no vmcnt throttle) + runtime half-N padding-quadrant skip. False = legacy A1@k+1 + vmcnt drain
    persistent: bool = True,  # True = scf.for tile loop (fixed grid, cap_cu reserves CUs); False = one tile/WG + s_endpgm over-launch guard (full-device default)
    cap_cu: int = -1,  # >0: cap grid to this many WGs (= reserve device CUs for comm-compute overlap). <=0: use the full device CU count.
    N: int = 0,  # compile-time output width (0 = unknown): lets _col_safe prove the epilogue's column OOB select dead. Part of the autotune cache key
    n_stride: int = 0,  # >0: padded N row-count pitch for B_T (stored at [G,n_stride,KS], real width fed via N/c_n)
    nt_esplit: int = 4,  # epilogue store schedule (see _NN_E_SCHED), the twin of the NN dgrad's nn_esplit. 0 = one 128-store burst
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
    glu: bool = False,  # fuse a SwiGLU epilogue: B_T is [2I, K] gate||up, the tile pairs the two bands in registers and writes l1 [M,2I] + act [M,I]
    glu_i: int = 0,  # gate half width I (required when glu); N is this same I, i.e. the activation's width
    glu_act_aux: int = 0,  # aux immediate for the act store alone (it is pure streaming output, so evict-first may pay where it would not for l1)
):
    """Grouped NT forward (out = a @ b^T). persistent=True: a fixed grid of WGs strides the
    tile space via scf.for (cap_cu reserves CUs for comm overlap); persistent=False: one tile
    per WG + s_endpgm over-launch guard (full-device default). ``num_xcd`` remaps the global
    tile id so same-XCD WGs cluster for per-XCD L2 reuse; ``group_m``/``group_n`` add the
    L2-reuse tile swizzle (see _grouped_block_mn)."""
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert G >= 1
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K
    KS = k_stride if k_stride else K  # addressing row stride (>= K); compute dim stays K
    assert KS >= K
    # GLU mode retiles the N axis. A tile owns _NBLK = LDS_BLOCK_N columns of the
    # *gate* half and pairs them with the same columns of the up half, so its two B
    # LDS pools hold two 128-column *bands* of B_T (offset I apart) rather than the
    # two halves of one 256-wide block. Total tiles are unchanged -- ceildiv(I, 128)
    # equals ceildiv(2I, 256) -- so the mainloop does the same work; only where the
    # second pool reads and where the results land differ.
    if glu:
        assert glu_i > 0 and N == glu_i, f"glu needs N == glu_i (the gate width); got N={N} glu_i={glu_i}"
        # The pairing lives in the dist2 body's quadrant registers, and the pair
        # store is the scalar path's (CShuffle stages one fragment through LDS).
        assert nt_dist2 and not store_cshuffle, "fused GLU epilogue rides the scalar dist2 store path"
        assert not beta_is_one, "fused GLU epilogue overwrites; there is nothing to accumulate into"
    _NBLK = LDS_BLOCK_N if glu else BLOCK_N
    # Known N makes the scalar epilogue per-element OOB select dead (nt_dist2 supplies the in-bounds half-N boundary body).
    _col_safe = (
        (N > 0 and N % _NBLK == 0)
        if glu
        else (N > 0 and (N % BLOCK_N == 0 or (nt_dist2 and N % LDS_BLOCK_N == 0)))
    )
    _nb_c = ceildiv(N, _NBLK) if N > 0 else 0  # compile-time N-block count (0 = take it from c_n)
    # Group-offs table form (see _SGPR_GO_MAX_G): SGPR/s_buffer_load vs lane-resident gather.
    _sgo = G <= _SGPR_GO_MAX_G
    # Boundary-N body width, in B column-tiles per wave (N_TILES_B=2 is the full tile). When
    # the last N-block's valid width fits the lower half of the b0 LDS pool its upper columns
    # are pure padding, so one column-tile per wave covers it. Needs the runtime half-N branch.
    # GLU never narrows: the second pool is the up band, not padding columns, so both
    # column-tiles are always live and the N tail is handled by masking instead.
    _bnd_ntb = (
        N_TILES_B
        if glu
        else (1 if (nt_dist2 and N > 0 and 0 < N % BLOCK_N <= LDS_BLOCK_N // 2) else N_TILES_B)
    )
    NS = n_stride if n_stride else N
    assert n_stride == 0 or NS >= N > 0, f"n_stride={n_stride} must be >= real N={N}"
    _bn_rows = 2 * glu_i if glu else NS
    _b32 = N > 0 and G * _bn_rows * KS < 2**31
    _B_GRP = _bn_rows * KS if _b32 else 1

    _cshuf_ty = fx.Float16 if out_fp16 else fx.BFloat16
    _cshuf_n = 8 * 16 * (N_TILES_B * 16)
    _cs_pipe = persistent and store_cshuffle and bool(cs_pipe)
    # The split reorders the quadrants of the scalar epilogue; the LDS-staged CShuffle one
    # shares a single staging buffer across them and has to keep the emitted order.
    _esplit = 0 if store_cshuffle else nt_esplit
    _cshuf_alloc = (2 * _cshuf_n) if _cs_pipe else _cshuf_n
    _cstore_aux = 0 if cstore_aux is None else int(cstore_aux)

    _ss_anns = {
        "A_lds_cur_0": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_cur_1": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_next_0": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_next_1": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "B_lds_cur_0": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_cur_1": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_next_0": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_next_1": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "C_lds_shuffle": fx.Array[_cshuf_ty, _cshuf_alloc, 16],
    }
    SharedStorage = fx.struct(type("SharedStorage", (), {"__annotations__": _ss_anns}))

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_nt_persistent(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,  # glu: l1 [M, 2I]
        ACT: fx.Tensor,  # glu: act [M, I]. Otherwise unused -- the launch aliases it to C.
        PROBS: fx.Tensor,  # glu: routing probs [M] fp32. Otherwise unused, aliased to C.
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,  # int32 view of int64 [G+1]; _load_go reads low word at i32[2*idx]
        c_n: fx.Int32,  # glu: the gate width I, not the 2I the GEMM writes
    ):
        # ACT/PROBS ride along in both modes because the body has to stay inside this
        # decorator: @flyc.kernel rewrites the AST (dynamic `if` -> dispatch), and a
        # body hoisted into a plain shared function would lose that. Under const_expr
        # the plain GEMM never reads them, so they cost two dropped kernel args and
        # nothing in the mainloop.
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        # Compile-time N (same autotune key supplies c_n): see the NN twin -- constant divisors
        # turn the pre-first-g2s tile-swizzle chain into magic multiplies.
        n_blocks = fx.Int32(_nb_c) if _nb_c else ceildiv(c_n, BLOCK_N)

        # On-device group scan (mirrors kernel_grouped_nn): _tcs = per-group first tile id; _g0/_g1 = group start/end.
        go_rs = _buffer_ops.create_buffer_resource(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        # int32 view of the int64 [G+1] table: entry g at i32 element 2*g (offsets < 2^31).
        if const_expr(_sgo):
            # Small G: uniform s_buffer_load -> the scan/decode is pure SALU off lgkmcnt.
            _gt = _sgpr_tbl_load(go_rs, G + 1, stride=2)
            _g0, _g1 = _gt[:G], _gt[1:]
            _nt = [ceildiv_pow2(_g1[g] - _g0[g], BLOCK_M) * n_blocks for g in range_constexpr(G)]
            _tcs_end = _sgpr_tbl_scan(_nt)  # entry g = tiles owned by groups <= g
            _tcs = [_tcs_end[g] - _nt[g] for g in range_constexpr(G)]
            total_tiles = _tcs_end[-1]
            _m_total_v = _gt[G]
        else:
            # Lane-parallel scan: tile prefix _tcs resident in lanes (one wave scan, no O(G) carry).
            lane_g = fx.thread_idx.x % 64
            _g0 = _lane_tbl_load(go_rs, lane_g, G + 1, stride=2)
            _g1 = _lane_tbl_load(go_rs, lane_g, G + 1, stride=2, first=1)
            _own = [lane_g + fx.Int32(64 * c) < fx.Int32(G) for c in range_constexpr(len(_g0))]
            _nt = [
                arith.select(_own[c], ceildiv_pow2(_g1[c] - _g0[c], BLOCK_M) * n_blocks, fx.Int32(0))
                for c in range_constexpr(len(_g0))
            ]
            _tcs_end = _lane_tbl_scan(_nt)  # entry g = tiles owned by groups <= g
            _tcs = [_tcs_end[c] - _nt[c] for c in range_constexpr(len(_nt))]
            total_tiles = _readlane_i32(_tcs_end[-1], 63)
            _m_total_v = _lane_tbl_get(_g0, G)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x
        nsms = fx.grid_dim.x  # persistent stride = number of launched WGs

        if const_expr(not persistent):
            # one tile per WG: pin total_tiles to SGPR and s_endpgm the over-launched WGs.
            if const_expr(not _sgo):
                total_tiles = _readfirstlane_i32(total_tiles)
            _llvm.inline_asm(
                None,
                [pid.ir_value(), arith._to_raw(total_tiles)],
                "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
                "s,s,~{scc},~{memory}",
                has_side_effects=True,
            )

        def _find_group(tt):
            """tile id -> (group B base, its first tile id, its m range), all off the same G-1
            boundary compares (no group index materialised, no per-table select chain); with
            ``_b32`` the B base is a per-group literal so the chain never rebuilds the index."""
            if const_expr(_sgo):
                _gb = [fx.Int32(g * _B_GRP) for g in range_constexpr(G)]
                return _sgpr_tbl_pick(_tcs_end[:-1], tt, (_gb, _tcs, _g0, _g1))
            gi = _lane_tbl_count_le(_tcs_end, tt)
            gb = gi if const_expr(_B_GRP == 1) else gi * fx.Int32(_B_GRP)
            return gb, _lane_tbl_get(_tcs, gi), _lane_tbl_get(_g0, gi), _lane_tbl_get(_g1, gi)

        def _do_tile(tt, group_b, tile_start, m_start, m_end):
            local = tt - tile_start
            # L2-reuse tile swizzle: group_n band -> group_m 1D -> row-major fallback.
            local_block_m, block_n = _grouped_block_mn(
                local, m_start, m_end, n_blocks, BLOCK_M, group_m, group_n, n_blocks_c=_nb_c, unsigned=True
            )

            a_cur0 = lds.A_lds_cur_0
            a_cur1 = lds.A_lds_cur_1
            a_next0 = lds.A_lds_next_0
            a_next1 = lds.A_lds_next_1
            b_cur0 = lds.B_lds_cur_0
            b_cur1 = lds.B_lds_cur_1
            b_next0 = lds.B_lds_next_0
            b_next1 = lds.B_lds_next_1

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            m_row = m_start + local_block_m * BLOCK_M
            # Fold the tile's element base into the i64 SRD base; in-tile offsets stay int32.
            # All of these are non-negative, so index_castui: the signed cast sign-extends, which
            # drags a 64-bit multiply-and-carry chain behind every base and extent.
            m_total = _m_total_v if const_expr(_sgo) else _readfirstlane_i32(_m_total_v)
            a_base = uindex(m_row) * arith.index(KS)
            a_nrec = (uindex(m_total) - uindex(m_row)) * arith.index(KS)
            if const_expr(_b32):
                # Whole B_T fits in int32 -> base/extent are one int32 add and one int32 subtract.
                _b_off = group_b + block_n * fx.Int32(_NBLK * KS)
                b_base = uindex(_b_off)
                b_nrec = arith.index(G * _bn_rows * KS) - b_base
            else:
                bn_i = (
                    arith.index_cast(T.index, c_n * fx.Int32(2))
                    if const_expr(glu)
                    else (arith.index(NS) if const_expr(n_stride) else arith.index_cast(T.index, c_n))
                )
                b_base = (
                    arith.index_cast(T.index, group_b) * bn_i + arith.index_cast(T.index, block_n * _NBLK)
                ) * arith.index(KS)
                b_nrec = arith.index(G) * bn_i * arith.index(KS) - b_base
            A0_gl_offset = 0
            A1_gl_offset = LDS_BLOCK_M * KS
            B0_gl_offset = 0
            # The one offset the whole fusion turns on: under glu the second B pool is
            # the *up* band, I weight rows further in, not the next 128 columns. That is
            # what puts gate[m, j] and up[m, j] in the same lane at the same fragment
            # index, so the epilogue pairs them without a shuffle. Past the last full
            # band both bands run off the end of B_T; the SRD clamps those reads and the
            # epilogue's column mask drops the results.
            B1_gl_offset = (glu_i * KS) if glu else (LDS_BLOCK_N * KS)

            gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
            gB = make_fp8_buffer_tensor_rebased(B_T, F8_IR_t, b_base, b_nrec)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, KS, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle(lane_id, wave_id, KS, N_LDS_ROUNDS, preshuffled=False)

            # AGPR in-place accum (mode 2) when agpr_inplace -> off the VGPR file (spill-free).
            mfma = _build_mfma(
                N_TILES_A,
                N_TILES_B,
                cbsz,
                blgp,
                asm_mode=("2" if acc_mode == "agpr" else "3") if agpr_inplace else None,
            )

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = S2RLoader(wave_n, N_TILES_B)
            if const_expr(glu):
                store_c = StoreCSwiGLU(
                    A_scale,
                    B_scale,
                    C,  # l1 [M, 2I]
                    ACT,  # act [M, I]
                    PROBS,
                    m_end,
                    glu_i,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    col_safe=_col_safe,
                    store_aux=_cstore_aux,
                    act_aux=glu_act_aux,
                )
            elif const_expr(store_cshuffle):
                store_c = StoreCPerTensorCShuffle(
                    A_scale,
                    B_scale,
                    C,
                    m_end,
                    c_n,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    lds.C_lds_shuffle,
                    wave_id,
                    pipe=_cs_pipe,
                    store_aux=_cstore_aux,
                    beta_is_one=beta_is_one,
                )
            else:
                store_c = StoreCPerTensor(
                    A_scale,
                    B_scale,
                    C,
                    m_end,
                    c_n,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    col_safe=_col_safe,
                    beta_is_one=beta_is_one,
                )

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            # Per-mfma scheduling barrier; sched_schedbar swaps it for a compile-time sched_barrier(0). Prologue/cross-iter/epilog barriers stay real.
            def _ibar():
                if const_expr(sched_schedbar):
                    rocdl.sched_barrier(0)
                else:
                    rocdl.s_barrier()

            # Extra per-iter g2s drain after c11 caps how far the next iter runs ahead (-1 = off).
            _nt_vmcnt = nt_vmcnt

            if const_expr(nt_dist2):
                # Uniform distance-2 mainloop: A1 staged at k+2 so all four LDS pools prefetch a K-iter ahead; nq==1 drops the all-padding b1 half.
                _NB_DRAIN = 2 * N_LDS_STEPS_A + N_LDS_STEPS_B
                _NB_DRAIN_HALF = 2 * N_LDS_STEPS_A
                wave_n_offset = wave_n * (N_TILES_B * 16)
                wave_m_offset = wave_m * (N_TILES_A * 16)
                _base_row = m_row + wave_m_offset
                # _NBLK == BLOCK_N unless glu, where a tile spans 128 gate columns.
                _base_col = block_n * _NBLK + wave_n_offset
                if const_expr(_bnd_ntb < N_TILES_B):
                    # Narrow twin of (mfma, b_s2r, b_g2s, store_c) for the boundary body: one B
                    # column-tile per wave and one g2s load step cover the same pool rows, so the
                    # step count equals the column-tile count and the loaded rows are the ones read.
                    assert N_LDS_STEPS_B == N_TILES_B  # both BLOCK_N/128; keeps the mapping 1:1
                    mfma_b = _build_mfma(
                        N_TILES_A,
                        _bnd_ntb,
                        cbsz,
                        blgp,
                        asm_mode=("2" if acc_mode == "agpr" else "3") if agpr_inplace else None,
                    )
                    b_g2s_b = G2SLoader(b_div, gl_off_b, _bnd_ntb, F8_IR_t, wave_id)
                    b_s2r_b = S2RLoader(wave_n, _bnd_ntb)
                    store_c_b = StoreCPerTensor(
                        A_scale,
                        B_scale,
                        C,
                        m_end,
                        c_n,
                        mfma_b.idx,
                        N_TILES_A,
                        _bnd_ntb,
                        _out_ty,
                        col_safe=_col_safe,
                        beta_is_one=beta_is_one,
                    )

                def _body_d2(nq):
                    """One 256xBLOCK_N output tile on the distance-2 mainloop. nq=2 full tile;
                    nq=1 drops the all-padding b1 half of the last N-block; nq=0 is the narrow
                    boundary body (drops the padding upper half of b0 too)."""
                    _full = nq == 2
                    _bnd = nq == 0
                    _mm = mfma_b if _bnd else mfma
                    _bg2s = b_g2s_b if _bnd else b_g2s
                    _bs2r = b_s2r_b if _bnd else b_s2r
                    _st = store_c_b if _bnd else store_c
                    _bcol = block_n * BLOCK_N + wave_n * (_bnd_ntb * 16) if _bnd else _base_col
                    _nacc = N_TILES_A * (_bnd_ntb if _bnd else N_TILES_B)
                    # Same drain as the half body ("b_c0 landed + everything older"): the narrow
                    # body only deletes ds_read/mfma/g2s work, the vmcnt+barrier sequence is
                    # unchanged (pitfalls/05: never move synchronization in a boundary variant).
                    _nd = _NB_DRAIN if _full else _NB_DRAIN_HALF
                    a_c0, a_c1, a_n0, a_n1 = a_cur0, a_cur1, a_next0, a_next1
                    b_c0, b_c1, b_n0, b_n1 = b_cur0, b_cur1, b_next0, b_next1

                    c00 = [_mm.zero_value] * _nacc
                    c10 = [_mm.zero_value] * _nacc
                    if const_expr(_full):
                        c01 = [_mm.zero_value] * _nacc
                        c11 = [_mm.zero_value] * _nacc

                    _bg2s.load(b_c0, B0_gl_offset + 0 * BLOCK_K)
                    a_g2s.load(a_c0, A0_gl_offset + 0 * BLOCK_K)
                    if const_expr(_full):
                        _bg2s.load(b_c1, B1_gl_offset + 0 * BLOCK_K)
                    a_g2s.load(a_c1, A1_gl_offset + 0 * BLOCK_K)
                    if const_expr(persistent):
                        rocdl.s_barrier()
                    else:
                        if wave_m == 1:
                            rocdl.s_barrier()
                    wait_barrier(_nd)
                    _bg2s.load(b_n0, B0_gl_offset + 1 * BLOCK_K)
                    a_g2s.load(a_n0, A0_gl_offset + 1 * BLOCK_K)
                    if const_expr(_full):
                        _bg2s.load(b_n1, B1_gl_offset + 1 * BLOCK_K)
                    a_g2s.load(a_n1, A1_gl_offset + 1 * BLOCK_K)
                    wait_barrier(_nd)

                    for k in range_constexpr(K_ITERS - 2):
                        b0_frag = _bs2r.load(b_c0)
                        a0_frag = a_s2r.load(a_c0)
                        if const_expr(_full):
                            b1_frag = _bs2r.load(b_c1)
                        rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c00 = _mm.call(a0_frag, b0_frag, c00)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                        _bg2s.load(b_c0, B0_gl_offset + (k + 2) * BLOCK_K)
                        if const_expr(_full):
                            rocdl.s_barrier()
                            rocdl.s_setprio(1)
                            c01 = _mm.call(a0_frag, b1_frag, c01)
                            rocdl.s_setprio(0)
                            rocdl.s_barrier()
                        a1_frag = a_s2r.load(a_c1)
                        a_g2s.load(a_c0, A0_gl_offset + (k + 2) * BLOCK_K)
                        rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c10 = _mm.call(a1_frag, b0_frag, c10)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                        if const_expr(_full):
                            _bg2s.load(b_c1, B1_gl_offset + (k + 2) * BLOCK_K)
                        a_g2s.load(a_c1, A1_gl_offset + (k + 2) * BLOCK_K)
                        wait_barrier(_nd)
                        if const_expr(_full):
                            rocdl.s_setprio(1)
                            c11 = _mm.call(a1_frag, b1_frag, c11)
                            rocdl.s_setprio(0)
                            rocdl.s_barrier()
                        a_c0, a_n0 = a_n0, a_c0
                        a_c1, a_n1 = a_n1, a_c1
                        b_c0, b_n0 = b_n0, b_c0
                        b_c1, b_n1 = b_n1, b_c1

                    # Tail step K_ITERS-2: every stage already issued (distance 2), read only.
                    b0_frag = _bs2r.load(b_c0)
                    a0_frag = a_s2r.load(a_c0)
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c00 = _mm.call(a0_frag, b0_frag, c00)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    if const_expr(_full):
                        b1_frag = _bs2r.load(b_c1)
                        rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c01 = _mm.call(a0_frag, b1_frag, c01)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                    a1_frag = a_s2r.load(a_c1)
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c10 = _mm.call(a1_frag, b0_frag, c10)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    if const_expr(_full):
                        rocdl.s_setprio(1)
                        c11 = _mm.call(a1_frag, b1_frag, c11)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                    a_c0, a_n0 = a_n0, a_c0
                    a_c1, a_n1 = a_n1, a_c1
                    b_c0, b_n0 = b_n0, b_c0
                    b_c1, b_n1 = b_n1, b_c1

                    # Tail step K_ITERS-1: last stage issued a K-iter ago -> drain then read; K-tail mask applies to the final iter only.
                    wait_barrier(0)
                    b0_frag = _bs2r.load(b_c0)
                    a0_frag = a_s2r.load(a_c0)
                    a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
                    rocdl.s_setprio(1)
                    c00 = _mm.call(a0_frag, b0_frag, c00)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    if const_expr(_full):
                        b1_frag = _bs2r.load(b_c1)
                        rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c01 = _mm.call(a0_frag, b1_frag, c01)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                    a1_frag = a_s2r.load(a_c1)
                    a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c10 = _mm.call(a1_frag, b0_frag, c10)
                    if const_expr(_full):
                        c11 = _mm.call(a1_frag, b1_frag, c11)
                    rocdl.s_setprio(0)
                    if const_expr(glu):
                        # The quadrants are (gate, up) pairs sharing rows rather than
                        # four corners of one tile, so they leave in two pair stores.
                        _st.store_pair(c00, c01, _base_row, _bcol)
                        _st.store_pair(c10, c11, _base_row + LDS_BLOCK_M, _bcol)
                    else:
                        # Batched epilogue store, same lever as the NN dgrad body's; this one owns
                        # the forward cells of the tw suite.
                        _store_split(
                            _st,
                            (
                                (c00, 0, 0),
                                (c01 if _full else None, 0, LDS_BLOCK_N),
                                (c10, LDS_BLOCK_M, 0),
                                (c11 if _full else None, LDS_BLOCK_M, LDS_BLOCK_N),
                            ),
                            _base_row,
                            _bcol,
                            _esplit,
                            _full,
                        )

                if const_expr(glu):
                    # No half-N body under glu: the second pool holds the up band, not
                    # padding columns, so both are always live and the N tail is masked
                    # in the epilogue instead of being skipped here.
                    _body_d2(2)
                    return

                # Wave-uniform runtime half-N predicate: last N-block whose valid width fits the b0 LDS half (b1 all padding).
                _nb_last = n_blocks - fx.Int32(1)
                _lastw = c_n - _nb_last * fx.Int32(BLOCK_N)
                _is_half = arith.select(
                    block_n == _nb_last,
                    arith.select(_lastw <= fx.Int32(LDS_BLOCK_N), fx.Int32(1), fx.Int32(0)),
                    fx.Int32(0),
                )
                # _bnd_ntb<N_TILES_B narrows the SAME boundary branch instead of adding a third
                # body: c_n equals the compiled N (both come from the one autotune key), so
                # _is_half can only fire on a block whose valid width is <= the narrow span.
                _bnd_nq = 0 if _bnd_ntb < N_TILES_B else 1
                if _readfirstlane_i32(_is_half) == fx.Int32(1):
                    _body_d2(_bnd_nq)
                else:
                    _body_d2(2)
                return

            b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)
            # persistent: unconditional barrier for cross-tile phase-correctness; 8w: one tile/WG so the divergent barrier is correct.
            if const_expr(persistent):
                rocdl.s_barrier()
            else:
                if wave_m == 1:
                    rocdl.s_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            for k in range_constexpr(K_ITERS - 2):
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                _ibar()
                rocdl.s_setprio(1)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
                _ibar()
                rocdl.s_setprio(1)
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                _ibar()
                rocdl.s_setprio(1)
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
                rocdl.s_setprio(1)
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if _nt_vmcnt >= 0:
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string=f"s_waitcnt vmcnt({_nt_vmcnt})",
                        constraints="",
                        has_side_effects=True,
                    )
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            k = K_ITERS - 2
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b0_frag = b_s2r.load(b_next0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            a0_frag = a_s2r.load(a_cur0)
            a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
            wait_barrier(0)
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            wave_n_offset = wave_n * (N_TILES_B * 16)
            wave_m_offset = wave_m * (N_TILES_A * 16)
            base_row = m_row + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
            _store_quadrants(
                store_c, c00_frag, c01_frag, c10_frag, c11_frag, base_row, base_col, LDS_BLOCK_M, LDS_BLOCK_N
            )

        if const_expr(persistent):
            for t in range(pid, total_tiles, nsms):
                tt = xcd_remap_pid_u(t, total_tiles, num_xcd)
                gb, ts, ms, me = _find_group(tt)
                _do_tile(tt, gb, ts, ms, me)
        else:
            tt = xcd_remap_pid_u(pid, total_tiles, num_xcd)
            gb, ts, ms, me = _find_group(tt)
            _do_tile(tt, gb, ts, ms, me)

    def _grid_and_attrs(m_total, c_n):
        """Tile count -> grid width, plus the launch value attrs. In GLU mode a tile
        owns ``_NBLK`` columns of the *gate* half instead of BLOCK_N of the raw
        output, and ``c_n`` is that gate width -- the tile count works out the same,
        since two 128-wide bands replace one 256-wide block."""
        n_blocks = ceildiv(c_n, _NBLK)
        upper = (ceildiv(m_total, BLOCK_M) + G) * n_blocks
        ncus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        # cap_cu>0 reserves CUs for comm-compute overlap; cap_cu<=0 = full device.
        _cap = ncus if cap_cu <= 0 else min(int(cap_cu), ncus)
        # persistent: cap to _cap WGs. non-persistent: full grid, over-launched WGs s_endpgm.
        grid_x = arith.select(upper < _cap, upper, fx.Int32(_cap)) if persistent else upper
        attrs = make_value_attrs(waves_per_eu, 128 if (agpr_inplace and acc_mode == "agpr") else 0, "512,512")
        return grid_x, attrs

    if const_expr(glu):

        @flyc.jit
        def launch_grouped_nt_glu_persistent(
            A: fx.Tensor,
            B_T: fx.Tensor,
            L1: fx.Tensor,
            ACT: fx.Tensor,
            PROBS: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            group_offs: fx.Tensor,
            m_total: int,
            c_n: fx.Int32,
            stream: fx.Stream,
        ):
            grid_x, attrs = _grid_and_attrs(m_total, c_n)
            kernel_grouped_nt_persistent(
                A,
                B_T,
                L1,
                ACT,
                PROBS,
                A_scale,
                B_scale,
                group_offs,
                c_n,
                value_attrs=attrs,
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_grouped_nt_glu_persistent

    @flyc.jit
    def launch_grouped_nt_persistent(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,
        m_total: int,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x, attrs = _grid_and_attrs(m_total, c_n)
        kernel_grouped_nt_persistent(
            A,
            B_T,
            C,
            C,  # ACT unused without glu
            C,  # PROBS unused without glu
            A_scale,
            B_scale,
            group_offs,
            c_n,
            value_attrs=attrs,
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_nt_persistent


# wgrad: variable-K grouped GEMM (TN), C[g]=lhs_g^T@rhs_g; per-group runtime contraction (scf.for), K-tail clamped by the SRD num_records bound.


def _wgrad_accum(mfma, a_frags, b_frags, acc_regs):
    """One quadrant's mma accumulate, reading/writing the rmem accumulators
    in place (so the value survives the scf.for iteration boundary). Plain
    free function -> may use obj.method() (mfma.call); only the kernel-level
    scf.for body is forbidden from doing so."""
    c = [Vec(fx.memref_load_vec(r)) for r in acc_regs]
    c = mfma.call(a_frags, b_frags, c)
    for idx in range_constexpr(len(acc_regs)):
        fx.memref_store_vec(c[idx], acc_regs[idx])


def _wgrad_body_4buf(
    k,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    mfma,
    a_cur0,
    a_cur1,
    b_cur0,
    b_cur1,
    a_next0,
    a_next1,
    b_next0,
    b_next1,
    acc00,
    acc01,
    acc10,
    acc11,
    A0_off,
    A1_off,
    B0_off,
    B1_off,
    AM,
    BNs,
    NA,
    NB,
):
    """One K-tile of the masked 4-buffer distance-2 inline pipeline, as a FREE
    FUNCTION (obj.method allowed; only the kernel-level scf.for body forbids it) so
    it can run inside a runtime chunk scf.for. Identical staging to
    _compile_grouped_tn_wgrad_masked's main loop (read cur tile k, complete tile k+1's
    A-half into a_next1, prefetch tile k+2 into cur/b — caller swaps after so the
    next call's cur = this call's next), but accumulates via memref (_wgrad_accum) so the
    acc survives the scf.for boundary. Reads/over-reads past the group's tokens are
    SRD-clamped to 0 by the per-group num_records bound. Inline ds_read drain-removal
    works here because the body is straight-line within the (compile-time unrolled)
    chunk — the masked graded wait_barrier(2*NA+NB) is the only iter drain."""
    # Cast the i32 K-step to index before scaling by AM/BNs so the product/sum stay i64 (no overflow at large mg*OUT).
    k1 = arith.index_cast(T.index, k + 1)
    k2 = arith.index_cast(T.index, k + 2)
    b0 = b_s2r.load(b_cur0, drain=False)
    a0 = a_s2r.load(a_cur0)
    a_g2s.load(a_next1, A1_off + k1 * AM)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a0, b0, acc00)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    b1 = b_s2r.load(b_cur1)
    b_g2s.load(b_cur0, B0_off + k2 * BNs)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a0, b1, acc01)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    a1 = a_s2r.load(a_cur1)
    a_g2s.load(a_cur0, A0_off + k2 * AM)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a1, b0, acc10)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    b_g2s.load(b_cur1, B1_off + k2 * BNs)
    wait_barrier(2 * NA + NB)
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a1, b1, acc11)
    rocdl.s_setprio(0)
    rocdl.s_barrier()


def _band_block_mn(pid, num_pid_m, n_blocks, GM, GN):
    """2D super-block (band) tile swizzle for the wgrad per-group grid (port of dense TN
    _tn_block_mn): N split into width-GN bands, GM inside each, so the (GM A-slab + GN B-slab)
    working set stays L2-resident. Trace-time bijection over num_pid_m*n_blocks tiles."""
    band_tiles = num_pid_m * GN
    band = pid // band_tiles
    pid_in_band = pid % band_tiles
    band_n0 = band * GN
    rem_n = fx.Int32(n_blocks) - band_n0
    band_w = arith.select(rem_n < fx.Int32(GN), rem_n, fx.Int32(GN))
    nig = fx.Int32(GM) * band_w
    gid = pid_in_band // nig
    pig = pid_in_band % nig
    fpm = gid * fx.Int32(GM)
    rem_m = fx.Int32(num_pid_m) - fpm
    gsm = arith.select(rem_m < fx.Int32(GM), rem_m, fx.Int32(GM))
    return fpm + (pig % gsm), band_n0 + (pig // gsm)


def _grouped_block_mn(
    local, m_start, m_end, n_blocks, block_m_size, group_m, group_n, n_blocks_c=0, unsigned=False
):
    """Map a within-group linear tile index ``local`` to (block_m, block_n) under the
    L2-reuse tile swizzle: group_n band (2D super-block) -> group_m 1D super-block ->
    row-major. The per-group runtime guards (bpr_g>group_m / n_blocks>group_n)
    degenerate to row-major for small/skewed groups so they can never corrupt tiny
    groups (skew-safe). Shared by the fwd (NT) and dgrad (NN) kernels, persistent and
    non-persistent. group_m/group_n are per-shape autotuned (the small-K L2 lever).

    ``n_blocks_c`` = compile-time value of ``n_blocks`` (0 = unknown). Given it, every divisor
    on this path is a constant, so the tile-id decode that gates a tile's first g2s costs magic
    multiplies instead of v_rcp_iflag sequences (each of which also round-trips VALU->SGPR).

    ``unsigned``: caller-proven "local / m_end-m_start are non-negative", which drops the
    floor-div sign fixup on every divide here (see ``udiv``). Opt-in so the other callers of
    this helper keep byte-identical ISA."""
    _dv = udiv if unsigned else (lambda a, b: a // b)
    _md = umod if unsigned else (lambda a, b: a % b)
    _cdiv = ceildiv_pow2 if unsigned else ceildiv
    lm_r = _dv(local, n_blocks)
    bn_r = _md(local, n_blocks)
    if const_expr(group_n > 0 and group_m > 0):
        bpr_g = _cdiv(m_end - m_start, block_m_size)
        bm_b, bn_b = _band_block_mn(local, bpr_g, n_blocks, group_m, group_n)
        use_band = (bpr_g > fx.Int32(group_m)) & (fx.Int32(n_blocks) > fx.Int32(group_n))
        return arith.select(use_band, bm_b, lm_r), arith.select(use_band, bn_b, bn_r)
    elif const_expr(group_m > 0):
        GM_c = fx.Int32(group_m)
        bpr_g = _cdiv(m_end - m_start, block_m_size)
        npg = GM_c * n_blocks
        grp = _dv(local, npg)
        first_m = grp * GM_c
        rem_m = bpr_g - first_m
        in_grp = _md(local, npg)
        _part = rem_m < GM_c  # trailing band: fewer than group_m M-blocks left in this group
        if const_expr(n_blocks_c > 0):
            # Walk the trailing band row-major (divisor n_blocks) instead of band-major (divisor
            # rem_m, a runtime value): same bijection over its rem_m*n_blocks tiles, but both
            # divisors stay compile-time constants.
            NB_c = fx.Int32(n_blocks_c)
            lm_g = first_m + arith.select(_part, _dv(in_grp, NB_c), _md(in_grp, GM_c))
            bn_g = arith.select(_part, _md(in_grp, NB_c), _dv(in_grp, GM_c))
        else:
            gsize_m = arith.select(_part, rem_m, GM_c)
            lm_g = first_m + _md(in_grp, gsize_m)
            bn_g = _dv(in_grp, gsize_m)
        use_gm = bpr_g > GM_c
        return arith.select(use_gm, lm_g, lm_r), arith.select(use_gm, bn_g, bn_r)
    return lm_r, bn_r


_WGRAD_XCD_HW = 8  # gfx950 dispatcher: workgroup bid runs on XCD bid % _WGRAD_XCD_HW
_WGRAD_XCD_RCP_SHIFT = 16  # fixed-point reciprocal of the compile-time swizzle divisors


def _wgrad_xcd_aff_geom(n_blocks_m, n_blocks_n, tiles_per_group, nxcd=_WGRAD_XCD_HW):
    """(h, w) for the XCD-affine wgrad swizzle, or None when the grid is too small. Reorders each
    XCD's residue class into a contiguous width-w column band (h>1 reuses A-slabs); the rectangle's
    two sides are CONCURRENT operand streams, so its run is set by the LARGER side, not their sum."""
    sz = tiles_per_group // nxcd
    if sz < 2 or n_blocks_m < 2 or n_blocks_n < 2:
        return None
    best = None
    for w in (d for d in range(1, n_blocks_n + 1) if n_blocks_n % d == 0):
        rows = min(n_blocks_m, -(-sz // w))
        cols = w * -(-sz // (n_blocks_m * w))
        key = (max(rows, cols), rows + cols, w)
        if best is None or key < best[0]:
            best = (key, w, rows)
    _, w, rows = best
    return (2 if w > 1 and rows % 2 == 0 and n_blocks_m % 2 == 0 else 1), w


def _wgrad_band_is_xcd_aff(n_blocks_m, n_blocks_n, group_m, group_n, nxcd=_WGRAD_XCD_HW):
    """True when the (group_m, group_n) band already hands every XCD the minimal operand
    footprint (super-block of nxcd cells tiling the grid). Equivalent to the compact rectangle
    only while the launch is shallow, so the caller trusts this gate only then."""
    return group_m * group_n == nxcd and n_blocks_m % group_m == 0 and n_blocks_n % group_n == 0


def _wgrad_xcd_div(x, d, xmax):
    """``x // d`` for a compile-time d, as one multiply plus one shift by a fixed-point reciprocal
    exact over [0, xmax]. Same trick as _wgrad_split_div, same reason: a real divide in
    the per-tile prologue is a latency-exposed serial chain at occ=1."""
    m = -(-(1 << _WGRAD_XCD_RCP_SHIFT) // d)
    assert all((v * m) >> _WGRAD_XCD_RCP_SHIFT == v // d for v in range(xmax + 1)), (d, xmax)
    return fx.Int32(fx.Int32(x * fx.Int32(m)) >> _WGRAD_XCD_RCP_SHIFT)


def _wgrad_xcd_rot_ok(TILES_PER_GROUP, gp=1, nxcd=_WGRAD_XCD_HW):
    """True when the per-super-block class rotation of _wgrad_xcd_tile is a bijection. It permutes
    the residue classes, so every class must own the same number of tiles, i.e. nxcd | gp*TPG."""
    return (gp * TILES_PER_GROUP) % nxcd == 0


def _wgrad_xcd_tile(idx, N_BLOCKS_M, N_BLOCKS_N, h, w, gp, TILES_PER_GROUP, rot=False, nxcd=_WGRAD_XCD_HW):
    """dispatch id -> (group_idx, block_m, block_n) under the XCD-affine swizzle; (h, w) from
    _wgrad_xcd_aff_geom, gp = groups a super-block spans. De-interleaves the id by nxcd so a class
    is a contiguous run, then walks (h x w) blocks down a width-w band. rot spreads boundary load."""
    k = nxcd // gp
    SB = gp * TILES_PER_GROUP
    per, rem = divmod(SB, nxcd)
    cbs, bsz = N_BLOCKS_M * w, h * w
    sb = idx // SB
    r = idx - sb * fx.Int32(SB)
    c = r & fx.Int32(nxcd - 1)
    if const_expr(rot):
        assert _wgrad_xcd_rot_ok(TILES_PER_GROUP, gp, nxcd)
        c = (c + sb) & fx.Int32(nxcd - 1)
    # low log2(k) bits of the class pick the run inside a group, the high bits pick the group.
    j = r >> (nxcd.bit_length() - 1)
    lin = (c & fx.Int32(k - 1)) * fx.Int32(per) + j
    if const_expr(rem):
        lin = lin + arith.select(c < fx.Int32(rem), c, fx.Int32(rem))
    group_idx = sb * fx.Int32(gp) + (c >> (k.bit_length() - 1))
    cb = _wgrad_xcd_div(lin, cbs, per * k + rem)
    p = lin - cb * fx.Int32(cbs)
    blk = _wgrad_xcd_div(p, bsz, cbs - 1)
    q = p - blk * fx.Int32(bsz)
    if const_expr(h == 1):
        return group_idx, blk, cb * fx.Int32(w) + q
    col = _wgrad_xcd_div(q, h, bsz - 1)
    return group_idx, blk * fx.Int32(h) + (q - col * fx.Int32(h)), cb * fx.Int32(w) + col


def _wgrad_block_mn(
    idx,
    G,
    TILES_PER_GROUP,
    N_BLOCKS_M,
    N_BLOCKS_N,
    group_m,
    group_n,
    interleave,
    tile_rot=0,
    xcd_aff=None,
    xcd_rot=False,
):
    """idx -> (group_idx, block_m, block_n) for the wgrad output grid. interleave=True is the
    band-cyclic group interleave (skew load-balance, keeps B-stripe L2 reuse); interleave=False
    picks XCD-affine / group_n band / group_m cluster / row-major."""
    if const_expr(interleave and group_m > 0 and N_BLOCKS_M > group_m and N_BLOCKS_M % group_m == 0):
        BAND = const_expr(group_m * N_BLOCKS_N)
        bg = idx // BAND
        in_band = idx % BAND
        return bg % G, (bg // G) * group_m + (in_band % group_m), in_band // group_m
    if const_expr(interleave):
        cl = idx // N_BLOCKS_N
        return cl % G, cl // G, idx % N_BLOCKS_N
    if const_expr(xcd_aff is not None):
        return _wgrad_xcd_tile(idx, N_BLOCKS_M, N_BLOCKS_N, *xcd_aff, TILES_PER_GROUP, xcd_rot)
    group_idx = idx // TILES_PER_GROUP
    local = idx % TILES_PER_GROUP
    if const_expr(tile_rot):
        # gcd(TILES_PER_GROUP, num_xcd) pins boundary tiles to fixed XCD residues; the per-group tile_rot rotation spreads them (a per-group bijection).
        local = (local + group_idx * fx.Int32(tile_rot)) % fx.Int32(TILES_PER_GROUP)
    if const_expr(group_n > 0 and group_m > 0 and N_BLOCKS_N > group_n):
        block_m, block_n = _band_block_mn(local, N_BLOCKS_M, N_BLOCKS_N, group_m, group_n)
    elif const_expr(group_m > 0 and N_BLOCKS_M > group_m):
        GM_c = fx.Int32(group_m)
        npg = group_m * N_BLOCKS_N
        first_m = (local // npg) * GM_c
        rem_m = fx.Int32(N_BLOCKS_M) - first_m
        gsize_m = arith.select(rem_m < GM_c, rem_m, GM_c)
        in_grp = local % npg
        block_m = first_m + (in_grp % gsize_m)
        block_n = in_grp // gsize_m
    else:
        block_m = local // N_BLOCKS_N
        block_n = local % N_BLOCKS_N
    return group_idx, block_m, block_n


def _wgrad_rebase(A, B, m_start, m_end, OUT_M, OUT_N, F8_IR_t):
    """Fold m_start*OUT into the i64 SRD base + per-group num_records (cumulative m_end*OUT
    overflows int32 for large-G MoE). Returns (a_div, b_div, a_rebase, b_rebase) for G2SLoader's
    i64-traverse mode; needed because the per-group span mg*OUT can pass the 32-bit soffset."""
    a_base = arith.index_cast(T.index, m_start) * arith.index(OUT_M)
    b_base = arith.index_cast(T.index, m_start) * arith.index(OUT_N)
    mg = arith.index_cast(T.index, m_end) - arith.index_cast(T.index, m_start)
    a_nrec = arith.maxsi(mg * arith.index(OUT_M), arith.index(0))
    b_nrec = arith.maxsi(mg * arith.index(OUT_N), arith.index(0))
    gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
    gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
    return a_div, b_div, (A, F8_IR_t, a_base, a_nrec), (B, F8_IR_t, b_base, b_nrec)


def _compile_grouped_tn_wgrad_masked(
    *,
    OUT_M: int,
    OUT_N: int,
    G: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    waves_per_eu: int = 2,
    nt_vmcnt: int = 3,
    num_xcd: int = 8,
    acc_mode: str = "agpr",  # "vgpr"=VGPR in-place (mode 3); "agpr"=AGPR in-place (mode 2)
    s2r_inline: bool = True,  # True = inline-asm packed ds_read_tr8 + manual lgkmcnt (dense TN path; needs agpr_alloc>0)
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
    group_m: int = 0,
    store_cshuffle: bool = True,
    chunk: int = 8,  # capacity-free chunked K-loop: outer runtime scf.for over
    # ceildiv(k_iters,chunk) x range_constexpr(chunk) of the 4-buffer body; over-run is SRD-clamped.
    i64_traverse: bool = False,  # A[m,OUT_M] & B[m,OUT_N] traversal via per-load i64 SRD re-base (lifts mg*OUT < 2^32 cap)
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
):
    """Masked grouped TN wgrad: a capacity-free chunked K-loop over the 4-buffer inline body,
    with the per-group contraction masked by the SRD num_records clamp (over-read -> 0). acc_mode
    picks the MFMA accumulator class: "vgpr"=mode 3 (in-place vacc); "agpr"=mode 2 (off-VGPR)."""
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert G >= 1
    assert acc_mode in ("vgpr", "agpr")
    _agpr = acc_mode == "agpr"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    _LDS_CS = 1056
    a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
    b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

    N_BLOCKS_M = (OUT_M + BLOCK_M - 1) // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N

    _cshuf_ty = fx.Float16 if out_fp16 else fx.BFloat16
    _cshuf_n = 8 * 16 * (N_TILES_B * 16)

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        C_lds_shuffle: fx.Array[_cshuf_ty, _cshuf_n, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_tn_masked(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16

        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))

        pid = xcd_remap_pid(fx.block_idx.x, G * TILES_PER_GROUP, num_xcd)
        group_idx, block_m, block_n = _wgrad_block_mn(
            pid, G, TILES_PER_GROUP, N_BLOCKS_M, N_BLOCKS_N, group_m, 0, _WG_INTERLEAVE
        )
        m_start = _load_go(go_div, group_idx)
        m_end = _load_go(go_div, group_idx + 1)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4

        a_div, b_div, a_rb, b_rb = _wgrad_rebase(A, B, m_start, m_end, OUT_M, OUT_N, F8_IR_t)

        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, OUT_M, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, OUT_N, N_LDS_ROUNDS)

        mfma = _build_mfma(N_TILES_A, N_TILES_B, cbsz, blgp, asm_mode="2" if _agpr else "3")

        # A and B both stride the contraction (token) dim: re-base both SRDs per load in i64 mode.
        a_rebase = a_rb if i64_traverse else None
        b_rebase = b_rb if i64_traverse else None
        a_g2s = G2SLoader(
            a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS, rebase=a_rebase
        )
        b_g2s = G2SLoader(
            b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS, rebase=b_rebase
        )
        a_s2r = S2RLoaderTr(
            wave_m,
            N_TILES_A,
            LDS_BLOCK_M // 2,
            inline_asm=s2r_inline,
            vmcnt_hint=nt_vmcnt,
            chunk_stride=_LDS_CS,
        )
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=s2r_inline, vmcnt_hint=nt_vmcnt, chunk_stride=_LDS_CS
        )
        if const_expr(store_cshuffle):
            store_c = StoreCPerTensorCShuffle(
                A_scale,
                B_scale,
                C,
                (group_idx + 1) * OUT_M,
                OUT_N,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                _out_ty,
                lds.C_lds_shuffle,
                wave_id,
                beta_is_one=beta_is_one,
            )
        else:
            store_c = StoreCPerTensor(
                A_scale,
                B_scale,
                C,
                (group_idx + 1) * OUT_M,
                OUT_N,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                _out_ty,
                beta_is_one=beta_is_one,
            )

        # i64 index so A0_off + (k+2)*AM does not truncate when the per-group token span mg*OUT exceeds 2^31.
        A0_off = arith.index_cast(T.index, block_m * BLOCK_M)  # relative to the m_start-folded SRD base
        A1_off = A0_off + LDS_BLOCK_M
        B0_off = arith.index_cast(T.index, block_n * BLOCK_N)
        B1_off = B0_off + LDS_BLOCK_N
        AM = arith.index(BLOCK_K * OUT_M)
        BNs = arith.index(BLOCK_K * OUT_N)

        b_g2s.load(b_cur0, B0_off + 0 * BNs)
        a_g2s.load(a_cur0, A0_off + 0 * AM)
        b_g2s.load(b_cur1, B1_off + 0 * BNs)
        a_g2s.load(a_cur1, A1_off + 0 * AM)
        if wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
        b_g2s.load(b_next0, B0_off + 1 * BNs)
        a_g2s.load(a_next0, A0_off + 1 * AM)
        b_g2s.load(b_next1, B1_off + 1 * BNs)
        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Capacity-free chunked path: runtime k_iters, even-chunk unroll of the 4-buffer body; over-run SRD-clamped to 0.
        acc00 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_ACCUMS)]
        acc01 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_ACCUMS)]
        acc10 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_ACCUMS)]
        acc11 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_ACCUMS)]
        for _q in (acc00, acc01, acc10, acc11):
            for _r in _q:
                fx.memref_store_vec(mfma.zero_value, _r)
        _kit = (m_end - m_start + (BLOCK_K - 1)) // BLOCK_K
        _nchunks = (_kit + (chunk - 1)) // chunk
        for _c in range(_nchunks):
            for _j in range_constexpr(chunk):
                _wgrad_body_4buf(
                    _c * chunk + _j,
                    a_g2s,
                    b_g2s,
                    a_s2r,
                    b_s2r,
                    mfma,
                    a_cur0,
                    a_cur1,
                    b_cur0,
                    b_cur1,
                    a_next0,
                    a_next1,
                    b_next0,
                    b_next1,
                    acc00,
                    acc01,
                    acc10,
                    acc11,
                    A0_off,
                    A1_off,
                    B0_off,
                    B1_off,
                    AM,
                    BNs,
                    N_LDS_STEPS_A,
                    N_LDS_STEPS_B,
                )
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
        c00_frag = [Vec(fx.memref_load_vec(_r)) for _r in acc00]
        c01_frag = [Vec(fx.memref_load_vec(_r)) for _r in acc01]
        c10_frag = [Vec(fx.memref_load_vec(_r)) for _r in acc10]
        c11_frag = [Vec(fx.memref_load_vec(_r)) for _r in acc11]

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = group_idx * OUT_M + block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        _store_quadrants(
            store_c, c00_frag, c01_frag, c10_frag, c11_frag, base_row, base_col, LDS_BLOCK_M, LDS_BLOCK_N
        )

    @flyc.jit
    def launch_grouped_tn_masked(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,
        stream: fx.Stream,
    ):
        grid_x = G * TILES_PER_GROUP
        # AGPR alloc needed for mode-2 acc AND for the inline-asm S2R packed reads.
        attrs = make_value_attrs(waves_per_eu, 128 if (_agpr or s2r_inline) else 0, "512,512")
        kernel_grouped_tn_masked(
            A,
            B,
            C,
            A_scale,
            B_scale,
            group_offs,
            value_attrs=attrs,
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_tn_masked


# Caches the compiled kernel per config key (not results) for reuse without re-tracing.
_GROUPED_LAUNCH_CACHE: dict = {}

_GROUPED_AGPR = True  # AGPR in-place accumulation (off-VGPR, spill-free)


# Per-shape online autotune: time a small candidate set on a balanced distribution, cache the winner keyed on static dims only.
_GROUPED_AT_CACHE: dict = {}


def _grouped_compile_cfg(
    trans_b,
    K,
    G,
    bm,
    xcd,
    grp_agpr,
    out_fp16,
    cbsz,
    blgp,
    nt_group_m,
    acc_mode,
    store_cshuffle=False,
    sched_schedbar=False,
    bn=256,
    nt_group_n=0,
    cap_cu=-1,
    i64_traverse=False,
    beta_is_one=False,
    n_stride=0,
    N=0,
):
    assert n_stride == 0 or N > 0, f"n_stride={n_stride} requires real N>0"
    ckey = (
        "nt" if trans_b else "nn",
        K,
        G,
        bm,
        xcd,
        grp_agpr,
        out_fp16,
        cbsz,
        blgp,
        nt_group_m,
        acc_mode,
        store_cshuffle,
        sched_schedbar,
        bn,
        nt_group_n,
        cap_cu,
        i64_traverse,
        beta_is_one,  # baked into the epilogue (read-back), so it keys the artifact
        n_stride,  # padded N pitch: bakes a different B addressing, so it keys the artifact
        N,  # real N compile const (paired with n_stride)
    )
    l = _GROUPED_LAUNCH_CACHE.get(ckey)
    if l is None:
        if trans_b:
            l = _compile_grouped_nt(
                K=K,
                G=G,
                BLOCK_M=bm,
                BLOCK_N=bn,
                nt_vmcnt=3,
                num_xcd=xcd,
                agpr_inplace=grp_agpr,
                out_fp16=out_fp16,
                cbsz=cbsz,
                blgp=blgp,
                group_m=nt_group_m,
                group_n=nt_group_n,
                store_cshuffle=store_cshuffle,
                sched_schedbar=sched_schedbar,
                persistent=True,
                cap_cu=cap_cu,
                beta_is_one=beta_is_one,
                n_stride=n_stride,
                N=N,
            )
        else:
            l = _compile_grouped_nn(
                K=K,
                G=G,
                BLOCK_M=bm,
                BLOCK_N=bn,
                nt_vmcnt=3,
                num_xcd=xcd,
                agpr_inplace=grp_agpr,
                out_fp16=out_fp16,
                cbsz=cbsz,
                blgp=blgp,
                group_m=nt_group_m,
                group_n=nt_group_n,
                store_cshuffle=store_cshuffle,
                sched_schedbar=sched_schedbar,
                persistent=True,
                cap_cu=cap_cu,
                i64_traverse=i64_traverse,
                beta_is_one=beta_is_one,
                n_stride=n_stride,
                N=N,
            )
        _GROUPED_LAUNCH_CACHE[ckey] = l
    return l


def _balanced_group_offs(m_total, G, device):
    """Synthetic balanced group_offs [G+1] int64 (int32-view, matching the dispatch's
    free reinterpret): M_total split into G near-equal groups. The autotune times on
    this canonical distribution so the chosen config depends ONLY on the static shape
    (op, N, K, G, M_total), never on the (possibly skewed) token distribution the first
    real call carries — we cannot tell balanced from skewed at dispatch, so every input
    is timed as balanced."""
    base = m_total // G
    sizes = torch.full((G,), base, dtype=torch.int64, device=device)
    rem = m_total - base * G
    if rem:
        sizes[:rem] += 1
    offs = torch.zeros(G + 1, dtype=torch.int64, device=device)
    offs[1:] = sizes.cumsum(0)
    return offs.view(torch.int32)


_NP_8WAVE_CANDS = ((256, 8, 4, 0), (256, 8, 8, 0))
# (num_xcd=4, group_m=8) is the measured static optimum across the deployed MoE shape family.
# The online race cannot find it: it scores on synthetic *balanced* tensors, where this arm
# ties xcd8_gm4, but the real skewed deployed distributions separate them.
_NP_STATIC_CAND = (256, 4, 8, 0)
_NP_DRAM_WIDE_CAND = (256, 4, 4, 0)
_NP_DRAM_NARROW_CAND = (256, 8, 4, 0)
_NP_B_LLC = 256 << 20
# xcd1 (group-major, B[g] L2-resident) only pays off for large B[g]; K>=4096 is the observed crossover.
_NP_LARGE_K = 4096
_NP_PM_CANON = (1024, 8192)


def _np_regime(trans_b, N, K, G, M_total):
    """Coarse M-derived regime bucket for the autotune key (a rule, not a per-M retune):
    1 = underfilled NN dgrad grid (wants small-M bm128), 0 = steady NK-autotuned."""
    if not trans_b:
        pm = M_total // G
        if G * ((pm + 127) // 128) * ((N + 255) // 256) <= _num_cus():
            return 1  # small-M dgrad -> bm128
    return 0  # steady -> NK autotune


def _autotune_np_dispatch(
    trans_b, N, K, G, out_fp16, cbsz, blgp, args, regime, beta_is_one=False, n_stride=0
):
    """Race the NT/NN candidates on balanced synthetic tensors, cached per static
    (op,N,K,G,dtype,regime), never per M_total. regime==1 -> fixed bm128. Candidates carry the
    caller's ``beta_is_one``; the probe owns its buffer and zeroes it per launch (C = 0 + acc)."""
    # NN B[K,N] traversal: re-base B SRD per load in i64 when the per-group span K*N reaches 2^32 fp8.
    i64_tr = (not trans_b) and (K * max(N, n_stride) >= 2**32)

    def mk(bm, xcd, gm, gn):
        if trans_b:  # NT: merged factory, non-persistent mode (intrinsic MMA, scalar store)
            return _compile_grouped_nt(
                K=K,
                G=G,
                BLOCK_M=bm,
                BLOCK_N=256,
                out_fp16=out_fp16,
                cbsz=cbsz,
                blgp=blgp,
                num_xcd=xcd,
                group_m=gm,
                group_n=gn,
                persistent=False,
                agpr_inplace=False,
                store_cshuffle=False,
                sched_schedbar=False,
                beta_is_one=beta_is_one,
                # Keep the loop-tail vmcnt on NT: measured net-negative to drop it here (unlike
                # the NN twin, where this drain is redundant) -- see pitfalls/05 "NN 上能删,NT 上不能".
                nt_vmcnt=3,
                N=N,
                n_stride=n_stride,
            )
        # NN: merged factory, non-persistent mode (AGPR in-place, scalar store).
        return _compile_grouped_nn(
            K=K,
            G=G,
            BLOCK_M=bm,
            BLOCK_N=256,
            out_fp16=out_fp16,
            cbsz=cbsz,
            blgp=blgp,
            num_xcd=xcd,
            group_m=gm,
            group_n=gn,
            persistent=False,
            agpr_inplace=True,
            store_cshuffle=False,
            sched_schedbar=False,
            nt_vmcnt=-1,
            i64_traverse=i64_tr,
            N=N,
            beta_is_one=beta_is_one,
            n_stride=n_stride,
        )

    if not trans_b and regime == 1:
        # small-M dgrad: BLOCK_M=128 doubles the M-tiles and wins here, so a single config, no autotune.
        return mk(128, 1, 0, 0)
    a_live, b_i8, out_live = args[0], args[1], args[2]
    mps = []
    # Production-magnitude fp8 probe A: fp8 GEMM wall time is data-magnitude/DVFS sensitive, so an all-zero probe would misrank candidates.
    _g = torch.Generator(device=a_live.device).manual_seed(0)
    for pm in _NP_PM_CANON:
        M_c = G * pm
        a_c = torch.empty((M_c, a_live.shape[1]), device=a_live.device, dtype=a_live.dtype)
        a_c.view(torch.uint8).random_(0, 64, generator=_g)
        out_c = torch.empty((M_c, N), device=out_live.device, dtype=out_live.dtype)
        offs_c = _balanced_group_offs(M_c, G, a_live.device)
        mps.append(
            [
                (a_c.view(torch.int8), b_i8, out_c, args[3], args[4], offs_c, M_c, N, args[8]),
                out_c,
                None,
                None,
            ]
        )

    # <=4 candidates/op, all measured-competitive: arms that lose on the deployed shapes lose on
    # every distribution, so they only spend a candidate slot and hand the race a chance to pick
    # a loser. cand[0] is the static optimum; NT deep-K keeps its own arms (unmeasured family,
    # xcd1 is its group-major lever).
    if trans_b:
        cands = list(_NP_8WAVE_CANDS)
        if N <= K and K >= _NP_LARGE_K:
            cands.append((256, 1, 4, 0))
        elif G * N * K > _NP_B_LLC:
            lead = _NP_DRAM_WIDE_CAND if N > K else _NP_DRAM_NARROW_CAND
            cands = [lead] + [c for c in cands if c != lead] + [_NP_STATIC_CAND]
        else:
            cands.insert(0, _NP_STATIC_CAND)
    else:
        # dgrad NN: N-bands do not help (deep-K transpose-load core), so diversity is in (num_xcd, group_m).
        cands = [_NP_STATIC_CAND, (256, 8, 4, 0)]

    def _score(launch):
        """Geomean of the launch time at every canonical M, or None if it drifts/NaNs at
        any M (numeric guard). Timing each candidate at both ends picks an M-robust config."""
        prod = 1.0
        for targs, out_view, ref, refnorm in mps:
            if beta_is_one:
                out_view.zero_()  # C = 0 + acc, so the overwrite reference still applies
            launch(*targs)
            torch.cuda.synchronize()
            if ref is not None:
                o = out_view.detach().float()
                e = float(((o - ref) * (o - ref)).sum().item())
                if (e / refnorm) >= (2e-2**2) or not torch.isfinite(o.view(-1)[:1024]).all().item():
                    return None
            prod *= _robust_time(launch, targs)
        return prod ** (1.0 / len(mps))

    base = mk(*cands[0])
    for mp in mps:  # establish the per-M numeric reference from the base config
        if beta_is_one:
            mp[1].zero_()
        base(*mp[0])
        torch.cuda.synchronize()
        r = mp[1].detach().clone().float()
        mp[2], mp[3] = r, (float((r * r).sum().item()) or 1.0)
    best, bs = base, _score(base)
    for cand in cands[1:]:
        l = mk(*cand)
        s = _score(l)  # numeric guard folded in: None -> skip
        if s is not None and s < bs * 0.985:  # adopt only past the noise margin (geomean)
            best, bs = l, s

    return best


def grouped_gemm_fp8_tensorwise_flydsl_kernel(
    a: "torch.Tensor",
    b: "torch.Tensor",
    a_scale: "torch.Tensor",
    b_scale: "torch.Tensor",
    group_offs: "torch.Tensor",
    trans_b: bool = False,
    out_dtype=torch.bfloat16,
    num_cu: "int | None" = -1,
    beta: float = 0.0,
    out: "torch.Tensor | None" = None,
    n_real: "int | None" = None,
) -> "torch.Tensor":
    """FlyDSL per-tensor grouped fp8 GEMM (M-grouped), matching the Triton entry.
    out[g] = a[g] @ B_view[g] * a_scale * b_scale. trans_b=True (forward) uses b [G, N, K] and
    the NT kernel; trans_b=False (dgrad) uses b [G, K, N] and the NN kernel. group_offs [G+1]."""
    assert a.ndim == 2 and b.ndim == 3
    M_total, K = a.shape
    G = b.shape[0]
    N = b.shape[1] if trans_b else b.shape[2]
    K_b = b.shape[2] if trans_b else b.shape[1]
    assert K == K_b, f"K mismatch a={K} b={K_b}"

    # Pad-both: n_real is the real width -> n_stride carries the padded pitch, N collapses to real.
    n_stride = 0
    if n_real is not None and n_real != N:
        assert 0 < n_real < N, f"n_real={n_real} must be in (0, N={N})"
        n_stride, N = N, n_real

    out = resolve_accum_out(out, beta, (M_total, N), a.device, out_dtype)
    beta_is_one = beta == 1.0
    # kernel reads group_offs as int64 low-words via a free int32-view; int32 callers are upcast once.
    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0

    grp_agpr = _GROUPED_AGPR
    nt_group_m = _GROUPED_NT_GROUPM  # 0 = row-major; the autotune sweeps group_m per shape
    op = "nt" if trans_b else "nn"
    # num_cu<=0: whole device via non-persistent nt8w/nn8w (one tile/WG). num_cu>0: reserve CUs -> persistent fixed grid. M_total is in the key.
    capped = num_cu is not None and num_cu > 0
    nonpersist = not capped
    # NK autotune key excludes M_total; a coarse M-derived regime bucket covers every M_total (see _np_regime).
    regime = _np_regime(trans_b, N, K, G, M_total) if nonpersist else 0
    at_key = (
        op,
        N,
        K,
        G,
        out_fp16,
        cbsz,
        blgp,
        regime,
        nonpersist,
        num_cu if capped else 0,
        beta_is_one,
        n_stride,
    )
    # Full rank (not flattened): a flat reshape(-1) overflows the int32 shape pack when M_total*K > 2^31; the kernel re-bases via i64.
    a_i8 = a.view(torch.int8)
    b_i8 = b.view(torch.int8)
    args = (
        a_i8,
        b_i8,
        out,
        a_scale.float().reshape(1),
        b_scale.float().reshape(1),
        go32,
        M_total,
        N,
        torch.cuda.current_stream(),
    )
    entry = _GROUPED_AT_CACHE.get(at_key)
    if entry is None:
        if nonpersist:
            # num_cu<=0 (full device): autotune the non-persistent nt8w/nn8w swizzle (straight-line one-tile/WG body, no scf.for penalty).
            launch = _autotune_np_dispatch(
                trans_b,
                N,
                K,
                G,
                out_fp16,
                cbsz,
                blgp,
                args,
                regime,
                beta_is_one=beta_is_one,
                n_stride=n_stride,
            )
        else:
            # Single persistent prod config (no autotune); reached only when num_cu>0 reserves CUs. Default goes to nt8w/nn8w.
            launch = _grouped_compile_cfg(
                trans_b,
                K,
                G,
                256,
                8,
                grp_agpr,
                out_fp16,
                cbsz,
                blgp,
                nt_group_m,
                "agpr",
                store_cshuffle=True,
                sched_schedbar=True,
                cap_cu=(num_cu if capped else -1),
                # NN B[K,N] per-group traversal: i64 re-base when K*N (padded pitch) reaches 2^32 fp8.
                i64_traverse=((not trans_b) and (K * max(N, n_stride) >= 2**32)),
                beta_is_one=beta_is_one,
                n_stride=n_stride,
                N=(N if n_stride else 0),
            )
        entry = [launch, None]  # [raw @flyc.jit closure, flyc.compile'd object (lazy)]
        _GROUPED_AT_CACHE[at_key] = entry
    raw, compiled = entry
    # Mode-split: CUDA-graph capture uses the raw @flyc.jit closure (compile-d regresses under capture); eager uses a flyc.compile-d object (skips per-call drift-check).
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        if compiled is None:
            with CompilationContext.compile_hints(_GG_SCHED_HINTS):
                compiled = compile_with_scratch_out(raw, args, out_index=2)
            entry[1] = compiled
        compiled(*args)
    return out


_GROUPED_WGRAD_LAUNCH_CACHE: dict = {}
_GROUPED_WGRAD_AT_CACHE: dict = {}
# Winning wgrad config keyed beta-INDEPENDENTLY: raced at beta=0, so beta=0/beta=1 reuse it (a tight beta=1 build would fault in the race).
_GROUPED_WGRAD_CFG_CACHE: dict = {}


# 3-buffer whole-loop: one pool at depth 3, rest depth 2; n_phases = lcm(nbuf), statically unrolled.
_WL_ASM_CACHE_3BUF = {}
# Phase drain leaves the last _WL_ELGK ds_reads in flight across the barrier; safe only because the next rewrite is a full global round trip away (short tail safe, long tail races).
_WL_ELGK = 15


def _wholeloop_asm_3buf(
    *,
    nta,
    ntb,
    bases,  # [4][2*tiles] -- ONE per-lane address set per pool (its buffer 0)
    buf_off,  # [4][nbuf_p] compile-time LDS byte delta of buffer b from the pool's buffer 0
    gbases,  # [4][nbuf_p]
    gl_a,
    gl_b,
    rsrc_a,
    rsrc_b,
    soff0,  # [4] initial gmem soffset for the first in-loop write (targets K-block nbuf_p[p])
    kstep,
    kstep_b,
    nval,  # runtime trip count, must be a multiple of n_phases = lcm(nbuf_p)
    acc,
    nsa,
    nsb,
    rs,
    cs,
    nw,
    cbsz=0,
    blgp=0,
    tail_nval=None,  # SGPR i32: 0..5 extra single-block passes fused into this asm block.
    a_plain=False,  # see _wholeloop_asm_3buf's a_plain
    a_halves=2,  # 2 = full 256-row tile (a0,a1); 1 = a0-only (skip fully-masked a-half1 + c10/c11)
    b_halves=2,  # 2 = full 256-col tile (b0,b1); 1 = b0-only (skip fully-masked b-half1 + c01/c11)
    nval_can_be_zero=False,  # variable-K: guard the do-while main loop with an nval==0 entry
    # branch so groups shorter than n_phases run only the fused tail (no wasted K-blocks).
):
    from functools import reduce
    from math import gcd

    nq = nta * ntb
    # a0-only / b0-only drop the dead half pool entirely; folding the buffer select into the ds_read immediate frees the address VGPRs.
    n_pools = a_halves + b_halves
    # QUADS[qi] = (a-pool, b-pool); a0-only drops the (1,*) quads, b0-only the (*,1) ones.
    QUADS = tuple((ah, a_halves + bh) for ah in range(a_halves) for bh in range(b_halves))
    NT = len(QUADS) * nq
    PIN = 8
    tiles = tuple([nta] * a_halves + [ntb] * b_halves)
    ntmp = sum(tiles)
    nbase = [2 * tiles[p] for p in range(n_pools)]
    buf_off = tuple(tuple(o) for o in buf_off)
    nbuf_p = tuple(len(buf_off[p]) for p in range(n_pools))
    n_phases = reduce(lambda a, b: a * b // gcd(a, b), nbuf_p, 1)
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    _vmcnt_mode = "partial"
    _has_tail = tail_nval is not None
    _cs_t = tuple(cs) if isinstance(cs, (list, tuple)) else (cs,) * n_pools
    _rs_t = tuple(rs) if isinstance(rs, (list, tuple)) else (rs,) * n_pools
    # The buffer delta rides the ds_read 16-bit immediate, so one address set feeds all of a pool buffers.
    assert max(o for pol in buf_off for o in pol) + max(_rs_t) < 65536, "buffer delta overflows ds offset"
    key = (
        "3buf",
        nta,
        ntb,
        nsa,
        nsb,
        nbuf_p,
        buf_off,
        mods,
        _rs_t,
        _cs_t,
        nw,
        _vmcnt_mode,
        _has_tail,
        a_plain,
        a_halves,
        b_halves,
        nval_can_be_zero,
        _WL_ELGK,
    )
    if key not in _WL_ASM_CACHE_3BUF:
        o_acc = list(range(NT))
        t_pool = [NT]
        for p in range(n_pools):
            t_pool.append(t_pool[-1] + tiles[p])
        o_cnt = NT + ntmp
        o_wsoff = [o_cnt + 1 + p for p in range(n_pools)]  # per-pool running gmem write soffset
        # Tail = 5 statically-unrolled gated phases. Never add an unwritten "=&s" output here: uninitialized-output hazard for regalloc.

        # Ordered input-operand schema (name, width) above the output base.
        i = o_cnt + 1 + n_pools
        _in_schema = (
            [("base", nbase[p]) for p in range(n_pools)]
            + [("gbase", nbuf_p[p]) for p in range(n_pools)]
            + [("gl_a", nsa), ("gl_b", nsb)]
            + [("rsrc_a", 1), ("rsrc_b", 1), ("kstep", 1), ("kstep_b", 1), ("nval", 1)]
            + ([("tail_nval", 1)] if _has_tail else [])
            + [("soff0", n_pools)]
        )
        _blocks = []
        for _name, _w in _in_schema:
            _blocks.append(list(range(i, i + _w)))
            i += _w
        _it = iter(_blocks)
        i_base = [next(_it) for _p in range(n_pools)]
        i_gbase = [next(_it) for _p in range(n_pools)]
        i_gla = next(_it)
        i_glb = next(_it)
        i_rsa, i_rsb = next(_it)[0], next(_it)[0]
        i_kstep, i_kstepb, i_nval = next(_it)[0], next(_it)[0], next(_it)[0]
        if _has_tail:
            i_tailval = next(_it)[0]
        i_soff0 = next(_it)
        i_ks = [i_kstep] * a_halves + [i_kstepb] * b_halves

        def pool_of(tt):
            for p in range(n_pools):
                if t_pool[p] <= tt < t_pool[p + 1]:
                    return p, tt - t_pool[p]
            raise AssertionError

        def ds_line(buf_per_pool, tt):
            p, ti = pool_of(tt)
            bo = buf_off[p][buf_per_pool[p]]  # buffer select folded into the DS immediate
            rs = _rs_t[p]
            vb = PIN + (tt - NT) * 8
            p0, p1 = i_base[p][2 * ti], i_base[p][2 * ti + 1]
            if a_plain and p < a_halves:
                # plain (no-transpose) read -- see _wholeloop_asm_3buf a_plain.
                return (
                    f"ds_read_b128 v[{vb}:{vb + 3}], ${p0} offset:{bo}\n"
                    f"ds_read_b128 v[{vb + 4}:{vb + 7}], ${p1} offset:{bo}"
                )
            return (
                f"ds_read_b64_tr_b8 v[{vb}:{vb + 1}], ${p0} offset:{bo}\n"
                f"ds_read_b64_tr_b8 v[{vb + 2}:{vb + 3}], ${p1} offset:{bo}\n"
                f"ds_read_b64_tr_b8 v[{vb + 4}:{vb + 5}], ${p0} offset:{bo + rs}\n"
                f"ds_read_b64_tr_b8 v[{vb + 6}:{vb + 7}], ${p1} offset:{bo + rs}"
            )

        def emit_g2s(write_buf_per_pool):
            r = []
            if a_plain:
                # dgrad (NN) path keeps the original per-half (p-outer) A issue order: the A-pool
                # step-interleave below is a wgrad-only L2-line win and must not perturb the shared
                # NN dgrad trace (this helper is shared by _wave4_do_tile_tn and _wave4_do_tile_nn).
                order = [(p, st) for p in range(a_halves) for st in range(nsa)]
            else:
                # wgrad (TN): A pools step-interleaved -- the two halves of one K-row are adjacent in
                # the issue stream, so a row stride that is not a multiple of the 128B line lets the
                # second half hit the line the first half already pulled in.
                order = [(p, st) for st in range(nsa) for p in range(a_halves)]
            # B pools stay last -- the partial drain below counts on exactly their loads being the
            # outstanding ones. Within that block the wgrad path step-interleaves them for the same
            # adjacent-line reason as the A pools above.
            if a_plain:
                order += [(p, st) for p in range(a_halves, n_pools) for st in range(nsb)]
            else:
                order += [(p, st) for st in range(nsb) for p in range(a_halves, n_pools)]
            for p, st in order:
                rsrc = i_rsa if p < a_halves else i_rsb
                gl = i_gla if p < a_halves else i_glb
                buf = write_buf_per_pool[p]
                r.append(
                    f"s_add_u32 m0, ${i_gbase[p][buf]}, {st * nw * _cs_t[p]}\n"
                    f"buffer_load_dwordx4 ${gl[st]}, ${rsrc}, ${o_wsoff[p]} offen lds"
                )
            return r

        def _mfma_line(qi, ii, ji):
            ap, bp = QUADS[qi]
            q = qi * nq + ii * ntb + ji
            at = t_pool[ap] + ii
            bt = t_pool[bp] + ji
            return (f"v_mfma_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}{mods}", at, bt)

        def _diag_cells():
            # MFMA emission order: srcA pool is the outer loop so srcA stays on one fragment per bn-run (this MFMA is srcA-movement sensitive); bm x bn diagonal blocking spreads the ds_read refills.
            bm = 2
            ncol = b_halves * ntb
            # A short last N-block (see _wgrad_bnd_tiles) holds fewer columns; take the widest fit.
            bn = gcd(4, ncol)
            nib, ncb = nta // bm, ncol // bn
            cells = []
            for D in range(nib + ncb - 1):
                for iib in range(nib):
                    cb = D - iib
                    if 0 <= cb < ncb:
                        for di in range(bm):
                            for ah in range(a_halves):
                                for dj in range(bn):
                                    ii = iib * bm + di
                                    col = cb * bn + dj
                                    cells.append((ii, col // ntb, col % ntb, ah))
            return cells

        def emit_quadrant_mfmas():
            seq = []
            for ii, bh, ji, ah in _diag_cells():
                qi = ah * b_halves + bh
                seq.append(_mfma_line(qi, ii, ji))
            return seq

        def emit_phase(refill_bp, write_bp):
            g2sl = emit_g2s(write_bp)
            mlist = emit_quadrant_mfmas()
            last = {}
            for mi, (_ml, at, bt) in enumerate(mlist):
                last[at] = mi
                last[bt] = mi
            _gset = {}
            if g2sl:
                rfslot, rf = set(), set()
                for mi, (_ml, at, bt) in enumerate(mlist):
                    for rt in (at, bt):
                        if last[rt] == mi and rt not in rf:
                            rfslot.add(mi)
                            rf.add(rt)
                free = [mi for mi in range(len(mlist)) if mi not in rfslot]
                fgap = max(len(free) // max(len(g2sl), 1), 1)
                for k, fi in enumerate(free):
                    if (k % fgap == 0) and len(_gset) < len(g2sl):
                        _gset[fi] = len(_gset)
            out, gi, refilled = [], 0, set()
            for mi, (ml, at, bt) in enumerate(mlist):
                out.append(ml)
                for rt in (at, bt):
                    if last[rt] == mi and rt not in refilled:
                        out.append(ds_line(refill_bp, rt))
                        refilled.add(rt)
                if g2sl and mi in _gset and gi < len(g2sl):
                    out.append(g2sl[gi])
                    gi += 1
            while gi < len(g2sl):
                out.append(g2sl[gi])
                gi += 1
            for tt in range(NT, NT + ntmp):
                if tt not in refilled:
                    out.append(ds_line(refill_bp, tt))
            return out

        _3buf_pools = [p for p in range(n_pools) if nbuf_p[p] == 3]
        _3buf_pool = _3buf_pools[0] if _3buf_pools else None
        if _vmcnt_mode == "partial" and _3buf_pool is not None:
            # Partial drain keeps all 3-buffered pools writes in flight; emit order (0,1,2,3) leaves exactly those outstanding.
            _n_outstanding = sum((nsa if p < a_halves else nsb) for p in _3buf_pools)
            _ipend = f"s_waitcnt vmcnt({_n_outstanding}) lgkmcnt({_WL_ELGK})\ns_barrier"
        else:
            _n_outstanding = 0
            _ipend = "s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier"
        # Prologue drain: the 3rd-buffer primes are read at phase 1, one partial drain away.
        _pro_vm = _n_outstanding

        def _emit_phase_block(ph, drain_line):
            refill_bp = [(ph + 1) % nbuf_p[p] for p in range(n_pools)]
            write_bp = [ph % nbuf_p[p] for p in range(n_pools)]
            blk = emit_phase(refill_bp, write_bp)
            blk.append(drain_line)
            for p in range(n_pools):
                blk.append(f"s_add_u32 ${o_wsoff[p]}, ${o_wsoff[p]}, ${i_ks[p]}")
            return blk

        refill0 = [(-1 + 1) % nbuf_p[p] for p in range(n_pools)]
        L = [f"s_mov_b32 ${o_cnt}, 0"]
        for p in range(n_pools):
            L.append(f"s_mov_b32 ${o_wsoff[p]}, ${i_soff0[p]}")
        L += [ds_line(refill0, tt) for tt in range(NT, NT + ntmp)]
        # The pre-asm rendezvous covers only the buf0 primes; the buf1 primes wait sinks here to overlap with the ntmp ds_read issue window.
        L.append(f"s_waitcnt vmcnt({_pro_vm}) lgkmcnt(0)")
        # Rendezvous the prologue reads: lgkmcnt(0) is per-wave, so without this a wave could overwrite buf0 while a lagging wave prologue ds_reads are still in flight.
        L.append("s_barrier")
        if nval_can_be_zero:
            # The main loop is a do-while, so without this nval==0 still costs n_phases wasted K-blocks.
            L.append(f"s_cmp_eq_u32 ${i_nval}, 0")
            L.append("s_cbranch_scc1 2f")
        L.append("1:")
        for ph in range(n_phases):
            L += _emit_phase_block(ph, _ipend)
        L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, {n_phases}")
        L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}")
        L.append("s_cbranch_scc1 1b")
        if nval_can_be_zero:
            L.append("2:")

        if _has_tail and _vmcnt_mode == "partial" and _3buf_pool is not None:
            # Full drain before the tail reuses the loop LDS/regs (partial drain needs a next phase, gone after exit); skipped when tail_nval==0.
            L.append(f"s_cmp_eq_u32 ${i_tailval}, 0")
            L.append("s_cbranch_scc1 3f")
            L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            L.append("s_barrier")
            L.append("3:")

        if _has_tail:
            # In-asm tail: up to 5 gated single-block phases reusing the main-loop block; the trailing full drain catches whatever is left in flight.
            for j in range(5):
                skip_lbl = f"{j + 4}"  # distinct numeric local labels, unused elsewhere
                L.append(f"s_cmp_le_u32 ${i_tailval}, {j}")  # tail_nval<=j -> no phase j
                L.append(f"s_cbranch_scc1 {skip_lbl}f")
                L += _emit_phase_block(j, _ipend)
                L.append(f"{skip_lbl}:")

        L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
        asm = "\n".join(L)

        vtmp = [f"=&{{v[{PIN + f * 8}:{PIN + f * 8 + 7}]}}" for f in range(ntmp)]
        cons = ",".join(
            ["=a"] * NT
            + vtmp
            + ["=&s"] * (1 + n_pools)
            + ["v"] * sum(nbase[p] for p in range(n_pools))
            + ["s"] * sum(nbuf_p)
            + ["v"] * nsa
            + ["v"] * nsb
            + (["s", "s", "s", "s", "s", "s"] if _has_tail else ["s", "s", "s", "s", "s"])
            + ["s"] * n_pools
            + [str(q) for q in o_acc]
        )
        st = (
            "!llvm.struct<("
            + ", ".join(["vector<4xf32>"] * NT + ["vector<8xi32>"] * ntmp + ["i32"] * (1 + n_pools))
            + ")>"
        )
        _WL_ASM_CACHE_3BUF[key] = (asm, cons, st)
    asm, cons, st = _WL_ASM_CACHE_3BUF[key]

    n_pools = a_halves + b_halves
    n_quads = a_halves * b_halves
    ins = []
    for p in range(n_pools):
        ins += list(bases[p])
    for p in range(n_pools):
        ins += list(gbases[p])
    ins += list(gl_a) + list(gl_b)
    ins += [rsrc_a, rsrc_b, kstep, kstep_b, nval]
    if _has_tail:
        ins += [tail_nval]
    ins += list(soff0)
    for qi in range_constexpr(n_quads):
        ins += [acc[qi][q] for q in range_constexpr(nq)]
    ins = [arith._to_raw(v) for v in ins]

    r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
    o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(NT)]
    return [o[qi * nq : (qi + 1) * nq] for qi in range(n_quads)]


def _wholeloop_tile_3buf(
    *,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    lds,
    gl_off_a,
    gl_off_b,
    A,
    B,
    a_base,
    b_base,
    a_nrec,
    b_nrec,
    c_n,
    c_m,
    wave_id,
    mfma,
    store_c,
    nta,
    ntb,
    n_accums,
    nsa,
    nsb,
    block_k,
    cs,
    nw,
    cbsz,
    blgp,
    base_row,
    base_col,
    lds_block_m,
    lds_block_n,
    nval,
    do_store=True,  # False = return res, caller stores after the tail
    tail_nval=None,  # pass through to _wholeloop_asm_3buf
    a_plain=False,  # see _wholeloop_tile_3buf's a_plain/a_row_stride
    a_row_stride=None,
    b0_extra_buf,  # pool2's (B0) 3rd buffer, so both B pools are 3-deep, not just pool3 (B1)
    a_halves=2,  # 2 = full a0+a1; 1 = a0-only (last <=128-valid M-block boundary skip)
    b_halves=2,  # 2 = full b0+b1; 1 = b0-only (last <=128-valid N-block boundary skip)
    nval_can_be_zero=False,  # see _wholeloop_asm_3buf
):
    assert not a_plain or a_row_stride is not None, "a_plain=True requires a_row_stride"
    assert a_halves in (1, 2) and b_halves in (1, 2)
    a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
    a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
    b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
    b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
    b_extra1 = lds.B_lds_extra_1  # pool3's 3rd buffer
    cm_i = arith.index_cast(T.index, c_m)
    cn_i = arith.index_cast(T.index, c_n)
    A_K_STEP = arith.index(block_k) * cm_i
    B_K_STEP = arith.index(block_k) * cn_i
    if a_plain:
        A0_gl_offset = 0
        A1_gl_offset = fx.Int32(lds_block_m) * a_row_stride
        A1_soff0 = A1_gl_offset
    else:
        A0_gl_offset, A1_gl_offset = 0, lds_block_m
        A1_soff0 = fx.Int32(lds_block_m)
    B0_gl_offset, B1_gl_offset = 0, lds_block_n

    # Prologue: pools 0-2 prime 2 K-blocks, pool3 primes 3; a_halves/b_halves=1 skip a fully-masked half loads and pool.
    a_g2s.load(a_cur0, A0_gl_offset + 0 * A_K_STEP)
    b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
    if b_halves == 2:
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
    if a_halves == 2:
        a_g2s.load(a_cur1, A1_gl_offset + 0 * A_K_STEP)
    a_g2s.load(a_next0, A0_gl_offset + 1 * A_K_STEP)
    b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
    if b_halves == 2:
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
    if a_halves == 2:
        a_g2s.load(a_next1, A1_gl_offset + 1 * A_K_STEP)
    if b_halves == 2:
        b_g2s.load(b_extra1, B1_gl_offset + 2 * B_K_STEP)  # pool3's 3rd prime (K-block 2)
    b_g2s.load(b0_extra_buf, B0_gl_offset + 2 * B_K_STEP)  # pool2's 3rd prime
    # This rendezvous covers only the buf0 primes (consumed by the whole-loop own prologue reads); buf1 and 3rd-buffer primes are waited later inside the asm.
    _n_deep_b = 1 + (1 if b_halves == 2 else 0)
    wait_barrier(a_halves * nsa + b_halves * nsb + _n_deep_b * nsb)

    # pools[p] = (buf_tuple, s2r); 3buf pools buf_tuple has 3 entries, others 2.
    pool2_bufs = (b_cur0, b_next0, b0_extra_buf)
    a_pools = [((a_cur0, a_next0), a_s2r)]
    if a_halves == 2:
        a_pools.append(((a_cur1, a_next1), a_s2r))
    pools = a_pools + [(pool2_bufs, b_s2r)]
    if b_halves == 2:
        pools.append(((b_cur1, b_next1, b_extra1), b_s2r))
    # A pool buffers are consecutive equal-size LDS fields with a buffer-independent read pattern, so only buffer 0 needs live address VGPRs (rest ride the ds_read immediate).
    bases = [[v for pair in s2r.base_addr(buf_tuple[0]) for v in pair] for buf_tuple, s2r in pools]
    _cs_p = list(cs) if isinstance(cs, (list, tuple)) else [cs] * (a_halves + b_halves)
    _a_buf_bytes = (lds_block_m * block_k) // 1024 * _cs_p[0]
    _b_buf_bytes = (lds_block_n * block_k) // 1024 * _cs_p[a_halves]
    buf_off = [
        [b * (_a_buf_bytes if p < a_halves else _b_buf_bytes) for b in range(len(buf_tuple))]
        for p, (buf_tuple, _s2r) in enumerate(pools)
    ]
    gbases = [
        [
            rocdl.readfirstlane(
                T.i32, fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(_cs_p[p])
            )
            for buf in buf_tuple
        ]
        for p, (buf_tuple, _s2r) in enumerate(pools)
    ]
    gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(nsa)]
    gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(nsb)]
    rsrc_a = _buffer_ops.create_buffer_resource(
        A, max_size=False, num_records_bytes=a_nrec, base_byte_offset=a_base
    )
    rsrc_b = _buffer_ops.create_buffer_resource(
        B, max_size=False, num_records_bytes=b_nrec, base_byte_offset=b_base
    )
    kstep_a = rocdl.readfirstlane(T.i32, fx.Int32(block_k) * c_m)
    kstep_b = rocdl.readfirstlane(T.i32, fx.Int32(block_k) * c_n)
    # soff0[p] = gmem offset for the first in-loop write, targeting K-block nbuf[p] (matches W(0)=nbuf[p] in the asm schedule).
    soff0_a = [rocdl.readfirstlane(T.i32, fx.Int32(A0_gl_offset) + fx.Int32(2) * kstep_a)]
    if a_halves == 2:
        soff0_a.append(rocdl.readfirstlane(T.i32, A1_soff0 + fx.Int32(2) * kstep_a))
    soff0_b = [rocdl.readfirstlane(T.i32, fx.Int32(B0_gl_offset) + fx.Int32(3) * kstep_b)]
    if b_halves == 2:
        soff0_b.append(rocdl.readfirstlane(T.i32, fx.Int32(B1_gl_offset) + fx.Int32(3) * kstep_b))
    acc0 = [[mfma.zero_value] * n_accums for _ in range_constexpr(a_halves * b_halves)]
    res = _wholeloop_asm_3buf(
        nta=nta,
        ntb=ntb,
        bases=bases,
        buf_off=buf_off,
        gbases=gbases,
        gl_a=gl_a6,
        gl_b=gl_b6,
        rsrc_a=rsrc_a,
        rsrc_b=rsrc_b,
        soff0=soff0_a + soff0_b,
        kstep=kstep_a,
        kstep_b=kstep_b,
        nval=nval,
        acc=acc0,
        nsa=nsa,
        nsb=nsb,
        rs=[(a_s2r.width // 16) * a_s2r.chunk_stride] * a_halves
        + [(b_s2r.width // 16) * b_s2r.chunk_stride] * b_halves,
        cs=_cs_p,
        nw=nw,
        cbsz=cbsz,
        blgp=blgp,
        tail_nval=tail_nval,
        a_plain=a_plain,
        a_halves=a_halves,
        b_halves=b_halves,
        nval_can_be_zero=nval_can_be_zero,
    )
    if not do_store:
        return res
    if a_halves == 2 and b_halves == 2:
        _store_quadrants(
            store_c, res[0], res[1], res[2], res[3], base_row, base_col, lds_block_m, lds_block_n
        )
    else:
        # Boundary tile: skipped rows/cols lie outside StoreC row/column clamp, so dropping their stores is a no-op.
        for ah in range_constexpr(a_halves):
            for bh in range_constexpr(b_halves):
                store_c.store(
                    res[ah * b_halves + bh],
                    base_row + ah * lds_block_m,
                    base_col + bh * lds_block_n,
                )
    return res


# Must stay top-level, not nested: FlyDSL @flyc.kernel tracer processes nested defs and would trip @flyc.jit global-drift check on repeat launches.
def _wave4_do_tile_tn(
    t,
    *,
    TOTAL,
    num_xcd,
    G,
    TILES_PER_GROUP,
    N_BLOCKS_M,
    N_BLOCKS_N,
    group_m,
    group_n,
    go_div,
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    OUT_M,
    OUT_N,
    F8_IR_t,
    N_TILES_A,
    N_TILES_B,
    N_ACCUMS,
    N_LDS_STEPS_A,
    N_LDS_STEPS_B,
    _CS,
    N_WAVES,
    cbsz,
    blgp,
    LDS_BLOCK_M,
    LDS_BLOCK_N,
    vmcnt_hint,
    _out_ty,
    gl_off_a,
    gl_off_b,
    A,
    B,
    C,
    A_scale,
    B_scale,
    wave_id,
    wave_m,
    wave_n,
    lds,
    _cm,
    _cn,
    tile_rot=0,
    xcd_aff=None,
    xcd_rot=False,
    a_halves=2,
    b_halves=2,
    a_width=None,
    b_width=None,
    swap_n=False,
    col_safe=False,
    split_kb0=None,
    split_kbe=None,
    band_row=None,
    fold_band=None,
    bounds=None,
    beta_is_one=False,
    tile_mn=None,
    WS=None,
    C_M=None,
    C_N=None,
):
    # Pad-both: A/B stay at padded OUT_M/OUT_N; C is written at the tight C_M/C_N (else padded).
    if C_M is None:
        C_M = OUT_M
    if C_N is None:
        C_N = OUT_N
    if tile_mn is None:
        tt = xcd_remap_pid(t, TOTAL, num_xcd)
        group_idx, block_m, block_n = _wgrad_block_mn(
            tt,
            G,
            TILES_PER_GROUP,
            N_BLOCKS_M,
            N_BLOCKS_N,
            group_m,
            group_n,
            False,
            tile_rot,
            xcd_aff,
            xcd_rot,
        )
    else:
        group_idx, block_m, block_n = tile_mn
    group_idx = _readfirstlane_i32(group_idx)
    block_m = _readfirstlane_i32(block_m)
    block_n = _readfirstlane_i32(block_n)
    if bounds is None:
        m_start = _readfirstlane_i32(_load_go(go_div, group_idx))
        m_end = _readfirstlane_i32(_load_go(go_div, group_idx + 1))
    else:
        # The dispatch already had this group's row range in SGPRs (it decides there whether the
        # group is cut), so the body takes it instead of re-reading the table through a vector
        # load it would then have to readfirstlane.
        m_start, m_end = (_readfirstlane_i32(v) for v in bounds)
    mg = _readfirstlane_i32(m_end - m_start)
    k_iters = ceildiv(mg, BLOCK_K)
    store_base = None
    assert split_kb0 is None or not swap_n, "deep-K slicing is incompatible with the swap_n body"
    assert fold_band is None or (split_kb0 is not None and not beta_is_one), "the in-GEMM fold rides a slice"
    if split_kb0 is not None:
        # Deep-K token split: this work item owns one piece of its own group. The leading piece
        # (band_row < 0) keeps C and the row origin; a later one shifts the origin by its own kb0 and
        # banks its partial in scratch band band_row, summed back by the reduce pass. Own rows only.
        m_off = fx.Int32(split_kb0) * fx.Int32(BLOCK_K)
        m_start = _readfirstlane_i32(m_start + m_off)
        mg = _readfirstlane_i32(mg - m_off)
        k_iters = fx.Int32(_readfirstlane_i32(fx.Int32(split_kbe) - fx.Int32(split_kb0)))
        # The band offset rides on the SRD BASE, not the row index, so the epilogue computes row
        # addresses exactly as the unsliced store does (shifting base_row instead would put the
        # split's scalar chain in front of every store address, penalising launches that slice nothing).
        _shift = _readfirstlane_i32(fx.Int32(band_row) - fx.Int32(group_idx) * fx.Int32(C_M))
        _obytes = 4 if _out_ty is fx.Float32 else 2  # WS bands match C's dtype/width (fp32 when fused)
        # band_row < 0 = the group's leading piece: it holds K from 0, so it owns C and only the
        # pieces behind it bank. Same store instructions either way -- the piece picks the SRD base,
        # and the row addressing below is untouched.
        store_base = arith.select(
            fx.Int32(band_row) >= fx.Int32(0),
            _buffer_ops.extract_base_index(WS)
            + arith.index_cast(T.index, _shift) * arith.index(C_N * _obytes),
            _buffer_ops.extract_base_index(C),
        )
    # main loop takes the largest multiple of 6; the remainder (0..5) is the in-asm fused tail.
    n6 = (k_iters // 6) * 6
    nval_main = _readfirstlane_i32(n6)
    tail_k_u = _readfirstlane_i32(k_iters - n6)  # raw 0..5, for the in-asm fused tail

    bm_off = block_m * BLOCK_M
    bn_off = block_n * BLOCK_N
    # swap_n (square OUT_M==OUT_N half-N boundary tiles) feeds B->a-side and A->b-side so the short N rides the lean a_halves=1 drop; the frag is written transposed.
    baseA = arith.index_cast(T.index, m_start) * arith.index(OUT_M) + arith.index_cast(T.index, bm_off)
    nrecA = arith.maxsi(
        arith.index_cast(T.index, mg) * arith.index(OUT_M) - arith.index_cast(T.index, bm_off),
        arith.index(0),
    )
    baseB = arith.index_cast(T.index, m_start) * arith.index(OUT_N) + arith.index_cast(T.index, bn_off)
    nrecB = arith.maxsi(
        arith.index_cast(T.index, mg) * arith.index(OUT_N) - arith.index_cast(T.index, bn_off),
        arith.index(0),
    )
    if swap_n:
        a_op, b_op = B, A
        a_base, a_nrec, b_base, b_nrec = baseB, nrecB, baseA, nrecA
        a_wave, b_wave = wave_n, wave_m
        c_m_body, c_n_body = _cn, _cm  # a-side(B) K-stride=OUT_N, b-side(A) K-stride=OUT_M
        base_row = bn_off + wave_n * (N_TILES_A * 16)  # a-side -> N origin
        base_col = group_idx * C_M + bm_off + wave_m * (N_TILES_B * 16)  # b-side -> M origin (tight C)
    else:
        a_op, b_op = A, B
        a_base, a_nrec, b_base, b_nrec = baseA, nrecA, baseB, nrecB
        a_wave, b_wave = wave_m, wave_n
        c_m_body, c_n_body = _cm, _cn
        base_row = group_idx * C_M + bm_off + wave_m * (N_TILES_A * 16)  # tight C row origin
        base_col = bn_off + wave_n * (N_TILES_B * 16)

    gA = make_fp8_buffer_tensor_rebased(a_op, F8_IR_t, a_base, a_nrec)
    gB = make_fp8_buffer_tensor_rebased(b_op, F8_IR_t, b_base, b_nrec)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
    mfma._do_mma = lambda _a, _b, _c: asm_mma_do(_a, _b, _c, mode="2", cbsz=cbsz, blgp=blgp)

    # wave bank-swizzle on (matches gl_off_a/b wswz in the kernel body).
    _wswz = True
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_CS)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_CS)
    a_s2r = S2RLoaderTr(
        a_wave,
        N_TILES_A,
        N_TILES_A * 16,
        inline_asm=True,
        vmcnt_hint=vmcnt_hint,
        n_waves=N_WAVES,
        chunk_stride=_CS,
        width=LDS_BLOCK_M if a_width is None else a_width,
        wswz=_wswz,
    )
    b_s2r = S2RLoaderTr(
        b_wave,
        N_TILES_B,
        N_TILES_B * 16,
        inline_asm=True,
        vmcnt_hint=vmcnt_hint,
        n_waves=N_WAVES,
        chunk_stride=_CS,
        width=LDS_BLOCK_N if b_width is None else b_width,
        wswz=_wswz,
    )

    _c_rows = (group_idx + 1) * C_M
    _store_cls = (
        StoreCPerTensorRowN
        if (col_safe and not swap_n and _out_ty is fx.BFloat16 and N_TILES_B % 2 == 0)
        else StoreCPerTensor
    )
    store_c = _store_cls(
        A_scale,
        B_scale,
        C,
        _c_rows,
        fx.Int32(C_N),
        mfma.idx,
        N_TILES_A,
        N_TILES_B,
        _out_ty,
        trans=swap_n,
        col_safe=col_safe and not swap_n,
        c_base=store_base,
        beta_is_one=beta_is_one or fold_band is not None,
        accum_mask=(fx.Int32(band_row) < fx.Int32(0)) if (beta_is_one and split_kb0 is not None) else None,
        # In-GEMM fold: the leading piece adds its group's first scratch band into its own
        # accumulators through the beta=1 read-back, so the reduce pass never touches this tile.
        # (band row origin, row bound) -- a bound of 0 leaves a piece with nothing to fold reading
        # a zero-record descriptor, i.e. adding zero.
        rd_base=_buffer_ops.extract_base_index(WS) if fold_band is not None else None,
        rd_shift=(
            _readfirstlane_i32(fold_band[0] - group_idx * fx.Int32(C_M)) if fold_band is not None else None
        ),
        rd_rows=fold_band[1] if fold_band is not None else None,
    )
    _common = dict(
        a_g2s=a_g2s,
        b_g2s=b_g2s,
        a_s2r=a_s2r,
        b_s2r=b_s2r,
        lds=lds,
        gl_off_a=gl_off_a,
        gl_off_b=gl_off_b,
        A=a_op,
        B=b_op,
        a_base=a_base,
        b_base=b_base,
        a_nrec=a_nrec,
        b_nrec=b_nrec,
        c_n=c_n_body,
        c_m=c_m_body,
        wave_id=wave_id,
        mfma=mfma,
        store_c=store_c,
        nta=N_TILES_A,
        ntb=N_TILES_B,
        n_accums=N_ACCUMS,
        nsa=N_LDS_STEPS_A,
        nsb=N_LDS_STEPS_B,
        block_k=BLOCK_K,
        cs=_CS,
        nw=N_WAVES,
        cbsz=cbsz,
        blgp=blgp,
        base_row=base_row,
        base_col=base_col,
        lds_block_m=LDS_BLOCK_M,
        lds_block_n=LDS_BLOCK_N,
    )

    _b0x = lds.B_lds_extra_0
    _wholeloop_tile_3buf(
        **_common,
        nval=nval_main,
        do_store=True,
        tail_nval=tail_k_u,
        b0_extra_buf=_b0x,
        a_halves=a_halves,
        b_halves=b_halves,
        nval_can_be_zero=True,  # variable-K: groups with k_iters<6 run tail-only
    )


def _make_wave4_smem(*, a_lds_size, b_lds_size, cshuf_ty, cshuf_n):
    """Build the 4-wave SharedStorage @fx.struct. Field ORDER fixes LDS offsets and is
    POOL-MAJOR (a pool's buffers consecutive and equal-size) so the whole-loop addresses every
    buffer off one register set plus a ds_read immediate (see _wholeloop_asm_3buf.ds_line)."""
    F8 = fx.Float8E4M3FN
    ann = {
        "A_lds_cur_0": fx.Array[F8, a_lds_size, 16],
        "A_lds_next_0": fx.Array[F8, a_lds_size, 16],
        "A_lds_cur_1": fx.Array[F8, a_lds_size, 16],
        "A_lds_next_1": fx.Array[F8, a_lds_size, 16],
        "B_lds_cur_0": fx.Array[F8, b_lds_size, 16],
        "B_lds_next_0": fx.Array[F8, b_lds_size, 16],
        "B_lds_extra_0": fx.Array[F8, b_lds_size, 16],
        "B_lds_cur_1": fx.Array[F8, b_lds_size, 16],
        "B_lds_next_1": fx.Array[F8, b_lds_size, 16],
        "B_lds_extra_1": fx.Array[F8, b_lds_size, 16],
        "C_lds_shuffle": fx.Array[cshuf_ty, cshuf_n, 16],
    }
    return fx.struct(type("SharedStorage", (), {"__annotations__": ann}))


_Wave4Geometry = namedtuple(
    "_Wave4Geometry",
    "N_WAVES N_TILES_A N_TILES_B N_ACCUMS LDS_BLOCK_M LDS_BLOCK_N "
    "N_LDS_STEPS_A N_LDS_STEPS_B N_LDS_ROUNDS a_lds_size b_lds_size EPI_PAD cshuf_n cshuf_ty",
)


def _wave4_geometry(*, block_m, block_n, block_k, cs, csa, out_fp16, out_fp32=False):
    """Derived 4-wave tile/LDS geometry shared by both grouped factories (trans_b-agnostic,
    factory-scope Python). ``csa``/``cs`` are the A/B LDS column strides (wgrad shares one
    _CS, dgrad uses _CSA/_CS); EPI_PAD keeps the CShuffle epilogue LDS-bank-conflict-free.
    fp32 doubles the CShuffle staging width, so EPI_PAD drops to 0 to keep C staging in one A buffer."""
    n_waves = 4
    n_tiles_a = block_m // 64
    n_tiles_b = block_n // 64
    lds_block_m = block_m // 2
    lds_block_n = block_n // 2
    n_lds_steps_a = (lds_block_m * block_k) // (256 * 16)
    n_lds_steps_b = (lds_block_n * block_k) // (256 * 16)
    epi_pad = 0 if out_fp32 else 4
    return _Wave4Geometry(
        N_WAVES=n_waves,
        N_TILES_A=n_tiles_a,
        N_TILES_B=n_tiles_b,
        N_ACCUMS=n_tiles_a * n_tiles_b,
        LDS_BLOCK_M=lds_block_m,
        LDS_BLOCK_N=lds_block_n,
        N_LDS_STEPS_A=n_lds_steps_a,
        N_LDS_STEPS_B=n_lds_steps_b,
        N_LDS_ROUNDS=max(n_lds_steps_a, n_lds_steps_b),
        a_lds_size=(lds_block_m * block_k) // 1024 * csa,
        b_lds_size=(lds_block_n * block_k) // 1024 * cs,
        EPI_PAD=epi_pad,
        cshuf_n=n_waves * 16 * (n_tiles_b * 16 + epi_pad),
        cshuf_ty=fx.Float32 if out_fp32 else (fx.Float16 if out_fp16 else fx.BFloat16),
    )


# Distribution-agnostic deep-K wgrad split: a group deep in its own K is cut into equal pieces (no atom exceeds kg/pieces) regardless of the token distribution.
_WGRAD_WAVE = 64  # the [G+1] offset table a head id scans must fit one wave
_WGRAD_SPLIT_NB = 8  # max token chunks the contraction length is cut into (chunk = ktot/NB K-blocks)
_WGRAD_SPLIT_NB_MIN = 4  # below this the split has no interior boundary with a trailing piece (no-op)
_WGRAD_SPLIT_FILL = 4  # target device fills on the hot group; deeper chunks are pure fold overhead.
# 4 (was 2): a shape with few tiles per group (e.g. down-proj, 144) drops to ~2 waves when only a
# handful of experts route (capacity-limited MoE), and NB=4 was too shallow to refill. FILL=4 lifts
# it to NB=8 there while a wide shape (gate_up, 276) still stays NB=4 -- deeper would be fold overhead.
_WGRAD_SPLIT_FIRE = 3  # cut bar (half-chunks of own K): only a group starving the CUs is cut
_WGRAD_SPLIT_HOLD = 1  # promote bar: reorder tiles to head ids without banking; kept below FIRE
_WGRAD_SPLIT_KMIN = 6  # K-blocks a chunk must keep (the whole-loop fused-tail floor)
_WGRAD_TIER_BARS = (256,)  # tier bars (contraction-length divisors) where plain dispatch adds an id class
_WGRAD_RED_JS = 2  # live groups the latency-bound fold walks in parallel
_WGRAD_RED_WPT = 8  # reduce WGs per sliced tile
_WGRAD_RED_VEC = 8  # out_ty elements (128b) each reduce lane moves per pass
_WGRAD_ZERO_JS = 4  # empty groups the zero pass walks in parallel; pow2 (the id splits by shift/mask)
_WGRAD_ZERO_PPW = 3  # contiguous 256-thread passes one zero-fill workgroup streams per empty group
_WGRAD_HAND_AUX = 1  # sc0 on the handoff flag: poll past L1, and producer/consumer share an XCD L2
_WGRAD_HEAD_HW = (2, 3)  # (h, w) rectangle a split piece's XCD class walks; None reuses the tile band
_WGRAD_HEAD_ROT = True  # rotate which run an XCD class takes by the group index (edge-tile balance)


def _wgrad_flag_rows(ntiles, out_n, elem_bytes=2):
    """Scratch rows the in-GEMM fold's i32 handoff flags occupy, one slot per (group, tile)."""
    return ceildiv(ntiles * 4, out_n * elem_bytes)


def _wgrad_split_head_ids(tiles_per_group, total, ncu, nb, nxcd=_WGRAD_XCD_HW):
    """Slice-head dispatch ids for ``nb``, matching ``_compile_grouped_tn_wgrad_4wave``.
    Must be ``<= total`` so the head fits one grid-stride turn; otherwise the 4-wave
    factory asserts and autotune falls through to masked (which cannot serve a tight C)."""
    if nb <= 1:
        return 0
    sp_a = (nb - 1) * tiles_per_group
    sp_2x = ceildiv(2 * sp_a, nxcd) * nxcd
    sp_lead = sp_a if tiles_per_group < ncu and sp_2x <= total else 0
    return ceildiv(sp_a + sp_lead, nxcd) * nxcd


def _wgrad_split_geom(tiles_per_group, total, ncu):
    """Compile-time deep-K split geometry ``(NB, BANDS, FIRE, HOLD)`` shared by factory and host
    entry: NB token chunks, BANDS = NB-1 scratch bands, FIRE/HOLD the cut/promote bars. NB == 1
    disables the path. Static shape only (tiles per group vs CUs); never inspects the distribution."""
    if total <= ncu or ncu <= 0 or tiles_per_group <= 0:
        return 1, 0, 0, 0
    if total // tiles_per_group + 1 > _WGRAD_WAVE:  # offset-table scan must fit one wave
        return 1, 0, 0, 0
    nb = _WGRAD_SPLIT_NB_MIN
    while nb < _WGRAD_SPLIT_NB and tiles_per_group * nb < _WGRAD_SPLIT_FILL * ncu:
        nb *= 2
    # FILL may ask for NB=8 on a 144-tile down-proj; with few local experts (EP>1, G=4)
    # that head is larger than G*tiles and the 4-wave persist compile fails. Lower NB
    # until the head fits; below NB_MIN the split is a no-op so disable it.
    while nb > 1 and _wgrad_split_head_ids(tiles_per_group, total, ncu, nb) > total:
        nb //= 2
    if nb < _WGRAD_SPLIT_NB_MIN:
        return 1, 0, 0, 0
    return nb, nb - 1, _WGRAD_SPLIT_FIRE, _WGRAD_SPLIT_HOLD


def _wgrad_go_read(rsrc, i):
    """Entry ``i`` of the int64 [G+1] offset table (low word at i32 element 2*i) on the SCALAR path.
    s_buffer_load lands in an SGPR under lgkmcnt, which makes the deep-K rule affordable; a per-lane
    gather instead drains vmcnt or keeps a VGPR live across the MAC loop."""
    return _sload_i32(rsrc, 2 * i)


def _wgrad_chunk_geom(m_total, BLOCK_K, NB):
    """``(CHD, CHR)``: the global token-chunk grid, CHD the chunk width in K-blocks (contraction
    length over NB, floored at the fused-tail minimum) and CHR the same in rows. Cut from
    ``m_total`` = lhs.shape[0], a plain SHAPE the host already reads -- a scale, not a distribution."""
    ktot = ceildiv_pow2(m_total, BLOCK_K)
    ch = ceildiv_pow2(ktot, NB)
    chd = arith.select(ch > fx.Int32(_WGRAD_SPLIT_KMIN), ch, fx.Int32(_WGRAD_SPLIT_KMIN))
    return chd, fx.Int32(chd) * fx.Int32(BLOCK_K)


def _wgrad_chunk_of(off, chr_, NB, uni=True):
    """Token chunk holding global row ``off`` (``off // CHR`` saturated at the last chunk), as NB-1
    compares so GEMM and fold re-derive chunk indices identically and cheaply on the scalar path.
    ``uni=False`` keeps it per-lane for the fold's one-lane-per-group enumeration."""
    c0 = fx.Int32(0)
    for c in range_constexpr(1, NB):
        c0 = c0 + fx.Int32(arith.select(off >= fx.Int32(c) * chr_, fx.Int32(1), fx.Int32(0)))
    return _readfirstlane_i32(c0) if uni else c0


def _wgrad_fire_bar(chd, FIRE):
    """K-blocks of its own K a group must hold to be cut: FIRE half-chunks, folded at trace time."""
    return fx.Int32(FIRE // 2) * chd if FIRE % 2 == 0 else (fx.Int32(FIRE) * chd) // fx.Int32(2)


def _wgrad_split_pieces(r0, r1, chr_, NB, LEAD, uni=True):
    """``(c0, np)`` for the group spanning rows ``[r0, r1)``: the chunk it starts in and how many
    work items the head id space hands it (one per interior boundary, plus ``LEAD``). The pieces
    are its own K cut into np equal slices, so the longest atom is kg/np, not the fixed grid's."""
    c0 = _wgrad_chunk_of(r0, chr_, NB, uni=uni)
    return c0, _wgrad_chunk_of(r1 - fx.Int32(1), chr_, NB, uni=uni) - c0 + fx.Int32(LEAD)


def _wgrad_is_cut(np_, kg, chd, FIRE, HOLD, LEAD):
    """``(held, cut)`` off the piece count ``np_`` and the group's own K. HELD iff a boundary falls
    inside and it clears the promote bar (head ids own it); CUT iff it also clears the FIRE bar and
    the id space gives it more than one piece. Every side re-derives the split from these two."""
    held = arith.andi(np_ > fx.Int32(LEAD), kg >= _wgrad_fire_bar(chd, HOLD))
    return held, arith.andi(arith.andi(kg >= _wgrad_fire_bar(chd, FIRE), np_ >= fx.Int32(2)), held)


_WGRAD_SPLIT_WS_CACHE = {}
_WGRAD_SPLIT_WS_SHAPE = {}


def _wgrad_split_ws(OUT_M, OUT_N, G, device, dtype, BLOCK_M=256, BLOCK_N=256):
    """Scratch for the split-K slice partials: one band of OUT_M rows per bankable token chunk, at
    C's row pitch, so a slice store only swaps the band SRD's base. Persistent per (shape, device)
    -- a fixed buffer is what CUDA-graph capture needs; the band count is memoized too."""
    gk = (device.index, OUT_M, OUT_N, G, BLOCK_M, BLOCK_N)
    tpg = ceildiv(OUT_M, BLOCK_M) * ceildiv(OUT_N, BLOCK_N)
    # Tail rows past the bands hold the in-GEMM fold's handoff flags, one i32 per (group, tile)
    # and zero at rest: the folding piece clears its own slot, so one buffer serves every launch.
    frows = _wgrad_flag_rows(G * tpg, OUT_N)
    shape = _WGRAD_SPLIT_WS_SHAPE.get(gk)
    if shape is None:
        ncu = torch.cuda.get_device_properties(device).multi_processor_count
        # Sized for the widest geometry any candidate can pick (the XCD-affine one splits deepest).
        bands = _wgrad_split_geom(tpg, G * tpg, ncu)[1]
        shape = (max(bands, 1) * OUT_M + frows, OUT_N)
        _WGRAD_SPLIT_WS_SHAPE[gk] = shape
    key = (device.index, dtype) + shape
    ws = _WGRAD_SPLIT_WS_CACHE.get(key)
    if ws is None:
        ws = torch.empty(shape, device=device, dtype=dtype)
        ws[shape[0] - frows :].zero_()
        _WGRAD_SPLIT_WS_CACHE[key] = ws
    return ws


def _wgrad_bnd_tiles(rem, waves, n_tiles, quantum=1):
    """MFMA tiles per wave a boundary body needs to cover ``rem`` valid rows/cols of the last block,
    the wgrad twin of the NT kernel's ``_bnd_ntb``. Drops tiles the remainder never reaches; rem=0
    keeps the full half. ``quantum`` rounds up to that side's MFMA emit-order blocking factor."""
    if rem <= 0:
        return n_tiles
    t = -(-min(n_tiles, -(-rem // (waves * 16))) // quantum) * quantum
    return min(n_tiles, t)


def _compile_grouped_tn_wgrad_4wave(
    *,
    OUT_M: int,
    OUT_N: int,
    G: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    num_xcd: int = 8,
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
    group_m: int = 0,
    group_n: int = 0,
    xcd_aff: bool = False,
    xcd_rot: bool = True,
    vmcnt_hint: int = 2,
    cap_cu: int = -1,
    half_bnd: int = -1,
    split_k: bool = True,
    beta_is_one: bool = False,
    out_fp32: bool = False,  # C is fp32 (fused bgrad-accum target); scalar store is fp32-aware
    m_real: int = 0,  # >0: real N (hidden) extent; A/B operands stay padded to OUT_M/OUT_N
    n_real: int = 0,  # >0: real K extent; with c_tight, C collapses onto [G, m_real, n_real]
    c_tight: bool = False,  # C output has the real (tight) pitch, not the padded OUT_M/OUT_N
    _probe: int = 0,
):
    """4-wave (occ=1) grouped TN wgrad dW[g]=A[g]^T@B[g], variable-K per group. 256x256
    whole-loop bare-asm body: runtime nval (floored to x6) + in-asm fused tail; partial
    K-blocks zeroed by per-group SRD num_records clamp. C=[G*C_M, C_N] (tight or padded)."""

    BLOCK_K = 128
    # BLOCK_N=128 unsupported: ds_read_b64_tr_b8 hardware transpose at that width gives a wrong (finite) result. Keep this assert at 256.
    assert BLOCK_M == 256 and BLOCK_N == 256, "4-wave grouped wgrad is 256x256-only"
    assert G >= 1
    # 2-pool (both B pools 3-buffered) + scalar store; _CS=1024 fits 10 buffers.
    _CS = 1024
    # geo.cshuf_n bakes EPI_PAD=4 (row_pad=4 at the StoreCPerTensorCShuffle call sites).
    _geo = _wave4_geometry(
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        block_k=BLOCK_K,
        cs=_CS,
        csa=_CS,
        out_fp16=out_fp16,
        out_fp32=out_fp32,
    )
    N_WAVES = _geo.N_WAVES
    N_TILES_A = _geo.N_TILES_A
    N_TILES_B = _geo.N_TILES_B
    LDS_BLOCK_M = _geo.LDS_BLOCK_M
    LDS_BLOCK_N = _geo.LDS_BLOCK_N
    N_LDS_STEPS_A = _geo.N_LDS_STEPS_A
    N_LDS_STEPS_B = _geo.N_LDS_STEPS_B
    N_LDS_ROUNDS = _geo.N_LDS_ROUNDS
    a_lds_size = _geo.a_lds_size
    b_lds_size = _geo.b_lds_size
    _cshuf_ty = _geo.cshuf_ty
    _M_VALID = m_real if m_real else OUT_M
    _N_VALID = n_real if n_real else OUT_N
    assert 0 < _M_VALID <= OUT_M, f"m_real={m_real} out of range for OUT_M={OUT_M}"
    assert 0 < _N_VALID <= OUT_N, f"n_real={n_real} out of range for OUT_N={OUT_N}"
    _C_M = _M_VALID if c_tight else OUT_M
    _C_N = _N_VALID if c_tight else OUT_N
    N_BLOCKS_M = (_C_M + BLOCK_M - 1) // BLOCK_M
    N_BLOCKS_N = (_C_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    _NCU = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    # (group_m, group_n) are the XCD-affine (h, w) under xcd_aff, carrying their own class rotation
    # (_XCD_ROT) instead of the band maps' tile_rot. The super-block spans ONE group so every XCD
    # holds the same fraction of every group's K -- loads stay equal under ANY token split.
    _XCD_GP, _XCD_K = 1, _WGRAD_XCD_HW
    _XCD_AFF = (group_m, group_n, _XCD_GP) if xcd_aff else None
    _XCD_ROT = _XCD_AFF is not None and xcd_rot and _wgrad_xcd_rot_ok(TILES_PER_GROUP, _XCD_GP)
    if _XCD_AFF is not None:
        assert num_xcd <= 1, "xcd_aff assumes the hardware bid % 8 XCD split, not a pid remap"
        assert (
            _wgrad_xcd_aff_geom(N_BLOCKS_M, N_BLOCKS_N, TILES_PER_GROUP, _XCD_K) is not None
            and N_BLOCKS_M % group_m == 0
            and N_BLOCKS_N % group_n == 0
            and group_m * group_n <= TILES_PER_GROUP // _XCD_K
        ), f"bad xcd_aff geometry ({group_m},{group_n}) for {N_BLOCKS_M}x{N_BLOCKS_N}"
    _TILE_ROT = 0 if _XCD_AFF is not None else (_WG_TILE_ROT if TILES_PER_GROUP > _WG_TILE_ROT else 0)
    # Split pieces run the plain map over a single group, so their band spans all XCDs (gp=1). A
    # piece is the whole of a cut launch's work and wants its own tile->XCD rectangle: the band
    # tuned for the plain tiles is not the one that suits a piece (measured, collapsed-only).
    _HEAD_HW = _WGRAD_HEAD_HW or (group_m, group_n)
    if N_BLOCKS_M % _HEAD_HW[0] or N_BLOCKS_N % _HEAD_HW[1] or TILES_PER_GROUP % (_HEAD_HW[0] * _HEAD_HW[1]):
        _HEAD_HW = (group_m, group_n)  # the override must tile the grid; otherwise keep the band
    _HEAD_AFF = (*_HEAD_HW, 1) if _XCD_AFF is not None else None
    _HEAD_ROT = _HEAD_AFF is not None and _WGRAD_HEAD_ROT and _wgrad_xcd_rot_ok(TILES_PER_GROUP, 1)
    # The boundary bodies drop a short last block's masked MFMA, but a cheap tile frees its CU
    # early and desyncs the L2-slab phase cohorts (no resync at occ=1) -- pays off shallow, not
    # deep. M side stays behind the launch-depth gate, N behind the cost gate. half_bnd: 1=M, 2=N.
    _BND_GATED = half_bnd < 0 and TOTAL >= _WGRAD_AFF_ROUNDS * _NCU
    _BND_MASK = 3 if half_bnd < 0 else half_bnd
    _HALF_M = bool(_BND_MASK & 1) and not _BND_GATED and 0 < _C_M % BLOCK_M <= BLOCK_M // 2
    _WAVE_N = BLOCK_N // LDS_BLOCK_N
    _N_REM = _C_N % BLOCK_N if 0 < _C_N % BLOCK_N <= LDS_BLOCK_N else 0
    # The short N body drops _BND_COLS columns off one N-block, buying that fraction of MFMA
    # against a phase cost ~1/_WGRAD_BND_PHASE_INV of the wall. Enable when the MFMA saved beats
    # that cost (deep); shallow, the cost is off the wall (see _HALF_M).
    _BND_COLS = BLOCK_N - _wgrad_bnd_tiles(_N_REM, _WAVE_N, N_TILES_B) * _WAVE_N * 16
    _HALF_N = (
        bool(_BND_MASK & 2)
        and _N_REM > 0
        and (not _BND_GATED or _BND_COLS * _WGRAD_BND_PHASE_INV > N_BLOCKS_N * BLOCK_N)
    )
    _BND_NTB = _wgrad_bnd_tiles(_N_REM if _HALF_N else 0, _WAVE_N, N_TILES_B)
    _WAVE_M = BLOCK_M // LDS_BLOCK_M
    _M_REM = _C_M % BLOCK_M if 0 < _C_M % BLOCK_M <= LDS_BLOCK_M else 0
    _BND_NTA = _wgrad_bnd_tiles(_M_REM if _HALF_M else 0, _WAVE_M, N_TILES_A, quantum=2)
    _BND_WM = LDS_BLOCK_M // 2
    _BND_A_NARROW = _HALF_M and _BND_NTA * _WAVE_M * 16 <= _BND_WM and (_BND_WM * BLOCK_K) % (256 * 16) == 0
    _BND_NSA = (_BND_WM * BLOCK_K) // (256 * 16) if _BND_A_NARROW else N_LDS_STEPS_A
    _BND_WN = LDS_BLOCK_N // 2
    _BND_B_NARROW = _HALF_N and _BND_NTB * _WAVE_N * 16 <= _BND_WN and (_BND_WN * BLOCK_K) % (256 * 16) == 0
    _BND_NSB = (_BND_WN * BLOCK_K) // (256 * 16) if _BND_B_NARROW else N_LDS_STEPS_B
    # Every stored column is < _C_N at compile time, so the epilogue per-element OOB select is dead.
    _COL_SAFE = (_C_N % BLOCK_N == 0) or (_HALF_N and _BND_NTB * _WAVE_N * 16 == _C_N % BLOCK_N)
    # Deep-K split geometry, independent of the tile map and the chosen candidate: a slice brings its
    # own group index and a plain in-group block position, so the host scratch and every candidate
    # agree on bands.
    _NB, _SP_BANDS, _FIRE, _HOLD = (
        _wgrad_split_geom(TILES_PER_GROUP, TOTAL, _NCU) if split_k else (1, 0, 0, 0)
    )
    _SPLIT = _NB > 1
    # One dispatch id per (chunk boundary, in-group block), AHEAD of the plain tile ids since a chunk
    # is the longest atom; padded to the XCD count so a plain tile keeps its frozen-grid class.
    _SP_A = (_NB - 1) * TILES_PER_GROUP if _SPLIT else 0
    _SP_2X = ceildiv(2 * _SP_A, _WGRAD_XCD_HW) * _WGRAD_XCD_HW
    _SP_LEAD = _SP_A if _SPLIT and TILES_PER_GROUP < _NCU and _SP_2X <= TOTAL else 0
    _LEAD_PC = 1 if _SP_LEAD else 0  # work items a cut group gets beyond its interior boundaries
    _SP_SPS = _SP_A + _SP_LEAD
    _SP_HEAD = ceildiv(_SP_SPS, _WGRAD_XCD_HW) * _WGRAD_XCD_HW if _SPLIT else 0
    _GRID_EXT = _SP_HEAD + TOTAL
    assert _SP_HEAD <= TOTAL, "slice head must fit one grid-stride turn"
    _PLAIN_ROT = min(_SP_2X, TOTAL) if _SPLIT else 0
    _ONE_TURN = cap_cu <= 0  # grid covers the tile space: every workgroup takes exactly one tile id
    # In-GEMM fold: a cut group's LEADING piece folds its first peer band into its own accumulators
    # via the beta=1 read-back, dropping that tile's HBM round trip to one band read in the epilogue.
    # Preconditions (compile time): lead ids rank ABOVE every boundary id (_SP_LEAD) so a consumer
    # never dispatches before its producer -- with _ONE_TURN that makes the spin deadlock-free; and
    # their id distance is an XCD-count multiple, so producer/consumer share an L2 (no writeback).
    _FUSE = _SPLIT and _ONE_TURN and bool(_SP_LEAD) and _SP_A % _WGRAD_XCD_HW == 0 and not beta_is_one
    _FLAG_N = TOTAL  # one slot per (group, in-group block)
    # The flags sit past the last band; the host sizes the scratch off the same tight C extents.
    _FLAG_OFF = _SP_BANDS * _C_M * _C_N * (4 if out_fp32 else 2)
    _TIER = _SPLIT and _ONE_TURN and TOTAL % _WGRAD_XCD_HW == 0
    _PLAIN_GRID = (len(_WGRAD_TIER_BARS) + 1) * TOTAL if _TIER else TOTAL
    # Reduce: one workgroup covers BLOCK_M//_WGRAD_RED_WPT rows x BLOCK_N cols of a sliced tile. Its
    # id space is (tile, sub) only -- the banked chunks are walked inside the workgroup, so the count
    # does not grow with NB and a launch that slices nothing stays at the frozen cost.
    _RED_VEC = _WGRAD_RED_VEC // 2 if out_fp32 else _WGRAD_RED_VEC  # fp32 halves the dwordx4 store vector
    _RED_ROWS = BLOCK_M // _WGRAD_RED_WPT
    _RED_LPR = BLOCK_N // _RED_VEC  # lanes spanning one tile row
    _RED_RPP = 256 // _RED_LPR  # rows one 256-thread pass covers
    _RED_L2WPT = _WGRAD_RED_WPT.bit_length() - 1
    _RED_JS = max(1, min(_WGRAD_RED_JS, G))
    _RED_GRID = max(1, TILES_PER_GROUP * _WGRAD_RED_WPT * _RED_JS)
    # An EMPTY group's output is a plain zero fill. Run as a GEMM tile body it has no MFMA to hide
    # under, so the store burst streams at ~0.27 us/MB against a 0.17 us/MB write roofline; instead
    # it rides the already-launched thin fold pass and the GEMM stops dispatching those tiles. Its C
    # slice is contiguous, so the fill walks it FLAT: one wavefront store per 1KB run, no ragged-tile
    # lane waste. (beta=1 accumulates into C and must leave empty rows untouched -> keeps in-GEMM.)
    _ZERO = _SPLIT and _probe != 1 and not beta_is_one
    _Z_PASS = 256 * _RED_VEC  # C elements one 256-thread pass writes, contiguous
    _ZSTRIPE = ceildiv(ceildiv(_C_M * _C_N, _Z_PASS), _WGRAD_ZERO_PPW)
    _ZJS = min(_WGRAD_ZERO_JS, 1 << (G.bit_length() - 1))
    _ZJS_L2 = _ZJS.bit_length() - 1
    _ZERO_GRID = _ZSTRIPE * _ZJS if _ZERO else 0
    # With the in-GEMM fold active, the reduce pass's fold ids are empty on any launch whose groups
    # split in two, so they ride the zero pass's workgroups through the grid-stride loop instead of
    # costing 2304 dispatches of their own.
    _RED_LAUNCH = max(_RED_GRID, _ZERO_GRID) if (_FUSE and _ZERO) else _RED_GRID + _ZERO_GRID
    assert not _ZERO or _ZJS == 1 << _ZJS_L2, "the zero pass splits its id by shift/mask"
    assert not _SPLIT or (
        _WGRAD_RED_WPT & (_WGRAD_RED_WPT - 1) == 0 and _RED_ROWS % _RED_RPP == 0 and _C_N % _RED_VEC == 0
    ), "split-K reduce needs a pow2 WPT, row-aligned passes and a vector-aligned C_N"

    # Scalar store shrinks C_lds_shuffle to a stub so the 2nd deferred-write B buffer fits.
    SharedStorage = _make_wave4_smem(
        a_lds_size=a_lds_size,
        b_lds_size=b_lds_size,
        cshuf_ty=_cshuf_ty,
        cshuf_n=16,
    )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_grouped_tn_wgrad_4wave(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,
        WS: fx.Tensor,
        m_total: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float32 if out_fp32 else (fx.Float16 if out_fp16 else fx.BFloat16)
        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))
        if const_expr(_SPLIT):
            go_rs = _buffer_ops.create_buffer_resource(
                group_offs, max_size=False, num_records_bytes=(G + 1) * 8
            )

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x
        nsms = fx.grid_dim.x

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        # wave bank-swizzle on (write side; matches S2RLoaderTr wswz).
        gl_off_a = compute_global_swizzle_nn(
            lane_id, wave_id, OUT_M, N_LDS_ROUNDS, width=LDS_BLOCK_M, wswz=True
        )
        gl_off_b = compute_global_swizzle_nn(
            lane_id, wave_id, OUT_N, N_LDS_ROUNDS, width=LDS_BLOCK_N, wswz=True
        )
        if const_expr(_BND_A_NARROW):
            gl_off_a_bnd = compute_global_swizzle_nn(
                lane_id, wave_id, OUT_M, _BND_NSA, width=_BND_WM, wswz=True
            )
        if const_expr(_BND_B_NARROW):
            gl_off_b_bnd = compute_global_swizzle_nn(
                lane_id, wave_id, OUT_N, _BND_NSB, width=_BND_WN, wswz=True
            )
        _cm = fx.Int32(OUT_M)
        _cn = fx.Int32(OUT_N)
        _gt = None

        def _do_tile_3buf(
            t,
            tile_a_halves,
            tile_b_halves,
            tile_swap_n=False,
            split_kb0=None,
            split_kbe=None,
            band_row=None,
            fold_band=None,
            bounds=None,
            tile_mn=None,
        ):
            _narrow_a = _BND_A_NARROW and tile_a_halves == 1
            _narrow_b = _BND_B_NARROW and tile_b_halves == 1
            _nta = _BND_NTA if tile_a_halves == 1 else N_TILES_A
            _ntb = _BND_NTB if tile_b_halves == 1 else N_TILES_B
            _wave4_do_tile_tn(
                t,
                beta_is_one=beta_is_one,
                TOTAL=TOTAL,
                num_xcd=num_xcd,
                G=G,
                TILES_PER_GROUP=TILES_PER_GROUP,
                N_BLOCKS_M=N_BLOCKS_M,
                N_BLOCKS_N=N_BLOCKS_N,
                group_m=group_m,
                group_n=group_n,
                go_div=go_div,
                BLOCK_K=BLOCK_K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                OUT_M=OUT_M,
                OUT_N=OUT_N,
                F8_IR_t=F8_IR_t,
                N_TILES_A=_nta,
                N_TILES_B=_ntb,
                N_ACCUMS=_nta * _ntb,
                N_LDS_STEPS_A=_BND_NSA if _narrow_a else N_LDS_STEPS_A,
                N_LDS_STEPS_B=_BND_NSB if _narrow_b else N_LDS_STEPS_B,
                _CS=_CS,
                N_WAVES=N_WAVES,
                cbsz=cbsz,
                blgp=blgp,
                LDS_BLOCK_M=LDS_BLOCK_M,
                LDS_BLOCK_N=LDS_BLOCK_N,
                vmcnt_hint=vmcnt_hint,
                _out_ty=_out_ty,
                gl_off_a=gl_off_a_bnd if _narrow_a else gl_off_a,
                gl_off_b=gl_off_b_bnd if _narrow_b else gl_off_b,
                a_width=_BND_WM if _narrow_a else None,
                b_width=_BND_WN if _narrow_b else None,
                A=A,
                B=B,
                C=C,
                A_scale=A_scale,
                B_scale=B_scale,
                wave_id=wave_id,
                wave_m=wave_m,
                wave_n=wave_n,
                lds=lds,
                _cm=_cm,
                _cn=_cn,
                tile_rot=_TILE_ROT,
                xcd_aff=_XCD_AFF,
                xcd_rot=_XCD_ROT,
                a_halves=tile_a_halves,
                b_halves=tile_b_halves,
                swap_n=tile_swap_n,
                col_safe=_COL_SAFE,
                split_kb0=split_kb0,
                split_kbe=split_kbe,
                band_row=band_row,
                fold_band=fold_band,
                bounds=bounds,
                tile_mn=tile_mn,
                WS=WS,
                C_M=_C_M,
                C_N=_C_N,
            )

        # The tile id travels as an ARGUMENT and every dynamic branch body holds a CALL only: a
        # dynamic if/else body is rewritten into its own function, so a variable assigned there
        # does not reach a closure called from the same branch (it only re-binds after the if).
        def _emit_tile(t, _sl, _mn=None):
            if const_expr(_HALF_M or _HALF_N):
                if const_expr(_mn is not None):
                    _blk_m, _blk_n = _mn
                else:
                    _tt = xcd_remap_pid(t, TOTAL, num_xcd)
                    _, _blk_m, _blk_n = _wgrad_block_mn(
                        _tt,
                        G,
                        TILES_PER_GROUP,
                        N_BLOCKS_M,
                        N_BLOCKS_N,
                        group_m,
                        group_n,
                        False,
                        _TILE_ROT,
                        _XCD_AFF,
                        _XCD_ROT,
                    )
                if const_expr(_HALF_M and _HALF_N):
                    if _readfirstlane_i32(_blk_m) == fx.Int32(N_BLOCKS_M - 1):
                        if _readfirstlane_i32(_blk_n) == fx.Int32(N_BLOCKS_N - 1):
                            # The 2-pool (a0-only,b0-only) corner faults the tight beta=1 read-back, so tight routes it to the 3-pool body.
                            if const_expr(c_tight):
                                _do_tile_3buf(t, 2, 1, **_sl)
                            else:
                                _do_tile_3buf(t, 1, 1, **_sl)
                        else:
                            _do_tile_3buf(t, 1, 2, **_sl)
                    else:
                        if _readfirstlane_i32(_blk_n) == fx.Int32(N_BLOCKS_N - 1):
                            _do_tile_3buf(t, 2, 1, **_sl)
                        else:
                            _do_tile_3buf(t, 2, 2, **_sl)
                elif const_expr(_HALF_M):
                    if _readfirstlane_i32(_blk_m) == fx.Int32(N_BLOCKS_M - 1):
                        _do_tile_3buf(t, 1, 2, **_sl)
                    else:
                        _do_tile_3buf(t, 2, 2, **_sl)
                else:
                    if _readfirstlane_i32(_blk_n) == fx.Int32(N_BLOCKS_N - 1):
                        _do_tile_3buf(t, 2, 1, **_sl)
                    else:
                        _do_tile_3buf(t, 2, 2, **_sl)
            else:
                _do_tile_3buf(t, 2, 2, **_sl)

        if const_expr(_SPLIT):
            # The chunk grid, once, on the scalar path, out of the launch's contraction length
            # only -- the tile ids need it to cut their own group's trip count at the boundary and
            # the slice ids to place themselves, and neither reads the offset table for it.
            _chd, _chr = _wgrad_chunk_geom(m_total, BLOCK_K, _NB)
            _ktot = ceildiv_pow2(m_total, BLOCK_K)
            _tier_bar = [_readfirstlane_i32(udiv(_ktot, d)) for d in _WGRAD_TIER_BARS]

            def _emit_chunk(d):
                """A head id carries (chunk boundary, in-group block) and runs the piece that boundary
                starts, inverting to its group with a wave-wide offset-table gather. Ids past _SP_A
                form the lead class: same decode, but the piece is the group's leading segment."""
                _cls = d >= fx.Int32(_SP_A)  # lead class

                def _pick(a, b):  # a on the lead class, b on the boundary class
                    return arith.select(_cls, a, b) if const_expr(_SP_LEAD) else b

                _i = _readfirstlane_i32(d - _pick(fx.Int32(_SP_A), fx.Int32(0)))
                _cb = _readfirstlane_i32(udiv(_i, TILES_PER_GROUP))
                _q = _readfirstlane_i32(_i - _cb * fx.Int32(TILES_PER_GROUP))
                _key = _readfirstlane_i32((_cb + fx.Int32(1)) * _chr)
                _tb = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2)[0]
                _tb = fx.Int32(arith.select(lane_id <= fx.Int32(G), _tb, fx.Int32(0x7FFFFFFF)))
                _gi = _readfirstlane_i32(_wave_count_le_i32(_tb, _key) - fx.Int32(1))
                _r0 = _readfirstlane_i32(_readlane_i32(_tb, _gi))
                _r1 = _readfirstlane_i32(_readlane_i32(_tb, _gi + fx.Int32(1)))
                _kg = ceildiv_pow2(_r1 - _r0, BLOCK_K)
                # The in-group block takes the XCD-affine swizzle over a single group, on its own
                # _HEAD_HW rectangle (row-major would hand the pieces non-overlapping operand slabs).
                # Boundary bodies all sit in the last block row/column, so rotating which run a class
                # takes by the GROUP index spreads that cheaper edge share evenly across the XCDs.
                _hq = (_gi * fx.Int32(TILES_PER_GROUP) + _q) if const_expr(_HEAD_ROT) else _q
                _, _mb, _nb = _wgrad_block_mn(
                    _readfirstlane_i32(_hq),
                    1,
                    TILES_PER_GROUP,
                    N_BLOCKS_M,
                    N_BLOCKS_N,
                    group_m,
                    group_n,
                    False,
                    0,
                    _HEAD_AFF,
                    _HEAD_ROT,
                )
                _mb = _readfirstlane_i32(_mb)
                _nb = _readfirstlane_i32(_nb)
                _c0, _np = _wgrad_split_pieces(_r0, _r1, _chr, _NB, _LEAD_PC)
                _np = _readfirstlane_i32(_np)
                _lead = _cb == _c0  # chunk-index test: a group opening on a boundary starts next chunk
                _j = _pick(fx.Int32(0), _readfirstlane_i32(_cb - _c0 + fx.Int32(_LEAD_PC)))
                _held, _cut = _wgrad_is_cut(_np, _kg, _chd, _FIRE, _HOLD, _LEAD_PC)
                _w = _kg
                for _s in range_constexpr(2, _NB + 1):
                    _w = arith.select(_np == fx.Int32(_s), udiv(_kg, _s), _w)
                _w = _readfirstlane_i32(_w)
                # an uncut but held group is carried whole by piece 0; stopping short would drop K
                _kbs = arith.select(_cut, _j * _w, fx.Int32(0))
                _kbe = arith.select(arith.andi(_cut, _j + fx.Int32(1) < _np), (_j + fx.Int32(1)) * _w, _kg)
                _bnd = arith.select(arith.andi(_cut, _j > fx.Int32(0)), _readfirstlane_i32(_cb), fx.Int32(-1))
                _live = arith.andi(
                    arith.andi(_r0 < _key, _key < m_total),
                    arith.andi(arith.ori(arith.andi(_lead, _held), _cut), _j < _np),
                )
                if const_expr(_SP_LEAD):
                    _live = _pick(
                        arith.andi(arith.andi(_lead, _key < m_total), _cut),
                        _live,
                    )
                _kbe = _readfirstlane_i32(_kbe)
                _kbs = _readfirstlane_i32(_kbs)
                if const_expr(_SP_HEAD > _SP_SPS):
                    _live = arith.andi(_live, d < fx.Int32(_SP_SPS))
                _sl = dict(
                    split_kb0=_kbs,
                    split_kbe=_kbe,
                    band_row=_readfirstlane_i32(_bnd * fx.Int32(_C_M)),
                    bounds=(_r0, _r1),
                    tile_mn=(_gi, _mb, _nb),
                )
                if const_expr(_FUSE):
                    # FOLD = the lead piece of a cut group; it adds the band its j==1 peer banks.
                    # PUB = that peer. Both derive the same (group, in-group block) slot, and the
                    # peer's id is exactly _SP_A below the lead's, so they land on one XCD.
                    _fold = arith.andi(_cls, _live) if const_expr(_SP_LEAD) else _live
                    _pub = arith.andi(_live, arith.andi(_cut, _j == fx.Int32(1)))
                    _fb = _readfirstlane_i32(arith.select(_fold, _c0 + fx.Int32(1 - _LEAD_PC), fx.Int32(0)))
                    _sl["fold_band"] = (
                        _readfirstlane_i32(_fb * fx.Int32(_C_M)),
                        _readfirstlane_i32(
                            arith.select(_fold, (_fb + fx.Int32(1)) * fx.Int32(_C_M), fx.Int32(0))
                        ),
                    )
                    _slot = _readfirstlane_i32((_gi * fx.Int32(TILES_PER_GROUP) + _q) * fx.Int32(4))
                    _fl_rs = _buffer_ops.create_buffer_resource(
                        WS,
                        max_size=False,
                        num_records_bytes=arith.index(_FLAG_N * 4),
                        base_byte_offset=arith.index(_FLAG_OFF),
                    )
                    # A piece with nothing to fold polls a zero-record descriptor for zero, so the
                    # spin costs it one load and never blocks.
                    _poll_rs = _buffer_ops.create_buffer_resource(
                        WS,
                        max_size=False,
                        num_records_bytes=arith.index_cast(
                            T.index,
                            _readfirstlane_i32(arith.select(_fold, fx.Int32(_FLAG_N * 4), fx.Int32(0))),
                        ),
                        base_byte_offset=arith.index(_FLAG_OFF),
                    )
                # The poll stays OUT of the dispatch branch below: an inline-asm block inside a
                # dynamic-if body (which the rewriter lifts into its own function) faults. A dead
                # id polls a zero-record descriptor for zero and falls straight through.
                if const_expr(_FUSE):
                    spin_flag_eq(
                        _poll_rs,
                        _slot,
                        _readfirstlane_i32(arith.select(_fold, fx.Int32(1), fx.Int32(0))),
                    )
                if _readfirstlane_i32(arith.select(_live, fx.Int32(1), fx.Int32(0))) > fx.Int32(0):
                    _emit_tile(d, _sl, (_mb, _nb))
                    if const_expr(_FUSE):
                        # One store raises the peer's flag and clears the folded one: the drain
                        # ahead of it is free here, the workgroup is about to retire either way.
                        wait_barrier(0)
                        _buffer_ops.buffer_store(
                            fx.Int32(arith.select(_pub, fx.Int32(1), fx.Int32(0))),
                            _fl_rs,
                            _slot,
                            mask=arith.andi(fx.thread_idx.x == fx.Int32(0), arith.ori(_pub, _fold)),
                            cache_modifier=_WGRAD_HAND_AUX,
                            offset_is_bytes=True,
                        )

        def _emit_plain(t, cls=None):
            """A plain tile id runs the frozen body over its group's whole K, needing one scalar
            verdict to know whether its group is cut. Under the tiered dispatch it also compares its
            own K against the depth bars to know whether this id class owns it."""
            _tt = xcd_remap_pid(t, TOTAL, num_xcd)
            _pg, _pm, _pn = _wgrad_block_mn(
                _tt,
                G,
                TILES_PER_GROUP,
                N_BLOCKS_M,
                N_BLOCKS_N,
                group_m,
                group_n,
                False,
                _TILE_ROT,
                _XCD_AFF,
                _XCD_ROT,
            )
            _pg = _readfirstlane_i32(_pg)
            _pr0 = _wgrad_go_read(go_rs, _pg)
            _pr1 = _wgrad_go_read(go_rs, _pg + fx.Int32(1))
            _pkg = ceildiv_pow2(_pr1 - _pr0, BLOCK_K)
            _pheld, _ = _wgrad_is_cut(
                _wgrad_split_pieces(_pr0, _pr1, _chr, _NB, _LEAD_PC)[1],
                _pkg,
                _chd,
                _FIRE,
                _HOLD,
                _LEAD_PC,
            )
            _pgo = arith.select(_pheld, fx.Int32(0), fx.Int32(1))
            if cls is not None:
                _ptier = fx.Int32(0)
                for _b in _tier_bar:
                    _ptier = _ptier + arith.select(_pkg >= _b, fx.Int32(0), fx.Int32(1))
                _pgo = arith.select(
                    arith.select(_pheld, fx.Int32(-1), _ptier) == cls, fx.Int32(1), fx.Int32(0)
                )
            if const_expr(_ZERO):
                _pgo = arith.select(_pr1 > _pr0, _pgo, fx.Int32(0))
            if _readfirstlane_i32(_pgo) > fx.Int32(0):
                _emit_tile(
                    t,
                    dict(tile_mn=(_pg, _pm, _pn), bounds=(_pr0, _pr1)),
                    (_pm, _pn),
                )

        def _plain_rot(t):
            """Plain ids ride a rotation of their own: where the tile stream wraps is a co-residency
            choice, not a consequence of how many slice ids there are. The shift is an XCD-count
            multiple, so a tile keeps its class through the wrap."""
            _p = t - fx.Int32(_PLAIN_ROT)
            return _readfirstlane_i32(_p + arith.select(_p < fx.Int32(0), fx.Int32(TOTAL), fx.Int32(0)))

        if const_expr(_SPLIT and _ONE_TURN):
            _plain_wg = pid >= fx.Int32(_SP_HEAD)
            if _readfirstlane_i32(arith.select(_plain_wg, fx.Int32(1), fx.Int32(0))) > fx.Int32(0):
                _d = _readfirstlane_i32(pid - fx.Int32(_SP_HEAD))
                if const_expr(_TIER):
                    _cls = _readfirstlane_i32(udiv(_d, TOTAL))
                    _emit_plain(_plain_rot(_readfirstlane_i32(_d - _cls * fx.Int32(TOTAL))), _cls)
                else:
                    _emit_plain(_plain_rot(_d))
            else:
                _emit_chunk(pid)
            return
        for d in range(pid, _GRID_EXT, nsms):
            if const_expr(_SPLIT):
                # TWO BODIES chosen by a wave-uniform scalar test on the id alone, so a plain tile id
                # runs the frozen body and reads nothing but its own group's bounds. Merging them via
                # selects would make every workgroup carry the banked store's scalar chain.
                if _readfirstlane_i32(
                    arith.select(d >= fx.Int32(_SP_HEAD), fx.Int32(1), fx.Int32(0))
                ) > fx.Int32(0):
                    _emit_plain(d - fx.Int32(_SP_HEAD))
                else:
                    _emit_chunk(d)
            else:
                _emit_tile(d, {})

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_grouped_tn_wgrad_reduce(C: fx.Tensor, group_offs: fx.Tensor, WS: fx.Tensor, m_total: fx.Int32):
        """Fold the deep-K scratch bands back into C: work item (tile q, sub) sums the banked groups'
        bands into its tile in ascending chunk order, in fp32 (no atomics, bit-reproducible). The
        walk is sized by a wave scan of the per-group rule, so a launch with nothing to fold exits at once."""
        _ = str(fx.thread_idx.x)
        _out_ty = fx.Float32 if out_fp32 else (fx.Float16 if out_fp16 else fx.BFloat16)
        _ir_ty = _out_ty.ir_type
        _obytes = 4 if out_fp32 else 2  # C/band element width (fp32 under fused bgrad-accum)
        f32v = fx.T.VectorType.get([_RED_VEC], fx.T.f32())
        outv = fx.T.VectorType.get([_RED_VEC], _ir_ty)
        go_rs = _buffer_ops.create_buffer_resource(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        c_base = _buffer_ops.extract_base_index(C)
        ws_base = _buffer_ops.extract_base_index(WS)
        tid = fx.thread_idx.x
        _chd, _chr = _wgrad_chunk_geom(m_total, BLOCK_K, _NB)
        # Lane g = group g, so the cut rule runs for all G groups at once (one gather pair plus one
        # wave scan). The row count is clamped at zero first: an unclamped negative span (lane G's
        # 0 upper bound, logical shift) would score as a huge K and put a group index past C.
        _lane_g = tid % fx.Int32(64)
        _g0 = _lane_tbl_load(go_rs, _lane_g, G + 1, stride=2)[0]
        _g1 = _lane_tbl_load(go_rs, _lane_g, G + 1, stride=2, first=1)[0]
        _lk = ceildiv_pow2(fx.Int32(arith.select(_g1 > _g0, _g1 - _g0, fx.Int32(0))), BLOCK_K)
        _lnp = _wgrad_split_pieces(_g0, _g1, _chr, _NB, _LEAD_PC, uni=False)[1]
        _, _lcut = _wgrad_is_cut(_lnp, _lk, _chd, _FIRE, _HOLD, _LEAD_PC)
        if const_expr(_FUSE):
            # The GEMM's lead piece already folded the group's FIRST band, so a group cut in two
            # has nothing left here and drops out of the walk entirely.
            _lcut = arith.andi(_lcut, _lnp >= fx.Int32(3))
        _lp = _lane_tbl_scan([fx.Int32(arith.select(_lcut, fx.Int32(1), fx.Int32(0)))])[0]
        _nlive = _readlane_i32(_lp, 63)
        if const_expr(_ZERO):
            # Lane g holds group g's row span, so one more scan enumerates the EMPTY groups the
            # same way _lp enumerates the banked ones. Lanes past G read 0 and must be masked off.
            _ep = _lane_tbl_scan(
                [
                    fx.Int32(
                        arith.select(arith.andi(_lane_g < fx.Int32(G), _g1 <= _g0), fx.Int32(1), fx.Int32(0))
                    )
                ]
            )[0]
            _nempty = _readlane_i32(_ep, 63)
        col_l = (tid % fx.Int32(_RED_LPR)) * fx.Int32(_RED_VEC)
        row_l = tid // fx.Int32(_RED_LPR)

        def _fold_slot(w0):
            """Sum every banked group's bands into this work item's (tile, sub) slice of C."""
            # Walked back to front: the GEMM stores the split pieces in ascending id order, so the
            # tiles it wrote last are the ones still hot in the LLC when the fold starts.
            w = fx.Int32(_RED_GRID - 1) - w0
            _wslot = _readfirstlane_i32(w >> _RED_L2WPT)
            sub = _readfirstlane_i32(w & fx.Int32(_WGRAD_RED_WPT - 1))
            _wj = _readfirstlane_i32(udiv(_wslot, TILES_PER_GROUP))
            _wq = _readfirstlane_i32(_wslot - _wj * fx.Int32(TILES_PER_GROUP))
            # Same block map the slice ids used (row-major inside the group), not the tile map.
            _bm = _readfirstlane_i32(udiv(_wq, N_BLOCKS_N))
            bm_off = _readfirstlane_i32(_bm * fx.Int32(BLOCK_M))
            bn_off = _readfirstlane_i32((_wq - _bm * fx.Int32(N_BLOCKS_N)) * fx.Int32(BLOCK_N))
            col = bn_off + col_l
            col_ok = col < fx.Int32(_C_N)
            off0 = (sub * fx.Int32(_RED_ROWS) + row_l) * fx.Int32(_C_N) + col
            # Zero trips when no group banked anything, which is every launch whose groups all hold
            # less than two token chunks -- the balanced deploy point included.
            for _jv in range(_wj, _nlive, fx.Int32(_RED_JS)):
                # The prefix-sum is monotone, so counting the lanes at or below a value gives the
                # index of the _j-th live group directly -- one ballot, no search. Counted down for
                # the same reason the tile walk is, and still a permutation of the groups.
                gi = _wave_count_le_i32(_lp, _readfirstlane_i32(_nlive - fx.Int32(1) - _jv))
                _r0 = _readfirstlane_i32(_readlane_i32(_g0, gi))
                # The group banks chunks (first chunk, last chunk]: the same span the GEMM's slice
                # ids cover, re-derived from the same row range.
                _c0 = _wgrad_chunk_of(_r0, _chr, _NB)
                _c1 = _wgrad_chunk_of(_readfirstlane_i32(_readlane_i32(_g1, gi)) - fx.Int32(1), _chr, _NB)
                rs_c = make_row_band_resource(c_base, gi * _C_M + bm_off, (gi + 1) * _C_M, _C_N, _obytes)
                # Compile-time band sweep: unwritten bands are out of SRD (no request); the lead class shifts the first band one chunk later.
                _cf = _c0 + fx.Int32(1 - _LEAD_PC + (1 if _FUSE else 0))
                _sel = [arith.andi(fx.Int32(c) > _cf, fx.Int32(c) <= _c1) for c in range_constexpr(1, _NB)]
                for p in range_constexpr(_RED_ROWS // _RED_RPP):
                    off = off0 + fx.Int32(p * _RED_RPP * _C_N)
                    parts = [
                        _buffer_ops.buffer_load(rs_c, off, vec_width=_RED_VEC, dtype=_ir_ty, mask=col_ok)
                    ]
                    for c in range_constexpr(1, _NB):
                        # Chunk c banks row band c-1, a compile-time SRD offset. A band outside the
                        # group's chunk span, or one whose boundary fell in the group's last
                        # K-block (an empty slice the GEMM never stored), reads out of bounds.
                        rs_w = make_row_band_resource(
                            ws_base, bm_off + fx.Int32((c - 1) * _C_M), c * _C_M, _C_N, _obytes
                        )
                        parts.append(
                            _buffer_ops.buffer_load(
                                rs_w,
                                arith.select(_sel[c - 1], off, fx.Int32(0x3FFFFFFF)),
                                vec_width=_RED_VEC,
                                dtype=_ir_ty,
                                mask=col_ok,
                            )
                        )
                    if const_expr(out_fp32):
                        acc = parts[0]
                        for v in parts[1:]:
                            acc = arith.addf(acc, v)
                        _buffer_ops.buffer_store(acc, rs_c, off, mask=col_ok)
                    else:
                        acc = arith.extf(f32v, parts[0])
                        for v in parts[1:]:
                            acc = arith.addf(acc, arith.extf(f32v, v))
                        _buffer_ops.buffer_store(arith.trunc_f(outv, acc), rs_c, off, mask=col_ok)

        def _zero_fill(z):
            """Zero one flat run of every EMPTY group. The GEMM used to run these as full tile
            bodies, where a store burst with no MFMA behind it streams at well under the device
            write rate at occupancy 1; here the fill rides an already-launched thin kernel. The
            empty groups come off the same [G+1] table scan the fold uses -- device side, no
            dependence on the distribution."""
            # Stripe index in the LOW bits so both fields fall out of one shift/mask.
            _zj = _readfirstlane_i32(z & fx.Int32(_ZJS - 1))
            _zr = _readfirstlane_i32(z >> _ZJS_L2)
            zoff0 = _zr * fx.Int32(_WGRAD_ZERO_PPW * _Z_PASS) + tid * fx.Int32(_RED_VEC)
            # One zero vector for the whole walk, read out of a zero-record descriptor: a per-group
            # load would put a vmcnt drain of the previous group's stores in front of every batch.
            zv = _buffer_ops.buffer_load(go_rs, fx.Int32(0x3FFFFFFF), vec_width=_RED_VEC, dtype=_ir_ty)
            for _zv in range(_zj, _nempty, fx.Int32(_ZJS)):
                gi = _wave_count_le_i32(_ep, _readfirstlane_i32(_zv))
                # The group's whole slice in one SRD: the run past C's last element falls out of
                # records and needs no lane mask.
                rs_z = make_row_band_resource(c_base, gi * _C_M, (gi + 1) * _C_M, _C_N, _obytes)
                for p in range_constexpr(_WGRAD_ZERO_PPW):
                    _buffer_ops.buffer_store(zv, rs_z, zoff0 + fx.Int32(p * _Z_PASS))

        for w in range(fx.block_idx.x, fx.Int32(_RED_GRID + _ZERO_GRID), fx.grid_dim.x):
            if const_expr(_ZERO):
                if _readfirstlane_i32(w) >= fx.Int32(_RED_GRID):
                    _zero_fill(_readfirstlane_i32(w - fx.Int32(_RED_GRID)))
                else:
                    _fold_slot(w)
            else:
                _fold_slot(w)

    _GRID_X = (_SP_HEAD + _PLAIN_GRID) if cap_cu <= 0 else min(TOTAL, int(cap_cu), _NCU)
    _ATTRS = make_value_attrs(1, 0, "256,256")

    @flyc.jit
    def launch_grouped_tn_wgrad_4wave(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,
        WS: fx.Tensor,
        m_total: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_grouped_tn_wgrad_4wave(
            A,
            B,
            C,
            A_scale,
            B_scale,
            group_offs,
            WS,
            m_total,
            value_attrs=_ATTRS,
        ).launch(grid=(_GRID_X, 1, 1), block=(256, 1, 1), stream=stream)
        if const_expr(_SPLIT and _probe != 1):
            # Same stream: the fold sees every slice partial. The grid is the compile-time tile
            # space; a launch whose groups all pick S=1 finds nothing live and exits at once.
            kernel_grouped_tn_wgrad_reduce(C, group_offs, WS, m_total).launch(
                grid=(_RED_LAUNCH, 1, 1), block=(256, 1, 1), stream=stream
            )

    return launch_grouped_tn_wgrad_4wave


def _wgrad_masked_cfg(
    OUT_M, OUT_N, G, out_fp16, cbsz, blgp, chunk, group_m, num_xcd, i64_traverse=False, beta_is_one=False
):
    """Compile (or cache-hit) the masked chunked wgrad for one (chunk, group_m, num_xcd)."""
    ck = ("masked", OUT_M, OUT_N, G, out_fp16, cbsz, blgp, chunk, group_m, num_xcd, i64_traverse, beta_is_one)
    l = _GROUPED_WGRAD_LAUNCH_CACHE.get(ck)
    if l is None:
        l = _compile_grouped_tn_wgrad_masked(
            OUT_M=OUT_M,
            OUT_N=OUT_N,
            G=G,
            num_xcd=num_xcd,
            acc_mode="agpr",
            s2r_inline=True,
            out_fp16=out_fp16,
            cbsz=cbsz,
            blgp=blgp,
            group_m=group_m,
            store_cshuffle=True,
            chunk=chunk,
            i64_traverse=i64_traverse,
            beta_is_one=beta_is_one,
        )
        _GROUPED_WGRAD_LAUNCH_CACHE[ck] = l
    return l


# Per-group tile rotation: odd and coprime with the 16-tile CU-residue class, so it spreads the boundary tiles.
_WG_TILE_ROT = 5

# 4-wave (group_m, group_n, num_xcd, xcd_aff) candidates; xcd=1 keeps group-major LPT order (a
# skewed load needs it, unknowable at dispatch).
_WGRAD_4WAVE_CANDS = ((4, 2, 1, 0), (4, 4, 1, 0), (8, 4, 1, 0))
# Dispatch rounds (G*tiles_per_group / ncu) from which the steady state, rather than the per-tile
# work, sets the wall: the XCD-affine run leads the candidate list from here, and from here the
# short last-M-block body stops paying for itself (see _HALF_M).
_WGRAD_AFF_ROUNDS = 8
# Reciprocal of the wall fraction a boundary body costs once the launch is that deep: a cheap tile
# frees its CU early and the WG that refills it runs out of L2 phase with its neighbours (_HALF_N).
_WGRAD_BND_PHASE_INV = 20
# Margin a candidate must beat the standing best by to take its place (see _autotune_wgrad_dispatch).
_WGRAD_RACE_MARGIN = 0.985


def _wgrad_4wave_cands(OUT_M, OUT_N, G, ncu, block=256):
    """Order the candidates so cands[0] -- the incumbent the race only displaces on a real margin --
    is the one the tile geometry argues for: the XCD-affine run leads once the launch is deep, and
    the band behind it is the NARROW one (the larger rectangle side sets the operand it streams)."""
    n_blocks_m = (OUT_M + block - 1) // block
    n_blocks_n = (OUT_N + block - 1) // block
    tiles_per_group = n_blocks_m * n_blocks_n
    head = (4, 2)
    band = tuple(sorted(_WGRAD_4WAVE_CANDS, key=lambda c: c[:2] != head))
    deep = G * tiles_per_group >= _WGRAD_AFF_ROUNDS * ncu
    xcd_k = _WGRAD_XCD_HW  # one group per super-block, so every XCD takes a share of every group
    aff = (
        None
        if not deep and _wgrad_band_is_xcd_aff(n_blocks_m, n_blocks_n, *head)
        else _wgrad_xcd_aff_geom(n_blocks_m, n_blocks_n, tiles_per_group, xcd_k)
    )
    # The affine run leads once the launch is deep: it hands every XCD a compact rectangle of the
    # group's tiles, where a band map's class is every nxcd-th tile of the band order.
    lead_aff = deep and aff is not None and aff != (1, 1)
    if aff is None:
        geom = band
    else:
        aff_c = (aff[0], aff[1], 1, 1)
        geom = (aff_c,) + band[:3] if lead_aff else band + (aff_c,)
    cands = tuple(c + (-1, True) for c in geom)
    if not deep:
        return cands
    # The boundary bodies, which a deep launch otherwise gates off, keep the deep-K rule: offering
    # them WITHOUT it lets the balanced race -- all the tuner sees -- trade the whole mechanism for
    # the boundary tiles' MFMA, which is worth far less than the deep-K rule.
    cands += tuple(c[:4] + (3, True) for c in cands[:2])
    if lead_aff:
        # A taller band than the geometry asks for only rode in front when a super-block spanning
        # several groups absorbed the extra operand rows; with one group per super-block it goes LAST
        # and must beat the geometry by the hysteresis rather than winning on dispatch order.
        h = 2 * aff[0]
        if n_blocks_m % h == 0 and h * aff[1] <= tiles_per_group // xcd_k:
            cands += ((h, aff[1], 1, 1, 3, True),)
    return cands


def _autotune_wgrad_dispatch(
    OUT_M,
    OUT_N,
    G,
    out_fp16,
    cbsz,
    blgp,
    args,
    i64_traverse=False,
    out_fp32=False,
    m_real=0,
    n_real=0,
    c_tight=False,
):
    """Race the wgrad candidates and return a ``finalize(beta_is_one) -> raw launch`` factory,
    cached per static (OUT_M,OUT_N,G,dtype,i64,tight,fp32), never per m_total. Racing at beta=0 keeps
    the winner beta-independent (a beta=1 build never enters the race, where a tight one would fault)."""

    lhs_live, rhs_live = args[0], args[1]
    M_total = lhs_live.shape[0]
    pm = max(1, M_total // G)
    M_c = G * pm
    # Fixed-seed production-magnitude fp8 operands: fp8 GEMM wall time is data-magnitude/DVFS sensitive, so an all-zero probe would misrank candidates.
    _g = torch.Generator(device=lhs_live.device).manual_seed(0)
    lhs_c = torch.empty((M_c, OUT_M), device=lhs_live.device, dtype=lhs_live.dtype)
    rhs_c = torch.empty((M_c, OUT_N), device=rhs_live.device, dtype=rhs_live.dtype)
    lhs_c.view(torch.uint8).random_(0, 64, generator=_g)
    rhs_c.view(torch.uint8).random_(0, 64, generator=_g)
    # Distribution-agnostic: score the tuner on a BALANCED load only. No skew probe in the tuner --
    # the kernel serves every runtime distribution with one strategy and must not be tuned to any
    # particular (skewed) one.
    bench_c = torch.empty_like(args[2])
    mps = []
    for offs_c in (_balanced_group_offs(M_c, G, lhs_live.device),):
        mps.append(
            [
                (
                    lhs_c.view(torch.int8),
                    rhs_c.view(torch.int8),
                    bench_c,
                    args[3],
                    args[4],
                    offs_c,
                    args[6],
                    M_c,
                    args[8],
                ),
                bench_c,
                None,
                None,
            ]
        )

    _ncu = torch.cuda.get_device_properties(lhs_live.device).multi_processor_count
    # Candidate tile geometry follows the C extent (real under c_tight) so group_m/group_n match the compile's N_BLOCKS.
    _GM = m_real if (c_tight and m_real) else OUT_M
    _GN = n_real if (c_tight and n_real) else OUT_N
    wave4_cands = _wgrad_4wave_cands(_GM, _GN, G, _ncu)

    def _compile_4wave(gm, gn, xcd, aff, half, split, beta_is_one=False):
        return _compile_grouped_tn_wgrad_4wave(
            OUT_M=OUT_M,
            OUT_N=OUT_N,
            G=G,
            out_fp16=out_fp16,
            cbsz=cbsz,
            blgp=blgp,
            num_xcd=xcd,
            group_m=gm,
            group_n=gn,
            xcd_aff=bool(aff),
            half_bnd=half,
            split_k=split,
            beta_is_one=beta_is_one,
            out_fp32=out_fp32,
            m_real=m_real,
            n_real=n_real,
            c_tight=c_tight,
        )

    def _build_masked(beta_is_one):
        # The masked kernel writes the padded pitch; fail loudly rather than corrupt a tight C.
        assert not (c_tight and ((m_real and m_real != OUT_M) or (n_real and n_real != OUT_N))), (
            "wgrad masked fallback cannot serve a tight C (padded-pitch store into real-extent buffer)"
        )
        _masked = _wgrad_masked_cfg(
            OUT_M,
            OUT_N,
            G,
            out_fp16,
            cbsz,
            blgp,
            8,
            4,
            1,
            i64_traverse=i64_traverse,
            beta_is_one=beta_is_one,
        )

        def prod(*a):  # masked fallback takes neither the split-K scratch nor m_total, keeps stream
            return _masked(*a[:6], a[-1])

        return prod

    def _try_4wave(cfg):
        """Compile+launch ``cfg`` on the balanced probe. None if the output is non-finite."""
        cand = _compile_4wave(*cfg)
        for mp in mps:
            cand(*mp[0])
            torch.cuda.synchronize()
            if not torch.isfinite(mp[1].view(-1)[:1024].float()).all().item():
                return None
        return cand

    print(
        f"[wgrad-autotune] OUT=({OUT_M},{OUT_N}) real=({m_real},{n_real}) G={G} "
        f"M_total={M_total} i64={i64_traverse} tight={c_tight} fp32={out_fp32} cands={wave4_cands}",
        flush=True,
    )
    prod = None
    best_cfg = None
    errors = []
    # i64 huge shapes skip persist 4-wave on padded C (masked is the ref). Tight C cannot
    # use masked, so still try 4-wave there -- better a clear raise than a padded-pitch store.
    skip_4wave = i64_traverse and not c_tight
    if not skip_4wave:
        for cfg in wave4_cands:
            try:
                cand = _try_4wave(cfg)
            except Exception as exc:
                errors.append((cfg, f"{type(exc).__name__}: {exc}"))
                if not c_tight:
                    break
                continue
            if cand is not None:
                prod, best_cfg = cand, cfg
                print(f"[wgrad-autotune] persist-ref {cfg}", flush=True)
                break
            errors.append((cfg, "non-finite"))
            if not c_tight:
                break
    if prod is None:  # i64 huge shape or 4-wave failed to compile/produced NaN -> masked ref
        if c_tight:
            raise RuntimeError(
                "wgrad persist 4-wave cannot serve a tight C "
                f"(padded-pitch masked store into real-extent buffer). "
                f"OUT=({OUT_M},{OUT_N}) real=({m_real},{n_real}) G={G} "
                f"M_total={M_total} i64={i64_traverse} fp32={out_fp32} "
                f"cands={wave4_cands} errors={errors}"
            )
        prod = _build_masked(False)
        best_cfg = None
        for mp in mps:
            prod(*mp[0])
            torch.cuda.synchronize()
            if not torch.isfinite(mp[1].view(-1)[:1024].float()).all().item():
                return _build_masked  # numeric guard: nothing else is safe to try

    for mp in mps:  # establish the per-M numeric reference from prod
        prod(*mp[0])
        torch.cuda.synchronize()
        r = mp[1].detach().clone().float()
        mp[2], mp[3] = r, (float((r * r).sum().item()) or 1.0)

    def _score(launch):
        """Max launch time over the canonical loads, or None on rel-MSE drift / NaN."""
        worst = 0.0
        for targs, ov, ref, refnorm in mps:
            launch(*targs)
            torch.cuda.synchronize()
            o = ov.detach().float()
            e = float(((o - ref) * (o - ref)).sum().item())
            rel = e / refnorm
            if rel >= (2e-2**2) or not torch.isfinite(o.view(-1)[:1024]).all().item():
                return None
            worst = max(worst, _robust_time(launch, targs))
        return worst

    best_s = _score(prod)
    race = tuple(cfg for cfg in wave4_cands if cfg != best_cfg)
    for cfg in race:
        try:
            l = _compile_4wave(*cfg)
        except Exception:
            continue
        s = _score(l)  # numeric guard folded in: None -> skip
        # EVERY displacement clears the hysteresis, not just the first: scoring candidates in a fixed
        # order under one clock makes a reading carry that order as a systematic bias, so a bar
        # frozen off the FIRST best would let the bias decide every later swap.
        if s is not None and (best_s is None or s < best_s * _WGRAD_RACE_MARGIN):
            best_s, best_cfg = s, cfg
    if best_cfg is None:
        if c_tight:
            raise RuntimeError(
                "wgrad persist 4-wave race left no tight-C winner "
                f"OUT=({OUT_M},{OUT_N}) real=({m_real},{n_real}) G={G} "
                f"M_total={M_total} i64={i64_traverse} fp32={out_fp32} errors={errors}"
            )
        return _build_masked
    _cfg = best_cfg
    return lambda beta_is_one: _compile_4wave(*_cfg, beta_is_one=beta_is_one)


def grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel(
    lhs: "torch.Tensor",
    rhs: "torch.Tensor",
    lhs_scale: "torch.Tensor",
    rhs_scale: "torch.Tensor",
    group_offs: "torch.Tensor",
    out_dtype=torch.bfloat16,
    num_cu: "int | None" = -1,
    beta: float = 0.0,
    out: "torch.Tensor | None" = None,
    m_real: "int | None" = None,
    n_real: "int | None" = None,
    c_tight: bool = False,
) -> "torch.Tensor":
    """FlyDSL per-tensor variable-K grouped fp8 GEMM (wgrad), matching the Triton entry.
    C[g] = lhs[g]^T @ rhs[g] * lhs_scale * rhs_scale, out [G, C_M, C_N]; group_offs [G+1]
    int splits M. The caller (backend) has already applied the trans_c lhs/rhs swap.
    Pad-both: lhs/rhs stay padded to OUT_M/OUT_N; c_tight + m_real/n_real collapse C to [G, m_real, n_real]. out_dtype may be fp32."""
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0], f"M_total mismatch lhs={lhs.shape[0]} rhs={rhs.shape[0]}"
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    _m_real = m_real if m_real is not None else OUT_M
    _n_real = n_real if n_real is not None else OUT_N
    assert 0 < _m_real <= OUT_M and 0 < _n_real <= OUT_N
    _tight = bool(c_tight) and (_m_real != OUT_M or _n_real != OUT_N)
    C_M = _m_real if _tight else OUT_M
    C_N = _n_real if _tight else OUT_N

    out = resolve_accum_out(out, beta, (G, C_M, C_N), lhs.device, out_dtype)
    beta_is_one = beta == 1.0
    ws = _wgrad_split_ws(C_M, C_N, G, lhs.device, out_dtype)
    # kernel reads group_offs as int64 low-words via a free int32-view (no .to(int32) cast).
    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    out_fp32 = out_dtype == torch.float32
    cbsz = 1 if lhs.dtype == torch.float8_e5m2 else 0
    blgp = 1 if rhs.dtype == torch.float8_e5m2 else 0

    # Full rank (not flattened): a flat reshape(-1) overflows the int32 shape pack when M_total*OUT_{M,N} > 2^31.
    lhs_i8 = lhs.view(torch.int8)
    rhs_i8 = rhs.view(torch.int8)
    lsf = lhs_scale.float().reshape(1)
    rsf = rhs_scale.float().reshape(1)
    stream = torch.cuda.current_stream()

    M_total = lhs.shape[0]
    i64_tr = (M_total * OUT_M >= 2**32) or (M_total * OUT_N >= 2**32)
    # out as 2D [G*C_M, C_N] (the kernel's stacked-group view). m_total = the contraction
    # length, a plain SHAPE (the same one the i64 test reads): the kernel cuts its token-chunk grid
    # out of it, never looking at the offset table's CONTENT.
    out2d = out.view(G * C_M, C_N)
    wargs = (lhs_i8, rhs_i8, out2d, lsf, rsf, go32, ws, M_total, stream)
    cfg_key = (OUT_M, OUT_N, G, out_fp16, cbsz, blgp, i64_tr, out_fp32, C_M, C_N, _tight)
    finalize = _GROUPED_WGRAD_CFG_CACHE.get(cfg_key)
    if finalize is None:
        finalize = _autotune_wgrad_dispatch(
            OUT_M,
            OUT_N,
            G,
            out_fp16,
            cbsz,
            blgp,
            wargs,
            i64_tr,
            out_fp32=out_fp32,
            m_real=(_m_real if _tight else 0),
            n_real=(_n_real if _tight else 0),
            c_tight=_tight,
        )
        _GROUPED_WGRAD_CFG_CACHE[cfg_key] = finalize
    # beta_is_one is baked into the kernel (the epilogue reads C back), so the compiled artifact keys on it.
    at_key = (cfg_key, beta_is_one)
    entry = _GROUPED_WGRAD_AT_CACHE.get(at_key)
    if entry is None:
        entry = [finalize(beta_is_one), None]
        _GROUPED_WGRAD_AT_CACHE[at_key] = entry
    raw, compiled = entry
    # Mode-split, same as the forward entry: CUDA-graph capture takes the raw @flyc.jit closure,
    # eager takes a flyc.compile-d object. The raw closure re-binds the signature every dispatch,
    # which costs more host time than eager -- worth it only under capture.
    if torch.cuda.is_current_stream_capturing():
        raw(*wargs)
    else:
        if compiled is None:
            try:
                # compile runs the kernel once to build its artifact -- against the caller's
                # buffer that would be a second accumulate, so beta=1 compiles on a scratch.
                compiled = compile_with_scratch_out(raw, wargs, out_index=2)
            except Exception:  # masked fallback is a plain closure, not a @flyc.jit function
                compiled = raw
            entry[1] = compiled
        compiled(*wargs)
    return out
