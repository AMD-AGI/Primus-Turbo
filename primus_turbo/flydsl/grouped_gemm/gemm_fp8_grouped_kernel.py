###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL fp8 per-tensor (TENSORWISE) GROUPED GEMM — M-grouped operator.

Covers the forward (NT: out = a @ b^T) and dgrad (NN: grad_a = grad_out @ b)
of grouped/MoE GEMM, where A is [M_total, K] (groups concatenated along M),
B is [G, N, K] (per-group weights), out is [M_total, N], and
``group_offs`` [G+1] int32 splits M_total into G groups.

Design (CPU-sync-free, reuses the dense kernel body verbatim):
  * Grid is over-launched to a host upper bound
    ``(ceil(M_total/BLOCK_M) + G) * n_blocks`` (no device read of group_lens);
    each WG computes the true ``total_tiles`` on-device via an O(G) scan and
    returns early (whole body guarded by ``if pid < total_tiles``) when its
    pid is past the end.
  * The same O(G) scan maps pid -> (group_idx, local tile) -> (local_block_m,
    block_n). Per-group addressing needs NO base-pointer shift:
      - A/B loads add the group element offset (m_start*K / group_idx*N*K); the
        full-tensor SRD clamps the last over-read to 0.
      - the C store passes ``c_rows = group_offs[group_idx+1]`` (the ABSOLUTE
        group-end row) so its SRD bound clamps a partial M-tile's extra rows
        (which belong to the next group) out — no spill across groups.
  * Per-tensor scale = scalar a_scale/b_scale (reused StoreCPerTensor).

Built on the dense kernel's primitives; see gemm_fp8_kernel.py for the K-loop /
K-tail / barrier rationale (identical here).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.fp8_gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    StoreCPerTensorCShuffle,
    _readfirstlane_i32,
    asm_mma_do,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor,
    make_value_attrs,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)

# Baked NT super-block tile swizzle width (0 = row-major; the autotune sweeps group_m
# per shape for B[g] N-stripe L2 reuse).
_GROUPED_NT_GROUPM = 0


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


def _store_quadrants(store_c, c00, c01, c10, c11, base_row, base_col, LDS_BLOCK_M, LDS_BLOCK_N):
    """Store the four output quadrants (shared by all 6 kernels; base_row/base_col are
    computed per-kernel by the caller)."""
    store_c.store(c00, base_row + 0, base_col + 0)
    store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
    store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
    store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


# ── PERSISTENT grouped NN dgrad: a fixed grid of num_sms WGs strides the tile space
#    via scf.for; total_tiles from an on-device O(G) scan (no host read); LDS reused
#    across tiles (per-tile entry barrier isolates prev-tile reads from next writes).
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
):
    """Persistent (CPU-sync-free) grouped NN dgrad. Same math as the dense NN
    kernel but a fixed grid of ``num_sms`` WGs strides over the
    tile space via scf.for, eliminating the over-launch wasted WGs and
    amortising the per-WG fixed cost (O(G) scan + prelude + epilog).

    ``group_m``/``group_n`` port the NT fwd L2-reuse tile swizzle (1D M-cluster /
    2D band): same-N-stripe M-tiles cluster so B[g]'s N-stripe stays L2-resident.
    The B[g]=[K,N] N-stripe is reused across the clustered M-tiles exactly like
    NT's B[g]=[N,K]. Both gated by the in-kernel bpr_g/n_blocks guards (row-major
    fallback for small/skewed groups) so they can never corrupt tiny groups."""
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

    # CShuffle epilogue staging (see NT): 8 waves x 16 rows x Cc(=N_TILES_B*16) out_ty.
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
    def kernel_grouped_nn_persistent(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,  # int32 [G+1]
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)  # materialize before S2RLoaderTr (dense NN note)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        n_blocks = ceildiv(c_n, BLOCK_N)

        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))

        # total_tiles on-device (O(G) scan; no host read of group lens).
        total_tiles = fx.Int32(0)
        prev_off = _load_go(go_div, 0)
        for g in range_constexpr(G):
            nxt_off = _load_go(go_div, g + 1)
            m_g = nxt_off - prev_off
            total_tiles = total_tiles + ceildiv(m_g, BLOCK_M) * n_blocks
            prev_off = nxt_off

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x
        nsms = fx.grid_dim.x  # persistent stride = number of launched WGs

        if const_expr(not persistent):
            # one tile per WG: pin total_tiles to SGPR and s_endpgm the over-launched WGs.
            total_tiles = _readfirstlane_i32(total_tiles)
            _llvm.inline_asm(
                None,
                [pid.ir_value(), arith._to_raw(total_tiles)],
                "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
                "s,s,~{scc},~{memory}",
                has_side_effects=True,
            )

        # Per-tile body (inlined free function so the ast-rewriter handles `if wave_m==1`
        # + range_constexpr and loaders/mfma/store aren't mis-collected as scf.for iter_args).
        def _do_tile(t):
            # XCD remap of the tile id (bijection; identity when num_xcd<=1): same-group
            # tiles cluster on one XCD for per-XCD L2 reuse of B[g].
            tt = xcd_remap_pid(t, total_tiles, num_xcd)
            # tt -> (group_idx, tile_start) via O(G) scan.
            cum = fx.Int32(0)
            group_idx = fx.Int32(0)
            tile_start = fx.Int32(0)
            p2 = _load_go(go_div, 0)
            for g in range_constexpr(G):
                nx = _load_go(go_div, g + 1)
                mg = nx - p2
                tg = ceildiv(mg, BLOCK_M) * n_blocks
                nc = cum + tg
                inq = (tt >= cum) & (tt < nc)
                group_idx = arith.select(inq, fx.Int32(g), group_idx)
                tile_start = arith.select(inq, cum, tile_start)
                cum = nc
                p2 = nx

            m_start = _load_go(go_div, group_idx)
            m_end = _load_go(go_div, group_idx + 1)
            local = tt - tile_start
            # L2-reuse tile swizzle (group_n band -> group_m 1D -> row-major); same
            # B[g] N-stripe reuse as NT, guards degenerate to row-major for small groups.
            local_block_m, block_n = _grouped_block_mn(
                local, m_start, m_end, n_blocks, BLOCK_M, group_m, group_n
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
            A0_gl_offset = m_row * K
            A1_gl_offset = (m_row + LDS_BLOCK_M) * K
            b_grp = group_idx * K * c_n
            B0_gl_offset = b_grp + block_n * BLOCK_N
            B1_gl_offset = b_grp + block_n * BLOCK_N + LDS_BLOCK_N

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

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
            # B transpose-load via inline-asm ds_read_b64_tr_b8: the opaque asm hides the
            # wave-coop transpose reads from the backend so it keeps load/mfma overlap
            # (the intrinsic would force a vmcnt(0) drain). Inline path needs agpr_alloc>0.
            b_s2r = S2RLoaderTr(wave_n, N_TILES_B, 32, inline_asm=(agpr_inplace and acc_mode == "agpr"))
            if const_expr(store_cshuffle):
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
                )
            else:
                store_c = StoreCPerTensor(
                    A_scale, B_scale, C, m_end, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty
                )

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            # Inner before-mfma scheduling barrier (see NT). sched_schedbar=True swaps
            # it for a compile-time sched_barrier(0) (no runtime WG sync). After-mfma
            # barriers stay real (gfx950 mfma-src/ds-read VGPR-overlap race).
            def _ibar():
                if const_expr(sched_schedbar):
                    rocdl.sched_barrier(0)
                else:
                    rocdl.s_barrier()

            # Prelude.
            b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)
            # persistent: unconditional barrier (cross-tile phase-correctness). 8w: one
            # tile per WG, so the dense divergent `if wave_m==1` barrier is correct.
            if const_expr(persistent):
                rocdl.s_barrier()
            else:
                if wave_m == 1:
                    rocdl.s_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)
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
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
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
                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
                rocdl.s_setprio(1)
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if nt_vmcnt >= 0:
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
                        constraints="",
                        has_side_effects=True,
                    )
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            # Epilog 1.
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

            # Epilog 2 (K-tail).
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
        # persistent: cap to _cap WGs (reserve CUs). non-persistent: full upper-bound grid,
        # one tile per WG (over-launched WGs s_endpgm in-kernel).
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
            c_n,
            value_attrs=attrs,
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_nn_persistent


def _compile_grouped_nt(
    *,
    K: int,
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
    persistent: bool = True,  # True = scf.for tile loop (fixed grid, cap_cu reserves CUs); False = one tile/WG + s_endpgm over-launch guard (full-device default)
    cap_cu: int = -1,  # >0: cap grid to this many WGs (= reserve device CUs for comm-compute overlap). <=0: use the full device CU count.
):
    """Grouped NT forward (out = a @ b^T). persistent=True: a fixed grid of WGs strides
    the tile space via scf.for (cap_cu reserves CUs for comm overlap). persistent=False:
    one tile per WG + s_endpgm over-launch guard (full-device default, no tile-loop
    penalty). The per-tile body is the same for both modes (a free function so loaders
    aren't mis-collected as scf.for iter_args).

    ``num_xcd`` optionally remaps the global tile id (bijection over [0,total_tiles))
    so same-XCD WGs cluster on contiguous tiles for per-XCD L2 reuse; num_xcd<=1 =
    identity (plain row-major scan). ``group_m``/``group_n`` add the L2-reuse tile
    swizzle (see _grouped_block_mn)."""
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

    # CShuffle epilogue staging (8 waves x 16 rows x N_TILES_B*16 out_ty elems); used
    # only when store_cshuffle=True (vectorized 128b store vs scalar buffer_store_short).
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
    def kernel_grouped_nt_persistent(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,  # int32 [G+1]
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        # c_n stays runtime (a compile-time N folds the per-tile int-div to shifts but is
        # perf-neutral and bloats the compile cache per N).
        n_blocks = ceildiv(c_n, BLOCK_N)

        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))

        # total_tiles on-device (O(G) scan; no host read of group lens). The offsets are
        # re-scanned per tile (L1-cached) rather than hoisted: keeping ~2*(G+1) values
        # live across the persistent loop costs more occupancy than the re-scan saves.
        total_tiles = fx.Int32(0)
        prev_off = _load_go(go_div, 0)
        for g in range_constexpr(G):
            nxt_off = _load_go(go_div, g + 1)
            m_g = nxt_off - prev_off
            total_tiles = total_tiles + ceildiv(m_g, BLOCK_M) * n_blocks
            prev_off = nxt_off

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x
        nsms = fx.grid_dim.x  # persistent stride = number of launched WGs

        if const_expr(not persistent):
            # one tile per WG: pin total_tiles to SGPR and s_endpgm the over-launched WGs.
            total_tiles = _readfirstlane_i32(total_tiles)
            _llvm.inline_asm(
                None,
                [pid.ir_value(), arith._to_raw(total_tiles)],
                "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
                "s,s,~{scc},~{memory}",
                has_side_effects=True,
            )

        def _do_tile(t):
            # XCD remap of the tile id (bijection; identity when num_xcd<=1).
            tt = xcd_remap_pid(t, total_tiles, num_xcd)
            cum = fx.Int32(0)
            group_idx = fx.Int32(0)
            tile_start = fx.Int32(0)
            p2 = _load_go(go_div, 0)
            for g in range_constexpr(G):
                nx = _load_go(go_div, g + 1)
                mg = nx - p2
                tg = ceildiv(mg, BLOCK_M) * n_blocks
                nc = cum + tg
                inq = (tt >= cum) & (tt < nc)
                group_idx = arith.select(inq, fx.Int32(g), group_idx)
                tile_start = arith.select(inq, cum, tile_start)
                cum = nc
                p2 = nx

            m_start = _load_go(go_div, group_idx)
            m_end = _load_go(go_div, group_idx + 1)
            local = tt - tile_start
            # L2-reuse tile swizzle (group_n band -> group_m 1D -> row-major). B[g]'s
            # N-stripe stays L2-resident across the clustered tiles; the per-group
            # runtime guards degenerate to row-major for small/skewed groups.
            local_block_m, block_n = _grouped_block_mn(
                local, m_start, m_end, n_blocks, BLOCK_M, group_m, group_n
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
            A0_gl_offset = m_row * K
            A1_gl_offset = (m_row + LDS_BLOCK_M) * K
            # B_T is [G, N, K]; group base = group_idx*c_n*K, N-row = block_n*BLOCK_N,
            # each N-row is K-contiguous.
            B0_gl_offset = (group_idx * c_n + block_n * BLOCK_N) * K
            B1_gl_offset = (group_idx * c_n + block_n * BLOCK_N + LDS_BLOCK_N) * K

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

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
            if const_expr(store_cshuffle):
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
                )
            else:
                store_c = StoreCPerTensor(
                    A_scale, B_scale, C, m_end, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty
                )

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            # Inner per-mfma scheduling barrier; sched_schedbar=True swaps it for a
            # compile-time sched_barrier(0) (no runtime WG sync). Prologue/cross-iter/
            # epilog barriers stay real (LDS coop-load + ping-pong correctness).
            def _ibar():
                if const_expr(sched_schedbar):
                    rocdl.sched_barrier(0)
                else:
                    rocdl.s_barrier()

            # Prelude.
            b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)
            # persistent: unconditional barrier (cross-tile phase-correctness). 8w: one
            # tile per WG, so the dense divergent `if wave_m==1` barrier is correct.
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
                if nt_vmcnt >= 0:
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
                        constraints="",
                        has_side_effects=True,
                    )
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            # Epilog 1.
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

            # Epilog 2 (K-tail).
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
                _do_tile(t)
        else:
            _do_tile(pid)

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
        n_blocks = ceildiv(c_n, BLOCK_N)
        upper = (ceildiv(m_total, BLOCK_M) + G) * n_blocks
        ncus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        # cap_cu>0 reserves device CUs for comm-compute overlap: launch exactly
        # min(upper, cap_cu) persistent WGs so only cap_cu CUs run the GEMM and the
        # rest are free for the overlapped comm kernel. cap_cu<=0 = full device.
        _cap = ncus if cap_cu <= 0 else min(int(cap_cu), ncus)
        # persistent: cap to _cap WGs (reserve CUs). non-persistent: full upper-bound grid,
        # one tile per WG (over-launched WGs s_endpgm in-kernel).
        grid_x = arith.select(upper < _cap, upper, fx.Int32(_cap)) if persistent else upper
        attrs = make_value_attrs(waves_per_eu, 128 if (agpr_inplace and acc_mode == "agpr") else 0, "512,512")
        kernel_grouped_nt_persistent(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            group_offs,
            c_n,
            value_attrs=attrs,
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_nt_persistent


# ── wgrad: variable-K grouped GEMM (TN). C[g]=lhs_g^T@rhs_g; contraction m_g is
#    per-group runtime (scf.for K-loop). Accumulators in rmem (the loop carries no
#    objects); per-group K-tail clamp via the SRD num_records bound (over-read -> 0).


def _make_fp8_buf_nr(arg_i8, fp8_ir_t, num_records_bytes):
    """make_fp8_buffer_tensor with an explicit (runtime) num_records bound, so the
    buffer SRD clamps reads past the bound to 0 — used for the per-group A/B
    K-tail clamp (bound = m_end * OUT_{M,N}).

    num_records (= m_end*OUT_{M,N}) is wave-uniform in value, but the compiler treats
    m_end (from the per-lane group scan) as VGPR -> the SRD lands in VGPRs and every
    K-loop buffer_load gets a readfirstlane/saveexec waterfall. readfirstlane pins
    num_records to an SGPR so the SRD stays scalar."""
    num_records_bytes = _readfirstlane_i32(num_records_bytes)
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False, num_records_bytes=num_records_bytes)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


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
    b0 = b_s2r.load(b_cur0, drain=False)
    a0 = a_s2r.load(a_cur0)
    a_g2s.load(a_next1, A1_off + (k + 1) * AM)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a0, b0, acc00)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    b1 = b_s2r.load(b_cur1)
    b_g2s.load(b_cur0, B0_off + (k + 2) * BNs)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a0, b1, acc01)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    a1 = a_s2r.load(a_cur1)
    a_g2s.load(a_cur0, A0_off + (k + 2) * AM)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a1, b0, acc10)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    b_g2s.load(b_cur1, B1_off + (k + 2) * BNs)
    wait_barrier(2 * NA + NB)
    rocdl.s_setprio(1)
    _wgrad_accum(mfma, a1, b1, acc11)
    rocdl.s_setprio(0)
    rocdl.s_barrier()


def _band_block_mn(pid, num_pid_m, n_blocks, GM, GN):
    """2D super-block (band) tile swizzle for the wgrad per-group grid (port of
    dense TN _tn_block_mn). N split into width-GN bands, GROUP_M (GM) inside each →
    A reused GN×, B reused GM× → working set (GM·A_slab + GN·B_slab) stays L2-
    resident. Plain Python (trace-time), bijection over num_pid_m*n_blocks tiles.
    pid=local within-group tile id (runtime); num_pid_m/n_blocks/GM/GN compile-time."""
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


def _grouped_block_mn(local, m_start, m_end, n_blocks, block_m_size, group_m, group_n):
    """Map a within-group linear tile index ``local`` to (block_m, block_n) under the
    L2-reuse tile swizzle: group_n band (2D super-block) -> group_m 1D super-block ->
    row-major. The per-group runtime guards (bpr_g>group_m / n_blocks>group_n)
    degenerate to row-major for small/skewed groups so they can never corrupt tiny
    groups (skew-safe). Shared by the fwd (NT) and dgrad (NN) kernels, persistent and
    non-persistent. group_m/group_n are per-shape autotuned (the small-K L2 lever)."""
    lm_r = local // n_blocks
    bn_r = local % n_blocks
    if const_expr(group_n > 0 and group_m > 0):
        bpr_g = ceildiv(m_end - m_start, block_m_size)
        bm_b, bn_b = _band_block_mn(local, bpr_g, n_blocks, group_m, group_n)
        use_band = (bpr_g > fx.Int32(group_m)) & (fx.Int32(n_blocks) > fx.Int32(group_n))
        return arith.select(use_band, bm_b, lm_r), arith.select(use_band, bn_b, bn_r)
    elif const_expr(group_m > 0):
        GM_c = fx.Int32(group_m)
        bpr_g = ceildiv(m_end - m_start, block_m_size)
        npg = GM_c * n_blocks
        grp = local // npg
        first_m = grp * GM_c
        rem_m = bpr_g - first_m
        gsize_m = arith.select(rem_m < GM_c, rem_m, GM_c)
        in_grp = local % npg
        lm_g = first_m + (in_grp % gsize_m)
        bn_g = in_grp // gsize_m
        use_gm = bpr_g > GM_c
        return arith.select(use_gm, lm_g, lm_r), arith.select(use_gm, bn_g, bn_r)
    return lm_r, bn_r


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
    # ceildiv(k_iters,chunk) x inner range_constexpr(chunk) of the 4-buffer body; even
    # chunk resets the ping-pong at the boundary; over-run is SRD-clamped (no host cap).
):
    """Masked grouped TN wgrad: a CAPACITY-FREE chunked K-loop (outer runtime
    scf.for over ceildiv(k_iters,chunk) x inner range_constexpr(chunk) of the
    4-buffer inline body) instead of a plain scf.for, with the actual per-group
    contraction masked by the per-group SRD num_records clamp (over-read past the
    group's last token -> 0). The inner compile-time chunk recovers dense's cross-
    iteration software pipelining without a host-known token capacity. acc_mode
    picks the MFMA accumulator register class: "vgpr"=inline-asm mode 3 (=v,v,v,0
    in-place vacc, no accvgpr shuffle — usually fastest, mirrors the +9% scf.for
    win); "agpr"=mode 2 (=a,v,v,0, off-VGPR)."""
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
        group_idx = pid // TILES_PER_GROUP
        local = pid % TILES_PER_GROUP
        if const_expr(group_m > 0 and N_BLOCKS_M > group_m):
            GM_c = fx.Int32(group_m)
            npg = group_m * N_BLOCKS_N
            grp = local // npg
            first_m = grp * GM_c
            rem_m = fx.Int32(N_BLOCKS_M) - first_m
            gsize_m = arith.select(rem_m < GM_c, rem_m, GM_c)
            in_grp = local % npg
            block_m = first_m + (in_grp % gsize_m)
            block_n = in_grp // gsize_m
        else:
            block_m = local // N_BLOCKS_N
            block_n = local % N_BLOCKS_N

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

        a_nr = m_end * OUT_M
        b_nr = m_end * OUT_N
        gA = _make_fp8_buf_nr(A, F8_IR_t, a_nr)
        gB = _make_fp8_buf_nr(B, F8_IR_t, b_nr)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, OUT_M, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, OUT_N, N_LDS_ROUNDS)

        mfma = _build_mfma(N_TILES_A, N_TILES_B, cbsz, blgp, asm_mode="2" if _agpr else "3")

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
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
            )
        else:
            store_c = StoreCPerTensor(
                A_scale, B_scale, C, (group_idx + 1) * OUT_M, OUT_N, mfma.idx, N_TILES_A, N_TILES_B, _out_ty
            )

        A0_off = m_start * OUT_M + block_m * BLOCK_M
        A1_off = A0_off + LDS_BLOCK_M
        B0_off = m_start * OUT_N + block_n * BLOCK_N
        B1_off = B0_off + LDS_BLOCK_N
        AM = BLOCK_K * OUT_M
        BNs = BLOCK_K * OUT_N

        # Prelude (tile 0 -> cur, tile 1 -> next).
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

        # CAPACITY-FREE chunked path (CK-style hardware loop equivalent): runtime
        # k_iters, even-chunk inner unroll of the 4-buffer inline body; memref accs
        # survive the runtime scf.for; over-run (k>=k_iters) SRD-clamped to 0.
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


# Caches the compiled kernel per config key (not any result/quant/transpose) so it is
# reused across calls without re-tracing.
_GROUPED_LAUNCH_CACHE: dict = {}

# Baked production constant.
_GROUPED_AGPR = True  # AGPR in-place accumulation (off-VGPR, spill-free)


# ── Per-shape online autotune: on first call for a static (op,N,K,G,M_total,dtype)
#    shape, time a small candidate set on a balanced token distribution and cache the
#    winner. Keyed on static dims only (never per-group counts) -> transfers across steps.
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
):
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


def _balanced_targs(args, m_total, G):
    """args with the group_offs slot (index 5) replaced by a balanced m_total/G split,
    for distribution-independent autotune timing."""
    bal = _balanced_group_offs(m_total, G, args[2].device)
    return args[:5] + (bal,) + args[6:]


def _robust_time(launch, targs, warmup=250, reps=5, iters=50):
    """Median-of-`reps` timing of launch(*targs) after `warmup` iters (the long warmup
    reaches boost clock; short-K kernels mis-pick the config otherwise)."""
    for _ in range(warmup):
        launch(*targs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            launch(*targs)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    ts.sort()
    return ts[len(ts) // 2]


def _autotune_np_dispatch(trans_b, K, G, out_fp16, cbsz, blgp, args):
    """num_cu<=0 (full device): per-shape autotune the NON-PERSISTENT kernel's L2-reuse
    swizzle, timed on a BALANCED token distribution (see _balanced_group_offs) so the
    pick is distribution-independent. 3 candidates (band dropped — never adopted under
    balanced timing): base (8,4,0) = common winner + correctness reference; (1,0,0)
    row-major (num_xcd=1 wins some down-proj shapes); (8,8,0) wide M-cluster.
    >=1.5% hysteresis. Cached per shape."""
    out_view = args[2]
    # time on a balanced group_offs (args[6] = M_total) so a skewed first call cannot
    # bias the config pick.
    targs = _balanced_targs(args, args[6], G)

    def mk(xcd, gm, gn):
        if trans_b:  # NT: merged factory, non-persistent mode (intrinsic MMA, scalar store)
            return _compile_grouped_nt(
                K=K,
                G=G,
                BLOCK_M=256,
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
                nt_vmcnt=-1,
            )
        # NN: merged factory, non-persistent mode (AGPR in-place, scalar store).
        return _compile_grouped_nn(
            K=K,
            G=G,
            BLOCK_M=256,
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
        )

    base = mk(8, 4, 0)
    base(*targs)
    torch.cuda.synchronize()
    _r = out_view.detach().clone().float()
    _rn = float((_r * _r).sum().item()) or 1.0

    def _ok():
        o = out_view.detach().float()
        e = float(((o - _r) * (o - _r)).sum().item())
        return (e / _rn) < (2e-2**2) and torch.isfinite(o.view(-1)[:1024]).all().item()

    best, bt = base, _robust_time(base, targs)
    for xcd, gm, gn in ((1, 0, 0), (8, 8, 0)):
        l = mk(xcd, gm, gn)
        l(*targs)
        torch.cuda.synchronize()
        if not _ok():  # numeric guard: never adopt a config that drifts from the base
            continue
        t = _robust_time(l, targs)
        if t < bt * 0.985:  # adopt only if >=1.5% faster (robust timing -> reliable)
            best, bt = l, t
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
) -> "torch.Tensor":
    """FlyDSL per-tensor grouped fp8 GEMM (M-grouped), matching the Triton entry.

    out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g] * a_scale * b_scale
      trans_b=True  (forward): b [G, N, K] (b[g]^T); NT kernel.
      trans_b=False (dgrad)  : b [G, K, N];          NN kernel.
    a [M_total, K] fp8; a_scale/b_scale scalar fp32; group_offs [G+1] int.
    """
    assert a.ndim == 2 and b.ndim == 3
    M_total, K = a.shape
    G = b.shape[0]
    N = b.shape[1] if trans_b else b.shape[2]
    K_b = b.shape[2] if trans_b else b.shape[1]
    assert K == K_b, f"K mismatch a={K} b={K_b}"

    out = torch.empty((M_total, N), device=a.device, dtype=out_dtype)
    # kernel reads group_offs as int64 low-words via a free int32-view (no .to(int32)
    # cast); int32 callers are upcast to int64 once.
    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0

    grp_agpr = _GROUPED_AGPR
    nt_group_m = _GROUPED_NT_GROUPM  # 0 = row-major; the autotune sweeps group_m per shape
    op = "nt" if trans_b else "nn"
    # num_cu<=0: whole device via the NON-PERSISTENT nt8w/nn8w (one tile/WG, no scf.for
    # tile-loop penalty). num_cu>0: reserve CUs for comm overlap -> persistent (fixed
    # grid). M_total is in the key (an underfilled grid prefers a different config).
    capped = num_cu is not None and num_cu > 0
    nonpersist = not capped
    at_key = (op, N, K, G, out_fp16, cbsz, blgp, M_total, nonpersist, num_cu if capped else 0)
    a_i8 = a.view(torch.int8).reshape(-1)
    b_i8 = b.view(torch.int8).reshape(-1)
    args = (
        a_i8,
        b_i8,
        out.view(-1),
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
            # num_cu<=0 (full device): per-shape autotune the NON-PERSISTENT nt8w/nn8w
            # L2-reuse swizzle (3 candidates, balanced-timed). The straight-line one-
            # tile/WG body avoids the persistent scf.for tile-loop scheduling penalty.
            launch = _autotune_np_dispatch(trans_b, K, G, out_fp16, cbsz, blgp, args)
        else:
            # Single persistent prod config (xcd8/agpr/cshuffle/sched), NO autotune.
            # Reached only by capped (num_cu>0 -> reserve CUs for comm overlap, grid
            # capped to num_cu). The default (any dtype) goes to nt8w/nn8w.
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
            )
        entry = [launch, None]  # [raw @flyc.jit closure, flyc.compile'd object (lazy)]
        _GROUPED_AT_CACHE[at_key] = entry
    raw, compiled = entry
    # Mode-split: CUDA-graph capture uses the raw @flyc.jit closure (graph-friendly; a
    # flyc.compile'd object regresses under capture); eager uses a one-time flyc.compile'd
    # object that skips @flyc.jit's per-call drift-check + arg-hash dispatch overhead.
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            entry[1] = compiled
        compiled(*args)
    return out


# wgrad compilation cache: (OUT_M, OUT_N, G, out_fp16, cbsz, blgp) -> launch.
_GROUPED_WGRAD_LAUNCH_CACHE: dict = {}
# wgrad per-shape autotune cache (static-dim key -> winning launch).
_GROUPED_WGRAD_AT_CACHE: dict = {}


def _wgrad_masked_cfg(OUT_M, OUT_N, G, out_fp16, cbsz, blgp, chunk, group_m, num_xcd):
    """Compile (or cache-hit) the masked chunked wgrad for one (chunk, group_m, num_xcd)."""
    ck = (OUT_M, OUT_N, G, out_fp16, cbsz, blgp, chunk, group_m, num_xcd)
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
        )
        _GROUPED_WGRAD_LAUNCH_CACHE[ck] = l
    return l


def _autotune_wgrad_dispatch(OUT_M, OUT_N, G, out_fp16, cbsz, blgp, args, m_total):
    """Per-shape wgrad config select over the masked chunked kernel only, timed on a
    BALANCED token distribution (>=1.5% hysteresis). 3 candidates as (chunk, group_m,
    num_xcd): (8,4,8) = prod / most frequent winner; (8,0,8) wins big-OUT_M / square
    shapes; (4,4,8) wins short-contraction. The masked kernel matches/beats the old
    persistent scf.for kernel on every MoE shape (2026-06-14 sweep, worst 1.0%)."""
    out_view = args[2]
    # time on a balanced group_offs (m_total split over G) so a skewed call can't bias it.
    targs = _balanced_targs(args, m_total, G)

    def _M(chunk, group_m, num_xcd):
        return _wgrad_masked_cfg(OUT_M, OUT_N, G, out_fp16, cbsz, blgp, chunk, group_m, num_xcd)

    prod = _M(8, 4, 8)  # most frequent per-shape winner + correctness reference
    prod(*targs)
    torch.cuda.synchronize()
    if not torch.isfinite(out_view.view(-1)[:1024].float()).all().item():
        return prod  # numeric guard: prod produced NaN/Inf -> don't time alts
    best_l, best_t = prod, _robust_time(prod, targs)
    for chunk, group_m, num_xcd in ((8, 0, 8), (4, 4, 8)):
        l = _M(chunk, group_m, num_xcd)
        t = _robust_time(l, targs)
        if t < best_t * 0.985:  # hysteresis: adopt only if >=1.5% faster (robust timing)
            best_l, best_t = l, t
    return best_l


def grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel(
    lhs: "torch.Tensor",
    rhs: "torch.Tensor",
    lhs_scale: "torch.Tensor",
    rhs_scale: "torch.Tensor",
    group_offs: "torch.Tensor",
    out_dtype=torch.bfloat16,
    num_cu: "int | None" = -1,
) -> "torch.Tensor":
    """FlyDSL per-tensor variable-K grouped fp8 GEMM (wgrad), matching the
    Triton variable-K entry.

    C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]] * lhs_scale * rhs_scale
    lhs [M_total, OUT_M] fp8, rhs [M_total, OUT_N] fp8, out [G, OUT_M, OUT_N].
    lhs_scale/rhs_scale scalar fp32; group_offs [G+1] int. The caller (backend)
    has already applied the trans_c lhs/rhs swap.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == rhs.shape[0], f"M_total mismatch lhs={lhs.shape[0]} rhs={rhs.shape[0]}"
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    # kernel reads group_offs as int64 low-words via a free int32-view (no .to(int32) cast).
    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if lhs.dtype == torch.float8_e5m2 else 0
    blgp = 1 if rhs.dtype == torch.float8_e5m2 else 0

    lhs_i8 = lhs.view(torch.int8).reshape(-1)
    rhs_i8 = rhs.view(torch.int8).reshape(-1)
    lsf = lhs_scale.float().reshape(1)
    rsf = rhs_scale.float().reshape(1)
    stream = torch.cuda.current_stream()

    # NOTE: no cap-fed masked wgrad path — a general dropless operator must not assume
    # a host-fed per-expert token capacity. Only the capacity-free chunked-masked variant
    # (reads group_offs, SRD-clamped over-run) is used, as an autotune candidate below.

    # ── Per-shape online autotune (wgrad): time 3 masked configs on a balanced token
    #    distribution, cache the winner. Keyed on static dims (OUT_M,OUT_N,G,dtype,M_total);
    #    M_total is in the key because the best config depends on the contraction length.
    #    num_cu is ignored: the masked kernel uses a full G*tiles grid (it isn't
    #    persistent-strided), so it can't reserve CUs for comm-overlap.
    M_total = lhs.shape[0]
    at_key = (OUT_M, OUT_N, G, out_fp16, cbsz, blgp, M_total)
    wargs = (lhs_i8, rhs_i8, out.view(-1), lsf, rsf, go32, stream)
    launch = _GROUPED_WGRAD_AT_CACHE.get(at_key)
    if launch is None:
        launch = _autotune_wgrad_dispatch(OUT_M, OUT_N, G, out_fp16, cbsz, blgp, wargs, M_total)
        _GROUPED_WGRAD_AT_CACHE[at_key] = launch
    launch(*wargs)
    return out
