###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL).

This file owns the kernel definition end-to-end; it does NOT delegate to
`kernels.fp8_gemm_8wave` in the FlyDSL repo. The lower-level FlyDSL helpers
(G2SLoader, S2RLoader, Mfma16x16x128, swizzle math) ARE imported from
`kernels.fp8_gemm_utils` since those are reusable primitives, not the kernel
orchestration itself. The output store (_StoreCPerTensor) is defined here.

`@flyc.kernel` decorated functions must reference their dependencies as MODULE
globals (not as closure cells from an enclosing factory), so FlyDSL is imported
at module load time. FlyDSL is a hard dependency (see requirements.txt); a
missing install raises ImportError on import rather than degrading silently.

Algorithm: 8-wave (512 thread WG, wave_m=2 × wave_n=4), BLOCK_M=BLOCK_N=256,
BLOCK_K=128, LDS_BLOCK_M=LDS_BLOCK_N=128, 2×2 mma ping-pong (c00/c01/c10/c11
per wave), `mfma_f32_16x16x128_f8f6f4`. NN B-operand load uses
ds_read_b64_tr_b8 transpose-load (S2RLoaderTr); NT shares the FlyDSL repo's
S2RLoader. Includes the c10/c11 "stale a_cur1" pipeline fix in epilog 1.

Constraints:
  - K_ITERS = ceil(K/128) >= 2 (i.e. K >= 129); arbitrary K via the native K-tail
  - out_dtype in {bf16, fp16}; E4M3 / E5M2 / hybrid fp8 inputs
  - per-tensor scale  (a_scale / b_scale scalar fp32, broadcast inside wrapper)
  - NT, NN, TN native; TT not supported; trans_c via post-hoc transpose
"""

import functools
import os
import sys

import torch

# isort: off
# FlyDSL utils must be importable as module globals (@flyc.kernel needs them as
# globals, not closure cells). The `kernels` package is NOT shipped in the
# `flydsl` pip wheel (it lives at the FlyDSL repo root), so it is vendored as the
# `3rdparty/FlyDSL` submodule (pinned to the same tag as the pip `flydsl`); put
# that submodule root on sys.path before importing `kernels`. The compiled
# `flydsl` compiler itself still comes from the pip install.
_flydsl_3p_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "3rdparty", "FlyDSL")
)
if not os.path.isdir(os.path.join(_flydsl_3p_root, "kernels")):
    raise ImportError(
        f"FlyDSL submodule not found at {_flydsl_3p_root} (missing 'kernels/'). "
        "Run `git submodule update --init 3rdparty/FlyDSL` to fetch it."
    )
if _flydsl_3p_root not in sys.path:
    sys.path.insert(0, _flydsl_3p_root)

from kernels.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    swizzle_128,
    wait_barrier,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

# isort: on


class _StoreCPerTensor:
    """Per-tensor (scalar) scaled output store: out = (acc * a_scale * b_scale).to(out_ty).

    TENSORWISE scaling is a single scalar, so the two scales are read ONCE per
    store from length-1 buffers and applied uniformly -- no per-row/col broadcast
    (the wrapper passes the scalar scales directly, with no length-M/N buffer to
    materialize). out_ty selects bf16 / fp16 output, produced from the f32
    accumulator via a single ``.to(out_ty)`` (matches CK/Triton
    ``acc.to(out_dtype)``). Tile indexing + the OOB column clamp mirror FlyDSL's
    read-only StoreC; we don't subclass it because the scale handling differs.
    """

    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        c_nbytes = c_rows * c_cols * 2  # bf16 / fp16 output = 2 bytes
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)  # 1 fp32
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)  # 1 fp32
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _store_one(self, value, c_index):
        fx.memref_store_vec(Vec.filled(1, value, self.out_ty), self.reg_out_1)
        fx.copy(self.out_atom_1, self.reg_out_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div)
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = (vec_f32[i] * scale).to(self.out_ty)
                    c_index = (row + i) * self.c_cols + col
                    self._store_one(scaled, arith.select(col_valid, c_index, oob))


def _a_tail_mask_vec(lane_id, r):
    """Per-lane i32x8 bit-mask that zeroes A-fragment bytes whose K-column
    is >= r (the valid K-tail length, r in [1,128)).

    S2RLoader packs each A tile as pack_i32x4_i32x8(half0, half1):
      bytes [0:16]  = K[col0 .. col0+16)
      bytes [16:32] = K[col0+64 .. col0+80)   with col0 = (lane//16)*16.
    Byte j of word w is valid iff its K-column < r; invalid bytes -> 0x00.
    AND-ing this mask into the A frag makes the tail terms a_k=0, so the
    MFMA dot product Σ a_k·b_k drops k>=r regardless of B (only A masked).
    """
    col0 = (lane_id // 16) * 16  # runtime, in {0,16,32,48}
    words = []
    for w in range_constexpr(8):
        run_off = 0 if w < 4 else 64
        ww = w if w < 4 else w - 4
        base = col0 + (run_off + 4 * ww)  # K-column of byte 0 of this word
        word = fx.Int32(0)
        for b in range_constexpr(4):
            valid = (base + fx.Int32(b)) < fx.Int32(r)
            cval = 0xFF << (8 * b)
            if cval >= (1 << 31):
                cval -= 1 << 32  # signed two's-complement bit pattern
            word = word + arith.select(valid, fx.Int32(cval), fx.Int32(0))
        words.append(word)
    return Vec.from_elements(words, fx.Int32)


def _mask_a_tail(frag_list, lane_id, r):
    """Return A frags with the K-tail (>= r) zeroed; r%128==0 -> unchanged."""
    if r % 128 == 0:
        return frag_list
    mask = _a_tail_mask_vec(lane_id, r % 128)
    return [f & mask for f in frag_list]


def _make_value_attrs(waves_per_eu, agpr_alloc, fwg):
    """Build kernel value_attrs dict.

    agpr_alloc encoding:
      0       — no AGPR hint (compiler default)
      N > 0   — force EXACT N AGPRs (passthrough "N,N")
      -N      — allow up to N AGPRs (passthrough "0,N")
    """
    d = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": fwg}
    if agpr_alloc != 0:
        if agpr_alloc < 0:
            alloc = f"0,{-agpr_alloc}"
        else:
            alloc = f"{agpr_alloc},{agpr_alloc}"
        d["passthrough"] = [
            ["amdgpu-agpr-alloc", alloc],
            ["amdgpu-mfma-vgpr-form", "false"],
        ]
    return d


def _asm_mma_do(a, b, c, mode="2", cbsz=0, blgp=0):
    """fp8 16x16x128 MFMA via inline asm. Replaces `mma_atom_call_ssa`
    (FlyDSL core, untouched) to control the dst register class — the lever
    the intrinsic path can't express — and so kill the MFMA dst/srcA WAR
    hazard that makes barrier-reduced TN schedules nondeterministic.

    ``mode`` (from the `asm_mma` kernel param):
      1  `=&v,v,v,0`  VGPR accum + early-clobber: dst forced disjoint from
         srcA/srcB. Deterministic but the early-clobber also removes the
         scheduling freedom (and accum eats VGPR → low occupancy).
      2  `=a,v,v,0`   AGPR accum: dst/srcC in AGPR, srcA/srcB in VGPR —
         physically separate register files, so dst can NEVER alias srcA
         (det=0 by construction, no early-clobber needed → full scheduling
         freedom) AND the accumulators leave the VGPR file (higher occ).
    """
    v4f32 = ir.VectorType.get([4], ir.F32Type.get())
    cons = "=a,v,v,0" if str(mode) == "2" else "=&v,v,v,0"
    # cbsz/blgp select srcA/srcB fp8 format (0=E4M3, 1=E5M2). Omit when both
    # are E4M3 to keep the e4m3 asm byte-identical (determinism).
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    op = _llvm.InlineAsmOp(
        res=v4f32,
        operands_=[_raw(a), _raw(b), _raw(c)],
        asm_string=f"v_mfma_f32_16x16x128_f8f6f4 $0, $1, $2, $0{mods}",
        constraints=cons,
        has_side_effects=False,
    )
    return Vec(op.result)


def _xcd_remap_pid(pid, total_pids, num_xcd):
    """XCD-aware PID remap (plain-Python expr builder, like _tn_block_mn).

    HW round-robins consecutive workgroups across XCDs (physical XCD =
    pid % num_xcd), and each XCD owns a private L2. Re-gather all same-XCD
    WGs into one contiguous logical-pid block so the GROUP_M super-block
    swizzle's L2 reuse lands within a single XCD. Bijection for any
    total_pids: XCDs [0,rem) hold per_xcd+1 tiles, [rem,num_xcd) hold per_xcd.
    """
    if num_xcd <= 1:
        return pid
    per_xcd = total_pids // num_xcd  # floor
    rem = total_pids - per_xcd * num_xcd
    xcd = pid % num_xcd
    local = pid // num_xcd
    offset = xcd * per_xcd + arith.select(xcd < rem, xcd, rem)
    return offset + local


@functools.lru_cache(maxsize=256)
def _compile_dense_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 3,  # end-of-iter s_waitcnt vmcnt(N): N=3 → det=0 (gfx950 G2S buffer_load_lds/ds_read LDS hazard), <=1.1% cost; N>=4 races, N<3 costlier; -1 disables
    num_xcd: int = 8,  # XCD-aware PID remap: cluster same-XCD WGs into contiguous logical tiles for per-XCD L2 reuse (gfx950 MI355X = 8 XCD); 1 disables
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # _StoreCPerTensor out dtype: True -> fp16, else bf16
):
    """Build & cache the (K, BLOCK_M, BLOCK_N, GROUP_M)-specialised launch.

    ``GROUP_M`` enables Triton-style super-block tile-id swizzle for L2
    reuse: WGs advance ``block_m`` first within each ``GROUP_M × n_blocks``
    band. ``GROUP_M == 1`` = plain row-major scan.

    The main K-loop emits 7 barriers (before/after each MFMA); all are
    load-bearing — dropping any risks a compiler-reorder race.
    """
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert GROUP_M >= 1

    # Odd-K (native K-tail): run ceil(K/128) iters; the final iter is the
    # partial K-tail of length K_TAIL (0 => exact multiple, no masking).
    # The tail block's invalid K-columns are zeroed on the A operand in
    # Epilog 2 via _mask_a_tail (see helper). G2S over-reads of the tail are
    # harmless: interior rows read next-row bytes (masked away), the last
    # row clamps to 0 via the buffer SRD num_records bound.
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

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

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # NT semantics: A is [M, K] row-major K-contig.
        #               B_T is [N, K] row-major K-contig (= B^T storage of [K, N]).
        # Output       C is [M, N] row-major bf16.
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

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
        # Triton-style super-block swizzle for L2 reuse. Tail handled via
        # group_size_m = min(num_pid_m - first_pid_m, GROUP_M) so any
        # GROUP_M ≥ 1 is correct regardless of num_pid_m % GROUP_M.
        # arith.select used since flydsl lacks an integer minimum op.
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = _xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = _StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude: k=0 → cur, k=1 → next (a_next1 lazily on first main iter).
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main K-loop. Each iter: s2r {a0,b0,b1,a1} → 4 mma (c00→c01→c10→c11)
        # interleaved with k+1 (a_next1) and k+2 (a_cur0, b_cur0, b_cur1) prefetches.
        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

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
                )  # end-of-iter G2S drain (race fix)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1 (k = K_ITERS - 2). The a_g2s.load(a_next1, A1 + (k+1)*BLOCK_K)
        # line is the c10/c11 stale-a1 pipeline fix -- without it epilog-2's
        # a1_frag would read older K-iter data and the bottom half of every
        # output tile loses the final K-tile contribution.
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
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 (k = K_ITERS - 1) -- the K-tail block. Mask the A operand
        # so invalid K-columns (>= K_TAIL) contribute 0. No-op when K_TAIL==0.
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = _mask_a_tail(a0_frag, lane_id, K_TAIL)
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
        a1_frag = _mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale + store.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nt(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nt


# ──────────────────────────────────────────────────────────────────────


def _inttoptr_lds(byte_addr):
    """Convert an integer byte address to !llvm.ptr<3> (LDS pointer).

    Same shape as the mla kernel helper -- ir.Type.parse'd per call;
    the parsed Type belongs to the current MLIRContext, and caching
    across compiles fails verification on the 2nd shape.
    """
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


_gep = _buffer_ops.get_element_ptr


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return ptr


def _packed_ds_read_tr_offsets(base_ptr, byte_offsets, vmcnt_hint=None):
    """Pack len(byte_offsets) ds_read_b64_tr_b8 sharing ONE base ptr VGPR,
    each at a compile-time immediate byte offset (HK CRR
    load_col_from_st_half pattern: 1 addr + offset:N).

    Why 1 shared ptr (not N independent ptrs): the backend needs 1 addr
    VGPR instead of N, avoiding the spill cascade (2026-05-28: N-distinct-
    ptr packs gave spill 14->222 on TN BM=256; this form gives spill 8).
    For S2RLoaderTr the 4 ds_read of one tile split into 2 pairs (c0,c2)
    and (c1,c3), each pair differing by exactly r_step*8192 (verified:
    _K_BASE 0->64 / 8->72 bumps r_step by 1 while W/K_mod_8/swz_K/j_chunk
    stay identical), so the second read of each pair is base+offset:8192
    with NO runtime address component.

    `=&v` early-clobber + `~{memory}` clobber preserve wave-coop
    ds_read_b64_tr_b8 correctness (single-output asm permutes lanes).
    NOTE: ds_read_b64_tr_b8 is async (completes on lgkmcnt). The opaque asm
    hides this from the backend, so the *caller* (S2RLoaderTr.load) must
    emit a trailing `s_waitcnt lgkmcnt(0)` before the consuming mfma —
    otherwise the mfma races the LDS read (K-dependent garbage/NaN).

    Returns list of v2i32 (one per offset)."""
    N = len(byte_offsets)
    v2i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    struct_t = _llvm.StructType.get_literal([v2i32] * N)
    lines = []
    if vmcnt_hint is not None and vmcnt_hint >= 0:
        lines.append(f"s_waitcnt vmcnt({vmcnt_hint})")
    for k in range(N):
        # ${N} is the single shared input ptr (after N outputs $0..$N-1).
        lines.append(f"ds_read_b64_tr_b8 ${k}, ${N} offset:{byte_offsets[k]}")
    asm = "\n".join(lines)
    constraints = ",".join(["=&v"] * N + ["v"] + ["~{memory}"])
    asm_op = _llvm.InlineAsmOp(
        res=struct_t,
        operands_=[_raw(base_ptr)],
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )
    return [_llvm.extractvalue(v2i32, asm_op.result, [k]) for k in range(N)]


class _G2SLoaderStride(G2SLoader):
    """G2SLoader with a tunable per-wave LDS chunk stride (default 1024 =
    identical to base). A stride of 1056 (=1024+32) makes the per-wave chunk
    base NOT bank-aligned on gfx950: W*1056/4 % 64 = W*8, so the wave-step
    dimension W spreads across banks instead of all collapsing to bank 0.
    This halves the TN transpose-read bank conflict (4-way -> 2-way), since
    the 4 lane-groups (lane//16, i.e. W=0,2,4,6) then hit 4 distinct banks.
    The within-chunk data layout (lane*16 from the copy atom) is unchanged;
    only the chunk base is padded by 32 bytes. Read side (S2RLoaderTr*) must
    use the SAME chunk_stride."""

    def __init__(self, *a, chunk_stride=1024, **kw):
        super().__init__(*a, **kw)
        self.chunk_stride = chunk_stride

    def _lds_dst_at(self, lds_dst, step):
        cs = self.chunk_stride
        step_off = self.wave_id * cs + step * (self.n_waves * cs)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def compute_global_swizzle_nn(lane_id, wave_id, N_out, n_rounds):
    """B in NN: [K_inner, N_out] row-major. Each round loads 64 K-rows ×
    128 N-bytes (per wave 8 K-rows × 128 N-bytes via BufferCopyLDS128b).
    Per-round per-lane offset = swizzle_128(k_row, n_col_byte_base) over
    the [K, N] flat byte view with N_out element stride.
    """
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        k_row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        n_col = (lane_id % 8) * 16
        rs, cs = swizzle_128(k_row, n_col)
        offsets.append(rs * N_out + cs)
    return offsets


class S2RLoaderTr:
    """LDS -> mfma B-operand transpose-load for NN-layout B (Session 3,
    probe-verified 2026-05-25, see scripts2/probe_transpose_nn_lds.py and
    scripts2/NN_TRANSPOSE_DESIGN.md).

    LDS layout (from G2SLoader + BufferCopyLDS128b + swizzle_128, written
    into a [LDS_BLOCK_N=128, BLOCK_K=128]-sized region) is NOT the natural
    ``LDS[K*128 + N]`` row-major layout assumed by an earlier version of
    this class. The actual per-wave 1024-byte block layout is:
        LDS[W*1024 + step*8192 + L*16 + n_byte]
          = B[K=L//8 + W*8 + step*64, N=(L%8)*16 XOR swz_K + n_byte]
    where W ∈ [0,8) is the writing wave_id, step ∈ {0,1} the LDS load
    round, L ∈ [0,64) the lane within that wave, n_byte ∈ [0,16) the byte
    index within the 16-byte BufferCopyLDS128b chunk, and
        swz_K = ((K % 16) // 2) * 16        (swizzle_128 XOR factor).

    We issue 4 ds_read_tr8_b64 per (lane, mfma_tile) with per-lane ptrs
    derived to match the mfma B operand layout (Session 1 ground truth:
    for B operand of mfma_f32_16x16x128_f8f6f4, lane L wants byte b ∈
    [0,32) to be B[K = (L//16)*16 + (b<16 ? b : 64+b-16),
    N = wave_n*32 + tile_i*16 + L%16]).

    Per-lane ptr formula (I = L//16, L_in_sg = L%16, c ∈ {0,1,2,3},
    K_BASE = [0, 8, 64, 72]):
        K_log = I*16 + K_BASE[c] + L_in_sg//2          ∈ [0, 128)
        r_step = K_log // 64                            ∈ {0, 1}
        W      = (K_log % 64) // 8                     ∈ [0, 8)
        K_mod_8 = K_log % 8                            ∈ [0, 8)
        swz_K  = ((K_log % 16) // 2) * 16              ∈ {0,16,...,112}
        tile_N_start = wave_n*32 + tile_i*16
        j_chunk = (tile_N_start // 16) ^ (swz_K // 16) # un-swizzle N chunk
        ptr = LDS + W*1024 + r_step*8192 + K_mod_8*128
                  + j_chunk*16 + (L_in_sg%2)*8
    """

    _K_BASE = (0, 8, 64, 72)
    # NOTE: MLIR types are context-scoped; caching TR_TYPE as a class
    # attribute fails on the 2nd compile (different K) because the cached
    # i32 type belongs to the prior MLIR context and Numeric.from_ir_type
    # can't look it up in the new context's map. Always rebuild per-call.

    def __init__(self, wave_n, n_tiles_b, inline_asm=False, vmcnt_hint=2, chunk_stride=1024):
        """``inline_asm=True`` switches ds_read_tr8_b64 from rocdl
        intrinsic to inline-asm block. backend SIInsertWaitcnts treats
        inline asm as opaque (not known wave-coop op) so does NOT
        auto-emit `vmcnt(0)` drain before each tr8 group. Caller-supplied
        `vmcnt_hint` is emitted once per tile_i (before first of 4 ds_read)
        to ensure LDS data sync. This eliminates 446/447 vmcnt(0) sites
        per K=28672 kernel, gaining ~+22% TFLOPS on big K-shapes.

        ``vmcnt_hint=2`` is the validated safe value (1000-run det check
        passed on 5 critical shapes incl. (8192,8192,28672)). vmcnt(6)
        shows occasional race on K=28672. Larger vmcnt = looser sync =
        faster but unsafe; smaller = stricter = closer to baseline drain.

        ⚠ When inline_asm=True, the kernel MUST set agpr_alloc>0 (any
        nonzero pin); compiler-decide AGPR (=0) conflicts with inline
        asm `=v` constraint and produces nan output. See path J debug.
        """
        self.wave_n = wave_n
        self.n_tiles_b = n_tiles_b
        self.lane_id = fx.thread_idx.x % 64
        self.inline_asm = inline_asm
        self.vmcnt_hint = vmcnt_hint
        # Per-wave LDS chunk stride (bank-spread; must match G2S writer).
        # round_stride = n_waves(8) * chunk_stride is the r_step K-sub-round
        # jump (1024->8192 default; 1056->8448 bank-spread).
        self.chunk_stride = chunk_stride
        self.round_stride = 8 * chunk_stride

    def load(self, lds_src, preshuffled=False, drain=True):
        """Returns list[N_TILES_B] of i32x8 (32 fp8/lane K-contig at N-col).
        inline_asm path emits a single trailing separate lgkmcnt(0) drain;
        correct because mma.call follows at a function-call boundary the
        backend doesn't reorder across.
        ``drain=False`` skips the trailing lgkmcnt(0) — caller guarantees a
        downstream full lgkm drain (e.g. the immediately-following A load)
        covers these reads before the consuming mfma."""
        assert not preshuffled, "S2RLoaderTr does not support preshuffled"
        tr_type = Vec.make_type(2, fx.Int32)

        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        I = self.lane_id // 16
        L_in_sg = self.lane_id % 16

        # 1-ptr + offset:8192 packing (mirror HK CRR load_col_from_st_half).
        # The 4 ds_read of a tile pair as (c0,c2) and (c1,c3): within each
        # pair the only formula difference is _K_BASE 0->64 (or 8->72),
        # which bumps r_step by exactly 1 (=8192 bytes) and leaves every
        # other term identical. So the 2nd read of each pair is just
        # base+offset:8192 — no runtime address, single shared input ptr.
        def _ptr_off_b(c, tile_i):
            K_log = I * 16 + S2RLoaderTr._K_BASE[c] + (L_in_sg // 2)
            r_step = K_log // 64
            W = (K_log % 64) // 8
            K_mod_8 = K_log % 8
            swz_K = ((K_log % 16) // 2) * 16
            tile_N_start = self.wave_n * 32 + tile_i * 16
            j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
            return (
                W * self.chunk_stride
                + r_step * self.round_stride
                + K_mod_8 * 128
                + j_chunk * 16
                + (L_in_sg % 2) * 8
            )

        RS = self.round_stride  # c0->c2 / c1->c3 packed-read jump (r_step+1)
        frag = []
        for tile_i in range_constexpr(self.n_tiles_b):
            if self.inline_asm:
                p0 = _lds_ptr_from_i32(base_i32 + fx.Int32(_ptr_off_b(0, tile_i)))
                p1 = _lds_ptr_from_i32(base_i32 + fx.Int32(_ptr_off_b(1, tile_i)))
                r02 = _packed_ds_read_tr_offsets(p0, [0, RS], vmcnt_hint=self.vmcnt_hint)
                r13 = _packed_ds_read_tr_offsets(p1, [0, RS], vmcnt_hint=None)
                # r02 = [c0, c2], r13 = [c1, c3] → reorder to [c0,c1,c2,c3]
                calls = [Vec(r02[0]), Vec(r13[0]), Vec(r02[1]), Vec(r13[1])]
            else:
                calls = [
                    Vec(
                        rocdl.ds_read_tr8_b64(
                            tr_type, _lds_ptr_from_i32(base_i32 + fx.Int32(_ptr_off_b(c, tile_i)))
                        ).result
                    )
                    for c in range_constexpr(4)
                ]

            # Concat 4 × v2i32 → v8i32. Byte order matches mfma B-operand
            # bytes 0..31 for lane L:
            #   v8i32[0..1] = call 0 → K = (L//16)*16 + L%2*4 + ...   (bytes 0..7)
            #   v8i32[2..3] = call 1 → bytes 8..15
            #   v8i32[4..5] = call 2 → bytes 16..23 (K + 64 jump)
            #   v8i32[6..7] = call 3 → bytes 24..31
            v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
            v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
            frag.append(v4_lo.shuffle(v4_hi, list(range(8))))
        if self.inline_asm and drain:
            # ds_read_b64_tr_b8 completes on lgkmcnt (async LDS read); the
            # opaque asm hides this so the backend won't auto-insert the
            # wait the mfma needs. Single trailing drain (mma.call follows
            # at a call boundary the backend doesn't reorder across).
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
        return frag


class S2RLoaderTr_A:
    """LDS → mfma A-operand wave-coop tr8 load (for TN-layout fp8 GEMM).
    Per HK fp8 CRR (kernel_fp8_layouts.cpp `crr_mma`), mfma A and B
    operand register byte layout is IDENTICAL — both call same
    v_mfma_f32_16x16x128_f8f6f4. HK does
    `reinterpret_cast<A_row_reg>(a_col_reg)` to use B-style transpose
    fragment as A-op. We replicate via per-wave_m S2RLoaderTr variant.
    """

    _K_BASE = (0, 8, 64, 72)

    def __init__(
        self,
        wave_m,
        n_tiles_a,
        lds_block_m,
        inline_asm=False,
        vmcnt_hint=2,
        chunk_stride=1024,
        n_waves=8,
        wave_stride=None,
    ):
        """``lds_block_m`` = BLOCK_M // 2 (LDS half-block M-size). Per-wave
        M coverage = lds_block_m / num_wave_m. For 8w with num_wave_m=2:
        BM=256 → lds_block_m=128 → stride=64; BM=128 → lds_block_m=64 → stride=32.
        ``n_waves`` = WG wave count (8 default; 16 for the 1024-thread layout
        where one K-round fills 128 K = 16 waves × 8). ``wave_stride`` overrides
        the M sub-half stride (16w: each wave owns 64 rows → stride=64)."""
        self.wave_m = wave_m
        self.n_tiles_a = n_tiles_a
        self.wave_stride = (lds_block_m // 2) if wave_stride is None else wave_stride
        self.lane_id = fx.thread_idx.x % 64
        self.inline_asm = inline_asm
        self.vmcnt_hint = vmcnt_hint
        self.chunk_stride = chunk_stride  # bank-spread LDS chunk stride
        self.n_waves = n_waves
        self.round_stride = n_waves * chunk_stride  # r_step K-sub-round jump

    def _ptr_off_a(self, c, tile_i, I, L_in_sg):
        K_log = I * 16 + S2RLoaderTr_A._K_BASE[c] + (L_in_sg // 2)
        KW = self.n_waves * 8
        r_step = K_log // KW
        W = (K_log % KW) // 8
        K_mod_8 = K_log % 8
        swz_K = ((K_log % 16) // 2) * 16
        # 8w A: wave_m * wave_stride (= LDS_BLOCK_M / num_wave_m)
        tile_M_start = self.wave_m * self.wave_stride + tile_i * 16
        j_chunk = (tile_M_start // 16) ^ (swz_K // 16)
        return (
            W * self.chunk_stride
            + r_step * self.round_stride
            + K_mod_8 * 128
            + j_chunk * 16
            + (L_in_sg % 2) * 8
        )

    def _issue_one(self, lds_src, tile_i):
        """Issue the 4 ds_read_b64_tr_b8 of one A tile WITHOUT draining
        lgkmcnt or assembling the frag. Returns the 4 raw v2i32 Vec; the
        caller (load_one) drains lgkmcnt then assembles."""
        tr_type = Vec.make_type(2, fx.Int32)
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        I = self.lane_id // 16
        L_in_sg = self.lane_id % 16
        RS = self.round_stride
        if self.inline_asm:
            p0 = _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off_a(0, tile_i, I, L_in_sg)))
            p1 = _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off_a(1, tile_i, I, L_in_sg)))
            r02 = _packed_ds_read_tr_offsets(p0, [0, RS], vmcnt_hint=self.vmcnt_hint)
            r13 = _packed_ds_read_tr_offsets(p1, [0, RS], vmcnt_hint=None)
            return [Vec(r02[0]), Vec(r13[0]), Vec(r02[1]), Vec(r13[1])]
        return [
            Vec(
                rocdl.ds_read_tr8_b64(
                    tr_type,
                    _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off_a(c, tile_i, I, L_in_sg))),
                ).result
            )
            for c in range_constexpr(4)
        ]

    @staticmethod
    def _assemble(calls):
        v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
        v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
        return v4_lo.shuffle(v4_hi, list(range(8)))

    @staticmethod
    def _wait_lgkmcnt(n):
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string=f"s_waitcnt lgkmcnt({n})",
            constraints="",
            has_side_effects=True,
        )

    def load_one(self, lds_src, tile_i):
        """Single-A-tile wave-coop tr8 load → one i32x8 frag (caps peak
        A-fragment liveness at 1 tile for the inline path)."""
        calls = self._issue_one(lds_src, tile_i)
        frag = self._assemble(calls)
        if self.inline_asm:
            self._wait_lgkmcnt(0)
        return frag

    def load(self, lds_src, preshuffled=False):
        assert not preshuffled
        # Issue all n_tiles_a tiles' ds_read, then one trailing lgkmcnt(0)
        # (as the B loader does). A per-tile drain serializes each tile's
        # reads before the next and blocks ds_read pipelining; batching
        # keeps the read->mfma dependency (one drain before the consuming
        # mma) while letting the loads overlap.
        if self.inline_asm:
            all_calls = [self._issue_one(lds_src, tile_i) for tile_i in range_constexpr(self.n_tiles_a)]
            self._wait_lgkmcnt(0)
            return [self._assemble(c) for c in all_calls]
        return [self.load_one(lds_src, tile_i) for tile_i in range_constexpr(self.n_tiles_a)]


@functools.lru_cache(maxsize=128)
def _compile_dense_nn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    num_xcd: int = 8,  # XCD-aware PID remap for per-XCD L2 reuse (MI355X = 8 XCD); 1 disables. See _xcd_remap_pid.
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    # path J: emit ds_read_tr8_b64 as inline asm so the backend treats it as
    # opaque and skips the auto vmcnt(0) drain; vmcnt_hint supplies the LDS
    # sync. MUST set agpr_alloc>0 (AGPR=0 + inline-asm =v constraint → nan).
    b_inline_asm_load: bool = False,
    vmcnt_hint: int = 2,
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # _StoreCPerTensor out dtype: True -> fp16, else bf16
):
    """NN-layout fp8 dense kernel. A [M, K], B [K, N], C [M, N].

    ``agpr_alloc`` / ``waves_per_eu`` mirror the NT kernel's knobs; see
    ``_make_value_attrs`` for ``agpr_alloc`` encoding (N>0 = exact N AGPRs,
    -N = up to N, 0 = compiler default)."""
    if b_inline_asm_load and agpr_alloc == 0:
        raise ValueError(
            "b_inline_asm_load=True requires agpr_alloc > 0. AGPR=0 "
            "(compiler-decide) combined with inline-asm `=v` constraint "
            "produces nan output. Pin AGPR to 16/32/48 (32 is the "
            "validated default)."
        )
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters; final iter masked on A (see NT note).
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K  # same byte count as NT, different layout

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

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Workaround: forces materialization of thread_idx.x in trace IR
        # before downstream S2RLoaderTr lazy-evaluates it inside
        # range_constexpr loops. Without this, flydsl emits an
        # out-of-order load schedule for ds_read_tr8_b64 that yields
        # garbage output (C-SNR ≈ -2.4 dB). Probe-verified ptr formula is
        # correct; bug is in flydsl's trace order, not our kernel.
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

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
        # Triton-style super-block swizzle for L2 reuse. Tail handling
        # identical to NT — group_size_m = min(remaining_m, GROUP_M) so any
        # GROUP_M is correct regardless of num_pid_m % GROUP_M. Without this
        # tail clamp, GM > num_pid_m emits block_m values past the row tile
        # bound, leaving most valid (m, n) tiles uncovered (SNR ≈ -50 dB).
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = _xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

        # A: same as NT.
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K

        # B: NN-specific. B is [K, N] row-major; per WG we load BLOCK_K K-rows
        # × BLOCK_N N-cols, split into 2 N-halves of LDS_BLOCK_N each. K-iter
        # step advances K-rows by BLOCK_K, which in element units is BLOCK_K * c_n.
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            # E5M2 / hybrid: rebuild the MFMA atom with per-operand fp8 fmt
            # (cbsz->srcA, blgp->srcB). Same instruction family / frag layout
            # as the default e4m3 atom, so loaders are unchanged.
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(wave_n, N_TILES_B, inline_asm=b_inline_asm_load, vmcnt_hint=vmcnt_hint)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = _StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main loop. Emits 7 barriers per K-iter (before/after each MFMA);
        # all are load-bearing — dropping any risks a compiler-reorder race.
        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

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
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 -- K-tail block. Mask A so K-cols >= K_TAIL contribute 0.
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = _mask_a_tail(a0_frag, lane_id, K_TAIL)
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
        a1_frag = _mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nn


@functools.lru_cache(maxsize=128)
def _compile_dense_tn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    waves_per_eu: int = 2,
    vmcnt_hint: int = 3,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D band (width group_n)
    num_xcd: int = 8,  # XCD-aware PID remap for per-XCD L2 reuse (MI355X = 8 XCD); 1 disables. See _xcd_remap_pid.
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # _StoreCPerTensor out dtype: True -> fp16, else bf16
):
    """TN-layout fp8 dense kernel: A [K, M], B [K, N], C [M, N] = A^T @ B.
    Both A and B are K-row strided → wave-coop ds_read_b64_tr_b8 on both
    sides (= NN B path x2; HK CRR shows the mfma A/B operand byte layout is
    identical, so S2RLoaderTr_A feeds the A operand directly). Path-J
    inline-asm tr8 on both operands + asm-inplace MFMA (=a,v,v,0; D aliases C
    in AGPR -> accumulators spill-free, no per-K-iter A-side vmcnt(0) drain)."""
    _a_inline = True
    _b_inline = True
    _asm_mma_mode = "2"  # asm-inplace MFMA (accum in AGPR)
    _inplace = True
    agpr_alloc = 128
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters. No A-mask needed here -- TN's A [K,M]
    # and B [K,N] are K-row-major, so the tail's invalid K-rows are fully out
    # of bounds and clamp to 0 via the buffer SRD num_records bound.
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    # TN uses wave-coop tr8 for A path; S2RLoaderTr_A formula reads
    # K_log ∈ [0, 128) requiring 2 G2S rounds = 16K LDS slot.
    # For BM=128 (natural N_LDS_STEPS_A=1, 8K slot), force to 2 rounds
    # and 16K slot to match S2RLoaderTr_A K=128 expectation.
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)  # ≥ 2 for tr8 K=128
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    # Bank-spread LDS chunk stride (gfx950 bank-conflict fix). 1024 = off
    # (bank-0-aligned per-wave chunk → transpose-read bank conflict). 1056
    # = +32B pad → W*1056/4%64 = W*8 spreads the wave-step dim across banks,
    # eliminating LDS bank conflicts. Load-bearing constant; both the G2S
    # writer (_G2SLoaderStride) and S2R reader use it.
    _LDS_CS = 1056
    # a_lds_size: N rounds × 8 waves × chunk_stride. Pad to stride.
    a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
    b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

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

    def _tn_block_mn(pid, num_pid_m, n_blocks, GM, GN):
        """Tile-id -> (block_m, block_n). Plain Python so the GN>0 branch is
        resolved at trace time (not as a kernel `if`). GN==0: 1D GROUP_M
        super-row swizzle (block_m inner, all-N sweep). GN>0: 2D band — N
        split into width-GN bands, GROUP_M 1D inside each band. Both dims
        blocked → A[block_m] reused GN×, B[block_n] reused GROUP_M× so the
        working set (GROUP_M·A_slab + GN·B_slab) stays L2-resident, cutting
        the big-N B re-stream. Always a bijection (full bands take
        num_pid_m·GN pids; remainder = narrower last band). det-neutral."""
        if GN > 0:
            band_tiles = num_pid_m * GN
            band = pid // band_tiles
            pid_in_band = pid % band_tiles
            band_n0 = band * GN
            rem_n = n_blocks - band_n0
            band_w = arith.select(rem_n < GN, rem_n, fx.Int32(GN))
            nig = GM * band_w
            gid = pid_in_band // nig
            pig = pid_in_band % nig
            fpm = gid * GM
            rem_m = num_pid_m - fpm
            gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
            return fpm + (pig % gsm), band_n0 + (pig // gsm)
        nig = GM * n_blocks
        gid = pid // nig
        pig = pid % nig
        fpm = gid * GM
        rem_m = num_pid_m - fpm
        gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
        return fpm + (pig % gsm), pig // gsm

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4

        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = _xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        # Swizzle via plain-Python helper (NOT a kernel `if`: @flyc.kernel
        # wraps each if-branch in its own fn so vars defined inside aren't
        # visible after — see prelude note). Helper builds the expr graph
        # for one Python-selected path (1D GROUP_M or 2D band).
        block_m, block_n = _tn_block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # TN A stored [K, M] row-major: stride M per K-row.
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M

        # B same as NN: stored [K, N] row-major.
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Both A+B use NN-style K-strided global swizzle.
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if _inplace:
            _mm = _asm_mma_mode
            mfma._do_mma = lambda _a, _b, _c, _m=_mm: _asm_mma_do(_a, _b, _c, mode=_m, cbsz=cbsz, blgp=blgp)

        a_g2s = _G2SLoaderStride(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        b_g2s = _G2SLoaderStride(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        a_s2r = S2RLoaderTr_A(
            wave_m,
            N_TILES_A,
            LDS_BLOCK_M,
            inline_asm=_a_inline,
            vmcnt_hint=vmcnt_hint,
            chunk_stride=_LDS_CS,
        )
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, inline_asm=_b_inline, vmcnt_hint=vmcnt_hint, chunk_stride=_LDS_CS
        )
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = _StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * c_m)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * c_m)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K * c_m)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Steady loop: per-iter A-half-0/A-half-1 × {b0,b1} MMA interleaved
        # with the next-tile G2S prefetch and one s_barrier per MMA quadrant.
        # All 7 barriers are load-bearing (dropping any races at the
        # MFMA-reorder level under some GROUP_M; gated by long det runs).
        for k in range_constexpr(K_ITERS - 2):
            # b0 drain=False: the b0 reads are covered by the immediately-
            # following a0 load's lgkmcnt(0) before c00 consumes b0, so the
            # b0 loader's own trailing drain is redundant. (b1 keeps its
            # drain — c01 consumes b1 with no covering drain between.)
            b0_frag = b_s2r.load(b_cur0, drain=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
            rocdl.s_barrier()
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
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2.
        a0_frag = a_s2r.load(a_cur0)
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
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_tn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_tn


# NN 4-wave kernel removed (consistently slower than 8w; HW caps 1
# wave/SIMD for this layout).


_COMPILED_DENSE_CACHE: dict = {}


def _get_compiled_dense(launch, args):
    """Cache compiled launcher by (shape, dtype, int-arg) tuple."""
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    # Zero-copy flat byte view. Recomputed every call (no id()-keyed cache: a
    # freed tensor's id + data_ptr can both be reused, and a recycled pair with a
    # different numel would alias the wrong length). The view ops are ~1us and
    # allocate nothing.
    if t.element_size() == 1 and t.dtype != torch.int8:  # fp8
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tensorwise scalar -> a length-1 fp32 buffer (no broadcast).

    The kernel applies it per-tensor (StoreCPerTensor reads the single value and
    multiplies uniformly), so there is no per-row/col vector to materialize --
    just an fp32/device cast (a no-op when the input already matches), avoiding
    the length-M/N copy kernel the old broadcast incurred every call.
    """
    assert scale.numel() == 1, f"per-tensor expects scalar, got {scale.shape}"
    return scale.to(dtype=torch.float32, device=device).reshape(1)


# NN per-shape autotune. GROUP_M/num_xcd fixed at the analytic L2-optimum
# (4, 8) for the same reason as NT (L2 effects the hot bench can't measure);
# only BLOCK_M and AGPR are swept. AGPR must be nonzero (path-J
# ds_read_b64_tr_b8 produces nan otherwise).
_NN_CANDIDATES = [
    (256, 4, 8, 32),
    (256, 4, 8, 64),
    (128, 4, 8, 48),
]
_NN_AUTOTUNE_CACHE: dict = {}


def _autotune_nn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NN candidates, cache best (launch, cfg) by (M,N,K).

    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate,
    finite-checks the output, times 2-warmup + 20-iter, and caches the
    fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NN_AUTOTUNE_CACHE:
        return _NN_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NN_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (_StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            # path J: inline-asm ds_read_b64_tr_b8 ON by default. Eliminates
            # 446/447 compiler-auto vmcnt(0) drain per K-iter.
            launch = _compile_dense_nn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                num_xcd=xcd,
                agpr_alloc=ag,
                b_inline_asm_load=True,
                vmcnt_hint=2,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NN autotune found no working cfg for ({M},{N},{K})")
    _NN_AUTOTUNE_CACHE[key] = best
    return best


# NT per-shape autotune. Format: (BLOCK_M, GROUP_M, num_xcd, AGPR).
# GROUP_M and num_xcd are FIXED at the analytic L2-optimum (4, 8): they are
# per-XCD L2-reuse effects the hot-cache bench cannot measure, so benching them
# just picks by noise (and was dropping B-streaming shapes ~5% whenever it
# mis-picked num_xcd=1). Only BLOCK_M and AGPR are swept -- both are
# occupancy/compute effects the hot bench measures reliably (BLOCK_M=128 wins
# tiny grids and loses big-M; AGPR trades occupancy).
_NT_CANDIDATES = [
    (256, 4, 8, 64),
    (256, 4, 8, 32),
    (128, 4, 8, 48),
    (128, 4, 8, 32),
]
_NT_AUTOTUNE_CACHE: dict = {}


def _autotune_nt_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NT candidates, cache best (launch, cfg) by (M,N,K).

    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate,
    finite-checks the output, times 2-warmup + 20-iter, and caches the
    fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NT_AUTOTUNE_CACHE:
        return _NT_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NT_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (_StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            launch = _compile_dense_nt(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                agpr_alloc=ag,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NT autotune found no working cfg for ({M},{N},{K})")
    _NT_AUTOTUNE_CACHE[key] = best
    return best


# TN dispatch: a single inplace-A kernel (inline-asm tr8 on both operands +
# asm_mma=2 → accumulators aliased into AGPR, spill-free, no per-K-iter A-side
# vmcnt(0) drain). Same 1D GROUP_M=4 + XCD-aware PID remap as NT/NN; only the
# num_xcd on/off is benched per shape (L2-resident shapes pick num_xcd=1).


_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_tn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K).

    1D GROUP_M=4 with num_xcd 8 vs 1 (XCD-aware PID remap); large
    (HBM-streaming) shapes expose the per-XCD L2 reuse on the hot bench,
    L2-resident shapes pick num_xcd=1.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _TN_AUTOTUNE_CACHE:
        return _TN_AUTOTUNE_CACHE[key]
    # Occupancy routing: BLOCK_M=BLOCK_N=256 yields ceil(M/256)*ceil(N/256)
    # tiles; below NUM_CUS the grid can't fill every CU, so BLOCK_M=128 doubles
    # the M-tile count. Above it the smaller block's per-tile overhead dominates.
    NUM_CUS = 256
    tiles_256 = ((M + 255) // 256) * ((N + 255) // 256)
    bm = 128 if tiles_256 < NUM_CUS else 256
    out_view = args[2]
    best_us = float("inf")
    best = None
    for xcd in (8, 1):
        try:
            launch = _compile_dense_tn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=4,
                vmcnt_hint=3,
                group_n=0,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, 4, 0, xcd))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"TN autotune found no working cfg for ({M},{N},{K})")
    _TN_AUTOTUNE_CACHE[key] = best
    return best


def gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Dense FP8 GEMM (per-tensor scaling).

    Inputs may be E4M3, E5M2 or hybrid (per-operand fp8 format threaded into the
    MFMA); out_dtype may be bf16 or fp16; contraction K is arbitrary (native
    K-tail). Layout dispatch:
      - NT (trans_a=F, trans_b=T): native NT kernel, no host transpose.
      - NN (trans_a=F, trans_b=F): native NN kernel (used by dgrad path), no
        host transpose; uses ds_read_b64_tr_b8 for B-operand load.
      - TN (trans_a=T, trans_b=F): native TN kernel, no host transpose.
      - TT (trans_a=T, trans_b=T): not supported (raises).
    trans_c=True returned as post-hoc out.t().contiguous().
    """
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"FlyDSL wrapper emits bf16 or fp16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    # fp16 vs bf16 output dtype for _StoreCPerTensor (both from the f32 accumulator).
    out_fp16 = out_dtype == torch.float16

    if trans_a and (not trans_b):
        # TN native: A [K, M], B [K, N]. Math C = A^T @ B.
        K_a, M = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # TN: per-shape autotune over the candidate families (general
        # intrinsic-A, big-N 2D-band swizzle, big-N asm-inplace, winK big-K
        # asm-inplace) picks the best cfg for this shape, caches by (M,N,K).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_tn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
        if trans_c:
            return out.t().contiguous()
        return out

    # Dispatch by layout.
    if (not trans_a) and (not trans_b):
        # NN native: A [M, K], B [K, N].
        M, K_a = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NN: per-shape runtime autotune over the candidate tiles, caches by
        # (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    elif (not trans_a) and trans_b:
        # NT native: A [M, K], B [N, K] (B^T storage of [K, N]).
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NT: per-shape runtime autotune over the 8w/v3 candidate tiles, caches
        # by (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nt_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    else:
        raise NotImplementedError(
            f"FlyDSL fp8 GEMM does not support the TT layout " f"(trans_a={trans_a}, trans_b={trans_b})."
        )
    if trans_c:
        return out.t().contiguous()
    return out
