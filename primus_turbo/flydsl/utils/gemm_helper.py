###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _as_index(v):
    # c_rows/c_cols may be a runtime value (dense/grouped NT/NN: N, m_end) or a
    # compile-time int (wgrad CShuffle: OUT_N). Coerce both to an MLIR index.
    return arith.index(v) if isinstance(v, int) else arith.index_cast(T.index, v)


def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False (no num_records_bytes): the buffer descriptor adapts to the
    # actual tensor extent instead of baking the first call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


def make_fp8_buffer_tensor_rebased(arg_i8, fp8_ir_t, base_elems, num_records_bytes):
    """make_fp8_buffer_tensor with the SRD base advanced by ``base_elems`` (fp8/int8
    = 1 byte/elem), in 64-bit. Folds a per-tile huge element offset into the
    descriptor base so the buffer voffset/soffset stay small int32 -> addresses
    inputs > 2^31 elems / > 4GB that the flat-shape pack and 32-bit voffset cannot.
    ``num_records_bytes`` bounds the SRD from the shifted base (HW OOB clamp)."""
    base = arith.index_cast(T.i64, _buffer_ops.extract_base_index(arg_i8))
    # Pin the wave-uniform shifted base + num_records to SGPRs: the group-scan base reads
    # as VGPR -> VGPR SRD -> readfirstlane waterfall per K-loop load. Pin keeps it scalar.
    base = _readfirstlane_i32(base + arith.index_cast(T.i64, base_elems))
    nr = arith.minui(arith.index_cast(T.index, num_records_bytes), arith.index(0xFFFFFFFF))
    nrec = fx.Int64(_readfirstlane_i32(arith.index_cast(T.i64, nr)))
    flags = _buffer_ops._get_buffer_flags()
    # global int8 ptr at the shifted addr -> int8 BufferDesc fat ptr -> recast fp8.
    base_ptr = fx.inttoptr(fx.PointerType.get(elem_ty=T.i8, address_space=1, alignment=16), base)
    i8_buf_ty = fx.PointerType.get(elem_ty=T.i8, address_space=TargetAddressSpace.BufferDesc, alignment=16)
    buf_ptr = fx.make_ptr(
        i8_buf_ty, [base_ptr, fx.Int16(0).ir_value(), nrec.ir_value(), fx.Int32(flags).ir_value()]
    )
    lay = fx.make_layout(0x40000000, 1)  # 1D flat; HW bounds via num_records
    iter_i8 = fx.get_iter(fx.make_view(buf_ptr, lay))
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, lay))


def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id, chunk_stride=1024, rebase=None):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64
        # Per-wave LDS chunk stride. A padded stride (e.g. 1056) un-aligns the
        # chunk base across LDS banks to cut transpose-read bank conflicts; the
        # read side (S2RLoaderTr) must use the same value.
        self.chunk_stride = chunk_stride
        # i64-traversal mode. None -> the contraction K-offset rides the 32-bit
        # soffset (caps the operand span at < 2^32 fp8). A tuple
        # (arg_i8, fp8_ir_t, base_elems, num_records_bytes) instead re-bases the
        # SRD per load: k_offset folds into the i64 descriptor base and soffset
        # stays 0, lifting the cap at the cost of one re-base per load.
        self.rebase = rebase

    def _src_div(self, k_offset):
        """(divided source tensor, soffset) for one load. int32 path returns the
        prebuilt source and rides k_offset on soffset; i64 path folds k_offset
        into the SRD base and returns soffset 0."""
        if self.rebase is None:
            return self.gl_src, k_offset
        arg_i8, fp8_t, base_elems, nrec = self.rebase
        off = _as_index(k_offset)
        # Clamp the shifted num_records to >= 0: an over-launched/masked tile (grouped
        # over-launch guard) can produce off > nrec; a signed-negative remainder would
        # wrap to a huge unsigned SRD bound (minui in make_fp8_buffer_tensor_rebased)
        # and read out of bounds. 0 records -> HW drops every load (matches int32 masking).
        rem = arith.maxsi(_as_index(nrec) - off, arith.index(0))
        g = make_fp8_buffer_tensor_rebased(arg_i8, fp8_t, _as_index(base_elems) + off, rem)
        return fx.logical_divide(g, fx.make_layout(1, 1)), 0

    def _lds_dst_at(self, lds_dst, step, base_off=None):
        cs = self.chunk_stride
        step_off = self.wave_id * cs + step * (self.n_waves * cs)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        if base_off is not None:  # runtime LDS-stage byte offset (double-buffer parity)
            base_i32 = base_i32 + base_off
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset, base_off=None):
        src_div, soff = self._src_div(k_offset)
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(src_div, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step, base_off)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(soff))


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


class S2RLoader:
    # Uses the intrinsic ds_read (no manual-lgkmcnt inline-asm path): the backend already
    # packs the reads onto shared base pointers and schedules per-tile lgkmcnt finer than a
    # single coarse drain.
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag


def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


class Mfma16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c):
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)])
        return c


# ── Reusable fp8 GEMM primitives (store, K-tail mask, value-attrs, AGPR MFMA, XCD
#    remap, LDS-ptr/transpose loaders, swizzle), shared by dense and grouped.


def _readfirstlane_i32(v):
    """Force a wave-uniform-in-value i32 into an SGPR via s_readfirstlane.

    For grouped GEMM the output buffer descriptor's num_records = m_end*c_n*2
    is uniform across a tile's wave (all lanes share the group), but the
    compiler's divergence analysis treats m_end (from the per-tile group scan)
    as divergent -> the SRD lands in VGPRs -> every buffer_store_short is
    wrapped in a readfirstlane/saveexec waterfall loop. Pinning the value to
    SGPR collapses the SRD to scalar regs and drops the per-store waterfall."""
    raw = _raw(v)
    r = rocdl.readfirstlane(res=raw.type, src=raw)
    rv = r.result if hasattr(r, "result") else r
    return ArithValue(rv)


class StoreCPerTensor:
    """Per-tensor scaled output store: out = (acc * a_scale * b_scale).to(out_ty).

    Both scales are read once from length-1 buffers and applied uniformly;
    out_ty is bf16 or fp16. Columns past c_cols clamp to an OOB index.
    """

    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        # C addressed via i64 per-tile re-basing (handles M*N > 2^31 / >4GB output);
        # pass C as 2D so its shape packs within int32.
        self.c_base = _buffer_ops.extract_base_index(C)  # index = byte base address
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)  # 1 fp32
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)  # 1 fp32
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def store(self, c_frag, base_row, base_col):
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div)
        # Re-base output at this row band (i64) so the per-store byte offset stays int32;
        # clamp band base to [0, c_rows] and num_records to the 32-bit SRD field.
        out_b = 2  # bf16/fp16 = 2 bytes
        cols_i = _as_index(self.c_cols)
        row_i = _as_index(base_row)
        rows_i = _as_index(self.c_rows)
        row_c = arith.minui(row_i, rows_i)
        band_base = self.c_base + row_c * cols_i * arith.index(out_b)
        # Cap at 0x7FFFFFFF so buffer_store(mask=False) → voffset=0x7FFFFFFF is always OOB;
        # valid tile offsets are at most BLOCK_M*c_cols*2 ≈ 30 MB << 2 GB.
        nrec = arith.minui((rows_i - row_c) * cols_i * arith.index(out_b), arith.index(0x7FFFFFFF))
        # Pin to SGPRs: base_row derives from the group scan which the compiler marks as
        # divergent, landing the SRD in VGPRs and waterfalling every buffer_store.
        band_base_i64 = _readfirstlane_i32(arith.index_cast(T.i64, band_base))
        nrec_pinned = arith.index_cast(T.index, _readfirstlane_i32(arith.index_cast(T.i64, nrec)))
        rsrc = _buffer_ops.create_buffer_resource_from_addr(band_base_i64, num_records_bytes=nrec_pinned)
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = (vec_f32[i] * scale).to(self.out_ty)
                    off = ((row_local + i) * self.c_cols + col) * out_b  # i32-small within band
                    _buffer_ops.buffer_store(scaled, rsrc, off, mask=col_valid, offset_is_bytes=True)


class StoreCPerTensorCShuffle:
    """CShuffle output store: same value->global-address mapping as StoreCPerTensor
    (byte-identical) but stages each 16-row sub-tile through per-wave LDS row-major,
    re-reads it N-contiguous, and emits one vectorized 128b global store per lane
    (vs 128 column-strided scalar buffer_store_short). Assumes BLOCK_N=256 (EPL=8
    out_ty/lane=128b) and c_cols % Cc == 0, base_col % Cc == 0 (true for FFN N dims);
    invalid runs clamp to an OOB element index (HW SRD drop), as the scalar path does."""

    def __init__(
        self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty, c_lds, wave_id
    ):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.wave_id = wave_id
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        self.Cc = n_tiles_b * 16  # columns in one 16-row shuffle tile
        self.EPL = (16 * self.Cc) // 64  # out_ty elements each lane re-reads (16*Cc rows/cols / 64 lanes)
        # One coalesced global store is 128 bits = 8 x 16b (buffer_store_dwordx4). If a
        # lane re-reads more than that (EPL > 8 -- e.g. the 4-wave 2x2 geometry has
        # n_tiles_b=4 -> Cc=64 -> EPL=16), emit EPL//8 back-to-back 128b stores; EPL==8
        # (the 8-wave path) is a single store. EPL must be a multiple of 8 and fit in one row.
        self.elems_per_store = 8  # 16b elements packed into one 128b vector store
        assert self.EPL % self.elems_per_store == 0 and self.EPL <= self.Cc, (
            f"CShuffle expects EPL a multiple of 8 within Cc={self.Cc}; got EPL={self.EPL}"
        )
        # The ds_write_b16 staging + 128b re-read aliases LDS banks, but the epilogue
        # store stall is hidden behind the MMA pipeline / next-tile prologue, so anti-
        # conflict row padding is perf-neutral here and is not used.
        self.row_stride = self.Cc  # logical == physical (no anti-conflict padding)
        self.wave_lds_elems = 16 * self.row_stride  # per-wave staging (one 16-row tile)
        self.c_lds = c_lds
        # C addressed via i64 per-band re-basing (handles OUT_M*OUT_N > 2^31 / >4GB);
        # the final 128b store re-bases at each 16-row sub-tile band (see store()).
        self.c_base = _buffer_ops.extract_base_index(C)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        # addr-space 2 (LDS), mirroring G2SLoader.LdsPtr_t. Separate scalar-store
        # (align 2) and vector-read (align 16) pointer types.
        self._store_ptr_t = fx.PointerType.get(out_ty.ir_type, 2, 2)
        self._read_ptr_t = fx.PointerType.get(out_ty.ir_type, 2, 16)

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def store(self, c_frag, base_row, base_col):
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div)
        lds_base = fx.Int32(fx.ptrtoint(self.c_lds.ptr))
        wave_off = self.wave_id * self.wave_lds_elems  # element offset of this wave's region
        out_b = 2  # bf16/fp16 = 2 bytes
        cols_i = _as_index(self.c_cols)
        rows_i = _as_index(self.c_rows)
        for ti in range_constexpr(self.n_tiles_a):
            # --- stage this 16-row sub-tile row-major into the per-wave LDS region ---
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                lds_col = tj * 16 + self.lane_id % 16
                for i in range_constexpr(4):
                    lds_row = (self.lane_id // 16) * 4 + i
                    e = wave_off + lds_row * self.row_stride + lds_col
                    val = (vec_f32[i] * scale).to(self.out_ty)
                    ptr = fx.inttoptr(self._store_ptr_t, lds_base + e * 2)
                    ptr.store(val)
            S2RLoaderTr._wait_lgkmcnt(0)
            # Re-base output at this 16-row band (i64), re-read N-contiguous (one EPL-col
            # run/lane) + one 128b store at a small in-band i32 offset; band num_records OOB-drops.
            band_row = arith.index_cast(T.index, base_row + ti * 16)
            row_c = arith.minui(band_row, rows_i)
            band_base = self.c_base + row_c * cols_i * arith.index(out_b)
            nrec = arith.minui((rows_i - row_c) * cols_i * arith.index(out_b), arith.index(0x7FFFFFFF))
            band_base_i64 = _readfirstlane_i32(arith.index_cast(T.i64, band_base))
            nrec_pinned = arith.index_cast(T.index, _readfirstlane_i32(arith.index_cast(T.i64, nrec)))
            rsrc = _buffer_ops.create_buffer_resource_from_addr(band_base_i64, num_records_bytes=nrec_pinned)
            row_in = (self.lane_id * self.EPL) // self.Cc
            col0 = (self.lane_id * self.EPL) % self.Cc
            for sub in range_constexpr(self.EPL // self.elems_per_store):
                col_in = col0 + sub * self.elems_per_store
                lane_e = wave_off + row_in * self.row_stride + col_in
                rptr = fx.inttoptr(self._read_ptr_t, lds_base + lane_e * 2)
                vec = fx.make_view(rptr, fx.make_layout(self.elems_per_store, 1)).load()
                gcol = base_col + col_in
                valid = (gcol + fx.Int32(self.elems_per_store)) <= self.c_cols
                off = (row_in * self.c_cols + gcol) * out_b  # i32-small within band
                _buffer_ops.buffer_store(vec, rsrc, off, mask=valid, offset_is_bytes=True)
            S2RLoaderTr._wait_lgkmcnt(0)  # drain re-read before next ti overwrites LDS


def _a_tail_mask_vec(lane_id, r):
    """Per-lane i32x8 byte-mask zeroing A-fragment bytes whose K-column >= r
    (r in [1,128)). AND-ing it into the A frag drops the K-tail terms (a_k=0)
    so the mfma ignores k>=r regardless of B."""
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


def mask_a_tail(frag_list, lane_id, r):
    """Return A frags with the K-tail (>= r) zeroed; r%128==0 -> unchanged."""
    if r % 128 == 0:
        return frag_list
    mask = _a_tail_mask_vec(lane_id, r % 128)
    return [f & mask for f in frag_list]


def make_value_attrs(waves_per_eu, agpr_alloc, fwg):
    """Kernel value_attrs. agpr_alloc: 0 = compiler default; N>0 = force exactly
    N AGPRs ("N,N"); -N = allow up to N ("0,N")."""
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


def asm_mma_do(a, b, c, mode="2", cbsz=0, blgp=0):
    """fp8 16x16x128 MFMA via inline asm, to pin the dst register class.
    mode "2" (=a,v,v,0): accumulator in AGPR (srcA/srcB in VGPR) — separate register
    files keep dst from aliasing srcA and free the VGPR file. mode "3" (=v,v,v,0): VGPR
    in-place (D=C, avoids the accvgpr shuffle). mode "1" (=&v,v,v,0): VGPR early-clobber."""
    v4f32 = ir.VectorType.get([4], ir.F32Type.get())
    cons = {"2": "=a,v,v,0", "3": "=v,v,v,0"}.get(str(mode), "=&v,v,v,0")
    # cbsz/blgp select srcA/srcB fp8 format (0=E4M3, 1=E5M2).
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    op = _llvm.InlineAsmOp(
        res=v4f32,
        operands_=[_raw(a), _raw(b), _raw(c)],
        asm_string=f"v_mfma_f32_16x16x128_f8f6f4 $0, $1, $2, $0{mods}",
        constraints=cons,
        has_side_effects=False,
    )
    return Vec(op.result)


def xcd_remap_pid(pid, total_pids, num_xcd):
    """Remap the tile id so same-XCD workgroups gather into one contiguous
    block, keeping each XCD's L2 reuse within that XCD. Bijection over
    [0, total_pids); identity when num_xcd <= 1."""
    if num_xcd <= 1:
        return pid
    per_xcd = total_pids // num_xcd  # floor
    rem = total_pids - per_xcd * num_xcd
    xcd = pid % num_xcd
    local = pid // num_xcd
    offset = xcd * per_xcd + arith.select(xcd < rem, xcd, rem)
    return offset + local


def _inttoptr_lds(byte_addr):
    """Integer byte address -> !llvm.ptr<3> (LDS). Parsed per call: the type is
    bound to the current MLIRContext and cannot be cached across compiles."""
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


_gep = _buffer_ops.get_element_ptr


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return ptr


def _packed_ds_read_tr_offsets(base_ptr, byte_offsets, vmcnt_hint=None):
    """Pack the ds_read_b64_tr_b8 reads onto ONE shared base-ptr VGPR, each at a
    compile-time immediate byte offset (1 addr VGPR instead of N avoids an
    address-register spill). Reads are async (complete on lgkmcnt); the caller
    must drain lgkmcnt before the consuming mfma. Returns one v2i32 per offset."""
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


def compute_global_swizzle_nn(lane_id, wave_id, N_out, n_rounds):
    """Per-lane global-load offsets for NN B [K_inner, N_out] row-major: each
    round loads 64 K-rows x 128 N-bytes via swizzle_128(k_row, n_col) over the
    flat [K, N] byte view with N_out element stride."""
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        k_row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        n_col = (lane_id % 8) * 16
        rs, cs = swizzle_128(k_row, n_col)
        offsets.append(rs * N_out + cs)
    return offsets


class S2RLoaderTr:
    """LDS -> mfma operand wave-coop transpose load via ds_read_b64_tr_b8.
    Serves K-major fp8 operands (NN B, TN A and B — their mfma operand byte
    layouts are identical); the operand is selected by the per-wave coordinate
    stride tile_stride and the WG wave count n_waves. See _ptr_off for the map.
    """

    _K_BASE = (0, 8, 64, 72)

    def __init__(
        self,
        wave_idx,
        n_tiles,
        tile_stride,
        inline_asm=False,
        vmcnt_hint=2,
        chunk_stride=1024,
        n_waves=8,
    ):
        """wave_idx: this wave's index along the transposed coordinate (wave_n
        for B, wave_m for A). tile_stride: per-wave coverage on that axis.
        chunk_stride must match the G2S writer. inline_asm issues the reads as
        opaque asm (caller drains via vmcnt_hint) and requires agpr_alloc>0."""
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.tile_stride = tile_stride
        self.lane_id = fx.thread_idx.x % 64
        self.inline_asm = inline_asm
        self.vmcnt_hint = vmcnt_hint
        self.chunk_stride = chunk_stride
        self.n_waves = n_waves
        self.round_stride = n_waves * chunk_stride

    def _ptr_off(self, c, tile_i, I, L_in_sg):
        KW = self.n_waves * 8
        K_log = I * 16 + S2RLoaderTr._K_BASE[c] + (L_in_sg // 2)
        r_step = K_log // KW
        W = (K_log % KW) // 8
        K_mod_8 = K_log % 8
        swz_K = ((K_log % 16) // 2) * 16
        coord_start = self.wave_idx * self.tile_stride + tile_i * 16
        j_chunk = (coord_start // 16) ^ (swz_K // 16)
        return (
            W * self.chunk_stride
            + r_step * self.round_stride
            + K_mod_8 * 128
            + j_chunk * 16
            + (L_in_sg % 2) * 8
        )

    def _issue_one(self, lds_src, tile_i, base_off=None):
        """Issue the 4 ds_read_b64_tr_b8 of one tile (no drain, no assemble).
        Returns the 4 raw v2i32 Vec."""
        tr_type = Vec.make_type(2, fx.Int32)
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        if base_off is not None:  # runtime LDS-stage byte offset (double-buffer parity)
            base_i32 = base_i32 + base_off
        I = self.lane_id // 16
        L_in_sg = self.lane_id % 16
        RS = self.round_stride  # c0->c2 / c1->c3 jump (one K-sub-round)
        if self.inline_asm:
            p0 = _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off(0, tile_i, I, L_in_sg)))
            p1 = _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off(1, tile_i, I, L_in_sg)))
            r02 = _packed_ds_read_tr_offsets(p0, [0, RS], vmcnt_hint=self.vmcnt_hint)
            r13 = _packed_ds_read_tr_offsets(p1, [0, RS], vmcnt_hint=None)
            # r02 = [c0, c2], r13 = [c1, c3] -> caller assembles as c0,c1,c2,c3
            return [Vec(r02[0]), Vec(r13[0]), Vec(r02[1]), Vec(r13[1])]
        return [
            Vec(
                rocdl.ds_read_tr8_b64(
                    tr_type,
                    _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off(c, tile_i, I, L_in_sg))),
                ).result
            )
            for c in range_constexpr(4)
        ]

    @staticmethod
    def _assemble(calls):
        # Concat 4 x v2i32 -> v8i32 = mfma operand bytes 0..31 for this lane.
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

    def load(self, lds_src, preshuffled=False, drain=True, base_off=None):
        """Return all n_tiles operand frags. Inline-asm path issues every tile's
        async reads then one trailing lgkmcnt(0) before the consuming mfma;
        drain=False skips it when a later drain covers these reads. The intrinsic
        path lets the backend insert the wait. base_off = runtime LDS-stage byte
        offset (double-buffer parity)."""
        assert not preshuffled, "S2RLoaderTr does not support preshuffled"
        if self.inline_asm:
            all_calls = [self._issue_one(lds_src, t, base_off) for t in range_constexpr(self.n_tiles)]
            if drain:
                self._wait_lgkmcnt(0)
            return [self._assemble(c) for c in all_calls]
        return [self._assemble(self._issue_one(lds_src, t, base_off)) for t in range_constexpr(self.n_tiles)]

    def base_addr(self, lds_src):
        """Per-lane LDS byte address pairs for the whole-loop transpose reads.

        Returns [[p0, p1]] * n_tiles: tile i's 4 ds_read_b64_tr_b8 are c0=p0[i]+0,
        c1=p1[i]+0, c2=p0[i]+RS, c3=p1[i]+RS (RS = 8*chunk_stride), assembled in-place
        as the 8xi32 operand (no shuffle). p0/p1 are NOT tile-strided (j_chunk carries
        an XOR), so each tile needs its own pair."""
        base = fx.Int32(fx.ptrtoint(lds_src.ptr))
        I = self.lane_id // 16
        L_in_sg = self.lane_id % 16
        out = []
        for t in range_constexpr(self.n_tiles):
            p0 = base + fx.Int32(self._ptr_off(0, t, I, L_in_sg))
            p1 = base + fx.Int32(self._ptr_off(1, t, I, L_in_sg))
            out.append([p0, p1])
        return out
