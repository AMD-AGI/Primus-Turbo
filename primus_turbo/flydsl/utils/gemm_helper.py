###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.buffer_ops import buffer_store, create_buffer_resource
from flydsl.expr.typing import AddressSpace, PointerType, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


_PRESHUF_KT = 16  # scale-preshuffle k-tile (rows*KT dwords staged in LDS per workgroup)


def scale_opsel(k, pack=1):
    return k % pack


def _as_index(v):
    # c_rows/c_cols may be a runtime value (dense/grouped NT/NN: N, m_end) or a
    # compile-time int (wgrad CShuffle: OUT_N). Coerce both to an MLIR index.
    return arith.index(v) if isinstance(v, int) else arith.index_cast(T.index, v)


def make_fp8_buffer_tensor_rebased(arg_i8, fp8_ir_t, base_elems, num_records_bytes):
    """Build an fp8 BufferDesc tensor with the SRD base advanced by ``base_elems`` (fp8/int8
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


def make_bf16_buffer_tensor_rebased(arg, bf16_ir_t, base_bytes, num_records_bytes):
    """make_fp8_buffer_tensor_rebased for a 2-byte (bf16/fp16) operand: SRD base
    advanced by ``base_bytes`` (i64), bounded by ``num_records_bytes`` (HW OOB clamp
    -> per-group K-tail reads 0). Folds the per-group token base into the descriptor
    so the buffer voffset/soffset stay int32 even past 2^31 elems."""
    base = arith.index_cast(T.i64, _buffer_ops.extract_base_index(arg))
    base_off = arith.index_cast(T.i64, arith.index_cast(T.index, base_bytes))
    base = _readfirstlane_i32(base + base_off)
    nr = arith.minui(arith.index_cast(T.index, num_records_bytes), arith.index(0xFFFFFFFF))
    nrec = fx.Int64(_readfirstlane_i32(arith.index_cast(T.i64, nr)))
    flags = _buffer_ops._get_buffer_flags()
    base_ptr = fx.inttoptr(fx.PointerType.get(elem_ty=T.i8, address_space=1, alignment=16), base)
    i8_buf_ty = fx.PointerType.get(elem_ty=T.i8, address_space=TargetAddressSpace.BufferDesc, alignment=16)
    buf_ptr = fx.make_ptr(
        i8_buf_ty, [base_ptr, fx.Int16(0).ir_value(), nrec.ir_value(), fx.Int32(flags).ir_value()]
    )
    lay = fx.make_layout(0x40000000, 1)
    iter_i8 = fx.get_iter(fx.make_view(buf_ptr, lay))
    bf_buf_ptr_ty = fx.PointerType.get(
        elem_ty=bf16_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_bf = fx.recast_iter(bf_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_bf, lay))


def make_bf16_fp16_tile_tensor(base_i64, byte_offset, elems):
    """Per-tile 2-byte (bf16/fp16) global view with base rebased by ``byte_offset`` in
    int64 (keeps per-lane voffset int32 past a >4GB pool). Returns a plain global view;
    the caller feeds it straight to gemm_tile, so the nt_tile API stays unchanged."""
    ptr_ty = PointerType.get(elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16)
    tile_ptr = fx.inttoptr(ptr_ty, base_i64 + byte_offset)
    return fx.make_view(tile_ptr, fx.make_layout(elems, 1))


def make_bf16_rebased_rsrc(arg, base_elems, num_records_bytes):
    """Return a raw !llvm.ptr<8> buffer SRD whose base = arg's base advanced by
    ``base_elems`` bf16 elements (2 bytes), in 64-bit, bounded by ``num_records_bytes``.
    For buffers > 4GB (e.g. pro-cr4 interm 4.8GB) where a flat entry*D_V element offset
    overflows the 32-bit voffset. Mirrors BufferResourceDescriptor.from_memref's
    MakeBufferRsrcOp construction but with a rebased base. Pass directly to
    buffer_load/buffer_store as `rsrc` (with dtype=bf16)."""
    base = arith.index_cast(T.i64, _buffer_ops.extract_base_index(arg))
    base = base + arith.index_cast(T.i64, base_elems) * fx.Int64(2)
    nr = arith.minui(arith.index_cast(T.index, num_records_bytes), arith.index(0xFFFFFFFF))
    nrec = _raw(fx.Int64(arith.index_cast(T.i64, nr)))
    flags = _buffer_ops._get_buffer_flags()
    llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
    base_ptr = _llvm.IntToPtrOp(llvm_ptr_ty, _raw(base)).result
    rsrc_ty = ir.Type.parse("!llvm.ptr<8>")
    stride_val = _buffer_ops._create_i16_constant(0)
    flags_val = _buffer_ops._create_i32_constant(flags)
    rsrc = rocdl.MakeBufferRsrcOp(rsrc_ty, base_ptr, stride_val, nrec, flags_val).result
    return rsrc


def swizzle_128(row, col, width=128):
    """XOR bank-swizzle over a `width`=2**k logical row (width=128 is byte-identical to
    the original fixed-128 form). The swizzle must stay within col's bits [0,k); for k<7
    we extract fewer bits (k-4) from the `row%16` window so the result is always < width."""
    k = width.bit_length() - 1
    period = 16 * width
    offset = row * width + col
    nbits = k - 4
    mask = (1 << nbits) - 1
    extracted = (offset % period) >> (k + 1)
    swizzle = (extracted & mask) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // width, swizzled_offset % width


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


class _S2RLoaderBase:
    """Shared ctor for LDS->register operand loaders: caches the per-lane id,
    this wave's tile index, and the tile count."""

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles


class S2RLoader(_S2RLoaderBase):
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

    def base_addr(self, lds_src, preshuffled=False):
        """Per-lane LDS byte address pairs for the bare-asm whole-loop: mirrors `load()`'s
        addressing but returns addresses so `ds_line` can emit 2x ds_read_b128 (no HW
        transpose) for an already-M/N-major operand. Same [[p0,p1]]*n_tiles shape as
        `S2RLoaderTr.base_addr` (interchangeable); p0/p1 are the two 16B halves of one
        32-elem fragment, not two lane-group bases."""
        base = fx.Int32(fx.ptrtoint(lds_src.ptr))
        out = []
        for i in range_constexpr(self.n_tiles):
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            addrs = []
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                addrs.append(base + fx.Int32(offset))
            out.append(addrs)
        return out


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

    def set_inplace_asm(self, cbsz, blgp):
        # inplace MFMA: accum in AGPR; cbsz/blgp select srcA/srcB fmt
        self._do_mma = lambda _a, _b, _c: asm_mma_do(_a, _b, _c, mode="2", cbsz=cbsz, blgp=blgp)

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


# ── MXFP8 scaled MFMA + per-1x32 E8M0 scale loaders (block-scaled dense GEMM);
#    scales are pre-shuffled by the quant / FlyDSL preshuffle.


def _asm_mma_scale_do(a, b, c, sa, sb, opsel, cbsz=0, blgp=0):
    """Inline-asm scaled MFMA v_mfma_scale_f32_16x16x128_f8f6f4. =&v early-clobber
    forces dst disjoint from srcA/srcB; opaque to the backend so it co-schedules with
    the asm ds_read_b64_tr_b8 loads. opsel (0..3) picks the packed E8M0 byte via
    op_sel (low bit) / op_sel_hi (high bit). cbsz/blgp select srcA/srcB fp8 format
    (0=E4M3, 1=E5M2)."""
    v4f32 = ir.VectorType.get([4], ir.F32Type.get())
    lo = opsel & 1
    hi = (opsel >> 1) & 1
    osel = f"op_sel:[{lo},{lo},0] op_sel_hi:[{hi},{hi},0]"
    cons = "=&v,v,v,0,v,v"  # VGPR early-clobber accumulator
    op = _llvm.InlineAsmOp(
        res=v4f32,
        operands_=[_raw(a), _raw(b), _raw(c), _raw(sa), _raw(sb)],
        asm_string=f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $4, $5 cbsz:{cbsz} blgp:{blgp} {osel}",
        constraints=cons,
        has_side_effects=False,
    )
    return Vec(op.result)


class MfmaScale16x16x128:
    """16x16x128 f8f6f4 MFMA with per-block E8M0 scale operands.

    Mirrors ``Mfma16x16x128`` but routes through the raw rocdl intrinsic so
    the (scale_a, scale_b) i32 operands can be supplied per call.
    """

    def __init__(self, n_tiles_a, n_tiles_b, asm_mma=False, cbsz=0, blgp=0):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        # opsel picks the packed dword's E8M0 byte (k%PACK); pack==1 -> stays 0.
        self.opsel = 0
        self.asm_mma = asm_mma
        self.cbsz = cbsz  # srcA fp8 format: 0=E4M3, 1=E5M2
        self.blgp = blgp  # srcB fp8 format: 0=E4M3, 1=E5M2

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c, sa, sb):
        # operand order: a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b
        if self.asm_mma:  # inline-asm scaled MFMA (co-schedules with asm tr8 loads)
            return _asm_mma_scale_do(a, b, c, sa, sb, self.opsel, self.cbsz, self.blgp)
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            self.res_ty,
            [a, b, c, self.cbsz, self.blgp, self.opsel, sa, self.opsel, sb],
        )

    def call(self, a, b, c, sa, sb):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        assert len(sa) == self.n_tiles_a
        assert len(sb) == self.n_tiles_b

        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])
        return c


class ScaleBComb:
    """Combined B scale loader (pairs with the combined-B preshuffle, layout 3:
    ``build_preshuffle_ab_kernel`` B region / C++ ``compute_preshuffle_scale_index``).

    One dwordx4 per lane returns [s0,s1,s2,s3]; (s0,s1)=b0 sub-tiles, (s2,s3)=b1.
    """

    def __init__(self, sp_tensor, dim, K, n_slabs=1, pack=1):
        self.K128 = K // 128  # number of K-groups (one i32 per K-iter)
        self.PACK = pack
        self.K128p = ceildiv(self.K128, self.PACK)  # packed K-groups (PACK scales / dword)
        self.lane = fx.thread_idx.x % 64
        # grp = (col//256)*4 + wn is block-strided, so the buffer holds cdiv(dim,256)*4
        # groups (matches the C++ preshuffle B sizing). A partial last 256-block reads
        # only its valid wn groups; OOB-col reads clamp to 0 and StoreC drops them.
        # dim%256==0 -> cdiv(dim,256)*4 == dim//64 (no change for aligned shapes).
        # n_slabs>1 (grouped): b_sp stacks G per-group slabs; load(slab=g) indexes group g.
        self.slab_elems = ((dim + 255) // 256) * 4 * self.K128p * 64 * 4  # i32 per group
        nbytes = self.slab_elems * n_slabs * 4  # int32 records
        self.rsrc = _buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k, slab=0):
        """base: sb_base0 (b0 region col base). Returns 4 i32 (b0:0,1  b1:2,3)."""
        grp = (base // 256) * 4 + (base % 256) // 32
        kk = k // self.PACK
        idx = ((grp * self.K128p + kk) * 64 + self.lane) * 4 + slab * self.slab_elems
        v = Vec(_buffer_ops.buffer_load(self.rsrc, idx, vec_width=4, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(4)]


class ScaleS2R:
    """Per-lane E8M0 scale loader for v_mfma_scale_f32_16x16x128 (preshuffled).

    The 16x16x128 MFMA distributes K=128 so lane ``(g, r)`` with
    ``g = lane//16`` (0..3) and ``r = lane%16`` holds the A/B data for matrix
    row/col ``r`` and the 32-K micro-block ``g``. With opsel==0 the hardware
    samples byte 0 of each lane's scale operand, so lane ``(g, r)`` just needs
    ``scale[r, 4k+g]`` in a register.

    To make that a single fully-coalesced dword load with no per-lane ALU, the
    host pre-shuffles the raw E8M0 [DIM, K//32] into

        SP[rt, k, lane] = broadcast_u8_to_u32( scale[rt*16 + lane%16, 4k + lane//16] )

    laid out int32 [DIM//16, K//128, 64]. For row-tile ``rt`` and K-iter ``k``
    the 64 lanes of a wave read 64 contiguous dwords. The A-operand preshuffle
    (layout 1) is produced by ``build_preshuffle_ab_kernel`` (A region), fused into
    the mxfp8 GEMM launch.
    """

    def __init__(self, sp_tensor, dim, K, n_tiles, pack=1):
        self.K128 = K // 128  # number of K-groups (one i32 per K-iter)
        self.PACK = pack
        self.K128p = ceildiv(self.K128, self.PACK)  # packed K-groups (PACK scales / dword)
        self.n_tiles = n_tiles
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64  # == (lane//16)*16 + lane%16
        # cdiv (not floor): a non-group_span-multiple ``dim`` (general M) still needs the
        # partial last 64-row group resident so its valid rows read real scales; the
        # group's OOB rows were preshuffle-masked to 0 and StoreC drops their output.
        nbytes = ceildiv(dim, self.group_span) * self.K128p * 64 * n_tiles * 4  # int32 records
        self.rsrc = _buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        """base: runtime global row/col base for this (region, wave). Returns n_tiles i32
        (packed dword for K-group k//PACK; caller selects byte k%PACK via MFMA op_sel)."""
        grp = base // self.group_span
        kk = k // self.PACK
        idx = ((grp * self.K128p + kk) * 64 + self.lane) * self.n_tiles
        v = Vec(_buffer_ops.buffer_load(self.rsrc, idx, vec_width=self.n_tiles, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(self.n_tiles)]


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
    """Scalar output store: out = (acc [* a_scale * b_scale]).to(out_ty).

    Shared by the per-tensor GEMM and the mxfp8 GEMM. ``A_scale``/``B_scale`` are
    optional: when given, both are read once from length-1 buffers and applied
    uniformly (per-tensor); when ``None`` the scale is already folded into the
    accumulator by the scaled MMA (mxfp8), so the store is plain. The output is
    re-based per row band in 64-bit index (int64-safe, M*N > 4GB) via
    ``make_row_band_resource``; columns past c_cols clamp to an OOB index (HW SRD
    drop). out_ty bf16/fp16; pass C as 2D so its shape packs within int32.
    """

    def __init__(
        self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty, elem_fn=None
    ):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        # Optional f32->f32 epilogue node chain (bias/act), post-scale pre-cast.
        self.elem_fn = elem_fn
        self.scaled = A_scale is not None
        self.c_base = _buffer_ops.extract_base_index(C)  # index = byte base address
        if self.scaled:
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
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div) if self.scaled else None
        # buffer_store row-band path (int64-safe); the band SRD is pinned to SGPRs inside.
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    val = vec_f32[i] * scale if self.scaled else vec_f32[i]
                    if self.elem_fn is not None:
                        val = self.elem_fn(val)  # bias/act epilogue node chain
                    val = val.to(self.out_ty)
                    off = ((row_local + i) * self.c_cols + col) * 2  # i32-small within band
                    _buffer_ops.buffer_store(val, rsrc, off, mask=col_valid, offset_is_bytes=True)


class StoreCPerTensorCShuffle:
    """CShuffle output store: same value->global-address mapping as StoreCPerTensor
    (byte-identical) but stages each 16-row sub-tile through per-wave LDS row-major,
    re-reads it N-contiguous, and emits one vectorized 128b global store per lane
    (vs 128 column-strided scalar buffer_store_short). Assumes BLOCK_N=256 (EPL=8
    out_ty/lane=128b) and c_cols % Cc == 0, base_col % Cc == 0 (true for FFN N dims);
    invalid runs clamp to an OOB element index (HW SRD drop), as the scalar path does."""

    def __init__(
        self,
        A_scale,
        B_scale,
        C,
        c_rows,
        c_cols,
        c_idx_fn,
        n_tiles_a,
        n_tiles_b,
        out_ty,
        c_lds,
        wave_id,
        row_pad=0,
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
        # The ds_write_b16 staging + 128b re-read aliases LDS banks. row_pad=0 (default)
        # keeps the historical behavior. row_pad is an explicit opt-in: the caller must
        # size its own C_lds_shuffle allocation as n_waves*16*(n_tiles_b*16 + row_pad)
        # (not just n_tiles_b*16), or this overflows the buffer.
        self.row_stride = self.Cc + row_pad
        self.row_pad = row_pad
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


def compute_global_swizzle_nn(lane_id, wave_id, N_out, n_rounds, width=128, wswz=False):
    """Per-lane global-load offsets for NN B [K_inner, N_out] row-major, via
    swizzle_128(k_row, n_col, width) over the flat [K, N] byte view. `width` must equal
    the destination buffer's LDS column span (e.g. LDS_BLOCK_N; defaults to 128,
    byte-identical there) or reads overlap. chunks=width//16 lanes cooperate per K-row,
    the remaining 64/chunks lanes per wave span distinct K-rows."""
    offsets = []
    n_waves = fx.block_dim.x // 64
    chunks = width // 16
    rows_per_wave = 64 // chunks
    for r in range_constexpr(n_rounds):
        k_row = lane_id // chunks + wave_id * rows_per_wave + r * (n_waves * rows_per_wave)
        n_col = (lane_id % chunks) * 16
        rs, cs = swizzle_128(k_row, n_col, width=width)
        if wswz:
            # Matching write for the read-side j_chunk^(W<<1) bank-spread swizzle:
            # XOR (wave_id<<1) into the column chunk so it lands where the read looks.
            cs = cs ^ (((wave_id << 1) & (chunks - 1)) * 16)
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
        width=128,
        wswz=False,
    ):
        """wave_idx: this wave's index along the transposed coord (wave_n for B, wave_m for
        A). tile_stride: per-wave coverage. chunk_stride must match the G2S writer.
        inline_asm issues opaque-asm reads (caller drains via vmcnt_hint, needs
        agpr_alloc>0). width: this buffer's LDS column span (defaults 128) -- MUST match the
        paired compute_global_swizzle_nn `width` (see swz_K for why it scales with width)."""
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.tile_stride = tile_stride
        self.lane_id = fx.thread_idx.x % 64
        self.inline_asm = inline_asm
        self.vmcnt_hint = vmcnt_hint
        self.chunk_stride = chunk_stride
        self.n_waves = n_waves
        self.round_stride = n_waves * chunk_stride
        self.width = width
        self.wswz = wswz  # wave bank-swizzle (j_chunk^(W<<1)); scoped to 2-pool@1024

    def _ptr_off(self, c, tile_i, I, L_in_sg):
        # rows_per_wave = 64/chunks must match compute_global_swizzle_nn's own
        # rows_per_wave (the write side's local K-row count per wave per round).
        chunks = self.width // 16
        rows_per_wave = 64 // chunks
        KW = self.n_waves * rows_per_wave
        K_log = I * 16 + S2RLoaderTr._K_BASE[c] + (L_in_sg // 2)
        r_step = K_log // KW
        W = (K_log % KW) // rows_per_wave
        K_local_row = K_log % rows_per_wave
        # swz_K reproduces swizzle_128's own (extracted & mask) term: extracted =
        # (K_log%16)//2 is width-independent; only the mask narrows with width.
        swz_K = (((K_log % 16) // 2) & (chunks - 1)) * 16
        coord_start = self.wave_idx * self.tile_stride + tile_i * 16
        j_chunk = (coord_start // 16) ^ (swz_K // 16)
        if self.wswz:
            # XOR (W<<1) into j_chunk: address bit shift W*32, matching the write
            # side's compute_global_swizzle_nn(wswz=True) -- drops LDSBankConflict
            # conflicts drop to 0 at _CS=1024 (2-pool).
            j_chunk = j_chunk ^ ((W << 1) & (chunks - 1))
        return (
            W * self.chunk_stride
            + r_step * self.round_stride
            + K_local_row * self.width
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
        """Per-lane LDS address pairs [[p0,p1]]*n_tiles for the whole-loop transpose reads:
        tile i's 4 ds_read_b64_tr_b8 are p0[i]+0, p1[i]+0, p0[i]+RS, p1[i]+RS (RS =
        8*chunk_stride). p0/p1 are not tile-strided (j_chunk carries an XOR), so each tile
        needs its own pair."""
        base = fx.Int32(fx.ptrtoint(lds_src.ptr))
        I = self.lane_id // 16
        L_in_sg = self.lane_id % 16
        out = []
        for t in range_constexpr(self.n_tiles):
            p0 = base + fx.Int32(self._ptr_off(0, t, I, L_in_sg))
            p1 = base + fx.Int32(self._ptr_off(1, t, I, L_in_sg))
            out.append([p0, p1])
        return out


def block_mn(pid, num_pid_m, n_blocks, GM, GN):
    """Tile-id -> (block_m, block_n), resolved at trace time. GN==0: 1D GROUP_M
    super-row swizzle (block_m inner). GN>0: 2D band — N split into width-GN bands
    with GROUP_M inside each, keeping both A and B slabs L2-resident. Bijection."""
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


def make_row_band_resource(c_base, base_row, c_rows, c_cols, elem_bytes):
    """Buffer resource re-based at this workgroup's row band [base_row, c_rows), in
    64-bit ``index`` arith, so a 32-bit offset only spans the band (handles outputs
    whose flat M*N exceeds 2^31 / 4GB). base_row clamped to [0, c_rows] so a
    partial/fully-OOB last row tile bases 0 records (its stores drop).

    base/num_records are pinned to SGPRs via ``_readfirstlane_i32``: base_row is
    uniform across a tile's wave but the compiler's divergence analysis lands the
    SRD in VGPRs, waterfalling every buffer_store. Pinning collapses it to scalar
    regs (see ``_readfirstlane_i32`` / ``StoreCPerTensor``)."""
    elem = arith.index(elem_bytes)
    cols_i = _as_index(c_cols)
    row_i = _as_index(base_row)
    rows_i = _as_index(c_rows)
    row_c = arith.minui(row_i, rows_i)
    band_base = c_base + row_c * cols_i * elem
    # cap at 0x7FFFFFFF so a masked-out buffer_store (voffset=0x7FFFFFFF) is always OOB
    nrec = arith.minui((rows_i - row_c) * cols_i * elem, arith.index(0x7FFFFFFF))
    band_base_i64 = _readfirstlane_i32(arith.index_cast(T.i64, band_base))
    nrec_pinned = arith.index_cast(T.index, _readfirstlane_i32(arith.index_cast(T.i64, nrec)))
    return _buffer_ops.create_buffer_resource_from_addr(band_base_i64, num_records_bytes=nrec_pinned)


def _robust_time(launch, args, warmup=250, reps=5, iters=50):
    """Median-of-`reps` timing of launch(*args) after `warmup` iters.
    The long warmup reaches boost clock; short-K kernels mis-pick configs otherwise."""
    for _ in range(warmup):
        launch(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            launch(*args)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    ts.sort()
    return ts[len(ts) // 2]


# E8M0 scale preshuffle (FlyDSL, LDS-tiled): raw E8M0 [DIM,K//32] -> preshuffled int32.
# Tile by k: coalesced load of 64 rows x KT cols into LDS, coalesced dwordx4 store of the
# [KT,64,4] block (wave-lane transpose via LDS, both DRAM sides coalesced). n_tiles=4.
#
# The preshuffle is NOT a standalone launch: ``build_preshuffle_ab_kernel`` returns the
# bare @flyc.kernel so the mxfp8 GEMM can launch it + the gemm kernel from ONE @flyc.jit
# host stub (turbo-style single dispatch, scales repacked into a caller-owned workspace
# in stream order right before the gemm reads them -- no separate Python/launch dispatch).


def _lds_barrier():
    # Drain outstanding LDS writes (lgkmcnt) BEFORE the workgroup barrier, else
    # readers may observe stale LDS (a bare s_barrier doesn't wait on ds_write).
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


def _emit_lds_repack(is_a, grp, k0, tile, rin, rout, dim, K128, KT, tid, BLK, rd_base=0, wr_base=0, pack=1):
    # LDS-tiled transpose body (one workgroup, one (grp,k-chunk)). rd_base/wr_base
    # (default 0) shift the flat read/write offset to a group's slab (0 = dense).
    NT = 4
    TILE = 64 * KT
    assert KT % pack == 0 and TILE % BLK == 0 and ((KT // pack) * 64) % BLK == 0
    for i in range_constexpr(TILE // BLK):
        idx = tid + i * BLK
        rr = idx // KT
        kk = idx % KT
        gk = k0 + kk
        if is_a:
            grow = grp * 64 + rr  # A: rows grp*64 + (s*16+r)
        else:
            s = rr // 16  # B-comb: row = nblk*256 + wn*32 + OFF[s] + rinner
            off = (s % 2) * fx.Int32(16) + (s // 2) * fx.Int32(128)
            grow = (grp // 4) * 256 + (grp % 4) * 32 + off + (rr % 16)
        dw = _buffer_ops.buffer_load(
            rin, grow * K128 + gk + rd_base, vec_width=1, dtype=T.i32, mask=(gk < K128) & (grow < dim)
        )
        fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)).store(
            Vec.from_elements([fx.Int32(dw)], fx.Int32)
        )
    _lds_barrier()
    # Packed store: pack PACK consecutive K-iters into one output dword per lane (the
    # reader mirrors this via kk=k//PACK + MFMA op_sel). PACK=1 = unpacked.
    PACK = pack
    K128p = ceildiv(K128, PACK)
    NGP = KT // PACK  # packed groups produced per KT-chunk
    NOUTp = NGP * 64
    for j in range_constexpr(NOUTp // BLK):
        ol = tid + j * BLK
        kkp = ol // 64
        lane = ol % 64
        r = lane % 16
        sh = (lane // 16) * fx.Int32(8)
        gkp = (k0 // PACK) + kkp
        elems = []
        for s in range_constexpr(NT):
            packed = fx.Int32(0)
            for bb in range_constexpr(PACK):
                so = (s * 16 + r) * KT + (kkp * PACK + bb)
                val = Vec(
                    fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(so)), fx.make_layout(1, 1)).load()
                )
                b = (fx.Int32(val[0]) >> sh) & fx.Int32(0xFF)
                packed = packed | (b << fx.Int32(bb * 8))
            elems.append(packed)
        vec = Vec.from_elements(elems, fx.Int32)
        _buffer_ops.buffer_store(
            vec.ir_value(),
            rout,
            ((grp * K128p + gkp) * 64 + lane) * 4 + wr_base,
            mask=(k0 + kkp * PACK) < K128,
        )


def build_preshuffle_ab_kernel(K128: int, KT: int = _PRESHUF_KT, BLK: int = 256, pack: int = 1):
    """Build the fused A (layout 1) + B-comb (layout 3) scale-preshuffle @flyc.kernel.

    Returns ``(kern, n_kt)``. ``kern`` is a bare KernelFunction (NOT a launch): the
    mxfp8 GEMM factory calls it inside its own @flyc.jit so the preshuffle + gemm
    issue from a single host stub. One workgroup repacks one (group, KT-chunk) of
    raw E8M0 [DIM, K//32] (viewed int32 [DIM, K128]) into the broadcast int32 layout
    the gemm's ScaleS2R / ScaleBComb consume; region by block id ([0,a_blocks)->A,
    rest->B), bid being workgroup-uniform so the branch + its LDS barrier are
    divergence-free. n_kt = ceildiv(K128, KT) is the per-group block count; the
    caller sizes the grid as ``a_blocks + b_ngrp * n_kt``.
    """
    TILE = 64 * KT
    n_kt = ceildiv(K128, KT)
    K128p = ceildiv(K128, pack)  # packed K-groups (PACK scales / dword)

    @fx.struct
    class Smem:
        tile: fx.Array[fx.Int32, TILE, 16]

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        m: fx.Int32,
        n: fx.Int32,
        a_blocks: fx.Int32,
        a_ngrp: fx.Int32,
        b_ngrp: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile = fx.SharedAllocator().allocate(Smem).peek().tile
        rin_a = _buffer_ops.create_buffer_resource(a_raw, max_size=False, num_records_bytes=m * K128 * 4)
        rin_b = _buffer_ops.create_buffer_resource(b_raw, max_size=False, num_records_bytes=n * K128 * 4)
        rout_a = _buffer_ops.create_buffer_resource(
            a_sp, max_size=False, num_records_bytes=a_ngrp * K128p * 256 * 4
        )
        rout_b = _buffer_ops.create_buffer_resource(
            b_sp, max_size=False, num_records_bytes=b_ngrp * K128p * 256 * 4
        )
        if bid < a_blocks:
            _emit_lds_repack(
                True, bid // n_kt, (bid % n_kt) * KT, tile, rin_a, rout_a, m, K128, KT, tid, BLK, pack=pack
            )
        if bid >= a_blocks:
            bb = bid - a_blocks
            _emit_lds_repack(
                False, bb // n_kt, (bb % n_kt) * KT, tile, rin_b, rout_b, n, K128, KT, tid, BLK, pack=pack
            )

    return kern, n_kt


# ───────────────────────────────────────────────────────────────────────
# Reusable bf16 GEMM primitives for gemm_bf16_kernel.py (mfma 32x32x16 / 16x16x32,
# tr_b16 loaders, swizzle, store). bf16 counterpart of the fp8 block above.
# ───────────────────────────────────────────────────────────────────────
BLOCK_K = 64  # K depth per LDS tile (exported to the bf16 kernels)


def make_fp16_bf16_buffer_tensor(arg):
    return fx.rocdl.make_buffer_tensor(arg, max_size=False)


def compute_global_swizzle_bf16(lane_id, wave_id, K, n_rounds):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        col_byte = (lane_id % 8) * 16
        _, c = swizzle_128(row, col_byte)
        offsets.append(row * K + c // 2)
    return offsets


def compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, n_steps):
    offsets = []
    n_waves = fx.block_dim.x // 64
    kk = (lane_id % 32) // 2
    g = lane_id // 32
    n_in = g * 16 + (lane_id % 2) * 8
    for step in range_constexpr(n_steps):
        idx = wave_id + step * n_waves
        n_tile = idx // 4
        ks = idx % 4
        offsets.append((ks * 16 + kk) * c_n + n_tile * 32 + n_in)
    return offsets


def _packed_ds_read_tr16(base_ptr, byte_offsets):
    n = len(byte_offsets)
    v2i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    struct_t = _llvm.StructType.get_literal([v2i32] * n)
    asm = "\n".join(f"ds_read_b64_tr_b16 ${k}, ${n} offset:{byte_offsets[k]}" for k in range(n))
    constraints = ",".join(["=&v"] * n + ["v"] + ["~{memory}"])
    op = _llvm.InlineAsmOp(
        res=struct_t,
        operands_=[_raw(base_ptr)],
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )
    return [Vec(_llvm.extractvalue(v2i32, op.result, [k])).bitcast(fx.BFloat16) for k in range(n)]


def _read_tr16_sub(base_i32, sub16, row_off):
    """One tr16 sub-block (512 elems/block): packed double-read, the pair 128 bytes
    apart, at sub16*512 + row_off, then assembled."""
    ptr = _lds_ptr_from_i32(base_i32 + (sub16 * 512 + row_off) * 2)
    r0, r1 = _packed_ds_read_tr16(ptr, [0, 128])
    return r0.shuffle(r1, list(range(8)))


class _S2RLoaderBf16(_S2RLoaderBase):
    """Shared skeleton for the bf16 operand loaders (cf. _MfmaBf16): n_tiles output
    tiles, each a list of k-sub fragments. Subclasses supply the per-sub offset table
    and _tile(), which holds the LDS address math -- transposed ds_read_tr_b16 or
    swizzled buffer load (too different to share beyond this loop)."""

    def load(self, lds_src):
        return [self._tile(lds_src, i) for i in range_constexpr(self.n_tiles)]


class S2RLoaderTrBf16(_S2RLoaderBf16):
    """mfma_f32_32x32x16 operand via ds_read_tr_b16 transpose. Like S2RLoaderTr's
    _K_BASE, _SUB lists the tr16 sub-block of each inst_k=16 mfma step (consecutive
    here); its length is both the sub count and the per-tile block stride."""

    _SUB = (0, 1, 2, 3)

    def _tile(self, lds_src, i):
        m, kblk = self.lane_id % 32, self.lane_id // 32
        row_off = (m // 16) * 256 + kblk * 128 + (m % 16) * 4
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        sub0 = (self.wave_idx * self.n_tiles + i) * len(self._SUB)
        return [
            _read_tr16_sub(base_i32, sub0 + self._SUB[c], row_off) for c in range_constexpr(len(self._SUB))
        ]


class _MfmaBf16:
    """Grouped bf16 mfma: accumulate n_tiles_a x n_tiles_b output tiles. The k-sub
    count is taken from each operand's fragment list (len(a[i])), so the atom's
    (m, n, inst_k) is the only shape this class needs -- no BLOCK_K coupling."""

    def __init__(self, n_tiles_a, n_tiles_b, m, n, inst_k):
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(m, n, inst_k, fx.BFloat16))
        acc_len = m * n // 64  # f32 accum lanes per wave
        self.accum_type = Vec.make_type(acc_len, fx.Float32)
        self.zero_value = Vec.filled(acc_len, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                acc = c[self.idx(i, j)]
                for ks in range_constexpr(len(a[i])):
                    acc = fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a[i][ks], b[j][ks], acc)
                c[self.idx(i, j)] = acc
        return c


class Mfma32x32x16(_MfmaBf16):
    def __init__(self, n_tiles_a, n_tiles_b):
        super().__init__(n_tiles_a, n_tiles_b, 32, 32, 16)


class Mfma16x16x32(_MfmaBf16):
    def __init__(self, n_tiles_a, n_tiles_b):
        super().__init__(n_tiles_a, n_tiles_b, 16, 16, 32)


class S2RLoaderTr16x32Bf16(_S2RLoaderBf16):
    """mfma_f32_16x16x32 operand via ds_read_tr_b16 transpose. Like S2RLoaderTr's
    _K_BASE, _SUB lists the tr16 sub-block offset of each inst_k=32 mfma step (each
    spans two tr16 blocks); _BLOCK is the tr16 blocks per n32-region. Both baked in."""

    _SUB = (0, 2)
    _BLOCK = 4

    def _tile(self, lds_src, i):
        octet, mm = self.lane_id // 16, self.lane_id % 16
        s_in_pair, kb = octet // 2, octet % 2
        row_off = kb * 128 + mm * 4
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        orig_tile = self.wave_idx * self.n_tiles + i
        n32_block, g16 = orig_tile // 2, orig_tile % 2
        row = g16 * 256 + row_off  # fold g16 block offset into row
        return [
            _read_tr16_sub(base_i32, n32_block * self._BLOCK + self._SUB[c] + s_in_pair, row)
            for c in range_constexpr(len(self._SUB))
        ]


def _load8_bf16(lds_src, byte_off):
    i8 = fx.recast_iter(fx.Uint8, lds_src.ptr)
    p = fx.add_offset(i8, fx.make_int_tuple(byte_off))
    v = fx.make_view(p, fx.make_layout(16, 1)).load()
    return v.bitcast(fx.BFloat16)


class S2RLoaderBf16(_S2RLoaderBf16):
    """mfma_f32_32x32x16 operand (swizzled, non-transposed). Mirroring S2RLoaderTr,
    _K_BASE lists the K-column (elems) of each inst_k=16 sub of a 32-row tile; its
    length is the sub count -- no BLOCK_K needed."""

    _K_BASE = (0, 16, 32, 48)

    def _tile(self, lds_src, i):
        m, kblk = self.lane_id % 32, self.lane_id // 32
        row = self.wave_idx * (self.n_tiles * 32) + i * 32 + m
        subs = []
        for c in range_constexpr(len(self._K_BASE)):
            col_byte = (self._K_BASE[c] + kblk * 8) * 2
            _, cs = swizzle_128(row, col_byte)
            subs.append(_load8_bf16(lds_src, row * 128 + cs))
        return subs


class StoreCBf16:
    def __init__(self, C, c_rows, c_cols, out_ty, cache_modifier=0):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.out_ty = out_ty
        self.cache_modifier = cache_modifier
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)
        self.c_rsrc = (
            create_buffer_resource(C, max_size=False, num_records_bytes=c_nbytes) if cache_modifier else None
        )
        self.oob = fx.Int32(c_rows * c_cols)  # out-of-bounds sink index

    def _store_masked(self, value, c_index, valid):
        """Store one element to c_index (masked to the OOB sink when invalid)."""
        idx = arith.select(valid, c_index, self.oob)
        val = value.to(self.out_ty)
        if self.cache_modifier:
            buffer_store(val, self.c_rsrc, fx.Int32(idx), cache_modifier=self.cache_modifier)
        else:
            fx.memref_store_vec(Vec.filled(1, val, self.out_ty), self.reg_out_1)
            fx.copy(self.out_atom_1, self.reg_out_1, fx.slice(self.c_div, (None, fx.Int32(idx))))

    def store(self, c_frag, base_row, base_col):
        n = self.lane_id % 32
        m_hi = (self.lane_id // 32) * 4
        col = base_col + n
        col_valid = col < self.c_cols
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(16):
                row = base_row + ti * 32 + (r // 4) * 8 + m_hi + (r % 4)
                self._store_masked(acc[r], row * self.c_cols + col, col_valid)

    def store16(self, c_frag, base_row, base_col):
        n = self.lane_id % 16
        m_hi = (self.lane_id // 16) * 4
        col = base_col + n
        col_valid = col < self.c_cols
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(4):
                row = base_row + ti * 16 + m_hi + r
                self._store_masked(acc[r], row * self.c_cols + col, col_valid)

    def store_trans16(self, c_frag, group_idx, base_m, base_n, out_m, out_n):
        n = self.lane_id % 16
        m_hi = (self.lane_id // 16) * 4
        glob_n = base_n + n
        n_valid = glob_n < out_n
        row_base = (group_idx * out_n + glob_n) * out_m
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(4):
                m = base_m + ti * 16 + m_hi + r
                self._store_masked(acc[r], row_base + m, n_valid)
