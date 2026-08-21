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

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.compiler.ast_rewriter import (
    InsertEmptyYieldForSCFFor,
    ReplaceIfWithDispatch,
)
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.buffer_ops import buffer_store, create_buffer_resource
from flydsl.expr.typing import AddressSpace, PointerType, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def ceildiv_pow2(a, b: int):
    """``ceildiv(a, b)`` for a power-of-two ``b`` and a non-negative device value ``a``.
    Signed ``a // b`` lowers to arith.floordivsi (divide + remainder + sign fixup); the shift
    is one. Use on runtime hot-path values; plain ``ceildiv`` stays for host-side ints."""
    assert b > 0 and (b & (b - 1)) == 0
    return (a + (b - 1)) >> (b.bit_length() - 1)


def _u32(v):
    return _raw(fx.Int32(v) if isinstance(v, int) else v)


def udiv(a, b):
    """``a // b`` for device values proven non-negative (tile ids, group tile counts).
    Python ``//`` on a device Int32 is arith.floordivsi (magic multiply plus a remainder and
    sign fixup); the unsigned form is far cheaper on the exposed grouped tile-decode chain."""
    return ArithValue(arith.divui(_u32(a), _u32(b)))


def umod(a, b):
    """``a % b`` for device values proven non-negative; see ``udiv``."""
    return ArithValue(arith.remui(_u32(a), _u32(b)))


def uindex(v):
    """``arith.index_cast(T.index, v)`` for a device value proven non-negative (row/tile/group
    offsets). The signed cast sign-extends into every derived SRD base/extent; the unsigned cast
    zero-extends so the high half folds to zero."""
    return ArithValue(arith.index_castui(T.index, _raw(v)))


def resolve_accum_out(out, beta, shape, device, out_dtype):
    """Validate an optional caller-owned ``out``, or allocate one.

    ``beta=0.0`` overwrites and is the default; ``beta=1.0`` makes the epilogue
    accumulate, which only works against a buffer the caller supplies. The scalar
    store derives its element byte width from ``out_ty`` (2 for bf16/fp16, 4 for
    fp32), so an accumulate target may be bf16/fp16/fp32 -- it only has to be
    contiguous; a non-contiguous ``out`` would be silently replaced by a
    ``.contiguous()`` copy and the accumulation would land nowhere the caller can see.
    """
    assert beta in (0.0, 1.0), f"Only beta=0 (overwrite) or beta=1 (accumulate) supported, got {beta}"
    if out is None:
        assert beta == 0.0, "beta=1.0 requires an explicit `out` buffer to accumulate into"
        return torch.empty(shape, device=device, dtype=out_dtype)
    assert tuple(out.shape) == tuple(shape), f"out shape {tuple(out.shape)} must equal {tuple(shape)}"
    assert out.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"FlyDSL accumulate epilogue writes bf16/fp16/fp32; got out dtype {out.dtype}"
    )
    assert out.dtype == out_dtype, f"out dtype {out.dtype} must match out_dtype {out_dtype}"
    assert out.device == device, "out must be on the same device as the inputs"
    assert out.is_contiguous(), "out must be contiguous to accumulate in place"
    return out


def compile_with_scratch_out(launch, args, out_index=2):
    """``flyc.compile`` the launch, with the output argument pointed at a scratch buffer.

    Compilation builds its artifact by running the kernel once (see ``_compile_impl``),
    which an accumulate (beta=1) build would fold into the caller's buffer as a second
    GEMM. The artifact is keyed on shapes, not pointers, so compiling against a scratch
    of the same shape and then calling it on the real buffer runs exactly once.
    """
    scratch = torch.zeros_like(args[out_index])
    return flyc.compile(launch, *args[:out_index], scratch, *args[out_index + 1 :])


def floordiv_pow2(a, b: int):
    """``a // b`` for a power-of-two ``b``, shifting past what ``floordivsi`` would lower to."""
    assert b > 0 and (b & (b - 1)) == 0
    return a >> (b.bit_length() - 1)


_PRESHUF_KT = 16  # scale-preshuffle k-tile (rows*KT dwords staged in LDS per workgroup)


def scale_opsel(k, pack=1):
    return k % pack


def _as_index(v):
    # c_rows/c_cols may be a runtime value (dense/grouped NT/NN: N, m_end) or a
    # compile-time int (wgrad CShuffle: OUT_N). Coerce both to an MLIR index.
    return arith.index(v) if isinstance(v, int) else arith.index_cast(T.index, v)


def _i64(v):
    # widen an i32 runtime value to i64 (avoids overflow in worst-case base offsets)
    return ArithValue(arith.extsi(T.i64, _buffer_ops._unwrap_value(v)), signed=True)


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


def make_fp8_rebased_tensor_and_srd(arg_i8, fp8_ir_t, base_elems, num_records_bytes):
    """``make_fp8_buffer_tensor_rebased`` (the G2S prologue tensor) plus a matching raw SRD on
    the same int64-folded base for ``call_mxfp4_wholeloop``'s K-loop (bare rsrc + int32
    soffset). The mxfp4 GEMMs reach the operand both ways past 2^31 elems."""
    tensor = make_fp8_buffer_tensor_rebased(arg_i8, fp8_ir_t, base_elems, num_records_bytes)
    base = _readfirstlane_i32(
        arith.index_cast(T.i64, _buffer_ops.extract_base_index(arg_i8)) + arith.index_cast(T.i64, base_elems)
    )
    nr = arith.minui(arith.index_cast(T.index, num_records_bytes), arith.index(0xFFFFFFFF))
    nrec = arith.index_cast(T.index, _readfirstlane_i32(arith.index_cast(T.i64, nr)))
    srd = _buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=nrec)
    return tensor, srd


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


def lds_row_swizzle(row, chunks):
    """XOR key, in 16B chunks, for a `chunks`-wide LDS row: it spreads the 16B column one
    ds_read_b64_tr_b8 gathers across the gfx950 LDS banks. Shared by the g2s write side
    (swizzle_128) and the transpose read side (S2RLoaderTr._ptr_off) so the two cannot drift."""
    return ((row % 16) // 2) & (chunks - 1)


def swizzle_128(row, col, width=128):
    """XOR bank-swizzle over a `width`=2**k logical row (width=128 is byte-identical to the
    original fixed-128 form). The key only permutes the width//16 chunks inside one row, so
    the row index passes through unchanged."""
    return row, col ^ (lds_row_swizzle(row, width // 16) * 16)


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


def compute_global_swizzle_shear(lane_id, wave_id, K, n_rounds, m_row, ksm, up):
    """compute_global_swizzle(preshuffled=False) with every row's 128B fetch snapped to its
    enclosing cache line, for a row pitch whose ``K % 128 == ksm != 0`` (raw K-block straddles
    two lines). ``S2RLoaderShear`` reassembles each K-block from two consecutive windows."""
    assert ksm % 16 == 0 and 0 < ksm < 128
    mbias = (m_row * fx.Int32(ksm)) % fx.Int32(128)
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
        col = (lane_id % 8) * 16
        r, c = swizzle_128(row, col)
        sh = ((m_row + row) * fx.Int32(ksm)) % fx.Int32(128)
        disp = ((fx.Int32(128) - sh) % fx.Int32(128)) if const_expr(up) else (fx.Int32(0) - sh)
        offsets.append(r * K + c + mbias + disp)
    return offsets


def shear_mbias(m_row, ksm):
    """Byte bias that ``compute_global_swizzle_shear`` assumes was taken off the A SRD base.
    Unsheared halves of the same operand add it back to their own offsets."""
    return (m_row * fx.Int32(ksm)) % fx.Int32(128)


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
        # i64-traversal mode. None -> the K-offset rides the 32-bit soffset (caps the operand
        # span at < 2^32 fp8). A (arg_i8, fp8_ir_t, base_elems, num_records_bytes) tuple re-bases
        # the SRD per load instead (k_offset folds into the i64 base), lifting the cap.
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


class S2RLoaderShear(S2RLoader):
    """S2RLoader twin for an operand fetched with ``compute_global_swizzle_shear``: each
    K-block's bytes are split across window k and its partner (k-1 or k+1 per ``up``), both
    already resident. ``ksm == 64`` folds the splice into the XOR bank-swizzle key for free."""

    def __init__(self, wave_idx, n_tiles, m_row, ksm, up):
        super().__init__(wave_idx, n_tiles)
        assert ksm == 64 and up, "only the round-up 64B shear is implemented"
        self.m_row = m_row

    def load(self, lds_cur, lds_partner):
        delta = fx.Int32(fx.ptrtoint(lds_partner.ptr)) - fx.Int32(fx.ptrtoint(lds_cur.ptr))
        # n_tiles*16 and i*16 are even, so a fragment row's parity is the lane's parity.
        odd = ((self.m_row + self.lane_id) & fx.Int32(1)) == fx.Int32(1)
        sh = arith.select(odd, fx.Int32(64), fx.Int32(0))
        pdelta = arith.select(odd, delta, fx.Int32(0))
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            key = (lds_row_swizzle(row, 8) * 16) ^ sh
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                offset = row * 128 + (col ^ key)
                if const_expr(step == 0):
                    offset = offset + pdelta
                v = self._vec_load_16xf8(lds_cur, offset)
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

    def __init__(self, sp_tensor, dim, K, n_slabs=1, pack=1, k128p=None):
        self.K128 = K // 128  # number of K-groups (one i32 per K-iter)
        self.PACK = pack
        self.K128p = ceildiv(self.K128, self.PACK) if k128p is None else k128p
        self.lane = fx.thread_idx.x % 64
        # grp = (col//256)*4 + wn is block-strided, so the buffer holds cdiv(dim,256)*4
        # groups (matches the C++ preshuffle B sizing). A partial last 256-block reads
        # only its valid wn groups; OOB-col reads clamp to 0 and StoreC drops them.
        # dim%256==0 -> cdiv(dim,256)*4 == dim//64 (no change for aligned shapes).
        # n_slabs>1 (grouped): b_sp stacks G per-group slabs; load(slab=g) indexes group g.
        self.slab_elems = ((dim + 255) // 256) * 4 * self.K128p * 64 * 4  # i32 per group
        nbytes = self.slab_elems * n_slabs * 4  # int32 records
        self.rsrc = _buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k, kbase=0, slab=0):
        """base: sb_base0 (b0 region col base). ``kbase``: packed-K base of the group's own
        region (0 = one global packing). Returns 4 i32 (b0:0,1  b1:2,3)."""
        grp = (base // 256) * 4 + (base % 256) // 32
        idx = ((grp * self.K128p + k // self.PACK + kbase) * 64 + self.lane) * 4 + slab * self.slab_elems
        v = Vec(_buffer_ops.buffer_load(self.rsrc, idx, vec_width=4, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(4)]


class ScaleS2R:
    """Per-lane E8M0 scale loader for v_mfma_scale_f32_16x16x128 (preshuffled). The host
    pre-shuffles the raw E8M0 [DIM, K//32] so each wave's 64 lanes read 64 contiguous dwords
    with no per-lane ALU; the A-operand preshuffle comes from ``build_preshuffle_ab_kernel``."""

    def __init__(self, sp_tensor, dim, K, n_tiles, pack=1, k128p=None):
        self.K128 = K // 128  # number of K-groups (one i32 per K-iter)
        self.PACK = pack
        self.K128p = ceildiv(self.K128, self.PACK) if k128p is None else k128p
        self.n_tiles = n_tiles
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64  # == (lane//16)*16 + lane%16
        # cdiv (not floor): a non-group_span-multiple ``dim`` (general M) still needs the
        # partial last 64-row group resident so its valid rows read real scales; the
        # group's OOB rows were preshuffle-masked to 0 and StoreC drops their output.
        nbytes = ceildiv(dim, self.group_span) * self.K128p * 64 * n_tiles * 4  # int32 records
        self.rsrc = _buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k, kbase=0):
        """base: runtime global row/col base for this (region, wave). Returns n_tiles i32
        (packed dword for K-group kbase + k//PACK; caller selects byte k%PACK via MFMA
        op_sel). ``kbase``: packed-K base of the group's own region (0 = global packing)."""
        grp = base // self.group_span
        idx = ((grp * self.K128p + k // self.PACK + kbase) * 64 + self.lane) * self.n_tiles
        v = Vec(_buffer_ops.buffer_load(self.rsrc, idx, vec_width=self.n_tiles, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(self.n_tiles)]


# ── Reusable fp8 GEMM primitives (store, K-tail mask, value-attrs, AGPR MFMA, XCD
#    remap, LDS-ptr/transpose loaders, swizzle), shared by dense and grouped.


def _res_of(op):
    """Unwrap an op builder's single result (some rocdl builders already return one)."""
    return op.result if hasattr(op, "result") else op


def _readfirstlane_i32(v):
    """Force a wave-uniform-in-value i32 into an SGPR via s_readfirstlane.

    For grouped GEMM the output buffer descriptor's num_records = m_end*c_n*2
    is uniform across a tile's wave (all lanes share the group), but the
    compiler's divergence analysis treats m_end (from the per-tile group scan)
    as divergent -> the SRD lands in VGPRs -> every buffer_store_short is
    wrapped in a readfirstlane/saveexec waterfall loop. Pinning the value to
    SGPR collapses the SRD to scalar regs and drops the per-store waterfall."""
    return ArithValue(_res_of(rocdl.readfirstlane(res=_raw(v).type, src=_raw(v))))


# gfx9 DPP controls: ROW_SHR|n shifts right by n within a 16-lane row; ROW_BCAST15/31 feed a row's last lane into following rows.
_DPP_ROW_SHR = 0x110
_DPP_ROW_BCAST15 = 0x142
_DPP_ROW_BCAST31 = 0x143


def _dpp_add_i32(acc, ctrl, row_mask=0xF):
    """acc + DPP(acc, ctrl); masked-off and shifted-in lanes contribute 0."""
    raw = _raw(acc)
    r = rocdl.update_dpp(raw.type, _raw(fx.Int32(0)), raw, ctrl, row_mask, 0xF, True)
    return acc + ArithValue(_res_of(r))


def _wave_prefix_add_i32(v):
    """Wave64 inclusive add-scan of a per-lane i32 (lane l ends with the sum of 0..l). Six DPP
    steps replace the serial carry (bound_ctrl zeroes shifted-in lanes). Requires a full EXEC
    mask (kernel entry)."""
    for _sh in (1, 2, 4, 8):
        v = _dpp_add_i32(v, _DPP_ROW_SHR + _sh)
    v = _dpp_add_i32(v, _DPP_ROW_BCAST15, row_mask=0xA)
    return _dpp_add_i32(v, _DPP_ROW_BCAST31, row_mask=0xC)


def _readlane_i32(v, lane):
    """Broadcast one lane of a per-lane i32 into an SGPR; lane must be wave-uniform."""
    raw = _raw(v)
    return ArithValue(_res_of(rocdl.readlane(res=raw.type, src=raw, lane=lane)))


def _wave_count_le_i32(v, bound):
    """Number of lanes whose per-lane i32 is <= the wave-uniform bound. One ballot plus one
    s_bcnt1; on a monotone table this is the first lane above bound, an O(1) stand-in for a
    G-wide boundary compare chain."""
    m = _res_of(rocdl.ballot(res=ir.IntegerType.get_signless(64), pred=_raw(v <= bound)))
    n = _res_of(_llvm.intr_ctpop(m))
    return ArithValue(arith.trunci(T.i32, n))


def _lane_load_i32(rsrc, idx):
    """One per-lane i32 gather from a buffer resource; out-of-range lanes read 0."""
    return ArithValue(_buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=T.i32))


def _sload_i32(rsrc, idx):
    """One wave-uniform i32 read on the scalar path (``s_buffer_load`` into an SGPR). The value
    never enters the VGPR file, so a consumer waits on lgkmcnt and the read hits the scalar
    cache, not the g2s-evicted vL1D. (Raw intrinsic: buffer_load(is_scalar=) is not universal.)"""
    i32_t = ir.IntegerType.get_signless(32)
    rsrc_v4 = _llvm.bitcast(
        ir.VectorType.get([4], i32_t), _llvm.ptrtoint(ir.IntegerType.get_signless(128), _raw(rsrc))
    )
    args = [rsrc_v4, _raw(fx.Int32(idx * 4)), _raw(fx.Int32(0))]  # rsrc, byte offset, cache policy
    return ArithValue(_llvm.call_intrinsic(i32_t, "llvm.amdgcn.s.buffer.load.i32", args, [], []))


# SGPR-resident int32 table (entry i in its own SGPR): small-table twin of the lane-resident
# table. Lookup/prefix-sum are O(n_entries) SALU, so it pays only while the table is short --
# but keeps the whole group scan off the vector path (no gather, no DPP scan, no readlane).
def _sgpr_tbl_load(rsrc, n_entries, stride=1, first=0):
    """Read entries [0, n_entries) of an i32 buffer view into SGPRs; entry i is i32 element
    ``(i + first) * stride``. All loads are in flight under one lgkmcnt."""
    return [_sload_i32(rsrc, (i + first) * stride) for i in range_constexpr(n_entries)]


def _sgpr_tbl_scan(tbl):
    """Inclusive add-scan of an SGPR-resident table (entry i = sum of entries 0..i)."""
    out = []
    acc = tbl[0]
    out.append(acc)
    for v in tbl[1:]:
        acc = acc + v
        out.append(acc)
    return out


def _sgpr_tbl_pick(bounds, key, tables):
    """Decode ``key`` against the monotone boundary table ``bounds`` and return one entry per
    table in ``tables``, all from the owning group. One compare per boundary drives every
    ``s_cselect`` directly -- cheaper than materialising the group index first."""
    outs = [t[0] for t in tables]
    for g in range_constexpr(1, len(tables[0])):
        take = key >= bounds[g - 1]
        outs = [ArithValue(arith.select(take, t[g], o)) for t, o in zip(tables, outs)]
    return outs


# Lane-resident int32 table (entry i in lane i%64 of chunk i//64): avoids SGPR overflow past ~64 entries and LDS publish-barrier/ds_read; lookup = one v_readlane, prefix sum = one wave scan.
def _lane_tbl_load(rsrc, lane, n_entries, stride=1, first=0):
    """Gather entries [0, n_entries) of an i32 buffer view into lane-resident chunks. Entry i
    is read from i32 element ``(i + first) * stride``; lanes past the buffer bound read 0."""
    n_chunk = ceildiv(n_entries, 64)
    return [_lane_load_i32(rsrc, (lane + 64 * c + first) * stride) for c in range_constexpr(n_chunk)]


def _lane_tbl_scan(tbl):
    """Inclusive add-scan across a lane-resident table (chunk totals carried forward)."""
    out = []
    base = fx.Int32(0)
    for v in tbl:
        s = _wave_prefix_add_i32(v) + base
        out.append(s)
        base = _readlane_i32(s, 63)
    return out


def _lane_tbl_get(tbl, idx):
    """Entry ``idx`` (wave-uniform, or a Python int) of a lane-resident table."""
    if isinstance(idx, int):
        return _readlane_i32(tbl[idx // 64], idx % 64)
    v = _readlane_i32(tbl[0], idx)
    for c in range(1, len(tbl)):
        hit = idx >= fx.Int32(64 * c)
        v = arith.select(hit, _readlane_i32(tbl[c], idx - fx.Int32(64 * c)), v)
    return v


def _lane_tbl_count_le(tbl, bound):
    """Number of table entries <= ``bound``."""
    n = _wave_count_le_i32(tbl[0], bound)
    for c in range(1, len(tbl)):
        n = n + _wave_count_le_i32(tbl[c], bound)
    return n


class StoreCPerTensor:
    """Scalar output store: out = (acc [* a_scale * b_scale]).to(out_ty). Shared by the
    per-tensor and mxfp8 GEMMs; scales optional (folded by the scaled MMA when None). Output
    re-based per row band in 64-bit index; see col_safe/c_base/trans/beta_is_one on __init__."""

    stages_lds = False  # subclasses that borrow LDS need the caller to fence its owner off

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
        elem_fn=None,
        col_safe=False,
        store_aux=0,
        trans=False,
        c_base=None,
        beta_is_one=False,
        accum_mask=None,
    ):
        self.beta_is_one = beta_is_one
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.col_safe = col_safe
        # trans: transposed scalar store for the A/B-swapped wgrad swap_n boundary body (frag row=N, col=M, written C[m,n]); square OUT_M==OUT_N tiles only.
        self.trans = trans
        self.store_aux = store_aux
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        # Element byte width drives the row-band address arithmetic; fp32 accumulate
        # targets (bf16/fp16 gemm out -> fp32 main_grad) store 4-byte elements.
        self.out_bytes = 4 if out_ty is fx.Float32 else 2
        # Runtime predicate for the accumulate: the deep-K wgrad picks C vs the split scratch
        # on the SRD base at runtime, and only the C piece may add the read-back -- a banked
        # slice that added its scratch back would fold the previous launch's partial in.
        self.accum_mask = accum_mask
        # Optional f32->f32 epilogue node chain (bias/act), post-scale pre-cast.
        self.elem_fn = elem_fn
        self.scaled = A_scale is not None
        self.c_base = _buffer_ops.extract_base_index(C) if c_base is None else c_base  # byte base address
        if self.scaled:
            gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)  # 1 fp32
            gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)  # 1 fp32
            self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
            self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
            self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self._scale_v = None

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _scale(self):
        """a_scale*b_scale, loaded once per emitting block so the quadrant stores share one
        instance instead of re-issuing both scalar loads. Cached per MLIR block: sibling regions
        each get their own load, since a value in one does not dominate a use in another."""
        if not self.scaled:
            return None
        blk = ir.InsertionPoint.current.block
        if self._scale_v is None or self._scale_v[0] != blk:
            self._scale_v = (blk, self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div))
        return self._scale_v[1]

    def flush(self):
        """Emit whatever a subclass left queued; a store that lands in its own call has none."""

    def _row_col(self, ti, i, tj, base_col):
        """Element address of the value at fragment (ti, tj), row ``i`` of this lane's four."""
        return (ti * 16 + (self.lane_id // 16) * 4 + i) * self.c_cols + base_col + tj * 16 + self.lane_id % 16

    def _read_back(self, rsrc, ti, i, tj, base_row, base_col):
        """One beta=1 read-back load. ``trans`` swaps the fragment axes (row = N, col = M), so
        the element still lands at C[m, n] and the N bound moves with (ti, i) instead of tj."""
        if self.trans:
            n = base_row + ti * 16 + (self.lane_id // 16) * 4 + i  # base_row is the N origin here
            off = (tj * 16 + self.lane_id % 16) * self.c_cols + n
        else:
            n = base_col + tj * 16 + self.lane_id % 16
            off = self._row_col(ti, i, tj, base_col)
        return _buffer_ops.buffer_load(
            rsrc,
            off,
            vec_width=1,
            dtype=self.out_ty.ir_type,
            mask=None if self.col_safe else n < self.c_cols,
        )

    def prefetch(self, base_row, base_col):
        """Read C back in value order, so a subclass indexes it the way it indexes accumulators.
        Every load is in flight before the first store waits on one; masked-out lanes read OOB
        (0) and their store is dropped, so the read-back needs no extra bounds logic."""
        if not const_expr(self.beta_is_one):
            return None
        band_row = base_col if self.trans else base_row  # trans pins the band to M
        rsrc = make_row_band_resource(self.c_base, band_row, self.c_rows, self.c_cols, self.out_bytes)
        return [
            [
                [self._read_back(rsrc, ti, i, tj, base_row, base_col) for i in range_constexpr(4)]
                for tj in range_constexpr(self.n_tiles_b)
            ]
            for ti in range_constexpr(self.n_tiles_a)
        ]

    def _accum(self, val, prev, ti, tj, i):
        """Add the read-back before the cast, so the accumulate rounds exactly once."""
        if prev is None:
            return val
        add = self.out_ty(prev[ti][tj][i]).to(fx.Float32)
        if self.accum_mask is not None:
            add = fx.Float32(arith.select(_raw(self.accum_mask), _raw(add), _raw(fx.Float32(0.0))))
        return val + add

    def store(self, c_frag, base_row, base_col, prev=None):
        scale = self._scale()
        if const_expr(self.beta_is_one) and prev is None:
            prev = self.prefetch(base_row, base_col)
        if self.trans:
            return self._store_trans(c_frag, base_row, base_col, scale, prev)
        # buffer_store row-band path (int64-safe); the band SRD is pinned to SGPRs inside.
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, self.out_bytes)
        col0 = base_col + self.lane_id % 16
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            # One byte address per row: the fragment column step rides the store's 12-bit immediate.
            row_off = [((row_local + i) * self.c_cols + col0) * self.out_bytes for i in range_constexpr(4)]
            for tj in range_constexpr(self.n_tiles_b):
                col_valid = None if self.col_safe else (col0 + tj * 16) < self.c_cols
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                if self.scaled:
                    vec_f32 = vec_f32 * scale  # wave-uniform scale packs to v_pk_mul_f32
                for i in range_constexpr(4):
                    val = vec_f32[i]
                    if self.elem_fn is not None:
                        val = self.elem_fn(val)  # bias/act epilogue node chain
                    val = self._accum(val, prev, ti, tj, i).to(self.out_ty)
                    off = row_off[i] if tj == 0 else row_off[i] + tj * 16 * self.out_bytes
                    _buffer_ops.buffer_store(
                        val,
                        rsrc,
                        off,
                        mask=col_valid,
                        cache_modifier=self.store_aux,
                        offset_is_bytes=True,
                    )

    def _store_trans(self, c_frag, base_row, base_col, scale, prev=None):
        """Transposed twin of store() for the A/B-swapped wgrad boundary body: c_frag holds
        acc[n,m], base_row = N origin, base_col = M origin (band pinned to M rows). Same scalar
        buffer_store path and value math -- only the global address is transposed."""
        rsrc = make_row_band_resource(self.c_base, base_col, self.c_rows, self.c_cols, self.out_bytes)
        for ti in range_constexpr(self.n_tiles_a):
            n_local = ti * 16 + (self.lane_id // 16) * 4  # a-side -> N (col within band)
            for tj in range_constexpr(self.n_tiles_b):
                m_in_band = tj * 16 + self.lane_id % 16  # b-side -> M (row within band)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                if self.scaled:
                    vec_f32 = vec_f32 * scale
                for i in range_constexpr(4):
                    n = base_row + n_local + i
                    n_valid = None if self.col_safe else n < self.c_cols
                    val = vec_f32[i]
                    if self.elem_fn is not None:
                        val = self.elem_fn(val)
                    val = self._accum(val, prev, ti, tj, i).to(self.out_ty)
                    off = (m_in_band * self.c_cols + n) * self.out_bytes
                    _buffer_ops.buffer_store(
                        val, rsrc, off, mask=n_valid, cache_modifier=self.store_aux, offset_is_bytes=True
                    )


_DPP_QUAD_SWAP1 = 0xB1  # quad_perm:[1,0,3,2] -- exchange with the neighbouring lane
_PERM_LO_PAIR = 0x05040100  # {own low half, right neighbour's low half}
_PERM_HI_PAIR = 0x03020706  # {left neighbour's high half, own high half}


class StoreCPerTensorPairN(StoreCPerTensor):
    """A row's two n-fragments folded into one dword store, widening the request a fragment
    row's 16 lanes leave with; the pair comes from a lane and its DPP neighbour."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_tiles_b % 2 == 0, "the pair is written as a unit"
        lane16 = self.lane_id % 16
        hi = (lane16 % 2) > 0
        # even lane e holds columns (e, e+1); odd lane o holds (16 + o - 1, 16 + o).
        self.pair_col = arith.select(hi, lane16 + 15, lane16)
        self.pair_sel = arith.select(hi, fx.Int32(_PERM_HI_PAIR), fx.Int32(_PERM_LO_PAIR))

    def _pack(self, lo, hi):
        """(lo, hi) as one dword of out_ty; bf16 takes the single packed convert."""
        if const_expr(self.out_ty is fx.BFloat16):
            return rocdl.cvt_pk_bf16_f32(lo, hi)
        pair = Vec.from_elements([lo.to(self.out_ty), hi.to(self.out_ty)], self.out_ty)
        return arith._to_raw(pair.bitcast(fx.Int32)[0])

    def store(self, c_frag, base_row, base_col, prev=None):
        scale = self._scale()
        if const_expr(self.beta_is_one) and prev is None:
            prev = self.prefetch(base_row, base_col)
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        col0 = base_col + self.pair_col
        zero = arith._to_raw(fx.Int32(0))
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            vecs = [
                (Vec(c_frag[self.c_idx_fn(ti, tj)]) * scale)
                if self.scaled
                else Vec(c_frag[self.c_idx_fn(ti, tj)])
                for tj in range_constexpr(self.n_tiles_b)
            ]
            for i in range_constexpr(4):
                row_off = ((row_local + i) * self.c_cols + col0) * 2  # i32-small within band
                for p in range_constexpr(self.n_tiles_b // 2):
                    dcol = p * 32
                    lo, hi = vecs[2 * p][i], vecs[2 * p + 1][i]
                    if self.elem_fn is not None:
                        lo, hi = self.elem_fn(lo), self.elem_fn(hi)
                    lo = self._accum(lo, prev, ti, 2 * p, i)
                    hi = self._accum(hi, prev, ti, 2 * p + 1, i)
                    pk = self._pack(lo, hi)
                    sw = rocdl.update_dpp(pk.type, zero, pk, _DPP_QUAD_SWAP1, 0xF, 0xF, True)
                    pair_ok = None if self.col_safe else (col0 + dcol + 1) < self.c_cols
                    _buffer_ops.buffer_store(
                        rocdl.perm_b32(sw, pk, self.pair_sel),
                        rsrc,
                        row_off if dcol == 0 else row_off + dcol * 2,
                        mask=pair_ok,
                        cache_modifier=self.store_aux,
                        offset_is_bytes=True,
                    )


XPOSE_SLOT = 512  # bytes one staged transpose takes: four lane groups, a 128 B block each
XPOSE_SLOTS = 8  # slots the staging rotates over, so a slot is reused long after it retires
_XPOSE_LAG = 4  # runs left in flight before the oldest is drained and stored


def _wait_lgkmcnt_mem(n):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt lgkmcnt({n})",
        constraints="~{memory}",
        has_side_effects=True,
    )


def load_per_tensor_scale(A_scale, B_scale):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    vals = []
    for t in (A_scale, B_scale):
        g = fx.rocdl.make_buffer_tensor(t, max_size=False, num_records_bytes=4)  # 1 fp32
        div = fx.logical_divide(g, fx.make_layout(1, 1))
        fx.copy(atom, fx.slice(div, (None, fx.Int32(0))), reg)
        vals.append(Vec(fx.memref_load_vec(reg))[0])
    return vals[0] * vals[1]


class StoreCPerTensorLineN(StoreCPerTensorPairN):
    """PairN widened to a whole-line request per fragment row, via an LDS transpose whose
    ``lds_xpose`` scratch the caller must fence against whatever it is borrowed from."""

    stages_lds = True

    def __init__(self, *args, lds_xpose, scale, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_tiles_b % 4 == 0, "a line run gathers four n-fragments"
        assert self.col_safe, "a whole-line run has no column mask"
        lane16 = self.lane_id % 16
        quad = lane16 % 4
        self.line_col = quad * 16 + (lane16 // 4) * 4
        self.scale = scale
        blk = (self.lane_id // 16) * 128
        self._xp_wr = _lds_ptr_from_i32(lds_xpose + blk + quad * 32 + (lane16 // 4) * 8)
        self._xp_rd = _lds_ptr_from_i32(lds_xpose + blk + lane16 * 8)
        self._slot = 0
        self._pend = []

    def _xpose(self, pk):
        off = self._slot * XPOSE_SLOT
        self._slot = (self._slot + 1) % XPOSE_SLOTS
        return _lds_xpose_tr16(self._xp_wr, self._xp_rd, Vec.from_elements(pk, fx.Int32), off)

    def _retire(self):
        rsrc, off, run = self._pend.pop(0)
        _wait_lgkmcnt_mem(2 * len(self._pend))  # a write and a read still queued per pending run
        _buffer_ops.buffer_store(
            run.bitcast(self.out_ty),
            rsrc,
            off,
            cache_modifier=self.store_aux,
            offset_is_bytes=True,
        )

    def flush(self):
        while self._pend:
            self._retire()

    def store(self, c_frag, base_row, base_col, prev=None, tap=None):
        scale = self.scale
        if const_expr(self.beta_is_one) and prev is None:
            prev = self.prefetch(base_row, base_col)
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        col0 = base_col + self.line_col
        for ti in range_constexpr(self.n_tiles_a):
            if tap is not None:
                tap()
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            vecs = [
                (Vec(c_frag[self.c_idx_fn(ti, tj)]) * scale)
                if self.scaled
                else Vec(c_frag[self.c_idx_fn(ti, tj)])
                for tj in range_constexpr(self.n_tiles_b)
            ]
            for i in range_constexpr(4):
                if tap is not None:
                    tap()
                row_off = ((row_local + i) * self.c_cols + col0) * 2  # i32-small within band
                for g in range_constexpr(self.n_tiles_b // 4):
                    el = [vecs[4 * g + f][i] for f in range_constexpr(4)]
                    if self.elem_fn is not None:
                        el = [self.elem_fn(v) for v in el]
                    el = [self._accum(v, prev, ti, 4 * g + f, i) for f, v in enumerate(el)]
                    run = self._xpose([self._pack(el[0], el[1]), self._pack(el[2], el[3])])
                    self._pend.append((rsrc, row_off if g == 0 else row_off + g * 128, run))
                    if len(self._pend) > _XPOSE_LAG:
                        self._retire()


class StoreCPerTensorQuadN(StoreCPerTensorPairN):
    """PairN for a kernel whose n-fragments arrived column-interleaved, so a lane already holds
    the adjacent columns and one unmasked store folds them with no cross-lane step. The caller
    owns the interleave; the tile must be column-safe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_tiles_b in (4, 8), "the interleave folds four or eight n-fragments"
        assert self.col_safe, "a partial fold has no column mask"

    def _row_col(self, ti, i, tj, base_col):
        """The fold hands this lane a whole run of columns, so tj steps by one, not by a tile."""
        row = ti * 16 + (self.lane_id // 16) * 4 + i
        return row * self.c_cols + base_col + (self.lane_id % 16) * self.n_tiles_b + tj

    def store(self, c_frag, base_row, base_col, prev=None):
        scale = self._scale()
        if const_expr(self.beta_is_one) and prev is None:
            prev = self.prefetch(base_row, base_col)
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        col0 = base_col + (self.lane_id % 16) * self.n_tiles_b
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            vecs = [
                (Vec(c_frag[self.c_idx_fn(ti, tj)]) * scale)
                if self.scaled
                else Vec(c_frag[self.c_idx_fn(ti, tj)])
                for tj in range_constexpr(self.n_tiles_b)
            ]
            for i in range_constexpr(4):
                el = [v[i] if self.elem_fn is None else self.elem_fn(v[i]) for v in vecs]
                el = [self._accum(v, prev, ti, tj, i) for tj, v in enumerate(el)]
                dw = [self._pack(el[2 * h], el[2 * h + 1]) for h in range_constexpr(self.n_tiles_b // 2)]
                _buffer_ops.buffer_store(
                    Vec.from_elements(dw, fx.Int32).bitcast(self.out_ty),
                    rsrc,
                    ((row_local + i) * self.c_cols + col0) * 2,  # i32-small within band
                    cache_modifier=self.store_aux,
                    offset_is_bytes=True,
                )


class StoreCPerTensorCShuffle:
    """CShuffle output store: same value->global-address mapping as StoreCPerTensor
    (byte-identical) but stages each 16-row sub-tile through per-wave LDS, re-reads it
    N-contiguous, and emits one vectorized 128b global store per lane. Assumes BLOCK_N=256."""

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
        pipe=False,
        store_aux=0,
        double_buffer=None,
        beta_is_one=False,
    ):
        # ``double_buffer`` is the tensorwise-NT caller's alias for ``pipe`` (staging double-buffer).
        if double_buffer is not None:
            pipe = double_buffer
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.wave_id = wave_id
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        self.beta_is_one = beta_is_one
        # Element byte width from out_ty: fp32=4, bf16/fp16=2. Everything below (LDS byte
        # offsets, the read-back _band pitch, and how many elements pack a 128b store) is
        # derived from it, so the same staging serves an fp32 fused-accum target.
        self.out_b = 4 if out_ty is fx.Float32 else 2
        self.Cc = n_tiles_b * 16  # columns in one 16-row shuffle tile
        self.EPL = (16 * self.Cc) // 64  # out_ty elements each lane re-reads (16*Cc rows/cols / 64 lanes)
        # One coalesced global store is 128 bits = 16/out_b elements (buffer_store_dwordx4):
        # 8 x bf16 or 4 x fp32. A lane re-reading more than that (EPL > elems_per_store) emits
        # EPL//elems_per_store back-to-back 128b stores. EPL must be a multiple of it and fit
        # in one row.
        self.elems_per_store = 16 // self.out_b  # elements packed into one 128b vector store
        assert self.EPL % self.elems_per_store == 0 and self.EPL <= self.Cc, (
            f"CShuffle expects EPL a multiple of {self.elems_per_store} within Cc={self.Cc}; got EPL={self.EPL}"
        )
        # The ds_write_b16 staging + 128b re-read aliases LDS banks; row_pad is an opt-in fix.
        # The caller must then size C_lds_shuffle as n_waves*16*(n_tiles_b*16 + row_pad), not
        # just n_tiles_b*16, or this overflows the buffer.
        self.row_stride = self.Cc + row_pad
        self.row_pad = row_pad
        self.wave_lds_elems = 16 * self.row_stride  # per-wave staging (one 16-row tile)
        # pipe=True double-buffers the staging region (parity ti%2) so ti+1's ds_write
        # overlaps ti's ds_read + store; needs 2x LDS (caller sizes C_lds_shuffle 2*_cshuf_n).
        self.pipe = pipe
        # Non-temporal aux immediate for the C store (keeps write-once C out of L2). 0 = default.
        self.store_aux = store_aux
        self.wave_stride = self.wave_lds_elems * (2 if pipe else 1)
        self.c_lds = c_lds
        # C addressed via i64 per-band re-basing (handles OUT_M*OUT_N > 2^31 / >4GB);
        # the final 128b store re-bases at each 16-row sub-tile band (see store()).
        self.c_base = _buffer_ops.extract_base_index(C)
        # A_scale/B_scale None -> scale already folded into the accumulator by the scaled MMA (mxfp8), so the store is plain.
        self.scaled = A_scale is not None
        if self.scaled:
            gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)
            gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)
            self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
            self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
            self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        # addr-space 2 (LDS), mirroring G2SLoader.LdsPtr_t. Separate scalar-store
        # (align = element width: b16 for 16-bit, b32 for fp32) and vector-read
        # (align 16 = one 128b run) pointer types.
        self._store_ptr_t = fx.PointerType.get(out_ty.ir_type, 2, self.out_b)
        self._read_ptr_t = fx.PointerType.get(out_ty.ir_type, 2, 16)

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _band(self, base_row, ti, cols_i, rows_i, out_b):
        """Output SRD re-based (i64) at the 16-row band this sub-tile writes."""
        band_row = arith.index_cast(T.index, base_row + ti * 16)
        row_c = arith.minui(band_row, rows_i)
        band_base = self.c_base + row_c * cols_i * arith.index(out_b)
        nrec = arith.minui((rows_i - row_c) * cols_i * arith.index(out_b), arith.index(0x7FFFFFFF))
        band_base_i64 = _readfirstlane_i32(arith.index_cast(T.i64, band_base))
        nrec_pinned = arith.index_cast(T.index, _readfirstlane_i32(arith.index_cast(T.i64, nrec)))
        return _buffer_ops.create_buffer_resource_from_addr(band_base_i64, num_records_bytes=nrec_pinned)

    def _runs(self, base_col):
        """(in-tile column, global column, valid) per 128b run, in store order."""
        row_in = (self.lane_id * self.EPL) // self.Cc
        col0 = (self.lane_id * self.EPL) % self.Cc
        runs = []
        for sub in range_constexpr(self.EPL // self.elems_per_store):
            col_in = col0 + sub * self.elems_per_store
            gcol = base_col + col_in
            runs.append((col_in, gcol, (gcol + fx.Int32(self.elems_per_store)) <= self.c_cols))
        return row_in, runs

    def prefetch(self, base_row, base_col):
        """Issue the beta=1 read-back for every sub-tile up front; ``store`` consumes it.
        One sub-tile owns only EPL//8 runs, far too few to cover HBM latency alone, so all
        of it goes out before the first staging round. Flat order (ti, run)."""
        if not const_expr(self.beta_is_one):
            return None
        cols_i = _as_index(self.c_cols)
        rows_i = _as_index(self.c_rows)
        row_in, runs = self._runs(base_col)
        prev = []
        for ti in range_constexpr(self.n_tiles_a):
            rsrc_i = self._band(base_row, ti, cols_i, rows_i, self.out_b)
            for _col_in, gcol, valid in runs:
                prev.append(
                    Vec(
                        _buffer_ops.buffer_load(
                            rsrc_i,
                            row_in * self.c_cols + gcol,
                            vec_width=self.elems_per_store,
                            dtype=self.out_ty.ir_type,
                            mask=valid,
                        )
                    )
                )
        return prev

    def store(self, c_frag, base_row, base_col, prev=None):
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div) if self.scaled else None
        if const_expr(self.beta_is_one) and prev is None:
            prev = self.prefetch(base_row, base_col)
        lds_base = fx.Int32(fx.ptrtoint(self.c_lds.ptr))
        wave_base = self.wave_id * self.wave_stride  # base of this wave's region(s)
        out_b = self.out_b  # element bytes (fp32=4, bf16/fp16=2)
        cols_i = _as_index(self.c_cols)
        rows_i = _as_index(self.c_rows)
        n_runs = self.EPL // self.elems_per_store

        def _write_ti(ti, roff):
            # stage this 16-row sub-tile row-major into the given per-wave LDS region
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                lds_col = tj * 16 + self.lane_id % 16
                for i in range_constexpr(4):
                    lds_row = (self.lane_id // 16) * 4 + i
                    e = roff + lds_row * self.row_stride + lds_col
                    val = (vec_f32[i] * scale if self.scaled else vec_f32[i]).to(self.out_ty)
                    ptr = fx.inttoptr(self._store_ptr_t, lds_base + e * out_b)
                    ptr.store(val)

        def _read_store_ti(ti, roff):
            # Re-base output at this 16-row band (i64), re-read N-contiguous (one EPL-col
            # run/lane) + one 128b store at a small in-band i32 offset; band num_records OOB-drops.
            rsrc = self._band(base_row, ti, cols_i, rows_i, out_b)
            row_in, runs = self._runs(base_col)
            for sub, (col_in, gcol, valid) in enumerate(runs):
                lane_e = roff + row_in * self.row_stride + col_in
                rptr = fx.inttoptr(self._read_ptr_t, lds_base + lane_e * out_b)
                vec = fx.make_view(rptr, fx.make_layout(self.elems_per_store, 1)).load()
                if const_expr(self.beta_is_one):
                    # the staged value is already out_ty, so this widen-add-round is the
                    # second rounding; the scalar path (StoreCPerTensor) keeps its single one.
                    vec = (Vec(vec).to(fx.Float32) + prev[ti * n_runs + sub].to(fx.Float32)).to(self.out_ty)
                off = (row_in * self.c_cols + gcol) * out_b  # i32-small within band
                _buffer_ops.buffer_store(
                    vec, rsrc, off, mask=valid, cache_modifier=self.store_aux, offset_is_bytes=True
                )

        if const_expr(self.pipe and self.n_tiles_a > 1):
            # Depth-2 softpipe over ti: prefetch ti+1's ds_write, store ti, drain in WPT
            # steps (lgkmcnt oldest-first) so ti+1's prefetched writes stay in flight.
            WPT = self.n_tiles_b * 4  # ds_write_b16 per ti (n_tiles_b tj * 4 rows)

            def _roff(ti):
                return wave_base + (ti % 2) * self.wave_lds_elems

            _write_ti(0, _roff(0))
            for ti in range_constexpr(self.n_tiles_a):
                if const_expr(ti + 1 < self.n_tiles_a):
                    _write_ti(ti + 1, _roff(ti + 1))
                    S2RLoaderTr._wait_lgkmcnt(WPT)  # ti writes done; keep ti+1's WPT
                    _read_store_ti(ti, _roff(ti))
                    S2RLoaderTr._wait_lgkmcnt(WPT)  # ti read done; region free for ti+2
                else:
                    S2RLoaderTr._wait_lgkmcnt(0)  # last: drain final writes
                    _read_store_ti(ti, _roff(ti))
                    S2RLoaderTr._wait_lgkmcnt(0)
        else:
            for ti in range_constexpr(self.n_tiles_a):
                _write_ti(ti, wave_base)
                S2RLoaderTr._wait_lgkmcnt(0)
                _read_store_ti(ti, wave_base)
                S2RLoaderTr._wait_lgkmcnt(0)  # drain re-read before next ti overwrites LDS


def _store_quadrants(store_c, c00, c01, c10, c11, base_row, base_col, LDS_BLOCK_M, LDS_BLOCK_N):
    """Store the four accumulator quadrants at their (row, col) sub-tile offsets.

    A beta=1 epilogue read-back is software-pipelined by one quadrant: quadrant i+1's
    loads go out before quadrant i stores, so they overlap without keeping all four
    quadrants' worth of loaded values live at once. The 4-wave wgrad runs at occupancy
    1, so it has neither a co-resident wave to hide the latency nor the registers to
    spare for prefetching everything up front.
    """
    quads = (
        (c00, base_row + 0, base_col + 0),
        (c01, base_row + 0, base_col + LDS_BLOCK_N),
        (c10, base_row + LDS_BLOCK_M, base_col + 0),
        (c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N),
    )
    nxt = store_c.prefetch(quads[0][1], quads[0][2])
    for i, (frag, r, c) in enumerate(quads):
        cur = nxt
        if i + 1 < len(quads):
            nxt = store_c.prefetch(quads[i + 1][1], quads[i + 1][2])
        store_c.store(frag, r, c, prev=cur)


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
    """fp8 16x16x128 MFMA via inline asm, to pin the dst register class. mode "2" (=a,v,v,0):
    accumulator in AGPR, freeing the VGPR file; mode "3" (=v,v,v,0): VGPR in-place (D=C, no
    accvgpr shuffle); mode "1" (=&v,v,v,0): VGPR early-clobber."""
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


def xcd_remap_pid_u(pid, total_pids, num_xcd):
    """``xcd_remap_pid`` on ids proven non-negative: same bijection, unsigned divides (a mask
    and a shift for pow2 num_xcd, vs a floor-div fixup each on the signed path). A separate
    entry point so the signed callers keep byte-identical ISA."""
    if num_xcd <= 1:
        return pid
    per_xcd = udiv(total_pids, num_xcd)  # floor
    rem = total_pids - per_xcd * num_xcd
    xcd = umod(pid, num_xcd)
    local = udiv(pid, num_xcd)
    offset = xcd * per_xcd + ArithValue(arith.minui(_u32(xcd), _u32(rem)))
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
    """LDS -> mfma operand wave-coop transpose load via ds_read_b64_tr_b8. Serves K-major fp8
    operands (NN B, TN A and B, whose mfma operand byte layouts are identical); the operand is
    selected by tile_stride and the WG wave count n_waves. See _ptr_off for the map."""

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
        """wave_idx: this wave's index along the transposed coord; tile_stride: per-wave
        coverage; chunk_stride must match the G2S writer. inline_asm issues opaque-asm reads
        (caller drains via vmcnt_hint). width: LDS column span, MUST match compute_global_swizzle_nn."""
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
        # swz_K reproduces swizzle_128's own key over this buffer's chunk count.
        swz_K = lds_row_swizzle(K_log, chunks) * 16
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

    def _issue_one(self, lds_src, tile_i, base_off=None, vmcnt=None):
        """Issue the 4 ds_read_b64_tr_b8 of one tile (no drain, no assemble).
        Returns the 4 raw v2i32 Vec. vmcnt overrides the instance g2s drain hint."""
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
            _vm = self.vmcnt_hint if vmcnt is None else vmcnt
            r02 = _packed_ds_read_tr_offsets(p0, [0, RS], vmcnt_hint=_vm)
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

    def load(self, lds_src, preshuffled=False, drain=True, base_off=None, vmcnt=None):
        """Return all n_tiles operand frags. Inline-asm path issues every tile's
        async reads then one trailing lgkmcnt(0) before the consuming mfma;
        drain=False skips it when a later drain covers these reads. The intrinsic
        path lets the backend insert the wait. base_off = runtime LDS-stage byte
        offset (double-buffer parity). vmcnt overrides the instance g2s drain hint
        for this call (-1 = none, when a caller-side rendezvous already covers it)."""
        assert not preshuffled, "S2RLoaderTr does not support preshuffled"
        if self.inline_asm:
            all_calls = [self._issue_one(lds_src, t, base_off, vmcnt) for t in range_constexpr(self.n_tiles)]
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
    """Buffer resource re-based at this workgroup's row band [base_row, c_rows) in 64-bit index
    arith, so a 32-bit offset only spans the band (handles M*N > 2^31). base/num_records are
    pinned to SGPRs via ``_readfirstlane_i32`` to stop the SRD waterfalling every buffer_store."""
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


def make_row_band_resource_div(c_base, base_row, c_rows, c_cols, elem_bytes):
    """Divergent-``base_row`` variant of ``make_row_band_resource`` (each lane owns a distinct
    row): same 64-bit re-base but the SRD base is a VGPR so the buffer op WATERFALLS. Use only
    when ``base_row`` is genuinely per-lane; base_row past c_rows -> 0 records (drop)."""
    elem = arith.index(elem_bytes)
    cols_i = _as_index(c_cols)
    row_i = _as_index(base_row)
    rows_i = _as_index(c_rows)
    row_c = arith.minui(row_i, rows_i)
    band_base = c_base + row_c * cols_i * elem
    nrec = arith.minui((rows_i - row_c) * cols_i * elem, arith.index(0x7FFFFFFF))
    band_base_i64 = arith.index_cast(T.i64, band_base)
    return _buffer_ops.create_buffer_resource_from_addr(band_base_i64, num_records_bytes=nrec)


def run_compiled(cache: dict, key, launch, *args):
    """``flyc.compile`` the launch once per ``key``, then call the compiled object.

    Calling a ``@flyc.jit`` launch directly re-pays its dispatch on every launch: signature bind over
    the launch's arguments, cache-key build from the bound values, and a globals-drift check. That is
    host time on the critical path, and it is not small -- for the fp8 combine's ~24-argument launch
    it measures 180 us per call against 37 us through the compiled object, and a training step makes
    that call twice per MoE layer. ``flyc.compile`` resolves all of it once; the object it returns
    only forwards to the call state.

    The key belongs to the caller because it cannot be derived here. It has to cover everything the
    compiled artifact depends on, which is the launch's own identity (usually an ``lru_cache``d
    builder, but not always -- some callers build the closure per call) PLUS any argument value the
    compile bakes in, such as an ``M`` that is not part of the builder's key.
    """
    compiled = cache.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        cache[key] = compiled
    compiled(*args)


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


def emit_if_then(cond, then_fn):
    """Emit a dynamic ``if cond: then_fn()`` (the body-only AST rewrite's primitive).

    ``then_fn`` must take no arguments, and closing over a loop variable in it is unsafe unless the
    call is immediate -- build a zero-arg emitter per iteration instead."""
    ReplaceIfWithDispatch.scf_if_dispatch(cond, then_fn)


def emit_for(stop, body):
    """Emit a dynamic ``for i in range(stop): body(i)`` with ``i`` as a signed Int32."""
    InsertEmptyYieldForSCFFor.scf_for_dispatch(
        fx.Int32(0), stop, fx.Int32(1), lambda iv, _names: body(fx.arith.ArithValue(iv, signed=True))
    )


def _emit_lds_repack(
    is_a,
    grp,
    k0,
    tile,
    rin,
    rout,
    dim,
    K128,
    KT,
    tid,
    BLK,
    rd_base=0,
    wr_base=0,
    pack=1,
    kbound=None,
    k128p=None,
):
    # LDS-tiled transpose body (one workgroup, one (grp,k-chunk)). rd_base/wr_base
    # (default 0) shift the flat read/write offset to a group's slab (0 = dense).
    # kbound (default K128) bounds this chunk's k index; k128p (default ceildiv(K128,pack)) is the output k-stride, so the variable-K wgrad packs each group from its own k0.
    NT = 4
    TILE = 64 * KT
    assert KT % pack == 0 and TILE % BLK == 0 and ((KT // pack) * 64) % BLK == 0
    KBND = K128 if kbound is None else kbound
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
            rin,
            grow * K128 + gk + rd_base,
            vec_width=1,
            dtype=T.i32,
            mask=(gk < KBND) & (grow < dim),
        )
        fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)).store(
            Vec.from_elements([fx.Int32(dw)], fx.Int32)
        )
    _lds_barrier()
    # Packed store: pack PACK consecutive K-iters into one output dword per lane (the
    # reader mirrors this via kk=k//PACK + MFMA op_sel). PACK=1 = unpacked.
    PACK = pack
    K128p = ceildiv(K128, PACK) if k128p is None else k128p
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
            mask=(k0 + kkp * PACK) < KBND,
        )


def build_preshuffle_ab_kernel(K128: int, KT: int = _PRESHUF_KT, BLK: int = 256, pack: int = 1):
    """Build the fused A (layout 1) + B-comb (layout 3) scale-preshuffle @flyc.kernel. Returns
    ``(kern, n_kt)`` where kern is a bare KernelFunction the mxfp8 GEMM factory calls inside its
    own @flyc.jit; one workgroup repacks one (group, KT-chunk) into the layout ScaleS2R consumes."""
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


# Reusable bf16 GEMM primitives for the {gemm,grouped_gemm}_bf16_kernel.py pair (mfma
# 32x32x16 / 16x16x32, tr_b16 loaders, swizzle, store). bf16 counterpart of the fp8 block.
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


def _lds_xpose_tr16(wr_ptr, rd_ptr, val, byte_offset=0):
    """Stage one lane's 8 B and take the 16-lane group's 128 B back transposed, async on lgkmcnt."""
    v2i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    op = _llvm.InlineAsmOp(
        res=v2i32,
        operands_=[_raw(wr_ptr), _raw(rd_ptr), _raw(val)],
        asm_string=(
            f"ds_write_b64 $1, $3 offset:{byte_offset}\nds_read_b64_tr_b16 $0, $2 offset:{byte_offset}"
        ),
        constraints="=&v,v,v,v,~{memory}",
        has_side_effects=True,
    )
    return Vec(op.result).bitcast(fx.BFloat16)


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
        # Rebuild a rank-1 view: store indices are linear, so C's host rank must not leak in.
        c_ptr_ty = PointerType.get(elem_ty=out_ty.ir_type, address_space=AddressSpace.Global, alignment=16)
        c_base = ArithValue(arith.index_cast(T.i64, _buffer_ops.extract_base_index(C)), signed=True)
        c_lin = fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, c_base), fx.make_layout(c_rows * c_cols, 1)))
        gC = fx.rocdl.make_buffer_tensor(c_lin, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)
        self.c_rsrc = (
            create_buffer_resource(c_lin, max_size=False, num_records_bytes=c_nbytes)
            if cache_modifier
            else None
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


# MXFP4 kernel helpers: lane-table group search (with its DPP scan primitives),
# power-of-two ceildiv/floordiv, and the XCD band remap.


def xcd_band_remap_pid(pid, total_pids, num_xcd, band):
    """Band-cyclic variant of ``xcd_remap_pid``: an XCD owns every ``num_xcd``-th run of
    ``band`` tiles, keeping intra-run L2 reuse while every XCD samples the full tile range.
    That balances a non-uniform per-tile cost the contiguous remap would strand on one XCD."""
    if num_xcd <= 1 or band <= 0:
        return pid
    assert num_xcd & (num_xcd - 1) == 0
    span = num_xcd * band
    local = floordiv_pow2(pid, num_xcd)
    xcd = pid - local * num_xcd
    rnd = local // band
    mapped = (rnd * num_xcd + xcd) * band + (local - rnd * band)
    return arith.select(pid < (total_pids // span) * span, mapped, pid)


def _lane_tbl_get(tbl, idx):
    """Entry ``idx`` (wave-uniform, or a Python int) of a lane-resident table."""
    if isinstance(idx, int):
        return _readlane_i32(tbl[idx // 64], idx % 64)
    v = _readlane_i32(tbl[0], idx)
    for c in range(1, len(tbl)):
        hit = idx >= fx.Int32(64 * c)
        v = arith.select(hit, _readlane_i32(tbl[c], idx - fx.Int32(64 * c)), v)
    return v


def _lane_tbl_load(rsrc, lane, n_entries, stride=1, first=0):
    """Gather entries [0, n_entries) of an i32 buffer view into lane-resident chunks.
    Entry i reads i32 element ``(i + first) * stride``; lanes past the buffer bound read 0."""
    n_chunk = ceildiv(n_entries, 64)
    return [_lane_load_i32(rsrc, (lane + 64 * c + first) * stride) for c in range_constexpr(n_chunk)]


def _lane_tbl_scan(tbl):
    """Inclusive add-scan across a lane-resident table (chunk totals carried forward)."""
    out = []
    base = fx.Int32(0)
    for v in tbl:
        s = _wave_prefix_add_i32(v) + base
        out.append(s)
        base = _readlane_i32(s, 63)
    return out


def _lane_tbl_count_le(tbl, bound):
    """Number of table entries <= ``bound``."""
    n = _wave_count_le_i32(tbl[0], bound)
    for c in range(1, len(tbl)):
        n = n + _wave_count_le_i32(tbl[c], bound)
    return n


def _readlane_i32(v, lane):
    """Broadcast one lane of a per-lane i32 into an SGPR; lane must be wave-uniform."""
    raw = _raw(v)
    return ArithValue(_res_of(rocdl.readlane(res=raw.type, src=raw, lane=lane)))


def _wave_count_le_i32(v, bound):
    """Number of lanes whose per-lane i32 is <= the wave-uniform bound. One ballot plus one
    s_bcnt1; on a monotone table this is the first lane above bound, an O(1) stand-in for a
    G-wide boundary compare chain."""
    m = _res_of(rocdl.ballot(res=ir.IntegerType.get_signless(64), pred=_raw(v <= bound)))
    n = _res_of(_llvm.intr_ctpop(m))
    return ArithValue(arith.trunci(T.i32, n))


def _res_of(op):
    """Unwrap an op builder's single result (some rocdl builders already return one)."""
    return op.result if hasattr(op, "result") else op


def _lane_load_i32(rsrc, idx):
    """One per-lane i32 gather from a buffer resource; out-of-range lanes read 0."""
    return ArithValue(_buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=T.i32))


# A lane-resident int32 table (entry i in lane i%64 of chunk i//64) makes a lookup one v_readlane and a prefix sum one wave scan, avoiding SGPR overflow / an LDS barrier.


def _wave_prefix_add_i32(v):
    """Wave64 inclusive add-scan of a per-lane i32 (lane l ends with the sum of 0..l).
    DPP steps replace the serial carry; bound_ctrl zeroes the shifted-in lanes. Requires a
    full EXEC mask (kernel entry)."""
    for _sh in (1, 2, 4, 8):
        v = _dpp_add_i32(v, _DPP_ROW_SHR + _sh)
    v = _dpp_add_i32(v, _DPP_ROW_BCAST15, row_mask=0xA)
    return _dpp_add_i32(v, _DPP_ROW_BCAST31, row_mask=0xC)


_DPP_ROW_SHR = 0x110


_DPP_ROW_BCAST31 = 0x143


_DPP_ROW_BCAST15 = 0x142


def _dpp_add_i32(acc, ctrl, row_mask=0xF):
    """acc + DPP(acc, ctrl); masked-off and shifted-in lanes contribute 0."""
    raw = _raw(acc)
    r = rocdl.update_dpp(raw.type, _raw(fx.Int32(0)), raw, ctrl, row_mask, 0xF, True)
    return acc + ArithValue(_res_of(r))
