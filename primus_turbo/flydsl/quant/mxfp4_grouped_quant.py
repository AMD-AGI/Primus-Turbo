###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Fused grouped MXFP4 dual-cast quant (rowwise tight-M + colwise 128-padded-M).

Drop-in for the HIP ``grouped_quantize_mxfp4_dual`` (non-shuffle, per-1x32 E8M0).
One bf16 read of the grouped activation ``x`` [total_M, N] emits both:
  * rowwise fp4 [total_M, N_pad/2] + E8M0 [total_M, N_pad/32] -- TIGHT M layout
    (row i == input row i), the fwd/dgrad operand;
  * colwise fp4 [N, M_pad_col/2] + E8M0 [N, M_pad_col/32] -- 128-padded per-group
    M layout (transposed), the variable-K wgrad operand.
The per-group 128-pad offsets are filled on-device by a fused ``pad`` prologue
(no D2H). Numerics reuse the bit-exact mxfp4 microblock primitives (RHT +
all-int E8M0 + native cvt_scalef32_pk_fp4), so the output matches the C++ dual
byte-for-byte on the non-SR / non-2d / non-shuffle recipes.

Tile [BM=128 x BK=128]: BM=128 == the col-pad align so every tile's 128 rows
belong to exactly one group (the meta prologue does one O(G) search per tile);
BK=128 == the N-pad align so each tile is one side of the N-pad boundary.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.quant.mxfp4_quant_kernel import (
    _compute_scale_native,
    _cvt_microblock_to_fp4,
    _microblock_amax,
    _microblock_vf,
)
from primus_turbo.flydsl.utils.gemm_helper import xcd_remap_pid

MB = 32  # MXFP4 microblock (elems per E8M0)
_OOB = 0x7FFFFFFF


def _lds_store_vec4(lds_ptr, off, vec):
    fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(4, 1)).store(vec)


def _lds_load_vec4(lds_ptr, off):
    return fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(4, 1)).load()


def _lds_load1(lds_ptr, off):
    return fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(1, 1)).load()[0]


def _store_words_vec4(rsrc, off, words):
    buffer_ops.buffer_store(Vec.from_elements(list(words), fx.Int32), rsrc, off)


def _load_i32_at(div, idx):
    """One int32 scalar at element ``idx`` from an i32 logical view (int64 offs low word)."""
    if isinstance(idx, int):
        idx = fx.Int32(idx)
    atom = fx.make_copy_atom(rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, idx)), reg)
    return Vec(fx.memref_load_vec(reg))[0]


def compile_grouped_mxfp4_qdual(total_M, N, G, M_pad_col, N_pad, row_rht, col_rht, bm=64, bk=256):
    """Compile the fused grouped mxfp4 dual quant. Shapes/recipes are baked.

    Prologue chain (one @flyc.jit stub, no host metadata ops / D2H):
      1) ``pad`` (1 thread): tight GO -> 128-padded col lens/offs (LC/OC);
      2) ``meta`` (1 thread/tile): O(G) group search -> per-bm-row-block
         (RB=abs input row of local 0, RE=abs input row end of the group);
      3) ``kern``: the fused dual tile.
    ``bm`` (tile rows) must divide 128 (one tile -> one 128-padded group)."""
    # mxfp8-quant-style kernel: concurrent ROW/COL halves (256+256 of nth=512) sharing
    # the LDS tile, then a coalesced transposed COL write-back from an LDS stage
    # (ldsc). BK=256 -> each row-output store is 32 contiguous i32 = 128B coalesced
    # (fp4 = 0.5B, so 256 cols = 128B); the COL transpose write is decoupled + coalesced
    # via ldsc. BM=64 keeps tile+ldsc within the 64KB LDS (one tile -> one 128-padded
    # group since BM<=128). BK divides N_pad via ceil + overshoot mask.
    assert 128 % bm == 0 and bm % 32 == 0 and bk % 32 == 0
    BM = bm
    BK = bk
    nth = 512  # threads/block
    HALF = nth // 2  # ROW half = threads [0,256), COL half = [256,512)
    _TCW = BK // 2  # i32 words per tile row (2 bf16/i32)
    _NW = BM * _TCW  # i32 words in the LDS tile
    _RMB = BM // MB  # col-phase row-microblocks per tile
    _CMB = BK // MB  # row-phase col-microblocks per tile
    DWPC = BM // 8  # i32 per feature's M-run in a tile (fp4: BM/2 bytes)
    LDSC_DW = BK * DWPC  # ldsc i32 words (staged col fp4 for the whole tile)
    _NROWT = (BM * _CMB + HALF - 1) // HALF  # row microblocks per ROW-half thread
    _NCOLT = (BK * _RMB + HALF - 1) // HALF  # col microblocks per COL-half thread
    _NLOAD = (_NW + nth * 4 - 1) // (nth * 4)  # vec4 i32 loads/thread
    _CWIT = (LDSC_DW + nth * 4 - 1) // (nth * 4)  # col write-back vec4 iters
    NBM = M_pad_col // BM  # padded-M blocks (col layout)
    NBK = (N_pad + BK - 1) // BK  # N blocks (ceil; BK may overshoot 128-aligned N_pad)
    ROW_SC_N = N_pad // 32  # rowwise scale cols
    COL_SC_N = M_pad_col // 32  # colwise scale cols
    ROW_OUT_W = N_pad // 8  # rowwise fp4 i32 words per row
    COL_OUT_W = M_pad_col // 8  # colwise fp4 i32 words per col

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int32, _NW, 16]
        ldsc: fx.Array[fx.Int32, LDSC_DW, 16]

    @flyc.kernel(known_block_size=[nth, 1, 1])
    def kern(
        X: fx.Tensor,  # int32 view of bf16 [total_M, N], logical [total_M, N/2]
        ROW_OUT: fx.Tensor,  # int32 view fp4 [total_M, N_pad/8]
        ROW_SC: fx.Tensor,  # uint8 [total_M, N_pad/32]
        COL_OUT: fx.Tensor,  # int32 view fp4 [N, M_pad_col/8]
        COL_SC: fx.Tensor,  # uint8 [N, M_pad_col/32]
        GO: fx.Tensor,  # tight per-group offs (int32 view of int64 [G+1])
        LC: fx.Tensor,  # OUT: 128-padded per-group lens (int64 [G])
        OC: fx.Tensor,  # OUT: 128-padded per-group offs (int64 [G+1])
    ):
        # Fused dual tile (one BM x BK tile / WG, one microblock/thread). The per-tile
        # group metadata is computed INLINE (no meta prologue kernel): each WG does the
        # O(G) 128-padded-offset scan from GO (loaded to registers first, so no dependent
        # load chain), yielding in_rebase (abs input row of local 0) / in_end (group input
        # end). The pid==0 WG also emits the padded lens/offs outputs (threads tid<=G).
        I32 = fx.Int32
        z = I32(0)
        lds = fx.SharedAllocator().allocate(Smem).peek()
        tid = fx.thread_idx.x
        # XCD-aware tile remap: spread WGs across the 8 XCDs for L2 locality + full CU
        # occupancy (the linear pid map left ~40% of CUs idle -> memory-bound lever).
        pid = xcd_remap_pid(fx.block_idx.x, I32(NBM * NBK), 8)
        bt = pid // I32(NBK)  # padded-M block; one tile -> one group
        bkc = pid - bt * I32(NBK)  # N block
        base_c = bt * I32(BM)

        # Parallel-load GO[0..G] then scan in registers (avoids the prev=nxt dependent
        # load chain the mxfp8 per-WG scan suffered from).
        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        go_vals = [_load_i32_at(go_div, 2 * g) for g in range_constexpr(G + 1)]
        found = z
        oc_g = z
        go_g = z
        go_g1 = z
        cap_off = z  # offs_col[tid] (pid==0 WG only)
        cap_len = z  # lens_col[tid] (pid==0 WG only)
        acc = z
        for g in range_constexpr(G):
            prev = go_vals[g]
            nxt = go_vals[g + 1]
            lpad = ((nxt - prev + I32(127)) // I32(128)) * I32(128)
            acc_next = acc + lpad
            inq = (base_c >= acc) & (base_c < acc_next)
            oc_g = arith.select(inq, acc, oc_g)
            go_g = arith.select(inq, prev, go_g)
            go_g1 = arith.select(inq, nxt, go_g1)
            found = arith.select(inq, I32(1), found)
            atg = tid == I32(g)
            cap_off = arith.select(atg, acc, cap_off)
            cap_len = arith.select(atg, lpad, cap_len)
            acc = acc_next
        cap_off = arith.select(tid == I32(G), acc, cap_off)  # offs_col[G] = total padded
        in_rebase = arith.select(found == I32(1), go_g + (base_c - oc_g), z)
        in_end = arith.select(found == I32(1), go_g1, z)
        if pid == z:  # one WG writes the padded lens/offs outputs (num_records masks tid>G)
            lc_r = buffer_ops.create_buffer_resource(LC, max_size=False, num_records_bytes=I32(G * 8))
            oc_r = buffer_ops.create_buffer_resource(OC, max_size=False, num_records_bytes=I32((G + 1) * 8))
            buffer_ops.buffer_store(cap_len, lc_r, 2 * tid)
            buffer_ops.buffer_store(z, lc_r, 2 * tid + I32(1))
            buffer_ops.buffer_store(cap_off, oc_r, 2 * tid)
            buffer_ops.buffer_store(z, oc_r, 2 * tid + I32(1))

        # ---- coalesced tile load: X[in_rebase + tr, bkc*BK + col] -> LDS (all
        # loads issued first for read MLP; past-group rows / >=N cols -> 0) ----
        xw = total_M * (N >> 1)
        rsrc = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=xw * 4)
        c0w = bkc * I32(_TCW)
        _vecs = []
        for chunk in range_constexpr(_NLOAD):
            tw = chunk * (nth * 4) + tid * 4
            tr = tw // I32(_TCW)
            wc = tw - tr * I32(_TCW)
            grow = in_rebase + tr
            fcolw = c0w + wc
            ioff = grow * I32(N >> 1) + fcolw
            ioff = ((grow < in_end) & (fcolw < I32(N >> 1))).select(ioff, I32(_OOB))
            _vecs.append(buffer_ops.buffer_load(rsrc, ioff, vec_width=4, dtype=T.i32))
        for chunk in range_constexpr(_NLOAD):
            _lds_store_vec4(lds.buf.ptr, chunk * (nth * 4) + tid * 4, _vecs[chunk])
        fx.barrier()

        orsrc = buffer_ops.create_buffer_resource(
            ROW_OUT, max_size=False, num_records_bytes=total_M * ROW_OUT_W * 4
        )
        rscrsrc = buffer_ops.create_buffer_resource(
            ROW_SC, max_size=False, num_records_bytes=total_M * ROW_SC_N
        )
        corsrc = buffer_ops.create_buffer_resource(
            COL_OUT, max_size=False, num_records_bytes=N * COL_OUT_W * 4
        )
        cscrsrc = buffer_ops.create_buffer_resource(COL_SC, max_size=False, num_records_bytes=N * COL_SC_N)

        # Concurrent halves: ROW half (tid<HALF) casts tight-M + 128B-coalesced store;
        # COL half (tid>=HALF) casts the transpose + stages fp4 to ldsc (c-major). The
        # two run in different warps so the row HBM writes overlap the col compute.
        half = tid // I32(HALF)
        lt = tid - half * I32(HALF)
        if half == z:  # ROW half
            for kk in range_constexpr(_NROWT):
                task = kk * I32(HALF) + lt
                r_row = task // I32(_CMB)
                r_cmb = task - r_row * I32(_CMB)
                base_w = r_row * I32(_TCW) + r_cmb * I32(16)
                rbits = []
                for q in range_constexpr(4):
                    v4 = _lds_load_vec4(lds.buf.ptr, base_w + q * 4)
                    for j in range_constexpr(4):
                        rbits.append(v4[j] << 16)
                        rbits.append(v4[j] & 0xFFFF0000)
                vf = _microblock_vf(rbits, row_rht)
                native_bits, rbiased = _compute_scale_native(_microblock_amax(vf))
                rwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits))
                grow = in_rebase + r_row
                gcmb = bkc * I32(_CMB) + r_cmb
                row_ok = (grow < in_end) & (gcmb * I32(4) < I32(ROW_OUT_W))  # mask N_pad overshoot
                ob = grow * I32(ROW_OUT_W) + gcmb * I32(4)
                for c in range_constexpr(4):
                    buffer_ops.buffer_store(rwords[c], orsrc, row_ok.select(ob + c, I32(_OOB)))
                buffer_ops.buffer_store(
                    arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, grow * I32(ROW_SC_N) + gcmb, mask=row_ok
                )
        if half != z:  # COL half: cast transpose -> stage to ldsc (col scale direct)
            for kk in range_constexpr(_NCOLT):
                task = kk * I32(HALF) + lt
                c_col = task // I32(_RMB)
                mblk = task - c_col * I32(_RMB)
                cw = c_col >> 1
                chalf = c_col & 1
                row0 = mblk * I32(32)
                cbits = []
                for row in range_constexpr(32):
                    word = _lds_load1(lds.buf.ptr, (row0 + row) * I32(_TCW) + cw)
                    cbits.append(arith.select(chalf != I32(0), word & I32(-65536), word << 16))
                cvf = _microblock_vf(cbits, col_rht)
                cnative, cbiased = _compute_scale_native(_microblock_amax(cvf))
                cwords = _cvt_microblock_to_fp4(cvf, arith.bitcast(T.f32, cnative))
                _lds_store_vec4(
                    lds.ldsc.ptr, c_col * I32(DWPC) + mblk * I32(4), Vec.from_elements(cwords, fx.Int32)
                )
                gcol = bkc * I32(BK) + c_col
                gmmb = bt * I32(_RMB) + mblk
                buffer_ops.buffer_store(
                    arith.trunci(T.i8, cbiased & 0xFF),
                    cscrsrc,
                    gcol * I32(COL_SC_N) + gmmb,
                    mask=gcol < I32(N),
                )
        fx.barrier()
        # ---- coalesced transposed COL write-back: ldsc -> COL_OUT (all threads) ----
        for it in range_constexpr(_CWIT):
            lo = (tid + it * I32(nth)) * I32(4)
            cc = lo // I32(DWPC)  # feature within tile
            dwi0 = lo - cc * I32(DWPC)  # i32 within feature's M-run
            v4 = _lds_load_vec4(lds.ldsc.ptr, lo)
            gcol = bkc * I32(BK) + cc
            cob = gcol * I32(COL_OUT_W) + bt * I32(DWPC) + dwi0
            buffer_ops.buffer_store(v4, corsrc, (gcol < I32(N)).select(cob, I32(_OOB)))

    @flyc.jit
    def launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        GO: fx.Tensor,
        LC: fx.Tensor,
        OC: fx.Tensor,
        stream: fx.Stream,
    ):
        # Single kernel: per-tile group metadata computed inline (no meta prologue),
        # padded lens/offs emitted by the pid==0 WG.
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, GO, LC, OC).launch(
            grid=(NBM * NBK, 1, 1), block=(nth, 1, 1), stream=stream
        )

    return launch


_GQ_MXFP4_CACHE: dict = {}


def grouped_quant_mxfp4_raw(x, group_lens, group_offs, fp4_dtype, row_rht, col_rht, bm=64, bk=256):
    """FlyDSL grouped mxfp4 dual quant, drop-in for the HIP grouped_quantize_mxfp4_dual
    (non-shuffle, non-SR, non-2d recipes). Returns the same 6-tuple:
      (rowwise_out [total_M, N_pad/2] fp4, rowwise_scale [total_M, N_pad/32] e8m0,
       colwise_out [N, M_pad_col/2] fp4, colwise_scale [N, M_pad_col/32] e8m0,
       group_lens_padded_col [G], group_offs_padded_col [G+1]).
    ``x`` [total_M, N] bf16/fp16 contiguous; group_lens [G] / group_offs [G+1] int64 GPU."""
    import flydsl.compiler as _flyc
    import torch

    assert x.ndim == 2 and x.is_contiguous()
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
    assert group_lens.is_cuda and group_offs.is_cuda
    total_M, N = int(x.shape[0]), int(x.shape[1])
    G = int(group_lens.shape[0])
    assert N % MB == 0, f"N must be a multiple of {MB}"
    N_pad = (N + 127) // 128 * 128
    M_pad_col = (total_M + G * 128 + 127) // 128 * 128

    dev = x.device
    row_out = torch.empty(total_M, N_pad // 2, dtype=torch.uint8, device=dev)
    row_sc = torch.empty(total_M, N_pad // 32, dtype=torch.uint8, device=dev)
    col_out = torch.empty(N, M_pad_col // 2, dtype=torch.uint8, device=dev)
    col_sc = torch.empty(N, M_pad_col // 32, dtype=torch.uint8, device=dev)
    lens_col = torch.empty(G, dtype=torch.int64, device=dev)
    offs_col = torch.empty(G + 1, dtype=torch.int64, device=dev)

    go = group_offs.to(torch.int64).view(torch.int32)
    lc = lens_col.view(torch.int32)
    oc = offs_col.view(torch.int32)

    key = (total_M, N, G, M_pad_col, N_pad, bool(row_rht), bool(col_rht), int(bm), int(bk), x.dtype)
    comp = _GQ_MXFP4_CACHE.get(key)
    stream = torch.cuda.current_stream()
    xi = x.view(torch.int32)
    roi = row_out.view(torch.int32)
    coi = col_out.view(torch.int32)
    if comp is None:
        launch = compile_grouped_mxfp4_qdual(
            total_M, N, G, M_pad_col, N_pad, bool(row_rht), bool(col_rht), bm=bm, bk=bk
        )
        comp = _flyc.compile(launch, xi, roi, row_sc, coi, col_sc, go, lc, oc, stream)
        _GQ_MXFP4_CACHE[key] = comp
    comp(xi, roi, row_sc, coi, col_sc, go, lc, oc, stream)

    e8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    return (
        row_out.view(fp4_dtype),
        row_sc.view(e8),
        col_out.view(fp4_dtype),
        col_sc.view(e8),
        lens_col,
        offs_col,
    )
