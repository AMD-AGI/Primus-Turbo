# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""dsa_bwd_dkv_interm_v2: DeepSeek-V4 sparse-MLA dKV-intermediate (multi-wave FlyDSL).

Native FlyDSL replacement for the Triton ``_bwd_compute_dkv_intermediate``.
Computes, per query token t (contracting over the head axis H):

    interm[t, key, d] = sum_h ( Q[t,h,d] * dS[t,h,key] + dO[t,h,d] * P[t,h,key] )

Output interm[T, TOPK, D_V], feeds the (unchanged) CSR dKV gather-reduce. dS/P are
REUSED from the dQ kernel's HBM buffers (no recompute).

Design (mirrors gluon warps_per_cta=[4,1]; see record/DESIGN_dkv_interm_flydsl_multiwave.md):
  * Grid = (T,). Each workgroup = NW waves, all on the SAME token.
  * The NW waves SPLIT the output D_V dimension: wave w owns d in
    [w*D_PER_WAVE, (w+1)*D_PER_WAVE). No cross-wave reduction (contraction H is
    done inside each wave).
  * Loop head groups (BLOCK_H heads at a time): stage Q/dO [BLOCK_H, D_V] ROW-MAJOR
    into SHARED LDS ([head][d], padded stride to dodge ds_read_tr bank conflicts),
    cooperatively by all NW*64 threads (contiguous vec8 loads — NOT a scatter, so
    unlike the fwd gather this staging vectorizes cleanly). Each wave then reads its
    own 128-d columns.
  * MFMA mfma_f32_16x16x32_bf16, K=32 head-contraction steps:
      A = Q_T[d, h] via ds_read_tr16 (row-major [head][d] tile -> transpose read),
      B = dS[h, key] natural (loaded from HBM per (key-tile, head-chunk), vec8 over
          head, reused across the DT_PER_WAVE d-tiles),
      C = interm[d, key] fp32.
    The SAME tile feeds Q_T; dO_T from the doT tile; two GEMMs accumulate into one C.

gfx950 / CDNA4 only. rope skipped (V4 zero-pad; interm rope cols never read/stored).
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _mfma_16x16x32(a, b, c):
    return rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])


def build_dsa_bwd_dkv_interm_v2_module(
    num_heads,
    kv_lora_rank,
    d_qk,
    topk,
    dtype_str="bf16",
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "dsa_bwd_dkv_interm_v2 targets gfx950 (CDNA4)"
    assert dtype_str == "bf16", "bf16 only"

    HEAD_DIM = int(kv_lora_rank)  # D_V = 512 (output d dim)
    D_QK = int(d_qk)              # interm row stride incl rope pad
    NUM_HEADS = int(num_heads)
    TOPK = int(topk)
    assert HEAD_DIM % 32 == 0
    assert NUM_HEADS % 16 == 0, "dkv-interm v2 needs NUM_HEADS % 16 == 0"

    WARP_SIZE = 64
    # NW waves split the D_V output. D_V=512 / NW must be a multiple of 16 (d-tiles).
    # NW=8 (512-thread CTA) is the swept optimum: it keeps per-wave acc [64,TILE_K]
    # small enough to avoid spilling (v214/s0 at BH32/TK64) while packing enough waves
    # to hide the Q/dO/interm memory latency. NW4 spills; NW16 loses occupancy.
    NW = int(os.environ.get("PRIMUS_DSA_INTERM_NW", "8"))
    while HEAD_DIM % NW != 0 or (HEAD_DIM // NW) % 16 != 0:
        NW -= 1
    D_PER_WAVE = HEAD_DIM // NW
    DT_PER_WAVE = D_PER_WAVE // 16           # output d-tiles per wave
    BLOCK_SIZE = NW * WARP_SIZE

    # Head group: BLOCK_H heads staged into LDS at a time. Must be a multiple of 32
    # (head-contraction chunk = 32). NUM_HG groups cover all heads (last padded).
    BLOCK_H = int(os.environ.get("PRIMUS_DSA_INTERM_BH", "32"))
    assert BLOCK_H % 32 == 0
    NUM_HG = (NUM_HEADS + BLOCK_H - 1) // BLOCK_H
    HC_STEPS = BLOCK_H // 32                  # head-contraction MFMA steps per group
    # Async global->LDS DMA (raw_ptr_buffer_load_lds) for Q/dO staging: bypasses VGPR
    # and fires asynchronously so the MFMA of head-group g overlaps the fetch of g+1
    # (the latency-hiding lever; interm is MemUnitStalled-bound at occ 1-2).
    _USE_DMA = os.environ.get("PRIMUS_DSA_INTERM_DMA", "1") == "1"
    _A_PF = int(os.environ.get("PRIMUS_DSA_INTERM_APF", "2"))  # A-operand prefetch depth
    _A_PF = max(1, min(_A_PF, DT_PER_WAVE))

    # Key tile: TILE_K keys per output tile (MFMA N=16 sub-tiles). TK64 swept optimum
    # (fewer Q/dO re-stagings than TK32; TK>=128 spills the per-wave acc). Falls back
    # to 32 when TOPK is not a multiple of 64.
    _tk_def = 64 if TOPK % 64 == 0 else 32
    TILE_K = int(os.environ.get("PRIMUS_DSA_INTERM_TK", str(_tk_def)))
    assert TILE_K % 16 == 0
    N_SUB = TILE_K // 16
    assert TOPK % TILE_K == 0, f"TOPK ({TOPK}) must be a multiple of TILE_K ({TILE_K})"
    NUM_TILES = TOPK // TILE_K

    # LDS: qT[BLOCK_H][D_V + PAD] ++ doT[BLOCK_H][D_V + PAD], row-major [head][d].
    # PAD=16 breaks ds_read_tr bank conflicts (proven in fwd/dq).
    SQ = HEAD_DIM + 16
    # dS/P staged SHARED (once per workgroup) as [key][head], stride SH (head axis
    # padded to dodge bank conflicts). Head-contiguous so the hot B-operand read is
    # ONE vec8 LDS load over the head contraction axis. Staged ONCE for all NW waves
    # -> the scattered per-wave HBM B-load (Stage-A's MemUnitStalled bottleneck) and
    # its redundant traffic are both gone.
    SH = BLOCK_H + 8
    # Double-buffer qT/doT (NBUF=2) so head-group g+1's async DMA overlaps g's GEMM
    # (gluon's num_stages pipeline; the last real latency-hiding lever). NBUF=1 falls
    # back to the fire-then-drain single-buffer path. Only enabled if the 2x qT/doT
    # LDS fits under 160KB and there is more than one head group to pipeline.
    # Pipeline nets ~0 (slightly worse): s_waitcnt(0) is a full drain (no per-op vmcnt
    # threshold, and the dS/P regular vector loads share the counter), so the hg+1
    # prefetch can't actually overlap hg's GEMM. Same limitation as the fwd pipeline.
    # Gated OFF; single-buffer fire-then-drain is the default.
    _WANT_PIPE = os.environ.get("PRIMUS_DSA_INTERM_PIPE", "0") == "1"
    _one_qt = BLOCK_H * SQ
    NBUF = 2 if (_WANT_PIPE and _USE_DMA and NUM_HG > 1
                 and (2 * 2 * _one_qt + 2 * TILE_K * SH) * 2 <= 163840) else 1
    LDS_QT = NBUF * BLOCK_H * SQ
    LDS_DOT = NBUF * BLOCK_H * SQ
    LDS_DS = TILE_K * SH
    LDS_P = TILE_K * SH
    LDS_TOTAL = LDS_QT + LDS_DOT + LDS_DS + LDS_P

    allocator = SmemAllocator(None, arch=gpu_arch,
                              global_sym_name=f"dsa_dkv_interm_v2_H{NUM_HEADS}_K{TOPK}_W{NW}_BH{BLOCK_H}_TK{TILE_K}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dsa_bwd_dkv_interm_v2_kernel(
        Q: fx.Tensor,       # [T, H, D_QK] bf16
        dO: fx.Tensor,      # [T, H, D_V]  bf16
        dS: fx.Tensor,      # [T, H, TOPK] bf16
        Pin: fx.Tensor,     # [T, H, TOPK] bf16
        Interm: fx.Tensor,  # [T, TOPK, D_QK] bf16 (out; rope cols unwritten)
        total_tokens: fx.Int32,
    ):
        f16_ty = T.bf16
        f32_ty = T.f32

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dO)
        ds_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dS)
        p_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Pin)
        interm_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Interm)
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True) if _USE_DMA else None
        do_rsrc = buffer_ops.create_buffer_resource(dO, max_size=True) if _USE_DMA else None

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, f16_ty, shape=(LDS_TOTAL,)).get()
        # qT/doT double-buffered: buffer b at base + b*(BLOCK_H*SQ).
        _QT0 = 0
        _DOT0 = LDS_QT
        C_DS = arith.index(LDS_QT + LDS_DOT)
        C_P = arith.index(LDS_QT + LDS_DOT + LDS_DS)
        def C_QT_buf(b):
            return arith.index(_QT0 + b * (BLOCK_H * SQ))
        def C_DOT_buf(b):
            return arith.index(_DOT0 + b * (BLOCK_H * SQ))

        def _gep64(base_p, elem_idx64, elem_t):
            return _llvm.GEPOp(_llvm_ptr_ty(), base_p, [elem_idx64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v64(base_p, elem_idx64, n):
            return _llvm.LoadOp(T.vec(n, f16_ty), _gep64(base_p, elem_idx64, f16_ty)).result

        def i64(v):
            return arith.index_cast(T.i64, v)

        tok = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        wave = tid // arith.index(WARP_SIZE)
        lane = tid % arith.index(WARP_SIZE)
        tt_v = arith.index_cast(T.index, total_tokens)
        tok_active = arith.cmpi(arith.CmpIPredicate.slt, tok, tt_v)
        tok_safe = arith.select(tok_active, tok, arith.index(0))

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)
        d_wave_base = wave * arith.index(D_PER_WAVE)

        NH = arith.index(NUM_HEADS)
        HD = arith.index(HEAD_DIM)
        DQK = arith.index(D_QK)
        TOPK_I = arith.index(TOPK)
        c_zero_f16 = arith.constant(0.0, type=f16_ty)
        zero_pack8 = arith.constant_vector(0.0, T.vec(8, f16_ty))
        c_zero_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))

        # ds_read_tr lane decomposition for a row-major tile[row][d] stride SQ.
        # HW 4x4 transpose delivers A[d = L%16 (col-tile), row = (L//16)*8 + e].
        # Here "row" = head. k_group / col_sub as in dq's _av_a.
        tr_k_group = lane_mod_16 // arith.index(4)   # (L%16)//4
        tr_col_sub = lane_mod_16 % arith.index(4)    # L%4  (== lane_mod_16 % 4)
        v_hrow = lane_div_16 * arith.index(8) + tr_k_group

        def _ds_read_tr_v4(elem_idx):
            byte = elem_idx * arith.index(2) + arith.index(lds_off)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), i64(byte)).result
            return rocdl.ds_read_tr16_b64(T.vec(4, f16_ty), ptr).result

        # A-operand for (tile-region C_base, head-chunk hc, d-tile dt): A[d, head=hc*32+..].
        def _a_pack(C_base, hc, dt):
            h_row = arith.index(hc * 32) + v_hrow
            d_col = d_wave_base + arith.index(dt * 16) + tr_col_sub * arith.index(4)
            base = C_base + h_row * arith.index(SQ) + d_col
            va = _ds_read_tr_v4(base)
            vb = _ds_read_tr_v4(base + arith.index(4 * SQ))  # +4 heads
            return vector.shuffle(va, vb, [0, 1, 2, 3, 4, 5, 6, 7])

        # tile loop (output [D_V, TILE_K] per tile; each wave its d-slice)
        for tstep in scf.for_(arith.index(0), arith.index(NUM_TILES), arith.index(1)):
            key_tile_base = tstep * arith.index(TILE_K)

            acc = [[c_zero_acc for _ in range_constexpr(N_SUB)] for _ in range_constexpr(DT_PER_WAVE)]

            _COLG = HEAD_DIM // 8          # vec8 groups per head row (64)
            _STOT = BLOCK_H * _COLG
            _ITERS = (_STOT + BLOCK_SIZE - 1) // BLOCK_SIZE
            _CKG = TILE_K // 8            # vec8 groups (of keys) per head
            _DTOT = BLOCK_H * _CKG
            _DITERS = (_DTOT + BLOCK_SIZE - 1) // BLOCK_SIZE
            _dma_size = arith.constant(16, type=T.i32)
            _dma_z = arith.constant(0, type=T.i32)
            _dma_aux = arith.constant(1, type=T.i32)
            lds_base_byte = buffer_ops.extract_base_index(lds, address_space=3) if const_expr(_USE_DMA) else None

            # Fire the async Q/dO DMA for head-group hg into LDS buffer `buf`.
            def _fire_qdo(hg, buf):
                hg_off = hg * BLOCK_H
                cqt = C_QT_buf(buf)
                cdot = C_DOT_buf(buf)
                for it in range_constexpr(_ITERS):
                    slot = tid + arith.index(it * BLOCK_SIZE)
                    in_r = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.index(_STOT)) if const_expr(_STOT % BLOCK_SIZE != 0) else None
                    hloc = slot // arith.index(_COLG)
                    col8 = (slot % arith.index(_COLG)) * arith.index(8)
                    h_glob = arith.index(hg_off) + hloc
                    h_in = arith.cmpi(arith.CmpIPredicate.slt, h_glob, NH)
                    h_safe = arith.select(h_in, h_glob, arith.index(0))
                    q_off = (tok_safe * NH + h_safe) * DQK + col8
                    o_off = (tok_safe * NH + h_safe) * HD + col8
                    q_voff = arith.index_cast(T.i32, q_off * arith.index(2))
                    o_voff = arith.index_cast(T.i32, o_off * arith.index(2))
                    qd = cqt + hloc * arith.index(SQ) + col8
                    od = cdot + hloc * arith.index(SQ) + col8
                    qaddr = arith.index_cast(T.i64, lds_base_byte + (arith.index(lds_off) + qd) * arith.index(2))
                    oaddr = arith.index_cast(T.i64, lds_base_byte + (arith.index(lds_off) + od) * arith.index(2))
                    qlp = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), qaddr).result
                    olp = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), oaddr).result
                    def _fire():
                        rocdl.raw_ptr_buffer_load_lds(q_rsrc, qlp, _dma_size, q_voff, _dma_z, _dma_z, _dma_aux)
                        rocdl.raw_ptr_buffer_load_lds(do_rsrc, olp, _dma_size, o_voff, _dma_z, _dma_z, _dma_aux)
                    if const_expr(in_r is not None):
                        _if_d = scf.IfOp(in_r, [], has_else=False)
                        with ir.InsertionPoint(_if_d.then_block):
                            _fire(); scf.YieldOp([])
                    else:
                        _fire()

            # Coop vec8 Q/dO staging (non-DMA fallback) for head-group hg into buffer 0.
            def _stage_qdo_coop(hg):
                hg_off = hg * BLOCK_H
                for it in range_constexpr(_ITERS):
                    slot = tid + arith.index(it * BLOCK_SIZE)
                    in_r = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.index(_STOT))
                    hloc = slot // arith.index(_COLG)
                    col8 = (slot % arith.index(_COLG)) * arith.index(8)
                    h_glob = arith.index(hg_off) + hloc
                    h_in = arith.cmpi(arith.CmpIPredicate.slt, h_glob, NH)
                    h_safe = arith.select(h_in, h_glob, arith.index(0))
                    valid = arith.AndIOp(in_r, h_in).result
                    q_off64 = arith.AddIOp(arith.MulIOp(i64((tok_safe * NH + h_safe)), i64(DQK)).result, i64(col8)).result
                    o_off64 = arith.AddIOp(arith.MulIOp(i64((tok_safe * NH + h_safe)), i64(HD)).result, i64(col8)).result
                    qv = arith.select(valid, load_f16_v64(q_ptr, q_off64, 8), zero_pack8)
                    ov = arith.select(valid, load_f16_v64(do_ptr, o_off64, 8), zero_pack8)
                    dst = C_QT_buf(0) + hloc * arith.index(SQ) + col8
                    dst_o = C_DOT_buf(0) + hloc * arith.index(SQ) + col8
                    _if = scf.IfOp(in_r, [], has_else=False)
                    with ir.InsertionPoint(_if.then_block):
                        vector.store(qv, lds, [dst])
                        vector.store(ov, lds, [dst_o])
                        scf.YieldOp([])

            # Stage dS/P for head-group hg into the SHARED [key][head] LDS tile.
            def _stage_dsp(hg):
                hg_off = hg * BLOCK_H
                for it in range_constexpr(_DITERS):
                    slot = tid + arith.index(it * BLOCK_SIZE)
                    in_r = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.index(_DTOT))
                    hloc = slot // arith.index(_CKG)
                    k8 = (slot % arith.index(_CKG)) * arith.index(8)
                    h_glob = arith.index(hg_off) + hloc
                    h_in = arith.cmpi(arith.CmpIPredicate.slt, h_glob, NH)
                    h_safe = arith.select(h_in, h_glob, arith.index(0))
                    key0 = key_tile_base + k8
                    valid = arith.AndIOp(in_r, h_in).result
                    dsoff64 = arith.AddIOp(arith.MulIOp(i64((tok_safe * NH + h_safe)), i64(TOPK_I)).result, i64(key0)).result
                    dsv = load_f16_v64(ds_ptr, dsoff64, 8)
                    pv = load_f16_v64(p_ptr, dsoff64, 8)
                    _if_b = scf.IfOp(valid, [], has_else=False)
                    with ir.InsertionPoint(_if_b.then_block):
                        for e in range_constexpr(8):
                            bidx = (k8 + arith.index(e)) * arith.index(SH) + hloc
                            vector.store(vector.from_elements(T.vec(1, f16_ty), [vector.extract(dsv, static_position=[e], dynamic_position=[])]), lds, [C_DS + bidx])
                            vector.store(vector.from_elements(T.vec(1, f16_ty), [vector.extract(pv, static_position=[e], dynamic_position=[])]), lds, [C_P + bidx])
                        scf.YieldOp([])

            # Output GEMM for the head group whose qT/doT live in buffer `buf`.
            def _gemm(buf):
                cqt = C_QT_buf(buf)
                cdot = C_DOT_buf(buf)
                for hc in range_constexpr(HC_STEPS):
                    hbase = arith.index(hc * 32) + lane_div_16 * arith.index(8)
                    b_ds = []
                    b_p = []
                    for kt in range_constexpr(N_SUB):
                        kloc = arith.index(kt * 16) + lane_mod_16
                        bidx = kloc * arith.index(SH) + hbase
                        b_ds.append(vector.load_op(T.vec(8, f16_ty), lds, [C_DS + bidx]))
                        b_p.append(vector.load_op(T.vec(8, f16_ty), lds, [C_P + bidx]))
                    aq_pf = [None] * DT_PER_WAVE
                    ao_pf = [None] * DT_PER_WAVE
                    for p in range_constexpr(_A_PF):
                        aq_pf[p] = _a_pack(cqt, hc, p)
                        ao_pf[p] = _a_pack(cdot, hc, p)
                    for dt in range_constexpr(DT_PER_WAVE):
                        aq = aq_pf[dt]
                        ao = ao_pf[dt]
                        if const_expr(dt + _A_PF < DT_PER_WAVE):
                            aq_pf[dt + _A_PF] = _a_pack(cqt, hc, dt + _A_PF)
                            ao_pf[dt + _A_PF] = _a_pack(cdot, hc, dt + _A_PF)
                        for kt in range_constexpr(N_SUB):
                            cur = acc[dt][kt]
                            cur = _mfma_16x16x32(aq, b_ds[kt], cur)
                            cur = _mfma_16x16x32(ao, b_p[kt], cur)
                            acc[dt][kt] = cur

            if const_expr(NBUF == 2):
                # Software-pipelined over head groups: fire hg's Q/dO DMA, then while it
                # is in flight stage dS/P; before computing hg, fire hg+1's DMA into the
                # other buffer so its fetch overlaps hg's GEMM (gluon's num_stages).
                _fire_qdo(0, 0)
                for hg in range_constexpr(NUM_HG):
                    buf = hg & 1
                    nbuf = (hg + 1) & 1
                    _stage_dsp(hg)              # dS/P for this hg (also masks pad heads)
                    if const_expr(hg + 1 < NUM_HG):
                        _fire_qdo(hg + 1, nbuf)  # prefetch next hg's Q/dO (overlaps GEMM)
                    rocdl.s_waitcnt(0)          # drain: this hg's Q/dO DMA is complete
                    gpu.barrier()               # qT/doT[buf] + dS/P visible to all waves
                    _gemm(buf)
                    gpu.barrier()               # protect dS/P + buf before next hg reuses
            else:
                for hg in range_constexpr(NUM_HG):
                    gpu.barrier()  # protect qT/doT LDS from previous group's reads
                    if const_expr(_USE_DMA):
                        _fire_qdo(hg, 0)
                    else:
                        _stage_qdo_coop(hg)
                    _stage_dsp(hg)
                    if const_expr(_USE_DMA):
                        rocdl.s_waitcnt(0)  # drain async Q/dO DMA before LDS reads
                    gpu.barrier()  # qT/doT + dS/P staged for this head group
                    _gemm(0)

            # ---- store interm[t, key, d] for this tile (each wave its d-slice) ----
            # C-acc lane L holds C[m = (L//16)*4 + r, n = L%16]; m=d-in-tile, n=key.
            for dt in range_constexpr(DT_PER_WAVE):
                for kt in range_constexpr(N_SUB):
                    key_glob = key_tile_base + arith.index(kt * 16) + lane_mod_16
                    key_in = arith.cmpi(arith.CmpIPredicate.slt, key_glob, TOPK_I)
                    guard = arith.AndIOp(key_in, tok_active).result
                    _if_st = scf.IfOp(guard, [], has_else=False)
                    with ir.InsertionPoint(_if_st.then_block):
                        interm_row64 = arith.MulIOp(
                            i64((tok_safe * TOPK_I + key_glob)), i64(DQK)).result
                        # d = d_wave_base + dt*16 + (L//16)*4 + r; the 4 r's are
                        # contiguous -> coalesce the 4 scalar stores into one vec4.
                        d0 = d_wave_base + arith.index(dt * 16) + lane_div_16 * arith.index(4)
                        off64 = arith.AddIOp(interm_row64, i64(d0)).result
                        vals16 = [arith.trunc_f(f16_ty, vector.extract(acc[dt][kt], static_position=[r], dynamic_position=[])) for r in range_constexpr(4)]
                        vec4 = vector.from_elements(T.vec(4, f16_ty), vals16)
                        _llvm.StoreOp(vec4, _gep64(interm_ptr, off64, f16_ty))
                        scf.YieldOp([])

    @flyc.jit
    def launch_dsa_bwd_dkv_interm_v2(
        Q: fx.Tensor,
        dO: fx.Tensor,
        dS: fx.Tensor,
        Pin: fx.Tensor,
        Interm: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        tt_idx = arith.index_cast(T.index, total_tokens)
        launcher = dsa_bwd_dkv_interm_v2_kernel(Q, dO, dS, Pin, Interm, total_tokens)
        if const_expr(waves_per_eu is not None):
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign")]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")]))
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(grid=(tt_idx, arith.index(1), arith.index(1)), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math,
              "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True}}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_dsa_bwd_dkv_interm_v2(*args, **kwargs)

    return _launch
