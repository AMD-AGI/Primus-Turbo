# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""dsa_bwd_dkv_interm: DeepSeek-V4 sparse-MLA dKV-intermediate (FlyDSL MFMA).

Ports the Triton ``_bwd_compute_dkv_intermediate`` to a native FlyDSL M=16
head-contraction MFMA (the same output-GEMM pattern proven in
``build_csa_pool_bwd_dpool_mfma_module``). Computes, per query token:

    interm[key, d] = sum_h ( Q[h, d] * dS[h, key] + dO[h, d] * P[h, key] )

contracting over the head axis H. dS / P are REUSED from the dQ kernel's HBM
buffers (no recompute). Output interm[T, TOPK, D_V] feeds the (unchanged) CSR
dKV gather-reduce kernel.

Grid: (T,) — one workgroup (single wave, 64 lanes) per token contracts ALL heads
into the [TOPK, D_V] intermediate for that token (atomic-free, unique owner).

Layout (mirrors the dpool MFMA output GEMM):
  * qT / doT staged transposed into LDS [D_V, H_PAD] (A-operand).
  * dS / P staged into LDS [BLOCK_K, H_PAD] (B-operand), loaded from HBM.
  * output GEMM: part[d, key] = qT[d,h] @ dS[h,key] + doT[d,h] @ P[h,key]
    (16x16x32, K=32 head-contraction steps; head axis padded to mult of 32).

gfx950 / CDNA4 only. rope skipped (V4 zero-pad; interm rope cols never read).
"""

import math

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


def _mfma_16x16x32(a, b, c):
    return rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])


def build_dsa_bwd_dkv_interm_module(
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
    assert gpu_arch.startswith("gfx950"), "dsa_bwd_dkv_interm targets gfx950 (CDNA4)"
    assert dtype_str == "bf16", "bf16 only"

    HEAD_DIM = int(kv_lora_rank)   # D_V = 512 (output d dim of interm)
    D_QK = int(d_qk)               # interm row stride (D_V + rope pad)
    NUM_HEADS = int(num_heads)
    TOPK = int(topk)
    assert HEAD_DIM % 32 == 0
    assert NUM_HEADS % 16 == 0, "dkv-interm MFMA needs NUM_HEADS % 16 == 0"

    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    BLOCK_K = 16                    # keys per output-GEMM tile (MFMA N)
    assert TOPK % BLOCK_K == 0, f"TOPK ({TOPK}) must be a multiple of {BLOCK_K}"
    H_KSTEPS = (NUM_HEADS + 31) // 32   # head-contraction MFMA steps (K=32)
    H_PAD = H_KSTEPS * 32
    HAS_HPAD = H_PAD != NUM_HEADS
    NUM_TILES = TOPK // BLOCK_K
    # Block the D_V output so qT/doT LDS ([D_BLOCK, H_PAD] each) fits 160KB. The
    # full [D_V=512, H_PAD] would be 2*512*128*2B=256KB. D_BLOCK=128 -> 2*128*128*2
    # =64KB + dS/P 8KB = 72KB. Loop D in DB_ITERS blocks; dS/P staged once per
    # key-tile and reused across D-blocks.
    import os as _os
    D_BLOCK = int(_os.environ.get("PRIMUS_DSA_INTERM_DBLOCK", "128"))
    assert HEAD_DIM % D_BLOCK == 0 and D_BLOCK % 16 == 0
    DB_ITERS = HEAD_DIM // D_BLOCK
    DT_PER_BLK = D_BLOCK // 16       # output d-tiles per D-block

    # LDS: qT [D_BLOCK, H_PAD] ++ doT [D_BLOCK, H_PAD] ++ dS [BLOCK_K, H_PAD] ++ P [BLOCK_K, H_PAD]
    LDS_QT = D_BLOCK * H_PAD
    LDS_DOT = D_BLOCK * H_PAD
    LDS_DS = BLOCK_K * H_PAD
    LDS_P = BLOCK_K * H_PAD
    LDS_TOTAL = LDS_QT + LDS_DOT + LDS_DS + LDS_P

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"dsa_dkv_interm_H{NUM_HEADS}_K{TOPK}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dsa_bwd_dkv_interm_kernel(
        Q: fx.Tensor,       # [T, H, D_QK] bf16
        dO: fx.Tensor,      # [T, H, D_V] bf16
        dS: fx.Tensor,      # [T, H, TOPK] bf16
        Pin: fx.Tensor,     # [T, H, TOPK] bf16
        Interm: fx.Tensor,  # [T, TOPK, D_QK] bf16 (out; rope cols unwritten)
        total_tokens: fx.Int32,
    ):
        f16_ty = T.bf16
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dO)
        ds_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dS)
        p_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Pin)
        interm_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Interm)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, f16_ty, shape=(LDS_TOTAL,)).get()
        C_QT = arith.index(0)
        C_DOT = arith.index(LDS_QT)
        C_DS = arith.index(LDS_QT + LDS_DOT)
        C_P = arith.index(LDS_QT + LDS_DOT + LDS_DS)

        def _gep(base_p, elem_idx, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(_llvm_ptr_ty(), base_p, [idx_i64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v(base_p, elem_idx, n):
            return _llvm.LoadOp(T.vec(n, f16_ty), _gep(base_p, elem_idx, f16_ty)).result

        def load_f16_v64(base_p, elem_idx64, n):
            # i64-indexed load for offsets that can exceed i32 (T*H*TOPK etc.).
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_p, [elem_idx64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=f16_ty, noWrapFlags=0).result
            return _llvm.LoadOp(T.vec(n, f16_ty), gep).result

        def store_f16(val, base_p, elem_idx):
            _llvm.StoreOp(val, _gep(base_p, elem_idx, f16_ty))

        def lds_store1(val, idx):
            vector.store(vector.from_elements(T.vec(1, f16_ty), [val]), lds, [idx])

        def lds_load8(idx):
            return vector.load_op(T.vec(8, f16_ty), lds, [idx])

        tok = arith.index_cast(T.index, gpu.block_idx.x)
        lane = arith.index_cast(T.index, gpu.thread_idx.x)
        tt_v = arith.index_cast(T.index, total_tokens)
        tok_active = arith.cmpi(arith.CmpIPredicate.slt, tok, tt_v)
        tok_safe = arith.select(tok_active, tok, arith.index(0))

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)  # group g 0..3

        NH = arith.index(NUM_HEADS)
        HD = arith.index(HEAD_DIM)
        DQK = arith.index(D_QK)
        TOPK_I = arith.index(TOPK)
        c_zero_f16 = arith.constant(0.0, type=f16_ty)
        c_zero_mfma_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))

        lane_d_base = lane * arith.index(8)  # 64 lanes * 8 = 512 = HEAD_DIM (full-D coop load)
        c_zero_mfma_acc_ = c_zero_mfma_acc

        # ── D-block outer loop: stage qT/doT for a D_BLOCK slice into LDS, then loop
        # all key-tiles doing the output GEMM for that D slice. Bounds LDS to fit.
        for db in range_constexpr(DB_ITERS):
            d0 = db * D_BLOCK  # first d of this block
            gpu.barrier()  # protect qT/doT LDS from previous D-block's GEMM reads

            # Stage qT/doT[d0:d0+D_BLOCK, :] transposed into LDS [D_BLOCK, H_PAD].
            # Each lane owns 8 contiguous global-d; keep only those in [d0, d0+D_BLOCK).
            for h in range_constexpr(NUM_HEADS):
                q_row64 = arith.MulIOp(arith.index_cast(T.i64, tok_safe * NH + arith.index(h)), arith.index_cast(T.i64, DQK)).result
                o_row64 = arith.MulIOp(arith.index_cast(T.i64, tok_safe * NH + arith.index(h)), arith.index_cast(T.i64, HD)).result
                qv = load_f16_v64(q_ptr, arith.AddIOp(q_row64, arith.index_cast(T.i64, lane_d_base)).result, 8)
                dv = load_f16_v64(do_ptr, arith.AddIOp(o_row64, arith.index_cast(T.i64, lane_d_base)).result, 8)
                for e in range_constexpr(8):
                    d = lane_d_base + arith.index(e)
                    d_in = arith.AndIOp(
                        arith.cmpi(arith.CmpIPredicate.sge, d, arith.index(d0)),
                        arith.cmpi(arith.CmpIPredicate.slt, d, arith.index(d0 + D_BLOCK))).result
                    dloc = d - arith.index(d0)
                    dst = dloc * arith.index(H_PAD) + arith.index(h)
                    _if_q = scf.IfOp(d_in, [], has_else=False)
                    with ir.InsertionPoint(_if_q.then_block):
                        lds_store1(vector.extract(qv, static_position=[e], dynamic_position=[]), C_QT + dst)
                        lds_store1(vector.extract(dv, static_position=[e], dynamic_position=[]), C_DOT + dst)
                        scf.YieldOp([])
            if const_expr(HAS_HPAD):
                for hp in range_constexpr(H_PAD - NUM_HEADS):
                    h = NUM_HEADS + hp
                    for e in range_constexpr(8):
                        d = lane_d_base + arith.index(e)
                        d_in = arith.AndIOp(
                            arith.cmpi(arith.CmpIPredicate.sge, d, arith.index(d0)),
                            arith.cmpi(arith.CmpIPredicate.slt, d, arith.index(d0 + D_BLOCK))).result
                        dloc = d - arith.index(d0)
                        dst = dloc * arith.index(H_PAD) + arith.index(h)
                        _if_qp = scf.IfOp(d_in, [], has_else=False)
                        with ir.InsertionPoint(_if_qp.then_block):
                            lds_store1(c_zero_f16, C_QT + dst)
                            lds_store1(c_zero_f16, C_DOT + dst)
                            scf.YieldOp([])
            gpu.barrier()  # qT/doT staged for this D-block

            for kt in scf.for_(arith.index(0), TOPK_I, arith.index(BLOCK_K)):
                gpu.barrier()  # protect dS/P LDS from previous tile's GEMM reads

                # Stage dS/P for this key-tile into LDS [BLOCK_K, H_PAD] (re-staged
                # per D-block; small: BLOCK_K*H elems).
                _DSTOT = BLOCK_K * NUM_HEADS
                _ITERS = (_DSTOT + WARP_SIZE - 1) // WARP_SIZE
                for it in range_constexpr(_ITERS):
                    slot = lane + arith.index(it * WARP_SIZE)
                    in_r = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.index(_DSTOT))
                    kloc = slot // arith.index(NUM_HEADS)
                    h = slot % arith.index(NUM_HEADS)
                    key_glob = kt + kloc
                    key_in = arith.cmpi(arith.CmpIPredicate.slt, key_glob, TOPK_I)
                    key_safe = arith.select(key_in, key_glob, arith.index(0))
                    src64 = arith.AddIOp(
                        arith.MulIOp(arith.index_cast(T.i64, tok_safe * NH + h), arith.index_cast(T.i64, TOPK_I)).result,
                        arith.index_cast(T.i64, key_safe)).result
                    dsv0 = vector.extract(load_f16_v64(ds_ptr, src64, 1), static_position=[0], dynamic_position=[])
                    pv0 = vector.extract(load_f16_v64(p_ptr, src64, 1), static_position=[0], dynamic_position=[])
                    valid = arith.AndIOp(in_r, key_in).result
                    dsv0 = arith.select(valid, dsv0, c_zero_f16)
                    pv0 = arith.select(valid, pv0, c_zero_f16)
                    dst = kloc * arith.index(H_PAD) + h
                    _if = scf.IfOp(in_r, [], has_else=False)
                    with ir.InsertionPoint(_if.then_block):
                        lds_store1(dsv0, C_DS + dst)
                        lds_store1(pv0, C_P + dst)
                        scf.YieldOp([])
                if const_expr(HAS_HPAD):
                    for hp in range_constexpr(H_PAD - NUM_HEADS):
                        h = arith.index(NUM_HEADS + hp)
                        for kloc0 in range_constexpr(BLOCK_K):
                            dst = arith.index(kloc0) * arith.index(H_PAD) + h
                            lds_store1(c_zero_f16, C_DS + dst)
                            lds_store1(c_zero_f16, C_P + dst)

                gpu.barrier()  # dS/P staged

                # Output GEMM for this D-block: part[d,key] = sum_h qT[d,h]*dS + doT*P.
                kt_i32 = arith.index_cast(T.i32, kt)
                for dtl in range_constexpr(DT_PER_BLK):
                    part_acc = c_zero_mfma_acc_
                    for ks in range_constexpr(H_KSTEPS):
                        a_off = (arith.index(dtl * 16) + lane_mod_16) * arith.index(H_PAD) + arith.index(ks * 32) + lane_div_16 * arith.index(8)
                        b_off = lane_mod_16 * arith.index(H_PAD) + arith.index(ks * 32) + lane_div_16 * arith.index(8)
                        part_acc = _mfma_16x16x32(lds_load8(C_QT + a_off), lds_load8(C_DS + b_off), part_acc)
                        part_acc = _mfma_16x16x32(lds_load8(C_DOT + a_off), lds_load8(C_P + b_off), part_acc)
                    key_glob = arith.AddIOp(kt_i32, arith.index_cast(T.i32, lane_mod_16)).result
                    key_in = arith.cmpi(arith.CmpIPredicate.slt, key_glob, arith.constant(TOPK, type=T.i32))
                    _guard = scf.IfOp(arith.AndIOp(key_in, tok_active).result, [], has_else=False)
                    with ir.InsertionPoint(_guard.then_block):
                        key_idx = arith.index_cast(T.index, key_glob)
                        # i64 row offset: (tok*TOPK + key)*DQK can exceed i32 index range.
                        interm_row64 = arith.MulIOp(
                            arith.index_cast(T.i64, tok_safe * TOPK_I + key_idx),
                            arith.index_cast(T.i64, DQK)).result
                        for r in range_constexpr(4):
                            d = arith.index(d0 + dtl * 16) + lane_div_16 * arith.index(4) + arith.index(r)
                            off64 = arith.AddIOp(interm_row64, arith.index_cast(T.i64, d)).result
                            pv = vector.extract(part_acc, static_position=[r], dynamic_position=[])
                            gep = _llvm.GEPOp(_llvm_ptr_ty(), interm_ptr, [off64],
                                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                              elem_type=f16_ty, noWrapFlags=0).result
                            _llvm.StoreOp(arith.trunc_f(f16_ty, pv), gep)
                        scf.YieldOp([])

    @flyc.jit
    def launch_dsa_bwd_dkv_interm(
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
        launcher = dsa_bwd_dkv_interm_kernel(Q, dO, dS, Pin, Interm, total_tokens)
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
            return launch_dsa_bwd_dkv_interm(*args, **kwargs)

    return _launch
