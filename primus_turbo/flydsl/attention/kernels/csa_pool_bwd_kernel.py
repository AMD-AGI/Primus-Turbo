# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""csa_pool_bwd: V4 CSA backward with in-kernel pool scatter-add (FlyDSL per-row).

Mirrors the Triton reference ``_csa_attention_pool_bwd_kernel``: one program per
``(b, qhid, m)`` query row, re-materialising the joint online-softmax row from
the saved fp32 LSE (raw ``qk*scale`` domain) and the preprocess delta
``D[b,h,m] = sum_d dout*out``. Emits:

* ``dq``       [B, H, Sq, D]      direct fp32 store (one program owns the row)
* ``dk_local`` [B, H, Sq, D]      atomic-add (many m hit the same n)
* ``dv_local`` [B, H, Sq, D]      atomic-add
* ``dpool``    [B, P, D]          atomic-add scatter into the compressed pool
* ``dsink``    [H]                atomic-add per query row (when has_sink)

Per-row / per-lane layout (identical to ``csa_pool_fwd_kernel``): one wave (64
lanes) per program, each lane owns ``D_PER_LANE = D // 64`` contiguous head-dim
elements (lane ``L`` → dims ``L*D_PER_LANE .. +D_PER_LANE-1``). The QK / dP dot
products reduce across the wave via ``shuffle_xor``; the resulting scalar scores
are warp-uniform, so the softmax / dS math is replicated per lane and the
per-dim gradient terms stay lane-local.

Math (per query (b, h, m), see the Triton reference docstring):

  p_local[n]  = exp(qk_local[n]  - lse)   (masked to the SWA window)
  p_sparse[k] = exp(qk_sparse[k] - lse)   (masked to valid gathered slots)
  dP[.]       = sum_d dout[d] * {v_local | pool}[.,d]
  dS[.]       = p[.] * (dP[.] - delta)
  dq[d]       = scale * ( sum_n dS_local[n]*k_local[n,d]
                        + sum_k dS_sparse[k]*pool[k,d] )
  dk_local[n,d] += dS_local[n]  * scale * q[d]
  dv_local[n,d] += p_local[n]   * dout[d]
  dpool[idx,d]  += dS_sparse[k] * scale * q[d] + p_sparse[k] * dout[d]
  dsink[h]      += -p_sink * delta          (p_sink = exp(sink_h - lse))

dtype contract: all tensors loaded in input dtype; dots reduce in fp32; the
online P / dP / dS re-materialisation is fp32. Gradient buffers are fp32
(the launcher casts to input dtype). scale is applied to dq / dk / dpool once
(folded out of the accumulation loop where possible).
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from primus_turbo.flydsl.attention.kernels.kernels_common import dtype_to_elem_type, mfma_f32_16x16x32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm, math as math_dialect
from flydsl.expr import const_expr  # noqa: E402


KERNEL_NAME = "csa_pool_bwd_kernel"

_LLVM_GEP_DYNAMIC = -2147483648
_LOG2E_BWD = math.log2(math.e)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_csa_pool_bwd_module(
    num_heads,
    head_dim,
    swa_window,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    mqa_kv=False,
    has_sink=False,
    has_local=True,
    has_sparse=True,
    store_dpool=True,
):
    """Build the V4 CSA-from-pool backward per-row launcher (in-kernel scatter).

    ``mqa_kv`` selects whether ``k_local`` / ``v_local`` / ``dk`` / ``dv`` are
    shared across heads (MQA, ``HK == 1``) or per-head (MHA, ``HK == H``).
    ``dpool`` is always shared across heads.
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    assert HEAD_DIM % WARP_SIZE == 0, f"head_dim must be divisible by {WARP_SIZE}"
    D_PER_LANE = HEAD_DIM // WARP_SIZE  # 8 for D=512
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_pool_bwd_kernel(
        Q: fx.Tensor,
        K_LOCAL: fx.Tensor,
        V_LOCAL: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        Sink: fx.Tensor,
        DQ: fx.Tensor,
        DK_LOCAL: fx.Tensor,
        DV_LOCAL: fx.Tensor,
        DPOOL: fx.Tensor,
        DSINK: fx.Tensor,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K_LOCAL)
        vl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V_LOCAL)
        pool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), POOL)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TOPK)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DQ)
        dk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DK_LOCAL)
        dv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DV_LOCAL)
        dpool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DPOOL)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=True)
        if const_expr(has_sink):
            sink_rsrc = buffer_ops.create_buffer_resource(Sink, max_size=True)
            dsink_rsrc = buffer_ops.create_buffer_resource(DSINK, max_size=True)

        # ---- Load / store helpers ----
        def _gep(base_ptr, elem_idx, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v(base_ptr, elem_idx, n):
            vt = T.vec(n, f16_ty)
            return _llvm.LoadOp(vt, _gep(base_ptr, elem_idx, f16_ty)).result

        def load_i32_scalar(base_ptr, elem_idx):
            return _llvm.LoadOp(T.i32, _gep(base_ptr, elem_idx, T.i32)).result

        def store_f32_scalar(val, base_ptr, elem_idx):
            _llvm.StoreOp(val, _gep(base_ptr, elem_idx, f32_ty))

        def atomic_add_f32(base_ptr, elem_idx, val):
            _llvm.AtomicRMWOp(
                _llvm.AtomicBinOp.fadd,
                _gep(base_ptr, elem_idx, f32_ty),
                val,
                _llvm.AtomicOrdering.monotonic,
            )

        # ---- Thread / program ----
        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        pid_bh = arith.index_cast(T.index, gpu.block_idx.y)
        lane = arith.index_cast(T.index, gpu.thread_idx.x)

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)
        pool_size_v = arith.index_cast(T.index, pool_size)
        pool_size_i32 = pool_size
        if const_expr(hasattr(pool_size_i32, "ir_value")):
            pool_size_i32 = pool_size_i32.ir_value()
        seq_len_i32 = seq_len
        if const_expr(hasattr(seq_len_i32, "ir_value")):
            seq_len_i32 = seq_len_i32.ir_value()
        K_topk_i32 = K_topk
        if const_expr(hasattr(K_topk_i32, "ir_value")):
            K_topk_i32 = K_topk_i32.ir_value()

        bid = pid_bh // arith.index(NUM_HEADS)
        qhid = pid_bh % arith.index(NUM_HEADS)

        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        # ---- Constants ----
        NEG_INF_F = -1.0e30
        c_neg_inf = arith.constant(NEG_INF_F, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        zero_vec = arith.constant_vector(0.0, T.vec(D_PER_LANE, f32_ty))

        lane_i32 = arith.index_cast(T.i32, lane)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)

        def warp_reduce_sum_f32(v):
            cur = v
            for off in [32, 16, 8, 4, 2, 1]:
                xor_amt = arith.constant(off, type=T.i32)
                peer = arith.ArithValue(cur).shuffle_xor(xor_amt, width_i32)
                cur = arith.AddFOp(cur, peer, fastmath=fm_fast).result
            return cur

        def vec_dot_f32(a_vec, b_vec):
            s = c_zero_f
            for i in range_constexpr(D_PER_LANE):
                av = vector.extract(a_vec, static_position=[i], dynamic_position=[])
                bv = vector.extract(b_vec, static_position=[i], dynamic_position=[])
                p = arith.MulFOp(av, bv, fastmath=fm_fast).result
                s = arith.AddFOp(s, p, fastmath=fm_fast).result
            return s

        # ---- Per-row Q / DOUT lane vectors + scalars ----
        q_row_base = ((bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
        q_lane_off = q_row_base + lane * arith.index(D_PER_LANE)
        q_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(q_ptr, q_lane_off, D_PER_LANE))
        q_vec = arith.select(q_active, q_vec, zero_vec)
        do_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(do_ptr, q_lane_off, D_PER_LANE))
        do_vec = arith.select(q_active, do_vec, zero_vec)

        lse_off = (bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + pid_m_safe
        lse_off_i32 = arith.index_cast(T.i32, lse_off)
        lse_val = buffer_ops.buffer_load(lse_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)
        delta_val = buffer_ops.buffer_load(delta_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)

        # ---- dq lane accumulator (D_PER_LANE elems) ----
        dq_acc = [c_zero_f for _ in range_constexpr(D_PER_LANE)]

        # ---- Sink gradient (once per row, lane 0) ----
        if const_expr(has_sink):
            qhid_i32 = arith.index_cast(T.i32, qhid)
            sink_h = buffer_ops.buffer_load(sink_rsrc, qhid_i32, vec_width=1, dtype=f32_ty)
            ps_diff = arith.SubFOp(sink_h, lse_val, fastmath=fm_fast).result
            p_sink = math_dialect.exp(ps_diff, fastmath=fm_fast)
            neg = arith.SubFOp(c_zero_f, p_sink, fastmath=fm_fast).result
            dsink_contrib = arith.MulFOp(neg, delta_val, fastmath=fm_fast).result
            is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.index(0))
            do_sink = arith.AndIOp(is_lane0, q_active).result
            dsink_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DSINK)
            _if_sink = scf.IfOp(do_sink, [], has_else=False)
            with ir.InsertionPoint(_if_sink.then_block):
                atomic_add_f32(dsink_ptr, qhid, dsink_contrib)
                scf.YieldOp([])

        # ---- Local SWA loop bounds (match forward) ----
        _pid_p1 = pid_m + arith.index(1)
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _pid_p1, seq_len_v)
        n_loop_end = arith.select(_le_seq, _pid_p1, seq_len_v)
        SWA = arith.index(int(swa_window))
        _ge_w = arith.cmpi(arith.CmpIPredicate.sge, _pid_p1, SWA)
        n_loop_start = arith.select(_ge_w, _pid_p1 - SWA, arith.index(0))

        pid_m_i32 = arith.index_cast(T.i32, pid_m)
        w_i32 = arith.constant(int(swa_window), type=T.i32)

        # ==== LOCAL SWA loop (stride 1) ====
        if const_expr(has_local):
          dq_res_local = None
          for n_idx, dq_carry, dq_res_local in scf.for_(
            n_loop_start, n_loop_end, arith.index(1), iter_args=dq_acc,
          ):
            dq_acc = [dq_carry[d] for d in range_constexpr(D_PER_LANE)]
            n_i32 = arith.index_cast(T.i32, n_idx)

            # mask: causal (n>m) | swa (n+W<=m) | oob (n>=seq)
            is_causal = arith.cmpi(arith.CmpIPredicate.sgt, n_i32, pid_m_i32)
            _npw = arith.AddIOp(n_i32, w_i32).result
            is_swa = arith.cmpi(arith.CmpIPredicate.sle, _npw, pid_m_i32)
            is_oob = arith.cmpi(arith.CmpIPredicate.sge, n_i32, seq_len_i32)
            bad = arith.OrIOp(arith.OrIOp(is_causal, is_swa).result, is_oob).result

            if const_expr(mqa_kv):
                kl_row_base = (bid * seq_len_v + n_idx) * arith.index(HEAD_DIM)
            else:
                kl_row_base = ((bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + n_idx) * arith.index(HEAD_DIM)
            kl_lane_off = kl_row_base + lane * arith.index(D_PER_LANE)
            kl_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(kl_ptr, kl_lane_off, D_PER_LANE))
            vl_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(vl_ptr, kl_lane_off, D_PER_LANE))

            qk = warp_reduce_sum_f32(vec_dot_f32(kl_vec, q_vec))
            qk = arith.MulFOp(qk, c_sm_scale_f, fastmath=fm_fast).result
            qk = arith.select(bad, c_neg_inf, qk)
            p = math_dialect.exp(arith.SubFOp(qk, lse_val, fastmath=fm_fast).result, fastmath=fm_fast)

            dp = warp_reduce_sum_f32(vec_dot_f32(do_vec, vl_vec))
            ds = arith.MulFOp(p, arith.SubFOp(dp, delta_val, fastmath=fm_fast).result, fastmath=fm_fast).result
            ds_scaled = arith.MulFOp(ds, c_sm_scale_f, fastmath=fm_fast).result

            new_dq = []
            dk_cs = []
            dv_cs = []
            for d in range_constexpr(D_PER_LANE):
                kd = vector.extract(kl_vec, static_position=[d], dynamic_position=[])
                qd = vector.extract(q_vec, static_position=[d], dynamic_position=[])
                dod = vector.extract(do_vec, static_position=[d], dynamic_position=[])
                # dq += ds * scale * k
                contrib = arith.MulFOp(ds_scaled, kd, fastmath=fm_fast).result
                new_dq.append(arith.AddFOp(dq_acc[d], contrib, fastmath=fm_fast).result)
                # dk += ds * scale * q ; dv += p * dout
                dk_cs.append(arith.MulFOp(ds_scaled, qd, fastmath=fm_fast).result)
                dv_cs.append(arith.MulFOp(p, dod, fastmath=fm_fast).result)
            # Single uniform guard around all atomics for this key.
            _if_kv = scf.IfOp(q_active, [], has_else=False)
            with ir.InsertionPoint(_if_kv.then_block):
                for d in range_constexpr(D_PER_LANE):
                    elem_off = kl_lane_off + arith.index(d)
                    atomic_add_f32(dk_ptr, elem_off, dk_cs[d])
                    atomic_add_f32(dv_ptr, elem_off, dv_cs[d])
                scf.YieldOp([])
            yield new_dq

          dq_acc = [dq_res_local[d] for d in range_constexpr(D_PER_LANE)]

        # ==== SPARSE loop (stride 1) ====
        if const_expr(has_sparse):
          dq_res_sparse = None
          for k_idx, dq_carry_s, dq_res_sparse in scf.for_(
            arith.index(0), K_topk_v, arith.index(1), iter_args=dq_acc,
          ):
            dq_acc = [dq_carry_s[d] for d in range_constexpr(D_PER_LANE)]

            topk_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_idx
            idx_i32 = load_i32_scalar(topk_ptr, topk_off)
            is_nonneg = arith.cmpi(arith.CmpIPredicate.sge, idx_i32, arith.constant(0, type=T.i32))
            is_lt_p = arith.cmpi(arith.CmpIPredicate.slt, idx_i32, pool_size_i32)
            valid = arith.AndIOp(is_nonneg, is_lt_p).result
            valid_active = arith.AndIOp(valid, q_active).result
            idx_safe_i32 = arith.select(valid, idx_i32, arith.constant(0, type=T.i32))
            idx_safe = arith.index_cast(T.index, idx_safe_i32)
            pool_row_base = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
            pool_lane_off = pool_row_base + lane * arith.index(D_PER_LANE)
            g_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(pool_ptr, pool_lane_off, D_PER_LANE))

            qk = warp_reduce_sum_f32(vec_dot_f32(g_vec, q_vec))
            qk = arith.MulFOp(qk, c_sm_scale_f, fastmath=fm_fast).result
            qk = arith.select(valid_active, qk, c_neg_inf)
            p = math_dialect.exp(arith.SubFOp(qk, lse_val, fastmath=fm_fast).result, fastmath=fm_fast)

            dp = warp_reduce_sum_f32(vec_dot_f32(do_vec, g_vec))
            ds = arith.MulFOp(p, arith.SubFOp(dp, delta_val, fastmath=fm_fast).result, fastmath=fm_fast).result
            ds_scaled = arith.MulFOp(ds, c_sm_scale_f, fastmath=fm_fast).result

            new_dq = []
            for d in range_constexpr(D_PER_LANE):
                gd = vector.extract(g_vec, static_position=[d], dynamic_position=[])
                contrib = arith.MulFOp(ds_scaled, gd, fastmath=fm_fast).result
                new_dq.append(arith.AddFOp(dq_acc[d], contrib, fastmath=fm_fast).result)
            if const_expr(store_dpool):
                # dpool += ds*scale*q + p*dout (only when this kernel owns dpool;
                # the split launcher uses the dedicated dpool kernel instead).
                dpool_cs = []
                for d in range_constexpr(D_PER_LANE):
                    qd = vector.extract(q_vec, static_position=[d], dynamic_position=[])
                    dod = vector.extract(do_vec, static_position=[d], dynamic_position=[])
                    t0 = arith.MulFOp(ds_scaled, qd, fastmath=fm_fast).result
                    t1 = arith.MulFOp(p, dod, fastmath=fm_fast).result
                    dpool_cs.append(arith.AddFOp(t0, t1, fastmath=fm_fast).result)
                _if_dp = scf.IfOp(valid_active, [], has_else=False)
                with ir.InsertionPoint(_if_dp.then_block):
                    for d in range_constexpr(D_PER_LANE):
                        elem_off = pool_lane_off + arith.index(d)
                        atomic_add_f32(dpool_ptr, elem_off, dpool_cs[d])
                    scf.YieldOp([])
            yield new_dq

          dq_acc = [dq_res_sparse[d] for d in range_constexpr(D_PER_LANE)]

        # ==== Store dq (direct, fp32) ====
        _o_guard = scf.IfOp(q_active, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            dq_row_base = ((bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            dq_lane_off = dq_row_base + lane * arith.index(D_PER_LANE)
            for d in range_constexpr(D_PER_LANE):
                store_f32_scalar(dq_acc[d], dq_ptr, dq_lane_off + arith.index(d))
            scf.YieldOp([])

    @flyc.jit
    def launch_csa_pool_bwd(
        Q: fx.Tensor,
        K_LOCAL: fx.Tensor,
        V_LOCAL: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        Sink: fx.Tensor,
        DQ: fx.Tensor,
        DK_LOCAL: fx.Tensor,
        DV_LOCAL: fx.Tensor,
        DPOOL: fx.Tensor,
        DSINK: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        grid_x = sl_idx
        grid_y = bs_idx * arith.index(NUM_HEADS)

        launcher = csa_pool_bwd_kernel(
            Q, K_LOCAL, V_LOCAL, POOL, TOPK, DOUT, LSE, DELTA, Sink,
            DQ, DK_LOCAL, DV_LOCAL, DPOOL, DSINK,
            seq_len, K_topk, pool_size,
        )

        ctx = CompilationContext.get_current()
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)

        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"),
                ir.StringAttr.get("true"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"),
                ir.StringAttr.get("true"),
            ]))
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_pool_bwd(*args, **kwargs)

    return _launch


build_csa_pool_bwd_module_primary = build_csa_pool_bwd_module


def build_csa_pool_bwd_dpool_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    mqa_kv=True,
    k_block=1,
):
    """CSA dpool-only backward: head-summed, atomic-free partial buffer.

    Grid: (ceil(K_topk / K_BLOCK), Sq, B). One wave per (b, m, k-block). For each
    of the ``K_BLOCK`` sparse slots it owns, the wave loops over ALL heads and
    accumulates ``dpool_contrib = sum_h (dS[h,k]*scale*q[h] + p[h,k]*dout[h])``
    in registers (per-lane ``D_PER_LANE`` elems), then PLAIN-STORES it to a
    partial buffer ``DPOOL_PART[B, Sq, K_topk, D]`` (each (b,m,k) owned by exactly
    one program — no atomics). The host reduces the partial into ``dpool[B,P,D]``
    via ``index_add`` over the ``topk`` indices.

    This removes the pool-row atomic contention (the dominant CSA-bwd cost): the
    per-head atomic kernel serialises ``Sq*H*K`` atomics onto ``P`` rows; here
    there are zero atomics and the head sum is register-local.
    """
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    K_BLOCK = int(k_block)
    assert HEAD_DIM % WARP_SIZE == 0
    D_PER_LANE = HEAD_DIM // WARP_SIZE
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_pool_bwd_dpool_kernel(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DPOOL_PART: fx.Tensor,  # [B, Sq, K_topk, D] fp32
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        pool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), POOL)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TOPK)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        part_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DPOOL_PART)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=True)

        def _gep(base_ptr, elem_idx, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v(base_ptr, elem_idx, n):
            return _llvm.LoadOp(T.vec(n, f16_ty), _gep(base_ptr, elem_idx, f16_ty)).result

        def load_i32_scalar(base_ptr, elem_idx):
            return _llvm.LoadOp(T.i32, _gep(base_ptr, elem_idx, T.i32)).result

        def store_f32(val, base_ptr, elem_idx):
            _llvm.StoreOp(val, _gep(base_ptr, elem_idx, f32_ty))

        pid_kb = arith.index_cast(T.index, gpu.block_idx.x)
        pid_m = arith.index_cast(T.index, gpu.block_idx.y)
        bid = arith.index_cast(T.index, gpu.block_idx.z)
        lane = arith.index_cast(T.index, gpu.thread_idx.x)

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)
        pool_size_v = arith.index_cast(T.index, pool_size)
        pool_size_i32 = pool_size
        if const_expr(hasattr(pool_size_i32, "ir_value")):
            pool_size_i32 = pool_size_i32.ir_value()

        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_neg_inf = arith.constant(-1.0e30, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        zero_vec = arith.constant_vector(0.0, T.vec(D_PER_LANE, f32_ty))
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)

        def warp_reduce_sum_f32(v):
            cur = v
            for off in [32, 16, 8, 4, 2, 1]:
                peer = arith.ArithValue(cur).shuffle_xor(arith.constant(off, type=T.i32), width_i32)
                cur = arith.AddFOp(cur, peer, fastmath=fm_fast).result
            return cur

        def vec_dot_f32(a_vec, b_vec):
            s = c_zero_f
            for i in range_constexpr(D_PER_LANE):
                av = vector.extract(a_vec, static_position=[i], dynamic_position=[])
                bv = vector.extract(b_vec, static_position=[i], dynamic_position=[])
                s = arith.AddFOp(s, arith.MulFOp(av, bv, fastmath=fm_fast).result, fastmath=fm_fast).result
            return s

        lane_off = lane * arith.index(D_PER_LANE)

        # For each owned k-slot: gather pool row, sum dpool over heads, store.
        for kk in range_constexpr(K_BLOCK):
            k_idx = pid_kb * arith.index(K_BLOCK) + arith.index(kk)
            k_in_range = arith.cmpi(arith.CmpIPredicate.slt, k_idx, K_topk_v)
            k_idx_safe = arith.select(k_in_range, k_idx, arith.index(0))

            topk_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_idx_safe
            idx_i32 = load_i32_scalar(topk_ptr, topk_off)
            is_nonneg = arith.cmpi(arith.CmpIPredicate.sge, idx_i32, arith.constant(0, type=T.i32))
            is_lt_p = arith.cmpi(arith.CmpIPredicate.slt, idx_i32, pool_size_i32)
            valid = arith.AndIOp(arith.AndIOp(is_nonneg, is_lt_p).result, k_in_range).result
            valid_active = arith.AndIOp(valid, q_active).result
            idx_safe_i32 = arith.select(valid, idx_i32, arith.constant(0, type=T.i32))
            idx_safe = arith.index_cast(T.index, idx_safe_i32)
            pool_lane_off = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM) + lane_off
            g_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(pool_ptr, pool_lane_off, D_PER_LANE))

            dpool_acc = [c_zero_f for _ in range_constexpr(D_PER_LANE)]
            for h in range_constexpr(NUM_HEADS):
                q_lane_off = ((bid * arith.index(NUM_HEADS) + arith.index(h)) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM) + lane_off
                q_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(q_ptr, q_lane_off, D_PER_LANE))
                do_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), load_f16_v(do_ptr, q_lane_off, D_PER_LANE))
                lse_off = (bid * arith.index(NUM_HEADS) + arith.index(h)) * seq_len_v + pid_m_safe
                lse_off_i32 = arith.index_cast(T.i32, lse_off)
                lse_h = buffer_ops.buffer_load(lse_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)
                delta_h = buffer_ops.buffer_load(delta_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)

                qk = warp_reduce_sum_f32(vec_dot_f32(g_vec, q_vec))
                qk = arith.MulFOp(qk, c_sm_scale_f, fastmath=fm_fast).result
                qk = arith.select(valid_active, qk, c_neg_inf)
                p = math_dialect.exp(arith.SubFOp(qk, lse_h, fastmath=fm_fast).result, fastmath=fm_fast)
                dp = warp_reduce_sum_f32(vec_dot_f32(do_vec, g_vec))
                ds = arith.MulFOp(p, arith.SubFOp(dp, delta_h, fastmath=fm_fast).result, fastmath=fm_fast).result
                ds_scaled = arith.MulFOp(ds, c_sm_scale_f, fastmath=fm_fast).result
                for d in range_constexpr(D_PER_LANE):
                    qd = vector.extract(q_vec, static_position=[d], dynamic_position=[])
                    dod = vector.extract(do_vec, static_position=[d], dynamic_position=[])
                    t0 = arith.MulFOp(ds_scaled, qd, fastmath=fm_fast).result
                    t1 = arith.MulFOp(p, dod, fastmath=fm_fast).result
                    dpool_acc[d] = arith.AddFOp(dpool_acc[d], arith.AddFOp(t0, t1, fastmath=fm_fast).result, fastmath=fm_fast).result

            # Plain store to partial[b, m, k, :] (unique owner). Zero for
            # invalid/oob so host reduction sums blindly.
            part_row = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_idx_safe
            part_lane_off = part_row * arith.index(HEAD_DIM) + lane_off
            _if = scf.IfOp(arith.AndIOp(k_in_range, q_active).result, [], has_else=False)
            with ir.InsertionPoint(_if.then_block):
                for d in range_constexpr(D_PER_LANE):
                    c = arith.select(valid, dpool_acc[d], c_zero_f)
                    store_f32(c, part_ptr, part_lane_off + arith.index(d))
                scf.YieldOp([])

    @flyc.jit
    def launch_csa_pool_bwd_dpool(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DPOOL_PART: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        kt_idx = arith.index_cast(T.index, K_topk)
        grid_x = (kt_idx + arith.index(K_BLOCK - 1)) // arith.index(K_BLOCK)
        launcher = csa_pool_bwd_dpool_kernel(
            Q, POOL, TOPK, DOUT, LSE, DELTA, DPOOL_PART, seq_len, K_topk, pool_size,
        )
        ctx = CompilationContext.get_current()
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
        launcher.launch(grid=(grid_x, sl_idx, bs_idx), block=(BLOCK_SIZE, 1, 1), stream=stream)

    compile_hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_pool_bwd_dpool(*args, **kwargs)

    return _launch


def build_csa_pool_bwd_dpool_mfma_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    mqa_kv=True,
    k_block=1,
):
    """CSA dpool-only backward, all-MFMA (head-as-contraction).

    Same contract as :func:`build_csa_pool_bwd_dpool_module` (grid signature +
    partial buffer ``DPOOL_PART[B, Sq, K_topk, D]``, host ``index_add`` reduce),
    but every GEMM runs on the matrix cores instead of per-head VALU dots:

    * ``qk[h,k] = q[h,d] @ gathered[k,d]``   (contract d, head-as-M MFMA)
    * ``dp[h,k] = dout[h,d] @ gathered[k,d]`` (contract d)
    * ``part[d,k] = sum_h qT[d,h]*ds_scaled[h,k] + doutT[d,h]*p[h,k]``
      (contract **h**, the AV-MFMA step)

    Grid: ``(Sq, B)`` — one wave per ``(b, m)`` contracts ALL heads so the
    partial stays atomic-free (unique ``(b,m,k)`` owner). ``q`` / ``dout`` are
    staged transposed into LDS (``[D, H]``) for the output GEMM's A-operand; the
    per-tile ``ds`` / ``p`` B-operands are staged ``[BLOCK_K, H]``. ``gathered``
    is read straight from HBM per key (d-contiguous).
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    BLOCK_K = 16
    assert HEAD_DIM % 32 == 0
    assert NUM_HEADS % 16 == 0, "dpool MFMA path needs NUM_HEADS % 16 == 0"
    K_CHUNKS = HEAD_DIM // 32       # d-contraction chunks for QK/DP
    D_TILES = HEAD_DIM // 16        # output d-tiles
    H_TILES = NUM_HEADS // 16       # head-tiles for QK/DP score layout
    H_KSTEPS = (NUM_HEADS + 31) // 32  # head-contraction MFMA steps (K=32)
    # The head-contraction GEMM contracts K=32 per step; pad the LDS head axis
    # up to a multiple of 32 and zero the padding so the extra head slots
    # contribute nothing (needed when NUM_HEADS % 32 != 0, e.g. small tests).
    H_PAD = H_KSTEPS * 32
    HAS_HPAD = H_PAD != NUM_HEADS
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    LDS_QT = HEAD_DIM * H_PAD       # qT [D, H_PAD]
    LDS_DOT = HEAD_DIM * H_PAD      # doutT [D, H_PAD]
    LDS_DS = BLOCK_K * H_PAD        # ds_scaled [BLOCK_K, H_PAD]
    LDS_P = BLOCK_K * H_PAD         # p [BLOCK_K, H_PAD]
    LDS_TOTAL = LDS_QT + LDS_DOT + LDS_DS + LDS_P

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name=f"csa_dpool_mfma_smem_H{NUM_HEADS}",
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_pool_bwd_dpool_mfma_kernel(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DPOOL_PART: fx.Tensor,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        pool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), POOL)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TOPK)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        part_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DPOOL_PART)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=True)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()
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

        def load_i32_scalar(base_p, elem_idx):
            return _llvm.LoadOp(T.i32, _gep(base_p, elem_idx, T.i32)).result

        def store_f32(val, base_p, elem_idx):
            _llvm.StoreOp(val, _gep(base_p, elem_idx, f32_ty))

        def lds_store1(val, idx):
            vector.store(vector.from_elements(T.vec(1, f16_ty), [val]), lds, [idx])

        def lds_load8(idx):
            return vector.load_op(T.vec(8, f16_ty), lds, [idx])

        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        bid = arith.index_cast(T.index, gpu.block_idx.y)
        lane = arith.index_cast(T.index, gpu.thread_idx.x)

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)
        pool_size_v = arith.index_cast(T.index, pool_size)
        pool_size_i32 = pool_size
        if const_expr(hasattr(pool_size_i32, "ir_value")):
            pool_size_i32 = pool_size_i32.ir_value()
        K_topk_i32 = K_topk
        if const_expr(hasattr(K_topk_i32, "ir_value")):
            K_topk_i32 = K_topk_i32.ir_value()

        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)   # group g 0..3
        lane_mod_16_i32 = arith.index_cast(T.i32, lane_mod_16)
        lane_div_16_i32 = arith.index_cast(T.i32, lane_div_16)

        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_neg_inf = arith.constant(-1.0e30, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        c_log2e_f = arith.constant(_LOG2E_BWD, type=f32_ty)
        zero_mfma_pack = arith.constant_vector(0.0, T.vec(8, f16_ty))
        c_zero_mfma_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))

        # ── Stage qT / doutT into LDS [D, H] (transpose). Each lane loads vec8
        # of d for a head, scatters to lds[d*H + h]. Loop all heads. ──
        lane_d_base = lane * arith.index(8)   # 64 lanes * 8 = 512 = HEAD_DIM
        for h in range_constexpr(NUM_HEADS):
            row = ((bid * arith.index(NUM_HEADS) + arith.index(h)) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            qv = load_f16_v(q_ptr, row + lane_d_base, 8)
            dv = load_f16_v(do_ptr, row + lane_d_base, 8)
            for e in range_constexpr(8):
                d = lane_d_base + arith.index(e)
                dst = d * arith.index(H_PAD) + arith.index(h)
                lds_store1(vector.extract(qv, static_position=[e], dynamic_position=[]), C_QT + dst)
                lds_store1(vector.extract(dv, static_position=[e], dynamic_position=[]), C_DOT + dst)
        c_zero_f16 = arith.constant(0.0, type=f16_ty)
        if const_expr(HAS_HPAD):
            # Zero the padding head columns [NUM_HEADS, H_PAD) of qT / doutT so
            # the head-contraction GEMM's extra K lanes contribute nothing.
            for hp in range_constexpr(H_PAD - NUM_HEADS):
                h = NUM_HEADS + hp
                for e in range_constexpr(8):
                    d = lane_d_base + arith.index(e)
                    dst = d * arith.index(H_PAD) + arith.index(h)
                    lds_store1(c_zero_f16, C_QT + dst)
                    lds_store1(c_zero_f16, C_DOT + dst)

        # Q/DOUT B-operand packs for QK/DP, per head-tile: lane L -> head =
        # ht*16 + L%16, d = ck*32 + g*8. Read straight from HBM.
        def qdo_bpack(base_p, ht, ck):
            head = arith.index(ht * 16) + lane_mod_16
            row = ((bid * arith.index(NUM_HEADS) + head) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            off = row + arith.index(ck * 32) + lane_div_16 * arith.index(8)
            return load_f16_v(base_p, off, 8)

        # Validity + pool row base for global key kk (i32).
        def key_meta(kk_i32):
            ko = arith.cmpi(arith.CmpIPredicate.sge, kk_i32, K_topk_i32)
            kk_idx = arith.index_cast(T.index, arith.select(ko, arith.constant(0, type=T.i32), kk_i32))
            toff = (bid * seq_len_v + pid_m_safe) * K_topk_v + kk_idx
            idr = load_i32_scalar(topk_ptr, toff)
            ineg = arith.cmpi(arith.CmpIPredicate.slt, idr, arith.constant(0, type=T.i32))
            igp = arith.cmpi(arith.CmpIPredicate.sge, idr, pool_size_i32)
            inv = arith.OrIOp(arith.OrIOp(ko, ineg).result, igp).result
            idx_safe = arith.index_cast(T.index, arith.select(inv, arith.constant(0, type=T.i32), idr))
            prow = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
            return inv, prow

        gpu.barrier()  # qT/doutT staged

        # ── Loop over K-tiles of BLOCK_K keys ──
        for kt in scf.for_(arith.index(0), K_topk_v, arith.index(BLOCK_K)):
            kt_i32 = arith.index_cast(T.i32, kt)
            gpu.barrier()  # protect ds/p LDS from previous tile's output reads

            # A-operand gathered: lane L -> key = L%16, d = ck*32 + g*8.
            key_pos_i32 = arith.AddIOp(kt_i32, lane_mod_16_i32).result
            _inv_a, pool_row_base = key_meta(key_pos_i32)
            g_packs = []
            for ck in range_constexpr(K_CHUNKS):
                poff = pool_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                g_packs.append(load_f16_v(pool_ptr, poff, 8))

            # For each head-tile: QK + DP MFMA, then compute p / ds and stage.
            for ht in range_constexpr(H_TILES):
                qk_acc = c_zero_mfma_acc
                dp_acc = c_zero_mfma_acc
                for ck in range_constexpr(K_CHUNKS):
                    qb = qdo_bpack(q_ptr, ht, ck)
                    dob = qdo_bpack(do_ptr, ht, ck)
                    qk_acc = mfma_f32_16x16x32(g_packs[ck], qb, qk_acc, dtype_str)
                    dp_acc = mfma_f32_16x16x32(g_packs[ck], dob, dp_acc, dtype_str)
                # C-layout: lane L holds [key = g*4+r, head = ht*16 + L%16].
                head = arith.index(ht * 16) + lane_mod_16
                lse_off = (bid * arith.index(NUM_HEADS) + head) * seq_len_v + pid_m_safe
                lse_off_i32 = arith.index_cast(T.i32, lse_off)
                lse_h = buffer_ops.buffer_load(lse_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)
                delta_h = buffer_ops.buffer_load(delta_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)
                for r in range_constexpr(4):
                    key_r_i32 = arith.AddIOp(kt_i32, arith.AddIOp(
                        arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                        arith.constant(r, type=T.i32)).result).result
                    inv_r, _pr = key_meta(key_r_i32)
                    qk_raw = vector.extract(qk_acc, static_position=[r], dynamic_position=[])
                    dp_raw = vector.extract(dp_acc, static_position=[r], dynamic_position=[])
                    qk_s = arith.MulFOp(qk_raw, c_sm_scale_f, fastmath=fm_fast).result
                    qk_s = arith.select(inv_r, c_neg_inf, qk_s)
                    p = arith.ArithValue(arith.MulFOp(arith.SubFOp(qk_s, lse_h, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)
                    ds = arith.MulFOp(p, arith.SubFOp(dp_raw, delta_h, fastmath=fm_fast).result, fastmath=fm_fast).result
                    ds_scaled = arith.MulFOp(ds, c_sm_scale_f, fastmath=fm_fast).result
                    # zero invalid explicitly (p already 0, but ds uses dp_raw).
                    ds_scaled = arith.select(inv_r, c_zero_f, ds_scaled)
                    p = arith.select(inv_r, c_zero_f, p)
                    # Stage into [BLOCK_K, H]: kloc = g*4+r, head = ht*16+L%16.
                    kloc = arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result
                    kloc = arith.index_cast(T.index, arith.AddIOp(kloc, arith.constant(r, type=T.i32)).result)
                    dst = kloc * arith.index(H_PAD) + head
                    lds_store1(arith.trunc_f(f16_ty, ds_scaled), C_DS + dst)
                    lds_store1(arith.trunc_f(f16_ty, p), C_P + dst)

            gpu.barrier()  # ds/p staged

            # ── Output GEMM: part[d, key] = sum_h qT[d,h]*ds[h,k] + doutT[d,h]*p[h,k]
            # A pack qT: lane L -> d = dt*16 + L%16, head = ks*32 + g*8.
            # B pack ds: lane L -> key = L%16, head = ks*32 + g*8.
            for dt in range_constexpr(D_TILES):
                part_acc = c_zero_mfma_acc
                for ks in range_constexpr(H_KSTEPS):
                    a_qt = (arith.index(dt * 16) + lane_mod_16) * arith.index(H_PAD) + arith.index(ks * 32) + lane_div_16 * arith.index(8)
                    a_dot = a_qt
                    b_off = lane_mod_16 * arith.index(H_PAD) + arith.index(ks * 32) + lane_div_16 * arith.index(8)
                    qt_pack = lds_load8(C_QT + a_qt)
                    ds_pack = lds_load8(C_DS + b_off)
                    part_acc = mfma_f32_16x16x32(qt_pack, ds_pack, part_acc, dtype_str)
                    dot_pack = lds_load8(C_DOT + a_dot)
                    p_pack = lds_load8(C_P + b_off)
                    part_acc = mfma_f32_16x16x32(dot_pack, p_pack, part_acc, dtype_str)
                # C: lane L -> part[d = dt*16 + g*4+r, key = L%16].
                key_glob = arith.AddIOp(kt_i32, lane_mod_16_i32).result
                key_in = arith.cmpi(arith.CmpIPredicate.slt, key_glob, K_topk_i32)
                _guard = scf.IfOp(arith.AndIOp(key_in, q_active).result, [], has_else=False)
                with ir.InsertionPoint(_guard.then_block):
                    key_idx = arith.index_cast(T.index, key_glob)
                    part_row = ((bid * seq_len_v + pid_m_safe) * K_topk_v + key_idx) * arith.index(HEAD_DIM)
                    for r in range_constexpr(4):
                        d = arith.index(dt * 16) + lane_div_16 * arith.index(4) + arith.index(r)
                        pv = vector.extract(part_acc, static_position=[r], dynamic_position=[])
                        store_f32(pv, part_ptr, part_row + d)
                    scf.YieldOp([])

    @flyc.jit
    def launch_csa_pool_bwd_dpool_mfma(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DPOOL_PART: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        launcher = csa_pool_bwd_dpool_mfma_kernel(
            Q, POOL, TOPK, DOUT, LSE, DELTA, DPOOL_PART, seq_len, K_topk, pool_size,
        )
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
        launcher.launch(grid=(sl_idx, bs_idx, arith.index(1)), block=(BLOCK_SIZE, 1, 1), stream=stream)

    compile_hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_pool_bwd_dpool_mfma(*args, **kwargs)

    return _launch


def build_csa_pool_bwd_dq_mfma_module(
    num_heads,
    head_dim,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    """CSA sparse-branch dq, all-MFMA (head-as-M QK/DP + AV over keys).

    Computes the sparse pool contribution to ``dq``:
        ``dq[h,d] = scale * sum_k ds_sparse[h,k] * gathered[k,d]``
    where ``p[h,k]=exp(qk*scale - lse_h)``, ``dp[h,k]=dout_h · gathered_k``,
    ``ds[h,k]=p*(dp - delta_h)``. Mirrors the sparse-forward AV structure but
    consumes the saved joint ``lse`` / preprocess ``delta`` instead of an online
    softmax, and adds a second score GEMM for ``dp``.

    Grid: ``(Sq, cdiv(HEAD_Q, 16), B)``. Writes ``dq_sparse[B, H, Sq, D]`` fp32
    (the launcher adds it into the local-stream dq). Same head-block tiling and
    LDS gathered^T / ds^T staging as :func:`build_csa_pool_sparse_fwd_module`.
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    BLOCK_H = 16
    BLOCK_K = 32
    assert HEAD_DIM % 32 == 0
    K_CHUNKS = HEAD_DIM // 32
    D_TILES = HEAD_DIM // 16
    N_SUB = BLOCK_K // 16
    if const_expr(sm_scale is None):
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    LDS_GT = HEAD_DIM * BLOCK_K       # gathered^T [d, kloc]
    LDS_DS = BLOCK_H * BLOCK_K        # ds_scaled [head, kloc]
    LDS_TOTAL = LDS_GT + LDS_DS

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="csa_dq_mfma_smem",
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def csa_pool_bwd_dq_mfma_kernel(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,  # [B, H, Sq, D] fp32
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        pool_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), POOL)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TOPK)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DQ)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=True)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, elem_type, shape=(LDS_TOTAL,)).get()
        C_DS = arith.index(LDS_GT)

        def _gep(base_p, elem_idx, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(_llvm_ptr_ty(), base_p, [idx_i64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v(base_p, elem_idx, n):
            return _llvm.LoadOp(T.vec(n, f16_ty), _gep(base_p, elem_idx, f16_ty)).result

        def load_i32_scalar(base_p, elem_idx):
            return _llvm.LoadOp(T.i32, _gep(base_p, elem_idx, T.i32)).result

        def store_f32(val, base_p, elem_idx):
            _llvm.StoreOp(val, _gep(base_p, elem_idx, f32_ty))

        def lds_store1(val, idx):
            vector.store(vector.from_elements(T.vec(1, f16_ty), [val]), lds, [idx])

        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        pid_hblk = arith.index_cast(T.index, gpu.block_idx.y)
        bid = arith.index_cast(T.index, gpu.block_idx.z)
        lane = arith.index_cast(T.index, gpu.thread_idx.x)

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)
        pool_size_v = arith.index_cast(T.index, pool_size)
        pool_size_i32 = pool_size
        if const_expr(hasattr(pool_size_i32, "ir_value")):
            pool_size_i32 = pool_size_i32.ir_value()
        K_topk_i32 = K_topk
        if const_expr(hasattr(K_topk_i32, "ir_value")):
            K_topk_i32 = K_topk_i32.ir_value()

        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)
        lane_mod_16_i32 = arith.index_cast(T.i32, lane_mod_16)
        lane_div_16_i32 = arith.index_cast(T.i32, lane_div_16)
        h_base = pid_hblk * arith.index(BLOCK_H)

        c_neg_inf = arith.constant(-1.0e30, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        c_log2e_f = arith.constant(_LOG2E_BWD, type=f32_ty)
        zero_mfma_pack = arith.constant_vector(0.0, T.vec(8, f16_ty))
        c_zero_mfma_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))

        # Q / DOUT B-operand packs (register-resident): lane L -> head = L%16, d chunk.
        my_head = h_base + lane_mod_16
        head_ib = arith.cmpi(arith.CmpIPredicate.slt, my_head, arith.index(NUM_HEADS))
        head_safe = arith.select(head_ib, my_head, arith.index(0))
        row_base = ((bid * arith.index(NUM_HEADS) + head_safe) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
        valid_hq = arith.AndIOp(head_ib, q_active).result
        q_packs = []
        do_packs = []
        for ck in range_constexpr(K_CHUNKS):
            off = row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
            qp = load_f16_v(q_ptr, off, 8)
            dp = load_f16_v(do_ptr, off, 8)
            q_packs.append(arith.select(valid_hq, qp, zero_mfma_pack))
            do_packs.append(arith.select(valid_hq, dp, zero_mfma_pack))

        # lse / delta for this lane's head.
        lse_off = (bid * arith.index(NUM_HEADS) + head_safe) * seq_len_v + pid_m_safe
        lse_off_i32 = arith.index_cast(T.i32, lse_off)
        lse_h = buffer_ops.buffer_load(lse_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)
        delta_h = buffer_ops.buffer_load(delta_rsrc, lse_off_i32, vec_width=1, dtype=f32_ty)

        def key_meta(kk_i32):
            ko = arith.cmpi(arith.CmpIPredicate.sge, kk_i32, K_topk_i32)
            kk_idx = arith.index_cast(T.index, arith.select(ko, arith.constant(0, type=T.i32), kk_i32))
            toff = (bid * seq_len_v + pid_m_safe) * K_topk_v + kk_idx
            idr = load_i32_scalar(topk_ptr, toff)
            ineg = arith.cmpi(arith.CmpIPredicate.slt, idr, arith.constant(0, type=T.i32))
            igp = arith.cmpi(arith.CmpIPredicate.sge, idr, pool_size_i32)
            inv = arith.OrIOp(arith.OrIOp(ko, ineg).result, igp).result
            idx_safe = arith.index_cast(T.index, arith.select(inv, arith.constant(0, type=T.i32), idr))
            prow = (bid * pool_size_v + idx_safe) * arith.index(HEAD_DIM)
            return inv, prow

        acc = [c_zero_mfma_acc for _ in range_constexpr(D_TILES)]

        for k_start, carry, results in scf.for_(
            arith.index(0), K_topk_v, arith.index(BLOCK_K), iter_args=acc,
        ):
            acc = [carry[d] for d in range_constexpr(D_TILES)]
            k_start_i32 = arith.index_cast(T.i32, k_start)
            gpu.barrier()

            for s in range_constexpr(N_SUB):
                key_pos_i32 = arith.AddIOp(
                    arith.AddIOp(k_start_i32, arith.constant(s * 16, type=T.i32)).result,
                    lane_mod_16_i32).result
                _inv_a, pool_row_base = key_meta(key_pos_i32)
                kloc_base = arith.index(s * 16) + lane_mod_16
                qk_acc = c_zero_mfma_acc
                dp_acc = c_zero_mfma_acc
                for ck in range_constexpr(K_CHUNKS):
                    poff = pool_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    a_pack = load_f16_v(pool_ptr, poff, 8)
                    qk_acc = mfma_f32_16x16x32(a_pack, q_packs[ck], qk_acc, dtype_str)
                    dp_acc = mfma_f32_16x16x32(a_pack, do_packs[ck], dp_acc, dtype_str)
                    # Scatter gathered^T for the AV step (reuse the QK A-operand).
                    d_base = arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    for e in range_constexpr(8):
                        gv = vector.extract(a_pack, static_position=[e], dynamic_position=[])
                        lds_idx = (d_base + arith.index(e)) * arith.index(BLOCK_K) + kloc_base
                        lds_store1(gv, lds_idx)
                # C-layout: lane L holds [key = g*4+r, head = L%16].
                for r in range_constexpr(4):
                    krow_i32 = arith.AddIOp(
                        arith.AddIOp(k_start_i32, arith.constant(s * 16, type=T.i32)).result,
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    inv_r, _pr = key_meta(krow_i32)
                    qk_raw = vector.extract(qk_acc, static_position=[r], dynamic_position=[])
                    dp_raw = vector.extract(dp_acc, static_position=[r], dynamic_position=[])
                    qk_s = arith.MulFOp(qk_raw, c_sm_scale_f, fastmath=fm_fast).result
                    qk_s = arith.select(inv_r, c_neg_inf, qk_s)
                    p = arith.ArithValue(arith.MulFOp(arith.SubFOp(qk_s, lse_h, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)
                    ds = arith.MulFOp(p, arith.SubFOp(dp_raw, delta_h, fastmath=fm_fast).result, fastmath=fm_fast).result
                    ds_scaled = arith.MulFOp(ds, c_sm_scale_f, fastmath=fm_fast).result
                    ds_scaled = arith.select(inv_r, c_zero_f, ds_scaled)
                    # Stage ds_scaled^T: [head = L%16, kloc = s*16 + g*4 + r].
                    kloc = arith.AddIOp(
                        arith.constant(s * 16, type=T.i32),
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    kloc_idx = arith.index_cast(T.index, kloc)
                    dst = C_DS + lane_mod_16 * arith.index(BLOCK_K) + kloc_idx
                    lds_store1(arith.trunc_f(f16_ty, ds_scaled), dst)

            gpu.barrier()

            # AV MFMA: dq[d, head] += gathered^T[d, kloc] @ ds_scaled[kloc, head].
            g_koff = lane_div_16 * arith.index(8)
            b_off = C_DS + lane_mod_16 * arith.index(BLOCK_K) + g_koff
            b_pack = vector.load_op(T.vec(8, f16_ty), lds, [b_off])
            new_acc = []
            for dt in range_constexpr(D_TILES):
                a_d = arith.index(dt * 16) + lane_mod_16
                a_idx = a_d * arith.index(BLOCK_K) + g_koff
                a_pack = vector.load_op(T.vec(8, f16_ty), lds, [a_idx])
                new_acc.append(mfma_f32_16x16x32(a_pack, b_pack, acc[dt], dtype_str))
            acc = new_acc

            yield acc

        acc = [results[d] for d in range_constexpr(D_TILES)]

        # Store dq fp32: acc[dt][r] = dq[d = dt*16 + g*4 + r, head = L%16].
        _if = scf.IfOp(valid_hq, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            dq_row_base = ((bid * arith.index(NUM_HEADS) + my_head) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            for dt in range_constexpr(D_TILES):
                for r in range_constexpr(4):
                    dv = vector.extract(acc[dt], static_position=[r], dynamic_position=[])
                    d = arith.index(dt * 16) + lane_div_16 * arith.index(4) + arith.index(r)
                    store_f32(dv, dq_ptr, dq_row_base + d)
            scf.YieldOp([])

    @flyc.jit
    def launch_csa_pool_bwd_dq_mfma(
        Q: fx.Tensor,
        POOL: fx.Tensor,
        TOPK: fx.Tensor,
        DOUT: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        pool_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        n_hblk = (NUM_HEADS + BLOCK_H - 1) // BLOCK_H
        launcher = csa_pool_bwd_dq_mfma_kernel(
            Q, POOL, TOPK, DOUT, LSE, DELTA, DQ, seq_len, K_topk, pool_size,
        )
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
        launcher.launch(grid=(sl_idx, arith.index(n_hblk), bs_idx), block=(BLOCK_SIZE, 1, 1), stream=stream)

    compile_hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_csa_pool_bwd_dq_mfma(*args, **kwargs)

    return _launch
