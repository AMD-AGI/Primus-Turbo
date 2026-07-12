"""DeepSeek-V4 sparse-MLA attention backward (flydsl, gfx950/MI355X).
All backward kernels (delta / dq / interm / gather / fused) + host dispatch, single file.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from primus_turbo.flydsl.utils.gemm_helper import make_bf16_rebased_rsrc


# ---- shared physical constants ----
_LOG2E = 1.4426950408889634
D = 512      # value head_dim
DQK = 576    # QK head_dim (incl. rope)
D_V = D      # alias used by gather / interm sections


def _attach(launch):
    """Attach the standard .compile() wrapper and return the launch closure."""
    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def _c8(a, b):
    """Concat two v4 int16 into a direct v8 bf16 (plain register concat, no shuffle
    crossbar; 16x16x32 MFMA operands must be direct-v8)."""
    va = Vec(a); vb = Vec(b)
    return _raw(Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], fx.Int16).bitcast(fx.BFloat16))

# ============================================================================
# kernel: delta / dsink
# ============================================================================

DELTA_THREADS = 64
DELTA_EPL = D // DELTA_THREADS  # 8 elems per lane
DSINK_THREADS = 256


def build_delta():
    elem = fx.BFloat16
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_delta_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + DELTA_THREADS * 4  # 64 fp32 partials

    @flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
    def k_fn(O: fx.Tensor, DO: fx.Tensor, DELTA: fx.Tensor, NROWS: fx.Int32):
        v8 = Vec.make_type(8, elem)
        lds = SmemPtr(allocator.get_base(), lds_off, fx.Float32.ir_type, shape=(DELTA_THREADS,)).get()
        row = fx.Index(gpu.block_idx.x)
        lane = fx.Index(gpu.thread_idx.x)
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(fx.Index(NROWS) * fx.Index(D * 2)))
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(NROWS) * fx.Index(D * 2)))
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(NROWS) * fx.Index(4)))

        base = row * fx.Index(D) + lane * fx.Index(DELTA_EPL)
        ov = buffer_ops.buffer_load(o_rsrc, base, vec_width=8, dtype=elem)
        dov = buffer_ops.buffer_load(do_rsrc, base, vec_width=8, dtype=elem)

        partial = fx.Float32(0.0)
        for i in range_constexpr(DELTA_EPL):
            oi = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(ov)[i])).result)
            di = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(dov)[i])).result)
            partial = fx.Float32(arith.AddFOp(_raw(partial), _raw(oi * di)).result)

        Vec.from_elements([partial], fx.Float32).store(lds, [lane])
        gpu.barrier()
        # full-wave reduction by lane 0 (delta kernel is tiny; serial 64-sum is fine).
        total = fx.Float32(0.0)
        for j in range_constexpr(DELTA_THREADS):
            total = fx.Float32(arith.AddFOp(
                _raw(total), _raw(Vec.load(Vec.make_type(1, fx.Float32), lds, [fx.Index(j)])[0])).result)
        buffer_ops.buffer_store(
            total, delta_rsrc, row * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.eq, _raw(lane), _raw(fx.Index(0))).result),
            offset_is_bytes=True)

    @flyc.jit
    def launch(O, DO, DELTA, NROWS, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(O, DO, DELTA, NROWS).launch(grid=(fx.Index(NROWS), 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream)

    return _attach(launch)


DSINK_TB = 64  # tokens per pass-1 WG (tb*H must be a multiple of DSINK_THREADS)


def build_dsink_split(T_LEN, H, tb=DSINK_TB):
    """Pass 1 (coalesced) of split d_sink: WG b owns a tb-token slice, reads full rows
    contiguously, each thread accumulates head (tid % H) across its rows, combined via
    LDS into partial[b, h]. OOB tail reads return 0; fp32 accumulate."""
    assert DSINK_THREADS % H == 0 and (tb * H) % DSINK_THREADS == 0
    TPH = DSINK_THREADS // H
    STEPS = (tb * H) // DSINK_THREADS
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dsink_p1_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + DSINK_THREADS * 4

    @flyc.kernel(known_block_size=[DSINK_THREADS, 1, 1])
    def k_fn(LSE: fx.Tensor, DELTA: fx.Tensor, SINK: fx.Tensor, PARTIAL: fx.Tensor,
             T: fx.Int32, NBLK: fx.Int32):
        lds = SmemPtr(allocator.get_base(), lds_off, fx.Float32.ir_type, shape=(DSINK_THREADS,)).get()
        b = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        Hn = fx.Index(H)
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        sink_rsrc = buffer_ops.create_buffer_resource(
            SINK, max_size=False, num_records_bytes=_raw(Hn * fx.Index(4)))
        part_rsrc = buffer_ops.create_buffer_resource(
            PARTIAL, max_size=False, num_records_bytes=_raw(fx.Index(NBLK) * Hn * fx.Index(4)))
        c_log2e = fx.Float32(_LOG2E)
        head = tid % Hn
        sink_h = fx.Float32(buffer_ops.buffer_load(sink_rsrc, head, vec_width=1, dtype=fx.Float32))

        base = b * fx.Index(tb) * Hn + tid  # first (contiguous) element this thread reads
        acc = fx.Float32(0.0)
        for s in range_constexpr(STEPS):
            g = base + fx.Index(s * DSINK_THREADS)
            lse_g = fx.Float32(buffer_ops.buffer_load(lse_rsrc, g, vec_width=1, dtype=fx.Float32))
            delta_g = fx.Float32(buffer_ops.buffer_load(delta_rsrc, g, vec_width=1, dtype=fx.Float32))
            e = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sink_h - lse_g) * c_log2e)))
            acc = fx.Float32(arith.AddFOp(_raw(acc), _raw(e * delta_g)).result)

        Vec.from_elements([acc], fx.Float32).store(lds, [tid])
        gpu.barrier()
        # combine the TPH threads sharing head=tid (tid<H): lds[tid + k*H], k in 0..TPH-1
        psum = fx.Float32(0.0)
        for k in range_constexpr(TPH):
            psum = fx.Float32(arith.AddFOp(
                _raw(psum),
                _raw(Vec.load(Vec.make_type(1, fx.Float32), lds, [tid + fx.Index(k * H)])[0])).result)
        buffer_ops.buffer_store(
            psum, part_rsrc, (b * Hn + tid) * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.slt, _raw(tid), _raw(Hn)).result),
            offset_is_bytes=True)

    @flyc.jit
    def launch(LSE, DELTA, SINK, PARTIAL, T, NBLK, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(LSE, DELTA, SINK, PARTIAL, T, NBLK).launch(
            grid=(fx.Index(NBLK), 1, 1), block=(DSINK_THREADS, 1, 1), stream=stream)

    return _attach(launch)


def build_dsink_reduce(NBLK, H):
    """Pass 2 of the coalesced split d_sink: d_sink[h] = -sum_b partial[b, h]. One WG of
    DSINK_THREADS; thread tid (< H) sums the NBLK block-partials of head tid, then negates."""
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dsink_p2_smem")

    @flyc.kernel(known_block_size=[DSINK_THREADS, 1, 1])
    def k_fn(PARTIAL: fx.Tensor, DSINK: fx.Tensor):
        tid = fx.Index(gpu.thread_idx.x)
        Hn = fx.Index(H)
        part_rsrc = buffer_ops.create_buffer_resource(
            PARTIAL, max_size=False, num_records_bytes=_raw(fx.Index(NBLK) * Hn * fx.Index(4)))
        dsink_rsrc = buffer_ops.create_buffer_resource(
            DSINK, max_size=False, num_records_bytes=_raw(Hn * fx.Index(4)))
        total = fx.Float32(0.0)
        for b in range_constexpr(NBLK):
            v = fx.Float32(buffer_ops.buffer_load(
                part_rsrc, fx.Index(b * H) + tid, vec_width=1, dtype=fx.Float32))
            total = fx.Float32(arith.AddFOp(_raw(total), _raw(v)).result)
        neg = fx.Float32(0.0) - total
        buffer_ops.buffer_store(
            neg, dsink_rsrc, tid * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.slt, _raw(tid), _raw(Hn)).result),
            offset_is_bytes=True)

    @flyc.jit
    def launch(PARTIAL, DSINK, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(PARTIAL, DSINK).launch(grid=(1, 1, 1), block=(DSINK_THREADS, 1, 1), stream=stream)

    return _attach(launch)


# ============================================================================
# kernel: dq
# ============================================================================

BLOCK_H = 64
HPW = 16
WAVES = BLOCK_H // HPW  # 4
THREADS = WAVES * 64    # 256
TILE_K = 16
TKP = 16
KS = D // 32            # 16 QK MFMA K-steps
DT = D // 16            # 32 PV d-tiles


def build_bwd_dq(topk_len, scale, pf=6, num_heads=None, pvpf=8):
    elem = fx.BFloat16
    # D_LDS: KV LDS row pad. 528 (264 dword %32=8) avoids the QK natural-v8 read's 2-way
    # bank conflict.
    D_LDS = 528
    DTE = DT
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    LDS_ELEMS = TKP * D_LDS
    # KVBUF: KV-LDS double-buffer depth (2 overlaps next-tile LDS store with current-tile
    # LDS reads, no read->store WAR barrier).
    KVBUF = 2
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dq_smem")
    kv_off = allocator._align(allocator.ptr, 16)
    mask_off = allocator._align(kv_off + KVBUF * LDS_ELEMS * 2, 16)
    allocator.ptr = allocator._align(mask_off + KVBUF * TILE_K * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, KV: fx.Tensor, DO: fx.Tensor, TOPK: fx.Tensor,
             LSE: fx.Tensor, DELTA: fx.Tensor, DQ: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
             T: fx.Int32, H: fx.Int32, NKV: fx.Int32):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_kv = SmemPtr(allocator.get_base(), kv_off, elem.ir_type, shape=(KVBUF * LDS_ELEMS,)).get()
        lds_mask = SmemPtr(allocator.get_base(), mask_off, fx.Float32.ir_type, shape=(KVBUF * TILE_K,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)

        token = fx.Index(gpu.block_idx.x)
        hg = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(H)
        head_wave_base = hg * fx.Index(BLOCK_H) + wave * fx.Index(HPW)
        head_A = head_wave_base + lo

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)))
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)))
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4)))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)))
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)))
        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)  # scale folded into exp2 base (saves a softmax mul)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)

        # ---- Q and dO (B operands): head=head_A ----
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)

        def load_q(ks):
            return buffer_ops.buffer_load(q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)

        def load_do(ks):
            return buffer_ops.buffer_load(do_rsrc, do_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)

        # Q/dO register-resident (reused across tiles), amortizing the load.
        q_packs = [load_q(ks) for ks in range_constexpr(KS)]
        do_packs = [load_do(ks) for ks in range_constexpr(KS)]

        lse_h = fx.Float32(buffer_ops.buffer_load(lse_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        # Clamp lse to finite: a fully-degenerate head (all kv invalid) has lse=-inf, giving
        # NaN P; clamping makes -inf-(-3e38)=-inf -> P=0 (matches triton's P mask).
        lse_h = fx.Float32(arith.MaxNumFOp(_raw(lse_h), _raw(fx.Float32(-3.0e38))).result)
        lse_l2 = fx.Float32(lse_h * c_log2e)  # loop-invariant: hoist lse*log2e out of tile loop
        delta_h = fx.Float32(buffer_ops.buffer_load(delta_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))

        g_row = tid // fx.Index(16)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def load_topk(tbase):
            return fx.Int32(buffer_ops.buffer_load(tk_rsrc, tk_row + tbase + g_row, vec_width=1, dtype=fx.Int32))

        def gather_load(idx):
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [buffer_ops.buffer_load(kv_rsrc, src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                                           vec_width=8, dtype=elem) for c in range_constexpr(4)]

        def gather_store(vvs, idx, buf_off, mbuf_off):
            valid = ArithValue(idx >= fx.Int32(0))
            for c in range_constexpr(4):
                Vec(vvs[c]).store(lds_kv, [buf_off + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)])
            if g_within == fx.Index(0):
                m = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(lds_mask, [mbuf_off + g_row])

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DTE)]
        idxA0 = load_topk(fx.Index(0))
        kv0 = gather_load(idxA0)
        idxB0 = load_topk(fx.Index(TILE_K))
        init = list(dq_acc0) + list(kv0) + [idxA0, idxB0]

        loop_results = init
        for t, iter_args in range(fx.Index(0), fx.Index(NUM_TILES), fx.Index(1), init=init):
            dq_acc = [iter_args[dt] for dt in range_constexpr(DTE)]
            kv_cur = [iter_args[DTE + c] for c in range_constexpr(4)]
            idxA = iter_args[DTE + 4]
            idxB = iter_args[DTE + 5]

            buf_off = (t % fx.Index(KVBUF)) * fx.Index(LDS_ELEMS)
            mbuf_off = (t % fx.Index(KVBUF)) * fx.Index(TILE_K)
            gather_store(kv_cur, idxA, buf_off, mbuf_off)
            # Issue next-tile HBM prefetches (KV data + tile+2 topk index) BEFORE the barrier:
            # they are vmcnt loads independent of the LDS store/barrier, so their issue overlaps
            # the barrier's lgkmcnt drain + s_barrier wait (fills the occ-1 sync bubble).
            kv_next = gather_load(idxB)
            idxB2 = load_topk((t + fx.Index(2)) * fx.Index(TILE_K))
            gpu.barrier()

            mask4 = Vec.load(v4f, lds_mask, [mbuf_off + grp * fx.Index(4)])

            # QK x2: S (B=Q) and dP (B=dO), sharing the gathered KV bv[ks]. occ-1 (single WG):
            # split each depth-16 MFMA chain into 2 accumulators so the RAW MFMA latency is
            # hidden by ILP (dQ has no neighbour WG to hide the chain).
            acc_s0 = Vec.filled(4, 0.0, fx.Float32)
            acc_s1 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp0 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp1 = Vec.filled(4, 0.0, fx.Float32)
            # QK4: 4 acc/operand (8 chains) vs 2 -> deeper ILP hides QK MFMA RAW latency using
            # the VGPR headroom. Gated to pro (num_heads>64) or large-topk (topk>=512); short-R
            # flash sees no benefit (fewer QK mfmas).
            _QK4 = 1 if ((num_heads is not None and num_heads > 64) or topk_len >= 512) else 0
            acc_s2 = Vec.filled(4, 0.0, fx.Float32)
            acc_s3 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp2 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp3 = Vec.filled(4, 0.0, fx.Float32)

            def _bv(ks):
                return Vec.load(v8, lds_kv, [buf_off + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)])
            def _q(ks):
                return q_packs[ks]
            def _do(ks):
                return do_packs[ks]
            PF = pf  # KV-read prefetch depth (build param, tuned for large-topk pro).
            bvq = [_bv(k) for k in range_constexpr(PF)]
            for ks in range_constexpr(KS):
                if ks + PF < KS:
                    bvq.append(_bv(ks + PF))
                if const_expr(_QK4):
                    r = ks % 4
                    accs = [acc_s0, acc_s1, acc_s2, acc_s3]
                    accd = [acc_dp0, acc_dp1, acc_dp2, acc_dp3]
                    accs[r] = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), accs[r]])
                    accd[r] = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), accd[r]])
                    acc_s0, acc_s1, acc_s2, acc_s3 = accs
                    acc_dp0, acc_dp1, acc_dp2, acc_dp3 = accd
                elif ks % 2 == 0:
                    acc_s0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), acc_s0])
                    acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), acc_dp0])
                else:
                    acc_s1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), acc_s1])
                    acc_dp1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), acc_dp1])
            if const_expr(_QK4):
                acc_s = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i]))
                     + fx.Float32(_raw(Vec(acc_s2)[i])) + fx.Float32(_raw(Vec(acc_s3)[i])) for i in range_constexpr(4)], fx.Float32)
                acc_dp = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i]))
                     + fx.Float32(_raw(Vec(acc_dp2)[i])) + fx.Float32(_raw(Vec(acc_dp3)[i])) for i in range_constexpr(4)], fx.Float32)
            else:
                acc_s = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i])) for i in range_constexpr(4)], fx.Float32)
                acc_dp = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i])) for i in range_constexpr(4)], fx.Float32)

            # P = exp2(acc_s*c_sl + mask4 - lse_l2) (scale*log2e folded into c_sl, mask4 is
            # {0,-inf}). PV A-operand (V ds_read_tr16) address depends only on buf_off+lane, so
            # the first PVPF tr16 reads are hoisted ahead of the exp2 loop (overlap softmax VALU).
            lo_d4 = lo // fx.Index(4)
            lo_m4 = lo % fx.Index(4)
            pv_base = fx.Int64(
                (buf_off + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4))
                * fx.Index(2) + fx.Index(kv_off))

            def _bvv(dt):
                ptr = buffer_ops.create_llvm_ptr(_raw(pv_base + fx.Int64(dt * 32)), address_space=3)
                return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))
            _PVPF = pvpf if (pvpf and ((num_heads is not None and num_heads > 64) or topk_len >= 512)) else 0
            bvv_pf = [_bvv(dt) for dt in range_constexpr(_PVPF)] if const_expr(_PVPF) else None

            # dS = P*(dP - delta)*scale; when masked P=0 -> dS=0 (no explicit dP mask).
            pvals = [None] * 4
            dsvals = [None] * 4
            for i in range_constexpr(4):
                arg = fx.Float32(_raw(Vec(acc_s)[i])) * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                pvals[i] = p
                _dp = fx.Float32(_raw(Vec(acc_dp)[i]))
                dsvals[i] = p * (_dp - delta_h) * c_scale

            # dS as PV B-operand (k=kv=grp*4+i, n=head=lo).
            pB = _raw(Vec.from_elements([fx.BFloat16(_raw(dsvals[i])) for i in range_constexpr(4)], elem).bitcast(fx.Int16))

            # dQ += dS @ K (PV): B=dS, A=V(ds_read_tr16), accumulate into dq_acc[dt] over tiles.
            # dq_acc pinned in AGPR via inline-asm MFMA (=a,v,v,0, D=C in-place, no accvgpr
            # shuffle) -> frees 128 arch-VGPR -> occ-2.
            _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())
            def _mma_ag(a, b, c):
                op = _llvm.InlineAsmOp(res=_v4f32_ir, operands_=[_raw(a), _raw(b), _raw(c)],
                                       asm_string="v_mfma_f32_16x16x16_bf16 $0, $1, $2, $0",
                                       constraints="=a,v,v,0", has_side_effects=False)
                return op.result
            def _pv(bvv, c):
                return _mma_ag(bvv, pB, c)
            new_dq = [None] * DT
            if const_expr(_PVPF):
                for dt in range_constexpr(DT):
                    if dt + _PVPF < DT:
                        bvv_pf.append(_bvv(dt + _PVPF))
                    new_dq[dt] = _pv(bvv_pf[dt], dq_acc[dt])
            else:
                for dt in range_constexpr(DT):
                    new_dq[dt] = _pv(_bvv(dt), dq_acc[dt])

            # store dS, P to [T,H,topk] bf16 (rank = tile_start + grp*4 + i, 4 contiguous).
            ts = t * fx.Index(TILE_K)
            sp_base = ((token * Hn + head_A) * fx.Index(topk_len) + ts + grp * fx.Index(4)) * fx.Index(2)
            # pack the two adjacent bf16-pk dwords into one dwordx2 store (halves the dS/P
            # store count; rank = grp*4+i are 4 contiguous, so [pk0,pk1] cover 4 ranks).
            ds_pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[0]), _raw(dsvals[1]))))
            ds_pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[2]), _raw(dsvals[3]))))
            pp_pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[0]), _raw(pvals[1]))))
            pp_pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[2]), _raw(pvals[3]))))
            ds_v2 = Vec.from_elements([ds_pk0, ds_pk1], fx.Int32)
            pp_v2 = Vec.from_elements([pp_pk0, pp_pk1], fx.Int32)
            buffer_ops.buffer_store(_raw(ds_v2), ds_rsrc, sp_base, offset_is_bytes=True)
            buffer_ops.buffer_store(_raw(pp_v2), pp_rsrc, sp_base, offset_is_bytes=True)

            loop_results = yield (list(new_dq) + list(kv_next) + [idxB, idxB2])

        dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
        head_i = head_wave_base + lo
        # dQ[token, head_i, 0:512] bf16.
        for dt in range_constexpr(DT):
            ov = Vec(dq_acc[dt])
            base = (token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)) * fx.Index(2)
            pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))))
            pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))))
            buffer_ops.buffer_store(_raw(Vec.from_elements([pk0, pk1], fx.Int32)), dq_rsrc, base, offset_is_bytes=True)
        # Zero the 64 rope cols (512..575) in-kernel (rope grad is dead), replacing the
        # strided host dq[..., 512:].zero_(). 4 d-tiles x (grp*4+[0..3]) cover 512..575.
        zero_v2 = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        for rt in range_constexpr(4):
            rbase = (token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)) * fx.Index(2)
            buffer_ops.buffer_store(_raw(zero_v2), dq_rsrc, rbase, offset_is_bytes=True)

    @flyc.jit
    def launch(Q: fx.Tensor, KV: fx.Tensor, DO: fx.Tensor, TOPK: fx.Tensor,
               LSE: fx.Tensor, DELTA: fx.Tensor, DQ: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
               T: fx.Int32, H: fx.Int32, NKV: fx.Int32, stream: fx.Stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        gy = fx.Index(H) // fx.Index(BLOCK_H)
        k_fn(Q, KV, DO, TOPK, LSE, DELTA, DQ, DS, PP, T, H, NKV).launch(
            grid=(fx.Index(T), gy, 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


def build_bwd_dq_pvk32(topk_len, scale, num_heads=None, pf=6, pvpf=8):
    elem = fx.BFloat16
    D_LDS = 528
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    LDS_ELEMS = TILE_K * D_LDS
    KVBUF = 2
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dq_pvk32_smem")
    kv_off = allocator._align(allocator.ptr, 16)
    mask_off = allocator._align(kv_off + KVBUF * LDS_ELEMS * 2, 16)
    allocator.ptr = allocator._align(mask_off + KVBUF * TILE_K * 4, 16)
    _QK4 = 1 if ((num_heads is not None and num_heads > 64) or topk_len >= 512) else 0

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, KV: fx.Tensor, DO: fx.Tensor, TOPK: fx.Tensor,
             LSE: fx.Tensor, DELTA: fx.Tensor, DQ: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
             T: fx.Int32, H: fx.Int32, NKV: fx.Int32):
        v8 = Vec.make_type(8, elem); v4 = Vec.make_type(4, elem); v4f = Vec.make_type(4, fx.Float32)
        i16 = fx.Int16
        lds_kv = SmemPtr(allocator.get_base(), kv_off, elem.ir_type, shape=(KVBUF * LDS_ELEMS,)).get()
        lds_mask = SmemPtr(allocator.get_base(), mask_off, fx.Float32.ir_type, shape=(KVBUF * TILE_K,)).get()
        tid = fx.Index(gpu.thread_idx.x); lane = tid % fx.Index(64); wave = tid // fx.Index(64)
        lo = lane % fx.Index(16); grp = lane // fx.Index(16)
        token = fx.Index(gpu.block_idx.x); hg = fx.Index(gpu.block_idx.y); Hn = fx.Index(H)
        head_wave_base = hg * fx.Index(BLOCK_H) + wave * fx.Index(HPW); head_A = head_wave_base + lo
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        kv_rsrc = buffer_ops.create_buffer_resource(KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)))
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)))
        tk_rsrc = buffer_ops.create_buffer_resource(TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4)))
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        delta_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        ds_rsrc = buffer_ops.create_buffer_resource(DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)))
        pp_rsrc = buffer_ops.create_buffer_resource(PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)))
        c_log2e = fx.Float32(_LOG2E); c_sl = fx.Float32(scale * _LOG2E); c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf")); c_zero = fx.Float32(0.0)
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        q_packs = [buffer_ops.buffer_load(q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem) for ks in range_constexpr(KS)]
        do_packs = [buffer_ops.buffer_load(do_rsrc, do_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem) for ks in range_constexpr(KS)]
        lse_h = fx.Float32(buffer_ops.buffer_load(lse_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        lse_h = fx.Float32(arith.MaxNumFOp(_raw(lse_h), _raw(fx.Float32(-3.0e38))).result)
        lse_l2 = fx.Float32(lse_h * c_log2e)
        delta_h = fx.Float32(buffer_ops.buffer_load(delta_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        g_row = tid // fx.Index(16); g_within = tid % fx.Index(16); tk_row = token * fx.Index(topk_len)

        def load_topk(tb): return fx.Int32(buffer_ops.buffer_load(tk_rsrc, tk_row + tb + g_row, vec_width=1, dtype=fx.Int32))
        def gather_load(idx):
            valid = ArithValue(idx >= fx.Int32(0)); src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [buffer_ops.buffer_load(kv_rsrc, src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8), vec_width=8, dtype=elem) for c in range_constexpr(4)]
        def gather_store(vvs, idx, bo, mo):
            valid = ArithValue(idx >= fx.Int32(0))
            for c in range_constexpr(4):
                Vec(vvs[c]).store(lds_kv, [bo + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)])
            if g_within == fx.Index(0):
                m = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(lds_mask, [mo + g_row])

        def _qkmma(a, b, c):
            return rocdl.mfma_f32_16x16x32_bf16(v4f, [a, b, c])
        def qk_softmax(buf_off, mbuf_off, t):
            # per-16 QK + softmax -> dsvals(4), pvals(4); store dS/P for this tile.
            mask4 = Vec.load(v4f, lds_mask, [mbuf_off + grp * fx.Index(4)])
            accs = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
            accd = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
            def _bv(ks): return Vec.load(v8, lds_kv, [buf_off + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)])
            bvq = [_bv(k) for k in range_constexpr(pf)]
            NR = 4 if _QK4 else 2
            for ks in range_constexpr(KS):
                if ks + pf < KS: bvq.append(_bv(ks + pf))
                r = ks % NR
                accs[r] = _qkmma(_raw(bvq[ks]), q_packs[ks], accs[r])
                accd[r] = _qkmma(_raw(bvq[ks]), do_packs[ks], accd[r])
            acc_s = [sum([fx.Float32(_raw(Vec(accs[j])[i])) for j in range_constexpr(NR)]) for i in range_constexpr(4)]
            acc_dp = [sum([fx.Float32(_raw(Vec(accd[j])[i])) for j in range_constexpr(NR)]) for i in range_constexpr(4)]
            pvals = [None] * 4; dsvals = [None] * 4
            for i in range_constexpr(4):
                arg = acc_s[i] * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                pvals[i] = p
                dsvals[i] = p * (acc_dp[i] - delta_h) * c_scale
            ts = t * fx.Index(TILE_K)
            sp = ((token * Hn + head_A) * fx.Index(topk_len) + ts + grp * fx.Index(4)) * fx.Index(2)
            ds2 = Vec.from_elements([fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[0]), _raw(dsvals[1])))),
                                     fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[2]), _raw(dsvals[3]))))], fx.Int32)
            pp2 = Vec.from_elements([fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[0]), _raw(pvals[1])))),
                                     fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[2]), _raw(pvals[3])))) ], fx.Int32)
            buffer_ops.buffer_store(_raw(ds2), ds_rsrc, sp, offset_is_bytes=True)
            buffer_ops.buffer_store(_raw(pp2), pp_rsrc, sp, offset_is_bytes=True)
            return dsvals

        lo_d4 = lo // fx.Index(4); lo_m4 = lo % fx.Index(4)
        def v32(ba, bb, dt):
            pa = buffer_ops.create_llvm_ptr(_raw(ba + fx.Int64(dt * 32)), address_space=3)
            pb = buffer_ops.create_llvm_ptr(_raw(bb + fx.Int64(dt * 32)), address_space=3)
            va = Vec(rocdl.ds_read_tr16_b64(v4, pa).result).bitcast(i16)
            vb = Vec(rocdl.ds_read_tr16_b64(v4, pb).result).bitcast(i16)
            return _raw(Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], i16).bitcast(elem))

        # inline-asm MFMA with D=C in AGPR ("=a,v,v,0"): pins the loop-carried dq_acc in the
        # separate AGPR file in-place (no accvgpr shuffle) -> frees ~128 arch-VGPR -> occ-2.
        _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())
        def _mma_agpr(a, b, c):
            op = _llvm.InlineAsmOp(res=_v4f32_ir, operands_=[_raw(a), _raw(b), _raw(c)],
                                   asm_string="v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0",
                                   constraints="=a,v,v,0", has_side_effects=False)
            return op.result

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DT)]
        idx0 = load_topk(fx.Index(0)); kva0 = gather_load(idx0)
        idx1 = load_topk(fx.Index(TILE_K)); kvb0 = gather_load(idx1)
        idx2 = load_topk(fx.Index(2 * TILE_K))
        init = list(dq_acc0) + list(kva0) + list(kvb0) + [idx0, idx1, idx2]
        NPAIR = NUM_TILES // 2
        loop_results = init
        for p, iter_args in range(fx.Index(0), fx.Index(NPAIR), fx.Index(1), init=init):
            dq_acc = [iter_args[dt] for dt in range_constexpr(DT)]
            kva = [iter_args[DT + c] for c in range_constexpr(4)]
            kvb = [iter_args[DT + 4 + c] for c in range_constexpr(4)]
            idxa = iter_args[DT + 8]; idxb = iter_args[DT + 9]; idxn = iter_args[DT + 10]
            ta = p * fx.Index(2); tb = ta + fx.Index(1)
            gather_store(kva, idxa, fx.Index(0), fx.Index(0))
            gather_store(kvb, idxb, fx.Index(LDS_ELEMS), fx.Index(TILE_K))
            # prefetch next pair's KV (tiles 2p+2, 2p+3)
            kva_n = gather_load(idxn)
            idxn2 = load_topk((ta + fx.Index(3)) * fx.Index(TILE_K))
            kvb_n = gather_load(idxn2)
            idxn3 = load_topk((ta + fx.Index(4)) * fx.Index(TILE_K))
            gpu.barrier()
            ds_a = qk_softmax(fx.Index(0), fx.Index(0), ta)            # per-16 QK/softmax tile a
            ds_b = qk_softmax(fx.Index(LDS_ELEMS), fx.Index(TILE_K), tb)  # per-16 tile b
            # PV K=32: dq_acc += [dS_a | dS_b] @ [V_a | V_b]
            pB = _raw(Vec.from_elements([fx.BFloat16(_raw(ds_a[i])) for i in range_constexpr(4)]
                                        + [fx.BFloat16(_raw(ds_b[i])) for i in range_constexpr(4)], elem))
            base_a = fx.Int64((fx.Index(0) + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)) * fx.Index(2) + fx.Index(kv_off))
            base_b = fx.Int64((fx.Index(LDS_ELEMS) + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)) * fx.Index(2) + fx.Index(kv_off))
            new_dq = [None] * DT
            trq = [v32(base_a, base_b, dt) for dt in range_constexpr(pvpf)]
            for dt in range_constexpr(DT):
                if dt + pvpf < DT: trq.append(v32(base_a, base_b, dt + pvpf))
                new_dq[dt] = _mma_agpr(trq[dt], pB, dq_acc[dt])
            # WAR barrier: this pair reuses buf0/buf1 next iter (KVBUF=2, per-pair alternation),
            # so the next pair's gather_store must wait for this pair's PV tr16 reads to drain
            # (else cross-wave race).
            gpu.barrier()
            loop_results = yield (list(new_dq) + list(kva_n) + list(kvb_n) + [idxn, idxn2, idxn3])

        dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
        head_i = head_wave_base + lo
        for dt in range_constexpr(DT):
            ov = Vec(dq_acc[dt])
            base = (token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)) * fx.Index(2)
            pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))))
            pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))))
            buffer_ops.buffer_store(_raw(Vec.from_elements([pk0, pk1], fx.Int32)), dq_rsrc, base, offset_is_bytes=True)
        zero_v2 = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        for rt in range_constexpr(4):
            rbase = (token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)) * fx.Index(2)
            buffer_ops.buffer_store(_raw(zero_v2), dq_rsrc, rbase, offset_is_bytes=True)

    @flyc.jit
    def launch(Q: fx.Tensor, KV: fx.Tensor, DO: fx.Tensor, TOPK: fx.Tensor,
               LSE: fx.Tensor, DELTA: fx.Tensor, DQ: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
               T: fx.Int32, H: fx.Int32, NKV: fx.Int32, stream: fx.Stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        gy = fx.Index(H) // fx.Index(BLOCK_H)
        k_fn(Q, KV, DO, TOPK, LSE, DELTA, DQ, DS, PP, T, H, NKV).launch(grid=(fx.Index(T), gy, 1), block=(THREADS, 1, 1), stream=stream)
    return _attach(launch)


# ============================================================================
# kernel: interm
# ============================================================================

THREADS = 256
WAVES = THREADS // 64          # 4
D_TILES = D_V // 16            # 32
DT_PER_WAVE = D_TILES // WAVES  # 8


# build_interm_rtr: REGISTER-transpose Q/dO (no Q/dO LDS) -> BD=512 at occ-2. Each WAVE owns
# DT_PER_WAVE=8 contiguous d-tiles and loops all rank-tiles, holding only its 128 d of Q/dO
# compact. A[m=d,k=h] is built once via ds_bpermute 16x16 transpose; dS/P stay LDS-staged.
def build_interm_rtr(topk_len, num_heads, BD=512):
    elem = fx.BFloat16
    R_CHUNK = topk_len
    KS16 = num_heads // 16          # h-blocks of 16 (flash 4, pro 8)
    # GSZ: rank-group size. Larger -> fewer RAW barriers. Must divide R_CHUNK exactly (else the
    # tail ranks drop). Gated to R%128==0 & R>=256; R128 keeps GSZ64 to retain the 2-group
    # prefetch overlap that hides the dS/P HBM load; un-divisible topk (e.g. 160) falls to 32.
    GSZ = (128 if (topk_len % 128 == 0 and topk_len >= 256) else (64 if topk_len % 64 == 0 else 32))
    RGROUPS = R_CHUNK // GSZ        # rank-groups (GSZ//16 rank-tiles each)
    RT_PER_G = GSZ // 16
    DBLK = D_V // BD                # d-blocks (grid.y); flash BD512->1, pro BD256->2
    DTB = BD // 16                  # d-tiles in a block
    DTW = DTB // WAVES              # d-tiles per wave (a_q/a_do register footprint driver)
    SROW = GSZ + 8                  # dS/P LDS row pad (tr16 bank-conflict avoid)
    DS_LDS = num_heads * SROW
    SDATA8 = (num_heads * GSZ) // 8
    SIT = SDATA8 // THREADS
    # DBUF: double-buffer dS/P LDS. DBUF=1 halves the dS/P LDS -> lifts the LDS-capped
    # occupancy -> hides the mfma/read latency. Short R (R128) prefers DBUF2.
    DBUF = 1 if topk_len >= 192 else 2
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_interm_rtr_smem")
    ds_off = allocator._align(allocator.ptr, 16)
    p_off = allocator._align(ds_off + DBUF * DS_LDS * 2, 16)
    allocator.ptr = allocator._align(p_off + DBUF * DS_LDS * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, DO: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
             INTERM: fx.Tensor, T: fx.Int32):
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        i16 = fx.Int16
        lds_ds = SmemPtr(allocator.get_base(), ds_off, elem.ir_type, shape=(DS_LDS,)).get()
        lds_p = SmemPtr(allocator.get_base(), p_off, elem.ir_type, shape=(DS_LDS,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        token = fx.Index(gpu.block_idx.x)
        dblk = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(num_heads)
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D_V * 2)))
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2)))
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2)))
        interm_rsrc = make_bf16_rebased_rsrc(
            INTERM, token * fx.Index(R_CHUNK) * fx.Index(D_V), fx.Index(R_CHUNK) * fx.Index(D_V * 2))

        q_tok = token * Hn * fx.Index(DQK)
        do_tok = token * Hn * fx.Index(D_V)
        ds_tok = token * Hn * fx.Index(R_CHUNK)
        wbase_d = dblk * fx.Index(BD) + wave * fx.Index(DTW * 16)   # this wave's d-block start

        # register-transpose compact -> A[m=d,k=h] (_tr4 = one 16-h-block -> 4 i16). VPERM:
        # extract bf16 elem lo_m4 from the 2 bpermuted dwords via ONE v_perm_b32 byte-permute
        # (vs shift+select). Gated to short-topk (VALU-bound); neutral on large R.
        _VPERM = 1 if topk_len <= 192 else 0
        def _tr4(cv):
            vi = Vec(Vec(cv).bitcast(fx.Int32))       # 2 x i32 (dw0,dw1)
            dw0 = _raw(vi[0]); dw1 = _raw(vi[1])
            sel = _raw(fx.Int32(lo_m4 * fx.Index(514) + fx.Index(256)))  # v_perm byte selector for elem lo_m4
            outs = []
            for i in range_constexpr(4):
                si = fx.Index(16) * grp + fx.Index(4 * i) + lo_d4
                idx = _raw(fx.Int32(si) * fx.Int32(4))
                s0 = rocdl.ds_bpermute(fx.Int32.ir_type, idx, dw0)
                s1 = rocdl.ds_bpermute(fx.Int32.ir_type, idx, dw1)
                if const_expr(_VPERM):
                    perm = _raw(rocdl.perm_b32(s1, s0, sel))
                    outs.append(fx.Int16(arith.TruncIOp(i16.ir_type, perm).result))
                else:
                    dwsel = ArithValue(lo_m4 < fx.Index(2)).select(s0, s1)
                    shift = ArithValue(lo_m4 % fx.Index(2) == fx.Index(0)).select(_raw(fx.Int32(0)), _raw(fx.Int32(16)))
                    shifted = _raw(fx.Int32(_raw(dwsel)) >> fx.Int32(_raw(shift)))
                    outs.append(fx.Int16(arith.TruncIOp(i16.ir_type, shifted).result))
            return outs                                # list of 4 i16

        def transpose(cv):
            return _raw(Vec.from_elements(_tr4(cv), i16))            # v4 (K=16 A operand)

        # v8 DIRECT (2 h-blocks -> 8 i16, no v4 intermediate) for K=32 MFMA: building v8 straight
        # from 8 scalars avoids holding both v4 and v8 -> fits occ-2.
        def transpose8(cv0, cv1):
            return _raw(Vec.from_elements(_tr4(cv0) + _tr4(cv1), i16).bitcast(elem))  # v8 bf16 (K=32 A operand)

        def load_tile(rsrc, tok, dstride, u, ks):
            h = fx.Index(ks * 16) + lane // fx.Index(4)
            d = wbase_d + fx.Index(u * 16) + (lane % fx.Index(4)) * fx.Index(4)
            return buffer_ops.buffer_load(rsrc, tok + h * dstride + d, vec_width=4, dtype=elem)

        def tr_ks(off, coltile, ks):
            row = (fx.Index(ks * 16) + grp * fx.Index(4) + lo_d4) * fx.Index(SROW) + coltile * fx.Index(16) + lo_m4 * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(i16))

        # ASM: issue all KS16 ds_read_b64_tr_b16 of a rank-tile as ONE inline-asm block so the
        # backend does not drain lgkmcnt per read and their LDS latencies overlap (caller drains
        # once before the mfma). Gated to R>=192; returns KS16 v2i32 (2 i32 = 4 bf16).
        _ASM = 1 if topk_len >= 192 else 0
        # ILP: split acc chains -> 4-way mfma ILP hides the mfma RAW latency. Gated to
        # 192<=R<=640 (elsewhere the extra acc VGPR drops occ where the chain isn't the bound).
        _ILP = 1 if 192 <= topk_len <= 640 else 0
        # s_setprio(1) around mfma: prioritizes mfma issue.
        _PRIO = 1
        # BSTACK (stacked-K): accumulate both GEMMs (Q@dS + dO@P, same [d,r]) into ONE acc chain
        # -> removes the 4 fp32 combine adds per output (interm is issue-bound). Gated to
        # topk<192; ILP owns 192<=R<=640 (mutually exclusive, _ILP takes precedence).
        _BSTACK = 1 if topk_len < 192 else 0
        v2i32_ty = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
        def _tr16_packed(off, coltile):
            base_row = (grp * fx.Index(4) + lo_d4) * fx.Index(SROW) + fx.Index(coltile * 16) + lo_m4 * fx.Index(4)
            bptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(base_row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            N = KS16
            struct_t = _llvm.StructType.get_literal([v2i32_ty] * N)
            lines = [f"ds_read_b64_tr_b16 ${k}, ${N} offset:{k * 16 * SROW * 2}" for k in range(N)]
            constraints = ",".join(["=&v"] * N + ["v"] + ["~{memory}"])
            op = _llvm.InlineAsmOp(res=struct_t, operands_=[_raw(bptr)],
                                   asm_string="\n".join(lines), constraints=constraints, has_side_effects=True)
            return [_llvm.extractvalue(v2i32_ty, op.result, [k]) for k in range(N)]
        def _wait_lgkm(n):
            _llvm.inline_asm(res=None, operands_=[], asm_string=f"s_waitcnt lgkmcnt({n})",
                             constraints="", has_side_effects=True)

        # dS/P HBM->register prefetch (depth 1): a barrier separates the LDS store from compute,
        # so without prefetch each rank-group fully exposes the dS/P HBM load. Load group g+1
        # into registers during g's compute.
        def hbm_load(g):
            regs = []
            for it in range_constexpr(SIT):
                v = tid + fx.Index(it * THREADS)
                flat = v * fx.Index(8)
                h = flat // fx.Index(GSZ)
                r = flat % fx.Index(GSZ)
                dsv = buffer_ops.buffer_load(ds_rsrc, ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * GSZ) + r, vec_width=8, dtype=elem)
                ppv = buffer_ops.buffer_load(pp_rsrc, ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * GSZ) + r, vec_width=8, dtype=elem)
                regs.append((dsv, ppv, h, r))
            return regs

        # Issue all HBM loads up front (Q/dO compact for the transpose + dS/P group 0) so the
        # bpermute transpose burst (no MFMA to hide it) overlaps the HBM latency.
        q_cmp = [[load_tile(q_rsrc, q_tok, fx.Index(DQK), u, ks) for ks in range_constexpr(KS16)]
                 for u in range_constexpr(DTW)]
        do_cmp = [[load_tile(do_rsrc, do_tok, fx.Index(D_V), u, ks) for ks in range_constexpr(KS16)]
                  for u in range_constexpr(DTW)]
        loads = [None] * RGROUPS
        loads[0] = hbm_load(0)
        # K=32 A operands (v8, register-resident, reused across all rank-tiles). transpose8
        # builds v8 directly -> only 128 VGPR held -> K=32 AGPR fits occ-2.
        aq8 = [[transpose8(q_cmp[u][2*k2], q_cmp[u][2*k2+1]) for k2 in range_constexpr(KS16 // 2)] for u in range_constexpr(DTW)]
        ado8 = [[transpose8(do_cmp[u][2*k2], do_cmp[u][2*k2+1]) for k2 in range_constexpr(KS16 // 2)] for u in range_constexpr(DTW)]
        for g in range_constexpr(RGROUPS):
            bsel = g % DBUF                       # which dS/P LDS buffer this group uses
            bs_e = fx.Index(bsel * DS_LDS)        # element offset into lds_ds/lds_p
            bs_b = bsel * DS_LDS * 2              # byte offset for tr_ks
            for (dsv, ppv, h, r) in loads[g]:
                Vec(dsv).store(lds_ds, [bs_e + h * fx.Index(SROW) + r])
                Vec(ppv).store(lds_p, [bs_e + h * fx.Index(SROW) + r])
            gpu.barrier()
            if g + 1 < RGROUPS:                  # prefetch next group during this compute
                loads[g + 1] = hbm_load(g + 1)

            def _asm_pair(rt):
                return (_tr16_packed(ds_off + bs_b, rt), _tr16_packed(p_off + bs_b, rt))
            def _d8(a, b):  # 2 v2i32 (4 bf16 each) -> direct v8 bf16 (no shuffle crossbar)
                va = Vec(Vec(a).bitcast(i16)); vb = Vec(Vec(b).bitcast(i16))
                return _raw(Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], i16).bitcast(elem))
            def _asm_v8(pair):
                ds_r, p_r = pair
                bds8 = [_d8(ds_r[2*k2], ds_r[2*k2+1]) for k2 in range_constexpr(KS16 // 2)]
                bp8 = [_d8(p_r[2*k2], p_r[2*k2+1]) for k2 in range_constexpr(KS16 // 2)]
                return bds8, bp8
            def _imma(a, b, c):
                return rocdl.mfma_f32_16x16x32_bf16(v4f, [a, b, c])
            for rt in range_constexpr(RT_PER_G):   # rank-tiles per group (GSZ//16)
                rank = fx.Index(g * GSZ + rt * 16) + lo
                if const_expr(_ASM):
                    # within-tile: packed async reads + ONE drain -> the tile's latencies overlap
                    cur = _asm_pair(rt)
                    _wait_lgkm(0)
                    bds8, bp8 = _asm_v8(cur)
                else:
                    b_ds = [tr_ks(ds_off + bs_b, fx.Index(rt), ks) for ks in range_constexpr(KS16)]
                    b_p = [tr_ks(p_off + bs_b, fx.Index(rt), ks) for ks in range_constexpr(KS16)]
                    bds8 = [_c8(b_ds[2*k2], b_ds[2*k2+1]) for k2 in range_constexpr(KS16 // 2)]
                    bp8 = [_c8(b_p[2*k2], b_p[2*k2+1]) for k2 in range_constexpr(KS16 // 2)]
                for u in range_constexpr(DTW):
                    if const_expr(_PRIO):
                        rocdl.s_setprio(1)
                    if const_expr(_ILP):
                        # split each acc chain into even/odd k2 -> 4 independent mfma chains (vs
                        # 2) so the mfma RAW latency is hidden by more in-wave ILP.
                        a0a = Vec.filled(4, 0.0, fx.Float32); a0b = Vec.filled(4, 0.0, fx.Float32)
                        a1a = Vec.filled(4, 0.0, fx.Float32); a1b = Vec.filled(4, 0.0, fx.Float32)
                        for k2 in range_constexpr(KS16 // 2):
                            if k2 % 2 == 0:
                                a0a = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq8[u][k2], bds8[k2], a0a])
                                a1a = rocdl.mfma_f32_16x16x32_bf16(v4f, [ado8[u][k2], bp8[k2], a1a])
                            else:
                                a0b = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq8[u][k2], bds8[k2], a0b])
                                a1b = rocdl.mfma_f32_16x16x32_bf16(v4f, [ado8[u][k2], bp8[k2], a1b])
                        ov = Vec.from_elements(
                            [fx.Float32(_raw(Vec(a0a)[i])) + fx.Float32(_raw(Vec(a0b)[i]))
                             + fx.Float32(_raw(Vec(a1a)[i])) + fx.Float32(_raw(Vec(a1b)[i])) for i in range_constexpr(4)], fx.Float32)
                    elif const_expr(_BSTACK):
                        # stacked-K: both mfmas feed ONE acc (interm[d,r] = Q@dS + dO@P) -> no
                        # combine adds. Trades dual-acc ILP for fewer instructions (issue-bound).
                        acc = Vec.filled(4, 0.0, fx.Float32)
                        for k2 in range_constexpr(KS16 // 2):
                            acc = _imma(aq8[u][k2], bds8[k2], acc)
                            acc = _imma(ado8[u][k2], bp8[k2], acc)
                        ov = Vec.from_elements(
                            [fx.Float32(_raw(Vec(acc)[i])) for i in range_constexpr(4)], fx.Float32)
                    else:
                        acc0 = Vec.filled(4, 0.0, fx.Float32)
                        acc1 = Vec.filled(4, 0.0, fx.Float32)
                        for k2 in range_constexpr(KS16 // 2):   # K=32 MFMA: half the MFMA count
                            acc0 = _imma(aq8[u][k2], bds8[k2], acc0)
                            acc1 = _imma(ado8[u][k2], bp8[k2], acc1)
                        ov = Vec.from_elements(
                            [fx.Float32(_raw(Vec(acc0)[i])) + fx.Float32(_raw(Vec(acc1)[i])) for i in range_constexpr(4)], fx.Float32)
                    if const_expr(_PRIO):
                        rocdl.s_setprio(0)
                    d = wbase_d + fx.Index(u * 16) + grp * fx.Index(4)   # global d
                    base = (rank * fx.Index(D_V) + d) * fx.Index(2)     # token folded into SRD base
                    bf4 = Vec.from_elements([fx.BFloat16(_raw(Vec(ov)[i])) for i in range_constexpr(4)], elem)
                    buffer_ops.buffer_store(_raw(bf4.bitcast(fx.Int32)), interm_rsrc, base, offset_is_bytes=True)
            # WAR barrier only when single-buffered (DBUF=1): the next group's store reuses the
            # same buffer, so wait for this group's dS/P reads to drain. DBUF=2 stores the other
            # buffer -> no WAR barrier needed.
            if DBUF == 1:
                gpu.barrier()

    @flyc.jit
    def launch(Q: fx.Tensor, DO: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
               INTERM: fx.Tensor, T: fx.Int32, stream: fx.Stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, DO, DS, PP, INTERM, T).launch(
            grid=(fx.Index(T), fx.Index(DBLK), 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


def build_interm_blk(topk_len, num_heads, BD, qpad=16, spad=8):
    elem = fx.BFloat16
    R_CHUNK = topk_len
    KS16 = num_heads // 16
    DBLK = D_V // BD                 # number of d-blocks (grid.y)
    DT_BLK = BD // 16               # d-tiles per block
    _BLKPRIO = 1                     # s_setprio(1) around blk mfma
    # BLKSTACK (stacked-K, blk twin of rtr BSTACK): accumulate Q@dS + dO@P into ONE acc chain
    # -> removes the 4 fp32 combine adds/output (blk is issue-bound). Gated to topk<=128 or
    # topk>=512 (excludes R160 local-band). Only flash uses blk (pro is rtr).
    _BLKSTACK = 1 if (topk_len <= 128 or topk_len >= 512) else 0
    RGROUPS = R_CHUNK // 64         # rank-groups (each = 4 rank-tiles = 4 waves)
    # tr16 bank-pad: pad the LDS row stride so consecutive tr16 rows land in distinct banks
    # (gfx950 = 64 banks). Q/dO row BD->BD+16, dS/P row 64->72. Occupancy-gated: pro lands at
    # 76KB (2WG/CU); a wider pad crosses 80KB -> 1WG and collapses.
    QROW = BD + qpad                # padded LDS row stride for Q/dO
    SROW = 64 + spad                # padded LDS row stride for dS/P
    Q_LDS = num_heads * QROW        # [h][BD] padded
    DS_LDS = num_heads * SROW       # [h][64] padded
    QDATA8 = (num_heads * BD) // 8  # coalesced vec8 count for Q/dO stage (actual data)
    SDATA8 = (num_heads * 64) // 8  # for dS/P stage (actual data)
    QIT = QDATA8 // THREADS         # stage iters (must divide)
    SIT = SDATA8 // THREADS

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_interm_blk_smem")
    q_off = allocator._align(allocator.ptr, 16)
    do_off = allocator._align(q_off + Q_LDS * 2, 16)
    ds_off = allocator._align(do_off + Q_LDS * 2, 16)
    p_off = allocator._align(ds_off + DS_LDS * 2, 16)
    allocator.ptr = allocator._align(p_off + DS_LDS * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, DO: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
             INTERM: fx.Tensor, T: fx.Int32):
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        v8 = Vec.make_type(8, elem)
        lds_q = SmemPtr(allocator.get_base(), q_off, elem.ir_type, shape=(Q_LDS,)).get()
        lds_do = SmemPtr(allocator.get_base(), do_off, elem.ir_type, shape=(Q_LDS,)).get()
        lds_ds = SmemPtr(allocator.get_base(), ds_off, elem.ir_type, shape=(DS_LDS,)).get()
        lds_p = SmemPtr(allocator.get_base(), p_off, elem.ir_type, shape=(DS_LDS,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        token = fx.Index(gpu.block_idx.x)
        dblk = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(num_heads)

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D_V * 2)))
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2)))
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2)))
        # interm can exceed 4GB (pro cr4 = 4.8GB) -> flat entry offset overflows the 32-bit
        # buffer voffset. Rebase the SRD to THIS token's slab (i64 base) so the in-slab
        # offset (rank*D_V+d < R_CHUNK*D_V) stays int32-safe.
        interm_rsrc = make_bf16_rebased_rsrc(
            INTERM, token * fx.Index(R_CHUNK) * fx.Index(D_V), fx.Index(R_CHUNK) * fx.Index(D_V * 2))

        q_tok = token * Hn * fx.Index(DQK)
        do_tok = token * Hn * fx.Index(D_V)
        ds_tok = token * Hn * fx.Index(R_CHUNK)
        d0 = dblk * fx.Index(BD)

        # Software-pipeline the dS/P staging: the per-group flow (load -> LDS store -> barrier ->
        # compute) fully exposes the HBM load each group. Prefetch the next group's dS/P into
        # registers during this group's compute so it overlaps the MFMAs (single LDS buffer).
        def hbm_load(g):
            regs = []
            for it in range_constexpr(SIT):
                v = tid + fx.Index(it * THREADS)
                flat = v * fx.Index(8)
                h = flat // fx.Index(64)
                r = flat % fx.Index(64)
                dsv = buffer_ops.buffer_load(ds_rsrc, ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * 64) + r, vec_width=8, dtype=elem)
                ppv = buffer_ops.buffer_load(pp_rsrc, ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * 64) + r, vec_width=8, dtype=elem)
                regs.append((dsv, ppv, h, r))
            return regs

        # Issue group 0's dS/P HBM load FIRST (before the Q/dO stage) so it overlaps the whole
        # Q/dO staging + its barrier. Depth 1: one load in flight (prefetched during the
        # previous group's compute) already hides the load; deeper prefetch only adds registers.
        PF_DEPTH = 1
        pending = [hbm_load(0)]

        # ---- stage Q/dO[H][BD] once (coalesced vec8, [h][d] natural).
        for it in range_constexpr(QIT):
            v = tid + fx.Index(it * THREADS)
            flat = v * fx.Index(8)
            h = flat // fx.Index(BD)
            d = flat % fx.Index(BD)
            qv = buffer_ops.buffer_load(q_rsrc, q_tok + h * fx.Index(DQK) + d0 + d, vec_width=8, dtype=elem)
            dov = buffer_ops.buffer_load(do_rsrc, do_tok + h * fx.Index(D_V) + d0 + d, vec_width=8, dtype=elem)
            Vec(qv).store(lds_q, [h * fx.Index(QROW) + d])
            Vec(dov).store(lds_do, [h * fx.Index(QROW) + d])
        gpu.barrier()

        def tr_ks(off, stride, coltile, ks):
            # h-block ks -> rows ks*16.., col tile (d-tile or rank-tile) -> coltile*16 within row.
            row = (fx.Index(ks * 16) + grp * fx.Index(4) + lo_d4) * fx.Index(stride) + coltile * fx.Index(16) + lo_m4 * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

        # A-operand tr16 reads are kept inline (re-read per group): the kernel is dS/P-re-read
        # bound, not A-LDS-read bound. Any build-time branch here must use a ternary, NOT an
        # in-kernel `if` (flydsl rewrites `if` to scf.if, scoping out assigned values).
        for g in range_constexpr(RGROUPS):
            rbase = fx.Index(g * 64)
            # store this group's (already HBM-loaded) dS/P into LDS.
            for (dsv, ppv, h, r) in pending[0]:
                Vec(dsv).store(lds_ds, [h * fx.Index(SROW) + r])
                Vec(ppv).store(lds_p, [h * fx.Index(SROW) + r])
            gpu.barrier()

            # keep PF_DEPTH loads in flight: prefetch group g+PF_DEPTH during this compute.
            gnext = g + PF_DEPTH
            if gnext < RGROUPS:
                pending.append(hbm_load(gnext))

            rank = rbase + wave * fx.Index(16) + lo     # this wave's rank-tile, row=lo
            # tr16-read batching + 2-acc: (1) HOIST the B operands (dS/P, index only wave,ks) out
            # of the dt loop -> read once, reuse across d-tiles. (2) PRE-ISSUE the A operands per
            # d-tile so all KS16 tr16 reads are in flight. (3) 2-acc split summed at store.
            b_ds = [tr_ks(ds_off, SROW, wave, ks) for ks in range_constexpr(KS16)]
            b_p = [tr_ks(p_off, SROW, wave, ks) for ks in range_constexpr(KS16)]
            # This kernel is LDS-read-latency bound at occ-1 (MFMAs stall on tr16 reads); more
            # occupancy, not more in-flight depth, is what would hide it.
            for dt in range_constexpr(DT_BLK):
                a_q = [tr_ks(q_off, QROW, fx.Index(dt), ks) for ks in range_constexpr(KS16)]
                a_do = [tr_ks(do_off, QROW, fx.Index(dt), ks) for ks in range_constexpr(KS16)]
                if const_expr(_BLKPRIO):
                    rocdl.s_setprio(1)
                if const_expr(_BLKSTACK):
                    acc0 = Vec.filled(4, 0.0, fx.Float32)
                    for ks in range_constexpr(KS16):
                        acc0 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_q[ks], b_ds[ks], acc0])
                        acc0 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_do[ks], b_p[ks], acc0])
                    acc1 = None
                else:
                    acc0 = Vec.filled(4, 0.0, fx.Float32)
                    acc1 = Vec.filled(4, 0.0, fx.Float32)
                    for ks in range_constexpr(KS16):
                        acc0 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_q[ks], b_ds[ks], acc0])
                        acc1 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_do[ks], b_p[ks], acc1])
                if const_expr(_BLKPRIO):
                    rocdl.s_setprio(0)
                if const_expr(_BLKSTACK):
                    ov = Vec.from_elements(
                        [fx.Float32(_raw(Vec(acc0)[i])) for i in range_constexpr(4)], fx.Float32)
                else:
                    ov = Vec.from_elements(
                        [fx.Float32(_raw(Vec(acc0)[i])) + fx.Float32(_raw(Vec(acc1)[i]))
                         for i in range_constexpr(4)], fx.Float32)
                d = d0 + fx.Index(dt * 16) + grp * fx.Index(4)
                base = (rank * fx.Index(D_V) + d) * fx.Index(2)   # token folded into SRD base
                bf4 = Vec.from_elements([fx.BFloat16(_raw(Vec(ov)[i])) for i in range_constexpr(4)], elem)
                buffer_ops.buffer_store(_raw(bf4.bitcast(fx.Int32)), interm_rsrc, base, offset_is_bytes=True)
            gpu.barrier()  # before next group re-stages dS/P
            pending.pop(0)

    @flyc.jit
    def launch(Q: fx.Tensor, DO: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor,
               INTERM: fx.Tensor, T: fx.Int32, stream: fx.Stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, DO, DS, PP, INTERM, T).launch(
            grid=(fx.Index(T), fx.Index(DBLK), 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


# The transpose for this head-contract GEMM is HW-cheapest via ds_read_tr16 from LDS: the
# dS/P LDS staging is essential to coalesce the dS/P read, so dropping it to fit a larger BD
# loses, and global_load_tr* (a gfx12/RDNA WMMA intrinsic) is not in the CDNA4 (gfx950) ISA.

# v2: LDS-staged, coalesced. Each WG owns one token x one d-tile(16): 256 threads coalesced-
# stage that d-tile's Q/dO[H][16] into LDS once, then 4 waves loop rank-tiles feeding the MFMA
# A operand via ds_read_tr16_b64. MFMA = mfma_f32_16x16x16bf16_1k (K=16). Grid=(T, D_V//16).
QLDS = 16                       # staged d-width per h-row (one d-tile)


# ============================================================================
# kernel: gather
# ============================================================================

LANES = 64
EPL = D_V // LANES    # 8 fp32 cols per lane -> two v4 halves (lo/hi)


def _store_dkv_bf16(dkv_rsrc, col_base, raw_lo, raw_hi):
    """Cast this lane's fp32 v4 lo/hi dKV accumulators to bf16 and store the 8 dKV cols
    [col_base .. col_base+7]. Folds the host dkv_acc.to(bf16) cast into the final write
    (reduction stays fp32 in-register, store rounds to bf16); two dwordx2 stores."""
    elem = fx.BFloat16
    bf_lo = Vec.from_elements([fx.BFloat16(_raw(Vec(raw_lo)[i])) for i in range_constexpr(4)], elem)
    bf_hi = Vec.from_elements([fx.BFloat16(_raw(Vec(raw_hi)[i])) for i in range_constexpr(4)], elem)
    buffer_ops.buffer_store(_raw(bf_lo.bitcast(fx.Int32)), dkv_rsrc, col_base * fx.Index(2), offset_is_bytes=True)
    buffer_ops.buffer_store(_raw(bf_hi.bitcast(fx.Int32)), dkv_rsrc, (col_base + fx.Index(4)) * fx.Index(2), offset_is_bytes=True)


def build_gather(nw=8):
    elem = fx.BFloat16
    THREADS = nw * LANES
    RED_ELEMS = nw * D_V         # LDS: [NW][512] fp32 partials

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_gather_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(red_off + RED_ELEMS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Interm: fx.Tensor, InvPtr: fx.Tensor, InvData: fx.Tensor, dKV: fx.Tensor,
             NKV: fx.Int32, N_TR: fx.Int32):
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        lds_red = SmemPtr(allocator.get_base(), red_off, fx.Float32.ir_type, shape=(RED_ELEMS,)).get()
        kv = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(LANES)
        lane = tid % fx.Index(LANES)

        invptr_rsrc = buffer_ops.create_buffer_resource(
            InvPtr, max_size=False, num_records_bytes=_raw((fx.Index(NKV) + fx.Index(1)) * fx.Index(4)))
        invdata_rsrc = buffer_ops.create_buffer_resource(
            InvData, max_size=False, num_records_bytes=_raw(fx.Index(N_TR) * fx.Index(4)))
        dkv_rsrc = buffer_ops.create_buffer_resource(
            dKV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)))  # bf16 dKV

        start = fx.Index(fx.Int32(buffer_ops.buffer_load(invptr_rsrc, kv, vec_width=1, dtype=fx.Int32)))
        end = fx.Index(fx.Int32(buffer_ops.buffer_load(invptr_rsrc, kv + fx.Index(1), vec_width=1, dtype=fx.Int32)))

        col_lo = lane * fx.Index(EPL)                    # this lane's 8 d-cols (lo=0..3, hi=4..7)
        acc_lo = Vec.filled(4, 0.0, fx.Float32)
        acc_hi = Vec.filled(4, 0.0, fx.Float32)

        loop_res = [acc_lo, acc_hi]
        for e, iter_args in range(start + wave, end, fx.Index(nw), init=[acc_lo, acc_hi]):
            entry = fx.Index(fx.Int32(buffer_ops.buffer_load(invdata_rsrc, e, vec_width=1, dtype=fx.Int32)))
            i_reb = make_bf16_rebased_rsrc(Interm, entry * fx.Index(D_V), fx.Index(D_V * 2))
            # single vec8 load (dwordx4) of this lane's 8 contiguous d-cols, split lo/hi ->
            # halves the load instruction count vs 2x vec4 (gather is issue-bound).
            iv8 = Vec(buffer_ops.buffer_load(i_reb, col_lo, vec_width=8, dtype=elem))
            iv_lo = Vec.from_elements([iv8[i] for i in range_constexpr(4)], elem)
            iv_hi = Vec.from_elements([iv8[4 + i] for i in range_constexpr(4)], elem)
            n_lo = Vec(iter_args[0]) + Vec(arith.ExtFOp(v4f_ty, _raw(iv_lo)).result)
            n_hi = Vec(iter_args[1]) + Vec(arith.ExtFOp(v4f_ty, _raw(iv_hi)).result)
            loop_res = yield [_raw(n_lo), _raw(n_hi)]

        # write this wave's partial to LDS[wave][col], reduce over waves, wave0 -> dkv.
        Vec(loop_res[0]).store(lds_red, [wave * fx.Index(D_V) + col_lo])
        Vec(loop_res[1]).store(lds_red, [wave * fx.Index(D_V) + col_lo + fx.Index(4)])
        gpu.barrier()

        if nw > 1:
            if wave == fx.Index(0):
                s_lo = Vec(loop_res[0]); s_hi = Vec(loop_res[1])
                for w in range_constexpr(1, nw):
                    s_lo = s_lo + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo])
                    s_hi = s_hi + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo + fx.Index(4)])
                _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, _raw(s_lo), _raw(s_hi))
        else:
            _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, loop_res[0], loop_res[1])

    @flyc.jit
    def launch(Interm, InvPtr, InvData, dKV, NKV, N_TR, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Interm, InvPtr, InvData, dKV, NKV, N_TR).launch(
            grid=(fx.Index(NKV), 1, 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


def build_partial_reduce(ns):
    """Sum Partial[pidx, 0..ns-1, :D_V] -> dKV[KV_BASE+pidx, :D_V] (f32). One WG per
    pool kv, 64 lanes x EPL cols. Companion pass-2 for build_gather_pool (ns-split)."""
    def _b():
        @flyc.kernel(known_block_size=[LANES, 1, 1])
        def k_fn(Partial: fx.Tensor, dKV: fx.Tensor, NKV: fx.Int32, KV_BASE: fx.Int32, NPOOL: fx.Int32):
            pidx = fx.Index(gpu.block_idx.x)
            kv = pidx + fx.Index(KV_BASE)
            lane = fx.Index(gpu.thread_idx.x)
            col_lo = lane * fx.Index(EPL)
            part_rsrc = buffer_ops.create_buffer_resource(
                Partial, max_size=False, num_records_bytes=_raw(fx.Index(NPOOL) * fx.Index(ns) * fx.Index(D_V * 4)))
            dkv_rsrc = buffer_ops.create_buffer_resource(
                dKV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)))  # bf16 dKV
            base0 = (pidx * fx.Index(ns)) * fx.Index(D_V)
            acc_lo = Vec(buffer_ops.buffer_load(part_rsrc, base0 + col_lo, vec_width=4, dtype=fx.Float32))
            acc_hi = Vec(buffer_ops.buffer_load(part_rsrc, base0 + col_lo + fx.Index(4), vec_width=4, dtype=fx.Float32))
            for sx in range_constexpr(1, ns):
                b = (pidx * fx.Index(ns) + fx.Index(sx)) * fx.Index(D_V)
                acc_lo = acc_lo + Vec(buffer_ops.buffer_load(part_rsrc, b + col_lo, vec_width=4, dtype=fx.Float32))
                acc_hi = acc_hi + Vec(buffer_ops.buffer_load(part_rsrc, b + col_lo + fx.Index(4), vec_width=4, dtype=fx.Float32))
            _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, _raw(acc_lo), _raw(acc_hi))

        @flyc.jit
        def launch(Partial, dKV, NKV, KV_BASE, NPOOL, stream):
            k_fn(Partial, dKV, NKV, KV_BASE, NPOOL).launch(grid=(fx.Index(NPOOL), 1, 1), block=(LANES, 1, 1), stream=stream)

        return _attach(launch)
    return _b()


def build_gather_banded(nw=8, rc=None):
    """Closed-form banded gather for the SWA local window (W=128); the inverse window map
    is closed-form so the whole CSR path is skipped. For kv=j, flat interm row =
    (j+local)*RC + (W-1-local), count = min(W, T-j); ``rc`` = interm row stride (R_CHUNK)."""
    elem = fx.BFloat16
    W = 128
    RC = W if rc is None else rc     # interm row stride (ranks per token)
    THREADS = nw * LANES
    RED_ELEMS = nw * D_V

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_gather_banded_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(red_off + RED_ELEMS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Interm: fx.Tensor, dKV: fx.Tensor, NKV: fx.Int32, T: fx.Int32):
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        lds_red = SmemPtr(allocator.get_base(), red_off, fx.Float32.ir_type, shape=(RED_ELEMS,)).get()
        kv = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(LANES)
        lane = tid % fx.Index(LANES)

        dkv_rsrc = buffer_ops.create_buffer_resource(
            dKV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)))  # bf16 dKV
        # cr0 interm is 512MB < 4GB, so the flat element offset is int32-safe -> a single plain
        # buffer resource works; no per-entry SRD rebase (that i64 rebase is only for pro cr4's
        # >4GB interm, which never uses this banded path).
        interm_rsrc = buffer_ops.create_buffer_resource(
            Interm, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(RC) * fx.Index(D_V * 2)))

        # count = min(W, T - kv): only the last W-1 kv have a short (edge) window.
        rem = fx.Index(T) - kv
        count = fx.Index(arith.select(rem < fx.Index(W), _raw(rem), _raw(fx.Index(W))))

        col_lo = lane * fx.Index(EPL)
        acc_lo = Vec.filled(4, 0.0, fx.Float32)
        acc_hi = Vec.filled(4, 0.0, fx.Float32)

        loop_res = [acc_lo, acc_hi]
        for local, iter_args in range(wave, count, fx.Index(nw), init=[acc_lo, acc_hi]):
            a_lo = Vec(iter_args[0]); a_hi = Vec(iter_args[1])
            entry = (kv + local) * fx.Index(RC) + fx.Index(W - 1) - local
            erow = entry * fx.Index(D_V)
            iv8 = Vec(buffer_ops.buffer_load(interm_rsrc, erow + col_lo, vec_width=8, dtype=elem))
            iv_lo = Vec.from_elements([iv8[i] for i in range_constexpr(4)], elem)  # 1x dwordx4 vs 2x dwordx2
            iv_hi = Vec.from_elements([iv8[4 + i] for i in range_constexpr(4)], elem)
            n_lo = a_lo + Vec(arith.ExtFOp(v4f_ty, _raw(iv_lo)).result)
            n_hi = a_hi + Vec(arith.ExtFOp(v4f_ty, _raw(iv_hi)).result)
            loop_res = yield [_raw(n_lo), _raw(n_hi)]

        Vec(loop_res[0]).store(lds_red, [wave * fx.Index(D_V) + col_lo])
        Vec(loop_res[1]).store(lds_red, [wave * fx.Index(D_V) + col_lo + fx.Index(4)])
        gpu.barrier()

        if nw > 1:
            if wave == fx.Index(0):
                s_lo = Vec(loop_res[0]); s_hi = Vec(loop_res[1])
                for w in range_constexpr(1, nw):
                    s_lo = s_lo + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo])
                    s_hi = s_hi + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo + fx.Index(4)])
                _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, _raw(s_lo), _raw(s_hi))
        else:
            _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, loop_res[0], loop_res[1])

    @flyc.jit
    def launch(Interm, dKV, NKV, T, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        # Launch only the T LOCAL kv (banded window). For cr0 T==NKV (all kv are local);
        # for cr128 the P pool kv (>=T) are handled by the pool-only CSR gather separately.
        k_fn(Interm, dKV, NKV, T).launch(
            grid=(fx.Index(T), 1, 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


def build_gather_pool(nw=16, ns=32):
    """Closed-form pool gather for cr=128 (deterministic causal HCA pool): token i attends
    pool block b (kv=T+b) iff i >= (b+1)*cr_pool - 1 and visibility is monotone in b, so
    column b == block b == kv gives a closed-form inverse. Same NS-split + LDS reduce."""
    THREADS = nw * LANES
    RED_ELEMS = nw * D_V
    GW = ns * nw
    elem = fx.BFloat16

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_gather_pool_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(red_off + RED_ELEMS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Interm: fx.Tensor, Partial: fx.Tensor, NKV: fx.Int32, N_TR: fx.Int32,
             T: fx.Int32, NPOOL: fx.Int32, CR_POOL: fx.Int32, R_CHUNK: fx.Int32, R_OFF: fx.Int32):
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        lds_red = SmemPtr(allocator.get_base(), red_off, fx.Float32.ir_type, shape=(RED_ELEMS,)).get()
        pidx = fx.Index(gpu.block_idx.x)          # pool block b (0..NPOOL-1)
        s = fx.Index(gpu.block_idx.y)
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(LANES)
        lane = tid % fx.Index(LANES)
        gw = s * fx.Index(nw) + wave

        interm_rsrc = buffer_ops.create_buffer_resource(
            Interm, max_size=False, num_records_bytes=_raw(fx.Index(N_TR) * fx.Index(D_V * 2)))
        part_rsrc = buffer_ops.create_buffer_resource(
            Partial, max_size=False, num_records_bytes=_raw(fx.Index(NPOOL) * fx.Index(ns) * fx.Index(D_V * 4)))

        # start_i(b) = (b+1)*cr_pool - 1 (first token that sees block b); count = T - start_i
        # (>= 1 for every block since the last block's start_i = T - 1). rank = R_OFF + b.
        start_i = (pidx + fx.Index(1)) * fx.Index(CR_POOL) - fx.Index(1)
        count = fx.Index(T) - start_i
        rank = fx.Index(R_OFF) + pidx

        col_lo = lane * fx.Index(EPL)
        acc_lo = Vec.filled(4, 0.0, fx.Float32)
        acc_hi = Vec.filled(4, 0.0, fx.Float32)

        loop_res = [acc_lo, acc_hi]
        for j, iter_args in range(gw, count, fx.Index(GW), init=[acc_lo, acc_hi]):
            a_lo = Vec(iter_args[0]); a_hi = Vec(iter_args[1])
            tok = start_i + j
            erow = (tok * fx.Index(R_CHUNK) + rank) * fx.Index(D_V)
            iv8 = Vec(buffer_ops.buffer_load(interm_rsrc, erow + col_lo, vec_width=8, dtype=elem))
            iv_lo = Vec.from_elements([iv8[i] for i in range_constexpr(4)], elem)  # 1x dwordx4 vs 2x dwordx2
            iv_hi = Vec.from_elements([iv8[4 + i] for i in range_constexpr(4)], elem)
            n_lo = a_lo + Vec(arith.ExtFOp(v4f_ty, _raw(iv_lo)).result)
            n_hi = a_hi + Vec(arith.ExtFOp(v4f_ty, _raw(iv_hi)).result)
            loop_res = yield [_raw(n_lo), _raw(n_hi)]

        Vec(loop_res[0]).store(lds_red, [wave * fx.Index(D_V) + col_lo])
        Vec(loop_res[1]).store(lds_red, [wave * fx.Index(D_V) + col_lo + fx.Index(4)])
        gpu.barrier()

        if wave == fx.Index(0):
            s_lo = Vec(loop_res[0]); s_hi = Vec(loop_res[1])
            for w in range_constexpr(1, nw):
                s_lo = s_lo + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo])
                s_hi = s_hi + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo + fx.Index(4)])
            poff = (pidx * fx.Index(ns) + s) * fx.Index(D_V)
            buffer_ops.buffer_store(_raw(s_lo), part_rsrc, (poff + col_lo) * fx.Index(4), offset_is_bytes=True)
            buffer_ops.buffer_store(_raw(s_hi), part_rsrc, (poff + col_lo + fx.Index(4)) * fx.Index(4), offset_is_bytes=True)

    @flyc.jit
    def launch(Interm, Partial, NKV, N_TR, T, NPOOL, CR_POOL, R_CHUNK, R_OFF, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Interm, Partial, NKV, N_TR, T, NPOOL, CR_POOL, R_CHUNK, R_OFF).launch(
            grid=(fx.Index(NPOOL), ns, 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


# ============================================================================
# kernel: fused
# ============================================================================

HPW = 16
D_LDS = 528
TILE_K = 16
KS = D // 32            # QK K-steps (16)
DT = D // 16            # PV / interm d-tiles (32)
BD_INT = 128            # interm Q/dO resident d-block width
NDB = D // BD_INT       # 4 d-blocks
BD_LDSB = BD_INT + 8    # padded LDS row stride for the d-block (136)
KSB = BD_INT // 32      # QK-packs per d-block (4)


def build_bwd_fused(topk_len, scale, num_heads=64):
    # 3 = K32 2-chain ILP; gated to topk<=128 (short-R flash).
    interm_chains = 3 if topk_len <= 128 else 1
    elem = fx.BFloat16
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    # batch factor: kv-tiles per d-block stage, amortizing the Q/dO restage barriers. Larger
    # KB = fewer restage passes, but too large loses the dQ/interm-phase pipeline overlap; 2
    # restage passes is the sweet spot.
    KB = 5 if NUM_TILES % 5 == 0 else (4 if NUM_TILES % 4 == 0 else (2 if NUM_TILES % 2 == 0 else 1))
    BLOCK_H = num_heads          # flash: 64 -> 1 WG all heads
    WAVES = BLOCK_H // HPW        # 4
    THREADS = WAVES * 64         # 256
    KSH = num_heads // 16        # interm head K-steps (flash 4)
    DUW = DT // WAVES // NDB      # interm d-tiles per wave per d-block (32/4/4 = 2)
    KV1 = TILE_K * D_LDS         # one KV tile in LDS (elems)
    KV_ELEMS = 2 * KV1           # KV double-buffer (ping-pong by kt%2) -> drop overwrite barrier
    QB_ELEMS = num_heads * BD_LDSB   # one d-block of Q/dO resident (occ-2)

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_fused_smem")
    kv_off = allocator._align(allocator.ptr, 16)
    qb_off = allocator._align(kv_off + KV_ELEMS * 2, 16)
    dob_off = allocator._align(qb_off + QB_ELEMS * 2, 16)
    ds_off = allocator._align(dob_off + QB_ELEMS * 2, 16)
    pp_off = allocator._align(ds_off + KB * num_heads * TILE_K * 2, 16)
    mask_off = allocator._align(pp_off + KB * num_heads * TILE_K * 2, 16)
    allocator.ptr = allocator._align(mask_off + KB * TILE_K * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, KV: fx.Tensor, DO: fx.Tensor, TOPK: fx.Tensor,
             LSE: fx.Tensor, DELTA: fx.Tensor, DQ: fx.Tensor, INTERM: fx.Tensor,
             T: fx.Int32, H: fx.Int32, NKV: fx.Int32):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        base = allocator.get_base()
        lds_kv = SmemPtr(base, kv_off, elem.ir_type, shape=(KV_ELEMS,)).get()
        lds_qb = SmemPtr(base, qb_off, elem.ir_type, shape=(QB_ELEMS,)).get()
        lds_dob = SmemPtr(base, dob_off, elem.ir_type, shape=(QB_ELEMS,)).get()
        lds_ds = SmemPtr(base, ds_off, elem.ir_type, shape=(KB * num_heads * TILE_K,)).get()
        lds_p = SmemPtr(base, pp_off, elem.ir_type, shape=(KB * num_heads * TILE_K,)).get()
        lds_mask = SmemPtr(base, mask_off, fx.Float32.ir_type, shape=(KB * TILE_K,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        token = fx.Index(gpu.block_idx.x)
        Hn = fx.Index(H)
        head_A = wave * fx.Index(HPW) + lo

        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)))
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4)))
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)))
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        interm_rsrc = buffer_ops.create_buffer_resource(
            INTERM, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len) * fx.Index(D * 2)))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)))
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)))

        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)

        # ---- Q/dO register-resident (B operands for QK; also source for interm d-block stage) ----
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        q_packs = [buffer_ops.buffer_load(q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8),
                                          vec_width=8, dtype=elem) for ks in range_constexpr(KS)]
        do_packs = [buffer_ops.buffer_load(do_rsrc, do_row + fx.Index(ks * 32) + grp * fx.Index(8),
                                           vec_width=8, dtype=elem) for ks in range_constexpr(KS)]

        lse_h = fx.Float32(buffer_ops.buffer_load(lse_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        lse_h = fx.Float32(arith.MaxNumFOp(_raw(lse_h), _raw(fx.Float32(-3.0e38))).result)
        lse_l2 = fx.Float32(lse_h * c_log2e)
        delta_h = fx.Float32(buffer_ops.buffer_load(delta_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))

        g_row = tid // fx.Index(16)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def load_topk(tbase):
            return fx.Int32(buffer_ops.buffer_load(tk_rsrc, tk_row + tbase + g_row, vec_width=1, dtype=fx.Int32))

        def gather_regs(idx):  # HBM -> registers (overlaps compute)
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            vvs = [buffer_ops.buffer_load(kv_rsrc, src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                                          vec_width=8, dtype=elem) for c in range_constexpr(4)]
            mval = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
            return vvs, mval

        def store_kv(vvs, mval, kt):  # registers -> LDS (KV double-buffer; mask per-kt)
            bufo = (kt % 2) * KV1
            for c in range_constexpr(4):
                Vec(vvs[c]).store(lds_kv, [fx.Index(bufo) + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)])
            if g_within == fx.Index(0):
                Vec.from_elements([mval], fx.Float32).store(lds_mask, [fx.Index(kt * TILE_K) + g_row])

        # tr16 LDS transpose read helper (A/B operand for the interm head-contraction GEMM)
        def tr_h(off, stride, coltile, ks):
            row = (fx.Index(ks * 16) + grp * fx.Index(4) + lo_d4) * stride + coltile * fx.Index(16) + lo_m4 * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DT)]
        for kb, iter_args in range(fx.Index(0), fx.Index(NUM_TILES), fx.Index(KB), init=dq_acc0):
            dq_acc = [iter_args[dt] for dt in range_constexpr(DT)]

            # ============ dQ phase: per-tile QK/dS/dP/PV, stash dS/P to lds_ds[kt] ============
            for kt in range_constexpr(KB):
                t = kb + fx.Index(kt)
                bufo = (kt % 2) * KV1   # KV double-buffer offset (ping-pong by kt%2)
                kv_cur, m_cur = gather_regs(load_topk(t * fx.Index(TILE_K)))
                store_kv(kv_cur, m_cur, kt)
                # KV/mask visible; kt+2 reuses this buf only after the next tile's barrier below.
                gpu.barrier()

                mask4 = Vec.load(v4f, lds_mask, [fx.Index(kt * TILE_K) + grp * fx.Index(4)])

                # QK x2 (S, dP): A=KV(lds), B=Q/dO(regs). 2-acc ILP.
                acc_s0 = Vec.filled(4, 0.0, fx.Float32); acc_s1 = Vec.filled(4, 0.0, fx.Float32)
                acc_dp0 = Vec.filled(4, 0.0, fx.Float32); acc_dp1 = Vec.filled(4, 0.0, fx.Float32)

                def _bv(ks):
                    return Vec.load(v8, lds_kv, [fx.Index(bufo) + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)])
                PF = 4
                bvq = [_bv(k) for k in range_constexpr(PF)]
                for ks in range_constexpr(KS):
                    if ks + PF < KS:
                        bvq.append(_bv(ks + PF))
                    if ks % 2 == 0:
                        acc_s0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc_s0])
                        acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), do_packs[ks], acc_dp0])
                    else:
                        acc_s1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc_s1])
                        acc_dp1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), do_packs[ks], acc_dp1])
                acc_s = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i])) for i in range_constexpr(4)], fx.Float32)
                acc_dp = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i])) for i in range_constexpr(4)], fx.Float32)

                # PVPF: PV's V tr16 reads (bvv) depend only on the resident KV-LDS (bufo), not on
                # softmax -> hoist the first _PVPF reads before the exp2 loop so their LDS-read
                # latency overlaps the softmax VALU.
                pv_base = fx.Int64(
                    (lo_d4 * fx.Index(D_LDS) + grp * fx.Index(4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4))
                    * fx.Index(2) + fx.Index(kv_off + bufo * 2))
                def _bvv(dt):
                    ptr = buffer_ops.create_llvm_ptr(_raw(pv_base + fx.Int64(dt * 32)), address_space=3)
                    return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))
                _PVPF = 8
                bvv_pf = [_bvv(dt) for dt in range_constexpr(_PVPF)]

                pvals = [None] * 4
                dsvals = [None] * 4
                for i in range_constexpr(4):
                    arg = fx.Float32(_raw(Vec(acc_s)[i])) * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                    p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                    pvals[i] = p
                    dsvals[i] = p * (fx.Float32(_raw(Vec(acc_dp)[i])) - delta_h) * c_scale

                pB = _raw(Vec.from_elements([fx.BFloat16(_raw(dsvals[i])) for i in range_constexpr(4)], elem).bitcast(fx.Int16))

                # hand dS/P to LDS scratch [kt][head_A][kv=grp*4+i] for the interm head-contraction.
                base_ds = fx.Index(kt * num_heads * TILE_K) + head_A * fx.Index(TILE_K) + grp * fx.Index(4)
                for i in range_constexpr(4):
                    Vec.from_elements([fx.BFloat16(_raw(dsvals[i]))], elem).store(lds_ds, [base_ds + fx.Index(i)])
                    Vec.from_elements([fx.BFloat16(_raw(pvals[i]))], elem).store(lds_p, [base_ds + fx.Index(i)])

                # ---- PV: dQ += dS@K (A=V tr16, B=dS regs) ---- (per-tile, all DT d-tiles)
                # dq_acc (loop-carried, occ wall) pinned in AGPR via inline-asm MFMA (=a,v,v,0,
                # D=C in-place, no accvgpr shuffle) -> frees arch-VGPR.
                _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())
                def _mma_agpr_k16(a, b, c):
                    op = _llvm.InlineAsmOp(res=_v4f32_ir, operands_=[_raw(a), _raw(b), _raw(c)],
                                           asm_string="v_mfma_f32_16x16x16_bf16 $0, $1, $2, $0",
                                           constraints="=a,v,v,0", has_side_effects=False)
                    return op.result
                new_dq = [None] * DT
                for dt in range_constexpr(DT):
                    if dt + _PVPF < DT:
                        bvv_pf.append(_bvv(dt + _PVPF))
                    new_dq[dt] = _mma_agpr_k16(bvv_pf[dt], pB, dq_acc[dt])
                dq_acc = new_dq

            gpu.barrier()  # all KB tiles' dS/P visible before the interm phase


            # ============ interm phase: d-block outer, kv-tile inner (reuse staged Q/dO) ======
            for db in range_constexpr(NDB):
                # stage one 128-d block of Q/dO from registers to LDS (once per KB tiles).
                for kl in range_constexpr(KSB):
                    ks = db * KSB + kl
                    d_local = fx.Index(kl * 32) + grp * fx.Index(8)
                    Vec(q_packs[ks]).store(lds_qb, [head_A * fx.Index(BD_LDSB) + d_local])
                    Vec(do_packs[ks]).store(lds_dob, [head_A * fx.Index(BD_LDSB) + d_local])
                gpu.barrier()  # Q/dO d-block visible

                for kt in range_constexpr(KB):
                    t = kb + fx.Index(kt)
                    ds_off_kt = ds_off + kt * num_heads * TILE_K * 2
                    pp_off_kt = pp_off + kt * num_heads * TILE_K * 2
                    # The interm B operands (dS/P) index only (kt, ks), not the d-tile, so their
                    # tr16 reads are hoisted once per (db, kt) and reused across the d-tile loop;
                    # the A operands (Q/dO) stay per-d-tile.
                    bd_h = [_c8(tr_h(ds_off_kt, fx.Index(TILE_K), fx.Index(0), 2*k2),
                                tr_h(ds_off_kt, fx.Index(TILE_K), fx.Index(0), 2*k2+1)) for k2 in range_constexpr(KSH // 2)]
                    bp_h = [_c8(tr_h(pp_off_kt, fx.Index(TILE_K), fx.Index(0), 2*k2),
                                tr_h(pp_off_kt, fx.Index(TILE_K), fx.Index(0), 2*k2+1)) for k2 in range_constexpr(KSH // 2)]
                    for u in range_constexpr(DUW):
                        dt = db * (DT // NDB) + wave + fx.Index(u * WAVES)   # global d-tile
                        dt_local = wave + fx.Index(u * WAVES)                # 0..7 within block
                        if const_expr(interm_chains == 3):
                            # K=32 (halved MFMA count) + 2 independent iacc chains (Q@dS, dO@P)
                            # -> 2-way ILP hides the mfma RAW latency on this occ-2 head-contract
                            # GEMM.
                            iacc_a = Vec.filled(4, 0.0, fx.Float32)
                            iacc_b = Vec.filled(4, 0.0, fx.Float32)
                            for k2 in range_constexpr(KSH // 2):
                                aq = _c8(tr_h(qb_off, fx.Index(BD_LDSB), dt_local, 2*k2), tr_h(qb_off, fx.Index(BD_LDSB), dt_local, 2*k2+1))
                                iacc_a = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq, bd_h[k2], iacc_a])
                                ao = _c8(tr_h(dob_off, fx.Index(BD_LDSB), dt_local, 2*k2), tr_h(dob_off, fx.Index(BD_LDSB), dt_local, 2*k2+1))
                                iacc_b = rocdl.mfma_f32_16x16x32_bf16(v4f, [ao, bp_h[k2], iacc_b])
                            iacc = Vec.from_elements(
                                [fx.Float32(_raw(Vec(iacc_a)[i])) + fx.Float32(_raw(Vec(iacc_b)[i])) for i in range_constexpr(4)], fx.Float32)
                        else:
                            # K=32 MFMA: concat two K=16 tr16 h-blocks -> v8 (halves MFMA count).
                            iacc = Vec.filled(4, 0.0, fx.Float32)
                            for k2 in range_constexpr(KSH // 2):
                                aq = _c8(tr_h(qb_off, fx.Index(BD_LDSB), dt_local, 2*k2), tr_h(qb_off, fx.Index(BD_LDSB), dt_local, 2*k2+1))
                                iacc = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq, bd_h[k2], iacc])
                            for k2 in range_constexpr(KSH // 2):
                                ao = _c8(tr_h(dob_off, fx.Index(BD_LDSB), dt_local, 2*k2), tr_h(dob_off, fx.Index(BD_LDSB), dt_local, 2*k2+1))
                                iacc = rocdl.mfma_f32_16x16x32_bf16(v4f, [ao, bp_h[k2], iacc])
                        iov = Vec(iacc)
                        kv_g = t * fx.Index(TILE_K) + lo
                        d_g = dt * fx.Index(16) + grp * fx.Index(4)
                        ibase = (token * fx.Index(topk_len) * fx.Index(D) + kv_g * fx.Index(D) + d_g) * fx.Index(2)
                        bf4 = Vec.from_elements([fx.BFloat16(_raw(Vec(iov)[i])) for i in range_constexpr(4)], elem)
                        buffer_ops.buffer_store(_raw(bf4.bitcast(fx.Int32)), interm_rsrc, ibase, offset_is_bytes=True)
                gpu.barrier()  # before next d-block restages lds_qb / lds_dob

            loop_results = yield list(dq_acc)

        dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
        for dt in range_constexpr(DT):
            ov = Vec(dq_acc[dt])
            base = (token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)) * fx.Index(2)
            pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))))
            pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))))
            buffer_ops.buffer_store(_raw(Vec.from_elements([pk0, pk1], fx.Int32)), dq_rsrc, base, offset_is_bytes=True)
        # Zero the 64 rope cols (512..575) in-kernel (rope grad is dead).
        zero_v2 = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        for rt in range_constexpr(4):
            rbase = (token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)) * fx.Index(2)
            buffer_ops.buffer_store(_raw(zero_v2), dq_rsrc, rbase, offset_is_bytes=True)

    @flyc.jit
    def launch(Q: fx.Tensor, KV: fx.Tensor, DO: fx.Tensor, TOPK: fx.Tensor,
               LSE: fx.Tensor, DELTA: fx.Tensor, DQ: fx.Tensor, INTERM: fx.Tensor,
               T: fx.Int32, H: fx.Int32, NKV: fx.Int32, stream: fx.Stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, KV, DO, TOPK, LSE, DELTA, DQ, INTERM, T, H, NKV).launch(
            grid=(fx.Index(T), 1, 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


# ============================================================================
# host-side dispatch
# ============================================================================

_GATHER_NS = 32  # cr128 pool-only multi-WG split factor (grid.y); pool is ~32 kv so cheap


def _build_inverted_topk_fast(flat_kv, num_kv):
    """CSR inverted-topk index via sort+searchsorted: one stable sort yields the permutation
    (inv_data) and the sorted keys, inv_ptr[k] = searchsorted(sorted_vals, k, 'left'). Narrows
    the sort key to int16 when num_kv fits (bit-identical, less data moved)."""
    if num_kv < 32768:
        keys = flat_kv.to(torch.int16)
        ar = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int16)
    else:
        keys = flat_kv
        ar = torch.arange(num_kv + 1, device=flat_kv.device, dtype=flat_kv.dtype)
    sorted_vals, inv_data = torch.sort(keys, stable=True)
    inv_ptr = torch.searchsorted(sorted_vals, ar).to(torch.int32)
    return inv_ptr, inv_data.to(torch.int32)


_DQ_CACHE: dict = {}


def _get_dq(topk_len, scale, num_heads=None):
    key = (topk_len, float(scale), int(num_heads) if num_heads is not None else -1)
    fn = _DQ_CACHE.get(key)
    if fn is None:
        # PV-K32: per-16 QK/softmax kept, PV batches 2 tiles into mfma_16x16x32 with direct-v8
        # operands. Gated to large-topk (topk>=512, even tile count); small-topk pair-batch's
        # per-pair WAR barrier overhead loses. Both dQ kernels pin dq_acc in AGPR via inline-asm.
        if topk_len >= 512 and topk_len % 32 == 0:
            fn = build_bwd_dq_pvk32(topk_len, float(scale), num_heads=num_heads)
        else:
            fn = build_bwd_dq(topk_len, float(scale), num_heads=num_heads)
        _DQ_CACHE[key] = fn
    return fn


def _get_fused(topk_len, scale, num_heads):
    key = ("fused", topk_len, float(scale), int(num_heads))
    fn = _DQ_CACHE.get(key)
    if fn is None:
        fn = build_bwd_fused(topk_len, float(scale), int(num_heads))
        _DQ_CACHE[key] = fn
    return fn


_DELTA_CACHE: dict = {}
_DSINK_SPLIT_CACHE: dict = {}


def _get_delta():
    fn = _DELTA_CACHE.get("fn")
    if fn is None:
        fn = build_delta()
        _DELTA_CACHE["fn"] = fn
    return fn


_GATHER_CACHE: dict = {}


def _get_gather():
    fn = _GATHER_CACHE.get("fn")
    if fn is None:
        # nw=16: the HCA cr128 pool-entry-0 CSR list is ~4000 long and one WG per kv walked it
        # serially; 16 waves split the list. cr0/cr4 (balanced/BW-bound) neutral.
        fn = build_gather(16)
        _GATHER_CACHE["fn"] = fn
    return fn


def _get_gather_banded(rc=None):
    # rc = interm row stride (R_CHUNK). None -> W=128 (cr0). cr128 local band uses rc=192
    # (the padded topk width) since the local ranks 0..127 live in a wider interm row.
    key = ("banded", rc)
    fn = _GATHER_CACHE.get(key)
    if fn is None:
        fn = build_gather_banded(8, rc=rc)
        _GATHER_CACHE[key] = fn
    return fn


def _get_gather_pool():
    fn = _GATHER_CACHE.get("pool")
    if fn is None:
        fn = build_gather_pool(16, _GATHER_NS)
        _GATHER_CACHE["pool"] = fn
    return fn


def _get_partial_reduce():
    fn = _GATHER_CACHE.get("partred")
    if fn is None:
        fn = build_partial_reduce(_GATHER_NS)
        _GATHER_CACHE["partred"] = fn
    return fn


_INTERM_CACHE: dict = {}


def _get_interm(topk_len, num_heads):
    # rtr (register-transpose Q/dO, no Q/dO LDS -> BD256) for pro (num_heads>64); flash (H<=64)
    # uses blk (tuned wide BD already minimizes re-read; rtr's transpose overhead not amortized
    # at H=64).
    use_rtr = int(num_heads) > 64
    mode = "rtr" if use_rtr else "blk"
    key = (topk_len, num_heads, mode)
    fn = _INTERM_CACHE.get(key)
    if fn is None:
        if mode == "rtr":
            # interm stays baseline (functional rocdl.mfma): inline-asm in-place MFMA breaks det
            # on interm's per-output fresh accumulator (the AGPR win only holds for dQ's
            # loop-carried acc).
            fn = build_interm_rtr(topk_len, num_heads, 256)
        else:
            # BD is R_CHUNK-dependent (interm is latency-bound). flash (H<=64): wider BD reduces
            # the d-block count -> less dS/P HBM re-read; pro (H=128): BD128 (BD256=176KB > cap).
            if num_heads <= 64:
                # flash: cr4 (R=640) prefers BD512; small-R cr0/cr128 prefer BD256 (BD512
                # under-utilizes at 1WG). interm is latency-bound so ILP > occupancy.
                bd = 512 if topk_len >= 640 else 256
            else:
                # pro (H=128) BD128 beats BD64 at every R: the read-batching makes BD128's fewer
                # d-blocks (less dS/P re-read) dominate BD64's 2WG occupancy. BD256 = 176KB > cap.
                bd = 128
            fn = build_interm_blk(topk_len, num_heads, bd)
        _INTERM_CACHE[key] = fn
    return fn


def sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    # flydsl heads-split dQ + flydsl dKV-interm/gather.
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    D = kv_lora_rank
    rope_rank = d_qk - D
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]
    assert D == 512 and d_qk == 576, f"flydsl bwd fixed to D=512 d_qk=576, got D={D} d_qk={d_qk}"
    assert num_heads % 32 == 0 and topk % 32 == 0

    # cr128 padding drop: the caller pads topk to a multiple of 64; the pad ranks are always -1
    # (contribute 0 dS/P), so slicing to the real 128+npool width shrinks the dQ/interm work
    # with no behavior change. Guarded to the deterministic HCA shape; cr0/cr4 untouched.
    _npool = num_kv - total_tokens
    _topk_real = 128 + _npool
    if 0 < _npool and topk <= 256 and topk > _topk_real and _topk_real % 32 == 0:
        topk = _topk_real
        topk_indices = topk_indices[:, :topk].contiguous()

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # The CSR build is kept sequential (overlapping it on a side stream contends with the
    # compute kernels; the memory-bound sort + stream setup dominates the short flash path).

    # delta = rowsum(O*dO): fp32 accumulation with a bf16 product (avoids materializing two
    # fp32 copies of O/dO); SNR-safe. Kept a standalone kernel (folding into the occ-1 dQ
    # kernel extends its latency more than it saves).
    assert o.shape[-1] == D and o.is_contiguous()
    # do_lora dedup: delta and the dQ kernel both need a contiguous [T,H,512] copy of the lora
    # cols; build it once and share.
    do_lora = do[..., :D].contiguous()  # [T, H, 512]
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)  # [T, H]
    # flydsl delta = rowsum(O*dO), fp32 (replaces the triton _bwd_delta_kernel).
    _dstream = torch.cuda.current_stream()
    _dargs = (o.reshape(-1, D), do_lora.reshape(-1, D),
              delta.reshape(-1), int(total_tokens * num_heads), _dstream)
    _dc = _DELTA_CACHE.get("c")
    if _dc is None:
        _dc = _get_delta().compile(*_dargs)
        _DELTA_CACHE["c"] = _dc
    _dc(*_dargs)
    # The dQ/fused kernels write ALL D_V=512 lora cols for every (token,head) AND
    # zero the 64 rope cols (512..575) in-kernel (matches triton's zeroed rope),
    # folding away the strided host dq[..., D:].zero_().
    dq = torch.empty_like(q)
    stream = torch.cuda.current_stream()
    # FUSED dQ+interm (flash small-topk only): dS/P handed through LDS in one kernel (no
    # chunk_dS/P HBM), producing dq + interm together. Wins for small topk (occ-2
    # kv-block-batching); pro (H=128) never fuses (8 waves -> register spill, occ-2 lost).
    use_fused = int(num_heads) <= 64 and int(topk) <= 256
    if not use_fused:
        chunk_dS = torch.empty(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        chunk_P = torch.empty(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        # ---- flydsl dQ (also produces dS / P for the dKV-interm kernel) ----
        fn = _get_dq(topk, scale, num_heads)
        args = (q, kv, do_lora, topk_indices, lse, delta, dq, chunk_dS, chunk_P,
                int(total_tokens), int(num_heads), int(num_kv), stream)
        ckey = ("c", topk, float(scale))
        compiled = _DQ_CACHE.get(ckey)
        if compiled is None:
            compiled = fn.compile(*args)
            _DQ_CACHE[ckey] = compiled
        compiled(*args)

    # interm only ever writes/reads the D_V=512 lora cols (rope cols are dead) -> allocate it
    # D_V-wide (not d_qk=576): cuts the dominant dKV-interm HBM write AND the gather read by 11%.
    R_CHUNK = topk
    interm = torch.empty(total_tokens, R_CHUNK, D, dtype=torch.bfloat16, device=q.device)
    # dKV is written directly in bf16 by the gather kernels (reduction stays fp32 in-register,
    # final store rounds to bf16). The zeros init covers the rope cols and any kv row
    # unreferenced by a query.
    dkv = torch.zeros(num_kv, d_qk, dtype=kv.dtype, device=q.device)

    # ---- flydsl dKV-interm: register-transpose (rtr) for pro, 2D-blocked (blk) for flash.
    if use_fused:
        # one kernel produced dq + interm already (dS/P via LDS, no chunk_dS/P HBM).
        ffn = _get_fused(int(topk), float(scale), int(num_heads))
        fargs = (q, kv, do_lora, topk_indices, lse, delta, dq, interm,
                 int(total_tokens), int(num_heads), int(num_kv), stream)
        fckey = ("fc", int(topk), float(scale), int(num_heads))
        fc = _DQ_CACHE.get(fckey)
        if fc is None:
            fc = ffn.compile(*fargs)
            _DQ_CACHE[fckey] = fc
        fc(*fargs)
    else:
        ifn = _get_interm(int(topk), int(num_heads))
        iargs = (q, do_lora, chunk_dS, chunk_P, interm, int(total_tokens), stream)
        ickey = ("c", int(topk), int(num_heads))
        icompiled = _INTERM_CACHE.get(ickey)
        if icompiled is None:
            icompiled = ifn.compile(*iargs)
            _INTERM_CACHE[ickey] = icompiled
        icompiled(*iargs)

    # cr=0 (pure SWA): num_kv==T and topk==W=128 -> use the closed-form banded gather
    # (skips the CSR argsort/bincount/cumsum + InvPtr/InvData buffers entirely; the
    # inverse window map is closed-form). Bit-exact vs the CSR path.
    is_cr0 = (num_kv == total_tokens) and (topk == 128)
    is_cr128 = (num_kv > total_tokens) and (topk <= 256)
    if is_cr0:
        _gargs = (interm.reshape(-1, D), dkv, int(num_kv), int(total_tokens), stream)
        _gc = _GATHER_CACHE.get("cb")
        if _gc is None:
            _gc = _get_gather_banded().compile(*_gargs)
            _GATHER_CACHE["cb"] = _gc
        _gc(*_gargs)
    elif is_cr128:
        # cr=128 (HCA): both the local SWA band and the pool are deterministic, so both invert
        # closed-form. (1) banded-local gather (rc=R_CHUNK) for kv 0..T-1. (2) closed-form pool
        # gather for kv>=T. Local and pool kv write disjoint dkv rows -> no conflict.
        _bargs = (interm.reshape(-1, D), dkv, int(num_kv), int(total_tokens), stream)
        _bkey = ("cb128", int(R_CHUNK))
        _bc = _GATHER_CACHE.get(_bkey)
        if _bc is None:
            _bc = _get_gather_banded(rc=int(R_CHUNK)).compile(*_bargs)
            _GATHER_CACHE[_bkey] = _bc
        _bc(*_bargs)

        # The REAL pool occupies ranks 128..128+P-1 (P = npool); ranks 128+P..R_CHUNK are
        # -1 padding (topk padded to a multiple of 64).
        npool = int(num_kv - total_tokens)
        partial = torch.empty(npool, _GATHER_NS, D, dtype=torch.float32, device=q.device)
        # Closed-form pool: token i attends pool block b (kv T+b) iff i >= (b+1)*cr_pool - 1;
        # visibility is monotone in b so column b == block b, giving a closed-form inverse.
        # Production shapes always have clean pool blocks (total_tokens % npool == 0).
        assert total_tokens % npool == 0, "cr128 pool blocks must be clean"
        cr_pool = total_tokens // npool
        _pargs = (interm.reshape(-1, D), partial, int(num_kv), int(total_tokens * R_CHUNK),
                  int(total_tokens), int(npool), int(cr_pool), int(R_CHUNK), 128, stream)
        _pc = _GATHER_CACHE.get("cpool")
        if _pc is None:
            _pc = _get_gather_pool().compile(*_pargs)
            _GATHER_CACHE["cpool"] = _pc
        _pc(*_pargs)
        _rargs = (partial.reshape(-1, D), dkv, int(num_kv), int(total_tokens), int(npool), stream)
        _rc = _GATHER_CACHE.get("cr")
        if _rc is None:
            _rc = _get_partial_reduce().compile(*_rargs)
            _GATHER_CACHE["cr"] = _rc
        _rc(*_rargs)
    else:
        # cr4: full CSR single-WG gather. CSR inverted-topk scatter of interm -> dkv[:, :D]
        # (bf16 cast in-kernel), rope cols untouched. The NS-split loses here: cr4's long pool
        # lists already saturate all CUs (BW-bound), so it would only add partial traffic.
        inv_ptr, inv_data = _build_inverted_topk_fast(topk_indices.reshape(-1), num_kv)
        _gargs = (interm.reshape(-1, D), inv_ptr.contiguous(), inv_data.contiguous(), dkv,
                  int(num_kv), int(total_tokens * R_CHUNK), stream)
        _gc = _GATHER_CACHE.get("c")
        if _gc is None:
            _gc = _get_gather().compile(*_gargs)
            _GATHER_CACHE["c"] = _gc
        _gc(*_gargs)

    d_sink = None
    if has_sink:
        # flydsl d_sink[h] = -sum_t exp(sink[h]-lse[t,h])*delta[t,h], 2-pass coalesced: pass 1
        # reads full rows contiguously into per-block partials[nblk,H], pass 2 reduces the nblk
        # blocks -> d_sink. fp32.
        d_sink = torch.empty(num_heads, dtype=torch.float32, device=q.device)
        _nblk = (int(total_tokens) + DSINK_TB - 1) // DSINK_TB
        _dspart = torch.empty(_nblk, int(num_heads), dtype=torch.float32, device=q.device)
        _p1args = (lse.reshape(-1), delta.reshape(-1), attn_sink, _dspart.reshape(-1),
                   int(total_tokens), int(_nblk), stream)
        _p2args = (_dspart.reshape(-1), d_sink, stream)
        _dskey = (int(total_tokens), int(num_heads))
        _dsc = _DSINK_SPLIT_CACHE.get(_dskey)
        if _dsc is None:
            _p1 = build_dsink_split(int(total_tokens), int(num_heads), DSINK_TB).compile(*_p1args)
            _p2 = build_dsink_reduce(int(_nblk), int(num_heads)).compile(*_p2args)
            _dsc = (_p1, _p2)
            _DSINK_SPLIT_CACHE[_dskey] = _dsc
        _dsc[0](*_p1args)
        _dsc[1](*_p2args)

    dkv = dkv.unsqueeze(1)
    return dq, dkv, d_sink
