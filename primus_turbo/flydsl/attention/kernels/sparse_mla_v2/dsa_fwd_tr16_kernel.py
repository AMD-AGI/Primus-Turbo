# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 sparse-MLA forward, M=16 head-as-M MFMA + ds_read_tr PV (path B).

Third-generation native-FlyDSL fused single-latent (K==V) forward. Combines the
two levers that individually failed:
  * M=16 16x16x32 QK -> [16,512] fp32 acc = 128 VGPR/lane -> occupancy 2 (the
    ``dsa_fwd_m16_kernel`` win), AND
  * ``ds_read_tr16_b64`` hardware-transpose PV read from a ROW-MAJOR V tile (the
    ``dsa_fwd_pipe_kernel`` M=32 win) — instead of ``dsa_fwd_m16_kernel``'s 128
    scalar-store kv^T restaging, which was why M=16 regressed (162<197 on real
    structured topk).

The gathered kv tile (already in registers from the QK A-operand loads) is stored
ROW-MAJOR as V[key][d] with a single vec8 store per chunk (vs 8 scalar stores),
then read back transposed via ds_read_tr for the AV MFMA. The transpose delivers
the 16x16x32 A-operand A[d=L%16, key=(L//16)*8+e] directly, no cross-lane shuffle.

Derivation of the per-lane ds_read_tr address (row-major V[key][d], stride
V_STRIDE): the HW 4x4 transpose gives result[L][e]=Input[src][L%4] with
src=(L//16)*16 + e*4 + (L%16)//4. Solving result[L][e]=V[key=(L//16)*8+e][d=L%16]
=> lane L addresses k_row=(L//16)*8 + (L%16)//4, d_col=dt*16 + (L%4)*4; a second
read at +4 keys + vec4 shuffle yields the full 8-wide (32-key) A-operand.

Contract (same kernel-pair API as gluon_v2 / triton_v2, single MQA latent):
  q    : [T, H, D_QK] bf16   (D_QK = kv_lora_rank + rope pad; rope skipped)
  kv   : [num_kv, 1, D_QK] bf16  (V == K[:kv_lora_rank]; flat latent ++ pool)
  topk : [T, TOPK] int32    (flat window ++ pool indices into kv; -1 = invalid)
  sink : [H] fp32           (optional per-head softmax sink)
  -> o : [T, H, kv_lora_rank] bf16 ; lse : [T, H] fp32 (sink-inclusive)

Grid: (T, cdiv(H, 16)). One wave (64 lanes) per (token, head-block-of-16).

MFMA v_mfma_f32_16x16x32 layouts (lane L, group g = L//16):
  A/B pack (vec8 bf16): lane L holds row/col L%16 and K-subgroup g*8 + 0..7.
  C acc (vec4 f32):     lane L holds C[g*4 + r, L%16] for r in 0..3.

QK: score[key, head] = kv[key, d] @ q[head, d]^T  (contract d, K=32 chunks).
  BLOCK_K=32 keys split into 2 sub-tiles s=0,1; C-layout key = s*16 + g*4 + r,
  head = L%16. Online softmax reduces across the 4 lane groups (shuffle_xor
  16, 32) to a per-head (L%16) running max/sum.
AV: acc[d, head] = V[key, d]^T @ p[key, head]  (contract key). head stays N=L%16
  so no cross-lane head remap. V staged ROW-MAJOR to LDS (vec8 store, reusing the
  QK A-operand loads — no 2nd HBM read); read back transposed via ds_read_tr for
  the A-operand; p transposed C->B-operand via LDS.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LOG2E = math.log2(math.e)
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _lds_fence():
    # s_waitcnt lgkmcnt(0): drain LDS ops for THIS wave (no cross-wave sync).
    # Each wave owns a disjoint LDS region, so a workgroup barrier is unnecessary;
    # a wave-local LDS counter drain is enough to order this wave's stores/loads.
    rocdl.s_waitcnt(0xF | (0x7 << 4) | (0 << 8) | (0x3 << 14))


def _mfma_16x16x32(a, b, c):
    return rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])


def build_dsa_fwd_tr16_module(
    num_heads,
    kv_lora_rank,
    d_qk,
    topk,
    dtype_str="bf16",
    sm_scale=None,
    has_sink=True,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "dsa_fwd_tr16 targets gfx950 (CDNA4)"
    assert dtype_str == "bf16", "bf16 only"

    HEAD_DIM = int(kv_lora_rank)  # D_V = 512
    D_QK = int(d_qk)              # row stride incl rope pad (e.g. 576)
    NUM_HEADS = int(num_heads)
    TOPK = int(topk)
    assert HEAD_DIM % 32 == 0
    BLOCK_H = 16
    # BLOCK_K keys per tile. Larger => fewer tiles => less per-tile softmax/fence
    # overhead (the dominant cost at occupancy 1). Must be a multiple of 16.
    BLOCK_K = int(os.environ.get("PRIMUS_DSA_TR16_BLOCK_K", "32"))
    assert BLOCK_K % 16 == 0
    # QK LDS-load prefetch depth. Higher => more LDS/MFMA overlap. 3 measured best
    # (373->376); AV prefetch depth is insensitive (kept 2).
    _QK_PF = int(os.environ.get("PRIMUS_DSA_TR16_QK_PF", "3"))
    K_CHUNKS = HEAD_DIM // 32   # QK d-contraction chunks (16)
    D_TILES = HEAD_DIM // 16    # AV output d-tiles (32)
    N_SUB = BLOCK_K // 16       # QK key sub-tiles (16 keys each)
    N_AVSUB = BLOCK_K // 32     # AV key sub-blocks (32-key contraction each)
    assert TOPK % BLOCK_K == 0, f"TOPK ({TOPK}) must be a multiple of {BLOCK_K}"
    N_HBLK = (NUM_HEADS + BLOCK_H - 1) // BLOCK_H

    WARP_SIZE = 64
    # Multi-wave workgroup: NUM_WAVES waves, each owning a distinct 16-head sub-block
    # of the SAME token, so the hardware overlaps their MFMA/LDS latency (gluon_v2's
    # 4-warp latency hiding). The kv tile is SHARED across all waves (topk depends
    # only on the token, so every wave reads the same gathered latent rows) — it is
    # gathered ONCE cooperatively per workgroup, eliminating the redundant per-wave
    # HBM gather AND the redundant per-wave QK A-operand HBM loads. Only p (and
    # optionally Q) are per-wave. kv tile is ROW-MAJOR kv[key][d], padded stride to
    # dodge ds_read_tr bank conflicts; the SAME tile feeds the QK A-operand (vec8
    # load) and the PV ds_read_tr transpose read (K==V single latent).
    # DMA (raw_ptr_buffer_load_lds) needs an UNPADDED 16B-aligned stride (each lane's
    # 16B write lands at key*STRIDE + lane*8). Then a (key&3)<<4 swizzle on the GLOBAL
    # fetch col is compensated on BOTH the QK vec8 read and the PV ds_read_tr read
    # (single shared tile serves both). Coop path keeps the +4 pad, no swizzle.
    _USE_DMA = os.environ.get("PRIMUS_DSA_TR16_DMA", "0") == "1"
    V_STRIDE = HEAD_DIM if _USE_DMA else HEAD_DIM + 4
    _LDS_LIMIT = 163840
    # Optionally stage Q in LDS (frees the 64 persistent q_packs VGPR -> occ 2).
    _Q_IN_LDS = os.environ.get("PRIMUS_DSA_TR16_QLDS", "0") == "1"
    Q_STRIDE = HEAD_DIM  # only the D_V cols are read for QK (rope skipped)
    _shared_kv_bytes = BLOCK_K * V_STRIDE * 2
    _per_wave_lds_bytes = (BLOCK_H * BLOCK_K + (BLOCK_H * Q_STRIDE if _Q_IN_LDS else 0)) * 2
    _wv_env = int(os.environ.get("PRIMUS_DSA_M16_WAVES", "0"))
    if _wv_env >= 1:
        NUM_WAVES = _wv_env
    else:
        NUM_WAVES = max(1, min(N_HBLK, (_LDS_LIMIT - _shared_kv_bytes) // _per_wave_lds_bytes))
    while N_HBLK % NUM_WAVES != 0:
        NUM_WAVES -= 1
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    N_WG_HBLK = N_HBLK // NUM_WAVES  # head-block groups along grid.y
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Software pipeline: double-buffer the SHARED kv tile so the async DMA gather of
    # tile t+1 overlaps the compute (QK+softmax+PV) of tile t (gluon's lever). Only
    # engaged with DMA (coop gather is VGPR-staged and can't overlap). NBUF buffers.
    _PIPELINE = os.environ.get("PRIMUS_DSA_TR16_PIPE", "0") == "1"
    NBUF = 2 if _PIPELINE else 1
    # LDS: [ SHARED kv row-major NBUF*(BLOCK_K*V_STRIDE) ] ++ per-wave [ p
    #        (BLOCK_H*BLOCK_K) ++ optional Q (BLOCK_H*Q_STRIDE) ].
    LDS_KV_ONE = BLOCK_K * V_STRIDE
    LDS_KV = NBUF * LDS_KV_ONE
    LDS_P = BLOCK_H * BLOCK_K
    LDS_Q = BLOCK_H * Q_STRIDE if _Q_IN_LDS else 0
    LDS_PER_WAVE = LDS_P + LDS_Q
    LDS_TOTAL = LDS_KV + NUM_WAVES * LDS_PER_WAVE

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"dsa_fwd_tr16_H{NUM_HEADS}_K{TOPK}_W{NUM_WAVES}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dsa_fwd_tr16_kernel(
        Q: fx.Tensor,      # [T, H, D_QK] bf16
        KV: fx.Tensor,     # [num_kv, D_QK] bf16 (flat)
        TopK: fx.Tensor,   # [T, TOPK] int32
        Sink: fx.Tensor,   # [H] fp32
        O: fx.Tensor,      # [T, H, D_V] bf16
        LSE: fx.Tensor,    # [T, H] fp32
        total_tokens: fx.Int32,
    ):
        f16_ty = T.bf16
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), KV)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TopK)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        sink_rsrc = buffer_ops.create_buffer_resource(Sink, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(KV, max_size=True) if _USE_DMA else None

        # DMA-path swizzle: (key&3)<<4 XOR on the d-col, applied to the GLOBAL fetch
        # and compensated identically on the QK read and PV ds_read_tr read (single
        # shared tile). Coop path: no swizzle. (key row offsets +8/+32 keep &3 mask.)
        def _kv_swz(key_idx, col_idx):
            if const_expr(_USE_DMA):
                return col_idx ^ ((key_idx & arith.index(0x3)) << arith.index(4))
            return col_idx

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, f16_ty, shape=(LDS_TOTAL,)).get()

        def _gep(base_p, elem_idx, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(_llvm_ptr_ty(), base_p, [idx_i64],
                               rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                               elem_type=elem_t, noWrapFlags=0).result

        def load_f16_v(base_p, elem_idx, n):
            return _llvm.LoadOp(T.vec(n, f16_ty), _gep(base_p, elem_idx, f16_ty)).result

        def load_i32(base_p, elem_idx):
            return _llvm.LoadOp(T.i32, _gep(base_p, elem_idx, T.i32)).result

        def store_f16(val, base_p, elem_idx):
            _llvm.StoreOp(val, _gep(base_p, elem_idx, f16_ty))

        def lds_store1(val, idx):
            vector.store(vector.from_elements(T.vec(1, f16_ty), [val]), lds, [idx])

        tok = arith.index_cast(T.index, gpu.block_idx.x)
        pid_wg_hblk = arith.index_cast(T.index, gpu.block_idx.y)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        wave_id = tid // arith.index(WARP_SIZE)
        lane = tid % arith.index(WARP_SIZE)

        # SHARED kv tile at LDS offset 0 (all waves read the same gathered latent).
        kv_lds_base = arith.index(0)
        # per-wave p (+ optional Q) region after the shared kv tile.
        wave_lds_base = arith.index(LDS_KV) + wave_id * arith.index(LDS_PER_WAVE)
        c_lds_p = wave_lds_base
        c_lds_q = wave_lds_base + arith.index(LDS_P)  # Q region (if _Q_IN_LDS)

        tt_v = arith.index_cast(T.index, total_tokens)
        tok_active = arith.cmpi(arith.CmpIPredicate.slt, tok, tt_v)
        tok_safe = arith.select(tok_active, tok, arith.index(0))

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane // arith.index(16)
        lane_mod_16_i32 = arith.index_cast(T.i32, lane_mod_16)
        lane_div_16_i32 = arith.index_cast(T.i32, lane_div_16)
        # This wave's 16-head block: (grid.y group) * NUM_WAVES + wave_id.
        h_base = (pid_wg_hblk * arith.index(NUM_WAVES) + wave_id) * arith.index(BLOCK_H)

        c_neg_inf = arith.constant(-1.0e30, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_one_f = arith.constant(1.0, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        c_log2e_f = arith.constant(_LOG2E, type=f32_ty)
        zero_pack = arith.constant_vector(0.0, T.vec(8, f16_ty))
        c_zero_acc = arith.constant_vector(0.0, T.vec(4, f32_ty))
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)

        NH = arith.index(NUM_HEADS)
        DQK = arith.index(D_QK)
        HD = arith.index(HEAD_DIM)
        TOPK_I = arith.index(TOPK)
        TOPK_i32 = arith.constant(TOPK, type=T.i32)

        # Q B-operand packs: lane L -> head (h_base + L%16), d chunk = ck*32 + g*8.
        my_head = h_base + lane_mod_16
        head_ib = arith.cmpi(arith.CmpIPredicate.slt, my_head, NH)
        head_safe = arith.select(head_ib, my_head, arith.index(0))
        q_row_base = (tok_safe * NH + head_safe) * DQK
        q_valid = arith.AndIOp(head_ib, tok_active).result
        if const_expr(_Q_IN_LDS):
            # Stage Q row-major in LDS once (Q[head=L%16][d]); read transient packs
            # per tile inside the k-loop. Frees the 64 persistent q_packs VGPR.
            for ck in range_constexpr(K_CHUNKS):
                q_off = q_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                qp = load_f16_v(q_ptr, q_off, 8)
                qp = arith.select(q_valid, qp, zero_pack)
                q_lds_idx = c_lds_q + lane_mod_16 * arith.index(Q_STRIDE) + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                vector.store(qp, lds, [q_lds_idx])
            _lds_fence()
            q_packs = None
            def _q_pack(ck):
                q_lds_idx = c_lds_q + lane_mod_16 * arith.index(Q_STRIDE) + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                return vector.load_op(T.vec(8, f16_ty), lds, [q_lds_idx])
        else:
            q_packs = []
            for ck in range_constexpr(K_CHUNKS):
                q_off = q_row_base + arith.index(ck * 32) + lane_div_16 * arith.index(8)
                qp = load_f16_v(q_ptr, q_off, 8)
                q_packs.append(arith.select(q_valid, qp, zero_pack))
            def _q_pack(ck):
                return q_packs[ck]

        def group_reduce_max(v):
            cur = v
            for off in (16, 32):
                peer = arith.ArithValue(cur).shuffle_xor(arith.constant(off, type=T.i32), width_i32)
                cur = arith.MaxNumFOp(cur, peer, fastmath=fm_fast).result
            return cur

        def group_reduce_add(v):
            cur = v
            for off in (16, 32):
                peer = arith.ArithValue(cur).shuffle_xor(arith.constant(off, type=T.i32), width_i32)
                cur = arith.AddFOp(cur, peer, fastmath=fm_fast).result
            return cur

        # topk[tok, kk] -> (invalid, kv_row_base). kv index in [0, num_kv), -1 invalid.
        def key_meta(kk_i32):
            ko = arith.cmpi(arith.CmpIPredicate.sge, kk_i32, TOPK_i32)
            kk_idx = arith.index_cast(T.index, arith.select(ko, arith.constant(0, type=T.i32), kk_i32))
            toff = tok_safe * TOPK_I + kk_idx
            idr = load_i32(topk_ptr, toff)
            ineg = arith.cmpi(arith.CmpIPredicate.slt, idr, arith.constant(0, type=T.i32))
            inv = arith.OrIOp(ko, ineg).result
            kv_row = arith.index_cast(T.index, arith.select(inv, arith.constant(0, type=T.i32), idr))
            return inv, kv_row * DQK

        # ---- factored tile primitives (buf = kv LDS buffer element base) ----
        _COLG = HEAD_DIM // 8  # vec8 groups per key row (64)
        _GTOT = BLOCK_K * _COLG
        GATHER_ITERS = (_GTOT + BLOCK_SIZE - 1) // BLOCK_SIZE
        _AV_PF = int(os.environ.get("PRIMUS_DSA_TR16_AV_PF", "2"))
        g_koff = lane_div_16 * arith.index(8)
        tr_k_group = (lane % arith.index(16)) // arith.index(4)
        tr_col_sub = lane % arith.index(4)
        v_krow = lane_div_16 * arith.index(8) + tr_k_group

        def _buf_base(buf):
            if isinstance(buf, int):
                return arith.index(buf * LDS_KV_ONE)
            return buf * arith.index(LDS_KV_ONE)

        def _ds_read_tr_v4(elem_idx):
            byte = elem_idx * arith.index(2) + arith.index(lds_off)
            byte_i64 = arith.index_cast(T.i64, byte)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(T.vec(4, f16_ty), ptr).result

        # ---- SHARED gather of tile k_start's kv into LDS buffer `buf`. Row-major
        # kv[key][d]; all BLOCK_SIZE threads cooperate; -1 topk rows zeroed/clamped.
        def _gather(buf, k_start_i32):
            bb = _buf_base(buf)
            if const_expr(_USE_DMA):
                # M=32-style DMA: lds_ptr is UNIFORM per wave (readfirstlane); the HW
                # writes each lane's 16B implicitly to lds_ptr + lane*16B. Each
                # wave-batch DMAs one full key row (64 lanes x 8 elems = 512 =
                # HEAD_DIM = V_STRIDE). Waves split the BLOCK_K key rows.
                # V_STRIDE == HEAD_DIM (unpadded) is required for the implicit write
                # to land at key*V_STRIDE + lane*8.
                assert V_STRIDE == HEAD_DIM, "DMA needs unpadded V_STRIDE"
                _LPR = HEAD_DIM * 2 // 16  # lanes per key row = 64
                assert _LPR == WARP_SIZE, "DMA assumes 64 lanes == one key row"
                assert BLOCK_K % NUM_WAVES == 0
                _NDMA = BLOCK_K // NUM_WAVES  # key rows each wave DMAs
                _dma_size = arith.constant(16, type=T.i32)
                _dma_z = arith.constant(0, type=T.i32)
                _dma_aux = arith.constant(1, type=T.i32)
                lds_base_byte = buffer_ops.extract_base_index(lds, address_space=3)
                # this lane's column within a key row (0..63) -> 8 elems each
                swiz_col = lane * arith.index(8)
                for dki in range_constexpr(_NDMA):
                    key_in_tile = wave_id * arith.index(_NDMA) + arith.index(dki)
                    # uniform per-wave LDS dest: base + key_row*V_STRIDE; lane*16B added by HW
                    v_row_elem = bb + key_in_tile * arith.index(V_STRIDE)
                    lds_addr = lds_base_byte + (arith.index(lds_off) + v_row_elem) * arith.index(2)
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_lane0).result
                    # per-lane global source: swizzle the fetched column (round-trips
                    # with the QK/PV read swizzle). key row from topk (clamp -1 -> 0).
                    unsw_col = _kv_swz(key_in_tile, swiz_col)
                    key_pos_i32 = arith.AddIOp(k_start_i32, arith.index_cast(T.i32, key_in_tile)).result
                    _inv_g, kv_row_base = key_meta(key_pos_i32)
                    global_elem = kv_row_base + unsw_col
                    voffset = arith.index_cast(T.i32, global_elem * arith.index(2))
                    rocdl.raw_ptr_buffer_load_lds(kv_rsrc, lds_ptr, _dma_size, voffset, _dma_z, _dma_z, _dma_aux)
            else:
                for gi in range_constexpr(GATHER_ITERS):
                    slot = tid + arith.index(gi * BLOCK_SIZE)
                    in_range = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.index(_GTOT))
                    key_in_tile = slot // arith.index(_COLG)
                    col8 = (slot % arith.index(_COLG)) * arith.index(8)
                    key_pos_i32 = arith.AddIOp(k_start_i32, arith.index_cast(T.i32, key_in_tile)).result
                    _inv_g, kv_row_base = key_meta(key_pos_i32)
                    koff = kv_row_base + col8
                    vec = load_f16_v(kv_ptr, koff, 8)
                    vec = arith.select(in_range, vec, zero_pack)
                    v_idx = bb + key_in_tile * arith.index(V_STRIDE) + col8
                    _if_g = scf.IfOp(in_range, [], has_else=False)
                    with ir.InsertionPoint(_if_g.then_block):
                        vector.store(vec, lds, [v_idx])
                        scf.YieldOp([])

        # ---- QK: read kv from buffer `buf`, MFMA against Q, return scaled/masked
        # scores (N_SUB x 4 per-lane fp32). This is the MATRIX phase to overlap.
        def _qk(buf, k_start_i32):
            bb = _buf_base(buf)
            owned = []
            for s in range_constexpr(N_SUB):
                kloc_base = arith.index(s * 16) + lane_mod_16
                def _load_a(ck):
                    base_col = arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    col = _kv_swz(kloc_base, base_col) if const_expr(_USE_DMA) else base_col
                    a_idx = bb + kloc_base * arith.index(V_STRIDE) + col
                    return vector.load_op(T.vec(8, f16_ty), lds, [a_idx])
                a_packs = [None] * K_CHUNKS
                for p in range_constexpr(_QK_PF):
                    a_packs[p] = _load_a(p)
                c_acc = c_zero_acc
                for ck in range_constexpr(K_CHUNKS):
                    a_pack = a_packs[ck]
                    if const_expr(ck + _QK_PF < K_CHUNKS):
                        a_packs[ck + _QK_PF] = _load_a(ck + _QK_PF)
                    c_acc = _mfma_16x16x32(a_pack, _q_pack(ck), c_acc)
                c_regs = [vector.extract(c_acc, static_position=[r], dynamic_position=[]) for r in range_constexpr(4)]
                s_row = []
                for r in range_constexpr(4):
                    krow_i32 = arith.AddIOp(
                        arith.AddIOp(k_start_i32, arith.constant(s * 16, type=T.i32)).result,
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    inv_r, _pr = key_meta(krow_i32)
                    sc = arith.MulFOp(c_regs[r], c_sm_scale_f, fastmath=fm_fast).result
                    sc = arith.select(inv_r, c_neg_inf, sc)
                    s_row.append(sc)
                owned.append(s_row)
            return owned

        # ---- softmax(owned_scores) + PV (reads kv from buffer `buf`). Returns
        # updated (m_i, l_i, acc). VALU (softmax) + MATRIX (PV) phase.
        def _softmax_pv(owned_scores, buf, m_i, l_i, acc):
            bb = _buf_base(buf)
            m_tile_local = owned_scores[0][0]
            for s in range_constexpr(N_SUB):
                for r in range_constexpr(4):
                    if const_expr(s == 0 and r == 0):
                        continue
                    m_tile_local = arith.MaxNumFOp(m_tile_local, owned_scores[s][r], fastmath=fm_fast).result
            m_tile = group_reduce_max(m_tile_local)
            m_new = arith.MaxNumFOp(m_i, m_tile, fastmath=fm_fast).result
            alpha = arith.ArithValue(arith.MulFOp(arith.SubFOp(m_i, m_new, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)

            p_owned = []
            l_tile_local = c_zero_f
            for s in range_constexpr(N_SUB):
                p_row = []
                for r in range_constexpr(4):
                    pe = arith.ArithValue(arith.MulFOp(arith.SubFOp(owned_scores[s][r], m_new, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)
                    p_row.append(pe)
                    l_tile_local = arith.AddFOp(l_tile_local, pe, fastmath=fm_fast).result
                p_owned.append(p_row)
            l_tile = group_reduce_add(l_tile_local)
            l_i = arith.AddFOp(arith.MulFOp(l_i, alpha, fastmath=fm_fast).result, l_tile, fastmath=fm_fast).result
            m_i = m_new

            for s in range_constexpr(N_SUB):
                for r in range_constexpr(4):
                    kloc = arith.AddIOp(
                        arith.constant(s * 16, type=T.i32),
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    kloc_idx = arith.index_cast(T.index, kloc)
                    p_f16 = arith.trunc_f(f16_ty, p_owned[s][r])
                    lds_pidx = c_lds_p + lane_mod_16 * arith.index(BLOCK_K) + kloc_idx
                    lds_store1(p_f16, lds_pidx)
            _lds_fence()

            alpha_vec = vector.broadcast(T.vec(4, f32_ty), alpha)

            def _b_pack(avs):
                b_pidx = c_lds_p + lane_mod_16 * arith.index(BLOCK_K) + arith.index(avs * 32) + g_koff
                return vector.load_op(T.vec(8, f16_ty), lds, [b_pidx])

            def _av_a(avs, dt):
                k_row_a = arith.index(avs * 32) + v_krow
                d_col = arith.index(dt * 16) + tr_col_sub * arith.index(4)
                if const_expr(_USE_DMA):
                    k_row_b = k_row_a + arith.index(4)
                    base_a = bb + k_row_a * arith.index(V_STRIDE) + _kv_swz(k_row_a, d_col)
                    base_b = bb + k_row_b * arith.index(V_STRIDE) + _kv_swz(k_row_b, d_col)
                    va = _ds_read_tr_v4(base_a)
                    vb = _ds_read_tr_v4(base_b)
                else:
                    base = bb + k_row_a * arith.index(V_STRIDE) + d_col
                    va = _ds_read_tr_v4(base)
                    vb = _ds_read_tr_v4(base + arith.index(4 * V_STRIDE))
                return vector.shuffle(va, vb, [0, 1, 2, 3, 4, 5, 6, 7])

            acc = [arith.MulFOp(acc[dt], alpha_vec, fastmath=fm_fast).result for dt in range_constexpr(D_TILES)]
            for avs in range_constexpr(N_AVSUB):
                b_pack = _b_pack(avs)
                av_pf = [None] * D_TILES
                for p in range_constexpr(_AV_PF):
                    av_pf[p] = _av_a(avs, p)
                new_acc = []
                for dt in range_constexpr(D_TILES):
                    new_acc.append(_mfma_16x16x32(av_pf[dt], b_pack, acc[dt]))
                    if const_expr(dt + _AV_PF < D_TILES):
                        av_pf[dt + _AV_PF] = _av_a(avs, dt + _AV_PF)
                acc = new_acc
            return m_i, l_i, acc

        m_i = c_neg_inf
        l_i = c_zero_f
        acc = [c_zero_acc for _ in range_constexpr(D_TILES)]

        if const_expr(not _PIPELINE):
            # ---- non-pipelined driver (default): gather; barrier; QK; softmax+PV.
            for k_start, carry, results in scf.for_(
                arith.index(0), TOPK_I, arith.index(BLOCK_K), iter_args=[m_i, l_i] + acc,
            ):
                m_i = carry[0]
                l_i = carry[1]
                acc = [carry[2 + d] for d in range_constexpr(D_TILES)]
                k_start_i32 = arith.index_cast(T.i32, k_start)
                gpu.barrier()  # prior tile's readers done before overwrite
                _gather(0, k_start_i32)
                if const_expr(_USE_DMA):
                    rocdl.s_waitcnt(0)  # drain async DMA before the tile is read
                gpu.barrier()  # full shared tile visible before QK
                owned = _qk(0, k_start_i32)
                m_i, l_i, acc = _softmax_pv(owned, 0, m_i, l_i, acc)
                yield [m_i, l_i] + acc
            m_i = results[0]
            l_i = results[1]
            acc = [results[2 + d] for d in range_constexpr(D_TILES)]
        else:
            # ---- software-pipelined driver: overlap softmax(t) VALU + QK(t+1)
            # MFMA. 2 kv buffers (parity by tile). PV of tile t reads buf[t%2] via
            # ds_read_tr, so tile t+1's gather goes into buf[(t+1)%2] (disjoint).
            # Prologue: gather(0)->buf0; QK(0)->prev.
            _k0 = arith.constant(0, type=T.i32)
            _gather(0, _k0)
            gpu.barrier()
            prev_scores = _qk(0, _k0)
            prev_k = _k0
            # main loop t = 0 .. NUM_TILES-2: gather(t+1)->buf[(t+1)%2];
            #   QK(t+1)->cur; softmax+PV(prev=t) reads buf[t%2]; promote.
            NUM_TILES = TOPK // BLOCK_K
            flat_prev = [prev_scores[s][r] for s in range(N_SUB) for r in range(4)]
            for t, carry, results in scf.for_(
                arith.index(0), arith.index(NUM_TILES - 1), arith.index(1),
                iter_args=[m_i, l_i] + acc + flat_prev + [arith.index_cast(T.index, prev_k)],
            ):
                m_i = carry[0]
                l_i = carry[1]
                acc = [carry[2 + d] for d in range_constexpr(D_TILES)]
                _pf = carry[2 + D_TILES: 2 + D_TILES + N_SUB * 4]
                prev_scores = [[_pf[s * 4 + r] for r in range(4)] for s in range(N_SUB)]
                prev_k_i32 = arith.index_cast(T.i32, carry[2 + D_TILES + N_SUB * 4])
                cur_buf = (t + arith.index(1)) % arith.index(2)
                prev_buf = t % arith.index(2)
                cur_k_i32 = arith.index_cast(T.i32, (t + arith.index(1)) * arith.index(BLOCK_K))
                # Barrier: prior iter's PV finished reading cur_buf (it was prev_buf two
                # iters ago) before this gather overwrites it; also makes the prologue/
                # previous gather visible. Single barrier per iter (vs 2 non-pipe).
                gpu.barrier()
                _gather(cur_buf, cur_k_i32)
                # QK(t+1) MFMA reads cur_buf (just gathered) -- needs gather visible, but
                # softmax_pv(prev) reads prev_buf (disjoint, already visible). Emit
                # softmax(prev) VALU FIRST so it overlaps the gather's LDS writes, THEN a
                # barrier, THEN QK(cur). PV(prev) also reads prev_buf (safe, disjoint).
                m_i, l_i, acc = _softmax_pv(prev_scores, prev_buf, m_i, l_i, acc)
                if const_expr(_USE_DMA):
                    rocdl.s_waitcnt(0)  # drain the async gather (overlapped w/ softmax_pv above)
                gpu.barrier()  # cur_buf gather visible before QK reads it
                cur_scores = _qk(cur_buf, cur_k_i32)
                flat_cur = [cur_scores[s][r] for s in range(N_SUB) for r in range(4)]
                yield [m_i, l_i] + acc + flat_cur + [(t + arith.index(1)) * arith.index(BLOCK_K)]
            m_i = results[0]
            l_i = results[1]
            acc = [results[2 + d] for d in range_constexpr(D_TILES)]
            _pf = results[2 + D_TILES: 2 + D_TILES + N_SUB * 4]
            last_scores = [[_pf[s * 4 + r] for r in range(4)] for s in range(N_SUB)]
            last_buf = (results[2 + D_TILES + N_SUB * 4] // arith.index(BLOCK_K)) % arith.index(2)
            # epilogue: softmax+PV of the last tile.
            m_i, l_i, acc = _softmax_pv(last_scores, last_buf, m_i, l_i, acc)

        # ---- sink fold (scaled domain) + normalize + store ----
        # NOTE: owned_scores are already scaled (c_regs * sm_scale), so m_i is in
        # the scaled domain — do NOT multiply by sm_scale again here.
        head_active = arith.AndIOp(head_ib, tok_active).result
        M_scaled = m_i
        head_i32 = arith.index_cast(T.i32, head_safe)
        sink_val = buffer_ops.buffer_load(sink_rsrc, head_i32, vec_width=1, dtype=f32_ty)
        m_fin = arith.MaxNumFOp(M_scaled, sink_val, fastmath=fm_fast).result
        af = arith.ArithValue(arith.MulFOp(arith.SubFOp(M_scaled, m_fin, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)
        l_af = arith.MulFOp(l_i, af, fastmath=fm_fast).result
        sink_e = arith.ArithValue(arith.MulFOp(arith.SubFOp(sink_val, m_fin, fastmath=fm_fast).result, c_log2e_f, fastmath=fm_fast).result).exp2(fastmath=fm_fast)
        l_total = arith.AddFOp(l_af, sink_e, fastmath=fm_fast).result
        empty = arith.cmpf(arith.CmpFPredicate.OLE, l_total, c_zero_f)
        safe_l = arith.select(empty, c_one_f, l_total)
        ln_l = math_dialect.log(safe_l, fastmath=fm_fast)
        lse_val = arith.AddFOp(m_fin, ln_l, fastmath=fm_fast).result
        inv_l = arith.DivFOp(c_one_f, safe_l, fastmath=fm_fast).result
        acc_scale = arith.MulFOp(af, inv_l, fastmath=fm_fast).result  # af/l_total

        _if = scf.IfOp(head_active, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            o_row_base = (tok_safe * NH + my_head) * HD
            for dt in range_constexpr(D_TILES):
                for r in range_constexpr(4):
                    ov = vector.extract(acc[dt], static_position=[r], dynamic_position=[])
                    ov = arith.MulFOp(ov, acc_scale, fastmath=fm_fast).result
                    of16 = arith.trunc_f(f16_ty, ov)
                    d = arith.index(dt * 16) + lane_div_16 * arith.index(4) + arith.index(r)
                    store_f16(of16, o_ptr, o_row_base + d)
            is_owner = arith.cmpi(arith.CmpIPredicate.eq, lane_div_16, arith.index(0))
            _if2 = scf.IfOp(is_owner, [], has_else=False)
            with ir.InsertionPoint(_if2.then_block):
                lse_off = tok_safe * NH + my_head
                lse_off_i32 = arith.index_cast(T.i32, lse_off)
                buffer_ops.buffer_store(lse_val, lse_rsrc, lse_off_i32)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_dsa_fwd_tr16(
        Q: fx.Tensor,
        KV: fx.Tensor,
        TopK: fx.Tensor,
        Sink: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        tt_idx = arith.index_cast(T.index, total_tokens)
        launcher = dsa_fwd_tr16_kernel(Q, KV, TopK, Sink, O, LSE, total_tokens)
        if const_expr(waves_per_eu is not None):
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
        passthrough_entries = []
        # Force occupancy 2 by capping architectural VGPRs: the [16,512] fp32 acc is
        # only 128 VGPR, and the AGPR pool is nearly idle (14/256), so the allocator
        # can stage the long-lived acc in AGPR and free arch VGPR below 256/2=... The
        # M=32 kernel needs this same knob. Env-gated for A/B.
        _amdgpu_wpe = os.environ.get("PRIMUS_DSA_TR16_AMDGPU_WPE", "")
        if _amdgpu_wpe:
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("amdgpu-waves-per-eu"), ir.StringAttr.get(_amdgpu_wpe)]))
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
        launcher.launch(grid=(tt_idx, arith.index(N_WG_HBLK), arith.index(1)), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math,
              "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True}}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_dsa_fwd_tr16(*args, **kwargs)

    return _launch
