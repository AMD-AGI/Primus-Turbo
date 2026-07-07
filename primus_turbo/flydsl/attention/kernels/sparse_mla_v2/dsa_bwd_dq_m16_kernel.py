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


def build_dsa_bwd_dq_m16_module(
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
    assert gpu_arch.startswith("gfx950"), "dsa_bwd_dq_m16 targets gfx950 (CDNA4)"
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
    # QK HBM-load prefetch depth. Lower => fewer transient VGPR (better occupancy),
    # higher => more VMEM/MFMA overlap. 2 is the M=32 kernel's sweet spot.
    _QK_PF = int(os.environ.get("PRIMUS_DSA_BWD_DQ_QK_PF", "1"))
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
    # Adaptive occupancy: when a workgroup already holds many waves (e.g. H64 packs
    # 4 head-blocks into one CTA), forcing waves_per_eu=2 makes the scheduler try to
    # co-resident 2 CTAs (8 waves) per EU, which thrashes the 256-VGPR dQ acc/spill.
    # wpe=1 lets each many-wave CTA own the EU alone -> H64 dQ 3.8->2.7ms. Few-wave
    # CTAs (H128 -> 2 waves) still benefit from wpe=2. Env override wins.
    _wpe_env = os.environ.get("PRIMUS_DSA_BWD_DQ_WPE", "")
    if _wpe_env:
        waves_per_eu = int(_wpe_env)
    elif NUM_WAVES >= 4:
        waves_per_eu = 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # LDS: [ SHARED kv row-major (BLOCK_K*V_STRIDE) ] ++ per-wave [ p (BLOCK_H*BLOCK_K)
    #        ++ optional Q (BLOCK_H*Q_STRIDE) ].
    LDS_KV = BLOCK_K * V_STRIDE
    LDS_P = BLOCK_H * BLOCK_K
    LDS_Q = BLOCK_H * Q_STRIDE if _Q_IN_LDS else 0
    LDS_PER_WAVE = LDS_P + LDS_Q
    LDS_TOTAL = LDS_KV + NUM_WAVES * LDS_PER_WAVE

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"dsa_fwd_tr16_H{NUM_HEADS}_K{TOPK}_W{NUM_WAVES}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_TOTAL * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dsa_bwd_dq_kernel(
        Q: fx.Tensor,      # [T, H, D_QK] bf16
        KV: fx.Tensor,     # [num_kv, D_QK] bf16 (flat)
        dO: fx.Tensor,     # [T, H, D_V] bf16
        TopK: fx.Tensor,   # [T, TOPK] int32
        LSE: fx.Tensor,    # [T, H] fp32 (sink-inclusive)
        Delta: fx.Tensor,  # [T, H] fp32 (rowsum(O*dO))
        dQ: fx.Tensor,     # [T, H, D_QK] bf16 (out; rope cols stay 0)
        dS: fx.Tensor,     # [T, H, TOPK] bf16 (out, for dKV kernel)
        Pout: fx.Tensor,   # [T, H, TOPK] bf16 (out, for dKV kernel)
        total_tokens: fx.Int32,
    ):
        f16_ty = T.bf16
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), KV)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dO)
        topk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TopK)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dQ)
        ds_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dS)
        pout_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Pout)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(Delta, max_size=True)
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

        def store_f16_v(val_vec, base_p, elem_idx):
            _llvm.StoreOp(val_vec, _gep(base_p, elem_idx, f16_ty))

        def lds_store1(val, idx):
            vector.store(vector.from_elements(T.vec(1, f16_ty), [val]), lds, [idx])

        def lds_store_v4(val_vec, idx):
            vector.store(val_vec, lds, [idx])

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

        # Q + dO B-operand packs: lane L -> head (h_base + L%16), d chunk = ck*32+g*8.
        # (dQ bwd: S=Q.K^T and dP=dO.K^T both contract d; dO is a 2nd stationary op.)
        my_head = h_base + lane_mod_16
        head_ib = arith.cmpi(arith.CmpIPredicate.slt, my_head, NH)
        head_safe = arith.select(head_ib, my_head, arith.index(0))
        q_row_base = (tok_safe * NH + head_safe) * DQK
        o_row_base = (tok_safe * NH + head_safe) * HD  # dO is [T,H,D_V] (no rope)
        q_valid = arith.AndIOp(head_ib, tok_active).result
        q_packs = []
        do_packs = []
        for ck in range_constexpr(K_CHUNKS):
            col_g = arith.index(ck * 32) + lane_div_16 * arith.index(8)
            qp = load_f16_v(q_ptr, q_row_base + col_g, 8)
            q_packs.append(arith.select(q_valid, qp, zero_pack))
            dp = load_f16_v(do_ptr, o_row_base + col_g, 8)
            do_packs.append(arith.select(q_valid, dp, zero_pack))
        def _q_pack(ck):
            return q_packs[ck]
        def _do_pack(ck):
            return do_packs[ck]

        # per-head lse (sink-inclusive) + delta; both scaled to log2 domain for P.
        head_flat_i32 = arith.index_cast(T.i32, tok_safe * NH + head_safe)
        lse_val = buffer_ops.buffer_load(lse_rsrc, head_flat_i32, vec_width=1, dtype=f32_ty)
        delta_val = buffer_ops.buffer_load(delta_rsrc, head_flat_i32, vec_width=1, dtype=f32_ty)
        c_sm_scale_log2e = arith.constant(float(sm_scale) * _LOG2E, type=f32_ty)
        neg_lse_log2e = arith.MulFOp(lse_val, arith.SubFOp(c_zero_f, c_log2e_f, fastmath=fm_fast).result, fastmath=fm_fast).result

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

        # dQ accumulators only (no online softmax: lse/delta are precomputed).
        acc = [c_zero_acc for _ in range_constexpr(D_TILES)]

        for k_start, carry, results in scf.for_(
            arith.index(0), TOPK_I, arith.index(BLOCK_K), iter_args=list(acc),
        ):
            acc = [carry[d] for d in range_constexpr(D_TILES)]
            k_start_i32 = arith.index_cast(T.i32, k_start)

            # ---- SHARED gather of this tile's kv into LDS (once per workgroup).
            # Row-major kv[key][d]; all BLOCK_SIZE threads cooperate. Unit of work =
            # one vec8 (8 d-cols) of one key row. -1 topk rows zeroed (masked).
            _COLG = HEAD_DIM // 8  # vec8 groups per key row (64)
            _GTOT = BLOCK_K * _COLG
            GATHER_ITERS = (_GTOT + BLOCK_SIZE - 1) // BLOCK_SIZE
            gpu.barrier()  # ensure prior tile's readers are done before overwrite
            if const_expr(_USE_DMA):
                # Async global->LDS DMA (raw_ptr_buffer_load_lds), bypassing VGPRs.
                # Each thread fires a 16B (vec8) load; LDS dest is implicit per-lane
                # (base + lane*16B). Swizzle (key&3)<<4 on the GLOBAL fetch col,
                # compensated on the QK + PV reads. Unpadded V_STRIDE==HEAD_DIM.
                # Branchless (GATHER_ITERS is exact: _GTOT % BLOCK_SIZE == 0).
                # Invalid (-1) topk rows clamp to global row 0; harmless because the
                # additive -inf validity mask zeroes those columns in softmax.
                assert _GTOT % BLOCK_SIZE == 0, "DMA path needs _GTOT divisible by BLOCK_SIZE"
                _dma_size = arith.constant(16, type=T.i32)
                _dma_z = arith.constant(0, type=T.i32)
                _dma_aux = arith.constant(1, type=T.i32)
                lds_base_byte = buffer_ops.extract_base_index(lds, address_space=3)
                for gi in range_constexpr(GATHER_ITERS):
                    slot = tid + arith.index(gi * BLOCK_SIZE)
                    key_in_tile = slot // arith.index(_COLG)
                    col8 = (slot % arith.index(_COLG)) * arith.index(8)
                    swz_col = _kv_swz(key_in_tile, col8)
                    key_pos_i32 = arith.AddIOp(k_start_i32, arith.index_cast(T.i32, key_in_tile)).result
                    _inv_g, kv_row_base = key_meta(key_pos_i32)  # already clamps -1 -> row 0
                    v_idx = kv_lds_base + key_in_tile * arith.index(V_STRIDE) + col8
                    lds_addr = lds_base_byte + (arith.index(lds_off) + v_idx) * arith.index(2)
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_i64).result
                    global_elem = kv_row_base + swz_col
                    voffset = arith.index_cast(T.i32, global_elem * arith.index(2))
                    rocdl.raw_ptr_buffer_load_lds(kv_rsrc, lds_ptr, _dma_size, voffset, _dma_z, _dma_z, _dma_aux)
                rocdl.s_waitcnt(0)  # drain the async DMA before LDS reads
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
                    v_idx = kv_lds_base + key_in_tile * arith.index(V_STRIDE) + col8
                    _if_g = scf.IfOp(in_range, [], has_else=False)
                    with ir.InsertionPoint(_if_g.then_block):
                        vector.store(vec, lds, [v_idx])
                        scf.YieldOp([])
            gpu.barrier()  # all waves see the full shared tile before QK reads

            # ==== GEMM1: S = Q.K^T AND dP = dO.K^T (share kv A-operand). ====
            ds_owned = []  # per (s, r): dS = P*(dP-delta)*scale, for the dQ AV below
            for s in range_constexpr(N_SUB):
                kloc_base = arith.index(s * 16) + lane_mod_16
                # kv A-operand from SHARED LDS tile: lane L -> key=s*16+L%16, d=ck*32+(L//16)*8+e.
                def _load_a(ck):
                    base_col = arith.index(ck * 32) + lane_div_16 * arith.index(8)
                    col = _kv_swz(kloc_base, base_col) if const_expr(_USE_DMA) else base_col
                    a_idx = kv_lds_base + kloc_base * arith.index(V_STRIDE) + col
                    return vector.load_op(T.vec(8, f16_ty), lds, [a_idx])
                a_packs = [None] * K_CHUNKS
                for p in range_constexpr(_QK_PF):
                    a_packs[p] = _load_a(p)
                s_acc = c_zero_acc
                dp_acc = c_zero_acc
                for ck in range_constexpr(K_CHUNKS):
                    a_pack = a_packs[ck]
                    if const_expr(ck + _QK_PF < K_CHUNKS):
                        a_packs[ck + _QK_PF] = _load_a(ck + _QK_PF)
                    s_acc = _mfma_16x16x32(a_pack, _q_pack(ck), s_acc)
                    dp_acc = _mfma_16x16x32(a_pack, _do_pack(ck), dp_acc)
                s_regs = [vector.extract(s_acc, static_position=[r], dynamic_position=[]) for r in range_constexpr(4)]
                dp_regs = [vector.extract(dp_acc, static_position=[r], dynamic_position=[]) for r in range_constexpr(4)]
                ds_row = []
                ds16_row = []
                p16_row = []
                for r in range_constexpr(4):
                    # this lane/r owns key = s*16 + (L//16)*4 + r; head = L%16
                    kloc_i32 = arith.AddIOp(
                        arith.constant(s * 16, type=T.i32),
                        arith.AddIOp(arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result,
                                     arith.constant(r, type=T.i32)).result).result
                    krow_i32 = arith.AddIOp(k_start_i32, kloc_i32).result
                    inv_r, _pr = key_meta(krow_i32)
                    # P = exp2(scale*log2e*S - lse*log2e); invalid key -> P=0
                    p_arg = math_dialect.fma(s_regs[r], c_sm_scale_log2e, neg_lse_log2e)
                    p_r = arith.ArithValue(p_arg).exp2(fastmath=fm_fast)
                    p_r = arith.select(inv_r, c_zero_f, p_r)
                    # dS = P*(dP - delta)*scale
                    dp_md = arith.SubFOp(dp_regs[r], delta_val, fastmath=fm_fast).result
                    ds_r = arith.MulFOp(arith.MulFOp(p_r, dp_md, fastmath=fm_fast).result, c_sm_scale_f, fastmath=fm_fast).result
                    ds_row.append(ds_r)
                    ds16_row.append(arith.trunc_f(f16_ty, ds_r))
                    p16_row.append(arith.trunc_f(f16_ty, p_r))
                ds_owned.append(ds_row)
                # The 4 keys r=0..3 are CONTIGUOUS (key = ...+r) in HBM [T,H,TOPK] and
                # LDS p-region -> coalesce the 8 scalar stores into vec4 stores.
                ds_vec = vector.from_elements(T.vec(4, f16_ty), ds16_row)
                p_vec = vector.from_elements(T.vec(4, f16_ty), p16_row)
                kloc0_i32 = arith.AddIOp(
                    arith.constant(s * 16, type=T.i32),
                    arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result).result
                kloc0_idx = arith.index_cast(T.index, kloc0_i32)
                kv_pos0 = k_start + kloc0_idx
                dsp_idx0 = (tok_safe * NH + my_head) * TOPK_I + kv_pos0
                _if_st = scf.IfOp(q_valid, [], has_else=False)
                with ir.InsertionPoint(_if_st.then_block):
                    store_f16_v(ds_vec, ds_ptr, dsp_idx0)
                    store_f16_v(p_vec, pout_ptr, dsp_idx0)
                    scf.YieldOp([])
                # dS into LDS p-region (B-operand for the dQ AV MFMA), vec4 contiguous
                lds_pidx0 = c_lds_p + lane_mod_16 * arith.index(BLOCK_K) + kloc0_idx
                lds_store_v4(ds_vec, lds_pidx0)

            _lds_fence()

            g_koff = lane_div_16 * arith.index(8)

            # ds_read_tr lane decomposition for the row-major V tile. HW 4x4
            # transpose: result[L][e] = V[src][L%4], src=(L//16)*16+e*4+(L%16)//4.
            # Address so result = A[d=dt*16+L%16, key=(L//16)*8+e]:
            #   k_row = (L//16)*8 + (L%16)//4 ; d_col = dt*16 + (L%4)*4
            # va gives keys g*8+0..3; a second read at +4 keys gives g*8+4..7;
            # shuffle to the full vec8 A-operand (32-key contraction chunk).
            tr_k_group = (lane % arith.index(16)) // arith.index(4)
            tr_col_sub = lane % arith.index(4)
            v_krow = lane_div_16 * arith.index(8) + tr_k_group

            def _ds_read_tr_v4(elem_idx):
                byte = elem_idx * arith.index(2) + arith.index(lds_off)
                byte_i64 = arith.index_cast(T.i64, byte)
                ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
                return rocdl.ds_read_tr16_b64(T.vec(4, f16_ty), ptr).result

            # AV over N_AVSUB 32-key sub-blocks (each 16x16x32 contracts 32 keys).
            # avs offsets the p B-operand (kloc) and the V read (key row) by 32.
            def _b_pack(avs):
                b_pidx = c_lds_p + lane_mod_16 * arith.index(BLOCK_K) + arith.index(avs * 32) + g_koff
                return vector.load_op(T.vec(8, f16_ty), lds, [b_pidx])

            def _av_a(avs, dt):
                k_row_a = arith.index(avs * 32) + v_krow
                d_col = arith.index(dt * 16) + tr_col_sub * arith.index(4)
                if const_expr(_USE_DMA):
                    # va reads keys g*8+tr; vb reads +4 keys. +4 flips key&3, so each
                    # read needs its own swizzle mask.
                    k_row_b = k_row_a + arith.index(4)
                    base_a = kv_lds_base + k_row_a * arith.index(V_STRIDE) + _kv_swz(k_row_a, d_col)
                    base_b = kv_lds_base + k_row_b * arith.index(V_STRIDE) + _kv_swz(k_row_b, d_col)
                    va = _ds_read_tr_v4(base_a)
                    vb = _ds_read_tr_v4(base_b)
                else:
                    base = kv_lds_base + k_row_a * arith.index(V_STRIDE) + d_col
                    va = _ds_read_tr_v4(base)
                    vb = _ds_read_tr_v4(base + arith.index(4 * V_STRIDE))
                return vector.shuffle(va, vb, [0, 1, 2, 3, 4, 5, 6, 7])

            # ==== GEMM2: dQ += dS @ K  (K==V shared tile, ds_read_tr; dS is B-op) ====
            # No online-softmax rescale (dQ accumulates directly across tiles).
            _AV_PF = 2
            for avs in range_constexpr(N_AVSUB):
                b_pack = _b_pack(avs)  # dS pack from LDS p-region
                av_pf = [None] * D_TILES
                for p in range_constexpr(_AV_PF):
                    av_pf[p] = _av_a(avs, p)
                new_acc = []
                for dt in range_constexpr(D_TILES):
                    new_acc.append(_mfma_16x16x32(av_pf[dt], b_pack, acc[dt]))
                    if const_expr(dt + _AV_PF < D_TILES):
                        av_pf[dt + _AV_PF] = _av_a(avs, dt + _AV_PF)
                acc = new_acc

            gpu.barrier()  # protect this tile's kv/dS LDS from next tile's gather
            yield list(acc)

        acc = [results[d] for d in range_constexpr(D_TILES)]

        # ---- epilogue: store dQ_lora [token, head, :HEAD_DIM] (rope cols stay 0) ----
        head_active = arith.AndIOp(head_ib, tok_active).result
        _if = scf.IfOp(head_active, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            dq_row_base = (tok_safe * NH + my_head) * DQK  # dQ is [T,H,D_QK] like q
            for dt in range_constexpr(D_TILES):
                for r in range_constexpr(4):
                    dv = vector.extract(acc[dt], static_position=[r], dynamic_position=[])
                    dv16 = arith.trunc_f(f16_ty, dv)
                    d = arith.index(dt * 16) + lane_div_16 * arith.index(4) + arith.index(r)
                    store_f16(dv16, dq_ptr, dq_row_base + d)
            scf.YieldOp([])

    @flyc.jit
    def launch_dsa_bwd_dq(
        Q: fx.Tensor,
        KV: fx.Tensor,
        dO: fx.Tensor,
        TopK: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        dQ: fx.Tensor,
        dS: fx.Tensor,
        Pout: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        tt_idx = arith.index_cast(T.index, total_tokens)
        launcher = dsa_bwd_dq_kernel(Q, KV, dO, TopK, LSE, Delta, dQ, dS, Pout, total_tokens)
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
            return launch_dsa_bwd_dq(*args, **kwargs)

    return _launch
