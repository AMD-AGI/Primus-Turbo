/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// f00_naive -- rung 00 of the Llama forward ladder: the naive baseline (no parent).
//
// Kernel Specification
//   work      workgroup = 64 query positions x ONE q head; wave w = positions [16 w, +16)
//   grid      B * Hq * Sq/64, q block fastest (natural order)
//   kv        64-row blocks, ONE LDS buffer: K tile (pitch 272) + V tile (pitch 288) = 35 KB
//   fill      every thread: global -> VGPR (buffer_load_b128) -> LDS (ds_store_b128), 8 + 8 per block
//   sync      per block: 2 full barriers (publish the fill; every wave done reading before the next)
//   Q         Q^T operands loaded once from global straight into registers (32 VGPR)
//   S^T       K Q^T: 16 WMMA per block, the K operand of every WMMA read from LDS (2 x ds_load_b128)
//   softmax   online, eager: every block rescales O and l; causal mask on every block
//   PV        O^T += V^T P^T: 16 WMMA per block, V^T operands by ds_load_tr16_b128
//
// The smallest kernel here that computes the right answer. Nothing overlaps: a block's fill waits
// for the previous block's PV, its S^T waits for the fill, and every WMMA waits for its own LDS
// operand.
#include "ladder/lf.h"

using namespace hk_attn;

namespace {

constexpr int K_OFF = 0;
constexpr int V_OFF = lf::K_TILE_B;
constexpr int LDS_B = lf::K_TILE_B + lf::V_TILE_B; // 35840

// global -> VGPR -> LDS: one 64 x 128 bf16 tile; thread t moves the 16-B chunk t % 16 of rows
// t / 16 + 8 j (j < 8), i.e. 128 threads x 16 B = 8 rows per round
HKA_DEV void fill_tile(lds_u8 *dst, int pitch, rsrc_t r, int row0_b, int stride_b, int tid) {
    const int c = tid % 16, r0 = tid / 16;
    u32x4     v[8];
    static_for<8>([&](auto j) { v[j] = buf_load_b128(r, row0_b + (r0 + 8 * j) * stride_b + 16 * c, 0); });
    static_for<8>([&](auto j) { ds_store_b128(dst, (r0 + 8 * j) * pitch + 16 * c, v[j]); });
}

} // namespace

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_fwd_f00(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long q_bytes,
    long long kv_bytes, long long o_bytes, long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv,
    int causal, float c2) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const lf::Args a{Q, K, V, O, LSE, q_bytes, kv_bytes, o_bytes, lse_bytes, B, Sq, Skv, Hq, Hkv, causal, c2};

    // ---- work: q block fastest ---------------------------------------------------------------------
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int nqb = Sq / lf::QB, i = (int) __builtin_amdgcn_workgroup_id_x();
    const int qb = i % nqb, hq = (i / nqb) % Hq, b = i / (nqb * Hq);
    const int hkv = hq / (Hq / Hkv);
    const int p0  = lf::QB * qb + lf::QW * w;
    const int n   = lf::n_kv_blocks(qb, Sq, Skv, causal);
    const int cshift = Skv - Sq, kv_stride_b = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    lds_u8 *const  kt_lds = lds_ptr(lds0 + K_OFF), *const vt_lds = lds_ptr(lds0 + V_OFF);
    const lds_u8  *kbase = rows_base<lf::K_PITCH>(lds0 + K_OFF, l);
    const lds_u8  *vbase = tr_base<lf::V_PITCH>(lds0 + V_OFF, l);
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);

    opnd_t qt[lf::NDK];
    lf::load_qt(qt, rq, lf::row_b(b, Sq, p0, Hq, hq), Hq * lf::D * 2, l);
    acc_t oacc[lf::NDT];
    static_for<lf::NDT>([&](auto dt) { oacc[dt] = zero_acc(); });
    float m2 = lf::BIG_NEG, lsum = 0.f;

    for (int j = 0; j < n; ++j) {
        // ---- fill block j (one buffer: every wave must be done with block j - 1 first) ----------------
        lds_barrier();
        const int kv0_b = lf::row_b(b, Skv, lf::KVB * j, Hkv, hkv);
        fill_tile(kt_lds, lf::K_PITCH, rk, kv0_b, kv_stride_b, tid);
        fill_tile(vt_lds, lf::V_PITCH, rv, kv0_b, kv_stride_b, tid);
        lds_barrier();

        // ---- S^T = K Q^T (rows kv, lane = query) -----------------------------------------------------
        acc_t s[lf::NKT];
        static_for<lf::NKT>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                const opnd_t ka = load_rows<lf::K_PITCH>(kbase, 16 * kt, 32 * k);
                if constexpr (decltype(k)::value == 0) s[kt] = wmma0(ka, qt[0]);
                else s[kt] = wmma(ka, qt[k], s[kt]);
            });
        });
        if (causal) // kv tile origin 64 j + 16 kt, q tile origin p0 (frag.h mask_row_gt_col)
            static_for<lf::NKT>([&](auto kt) {
                s[kt] = mask_row_gt_col(s[kt], l, p0 + cshift - (lf::KVB * j + 16 * kt));
            });

        // ---- online softmax, eager rescale ---------------------------------------------------------------
        const float m2n  = __builtin_fmaxf(m2, lf::qmax(s) * c2);
        const float corr = exp2_hw(m2 - m2n);
        lsum             = lsum * corr + lf::exp_tiles(s, c2, m2n);
        static_for<lf::NDT>([&](auto dt) { oacc[dt] = oacc[dt] * corr; });
        m2 = m2n;

        // ---- O^T += V^T P^T ---------------------------------------------------------------------------
        static_for<lf::NKC>([&](auto c) {
            const opnd_t pt = pack_pt_fwd(s[2 * c], s[2 * c + 1]);
            static_for<lf::NDT>([&](auto dt) {
                oacc[dt] = wmma(load_tr<lf::V_PITCH>(vbase, 32 * c, 16 * dt), pt, oacc[dt]);
            });
        });
    }
    lf::store_out(a, oacc, lsum, m2, b, hq, p0, l);
}
