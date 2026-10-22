/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// t_ext_fwd.hip -- bit-exact tests of hk_attn/ext_fwd.h and of the forward v0's data layout (fwd/idx.h), one
// workgroup each, small-integer / power-of-two inputs so every expectation is exact. Driver:
// hk/lab/test_ext_fwd.py (--cpu default: no GPU; --gpu: main session only).
//
//   t_fx_chain  4 waves = k_attn_fwd_v0's memory program on one stage, with the kernel's own f0:: addresses
//               and descriptors: per wave the Q op (tdm_tile<16, 512, 1040>), one real K / V op of stage 1 by
//               role (waves 0 / 1 K halves, 2 / 3 V halves; hot forms) and one 16-B dummy op (counted by
//               TENSORCNT: s_wait_tensorcnt 1 must retire the real op), LDS prefilled 0xABCD. Then per wave:
//               Q^T B operands (b128 @1040), K A operands (b128 @272) and V^T A operands (tr16 @288) of
//               chunk 2 + (w & 1), S^T = K Q^T, P^T = bf16(S^T), O^T = V^T P^T (C = 0), the O image in the
//               wave's dead Q region and the whole-line read-back stored to O at f0::o_store_b.
//   t_fx_qclamp 1 wave: the Q tile <16, 512, 1040> with tensor_dim1 = f0::q_op_dim1 = 8 (rows 8..15 zero
//               filled, rows past the tensor never read, pad bytes untouched)
//   t_fx_misc   1 wave: ballot + uniform branch, v_log_f32, IEEE 1/x, v_pk_fma (broadcast + negated
//               operand), v_pk_mul, v_pk_add, v_max3, the 16-value max tree with permlanex16
#include "fwd/idx.h"
#include "hk_attn/ext_fwd.h"
#include "hk_attn/prims.h"

using namespace hk_attn;

namespace {

HKA_DEV void fill_lds(uint32_t a, int nbytes, uint32_t v, int tid, int nthr) {
    for (int o = 16 * tid; o < nbytes; o += 16 * nthr) ds_store_b128(lds_ptr(a), o, u32x4{v, v, v, v});
}
HKA_DEV void dump_lds(rsrc_t r, int dst_off, uint32_t a, int nbytes, int tid, int nthr) {
    for (int o = 16 * tid; o < nbytes; o += 16 * nthr) buf_store_b128(r, dst_off + o, 0, ds_load_b128_u32(lds_ptr(a), o));
}
HKA_DEV void put_acc(rsrc_t r, int off, acc_t c, int lane) {
    buf_store_b128(r, off + 32 * lane, 0, lo_u32(c));
    buf_store_b128(r, off + 32 * lane + 16, 0, hi_u32(c));
}
HKA_DEV void put_opnd(rsrc_t r, int off, opnd_t v, int lane) {
    const u32x8 w = __builtin_bit_cast(u32x8, v);
    buf_store_b128(r, off + 32 * lane, 0, __builtin_shufflevector(w, w, 0, 1, 2, 3));
    buf_store_b128(r, off + 32 * lane + 16, 0, __builtin_shufflevector(w, w, 4, 5, 6, 7));
}

// t_fx_chain problem: B 1, Sq 64, Skv 128, Hq 8, Hkv 2 (G 4), hq4 1 -> hkv 1, qb 0, n = 2 stages; the ops of
// stage k = 1 are real, op k = 2 is a dummy
constexpr int CSQ = 64, CSKV = 128, CHQ = 8, CHKV = 2, CHQ4 = 1, CQB = 0, CN = 2;
// debug image offsets in `dbg`
constexpr int DBG_QO = 0;                                    // 4 x 16640 B: Q regions after the TDM
constexpr int DBG_K1 = DBG_QO + f0::NWAVE * f0::QO_W_B;      // 66560: K ring slot 1
constexpr int DBG_V1 = DBG_K1 + f0::K_SLOT_B;                // 83968: V ring slot 1
constexpr int DBG_DM = DBG_V1 + f0::V_SLOT_B;                // 102400: the 4 dummy cells
constexpr int DBG_S  = 103424;                               // 4 waves x 8 S^T accumulators x 1024 B
constexpr int DBG_QB = DBG_S + 4 * 8 * 1024;                 // wave 0: 16 Q^T operands
constexpr int DBG_KA = DBG_QB + 16 * 1024;                   // wave 0: 8 K operands
constexpr int DBG_VA = DBG_KA + 8 * 1024;                    // wave 0: 8 V^T operands
constexpr int DBG_BYTES = DBG_VA + 8 * 1024;                 // 168960
static_assert(DBG_DM + 64 <= DBG_S && DBG_BYTES == 168960, "t_fx_chain debug map");

} // namespace

// ---- t_fx_chain (4 waves): Q [1][64][8][128], K / V [1][128][2][128] bf16 (q/k/v_bytes = their sizes);
// O [1][64][8][128] (prefilled by the host; only heads 4..7 are written); dbg (DBG_BYTES)
extern "C" __global__ __launch_bounds__(128) void t_fx_chain(const __bf16 *Q, int q_bytes, const __bf16 *K,
                                                            const __bf16 *V, int kv_bytes, __bf16 *O,
                                                            int o_bytes, void *dbg, int dbg_bytes) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[f0::LDS_BYTES];
    (void) q_bytes;
    (void) kv_bytes;
    const int      tid = (int) __builtin_amdgcn_workitem_id_x(), w = wave_id(), l = lane_id();
    const uint32_t lds0 = lds_addr(smem);
    const rsrc_t   ro = make_rsrc(O, o_bytes), rd = make_rsrc(dbg, dbg_bytes);
    const int      rv = f0::op_role_v(w), hkv = f0::hkv_of(CHQ4, CHQ / CHKV);
    fill_lds(lds0, f0::LDS_BYTES, 0xABCDABCDu, tid, 128);
    lds_barrier(); // the fill's DS stores complete before any TDM write
    // the kernel's descriptors: Q op, real op k = 1, dummy op k = 2 (n = 2)
    tdm_load(tdm_g0(lds0 + (uint32_t) f0::q_op_lds(w),
                    (uint64_t) (uintptr_t) Q + 2ull * (uint64_t) f0::q_op_el(0, CSQ, CHQ, CHQ4, CQB, w)),
             tdm_g1_words(f0::q_g1(CHQ, f0::q_op_dim1(CSQ, CQB, w)).w));
    const uint64_t kv_base = (uint64_t) (uintptr_t) (rv != 0 ? V : K);
    static_for<2>([&](auto KK) {
        constexpr int k = 1 + KK;
        tdm_load(tdm_g0(lds0 + (uint32_t) f0::kv_op_lds_hot(w, k, CN),
                        kv_base + 2ull * (uint64_t) f0::kv_op_el_hot(0, CSKV, CHKV, hkv, w, k, CN)),
                 tdm_g1_words(f0::kv_op_g1_hot(rv, CHKV, f0::kv_op_dim1_hot(CSKV, w, k), f0::op_real(k, CN)).w));
    });
    wait_tensorcnt<1>(); // Q and the real op landed: the dummy (the newest) must be counted for this to hold
    lds_barrier();
    // LDS images: every wave its own Q region, K / V ring slot 1 split over the 4 waves
    dump_lds(rd, DBG_QO + w * f0::QO_W_B, lds0 + f0::q_op_lds(w), f0::QO_W_B, l, 32);
    dump_lds(rd, DBG_K1 + w * (f0::K_SLOT_B / 4), lds0 + f0::k_slot_off(1) + w * (f0::K_SLOT_B / 4), f0::K_SLOT_B / 4, l, 32);
    dump_lds(rd, DBG_V1 + w * (f0::V_SLOT_B / 4), lds0 + f0::v_slot_off(1) + w * (f0::V_SLOT_B / 4), f0::V_SLOT_B / 4, l, 32);
    // operands (the kernel's bursts)
    opnd_t qb[4][4], ka[2][4], va[8];
    {
        const lds_u8 *qp = lds_ptr(lds0 + (uint32_t) f0::q_lane(w, l));
        static_for<4>([&](auto h) {
            static_for<4>([&](auto k) {
                qb[h][k] = cat(ds_load_b128(qp, f0::q_imm(h, k, 0)), ds_load_b128(qp, f0::q_imm(h, k, 1)));
            });
        });
    }
    const int hc = w & 1; // chunk 2 + hc = stage 1, half hc
    auto      burst = [&](auto HC) {
        const lds_u8 *kp = lds_ptr(lds0 + (uint32_t) (f0::k_slot_off(1) + f0::k_lane(l)));
        static_for<4>([&](auto k) {
            static_for<2>([&](auto kt) {
                ka[kt][k] = cat(ds_load_b128(kp, f0::k_imm(HC, kt, k, 0)), ds_load_b128(kp, f0::k_imm(HC, kt, k, 1)));
            });
        });
        const lds_u8 *vp = lds_ptr(lds0 + (uint32_t) (f0::v_slot_off(1) + f0::v_lane(l)));
        static_for<8>([&](auto dt) {
            va[dt] = cat(ds_load_tr16_b128(vp, f0::v_imm(HC, 0, dt)), ds_load_tr16_b128(vp, f0::v_imm(HC, 1, dt)));
        });
    };
    if (hc == 0) burst(std::integral_constant<int, 0>{});
    else burst(std::integral_constant<int, 1>{});
    if (w == 0) {
        static_for<4>([&](auto h) { static_for<4>([&](auto k) { put_opnd(rd, DBG_QB + (4 * h + k) * 1024, qb[h][k], l); }); });
        static_for<2>([&](auto kt) { static_for<4>([&](auto k) { put_opnd(rd, DBG_KA + (4 * kt + k) * 1024, ka[kt][k], l); }); });
        static_for<8>([&](auto dt) { put_opnd(rd, DBG_VA + dt * 1024, va[dt], l); });
    }
    // S^T = K Q^T, P^T, O^T = V^T P^T
    acc_t S[4][2], Ot[4][8];
    static_for<4>([&](auto k) {
        static_for<2>([&](auto kt) {
            static_for<4>([&](auto h) {
                if constexpr (decltype(k)::value == 0) S[h][kt] = wmma0(ka[kt][0], qb[h][0]);
                else S[h][kt] = wmma(ka[kt][k], qb[h][k], S[h][kt]);
            });
        });
    });
    static_for<4>([&](auto h) {
        static_for<2>([&](auto kt) { put_acc(rd, DBG_S + (w * 8 + 2 * h + kt) * 1024, S[h][kt], l); });
        const opnd_t P = pack_pt_fwd(S[h][0], S[h][1]);
        static_for<8>([&](auto dt) { Ot[h][dt] = wmma0(va[dt], P); });
    });
    // O image (the wave's dead Q region) + whole-line read-back + O stores (the kernel's epilogue maps)
    lds_u8 *ow = lds_ptr(lds0 + (uint32_t) f0::ow_lane(w, l));
    static_for<4>([&](auto h) {
        static_for<8>([&](auto dt) { ds_store_b128(ow, f0::ow_imm(h, dt), cvt_pk_bf16(Ot[h][dt])); });
    });
    compiler_fence();
    const lds_u8 *rp = lds_ptr(lds0 + (uint32_t) f0::or_lane(w, l));
    static_for<2 * f0::QW>([&](auto s) {
        buf_store_b128(ro, (int) f0::o_store_b(0, CSQ, CHQ, CHQ4, CQB, w, s, l), 0, ds_load_b128_u32(rp, f0::or_imm(s)));
    });
    // drain the dummies, then the dummy cells
    wait_tensorcnt<0>();
    lds_barrier();
    if (w == 0 && l < 4) buf_store_b128(rd, DBG_DM + 16 * l, 0, ds_load_b128_u32(lds_ptr(lds0 + (uint32_t) f0::dummy_off(l)), 0));
}

// ---- t_fx_qclamp (1 wave): src Q [nrows][8][128] bf16 (allocation has 8 extra rows of 0x7777), Sq = 72:
// the Q op of (qb 1, wave 0) = positions 64.. with tensor_dim1 = q_op_dim1(72, 1, 0) = 8; out = 16640 B image
extern "C" __global__ __launch_bounds__(32) void t_fx_qclamp(const __bf16 *Q, int sq, void *out, int out_bytes) {
    __shared__ __attribute__((aligned(1024))) unsigned char sm[f0::QO_W_B];
    const int      l    = lane_id();
    const uint32_t l0   = lds_addr(sm);
    const rsrc_t   ro   = make_rsrc(out, out_bytes);
    const int      nsq  = uniform(sq);
    fill_lds(l0, f0::QO_W_B, 0xABCDABCDu, l, 32);
    lds_barrier();
    tdm_load(tdm_g0(l0, (uint64_t) (uintptr_t) Q + 2ull * (uint64_t) f0::q_op_el(0, nsq, 8, 1, 1, 0)),
             tdm_g1_words(f0::q_g1(8, f0::q_op_dim1(nsq, 1, 0)).w));
    split_signal_tdm_lds<0>();
    split_wait_lds();
    dump_lds(ro, 0, l0, f0::QO_W_B, l, 32);
}

// ---- t_fx_misc (1 wave): src f32: X[32] | T[2] (pad to 64) | LG[32] | RC[32] | S[64] | M[32] | A[32] | B3[96]
// (float offsets 0, 32, 64, 96, 128, 192, 224, 256), plus c (kernel arg, fp32 bits)
// out (u32 / f32, 32 lanes each): 0 ballot(X > T0) | 1 branch(T0) | 2 ballot(X > T1) | 3 branch(T1) |
// 4 log2_hw(LG) | 5 rcp_ieee(RC) | 6,7 pk_fma(S pair, c, -M) | 8,9 pk_mul(S pair, A) | 10,11 pk_add(S pair, A)
// | 12 max3(B3) | 13 column max of 16 values (max16 tree + permlanex16) of S x 0.5^lane... (see driver)
extern "C" __global__ __launch_bounds__(32) void t_fx_misc(const float *src, int src_bytes, void *out, int out_bytes,
                                                           int c_bits) {
    const int    l  = lane_id();
    const rsrc_t rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    const float  c  = __builtin_bit_cast(float, uniform(c_bits));
    auto ld = [&](int fo) { return buf_load_f32(rs, 4 * (fo + l), 0); };
    auto st = [&](int slot, float v) { buf_store_f32(ro, 128 * slot + 4 * l, 0, v); };
    auto stu = [&](int slot, uint32_t v) { buf_store_b32(ro, 128 * slot + 4 * l, 0, v); };
    const float x = ld(0), t0 = buf_load_f32(rs, 4 * 32, 0), t1 = buf_load_f32(rs, 4 * 33, 0);
    static_for<2>([&](auto j) {
        const uint32_t m = ballot(x > (j == 0 ? t0 : t1));
        stu(2 * j, m);
        uint32_t br = 2u;
        if (any_set(m)) br = 1u; // uniform branch
        stu(2 * j + 1, br);
    });
    st(4, log2_hw(ld(64)));
    st(5, rcp_ieee(ld(96)));
    const f32x2 s2 = {buf_load_f32(rs, 4 * (128 + 2 * l), 0), buf_load_f32(rs, 4 * (128 + 2 * l + 1), 0)};
    const float m = ld(192), a = ld(224);
    const f32x2 r0 = pk_fma(s2, f32x2{c, c}, f32x2{-m, -m});
    const f32x2 r1 = pk_mul(s2, f32x2{a, a});
    const f32x2 r2 = pk_add(s2, f32x2{a, a});
    st(6, r0[0]);
    st(7, r0[1]);
    st(8, r1[0]);
    st(9, r1[1]);
    st(10, r2[0]);
    st(11, r2[1]);
    st(12, max3(ld(256), ld(288), ld(320)));
    // the kernel's SM1 column max: 16 values of this lane (2 accumulators) -> max3 tree -> permlanex16
    acc_t a0, a1;
    static_for<8>([&](auto i) {
        a0[(int) i] = buf_load_f32(rs, 4 * (352 + 16 * l + i), 0);
        a1[(int) i] = buf_load_f32(rs, 4 * (352 + 16 * l + 8 + i), 0);
    });
    const float x0 = max3(a0[0], a0[1], a0[2]), x1 = max3(a0[3], a0[4], a0[5]), x2 = max3(a0[6], a0[7], a1[0]);
    const float x3 = max3(a1[1], a1[2], a1[3]), x4 = max3(a1[4], a1[5], a1[6]);
    float mx = __builtin_fmaxf(max3(x0, x1, x2), max3(x3, x4, a1[7]));
    mx = __builtin_fmaxf(mx, xhalf(mx));
    st(13, mx);
}
