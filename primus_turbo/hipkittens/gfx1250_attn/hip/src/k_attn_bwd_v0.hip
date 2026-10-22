/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_attn_bwd_v0.hip -- HK gfx1250 d128 GQA attention backward v0 (hk/notes/bwd_v0_design.md; implementation
// note hk/notes/bwd_v0_impl.md). Four kernels, launched in this order on one stream by bwd_v0.py:
//
//   k_hk_bwd_prep / _prep_z  LD records {NL = -LSE log2e, delta = rowsum(dO o O)} per 4 queries (+ _z: zero
//                            the WG's 32 KB of dQ_acc)                     grid (Sq/64, Hq, B), block 256
//   k_hk_bwd_g4              dK (scaled), dV, dQ_acc += dS K (fp32 atomics, C-fragment order), G = 4
//                                                                          grid (Hkv, Skv/128, B), block 128
//   k_hk_bwd_dqcvt           dq = bf16(scale dQ_acc), de-swizzled          grid (Sq/32, Hq, B), block 256
//
// Every index / address / loop bound comes from bwd/idx.h (b0::), which the CPU bounds checker
// hk/lab/bounds/bwd_v0_check.cpp replays. Every global access is a buffer access with a TRUE byte extent or a
// TDM tile whose tensor dims equal the tile dims. No hidden kernel arguments; static LDS.
//
// k_hk_bwd_g4, one WG = 128 kv rows of one (b, hkv), wave w (one per SIMD) owns kv rows kvw0 + 32 w + [0, 32);
// K / V B-operands (kf, vf), the K operand of dQ (kt) and dV^T / dK^T accumulators stay in VGPRs. A step = one
// q pair (32 queries) x one q head of the group; heads inner, pairs descending; unmasked steps first. Step s
// (parity P, design 6.6):
//   P1  S/dP of kv tile 0: 4 chains x 4 dt                                                        16 WMMA
//   P2  S/dP of kv tile 1 || softmax of kv tile 0                                                  16 WMMA
//   P3  dQ(s-1) = dS(s-1) K (A = dSa, read at the end of step s-1) || softmax of kv tile 1         16 WMMA
//   P4  4 ds_store_b128 of dS^T(s) -> dS buffer P; order fence; 32 tr16 of slot s (dO^T, Q^T);
//       32 fp32 atomics of dQ(s-1) (never waited in the loop)
//   P5  s_wait_tensorcnt 1; s_wait_dscnt 32 (the 32 tr16 stay in flight); s_barrier_signal       (B_s)
//   P6  dV^T += dO^T P, dK^T += Q^T dS                                                             32 WMMA
//   P7  s_barrier_wait                                                                             (B_s)
//   P8  waves 0..2: TDM op(s+3) (dO / Q / LD of tile clamp(s+3)) -> ring slot (s+3) % 4
//   P9  readback of slot (s+1) % 4 (qA, dA, LD) into the other register set; dSa = 16 tr16 of dS buffer P
// Diagonal steps (s >= nf): per-wave kind FULL / DIAG / DEAD (uniform branch), the softmax argument masked
// with MASK_NEG; the signal drains DS (dscnt 0) and the E tr16 are loaded after it; DEAD waves skip S/dP,
// softmax, dV/dK and store a zero dS^T, but keep every barrier, dQ WMMA and atomic.
//
// Knobs (compile-time): BWD_SKIP_DEAD (1; 0 = DEAD waves run the masked DIAG path: dk / dv must stay bitwise
// equal), BWD_INTERLEAVE (1; sched_group_barrier interleave of the softmax into P2 / P3; 0 = phases only).
#include "bwd/idx.h"
#include "hk_attn/ext_bwd.h"
#include "hk_attn/prims.h"

#ifndef BWD_SKIP_DEAD
#define BWD_SKIP_DEAD 1
#endif
#ifndef BWD_INTERLEAVE
#define BWD_INTERLEAVE 1
#endif
#ifndef BWD_HOLD
#define BWD_HOLD 0
#endif

using namespace hk_attn;

namespace {

constexpr int G  = b0::G4;
constexpr int XP = b0::XP, SP = b0::SP;
using T_QDO = tdm_tile<b0::QS, b0::D, XP>;     // dO / Q tile [32][128] bf16 -> pitch XP
using T_LD  = tdm_tile<1, 64, b0::LD_B, 4>;    // 32 queries' LD records: one row of 64 fp32
using T_KV  = tdm_tile<b0::BKV, b0::D, XP>;    // K / V image [128][128] bf16 -> pitch XP
static_assert(T_QDO::w0 == b0::W0_QDO && T_LD::w0 == b0::W0_LD && T_KV::w0 == b0::W0_KV, "TDM w0 == idx.h");
static_assert(T_QDO::bytes == b0::QT_B && T_KV::bytes == b0::KV_B && T_LD::bytes == b0::LD_B, "TDM footprints");
static_assert(b0::RING == 4, "slot = s & 3");

// one step's readback set: S / dP A operands (by hh, dt) and the LD records (by hh, k: NL lo, delta lo, NL hi,
// delta hi of accumulator rows 8h + 0..3 / 4..7)
struct XSet {
    opnd_t qA[2][4], dA[2][4];
    f32x4  ld[2][4];
};

// sched_group_barrier interleave of a 16-WMMA phase with softmax VALU / TRANS (design 6.6): `skip` leading
// WMMA-only groups, then one WMMA + NV VALU-class ops per group
template <int SKIP, int NV> HKA_DEV void interleave16() {
#if BWD_INTERLEAVE
    static_for<16>([&](auto g) {
        sched_group_barrier<sg::WMMA, 1, 0>();
        if constexpr ((int) g >= SKIP) sched_group_barrier<sg::VALU | sg::TRANS, NV, 0>();
    });
#endif
}

} // namespace

// ============================================================================================================
// k_hk_bwd_prep[_z]: grid (Sq / 64, Hq, B), block 256 (8 waves). WG (b, h, q0 = 64 bx); thread t, pass p:
// row q0 + 16 p + t / 16, 16-B chunk t % 16 of dO and O. delta = sum over d of dO * O in fp32 (per lane: even /
// odd fma chains, then their sum, then the 16-lane butterfly), NL = -(LSE * log2e). Thread g < 16 writes the
// 32-B record of queries q0 + 4g. ZERO: also zeroes dQ_acc blocks 2 bx, 2 bx + 1 of (b, h) (32 KB).
template <bool ZERO>
HKA_DEV void prep_body(const bf16 *DO, const bf16 *O, const float *LSE, float *LD, float *DQA, int Sq, int Hq,
                       int B) {
    __shared__ __attribute__((aligned(16))) unsigned char sdelta[b0::PREP_LDS_B];
    const int      t = (int) __builtin_amdgcn_workitem_id_x();
    const int      bx = (int) blockIdx.x, h = (int) blockIdx.y, b = (int) blockIdx.z;
    const int      q0 = b0::PREP_ROWS * bx;
    const uint32_t lds0 = lds_addr(sdelta);
    const rsrc_t   r_do = make_rsrc(DO, b0::qdo_bytes(B, Sq, Hq)), r_o = make_rsrc(O, b0::qdo_bytes(B, Sq, Hq));
    const rsrc_t   r_lse = make_rsrc(LSE, (int64_t) B * Hq * Sq * 4), r_ld = make_rsrc(LD, b0::ld_bytes(B, Hq, Sq));
    const int      c = b0::prep_chunk(t);
    u32x4          vdo[4], vo[4];
    static_for<4>([&](auto p) {
        const int off = (int) b0::prep_row_byte(b, Sq, Hq, h, q0 + b0::prep_row(t, p)) + 16 * c;
        vdo[p] = buf_load_b128(r_do, off, 0);
        vo[p]  = buf_load_b128(r_o, off, 0);
    });
    static_for<4>([&](auto p) {
        const bf16x8 a = __builtin_bit_cast(bf16x8, vdo[p]), o = __builtin_bit_cast(bf16x8, vo[p]);
        float        e0 = 0.f, e1 = 0.f;
        static_for<4>([&](auto k) {
            e0 = __builtin_fmaf((float) a[2 * k], (float) o[2 * k], e0);
            e1 = __builtin_fmaf((float) a[2 * k + 1], (float) o[2 * k + 1], e1);
        });
        float v = e0 + e1;
        v       = v + xor16<1>(v);
        v       = v + xor16<2>(v);
        v       = v + xor16<4>(v);
        v       = v + xor16<8>(v);
        if (c == 0) ds_store_b32(lds_ptr(lds0), 4 * b0::prep_row(t, p), __builtin_bit_cast(uint32_t, v));
    });
    lds_barrier();
    if (t < 16) {
        const u32x4 d4 = ds_load_b128_u32(lds_ptr(lds0), 16 * t);
        const f32x4 l4 = buf_load_f32x4(r_lse, (int) b0::lse_byte(b, Hq, h, Sq, q0 + 4 * t), 0);
        const f32x4 nl = -(l4 * b0::LOG2E);
        const int   rec = (int) b0::ld_rec_byte(b, Hq, h, Sq, q0 + 4 * t);
        buf_store_b128(r_ld, rec, 0, __builtin_bit_cast(u32x4, nl));
        buf_store_b128(r_ld, rec + 16, 0, d4);
    }
    if constexpr (ZERO) {
        const rsrc_t r_dqa = make_rsrc(DQA, b0::dqa_bytes(B, Hq, Sq));
        const int    base = (int) b0::cvt_src_byte(b, Hq, h, b0::nqp_of(Sq), 2 * bx);
        static_for<8>([&](auto i) { buf_store_b128(r_dqa, base + b0::prep_zero_off(t, i), 0, zero_u32x4()); });
    } else {
        (void) DQA;
    }
}

extern "C" __global__ __launch_bounds__(b0::PREP_THREADS) void k_hk_bwd_prep(const bf16 *__restrict__ DO,
                                                                            const bf16 *__restrict__ O,
                                                                            const float *__restrict__ LSE,
                                                                            float *__restrict__ LD,
                                                                            float *__restrict__ DQA, int Sq,
                                                                            int Hq, int B) {
    prep_body<false>(DO, O, LSE, LD, DQA, Sq, Hq, B);
}
extern "C" __global__ __launch_bounds__(b0::PREP_THREADS) void k_hk_bwd_prep_z(const bf16 *__restrict__ DO,
                                                                              const bf16 *__restrict__ O,
                                                                              const float *__restrict__ LSE,
                                                                              float *__restrict__ LD,
                                                                              float *__restrict__ DQA, int Sq,
                                                                              int Hq, int B) {
    prep_body<true>(DO, O, LSE, LD, DQA, Sq, Hq, B);
}

// ============================================================================================================
// k_hk_bwd_dqcvt: grid (Sq / 32, Hq, B), block 256. The 16 KB C-fragment block of (b, qh, qt): every 128-B
// line read once, whole (4 b128 per thread); bf16(x * scale) (fp32 multiply, RNE) scattered into an LDS image
// [32 q][128 d] of pitch 272 (one ds_store_b64 per 4 consecutive d); then whole 256-B dq rows stored.
extern "C" __global__ __launch_bounds__(b0::CVT_THREADS) void k_hk_bwd_dqcvt(const float *__restrict__ DQA,
                                                                            bf16 *__restrict__ DQ, float scale,
                                                                            int Sq, int Hq, int B) {
    __shared__ __attribute__((aligned(16))) unsigned char img[b0::CVT_LDS_B];
    const int      t = (int) __builtin_amdgcn_workitem_id_x();
    const int      qt = (int) blockIdx.x, qh = (int) blockIdx.y, b = (int) blockIdx.z;
    const uint32_t lds0 = lds_addr(img);
    const rsrc_t   r_src = make_rsrc(DQA, b0::dqa_bytes(B, Hq, Sq)), r_dq = make_rsrc(DQ, b0::qdo_bytes(B, Sq, Hq));
    const int      src = (int) b0::cvt_src_byte(b, Hq, qh, b0::nqp_of(Sq), qt);
    f32x4          x[4];
    static_for<4>([&](auto i) { x[i] = buf_load_f32x4(r_src, src + b0::cvt_src_off(t, i), 0); });
    static_for<4>([&](auto i) {
        const int p = b0::cvt_src_off(t, i) / 4;
        const f32x4 y = x[i] * scale;
        ds_store_b64(lds_ptr(lds0), b0::cvt_lds(b0::cvt_q(p), b0::cvt_d(p)),
                     u32x2{cvt_pk_bf16(y[0], y[1]), cvt_pk_bf16(y[2], y[3])});
    });
    lds_barrier();
    static_for<2>([&](auto i) {
        const int row = b0::cvt_row(t, i), chunk = b0::cvt_chunk(t, i);
        buf_store_b128(r_dq, (int) b0::cvt_dst_byte(b, Sq, Hq, qh, qt, row, chunk), 0,
                       ds_load_b128_u32(lds_ptr(lds0), row * b0::CVT_PITCH + 16 * chunk));
    });
}

// ============================================================================================================
// k_hk_bwd_g4: the main kernel (header comment). Kernel arguments (offsets): Q 0, K 8, V 16, DO 24, LD 32,
// DK 40, DV 48, DQA 56 (pointers), scale 64 (f32), Sq 68, Skv 72, Hq 76, Hkv 80, B 84, causal 88 (i32).
extern "C" __global__ __launch_bounds__(b0::BLOCK) void k_hk_bwd_g4(
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const bf16 *__restrict__ DO, const float *__restrict__ LD, bf16 *__restrict__ DK, bf16 *__restrict__ DV,
    float *__restrict__ DQA, float scale, int Sq, int Skv, int Hq, int Hkv, int B, int causal) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[b0::LDS_BYTES];
    lock_simd();
    (void) B; // host-gated; every extent below is per WG
    const uint32_t lds0 = lds_addr(smem);
    const int      l = lane_id(), w = wave_id();
    const b0::Work wk = b0::work_v0((int) blockIdx.x, (int) blockIdx.y, (int) blockIdx.z);
    const int      kvw0 = b0::kvw0_of(wk.blk), nqp = b0::nqp_of(Sq), cshift = Skv - Sq;
    const int      n = b0::n_all(kvw0, cshift, causal, nqp, G), nf = b0::n_full(kvw0, cshift, causal, nqp, G);
    if (n < G) return; // all waves, before any TDM / barrier; unreachable behind the host gate
    const float c1 = scale * b0::LOG2E;

    // ---- buffer resources (TRUE extents: the WG's dQ_acc group, its dK / dV slabs) ------------------------------
    const rsrc_t  r_dqa = make_rsrc((const char *) DQA + b0::dqa_grp_byte(wk.b, Hq, wk.hkv, G, Sq),
                                    b0::dqa_grp_bytes(G, Sq));
    const int64_t epib = b0::epi_base_byte(wk.b, Skv, Hkv, wk.hkv, kvw0);
    const rsrc_t  r_dk = make_rsrc((const char *) DK + epib, b0::epi_bytes(Hkv));
    const rsrc_t  r_dv = make_rsrc((const char *) DV + epib, b0::epi_bytes(Hkv));

    // ---- TDM roles: wave 0 dO, 1 Q, 2 LD records (one op per step each); wave 3 the K / V images ----------------
    const int      role = w;
    const char    *rten = role == b0::ISS_DO ? (const char *) DO : role == b0::ISS_Q ? (const char *) Q : (const char *) LD;
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) b0::role_tile_byte(role, wk.b, Sq, Hq, wk.hkv, G, 0, 0);
    const int64_t  rqstep = b0::role_qstep(role, Sq, Hq), rhstep = b0::role_hstep(role, Sq, Hq);
    const u32x8    rg1 = role == b0::ISS_LD ? tdm_g1<T_LD>(b0::role_stride_el(role, Hq), b0::role_rows(role))
                                            : tdm_g1<T_QDO>(b0::role_stride_el(role, Hq), b0::role_rows(role));
    const int      rlds0 = b0::role_lds0(role), rlstep = b0::role_lds_step(role);
    auto           ring_g0 = [&](int k) { // group 0 of the ring op of step k (tile clamp(k)) -> slot k % 4
        const int qt = b0::tile_qt(k, n, G, nqp), gh = b0::tile_gh(k, n, G);
        return tdm_g0(lds0 + (uint32_t) b0::role_lds_at(rlds0, rlstep, k & 3),
                      rbase + (uint64_t) b0::role_step_byte(0, rqstep, rhstep, qt, gh));
    };
    auto issue = [&](int k) { tdm_load(ring_g0(k), rg1); }; // waves 0..2 only

    // ---- prologue: ops 0..2 (waves 0..2), K / V images (wave 3); P0 barrier -------------------------------------
    if (w < b0::ISS_KV) {
        issue(0);
        issue(1);
        issue(2);
        wait_tensorcnt<2>();
    } else {
        const int64_t kvb = b0::kv_tile_byte(wk.b, Skv, Hkv, wk.hkv, kvw0);
        const u32x8   g1 = tdm_g1<T_KV>(b0::kv_stride_el(Hkv), b0::BKV);
        tdm_load(tdm_g0(lds0 + b0::OFF_K, (uint64_t) (uintptr_t) K + (uint64_t) kvb), g1);
        tdm_load(tdm_g0(lds0 + b0::OFF_V, (uint64_t) (uintptr_t) V + (uint64_t) kvb), g1);
        wait_tensorcnt<0>();
    }
    lds_barrier();

    // ---- resident operands ---------------------------------------------------------------------------------------
    opnd_t kf[2][4], vf[2][4], kt[2][4];
    {
        const lds_u8 *kr = rows_base<XP>(lds0 + b0::OFF_K + b0::kimg_rows_wave(w), l);
        const lds_u8 *vr = rows_base<XP>(lds0 + b0::OFF_V + b0::kimg_rows_wave(w), l);
        const lds_u8 *kc = tr_base<XP>(lds0 + b0::OFF_K + b0::kimg_tr_wave(w), l);
        static_for<2>([&](auto kh) {
            static_for<4>([&](auto dt) {
                kf[kh][dt] = load_rows<XP>(kr, 16 * kh, 32 * dt);
                vf[kh][dt] = load_rows<XP>(vr, 16 * kh, 32 * dt);
            });
        });
        static_for<2>([&](auto j) { static_for<4>([&](auto c) { kt[j][c] = load_tr<XP>(kc, 32 * c, 16 * j); }); });
    }
    acc_t dVt[8][2], dKt[8][2];
    static_for<8>([&](auto dt) {
        static_for<2>([&](auto kh) {
            dVt[dt][kh] = zero_acc();
            dKt[dt][kh] = zero_acc();
        });
    });

    // ---- per-step helpers ---------------------------------------------------------------------------------------------
    auto readback = [&](XSet &X, int slot) { // P9 part 1: qA / dA by dt, then LD
        const lds_u8 *qr = rows_base<XP>(lds0 + b0::lds_q(slot), l), *dr = rows_base<XP>(lds0 + b0::lds_do(slot), l);
        const lds_u8 *lr = lds_ptr(lds0 + b0::lds_ld(slot) + b0::ld_lane(l));
        static_for<4>([&](auto dt) {
            static_for<2>([&](auto hh) {
                X.qA[hh][dt] = load_rows<XP>(qr, 16 * hh, 32 * dt);
                X.dA[hh][dt] = load_rows<XP>(dr, 16 * hh, 32 * dt);
            });
        });
        static_for<2>([&](auto hh) {
            static_for<4>([&](auto k) {
                X.ld[hh][k] = __builtin_bit_cast(f32x4, ds_load_b128_u32(lr, b0::ld_imm(hh, k)));
            });
        });
    };
    opnd_t dSa[2][4]; // A operands of dQ(s - 1): dS(s - 1) [q][kv], read at the end of step s - 1
    auto   load_dsa = [&](int buf) { // P9 part 2
        const lds_u8 *dt_ = tr_base<SP>(lds0 + b0::lds_ds(buf), l);
        static_for<2>([&](auto qh) {
            static_for<4>([&](auto c) { dSa[qh][c] = load_tr<SP>(dt_, 32 * c, 16 * qh); });
        });
    };
    const int voff_lane = b0::dqa_voff(l, w, 0, 0, nqp); // (w << 7) | (l << 2)
    auto      dq_wmma = [&](acc_t (&dq)[2][2]) { // dQ = dS K: kc outer, 4 chains (qh, j)
        static_for<4>([&](auto c) {
            static_for<2>([&](auto qh) {
                static_for<2>([&](auto j) {
                    if constexpr (c == 0) dq[qh][j] = wmma0(dSa[qh][c], kt[j][c]);
                    else dq[qh][j] = wmma(dSa[qh][c], kt[j][c], dq[qh][j]);
                });
            });
        });
    };
    auto dq_atomics = [&](const acc_t (&dq)[2][2], int vo) { // the 32 atomics of dQ(s - 1), vo = their step voffset
        static_for<2>([&](auto qh) {
            static_for<2>([&](auto j) { atomic_add_c<512>(r_dqa, vo + b0::dqa_imm(qh, j, 0), 0, dq[qh][j]); });
        });
    };

    // ---- one step (KIND: 0 FULL, 1 MASKED; X = this step's set, Xn = the readback target) --------------------------
    auto step = [&](auto masked_c, auto par_c, int s, XSet &X, XSet &Xn) __attribute__((always_inline)) {
        constexpr bool MASKED = decltype(masked_c)::value;
        constexpr int  P      = decltype(par_c)::value;
        const int      qt = b0::tile_qt(s, n, G, nqp);
        const int      avo = voff_nonneg(voff_lane + b0::dqa_voff(0, 0, b0::atom_gh(s, n, G), b0::atom_qt(s, n, G, nqp), nqp));
        int            kind = b0::K_FULL;
        if constexpr (MASKED) kind = b0::wave_kind(kvw0, w, qt, cshift, causal);
        const bool live = !MASKED || !BWD_SKIP_DEAD || kind != b0::K_DEAD;
        acc_t      S[2][2], dP[2][2], dq[2][2];
        u32x4      pu[2][2], du[2][2];
        auto       sm = [&](auto hh, auto kh) { // softmax / dS of block (hh, kh), design 6.2
            acc_t t = softmax_arg_rows(S[hh][kh], c1, X.ld[hh][0], X.ld[hh][2]);
            if constexpr (MASKED) t = mask_col_gt_row(t, l, b0::mask_off(qt, hh, kvw0, w, kh, cshift), MASK_NEG);
            const acc_t p = exp2_acc(t), d = dsoftmax(p, dP[hh][kh], X.ld[hh][1], X.ld[hh][3]);
            pu[hh][kh] = cvt_pk_bf16(p);
            du[hh][kh] = cvt_pk_bf16(d);
        };
        auto sdp = [&](auto kh) { // S / dP of kv tile kh: 4 independent chains over dt
            static_for<4>([&](auto dt) {
                static_for<2>([&](auto hh) {
                    if constexpr (dt == 0) S[hh][kh] = wmma0(X.qA[hh][dt], kf[kh][dt]);
                    else S[hh][kh] = wmma(X.qA[hh][dt], kf[kh][dt], S[hh][kh]);
                });
                static_for<2>([&](auto hh) {
                    if constexpr (dt == 0) dP[hh][kh] = wmma0(X.dA[hh][dt], vf[kh][dt]);
                    else dP[hh][kh] = wmma(X.dA[hh][dt], vf[kh][dt], dP[hh][kh]);
                });
            });
        };
        constexpr std::integral_constant<int, 0> I0{};
        constexpr std::integral_constant<int, 1> I1{};
        if (live) {
            // P1: G0
            sdp(I0);
            sched_barrier<0>();
            // P2: G1 || softmax kh 0
            sdp(I1);
            sm(I0, I0);
            sm(I1, I0);
#if BWD_HOLD >= 2
            static_for<2>([&](auto hh) { static_for<4>([&](auto dt) { hold_vgprs(X.qA[hh][dt], du[1][0]); hold_vgprs(X.dA[hh][dt], du[1][0]); }); });
#endif
            interleave16<0, 4>();
            sched_barrier<0>();
            // P3: dQ(s - 1) || softmax kh 1
            dq_wmma(dq);
            sm(I0, I1);
            sm(I1, I1);
#if BWD_HOLD >= 1
            static_for<2>([&](auto qh) { static_for<4>([&](auto c) { hold_vgprs(dSa[qh][c], du[1][1]); }); });
#endif
            interleave16<2, 4>();
        } else {
            dq_wmma(dq);
            static_for<2>([&](auto hh) { static_for<2>([&](auto kh) { du[hh][kh] = zero_u32x4(); }); });
        }
        sched_barrier<0>();
        // P4: dS^T stores, fence, E operands (FULL), atomics of dQ(s - 1)
        {
            lds_u8 *dr = rows_base<SP>(lds0 + b0::lds_ds(P) + b0::ds_rows_wave(w), l);
            static_for<2>([&](auto kh) {
                static_for<2>([&](auto hh) { ds_store_b128(dr, b0::ct_imm(SP, 16 * kh, 16 * hh), du[hh][kh]); });
            });
        }
        order_fence();
        opnd_t        dOT[8], QT[8];
        const lds_u8 *dtr = tr_base<XP>(lds0 + b0::lds_do(s & 3), l), *qtr = tr_base<XP>(lds0 + b0::lds_q(s & 3), l);
        auto          load_e = [&] {
            static_for<8>([&](auto dt) { dOT[dt] = load_tr<XP>(dtr, 0, 16 * dt); });
            static_for<8>([&](auto dt) { QT[dt] = load_tr<XP>(qtr, 0, 16 * dt); });
        };
        if constexpr (!MASKED) load_e();
        sched_barrier<0>();
        dq_atomics(dq, avo);
        // P5: signal B_s
        if constexpr (!MASKED) split_signal_tdm_dscnt<b0::TCNT_STEADY, b0::FULL_UNCOVERED>();
        else split_signal_tdm_lds<b0::TCNT_STEADY>();
        // P6: E (+ the SALU of the next ring op's group 0, hidden under the WMMAs)
        const u32x4 g0n = ring_g0(s + b0::LEAD);
        asm volatile("" ::"s"(g0n)); // materialize here (IR sinking would move it behind the wait)
        if (live) {
            if constexpr (MASKED) load_e();
            const opnd_t Pb[2]  = {cat_u32(pu[0][0], pu[1][0]), cat_u32(pu[0][1], pu[1][1])};
            const opnd_t dSb[2] = {cat_u32(du[0][0], du[1][0]), cat_u32(du[0][1], du[1][1])};
            static_for<8>([&](auto dt) {
                static_for<2>([&](auto kh) { dVt[dt][kh] = wmma(dOT[dt], Pb[kh], dVt[dt][kh]); });
            });
            static_for<8>([&](auto dt) {
                static_for<2>([&](auto kh) { dKt[dt][kh] = wmma(QT[dt], dSb[kh], dKt[dt][kh]); });
            });
        }
        // P7: wait B_s
        split_wait_lds();
        // P8: ring op s + 3
        if (w < b0::ISS_KV) tdm_load(g0n, rg1);
        sched_barrier<0>();
        // P9: readback of tile s + 1, then dSa = dS(s)
        readback(Xn, (s + 1) & 3);
        sched_barrier<0>();
        load_dsa(P);
        sched_barrier<0>();
    };

    // ---- main loop: x2 unrolled (n, nf even), FULL steps then the diagonal steps ---------------------------------
    XSet X0, X1;
    readback(X0, 0);
    static_for<2>([&](auto qh) { static_for<4>([&](auto c) { dSa[qh][c] = opnd_t{}; }); });
    constexpr auto F0 = std::false_type{};
    constexpr auto T1 = std::true_type{};
    constexpr std::integral_constant<int, 0> P0{};
    constexpr std::integral_constant<int, 1> P1{};
    int s = 0;
    for (; s < nf; s += 2) {
        step(F0, P0, s, X0, X1);
        step(F0, P1, s + 1, X1, X0);
    }
    for (; s < n; s += 2) {
        step(T1, P0, s, X0, X1);
        step(T1, P1, s + 1, X1, X0);
    }

    // ---- drain: dQ(n - 1) and its atomics; epilogue (dV, scale * dK) straight from registers; TDM drain + barrier ---
    {
        acc_t dq[2][2];
        dq_wmma(dq);
        dq_atomics(dq, voff_nonneg(voff_lane + b0::dqa_voff(0, 0, b0::atom_gh(n, n, G), b0::atom_qt(n, n, G, nqp), nqp)));
    }
    static_for<2>([&](auto kh) {
        const int vo = voff_nonneg(b0::epi_voff_kh(l, w, kh, Hkv));
        static_for<8>([&](auto dt) {
            buf_store_ct_bf16(r_dv, vo + b0::epi_imm(dt), 0, dVt[dt][kh]);
            buf_store_ct_bf16(r_dk, vo + b0::epi_imm(dt), 0, scale_acc(dKt[dt][kh], scale));
        });
    });
    split_signal_tdm_lds<0>();
    split_wait_lds();
}
