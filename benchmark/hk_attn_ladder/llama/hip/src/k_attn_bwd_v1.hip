/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_attn_bwd_v1.hip -- HK gfx1250 d128 GQA attention backward v1 (hk/notes/bwd_v1_design.md; implementation
// note hk/notes/bwd_v1_impl.md). Same math, ABI, LDS map, dQ_acc layout and prep / convert kernels as v0
// (k_attn_bwd_v0.hip, bwd/idx.h); the main kernel's step is re-timed so that every memory op can run under a
// WMMA phase (expert scheduling mode build), plus head rotation of the dQ atomics. Kernels, in launch order:
//
//   k_hk_bwd1_prep / _prep_z  LD records {NL = -LSE log2e, delta} (+ _z: zero the WG's 32 KB of dQ_acc)
//                                                                         grid (Sq/64, Hq, B), block 256
//   k_hk_bwd1_g4              dK (scaled), dV, dQ_acc += dS K (fp32 atomics)  grid (Hkv, Skv/128, B), block 128
//   k_hk_bwd1_dqcvt           dq = bf16(scale dQ_acc)                       grid (Sq/32, Hq, B), block 256
//
// Every index comes from bwd1/idx.h (b1::, on top of bwd/idx.h b0::), which hk/lab/bounds/bwd_v1_check.cpp
// replays with the same -D knobs. Buffer accesses with TRUE extents, TDM tiles == tensor dims, int64 offsets,
// no hidden kernel arguments, static LDS, uniform barrier counts, TDM drained before exit.
//
// k_hk_bwd1_g4 step s (x4 unrolled: P4 = s & 3 is compile-time, so every ring slot / dS^T buffer / LDS offset is a
// constant; P = s & 1; design 2.-3.):
//   T   waves 0..2: TDM op(s+3) -> slot (s+3) & 3 (group 0 built in E2 of step s-1)
//   G0  S/dP(s) kv tile 0 (16 WMMA; their A operands do not die here) || dSa(s-1) 16 tr16 (buffer P^1) + dOT(s)
//       16 tr16 (slot s), 2 per WMMA gap
//   G1  S/dP(s) kv tile 1 (16 WMMA)          || softmax kv tile 0
//   C   dQ(s-1) = dSa K (16 WMMA)            || softmax kv tile 1; atomic voffset of tile s-1; s_wait_dscnt 0
//   ST  4 ds_store_b128 dS^T(s) -> buffer P (behind the natural va_vdst(0) on the cvt results)
//   E1  dV^T(s) (16 WMMA, kv-tile-major)     || QT(s) 16 tr16 in the first 8 gaps (no A operand dies there) +
//                                               the 32 dQ(s-1) atomics, 2 per gap (BWD1_ATOM)
//   SIG s_wait_tensorcnt 1; s_wait_dscnt NUNC (0 at NWIN 16); s_barrier_signal
//   E2  dK^T(s) (16 WMMA)                    || readback(s+1) 40 ds_load_b128 into the loop-carried X set
//   WAIT s_barrier_wait
// Diagonal steps (s >= nf): per-wave kind FULL / DIAG / DEAD (uniform branch), masked softmax argument, memory as
// bursts at the same protocol positions, s_wait_dscnt 0 at the signal; DEAD waves skip S/dP, softmax, dV/dK and the
// E-operand tr16 and store a zero dS^T, but keep dSa, dQ, atomics, readback and every barrier.
// Knobs: BWD1_SCHED, BWD1_ATOM, BWD1_NWIN, BWD1_HROT, BWD1_SKIP_DEAD (bwd1/idx.h); v1.1 variants (hk/notes/bwd_v11.md,
// all 0 = the validated v1 code): BWD1_OPT_MASKED (diagonal steps interleaved, every wave kind computes),
// BWD1_OPT_TDM3 (wave 3 issues a 16-B dummy TDM op per step: no role branch), BWD1_OPT_REGS (llvm.fake.use register
// holds against expert-mode WAR drains and WMMA-source v_nops; the steps are then expanded at kernel scope from
// bwd1/v1_step.inc, the same text the step lambda includes). Intended build: expert
// scheduling mode (-mllvm -amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD); correct in both modes
// (hk/build/bwd_v1/tools/expert_audit.py: 0 unguarded memory / ALU hazards in every loop of the expert builds).
#include "bwd1/idx.h"
#include "hk_attn/ext_bwd.h"
#include "hk_attn/prims.h"

using namespace hk_attn;

#if BWD1_OPT_REGS
// llvm.fake.use through an asm label (k_attn_bwd_v0.hip HK_BWD_OPT_REGS): a register use with no instruction and no
// register-class constraint (an empty inline asm "v" operand would have to live in v0..v255). The register allocator
// keeps the operand's VGPRs allocated up to this point. Only effective lexically in the kernel (the inliner drops the
// fake uses of an inlined callee), hence the kernel-scope step expansion below.
extern "C" __device__ void hka_llvm_fake_use(...) __asm("llvm.fake.use");
#define BWD1_KEEP8(x)                                                                                                \
    do {                                                                                                             \
        hka_llvm_fake_use((x)[0][0]);                                                                                \
        hka_llvm_fake_use((x)[0][1]);                                                                                \
        hka_llvm_fake_use((x)[0][2]);                                                                                \
        hka_llvm_fake_use((x)[0][3]);                                                                                \
        hka_llvm_fake_use((x)[1][0]);                                                                                \
        hka_llvm_fake_use((x)[1][1]);                                                                                \
        hka_llvm_fake_use((x)[1][2]);                                                                                \
        hka_llvm_fake_use((x)[1][3]);                                                                                \
    } while (0)
#define BWD1_KEEP8_1(x)                                                                                              \
    do {                                                                                                             \
        hka_llvm_fake_use((x)[0]);                                                                                   \
        hka_llvm_fake_use((x)[1]);                                                                                   \
        hka_llvm_fake_use((x)[2]);                                                                                   \
        hka_llvm_fake_use((x)[3]);                                                                                   \
        hka_llvm_fake_use((x)[4]);                                                                                   \
        hka_llvm_fake_use((x)[5]);                                                                                   \
        hka_llvm_fake_use((x)[6]);                                                                                   \
        hka_llvm_fake_use((x)[7]);                                                                                   \
    } while (0)
#define BWD1_KEEP4_U(x)                                                                                              \
    do {                                                                                                             \
        hka_llvm_fake_use((x)[0][0]);                                                                                \
        hka_llvm_fake_use((x)[0][1]);                                                                                \
        hka_llvm_fake_use((x)[1][0]);                                                                                \
        hka_llvm_fake_use((x)[1][1]);                                                                                \
    } while (0)
#define BWD1_KEEP_ACC4(x) BWD1_KEEP4_U(x)
#endif

namespace {

constexpr int G  = b0::G4;
constexpr int XP = b0::XP, SP = b0::SP;
using T_QDO = tdm_tile<b0::QS, b0::D, XP>;
using T_LD  = tdm_tile<1, 64, b0::LD_B, 4>;
using T_KV  = tdm_tile<b0::BKV, b0::D, XP>;
static_assert(T_QDO::w0 == b0::W0_QDO && T_LD::w0 == b0::W0_LD && T_KV::w0 == b0::W0_KV, "TDM w0 == idx.h");
static_assert(T_QDO::bytes == b0::QT_B && T_KV::bytes == b0::KV_B && T_LD::bytes == b0::LD_B, "TDM footprints");
static_assert(b1::RING == 4 && b1::NDS == 2, "slot = k & 3, dS buffer = s & 1");

struct XSet {
    opnd_t qA[2][4], dA[2][4];
    f32x4  ld[2][4];
};

// one sched group triple per WMMA: {1 WMMA, NV VALU|TRANS, NA VMEM, ND DS_READ}; counts of group g spread
// evenly (b1::share) over the phase's NW groups; SKIP leading groups carry no VALU (accumulators in flight)
template <int NW, int XV, int XA, int XD, int SKIPV = 0> HKA_DEV void pattern() {
    static_for<NW>([&](auto g) {
        sched_group_barrier<sg::WMMA, 1, 0>();
        if constexpr ((int) g >= SKIPV && XV > 0) {
            constexpr int nv = b1::share(XV, NW - SKIPV, (int) g - SKIPV);
            if constexpr (nv > 0) sched_group_barrier<sg::VALU | sg::TRANS, nv, 0>();
        }
        if constexpr (XA > 0) {
            constexpr int na = b1::share(XA, NW, g);
            if constexpr (na > 0) sched_group_barrier<sg::VMEM, na, 0>();
        }
        if constexpr (XD > 0) {
            constexpr int nd = b1::share(XD, NW, g);
            if constexpr (nd > 0) sched_group_barrier<sg::DS_READ, nd, 0>();
        }
    });
}

// opaque copy of an LDS address (an empty asm, no instruction): the per-segment lane bases of the loop are
// computed once and stay live in VGPRs (under pressure LLVM would otherwise re-add them inside the loop,
// and every such v_add feeding a DS address costs an expert-mode va_vdst wait)
HKA_DEV const lds_u8 *opaque_lds(const lds_u8 *p) {
    uint32_t a = (uint32_t) (uintptr_t) p;
    asm volatile("" : "+v"(a));
    return lds_ptr(a);
}

} // namespace

// ============================================================================================================
// k_hk_bwd1_prep[_z] / k_hk_bwd1_dqcvt: the v0 kernels (k_attn_bwd_v0.hip), unchanged apart from the names.
template <bool ZERO>
HKA_DEV void prep_body(const bf16 *DO, const bf16 *O, const float *LSE, float *LD, float *DQA, int Sq, int Hq,
                       int B
#if HK_STRIDES
                       , int64_t do_sb, int64_t do_ss, int64_t o_sb, int64_t o_ss
#endif
                       ) {
    __shared__ __attribute__((aligned(16))) unsigned char sdelta[b0::PREP_LDS_B];
    const int      t = (int) __builtin_amdgcn_workitem_id_x();
    const int      bx = (int) blockIdx.x, h = (int) blockIdx.y, b = (int) blockIdx.z;
    const int      q0 = b0::PREP_ROWS * bx;
    const uint32_t lds0 = lds_addr(sdelta);
    // whole-tensor resources: B Sq Hq D bf16 in either layout (dense; bwd1/idx.h HK_STRIDES)
    const rsrc_t   r_do = make_rsrc(DO, b0::qdo_bytes(B, Sq, Hq)), r_o = make_rsrc(O, b0::qdo_bytes(B, Sq, Hq));
    const rsrc_t   r_lse = make_rsrc(LSE, (int64_t) B * Hq * Sq * 4), r_ld = make_rsrc(LD, b0::ld_bytes(B, Hq, Sq));
    const int      c = b0::prep_chunk(t);
    u32x4          vdo[4], vo[4];
    static_for<4>([&](auto p) {
#if HK_STRIDES
        // dO and O each in their own layout (row byte < B Sq Hq D 2 < 2^31: host gate)
        const int row = q0 + b0::prep_row(t, p);
        vdo[p] = buf_load_b128(r_do, (int) b1::prep_row_byte_s(do_sb, do_ss, b, h, row) + 16 * c, 0);
        vo[p]  = buf_load_b128(r_o, (int) b1::prep_row_byte_s(o_sb, o_ss, b, h, row) + 16 * c, 0);
#else
        const int off = (int) b0::prep_row_byte(b, Sq, Hq, h, q0 + b0::prep_row(t, p)) + 16 * c;
        vdo[p] = buf_load_b128(r_do, off, 0);
        vo[p]  = buf_load_b128(r_o, off, 0);
#endif
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

// HK_STRIDES=1 (bwd1/idx.h): the int64 element strides (batch, seq) of each BSHD-family tensor follow the v1
// arguments, in this order (bwd_v1.py STRIDE_TENSORS): prep / prep_z dO, O; g4 Q, K, V, dO, dK, dV; dqcvt dQ
#if HK_STRIDES
#define BWD1_PREP_SARGS , int64_t do_sb, int64_t do_ss, int64_t o_sb, int64_t o_ss
#define BWD1_PREP_SVALS , do_sb, do_ss, o_sb, o_ss
#else
#define BWD1_PREP_SARGS
#define BWD1_PREP_SVALS
#endif
extern "C" __global__ __launch_bounds__(b0::PREP_THREADS) void k_hk_bwd1_prep(const bf16 *__restrict__ DO,
                                                                             const bf16 *__restrict__ O,
                                                                             const float *__restrict__ LSE,
                                                                             float *__restrict__ LD,
                                                                             float *__restrict__ DQA, int Sq,
                                                                             int Hq, int B BWD1_PREP_SARGS) {
    prep_body<false>(DO, O, LSE, LD, DQA, Sq, Hq, B BWD1_PREP_SVALS);
}
extern "C" __global__ __launch_bounds__(b0::PREP_THREADS) void k_hk_bwd1_prep_z(const bf16 *__restrict__ DO,
                                                                               const bf16 *__restrict__ O,
                                                                               const float *__restrict__ LSE,
                                                                               float *__restrict__ LD,
                                                                               float *__restrict__ DQA, int Sq,
                                                                               int Hq, int B BWD1_PREP_SARGS) {
    prep_body<true>(DO, O, LSE, LD, DQA, Sq, Hq, B BWD1_PREP_SVALS);
}
#undef BWD1_PREP_SARGS
#undef BWD1_PREP_SVALS

extern "C" __global__ __launch_bounds__(b0::CVT_THREADS) void k_hk_bwd1_dqcvt(const float *__restrict__ DQA,
                                                                             bf16 *__restrict__ DQ, float scale,
                                                                             int Sq, int Hq, int B
#if HK_STRIDES
                                                                             , int64_t dq_sb, int64_t dq_ss
#endif
                                                                             ) {
    __shared__ __attribute__((aligned(16))) unsigned char img[b0::CVT_LDS_B];
    const int      t = (int) __builtin_amdgcn_workitem_id_x();
    const int      qt = (int) blockIdx.x, qh = (int) blockIdx.y, b = (int) blockIdx.z;
    const uint32_t lds0 = lds_addr(img);
    const rsrc_t   r_src = make_rsrc(DQA, b0::dqa_bytes(B, Hq, Sq)), r_dq = make_rsrc(DQ, b0::qdo_bytes(B, Sq, Hq));
    const int      src = (int) b0::cvt_src_byte(b, Hq, qh, b0::nqp_of(Sq), qt);
    f32x4          x[4];
    static_for<4>([&](auto i) { x[i] = buf_load_f32x4(r_src, src + b0::cvt_src_off(t, i), 0); });
    static_for<4>([&](auto i) {
        const int   p = b0::cvt_src_off(t, i) / 4;
        const f32x4 y = x[i] * scale;
        ds_store_b64(lds_ptr(lds0), b0::cvt_lds(b0::cvt_q(p), b0::cvt_d(p)),
                     u32x2{cvt_pk_bf16(y[0], y[1]), cvt_pk_bf16(y[2], y[3])});
    });
    lds_barrier();
    static_for<2>([&](auto i) {
        const int row = b0::cvt_row(t, i), chunk = b0::cvt_chunk(t, i);
#if HK_STRIDES
        // dQ in its own layout (whole-tensor resource above: dense in both)
        buf_store_b128(r_dq, (int) b1::cvt_dst_byte_s(dq_sb, dq_ss, b, qh, qt, row, chunk), 0,
                       ds_load_b128_u32(lds_ptr(lds0), row * b0::CVT_PITCH + 16 * chunk));
#else
        buf_store_b128(r_dq, (int) b0::cvt_dst_byte(b, Sq, Hq, qh, qt, row, chunk), 0,
                       ds_load_b128_u32(lds_ptr(lds0), row * b0::CVT_PITCH + 16 * chunk));
#endif
    });
}

// ============================================================================================================
// k_hk_bwd1_g4: the main kernel (header comment). Kernel arguments (offsets) == v0: Q 0, K 8, V 16, DO 24,
// LD 32, DK 40, DV 48, DQA 56 (pointers), scale 64 (f32), Sq 68, Skv 72, Hq 76, Hkv 80, B 84, causal 88 (i32);
// HK_STRIDES=1 appends the int64 (batch, seq) element strides of Q 96, K 112, V 128, DO 144, DK 160, DV 176 (192 B).
extern "C" __global__ __launch_bounds__(b0::BLOCK) void k_hk_bwd1_g4(
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const bf16 *__restrict__ DO, const float *__restrict__ LD, bf16 *__restrict__ DK, bf16 *__restrict__ DV,
    float *__restrict__ DQA, float scale, int Sq, int Skv, int Hq, int Hkv, int B, int causal
#if HK_STRIDES
    , int64_t q_sb, int64_t q_ss, int64_t k_sb, int64_t k_ss, int64_t v_sb, int64_t v_ss, int64_t do_sb,
    int64_t do_ss, int64_t dk_sb, int64_t dk_ss, int64_t dv_sb, int64_t dv_ss
#endif
    ) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[b0::LDS_BYTES];
    lock_simd();
    (void) B; // host-gated; every extent below is per WG
    const uint32_t lds0 = lds_addr(smem);
    const int      l = lane_id(), w = wave_id();
    const b0::Work wk = b0::work_v0((int) blockIdx.x, (int) blockIdx.y, (int) blockIdx.z);
    const int      kvw0 = b0::kvw0_of(wk.blk), nqp = b0::nqp_of(Sq), cshift = Skv - Sq;
    const int      n = b0::n_all(kvw0, cshift, causal, nqp, G), nf = b0::n_full(kvw0, cshift, causal, nqp, G);
    if (n < G) return; // all waves, before any TDM / barrier; unreachable behind the host gate
    const int   rot = b1::rot_of(wk.blk, G);
    const float c1 = scale * b0::LOG2E;

    // ---- buffer resources (TRUE extents: the WG's dQ_acc group, its dK / dV slabs) ------------------------------
    const rsrc_t  r_dqa = make_rsrc((const char *) DQA + b0::dqa_grp_byte(wk.b, Hq, wk.hkv, G, Sq),
                                    b0::dqa_grp_bytes(G, Sq));
#if HK_STRIDES
    // dK / dV slabs, each in its own layout: base = row kvw0 of head hkv, extent = 127 seq strides + one row (the
    // WG's write footprint; inside the dense tensor in both layouts, bwd1/idx.h epi_bytes_s)
    (void) Hkv;
    const rsrc_t  r_dk = make_rsrc((const char *) DK + b1::kv_tile_byte_s(dk_sb, dk_ss, wk.b, wk.hkv, kvw0),
                                   b1::epi_bytes_s(dk_ss));
    const rsrc_t  r_dv = make_rsrc((const char *) DV + b1::kv_tile_byte_s(dv_sb, dv_ss, wk.b, wk.hkv, kvw0),
                                   b1::epi_bytes_s(dv_ss));
#else
    const int64_t epib = b0::epi_base_byte(wk.b, Skv, Hkv, wk.hkv, kvw0);
    const rsrc_t  r_dk = make_rsrc((const char *) DK + epib, b0::epi_bytes(Hkv));
    const rsrc_t  r_dv = make_rsrc((const char *) DV + epib, b0::epi_bytes(Hkv));
#endif

    // ---- TDM roles (v0): wave 0 dO, 1 Q, 2 LD records (one op per step each); wave 3 the K / V images ---------
    const int      role = w;
#if BWD1_OPT_TDM3
    // BWD1_OPT_TDM3 (k_attn_bwd_v0.hip HK_BWD_OPT_TDM3): wave 3's per-step op moves the 16 B of K row kvw0 (always in
    // bounds, read-only) into its own cell b0opt::DUMMY_OFF (nothing reads it; outside every window of the v1 map:
    // RING 4 <= RING_MAX); constant address (q / head / slot steps 0), tensor dims == tile dims (8 x 1)
    using T_DUM = tdm_tile<1, b0opt::DUMMY_COLS, b0opt::DUMMY_B>;
    const bool     dum  = role == b0::ISS_KV;
    const char    *rten = role == b0::ISS_DO ? (const char *) DO
                          : role == b0::ISS_Q ? (const char *) Q
                          : role == b0::ISS_LD ? (const char *) LD : (const char *) K;
#if HK_STRIDES
    // the role tensor's strides (dO: wave 0, Q: wave 1; LD and the dummy: unused 0); dummy = K row kvw0 in K's layout
    const int64_t  rsb = role == b0::ISS_DO ? do_sb : role == b0::ISS_Q ? q_sb : 0;
    const int64_t  rss = role == b0::ISS_DO ? do_ss : role == b0::ISS_Q ? q_ss : 0;
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) (dum ? b1::dummy_src_byte_s(k_sb, k_ss, wk.b, wk.hkv, kvw0)
                                           : b1::role_tile_byte_s(role, rsb, rss, wk.b, Sq, Hq, wk.hkv, G, 0, 0));
    const int64_t  rqstep = dum ? 0 : b1::role_qstep_s(role, rss, Sq, Hq), rhstep = dum ? 0 : b1::role_hstep_s(role, rss, Sq, Hq);
    const u32x8    rg1 = dum ? tdm_g1<T_DUM>(0, 1)
                         : role == b0::ISS_LD ? tdm_g1<T_LD>(b1::role_stride_el_s(role, rss), b0::role_rows(role))
                                              : tdm_g1<T_QDO>(b1::role_stride_el_s(role, rss), b0::role_rows(role));
#else
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) (dum ? b0opt::dummy_src_byte(wk.b, Skv, Hkv, wk.hkv, kvw0)
                                           : b0::role_tile_byte(role, wk.b, Sq, Hq, wk.hkv, G, 0, 0));
    const int64_t  rqstep = dum ? 0 : b0::role_qstep(role, Sq, Hq), rhstep = dum ? 0 : b0::role_hstep(role, Sq, Hq);
    const u32x8    rg1 = dum ? tdm_g1<T_DUM>(0, 1)
                         : role == b0::ISS_LD ? tdm_g1<T_LD>(b0::role_stride_el(role, Hq), b0::role_rows(role))
                                              : tdm_g1<T_QDO>(b0::role_stride_el(role, Hq), b0::role_rows(role));
#endif
    const int      rlds0 = dum ? b0opt::DUMMY_OFF : b0::role_lds0(role), rlstep = dum ? 0 : b0::role_lds_step(role);
#else
    const char    *rten = role == b0::ISS_DO ? (const char *) DO : role == b0::ISS_Q ? (const char *) Q : (const char *) LD;
#if HK_STRIDES
    // the role tensor's strides (dO: wave 0, Q: wave 1; LD: unused 0; wave 3 issues no ring op)
    const int64_t  rsb = role == b0::ISS_DO ? do_sb : role == b0::ISS_Q ? q_sb : 0;
    const int64_t  rss = role == b0::ISS_DO ? do_ss : role == b0::ISS_Q ? q_ss : 0;
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) b1::role_tile_byte_s(role, rsb, rss, wk.b, Sq, Hq, wk.hkv, G, 0, 0);
    const int64_t  rqstep = b1::role_qstep_s(role, rss, Sq, Hq), rhstep = b1::role_hstep_s(role, rss, Sq, Hq);
    const u32x8    rg1 = role == b0::ISS_LD ? tdm_g1<T_LD>(b1::role_stride_el_s(role, rss), b0::role_rows(role))
                                            : tdm_g1<T_QDO>(b1::role_stride_el_s(role, rss), b0::role_rows(role));
#else
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) b0::role_tile_byte(role, wk.b, Sq, Hq, wk.hkv, G, 0, 0);
    const int64_t  rqstep = b0::role_qstep(role, Sq, Hq), rhstep = b0::role_hstep(role, Sq, Hq);
    const u32x8    rg1 = role == b0::ISS_LD ? tdm_g1<T_LD>(b0::role_stride_el(role, Hq), b0::role_rows(role))
                                            : tdm_g1<T_QDO>(b0::role_stride_el(role, Hq), b0::role_rows(role));
#endif
    const int      rlds0 = b0::role_lds0(role), rlstep = b0::role_lds_step(role);
#endif
    auto           ring_g0 = [&](int k) { // group 0 of the ring op of step k (tile clamp(k)) -> slot k & 3
        const int qt = b0::tile_qt(k, n, G, nqp), gh = b1::tile_gh(k, n, G, rot);
        return tdm_g0(lds0 + (uint32_t) b0::role_lds_at(rlds0, rlstep, k & 3),
                      rbase + (uint64_t) b0::role_step_byte(0, rqstep, rhstep, qt, gh));
    };

    // ---- prologue: ops 0..2 (waves 0..2), K / V images (wave 3); dS^T buffer 1 zeroed (dQ(-1) = 0); P0 ------
    if (w < b0::ISS_KV) {
        tdm_load(ring_g0(0), rg1);
        tdm_load(ring_g0(1), rg1);
        tdm_load(ring_g0(2), rg1);
    } else {
#if HK_STRIDES
        // K and V each in their own layout: tile base at row kvw0 of head hkv, TDM row stride = the seq stride
        tdm_load(tdm_g0(lds0 + b0::OFF_K, (uint64_t) (uintptr_t) K + (uint64_t) b1::kv_tile_byte_s(k_sb, k_ss, wk.b, wk.hkv, kvw0)),
                 tdm_g1<T_KV>(k_ss, b0::BKV));
        tdm_load(tdm_g0(lds0 + b0::OFF_V, (uint64_t) (uintptr_t) V + (uint64_t) b1::kv_tile_byte_s(v_sb, v_ss, wk.b, wk.hkv, kvw0)),
                 tdm_g1<T_KV>(v_ss, b0::BKV));
#else
        const int64_t kvb = b0::kv_tile_byte(wk.b, Skv, Hkv, wk.hkv, kvw0);
        const u32x8   g1 = tdm_g1<T_KV>(b0::kv_stride_el(Hkv), b0::BKV);
        tdm_load(tdm_g0(lds0 + b0::OFF_K, (uint64_t) (uintptr_t) K + (uint64_t) kvb), g1);
        tdm_load(tdm_g0(lds0 + b0::OFF_V, (uint64_t) (uintptr_t) V + (uint64_t) kvb), g1);
#endif
    }
    {
        lds_u8 *dr = rows_base<SP>(lds0 + b0::lds_ds(1) + b0::ds_rows_wave(w), l);
        static_for<2>([&](auto kh) {
            static_for<2>([&](auto hh) { ds_store_b128(dr, b0::ct_imm(SP, 16 * kh, 16 * hh), zero_u32x4()); });
        });
    }
    if (w < b0::ISS_KV) wait_tensorcnt<b1::TCNT>(); // op(0), op(1) landed; op(2) flies
    else wait_tensorcnt<0>();
    lds_barrier();

    // ---- resident operands (v0) -----------------------------------------------------------------------------------
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
    // per-segment lane bases (loop-invariant VGPRs); every per-step slot / buffer offset below is a compile-time
    // constant (x4 unroll) that lands in the 16-bit DS offset field
    const lds_u8 *q_rows = opaque_lds(rows_base<XP>(lds0 + b0::OFF_Q, l)), *q_tr = opaque_lds(tr_base<XP>(lds0 + b0::OFF_Q, l));
    const lds_u8 *do_rows = opaque_lds(rows_base<XP>(lds0 + b0::OFF_DO, l)), *do_tr = opaque_lds(tr_base<XP>(lds0 + b0::OFF_DO, l));
    const lds_u8 *ld_b = opaque_lds(lds_ptr(lds0 + b0::OFF_LD + b0::ld_lane(l)));
    const lds_u8 *ds_tr = opaque_lds(tr_base<SP>(lds0 + b0::OFF_DS, l));
    lds_u8       *ds_st = (lds_u8 *) opaque_lds(rows_base<SP>(lds0 + b0::OFF_DS + b0::ds_rows_wave(w), l));
    // readback(t) items [LO, HI) into X from ring slot SLOT (b1 item order: qA / dA by dt, then the LD records)
    auto readback = [&](auto lo_c, auto hi_c, auto slot_c, XSet &X) __attribute__((always_inline)) {
        constexpr int LO = decltype(lo_c)::value, HI = decltype(hi_c)::value, SL = decltype(slot_c)::value;
        static_for<b1::RB_ITEMS>([&](auto it) {
            if constexpr ((int) it >= LO && (int) it < HI) {
                constexpr int kd = b1::rb_kind(it), hh = b1::rb_hh(it);
                if constexpr (kd == 0)
                    X.qA[hh][b1::rb_dt(it)] = load_rows<XP>(q_rows + SL * b0::QT_B, 16 * hh, 32 * b1::rb_dt(it));
                else if constexpr (kd == 1)
                    X.dA[hh][b1::rb_dt(it)] = load_rows<XP>(do_rows + SL * b0::QT_B, 16 * hh, 32 * b1::rb_dt(it));
                else
                    X.ld[hh][b1::rb_k(it)] = __builtin_bit_cast(f32x4, ds_load_b128_u32(ld_b, SL * b0::LD_B + b0::ld_imm(hh, b1::rb_k(it))));
            }
        });
    };
    opnd_t dSa[2][4]; // A operands of dQ(s - 1), read at the start of step s
    auto   load_dsa = [&](auto buf_c) {
        constexpr int BUF = decltype(buf_c)::value;
        static_for<2>([&](auto qh) {
            static_for<4>([&](auto c) { dSa[qh][c] = load_tr<SP>(ds_tr + BUF * b0::DS_B, 32 * c, 16 * qh); });
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
    // atomics [LO, HI) of the 32 (order qh, j, i), vo = their step voffset; each its own instruction
    auto atomics = [&](auto lo_c, auto hi_c, const acc_t (&dq)[2][2], int vo) __attribute__((always_inline)) {
        constexpr int LO = decltype(lo_c)::value, HI = decltype(hi_c)::value;
        static_for<b1::NATOM>([&](auto a) {
            if constexpr ((int) a >= LO && (int) a < HI) {
                constexpr int qh = a / 16, j = (a / 8) % 2, i = a % 8;
                buf_atomic_add_f32(r_dqa, vo + b0::dqa_imm(qh, j, i), 0, dq[qh][j][i]);
            }
        });
    };

    constexpr std::integral_constant<int, 0> I0{};
    constexpr std::integral_constant<int, 1> I1{};

    // ---- one step (masked_c: diagonal body; p4_c: s & 3, so ring slots / dS buffers / LDS offsets are
    // compile-time; X: tile s on entry, tile s + 1 on exit (one loop-carried set: the readback lands after X's
    // last use, in registers the store-side va_vdst(0) already cleared) ------------------------------------------
    u32x4 g0n = ring_g0(b1::LEAD); // group 0 of op(3), issued at the top of step 0
#if !(BWD1_OPT_REGS && BWD1_OPT_MASKED == 1)
    auto  step = [&](auto masked_c, auto p4_c, int s, XSet &X) __attribute__((always_inline)) {
        constexpr bool MASKED = decltype(masked_c)::value;
        constexpr int  P4 = decltype(p4_c)::value, P = P4 & 1, NX = (P4 + 1) & 3;
#if BWD1_OPT_MASKED == 2
        constexpr bool MDEAD = (int) decltype(masked_c)::value == 2; // masked_c 2: a diagonal step's DEAD-wave body
#endif
#define BWD1_KS 0
#include "bwd1/v1_step.inc"
#undef BWD1_KS
    };
#endif

    // ---- main loop: x4 unrolled (n, nf multiples of G = 4), FULL steps then the diagonal steps ------------------
    static_assert(G % 4 == 0, "x4 unroll needs n, nf multiples of 4");
    XSet X;
    readback(std::integral_constant<int, 0>{}, std::integral_constant<int, b1::RB_ITEMS>{}, std::integral_constant<int, 0>{}, X);
    constexpr auto F0 = std::false_type{};
    constexpr auto T1 = std::true_type{};
    constexpr std::integral_constant<int, 0> Q0{};
    constexpr std::integral_constant<int, 1> Q1{};
    constexpr std::integral_constant<int, 2> Q2{};
    constexpr std::integral_constant<int, 3> Q3{};
    int s = 0;
#if BWD1_OPT_REGS
    // BWD1_OPT_REGS: the FULL step instances expanded at kernel scope (the fake-use holds of v1_step.inc must be
    // lexically in the kernel); the diagonal steps too when every wave kind runs the FULL body (BWD1_OPT_MASKED=1)
#if BWD1_OPT_REGS & 8
    acc_t dq_c[2][2]; // dQ(s - 1) and its voffset, loop-carried (v1_step.inc); the initial values only feed the
    int   avo_c = 0;  // holds in G0 of step 0
    static_for<2>([&](auto qh) { static_for<2>([&](auto j) { dq_c[qh][j] = zero_acc(); }); });
#endif
#if BWD1_OPT_REGS & 16
    opnd_t qt_c[8];   // Q^T(s - 1) / dS(s - 1) packs (E2's operands), loop-carried through G0 of step s
    u32x4  du_c[2][2];
    static_for<8>([&](auto dt) { qt_c[dt] = opnd_t{}; });
    static_for<2>([&](auto hh) { static_for<2>([&](auto kh) { du_c[hh][kh] = zero_u32x4(); }); });
#endif
#define BWD1_STEP_AT(MASKED_, P4_, S_)                                                                               \
    constexpr bool MASKED = MASKED_;                                                                                 \
    constexpr int  P4 = P4_, P = P4 & 1, NX = (P4 + 1) & 3;                                                          \
    const int      s_at = S_;
#define BWD1_KS 1
    (void) F0;
#if BWD1_OPT_MASKED == 2
#define BWD1_MD0 constexpr bool MDEAD = false;
#else
#define BWD1_MD0
#endif
    for (; s < nf; s += 4) {
        { BWD1_STEP_AT(false, 0, s) BWD1_MD0 { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
        { BWD1_STEP_AT(false, 1, s + 1) BWD1_MD0 { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
        { BWD1_STEP_AT(false, 2, s + 2) BWD1_MD0 { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
        { BWD1_STEP_AT(false, 3, s + 3) BWD1_MD0 { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
    }
#undef BWD1_MD0
#if BWD1_OPT_MASKED == 1
    (void) T1, (void) Q0, (void) Q1, (void) Q2, (void) Q3;
    for (; s < n; s += 4) {
        { BWD1_STEP_AT(true, 0, s) { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
        { BWD1_STEP_AT(true, 1, s + 1) { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
        { BWD1_STEP_AT(true, 2, s + 2) { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
        { BWD1_STEP_AT(true, 3, s + 3) { const int s = s_at;
#include "bwd1/v1_step.inc"
        } }
    }
#endif
#undef BWD1_KS
#undef BWD1_STEP_AT
#else
    for (; s < nf; s += 4) {
        step(F0, Q0, s, X);
        step(F0, Q1, s + 1, X);
        step(F0, Q2, s + 2, X);
        step(F0, Q3, s + 3, X);
    }
#endif
#if BWD1_OPT_MASKED == 2
    // BWD1_OPT_MASKED=2: per diagonal step a wave-uniform kind branch between the DEAD body (masked_c 2: v1's burst
    // body, no compute) and the interleaved live body with the mask (masked_c 1); lambda steps (no holds)
    auto mstep = [&](auto p4_c, int sm_) __attribute__((always_inline)) {
        if (b0::wave_kind(kvw0, w, b0::tile_qt(sm_, n, G, nqp), cshift, causal) == b0::K_DEAD)
            step(std::integral_constant<int, 2>{}, p4_c, sm_, X);
        else
            step(std::integral_constant<int, 1>{}, p4_c, sm_, X);
    };
    (void) T1;
    for (; s < n; s += 4) {
        mstep(Q0, s);
        mstep(Q1, s + 1);
        mstep(Q2, s + 2);
        mstep(Q3, s + 3);
    }
#elif !(BWD1_OPT_REGS && BWD1_OPT_MASKED == 1)
    for (; s < n; s += 4) {
        step(T1, Q0, s, X);
        step(T1, Q1, s + 1, X);
        step(T1, Q2, s + 2, X);
        step(T1, Q3, s + 3, X);
    }
#endif

    // ---- drain: dQ(n - 1) (dSa from buffer (n - 1) & 1 = 1) and its atomics; epilogue; TDM drain + barrier -------
    {
        load_dsa(std::integral_constant<int, 1>{});
        acc_t dq[2][2];
        dq_wmma(dq);
        const int vo = voff_nonneg(voff_lane + b0::dqa_voff(0, 0, b1::atom_gh(n, n, G, rot), b1::atom_qt(n, n, G, nqp), nqp));
        static_for<2>([&](auto qh) {
            static_for<2>([&](auto j) { atomic_add_c<512>(r_dqa, vo + b0::dqa_imm(qh, j, 0), 0, dq[qh][j]); });
        });
    }
#if HK_STRIDES
    // dK in dK's layout, dV in dV's: per-lane voffsets with the int32 byte row strides (host gate: slab_ok)
    const int dk_rb = (int) (2 * dk_ss), dv_rb = (int) (2 * dv_ss);
    static_for<2>([&](auto kh) {
        const int vk = voff_nonneg(b1::epi_voff_kh_s(l, w, kh, dk_rb)), vv = voff_nonneg(b1::epi_voff_kh_s(l, w, kh, dv_rb));
        static_for<8>([&](auto dt) {
            buf_store_ct_bf16(r_dv, vv + b0::epi_imm(dt), 0, dVt[dt][kh]);
            buf_store_ct_bf16(r_dk, vk + b0::epi_imm(dt), 0, scale_acc(dKt[dt][kh], scale));
        });
    });
#else
    static_for<2>([&](auto kh) {
        const int vo = voff_nonneg(b0::epi_voff_kh(l, w, kh, Hkv));
        static_for<8>([&](auto dt) {
            buf_store_ct_bf16(r_dv, vo + b0::epi_imm(dt), 0, dVt[dt][kh]);
            buf_store_ct_bf16(r_dk, vo + b0::epi_imm(dt), 0, scale_acc(dKt[dt][kh], scale));
        });
    });
#endif
    split_signal_tdm_lds<0>();
    split_wait_lds();
}
