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

// ---- optimisation variants of k_hk_bwd_g4 (hk/notes/bwd_v0_opt.md). Every knob defaults to 0, and with all of them
// 0 the code object is byte-identical to the pinned v0 (manifest sha256). None changes the kernel ABI (arguments,
// block, grid, static LDS size). hk/lab/bounds/bwd_v0_check.cpp replays each one (`-- variant <name>`).
//   HK_BWD_OPT_MASKED    diagonal steps use the FULL step's E-operand layout: all 32 tr16 issued in P4 (every wave
//                        kind, DEAD included) and left in flight past the signal, E WMMAs only `if (live)`
//   HK_BWD_OPT_ATOM      dQ atomic sub-bursts: 2 = 2 x 16 (P4 | after the P8 TDM, or after the readback with
//                        ONEPHASE), 4 = 4 x 8 (P4 before / after the E tr16 | after the TDM | after the readback)
//   HK_BWD_OPT_TDM3      wave 3 issues a real 16-B TDM op per step into its own cell (b0opt::DUMMY_OFF) instead of
//                        the `w < 3` role branch
//   HK_BWD_OPT_REGS      register holds (llvm.fake.use: no instruction, any VGPR) against the WMMA-source WAR
//                        v_nops: bit 0 = qA / dA held to the end of P2, bit 1 = dSa held to the end of P3, bit 2 =
//                        in both step parities (else the P = 0 step only), bit 3 = in the diagonal steps too (else
//                        FULL steps only; the diagonal loop then stays on the lambda). 7 = the variant `regs`
//   HK_BWD_OPT_ONEPHASE  one memory phase per step: P5 signal (<= 7 E tr16 uncovered), P7 wait, P8 TDM, P9
//                        readback, then P6 E; the op(s+3) descriptor SALU is built under P1's WMMAs
//   HK_BWD_OPT_ATOM_LATE with HK_BWD_OPT_ATOM=2: the second sub-burst after the P9 readback + dSa loads instead
//                        of between the P8 TDM and the readback (its issue hides under the readback latency)
//   HK_BWD_OPT_ENTRY     s_wait_dscnt 0 after the prologue readback: the loop-entry DS state then matches the
//                        back edge's, so the first P1 WMMAs of the step after the loop head wait for their own
//                        2-4 loads instead of 16 more (v0: the merge with the prologue path, 40 loads pending
//                        instead of 56, makes every first-P1 wait of that step 16 loads too deep)
//   HK_BWD_OPT_ADDR      the LDS base VGPRs of the P4 E-operand burst (at the step start) and of the P9 readback
//                        (under E; ONEPHASE: at the step start) computed early and pinned (empty asm, no
//                        instruction), so no DS burst waits 17-21 cycles for its address VALU
#ifndef HK_BWD_OPT_MASKED
#define HK_BWD_OPT_MASKED 0
#endif
#ifndef HK_BWD_OPT_ATOM
#define HK_BWD_OPT_ATOM 0
#endif
#ifndef HK_BWD_OPT_TDM3
#define HK_BWD_OPT_TDM3 0
#endif
#ifndef HK_BWD_OPT_REGS
#define HK_BWD_OPT_REGS 0
#endif
#ifndef HK_BWD_OPT_ONEPHASE
#define HK_BWD_OPT_ONEPHASE 0
#endif
#ifndef HK_BWD_OPT_ATOM_LATE
#define HK_BWD_OPT_ATOM_LATE 0
#endif
#ifndef HK_BWD_OPT_ENTRY
#define HK_BWD_OPT_ENTRY 0
#endif
#ifndef HK_BWD_OPT_ADDR
#define HK_BWD_OPT_ADDR 0
#endif
static_assert(HK_BWD_OPT_ATOM == 0 || HK_BWD_OPT_ATOM == 2 || HK_BWD_OPT_ATOM == 4, "HK_BWD_OPT_ATOM: 0, 2 or 4");
static_assert(HK_BWD_OPT_REGS >= 0 && HK_BWD_OPT_REGS <= 15, "HK_BWD_OPT_REGS: bit mask 0..15");
static_assert(!HK_BWD_OPT_ATOM_LATE || HK_BWD_OPT_ATOM == 2, "HK_BWD_OPT_ATOM_LATE needs HK_BWD_OPT_ATOM=2");
#define HK_BWD_OPT_ANY                                                                                               \
    (HK_BWD_OPT_MASKED || HK_BWD_OPT_ATOM || HK_BWD_OPT_TDM3 || HK_BWD_OPT_REGS || HK_BWD_OPT_ONEPHASE ||            \
     HK_BWD_OPT_ENTRY || HK_BWD_OPT_ADDR)
#if HK_BWD_OPT_ANY
#include "bwd/opt_idx.h"
// llvm.fake.use through an asm label (as sync.h reaches s_wait_dscnt): a register use with no instruction and no
// register-class constraint (unlike an empty inline asm, whose "v" operands must live in v0..v255). The register
// allocator keeps the operands' VGPRs allocated up to this point; the AsmPrinter emits only a comment.
extern "C" __device__ void hka_llvm_fake_use(...) __asm("llvm.fake.use");
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

#if HK_BWD_OPT_REGS
// HK_BWD_OPT_REGS: keep the VGPRs of the 8 WMMA A operands x[2][4] allocated up to this point (one llvm.fake.use
// each: no instruction). Written right after a phase's closing sched_barrier, so it cannot be scheduled into the
// phase, and the phase's softmax temporaries cannot be given the registers its last WMMAs read (4 v_nop each).
// A macro, not a function: the inliner drops the llvm.fake.use calls of an inlined callee, so the calls must be
// lexically in the kernel (bwd/v0_step.inc expanded at kernel scope).
#define HK_BWD_KEEP8(x)                                                                                              \
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
#endif

#if HK_BWD_OPT_ADDR
// HK_BWD_OPT_ADDR: materialize an LDS base address in a VGPR here (empty asm with a read-write VGPR operand: no
// instruction). Its VALU must run before this point, and later uses read the pinned copy (no rematerialization
// next to the DS burst).
HKA_DEV lds_u8 *pin_lds(const lds_u8 *p) {
    uint32_t a = (uint32_t) (uintptr_t) p;
    asm volatile("" : "+v"(a));
    return lds_ptr(a);
}
#endif

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
#if HK_BWD_OPT_TDM3
    // wave 3: the per-step dummy op (b0opt::DUMMY_OFF; constant address, slot stride 0), selected once per WG
    using T_DUM = tdm_tile<1, b0opt::DUMMY_COLS, b0opt::DUMMY_B>;
    const bool     dum  = role == b0::ISS_KV;
    const char    *rten = role == b0::ISS_DO ? (const char *) DO
                          : role == b0::ISS_Q ? (const char *) Q
                          : role == b0::ISS_LD ? (const char *) LD : (const char *) K;
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) (dum ? b0opt::dummy_src_byte(wk.b, Skv, Hkv, wk.hkv, kvw0)
                                           : b0::role_tile_byte(role, wk.b, Sq, Hq, wk.hkv, G, 0, 0));
    const int64_t  rqstep = dum ? 0 : b0::role_qstep(role, Sq, Hq), rhstep = dum ? 0 : b0::role_hstep(role, Sq, Hq);
    const u32x8    rg1 = dum ? tdm_g1<T_DUM>(0, 1)
                         : role == b0::ISS_LD ? tdm_g1<T_LD>(b0::role_stride_el(role, Hq), b0::role_rows(role))
                                              : tdm_g1<T_QDO>(b0::role_stride_el(role, Hq), b0::role_rows(role));
    const int      rlds0 = dum ? b0opt::DUMMY_OFF : b0::role_lds0(role), rlstep = dum ? 0 : b0::role_lds_step(role);
#else
    const char    *rten = role == b0::ISS_DO ? (const char *) DO : role == b0::ISS_Q ? (const char *) Q : (const char *) LD;
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) b0::role_tile_byte(role, wk.b, Sq, Hq, wk.hkv, G, 0, 0);
    const int64_t  rqstep = b0::role_qstep(role, Sq, Hq), rhstep = b0::role_hstep(role, Sq, Hq);
    const u32x8    rg1 = role == b0::ISS_LD ? tdm_g1<T_LD>(b0::role_stride_el(role, Hq), b0::role_rows(role))
                                            : tdm_g1<T_QDO>(b0::role_stride_el(role, Hq), b0::role_rows(role));
    const int      rlds0 = b0::role_lds0(role), rlstep = b0::role_lds_step(role);
#endif
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
#if HK_BWD_OPT_ADDR
    // HK_BWD_OPT_ADDR: P9 part 1 from bases computed earlier (rb_bases, pinned): the same loads as readback
    struct RbBase {
        const lds_u8 *q, *d, *l;
    };
    auto rb_bases = [&](int slot) {
        return RbBase{pin_lds(rows_base<XP>(lds0 + b0::lds_q(slot), l)), pin_lds(rows_base<XP>(lds0 + b0::lds_do(slot), l)),
                      pin_lds(lds_ptr(lds0 + b0::lds_ld(slot) + b0::ld_lane(l)))};
    };
    auto readback_at = [&](XSet &X, const RbBase &rb) {
        static_for<4>([&](auto dt) {
            static_for<2>([&](auto hh) {
                X.qA[hh][dt] = load_rows<XP>(rb.q, 16 * hh, 32 * dt);
                X.dA[hh][dt] = load_rows<XP>(rb.d, 16 * hh, 32 * dt);
            });
        });
        static_for<2>([&](auto hh) {
            static_for<4>([&](auto k) {
                X.ld[hh][k] = __builtin_bit_cast(f32x4, ds_load_b128_u32(rb.l, b0::ld_imm(hh, k)));
            });
        });
    };
#endif
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
#if HK_BWD_OPT_ATOM
    // HK_BWD_OPT_ATOM: one accumulator (qh, j) of dQ(s - 1) = 8 atomics (same instructions as dq_atomics); the
    // step body puts a sched_barrier after each sub-burst so that it stays where it is placed
    auto dq_atom8 = [&](const acc_t (&dq)[2][2], int vo, auto qh, auto j) {
        atomic_add_c<512>(r_dqa, vo + b0::dqa_imm(qh, j, 0), 0, dq[qh][j]);
    };
    constexpr std::integral_constant<int, 0> A0{};
    constexpr std::integral_constant<int, 1> A1{};
#endif

    // ---- one step (KIND: 0 FULL, 1 MASKED; X = this step's set, Xn = the readback target) --------------------------
#if !HK_BWD_OPT_REGS || !(HK_BWD_OPT_REGS & 8)
    auto step = [&](auto masked_c, auto par_c, int s, XSet &X, XSet &Xn) __attribute__((always_inline)) {
        constexpr bool MASKED = decltype(masked_c)::value;
        constexpr int  P      = decltype(par_c)::value;
#include "bwd/v0_step.inc"
    };
#endif

    // ---- main loop: x2 unrolled (n, nf even), FULL steps then the diagonal steps ---------------------------------
    XSet X0, X1;
    readback(X0, 0);
    static_for<2>([&](auto qh) { static_for<4>([&](auto c) { dSa[qh][c] = opnd_t{}; }); });
#if HK_BWD_OPT_ENTRY
    // HK_BWD_OPT_ENTRY: retire the prologue's DS loads (once per WG), so that the loop-entry DS state is the back
    // edge's (the 56 readback + dSa loads of the previous step) and not a merge with 40 pending prologue loads
    sched_barrier<0>();
    wait_dscnt<0>();
    sched_barrier<0>();
#endif
    constexpr auto F0 = std::false_type{};
    constexpr auto T1 = std::true_type{};
    constexpr std::integral_constant<int, 0> P0{};
    constexpr std::integral_constant<int, 1> P1{};
    int s = 0;
#if HK_BWD_OPT_REGS
    // the same four step instances as the #else branch, expanded at kernel scope (bwd/v0_step.inc header)
    (void) F0, (void) P0, (void) P1;
#define HK_BWD_STEP_AT(MASKED_, P_, S_, X_, XN_)                                                                     \
    constexpr bool MASKED  = MASKED_;                                                                                \
    constexpr int  P       = P_;                                                                                     \
    const int      s_at    = S_;                                                                                     \
    XSet          &X = X_, &Xn = XN_;
    for (; s < nf; s += 2) {
        {
            HK_BWD_STEP_AT(false, 0, s, X0, X1)
            {
                const int s = s_at;
#include "bwd/v0_step.inc"
            }
        }
        {
            HK_BWD_STEP_AT(false, 1, s + 1, X1, X0)
            {
                const int s = s_at;
#include "bwd/v0_step.inc"
            }
        }
    }
#if HK_BWD_OPT_REGS & 8
    for (; s < n; s += 2) {
        {
            HK_BWD_STEP_AT(true, 0, s, X0, X1)
            {
                const int s = s_at;
#include "bwd/v0_step.inc"
            }
        }
        {
            HK_BWD_STEP_AT(true, 1, s + 1, X1, X0)
            {
                const int s = s_at;
#include "bwd/v0_step.inc"
            }
        }
    }
#else
    for (; s < n; s += 2) { // diagonal steps through the lambda (no holds; kernel scope adds ~30 v_mov per step here)
        step(T1, P0, s, X0, X1);
        step(T1, P1, s + 1, X1, X0);
    }
#endif
#undef HK_BWD_STEP_AT
#else
    for (; s < nf; s += 2) {
        step(F0, P0, s, X0, X1);
        step(F0, P1, s + 1, X1, X0);
    }
    for (; s < n; s += 2) {
        step(T1, P0, s, X0, X1);
        step(T1, P1, s + 1, X1, X0);
    }
#endif

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
