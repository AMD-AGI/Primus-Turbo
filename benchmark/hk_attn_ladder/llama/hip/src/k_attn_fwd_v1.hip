/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_attn_fwd_v1 -- HK gfx1250 flash-attention forward v1 for Llama-3.1 attention: head dim 128, bf16
// BSHD, GQA with G in {4, 8, 16} (a power of two: f1::hkv_of shifts), bottom-right causal (query p sees
// kv <= p + Skv - Sq). The kernel also has the full-attention path (causal = 0), proven by the checker, but the
// host gate (fwd_v1.py) does not admit it until a GPU check of it exists.
//
//   O[b, p, hq, :] = softmax_kv(scale Q K^T) V (bf16),  LSE[b, hq, p] = ln sum_kv exp(scale s) (fp32)
//
// v0's item (k_attn_fwd_v0: 64 positions x one head quad, 4 waves = 1 per SIMD, Q^T resident, S^T = K Q^T,
// O^T += V^T P^T, depth-4 TDM ring with one 32-row op per wave per stage, one split-barrier phase per
// stage) run by a PERSISTENT grid. Every index formula is fwd1/idx.h (f1::, replayed on the host by
// hk/lab/bounds/fwd_v1_check.cpp); notes hk/notes/fwd_v1_impl.md.
//
// grid (W, 1, 1) with W = #CUs (MODE_PAIRS: zig-zag pairs of q blocks, heavy first, XCD-major) or W =
// #items (MODE_ITEMS: v0's longest-first grid, one item per workgroup); block 128; static LDS 320 KB.
// Workgroup g runs items wg_item(g, 0 .. J-1). Ring slot, TDM op stream and barrier phases are
// workgroup-lifetime counters over the global stage T = T0_j + t: every stage of every item has
// WAIT(P_{T+1}) + op(T+3) at its even barrier point and tensorcnt + SIGNAL(P_{T+2}) at its odd one, so the
// next item's first three stages are loaded under the current item's last ones; Q(j+1) is issued at the
// even point of item j's last stage (tensorcnt 2 at its odd point, 1 at the next item's prologue) into the
// wave's Q region, which is separate from its O image region. Exit: tensorcnt 0, WAIT(P_{N+1}).
//
// Iteration c (32-kv chunk of the current item; S^T double-buffered by chunk parity):
//   A  [FWD1_MDS 1: K(c+1) + V(c) burst]  [MDS 0: V(c) burst]  [MDS 3: none]
//   B  QK(c+1) -> S[c+1 & 1]  ||  exp of q-tiles 0..2, P^T pack of 0..1  (+ op(T+3) descriptor, even c)
//   D  even: WAIT + [Q(j+1)] + op(T+3);  odd: tensorcnt 1 (2 in the last stage) + LDS release + SIGNAL
//   E  [MDS 0: K(c+2) burst]  [MDS 2: K(c+2) + V(c+1) burst into the other V set]
//      [MDS 3, even c only: K(c+2), V(c+1), K(c+3), V(c+2): one 64-load burst per stage]
//   F1 PV(c) q-tiles 0, 1 (+ row-sum WMMAs) || SM1(c+1) (max, rescale ballots), P^T pack of q-tile 2
//   F2 PV(c) q-tiles 2, 3 || exp + pack of q-tile 3
//   C  lazy rescale of chunk c+1 (the only branch in a stage: wave-uniform, rarely taken, out of line)
// Per item: prologue (tensorcnt 1 for its Q, Q^T burst, K(0) [+K(1), V(0)] bursts, QK(0), SM1(0)), stages
// [0, n-2) (op(T+3) = own stage t+3, or the stream's first stage after the item at t = n-3), stage n-2
// (op = second stream stage; masked SM1(2n-2)), stage n-1 (Q(j+1) + third stream stage; masked
// SM1(2n-1); no QK(2n)), epilogue. No wave-dependent branch anywhere; all branches are WG-uniform.
//
// Compile-time levers (each a cc.sh -D flag; default first; hk/build/fwd_v1/build_variants.sh builds the A/B set,
// hk/notes/fwd_v1_impl.md section 4 gives the cycle estimate of each):
//   FWD1_MDS    2 | 0 | 1 | 3  DS bursts (fwd1/idx.h): one merged burst at E (K(c+2) + V(c+1), V double-buffered)
//                          / v0's two bursts / one merged burst at A (K(c+1) + V(c)) / one burst per STAGE at the
//                          even E (K(c+2), V(c+1), K(c+3), V(c+2); 2 K sets, 3 V sets, +128 VGPR; FWD1_OPT_UNROLL 1
//                          so the third V set alternates without copies; hk/notes/fwd_v11.md)
//   FWD1_LSUM   1 | 0      row sums: one WMMA per q-tile with an all-ones A (sums of the bf16 P^T the PV WMMA
//                          consumes; no row-sum VALU) / per-lane fp32 v_pk_add (v0). LSUM 1 changes the LSE by
//                          at most -ln(1 - 2^-9) = 1.955e-3 nat vs fp32 sums (lab-like inputs: max 0.8-1.6e-3,
//                          mean 0.5-2.2e-4; FlyDSL measures 1.3-3.8e-3 max). LSUM 0 + SM1L 0 is v0's arithmetic
//                          operation for operation (bit-identical O / LSE); LSUM 0 + SM1L 1 differs from v0 only
//                          when a lazy rescale triggers (never on lab's randn inputs)
//   FWD1_SM1L   1 | 0      SM1: lean (own-half max vs m2th = (m2 + TH2) c2inv, one compare per tile; the cross-
//                          half max and m2n only inside the rescale block) / v0 (cross-half max, per-tile select)
//   FWD1_FSPLIT 1 | 0      F split after SM1 (decision chain under PV 0..1) / F as one scheduling region
//   FWD1_F1SGB  3 | 0      F1 interleave {1 WMMA, 3 VALU} per WMMA (sched_group_barrier; FSPLIT 1 only) / none
//   FWD1_EPI    0 | 1      epilogue before the next item's front / its O normalize under the next item's QK(0)
//                          (1 is not admitted: 22-26 SGPR spills)
// Runtime: kernel argument `mode` (f1::MODE_PAIRS persistent zig-zag pairs, f1::MODE_ITEMS v0's grid); every
// runtime divisor of the schedule decode is a host-computed magic number (f1::fdiv, s_mul_u64 + s_lshr_b64).
//
// FWD1_OPT_* (agent fwd-v11, hk/notes/fwd_v11.md): the W3b fixes of k_attn_fwd_v0 (hk/notes/fwd_v0_opt.md) that
// v1's levers did not cover, as compile-time variants, each default 0. With every one 0 (and FWD1_MDS 0..2) the code
// objects are the validated ones byte for byte (lsum0 = -DFWD1_LSUM=0; the default build). None changes an
// arithmetic operation, its operands or an accumulation order: every variant gives the o / lse bits of the same
// build without it. FWD1_MDS 3 changes where the K / V operand loads sit in the barrier program (proven by
// fwd1/idx.h check_protocol and the checker); the other variants do not touch the barrier / TDM / LDS program.
//   RESC    (SM1L 1) W3b RESC: the rescale decision as ONE compare, d = max_h (mx_h - m2th_h) > 0 (one v_cmp to VCC,
//           branch on VCC; decided in F1 under PV 0..1), instead of 4 per-q-tile ballots to SGPRs ORed by SALU at C;
//           the per-q-tile ballots ballot(mx_h > m2th_h) move into the rarely taken block. Exact: fp32 denormals are
//           on (.amdhsa_float_denorm_mode_32 3), so x - y > 0 <=> x > y for all non-NaN x, y (inf - inf = NaN
//           compares false like inf > inf), and v_max_num drops NaN: the block runs exactly when SM1L's OR is set.
//   NM2     (SM1L 1) the running max without per-chunk negation / phi moves: 1 = W3b's q-tile pairs m2p[2] (the
//           tails take -m2 as one half of the pair: op_sel broadcast + neg modifier); 2 = the broadcast pairs
//           {-m2, -m2} kept as state (the fma operand as is) and c2 in a VGPR that cannot be rematerialized from its
//           SGPR every chunk. The rarely taken block writes through an identity DPP move (InstCombine cannot push
//           the tails' fneg through the loop phi). Same fma operands, same bits.
//   RSB     (LSUM 0) the TAIL=0 placement of the production p1 build: tail + row sum of q-tiles 0..2 and the P^T
//           pack of q-tile 2 in the QK run B (they overlap the WMMA drain + 22 before the D/E burst; W3b measured
//           TAIL 1 at +1.9 % inside `all`), F keeps SM1 and q-tile 3.
//   UNROLL  the steady stage loop runs 2 stages per iteration (one back-edge per 2 stages; an odd stage count runs
//           stage 0 first). Required by FWD1_MDS 3.
//
// HK_STRIDES 0 | 1 (agent fwd-strides, hk/notes/fwd_v1_strides.md): 1 = Q, K, V and O each in either layout, chosen per
// tensor by the host: BSHD contiguous or SBHD storage ([S, B, H, D] contiguous seen as [B, S, H, D]; Megatron's q / k / v
// / o), so the Turbo glue makes no layout copy. Eight int64 kernel arguments follow the 30 of the default build, in this
// order (the element strides of batch and seq of each tensor; head stride D, element stride 1):
//   sQb, sQs, sKb, sKs, sVb, sVs, sOb, sOs   (kernarg offsets 152, 160, ..., 208; 216 B in all)
// The seq stride is the TDM row stride of the tensor's ops (Q: sQs; a K / V wave: its role's), every element offset is
// b sb + s ss + h D (fwd1/idx.h *_s, int64, < 2^30 by the host gate), the O resource keeps the whole tensor's extent.
// Nothing else changes: same item schedule, barrier / TDM / LDS program, arithmetic and bits as the build without it
// (o / lse identical whatever the layouts). Composes with every FWD1_* / FWD1_OPT_* flag. With HK_STRIDES 0 (default)
// every line of the strided path is compiled out: the code objects are the validated ones byte for byte.
#include "fwd1/idx.h"
#include "hk_attn/ext_fwd.h"
#include "hk_attn/prims.h"

#pragma clang fp contract(off)

#ifndef FWD1_LSUM
#define FWD1_LSUM 1
#endif
#ifndef FWD1_SM1L
#define FWD1_SM1L 1
#endif
#ifndef FWD1_FSPLIT
#define FWD1_FSPLIT 1
#endif
#ifndef FWD1_F1SGB
#define FWD1_F1SGB 3
#endif
#ifndef FWD1_OPT_RESC
#define FWD1_OPT_RESC 0
#endif
#ifndef FWD1_OPT_NM2
#define FWD1_OPT_NM2 0
#endif
#ifndef FWD1_OPT_RSB
#define FWD1_OPT_RSB 0
#endif
#ifndef FWD1_OPT_UNROLL
#define FWD1_OPT_UNROLL 0
#endif
#ifndef HK_STRIDES
#define HK_STRIDES 0
#endif
static_assert(HK_STRIDES == 0 || HK_STRIDES == 1, "HK_STRIDES 0 | 1");
static_assert(FWD1_MDS >= 0 && FWD1_MDS <= 3 && (FWD1_EPI == 0 || FWD1_EPI == 1), "FWD1 variant");
static_assert((!FWD1_OPT_RESC && !FWD1_OPT_NM2) || FWD1_SM1L, "FWD1_OPT_RESC / NM2 need FWD1_SM1L 1");
static_assert(FWD1_OPT_NM2 >= 0 && FWD1_OPT_NM2 <= 2, "FWD1_OPT_NM2 0 | 1 | 2");
static_assert(!FWD1_OPT_RSB || !FWD1_LSUM, "FWD1_OPT_RSB is the LSUM 0 row-sum placement");
static_assert(FWD1_MDS != 3 || (FWD1_OPT_UNROLL && FWD1_EPI == 0), "FWD1_MDS 3 needs FWD1_OPT_UNROLL 1, FWD1_EPI 0");

#define FWD_INL __attribute__((always_inline))

using namespace hk_attn;

namespace {

constexpr int NH = f1::NH, NDK = f1::NDK, NDT = f1::NDT;
template <int N> using ic = std::integral_constant<int, N>;

static_assert(f1::BLOCK == 128 && f1::LDS_BYTES == 327680, "4 waves, one workgroup per CU");

struct Tdm { // one TDM op: groups 0 / 1 (wave-uniform SGPRs)
    u32x4 g0;
    u32x8 g1;
};
HKA_DEV Tdm make_tdm(uint32_t lds, uint64_t ga, const f1::G1 &g) {
    return Tdm{tdm_g0(lds, ga), tdm_g1_words(g.w)};
}
HKA_DEV void issue(const Tdm &d) {
    tdm_load(d.g0, d.g1);
}

// max over the 16 values of one query in this lane (kv tiles 0 and 1): 7 v_max3 + 1 v_max
HKA_DEV float max16(const acc_t &a, const acc_t &b) {
    const float x0 = max3(a[0], a[1], a[2]), x1 = max3(a[3], a[4], a[5]), x2 = max3(a[6], a[7], b[0]);
    const float x3 = max3(b[1], b[2], b[3]), x4 = max3(b[4], b[5], b[6]);
    return __builtin_fmaxf(max3(x0, x1, x2), max3(x3, x4, b[7]));
}

// SGPR value fence (no instruction): x is "redefined" here, so nothing computed from it is placed before
// this point by IR passes; pins the op descriptor and the next stage's burst bases into region B
HKA_DEV void sgpr_fence(int &x) {
    asm volatile("" : "+s"(x));
}

#if FWD1_OPT_NM2
// identity v_mov_b32_dpp quad_perm:[0,1,2,3] (exact): the rarely taken block's new m2 goes through it, so InstCombine
// cannot push the tails' fneg(m2) through the loop phi (fwd_v0 HK_FWD_OPT_RESC's device); NM2 2 also pins c2 in a VGPR
HKA_DEV float opaque_f32(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0xE4, 0xF, 0xF, false));
}
#endif

// The kernel arguments as laid out in the kernarg segment (natural alignment; == the ABI of the entry,
// checked by hk/lab/test_ext_fwd1.py against the code object's metadata). Parameters that are needed only at
// item boundaries (decode, output pointers) are re-read from the segment there through an opaque copy of the
// segment pointer (scalar loads), so they hold no SGPR across the stage loop (no SGPR spill).
struct KArgs {
    const __bf16 *Q, *K, *V;
    __bf16       *O;
    float        *LSE;
    long long     o_bytes, lse_bytes;
    int           B, Sq, Skv, Hq, Hkv, G, nqb, nhq4, nkv_st, cst, causal;
    float         c2, c2inv;
    int           W, mode;
    uint32_t      mW;   // fdiv magic numbers (f1::fdiv_of, host-computed): W, max(nqb / 2, 1), nhq4, B nhq4
    int           shW;
    uint32_t      mNP;
    int           shNP;
    uint32_t      mH4;
    int           shH4;
    uint32_t      mNH;
    int           shNH;
#if HK_STRIDES
    long long     sQb, sQs, sKb, sKs, sVb, sVs, sOb, sOs; // element strides (batch, seq) of Q, K, V, O
#endif
};
static_assert(__builtin_offsetof(KArgs, B) == 56 && __builtin_offsetof(KArgs, c2) == 100 &&
                  __builtin_offsetof(KArgs, mode) == 112 && __builtin_offsetof(KArgs, shNH) == 144 &&
                  sizeof(KArgs) == (HK_STRIDES ? 216 : 152),
              "kernarg layout (148 B of arguments; the struct pads to 8)");
#if HK_STRIDES
static_assert(__builtin_offsetof(KArgs, sQb) == 152 && __builtin_offsetof(KArgs, sKb) == 168 &&
                  __builtin_offsetof(KArgs, sVb) == 184 && __builtin_offsetof(KArgs, sOs) == 208,
              "HK_STRIDES kernarg layout: 8 int64 strides at 152 .. 215 (fwd_v1.py STRIDE_ARGS)");
#endif
typedef const __attribute__((address_space(4))) KArgs *kargs_p;
// the segment pointer, made opaque (no instruction): loads through it are not merged with the entry's
// argument loads, nor hoisted out of the item loop
HKA_DEV kargs_p kargs() {
    kargs_p p = (kargs_p) __builtin_amdgcn_kernarg_segment_ptr();
    asm volatile("" : "+s"(p));
    return p;
}

#if FWD1_LSUM
// A operand of all-ones (bf16 1.0 = 0x3F80): D[m][n] = sum_k B[k][n] + C[m][n] in every row m
HKA_DEV opnd_t ones_opnd() {
    const u32x8 o = {0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u,
                     0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u};
    return __builtin_bit_cast(opnd_t, o);
}
#endif

} // namespace

extern "C" __global__ __launch_bounds__(f1::BLOCK) void k_attn_fwd_v1(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long o_bytes,
    long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int G, int nqb, int nhq4, int nkv_st,
    int cst, int causal, float c2, float c2inv, int W, int mode, uint32_t mW, int shW, uint32_t mNP, int shNP,
    uint32_t mH4, int shH4, uint32_t mNH, int shNH
#if HK_STRIDES
    ,
    long long sQb, long long sQs, long long sKb, long long sKs, long long sVb, long long sVs, long long sOb,
    long long sOs
#endif
) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[f1::LDS_BYTES];
    lock_simd();

    // ---- identity (wave-uniform except the lane) ------------------------------------------------------------
    const int      w       = wave_id();
    const int      l       = lane_id();
    const int      rv      = f1::op_role_v(w); // 0: this wave moves K halves, 1: V halves (SALU selects)
    const uint32_t lds0    = lds_addr(smem);
    const uint64_t kv_base = (uint64_t) (uintptr_t) (rv != 0 ? V : K);
#if HK_STRIDES
    // the role tensor's seq stride: the TDM row stride of this wave's K / V ops, held across the stage loop in one
    // SGPR as Hkv is in the default build (f1::kv_ss32: exact, the host keeps every stride < 2^31); the batch strides
    // and Q / O strides are re-read at item boundaries (lay_*)
    const uint32_t kv_ss = f1::kv_ss32(rv != 0 ? sVs : sKs);
    const uint32_t sst   = f1::kv_sst_s(kv_ss); // K / V elements per 64-row stage
    auto lay_kv = [&](kargs_p ka) FWD_INL { return rv != 0 ? f1::Lay{ka->sVb, ka->sVs} : f1::Lay{ka->sKb, ka->sKs}; };
    auto lay_q  = [&](kargs_p ka) FWD_INL { return f1::Lay{ka->sQb, ka->sQs}; };
    auto lay_o  = [&](kargs_p ka) FWD_INL { return f1::Lay{ka->sOb, ka->sOs}; };
#else
    const uint32_t sst     = f1::kv_sst(Hkv); // K / V elements per 64-row stage
#endif
    // schedule / shape parameters re-read from the kernarg segment (SGPR, short-lived)
    auto sched = [&](kargs_p ka) FWD_INL {
        return f1::Sched{ka->mode, ka->W, ka->B, ka->nhq4, ka->nqb, f1::Fdiv{ka->mW, ka->shW}, f1::Fdiv{ka->mNP, ka->shNP},
                         f1::Fdiv{ka->mH4, ka->shH4}, f1::Fdiv{ka->mNH, ka->shNH}};
    };
    auto shape = [&](kargs_p ka) FWD_INL {
        return f1::Shape{ka->B, ka->Sq, Skv, ka->Hq, Hkv, ka->G, ka->nqb, ka->nhq4, ka->nkv_st, ka->cst, causal};
    };
    const int g = (int) __builtin_amdgcn_workgroup_id_x();
    // ---- TDM descriptors (pure SALU: both candidates, then scalar selects) ----------------------------------
    // K / V op (compact form f1::OpC: d1 > 0 real, 0 dummy) at global stage T
    auto op_desc = [&](const f1::OpC &o, int T) FWD_INL -> Tdm {
        const bool     real = o.d1 > 0;
#if HK_STRIDES
        const f1::G1   a = f1::kv_g1_s(rv, kv_ss, o.d1), d = f1::dummy_g1(rv);
#else
        const f1::G1   a = f1::kv_g1(rv, Hkv, o.d1), d = f1::dummy_g1(rv);
#endif
        f1::G1         s1{};
        for (int i = 0; i < 8; ++i) s1.w[i] = real ? a.w[i] : d.w[i];
        const uint32_t la_r = (uint32_t) f1::kv_op_lds(w, T), la_d = (uint32_t) f1::dummy_off(w);
        return make_tdm(lds0 + (real ? la_r : la_d), kv_base + 2ull * (uint64_t) o.el, s1);
    };
    // Q op (compact): the item's Q tile into the wave's Q region, or a 16-B dummy of Q row 0 into its Q dummy
    // cell; Hq / Q from the stash
    auto q_desc = [&](const f1::OpC &o) FWD_INL -> Tdm {
        const bool     real = o.d1 > 0;
        const kargs_p  ka = kargs();
#if HK_STRIDES
        const f1::G1   a = f1::q_g1_s(ka->sQs, o.d1), d = f1::dummy_g1(0);
#else
        const f1::G1   a = f1::q_g1(ka->Hq, o.d1), d = f1::dummy_g1(0);
#endif
        f1::G1         s1{};
        for (int i = 0; i < 8; ++i) s1.w[i] = real ? a.w[i] : d.w[i];
        const uint32_t la_r = (uint32_t) f1::q_op_lds(w), la_d = (uint32_t) f1::q_dummy_off(w);
        return make_tdm(lds0 + (real ? la_r : la_d), (uint64_t) (uintptr_t) ka->Q + 2ull * (uint64_t) o.el, s1);
    };

    // ---- LDS operand bursts (per-lane base VGPR + compile-time DS immediates) --------------------------------
    // burst bases of global stage T (one v_add each; computed a stage ahead, off the DS address path)
    auto kbase = [&](int T) FWD_INL { return lds0 + (uint32_t) (f1::k_slot_off(T) + f1::k_lane(l)); };
    auto vbase = [&](int T) FWD_INL { return lds0 + (uint32_t) (f1::v_slot_off(T) + f1::v_lane(l)); };
    // K A operands of stage half HC at base kb (consumption order of qk: k outer, kt inner)
    auto ld_k = [&](opnd_t(&kk)[2][NDK], uint32_t kb, auto HC) FWD_INL {
        const lds_u8 *kp = lds_ptr(kb);
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                kk[kt][k] = cat(ds_load_b128(kp, f1::k_imm(HC, kt, k, 0)), ds_load_b128(kp, f1::k_imm(HC, kt, k, 1)));
            });
        });
    };
    // V^T A operands of stage half HC at base vb (tr16 at pitch 288)
    auto ld_v = [&](opnd_t(&vv)[NDT], uint32_t vb, auto HC) FWD_INL {
        const lds_u8 *vp = lds_ptr(vb);
        static_for<NDT>([&](auto dt) {
            vv[dt] = cat(ds_load_tr16_b128(vp, f1::v_imm(HC, 0, dt)), ds_load_tr16_b128(vp, f1::v_imm(HC, 1, dt)));
        });
    };

    // ---- state -------------------------------------------------------------------------------------------------
    opnd_t qb[NH][NDK];   // Q^T B operands (resident per item)
#if FWD1_MDS == 3
    opnd_t kab[2][2][NDK]; // K A operands by chunk parity: QK(c) reads kab[c & 1]; the even E(c) loads K(c+2), K(c+3)
    opnd_t vE[NDT], vO[NDT], vN[NDT]; // V^T A operands of PV(2t), PV(2t+1), PV(2t+2) (vE = vN after the odd chunk)
#else
    opnd_t ka[2][NDK];    // K A operands of the next QK
#endif
#if FWD1_MDS == 2
    opnd_t vab[2][NDT];   // V^T A operands by chunk parity (PV(c) reads vab[c & 1], E(c) loads V(c+1))
#elif FWD1_MDS == 3
    // (V sets vE / vO / vN above)
#else
    opnd_t va[NDT];       // V^T A operands of the current PV
#endif
    opnd_t P[NH];         // P^T B operands
    acc_t  S[2][NH][2];   // S^T of chunks c (exp in place) and c + 1
    acc_t  oacc[NH][NDT]; // O^T
#if FWD1_LSUM
    acc_t        lacc[NH]; // row sums by WMMA (every element of a lane = the sum of its column)
    const opnd_t ones = ones_opnd();
#else
    f32x2 lsum[NH]; // per-lane partial row sums (halves added in the epilogue)
#endif
#if FWD1_OPT_NM2 == 2
    f32x2 nmb[NH]; // -(running max) (log2 domain) as the broadcast pair {-m2[h], -m2[h]}: the tails' fma operand as is
#define M2(h) (-nmb[h][0])
    const float c2v = opaque_f32(c2); // c2 in a VGPR (not rematerialized from its SGPR every chunk)
#elif FWD1_OPT_NM2
    f32x2 m2p[NH / 2]; // running max (log2 domain) as q-tile pairs {m2[2 i], m2[2 i + 1]}
#define M2(h) m2p[(h) >> 1][(h) & 1]
#else
    float m2[NH]; // running max (log2 domain)
#define M2(h) m2[h]
#endif
#if FWD1_OPT_RESC
    uint32_t need = 0u; // ballot of "some q-tile's own-half max exceeds its threshold" (wave-uniform)
#else
    uint32_t bal[NH]; // rescale ballots (wave-uniform)
#endif
#if FWD1_SM1L
    float m2th[NH]; // (m2 + TH2) c2inv: the unscaled own-half max that triggers a rescale
    float mxh[NH];  // own-half column max of chunk c + 1 (for the rescale block)
#else
    float m2n[NH]; // candidate max of chunk c + 1
#endif
    // current item (n, T0, el0 in SGPRs; its b / hq4 / qb are decoded again for the epilogue) and lookahead
    int      n = 1, T0 = 0;
    uint32_t el0 = 0; // K / V element of the item's stage 0 for this wave's op (f1::kv_el0)
    f1::OpC  nx[3];  // ops of stream stages T0 + n .. T0 + n + 2 (the next items' first stages, or dummies)
    f1::OpC  qn{};   // Q op of item j + 1 (issued in item j's last stage; a dummy after the last item)
    uint32_t kb0 = 0, kb1 = 0, vb0 = 0, vb1 = 0, kb2 = 0, vb2 = 0; // burst bases of stages T, T+1 (T+2)

    // items of this workgroup (>= 1 for every grid fwd_v1.plan launches: W <= #units). A larger grid gives some
    // workgroups J = 0; they return here, before any TDM op, LDS access or barrier (J is workgroup-uniform, so
    // all 4 waves leave together; items mode would otherwise decode a q block < 0 for the prologue's Q op)
    const int J = f1::wg_items(sched(kargs()), g);
    if (J <= 0) return;
    // item j of this workgroup: its stage count and stage-0 K / V element (needed from its prologue on).
    // Every quotient is an fdiv (s_mul_u64 + s_lshr_b64): ~25 SALU per decode, no division sequence.
    auto set_cur = [&](int j) FWD_INL {
        const kargs_p   ka = kargs();
        const f1::Work  x  = f1::wg_item(sched(ka), g, j);
        n   = f1::n_stages(x.qb, ka->cst, ka->nkv_st, causal);
#if HK_STRIDES
        el0 = f1::kv_el0_s(lay_kv(ka), x.b, f1::hkv_of(x.hq4, ka->G), w);
#else
        el0 = f1::kv_el0(x.b, Skv, Hkv, f1::hkv_of(x.hq4, ka->G), w);
#endif
    };
    // lookahead of item j (first used at its stage n-3): the ops of the 3 stream stages after it and Q(j+1).
    // Called right before the item's QK(0), in the same scheduling region, so its SALU fills WMMA issue gaps.
    auto set_ahead = [&](int j) FWD_INL {
        const kargs_p   ka = kargs();
        const f1::Sched sc = sched(ka);
        const f1::Shape sh = shape(ka);
        f1::StRef r[3];
        f1::stream3(sh, sc, g, J, j + 1, r);
#if HK_STRIDES
        const f1::Lay lk = lay_kv(ka);
        static_for<3>([&](auto k) { nx[k] = f1::op_c_s(sh, lk, r[k], w); });
        qn = f1::q_c_s(j + 1 < J, f1::wg_item(sc, g, f1::imin(j + 1, J - 1)), sh.Sq, lay_q(ka), w);
#else
        static_for<3>([&](auto k) { nx[k] = f1::op_c(sh, r[k], w); });
        qn = f1::q_c(j + 1 < J, f1::wg_item(sc, g, f1::imin(j + 1, J - 1)), sh.Sq, sh.Hq, w);
#endif
    };

    // ---- phases ----------------------------------------------------------------------------------------------
    auto qk = [&](acc_t(&s)[NH][2], const opnd_t(&kk)[2][NDK]) FWD_INL { // S^T(c') = K(c') Q^T
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                static_for<NH>([&](auto h) {
                    if constexpr (decltype(k)::value == 0) s[h][kt] = wmma0(kk[kt][0], qb[h][0]);
                    else s[h][kt] = wmma(kk[kt][k], qb[h][k], s[h][kt]);
                });
            });
        });
    };
    auto pv = [&](const opnd_t(&vv)[NDT], auto H) FWD_INL { // O^T += V^T P^T of q-tile H (+ its row sums)
        static_for<NDT>([&](auto dt) { oacc[H][dt] = wmma(vv[dt], P[H], oacc[H][dt]); });
#if FWD1_LSUM
        lacc[H] = wmma(ones, P[H], lacc[H]);
#endif
    };
    // SM1 of chunk c1 (S buffer s): [causal mask] -> column max -> rescale ballot; FIRST (item prologue):
    // m2 = the chunk's max directly (O and l are 0)
    auto sm1 = [&](acc_t(&s)[NH][2], int c1, auto MASKED, auto FIRST) FWD_INL {
        if constexpr (decltype(MASKED)::value != 0) {
            const int off0 = f1::mask_off(w, c1, n, 0, causal), off1 = f1::mask_off(w, c1, n, 1, causal);
            static_for<NH>([&](auto h) {
                s[h][0] = mask_row_gt_col(s[h][0], l, off0);
                s[h][1] = mask_row_gt_col(s[h][1], l, off1);
            });
        }
        static_for<NH>([&](auto h) {
            const float mx = max16(s[h][0], s[h][1]);
            if constexpr (decltype(FIRST)::value != 0) {
#if FWD1_OPT_NM2
                const float m = __builtin_fmaxf(mx, xhalf(mx)) * c2;
#if FWD1_OPT_NM2 == 2
                nmb[h] = f32x2{-m, -m};
#else
                M2(h) = m;
#endif
                m2th[h] = (m + f1::TH2) * c2inv;
#else
                m2[h] = __builtin_fmaxf(mx, xhalf(mx)) * c2;
#if FWD1_SM1L
                m2th[h] = (m2[h] + f1::TH2) * c2inv;
#endif
#endif
            } else {
#if FWD1_SM1L
                mxh[h] = mx;
#if !FWD1_OPT_RESC
                bal[h] = ballot(mx > m2th[h]);
#endif
#else
                const float x2 = __builtin_fmaxf(mx, xhalf(mx)) * c2;
                bal[h]         = ballot(x2 - m2[h] > f1::TH2);
                m2n[h]         = any_set(bal[h]) ? __builtin_fmaxf(m2[h], x2) : m2[h];
#endif
            }
        });
#if FWD1_OPT_RESC
        // one compare for the branch: max_h (mx_h - m2th_h) > 0 <=> some q-tile's ballot(mx_h > m2th_h) of SM1L is set
        if constexpr (decltype(FIRST)::value == 0) {
            const float d = __builtin_fmaxf(__builtin_fmaxf(mxh[0] - m2th[0], mxh[1] - m2th[1]),
                                            __builtin_fmaxf(mxh[2] - m2th[2], mxh[3] - m2th[3]));
            need          = ballot(d > 0.f);
        }
#endif
    };
    // p = exp2(s c2 - m2) in place (8 v_pk_fma + 16 v_exp), then the P^T pack (8 v_cvt_pk_bf16)
    auto tail_exp = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h  = decltype(H)::value;
#if FWD1_OPT_NM2 == 2
        const f32x2 cc = {c2v, c2v}, nm = nmb[h]; // -m2 as kept: no negation, no pair build
#elif FWD1_OPT_NM2
        // -m2 of the q-tile pair, then the half (op_sel broadcast + neg modifiers, no {-m2, -m2} moves)
        const f32x2 nmp = -m2p[h >> 1], cc = {c2, c2};
        const f32x2 nm  = __builtin_shufflevector(nmp, nmp, h & 1, h & 1);
#else
        const f32x2   cc = {c2, c2}, nm = {-m2[h], -m2[h]};
#endif
        static_for<2>([&](auto kt) {
            static_for<4>([&](auto j) {
                constexpr int jj = decltype(j)::value;
                const f32x2   a  = pk_fma(acc_pair<jj>(s[h][kt]), cc, nm);
                s[h][kt][2 * jj]     = exp2_hw(a[0]);
                s[h][kt][2 * jj + 1] = exp2_hw(a[1]);
            });
        });
    };
    auto tail_pack = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h = decltype(H)::value;
        P[h]            = pack_pt_fwd(s[h][0], s[h][1]);
    };
#if !FWD1_LSUM
    auto rsum = [&](acc_t(&s)[NH][2], auto H) FWD_INL { // 7 v_pk_add tree + 1 accumulate
        constexpr int h = decltype(H)::value;
        const f32x2   a = pk_add(pk_add(acc_pair<0>(s[h][0]), acc_pair<1>(s[h][0])),
                                 pk_add(acc_pair<2>(s[h][0]), acc_pair<3>(s[h][0])));
        const f32x2   b = pk_add(pk_add(acc_pair<0>(s[h][1]), acc_pair<1>(s[h][1])),
                                 pk_add(acc_pair<2>(s[h][1]), acc_pair<3>(s[h][1])));
        lsum[h]         = pk_add(lsum[h], pk_add(a, b));
    };
#endif
    auto scale_state = [&](auto H, float corr) FWD_INL {
        constexpr int h = decltype(H)::value;
        static_for<NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
#if FWD1_LSUM
        lacc[h] = lacc[h] * corr;
#else
        lsum[h] = pk_mul(lsum[h], f32x2{corr, corr});
#endif
    };
    // C: lazy rescale (uniform branch, rarely taken, laid out of line); corr = 1 exactly for q-tiles whose
    // m2 stays
    auto rescale = [&]() FWD_INL {
#if FWD1_SM1L && (FWD1_OPT_RESC || FWD1_OPT_NM2)
        // SM1L's block with RESC (decision = need; the per-q-tile ballots computed here, as SM1L's) and / or NM2
        // (m2 kept as pairs, written through the identity DPP move): the same mn, corr and m2th bits
#if FWD1_OPT_RESC
        if (__builtin_expect(any_set(need), 0)) {
#else
        if (__builtin_expect(any_set(bal[0] | bal[1] | bal[2] | bal[3]), 0)) {
#endif
            static_for<NH>([&](auto h) {
#if FWD1_OPT_RESC
                const uint32_t b = ballot(mxh[h] > m2th[h]);
#else
                const uint32_t b = bal[h];
#endif
                const float mo   = M2(h);
                const float mf   = __builtin_fmaxf(mxh[h], xhalf(mxh[h]));
                const float mn   = any_set(b) ? __builtin_fmaxf(mo, mf * c2) : mo;
                const float corr = exp2_hw(mo - mn);
                scale_state(h, corr);
#if FWD1_OPT_NM2 == 2
                const float nn = opaque_f32(-mn);
                nmb[h]         = f32x2{nn, nn};
#elif FWD1_OPT_NM2
                M2(h) = opaque_f32(mn);
#else
                M2(h) = mn;
#endif
                m2th[h] = (mn + f1::TH2) * c2inv;
            });
        }
#elif FWD1_SM1L
        if (__builtin_expect(any_set(bal[0] | bal[1] | bal[2] | bal[3]), 0)) {
            static_for<NH>([&](auto h) {
                const float mf   = __builtin_fmaxf(mxh[h], xhalf(mxh[h]));
                const float mn   = any_set(bal[h]) ? __builtin_fmaxf(m2[h], mf * c2) : m2[h];
                const float corr = exp2_hw(m2[h] - mn);
                scale_state(h, corr);
                m2[h]   = mn;
                m2th[h] = (mn + f1::TH2) * c2inv;
            });
        }
#else
        if (__builtin_expect(any_set(bal[0] | bal[1] | bal[2] | bal[3]), 0)) {
            static_for<NH>([&](auto h) { scale_state(h, exp2_hw(m2[h] - m2n[h])); });
        }
        static_for<NH>([&](auto h) { m2[h] = m2n[h]; });
#endif
    };
    auto zero_state = [&]() FWD_INL {
        static_for<NH>([&](auto h) {
            static_for<NDT>([&](auto dt) { oacc[h][dt] = zero_acc(); });
#if FWD1_LSUM
            lacc[h] = zero_acc();
#else
            lsum[h] = f32x2{0.f, 0.f};
#endif
        });
    };

    // ---- one iteration on chunk c = 2 t + PAR of the current item (global stage T = T0 + t) --------------------
    // QKN: QK(c+1) present (false only in the item's last odd chunk); BAR: 1 WAIT(P_{T+1}) + op(T+3),
    // 2 tensorcnt 1 + SIGNAL(P_{T+2}), 3 WAIT + Q(j+1) + op(T+3) (last stage, even), 4 tensorcnt 2 + SIGNAL
    // (last stage, odd); SM: 0 none, 1 SM1(c+1) + C(c+1), 2 the same with the causal mask; OPK (BAR 1 / 3):
    // 0 own stage t+3 or nx[0], 1 nx[1], 2 nx[2]
    auto iter = [&](int t, auto PAR, auto QKN, auto BAR, auto SM, auto OPK) FWD_INL {
        constexpr int  par = decltype(PAR)::value, nxp = 1 - par;
        constexpr bool qkn = decltype(QKN)::value != 0;
        constexpr int  bar = decltype(BAR)::value, sm = decltype(SM)::value, opk = decltype(OPK)::value;
        constexpr bool last = bar == 3 || bar == 4;
        const int      c    = 2 * t + par;
#if FWD1_MDS == 2
        opnd_t(&va)[NDT] = vab[par];
#elif FWD1_MDS == 3
        opnd_t(&va)[NDT] = *(par == 0 ? &vE : &vO);
#endif
        // A: one burst (MDS 1: K(c+1) first, then V(c)) / V(c) (MDS 0)
#if FWD1_MDS == 1
        if constexpr (qkn) ld_k(ka, par == 0 ? kb0 : kb1, ic<nxp>{});
#endif
#if FWD1_MDS == 0 || FWD1_MDS == 1
        ld_v(va, vb0, PAR);
        sched_barrier<0>();
#endif
        // B: QK(c+1) || exp of q-tiles 0..2, pack of 0..1 (+ op(T+3)'s descriptor and the next stage's
        // burst bases on even c)
        Tdm dn{}, dq{};
        if constexpr (bar == 1 || bar == 3) {
            int tk = t; // fenced: computed here, in B
            sgpr_fence(tk);
            __builtin_assume(tk >= 0);
            f1::OpC r = nx[opk];
            if constexpr (opk == 0) {
                const f1::OpC own = f1::own_c(Skv, el0, sst, w, tk + 3);
                r                 = tk + 3 < n ? own : nx[0];
            }
            dn = op_desc(r, T0 + tk + 3);
            if constexpr (bar == 3) dq = q_desc(qn);
            kb2 = kbase(T0 + tk + 2);
            vb2 = vbase(T0 + tk + 2);
        }
#if FWD1_MDS == 3
        if constexpr (qkn) qk(S[nxp], kab[nxp]); // K(c+1): loaded at the even E of the previous stage (front: K(1))
#else
        if constexpr (qkn) qk(S[nxp], ka);
#endif
#if FWD1_OPT_RSB
        // p1's (TAIL 0) B: tail and row sum of q-tiles 0..2, incl. q-tile 2's pack
        tail_exp(S[par], ic<0>{});
        tail_pack(S[par], ic<0>{});
        rsum(S[par], ic<0>{});
        tail_exp(S[par], ic<1>{});
        tail_pack(S[par], ic<1>{});
        rsum(S[par], ic<1>{});
        tail_exp(S[par], ic<2>{});
        tail_pack(S[par], ic<2>{});
        rsum(S[par], ic<2>{});
#else
        tail_exp(S[par], ic<0>{});
        tail_pack(S[par], ic<0>{});
        tail_exp(S[par], ic<1>{});
        tail_pack(S[par], ic<1>{});
        tail_exp(S[par], ic<2>{});
#endif
        sched_barrier<0>();
        // D: barrier point
        if constexpr (bar == 1 || bar == 3) {
            split_wait_lds(); // WAIT(P_{T+1}): stage T+1 published, stage T-1 free
            if constexpr (bar == 3) issue(dq); // Q(j+1) (or a dummy) before op(T+3)
            issue(dn);                          // op(T+3) into slot (T+3) % 4
            sched_barrier<0>();
        } else if constexpr (bar == 2) {
            split_signal_tdm_lds<f1::TCNT_STEADY>(); // SIGNAL(P_{T+2}): own op(T+2) landed, stage T read
        } else {
            split_signal_tdm_lds<f1::TCNT_LAST>(); // the same; Q(j+1), op(T+3) may fly
        }
        // E
#if FWD1_MDS == 0
        if constexpr (!last) { // K(c+2): stage T+1, half PAR (ka of K(c+1) is dead after B)
            ld_k(ka, kb1, PAR);
            sched_barrier<0>();
        }
#elif FWD1_MDS == 2
        if constexpr (!last) ld_k(ka, kb1, PAR);                      // K(c+2): stage T+1, half PAR
        if constexpr (qkn) ld_v(vab[nxp], par == 0 ? vb0 : vb1, ic<nxp>{}); // V(c+1)
        if constexpr (!last || qkn) sched_barrier<0>();
#elif FWD1_MDS == 3
        if constexpr (par == 0) { // the stage's one burst, in the order of first use (pinned: in-order DS counter)
            if constexpr (!last) {
                ld_k(kab[0], kb1, ic<0>{}); // K(c+2): stage T+1, half 0 (QK in the odd B)
                sched_barrier<0>();
            }
            ld_v(vO, vb0, ic<1>{}); // V(c+1): stage T, half 1 (PV in the odd F)
            sched_barrier<0>();
            if constexpr (!last) {
                ld_k(kab[1], kb1, ic<1>{}); // K(c+3): stage T+1, half 1 (QK in the next even B)
                sched_barrier<0>();
                ld_v(vN, vb1, ic<0>{}); // V(c+2): stage T+1, half 0 (PV in the next even F)
                sched_barrier<0>();
            }
        }
#endif
        // F1: PV(c) q-tiles 0, 1 || SM1(c+1), pack of q-tile 2
        pv(va, ic<0>{});
        pv(va, ic<1>{});
        if constexpr (sm != 0) sm1(S[nxp], c + 1, ic<sm == 2>{}, ic<0>{});
#if !FWD1_OPT_RSB
        tail_pack(S[par], ic<2>{});
#if !FWD1_LSUM
        rsum(S[par], ic<0>{});
        rsum(S[par], ic<1>{});
#endif
#endif
#if FWD1_F1SGB > 0 && FWD1_FSPLIT
        // interleave pattern of F1: {1 WMMA, FWD1_F1SGB VALU} per PV / row-sum WMMA, so the SM1 max trees do
        // not run ahead of the first PV WMMA (the census shows a 16 / 39-VALU head otherwise)
        static_for<2 * NDT + 2 * FWD1_LSUM>([&](auto) {
            sched_group_barrier<sg::WMMA, 1, 0>();
            sched_group_barrier<sg::VALU, FWD1_F1SGB, 0>();
        });
#endif
#if FWD1_FSPLIT
        sched_barrier<0>();
#endif
        // F2: PV(c) q-tiles 2, 3 || exp + pack of q-tile 3
        pv(va, ic<2>{});
        tail_exp(S[par], ic<3>{});
        tail_pack(S[par], ic<3>{});
        pv(va, ic<3>{});
#if !FWD1_LSUM
#if !FWD1_OPT_RSB
        rsum(S[par], ic<2>{});
#endif
        rsum(S[par], ic<3>{});
#endif
        sched_barrier<0>();
        // C: rescale for chunk c+1
        if constexpr (sm != 0) rescale();
        if constexpr (par == 1) { // the next stage's burst bases
            kb0 = kb1;
            kb1 = kb2;
            vb0 = vb1;
            vb1 = vb2;
        }
#if FWD1_MDS == 3
        if constexpr (par == 1 && !last) static_for<NDT>([&](auto dt) { vE[dt] = vN[dt]; }); // V(2t+2): the next PV(even)
#endif
    };

    // ---- item phases -----------------------------------------------------------------------------------------
    // prologue loads of the current item (global stage T0 = this item's stage 0, published): own Q landed,
    // Q^T burst, K(0) (+ K(1); MDS 2: + V(0))
#if FWD1_MDS == 0 || FWD1_MDS == 2
    opnd_t k0[2][NDK];
#endif
    auto front_loads = [&]() FWD_INL {
        wait_tensorcnt<f1::TCNT_ITEM>(); // own Q(j) landed (op(T0+2) may fly)
        const lds_u8 *qp = lds_ptr(lds0 + (uint32_t) f1::q_lane(w, l));
        static_for<NH>([&](auto h) {
            static_for<NDK>([&](auto k) {
                qb[h][k] = cat(ds_load_b128(qp, f1::q_imm(h, k, 0)), ds_load_b128(qp, f1::q_imm(h, k, 1)));
            });
        });
#if FWD1_MDS == 1
        ld_k(ka, kb0, ic<0>{}); // K(0) -> ka (QK(0)); A(0) reloads ka with K(1)
#elif FWD1_MDS == 3
        ld_k(kab[0], kb0, ic<0>{}); // K(0), consumed by QK(0)
        ld_k(kab[1], kb0, ic<1>{}); // K(1), consumed by QK(1) in iteration 0
        ld_v(vE, vb0, ic<0>{});     // V(0), consumed by PV(0) in iteration 0
#else
        ld_k(k0, kb0, ic<0>{}); // K(0)
        ld_k(ka, kb0, ic<1>{}); // K(1), consumed by QK(1) in iteration 0
#endif
#if FWD1_MDS == 2
        ld_v(vab[0], vb0, ic<0>{}); // V(0), consumed by PV(0) in iteration 0
#endif
        sched_barrier<0>();
    };
    auto front_qk = [&]() FWD_INL {
#if FWD1_MDS == 1
        qk(S[0], ka);
#elif FWD1_MDS == 3
        qk(S[0], kab[0]);
#else
        qk(S[0], k0);
#endif
    };
    auto front_sm1 = [&]() FWD_INL {
        sm1(S[0], 0, ic<1>{}, ic<1>{}); // SM1(0): masked (effective iff n == 1), m2 = max
#if FWD1_MDS == 3
        // the front's loads (V(0) last) retired here, after QK(0) (free at run time): otherwise V(0) is still
        // pending in LLVM's DS score window at the stage loop's entry, and once the 64-load stage burst pushes it out of
        // the 63-deep counter range LLVM guards PV(0) with s_wait_dscnt 62 in EVERY iteration (a ~35-cycle stall)
        wait_dscnt<0>();
#endif
        sched_barrier<0>();
    };
    // stages of the current item
    auto stages = [&]() FWD_INL {
        const int nmain = f1::imax(n - 2, 0);
#if FWD1_OPT_UNROLL
        // 2 stages per loop iteration; an odd count runs stage 0 first (WG-uniform branch)
        int t0 = 0;
        if (nmain & 1) {
            iter(0, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(0, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
            t0 = 1;
        }
        for (int t = t0; t < nmain; t += 2) {
            iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
            iter(t + 1, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(t + 1, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
        }
#else
        for (int t = 0; t < nmain; ++t) {
            iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
        }
#endif
        if (f1::has_pen_stage(n)) { // WG-uniform, outside any loop: stage n-2, its SM1(2n-2) masked
            const int t = n - 2;
            iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<1>{});
            iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<2>{}, ic<0>{});
        }
        {
            const int t = n - 1; // last stage: Q(j+1) + nx[2]; SM1(2n-1) masked; no QK(2n)
            iter(t, ic<0>{}, ic<1>{}, ic<3>{}, ic<2>{}, ic<2>{});
            iter(t, ic<1>{}, ic<0>{}, ic<4>{}, ic<0>{}, ic<0>{});
        }
    };
    // epilogue pieces (wave-private O region + VMEM; no barrier)
    auto epi_lt = [&](float (&lt)[NH]) FWD_INL {
        static_for<NH>([&](auto h) {
#if FWD1_LSUM
            lt[h] = lacc[h][0]; // every element of the lane holds the full column sum
#else
            const float ll = lsum[h][0] + lsum[h][1];
            lt[h]          = ll + xhalf(ll); // identical bits in lanes l and l ^ 16
#endif
        });
    };
    auto epi_norm = [&](u32x4(&pk)[NH][NDT], const float (&lt)[NH]) FWD_INL {
        static_for<NH>([&](auto h) {
            const float inv = lt[h] > 0.f ? rcp_ieee(lt[h]) : 0.f;
            static_for<NDT>([&](auto dt) { pk[h][dt] = cvt_pk_bf16(oacc[h][dt] * inv); });
        });
    };
    auto epi_image = [&](const u32x4(&pk)[NH][NDT]) FWD_INL {
        lds_u8 *ow = lds_ptr(lds0 + (uint32_t) f1::ow_lane(w, l));
        static_for<NH>([&](auto h) { static_for<NDT>([&](auto dt) { ds_store_b128(ow, f1::ow_imm(h, dt), pk[h][dt]); }); });
    };
    auto epi_out = [&](const f1::Work &x, const float (&lt)[NH]) FWD_INL {
        compiler_fence(); // read-back after the image writes (same-wave DS ops complete in order)
        const kargs_p ka = kargs();
        const int     Sq = ka->Sq, Hq = ka->Hq;
        const rsrc_t  ro = make_rsrc(ka->O, ka->o_bytes); // the whole tensor (SBHD: a batch's rows interleave)
        const lds_u8 *rp = lds_ptr(lds0 + (uint32_t) f1::or_lane(w, l));
#if HK_STRIDES
        const f1::Lay lo = lay_o(ka);
#endif
        static_for<2 * f1::QW>([&](auto s) {
            const u32x4 v = ds_load_b128_u32(rp, f1::or_imm(s));
#if HK_STRIDES
            buf_store_b128(ro, (int) f1::o_store_b_s(lo, x.b, x.hq4, x.qb, w, s, l), 0, v);
#else
            buf_store_b128(ro, (int) f1::o_store_b(x.b, Sq, Hq, x.hq4, x.qb, w, s, l), 0, v);
#endif
        });
        const rsrc_t rl = make_rsrc(ka->LSE, ka->lse_bytes);
        float        lse[NH];
        static_for<NH>([&](auto h) { lse[h] = (M2(h) + log2_hw(lt[h])) * f1::LN2; });
        const bool hi = l >= 16;
        static_for<2>([&](auto k) {
            buf_store_f32(rl, (int) f1::lse_store_b(x.b, Hq, Sq, x.hq4, x.qb, w, k, l), 0, hi ? lse[2 + k] : lse[k]);
        });
    };

    auto item_of = [&](int j) FWD_INL { return f1::wg_item(sched(kargs()), g, j); };

    // ---- workgroup prologue -------------------------------------------------------------------------------------
    {
        const kargs_p   ka = kargs();
        const f1::Sched sc = sched(ka);
        const f1::Shape sh = shape(ka);
        f1::StRef       r[3];
        f1::stream3(sh, sc, g, J, 0, r); // global stages 0, 1, 2
#if HK_STRIDES
        const f1::Lay lk = lay_kv(ka);
        issue(q_desc(f1::q_c_s(1, f1::wg_item(sc, g, 0), sh.Sq, lay_q(ka), w)));
        issue(op_desc(f1::op_c_s(sh, lk, r[0], w), 0));
        issue(op_desc(f1::op_c_s(sh, lk, r[1], w), 1));
#else
        issue(q_desc(f1::q_c(1, f1::wg_item(sc, g, 0), sh.Sq, sh.Hq, w)));
        issue(op_desc(f1::op_c(sh, r[0], w), 0));
        issue(op_desc(f1::op_c(sh, r[1], w), 1));
#endif
        sched_barrier<0>();
        wait_tensorcnt<f1::TCNT_STEADY>(); // own Q(0) and op 0 landed
        lds_barrier();                     // P_0: every wave's op 0 (stage 0) published
#if HK_STRIDES
        issue(op_desc(f1::op_c_s(sh, lk, r[2], w), 2));
#else
        issue(op_desc(f1::op_c(sh, r[2], w), 2));
#endif
        split_signal_tdm_lds<f1::TCNT_STEADY>(); // SIGNAL(P_1): own op 1 landed
    }
    kb0 = kbase(0);
    kb1 = kbase(1);
    vb0 = vbase(0);
    vb1 = vbase(1);
#if FWD1_EPI == 0
    for (int j = 0; j < J; ++j) {
        set_cur(j);
        front_loads();
        zero_state();
        set_ahead(j);
        front_qk();
        front_sm1();
        stages();
        float lt[NH];
        epi_lt(lt);
        u32x4 pk[NH][NDT];
        epi_norm(pk, lt);
        epi_image(pk);
        epi_out(item_of(j), lt);
        sched_barrier<0>();
        T0 += n;
    }
#else
    set_cur(0);
    front_loads();
    zero_state();
    set_ahead(0);
    front_qk();
    front_sm1();
    for (int j = 0; j < J; ++j) {
        stages();
        float lt[NH];
        epi_lt(lt);
        if (j + 1 < J) { // WG-uniform: the next item's loads, then its QK(0) under this item's O normalize
            T0 += n;
            set_cur(j + 1);
            front_loads();
            u32x4 pk[NH][NDT];
            front_qk();
            epi_norm(pk, lt);
            sched_barrier<0>();
            epi_image(pk);
            epi_out(item_of(j), lt); // decoded here (no SGPRs held across the next item's front)
            zero_state();
            set_ahead(j + 1); // after the epilogue: its 8 SGPRs are not live across the O stores
            front_sm1();
        } else {
            u32x4 pk[NH][NDT];
            epi_norm(pk, lt);
            epi_image(pk);
            epi_out(item_of(j), lt);
            sched_barrier<0>();
        }
    }
#endif
    // ---- exit: every TDM op of this wave landed (incl. dummies), then WAIT(P_{N+1}) -------------------------------
    sched_barrier<0>();
    wait_tensorcnt<0>();
    split_wait_lds();
}
