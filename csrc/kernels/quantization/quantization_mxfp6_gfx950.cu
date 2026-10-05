/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Fused MXFP6 (E2M3) quantize + pack into AITER's mxfp6_c0c1_256_padk2 layout,
// in both contraction directions from a single read of the input.
//
// Why this exists
// ---------------
// Training needs every tensor packed along two axes (fprop contracts K, dgrad
// contracts N, wgrad contracts M). AITER only ships a row-direction packer, so the
// column direction previously went through `pack(x.t().contiguous())`, and that
// materialised transpose cost more than both the GEMM difference and the packing itself.
//
// The fix is to never materialise it. A block stages one TILE_M x TILE_N patch of the
// input in LDS with coalesced reads, then packs rows and columns out of that patch.
// Everything downstream of the gather -- Hadamard, scale, conversion, addressing -- is
// identical for the two directions, because packing a column of `x` is by definition
// packing a row of `x.T`.
//
// Bit-exactness
// -------------
// The per-group math is a deliberate transliteration of AITER's `quant_mxfp6_group`
// (csrc/kernels/quant_mxfp6_gemm.cu), down to the H8-then-two-lane-butterflies
// factorisation of H32 and the gfx950 conversion intrinsic. That is not incidental:
// the row direction must reproduce AITER's packer byte for byte, since that packer is
// the oracle the A6W6 assembly was validated against, and the E2M3 rounding of an exact
// tie differs between plausible implementations (`floor(x + 0.5)` disagrees with the
// hardware's round-to-nearest-even, which is what the intrinsic does). Reusing the
// intrinsic makes the agreement structural rather than something tests have to police.

#include <hip/hip_runtime.h>

#include <atomic>
#include <type_traits>

#include "primus_turbo/common.h"
#include "primus_turbo/mxfp4_emit.hpp"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

namespace {

// ---------------------------------------------------------------------------
// Layout constants. These describe AITER's packed blob and cannot be retuned:
// the A6W6 assembly derives its strides from them.
// ---------------------------------------------------------------------------
constexpr int kGroupSize       = 32;  // values per E8M0 scale, and the Hadamard size
constexpr int kTileRows        = 256; // rows per packed tile
constexpr int kKTile           = 128; // K values per packed tile
constexpr int kGroupsPerKTile  = kKTile / kGroupSize; // 4
constexpr int kPackedTileBytes = 24576;
constexpr int kScaleTileBytes  = 1024;
constexpr int kC1PlaneOffset   = 16384; // byte offset of the C1 plane within a tile
constexpr int kC0BytesPerBlock = 16;    // of 24 bytes per group, 16 land in C0
constexpr int kC1BytesPerBlock = 8;

// 1/sqrt(32) at fp32 precision, so the rotation is orthonormal to fp32 rounding: both GEMM
// operands carry the factor and the product scales by 32 c^2 = 1 - O(1e-8). An earlier
// revision used 1/sqrt(32) rounded to bf16 (0.1767578125), copied from aiter's Triton packer
// that applied H as a bf16 dot; that scaled every MXFP6 GEMM result by 0.99979 -- a change to
// the real-valued function, which MLPerf closed division does not allow for a numerically
// safe transform. The switch changes ~0.1% of codes relative to that revision.
constexpr float kHadamard32Norm = 0.17677669529663687f;

// ---------------------------------------------------------------------------
// Blocking. TILE_M/TILE_N are multiples of kGroupSize so every staged patch holds
// whole 32-value groups in both directions, and divide 256 so a 256-aligned operand
// tiles exactly.
//
// 64x64 with 128 threads was picked by measurement, not assumption: a 64x64 patch holds
// TILE_M * TILE_N/32 = 128 groups in each direction, so at one thread per group a
// 128-thread block has every thread emit exactly one row group and one column group with
// none left idle. Sweeping TILE_M and TILE_N over 64..256 and the block over 128..512
// found nothing faster -- once the emit path below stopped being the bottleneck the
// kernel is bandwidth-bound, and larger patches only cost LDS and occupancy.
// ---------------------------------------------------------------------------
// Overridable only so a sweep can be driven from the command line without editing this
// file; the defaults below are the shipped, measured values and are what any normal
// build uses; they were checked at the row extents the training shapes present.
#ifndef MXFP6_TILE_M
#define MXFP6_TILE_M 64
#endif
// A 256-thread block, from a cold-operand sweep of TILE_M 32..128 x TILE_N 64..256 x
// block 128..512 at the row extents the training shapes present. Cold is the
// operative word: the previous 128-thread default came from a sweep with the input
// resident, which is not the state a packer call ever finds its operand in -- every one
// of them reads a tensor a GEMM or an optimizer step has just written.
//
// The sweep has to score both entry points, and this is the part that is easy to get
// wrong. quantize_mxfp6_impl and the fused prologue path do not rank tiles the same way,
// and the widest, largest-block configurations win the plain packer clearly while losing
// the fused one -- which is the path the MLP takes, and the more expensive of the two per
// call. Tuning on the plain packer alone selects a configuration that is a net regression
// in a real step. This one is faster than the previous default on both.
//
// TILE_M stays 64. Halving it is tempting on the plain packer, but TILE_M is
// MXFP6_COL_SUM_TILE_M, so it sets the bias-gradient partial buffer's row count: halving
// it doubles those rows and the reduction behind them, which this sweep does not charge
// for because that reduction is a separate kernel.
//
// TILE_M stays 64 deliberately: 32x256/512 is another ~2 points faster but TILE_M is
// tied to MXFP6_COL_SUM_TILE_M, which Python sizes the bias-gradient partial buffer
// from, so moving it is a coupled change rather than a blocking one.
#ifndef MXFP6_TILE_N
#define MXFP6_TILE_N 64
#endif
#ifndef MXFP6_THREADS_PER_BLOCK
#define MXFP6_THREADS_PER_BLOCK 256
#endif
#ifndef MXFP6_ASYNC_STAGE
#define MXFP6_ASYNC_STAGE 3
#endif

constexpr int TILE_M = MXFP6_TILE_M;
// The shipped width, and a default rather than the only value: the QK-norm+RoPE prologue's
// row reduction spans head_dim and has to be block-local, so that prologue instantiates the
// kernel at TILE_N = head_dim instead. Everything below is written against the template
// parameter; this constant only names the width the ordinary entry points use.
constexpr int kDefaultTileN     = MXFP6_TILE_N;
constexpr int THREADS_PER_BLOCK = MXFP6_THREADS_PER_BLOCK;

// The bias-gradient partial buffer has one row per M-tile, so its geometry is this tile
// height. The header carries the value because the host and Python size the buffer.
static_assert(TILE_M == MXFP6_COL_SUM_TILE_M,
              "MXFP6_COL_SUM_TILE_M must track TILE_M or the partial buffer is mis-sized");

// Pad the LDS row pitch so the column-direction gather, which walks the pitch, spreads
// across banks instead of piling onto one. 8 gives a 72-uint16 pitch, putting consecutive
// rows 4 banks apart, and also makes every row 16-byte aligned. Swept against pads of 2,
// 4 and 16 the choice turns out to be worth nothing measurable on its own -- once the
// emit path below is right the kernel is bandwidth-bound -- so this is kept for the
// alignment property rather than for any observed gain.
constexpr int LDS_PAD = 8;

constexpr int lds_pitch(const int tile_n) {
    return tile_n + LDS_PAD;
}

// Stands in for MXFP6QkNormRopeArgs on the instantiations that have no use for it, so a
// prologue that needs nine operands does not put them in every other prologue's kernarg
// segment. Passing the real struct unconditionally is correct but takes kernarg from 80
// bytes to 176 on kernels that never read it; this keeps that at 8. The generated
// instruction streams are byte-identical either way -- verified by diffing device assembly
// against the pre-change file, all 16 shipped instantiations -- so this is about not
// leaving an unused cost behind, not about speed.
struct MXFP6NoPrologueArgs {};

template <typename DType, MXFP6Prologue PROLOGUE>
using prologue_args_t = std::conditional_t<
    PROLOGUE == MXFP6Prologue::QkNormRopeBackward, MXFP6QkNormRopeArgs<DType>,
    std::conditional_t<
        PROLOGUE == MXFP6Prologue::LnModulate, MXFP6LnModulateArgs<DType>,
        std::conditional_t<PROLOGUE == MXFP6Prologue::GateMul, MXFP6GateMulArgs<DType>,
                           MXFP6NoPrologueArgs>>>;

using packed_fp6x32_t = uint32_t __attribute__((ext_vector_type(6)));
using uint4_t         = uint32_t __attribute__((ext_vector_type(4)));
using uint2_t         = uint32_t __attribute__((ext_vector_type(2)));
using float16_t       = float __attribute__((ext_vector_type(16)));
using as3_uint32_ptr  = uint32_t __attribute__((address_space(3))) *;
using int32x4_t       = int32_t __attribute__((ext_vector_type(4)));

// gfx950's buffer_load_dword ... lds path. Unlike a normal vector load it does
// not allocate a VGPR for the payload: each lane writes its dword directly to
// lane*4 from the wave's LDS base. Four 256-byte strips fill each wave's 1 KiB
// share of a stage. This experiment deliberately keeps the wrapper local
// rather than pulling the packer into a larger kernel framework.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_resource(const void *ptr,
                                                                       uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(ptr), 0, bytes, 0x00020000);
}

__device__ __forceinline__ void async_load_lds_4(__amdgpu_buffer_rsrc_t resource,
                                                 as3_uint32_ptr lds_base,
                                                 int32_t byte_offset) {
    __builtin_amdgcn_raw_ptr_buffer_load_lds(resource, lds_base, 4, byte_offset, 0, 0, 0);
}

// Clang 20's raw_ptr builtin only accepts 1/2/4-byte widths, while the LLVM
// intrinsic and gfx950 ISA accept a 16-byte lane payload. Keep that wider
// instruction as a separate control rather than conflating instruction width
// with direct-to-LDS itself.
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    int32x4_t resource, as3_uint32_ptr lds_base, int size, int voffset, int soffset, int offset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ __forceinline__ int32x4_t make_buffer_resource_vec(const void *ptr, uint32_t bytes) {
    const uint64_t address = reinterpret_cast<uint64_t>(ptr);
    return int32x4_t{static_cast<int32_t>(address), static_cast<int32_t>(address >> 32),
                     static_cast<int32_t>(bytes), 0x00020000};
}

__device__ __forceinline__ void async_load_lds_16(int32x4_t resource,
                                                  as3_uint32_ptr lds_base,
                                                  int32_t byte_offset) {
    llvm_amdgcn_raw_buffer_load_lds(resource, lds_base, 16, byte_offset, 0, 0, 0);
}

// The packer's Hadamard is a bf16 dot with fp32 accumulate, so an fp16 input has to be
// rounded through bf16 first or the codes drift from AITER's.
template <typename DType> __device__ __forceinline__ float to_dot_operand(const uint16_t bits) {
    if constexpr (std::is_same_v<DType, bfloat16>) {
        const uint32_t widened = static_cast<uint32_t>(bits) << 16;
        return __builtin_bit_cast(float, widened);
    } else {
        const float value = __half2float(__builtin_bit_cast(half, bits));
        return static_cast<float>(static_cast<bfloat16>(value));
    }
}

// ---------------------------------------------------------------------------
// Prologue support. Distinct from to_dot_operand above: that one deliberately rounds fp16
// through bf16 because it feeds the Hadamard, whereas the epilogue arithmetic has to see
// the value the producing kernel would have written, so it widens exactly.
// ---------------------------------------------------------------------------
template <typename DType> __device__ __forceinline__ float to_float(const uint16_t bits) {
    if constexpr (std::is_same_v<DType, bfloat16>) {
        const uint32_t widened = static_cast<uint32_t>(bits) << 16;
        return __builtin_bit_cast(float, widened);
    } else {
        return __half2float(__builtin_bit_cast(half, bits));
    }
}

// Rounding back to DType before staging is what makes the fusion reproduce the epilogue it
// replaces rather than merely resemble it: the bytes that enter s_tile are then the same ones
// the unfused epilogue kernel would have stored to HBM for the packer to read back.
template <typename DType> __device__ __forceinline__ uint16_t from_float(const float value) {
    if constexpr (std::is_same_v<DType, bfloat16>) {
        // Round-to-nearest-even, matching c10::BFloat16 and LLVM's fptrunc-to-bfloat. No
        // NaN special case: the bias below already leaves a NaN input as some NaN, and
        // branching on it in the staging loop would cost more than the payload is worth.
        const uint32_t bits = __builtin_bit_cast(uint32_t, value);
        const uint32_t lsb  = (bits >> 16) & 1u;
        return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
    } else {
        return __builtin_bit_cast(uint16_t, __float2half(value));
    }
}

// The DType-rounded value, still in fp32. The bias-add has to be rounded to DType before the
// activation reads it, but nothing needs the narrow bits themselves, so for bf16 the round
// trip through uint16_t collapses to masking the low half off in place.
template <typename DType> __device__ __forceinline__ float round_to_dtype(const float value) {
    if constexpr (std::is_same_v<DType, bfloat16>) {
        const uint32_t bits = __builtin_bit_cast(uint32_t, value);
        const uint32_t lsb  = (bits >> 16) & 1u;
        return __builtin_bit_cast(float, (bits + 0x7fffu + lsb) & 0xffff0000u);
    } else {
        return __half2float(__float2half(value));
    }
}

/*
 * GELU and its derivative, evaluated without ever forming tanh.
 *
 * The tanh GELU is usually written 0.5x(1 + tanh(u)) with u = beta(x + kappa x^3), and both
 * it and its derivative need tanh only through 1 + tanh(u) and 1 - tanh(u)^2. Writing
 * E = e^{-2u}, which one hardware exp2 supplies,
 *
 *     1 + tanh(u)   = 2 / (1 + E)          1 - tanh(u)^2 = 4E / (1 + E)^2
 *
 * so the forward collapses all the way to x / (1 + E) and the derivative needs no tanh
 * either. That matters for two independent reasons.
 *
 * Speed: the packer is VALU bound the moment a prologue is switched on, so the activation's
 * instruction count is what decides whether fusing it beats the separate kernel it replaces
 * at all. A libm tanh costs 54 instructions per element, more than everything else in the
 * epilogue put together; this costs about 9.
 *
 * Conditioning: going through tanh means forming 1 + tanh(u) for u in the left tail, where
 * tanh has already saturated to -1 and the sum has no significant bits left. The closed
 * form cancels nowhere. It is a different rounding of the activation than ATen's, not a
 * worse one, which is the property test_fused_prologue_is_no_less_accurate_than_aten pins.
 *
 * `u` is still formed exactly as inductor's decomposition forms it, beta applied on its own
 * and only then scaled to the exp2 argument. Folding beta into the change of base saves the
 * multiply and is measurably worse: it is a more accurate `u` than ATen's, so it disagrees
 * with ATen a thousand times more often (0.27% of packed codes against 0.0003%) while
 * benchmarking identically. The point of matching is matching, not accuracy.
 */
constexpr float kGeluBeta  = 0.7978845608028654f; // sqrt(2/pi)
constexpr float kGeluKappa = 0.044715f;

// -2 * log2(e): the change of base that lets one v_exp_f32 supply E = e^-2u.
constexpr float kGeluNegTwoLog2e = -2.0f * 1.4426950408889634f;

// Below this, tanhf returns exactly -1, so the activation and its derivative are exactly
// zero. Reproducing that explicitly costs one compare and keeps the whole left tail
// identical to the graph being replaced, which is otherwise where nearly all of the
// disagreement with it would live.
constexpr float kGeluSaturate = -9.02f;

__device__ __forceinline__ float gelu_tanh(const float x) {
    const float inner = kGeluBeta * (x + kGeluKappa * (x * x * x));
    if (inner < kGeluSaturate)
        return 0.0f;
    // 0.5x * (1 + tanh u) = 0.5x * 2/(1 + E) = x/(1 + E).
    const float e = __builtin_amdgcn_exp2f(inner * kGeluNegTwoLog2e);
    return x * __builtin_amdgcn_rcpf(1.0f + e);
}

__device__ __forceinline__ float gelu_tanh_backward(const float grad, const float x) {
    const float x_sq  = x * x;
    const float inner = kGeluBeta * (x + kGeluKappa * (x_sq * x));
    if (inner < kGeluSaturate)
        return 0.0f;

    const float e = __builtin_amdgcn_exp2f(inner * kGeluNegTwoLog2e);
    const float d = __builtin_amdgcn_rcpf(1.0f + e);

    // d is (1 + tanh)/2, which is the derivative of the 0.5x factor. The other term is
    // 0.5x * (1 - tanh^2) * du/dx, with the 4 of 4E/(1+E)^2 and the 0.5 folded into the
    // 2*beta below.
    const float inner_derivative = 1.5957691216057308f * (1.0f + 3.0f * kGeluKappa * x_sq);
    const float right_derivative = x * e * d * d * inner_derivative;
    return grad * (d + right_derivative);
}

// Coalesced 8-wide staged read of one row segment, zero-filling past N. Factored out
// because the prologue modes need it for a second operand as well.
constexpr int kStageVec = 8;

__device__ __forceinline__ void stage_vector(uint16_t (&dst)[kStageVec],
                                             const uint16_t *__restrict__ src, const int32_t row,
                                             const int32_t col, const int32_t N) {
    const int64_t offset = static_cast<int64_t>(row) * N + col;
    if (col + kStageVec <= N) {
        *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(&src[offset]);
    } else {
#pragma unroll
        for (int i = 0; i < kStageVec; ++i)
            dst[i] = (col + i < N) ? src[offset + i] : uint16_t{0};
    }
}

// The bias+GELU prologues over a whole bf16 tile already in LDS, in place, for the
// async-staged arm.
//
// Same arithmetic as the generic staging loop, element for element: the only change is how
// values move. The generic loop, as compiled, spent ~29 instructions per element of which
// the GELU itself is ~12 -- one ds_read_u16 and ds_write_b16 per element, the bias
// re-staged from global on every pass, and bf16 round-to-nearest-even done in integer ALU
// twice per element. Here the tile moves as 16-byte vectors, the bias once per thread (a
// thread's columns are the same on every pass), and rounding is v_cvt_pk_bf16_f32, which
// agrees with the integer RNE on every non-NaN fp32 (checked exhaustively, all 2^32
// inputs). A NaN stays a NaN either way, which the integer form
// did not quite promise.
using bf16x2_t  = __bf16 __attribute__((ext_vector_type(2)));
using float2_t_ = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t pack_bf16x2_rne(const float lo, const float hi) {
    const float2_t_ v = {lo, hi};
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2_t));
}

__device__ __forceinline__ float bf16_lo(const uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float bf16_hi(const uint32_t w) {
    return __builtin_bit_cast(float, w & 0xffff0000u);
}

template <MXFP6Prologue PROLOGUE, int TILE_N, int LDS_PITCH, int TILE_M_ = TILE_M>
__device__ __forceinline__ void gelu_prologue_bf16(uint16_t (*s_tile)[LDS_PITCH],
                                                   const uint16_t *s_aux,
                                                   const uint16_t *__restrict__ bias,
                                                   const int tile_n, const int N) {
    constexpr int VEC  = kStageVec;
    constexpr int STEP = THREADS_PER_BLOCK * VEC;
    static_assert(STEP % TILE_N == 0, "a thread's columns must not move between passes");
    const int local_n = (threadIdx.x * VEC) % TILE_N;

    float bias_f[VEC];
    if (bias != nullptr) {
        uint16_t b[VEC];
        stage_vector(b, bias, 0, tile_n + local_n, N);
#pragma unroll
        for (int i = 0; i < VEC; ++i)
            bias_f[i] = bf16_lo(b[i]);
    }
    // bias is uniform, but a test of it inside the loop compiled to a compare and a branch per
    // element, so the loop is instantiated twice instead. The no-bias loop cannot become an add
    // of zero: -0 + 0 is +0, and gelu keeps the sign of a zero.
    const auto run = [&](auto has_bias) {
#pragma unroll
        for (int base = threadIdx.x * VEC; base < TILE_M_ * TILE_N; base += STEP) {
            const int local_m = base / TILE_N;
            const uint4 in = *reinterpret_cast<const uint4 *>(&s_tile[local_m][local_n]);
            const uint32_t *w = reinterpret_cast<const uint32_t *>(&in);
            uint4 g = {};
            if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward)
                g = *reinterpret_cast<const uint4 *>(&s_aux[local_m * TILE_N + local_n]);
            const uint32_t *gw = reinterpret_cast<const uint32_t *>(&g);
            uint4 out;
            uint32_t *o = reinterpret_cast<uint32_t *>(&out);
#pragma unroll
            for (int j = 0; j < VEC / 2; ++j) {
                float x0 = bf16_lo(w[j]);
                float x1 = bf16_hi(w[j]);
                if constexpr (decltype(has_bias)::value) {
                    // Both bias sums rounded by one conversion; each lane rounds on its own.
                    const uint32_t r = pack_bf16x2_rne(x0 + bias_f[2 * j], x1 + bias_f[2 * j + 1]);
                    x0 = bf16_lo(r);
                    x1 = bf16_hi(r);
                }
                if constexpr (PROLOGUE == MXFP6Prologue::BiasGelu)
                    o[j] = pack_bf16x2_rne(gelu_tanh(x0), gelu_tanh(x1));
                else
                    o[j] = pack_bf16x2_rne(gelu_tanh_backward(bf16_lo(gw[j]), x0),
                                           gelu_tanh_backward(bf16_hi(gw[j]), x1));
            }
            *reinterpret_cast<uint4 *>(&s_tile[local_m][local_n]) = out;
        }
    };
    if (bias != nullptr)
        run(std::true_type{});
    else
        run(std::false_type{});
}

// AdaLN's modulated layer norm over one staged vector, in place.
//
// Factored out rather than written into each staging arm because the three arms differ
// only in how the tile reaches LDS, not in what happens to it once there -- and because
// there is exactly one expression here whose association has to match the producing
// kernel's, so it should exist once.
//
// The row statistics are read, not recomputed. They reduce over the whole hidden
// dimension and a block only sees TILE_N of it, so recomputing is not on offer; taking
// them also means the normalisation is bit-identical to the producer's by construction
// rather than by agreement, leaving only the affine below to match.
template <typename DType>
__device__ __forceinline__ void apply_ln_modulate(uint16_t (&staged)[kStageVec],
                                                  const MXFP6LnModulateArgs<DType> &args,
                                                  const int32_t global_m, const int32_t global_n,
                                                  const int32_t N) {
    // Contraction off, deliberately. The epilogue being replaced is a sequence of tensor
    // ops, so its multiply and its add round separately; letting the compiler fuse them
    // into an FMA keeps one extra bit of the product and changes roughly one packed code in
    // 10^5 -- measured, not feared. Cheap to give up: this prologue is bandwidth bound, and
    // it is what lets the blobs be claimed bit-identical rather than close.
#pragma clang fp contract(off)
    const float   mean = args.mean[global_m];
    const float   rstd = args.rstd[global_m];
    // m = s * B + b over a power-of-two B, so the batch index is the low bits of the row.
    const int32_t b    = global_m & args.batch_mask;

    uint16_t scale_staged[kStageVec];
    uint16_t shift_staged[kStageVec];
    stage_vector(scale_staged, reinterpret_cast<const uint16_t *>(args.scale), b, global_n, N);
    stage_vector(shift_staged, reinterpret_cast<const uint16_t *>(args.shift), b, global_n, N);

#pragma unroll
    for (int i = 0; i < kStageVec; ++i) {
        const float x_hat = (to_float<DType>(staged[i]) - mean) * rstd;
        const float sc    = to_float<DType>(scale_staged[i]);
        const float sh    = to_float<DType>(shift_staged[i]);
        staged[i]         = from_float<DType>(x_hat * (1.0f + sc) + sh);
    }
}

// AdaLN's gate over one staged vector, in place: the product in fp32, rounded once.
template <typename DType>
__device__ __forceinline__ void apply_gate_mul(uint16_t (&staged)[kStageVec],
                                               const MXFP6GateMulArgs<DType> &args,
                                               const int32_t global_m, const int32_t global_n,
                                               const int32_t N) {
    uint16_t gate_staged[kStageVec];
    stage_vector(gate_staged, reinterpret_cast<const uint16_t *>(args.gate),
                 global_m & args.batch_mask, global_n, N);
#pragma unroll
    for (int i = 0; i < kStageVec; ++i)
        staged[i] = from_float<DType>(to_float<DType>(staged[i]) * to_float<DType>(gate_staged[i]));
}

// The gate prologue over a whole bf16 tile already in LDS, for the async-staged arm. The
// data movement of gelu_prologue_bf16 (16-byte LDS vectors, v_cvt_pk_bf16_f32) with
// apply_gate_mul's arithmetic. The gate row changes with the tile row, so it is loaded per
// pass: 16 bytes from a [B, N] table that stays in cache.
template <int TILE_N, int LDS_PITCH, int TILE_M_ = TILE_M>
__device__ __forceinline__ void gate_mul_prologue_bf16(uint16_t (*s_tile)[LDS_PITCH],
                                                       const MXFP6GateMulArgs<bfloat16> &args,
                                                       const int tile_m, const int tile_n,
                                                       const int N) {
    constexpr int VEC  = kStageVec;
    constexpr int STEP = THREADS_PER_BLOCK * VEC;
    static_assert(STEP % TILE_N == 0, "a thread's columns must not move between passes");
    const int local_n = (threadIdx.x * VEC) % TILE_N;
    const uint16_t *gate = reinterpret_cast<const uint16_t *>(args.gate);

#pragma unroll
    for (int base = threadIdx.x * VEC; base < TILE_M_ * TILE_N; base += STEP) {
        const int local_m = base / TILE_N;
        uint16_t g16[VEC];
        stage_vector(g16, gate, (tile_m + local_m) & args.batch_mask, tile_n + local_n, N);
        const uint32_t *gw = reinterpret_cast<const uint32_t *>(g16);
        const uint4 in = *reinterpret_cast<const uint4 *>(&s_tile[local_m][local_n]);
        const uint32_t *w = reinterpret_cast<const uint32_t *>(&in);
        uint4 out;
        uint32_t *o = reinterpret_cast<uint32_t *>(&out);
#pragma unroll
        for (int j = 0; j < VEC / 2; ++j)
            o[j] = pack_bf16x2_rne(bf16_lo(w[j]) * bf16_lo(gw[j]), bf16_hi(w[j]) * bf16_hi(gw[j]));
        *reinterpret_cast<uint4 *>(&s_tile[local_m][local_n]) = out;
    }
}

/*
 * Quantize one 32-value group and scatter it into the packed blob.
 *
 * `out_row` is the row index in the packed operand and `group` the 32-block index along
 * the contraction axis; the caller decides what those mean, which is the only thing that
 * distinguishes the row direction from the column direction.
 *
 * One thread owns the whole group. The obvious alternative -- spreading the group over
 * four adjacent lanes, eight values each -- is what this kernel originally did, and it
 * measured 1.8x slower. Two reasons, both structural. It needs 42 cross-lane exchanges
 * per group (16 for the outer two Hadamard stages, 2 for the amax reduction, and 24
 * purely to broadcast all 32 values into lane 0, since the conversion intrinsic consumes
 * 32 values from a single lane), and `__shfl_xor` lowers those to `ds_bpermute_b32`,
 * which contends with the staged tile for the LDS pipe. And three of the four lanes then
 * throw their work away at the store. Owning the group outright removes every exchange
 * and lets a wave cover 64 groups instead of 16.
 *
 * Doing it this way is bit-exact with the four-lane version rather than merely
 * equivalent, which matters because the row direction has to reproduce AITER's packer
 * byte for byte. H32 factorises into butterfly stages h = 1, 2, 4, 8, 16; the four-lane
 * version ran h = 1, 2, 4 in-lane and h = 8, 16 as the lane^1 and lane^2 exchanges,
 * because lane L held elements [8L, 8L+8). Running all five in-lane walks the same
 * network in the same order, so every floating-point addition sees the same operands.
 * The amax reduction is a max tree and so is order-independent, and the even/odd split
 * that version assembled from four lanes' strided pieces collapses to the identity
 * even[i] = v[2i], odd[i] = v[2i+1].
 */
template <bool KBLK = false>
__device__ __forceinline__ void
mxfp6_emit_group(float (&values)[kGroupSize], const int64_t out_row, const int32_t group,
                 const int32_t nk_pad, uint8_t *__restrict__ packed,
                 uint8_t *__restrict__ packed_scale, const mxfp4_emit::FlyPackArgs fly = {}) {
#pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
        const int h = 1 << stage;
#pragma unroll
        for (int pair = 0; pair < kGroupSize / 2; ++pair) {
            const int   butterfly = pair / h;
            const int   offset    = pair % h;
            const int   i0        = butterfly * (2 * h) + offset;
            const int   i1        = i0 + h;
            const float x0        = values[i0];
            const float x1        = values[i1];
            values[i0]            = x0 + x1;
            values[i1]            = x0 - x1;
        }
    }
#pragma unroll
    for (int i = 0; i < kGroupSize; ++i)
        values[i] *= kHadamard32Norm;

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kGroupSize; ++i)
        amax = fmaxf(amax, fabsf(values[i]));

    int32_t scale_unbiased;
    if (amax == 0.0f) {
        scale_unbiased = 0;
    } else {
        const uint32_t exponent = (__builtin_bit_cast(uint32_t, amax) >> 23) & 0xFFu;
        scale_unbiased          = exponent == 0u
                                      ? -127
                                      : (exponent == 0xFFu ? 127 : static_cast<int32_t>(exponent) - 129);
        scale_unbiased          = scale_unbiased < -127 ? -127 : scale_unbiased;
        scale_unbiased          = scale_unbiased > 127 ? 127 : scale_unbiased;
    }
    const uint8_t scale_exp = static_cast<uint8_t>(scale_unbiased + 127);

    float16_t even;
    float16_t odd;
#pragma unroll
    for (int i = 0; i < kGroupSize / 2; ++i) {
        even[i] = values[2 * i];
        odd[i]  = values[2 * i + 1];
    }

    const uint32_t scale_bits =
        scale_exp == 0 ? 0x00400000u : static_cast<uint32_t>(scale_exp) << 23;
    const float mx_scale = __builtin_bit_cast(float, scale_bits);
#if defined(__gfx950__)
    const packed_fp6x32_t fp6 =
        amax == 0.0f ? packed_fp6x32_t{}
                     : __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(even, odd, mx_scale);
#else
    const packed_fp6x32_t fp6{};
#endif

    if constexpr (KBLK) {
        // FlyDSL A6W6: the same 24 bytes (C0 = bytes 0..15, C1 = 16..23) stored K128-blocked so one
        // LDS-DMA instruction reads a contiguous KiB: C0 [rows/16, K/128, 16, 64], C1 [rows/32,
        // K/128, 32, 32], C1 after C0 in one buffer; scales in FlyDSL's packed layout (nt 4, ilv
        // 0).
        if (out_row >= fly.rows || group >= fly.k128 * 4)
            return;
        const int64_t nk = fly.k128;
        const int64_t rpad =
            (static_cast<int64_t>(fly.rows) + kTileRows - 1) / kTileRows * kTileRows;
        const int64_t s = group >> 2, g = group & 3;
        const int64_t c0 = (((out_row >> 4) * nk + s) * 16 + (out_row & 15)) * 64 + g * 16;
        const int64_t c1 =
            rpad * nk * 64 + (((out_row >> 5) * nk + s) * 32 + (out_row & 31)) * 32 + g * 8;
        *reinterpret_cast<uint4_t *>(packed + c0) = *reinterpret_cast<const uint4_t *>(&fp6);
        *reinterpret_cast<uint2_t *>(packed + c1) =
            *reinterpret_cast<const uint2_t *>(reinterpret_cast<const uint8_t *>(&fp6) + 16);
        packed_scale[mxfp4_emit::fly_scale_byte(out_row, group, fly)] = scale_exp;
        return;
    }
    const int32_t tile_row  = static_cast<int32_t>(out_row / kTileRows);
    const int32_t rem       = static_cast<int32_t>(out_row % kTileRows);
    const int32_t row_block = rem / 16;
    const int32_t row16     = rem % 16;
    const int32_t step      = group / kGroupsPerKTile;
    const int32_t k_group   = group % kGroupsPerKTile;
    const int32_t block     = row_block * 64 + k_group * 16 + row16;
    const int64_t tile_base = (static_cast<int64_t>(tile_row) * nk_pad + step) * kPackedTileBytes;
    const int64_t c0_base   = tile_base + block * kC0BytesPerBlock;
    const int64_t c1_base   = tile_base + kC1PlaneOffset + block * kC1BytesPerBlock;
    *reinterpret_cast<uint4_t *>(packed + c0_base) = *reinterpret_cast<const uint4_t *>(&fp6);
    *reinterpret_cast<uint2_t *>(packed + c1_base) =
        *reinterpret_cast<const uint2_t *>(reinterpret_cast<const uint8_t *>(&fp6) + 16);

    const int32_t scale_upper = rem / 128;
    const int32_t scale_sub   = (rem % 128) / 16;
    const int64_t scale_address =
        (static_cast<int64_t>(tile_row) * nk_pad + step) * kScaleTileBytes + scale_upper * 512 +
        k_group * 128 + row16 * 8 + scale_sub;
    packed_scale[scale_address] = scale_exp;
}

// Logical block coordinates. MI355X dispatches consecutive workgroups round-robin over its 8
// XCDs, each with its own L2, so with a grid width that is a multiple of 8 a block's XCD is
// blockIdx.x % 8. The A4W4 A layout writes each block's output in short runs that a neighbour
// along the contraction completes to a 128-byte line:
//   * row direction: blocks x and x+1 each write 64 bytes of the same rows -- on different XCDs
//     in dispatch order, so no L2 ever holds the whole line;
//   * column direction: blocks y .. y+3 write 32 bytes each -- same x, so already one XCD.
// Within every 8 P blocks along x (P = 256 / TILE_N: the blocks one row-direction line spans),
// logical blocks P c .. P c + P - 1 are taken from the hardware positions on XCD c. Columns keep
// their XCD, a row line's writers share one. A trailing partial group keeps the identity. With a
// grid width that is a multiple of 16 and a height that is a multiple of 4, the dispatch order also
// walks bands of 4 block rows, so the column direction's four writers of a line run close together
// in time. Order only: every block computes what it did. (A contiguous-range-per-XCD remap fixes
// rows but breaks column locality.)
template <bool REMAP, int TILE_N>
__device__ __forceinline__ void logical_block(int32_t &bx, int32_t &by) {
    by = int32_t(blockIdx.y);
    bx = int32_t(blockIdx.x);
    if constexpr (REMAP) {
        // A row-direction line (128 bytes) takes P = 256 / TILE_N blocks along x (64-byte runs
        // at TILE_N 128, 32-byte runs at 64). Give each XCD P consecutive x.
        constexpr uint32_t P  = 256 / TILE_N;
        constexpr uint32_t G  = 8 * P; // x-extent of one round over the 8 XCDs
        const uint32_t     gx = gridDim.x, gy = gridDim.y;
        if (gx % G == 0 && gy % 4 == 0) {
            // Bands of 4 block rows; in each, a chunk of 32 dispatch slots covers rows y .. y+3
            // of 8 x-values, one per XCD, and P chunks complete G x-values. Slot w runs on XCD
            // w % 8 and becomes x = G (q / P) + P (w % 8) + q % P: the XCD depends on x alone,
            // and a column line's four writers finish within ~32 P dispatches.
            const uint32_t hw   = blockIdx.y * gx + blockIdx.x;
            const uint32_t band = hw / (4 * gx), idx = hw % (4 * gx);
            const uint32_t q = idx / 32, w = idx % 32;
            bx = int32_t(G * (q / P) + P * (w % 8) + q % P);
            by = int32_t(4 * band + w / 8);
        } else if (gx % 8 == 0) {
            const uint32_t x = blockIdx.x;
            if ((x / G + 1) * G <= gx) {
                const uint32_t r = x % G;
                bx               = int32_t((x / G) * G + P * (r % 8) + r / 8);
            }
        }
    }
}

// One emit per output format. The FP4 formats share mxfp4_emit_group's quantization and
// differ only in the store address (see mxfp4_emit::Layout).
template <MXPackFmt FMT>
__device__ __forceinline__ void
emit_group_fmt(float (&values)[kGroupSize], const int64_t out_row, const int32_t group,
               const int32_t  nk, uint8_t *__restrict__ packed, uint8_t *__restrict__ packed_scale,
               const uint32_t sr_seed, const mxfp4_emit::FlyPackArgs fly,
               const float tile_amax = -1.0f) {
    if constexpr (FMT == MXPackFmt::Fp6) {
        mxfp6_emit_group(values, out_row, group, nk, packed, packed_scale);
    } else if constexpr (FMT == MXPackFmt::Fp6KBlk) {
        mxfp6_emit_group<true>(values, out_row, group, nk, packed, packed_scale, fly);
    } else if constexpr (FMT == MXPackFmt::Fp4Blob) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::A6W4Blob>(
            values, out_row, group, nk, packed, packed_scale, 0u, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4A) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::A4W4A>(values, out_row, group, nk, packed,
                                                                packed_scale, 0u, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4ASr) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::A4W4A, true>(
            values, out_row, group, nk, packed, packed_scale, sr_seed, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4Plain) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::Plain>(values, out_row, group, nk, packed,
                                                                packed_scale, 0u, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4PlainSr) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::Plain, true>(
            values, out_row, group, nk, packed, packed_scale, sr_seed, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4BlobSr) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::A6W4Blob, true>(
            values, out_row, group, nk, packed, packed_scale, sr_seed, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4Fly) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::Fly>(values, out_row, group, nk, packed,
                                                              packed_scale, 0u, fly, tile_amax);
    } else if constexpr (FMT == MXPackFmt::Fp4FlySr) {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::Fly, true>(
            values, out_row, group, nk, packed, packed_scale, sr_seed, fly, tile_amax);
    } else {
        mxfp4_emit::mxfp4_emit_group<mxfp4_emit::Layout::A4W4B>(values, out_row, group, nk, packed,
                                                                packed_scale, 0u, fly, tile_amax);
    }
}

/*
 * Fused dual MXFP6 packer.
 *
 * The grid covers the operand padded to 256 in both dimensions rather than its logical
 * extent, so the padded rows and columns are packed too. That is not wasted work: the
 * blob has to contain a well-defined encoding of zero there, and letting the OOB reads
 * fall out of the LDS zero-fill produces it for free -- otherwise the host would have to
 * memset the whole blob, which costs more than the packing.
 */
// ROW_FMT / COL_FMT pick each direction's output format (MXPackFmt). The hybrid's FP4
// column direction (Fp4Blob) emits the column direction as MXFP4 instead of MXFP6, from the same
// staged tile. That is what makes wgrad eligible for a mixed-format GEMM: wgrad contracts the token
// dimension, so its operands are a gradient and an activation rather than the weight, and narrowing
// one of them needs a tensor packed fp6 one way and fp4 the other. Nothing else changes -- the two
// blobs share tile geometry, block indexing and the scale plane, and differ only in the code plane,
// so the switch is local to the emit.
template <typename DType, bool DO_ROW, bool DO_COL, MXFP6Prologue PROLOGUE, bool DO_COL_SUM,
          int TILE_N = kDefaultTileN, MXPackFmt ROW_FMT = MXPackFmt::Fp6,
          MXPackFmt COL_FMT = MXPackFmt::Fp6>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void quantize_mxfp6_dual_kernel(
    const DType *__restrict__ input, const DType *__restrict__ aux, const DType *__restrict__ bias,
    uint8_t *__restrict__ row_packed, uint8_t *__restrict__ row_scale,
    uint8_t *__restrict__ col_packed, uint8_t *__restrict__ col_scale, float *__restrict__ col_sum,
    const int32_t M, const int32_t N, const int32_t row_nk_pad, const int32_t col_nk_pad,
    const prologue_args_t<DType, PROLOGUE> pargs, const uint32_t sr_seed,
    const mxfp4_emit::FlyPackArgs row_fly, const mxfp4_emit::FlyPackArgs col_fly) {
    static_assert(TILE_N % kGroupSize == 0, "a staged patch must hold whole groups both ways");
    static_assert(TILE_N <= THREADS_PER_BLOCK,
                  "the column-sum pass assigns one column per thread");
    constexpr bool kQkr = PROLOGUE == MXFP6Prologue::QkNormRopeBackward;
    // The direct-to-LDS instruction lays each wave's lane payloads contiguously.
    // Use a compact pitch for that arm; the production path retains its aligned
    // padded pitch unchanged.
    // QkNormRopeBackward opts out deliberately: its head-pair reduction wants a row's
    // chunks laid out the way the padded pitch leaves them, and staging it this way
    // is markedly slower than the padded pitch on the Flux QKV shapes.
    constexpr bool kAsyncStage = MXFP6_ASYNC_STAGE && !kQkr && TILE_M == 64 &&
                                 (TILE_N == 64 || TILE_N == 128) &&
                                 THREADS_PER_BLOCK == 256;
    constexpr int       LDS_PITCH = kAsyncStage ? TILE_N : lds_pitch(TILE_N);
    __shared__ uint16_t s_tile[TILE_M][LDS_PITCH];
    // BiasGeluBackward needs its second tensor after the direct load has landed.
    // Other instantiations pay one uint16_t, not a dormant second tile.
    __shared__ uint16_t
        s_aux[kAsyncStage && PROLOGUE == MXFP6Prologue::BiasGeluBackward ? TILE_M * TILE_N : 1];

    // The QK-norm+RoPE prologue's private state. Costs the other prologues nothing: the
    // array degenerates to one element and every pass below is `if constexpr`-dead.
    constexpr int  kChunksPerRow = TILE_N / kStageVec;
    constexpr int  kDwGroups     = THREADS_PER_BLOCK / kChunksPerRow;
    // dw partials, one float per (thread, column-it-owns). See the accumulator below for why
    // a thread owns a fixed set of columns; kDwGroups threads share each one.
    __shared__ float s_dw[kQkr ? kChunksPerRow * kDwGroups * kStageVec : 1];

    constexpr bool kXcdRemap = ROW_FMT == MXPackFmt::Fp4A || ROW_FMT == MXPackFmt::Fp4B ||
                               ROW_FMT == MXPackFmt::Fp4ASr || ROW_FMT == MXPackFmt::Fp4Plain ||
                               ROW_FMT == MXPackFmt::Fp4PlainSr || COL_FMT == MXPackFmt::Fp4A ||
                               COL_FMT == MXPackFmt::Fp4B || COL_FMT == MXPackFmt::Fp4ASr ||
                               COL_FMT == MXPackFmt::Fp4Plain || COL_FMT == MXPackFmt::Fp4PlainSr ||
                               ROW_FMT == MXPackFmt::Fp4Fly || ROW_FMT == MXPackFmt::Fp4FlySr ||
                               COL_FMT == MXPackFmt::Fp4Fly || COL_FMT == MXPackFmt::Fp4FlySr;
    // Row and column packs of one launch draw from different streams.
    const uint32_t row_seed = sr_seed, col_seed = sr_seed ^ 0x5bd1e995u;
    int32_t        bx, by;
    logical_block<kXcdRemap, TILE_N>(bx, by);
    const int32_t tile_m          = by * TILE_M;
    const int32_t tile_n          = bx * TILE_N;
    const bool async_full_tile = tile_m + TILE_M <= M && tile_n + TILE_N <= N;

    // q, k and v are per-head slices of mixed_qkv at stride 3D, and TILE_N is D, so a block
    // lies wholly inside one of them. Spelled on blockIdx.x against the literal 3 rather
    // than as tile_n % (3 * head_dim): gfx950 has no integer divide, so a modulo by a
    // runtime value compiles to a ~20-instruction reciprocal sequence that every block would
    // pay before its first load could issue. The host asserts the geometry that makes this
    // valid, so the kernel does not have to re-derive it.
    const int32_t qkv_slot = kQkr ? int32_t(bx % 3) : 0; // 0 = q, 1 = k, 2 = v
    const int32_t head     = kQkr ? int32_t(bx / 3) : 0;
    const bool    normed   = kQkr && qkv_slot < 2;               // v is packed plainly

    /*
     * The QK-norm + RoPE backward, in one pass over the patch.
     *
     * This replaces the staging loop below rather than decorating it, because unlike the
     * bias/GELU prologues it does not act elementwise on `input`: it computes the tensor to
     * be packed out of nine operands, and the last term of it depends on a reduction.
     *
     *   dn   = rope_backward(g, cos, sin)      pairs adjacent, Flux is interleaved
     *   u    = x * rstd                        x is mixed_qkv, the norm's saved input
     *   m    = sum_d(dn * w * u) / D           over head_dim, i.e. this block's whole width
     *   dx   = rstd * (dn * w - u * m)         what gets packed
     *   dw  += sum_rows(dn * u)                the norm weight gradient
     *
     * Two things decide the shape of this loop, and both were measured rather than reasoned
     * about (packer/RESULTS_prologue_reduce.md sections 3 and 3a).
     *
     * It is phased over rows. `dx` needs dn and u together *after* the reduction, so one of
     * them has to survive it. Staging u in a second LDS tile is the obvious answer and takes
     * occupancy from 7 waves/SIMD to 4, which costs more than the arithmetic it saves. But
     * the loop's stride is a whole number of tile rows, so a thread visits exactly one row
     * per iteration: if it finishes that row's dx before staging the next, only the current
     * row's operands are live -- 25 floats rather than 64 -- and nothing is staged twice.
     * s_tile is written once, already holding dx, which also means no separate norm pass has
     * to read the tile back out of LDS. That saved pass is why the whole thing measures
     * cheaper than the sum of its parts measured incrementally.
     *
     * And the reduction needs no barrier. Thread t owns row t / kChunksPerRow and chunk
     * t % kChunksPerRow, so the kChunksPerRow threads sharing a row are consecutive lanes,
     * inside one wave. A butterfly over lane^1 .. lane^(kChunksPerRow/2) sums exactly them:
     * no LDS, no __syncthreads, and a fixed order, so a gradient does not depend on wave
     * arrival. Measurably faster than the LDS-and-two-barriers version.
     */
    float dw_acc[kQkr ? kStageVec : 1] = {};

    /*
     * Experimental TE-style two-stage global->LDS pipeline.
     *
     * A 64x64 patch is two independent 32x64 stages: 32 is exactly one
     * MXFP6 group in the column direction, while each row still contains two
     * complete groups. Four waves issue four dword direct-to-LDS loads per
     * lane, filling one 32x64 stage (4096 bytes) with no payload VGPRs. After
     * stage 0 lands, stage 1 is issued before stage 0's prologue/Hadamard/pack
     * work. The next vmcnt(0) is therefore below useful work, matching the
     * structure of Transformer Engine's two-buffer MXFP8 cast.
     *
     * Tail tiles retain the production staging path below. Flux's dimensions
     * are multiples of 256, so the measured call table takes this arm on every
     * block; the fallback keeps the general operator contract intact.
     */
    if constexpr (kAsyncStage && MXFP6_ASYNC_STAGE == 2) {
        if (async_full_tile) {
            constexpr int kStageRows = 32;
            constexpr int kStageBytes = kStageRows * TILE_N * sizeof(uint16_t);
            static_assert(kStageBytes == THREADS_PER_BLOCK * 16);

            const auto input_resource =
                make_buffer_resource(input, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));
            const auto aux_resource =
                make_buffer_resource(aux, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));

            const int lane      = threadIdx.x & 63;
            const int wave      = threadIdx.x >> 6;
            const int local_m32 = threadIdx.x / (TILE_N / kStageVec);
            const int local_n   = (threadIdx.x % (TILE_N / kStageVec)) * kStageVec;

            auto issue_stage = [&](const int stage) {
#pragma unroll
                for (int strip = 0; strip < 4; ++strip) {
                    // One wave instruction writes 64 contiguous dwords = 256
                    // bytes = two compact 64-element bf16 rows.
                    const int strip_row = wave * 8 + strip * 2 + lane / 32;
                    const int strip_col = (lane % 32) * 2;
                    const int global_m  = tile_m + stage * kStageRows + strip_row;
                    const int byte_offset = static_cast<int>(
                        (int64_t(global_m) * N + tile_n + strip_col) * sizeof(DType));
                    const uintptr_t tile_lds_base = reinterpret_cast<uintptr_t>(
                        &s_tile[stage * kStageRows + wave * 8 + strip * 2][0]);
                    async_load_lds_4(input_resource,
                                     reinterpret_cast<as3_uint32_ptr>(tile_lds_base), byte_offset);
                    if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward) {
                        const uintptr_t aux_lds_base = reinterpret_cast<uintptr_t>(
                            &s_aux[(stage * kStageRows + wave * 8 + strip * 2) * TILE_N]);
                        async_load_lds_4(aux_resource,
                                         reinterpret_cast<as3_uint32_ptr>(aux_lds_base),
                                         byte_offset);
                    }
                }
            };

            issue_stage(0);
            float col_sum_acc = 0.0f;

#pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                // Drain only the current stage before any lane reads it. Stage
                // 1 is issued immediately afterwards and remains in flight
                // through all of stage 0's useful work.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
                if (stage + 1 < 2)
                    issue_stage(stage + 1);

                if constexpr (PROLOGUE != MXFP6Prologue::Identity) {
                    const int local_m = stage * kStageRows + local_m32;
                    uint16_t staged[kStageVec];
                    uint16_t aux_staged[kStageVec] = {};
                    uint16_t bias_staged[kStageVec] = {};
#pragma unroll
                    for (int i = 0; i < kStageVec; ++i) {
                        staged[i] = s_tile[local_m][local_n + i];
                        if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward)
                            aux_staged[i] =
                                s_aux[local_m * TILE_N + local_n + i];
                    }
                    if constexpr (PROLOGUE == MXFP6Prologue::LnModulate) {
                        apply_ln_modulate<DType>(staged, pargs, tile_m + local_m,
                                                 tile_n + local_n, N);
                    } else if constexpr (PROLOGUE == MXFP6Prologue::GateMul) {
                        apply_gate_mul<DType>(staged, pargs, tile_m + local_m, tile_n + local_n,
                                              N);
                    } else {
                        if (bias != nullptr)
                            stage_vector(bias_staged, reinterpret_cast<const uint16_t *>(bias), 0,
                                         tile_n + local_n, N);
#pragma unroll
                        for (int i = 0; i < kStageVec; ++i) {
                            float x = to_float<DType>(staged[i]);
                            if (bias != nullptr)
                                x = round_to_dtype<DType>(x + to_float<DType>(bias_staged[i]));
                            if constexpr (PROLOGUE == MXFP6Prologue::BiasGelu) {
                                staged[i] = from_float<DType>(gelu_tanh(x));
                            } else {
                                staged[i] = from_float<DType>(
                                    gelu_tanh_backward(to_float<DType>(aux_staged[i]), x));
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < kStageVec; ++i)
                        s_tile[local_m][local_n + i] = staged[i];
                    __syncthreads();
                }

                if constexpr (DO_COL_SUM) {
                    if (threadIdx.x < TILE_N) {
#pragma unroll
                        for (int i = 0; i < kStageRows; ++i)
                            col_sum_acc +=
                                to_float<DType>(s_tile[stage * kStageRows + i][threadIdx.x]);
                    }
                }

                constexpr int kRowGroups = DO_ROW ? kStageRows * (TILE_N / kGroupSize) : 0;
                constexpr int kColGroups = DO_COL ? TILE_N : 0;
                constexpr int kGroups    = kRowGroups + kColGroups;
                static_assert(kGroups <= THREADS_PER_BLOCK);

                const int slot = threadIdx.x;
                if constexpr (DO_ROW) {
                    if (slot < kRowGroups) {
                        constexpr int kBlocksPerRow = TILE_N / kGroupSize;
                        const int local_m =
                            stage * kStageRows + slot / kBlocksPerRow;
                        const int k_block = slot % kBlocksPerRow;
                        const int n_offset = k_block * kGroupSize;
                        float values[kGroupSize];
#pragma unroll
                        for (int i = 0; i < kGroupSize; ++i)
                            values[i] =
                                to_dot_operand<DType>(s_tile[local_m][n_offset + i]);
                        emit_group_fmt<ROW_FMT>(values, tile_m + local_m,
                                                tile_n / kGroupSize + k_block, row_nk_pad,
                                                row_packed, row_scale, row_seed, row_fly);
                    }
                }
                if constexpr (DO_COL) {
                    const int col_slot = slot - kRowGroups;
                    if (col_slot >= 0 && col_slot < kColGroups) {
                        float values[kGroupSize];
#pragma unroll
                        for (int i = 0; i < kGroupSize; ++i)
                            values[i] = to_dot_operand<DType>(
                                s_tile[stage * kStageRows + i][col_slot]);
                        emit_group_fmt<COL_FMT>(values, tile_n + col_slot,
                                                tile_m / kGroupSize + stage, col_nk_pad, col_packed,
                                                col_scale, col_seed, col_fly);
                    }
                }
            }

            if constexpr (DO_COL_SUM) {
                if (threadIdx.x < TILE_N)
                    col_sum[static_cast<int64_t>(by) * N + tile_n + threadIdx.x] = col_sum_acc;
            }
            return;
        }
    }

    // Control arm for the mechanism above: use the same direct-to-LDS
    // instructions and compact layout, but issue both halves up front and
    // retain the production whole-tile prologue/emit order. Mode 1 uses four
    // dword instructions per lane; mode 3 uses one dwordx4. Comparing either
    // with mode 0 prices the load mechanism; mode 2 minus mode 1 prices the
    // staged schedule.
    if constexpr (kAsyncStage && (MXFP6_ASYNC_STAGE == 1 || MXFP6_ASYNC_STAGE == 3)) {
        if (async_full_tile) {
            constexpr int kAsyncStageRows = THREADS_PER_BLOCK * kStageVec / TILE_N;
            constexpr int kAsyncStages    = TILE_M / kAsyncStageRows;
            constexpr int kWaveRows       = 64 * kStageVec / TILE_N;
            static_assert(TILE_M % kAsyncStageRows == 0);
            const auto input_resource =
                make_buffer_resource(input, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));
            const auto aux_resource =
                make_buffer_resource(aux, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));
            const auto input_resource_vec = make_buffer_resource_vec(
                input, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));
            const auto aux_resource_vec = make_buffer_resource_vec(
                aux, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));
            const int lane = threadIdx.x & 63;
            const int wave = threadIdx.x >> 6;

#pragma unroll
            for (int stage = 0; stage < kAsyncStages; ++stage) {
                if constexpr (MXFP6_ASYNC_STAGE == 3) {
                    constexpr int kVecsPerRow = TILE_N / kStageVec;
                    const int local_m = threadIdx.x / kVecsPerRow;
                    const int local_n = (threadIdx.x % kVecsPerRow) * kStageVec;
                    const int global_m = tile_m + stage * kAsyncStageRows + local_m;
                    const int byte_offset = static_cast<int>(
                        (int64_t(global_m) * N + tile_n + local_n) * sizeof(DType));
                    const uintptr_t tile_lds_base =
                        reinterpret_cast<uintptr_t>(
                            &s_tile[stage * kAsyncStageRows + wave * kWaveRows][0]);
                    async_load_lds_16(input_resource_vec,
                                      reinterpret_cast<as3_uint32_ptr>(tile_lds_base), byte_offset);
                    if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward) {
                        const uintptr_t aux_lds_base = reinterpret_cast<uintptr_t>(
                            &s_aux[(stage * kAsyncStageRows + wave * kWaveRows) * TILE_N]);
                        async_load_lds_16(aux_resource_vec,
                                          reinterpret_cast<as3_uint32_ptr>(aux_lds_base),
                                          byte_offset);
                    }
                } else {
#pragma unroll
                    for (int strip = 0; strip < 4; ++strip) {
                        const int strip_row = wave * 8 + strip * 2 + lane / 32;
                        const int strip_col = (lane % 32) * 2;
                        const int global_m  = tile_m + stage * 32 + strip_row;
                        const int byte_offset = static_cast<int>(
                            (int64_t(global_m) * N + tile_n + strip_col) * sizeof(DType));
                        const uintptr_t tile_lds_base = reinterpret_cast<uintptr_t>(
                            &s_tile[stage * 32 + wave * 8 + strip * 2][0]);
                        async_load_lds_4(input_resource,
                                         reinterpret_cast<as3_uint32_ptr>(tile_lds_base),
                                         byte_offset);
                        if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward) {
                            const uintptr_t aux_lds_base = reinterpret_cast<uintptr_t>(
                                &s_aux[(stage * 32 + wave * 8 + strip * 2) * TILE_N]);
                            async_load_lds_4(aux_resource,
                                             reinterpret_cast<as3_uint32_ptr>(aux_lds_base),
                                             byte_offset);
                        }
                    }
                }
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();

            if constexpr (std::is_same_v<DType, bfloat16> &&
                          (PROLOGUE == MXFP6Prologue::BiasGelu ||
                           PROLOGUE == MXFP6Prologue::BiasGeluBackward)) {
                gelu_prologue_bf16<PROLOGUE, TILE_N, LDS_PITCH>(
                    s_tile, s_aux, reinterpret_cast<const uint16_t *>(bias), tile_n, N);
                __syncthreads();
            } else if constexpr (std::is_same_v<DType, bfloat16> &&
                                 PROLOGUE == MXFP6Prologue::GateMul) {
                gate_mul_prologue_bf16<TILE_N, LDS_PITCH>(s_tile, pargs, tile_m, tile_n, N);
                __syncthreads();
            } else if constexpr (PROLOGUE != MXFP6Prologue::Identity) {
                constexpr int VEC   = kStageVec;
                constexpr int ELEMS = TILE_M * TILE_N;
#pragma unroll
                for (int base = threadIdx.x * VEC; base < ELEMS;
                     base += THREADS_PER_BLOCK * VEC) {
                    const int local_m = base / TILE_N;
                    const int local_n = base % TILE_N;
                    uint16_t staged[VEC];
                    uint16_t aux_staged[VEC] = {};
                    uint16_t bias_staged[VEC] = {};
#pragma unroll
                    for (int i = 0; i < VEC; ++i) {
                        staged[i] = s_tile[local_m][local_n + i];
                        if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward)
                            aux_staged[i] = s_aux[local_m * TILE_N + local_n + i];
                    }
                    if constexpr (PROLOGUE == MXFP6Prologue::LnModulate) {
                        apply_ln_modulate<DType>(staged, pargs, tile_m + local_m,
                                                 tile_n + local_n, N);
                    } else if constexpr (PROLOGUE == MXFP6Prologue::GateMul) {
                        apply_gate_mul<DType>(staged, pargs, tile_m + local_m, tile_n + local_n,
                                              N);
                    } else {
                        if (bias != nullptr)
                            stage_vector(bias_staged, reinterpret_cast<const uint16_t *>(bias), 0,
                                         tile_n + local_n, N);
#pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            float x = to_float<DType>(staged[i]);
                            if (bias != nullptr)
                                x = round_to_dtype<DType>(x + to_float<DType>(bias_staged[i]));
                            if constexpr (PROLOGUE == MXFP6Prologue::BiasGelu) {
                                staged[i] = from_float<DType>(gelu_tanh(x));
                            } else {
                                staged[i] = from_float<DType>(
                                    gelu_tanh_backward(to_float<DType>(aux_staged[i]), x));
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < VEC; ++i)
                        s_tile[local_m][local_n + i] = staged[i];
                }
                __syncthreads();
            }
        }
    }

    if constexpr (kQkr) {
        // Named locally so the body below keeps reading as the QK-norm+RoPE code it is; the
        // kernel parameter is generic because three prologues now carry operand structs.
        const auto &qkr = pargs;
        // Which slice this block owns. Inside the `if constexpr` because the operands only
        // exist on this instantiation -- the others are handed an empty struct.
        const DType *__restrict__ qkr_grad =
            qkv_slot == 0 ? qkr.dq : (qkv_slot == 1 ? qkr.dk : qkr.dv);
        const DType *__restrict__ qkr_w    = qkv_slot == 1 ? qkr.wk : qkr.wq;
        const float *__restrict__ qkr_rstd = qkv_slot == 1 ? qkr.rstd_k : qkr.rstd_q;

        const auto *__restrict__ input_u16 = reinterpret_cast<const uint16_t *>(input);
        const auto *__restrict__ grad_u16  = reinterpret_cast<const uint16_t *>(qkr_grad);
        const auto *__restrict__ cos_u16   = reinterpret_cast<const uint16_t *>(qkr.cos);
        const auto *__restrict__ sin_u16   = reinterpret_cast<const uint16_t *>(qkr.sin);
        const auto *__restrict__ w_u16     = reinterpret_cast<const uint16_t *>(qkr_w);
        constexpr int   VEC   = kStageVec;
        constexpr int   ELEMS = TILE_M * TILE_N;
        constexpr float kInvD = 1.0f / float(TILE_N);
        // head_dim is TILE_N, which the entry point checks, so every per-head column index
        // below is the tile-local one and every head_dim stride is a compile-time constant.
        // Worth being deliberate about: writing the column as global_n % qkr.head_dim is
        // equivalent and reads more explicitly, but head_dim is a runtime value and gfx950
        // has no integer divide, so each such modulo becomes a ~20-instruction reciprocal
        // sequence in the inner loop -- and there are four operands addressed this way, all
        // of them before the first load can issue. Measured, and expensive enough to matter.
        const int32_t gN = qkr.num_heads * TILE_N; // row stride of dq/dk/dv

#pragma unroll
        for (int base = threadIdx.x * VEC; base < ELEMS; base += THREADS_PER_BLOCK * VEC) {
            const int local_m  = base / TILE_N;
            const int local_n  = base % TILE_N;
            const int global_m = tile_m + local_m;
            const int global_n = tile_n + local_n;
            // Uniform across the kChunksPerRow lanes that share this row, which is what
            // makes the butterfly below legal: they all take the same branch.
            const bool live = global_m < M;

            // The gradient is read on every block, q, k and v alike. v is packed plainly but
            // it is still packed, and this read is the `d_qkv[..., 2D:].copy_(dv)` the
            // fusion absorbs. Only the *norm's* operands are q/k-only -- guarding this one
            // on `normed` too drops a third of the largest operand, which shows up as a
            // spurious saving.
            uint16_t staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (live)
                stage_vector(staged, grad_u16, global_m, head * TILE_N + local_n, gN);

            if (normed && live) {
                // x at the packer's own (row, column); cos, sin and w at column n % D. The
                // table row is the packer's row because Flux's frequencies are per
                // (position, batch) -- which is also why there is no reuse here to exploit.
                uint16_t x_staged[VEC], c_staged[VEC], s_staged[VEC], w_staged[VEC];
                stage_vector(x_staged, input_u16, global_m, global_n, N);
                stage_vector(c_staged, cos_u16, global_m, local_n, TILE_N);
                stage_vector(s_staged, sin_u16, global_m, local_n, TILE_N);
                // One vector load: w is a single row of length D, so row 0 addresses it with
                // the same coalescing the input gets.
                stage_vector(w_staged, w_u16, 0, local_n, TILE_N);
                const float rstd = qkr_rstd[int64_t(global_m) * qkr.num_heads + head];

                // The rotation's backward. Flux sets rotary_interleaved, which pairs index
                // 2i with 2i+1 -- adjacent -- so a thread's VEC-element chunk holds VEC/2
                // whole pairs and needs nothing from any other lane. The half-split
                // convention would pair local_n with local_n + D/2, in a different thread,
                // and force a shuffle or an LDS round trip. Nothing here survives
                // rotary_interleaved = False, which the host checks.
                float dn[VEC], uh[VEC], wv[VEC];
#pragma unroll
                for (int i = 0; i < VEC; i += 2) {
                    const float g_lo = to_float<DType>(staged[i]);
                    const float g_hi = to_float<DType>(staged[i + 1]);
                    const float c_lo = to_float<DType>(c_staged[i]);
                    const float c_hi = to_float<DType>(c_staged[i + 1]);
                    const float s_lo = to_float<DType>(s_staged[i]);
                    const float s_hi = to_float<DType>(s_staged[i + 1]);
                    dn[i]            = __builtin_fmaf(g_hi, s_hi, g_lo * c_lo);
                    dn[i + 1]        = __builtin_fmaf(-g_lo, s_lo, g_hi * c_hi);
                }

                float acc = 0.0f;
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    uh[i] = to_float<DType>(x_staged[i]) * rstd;
                    wv[i] = to_float<DType>(w_staged[i]);
                    acc   = __builtin_fmaf(dn[i] * wv[i], uh[i], acc);
                    // dw = sum_rows(dn * u). Accumulated here because staging is the only
                    // point where both are live, and because this is emphatically *not* the
                    // column sum below: that one reduces the staged tile, which by then
                    // holds dx, and is the projection's bias gradient.
                    dw_acc[i] = __builtin_fmaf(dn[i], uh[i], dw_acc[i]);
                }

#pragma unroll
                for (int h = 1; h < kChunksPerRow; h <<= 1)
                    acc += __shfl_xor(acc, h, kChunksPerRow);
                const float m_row = acc * kInvD;

#pragma unroll
                for (int i = 0; i < VEC; ++i)
                    staged[i] =
                        from_float<DType>(rstd * __builtin_fmaf(-uh[i], m_row, dn[i] * wv[i]));
            }

#pragma unroll
            for (int i = 0; i < VEC; ++i)
                s_tile[local_m][local_n + i] = staged[i];
        }
    }

    // Stage the patch. Reads are coalesced along N; anything outside the logical tensor
    // becomes zero, which is what the padded region of the blob must encode.
    if constexpr (!kQkr) {
        if (!(kAsyncStage && (MXFP6_ASYNC_STAGE == 1 || MXFP6_ASYNC_STAGE == 3) &&
              async_full_tile)) {
            const auto *__restrict__ input_u16 = reinterpret_cast<const uint16_t *>(input);
            const auto *__restrict__ aux_u16   = reinterpret_cast<const uint16_t *>(aux);
            const auto *__restrict__ bias_u16  = reinterpret_cast<const uint16_t *>(bias);
            constexpr int VEC                  = kStageVec;
            constexpr int ELEMS                = TILE_M * TILE_N;
#pragma unroll
            for (int base = threadIdx.x * VEC; base < ELEMS; base += THREADS_PER_BLOCK * VEC) {
                const int local_m  = base / TILE_N;
                const int local_n  = base % TILE_N;
                const int global_m = tile_m + local_m;
                const int global_n = tile_n + local_n;

                uint16_t staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
                if (global_m < M) {
                    stage_vector(staged, input_u16, global_m, global_n, N);

                    if constexpr (PROLOGUE == MXFP6Prologue::LnModulate) {
                        // Unlike the bias/GELU prologues this one does not map a zero input
                        // to zero -- a padded column would stage -mean * rstd rather than
                        // the zero the blob's padding has to encode. The entry point
                        // requires N to be a multiple of 256 so no padded column tile
                        // exists, which is why there is no per-element guard here.
                        apply_ln_modulate<DType>(staged, pargs, global_m, global_n, N);
                    } else if constexpr (PROLOGUE == MXFP6Prologue::GateMul) {
                        // Zero in, zero out, so padded columns need no guard.
                        apply_gate_mul<DType>(staged, pargs, global_m, global_n, N);
                    } else if constexpr (PROLOGUE != MXFP6Prologue::Identity) {
                        // Every operand the prologue reads comes in through stage_vector,
                        // which zero-fills past N. That is what lets the epilogue run
                        // unguarded over the whole vector: both prologues map an all-zero
                        // input to exactly zero, so the padded columns stage as zero on
                        // their own.
                        //
                        // Zero there is not cosmetic. The grid covers the operand padded to
                        // 256 on both axes and the dual pack contracts N one way and M the
                        // other, so padding on either axis lands on a contraction axis,
                        // where a nonzero code would add a spurious term to the dot
                        // product. An earlier version read the bias directly as
                        // bias_u16[global_n + i] and needed a per-element bounds branch to
                        // stop gelu(0 + bias) from landing there; the branch cost 13
                        // instructions per element in exec mask manipulation alone, more
                        // than the activation it guarded.
                        uint16_t aux_staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
                        if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward)
                            stage_vector(aux_staged, aux_u16, global_m, global_n, N);

                        // One vector load, not one load per element. The bias is a single
                        // row of length N, so the row-staging helper addresses it correctly
                        // with row 0 and gives the same coalescing the input gets.
                        uint16_t bias_staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
                        if (bias_u16 != nullptr)
                            stage_vector(bias_staged, bias_u16, 0, global_n, N);
#pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            // The bias-add is rounded back to DType before the activation
                            // reads it. That rounding looks redundant and is not: in the
                            // graph being replaced the add is a DType tensor op, so its
                            // result is a DType value, and carrying the sum on to the
                            // activation in fp32 instead changes 21% of the bf16 codes it
                            // produces.
                            float x = to_float<DType>(staged[i]);
                            if (bias_u16 != nullptr)
                                x = round_to_dtype<DType>(x + to_float<DType>(bias_staged[i]));
                            if constexpr (PROLOGUE == MXFP6Prologue::BiasGelu) {
                                staged[i] = from_float<DType>(gelu_tanh(x));
                            } else {
                                staged[i] = from_float<DType>(
                                    gelu_tanh_backward(to_float<DType>(aux_staged[i]), x));
                            }
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < VEC; ++i)
                    s_tile[local_m][local_n + i] = staged[i];
            }
        }
    }
    if constexpr (kAsyncStage && (MXFP6_ASYNC_STAGE == 1 || MXFP6_ASYNC_STAGE == 3)) {
        if (!async_full_tile)
            __syncthreads();
    } else {
        __syncthreads();
    }

    // Combine the weight-gradient partials. Thread t held chunk t % kChunksPerRow and
    // row-group t / kChunksPerRow, so laying its VEC floats out at that (chunk, group) puts
    // one column's partials contiguous, and one thread per column then sums kDwGroups of
    // them in a fixed order. No atomics, no cross-lane op, and an order that does not depend
    // on wave arrival.
    //
    // The destination has a head axis as well as a tile-row axis because dw reduces over
    // rows *and* heads, and a block owns one head. The caller sums both.
    if constexpr (kQkr) {
        if (normed) {
            const int chunk = threadIdx.x % kChunksPerRow;
            const int group = threadIdx.x / kChunksPerRow;
#pragma unroll
            for (int i = 0; i < kStageVec; ++i)
                s_dw[(chunk * kDwGroups + group) * kStageVec + i] = dw_acc[i];
        }
        __syncthreads();
        if (normed && threadIdx.x < TILE_N) {
            const int chunk = threadIdx.x / kStageVec;
            const int j     = threadIdx.x % kStageVec;
            float     acc   = 0.0f;
#pragma unroll
            for (int gr = 0; gr < kDwGroups; ++gr)
                acc += s_dw[(chunk * kDwGroups + gr) * kStageVec + j];
            float *__restrict__ dst = qkv_slot == 0 ? pargs.dw_q : pargs.dw_k;
            dst[(int64_t(by) * pargs.num_heads + head) * TILE_N + threadIdx.x] = acc;
        }
        __syncthreads();
    }

    // Per-column sums of the staged tile, for a bias gradient whose source tensor the
    // fusion has removed from HBM. Taken on the staged values, which are post-prologue and
    // pre-Hadamard -- exactly the tensor the separate reduction kernel used to read. Rows
    // past M staged as zero and contribute nothing, so no masking is needed on M.
    if constexpr (DO_COL_SUM) {
        if (threadIdx.x < TILE_N) {
            const int32_t global_n = tile_n + threadIdx.x;
            if (global_n < N) {
                float acc = 0.0f;
                // Left rolled. Unrolling the 64 LDS reads measures identical in registers
                // and occupancy, and this pass is a second read of a tile that is already
                // resident, so there is nothing here to schedule around.
                for (int i = 0; i < TILE_M; ++i)
                    acc += to_float<DType>(s_tile[i][threadIdx.x]);
                col_sum[static_cast<int64_t>(by) * N + global_n] = acc;
            }
        }
    }

    const int32_t slot = threadIdx.x;

    // 2-D block scaling (weights): one amax per 32x32 tile of the staged patch, shared by the
    // tile's 32 row groups and 32 column groups, so both directions get the same scale and -- with
    // no Hadamard, which the host enforces -- the column codes are the row codes transposed. Taken
    // on the bf16-rounded values the emit converts. The flag is a launch constant, so every thread
    // takes the same branch and reaches both barriers.
    constexpr int    kTiles2dM = TILE_M / kGroupSize, kTiles2dN = TILE_N / kGroupSize;
    __shared__ float s_colmax2d[kTiles2dM][TILE_N];
    __shared__ float s_tilemax2d[kTiles2dM][kTiles2dN];
    const bool       tile2d = row_fly.fp4_tile2d || col_fly.fp4_tile2d;
    if (tile2d) {
        for (int c = slot; c < kTiles2dM * TILE_N; c += THREADS_PER_BLOCK) {
            const int mb = c / TILE_N, col = c % TILE_N;
            float     m = 0.0f;
            for (int i = 0; i < kGroupSize; ++i)
                m = fmaxf(m, fabsf(static_cast<float>(static_cast<__bf16>(
                                 to_dot_operand<DType>(s_tile[mb * kGroupSize + i][col])))));
            s_colmax2d[mb][col] = m;
        }
        __syncthreads();
        for (int t = slot; t < kTiles2dM * kTiles2dN; t += THREADS_PER_BLOCK) {
            const int mb = t / kTiles2dN, nb = t % kTiles2dN;
            float     m = 0.0f;
            for (int j = 0; j < kGroupSize; ++j)
                m = fmaxf(m, s_colmax2d[mb][nb * kGroupSize + j]);
            s_tilemax2d[mb][nb] = m;
        }
        __syncthreads();
    }

    // Row direction: contract along N. Each staged row contributes TILE_N/32 groups.
    if constexpr (DO_ROW) {
        constexpr int kBlocksPerRow = TILE_N / kGroupSize;
        constexpr int kRowGroups    = TILE_M * kBlocksPerRow;
#pragma unroll
        for (int gi = slot; gi < kRowGroups; gi += THREADS_PER_BLOCK) {
            const int local_m  = gi / kBlocksPerRow;
            const int k_block  = gi % kBlocksPerRow;
            const int n_offset = k_block * kGroupSize;

            float values[kGroupSize];
#pragma unroll
            for (int i = 0; i < kGroupSize; ++i)
                values[i] = to_dot_operand<DType>(s_tile[local_m][n_offset + i]);

            emit_group_fmt<ROW_FMT>(values, tile_m + local_m, tile_n / kGroupSize + k_block,
                                    row_nk_pad, row_packed, row_scale, row_seed, row_fly,
                                    tile2d ? s_tilemax2d[local_m / kGroupSize][k_block] : -1.0f);
        }
    }

    // Column direction: contract along M, i.e. pack the rows of x.T. Same emit, different
    // gather -- this is the transpose that no longer has to be materialised.
    if constexpr (DO_COL) {
        constexpr int kBlocksPerCol = TILE_M / kGroupSize;
        constexpr int kColGroups    = TILE_N * kBlocksPerCol;
#pragma unroll
        for (int gi = slot; gi < kColGroups; gi += THREADS_PER_BLOCK) {
            const int local_n  = gi / kBlocksPerCol;
            const int k_block  = gi % kBlocksPerCol;
            const int m_offset = k_block * kGroupSize;

            float values[kGroupSize];
#pragma unroll
            for (int i = 0; i < kGroupSize; ++i)
                values[i] = to_dot_operand<DType>(s_tile[m_offset + i][local_n]);

            emit_group_fmt<COL_FMT>(values, tile_n + local_n, tile_m / kGroupSize + k_block,
                                    col_nk_pad, col_packed, col_scale, col_seed, col_fly,
                                    tile2d ? s_tilemax2d[k_block][local_n / kGroupSize] : -1.0f);
        }
    }
}

constexpr int ceil_div(const int x, const int m) {
    return (x + m - 1) / m;
}

// Grid and blob strides are a function of the logical shape alone, so both entry points
// derive them the same way.
struct launch_geometry {
    int  row_nk_pad;
    int  col_nk_pad;
    dim3 grid;
    dim3 block;
};

// The bias-gradient pass is a template parameter rather than a runtime null check, so that
// a caller who does not want it provably pays nothing rather than measurably nothing. The
// cost is six kernel instantiations per dtype instead of three.
// The format pairs the packers are instantiated for: both FP6 (the A6W6 blobs), FP6 + the
// MXFP4 blob (the A6W4 wgrad hybrid), and the two A4W4 pairs -- a gradient, packed as the A
// operand both ways (dgrad contracts its columns, wgrad its rows), and an activation or
// weight, FP6 for the forward and the B operand the other way.
inline bool is_a4w4(const MXPackFmt f) {
    return f == MXPackFmt::Fp4A || f == MXPackFmt::Fp4B || f == MXPackFmt::Fp4ASr ||
           f == MXPackFmt::Fp4Plain || f == MXPackFmt::Fp4PlainSr || f == MXPackFmt::Fp4Fly ||
           f == MXPackFmt::Fp4FlySr;
}

// The FlyDSL packed-scale parameters of the current launch (MXFlyPackScope, set by the op layer).
namespace {
MXFlyPack               g_fly_row, g_fly_col;
mxfp4_emit::FlyPackArgs to_args(const MXFlyPack &p) {
    return {p.is_b, p.nt, p.ilv, p.k128, p.rows, p.fp4_round, p.fp4_had, p.fp4_tile2d};
}
} // namespace

template <typename DType, bool DO_ROW, bool DO_COL, MXFP6Prologue PROLOGUE, int TILE_N>
void launch_dual(const dim3 grid, const dim3 block, hipStream_t stream, const DType *input,
                 const DType *aux, const DType *bias, uint8_t *row_packed, uint8_t *row_scale,
                 uint8_t *col_packed, uint8_t *col_scale, float *col_sum, const int32_t M,
                 const int32_t N, int32_t row_nk_pad, int32_t col_nk_pad,
                 const prologue_args_t<DType, PROLOGUE> &pargs, const MXPackFmt row_fmt,
                 const MXPackFmt col_fmt) {
    // AITER's A4W4 operands carry no guard tiles: a row is exactly ceil(K, 256) / 2 bytes.
    if (is_a4w4(row_fmt))
        row_nk_pad = ceil_div(N, kTileRows) * (kTileRows / kKTile);
    if (is_a4w4(col_fmt))
        col_nk_pad = ceil_div(M, kTileRows) * (kTileRows / kKTile);
    const uint32_t sr_seed =
        (row_fmt == MXPackFmt::Fp4ASr || col_fmt == MXPackFmt::Fp4ASr ||
         row_fmt == MXPackFmt::Fp4PlainSr || col_fmt == MXPackFmt::Fp4PlainSr ||
         row_fmt == MXPackFmt::Fp4FlySr || col_fmt == MXPackFmt::Fp4FlySr ||
         row_fmt == MXPackFmt::Fp4BlobSr || col_fmt == MXPackFmt::Fp4BlobSr)
            ? sr_next_seed(SRStream::MXPack)
            : 0u;
    const mxfp4_emit::FlyPackArgs row_fly = to_args(mx_fly_pack_row());
    const mxfp4_emit::FlyPackArgs col_fly = to_args(mx_fly_pack_col());
    if (row_fly.fp4_tile2d || col_fly.fp4_tile2d) {
        // The tile amax is taken on the staged values in the whole-tile emit path: it must see what
        // the emit converts (no prologue, no rotation), and the experimental per-stage schedule
        // (MXFP6_ASYNC_STAGE 2) has its own emit that does not take it.
        PRIMUS_TURBO_CHECK(PROLOGUE == MXFP6Prologue::Identity,
                           "2-D block scaling is for plain packs (weights), not prologue packs");
        PRIMUS_TURBO_CHECK(MXFP6_ASYNC_STAGE != 2, "2-D block scaling needs the whole-tile emit");
    }
    auto                          go      = [&](auto r, auto c) {
        constexpr MXPackFmt R = decltype(r)::value;
        constexpr MXPackFmt C = decltype(c)::value;
        if (col_sum != nullptr) {
            quantize_mxfp6_dual_kernel<DType, DO_ROW, DO_COL, PROLOGUE, true, TILE_N, R, C>
                <<<grid, block, 0, stream>>>(input, aux, bias, row_packed, row_scale, col_packed,
                                                                           col_scale, col_sum, M, N, row_nk_pad, col_nk_pad,
                                                                           pargs, sr_seed, row_fly, col_fly);
        } else {
            quantize_mxfp6_dual_kernel<DType, DO_ROW, DO_COL, PROLOGUE, false, TILE_N, R, C>
                <<<grid, block, 0, stream>>>(input, aux, bias, row_packed, row_scale, col_packed,
                                                                           col_scale, nullptr, M, N, row_nk_pad, col_nk_pad,
                                                                           pargs, sr_seed, row_fly, col_fly);
        }
    };
    using F6       = std::integral_constant<MXPackFmt, MXPackFmt::Fp6>;
    using F4Blob   = std::integral_constant<MXPackFmt, MXPackFmt::Fp4Blob>;
    using F4A      = std::integral_constant<MXPackFmt, MXPackFmt::Fp4A>;
    using F4B      = std::integral_constant<MXPackFmt, MXPackFmt::Fp4B>;
    using F4ASr    = std::integral_constant<MXPackFmt, MXPackFmt::Fp4ASr>;
    using F4P      = std::integral_constant<MXPackFmt, MXPackFmt::Fp4Plain>;
    using F4PSr    = std::integral_constant<MXPackFmt, MXPackFmt::Fp4PlainSr>;
    using F4F      = std::integral_constant<MXPackFmt, MXPackFmt::Fp4Fly>;
    using F4BlobSr = std::integral_constant<MXPackFmt, MXPackFmt::Fp4BlobSr>;
    using F4FSr    = std::integral_constant<MXPackFmt, MXPackFmt::Fp4FlySr>;
    using F6K      = std::integral_constant<MXPackFmt, MXPackFmt::Fp6KBlk>;
    // A direction that is not emitted does not constrain the pair.
    const MXPackFmt r = DO_ROW ? row_fmt : MXPackFmt::Fp6;
    const MXPackFmt c = DO_COL ? col_fmt : MXPackFmt::Fp6;
    if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp6)
        go(F6{}, F6{});
    else if (r == MXPackFmt::Fp4A && c == MXPackFmt::Fp4A)
        go(F4A{}, F4A{});
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4B)
        go(F6{}, F4B{});
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4Blob)
        go(F6{}, F4Blob{});
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4A) // gradient, wgrad-only MXFP4
        go(F6{}, F4A{});
    else if (r == MXPackFmt::Fp4A && c == MXPackFmt::Fp6) // gradient, dgrad-only MXFP4
        go(F4A{}, F6{});
    else if (r == MXPackFmt::Fp4ASr && c == MXPackFmt::Fp4ASr) // the same three, SR
        go(F4ASr{}, F4ASr{});
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4ASr)
        go(F6{}, F4ASr{});
    else if (r == MXPackFmt::Fp4ASr && c == MXPackFmt::Fp6)
        go(F4ASr{}, F6{});
    else if (r == MXPackFmt::Fp4Plain && c == MXPackFmt::Fp4Plain) // FlyDSL operands
        go(F4P{}, F4P{});
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4Plain)
        go(F6{}, F4P{});
    else if (r == MXPackFmt::Fp4Plain && c == MXPackFmt::Fp6) // row-only plain FP4 (forward-only)
        go(F4P{}, F6{});
    else if (r == MXPackFmt::Fp4PlainSr && c == MXPackFmt::Fp4PlainSr)
        go(F4PSr{}, F4PSr{});
    else if (r == MXPackFmt::Fp4Blob && c == MXPackFmt::Fp4Blob) // A4W4 tile-blob kernels
        go(F4Blob{}, F4Blob{});
    else if (r == MXPackFmt::Fp4BlobSr && c == MXPackFmt::Fp4BlobSr)
        go(F4BlobSr{}, F4BlobSr{});
    else if (r == MXPackFmt::Fp4Blob && c == MXPackFmt::Fp6) // row-only blob (forward / eval)
        go(F4Blob{}, F6{});
    else if (r == MXPackFmt::Fp4Fly && c == MXPackFmt::Fp4Fly) // FlyDSL packed scales
        go(F4F{}, F4F{});
    else if (r == MXPackFmt::Fp4FlySr && c == MXPackFmt::Fp4FlySr)
        go(F4FSr{}, F4FSr{});
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4Fly)
        go(F6{}, F4F{});
    // Column-only SR (fmt bit 25): the backward copy rounds stochastically, the forward rows do
    // not.
    else if (r == MXPackFmt::Fp6 && c == MXPackFmt::Fp4FlySr)
        go(F6{}, F4FSr{});
    else if (r == MXPackFmt::Fp6KBlk && c == MXPackFmt::Fp4FlySr)
        go(F6K{}, F4FSr{});
    else if (r == MXPackFmt::Fp4Fly && c == MXPackFmt::Fp4FlySr)
        go(F4F{}, F4FSr{});
    else if (r == MXPackFmt::Fp4Fly && c == MXPackFmt::Fp6) // row-only fly FP4 (forward-only)
        go(F4F{}, F6{});
    else if (r == MXPackFmt::Fp6KBlk &&
             c == MXPackFmt::Fp4Fly) // FlyDSL A6W6 forward + fly FP4 backward
        go(F6K{}, F4F{});
    else if (r == MXPackFmt::Fp6KBlk && c == MXPackFmt::Fp6) // row-only (forward / eval)
        go(F6K{}, F6{});
    else
        PRIMUS_TURBO_CHECK(false, "unsupported MX pack format pair (row ", int(row_fmt), ", col ",
                           int(col_fmt), ")");
}

template <typename DType, MXFP6Prologue PROLOGUE, int TILE_N = kDefaultTileN>
void launch_fused(const dim3 grid, const dim3 block, hipStream_t stream, const DType *input,
                  const DType *aux, const DType *bias, uint8_t *row_packed, uint8_t *row_scale,
                  uint8_t *col_packed, uint8_t *col_scale, float *col_sum, const int32_t M,
                  const int32_t N, const int32_t row_nk_pad, const int32_t col_nk_pad,
                  const prologue_args_t<DType, PROLOGUE> &pargs   = {},
                  const MXPackFmt                         row_fmt = MXPackFmt::Fp6,
                  const MXPackFmt                         col_fmt = MXPackFmt::Fp6) {
    launch_dual<DType, true, true, PROLOGUE, TILE_N>(
        grid, block, stream, input, aux, bias, row_packed, row_scale, col_packed, col_scale,
        col_sum, M, N, row_nk_pad, col_nk_pad, pargs, row_fmt, col_fmt);
}

template <int TILE_N> launch_geometry geometry_for(const int M, const int N) {
    // Cover the operand padded to whole 256-row tiles in both directions: M is the row
    // count of the row-direction blob and the K extent of the column-direction one, and
    // vice versa for N, so both have to be rounded up before tiling.
    //
    // Rounding K up to 256 rather than to its own 128 granularity can push one K-tile
    // past the blob's real extent. That is in bounds, not an overrun: the two mandatory
    // guard tiles absorb it, and the excess is at most one tile because ceil(k, 256) and
    // ceil(k, 128) differ by at most 128. Those writes are dead, like everything else in
    // the guard region.
    const int m_padded = ceil_div(M, kTileRows) * kTileRows;
    const int n_padded = ceil_div(N, kTileRows) * kTileRows;
    return {ceil_div(N, kKTile) + MXFP6_GUARD_K_TILES, ceil_div(M, kKTile) + MXFP6_GUARD_K_TILES,
            dim3(ceil_div(n_padded, TILE_N), ceil_div(m_padded, TILE_M)), dim3(THREADS_PER_BLOCK)};
}

} // namespace

template <typename DType>
void quantize_mxfp6_impl(const DType *input, uint8_t *row_packed, uint8_t *row_scale,
                         uint8_t *col_packed, uint8_t *col_scale, const int M, const int N,
                         const MXFP6Direction direction, hipStream_t stream,
                         const MXPackFmt row_fmt, const MXPackFmt col_fmt) {
    constexpr auto kNoPrologue = MXFP6Prologue::Identity;

    switch (direction) {
    case MXFP6Direction::Row: {
        const auto [row_nk_pad, col_nk_pad, grid, block] =
            geometry_for<kDefaultTileN>(M, N);
        launch_dual<DType, true, false, kNoPrologue, kDefaultTileN>(
            grid, block, stream, input, nullptr, nullptr, row_packed, row_scale, col_packed,
            col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, {}, row_fmt, col_fmt);
        break;
    }
    case MXFP6Direction::Col: {
        const auto [row_nk_pad, col_nk_pad, grid, block] =
            geometry_for<kDefaultTileN>(M, N);
        launch_dual<DType, false, true, kNoPrologue, kDefaultTileN>(
            grid, block, stream, input, nullptr, nullptr, row_packed, row_scale, col_packed,
            col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, {}, row_fmt, col_fmt);
        break;
    }
    case MXFP6Direction::Dual: {
        constexpr int kIdentityTileN = 128;
        const auto [row_nk_pad, col_nk_pad, grid, block] =
            geometry_for<kIdentityTileN>(M, N);
        launch_dual<DType, true, true, kNoPrologue, kIdentityTileN>(
            grid, block, stream, input, nullptr, nullptr, row_packed, row_scale, col_packed,
            col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, {}, row_fmt, col_fmt);
        break;
    }
    }
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

// Hybrid: MXFP6 row, MXFP4 column, from one pass over the input.
//
// wgrad is `grad_w = g_col @ x_col`, contracting the token dimension, so neither operand
// is the weight and A6W4 cannot reach it -- a third of GEMM time. Narrowing one of the two
// makes it eligible, and whichever tensor is narrowed needs exactly this: fp6 in the
// direction the forward or dgrad consumes, fp4 in the direction wgrad consumes.
//
// The column blob must be sized with MXFP4's 16384-byte tile, not MXFP6's 24576; that is
// the caller's job and mxfp4_gemm_pack_sizes is the helper.
template <typename DType>
void quantize_mxfp6_row_mxfp4_col_impl(const DType *input, uint8_t *row_packed,
                                       uint8_t *row_scale, uint8_t *col_packed,
                                       uint8_t *col_scale, const int M, const int N,
                                       hipStream_t stream) {
    constexpr int kIdentityTileN = 128;
    const auto [row_nk_pad, col_nk_pad, grid, block] = geometry_for<kIdentityTileN>(M, N);
    quantize_mxfp6_dual_kernel<DType, true, true, MXFP6Prologue::Identity, false, kIdentityTileN,
                               MXPackFmt::Fp6, MXPackFmt::Fp4Blob><<<grid, block, 0, stream>>>(
        input, nullptr, nullptr, row_packed, row_scale, col_packed, col_scale, nullptr, M, N,
        row_nk_pad, col_nk_pad, {}, 0u, {}, {});
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template void quantize_mxfp6_row_mxfp4_col_impl<bfloat16>(const bfloat16 *, uint8_t *, uint8_t *,
                                                          uint8_t *, uint8_t *, const int,
                                                          const int, hipStream_t);
template void quantize_mxfp6_row_mxfp4_col_impl<float16>(const float16 *, uint8_t *, uint8_t *,
                                                         uint8_t *, uint8_t *, const int,
                                                         const int, hipStream_t);

template <typename DType>
void quantize_mxfp6_fused_impl(const DType *input, const DType *aux, const DType *bias,
                               uint8_t *row_packed, uint8_t *row_scale, uint8_t *col_packed,
                               uint8_t *col_scale, float *col_sum, const int M, const int N,
                               const MXFP6Prologue prologue, hipStream_t stream,
                               const MXPackFmt row_fmt, const MXPackFmt col_fmt) {
    // Dual only, by design: see the declaration in quantization.h.
    switch (prologue) {
    case MXFP6Prologue::Identity: {
        constexpr int kIdentityTileN = 128;
        const auto [row_nk_pad, col_nk_pad, grid, block] =
            geometry_for<kIdentityTileN>(M, N);
        launch_fused<DType, MXFP6Prologue::Identity, kIdentityTileN>(
            grid, block, stream, input, aux, bias, row_packed, row_scale, col_packed, col_scale,
            col_sum, M, N, row_nk_pad, col_nk_pad, {}, row_fmt, col_fmt);
        break;
    }
    case MXFP6Prologue::BiasGelu: {
        constexpr int kBiasGeluTileN = 128;
        const auto [row_nk_pad, col_nk_pad, grid, block] =
            geometry_for<kBiasGeluTileN>(M, N);
        launch_fused<DType, MXFP6Prologue::BiasGelu, kBiasGeluTileN>(
            grid, block, stream, input, aux, bias, row_packed, row_scale, col_packed, col_scale,
            col_sum, M, N, row_nk_pad, col_nk_pad, {}, row_fmt, col_fmt);
        break;
    }
    case MXFP6Prologue::BiasGeluBackward: {
        const auto [row_nk_pad, col_nk_pad, grid, block] =
            geometry_for<kDefaultTileN>(M, N);
        launch_fused<DType, MXFP6Prologue::BiasGeluBackward>(
            grid, block, stream, input, aux, bias, row_packed, row_scale, col_packed, col_scale,
            col_sum, M, N, row_nk_pad, col_nk_pad, {}, row_fmt, col_fmt);
        break;
    }
    case MXFP6Prologue::QkNormRopeBackward:
        // Not reachable through this entry point, and listed rather than defaulted so that
        // adding a prologue keeps failing this switch until someone decides where it goes.
        // Its operands do not fit (input, aux, bias) and it runs at a different tile width;
        // quantize_mxfp6_qk_norm_rope_bwd_impl is its entry point.
        PRIMUS_TURBO_CHECK(false,
                           "QkNormRopeBackward has its own entry point, not the fused packer");
        break;
    case MXFP6Prologue::LnModulate:
        // Same reason: four operands that are not (aux, bias). It does run at the shipped
        // tile width, so the separation is about the signature alone.
        PRIMUS_TURBO_CHECK(false, "LnModulate has its own entry point, not the fused packer");
        break;
    case MXFP6Prologue::GateMul:
        PRIMUS_TURBO_CHECK(false, "GateMul has its own entry point, not the fused packer");
        break;
    }
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType>
void quantize_mxfp6_qk_norm_rope_bwd_impl(const DType                      *input,
                                          const MXFP6QkNormRopeArgs<DType> &args,
                                          uint8_t *row_packed, uint8_t *row_scale,
                                          uint8_t *col_packed, uint8_t *col_scale, float *col_sum,
                                          const int M, const int N, hipStream_t stream,
                                          const MXPackFmt row_fmt, const MXPackFmt col_fmt) {
    // The tile width *is* head_dim, because the norm's reduction spans head_dim and the
    // kernel only knows how to reduce inside a block. Every supported head_dim needs its own
    // instantiation; 128 is Flux's and the only one built. Anything else is a hard error
    // rather than a silent fallback, because a mismatched width would produce a plausible
    // wrong gradient -- the reduction would run over the wrong extent.
    PRIMUS_TURBO_CHECK(args.head_dim == 128,
                       "QkNormRopeBackward is instantiated for head_dim 128 only");
    // What lets the kernel derive q/k/v and the head from blockIdx.x alone: with the tile
    // width at head_dim, the x grid is exactly num_heads * 3 tiles and block b owns head
    // b / 3, slice b % 3. Checked here so the kernel does not carry the arithmetic.
    PRIMUS_TURBO_CHECK(N == args.num_heads * 3 * args.head_dim,
                       "N must be num_heads * 3 * head_dim for the QKV prologue");
    constexpr int kTileN = 128;
    static_assert(kTileN % kGroupSize == 0);
    const auto [row_nk_pad, col_nk_pad, grid, block] = geometry_for<kTileN>(M, N);
    // The grid must be exactly num_heads * 3 tiles wide, with no padded column tile, or
    // blockIdx.x stops naming a (head, slice) and the extra block reads a head that is not
    // there. The grid rounds N up to 256, so this asks that N = num_heads * 3 * head_dim be
    // a multiple of 256 -- at head_dim 128, that an even number of heads. Flux has 24.
    //
    // A padded column tile is not wrong for the ordinary packer, which zero-fills it. It is
    // wrong here only because this prologue derives its operands from blockIdx.x. Supporting
    // an odd head count would mean passing num_heads to the kernel and skipping the tail
    // block, which costs every block a comparison for a case no caller has.
    PRIMUS_TURBO_CHECK(int(grid.x) == args.num_heads * 3,
                       "QkNormRopeBackward needs num_heads * 3 * head_dim to be a multiple of "
                       "256, i.e. an even head count at head_dim 128");

    launch_fused<DType, MXFP6Prologue::QkNormRopeBackward, kTileN>(
        grid, block, stream, input, nullptr, nullptr, row_packed, row_scale, col_packed, col_scale,
        col_sum, M, N, row_nk_pad, col_nk_pad, args, row_fmt, col_fmt);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType>
void quantize_mxfp6_ln_modulate_impl(const DType *input, const MXFP6LnModulateArgs<DType> &args,
                                     uint8_t *row_packed, uint8_t *row_scale, uint8_t *col_packed,
                                     uint8_t *col_scale, float *col_sum, const int M, const int N,
                                     hipStream_t stream, const MXPackFmt row_fmt,
                                     const MXPackFmt col_fmt) {
    // The prologue leaves -mean * rstd where a zero-filled column should stage zero, and
    // that column is on the column-direction blob's contraction axis. Rather than pay a
    // per-element bounds branch in the innermost loop to mask it -- the same branch the
    // bias path measured at 13 instructions per element and removed -- require the shape
    // that makes padding impossible. Every AdaLN tensor in a DiT is a multiple of 256 wide.
    PRIMUS_TURBO_CHECK(N % kTileRows == 0,
                       "LnModulate needs N to be a multiple of 256 so the grid has no padded "
                       "column tile");
    // Rows past M are guarded by the staging path itself, which never runs the prologue on
    // them, so M has no such constraint.
    PRIMUS_TURBO_CHECK(args.batch_mask >= 0 && (args.batch_mask & (args.batch_mask + 1)) == 0,
                       "LnModulate needs a power-of-two batch, passed as batch_mask = B - 1");

    constexpr int kLnModulateTileN = 128;
    const auto [row_nk_pad, col_nk_pad, grid, block] = geometry_for<kLnModulateTileN>(M, N);
    launch_fused<DType, MXFP6Prologue::LnModulate, kLnModulateTileN>(
        grid, block, stream, input, nullptr, nullptr, row_packed, row_scale, col_packed, col_scale,
        col_sum, M, N, row_nk_pad, col_nk_pad, args, row_fmt, col_fmt);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType>
void quantize_mxfp6_gate_mul_impl(const DType *input, const MXFP6GateMulArgs<DType> &args,
                                  uint8_t *row_packed, uint8_t *row_scale, uint8_t *col_packed,
                                  uint8_t *col_scale, float *col_sum, const int M, const int N,
                                  hipStream_t stream, const MXPackFmt row_fmt,
                                  const MXPackFmt col_fmt) {
    PRIMUS_TURBO_CHECK(args.batch_mask >= 0 && (args.batch_mask & (args.batch_mask + 1)) == 0,
                       "GateMul needs a power-of-two batch, passed as batch_mask = B - 1");
    constexpr int kGateMulTileN = 128;
    const auto [row_nk_pad, col_nk_pad, grid, block] = geometry_for<kGateMulTileN>(M, N);
    launch_fused<DType, MXFP6Prologue::GateMul, kGateMulTileN>(
        grid, block, stream, input, nullptr, nullptr, row_packed, row_scale, col_packed, col_scale,
        col_sum, M, N, row_nk_pad, col_nk_pad, args, row_fmt, col_fmt);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template void quantize_mxfp6_impl<bfloat16>(const bfloat16 *, uint8_t *, uint8_t *, uint8_t *,
                                            uint8_t *, const int, const int, const MXFP6Direction,
                                            hipStream_t, MXPackFmt, MXPackFmt);
template void quantize_mxfp6_impl<float16>(const float16 *, uint8_t *, uint8_t *, uint8_t *,
                                           uint8_t *, const int, const int, const MXFP6Direction,
                                           hipStream_t, MXPackFmt, MXPackFmt);

template void quantize_mxfp6_fused_impl<bfloat16>(const bfloat16 *, const bfloat16 *,
                                                  const bfloat16 *, uint8_t *, uint8_t *, uint8_t *,
                                                  uint8_t *, float *, const int, const int,
                                                  const MXFP6Prologue, hipStream_t, MXPackFmt,
                                                  MXPackFmt);
template void quantize_mxfp6_fused_impl<float16>(const float16 *, const float16 *, const float16 *,
                                                 uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                                 float *, const int, const int, const MXFP6Prologue,
                                                 hipStream_t, MXPackFmt, MXPackFmt);

template void quantize_mxfp6_qk_norm_rope_bwd_impl<bfloat16>(
    const bfloat16 *, const MXFP6QkNormRopeArgs<bfloat16> &, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, float *, const int, const int, hipStream_t, MXPackFmt, MXPackFmt);
template void quantize_mxfp6_qk_norm_rope_bwd_impl<float16>(
    const float16 *, const MXFP6QkNormRopeArgs<float16> &, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, float *, const int, const int, hipStream_t, MXPackFmt, MXPackFmt);

template void quantize_mxfp6_ln_modulate_impl<bfloat16>(const bfloat16 *,
                                                        const MXFP6LnModulateArgs<bfloat16> &,
                                                        uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                                        float *, const int, const int, hipStream_t,
                                                        MXPackFmt, MXPackFmt);
template void quantize_mxfp6_ln_modulate_impl<float16>(const float16 *,
                                                       const MXFP6LnModulateArgs<float16> &,
                                                       uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                                       float *, const int, const int, hipStream_t,
                                                       MXPackFmt, MXPackFmt);

template void quantize_mxfp6_gate_mul_impl<bfloat16>(const bfloat16 *,
                                                     const MXFP6GateMulArgs<bfloat16> &, uint8_t *,
                                                     uint8_t *, uint8_t *, uint8_t *, float *,
                                                     const int, const int, hipStream_t, MXPackFmt,
                                                     MXPackFmt);
template void quantize_mxfp6_gate_mul_impl<float16>(const float16 *,
                                                    const MXFP6GateMulArgs<float16> &, uint8_t *,
                                                    uint8_t *, uint8_t *, uint8_t *, float *,
                                                    const int, const int, hipStream_t, MXPackFmt,
                                                    MXPackFmt);

// Outside the file's anonymous namespace: the op layer (another translation unit) calls these.
void mx_fly_pack_set(const MXFlyPack &row, const MXFlyPack &col) {
    g_fly_row = row;
    g_fly_col = col;
}
MXFlyPack mx_fly_pack_row() {
    return g_fly_row;
}
MXFlyPack mx_fly_pack_col() {
    return g_fly_col;
}

} // namespace primus_turbo
