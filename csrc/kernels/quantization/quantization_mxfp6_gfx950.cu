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
// column direction previously went through `pack(x.t().contiguous())`. Profiling a
// Flux 12B step attributed 82% of the MXFP6-vs-MXFP4 step-time gap to exactly that
// materialised transpose -- 420 ms of a 512 ms gap -- dwarfing both the GEMM
// difference and the packing itself.
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

#include "primus_turbo/common.h"
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

// 1/sqrt(32) rounded to bf16, NOT the nearest float. The reference packers apply the
// rotation as a bf16 dot, so their normalisation carries bf16 precision; using the exact
// fp32 constant instead shifts every rotated value by ~1e-4 relative, which is enough to
// push values sitting just under a rounding boundary over it and change ~0.1% of codes.
constexpr float kHadamard32Norm = 0.1767578125f;

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
// build uses. The sweep that chose them ran at MBS=64 row extents, which is why the
// MBS=32 extents (8192 rows) were re-examined separately.
#ifndef MXFP6_TILE_M
#define MXFP6_TILE_M 64
#endif
// A 256-thread block, from a cold-operand sweep of TILE_M 32..128 x TILE_N 64..256 x
// block 128..512 at the row extents MBS=32 and MBS=64 actually present. Cold is the
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
using prologue_args_t = std::conditional_t<PROLOGUE == MXFP6Prologue::QkNormRopeBackward,
                                           MXFP6QkNormRopeArgs<DType>, MXFP6NoPrologueArgs>;

using packed_fp6x32_t = uint32_t __attribute__((ext_vector_type(6)));
using uint4_t         = uint32_t __attribute__((ext_vector_type(4)));
using uint2_t         = uint32_t __attribute__((ext_vector_type(2)));
using float16_t       = float __attribute__((ext_vector_type(16)));

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
__device__ __forceinline__ void mxfp6_emit_group(float (&values)[kGroupSize], const int64_t out_row,
                                                 const int32_t group, const int32_t nk_pad,
                                                 uint8_t *__restrict__ packed,
                                                 uint8_t *__restrict__ packed_scale) {
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

/*
 * Fused dual MXFP6 packer.
 *
 * The grid covers the operand padded to 256 in both dimensions rather than its logical
 * extent, so the padded rows and columns are packed too. That is not wasted work: the
 * blob has to contain a well-defined encoding of zero there, and letting the OOB reads
 * fall out of the LDS zero-fill produces it for free -- otherwise the host would have to
 * memset the whole blob, which costs more than the packing.
 */
template <typename DType, bool DO_ROW, bool DO_COL, MXFP6Prologue PROLOGUE, bool DO_COL_SUM,
          int TILE_N = kDefaultTileN>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void quantize_mxfp6_dual_kernel(
    const DType *__restrict__ input, const DType *__restrict__ aux, const DType *__restrict__ bias,
    uint8_t *__restrict__ row_packed, uint8_t *__restrict__ row_scale,
    uint8_t *__restrict__ col_packed, uint8_t *__restrict__ col_scale, float *__restrict__ col_sum,
    const int32_t M, const int32_t N, const int32_t row_nk_pad, const int32_t col_nk_pad,
    const prologue_args_t<DType, PROLOGUE> qkr) {
    static_assert(TILE_N % kGroupSize == 0, "a staged patch must hold whole groups both ways");
    static_assert(TILE_N <= THREADS_PER_BLOCK,
                  "the column-sum pass assigns one column per thread");
    constexpr int       LDS_PITCH = lds_pitch(TILE_N);
    __shared__ uint16_t s_tile[TILE_M][LDS_PITCH];

    // The QK-norm+RoPE prologue's private state. Costs the other prologues nothing: the
    // array degenerates to one element and every pass below is `if constexpr`-dead.
    constexpr bool kQkr          = PROLOGUE == MXFP6Prologue::QkNormRopeBackward;
    constexpr int  kChunksPerRow = TILE_N / kStageVec;
    constexpr int  kDwGroups     = THREADS_PER_BLOCK / kChunksPerRow;
    // dw partials, one float per (thread, column-it-owns). See the accumulator below for why
    // a thread owns a fixed set of columns; kDwGroups threads share each one.
    __shared__ float s_dw[kQkr ? kChunksPerRow * kDwGroups * kStageVec : 1];

    const int32_t tile_m = blockIdx.y * TILE_M;
    const int32_t tile_n = blockIdx.x * TILE_N;

    // q, k and v are per-head slices of mixed_qkv at stride 3D, and TILE_N is D, so a block
    // lies wholly inside one of them. Spelled on blockIdx.x against the literal 3 rather
    // than as tile_n % (3 * head_dim): gfx950 has no integer divide, so a modulo by a
    // runtime value compiles to a ~20-instruction reciprocal sequence that every block would
    // pay before its first load could issue. The host asserts the geometry that makes this
    // valid, so the kernel does not have to re-derive it.
    const int32_t qkv_slot = kQkr ? int32_t(blockIdx.x % 3) : 0; // 0 = q, 1 = k, 2 = v
    const int32_t head     = kQkr ? int32_t(blockIdx.x / 3) : 0;
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

    if constexpr (kQkr) {
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

                if constexpr (PROLOGUE != MXFP6Prologue::Identity) {
                    // Every operand the prologue reads comes in through stage_vector, which
                    // zero-fills past N. That is what lets the epilogue run unguarded over
                    // the whole vector: both prologues map an all-zero input to exactly
                    // zero, so the padded columns stage as zero on their own.
                    //
                    // Zero there is not cosmetic. The grid covers the operand padded to 256
                    // on both axes and the dual pack contracts N one way and M the other,
                    // so padding on either axis lands on a contraction axis, where a
                    // nonzero code would add a spurious term to the dot product. An
                    // earlier version read the bias directly as bias_u16[global_n + i] and
                    // needed a per-element bounds branch to stop gelu(0 + bias) from
                    // landing there; the branch cost 13 instructions per element in exec
                    // mask manipulation alone, more than the activation it guarded.
                    uint16_t aux_staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if constexpr (PROLOGUE == MXFP6Prologue::BiasGeluBackward)
                        stage_vector(aux_staged, aux_u16, global_m, global_n, N);

                    // One vector load, not one load per element. The bias is a single row
                    // of length N, so the row-staging helper addresses it correctly with
                    // row 0 and gives the same coalescing the input gets.
                    uint16_t bias_staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (bias_u16 != nullptr)
                        stage_vector(bias_staged, bias_u16, 0, global_n, N);
#pragma unroll
                    for (int i = 0; i < VEC; ++i) {
                        // The bias-add is rounded back to DType before the activation reads
                        // it. That rounding looks redundant and is not: in the graph being
                        // replaced the add is a DType tensor op, so its result is a DType
                        // value, and carrying the sum on to the activation in fp32 instead
                        // changes 21% of the bf16 codes it produces.
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
    __syncthreads();

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
            float *__restrict__ dst = qkv_slot == 0 ? qkr.dw_q : qkr.dw_k;
            dst[(int64_t(blockIdx.y) * qkr.num_heads + head) * TILE_N + threadIdx.x] = acc;
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
                col_sum[static_cast<int64_t>(blockIdx.y) * N + global_n] = acc;
            }
        }
    }

    const int32_t slot = threadIdx.x;

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

            mxfp6_emit_group(values, tile_m + local_m, tile_n / kGroupSize + k_block, row_nk_pad,
                             row_packed, row_scale);
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

            mxfp6_emit_group(values, tile_n + local_n, tile_m / kGroupSize + k_block, col_nk_pad,
                             col_packed, col_scale);
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
template <typename DType, MXFP6Prologue PROLOGUE, int TILE_N = kDefaultTileN>
void launch_fused(const dim3 grid, const dim3 block, hipStream_t stream, const DType *input,
                  const DType *aux, const DType *bias, uint8_t *row_packed, uint8_t *row_scale,
                  uint8_t *col_packed, uint8_t *col_scale, float *col_sum, const int32_t M,
                  const int32_t N, const int32_t row_nk_pad, const int32_t col_nk_pad,
                  const prologue_args_t<DType, PROLOGUE> &qkr = {}) {
    if (col_sum != nullptr) {
        quantize_mxfp6_dual_kernel<DType, true, true, PROLOGUE, true, TILE_N>
            <<<grid, block, 0, stream>>>(input, aux, bias, row_packed, row_scale, col_packed,
                                         col_scale, col_sum, M, N, row_nk_pad, col_nk_pad, qkr);
    } else {
        quantize_mxfp6_dual_kernel<DType, true, true, PROLOGUE, false, TILE_N>
            <<<grid, block, 0, stream>>>(input, aux, bias, row_packed, row_scale, col_packed,
                                         col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, qkr);
    }
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
                         const MXFP6Direction direction, hipStream_t stream) {
    const auto [row_nk_pad, col_nk_pad, grid, block] = geometry_for<kDefaultTileN>(M, N);
    constexpr auto kNoPrologue                       = MXFP6Prologue::Identity;

    switch (direction) {
    case MXFP6Direction::Row:
        quantize_mxfp6_dual_kernel<DType, true, false, kNoPrologue, false>
            <<<grid, block, 0, stream>>>(input, nullptr, nullptr, row_packed, row_scale, col_packed,
                                         col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, {});
        break;
    case MXFP6Direction::Col:
        quantize_mxfp6_dual_kernel<DType, false, true, kNoPrologue, false>
            <<<grid, block, 0, stream>>>(input, nullptr, nullptr, row_packed, row_scale, col_packed,
                                         col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, {});
        break;
    case MXFP6Direction::Dual:
        quantize_mxfp6_dual_kernel<DType, true, true, kNoPrologue, false>
            <<<grid, block, 0, stream>>>(input, nullptr, nullptr, row_packed, row_scale, col_packed,
                                         col_scale, nullptr, M, N, row_nk_pad, col_nk_pad, {});
        break;
    }
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType>
void quantize_mxfp6_fused_impl(const DType *input, const DType *aux, const DType *bias,
                               uint8_t *row_packed, uint8_t *row_scale, uint8_t *col_packed,
                               uint8_t *col_scale, float *col_sum, const int M, const int N,
                               const MXFP6Prologue prologue, hipStream_t stream) {
    const auto [row_nk_pad, col_nk_pad, grid, block] = geometry_for<kDefaultTileN>(M, N);

    // Dual only, by design: see the declaration in quantization.h.
    switch (prologue) {
    case MXFP6Prologue::Identity:
        launch_fused<DType, MXFP6Prologue::Identity>(grid, block, stream, input, aux, bias,
                                                     row_packed, row_scale, col_packed, col_scale,
                                                     col_sum, M, N, row_nk_pad, col_nk_pad);
        break;
    case MXFP6Prologue::BiasGelu:
        launch_fused<DType, MXFP6Prologue::BiasGelu>(grid, block, stream, input, aux, bias,
                                                     row_packed, row_scale, col_packed, col_scale,
                                                     col_sum, M, N, row_nk_pad, col_nk_pad);
        break;
    case MXFP6Prologue::BiasGeluBackward:
        launch_fused<DType, MXFP6Prologue::BiasGeluBackward>(
            grid, block, stream, input, aux, bias, row_packed, row_scale, col_packed, col_scale,
            col_sum, M, N, row_nk_pad, col_nk_pad);
        break;
    case MXFP6Prologue::QkNormRopeBackward:
        // Not reachable through this entry point, and listed rather than defaulted so that
        // adding a prologue keeps failing this switch until someone decides where it goes.
        // Its operands do not fit (input, aux, bias) and it runs at a different tile width;
        // quantize_mxfp6_qk_norm_rope_bwd_impl is its entry point.
        PRIMUS_TURBO_CHECK(false,
                           "QkNormRopeBackward has its own entry point, not the fused packer");
        break;
    }
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template <typename DType>
void quantize_mxfp6_qk_norm_rope_bwd_impl(const DType *input,
                                          const MXFP6QkNormRopeArgs<DType> &args,
                                          uint8_t *row_packed, uint8_t *row_scale,
                                          uint8_t *col_packed, uint8_t *col_scale, float *col_sum,
                                          const int M, const int N, hipStream_t stream) {
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
        col_sum, M, N, row_nk_pad, col_nk_pad, args);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template void quantize_mxfp6_impl<bfloat16>(const bfloat16 *, uint8_t *, uint8_t *, uint8_t *,
                                            uint8_t *, const int, const int, const MXFP6Direction,
                                            hipStream_t);
template void quantize_mxfp6_impl<float16>(const float16 *, uint8_t *, uint8_t *, uint8_t *,
                                           uint8_t *, const int, const int, const MXFP6Direction,
                                           hipStream_t);

template void quantize_mxfp6_fused_impl<bfloat16>(const bfloat16 *, const bfloat16 *,
                                                  const bfloat16 *, uint8_t *, uint8_t *, uint8_t *,
                                                  uint8_t *, float *, const int, const int,
                                                  const MXFP6Prologue, hipStream_t);
template void quantize_mxfp6_fused_impl<float16>(const float16 *, const float16 *, const float16 *,
                                                 uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                                 float *, const int, const int, const MXFP6Prologue,
                                                 hipStream_t);

template void quantize_mxfp6_qk_norm_rope_bwd_impl<bfloat16>(const bfloat16 *,
                                                             const MXFP6QkNormRopeArgs<bfloat16> &,
                                                             uint8_t *, uint8_t *, uint8_t *,
                                                             uint8_t *, float *, const int,
                                                             const int, hipStream_t);
template void quantize_mxfp6_qk_norm_rope_bwd_impl<float16>(const float16 *,
                                                            const MXFP6QkNormRopeArgs<float16> &,
                                                            uint8_t *, uint8_t *, uint8_t *,
                                                            uint8_t *, float *, const int,
                                                            const int, hipStream_t);

} // namespace primus_turbo
