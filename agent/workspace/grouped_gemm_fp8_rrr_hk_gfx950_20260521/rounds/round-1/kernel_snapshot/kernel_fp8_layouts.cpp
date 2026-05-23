#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// =============================================================================
// Tuning constants (BF16-style: single block of constexpr ints + plain #defines
// for values that need stringification into inline asm).
// =============================================================================

// Block / tile dims
constexpr int BLOCK_SIZE        = 256;
constexpr int K_BLOCK           = 128;
constexpr int WARPS_M           = 2;
constexpr int WARPS_N           = 4;
constexpr int MIN_BLOCKS_PER_CU = 2;

// Block-swizzle (XCD-aware tile mapping)
constexpr int BLOCK_SWIZZLE_NUM_XCDS = 8;

// 4-wave RCR dispatch thresholds
constexpr int RCR_4WAVE_MIN_GRID  = 3200;
constexpr int RCR_4WAVE_MAX_K     = 8192;

constexpr int RCR_TWO_TILE_MIN_KI = 28;

// Derived block dims
constexpr int BLK = BLOCK_SIZE, BK = K_BLOCK;
constexpr int HB  = BLK / 2;
constexpr int _NUM_WARPS    = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS  = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
constexpr int TAIL_BLOCK_M  = 16;
constexpr int TAIL_BLOCK_N  = 16;

// Stringification utilities for inline-asm wait counters
#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x)  asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt("  TK_STRINGIFY(x) ")")
#define TK_PRAGMA_UNROLL(x) _Pragma(TK_STRINGIFY(unroll x))

#define RCR_PREFETCH_LGKM       8
#define RCR_INIT0_VMCNT         4
#define RCR_INIT1_VMCNT         6
#define RCR_STEADY_VMCNT        8
#define RCR_EPILOGUE_VMCNT      4
#define RCR_TWO_TILE_MID_VMCNT  6
#ifndef RCR_KTAIL_VMCNT
#define RCR_KTAIL_VMCNT         8
#endif
#define RRR_PREFETCH_LGKM       12
#define RRR_INIT0_VMCNT         4
#define RRR_INIT1_VMCNT         6
#define RRR_STEADY_VMCNT        4
#define RRR_EPILOGUE_VMCNT      2
// (Tried 2026-05-15: a two-K-tile main loop for RRR mirroring RCR's
// `main_loop_iter`. SNR check passed but perf was within run-to-run noise
// at dense 8K (best 2842T two-tile vs 2826-2843T single-tile across 3
// runs). Conclusion: RRR's bottleneck is LDS read bandwidth — B-load
// uses ds_read_b64_tr_b8 carrying 8B/lane vs RCR's ds_read_b128 16B/lane.
// Loop unrolling can't widen a saturated LDS pipe. Code reverted; the
// LDS-layout change for B (Option B) is the actionable lever.)
#define CRR_PREFETCH_LGKM       3
#define CRR_INIT0_VMCNT         2
#define CRR_INIT1_VMCNT         6
#define CRR_STEADY_VMCNT        4
#define CRR_EPILOGUE_VMCNT      2

// Per-layout main-loop unroll counts (stringified into #pragma unroll N).
#define RCR_MAIN_UNROLL 2
#define RRR_MAIN_UNROLL 2
#define CRR_MAIN_UNROLL 1
#define VARK_MAIN_UNROLL 2

// 2026-05-19: explicit lgkmcnt(0) drain after s_barrier is potentially redundant —
// barrier should imply lgkmcnt drain on gfx950. MAYBE_DRAIN_LGKM macro wraps
// all 79 explicit drains; toggle to test perf impact.
#ifndef DROP_REDUNDANT_LGKM_DRAIN
#define DROP_REDUNDANT_LGKM_DRAIN 1
#endif
#if DROP_REDUNDANT_LGKM_DRAIN
#define MAYBE_DRAIN_LGKM() ((void)0)
#else
#define MAYBE_DRAIN_LGKM() asm volatile("s_waitcnt lgkmcnt(0)")
#endif

#ifndef VARK_DROP_REDUNDANT_LGKM_DRAIN
#define VARK_DROP_REDUNDANT_LGKM_DRAIN 0
#endif

#ifndef VARK_SW_PIPE_HOIST_AHEAD
#define VARK_SW_PIPE_HOIST_AHEAD 0
#endif

#ifndef VARK_DROP_BARRIER_2
#define VARK_DROP_BARRIER_2 0
#endif

#ifndef VARK_HOIST_PREFETCH_INTO_HALF1
#define VARK_HOIST_PREFETCH_INTO_HALF1 0
#endif

#ifndef VARK_DROP_BARRIER_4
#define VARK_DROP_BARRIER_4 0
#endif

// Sched/wait barrier helpers
#define RRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define RCR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define CRR_STEADY_MID_BARRIER() __builtin_amdgcn_s_barrier()
#define CRR_MMA_BEGIN() __builtin_amdgcn_s_setprio(1)
#define CRR_MMA_END()   __builtin_amdgcn_s_setprio(0)

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

enum class Layout { RCR, RRR, CRR };

// Row-layout shared/register tiles (for A in RCR/RRR, B in RCR)
using ST_row    = st_fp8e4m3<HB, BK, st_16x128_s>;    // 128×128, M/N rows × K cols
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

// Col-layout register tiles (for B in RRR, A/B in CRR)
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;  // 128×64
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;  // 128×32

using ST_v2     = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a    = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;

static_assert(sizeof(A_row_reg) == sizeof(A_col_reg));
static_assert(alignof(A_row_reg) == alignof(A_col_reg));
static_assert(sizeof(B_row_reg) == sizeof(B_col_reg));
static_assert(alignof(B_row_reg) == alignof(B_col_reg));

// Cooperative col-major load from a v2/v2a-swizzled FP8 LDS tile.
// Two `ds_read_b64_tr_b8` per lane per K_HALF (offset:0 + offset:1024).
template<typename RT, int K_HALF, typename ST>
__device__ __forceinline__ void load_col_from_st_half(
    RT& dst, const ST& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;

    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k   = (k_row & 7) << 4;

    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);

        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:0\n"
            "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr)
            : "memory"
        );
    }
}

template<typename RT, typename ST>
__device__ __forceinline__ void load_col_from_st(
    RT& dst, const ST& tile, int col_start)
{
    load_col_from_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_st_half<RT, 1>(dst, tile, col_start);
}

__device__ __forceinline__ void rrr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_col_reg& b)
{
    mma_AB(acc, a, b, acc);
}

// AGPR-pinned variant template — selectable per template instantiation.
// For FUSED_KTAIL=false (8K-relevant), USE_AGPR=true gives spill -40%.
// For FUSED_KTAIL=true (gpt_oss), USE_AGPR=true gives spill +19% — bad.
// Caller dispatches: rrr_mma_agpr_t<!FUSED_KTAIL>(...).
template<bool USE_AGPR>
__device__ __forceinline__ void rrr_mma_agpr_t(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_col_reg& b)
{
    if constexpr (USE_AGPR) {
        using D_T = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
        #pragma unroll
        for (int n = 0; n < D_T::height; n++) {
            #pragma unroll
            for (int m = 0; m < D_T::width; m++) {
                #pragma unroll
                for (int k = 0; k < A_row_reg::width; k++) {
                    mfma1616128_agpr(
                        acc.tiles[n][m].data,
                        a.tiles[n][k].data,
                        b.tiles[k][m].data,
                        acc.tiles[n][m].data);
                }
            }
        }
    } else {
        mma_AB(acc, a, b, acc);
    }
}

// Backwards-compatible non-template name (always-AGPR for back-compat with
// existing main+epi sites).
__device__ __forceinline__ void rrr_mma_agpr(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_col_reg& b)
{
    rrr_mma_agpr_t<true>(acc, a, b);
}

__device__ __forceinline__ void rcr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_row_reg& b)
{
    mma_ABt(acc, a, b, acc);
}

// AGPR-pinned RCR variant — for dense `gemm_kernel<RCR>`. mma_ABt(A, B^T)
// uses same mfma1616128 underneath as mma_AB → mfma1616128_agpr works.
// For RCR the B operand is K-major in register (B_row_reg), no transpose
// at register level — the ABt name reflects mathematical form, not
// register layout. The asm is the same v_mfma_f32_16x16x128_f8f6f4.
__device__ __forceinline__ void rcr_mma_agpr(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_row_reg& b)
{
    using D_T = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
    #pragma unroll
    for (int n = 0; n < D_T::height; n++) {
        #pragma unroll
        for (int m = 0; m < D_T::width; m++) {
            #pragma unroll
            for (int k = 0; k < A_row_reg::width; k++) {
                mfma1616128_agpr(
                    acc.tiles[n][m].data,
                    a.tiles[n][k].data,
                    b.tiles[m][k].data,  // ABt: B's m-axis is OUTPUT N, k-axis matches A
                    acc.tiles[n][m].data);
            }
        }
    }
}

// Selectable template variant of rcr_mma_agpr — mirrors rrr_mma_agpr_t.
// USE_AGPR=true: cd-copy AGPR via mfma1616128_agpr (same as dense uses
// for gemm_kernel<RCR> with spill=0). USE_AGPR=false: builtin mma_ABt.
// For grouped_rcr_kernel: use rcr_mma_agpr_t<!FUSED_KTAIL>(...) so the
// FUSED_KTAIL=true instances stay on builtin (matching the ktail block
// at line ~2214, which also uses builtin to avoid AGPR/VGPR boundary
// cost). FUSED_KTAIL=false instances use AGPR throughout main+epi.
template<bool USE_AGPR>
__device__ __forceinline__ void rcr_mma_agpr_t(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a,
    const B_row_reg& b)
{
    if constexpr (USE_AGPR) {
        rcr_mma_agpr(acc, a, b);
    } else {
        mma_ABt(acc, a, b, acc);
    }
}

__device__ __forceinline__ void rcr_mma_32(
    rt_fl<RBM, RBN, col_l, rt_32x32_s>& acc,
    const rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>& a,
    const rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>& b)
{
    mma_ABt(acc, a, b, acc);
}

// 32×32×64 single-acc analog of rrr_mma_agpr_t (block 2 of 32×32 structural
// rewrite per [[fp8-rrr-32x32-foundation]]). For RBM=64, RBN=32, BK=128:
// 2×1×2 = 4 MFMAs/call vs 4×2×1 = 8 MFMAs/call in 16×16×128 wrapper.
// Same K throughput, half the issue count.
using A_row_reg_32 = rt_fp8e4m3<RBM, BK, row_l, rt_32x64_s>;
using B_col_reg_32 = rt_fp8e4m3<BK, RBN, col_l, rt_64x32_s>;

template<bool USE_AGPR>
__device__ __forceinline__ void rrr_mma_32_agpr_t(
    rt_fl<RBM, RBN, col_l, rt_32x32_s>& acc,
    const A_row_reg_32& a,
    const B_col_reg_32& b)
{
    if constexpr (USE_AGPR) {
        using D_T = rt_fl<RBM, RBN, col_l, rt_32x32_s>;
        #pragma unroll
        for (int n = 0; n < D_T::height; n++) {
            #pragma unroll
            for (int m = 0; m < D_T::width; m++) {
                #pragma unroll
                for (int k = 0; k < A_row_reg_32::width; k++) {
                    mfma323264_agpr_inplace(
                        acc.tiles[n][m].data,
                        a.tiles[n][k].data,
                        b.tiles[k][m].data);
                }
            }
        }
    } else {
        mma_AB(acc, a, b, acc);
    }
}

__attribute__((used)) static __device__ void
__lever_force_instantiate_rrr_mma_32_agpr() {
    rt_fl<RBM, RBN, col_l, rt_32x32_s> dummy_acc{};
    A_row_reg_32 dummy_a{};
    B_col_reg_32 dummy_b{};
    rrr_mma_32_agpr_t<true>(dummy_acc, dummy_a, dummy_b);
    rrr_mma_32_agpr_t<false>(dummy_acc, dummy_a, dummy_b);
}

// Register-cost probe (block 2a per [[fp8-rrr-32x32-foundation]]). Minimal
// __global__ kernel that loops rrr_mma_32_agpr_t<true>. Because __device__
// stubs don't appear in gfx950 KD notes, we need a __global__ to surface
// V/A/spill numbers via clang-offload-bundler + llvm-readobj --notes.
// Compares register footprint to existing grouped_gemm_fp8_kernel<RRR,256,*,*>
// baseline (V=256 A=128 spill=61 per [[fp8-rrr-attempt-h5-diag]]).
template<int LOOP_ITERS>
__global__ void __probe_rrr_mma_32_agpr(
    const fp8e4m3 *__restrict__ a_ptr,
    const fp8e4m3 *__restrict__ b_ptr,
    float        *__restrict__ d_ptr)
{
    rt_fl<RBM, RBN, col_l, rt_32x32_s> acc{};

    A_row_reg_32 a;
    B_col_reg_32 b;

    // Load A and B from gmem as raw bytes — the test cares about the
    // mma body's reg cost, not the load path's. One b128 load per lane
    // fills enough of each operand to keep the compiler honest.
    const int lane = kittens::laneid();
    const __uint128_t *a_src = reinterpret_cast<const __uint128_t*>(a_ptr) + lane;
    const __uint128_t *b_src = reinterpret_cast<const __uint128_t*>(b_ptr) + lane;
    #pragma unroll
    for (int n = 0; n < A_row_reg_32::height; ++n) {
        #pragma unroll
        for (int k = 0; k < A_row_reg_32::width; ++k) {
            *reinterpret_cast<__uint128_t*>(&a.tiles[n][k].data[0]) = a_src[n * 2 + k];
            *reinterpret_cast<__uint128_t*>(&a.tiles[n][k].data[4]) = a_src[(n * 2 + k) + 16];
        }
    }
    #pragma unroll
    for (int kk = 0; kk < B_col_reg_32::height; ++kk) {
        #pragma unroll
        for (int m = 0; m < B_col_reg_32::width; ++m) {
            *reinterpret_cast<__uint128_t*>(&b.tiles[kk][m].data[0]) = b_src[kk + m];
            *reinterpret_cast<__uint128_t*>(&b.tiles[kk][m].data[4]) = b_src[(kk + m) + 8];
        }
    }

    #pragma unroll 1
    for (int it = 0; it < LOOP_ITERS; ++it) {
        rrr_mma_32_agpr_t<true>(acc, a, b);
    }

    // Store back so acc isn't DCE'd. Strided write per lane.
    using D_T = rt_fl<RBM, RBN, col_l, rt_32x32_s>;
    float *d_dst = d_ptr + lane * 16;
    #pragma unroll
    for (int n = 0; n < D_T::height; ++n) {
        #pragma unroll
        for (int m = 0; m < D_T::width; ++m) {
            *reinterpret_cast<float2*>(d_dst + 0) = acc.tiles[n][m].data[0];
            *reinterpret_cast<float2*>(d_dst + 2) = acc.tiles[n][m].data[1];
            *reinterpret_cast<float2*>(d_dst + 4) = acc.tiles[n][m].data[2];
            *reinterpret_cast<float2*>(d_dst + 6) = acc.tiles[n][m].data[3];
            *reinterpret_cast<float2*>(d_dst + 8) = acc.tiles[n][m].data[4];
            *reinterpret_cast<float2*>(d_dst + 10) = acc.tiles[n][m].data[5];
            *reinterpret_cast<float2*>(d_dst + 12) = acc.tiles[n][m].data[6];
            *reinterpret_cast<float2*>(d_dst + 14) = acc.tiles[n][m].data[7];
            d_dst += 16 * 64;
        }
    }
}

template __global__ void __probe_rrr_mma_32_agpr<1>(const fp8e4m3*, const fp8e4m3*, float*);
template __global__ void __probe_rrr_mma_32_agpr<4>(const fp8e4m3*, const fp8e4m3*, float*);
template __global__ void __probe_rrr_mma_32_agpr<16>(const fp8e4m3*, const fp8e4m3*, float*);

template<typename A_RT_32x64>
__device__ __forceinline__ void load_a_kt_32x64(
    A_RT_32x64& A_tile,
    i32x4 a_srsrc_kt,
    int M_warp_base,
    int row_lane,
    int k_lane_byte,
    int a_row_stride_bytes,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h = 0; h < A_RT_32x64::height; ++h) {
        const int A_row_idx = M_warp_base + h * 32 + row_lane;
        const uint32_t v_base = static_cast<uint32_t>(
            A_row_idx * a_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            a_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
    }
}

template<typename B_RT_32x64>
__device__ __forceinline__ void load_b_kt_32x64(
    B_RT_32x64& B_tile,
    i32x4 b_srsrc_kt,
    int N_warp_base,
    int row_lane,
    int k_lane_byte,
    int b_row_stride_bytes,
    uint32_t b_group_byte_base,
    uint32_t K_tail_base_bytes,
    bool b128_lo_valid,
    bool b128_hi_valid)
{
    constexpr uint32_t SENTINEL = 0xFFFF0000u;

    #pragma unroll
    for (int h_b = 0; h_b < B_RT_32x64::height; ++h_b) {
        const int B_row_idx_in_group = N_warp_base + h_b * 32 + row_lane;
        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
            B_row_idx_in_group * b_row_stride_bytes +
            K_tail_base_bytes + k_lane_byte);
        const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
        const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_lo, 0, 0);
        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
            b_srsrc_kt, v_hi, 0, 0);
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
    }
}

__attribute__((used)) static __device__ void
__lever_d_round_b_force_instantiate_rcr_mma_32() {
    rt_fl<RBM, RBN, col_l, rt_32x32_s> dummy_acc{};
    rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s> dummy_a{};
    rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s> dummy_b{};
    rcr_mma_32(dummy_acc, dummy_a, dummy_b);

    i32x4 dummy_srsrc{};
    load_a_kt_32x64(dummy_a, dummy_srsrc,
                    /*M_warp_base=*/0,
                    /*row_lane=*/0,
                    /*k_lane_byte=*/0,
                    /*a_row_stride_bytes=*/0,
                    /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true,
                    /*b128_hi_valid=*/true);
    load_b_kt_32x64(dummy_b, dummy_srsrc,
                    /*N_warp_base=*/0,
                    /*row_lane=*/0,
                    /*k_lane_byte=*/0,
                    /*b_row_stride_bytes=*/0,
                    /*b_group_byte_base=*/0u,
                    /*K_tail_base_bytes=*/0u,
                    /*b128_lo_valid=*/true,
                    /*b128_hi_valid=*/true);
}

__attribute__((used)) static __device__ void
__lever_d_round_b_force_instantiate_st_32x64() {
    using ST_32x64 = st_fp8e4m3<HB, 64, st_32x64_s>;
    __shared__ ST_32x64 dummy_st;

    // Touch the type's static-member infrastructure to force full
    // template-parameter validation. Static-asserts mirror the
    // kittens-internal checks in ``st<>::`` body.
    static_assert(ST_32x64::rows == HB, "ST_32x64 rows should equal HB=128");
    static_assert(ST_32x64::cols == 64, "ST_32x64 cols should equal 64");
    static_assert(ST_32x64::underlying_subtile_rows == 32,
                  "ST_32x64 underlying subtile rows should equal 32");
    static_assert(ST_32x64::underlying_subtile_cols == 64,
                  "ST_32x64 underlying subtile cols should equal 64");
    static_assert(ST_32x64::underlying_subtile_bytes_per_thread == 16,
                  "ST_32x64 should dispatch the fp8 bytes_per_thread=16 branch");

    // Exercise the swizzle functor at compile time via a device call
    // path; LLVM DCE removes the dead reference after instantiation.
    (void)ST_32x64::swizzle({0, 0});
    (void)dummy_st;
}

__device__ __forceinline__ void crr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b)
{
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    mma_AB(acc, a_row, b, acc);
}

// In-place AGPR CRR — uses mfma1616128_agpr_inplace where the +a constraint
// reads/writes D directly. acc is both D and C in crr_mma; the inplace
// variant avoids the per-call cd-copy at the function boundary that
// otherwise inserts v_accvgpr_write/read across CRR's barrier-separated
// call sites. Without this, the compiler picks agpr_count=0 (VGPR-only)
// → vgpr_spill=24. With it: agpr=128, spill=0, +4–7% perf across 5 shapes.
__device__ __forceinline__ void crr_mma_agpr_inplace(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b)
{
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    using D_T = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
    #pragma unroll
    for (int n = 0; n < D_T::height; n++) {
        #pragma unroll
        for (int m = 0; m < D_T::width; m++) {
            #pragma unroll
            for (int k = 0; k < A_row_reg::width; k++) {
                mfma1616128_agpr_inplace(
                    acc.tiles[n][m].data,
                    a_row.tiles[n][k].data,
                    b.tiles[k][m].data);
            }
        }
    }
}

__device__ __forceinline__ float load_fp8_scalar(const _gl_fp8& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(fp8e4m3);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(fp8e4m3);
    const uint8_t bits = llvm_amdgcn_raw_buffer_load_b8(srsrc, voffset, 0, 0);
    return base_types::convertor<float, fp8e4m3>::convert(std::bit_cast<fp8e4m3>(bits));
}

__device__ __forceinline__ float load_bf16_scalar(const _gl_bf16& src, int row, int col) {
    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * src.cols() + col) * sizeof(bf16);
    const uint16_t bits = llvm_amdgcn_raw_buffer_load_b16(srsrc, voffset, 0, 0);
    return base_types::convertor<float, bf16>::convert(std::bit_cast<bf16>(bits));
}

// Packed 8 × fp8e4m3 = 8 bytes for vectorised dense tail-kernel K-loop:
// the HIP compiler emits a single ``global_load_dwordx2`` through this
// type when the source pointer is 8-byte aligned (8× fewer VMEM ops vs
// scalar fp8 loads). Extraction via ``convertor<float4, fp8e4m3_4>``
// gives 8 fp32 accumulator inputs per 8-byte load.
struct alignas(8) fp8e4m3_8 {
    fp8e4m3_4 lo, hi;
};

__device__ __forceinline__ void store_bf16_scalar(const _gl_bf16& dst, int row, int col, float value) {
    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(bf16);
    const std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst.raw_ptr);
    const std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    const buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    const i32x4 srsrc = std::bit_cast<i32x4>(br);
    const uint32_t voffset = (row * dst.cols() + col) * sizeof(bf16);
    const bf16 v = base_types::convertor<bf16, float>::convert(value);
    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v), srsrc, voffset, 0, 0);
}


template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL>
__device__ __forceinline__ void prefill_transpose_swizzled_offsets(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = src.template stride<2>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int lane_byte_offset =
            (laneid * bytes_per_thread) +
            (warpid * bytes_per_warp) +
            (i * num_warps * bytes_per_warp);
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset =
            lane_byte_offset % ST::underlying_subtile_bytes;

        const int row =
            subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        const int col =
            (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

        const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});
        const int shared_row =
            swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes;
        const int shared_col =
            (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

        const int transposed_global_row =
            shared_col + subtile_col * ST::underlying_subtile_cols;
        const int transposed_global_col =
            shared_row + subtile_row * ST::underlying_subtile_rows;

        swizzled_offsets[i] =
            (transposed_global_row * row_stride + transposed_global_col) * sizeof(T);
    }

    if constexpr (
        memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
        ST::rows * ST::cols * sizeof(T)
    ) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset =
                (laneid * bytes_per_thread) +
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset =
                lane_byte_offset % ST::underlying_subtile_bytes;

            const int row =
                subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col =
                (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});
            const int shared_row =
                swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes;
            const int shared_col =
                (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const int transposed_global_row =
                shared_col + subtile_col * ST::underlying_subtile_cols;
            const int transposed_global_col =
                shared_row + subtile_row * ST::underlying_subtile_rows;

            swizzled_offsets[memcpy_per_tile] =
                (transposed_global_row * row_stride + transposed_global_col) * sizeof(T);
        }
    }
}

template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void load_transpose(
    ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = src.template stride<2>();

    coord<> unit_coord(
        idx.template dim<0>(),
        idx.template dim<1>(),
        idx.template dim<3>() * ST::cols,
        idx.template dim<2>() * ST::rows
    );
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::cols * sizeof(T));

    const uintptr_t lds_tile_base = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int warp_linear_offset =
            (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        uintptr_t lds_addr =
            lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc,
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0,
            0,
            static_cast<int>(coherency::cache_all)
        );
    }

    if constexpr (
        memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
        ST::rows * ST::cols * sizeof(T)
    ) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST::underlying_subtile_bytes;
            uintptr_t lds_addr =
                lds_tile_base + warp_linear_offset + lds_subtile_id * ST::subtile_padding;
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0,
                0,
                static_cast<int>(coherency::cache_all)
            );
        }
    }
}

// P19 Dev A — m0-broadcast hoist for the 8-wave RCR DTL loads.
//
// Drop-in replacement for the kittens 4-arg G::load(dst, src, idx, swizzled_offsets)
// used inside the 8-wave RCR main loop. Behaviour:
//   * Pre-computes the per-iter LDS byte offset as an SGPR (readfirstlane
//     forces wave-uniform residency in SGPR class), bypassing LLVM's
//     tendency to recompute the address through a VGPR + v_readfirstlane
//     immediately before each DTL store.
//   * Issues `s_mov m0, <sgpr>` + `buffer_load_dwordx4 ... offen lds` via
//     inline asm. The LLVM intrinsic
//     `__builtin_amdgcn_raw_buffer_load_lds` lets the scheduler CSE the
//     m0 plumbing back into vector ops; inline asm forecloses that.
//   * Vector destination operand is a phantom — `buffer_load_dwordx4 ... lds`
//     does NOT write a VGPR, the operand merely satisfies LLVM's asm-binding
//     contract and gets dead-code-eliminated downstream.
//   * Preserves the leftover-warps tail handling from the kittens helper so
//     odd memcpy_per_tile counts still drain.
//
// Mirrors `kittens::load<2, false, ST, GL, COORD, N_THREADS>` from
// include/ops/warp/memory/tile/global_to_shared.cuh:187-249 (the 4-arg
// variant taking pre-computed swizzled_offsets).
template<int N_THREADS,
         ducks::st::all ST,
         ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __forceinline__ void rcr_8w_load_hoist(
    ST& dst, const GL& src, const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile  =
        ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(
        ST::rows * ST::cols * sizeof(T) >= bytes_per_warp,
        "shared tile must be at least 1024 bytes"
    );

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T* tensor_base = (T*)src.raw_ptr;
    T* global_ptr  = (T*)&src[unit_coord];
    // Full-tensor SRD: bound the ENTIRE source (A or B) tensor so that
    // OOB column-tile loads — induced by ``bpc = ceil_div(n, BLK)`` for
    // partial last-N tiles — clamp to 0 instead of touching unmapped
    // pages. The OG ``make_srsrc(global_ptr, ST::rows * row_stride *
    // sizeof(T))`` was tile-local: ``global_ptr`` past tensor end took
    // the SRD bound into unmapped memory, faulting on the swizzled
    // raw_buffer_load_lds path (verified on N=384 K=2048 RCR/RRR/CRR
    // cumulative test, MI350X). The full-tensor SRD makes every load
    // safe regardless of where ``unit_coord`` points; OOB bytes are
    // returned as 0, the masked C store drops them.
    const uint32_t total_bytes = static_cast<uint32_t>(
        size_t(src.batch()) * size_t(src.depth()) *
        size_t(src.rows())  * size_t(src.cols())  * sizeof(T));
    i32x4 srsrc = make_srsrc(tensor_base, total_bytes);
    // Tile byte offset from tensor base, hoisted into an SGPR.
    // Used as SOFFSET of the buffer_load_dwordx4 to recover the per-tile
    // base. Wave-uniform because ``unit_coord`` only depends on (br, bc,
    // sub-tile, k), all of which are uniform.
    const uint32_t tile_byte_offset = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(global_ptr) -
                              reinterpret_cast<uintptr_t>(tensor_base)));

    const uintptr_t lds_tile_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]);

    // A1: hoist scalar per-pass LDS-byte ramp into SGPRs in the prologue.
    // Each lds_addr below is wave-uniform (warpid + i are uniform), so we
    // make that explicit via readfirstlane and keep the value in SGPR.
    uint32_t lds_addrs[memcpy_per_tile + 1];  // +1 leftover slot (may be unused)
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const int warp_linear_offset =
            (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int lds_subtile_id = warp_linear_offset / ST::underlying_subtile_bytes;
        const uint32_t off32 = static_cast<uint32_t>(
            lds_tile_base + warp_linear_offset +
            lds_subtile_id * ST::subtile_padding);
        lds_addrs[i] = __builtin_amdgcn_readfirstlane(off32);
    }

    // A2: full inline-asm DTL — set m0 from the SGPR-hoisted per-pass offset
    // and issue buffer_load_dwordx4 ... offen lds. Operand binding mirrors
    // P18 Dev A's working 4-wave pattern (see rcr_4wave_dynamic.inc).
    // %0 = s "lds_off" (SGPR), %1 = v "goff" (per-lane VGPR offset),
    // %2 = s "srsrc" (4-SGPR buffer resource), %3 = s "tile_byte_offset"
    // (SGPR SOFFSET) — the per-tile base offset added to V_VOFFSET.
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        const uint32_t lds_off = lds_addrs[i];
        const uint32_t goff    = swizzled_offsets[i];
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
            :
            : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
            : "memory");
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) !=
                  ST::rows * ST::cols * sizeof(T)) {
        constexpr int leftover_bytes =
            ST::rows * ST::cols * sizeof(T) -
            memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;
        if (warpid < leftover_warps) {
            const int warp_linear_offset =
                (warpid * bytes_per_warp) +
                (memcpy_per_tile * num_warps * bytes_per_warp);
            const int lds_subtile_id =
                warp_linear_offset / ST::underlying_subtile_bytes;
            const uint32_t off32 = static_cast<uint32_t>(
                lds_tile_base + warp_linear_offset +
                lds_subtile_id * ST::subtile_padding);
            const uint32_t lds_off = __builtin_amdgcn_readfirstlane(off32);
            const uint32_t goff    = swizzled_offsets[memcpy_per_tile];
            asm volatile(
                "s_mov_b32 m0, %0\n\t"
                "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
                :
                : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
                : "memory");
        }
    }
}

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    hipStream_t stream;
    int m, n, k;
    int bpr, bpc, ki;
    int fast_m, fast_n, fast_k;
    int group_m;
    const float* dscale_a;
    const float* dscale_b;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// Resolve the combined per-tensor scale at kernel epilogue time. When the
// host wrapper passed device-side scale tensors (dscale_{a,b} non-null) we
// load them with a scalar global-memory read; otherwise we fall back to
// the host-known floats baked into `g`. The branch is uniform across the
// wave so the compiler keeps it scalar; the load itself is one b32 from
// global memory and hits cache after the first wave issues it.
__device__ __forceinline__ float resolve_combined_scale(const layout_globals &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

__device__ __forceinline__ int gemm_chiplet_swizzle_bid(int bid, int num_wgs) {
    if (num_wgs >= BLOCK_SWIZZLE_NUM_XCDS &&
        (num_wgs % BLOCK_SWIZZLE_NUM_XCDS) == 0) {
        return
            (bid % BLOCK_SWIZZLE_NUM_XCDS) *
                (num_wgs / BLOCK_SWIZZLE_NUM_XCDS) +
            (bid / BLOCK_SWIZZLE_NUM_XCDS);
    }
    return bid;
}

__device__ __forceinline__ void gemm_compute_block_coords(
    int bid, int bpr, int bpc, int group_m, int &br, int &bc) {
    bid = gemm_chiplet_swizzle_bid(bid, gridDim.x);
    const int num_wgid_in_group = group_m * bpc;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * group_m;
    const int group_size_m =
        (first_pid_m + group_m <= bpr)
            ? group_m
            : (bpr - first_pid_m);
    if (group_size_m <= 0) {
        br = bpr;
        bc = bpc;
        return;
    }
    br = first_pid_m + ((bid % num_wgid_in_group) % group_size_m);
    bc = (bid % num_wgid_in_group) / group_size_m;
}

// =============================================================================
// store_c_tile_n_masked — column-masked C-store for partial-N tiles (FP8 dense).
//
// Mirrors the BF16 helper in `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`.
// Templated on the rt accum type because the FP8 file has two main kernels with
// different RC tile widths (rcr_4w uses RBN=64; the generic 8-wave gemm_kernel
// uses RBN=32).
//
// 3-way fast path:
//   * fully OOB tile-column (n0 >= n_limit): no-op (block won't be writing).
//   * fully in-bounds (n1 <= n_limit): forward to the original `store(...)`.
//     Aligned shapes pay zero overhead — same instruction sequence as before.
//   * partial: lane-level skip on per-column OOB. Each lane checks its column
//     index against `n_limit`. C-tile is bf16 in global memory.
//
// MFMA still writes a result to RC's OOB columns, but those columns are
// dropped here and never reach global memory. SRD bounds for the global B
// (set up by the caller as full-tensor bounds) clamp OOB B-loads to 0, so
// the MFMA accumulator just contains a partial-sum garbage that we discard.
// =============================================================================
template<ducks::gl::all GL, ducks::rt::all RT>
__device__ __forceinline__ void store_c_tile_n_masked(
    const GL& g_c, const RT& src,
    int r_tile, int c_tile, int n_limit) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int n0 = c_tile * RT::cols;
    const int n1 = n0 + RT::cols;
    if (n0 >= n_limit) return;
    if (n1 <= n_limit) {
        store(g_c, src, {0, 0, r_tile, c_tile});
        return;
    }

    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<RT>{0, 0, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    uint32_t buffer_size = g_c.batch() * g_c.depth() * g_c.rows() * g_c.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            if (n0 + col >= n_limit) continue;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset +
                          k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    U v0 = base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].x);
                    U v1 = base_types::convertor<U, T>::convert(
                            src.tiles[i][j].data[idx].y);
                    const uint32_t off0 = ((row + l * 2)     * row_stride + col) * sizeof(U);
                    const uint32_t off1 = ((row + l * 2 + 1) * row_stride + col) * sizeof(U);
                    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v0), srsrc, off0, 0, 0);
                    llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v1), srsrc, off1, 0, 0);
                }
            }
        }
    }
}

template<ducks::gl::all GL, ducks::rt::all RT>
__device__ __forceinline__ void store_c_tile_mn_masked_grouped(
    const GL& g_c, const RT& src,
    int group_idx, int r_tile, int c_tile,
    int m_limit, int n_limit) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(std::is_same_v<U, bf16>, "C is bf16 global");

    const int m0 = r_tile * RT::rows;
    const int m1 = m0 + RT::rows;
    const int n0 = c_tile * RT::cols;
    const int n1 = n0 + RT::cols;

    if (m0 >= m_limit || n0 >= n_limit) return;
    if (m1 <= m_limit && n1 <= n_limit) {
        store(g_c, src, {0, group_idx, r_tile, c_tile});
        return;
    }

    constexpr int axis = 2;
    U* dst_ptr = (U*)&g_c[(coord<RT>{0, group_idx, r_tile, c_tile}
                            .template unit_coord<axis, 3>())];
    const int row_stride = g_c.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    uint32_t buffer_size = g_c.batch() * g_c.depth() * g_c.rows() * g_c.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            if (n0 + col >= n_limit) continue;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset +
                          k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    int row_a = row + l * 2;
                    int row_b = row + l * 2 + 1;
                    if (m0 + row_a < m_limit) {
                        U v0 = base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].x);
                        const uint32_t off0 = (row_a * row_stride + col) * sizeof(U);
                        llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v0), srsrc, off0, 0, 0);
                    }
                    if (m0 + row_b < m_limit) {
                        U v1 = base_types::convertor<U, T>::convert(
                                src.tiles[i][j].data[idx].y);
                        const uint32_t off1 = (row_b * row_stride + col) * sizeof(U);
                        llvm_amdgcn_raw_buffer_store_b16(std::bit_cast<uint16_t>(v1), srsrc, off1, 0, 0);
                    }
                }
            }
        }
    }
}

template<Layout L>
__global__ __launch_bounds__(_NUM_THREADS, MIN_BLOCKS_PER_CU)
void gemm_kernel(const layout_globals g) {
    int bid = blockIdx.x;
    int br, bc;
    gemm_compute_block_coords(bid, g.bpr, g.bpc, g.group_m, br, bc);
    const int ki_dyn = g.ki;
    if (br >= g.bpr || bc >= g.bpc || ki_dyn <= 0) {
        return;
    }
    int wm = warpid() / WARPS_N, wn = warpid() % WARPS_N;

    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    if constexpr (L == Layout::RCR) {
        using ST_rcr = ST_v2;
        __shared__ ST_rcr As[2][2];
        __shared__ ST_rcr Bs[2][2];
        A_row_reg a;
        B_row_reg b0, b1;

        constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
        constexpr int bpm = bpt * _NUM_THREADS;
        constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
        uint32_t soA[mpt], soB[mpt];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, s, k}; };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        int tic = 0, toc = 1;
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], g.a, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        if ((ki_dyn & 1) == 0 && ki_dyn >= RCR_TWO_TILE_MIN_KI) {
            auto main_loop_iter = [&](int tile) {
                load_b(b0, Bs[0][0], wn);
                load_a(a, As[0][0], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1][1], g.a, a_co(br*2+1, tile+1), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                MAYBE_DRAIN_LGKM();
                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[0][1], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0][0], g.b, b_co(bc*2, tile+2), soB);
                __builtin_amdgcn_s_barrier();

                MAYBE_DRAIN_LGKM();
                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cB, a, b1); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[0][1], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[0][0], g.a, a_co(br*2, tile+2), soA);
                __builtin_amdgcn_s_barrier();

                MAYBE_DRAIN_LGKM();
                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cC, a, b0); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b0, Bs[1][0], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[0][1], g.b, b_co(bc*2+1, tile+2), soB);
                TK_WAIT_VMCNT(RCR_TWO_TILE_MID_VMCNT); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cD, a, b1); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][0], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[0][1], g.a, a_co(br*2+1, tile+2), soA);
                TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();

                MAYBE_DRAIN_LGKM();
                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                load_b(b1, Bs[1][1], wn);
                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1][0], g.b, b_co(bc*2, tile+3), soB);
                __builtin_amdgcn_s_barrier();

                MAYBE_DRAIN_LGKM();
                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cB, a, b1); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();

                load_a(a, As[1][1], wm);
                rcr_8w_load_hoist<_NUM_THREADS>(As[1][0], g.a, a_co(br*2, tile+3), soA);
                __builtin_amdgcn_s_barrier();

                MAYBE_DRAIN_LGKM();
                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cC, a, b0); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

                rcr_8w_load_hoist<_NUM_THREADS>(Bs[1][1], g.b, b_co(bc*2+1, tile+3), soB);
                TK_WAIT_VMCNT(RCR_TWO_TILE_MID_VMCNT); __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cD, a, b1); __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_s_barrier();
            };

            TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
            for (int tile = 0; tile < ki_dyn - 2; tile += 2) {
                main_loop_iter(tile);
            }
            TK_WAIT_VMCNT(0);
            __builtin_amdgcn_s_barrier();
        } else
        {
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
        }

        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma_agpr(cC, a, b0);
            rcr_mma_agpr(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::RRR) {
        __shared__ ST_row As[2][2];
        __shared__ ST_v2 Bs[2][2];
        A_row_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB =
            ST_v2::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB =
            ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_v2> { return {0, 0, k, s}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rrr_mma_agpr(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rrr_mma_agpr(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rrr_mma_agpr(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier(); RRR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rrr_mma_agpr(cB, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rrr_mma_agpr(cC, a, b0);
            rrr_mma_agpr(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

    } else if constexpr (L == Layout::CRR) {
        using ST_crr_a = ST_v2a;
        using ST_crr_b = ST_v2;
        __shared__ ST_crr_a As[2][2];
        __shared__ ST_crr_b Bs[2][2];
        A_col_reg a;
        B_col_reg b0, b1;

        constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
        constexpr int bpmA = bptA * _NUM_THREADS;
        constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
        uint32_t soA[mptA];
        G::prefill_swizzled_offsets(As[0][0], g.a, soA);

        constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
        constexpr int bpmB = bptB * _NUM_THREADS;
        constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
        uint32_t soB[mptB];
        G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

        auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            G::load(tile, g.a, a_co(s, k), soA);
        };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            G::load(tile, g.b, b_co(s, k), soB);
        };

        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        int tic = 0, toc = 1;
        global_load_b(Bs[tic][0], bc*2,   0);
        global_load_a(As[tic][0], br*2,   0);
        global_load_b(Bs[tic][1], bc*2+1, 0);
        global_load_a(As[tic][1], br*2+1, 0);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        global_load_b(Bs[toc][0], bc*2,   1);
        global_load_a(As[toc][0], br*2,   1);
        global_load_b(Bs[toc][1], bc*2+1, 1);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            CRR_STEADY_MID_BARRIER();

            load_a(a, As[tic][1], wm);
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            global_load_b(Bs[tic][0], bc*2, k+2);
        }

        {
            load_b(b0, Bs[tic][0], wn);
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_dyn-1);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cA, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            const auto b1_keep = b1;
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cB, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cC, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cD, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        {
            const auto b0_keep = b0;
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cA, a, b0_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            const auto b1_keep = b1;
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cB, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma_agpr_inplace(cC, a, b0_keep);
            crr_mma_agpr_inplace(cD, a, b1_keep);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
        }    }

    const float combined_scale = resolve_combined_scale(g);
    mul(cA, cA, combined_scale);
    mul(cB, cB, combined_scale);
    mul(cC, cC, combined_scale);
    mul(cD, cD, combined_scale);

    // Store Output
    if (wm == 0) __builtin_amdgcn_s_barrier();
    store_c_tile_n_masked(g.c, cA, br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n);
    store_c_tile_n_masked(g.c, cB, br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n);
    store_c_tile_n_masked(g.c, cC, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n);
    store_c_tile_n_masked(g.c, cD, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n);
}

template<Layout L>
__global__ void gemm_tail_kernel(const layout_globals g) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    // ``main_covers_n`` mirrors the dispatch decision (Phase 4): when the
    // generic 8-wave kernel ran with ``bpc = ceil_div(n, BLK)``, the
    // ``store_c_tile_n_masked`` helper already wrote cols [fast_n, n)
    // for rows in [0, fast_m) with the FULL K reduction. Tail must NOT
    // redo those cells. Detected from ``g.bpc`` itself:
    //   * bpc * BLK > fast_n  →  ceil_div path → main covered N.
    //   * bpc * BLK == fast_n →  fast path (4-wave RCR or K misaligned).
    const bool main_covers_n = (g.bpc * BLK > g.fast_n);
    const int n_main_limit = main_covers_n ? g.n : g.fast_n;
    const bool interior_mn = row < g.fast_m && col < n_main_limit;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 && g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    float acc = 0.0f;

    if constexpr (L == Layout::RCR) {
        // Vec8 fast path for RCR: 8× fp8 (64 bits) per load reduces VMEM
        // ops 8× vs scalar fp8. Dense rarely runs this tail (LLM shapes
        // align 4096 / 8192) — mostly a code-symmetry win.
        const fp8e4m3* a_row = &g.a[coord<>(row, 0)];
        const fp8e4m3* b_row = &g.b[coord<>(col, 0)];
        int kk = k0;
        if ((g.k % 8 == 0) && ((k0 & 7) == 0)) {
            const fp8e4m3_8* a_v8 = reinterpret_cast<const fp8e4m3_8*>(a_row);
            const fp8e4m3_8* b_v8 = reinterpret_cast<const fp8e4m3_8*>(b_row);
            const int j_start = k0 >> 3;
            const int j_end   = g.k >> 3;
            #pragma unroll 4
            for (int j = j_start; j < j_end; ++j) {
                fp8e4m3_8 a8 = a_v8[j];
                fp8e4m3_8 b8 = b_v8[j];
                float4 a_lo = base_types::convertor<float4, fp8e4m3_4>::convert(a8.lo);
                float4 a_hi = base_types::convertor<float4, fp8e4m3_4>::convert(a8.hi);
                float4 b_lo = base_types::convertor<float4, fp8e4m3_4>::convert(b8.lo);
                float4 b_hi = base_types::convertor<float4, fp8e4m3_4>::convert(b8.hi);
                acc += a_lo.x * b_lo.x + a_lo.y * b_lo.y
                     + a_lo.z * b_lo.z + a_lo.w * b_lo.w
                     + a_hi.x * b_hi.x + a_hi.y * b_hi.y
                     + a_hi.z * b_hi.z + a_hi.w * b_hi.w;
            }
            kk = j_end << 3;
        }
        for (; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, col, kk);
        }
    } else if constexpr (L == Layout::RRR) {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, row, kk) * load_fp8_scalar(g.b, kk, col);
        }
    } else {
        for (int kk = k0; kk < g.k; ++kk) {
            acc += load_fp8_scalar(g.a, kk, row) * load_fp8_scalar(g.b, kk, col);
        }
    }

    const float scaled = acc * resolve_combined_scale(g);
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col, load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

// Single dynamic-K instantiation per layout: compile-time KI specialization
// caused VGPR spills (the unrolled two-tile loop body exceeded the register
// budget when KI was constexpr); the dynamic path holds at 0 spills.
template __global__ void gemm_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_kernel<Layout::CRR>(const layout_globals);

template __global__ void gemm_tail_kernel<Layout::RCR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::RRR>(const layout_globals);
template __global__ void gemm_tail_kernel<Layout::CRR>(const layout_globals);


struct grouped_layout_globals {
    _gl_fp8 a;                   // [M_total, K]
    _gl_fp8 b;                   // [G, N, K] (RCR)
    _gl_bf16 c;                  // [M_total, N]
    float scale_a, scale_b;
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;   // [G+1] int64 prefix-sum on device
    hipStream_t stream;
    int G;                       // number of groups
    int n;                       // N
    int k;                       // K
    int ki;                      // fast_k / K_BLOCK
    int bpc;                     // fast_n / BLOCK_SIZE
    int group_m;                 // tile-scheduling super-block factor
    int num_xcds;                // chiplet-swizzle XCD count (0 → default 8)
    int M_total;                 // sum of group sizes (= a.shape[0])
    // [grouped] Native non-aligned support (mirror of BF16 grouped Phase 3
    // and FP8 dense fast/tail). Main kernel only sweeps the largest aligned
    // Aligned interior dims (main kernel coverage):
    //   fast_n = (n / BLOCK_SIZE) * BLOCK_SIZE
    //   fast_k = (k / K_BLOCK)   * K_BLOCK
    // N-tail (n > fast_n) is covered by ceil_div(n, BLOCK_SIZE) bpc tiles
    // with N_MASKED_STORE masking the partial last column tile. K-tail
    // (K_rem == 64) is fused into the main kernel via FUSED_KTAIL. Other
    // K_rem values are not supported (no production shape exercises it).
    // Per-group M-tail (M_g % BLOCK_SIZE != 0) is handled via the per-
    // group shifted gl view + m_limit masked store.
    int fast_n, fast_k;
    int m_per_group;
    int num_slots;
    int chunk_size;              // Round-14 (gpt_oss FP8 kernel-only ceiling,
    int fuse_ktail_off;
    int sk_split_n;
    int* sk_partial_buf;
    int bn_block;                 // 0 (default) → 256 ; 128 → use BN=128 variant
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<typename GL>
__device__ __forceinline__ float resolve_combined_scale_grp(const GL &g) {
    const float sa_dev = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb_dev = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa_dev * sb_dev;
}

// =============================================================================
// Shared persistent-dispatch helpers for the 5 grouped FP8 kernels.
//
// init_group_cumsum_smem  : populates s_offs / s_cum_tiles / s_total_tiles
//                           with ceil_div bpr_g per group (M-tile count per
//                           group rounded UP to cover partial last M-tile).
//                           Caller passes M_BLOCK_DIV: BLOCK_SIZE for the
//                           bn=0 / bn128 kernels (BLK_M = 256), HB for the
//                           b128 kernel (BLK_M = 128).
//
// dispatch_tile_in_group  : 6-level binary search to map persistent tile
//                           index gt → (group_idx, local_tile, m_start_g,
//                           M_g, bpr_g). Returns the swizzled (br, bc).
//                           Returns false when the tile lands in a group-
//                           size-degenerate corner (continue-skip).
//
// make_per_group_gl_view  : copies g.a / g.c, patches raw_ptr + rows_internal
//                           so the SRD bound is group-local (M_g rows from
//                           m_start_g). Lets the caller use m_subtile_A=0 +
//                           m_limit=M_g for the masked store.
// =============================================================================

template<int MAX_G_PLUS_1>
__device__ __forceinline__ void init_group_cumsum_smem(
    const grouped_layout_globals& g,
    int* __restrict__ s_offs,
    int* __restrict__ s_cum_tiles,
    int& s_total_tiles,
    int  num_pid_n,
    int  M_BLOCK_DIV) {
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            t += kittens::ceil_div(next - prev, M_BLOCK_DIV) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles = t;
    }
    __syncthreads();
}

template<int MAX_G_PLUS_1>
__device__ __forceinline__ bool dispatch_tile_in_group(
    int  gt,
    const int* __restrict__ s_cum_tiles,
    const int* __restrict__ s_offs,
    int  num_pid_n,
    int  group_m,
    int  M_BLOCK_DIV,
    int& group_idx,
    int& m_start_g,
    int& M_g,
    int& bpr_g,
    int& br,
    int& bc) {
    int lo = 0;
    int hi = MAX_G_PLUS_1 - 1;
    #pragma unroll
    for (int level = 0; level < 6; ++level) {
        const int mid = (lo + hi + 1) >> 1;
        if (gt >= s_cum_tiles[mid]) lo = mid;
        else hi = mid - 1;
    }
    group_idx = lo;
    const int tile_start = s_cum_tiles[lo];
    const int local_tile = gt - tile_start;
    m_start_g = s_offs[group_idx];
    M_g = s_offs[group_idx + 1] - m_start_g;
    bpr_g = kittens::ceil_div(M_g, M_BLOCK_DIV);

    if (num_pid_n > bpr_g) {
        const int WGN = group_m;
        const int num_wgid_in_group = bpr_g * WGN;
        int group_id = local_tile / num_wgid_in_group;
        int first_pid_n = group_id * WGN;
        int group_size_n = min(num_pid_n - first_pid_n, WGN);
        if (group_size_n <= 0) return false;
        bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
        br = (local_tile % num_wgid_in_group) / group_size_n;
    } else {
        const int WGM = group_m;
        const int num_wgid_in_group = WGM * num_pid_n;
        int group_id = local_tile / num_wgid_in_group;
        int first_pid_m = group_id * WGM;
        int group_size_m = min(bpr_g - first_pid_m, WGM);
        if (group_size_m <= 0) return false;
        br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
        bc = (local_tile % num_wgid_in_group) / group_size_m;
    }
    return (br < bpr_g) && (bc < num_pid_n);
}

// Patches a single gl<> view in place so raw_ptr lands at row m_start
// and rows_internal = M_rows — i.e. the SRD bound becomes group-local.
// Caller copies first ( ``auto a_gl = g.a;`` ) because gl<> has no
// default constructor.
template<typename GL>
__device__ __forceinline__ void patch_gl_to_row_slice(
    GL& gl, int m_start, int M_rows) {
    using ptr_t = decltype(gl.raw_ptr);
    const auto* byte_base = reinterpret_cast<const uint8_t*>(gl.raw_ptr);
    const int row_stride_bytes = static_cast<int>(gl.template stride<2>()) * sizeof(*gl.raw_ptr);
    gl.raw_ptr = reinterpret_cast<ptr_t>(
        const_cast<uint8_t*>(byte_base + m_start * row_stride_bytes));
    gl.rows_internal = M_rows;
}

template<typename GLA, typename GLC>
__device__ __forceinline__ void patch_per_group_gl_view(
    GLA& a_gl, GLC& c_gl,
    int m_start_g, int M_g) {
    patch_gl_to_row_slice(a_gl, m_start_g, M_g);
    patch_gl_to_row_slice(c_gl, m_start_g, M_g);
}

template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rcr_kernel_body(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;
    // [kyle-L1 single-buf DEBUG] As/Bs decl LEFT AS [2][2] for now; loop
    // body single-buffered (always reads [0]) to isolate loop-structure
    // correctness from LDS-decl side effects. Will collapse to [1][2]
    // once loop is verified.
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/BLOCK_SIZE);
    const int total_tiles = s_total_tiles;

    // Prefill swizzled offsets ONCE (shared across all tiles & all groups —
    // depends only on the GL strides which are constant within the launch).
    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, m_start_g, M_g, bpr_g, br, bc;
        if (!dispatch_tile_in_group<MAX_G_PLUS_1>(
                gt, s_cum_tiles, s_offs, num_pid_n, g.group_m,
                /*M_BLOCK_DIV=*/BLOCK_SIZE,
                group_idx, m_start_g, M_g, bpr_g, br, bc)) continue;

        // Per-group shifted GL views (m_subtile_A=0, m_limit=M_g).
        // NOTE: m_start_g must be RBM-aligned (= 64) for the row_stride
        // pointer arithmetic to stay element-aligned; unbalanced shapes
        // with arbitrary byte-level m_start_g are a follow-up.
        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        auto b_tile = [&](int stage, int which) -> ST_rcr& {
            return Bs[stage][which];
        };

        // Reset accumulators per tile.
        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: load tile-0 + tile-1 (mirrors gemm_kernel<RCR> 1040-1054).
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], a_gl_g, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1: second-to-last K-tile (mirrors dense lines 1160-1187).
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile (mirrors dense lines 1189-1210).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0);
            rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr (FUSED_KTAIL) {
            // K-tail (K_rem == 64) fused into the main kernel. Original spill
            // budget kept ``a_base_ptr`` pointing at g.a (absolute base) so
            // ``g.a`` stays in const memory and doesn't add to the K-loop
            // register pressure. To support arbitrary group_idx > 0 we add
            // the m_start_g offset INSIDE the SRD voffset computation rather
            // than shifting a_base_ptr; this keeps the K-loop hot path
            // unchanged (a_gl_g is NOT referenced inside the FUSED_KTAIL
            // block) and only adds an SGPR-resident byte offset.
            if (g.fast_k < g.k) {
                const int laneid = kittens::laneid();
                const int row_lane = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                constexpr int KREM = 64;
                static_assert(KREM == 64,
                    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;

                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) *
                    static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);

                const uint32_t K_tail_base_bytes =
                    static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                auto load_a_kt = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    // m_start_g shifted into the byte offset; m_subtile_A=0
                    // with the per-group shifted gl view above, so the row
                    // index is m_start_g + slab*HB + warp_offset + lane.
                    const int M_warp_base =
                        m_start_g + (br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };

                // N_warp_base derivation:
                //   b_co(s, k) → coord {0, group_idx, s, k}
                //   unit_coord: N-row in tile = s * ST_rcr::rows = s * HB.
                //   warp wn picks rows wn*RBN..wn*RBN+RBN-1 within the 128-row tile.
                //   For h_b ∈ [0, B_row_reg::height = 2):
                //     B_row_idx_in_group = N_warp_base + h_b*16 + row_lane.
                //   Global byte = group_idx * N * K + B_row_idx_in_group * K + ...
                auto load_b_kt = [&](B_row_reg& B_tile, int n_strip) __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };

                load_b_kt(b0, 0);
                load_b_kt(b1, 1);
                load_a_kt(a,  0);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cA, a, b0);
                rcr_mma(cB, a, b1);
                load_a_kt(a,  1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cC, a, b0);
                rcr_mma(cD, a, b1);
            }
        }

        const float combined_scale = resolve_combined_scale_grp(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        // Masked store: m_limit drops rows past the partial-last-M-tile
        // (M_g % BLOCK_SIZE != 0 case introduced by the ceil_div bpr_g
        // above); n_limit drops cols past the partial-last-N-tile (covered
        // by N_MASKED_STORE on aligned shapes too). group_idx=0 because
        // g.c is [1, 1, M_total, N] flat — absolute m_limit + r0/r1
        // already encode the per-group position via m_subtile_C.
        mul(cA, cA, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        mul(cB, cB, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cB, /*group_idx=*/0, r0, c1, m_limit, g.n);
        mul(cC, cC, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, c0, m_limit, g.n);
        mul(cD, cD, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cD, /*group_idx=*/0, r1, c1, m_limit, g.n);

        MAYBE_DRAIN_LGKM();
        __builtin_amdgcn_s_barrier();
    }
}


// =============================================================================
// BLOCK_N=128 RCR variant.
// Goal: doubled N-tile parallelism for small-B / odd-N shapes where
// BLOCK_N=256 leaves CUs idle (e.g. Down_B4_M2048 = 1.5 tile/CU at BN=256).
// Per-tile MFMA work is HALVED (single n-strip instead of 2). Per-tile
// fixed overhead (binary search, scale mul, store) is unchanged → 2x more
// total overhead; this only wins if better load balance dominates.
//
// Caller contract: host must set g.bpc = ceil_div(g.n, 128).
// All M-side dims, K-block, accumulator types unchanged from BN=256 kernel.
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rcr_kernel_bn128_body(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];          // n-strip 1 unused; matches BN=256 LDS layout
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cC;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;          // host: ceil_div(g.n, 128)
    const int ki_dyn   = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/BLOCK_SIZE);
    const int total_tiles = s_total_tiles;

    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, m_start_g, M_g, bpr_g, br, bc;
        if (!dispatch_tile_in_group<MAX_G_PLUS_1>(
                gt, s_cum_tiles, s_offs, num_pid_n, g.group_m,
                /*M_BLOCK_DIV=*/BLOCK_SIZE,
                group_idx, m_start_g, M_g, bpr_g, br, bc)) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        // BN=128: ST_rcr cols = HB = 128, so bc directly addresses one ST tile
        // (no *2 factor as in BN=256 which packs 2 strips per BLOCK_N).
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        zero(cA); zero(cC);

        int tic = 0, toc = 1;
        // Prologue.
        rcr_8w_load_hoist<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], a_gl_g, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(Bs[toc][0], g.b, b_co(bc, 1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2,   1), soA);

        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        // Main loop (2 MFMAs per iter: cA, cC). Prefetch Bs[tic][0] @ k+2
        // BEFORE cC mma — this matches original BN=256 ordering where
        // b_tile(tic, 0) is prefetched between cA and cB mma. The original
        // pattern is: load b0 → mma cA → prefetch Bs[tic][0]@k+2 → mma cB →
        // ... → prefetch Bs[tic][1]@k+2 → mma cD. We do (load b0 → mma cA →
        // prefetch Bs[tic][0]@k+2 → mma cC).
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Prefetch Bs[tic][0] @ k+2 NOW (was at end of iter — moved earlier
            // to match original BN=256 ordering pattern).
            rcr_8w_load_hoist<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc, k+2), soB);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
        }

        // Epilog 1: K-iter ki_dyn-2.
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile (k = ki_dyn-1).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma(cC, a, b0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr (FUSED_KTAIL) {
            A_row_reg a_kt1;
            if (g.fast_k < g.k) {
                const int laneid = kittens::laneid();
                const int row_lane = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                constexpr int KREM = 64;
                static_assert(KREM == 64,
                    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;

                // Original absolute A SRD; m_start_g shifted into the byte
                // offset (mirrors grouped_rcr_kernel FUSED_KTAIL) so the K-
                // loop hot path doesn't have to keep a_gl_g alive.
                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) *
                    static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);

                const uint32_t K_tail_base_bytes =
                    static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                auto load_a_kt = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        m_start_g + (br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };

                // Single n-strip (n_strip=0 only). N_warp_base in BN=128:
                // bc * HB (= bc * 128) + wn * RBN.
                auto load_b_kt = [&](B_row_reg& B_tile) __attribute__((always_inline)) {
                    const int N_warp_base = bc * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };

                load_b_kt(b0);
                load_a_kt(a, 0);
                load_a_kt(a_kt1, 1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cA, a, b0);
                rcr_mma(cC, a_kt1, b0);
            }
        }

        const float combined_scale = resolve_combined_scale_grp(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        // BN=128: c0 in RBN units = bc*WARPS_N+wn (was bc*WARPS_N*2+wn for BN=256).
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N+wn);
        // Masked store via per-group shifted view + m_limit/n_limit; mirrors
        // main RCR kernel. N_MASKED_STORE template kept for ABI parity but
        // m_limit/n_limit already cover both partial-M and partial-N cells.
        mul(cA, cA, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        mul(cC, cC, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, c0, m_limit, g.n);

        MAYBE_DRAIN_LGKM();
        __builtin_amdgcn_s_barrier();
    }
}


// =============================================================================
// BLOCK_M=128 BLOCK_N=128 BLOCK_K=128 RCR variant (b128 = "BLK=128 both M, N").
// Single tile per (br, bc) covers 128M × 128N — no M-slab, no N-strip
// subdivision. 1 cA accumulator per warp, 1 mma per K-iter.
//
// 8 warps (WARPS_M=2 × WARPS_N=4): wm covers 64-row chunks (RBM=64),
// wn covers 32-col chunks (RBN=32). Total per tile: 2*64=128M × 4*32=128N.
//
// Caller contract: host sets g.bpc = ceil_div(g.n, 128) AND uses tile-count
// = (M_g/HB) × bpc per group (i.e., bpr_g = M_g/HB instead of M_g/BLOCK_SIZE).
//
// Prefetch ordering follows lesson learned from bn128 fix: prefetch is placed
// INSIDE the load_b/load_a + mma sequence (after load_b/load_a but BEFORE
// the wait+mma), to avoid end-of-iter overwrite hazard.
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rcr_kernel_b128_body(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;             // 128 rows (HB) × 128 cols (BK)
    __shared__ ST_rcr As[2];          // single M-slab per pipe stage
    __shared__ ST_rcr Bs[2];          // single N-strip per pipe stage
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    A_row_reg a;
    B_row_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;          // host: ceil_div(g.n, 128)
    const int ki_dyn   = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/HB);
    const int total_tiles = s_total_tiles;

    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, m_start_g, M_g, bpr_g, br, bc;
        if (!dispatch_tile_in_group<MAX_G_PLUS_1>(
                gt, s_cum_tiles, s_offs, num_pid_n, g.group_m,
                /*M_BLOCK_DIV=*/HB,
                group_idx, m_start_g, M_g, bpr_g, br, bc)) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, 0, m_subtile_A + s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> {
            return {0, group_idx, s, k};
        };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };

        zero(cA);

        int tic = 0, toc = 1;
        // Prologue: load tile-0 + tile-1.
        rcr_8w_load_hoist<_NUM_THREADS>(Bs[tic], g.b, b_co(bc, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic], a_gl_g, a_co(br, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(Bs[toc], g.b, b_co(bc, 1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc], a_gl_g, a_co(br, 1), soA);

        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        // Main loop: parallel of bn128 fixed main loop, with single mma.
        // Prefetch placement copied from bn128 (which works).
        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic], wn);
            load_a(a, As[tic], wm);
            // Prefetch As[tic] @ k+2 here (bn128's "As[toc][1] @ k+1" slot)
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic], a_gl_g, a_co(br, k+2), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Prefetch Bs[tic] @ k+2 here (bn128's "between-mmas" slot)
            rcr_8w_load_hoist<_NUM_THREADS>(Bs[tic], g.b, b_co(bc, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
        }

        // Epilog 1: K-iter ki_dyn-2.
        {
            load_b(b0, Bs[tic], wn);
            load_a(a, As[tic], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            TK_WAIT_VMCNT(0); __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2: last K-tile (k = ki_dyn-1).
        {
            load_b(b0, Bs[tic], wn);
            load_a(a, As[tic], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr (FUSED_KTAIL) {
            if (g.fast_k < g.k) {
                const int laneid = kittens::laneid();
                const int row_lane = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                constexpr int KREM = 64;
                static_assert(KREM == 64,
                    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;

                // Original absolute A SRD; m_start_g shifted into the byte
                // offset (mirrors grouped_rcr_kernel FUSED_KTAIL) so the K-
                // loop hot path doesn't have to keep a_gl_g alive.
                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) *
                    static_cast<uint32_t>(a_row_stride_bytes);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);

                const uint32_t K_tail_base_bytes =
                    static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) *
                    static_cast<uint32_t>(b_row_stride_bytes);

                // Single M-tile (br) and single N-tile (bc) for b128.
                // M_warp_base shifts by m_start_g (m_subtile_A=0 with view).
                // N_warp_base = bc * HB + wn * RBN
                auto load_a_kt = [&]() __attribute__((always_inline)) {
                    const int M_warp_base = m_start_g + br * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[4]) = v1;
                    }
                };

                auto load_b_kt = [&]() __attribute__((always_inline)) {
                    const int N_warp_base = bc * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes +
                            K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&b0.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&b0.tiles[h_b][0].data[4]) = v1;
                    }
                };

                load_b_kt();
                load_a_kt();
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma(cA, a, b0);
            }
        }

        const float combined_scale = resolve_combined_scale_grp(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        // r0 = m_subtile_C + br*WARPS_M + wm (no slab — single tile per br)
        // c0 = bc*WARPS_N + wn (single strip per bc)
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M + wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N + wn);
        // Masked store via per-group shifted view + m_limit/n_limit. Mirrors
        // main RCR. N_MASKED_STORE template retained for ABI parity.
        mul(cA, cA, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);

        MAYBE_DRAIN_LGKM();
        __builtin_amdgcn_s_barrier();
    }
}


// Force-instantiate. Compare resource report against the R57 step-2A
// baseline (V256 / A256 / Spill 0 / Scratch 0 — placeholder G::load).

#ifndef FP8_RRR_FUSE_PROBE
#define FP8_RRR_FUSE_PROBE 0
#endif
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rrr_kernel_body(const grouped_layout_globals g) {
    __shared__ ST_row As[2][2];
    __shared__ ST_v2  Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    // Uniform-group fast path (mirror reference at
    // mxfp8/Primus-Turbo/csrc/kernels/grouped_gemm/turbo/turbo_grouped_gemm_mxfp8_kernel.h).
    // If all groups have the same M (gpt_oss / DSV3 / most MoE training),
    // skip the 6-level binary search and use a single division per persistent
    // iteration to map gt → (group_idx, local_tile).
    __shared__ int s_uniform_M;     // > 0 iff all groups have same M; else -1
    __shared__ int s_tiles_per_g;   // (uniform_M / BLOCK_SIZE) * num_pid_n
    // K-tail cross-lane shuffle scratch (16 KB) — only allocated for
    // FUSED_KTAIL=true. Moved inside the if-constexpr branch below so
    // the FUSED_KTAIL=false instantiation can lift launch_bounds 1 → 2
    // (occupancy parity with dense gemm_kernel<RRR>).

    A_row_reg a;
    // H6 (2026-05-21, re-applied): b1 eliminated; main loop uses single b0
    // cycled strip0→strip1→strip0 with mma order cA→cB→cD→cC. Previously
    // measured spill 58→37 on NMASK=1 (worst-shape dispatch route per H3).
    // First measurement misread baseline; re-applying with H7 to compose.
    B_col_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR — read host-side
    // ``g.num_xcds`` knob, fall back to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    // 2026-05-20: use gridDim.x not NUM_CUS so we can launch with grid =
    // total_tiles (TK_*_GRID_MODE=tile experiment).
    const int slots_eff = gridDim.x;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, 64);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = g.ki;

    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        const int M0 = s_offs[1] - prev;
        bool uniform = (g.G > 0);
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            const int M_g_i = next - prev;
            if (M_g_i != M0) uniform = false;
            // ceil_div: partial last M-tile (M_g % BLOCK_SIZE != 0) gets its
            // own (br, bc) tile; per-group shifted gl view + m_limit masked
            // store below handle the OOB rows. Floor div silently skipped
            // small groups (M_g < BLOCK_SIZE) → garbage / zero output.
            t += kittens::ceil_div(M_g_i, BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles  = t;
        s_uniform_M    = uniform ? M0 : -1;
        s_tiles_per_g  = uniform ? kittens::ceil_div(M0, BLOCK_SIZE) * num_pid_n : 0;
    }
    __syncthreads();
    const int total_tiles  = s_total_tiles;
    const int uniform_M    = s_uniform_M;
    const int tiles_per_g  = s_tiles_per_g;

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, local_tile;
        if (uniform_M > 0) {
            // Uniform-group fast path: single division per persistent iter.
            // tiles_per_g is identical across groups so simple integer
            // division gives group_idx; remainder gives local_tile.
            group_idx  = gt / tiles_per_g;
            local_tile = gt - group_idx * tiles_per_g;
        } else {
            // Variable-M fallback: 6-level binary search over s_cum_tiles.
            int lo = 0;
            int hi = MAX_G_PLUS_1 - 1;
            #pragma unroll
            for (int level = 0; level < 6; ++level) {
                const int mid = (lo + hi + 1) >> 1;
                if (gt >= s_cum_tiles[mid]) lo = mid;
                else hi = mid - 1;
            }
            group_idx  = lo;
            local_tile = gt - s_cum_tiles[lo];
        }
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = kittens::ceil_div(M_g, BLOCK_SIZE);

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
        // H3 (2026-05-20): outer/inner ptr split for B-tensor — shift the
        // grouped B SRD base by `group_idx * K * N * sizeof(fp8)` once per
        // tile, then drop `group_idx` from `b_co` so it matches dense's
        // `{0, 0, k, s}`. Theory: lets the compiler reuse dense's exact
        // register/MMA schedule for B-loads, freeing the per-tile slot
        // currently held by `group_idx` inside the b_co lambda capture.
        // Safety: SRD bound (full-tensor depth=G*K*N) extends past tensor
        // end by (G-group_idx-1)*K*N bytes; HW returns 0 for those OOB
        // reads, and the k/bc loops stay within the current group's K*N
        // window so no garbage is ever read.
        auto b_gl_g = g.b;
        {
            using ptr_t = decltype(b_gl_g.raw_ptr);
            const int64_t b_group_stride_bytes =
                static_cast<int64_t>(g.b.template stride<1>()) * sizeof(*g.b.raw_ptr);
            auto* base = reinterpret_cast<uint8_t*>(b_gl_g.raw_ptr);
            b_gl_g.raw_ptr = reinterpret_cast<ptr_t>(
                base + static_cast<int64_t>(group_idx) * b_group_stride_bytes);
        }
        // 2026-05-20: a-load switched from G::load(.., g.a, ..) + m_subtile_A
        // shift to rcr_8w_load_hoist(.., a_gl_g, ..) — same loader RCR uses,
        // and the only one that survives the compiler optimizing away a
        // patched gl<>::raw_ptr (it computes tile_byte_offset =
        // global_ptr - tensor_base explicitly).
        //
        // Combined with the patched view (rows_internal = M_g), the SRD bound
        // becomes per-group byte-level — HW raw_buffer_load clamps OOB rows
        // to 0 so unbalanced M_g works without prior-iter HBM garbage leak.
        // (Pre-2026-05-20 state: SRD bound was full-tensor M_total, OOB rows
        // returned arbitrary VRAM contents that were freed-but-not-zeroed
        // from prior iters → reproducible memory access fault on iter 2 of
        // the HIPKITTEN sweep.)
        //
        // FUSED_KTAIL path still uses g.a + m_subtile_A unit-coord encoding;
        // see below.
        const int m_subtile_A = m_start_g / HB;  // FUSED_KTAIL only
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        // RRR coord conventions (mirror of dense gemm_kernel<RRR>):
        //   a_co(s, k) : A is per-group [M_g, K]  → unit_coord row index 0..bpr_g-1.
        //                Pointer-base shift via patched a_gl_g handles m_start_g.
        //   b_co(s, k) : B is per-group-shifted view (b_gl_g) — drop the
        //                group_idx dim to match dense's coord (H3 hoist).
        auto a_co = [&](int s, int k) -> coord<ST_row> {
            return {0, 0, s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_v2> {
            return {0, 0, k, s};
        };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: tile-0 + tile-1 (mirrors dense gemm_kernel<RRR>
        // lines 1421-1435). a-load uses g.a (unpatched) per fix above.
        G::load(Bs[tic][0], b_gl_g, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], b_gl_g, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], a_gl_g, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], b_gl_g, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], b_gl_g, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // Single-tile main loop (mirror dense lines 1437-1470).
        // 2026-05-15: AGPR variant — only main loop uses _agpr; epilog/ktail
        // stay on builtin so the cA-cD AGPR ↔ VGPR transition happens once
        // at end-of-main rather than every call.
        // H6+H7 (2026-05-21): single-b0 main loop (cA→cB→cD→cC), one
        // sched_barrier(0) at end-of-iter. H6 alone drops spill 58→37 on
        // NMASK=1 (worst-shape route). H7 removes redundant mid-iter
        // sched_barrier(0) drains so scheduler can reorder DS_READ/VMEM
        // around the mma pipeline.
        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            // Phase 1: cA = mma(slab0, strip0)
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Phase 2: cB = mma(slab0, strip1)
            load_b(b0, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cB, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Phase 3: cD = mma(slab1, strip1); prefetch strip1 (last use)
            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], b_gl_g, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cD, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Phase 4: cC = mma(slab1, strip0 reload); prefetch strip0 (last use)
            load_b(b0, Bs[tic][0], wn);
            G::load(Bs[tic][0], b_gl_g, b_co(bc*2, k+2), soB);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            RRR_SCHED_BARRIER();  // end-of-iter fence (CK pattern: 1 per K-iter)
        }

        // Epilog 1 (H6 single-b0): mma order cA→cB→cD→cC, single b0 cycled.
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cB, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cD, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[tic][0], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2 (H6 single-b0): mma order cA→cB→cD→cC, single b0 cycled.
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cB, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cD, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[tic][0], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_agpr_t<true>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // ===== FUSED_KTAIL (RRR FP8) =====
        // K-tail K_REM=64 reduction fused into main kernel epilog. Both A
        // AND B loaded direct-to-register via raw_buffer_load_b8 to avoid
        // the load_col_from_st (ds_read_b64_tr_b8) path that triggers a
        // mfma operand-forwarding quirk for RRR. Per-group SRD bounds
        // protect against cross-group/OOB garbage reads (K_row >= g.k → 0).
        // K-aligned-to-32 chunks: lane chunks 0,1 carry real data (K=0..64),
        // chunks 2,3 zeroed (K=64..128, mfma's K=128 zero-pad).
        if constexpr (FUSED_KTAIL) {
            // Per-warp scratch for cross-lane B shuffle (see step-2..4 below).
            // Layout per warp: 64 K-rows × 16 N-cols × 1 byte = 1024 bytes per
            // (warp, j); 8 warps × 2 j = 16 KB total. Only allocated for the
            // FUSED_KTAIL=true template instantiation.
            __shared__ uint8_t kt_b_scratch[8][2][1024];

            if (g.fast_k < g.k) {
                typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
                B_col_reg b0_kt, b1_kt;

                const int laneid_fk  = kittens::laneid();
                const int row_lane_fk = laneid_fk % 16;
                const int chunk_fk    = laneid_fk / 16;          // 0..3
                const int k_lane_byte_fk = chunk_fk * 32;
                const bool ab_chunk_valid = (chunk_fk < 2);      // K=0..64 chunks have real K-tail data
                constexpr uint32_t SENTINEL_FK = 0xFFFF0000u;

                // ---- Per-group SRDs (so OOB voffset returns 0) ----
                const fp8e4m3* a_base_ptr_fk =
                    (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                fp8e4m3* b_grp_base = const_cast<fp8e4m3*>(
                    &g.b[{0, group_idx, 0, 0}]);
                const uint32_t a_row_stride_bytes_fk = g.a.template stride<2>();
                const uint32_t b_row_stride_bytes_fk = g.b.template stride<2>();
                const uint32_t a_total_bytes_fk =
                    static_cast<uint32_t>(g.M_total) * a_row_stride_bytes_fk;
                const uint32_t b_grp_total_bytes_fk =
                    static_cast<uint32_t>(g.k) * b_row_stride_bytes_fk;
                i32x4 a_srsrc_fk = make_srsrc(
                    (const void*)a_base_ptr_fk, a_total_bytes_fk);
                i32x4 b_srsrc_fk = make_srsrc(
                    (const void*)b_grp_base, b_grp_total_bytes_fk);
                const uint32_t K_tail_byte_fk =
                    static_cast<uint32_t>(g.fast_k);

                // ---- Custom A K-tail load (direct-to-reg, mirror RCR) ----
                auto load_a_kt_fk = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx =
                            M_warp_base + h * 16 + row_lane_fk;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes_fk +
                            K_tail_byte_fk + k_lane_byte_fk);
                        const uint32_t v_lo = ab_chunk_valid ? v_base : SENTINEL_FK;
                        const uint32_t v_hi = ab_chunk_valid ? (v_base + 16) : SENTINEL_FK;
                        __uint128_t va0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_fk, v_lo, 0, 0);
                        __uint128_t va1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_fk, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][0].data[0]) = va0;
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][0].data[4]) = va1;
                    }
                };

                // ---- B K-tail load: cross-lane shuffle via LDS scratch ----
                // OLD: 32 strided byte_b8 loads per lane × 16 lanes = 512 ops/warp/(j,strip)
                // NEW: 1 b128/lane (16 contig N bytes for 1 K row) + LDS write +
                //      32 byte-reads from LDS column-strided.
                // Memory traffic same; instruction count cut ~2-3x; LDS bandwidth
                // is much higher than strided global so net throughput up.
                //
                // gfx950 quirk note: ds_read_b64_tr_b8 in mma chain b128(A)+ds_read(B)+mfma
                // silently zeros mfma. We use NORMAL ds_read (not _tr_b8 transposed)
                // so quirk should not fire.
                const int warp_id_fk = kittens::warpid();
                auto load_b_kt_fk = [&](B_col_reg& B_tile, int n_strip)
                        __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int j = 0; j < B_col_reg::width; ++j) {
                        // STEP 1: ALL 64 lanes load — lane l loads K=K_tail+l
                        // (covers K=K_tail+0..63 = the full real K-tail range).
                        const uint32_t K_pos = K_tail_byte_fk + laneid_fk;
                        const uint32_t N_col_start =
                            static_cast<uint32_t>(N_warp_base + j * 16);
                        const uint32_t voffset =
                            K_pos * b_row_stride_bytes_fk + N_col_start;
                        __uint128_t v = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_fk, voffset, 0, 0);
                        // STEP 2: write to LDS column-major: byte at (K, N) →
                        // LDS[N*64 + K]. Each lane scatters its 16 bytes:
                        // byte i of lane l → LDS[i*64 + l]. This makes per-lane
                        // strided writes (16 ds_write_b8/lane) but enables 8
                        // ds_read_b32/lane on read side (32 contig K bytes).
                        // Net: 24 LDS ops/lane vs 33 in row-major scheme.
                        uint8_t* v_bytes = (uint8_t*)&v;
                        uint8_t* lds_base = &kt_b_scratch[warp_id_fk][j][0];
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            lds_base[i * 64 + laneid_fk] = v_bytes[i];
                        }
                        // STEP 3: drain LDS commits.
                        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");

                        // STEP 4: contiguous 32-byte read per lane.
                        // Chunk 0/1 lane reads its 32 K bytes for N=row_lane_fk,
                        // K offset = chunk_fk*32. LDS layout col-major puts
                        // (K=0..63, N=row_lane_fk) at LDS[row_lane_fk*64..+63].
                        // Lane reads 32 contig bytes = 8 ds_read_b32.
                        intx8_t b_pack = intx8_t{};
                        if (ab_chunk_valid) {
                            const int K_offset = chunk_fk * 32;
                            uint8_t* lds_col_base =
                                &kt_b_scratch[warp_id_fk][j][row_lane_fk * 64 + K_offset];
                            // 32 contig bytes = 2 b128 reads.
                            __uint128_t r0 =
                                *reinterpret_cast<__uint128_t*>(lds_col_base);
                            __uint128_t r1 =
                                *reinterpret_cast<__uint128_t*>(lds_col_base + 16);
                            uint8_t* bp = (uint8_t*)&b_pack;
                            *reinterpret_cast<__uint128_t*>(bp) = r0;
                            *reinterpret_cast<__uint128_t*>(bp + 16) = r1;
                        }
                        *reinterpret_cast<intx8_t*>(
                            &B_tile.tiles[0][j].data[0]) = b_pack;
                    }
                };

                // ---- Runtime knob: LDS-based B load (avoids 32 byte-loads/lane).
                // Quirk-free strategy: pre-zero Bs[tic], coop-load B via LDS
                // with per-group SRD bound = K_REM*N (so K=K_REM..127 voffsets
                // OOB → no-op → preserve zero), keep A as direct-to-reg b128
                // (chunks 2,3 may read next-M-row garbage but multiplied by
                // B=0 in those chunks → harmless). Avoids the b128(A) →
                // ds_read(B) → mfma quirk because B's chunks 2,3 are zero
                // (so wrong contribution = 0).
                // Compile-time gated experiment. Toggle via #define.
                #ifndef HK_FP8_RRR_FUSED_KTAIL_LDS
                #define HK_FP8_RRR_FUSED_KTAIL_LDS 0
                #endif
                constexpr bool ktail_lds_active = (HK_FP8_RRR_FUSED_KTAIL_LDS != 0);

                if constexpr (ktail_lds_active) {
                    // ---- Cooperative pre-zero Bs[tic][0/1] ----
                    constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);
                    const int tid_z = threadIdx.x;
                    __uint128_t* Bs0_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][0].data[0]);
                    __uint128_t* Bs1_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][1].data[0]);
                    #pragma unroll
                    for (int idx = tid_z; idx < ST_V2_B128; idx += _NUM_THREADS) {
                        Bs0_ptr[idx] = 0;
                        Bs1_ptr[idx] = 0;
                    }
                    __syncthreads();

                    // ---- Custom coop load B with per-group SRD ----
                    // Per-group bound = K_REM * N bytes. Voffsets where K_local
                    // >= K_REM (chunks 2,3) auto-no-op → preserves zero.
                    const uint32_t b_row_stride = g.b.template stride<2>();
                    const uint32_t b_per_group_bound =
                        static_cast<uint32_t>(g.k - g.fast_k) * b_row_stride;

                    constexpr int bptB_lds = ST_v2::underlying_subtile_bytes_per_thread;
                    constexpr int bpwB_lds = bptB_lds * kittens::WARP_THREADS;
                    constexpr int mptB_lds = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / (bptB_lds * _NUM_THREADS);
                    constexpr int nwarps_lds = _NUM_THREADS / kittens::WARP_THREADS;
                    const int laneid_lds = kittens::laneid();
                    const int warpid_lds = kittens::warpid() % nwarps_lds;

                    #pragma unroll
                    for (int strip = 0; strip < 2; ++strip) {
                        // global_ptr at K-tail tile origin within this group
                        coord<ST_v2> tile_idx{0, group_idx, ki_dyn, bc*2+strip};
                        coord<> unit = tile_idx.template unit_coord<2, 3>();
                        fp8e4m3* gp = (fp8e4m3*)&g.b[unit];
                        i32x4 b_srsrc_lds = make_srsrc((const void*)gp, b_per_group_bound);

                        ST_v2& Bs_dst = Bs[tic][strip];
                        const uintptr_t lds_tile_base = reinterpret_cast<uintptr_t>(&Bs_dst.data[0]);

                        #pragma unroll
                        for (int i = 0; i < mptB_lds; i++) {
                            const int lane_byte_offset = (laneid_lds * bptB_lds) + (warpid_lds * bpwB_lds) + (i * nwarps_lds * bpwB_lds);
                            const int subtile_id = lane_byte_offset / ST_v2::underlying_subtile_bytes;
                            const int subtile_row = subtile_id / ST_v2::underlying_subtiles_per_row;
                            const int subtile_col = subtile_id % ST_v2::underlying_subtiles_per_row;
                            const int subtile_lane_byte_offset = lane_byte_offset % ST_v2::underlying_subtile_bytes;

                            const int row = subtile_lane_byte_offset / ST_v2::underlying_subtile_row_bytes;
                            const int col = (subtile_lane_byte_offset % ST_v2::underlying_subtile_row_bytes) / sizeof(fp8e4m3);

                            const uint32_t swizzled_shared_byte_offset = Bs_dst.swizzle({row, col});

                            const int swizzled_global_row = (swizzled_shared_byte_offset / ST_v2::underlying_subtile_row_bytes) + subtile_row * ST_v2::underlying_subtile_rows;
                            const int swizzled_global_col = (swizzled_shared_byte_offset % ST_v2::underlying_subtile_row_bytes) / sizeof(fp8e4m3) + subtile_col * ST_v2::underlying_subtile_cols;
                            const uint32_t swizzled_global_byte_offset =
                                (swizzled_global_row * b_row_stride + swizzled_global_col) * sizeof(fp8e4m3);

                            const int warp_linear_offset = (warpid_lds * bpwB_lds) + (i * nwarps_lds * bpwB_lds);
                            const int lds_subtile_id = warp_linear_offset / ST_v2::underlying_subtile_bytes;
                            uintptr_t lds_addr = lds_tile_base + warp_linear_offset + lds_subtile_id * ST_v2::subtile_padding;
                            kittens::as3_uint32_ptr lds_ptr = (kittens::as3_uint32_ptr)(lds_addr);

                            kittens::llvm_amdgcn_raw_buffer_load_lds(
                                b_srsrc_lds, lds_ptr, bptB_lds,
                                swizzled_global_byte_offset, 0, 0,
                                static_cast<int>(kittens::coherency::cache_all));
                        }
                    }

                    // A side: keep direct-to-reg b128 (working, fast).
                    // 2026-05-16 spill fix: serialize slabs to reuse `a`.
                    load_a_kt_fk(a, 0);

                    // Drain global loads (B coop load + A slab 0)
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    __syncthreads();

                    // ds_read B from LDS (col-layout via tr_b8)
                    load_col_from_st(b0_kt, Bs[tic][0], wn * RBN);
                    load_col_from_st(b1_kt, Bs[tic][1], wn * RBN);
                    MAYBE_DRAIN_LGKM();

                    rrr_mma(cA, a, b0_kt);
                    rrr_mma(cB, a, b1_kt);

                    load_a_kt_fk(a, 1);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    rrr_mma(cC, a, b0_kt);
                    rrr_mma(cD, a, b1_kt);
                    __builtin_amdgcn_s_barrier();
                } else {
                    // Production direct-to-reg byte-load path (slow but quirk-safe).
                    // 2026-05-16 spill fix: serialize slabs to reuse `a` (saves
                    // ~32 VGPRs of A_row_reg pressure from a_kt0+a_kt1).
                    load_b_kt_fk(b0_kt, 0);
                    load_b_kt_fk(b1_kt, 1);
                    load_a_kt_fk(a, 0);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    rrr_mma(cA, a, b0_kt);
                    rrr_mma(cB, a, b1_kt);
                    load_a_kt_fk(a, 1);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    rrr_mma(cC, a, b0_kt);
                    rrr_mma(cD, a, b1_kt);
                    __builtin_amdgcn_s_barrier();
                }
            }
        }

#if FP8_RRR_FUSE_PROBE
        if (g.fast_k < g.k) {
            // ---- Cooperative pre-zero of Bs[tic][0/1] ----
            // ST_v2 has swizzle padding (= 17408 bytes); strip to 16-byte
            // alignment so we can b128-store the entire region incl. padding.
            constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);
            const int tid = threadIdx.x;
            __uint128_t* Bs0_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][0].data[0]);
            __uint128_t* Bs1_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][1].data[0]);
            #pragma unroll
            for (int idx = tid; idx < ST_V2_B128; idx += _NUM_THREADS) {
                Bs0_ptr[idx] = 0;
                Bs1_ptr[idx] = 0;
            }
            __syncthreads();

            // ---- B side: cooperative G::load on K-tail iter ----
            // OOB voffsets (k_row >= K_global) no-op on raw_buffer_load_lds →
            // pre-zeroed bytes preserved → effective zero-pad for K=[K_global,
            // fast_k + K_BLOCK). Cross-group contamination is OK for the G=1
            // probe shape (full-tensor SRD == per-group SRD); production hybrid
            // would need per-group SRD construction.
            G::load(Bs[tic][0], g.b, b_co(bc*2,   ki_dyn), soB);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, ki_dyn), soB);

            const int laneid = kittens::laneid();
            const int row_lane = laneid % 16;
            const int k_lane_byte = (laneid / 16) * 32;
            const int K_REM = g.k - g.fast_k;
            const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
            const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;
            constexpr uint32_t SENTINEL = 0xFFFF0000u;
            // Original absolute A SRD; m_start_g shifted into the byte
            // offset (mirrors grouped_rcr_kernel FUSED_KTAIL).
            const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
            const uint32_t a_total_bytes =
                static_cast<uint32_t>(g.M_total) *
                static_cast<uint32_t>(a_row_stride_bytes);
            i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
            const uint32_t K_tail_base_bytes =
                static_cast<uint32_t>(g.fast_k);

            auto load_a_kt = [&](int slab) __attribute__((always_inline)) {
                const int M_warp_base =
                    m_start_g + (br * 2 + slab) * HB + wm * RBM;
                #pragma unroll
                for (int h = 0; h < A_row_reg::height; ++h) {
                    const int A_row_idx = M_warp_base + h * 16 + row_lane;
                    const uint32_t v_base = static_cast<uint32_t>(
                        A_row_idx * a_row_stride_bytes +
                        K_tail_base_bytes + k_lane_byte);
                    const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
                    const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
                    __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_lo, 0, 0);
                    __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_hi, 0, 0);
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[0]) = v0;
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[4]) = v1;
                }
            };

            // Wait B G::load + LDS visibility before B reads.
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

#if !FP8_RRR_FUSE_PROBE_SKIP_A_LOAD
            load_b(b0, Bs[tic][0], wn);
            load_a_kt(0);
            asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)" ::: "memory");
            rrr_mma(cA, a, b0);

            load_b(b1, Bs[tic][1], wn);
            rrr_mma(cB, a, b1);

            load_a_kt(1);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
#endif
            __builtin_amdgcn_s_barrier();
        }
#endif

        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        // Masked grouped store on the per-group shifted c_gl_g: m_limit=M_g
        // drops partial-last-M-tile OOB rows, n_limit=g.n drops partial-N.
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cB, /*group_idx=*/0, r0, c1, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, c0, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cD, /*group_idx=*/0, r1, c1, m_limit, g.n);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

// =============================================================================
// BLOCK_N=128 RRR variant (mirror of grouped_rcr_kernel_bn128_body).
// Output tile: BLK_M=256, BLK_N=128, K_BLOCK=128. Halves the N footprint
// vs the full RRR body — 2 accumulators (cA, cC) instead of 4 (cA-cD),
// 1 B-prefetch reg (b0) instead of 2 (b0, b1). Goal: V drop 256→~230,
// spill 61→0, eliminate 248B/lane scratch.
//
// Caller contract: host sets g.bpc = ceil_div(g.n, 128) (vs 256 for the
// full body); 2x more N-tiles per group, same M-tile count.
//
// FUSED_KTAIL path is omitted in this variant (template-skipped); revisit
// if K%128 != 0 shapes need bn128 routing — for now bn128 is gated to
// K-aligned (fast_k == k) shapes only at the dispatcher.
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rrr_kernel_bn128_body(const grouped_layout_globals g) {
    __shared__ ST_row As[2][2];
    __shared__ ST_v2  Bs[2];             // 2026-05-20: shrunk from [2][2]
                                         // (n-strip 1 dead — saves ~33KB LDS)
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    __shared__ int s_uniform_M;
    __shared__ int s_tiles_per_g;

    A_row_reg a;
    B_col_reg b0;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cC;

    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int slots_eff = gridDim.x;
    // bn128: chunk_size env-tunable to test 32/64/128/256
    const int chunk_size_eff_bn128 = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff_bn128);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;     // host: ceil_div(g.n, 128)
    const int ki_dyn   = g.ki;

    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        const int M0 = s_offs[1] - prev;
        bool uniform = (g.G > 0);
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            const int M_g_i = next - prev;
            if (M_g_i != M0) uniform = false;
            t += kittens::ceil_div(M_g_i, BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles  = t;
        s_uniform_M    = uniform ? M0 : -1;
        s_tiles_per_g  = uniform ? kittens::ceil_div(M0, BLOCK_SIZE) * num_pid_n : 0;
    }
    __syncthreads();
    const int total_tiles  = s_total_tiles;
    const int uniform_M    = s_uniform_M;
    const int tiles_per_g  = s_tiles_per_g;

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, local_tile;
        if (uniform_M > 0) {
            group_idx  = gt / tiles_per_g;
            local_tile = gt - group_idx * tiles_per_g;
        } else {
            int lo = 0;
            int hi = MAX_G_PLUS_1 - 1;
            #pragma unroll
            for (int level = 0; level < 6; ++level) {
                const int mid = (lo + hi + 1) >> 1;
                if (gt >= s_cum_tiles[mid]) lo = mid;
                else hi = mid - 1;
            }
            group_idx  = lo;
            local_tile = gt - s_cum_tiles[lo];
        }
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = kittens::ceil_div(M_g, BLOCK_SIZE);

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        // RRR bn128 coord conventions:
        //   a_co(s, k) : per-group [M_g, K], unit_coord row (patched view shifts base).
        //   b_co(s, k) : B is [1, G, K, N]; bc directly indexes 1 N-strip (no *2).
        auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_v2>  { return {0, group_idx, k, s}; };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        zero(cA); zero(cC);

        int tic = 0, toc = 1;
        // Prologue: tile 0 + tile 1 (1 b strip vs 2 for full body).
        G::load(Bs[tic], g.b, b_co(bc, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], a_gl_g, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc], g.b, b_co(bc, 1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2, 1), soA);

        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        // Main loop (2 mma per iter on cA, cC). Mirror RCR bn128 cadence.
        // 2026-05-20 Phase 2-C: dropped post-mma barriers (mma writes only
        // registers, no LDS — next iter's first LDS write has its own barrier).
        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);

            G::load(Bs[tic], g.b, b_co(bc, k+2), soB);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);

            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
        }

        // Epilog 1.
        {
            load_b(b0, Bs[tic], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc], wn);
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2 (last K-tile).
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rrr_mma(cC, a, b0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // FUSED_KTAIL omitted for bn128: bn128 dispatcher gates K%128==0.

        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cC, cC, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        // BN=128: c0 in RBN units = bc*WARPS_N+wn (was bc*WARPS_N*2+wn for BN=256).
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N+wn);
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, c0, m_limit, g.n);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

// =============================================================================
// Unified persistent grouped FP8 GEMM kernel — single ``__global__`` entry
// point dispatching to per-shape ``_body`` device functions.
//
//   Layout L | BLOCK_N | body
//   ---------+---------+--------------------------------
//   RCR      |    256  | grouped_rcr_kernel_body        (BLK_M = BLK_N = 256)
//   RCR      |    128  | grouped_rcr_kernel_bn128_body  (BLK_M = 256, BLK_N = 128)
//   RCR      |   -128  | grouped_rcr_kernel_b128_body   (BLK_M = BLK_N = 128)
//   RRR      |    256  | grouped_rrr_kernel_body        (BLK_M = BLK_N = 256)
//   RRR      |    128  | grouped_rrr_kernel_bn128_body  (BLK_M = 256, BLK_N = 128)
// =============================================================================
template<Layout L, int BLOCK_N = 256,
         bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_gemm_fp8_kernel(const grouped_layout_globals g) {
    if constexpr (L == Layout::RCR) {
        if constexpr (BLOCK_N == 128) {
            grouped_rcr_kernel_bn128_body<N_MASKED_STORE, FUSED_KTAIL>(g);
        } else if constexpr (BLOCK_N == -128) {
            grouped_rcr_kernel_b128_body<N_MASKED_STORE, FUSED_KTAIL>(g);
        } else {
            static_assert(BLOCK_N == 256,
                          "grouped_gemm_fp8_kernel: RCR BLOCK_N must be 256, 128, or -128");
            grouped_rcr_kernel_body<N_MASKED_STORE, FUSED_KTAIL>(g);
        }
    } else {
        static_assert(L == Layout::RRR,
                      "grouped_gemm_fp8_kernel: only Layout::RCR and Layout::RRR supported");
        if constexpr (BLOCK_N == 128) {
            grouped_rrr_kernel_bn128_body<N_MASKED_STORE, FUSED_KTAIL>(g);
        } else {
            static_assert(BLOCK_N == 256,
                          "grouped_gemm_fp8_kernel: RRR BLOCK_N must be 256 or 128");
            grouped_rrr_kernel_body<N_MASKED_STORE, FUSED_KTAIL>(g);
        }
    }
}

void dispatch_grouped_rcr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    // FUSED_KTAIL universal (mirrors bf16): the per-group shifted A SRD
    // works for any M_g, so the lds_k_tail_safe gate on m_per_group is no
    // longer needed. K_rem must be 64 (or 0) — the in-kernel K-tail block
    // is hard-coded for that width; non-K_BLOCK-aligned K with K_rem != 64
    // / != 0 is not supported (no production shape exercises it).
    const int K_rem_for_fuse = g.k - g.fast_k;
    const bool fuse_ktail_eligible =
        (g.bpc > 0) && (g.ki > 0) &&
        ((K_rem_for_fuse == 64) || (K_rem_for_fuse == 0));

    static const bool fuse_force_off = []() {
        if (const char* e = std::getenv("TK_GROUPED_RCR_FUSE_OFF")) {
            return std::atoi(e) >= 1;
        }
        return false;
    }();
    const bool fuse_ktail_active =
        fuse_ktail_eligible && !fuse_force_off && (g.fuse_ktail_off == 0);

    int* sk_partial_buf_owned = nullptr;
    if (g.sk_split_n > 0 && g.bpc > 0 && g.ki > 0 && g.sk_partial_buf == nullptr) {
        const int T_max = kittens::ceil_div(g.M_total, BLOCK_SIZE) * g.bpc;
        const size_t buf_bytes =
            static_cast<size_t>(T_max) * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
        hipMallocAsync(reinterpret_cast<void**>(&sk_partial_buf_owned),
                       buf_bytes, g.stream);
        hipMemsetAsync(sk_partial_buf_owned, 0, buf_bytes, g.stream);
        g.sk_partial_buf = sk_partial_buf_owned;
    }

    if (g.bpc > 0 && g.ki > 0) {
        // Round 2026-05-14: 3-way block-size autotune.
        //   g.bn_block = 0   (default) → BLK_M=BLK_N=256 (original kernel)
        //   g.bn_block = 128            → BN=128, BM=256 (bn128 variant)
        //   g.bn_block = -128           → BLK_M=BLK_N=128 (b128 variant)
        // Env knob TK_RCR_BN128 still works as a fallback override.
        static const int rcr_bn128_env = []() {
            if (const char* e = std::getenv("TK_RCR_BN128")) return std::atoi(e);
            return 0;
        }();
        int block_choice = g.bn_block;
        if (block_choice == 0 && rcr_bn128_env != 0) {
            // env: 1/128 → bn128 (256x128); -1/-128 → b128 (128x128)
            if (rcr_bn128_env == -1 || rcr_bn128_env == -128) block_choice = -128;
            else if (rcr_bn128_env == 1 || rcr_bn128_env == 128) block_choice = 128;
        }

        if (block_choice == 128) {
            g.bpc = kittens::ceil_div(g.n, 128);
        } else if (block_choice == -128) {
            g.bpc = kittens::ceil_div(g.n, 128);
        }

        const int N_BLK_eff = (block_choice == 128 || block_choice == -128) ? 128 : BLOCK_SIZE;
        const bool n_aligned = (g.bpc * N_BLK_eff == g.n);

        static const int rcr_slots_env = []() {
            if (const char* e = std::getenv("TK_RCR_NUM_CUS")) {
                const int v = std::atoi(e);
                if (v > 0 && v <= NUM_CUS) return v;
            }
            return NUM_CUS;
        }();
        const int rcr_slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
            ? g.num_slots : rcr_slots_env;

        if (g.chunk_size <= 0 || g.chunk_size > NUM_CUS) {
            static const int env_chunk_size = []() {
                if (const char* e = std::getenv("TK_RCR_CHUNK_SIZE")) {
                    const int v = std::atoi(e);
                    if (v >= 1 && v <= 256) return v;
                }
                return 0;
            }();
            g.chunk_size = env_chunk_size;
        }

        if (block_choice == -128) {
            // BLK_M=BLK_N=128 (b128 — single-mma per K-iter)
            if (fuse_ktail_active) {
                if (n_aligned) {
                    grouped_gemm_fp8_kernel<Layout::RCR, -128, false, true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                } else {
                    grouped_gemm_fp8_kernel<Layout::RCR, -128, true , true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                }
            } else {
                if (n_aligned) {
                    grouped_gemm_fp8_kernel<Layout::RCR, -128, false, false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                } else {
                    grouped_gemm_fp8_kernel<Layout::RCR, -128, true , false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                }
            }
        } else if (block_choice == 128) {
            // BN=128, BM=256 (bn128)
            if (fuse_ktail_active) {
                if (n_aligned) {
                    grouped_gemm_fp8_kernel<Layout::RCR, 128, false, true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                } else {
                    grouped_gemm_fp8_kernel<Layout::RCR, 128, true , true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                }
            } else {
                if (n_aligned) {
                    grouped_gemm_fp8_kernel<Layout::RCR, 128, false, false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                } else {
                    grouped_gemm_fp8_kernel<Layout::RCR, 128, true , false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
                }
            }
        } else if (fuse_ktail_active) {
            if (n_aligned) {
                grouped_gemm_fp8_kernel<Layout::RCR, 256, false, true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_gemm_fp8_kernel<Layout::RCR, 256, true , true><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            }
        } else {
            if (n_aligned) {
                grouped_gemm_fp8_kernel<Layout::RCR, 256, false, false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            } else {
                grouped_gemm_fp8_kernel<Layout::RCR, 256, true , false><<<dim3(rcr_slots), g.block(), 0, g.stream>>>(g);
            }
        }
    } else {
        // No aligned interior at all: main kernel cannot run; tail handles
        // every cell with a full-K reduction.
        g.fast_n = 0;
        g.fast_k = 0;
        g.bpc = 0;
        g.ki = 0;
    }

    // Universal FUSED_KTAIL in the main kernel covers K_rem == {0, 64}; for
    // K_rem != 0 && K_rem != 64 (non-production: every test shape has K %
    // K_BLOCK == 0) there is no fallback path here — caller must round K to
    // K_BLOCK before dispatch. Mirrors bf16: no standalone tail kernel.
    if (sk_partial_buf_owned != nullptr) {
        hipFreeAsync(sk_partial_buf_owned, g.stream);
    }
}

// =============================================================================
// Persistent grouped RRR dispatcher — FP8 (forward-A backward dA path).
//
// Mirror of ``dispatch_grouped_rcr``: the persistent ``grouped_rrr_kernel``
// covers the aligned interior; the N-tail partial column tile is captured
// by ``g.bpc = ceil_div(g.n, BLOCK_SIZE)`` + N_MASKED_STORE; the K-tail
// (K_rem == 64) folds into the main kernel via FUSED_KTAIL=true. Per-group
// M_g uses the per-group shifted gl view + m_limit masked store, so any
// M_g (incl. < BLOCK_SIZE) is supported.
//
// =========================================================================
// Perf ceiling: dgrad ~1.03x of Triton (overall avg, B={4,16}, M={2K,4K}).
// rocprofv3 PMC on os-DN-16-4096 (slowest shape):
//   VALUBusy           ~5.79%   <- kernel mostly idle, memory-stalled
//   MfmaUtil           ~0.00%   <- (counter unreliable for f8f6f4 mfma)
//   LDSBankConflict    0
//   TCC (L2) HIT       310,416 / call
//   TCC (L2) MISS      299,346 / call
//   L2 hit rate        51%      <- POOR; B's [G,K,N] K-stride access
//                                  pattern misses L1/L2 every K-iter on
//                                  short-K shapes (N_inner=2880).
//
// Path to 1.25x (per BF16 grouped_kernel comment line ~3325-3373):
// deeper B double-buffer Bs[2][2] -> Bs[3][2] so the main loop has
// 2 K-iters of compute (~512 cyc) to overlap each L2-miss latency vs
// current 1-iter (~256 cyc). LDS accounting (FP8):
//   Bs[3][2]   = 6 slots × 17KB = 102 KB
//   As[1][2]   = 2 slots × 17KB =  34 KB  (single-K-pair, sacrifices A prefetch)
//   s_offs etc =                 ~520 B
//   Total      =                  136 KB    (within 160 KB MI355X limit)
// Estimated cost of losing A prefetch: +6% per K-iter (A loads block,
// but A is L1-friendly so impact bounded). Estimated benefit of deep B:
// +20-50% if L2 miss latency hides successfully. Net: +14-44%.
//
// Implementation effort: ~200 lines new code (separate kernel function
// or DEEP_LDS template arm) + 3-stage pipeline + correctness validation.
// Profile script: scripts/_profile_fp8_rrr.py.
// =========================================================================
void dispatch_grouped_rrr(grouped_layout_globals g) {
    g.n = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k = static_cast<int>(g.a.cols());

    g.fast_n = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k = (g.k / K_BLOCK)    * K_BLOCK;
    // Mirror RCR: bpc rounds UP so the main kernel sweeps the (possibly
    // partial) last N tile and masks its OOB cells via N_MASKED_STORE.
    // OOB B reads inside the loop return 0 from raw_buffer_load_lds.
    g.bpc    = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki     = g.fast_k / K_BLOCK;

    if (g.M_total <= 0 || g.n <= 0 || g.k <= 0 || g.G <= 0) return;
    if (g.bpc <= 0 || g.ki <= 0) return;

    // bn_block routing (mirror RCR): 0 = default 256, 128 = bn128 variant.
    // 2026-05-20 Phase 2-C: RRR bn128 added to cut V from 256 → ~230 and
    // eliminate the 248B/lane scratch. FUSED_KTAIL not yet supported in
    // bn128; gate it to has_k_tail==false.
    static const int rrr_bn128_env = []() {
        if (const char* e = std::getenv("TK_RRR_BN128")) return std::atoi(e);
        return 0;
    }();
    int block_choice = g.bn_block;
    if (block_choice == 0 && (rrr_bn128_env == 1 || rrr_bn128_env == 128)) {
        block_choice = 128;
    }
    // FUSED_KTAIL universal: K_rem == 64 fuses into the main kernel via the
    // per-group shifted A SRD. K_rem != 0 && != 64 not supported (no
    // production shape; would need a separate fallback if ever needed).
    const bool has_k_tail = (g.k - g.fast_k) == 64;
    // bn128 doesn't support FUSED_KTAIL yet — fall back to 256 if K-tail present.
    if (block_choice == 128 && has_k_tail) block_choice = 0;

    if (block_choice == 128) {
        g.bpc = kittens::ceil_div(g.n, 128);
        const bool n_aligned_bn128 = (g.bpc * 128 == g.n);
        // env TK_RRR_BN128_CHUNK — chiplet swizzle chunk size (default 128;
        // 2026-05-20 sweep: chunk={32,64,128,256} → loss {18.0,20.4,15.2,15.8}%
        // — 128 best; default 64 had been left over from RCR bn128 mirror).
        static const int rrr_bn128_chunk = []() {
            if (const char* e = std::getenv("TK_RRR_BN128_CHUNK")) {
                const int v = std::atoi(e);
                if (v >= 16 && v <= 256) return v;
            }
            return 128;
        }();
        if (g.chunk_size <= 0) g.chunk_size = rrr_bn128_chunk;
        if (n_aligned_bn128) grouped_gemm_fp8_kernel<Layout::RRR, 128, false, false>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        else                 grouped_gemm_fp8_kernel<Layout::RRR, 128, true , false>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
        return;
    }

    // H17 (2026-05-21): RRR body doesn't branch on N_MASKED_STORE (dead
    // template arg — both NMASK=0/1 call store_c_tile_mn_masked_grouped
    // unconditionally). Yet llvm-readobj shows NMASK=0 V=256 A=128 sp=61
    // vs NMASK=1 V=256 A=128 sp=37 — pure compiler whim from differing
    // mangled name. Force-route to the NMASK=1 instance to take the
    // better reg-alloc. Correctness identical since body is identical.
    if (has_k_tail) {
        grouped_gemm_fp8_kernel<Layout::RRR, 256, true , true >
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    } else {
        grouped_gemm_fp8_kernel<Layout::RRR, 256, true , false>
            <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
    }
}

// =============================================================================
// Persistent grouped variable-K (CRR / dB) GEMM (CPU-sync-free) — FP8.
//
// Mirror of the BF16 ``grouped_variable_k_crr`` kernel in
// analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp (rounds 1-2). Drives
// the FP8 backward dB computation: given ``a`` = grad_out [M_total, N_fwd]
// fp8 and ``b`` = x [M_total, K_fwd] fp8, produce
// ``c`` = grad_b [G, N_fwd, K_fwd] bf16 in a single CPU-sync-free launch.
// Per-group K-reduction over M_g rows is realised by adding a per-group
// ``k_offset_tiles = m_start_g / HB`` to the kernel's K-axis tile coords.
//
// Native non-aligned (n, k): bpr = ceil_div(n, BLK), bpc = ceil_div(k, BLK).
// Partial last tiles use ``store_c_tile_mn_masked_grouped`` to drop OOB
// (m, n) cells. Safe because (a) A and B are 2D contiguous tensors, NOT
// 3D-grouped — the SRD-wrap-into-next-group issue blocking forward
// grouped does not apply; and (b) MMA cells at OOB output positions are
// dropped before any global store.
//
// Per-group M_g >= 2*HB = 256 is required (prologue + 2 epilogues each
// consume HB-many K-rows; ki_g = M_g / HB and we need ki_g >= 2). The
// Primus-side uniform-M >= 256 gate enforces this; the kernel ``ki_g <
// 2`` skip is a defensive fallback.
// =============================================================================
struct grouped_var_k_layout_globals_fp8 {
    _gl_fp8 a;                     // [1, 1, M_total, n] — grad_out fp8
    _gl_fp8 b;                     // [1, 1, M_total, k] — x fp8
    _gl_bf16 c;                    // [1, G, n, k]       — grad_b bf16
    float scale_a, scale_b;        // host-side scales (used when no dscale)
    const float* dscale_a;
    const float* dscale_b;
    const int64_t* group_offs;     // [G+1] int64 device prefix-sum
    hipStream_t stream;
    int G;                         // number of groups
    int M_total;                   // sum M_g across groups
    int n;                         // kernel M-output dim (= N_fwd)
    int k;                         // kernel N-output dim (= K_fwd)
    int group_m;
    int bpr;                       // ceil_div(n, BLOCK_SIZE)
    int bpc;                       // ceil_div(k, BLOCK_SIZE)
    int fast_n, fast_k;
    int num_xcds;                  // chiplet-swizzle XCD count (0 → default 8)
    int num_slots;                 // Round-3 (gpt_oss FP8 kernel-only ceiling,
    int chunk_size;                // Round-13 (gpt_oss FP8 kernel-only ceiling,
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__device__ __forceinline__ float resolve_combined_scale_var_k_fp8(
    const grouped_var_k_layout_globals_fp8 &g) {
    const float sa = g.dscale_a ? *g.dscale_a : g.scale_a;
    const float sb = g.dscale_b ? *g.dscale_b : g.scale_b;
    return sa * sb;
}

__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_var_k_kernel_fp8(const grouped_var_k_layout_globals_fp8 g) {
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    A_col_reg a;
#if VARK_SW_PIPE_HOIST_AHEAD
    // Round-24: extra A-tile register staging iter k+1's half-1 LDS A-read.
    A_col_reg a_next;
#endif
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR line ~2020 — host-side
    // ``g.num_xcds`` knob with fallback to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int slots_eff = gridDim.x;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;

    const int tiles_per_group = g.bpr * g.bpc;
    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
        s_cum_tiles[threadIdx.x] =
            static_cast<int>(threadIdx.x) * tiles_per_group;
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    if (threadIdx.x == 0) {
        s_total_tiles = g.G * tiles_per_group;
    }
    __syncthreads();
    const int total_tiles = s_total_tiles;

    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        const int group_idx = gt / tiles_per_group;
        const int tile_start = group_idx * tiles_per_group;
        const int local_tile = gt - tile_start;
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        // ceil_div + floor at 2: partial last K-tile (M_g % HB != 0) and
        // small groups (M_g <= HB) all get >= 2 K-tile iterations; per-group
        // shifted A/B view bounds the SRD so the second virtual K-tile load
        // hardware-clamps to 0 (no MMA contribution). Floor + skip silently
        // dropped these groups.
        const int ki_g_raw = kittens::ceil_div(M_g, HB);
        if (ki_g_raw <= 0) continue;
        const int ki_g = (ki_g_raw < 2) ? 2 : ki_g_raw;
        const int bpr_g = g.bpr;

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        // var_k shifts BOTH A and B (the K-reduction axis is the per-group
        // M_g segment of [M_total, *]) — see patch_gl_to_row_slice.
        auto a_gl_g = g.a;
        auto b_gl_g = g.b;
        patch_gl_to_row_slice(a_gl_g, m_start_g, M_g);
        patch_gl_to_row_slice(b_gl_g, m_start_g, M_g);
        constexpr int k_offset_tiles = 0;

        auto a_co = [&](int s, int k) -> coord<ST_crr_a> {
            return {0, 0, k_offset_tiles + k, s};
        };
        auto b_co = [&](int s, int k) -> coord<ST_crr_b> {
            return {0, 0, k_offset_tiles + k, s};
        };
        auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBM);
        };
        auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };
        auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
            rcr_8w_load_hoist<_NUM_THREADS>(tile, a_gl_g, a_co(s, k), soA);
        };
        auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
            rcr_8w_load_hoist<_NUM_THREADS>(tile, b_gl_g, b_co(s, k), soB);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        global_load_b(Bs[tic][0], bc*2,   0);
        global_load_a(As[tic][0], br*2,   0);
        global_load_b(Bs[tic][1], bc*2+1, 0);
        global_load_a(As[tic][1], br*2+1, 0);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        global_load_b(Bs[toc][0], bc*2,   1);
        global_load_a(As[toc][0], br*2,   1);
        global_load_b(Bs[toc][1], bc*2+1, 1);

        TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

#if VARK_SW_PIPE_HOIST_AHEAD
        load_a(a_next, As[tic][0], wm);
#endif

        TK_PRAGMA_UNROLL(VARK_MAIN_UNROLL)
        for (int k = 0; k < ki_g - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
#if VARK_SW_PIPE_HOIST_AHEAD
            a = a_next;
#else
            load_a(a, As[tic][0], wm);
#endif
            global_load_a(As[toc][1], br*2+1, k+1);
            TK_WAIT_LGKM(CRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
#if !VARK_DROP_REDUNDANT_LGKM_DRAIN
            MAYBE_DRAIN_LGKM();
#endif
#if VARK_HOIST_PREFETCH_INTO_HALF1
            __builtin_amdgcn_s_barrier();
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
#endif
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            crr_mma(cB, a, b1);
            CRR_MMA_END();
#if !VARK_DROP_BARRIER_2 && !VARK_HOIST_PREFETCH_INTO_HALF1
            CRR_STEADY_MID_BARRIER();
#endif

            load_a(a, As[tic][1], wm);
#if !VARK_HOIST_PREFETCH_INTO_HALF1
            global_load_a(As[tic][0], br*2, k+2);
            global_load_b(Bs[tic][1], bc*2+1, k+2);
#endif
            TK_WAIT_VMCNT(CRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
#if !VARK_DROP_REDUNDANT_LGKM_DRAIN
            MAYBE_DRAIN_LGKM();
#endif
#if VARK_SW_PIPE_HOIST_AHEAD
            load_a(a_next, As[toc][0], wm);
#endif
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
#if !VARK_DROP_BARRIER_4
            __builtin_amdgcn_s_barrier();
#endif
            global_load_b(Bs[tic][0], bc*2, k+2);
        }

        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            global_load_a(As[toc][1], br*2+1, ki_g-1);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }

        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cA, a, b0);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cB, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            CRR_MMA_BEGIN();
            crr_mma(cC, a, b0);
            crr_mma(cD, a, b1);
            CRR_MMA_END();
            __builtin_amdgcn_s_barrier();
        }

        // Apply scale + 2-axis-masked store. Output is per-group [n, k]
        // sub-tensor at depth = group_idx; no m_subtile_C row shift.
        const float combined_scale = resolve_combined_scale_var_k_fp8(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        store_c_tile_mn_masked_grouped(g.c, cA, group_idx,
            br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cB, group_idx,
            br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cC, group_idx,
            br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n, g.k);
        store_c_tile_mn_masked_grouped(g.c, cD, group_idx,
            br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}

void dispatch_grouped_var_k_fp8(grouped_var_k_layout_globals_fp8 g) {
    g.n = static_cast<int>(g.a.cols());
    g.k = static_cast<int>(g.b.cols());
    g.M_total = static_cast<int>(g.a.rows());

    g.fast_n = g.n;
    g.fast_k = g.k;
    g.bpr = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.bpc = kittens::ceil_div(g.k, BLOCK_SIZE);

    if (g.bpr <= 0 || g.bpc <= 0 || g.G <= 0) return;

    int slots_dispatch;
    if (g.num_slots > 0 && g.num_slots <= NUM_CUS) {
        slots_dispatch = g.num_slots;
    } else {
        static const int env_slots = []() {
            if (const char* e = std::getenv("TK_VARK_NUM_CUS")) {
                const int v = std::atoi(e);
                if (v > 0 && v <= NUM_CUS) return v;
            }
            return NUM_CUS;
        }();
        slots_dispatch = env_slots;
    }

    if (g.chunk_size <= 0 || g.chunk_size > NUM_CUS) {
        static const int env_chunk_size = []() {
            if (const char* e = std::getenv("TK_VARK_CHUNK_SIZE")) {
                const int v = std::atoi(e);
                if (v >= 1 && v <= 256) return v;
            }
            return 0;  // 0 → kernel uses default 64
        }();
        g.chunk_size = env_chunk_size;
    }

    grouped_var_k_kernel_fp8<<<dim3(slots_dispatch), g.block(), 0, g.stream>>>(g);
}

template<Layout L>
void dispatch(layout_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    if constexpr (L == Layout::RCR) {
        g.k = static_cast<int>(g.a.cols());
    } else if constexpr (L == Layout::RRR) {
        g.k = static_cast<int>(g.a.cols());
    } else {
        g.k = static_cast<int>(g.b.rows());
    }

    g.fast_m = (g.m / BLK) * BLK;
    g.fast_n = (g.n / BLK) * BLK;
    g.fast_k = (g.k / BK) * BK;
    g.bpr = g.fast_m / BLK;
    g.bpc = g.fast_n / BLK;
    g.ki = g.fast_k / BK;

    // FP8 dense Phase 4: enable native N-tail via main-kernel column-
    // masked C store + ``bpc = ceil_div(n, BLK)``, mirroring BF16 dense.
    // Safe because ``rcr_8w_load_hoist`` was refactored to use a FULL-
    // tensor SRD with per-tile SOFFSET (see helper near line 400) — OOB
    // column reads from a partial last tile clamp to 0 from the SRD
    // bound rather than faulting on unmapped memory.
    //
    // Conditions to enable ``ceil_div`` bpc (mirror BF16):
    //   1. The 8-wave generic ``gemm_kernel<L>`` is selected. The
    //      4-wave RCR fast path uses ``rcr_4w::prepare_g2s`` which still
    //      builds tile-local SRDs and is NOT yet refactored — keep its
    //      ``bpc = fast_n/BLK`` (its kernel is selected only for very
    //      large grids; tail kernel handles N misalignment in those
    //      large-grid cases).
    //   2. K is fully aligned (``fast_k == k``). When K is misaligned
    //      the tail kernel needs a deterministic fast/tail boundary in
    //      N; same as BF16 dense.
    bool main_covers_n = false;
    if (g.bpr > 0 && g.bpc > 0 && g.ki >= 2) {
        if (g.fast_k == g.k) {
            g.bpc = kittens::ceil_div(g.n, BLK);
            main_covers_n = true;
        }
        gemm_kernel<L><<<g.grid(), g.block(), 0, g.stream>>>(g);
    } else {
        g.fast_k = 0;
        g.ki = 0;
    }

    const bool need_tail = (g.fast_m != g.m) || (g.fast_k != g.k) ||
                           (!main_covers_n && g.fast_n != g.n) || (g.ki == 0);
    if (need_tail) {
        dim3 tail_block(TAIL_BLOCK_N, TAIL_BLOCK_M);
        dim3 tail_grid(
            kittens::ceil_div(g.n, TAIL_BLOCK_N),
            kittens::ceil_div(g.m, TAIL_BLOCK_M)
        );
        gemm_tail_kernel<L><<<tail_grid, tail_block, 0, g.stream>>>(g);
    }
}

#ifndef PRIMUS_TURBO_HK_INTEGRATION
static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

constexpr int DEFAULT_GROUP_M = 4;

template<Layout L>
static void gemm_wrapper(pybind11::object a, pybind11::object b, pybind11::object c,
                          pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                          int group_m) {
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        nullptr, nullptr,
    };
    dispatch<L>(g);
}

// Variant of gemm_wrapper that takes the per-tensor FP8 scales as 0-d device
// tensors (one element each) instead of host-side scalars / 0-d host tensors.
// The Python-side host wrapper used to call `(a_scale_inv * b_scale_inv).item()`
// on every dispatch which is a stream sync that costs ~18us on small dense
// FP8 shapes (≈ 30% of the kernel itself) — this entry skips that sync by
// passing the device pointers straight through to the kernel's epilogue, which
// reads one b32 from global memory at scale-application time. The two scales
// are both used in the epilogue only, so the cost of the load is fully hidden
// behind the GEMM main loop.
template<Layout L>
static void gemm_wrapper_dscale(pybind11::object a, pybind11::object b, pybind11::object c,
                                 pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                                 int group_m) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        group_m,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
    };
    dispatch<L>(g);
}

static void grouped_rcr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group,
                           int num_xcds,
                           int num_slots,
                           int chunk_size = 0,
                           int fuse_ktail_off = 0,
                           int sk_split_n = 0,
                           uint64_t sk_workspace_ptr = 0) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group,num_slots,chunk_size,fuse_ktail_off,sk_split_n */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group, num_slots, chunk_size, fuse_ktail_off, sk_split_n,
        // sk_partial_buf left default-init (nullptr); R13a alloc fills it when sk_split_n > 0.
    };
    // R17: caller-allocated workspace override. If sk_workspace_ptr != 0, the
    // dispatcher's per-call hipMallocAsync branch is skipped (gated below on
    // g.sk_partial_buf == nullptr). Cast through void* to silence -Wcast-align.
    if (sk_workspace_ptr != 0) {
        g.sk_partial_buf = reinterpret_cast<int*>(static_cast<uintptr_t>(sk_workspace_ptr));
    }
    dispatch_grouped_rcr(g);
}

static void grouped_rcr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds,
    int num_slots,
    int chunk_size = 0,
    int fuse_ktail_off = 0,
    int sk_split_n = 0,
    uint64_t sk_workspace_ptr = 0) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group,num_slots,chunk_size,fuse_ktail_off,sk_split_n */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group, num_slots, chunk_size, fuse_ktail_off, sk_split_n,
        // sk_partial_buf left default-init (nullptr); R13a alloc fills it when sk_split_n > 0.
    };
    // R17: caller-allocated workspace override (mirrors grouped_rcr_fn).
    if (sk_workspace_ptr != 0) {
        g.sk_partial_buf = reinterpret_cast<int*>(static_cast<uintptr_t>(sk_workspace_ptr));
    }
    dispatch_grouped_rcr(g);
}

static void grouped_rrr_fn(pybind11::object a, pybind11::object b, pybind11::object c,
                           pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                           pybind11::object group_offs_obj,
                           int group_m,
                           int m_per_group,
                           int num_xcds) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

static void grouped_rrr_dscale_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int m_per_group,
    int num_xcds) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G,n,k,ki,bpc,group_m,num_xcds,M_total,fast_n,fast_k,m_per_group */
        G, 0, 0, 0, 0, group_m, num_xcds, 0, 0, 0, m_per_group,
    };
    dispatch_grouped_rrr(g);
}

static void grouped_variable_k_crr_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int num_xcds,
    int num_slots,
    int chunk_size = 0) {
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_var_k_layout_globals_fp8 g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k, num_xcds, num_slots, chunk_size */
        G, 0, 0, 0, group_m, 0, 0, 0, 0, num_xcds, num_slots, chunk_size,
    };
    dispatch_grouped_var_k_fp8(g);
}

static void grouped_variable_k_crr_dscale_fp8_fn(
    pybind11::object a, pybind11::object b, pybind11::object c,
    pybind11::object scale_a_obj, pybind11::object scale_b_obj,
    pybind11::object group_offs_obj,
    int group_m,
    int num_xcds,
    int num_slots,
    int chunk_size = 0) {
    auto sa_ptr = scale_a_obj.attr("data_ptr")().cast<uintptr_t>();
    auto sb_ptr = scale_b_obj.attr("data_ptr")().cast<uintptr_t>();
    auto group_offs_ptr = group_offs_obj.attr("data_ptr")().cast<uintptr_t>();
    int G = group_offs_obj.attr("numel")().cast<int>() - 1;
    grouped_var_k_layout_globals_fp8 g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        0.f, 0.f,
        reinterpret_cast<const float*>(sa_ptr),
        reinterpret_cast<const float*>(sb_ptr),
        reinterpret_cast<const int64_t*>(group_offs_ptr),
        {},
        /* G, M_total, n, k, group_m, bpr, bpc, fast_n, fast_k, num_xcds, num_slots, chunk_size */
        G, 0, 0, 0, group_m, 0, 0, 0, 0, num_xcds, num_slots, chunk_size,
    };
    dispatch_grouped_var_k_fp8(g);
}
#endif  // !PRIMUS_TURBO_HK_INTEGRATION (close range #1)

#ifndef PRIMUS_TURBO_HK_INTEGRATION
PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.doc() = "FP8 per-tensor GEMM: C = A op B * scale_a * scale_b";
    m.def("gemm_rcr", &gemm_wrapper<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rrr", &gemm_wrapper<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_crr", &gemm_wrapper<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rcr_dscale", &gemm_wrapper_dscale<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_rrr_dscale", &gemm_wrapper_dscale<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("gemm_crr_dscale", &gemm_wrapper_dscale<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
    m.def("supports_shape", [](int m, int n, int k) -> bool {
        return m > 0 && n > 0 && k > 0;
    });
    // [fused-act] BF16 max-abs reduction → single fp32 device scalar. Used by
    // Primus-Turbo's ``_fused_act_grouped_fp8_forward`` (Round 1 of the FP8
    // grouped fused-activation-quant lever) to produce the activation scale
    // BEFORE calling the C++ ``quantize_fp8_tensorwise(input, scale=...)`` —
    m.def("grouped_rcr", &grouped_rcr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0,
          pybind11::arg("fuse_ktail_off") = 0,
          pybind11::arg("sk_split_n") = 0,
          pybind11::arg("sk_workspace_ptr") = uint64_t{0});
    m.def("grouped_rcr_dscale", &grouped_rcr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0,
          pybind11::arg("fuse_ktail_off") = 0,
          pybind11::arg("sk_split_n") = 0,
          // R17: see grouped_rcr m.def above.
          pybind11::arg("sk_workspace_ptr") = uint64_t{0});
    m.def("grouped_rrr", &grouped_rrr_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    m.def("grouped_rrr_dscale", &grouped_rrr_dscale_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("m_per_group") = 0,
          pybind11::arg("num_xcds") = 0);
    m.def("grouped_variable_k_crr", &grouped_variable_k_crr_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0);
    m.def("grouped_variable_k_crr_dscale",
          &grouped_variable_k_crr_dscale_fp8_fn,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_offs"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M,
          pybind11::arg("num_xcds") = 0,
          pybind11::arg("num_slots") = 0,
          pybind11::arg("chunk_size") = 0);
    m.attr("DEFAULT_GROUP_M") = DEFAULT_GROUP_M;
    m.attr("BLOCK_SIZE") = BLK;
    m.attr("K_BLOCK") = BK;

}
#endif  // !PRIMUS_TURBO_HK_INTEGRATION
