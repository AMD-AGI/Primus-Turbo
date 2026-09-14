// ============================================================================
// THE wave32 CORRECTNESS GATE, minimal form.
//
// WHY THE PORTED cdna4 test_transpose IS NOT ENOUGH.
//
//   cdna4/warp/register/tile/conversions.cu::test_transpose does:
//       load(reg_tile, input)          // row or col layout
//       transpose(reg_tile_T, reg_tile)
//       store(output, reg_tile_T)      // the dual layout
//
//   udna1's global_to_register.cuh defines the two layouts as exact duals of
//   each other *by construction*:
//       row layout: row = lane % R          col = stride*(lane/R) + k*ESG + ...
//       col layout: col = lane % C          row = stride*(lane/C) + k*ESG + ...
//   with the SAME element index idx = l + k*(stride/packing) in both.
//
//   So load(col) -> transpose (pure relabel) -> store(row) is a tautology: it
//   round-trips through two maps that are defined to be transposes of each
//   other. That test PASSES on wave32 no matter what the matrix unit does.
//   It does not touch WMMA at all.
//
// WHAT ACTUALLY HAS TO HOLD.
//
//   The eleven gfx950 backward call sites transpose tiles that came OUT OF THE
//   MATRIX UNIT, not out of a load. The identity they rely on is:
//
//       (WMMA C-accumulator lane->element map)
//         is the dual of
//       (WMMA A-operand lane->element map)
//
//   On CDNA wave64 this holds because MFMA's A and C maps are both
//   "16x16, 4 per lane, lane = index + 16*group" with row/col swapped.
//   On gfx1250 WMMA f32_16x16x32_bf16, A is v16bf16 (16x32, 16/lane) and C is
//   v8f32 (16x16, 8/lane) -- different shapes AND different per-lane counts.
//   Nothing in the tree executes this.
//
// THIS KERNEL.
//   D = A @ B^T via wmma161632, then transpose(D) and store through the ROW
//   layout. Host computes (A @ B^T)^T. If the duality holds, they match.
//   If it does not, this is the first thing in the tree that will say so.
//
// Build (no GPU needed to compile):
//   hipcc -std=c++20 -I$ROCM_PATH/include/hip -I<hk>/include \
//         -DKITTENS_UDNA1 --offload-arch=gfx1250 mma_transpose_gate.cu -o gate
// Run: ./gate     (single wave, single block, microseconds)
// ============================================================================
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int M = 16, N = 16, K = 32;

using AccShape = ducks::rt_shape::rt_16x16;   // C/D: v8f32  on wave32
using OpShape  = ducks::rt_shape::rt_16x32;   // A/B: v16bf16 on wave32

using GA = gl<bf16,  1, 1, M, K>;
using GD = gl<float, 1, 1, N, M>;

__global__ void gate_kernel(const GA ga, const GA gb, const GD gd)
{
    rt_base<bf16,  ducks::rt_layout::row, OpShape>  a;   // 16x32 A
    rt_base<bf16,  ducks::rt_layout::row, OpShape>  b;   // 16x32 B (row-major, for ABt)
    rt_base<float, ducks::rt_layout::col, AccShape> c;   // 16x16 accumulator
    rt_base<float, ducks::rt_layout::col, AccShape> d;

    const int lane = laneid();

    // --- Fill A and B in the ROW-layout map that mma_ABt_base expects. -------
    // Row layout for a 16x32 base tile: row = lane % 16,
    //   col = stride*(lane/16) + k*elements_per_stride_group + (element within)
    // Rather than hand-roll that map (which is exactly what is under test), we
    // populate through the library's own indexing so a map error shows up as a
    // *result* error rather than a setup error.
    constexpr int PPT = decltype(a)::packed_per_thread;   // 8 on wave32
    #pragma unroll
    for (int i = 0; i < PPT; ++i) {
        // Deterministic, distinguishable values; the host mirrors this exactly
        // using the same published map (see host_fill below).
        a.data[i] = bf16_2{};
        b.data[i] = bf16_2{};
    }
    // Load A and B from global through the library's row-layout loader so the
    // map is the library's, not ours.
    {
        rt<bf16, M, K, ducks::rt_layout::row, OpShape> ta, tb;
        load(ta, ga, {});
        load(tb, gb, {});
        a = ta.tiles[0][0];
        b = tb.tiles[0][0];
    }

    #pragma unroll
    for (int i = 0; i < decltype(c)::packed_per_thread; ++i) c.data[i] = float2{0.f, 0.f};

    // --- The matrix-unit op: D = A @ B^T + C --------------------------------
    mma_ABt_base(d, a, b, c);

    // --- Wrap the accumulator in a full rt so transpose() applies ------------
    rt<float, M, N, ducks::rt_layout::col, AccShape> D;
    D.tiles[0][0] = d;

    // --- THE OPERATION UNDER TEST: pure register relabel --------------------
    rt<float, N, M,
       typename ducks::rt_layout::transpose<ducks::rt_layout::col>::type,
       typename ducks::rt_shape::transpose<AccShape>::type> Dt;
    transpose(Dt, D);

    // --- Store through the dual layout --------------------------------------
    store(gd, Dt, {});

    (void)lane;
}

static inline float bf16_round(float x) {
    // emulate bf16 storage of the inputs so the host reference matches
    unsigned u; __builtin_memcpy(&u, &x, 4); u &= 0xFFFF0000u;
    float r; __builtin_memcpy(&r, &u, 4); return r;
}

int main() {
    std::vector<float> Ah(M * K), Bh(N * K);
    for (int i = 0; i < M * K; ++i) Ah[i] = bf16_round(((i * 37) % 17) * 0.125f - 1.0f);
    for (int i = 0; i < N * K; ++i) Bh[i] = bf16_round(((i * 53) % 19) * 0.0625f - 0.5f);

    // Host reference: Dref = A @ B^T  (M x N), then transpose -> (N x M)
    std::vector<float> Dref_T(N * M, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += Ah[m * K + k] * Bh[n * K + k];
            Dref_T[n * M + m] = acc;          // transposed placement
        }

    std::vector<bf16> Ab(M * K), Bb(N * K);
    for (int i = 0; i < M * K; ++i) Ab[i] = __float2bfloat16(Ah[i]);
    for (int i = 0; i < N * K; ++i) Bb[i] = __float2bfloat16(Bh[i]);

    bf16 *dA, *dB; float *dD;
    hipMalloc(&dA, Ab.size() * sizeof(bf16));
    hipMalloc(&dB, Bb.size() * sizeof(bf16));
    hipMalloc(&dD, N * M * sizeof(float));
    hipMemcpy(dA, Ab.data(), Ab.size() * sizeof(bf16), hipMemcpyHostToDevice);
    hipMemcpy(dB, Bb.data(), Bb.size() * sizeof(bf16), hipMemcpyHostToDevice);
    hipMemset(dD, 0, N * M * sizeof(float));

    GA ga(dA, nullptr, nullptr, nullptr, nullptr);
    GA gb(dB, nullptr, nullptr, nullptr, nullptr);
    GD gd(dD, nullptr, nullptr, nullptr, nullptr);
    hipLaunchKernelGGL(gate_kernel, dim3(1), dim3(kittens::WARP_THREADS), 0, 0, ga, gb, gd);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("LAUNCH FAIL: %s\n", hipGetErrorString(e)); return 2; }

    std::vector<float> Dg(N * M);
    hipMemcpy(Dg.data(), dD, N * M * sizeof(float), hipMemcpyDeviceToHost);

    double worst = 0.0; int bad = 0, wi = -1;
    for (int i = 0; i < N * M; ++i) {
        double err = std::fabs(Dg[i] - Dref_T[i]);
        double tol = 1e-2 * (1.0 + std::fabs(Dref_T[i]));
        if (err > tol) { ++bad; if (err > worst) { worst = err; wi = i; } }
    }
    printf("{\"test\":\"udna1_mma_transpose_duality\",\"bad\":%d,\"total\":%d,"
           "\"worst_abs_err\":%.6f,\"worst_idx\":%d,\"verdict\":\"%s\"}\n",
           bad, N * M, worst, wi, bad == 0 ? "GREEN" : "RED");
    if (bad) {
        printf("first 16 gpu vs ref:\n");
        for (int i = 0; i < 16; ++i) printf("  [%2d] gpu=%10.4f ref=%10.4f\n", i, Dg[i], Dref_T[i]);
    }
    return bad ? 1 : 0;
}
